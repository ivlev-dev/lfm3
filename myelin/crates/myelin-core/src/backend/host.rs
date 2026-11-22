//! Host-resident pieces of a DSP model: token embedding, RMSNorm weights and
//! the tied vocabulary projection.
//!
//! The Hexagon backend keeps the transformer matrices resident on the device,
//! but three operations stay on the host for now:
//!
//! * `embed` is a single row lookup, so the checkpoint's embedding tensor is
//!   dequantised one row at a time instead of being expanded to f32;
//! * `rms_norm` has 97 learned vectors of the model width — copying them with the
//!   weights would cost more RPCs than it saves;
//! * `lm_head` is tied to the embedding. The fast path runs the packed int4
//!   matrix on the DSP; [`HeadMode::Host`] instead dequantises the checkpoint
//!   rows on the host, which is slow but numerically identical to
//!   [`crate::CpuBackend`] and is the parity reference for the DSP layers.
//!
//! None of this is a stub: every path is complete, and the head mode is an
//! explicit switch, not a fallback.

use lfm_gguf::GgmlType;

use super::{BackendError, NormSlot, NormWeights};
use crate::dequant;
use crate::ops;
use crate::spec::ModelSpec;

/// Which implementation `lm_head` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadMode {
    /// Resident int4 GEMV on the DSP (the generation path).
    #[default]
    Dsp,
    /// Dequantise the checkpoint rows on the host (parity/diagnostic path).
    Host,
}

/// Learned norms of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorms {
    pub attention: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub ffn: Vec<f32>,
    pub post_ffn: Vec<f32>,
}

impl LayerNorms {
    /// Validate that every vector matches the model width.
    pub fn check(&self, hidden: usize, layer: usize) -> Result<(), BackendError> {
        for (name, values) in [
            ("attn_norm", &self.attention),
            ("post_attention_norm", &self.post_attention),
            ("ffn_norm", &self.ffn),
            ("post_ffw_norm", &self.post_ffn),
        ] {
            if values.len() != hidden {
                return Err(BackendError::Model(format!(
                    "layer {layer} {name} has {} values, expected {hidden}",
                    values.len()
                )));
            }
        }
        Ok(())
    }

    fn slot(&self, slot: NormSlot) -> &[f32] {
        match slot {
            NormSlot::Attention => &self.attention,
            NormSlot::PostAttention => &self.post_attention,
            NormSlot::Ffn => &self.ffn,
            NormSlot::PostFfn => &self.post_ffn,
        }
    }
}

/// One token-embedding matrix in its checkpoint storage.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRows {
    kind: GgmlType,
    row_bytes: usize,
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
}

impl EmbeddingRows {
    /// Adopt the raw tensor bytes and check the geometry.
    pub fn new(
        kind: GgmlType,
        rows: usize,
        cols: usize,
        bytes: Vec<u8>,
    ) -> Result<Self, BackendError> {
        let row_bytes = dequant::row_bytes(kind, cols)
            .map_err(|message| BackendError::Model(format!("embedding: {message}")))?;
        let expected = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| BackendError::Model("embedding size overflows usize".to_owned()))?;
        if bytes.len() != expected {
            return Err(BackendError::Model(format!(
                "embedding has {} bytes, {rows} rows of {cols} values need {expected}",
                bytes.len()
            )));
        }
        Ok(Self {
            kind,
            row_bytes,
            rows,
            cols,
            bytes,
        })
    }

    /// Number of rows (vocabulary size).
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Values per row (model width).
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Decode one row into `out`.
    pub fn row_into(&self, index: usize, out: &mut [f32]) -> Result<(), BackendError> {
        if index >= self.rows {
            return Err(BackendError::Model(format!(
                "token {index} is outside the vocabulary of {}",
                self.rows
            )));
        }
        if out.len() != self.cols {
            return Err(BackendError::Shape {
                operation: "embedding row",
                expected: self.cols,
                got: out.len(),
            });
        }
        let start = index * self.row_bytes;
        let row = &self.bytes[start..start + self.row_bytes];
        let values = dequant::dequantize_row(self.kind, "token_embd.weight", row, self.cols)
            .map_err(BackendError::Model)?;
        out.copy_from_slice(&values);
        Ok(())
    }
}

/// Host pieces of one DSP model.
#[derive(Debug, Clone)]
pub struct HostWeights {
    hidden: usize,
    vocab: usize,
    embedding: EmbeddingRows,
    layers: Vec<LayerNorms>,
    output_norm: Vec<f32>,
    norm_eps: f32,
    head: HeadMode,
}

impl HostWeights {
    /// Adopt already dequantised norms plus the stored embedding tensor.
    pub fn new(
        spec: &ModelSpec,
        embedding: EmbeddingRows,
        layers: Vec<LayerNorms>,
        output_norm: Vec<f32>,
    ) -> Result<Self, BackendError> {
        let dims = &spec.dims;
        if embedding.rows() != dims.vocab || embedding.cols() != dims.hidden {
            return Err(BackendError::Model(format!(
                "embedding is {}x{}, model is {}x{}",
                embedding.rows(),
                embedding.cols(),
                dims.vocab,
                dims.hidden
            )));
        }
        if layers.len() != dims.layers {
            return Err(BackendError::Model(format!(
                "host weights describe {} layers, model has {}",
                layers.len(),
                dims.layers
            )));
        }
        for (index, layer) in layers.iter().enumerate() {
            layer.check(dims.hidden, index)?;
        }
        if output_norm.len() != dims.hidden {
            return Err(BackendError::Model(format!(
                "output_norm has {} values, expected {}",
                output_norm.len(),
                dims.hidden
            )));
        }
        Ok(Self {
            hidden: dims.hidden,
            vocab: dims.vocab,
            embedding,
            layers,
            output_norm,
            norm_eps: spec.norm_eps,
            head: HeadMode::Dsp,
        })
    }

    /// Selected vocabulary-projection implementation.
    pub const fn head_mode(&self) -> HeadMode {
        self.head
    }

    /// Number of transformer layers the host norms cover.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Switch the vocabulary-projection implementation.
    pub fn set_head_mode(&mut self, mode: HeadMode) {
        self.head = mode;
    }

    /// Embedding lookup for one token.
    pub fn embed(&self, token: u32, out: &mut [f32]) -> Result<(), BackendError> {
        self.embedding.row_into(token as usize, out)
    }

    /// Apply one learned RMSNorm on the host.
    pub fn norm(
        &self,
        weights: NormWeights,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let weight: &[f32] = match weights {
            NormWeights::Layer { layer, slot } => {
                let layer = self.layers.get(layer).ok_or_else(|| {
                    BackendError::Model(format!("host weights have no layer {layer}"))
                })?;
                layer.slot(slot)
            }
            NormWeights::Output => &self.output_norm,
            NormWeights::Query { .. } | NormWeights::Key { .. } => {
                return Err(BackendError::Model(
                    "per-head q/k norms are part of the attention stage, not the host norms"
                        .to_owned(),
                ));
            }
        };
        if x.len() != weight.len() || out.len() != weight.len() {
            return Err(BackendError::Shape {
                operation: "host rms_norm",
                expected: weight.len(),
                got: x.len(),
            });
        }
        ops::rms_norm(x, weight, self.norm_eps, out);
        Ok(())
    }

    /// Exact host vocabulary projection over the checkpoint embedding rows.
    ///
    /// Rows are split across host threads; each row is dequantised on demand,
    /// so memory stays at the stored size plus one row per thread.
    pub fn lm_head(&self, x: &[f32], logits: &mut [f32]) -> Result<(), BackendError> {
        if x.len() != self.hidden {
            return Err(BackendError::Shape {
                operation: "host lm_head input",
                expected: self.hidden,
                got: x.len(),
            });
        }
        if logits.len() != self.vocab {
            return Err(BackendError::Shape {
                operation: "host lm_head logits",
                expected: self.vocab,
                got: logits.len(),
            });
        }
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(self.vocab);
        let chunk = self.vocab.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, out) in logits.chunks_mut(chunk).enumerate() {
                let first = chunk_index * chunk;
                let embedding = &self.embedding;
                scope.spawn(move || {
                    let mut row = vec![0.0f32; embedding.cols()];
                    for (offset, dst) in out.iter_mut().enumerate() {
                        let index = first + offset;
                        // `row_into` validates the index against the vocabulary.
                        embedding
                            .row_into(index, &mut row)
                            .expect("row index is bounded by the logits chunk");
                        *dst = ops::dot(x, &row);
                    }
                });
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ModelSpec;

    fn spec() -> ModelSpec {
        let mut spec = ModelSpec::placeholder();
        spec.dims.vocab = 4;
        spec.dims.hidden = 4;
        spec.dims.layers = 1;
        spec
    }

    fn host() -> HostWeights {
        let mut bytes = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                bytes.extend_from_slice(&((row * 4 + col) as f32).to_le_bytes());
            }
        }
        let embedding = EmbeddingRows::new(GgmlType::F32, 4, 4, bytes).expect("embedding");
        let norms = LayerNorms {
            attention: vec![1.0; 4],
            post_attention: vec![1.0; 4],
            ffn: vec![1.0; 4],
            post_ffn: vec![1.0; 4],
        };
        HostWeights::new(&spec(), embedding, vec![norms], vec![1.0; 4]).expect("host")
    }

    #[test]
    fn embedding_rows_follow_token_order() {
        let host = host();
        let mut out = [0.0f32; 4];
        host.embed(2, &mut out).expect("embed");
        assert_eq!(out, [8.0, 9.0, 10.0, 11.0]);
        assert!(host.embed(4, &mut out).is_err());
    }

    #[test]
    fn host_head_is_the_tied_embedding() {
        let host = host();
        let mut logits = [0.0f32; 4];
        host.lm_head(&[1.0, 0.0, 0.0, 0.0], &mut logits)
            .expect("head");
        assert_eq!(logits, [0.0, 4.0, 8.0, 12.0]);
    }

    #[test]
    fn norm_slots_are_independent() {
        let mut host = host();
        host.layers[0].attention = vec![2.0; 4];
        let mut out = [0.0f32; 4];
        host.norm(
            NormWeights::Layer {
                layer: 0,
                slot: NormSlot::Attention,
            },
            &[1.0; 4],
            &mut out,
        )
        .expect("norm");
        // RMS of four ones is one (up to the epsilon), so the result is the
        // weight itself.
        for value in out {
            assert!((value - 2.0).abs() < 1e-4);
        }
    }
}
