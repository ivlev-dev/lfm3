//! Host CPU backend.
//!
//! Loads a GGUF checkpoint, dequantizes the tensors LFM3 needs to f32 and runs
//! the reference math with [`crate::ops`]. This is the numerical oracle for the
//! DSP port: weights are the same values the device will hold, only the storage
//! layout differs.

use lfm_gguf::{GgmlType, GgufFile, TensorInfo};
use lfm_quant;

use crate::backend::{AttentionCall, Backend, BackendError, NormSlot, NormWeights};
use crate::kv::KvCache;
use crate::ops::{self, RopeTable};
use crate::spec::{AttentionKind, LayerSpec, ModelSpec};
use crate::RuntimeError;

/// Row-major f32 matrix: `rows` output channels of `cols` inputs each.
#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Wrap an already dequantized buffer.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, BackendError> {
        if rows == 0 || cols == 0 || data.len() != rows * cols {
            return Err(BackendError::Model(format!(
                "matrix {rows}x{cols} got {} values",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// One output channel.
    pub fn row(&self, index: usize) -> &[f32] {
        &self.data[index * self.cols..(index + 1) * self.cols]
    }

    fn multiply(&self, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
        if x.len() != self.cols {
            return Err(BackendError::Shape {
                operation: "matrix input",
                expected: self.cols,
                got: x.len(),
            });
        }
        if out.len() != self.rows {
            return Err(BackendError::Shape {
                operation: "matrix output",
                expected: self.rows,
                got: out.len(),
            });
        }
        ops::matvec(&self.data, self.rows, self.cols, x, out);
        Ok(())
    }
}

struct AttentionWeights {
    query: Matrix,
    key: Matrix,
    value: Matrix,
    gate: Matrix,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    output: Matrix,
    rope: RopeTable,
}

struct ConvWeights {
    in_proj: Matrix,
    template: Matrix,
    out_proj: Matrix,
}

struct FeedForward {
    up: Matrix,
    gate: Matrix,
    down: Matrix,
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    ffn: FeedForward,
    attention: Option<AttentionWeights>,
    conv: Option<ConvWeights>,
}

enum Head {
    Tied,
    Separate(Matrix),
}

/// Scratch activations reused across steps.
struct Scratch {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    gates: Vec<f32>,
    attended: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    acted: Vec<f32>,
    conv_mix: Vec<f32>,
    conv_input: Vec<f32>,
    conv_output: Vec<f32>,
    conv_gated: Vec<f32>,
}

/// f32 host implementation of the layer-level backend contract.
pub struct CpuBackend {
    spec: ModelSpec,
    embedding: Matrix,
    output_norm: Vec<f32>,
    head: Head,
    layers: Vec<LayerWeights>,
    scratch: Scratch,
}

impl CpuBackend {
    /// Open a checkpoint and load every tensor the spec describes.
    pub fn load(path: impl AsRef<std::path::Path>, spec: &ModelSpec) -> Result<Self, RuntimeError> {
        let file =
            GgufFile::open(path.as_ref()).map_err(|error| RuntimeError::Gguf(error.to_string()))?;
        Self::from_gguf(&file, spec)
    }

    /// Load tensors from an already opened checkpoint.
    pub fn from_gguf(file: &GgufFile, spec: &ModelSpec) -> Result<Self, RuntimeError> {
        let dims = &spec.dims;
        let hidden = dims.hidden;
        let kv_dim = dims.kv_dim();
        let embedding = load_matrix(file, "token_embd.weight", hidden, dims.vocab)?;
        let output_norm = load_vector(file, "output_norm.weight", hidden)?;
        let head = if spec.tied_embeddings {
            Head::Tied
        } else {
            Head::Separate(load_matrix(file, "output.weight", hidden, dims.vocab)?)
        };

        let mut layers = Vec::with_capacity(spec.layers.len());
        for (index, layer) in spec.layers.iter().copied().enumerate() {
            let prefix = format!("blk.{index}");
            let ffn = FeedForward {
                up: load_matrix(file, &format!("{prefix}.ffn_up.weight"), hidden, dims.ffn)?,
                gate: load_matrix(file, &format!("{prefix}.ffn_gate.weight"), hidden, dims.ffn)?,
                down: load_matrix(file, &format!("{prefix}.ffn_down.weight"), dims.ffn, hidden)?,
            };
            let (attention, conv) = match layer {
                LayerSpec::Attention { kind } => {
                    let rope = match kind {
                        AttentionKind::Sliding => {
                            spec.rope.sliding.as_ref().unwrap_or(&spec.rope.full)
                        }
                        AttentionKind::Full => &spec.rope.full,
                    };
                    (
                        Some(AttentionWeights {
                            query: load_matrix(
                                file,
                                &format!("{prefix}.attn_q.weight"),
                                hidden,
                                dims.q_dim(),
                            )?,
                            key: load_matrix(
                                file,
                                &format!("{prefix}.attn_k.weight"),
                                hidden,
                                kv_dim,
                            )?,
                            value: load_matrix(
                                file,
                                &format!("{prefix}.attn_v.weight"),
                                hidden,
                                kv_dim,
                            )?,
                            gate: load_matrix(
                                file,
                                &format!("{prefix}.attn_gate.weight"),
                                hidden,
                                dims.heads,
                            )?,
                            query_norm: load_vector(
                                file,
                                &format!("{prefix}.attn_q_norm.weight"),
                                dims.head_dim,
                            )?,
                            key_norm: load_vector(
                                file,
                                &format!("{prefix}.attn_k_norm.weight"),
                                dims.head_dim,
                            )?,
                            output: load_matrix(
                                file,
                                &format!("{prefix}.attn_output.weight"),
                                dims.q_dim(),
                                hidden,
                            )?,
                            rope: RopeTable::new(rope),
                        }),
                        None,
                    )
                }
                LayerSpec::Conv { ksize } => (
                    None,
                    Some(ConvWeights {
                        in_proj: load_matrix(
                            file,
                            &format!("{prefix}.shortconv.in_proj.weight"),
                            hidden,
                            3 * hidden,
                        )?,
                        template: load_matrix(
                            file,
                            &format!("{prefix}.shortconv.conv.weight"),
                            ksize,
                            hidden,
                        )?,
                        out_proj: load_matrix(
                            file,
                            &format!("{prefix}.shortconv.out_proj.weight"),
                            hidden,
                            hidden,
                        )?,
                    }),
                ),
                LayerSpec::Moe { .. } => {
                    return Err(RuntimeError::InvalidArgument(format!(
                        "layer {index} is a mixture-of-experts layer; the CPU backend only runs dense LFM3"
                    )));
                }
            };
            layers.push(LayerWeights {
                attn_norm: load_vector(file, &format!("{prefix}.attn_norm.weight"), hidden)?,
                post_attention_norm: load_vector(
                    file,
                    &format!("{prefix}.post_attention_norm.weight"),
                    hidden,
                )?,
                ffn_norm: load_vector(file, &format!("{prefix}.ffn_norm.weight"), hidden)?,
                post_ffw_norm: load_vector(
                    file,
                    &format!("{prefix}.post_ffw_norm.weight"),
                    hidden,
                )?,
                ffn,
                attention,
                conv,
            });
        }

        let scratch = Scratch {
            query: vec![0.0; dims.q_dim()],
            key: vec![0.0; kv_dim],
            value: vec![0.0; kv_dim],
            gates: vec![0.0; dims.heads],
            attended: vec![0.0; dims.q_dim()],
            up: vec![0.0; dims.ffn],
            gate: vec![0.0; dims.ffn],
            acted: vec![0.0; dims.ffn],
            conv_mix: vec![0.0; 3 * hidden],
            conv_input: vec![0.0; hidden],
            conv_output: vec![0.0; hidden],
            conv_gated: vec![0.0; hidden],
        };
        Ok(Self {
            spec: spec.clone(),
            embedding,
            output_norm,
            head,
            layers,
            scratch,
        })
    }

    fn norm_weight(&self, weights: NormWeights) -> Result<&[f32], BackendError> {
        match weights {
            NormWeights::Layer { layer, slot } => {
                let layer = self.layer(layer)?;
                Ok(match slot {
                    NormSlot::Attention => &layer.attn_norm,
                    NormSlot::PostAttention => &layer.post_attention_norm,
                    NormSlot::Ffn => &layer.ffn_norm,
                    NormSlot::PostFfn => &layer.post_ffw_norm,
                })
            }
            NormWeights::Query { layer } => Ok(&self.attention(layer)?.query_norm),
            NormWeights::Key { layer } => Ok(&self.attention(layer)?.key_norm),
            NormWeights::Output => Ok(&self.output_norm),
        }
    }

    fn layer(&self, index: usize) -> Result<&LayerWeights, BackendError> {
        self.layers
            .get(index)
            .ok_or_else(|| BackendError::Model(format!("no layer {index}")))
    }

    fn attention(&self, index: usize) -> Result<&AttentionWeights, BackendError> {
        self.layer(index)?
            .attention
            .as_ref()
            .ok_or_else(|| BackendError::Model(format!("layer {index} is not an attention layer")))
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn embed(&mut self, token: u32, out: &mut [f32]) -> Result<(), BackendError> {
        let token = token as usize;
        if token >= self.spec.dims.vocab {
            return Err(BackendError::Model(format!(
                "token {token} is outside the vocabulary of {}",
                self.spec.dims.vocab
            )));
        }
        if out.len() != self.spec.dims.hidden {
            return Err(BackendError::Shape {
                operation: "embed",
                expected: self.spec.dims.hidden,
                got: out.len(),
            });
        }
        out.copy_from_slice(self.embedding.row(token));
        Ok(())
    }

    fn rms_norm(
        &mut self,
        weights: NormWeights,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let weight = self.norm_weight(weights)?;
        if x.len() != weight.len() || out.len() != weight.len() {
            return Err(BackendError::Shape {
                operation: "rms_norm",
                expected: weight.len(),
                got: x.len(),
            });
        }
        ops::rms_norm(x, weight, self.spec.norm_eps, out);
        Ok(())
    }

    fn ffn_layer(&mut self, layer: usize, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
        let ffn = &self
            .layers
            .get(layer)
            .ok_or_else(|| BackendError::Model(format!("no layer {layer}")))?
            .ffn;
        if x.len() != ffn.up.cols() || out.len() != ffn.down.rows() {
            return Err(BackendError::Shape {
                operation: "ffn_layer",
                expected: ffn.up.cols(),
                got: x.len(),
            });
        }
        let scratch = &mut self.scratch;
        ffn.up.multiply(x, &mut scratch.up)?;
        ffn.gate.multiply(x, &mut scratch.gate)?;
        for index in 0..scratch.acted.len() {
            scratch.acted[index] = ops::silu(scratch.gate[index]) * scratch.up[index];
        }
        ffn.down.multiply(&scratch.acted, out)
    }

    fn attention_layer(
        &mut self,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        let attention = self
            .layers
            .get(call.layer)
            .and_then(|layer| layer.attention.as_ref())
            .ok_or_else(|| {
                BackendError::Model(format!("layer {} is not an attention layer", call.layer))
            })?;
        if call.input.len() != dims.hidden || out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "attention_layer",
                expected: dims.hidden,
                got: call.input.len(),
            });
        }
        if kv.len() != call.position {
            return Err(BackendError::Kv(format!(
                "cache has {} tokens before position {}, expected {}",
                kv.len(),
                call.position,
                call.position
            )));
        }
        let kv_dim = dims.kv_dim();
        if kv.kv_dim() != kv_dim {
            return Err(BackendError::Kv(format!(
                "cache width {} != model kv_dim {kv_dim}",
                kv.kv_dim()
            )));
        }
        let scratch = &mut self.scratch;
        attention.query.multiply(call.input, &mut scratch.query)?;
        attention.key.multiply(call.input, &mut scratch.key)?;
        attention.value.multiply(call.input, &mut scratch.value)?;

        let eps = self.spec.norm_eps;
        for head in 0..dims.heads {
            let head_slice = &mut scratch.query[head * dims.head_dim..(head + 1) * dims.head_dim];
            norm_one_head(head_slice, &attention.query_norm, eps);
            attention.rope.apply(head_slice, call.position);
        }
        for head in 0..dims.kv_heads {
            let head_slice = &mut scratch.key[head * dims.head_dim..(head + 1) * dims.head_dim];
            norm_one_head(head_slice, &attention.key_norm, eps);
            attention.rope.apply(head_slice, call.position);
        }
        kv.append(&scratch.key, &scratch.value)?;
        ops::attention(
            &scratch.query,
            kv,
            call.start,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            &mut scratch.attended,
        )?;
        attention.gate.multiply(call.input, &mut scratch.gates)?;
        for (head, gate) in scratch.gates.iter().enumerate() {
            let gate = ops::sigmoid(*gate);
            let head_slice =
                &mut scratch.attended[head * dims.head_dim..(head + 1) * dims.head_dim];
            for value in head_slice.iter_mut() {
                *value *= gate;
            }
        }
        attention.output.multiply(&scratch.attended, out)
    }

    fn conv_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        let conv = self
            .layers
            .get(layer)
            .and_then(|layer| layer.conv.as_ref())
            .ok_or_else(|| BackendError::Model(format!("layer {layer} is not a conv layer")))?;
        let ksize = conv.template.cols();
        let expected_state = (ksize - 1) * dims.hidden;
        if x.len() != dims.hidden || out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "conv_layer",
                expected: dims.hidden,
                got: x.len(),
            });
        }
        if state.len() != expected_state {
            return Err(BackendError::Shape {
                operation: "conv_layer state",
                expected: expected_state,
                got: state.len(),
            });
        }
        let scratch = &mut self.scratch;
        conv.in_proj.multiply(x, &mut scratch.conv_mix)?;
        let hidden = dims.hidden;
        let (mix_b, mix_c, mix_x) = {
            let mix = &scratch.conv_mix;
            (&mix[..hidden], &mix[hidden..2 * hidden], &mix[2 * hidden..])
        };
        for index in 0..hidden {
            scratch.conv_input[index] = mix_b[index] * mix_x[index];
        }
        for channel in 0..hidden {
            let template = conv.template.row(channel);
            let mut accumulator = template[ksize - 1] * scratch.conv_input[channel];
            for (lag, weight) in template.iter().take(ksize - 1).enumerate() {
                accumulator += weight * state[lag * hidden + channel];
            }
            scratch.conv_output[channel] = accumulator;
        }
        if ksize > 2 {
            state.copy_within(hidden.., 0);
        }
        state[(ksize - 2) * hidden..].copy_from_slice(&scratch.conv_input);
        for ((gated, c), value) in scratch
            .conv_gated
            .iter_mut()
            .zip(mix_c)
            .zip(&scratch.conv_output)
        {
            *gated = *c * *value;
        }
        conv.out_proj.multiply(&scratch.conv_gated, out)
    }

    fn lm_head(&mut self, x: &[f32], logits: &mut [f32]) -> Result<(), BackendError> {
        let matrix = match &self.head {
            Head::Tied => &self.embedding,
            Head::Separate(matrix) => matrix,
        };
        matrix.multiply(x, logits)
    }
}

/// RMSNorm of one head vector, used for q/k norm before rope.
fn norm_one_head(vector: &mut [f32], weight: &[f32], eps: f32) {
    let sum: f64 = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let mean = sum / vector.len() as f64;
    let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;
    for (value, weight) in vector.iter_mut().zip(weight) {
        *value = *value * scale * weight;
    }
}

fn tensor<'a>(file: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, RuntimeError> {
    file.tensor(name)
        .ok_or_else(|| RuntimeError::MissingTensor(name.to_owned()))
}

fn load_matrix(
    file: &GgufFile,
    name: &str,
    cols: usize,
    rows: usize,
) -> Result<Matrix, RuntimeError> {
    let info = tensor(file, name)?;
    let expected = vec![cols, rows];
    if info.dims.len() != 2 || info.dims != [cols as u64, rows as u64] {
        return Err(RuntimeError::TensorShape {
            name: name.to_owned(),
            expected,
            found: info.dims.clone(),
        });
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::Gguf(format!("tensor {name} size overflows usize")))?;
    let bytes = file
        .tensor_bytes(name)
        .map_err(|error| RuntimeError::Gguf(error.to_string()))?;
    let data = dequantize(info, bytes, count)?;
    Ok(Matrix { rows, cols, data })
}

fn load_vector(file: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, RuntimeError> {
    let info = tensor(file, name)?;
    let expected = vec![len];
    if info.dims.len() != 1 || info.dims[0] != len as u64 {
        return Err(RuntimeError::TensorShape {
            name: name.to_owned(),
            expected,
            found: info.dims.clone(),
        });
    }
    let bytes = file
        .tensor_bytes(name)
        .map_err(|error| RuntimeError::Gguf(error.to_string()))?;
    dequantize(info, bytes, len)
}

fn dequantize(info: &TensorInfo, bytes: &[u8], count: usize) -> Result<Vec<f32>, RuntimeError> {
    let values = match info.ggml_type {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>(),
        GgmlType::Q4_K => lfm_quant::dequant_row_q4_k(bytes, count)
            .map_err(|error| RuntimeError::Gguf(format!("{}: {error}", info.name)))?,
        GgmlType::Q6_K => lfm_quant::dequant_row_q6_k(bytes, count)
            .map_err(|error| RuntimeError::Gguf(format!("{}: {error}", info.name)))?,
        GgmlType::Q8_0 => {
            if count % 32 != 0 {
                return Err(RuntimeError::Gguf(format!(
                    "{}: Q8_0 length {count} is not a multiple of 32",
                    info.name
                )));
            }
            let mut out = Vec::with_capacity(count);
            for block in bytes.chunks_exact(34) {
                let mut values = [0u8; 34];
                values.copy_from_slice(block);
                out.extend_from_slice(&lfm_quant::dequant_q8_0(&values));
            }
            out
        }
        other => {
            return Err(RuntimeError::UnsupportedTensorType {
                name: info.name.clone(),
                ggml_type: format!("{other:?}"),
            });
        }
    };
    if values.len() != count {
        return Err(RuntimeError::Gguf(format!(
            "{}: dequantized {} values, expected {count}",
            info.name,
            values.len()
        )));
    }
    Ok(values)
}
