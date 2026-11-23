//! Host CPU backend.
//!
//! Loads a GGUF checkpoint, dequantizes the tensors LFM3 needs to f32 and runs
//! the reference math with [`crate::ops`]. This is the numerical oracle for the
//! DSP port: weights are the same values the device will hold, only the storage
//! layout differs.

use std::sync::Mutex;

use lfm_gguf::{GgufFile, TensorInfo};

use crate::backend::{AttentionCall, Backend, BackendError, NormSlot, NormWeights};
use crate::dequant;
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
    /// Hadamard block the rows are rotated by (`0` = natural basis). See
    /// [`WeightFormat`]; inputs are rotated by the same block before the dot.
    had: usize,
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
        Ok(Self {
            rows,
            cols,
            data,
            had: 0,
        })
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
        if self.had == 0 {
            ops::matvec(&self.data, self.rows, self.cols, x, out);
        } else {
            // `W' = W H`, `x' = H x` and `H H = I`, so `W' x' = W x`.
            let mut rotated = x.to_vec();
            hadamard_blocks(&mut rotated, self.had);
            ops::matvec(&self.data, self.rows, self.cols, &rotated, out);
        }
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
        let opts = weight_opts();
        *WEIGHT_STATS.lock().expect("weight stats") = WeightStats::default();
        if opts.active() {
            eprintln!(
                "cpu experiment: format={:?} had={} conv16={} head_f32={} kvk={:?} kvv={:?}",
                opts.format, opts.had, opts.conv16, opts.head_f32, opts.k_format, opts.v_format
            );
        }
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
        report_weight_stats(dims.hidden);
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
        if self.embedding.had != 0 {
            // Rows are stored rotated (`e H`); the embedding lookup returns the
            // natural vector, so undo the rotation for this one row.
            hadamard_blocks(out, self.embedding.had);
        }
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
        let mut x8 = x.to_vec();
        maybe_quantize_i8(&mut x8);
        ffn.up.multiply(&x8, &mut scratch.up)?;
        ffn.gate.multiply(&x8, &mut scratch.gate)?;
        for index in 0..scratch.acted.len() {
            scratch.acted[index] = ops::silu(scratch.gate[index]) * scratch.up[index];
        }
        maybe_quantize_i8(&mut scratch.acted);
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
        let kv_opts = weight_opts();
        if let Some(levels) = kv_levels(kv_opts.k_format) {
            // DSP `kvpage`: int4 per head per token.
            quantize_symmetric(&mut scratch.key, dims.head_dim, levels);
        }
        if let Some(levels) = kv_levels(kv_opts.v_format) {
            // DSP `kvpage` stores V int8 with a scale constant over tokens;
            // the CPU control uses one scale per token vector instead.
            quantize_symmetric(&mut scratch.value, kv_dim, levels);
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

/// `MYELIN_CPU_INT4=1` rounds every weight matrix to per-row int4 after
/// dequantisation, i.e. exactly what `myelin-pack` stores for the DSP. It is
/// the control experiment for "is the remaining error the DSP path or the
/// 4-bit weight format itself".
/// `MYELIN_CPU_INT8_ACT=1` rounds activations to int8 the way the DSP does
/// (one scale per vector, absmax/110), so the host can reproduce the device's
/// intermediate quantisation without the device.
fn maybe_quantize_i8(values: &mut [f32]) {
    let group: usize = match std::env::var("MYELIN_CPU_ACT_GROUP")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        Some(group) if group > 0 => group,
        _ if std::env::var_os("MYELIN_CPU_INT8_ACT").is_some() => values.len(),
        _ => return,
    };
    for chunk in values.chunks_mut(group) {
        let max = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if max <= 0.0 {
            continue;
        }
        let step = max / 110.0;
        for value in chunk.iter_mut() {
            *value = (*value / step).round().clamp(-127.0, 127.0) * step;
        }
    }
}

/// Temporary quantisation experiments for the int4 quality question.
///
/// All of these are read once from the environment when the checkpoint is
/// loaded and are **not** part of the production format. The controls are:
///
/// * `MYELIN_CPU_INT4=1` — symmetric int4 (absmax/7) per row, the format the
///   DSP currently gets (`myelin-pack` `gemv`), which reproduces the loop.
/// * `MYELIN_CPU_WBLOCK=32|64|128|256|row` — same int4 but one scale per block
///   of that many inputs instead of one per row. Implies int4.
/// * `MYELIN_CPU_WASYM=1` — asymmetric int4 with a min term in the same block
///   (`w = step*q + min`, `q` in `0..=15`); the CPU model of a Q4_K-style
///   min term. Combines with `WBLOCK`.
/// * `MYELIN_CPU_WHAD=32|128` — rotate each weight row and its input by a
///   normalised block Hadamard before quantising (`W H`, `H x`).
/// * `MYELIN_CPU_Q4K=1` — leave the GGUF values alone (this checkpoint is
///   Q4_K: per-256 blocks with a min term and f32 combine). Upper bound.
/// * `MYELIN_CPU_KVK=f32|int8|int4` / `MYELIN_CPU_KVV=f32|int8|int4` —
///   store K/V quantised at append time (int4 = absmax/7, int8 = absmax/127).
///   K uses one scale per `head_dim` block, V one per token vector, which is
///   the CPU analogue of the DSP `kvpage` layout, not bit-identical.
/// * `MYELIN_CPU_CONV16=1` — shortconv matrices use int16 per row instead of
///   int4 (CPU analogue of the DSP `CONV16`/Y16 path, not bit-identical).
/// * `MYELIN_CPU_WHF32=1` — leave the tied embedding/head matrix unquantised,
///   to separate "body format" from "logits format" in the diagnosis.
/// * `MYELIN_CPU_WSTATS=1` — always print the weight cosine for the chosen
///   format, even when it is the native one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightFormat {
    /// Values as stored in the GGUF (native Q4_K for the shipped checkpoint).
    Native,
    /// Symmetric int4, `block` values share one `absmax/7` scale (`0` = row).
    Int4 { block: usize },
    /// Asymmetric int4 with a min term: `w = step * q + min`, `q` in `0..=15`.
    /// One `(step, min)` pair per `block` values (`0` = row).
    Asym4 { block: usize },
    /// Symmetric int16, one `absmax/32767` scale per row.
    Int16Row,
}

impl WeightFormat {
    fn bytes_per_weight(self, cols: usize) -> f64 {
        let per_group = |block: usize| {
            let block = if block == 0 { cols } else { block };
            block as f64
        };
        match self {
            Self::Native => 144.0 / 256.0,
            Self::Int4 { block } => 0.5 + 4.0 / per_group(block),
            // One f32 scale and one f32 min per group.
            Self::Asym4 { block } => 0.5 + 8.0 / per_group(block),
            Self::Int16Row => 2.0 + 4.0 / cols as f64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvFormat {
    F32,
    Int8,
    Int4,
}

impl KvFormat {
    fn parse(name: &str, value: &str) -> Self {
        match value {
            "f32" => Self::F32,
            "int8" | "i8" => Self::Int8,
            "int4" | "i4" => Self::Int4,
            other => {
                eprintln!("warning: {name}={other} is not f32|int8|int4, ignored");
                Self::F32
            }
        }
    }

    fn active(self) -> bool {
        self != Self::F32
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightOpts {
    format: WeightFormat,
    /// Hadamard block for weights and activations (`0` = off).
    had: usize,
    conv16: bool,
    /// Leave the tied embedding / output head in the GGUF precision.
    head_f32: bool,
    k_format: KvFormat,
    v_format: KvFormat,
    stats: bool,
}

impl WeightOpts {
    fn from_env() -> Self {
        let flag = |name: &str| std::env::var_os(name).is_some();
        let q4k = flag("MYELIN_CPU_Q4K");
        let int4 = flag("MYELIN_CPU_INT4");
        let block = match std::env::var("MYELIN_CPU_WBLOCK").ok().as_deref() {
            Some("row") | Some("ROW") => Some(0),
            Some(value) => match value.parse::<usize>() {
                Ok(block) if block > 0 => Some(block),
                _ => {
                    eprintln!(
                        "warning: MYELIN_CPU_WBLOCK={value} is not 32|64|128|256|row, ignored"
                    );
                    None
                }
            },
            None => None,
        };
        let format = if q4k {
            WeightFormat::Native
        } else if flag("MYELIN_CPU_WASYM") {
            WeightFormat::Asym4 {
                block: block.unwrap_or(0),
            }
        } else if let Some(block) = block {
            WeightFormat::Int4 { block }
        } else if int4 {
            WeightFormat::Int4 { block: 0 }
        } else {
            WeightFormat::Native
        };
        let had = match std::env::var("MYELIN_CPU_WHAD").ok().as_deref() {
            Some("0") | None => 0,
            Some("32") => 32,
            Some("128") => 128,
            Some(other) => {
                eprintln!("warning: MYELIN_CPU_WHAD={other} is not 0|32|128, ignored");
                0
            }
        };
        let kv = |name: &str| {
            std::env::var(name)
                .map(|value| KvFormat::parse(name, &value))
                .unwrap_or(KvFormat::F32)
        };
        Self {
            format,
            had,
            conv16: flag("MYELIN_CPU_CONV16"),
            head_f32: flag("MYELIN_CPU_WHF32"),
            k_format: kv("MYELIN_CPU_KVK"),
            v_format: kv("MYELIN_CPU_KVV"),
            stats: flag("MYELIN_CPU_WSTATS"),
        }
    }

    fn active(&self) -> bool {
        self.format != WeightFormat::Native
            || self.had != 0
            || self.k_format.active()
            || self.v_format.active()
            || self.stats
    }
}

fn weight_opts() -> WeightOpts {
    static OPTS: std::sync::OnceLock<WeightOpts> = std::sync::OnceLock::new();
    *OPTS.get_or_init(WeightOpts::from_env)
}

#[derive(Debug, Clone, Copy, Default)]
struct WeightStats {
    dot: f64,
    orig_sq: f64,
    dec_sq: f64,
    err_sq: f64,
    max_rel: f64,
    values: usize,
}

impl WeightStats {
    fn add(&mut self, other: Self) {
        self.dot += other.dot;
        self.orig_sq += other.orig_sq;
        self.dec_sq += other.dec_sq;
        self.err_sq += other.err_sq;
        self.max_rel = self.max_rel.max(other.max_rel);
        self.values += other.values;
    }

    fn cosine(&self) -> f64 {
        if self.orig_sq == 0.0 || self.dec_sq == 0.0 {
            return 1.0;
        }
        self.dot / (self.orig_sq.sqrt() * self.dec_sq.sqrt())
    }

    /// `||w - w_hat|| / ||w||`: unlike cosine this is not dominated by the
    /// few largest weights, which is what exposes a too-coarse per-row scale.
    fn relative_rmse(&self) -> f64 {
        if self.orig_sq == 0.0 {
            return 0.0;
        }
        (self.err_sq / self.orig_sq).sqrt()
    }
}

static WEIGHT_STATS: Mutex<WeightStats> = Mutex::new(WeightStats {
    dot: 0.0,
    orig_sq: 0.0,
    dec_sq: 0.0,
    err_sq: 0.0,
    max_rel: 0.0,
    values: 0,
});

/// Normalised block Hadamard transform (fast Walsh-Hadamard) in place.
///
/// `block` must be a power of two and divide `values.len()`.
fn hadamard_blocks(values: &mut [f32], block: usize) {
    if block <= 1 {
        return;
    }
    debug_assert_eq!(values.len() % block, 0);
    let scale = 1.0 / (block as f32).sqrt();
    for chunk in values.chunks_mut(block) {
        let mut span = 1;
        while span < block {
            let mut base = 0;
            while base < block {
                for index in base..base + span {
                    let a = chunk[index];
                    let b = chunk[index + span];
                    chunk[index] = a + b;
                    chunk[index + span] = a - b;
                }
                base += 2 * span;
            }
            span *= 2;
        }
        for value in chunk.iter_mut() {
            *value *= scale;
        }
    }
}

/// Quantise one row to fixed point and measure the round trip.
///
/// Symmetric uses `step = max|w| / levels` and codes in `[-levels-1, levels]`
/// (7 or 32767, matching the existing `absmax/7` control and the DSP
/// offset-8 code range `[-8, 7]`). Asymmetric uses `step = (max - min) / 15`
/// and codes in `0..=15` decoded as `step*q + min` — the min term Q4_K has.
fn quantize_row_blocks(values: &mut [f32], block: usize, levels: f32, asym: bool) -> WeightStats {
    let mut stats = WeightStats::default();
    for chunk in values.chunks_mut(block) {
        let (low, step) = if asym {
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for value in chunk.iter() {
                low = low.min(*value);
                high = high.max(*value);
            }
            let span = high - low;
            if !span.is_finite() || span <= 0.0 {
                for value in chunk.iter_mut() {
                    *value = low;
                }
                continue;
            }
            (low, span / 15.0)
        } else {
            let max = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if max <= 0.0 {
                continue;
            }
            (0.0, max / levels)
        };
        for value in chunk.iter_mut() {
            let original = *value as f64;
            let quantized = if asym {
                ((*value - low) / step).round().clamp(0.0, 15.0) * step + low
            } else {
                (*value / step).round().clamp(-levels - 1.0, levels) * step
            };
            *value = quantized;
            let decoded = quantized as f64;
            stats.dot += original * decoded;
            stats.orig_sq += original * original;
            stats.dec_sq += decoded * decoded;
            stats.err_sq += (original - decoded) * (original - decoded);
            let relative = (original - decoded).abs() / step as f64;
            stats.max_rel = stats.max_rel.max(relative);
            stats.values += 1;
        }
    }
    stats
}

/// Symmetric fixed-point round trip of one vector, one `absmax/levels` scale
/// per `group` values. Used for the KV-store experiment.
fn quantize_symmetric(values: &mut [f32], group: usize, levels: f32) {
    for chunk in values.chunks_mut(group) {
        let max = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if max <= 0.0 {
            continue;
        }
        let step = max / levels;
        for value in chunk.iter_mut() {
            *value = (*value / step).round().clamp(-levels, levels) * step;
        }
    }
}

fn kv_levels(format: KvFormat) -> Option<f32> {
    match format {
        KvFormat::F32 => None,
        KvFormat::Int8 => Some(127.0),
        KvFormat::Int4 => Some(7.0),
    }
}

/// Apply the selected experimental format to one dequantised matrix.
///
/// Returns `None` for the untouched native path.
struct Prepared {
    stats: WeightStats,
    had: usize,
}

fn prepare_weights(name: &str, data: &mut [f32], rows: usize, cols: usize) -> Option<Prepared> {
    let opts = weight_opts();
    if opts.head_f32 && (name == "token_embd.weight" || name == "output.weight") {
        return None;
    }
    let mut format = opts.format;
    if opts.conv16 && name.contains(".shortconv.") {
        format = WeightFormat::Int16Row;
    }
    if format == WeightFormat::Native && opts.had == 0 {
        return None;
    }
    // A transform that does not tile the row is skipped for that matrix
    // (only the tiny `shortconv.conv` template hits this).
    let had = if opts.had != 0 && cols % opts.had == 0 {
        opts.had
    } else {
        0
    };
    let block = match format {
        WeightFormat::Int4 { block } | WeightFormat::Asym4 { block }
            if block != 0 && cols % block == 0 =>
        {
            block
        }
        WeightFormat::Int4 { .. } | WeightFormat::Asym4 { .. } => cols,
        WeightFormat::Int16Row => cols,
        WeightFormat::Native => cols,
    };
    let (levels, asym) = match format {
        WeightFormat::Int4 { .. } => (7.0, false),
        WeightFormat::Asym4 { .. } => (15.0, true),
        WeightFormat::Int16Row => (32_767.0, false),
        WeightFormat::Native => (0.0, false),
    };
    let mut stats = WeightStats::default();
    for row in 0..rows {
        let values = &mut data[row * cols..(row + 1) * cols];
        if had != 0 {
            hadamard_blocks(values, had);
        }
        if format != WeightFormat::Native {
            stats.add(quantize_row_blocks(values, block, levels, asym));
        }
    }
    Some(Prepared { stats, had })
}

fn report_weight_stats(hidden: usize) {
    let opts = weight_opts();
    if !opts.active() {
        return;
    }
    let stats = *WEIGHT_STATS.lock().expect("weight stats");
    let format = if opts.conv16 {
        format!("{:?}+conv16", opts.format)
    } else {
        format!("{:?}", opts.format)
    };
    eprintln!(
        "wstats: format={format} had={} cos={:.5} relrmse={:.4} maxrel={:.3} values={} bytes/weight={:.4}",
        opts.had,
        stats.cosine(),
        stats.relative_rmse(),
        stats.max_rel,
        stats.values,
        opts.format.bytes_per_weight(hidden),
    );
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
    let mut data = dequantize(info, bytes, count)?;
    let mut had = 0;
    if let Some(prepared) = prepare_weights(name, &mut data, rows, cols) {
        had = prepared.had;
        WEIGHT_STATS
            .lock()
            .expect("weight stats")
            .add(prepared.stats);
    }
    Ok(Matrix {
        rows,
        cols,
        data,
        had,
    })
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
    dequant::dequantize_row(info.ggml_type, &info.name, bytes, count).map_err(|message| match info
        .ggml_type
        .type_size()
    {
        Some(_) => RuntimeError::Gguf(message),
        None => RuntimeError::UnsupportedTensorType {
            name: info.name.clone(),
            ggml_type: format!("{:?}", info.ggml_type),
        },
    })
}

#[cfg(test)]
mod experiment_tests {
    use super::*;

    fn deterministic_row(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let original = deterministic_row(64, 7);
        let mut values = original.clone();
        hadamard_blocks(&mut values, 32);
        hadamard_blocks(&mut values, 32);
        for (a, b) in original.iter().zip(&values) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn asymmetric_codes_span_min_to_max() {
        let mut values = vec![-3.0, -1.0, 0.5, 2.0, 4.0];
        let len = values.len();
        let stats = quantize_row_blocks(&mut values, len, 15.0, true);
        assert_eq!(stats.values, 5);
        assert!(values[0] >= -3.0 - 1e-5);
        assert!(values[4] <= 4.0 + 1e-5);
        assert!(stats.relative_rmse() < 0.11);
    }

    #[test]
    fn block_scale_beats_row_scale_on_outliers() {
        // One large outlier sets a per-row absmax that is far too coarse for
        // the quiet values; a 32-wide block keeps them in their own range.
        let mut row = deterministic_row(256, 11);
        row[100] = 40.0;
        let mut per_row = row.clone();
        let mut blocked = row.clone();
        let row_stats = quantize_row_blocks(&mut per_row, 256, 7.0, false);
        let block_stats = quantize_row_blocks(&mut blocked, 32, 7.0, false);
        assert!(
            block_stats.relative_rmse() < row_stats.relative_rmse(),
            "block {} >= row {}",
            block_stats.relative_rmse(),
            row_stats.relative_rmse()
        );
    }
}
