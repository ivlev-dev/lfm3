//! On-disk model pack: everything the DSP needs for a full LFM3 generation.
//!
//! The pack is produced on the host by `examples/make_model_pack.rs` from a
//! GGUF checkpoint and the CPU oracle's calibration pass, then pushed to the
//! phone. It contains:
//!
//! * one int4 GEMV blob + manifest per transformer matrix (24 FFN gates/ups/
//!   downs, 11 attention q/k/v/o/gates, 13 conv in/out projections);
//! * one quantised shortconv blob per conv layer;
//! * the tied vocabulary projection as an int4 GEMV;
//! * the token embedding in its checkpoint storage (`token_embd` is Q6_K);
//! * `pack.json`: shapes, rope, norm epsilon and every calibrated activation
//!   scale.
//!
//! [`ModelPack::weight_plan`] turns the description into the resident plan
//! [`crate::ResidentWeights`] executes: it allocates K/KT/V images for the
//! requested context and refuses a plan that does not fit the ION budget
//! before anything reaches the device.

use std::fmt;
use std::path::{Path, PathBuf};

use lfm_gguf::GgmlType;
use myelin_core::backend::dsp::VQuant;
use myelin_core::backend::{EmbeddingRows, HostWeights, LayerNorms};
use myelin_core::spec::{
    AttentionKind, ChatTemplateSpec, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec, Quant,
    RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
};
use myelin_pack::gemv::GemvManifest;
use serde::{Deserialize, Serialize};

use crate::gemv_weights::PackedGemv;
use crate::segment::Segments;
use crate::weights::{
    AttentionFullPlan, AttentionParams, AttentionPlan, ConvPlan, FfnPlan, LmHeadPlan, PackedMatrix,
    ResidentImage, WeightBudget, WeightError, WeightPlan,
};

/// Format version this writer emits.
///
/// * 1: per-row int4 scales only.
/// * 2: adds `<blob>.seg` per-256 ratio sidecars (`PackGemv::seg_scales`).
///   Version 1 packs still load and keep the per-row kernel path.
pub const PACK_VERSION: u32 = 2;

/// Oldest pack version this reader understands.
pub const MIN_PACK_VERSION: u32 = 1;

/// Failure while reading or planning a model pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPackError {
    /// A file is missing or unreadable.
    Io(String),
    /// `pack.json` or a blob manifest is inconsistent.
    Layout(String),
    /// The plan does not fit the ION budget or a stage is missing.
    Weights(WeightError),
}

impl fmt::Display for ModelPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "model pack: {message}"),
            Self::Layout(message) => write!(f, "model pack: {message}"),
            Self::Weights(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ModelPackError {}

impl From<WeightError> for ModelPackError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

/// One rope configuration as written by the packer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackRope {
    /// `"full"` or `"partial"`.
    pub kind: String,
    pub theta: f32,
    pub dim: usize,
    /// Rotated pairs for a partial rope (`0` for full).
    pub rotated_dim: usize,
}

impl PackRope {
    /// Convert to the runtime description.
    pub fn spec(&self) -> RopeSpec {
        RopeSpec {
            kind: if self.kind == "partial" {
                RopeKind::Partial {
                    rotated_dim: self.rotated_dim,
                }
            } else {
                RopeKind::Full
            },
            theta: self.theta,
            dim: self.dim,
        }
    }
}

/// Transformer geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackDims {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub conv_cache: usize,
    pub sliding_window: usize,
}

/// One int4 matrix with the calibrated scales of the GEMV that reads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackGemv {
    /// File name inside the pack directory.
    pub file: String,
    /// `s_in`: real units per int8 activation code.
    pub input_scale: f32,
    /// `s_out`: real units per int8 result code.
    pub output_scale: f32,
    /// Per-256 ratio sidecar (`<file>.seg`) written by pack version 2.
    /// Absent in version 1 packs, which stay on the per-row path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seg_scales: Option<String>,
}

/// Feed-forward stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackFfn {
    pub gate: PackGemv,
    pub up: PackGemv,
    pub down: PackGemv,
    /// `s_mid`: real units per int8 code of `silu(gate) * up`.
    pub mid_scale: f32,
    pub reps: i32,
}

/// Short-convolution stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackConv {
    pub in_proj: PackGemv,
    /// `myelin-pack shortconv` blob.
    pub conv_file: String,
    pub out_proj: PackGemv,
    /// `s_bc`: real units per int8 code of the `bcx` buffer.
    pub mix_scale: f32,
    /// `s_y`: real units per int8 code of the conv output.
    pub y_scale: f32,
    pub ksh: i32,
    pub reps: i32,
}

/// Attention stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackAttention {
    pub query: PackGemv,
    pub key: PackGemv,
    pub value: PackGemv,
    pub output: PackGemv,
    pub gate: Option<PackGemv>,
    pub query_norm: Vec<f32>,
    pub key_norm: Vec<f32>,
    /// Q held in int8 for the paged-attention kernel.
    pub attn_q_scale: f32,
    /// Paged-attention output is Q12: `real = q12 / 4096`.
    pub attn_out_scale: f32,
    /// `s_attended`: real units per int8 code of the gated attention output.
    pub o_input_scale: f32,
    /// Per-channel V quantisation steps (`kv_heads * head_dim` values).
    pub v_channel: Vec<f32>,
    pub reps: i32,
}

/// One decoder layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackLayer {
    /// `"conv"` or `"attention"`.
    pub kind: String,
    /// `"full"` or `"sliding"` for an attention layer.
    pub attention_kind: Option<String>,
    pub ffn: PackFfn,
    pub conv: Option<PackConv>,
    pub attention: Option<PackAttention>,
}

/// Token embedding stored in its checkpoint format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackEmbed {
    pub file: String,
    /// `"Q6_K"`, `"F16"`, ... (GGML type name).
    pub dtype: String,
    pub rows: usize,
    pub cols: usize,
}

/// Tied vocabulary projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackLmHead {
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seg_scales: Option<String>,
    pub input_scale: f32,
    pub output_scale: f32,
    pub reps: i32,
}

/// Learned norms of one layer as stored in the pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackNorms {
    pub attention: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub ffn: Vec<f32>,
    pub post_ffn: Vec<f32>,
}

/// Root of `pack.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPack {
    pub version: u32,
    /// Source checkpoint, for diagnostics.
    pub model: String,
    pub dims: PackDims,
    pub norm_eps: f32,
    pub tied_embeddings: bool,
    pub rope_full: PackRope,
    pub rope_sliding: Option<PackRope>,
    pub layers: Vec<PackLayer>,
    pub embed: PackEmbed,
    pub lm_head: PackLmHead,
    /// The four RMSNorm vectors of every layer.
    pub norms: Vec<PackNorms>,
    /// Final `output_norm.weight`.
    pub output_norm: Vec<f32>,
}

impl ModelPack {
    /// Read `pack.json` from `dir`.
    pub fn read(dir: &Path) -> Result<Self, ModelPackError> {
        let path = dir.join("pack.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| ModelPackError::Io(format!("{}: {error}", path.display())))?;
        let pack: Self = serde_json::from_str(&text)
            .map_err(|error| ModelPackError::Layout(format!("{}: {error}", path.display())))?;
        pack.check()?;
        Ok(pack)
    }

    /// Write the manifest (used by the host packer and its tests).
    pub fn write(&self, dir: &Path) -> Result<(), ModelPackError> {
        std::fs::create_dir_all(dir)
            .map_err(|error| ModelPackError::Io(format!("{}: {error}", dir.display())))?;
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| ModelPackError::Layout(error.to_string()))?;
        std::fs::write(dir.join("pack.json"), text)
            .map_err(|error| ModelPackError::Io(format!("pack.json: {error}")))
    }

    /// Validate the cross-field invariants a plan depends on.
    pub fn check(&self) -> Result<(), ModelPackError> {
        if !(MIN_PACK_VERSION..=PACK_VERSION).contains(&self.version) {
            return Err(ModelPackError::Layout(format!(
                "pack version {} is not {MIN_PACK_VERSION}..={PACK_VERSION}",
                self.version
            )));
        }
        if self.layers.len() != self.dims.layers {
            return Err(ModelPackError::Layout(format!(
                "{} layers in the pack, {} in the dims",
                self.layers.len(),
                self.dims.layers
            )));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            match layer.kind.as_str() {
                "conv" => {
                    if layer.conv.is_none() || layer.attention.is_some() {
                        return Err(ModelPackError::Layout(format!(
                            "layer {index} is `conv` without a conv stage"
                        )));
                    }
                }
                "attention" => {
                    if layer.attention.is_none() || layer.conv.is_some() {
                        return Err(ModelPackError::Layout(format!(
                            "layer {index} is `attention` without an attention stage"
                        )));
                    }
                }
                other => {
                    return Err(ModelPackError::Layout(format!(
                        "layer {index} has unknown kind `{other}`"
                    )));
                }
            }
        }
        if self.embed.rows != self.dims.vocab || self.embed.cols != self.dims.hidden {
            return Err(ModelPackError::Layout(format!(
                "embedding is {}x{}, dims say {}x{}",
                self.embed.rows, self.embed.cols, self.dims.vocab, self.dims.hidden
            )));
        }
        if self.norms.len() != self.dims.layers {
            return Err(ModelPackError::Layout(format!(
                "{} norm blocks for {} layers",
                self.norms.len(),
                self.dims.layers
            )));
        }
        for (index, norms) in self.norms.iter().enumerate() {
            for (name, values) in [
                ("attn_norm", &norms.attention),
                ("post_attention_norm", &norms.post_attention),
                ("ffn_norm", &norms.ffn),
                ("post_ffw_norm", &norms.post_ffn),
            ] {
                if values.len() != self.dims.hidden {
                    return Err(ModelPackError::Layout(format!(
                        "layer {index} {name} has {} values, expected {}",
                        values.len(),
                        self.dims.hidden
                    )));
                }
            }
        }
        if self.output_norm.len() != self.dims.hidden {
            return Err(ModelPackError::Layout(format!(
                "output_norm has {} values, expected {}",
                self.output_norm.len(),
                self.dims.hidden
            )));
        }
        Ok(())
    }

    /// Rebuild the runtime spec.
    pub fn spec(&self) -> ModelSpec {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                if layer.kind == "conv" {
                    LayerSpec::Conv {
                        ksize: self.dims.conv_cache,
                    }
                } else {
                    LayerSpec::Attention {
                        kind: if layer.attention_kind.as_deref() == Some("sliding") {
                            AttentionKind::Sliding
                        } else {
                            AttentionKind::Full
                        },
                    }
                }
            })
            .collect();
        ModelSpec {
            family: ModelFamily::Lfm3,
            dims: Dims {
                hidden: self.dims.hidden,
                ffn: self.dims.ffn,
                layers: self.dims.layers,
                heads: self.dims.heads,
                kv_heads: self.dims.kv_heads,
                head_dim: self.dims.head_dim,
                vocab: self.dims.vocab,
                conv_cache: self.dims.conv_cache,
                sliding_window: self.dims.sliding_window,
            },
            rope: RopeConfig {
                full: self.rope_full.spec(),
                sliding: self.rope_sliding.as_ref().map(PackRope::spec),
            },
            layers,
            kv_format: KvFormat::F16,
            tokenizer: TokenizerSpec {
                model: String::new(),
                vocab_size: self.dims.vocab,
                bos_token_id: None,
                eos_token_id: None,
            },
            chat_template: ChatTemplateSpec {
                template: String::new(),
            },
            weight_format: WeightFormat::from_quant(Quant::Q4K),
            norm_eps: self.norm_eps,
            tied_embeddings: self.tied_embeddings,
        }
    }

    /// Path of a pack file.
    pub fn path(&self, dir: &Path, file: &str) -> PathBuf {
        dir.join(file)
    }

    /// Read the token embedding bytes.
    pub fn embed_bytes(&self, dir: &Path) -> Result<Vec<u8>, ModelPackError> {
        let path = self.path(dir, &self.embed.file);
        std::fs::read(&path)
            .map_err(|error| ModelPackError::Io(format!("{}: {error}", path.display())))
    }

    /// Build the host tensors: stored embedding, norms and output norm.
    pub fn host_weights(&self, dir: &Path) -> Result<HostWeights, ModelPackError> {
        let kind = ggml_type(&self.embed.dtype).ok_or_else(|| {
            ModelPackError::Layout(format!("unknown embedding dtype `{}`", self.embed.dtype))
        })?;
        let embedding = EmbeddingRows::new(
            kind,
            self.embed.rows,
            self.embed.cols,
            self.embed_bytes(dir)?,
        )
        .map_err(|error| ModelPackError::Layout(error.to_string()))?;
        let layers = self
            .norms
            .iter()
            .map(|norms| LayerNorms {
                attention: norms.attention.clone(),
                post_attention: norms.post_attention.clone(),
                ffn: norms.ffn.clone(),
                post_ffn: norms.post_ffn.clone(),
            })
            .collect();
        HostWeights::new(&self.spec(), embedding, layers, self.output_norm.clone())
            .map_err(|error| ModelPackError::Layout(error.to_string()))
    }

    /// Build the full resident plan for `max_context` tokens.
    ///
    /// Refuses a plan larger than `budget` before any allocation. The KV
    /// images are sized for the requested context; `cache_pages` follows the
    /// 32-token page granularity of the DSP kernel.
    pub fn weight_plan(
        &self,
        dir: &Path,
        max_context: usize,
        budget: WeightBudget,
    ) -> Result<WeightPlan, ModelPackError> {
        if max_context == 0 {
            return Err(ModelPackError::Layout(
                "max_context must be positive".to_owned(),
            ));
        }
        let cache_pages = max_context.div_ceil(myelin_core::kv::PAGE_TOKENS).max(1);
        let slots = cache_pages * myelin_core::kv::PAGE_TOKENS;
        let kv_dim = self.dims.kv_heads * self.dims.head_dim;
        let mut builder = WeightPlan::builder(budget);

        for (index, layer) in self.layers.iter().enumerate() {
            builder = builder.ffn(index, self.ffn_plan(dir, &layer.ffn)?);
            match (layer.kind.as_str(), &layer.conv, &layer.attention) {
                ("conv", Some(conv), None) => {
                    builder = builder.conv(index, self.conv_plan(dir, conv, index)?);
                }
                ("attention", None, Some(attention)) => {
                    let caches = AttentionPlan::new(
                        zero_image(&format!("blk.{index}.k"), self.dims.kv_heads * slots * 128)?,
                        zero_image(&format!("blk.{index}.kt"), self.dims.kv_heads * slots * 4)?,
                        zero_image(&format!("blk.{index}.v"), self.dims.kv_heads * slots * 128)?,
                        // Resident per-channel V scale table (512 f32): the
                        // `ATTN_SCALE` command reads it by handle, so a fused
                        // attention layer needs no host copy of it.
                        Some(f32_image(
                            &format!("blk.{index}.vscale"),
                            &attention.v_channel,
                        )?),
                        AttentionParams {
                            q_scale: attention.attn_q_scale,
                            out_scale: attention.attn_out_scale,
                            reps: attention.reps,
                            cache_pages,
                            preprojected: true,
                        },
                    )?;
                    if attention.v_channel.len() != kv_dim {
                        return Err(ModelPackError::Layout(format!(
                            "layer {index} has {} V channel scales, expected {kv_dim}",
                            attention.v_channel.len()
                        )));
                    }
                    builder = builder
                        .attention_full(index, self.attention_plan(dir, index, attention, caches)?);
                }
                _ => {
                    return Err(ModelPackError::Layout(format!(
                        "layer {index} has inconsistent kind and stages"
                    )));
                }
            }
        }

        let head = PackedMatrix::from_gemv(
            "lm_head",
            &self.read_gemv(dir, &self.lm_head.file, self.lm_head.seg_scales.as_deref())?,
        );
        builder = builder.lm_head(LmHeadPlan::new(
            head,
            self.lm_head.input_scale,
            self.lm_head.output_scale,
            self.lm_head.reps,
        )?);
        let plan = builder.build();
        plan.check_budget()?;
        Ok(plan)
    }

    fn ffn_plan(&self, dir: &Path, ffn: &PackFfn) -> Result<FfnPlan, ModelPackError> {
        if ffn.gate.input_scale.to_bits() != ffn.up.input_scale.to_bits() {
            return Err(ModelPackError::Layout(
                "FFN gate and up must share the input scale".to_owned(),
            ));
        }
        Ok(FfnPlan::new(
            self.matrix(dir, &ffn.gate)?,
            self.matrix(dir, &ffn.up)?,
            self.matrix(dir, &ffn.down)?,
            ffn.gate.input_scale,
            ffn.mid_scale,
            ffn.down.output_scale,
            ffn.reps,
        )?)
    }

    fn conv_plan(
        &self,
        dir: &Path,
        conv: &PackConv,
        layer: usize,
    ) -> Result<ConvPlan, ModelPackError> {
        let path = self.path(dir, &conv.conv_file);
        let blob = std::fs::read(&path)
            .map_err(|error| ModelPackError::Io(format!("{}: {error}", path.display())))?;
        let image = ResidentImage::new(format!("blk.{layer}.shortconv"), blob)?;
        Ok(ConvPlan::new(
            self.matrix(dir, &conv.in_proj)?,
            image,
            self.matrix(dir, &conv.out_proj)?,
            conv.in_proj.input_scale,
            conv.mix_scale,
            conv.y_scale,
            conv.out_proj.output_scale,
            conv.ksh,
            conv.reps,
            myelin_core::backend::dsp::conv_state_offset(layer),
        )?)
    }

    fn attention_plan(
        &self,
        dir: &Path,
        layer: usize,
        attention: &PackAttention,
        caches: AttentionPlan,
    ) -> Result<AttentionFullPlan, ModelPackError> {
        let gate = match &attention.gate {
            Some(gate) => Some(self.matrix(dir, gate)?),
            None => None,
        };
        AttentionFullPlan::new(
            caches,
            self.matrix(dir, &attention.query)?,
            self.matrix(dir, &attention.key)?,
            self.matrix(dir, &attention.value)?,
            self.matrix(dir, &attention.output)?,
            gate,
            attention.query_norm.clone(),
            attention.key_norm.clone(),
            attention.query.input_scale,
            attention.query.output_scale,
            attention.key.output_scale,
            attention.value.output_scale,
            attention
                .gate
                .as_ref()
                .map_or(1.0, |gate| gate.output_scale),
            attention.attn_q_scale,
            attention.attn_out_scale,
            attention.o_input_scale,
            attention.output.output_scale,
            VQuant::Channel(attention.v_channel.clone()),
            attention.reps,
        )
        .map_err(|error| ModelPackError::Layout(format!("layer {layer}: {error}")))
    }

    fn matrix(&self, dir: &Path, gemv: &PackGemv) -> Result<PackedMatrix, ModelPackError> {
        let packed = self.read_gemv(dir, &gemv.file, gemv.seg_scales.as_deref())?;
        Ok(PackedMatrix::from_gemv(gemv.file.clone(), &packed))
    }

    fn read_gemv(
        &self,
        dir: &Path,
        file: &str,
        seg_scales: Option<&str>,
    ) -> Result<PackedGemv, ModelPackError> {
        let blob_path = self.path(dir, file);
        let manifest_path = blob_path.with_extension("json");
        let blob = std::fs::read(&blob_path)
            .map_err(|error| ModelPackError::Io(format!("{}: {error}", blob_path.display())))?;
        let manifest = GemvManifest::read(&manifest_path).map_err(|error| {
            ModelPackError::Layout(format!("{}: {error}", manifest_path.display()))
        })?;
        let packed = PackedGemv::new(file.to_owned(), manifest, blob)
            .map_err(|error| ModelPackError::Layout(format!("{file}: {error}")))?;
        match seg_scales {
            Some(name) => {
                let path = self.path(dir, name);
                let bytes = std::fs::read(&path).map_err(|error| {
                    ModelPackError::Io(format!("{}: {error}", path.display()))
                })?;
                let segments = Segments::decode(&bytes).map_err(|error| {
                    ModelPackError::Layout(format!("{name}: {error}"))
                })?;
                packed
                    .with_segments(segments)
                    .map_err(|error| ModelPackError::Layout(format!("{file}: {error}")))
            }
            None => Ok(packed),
        }
    }
}

fn zero_image(label: &str, bytes: usize) -> Result<ResidentImage, ModelPackError> {
    // `alloc_in` needs a 128-byte multiple; K/V images already are, but the
    // per-token scale table is only 4-byte aligned.
    let padded = bytes.max(1).div_ceil(128) * 128;
    Ok(ResidentImage::new(label, vec![0u8; padded])?)
}

/// Little-endian f32 table padded to the 128-byte `alloc_in` alignment.
fn f32_image(label: &str, values: &[f32]) -> Result<ResidentImage, ModelPackError> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let padded = bytes.len().max(1).div_ceil(128) * 128;
    bytes.resize(padded, 0);
    Ok(ResidentImage::new(label, bytes)?)
}

/// Map a GGML type name to the enum used by the checkpoint readers.
fn ggml_type(name: &str) -> Option<GgmlType> {
    Some(match name {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "Q4_K" => GgmlType::Q4_K,
        "Q6_K" => GgmlType::Q6_K,
        "Q8_0" => GgmlType::Q8_0,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_pack() -> ModelPack {
        let gemv = PackGemv {
            file: "ffn_00_gate.gemv".to_owned(),
            input_scale: 0.5,
            output_scale: 1.0,
            seg_scales: None,
        };
        ModelPack {
            version: PACK_VERSION,
            model: "tiny".to_owned(),
            dims: PackDims {
                hidden: 2,
                ffn: 4,
                layers: 1,
                heads: 1,
                kv_heads: 1,
                head_dim: 2,
                vocab: 2,
                conv_cache: 7,
                sliding_window: 4,
            },
            norm_eps: 1e-5,
            tied_embeddings: true,
            rope_full: PackRope {
                kind: "full".to_owned(),
                theta: 10_000.0,
                dim: 2,
                rotated_dim: 0,
            },
            rope_sliding: None,
            layers: vec![PackLayer {
                kind: "conv".to_owned(),
                attention_kind: None,
                ffn: PackFfn {
                    gate: gemv.clone(),
                    up: gemv.clone(),
                    down: gemv.clone(),
                    mid_scale: 0.25,
                    reps: 1,
                },
                conv: Some(PackConv {
                    in_proj: gemv.clone(),
                    conv_file: "conv_00.blob".to_owned(),
                    out_proj: gemv,
                    mix_scale: 0.01,
                    y_scale: 0.05,
                    ksh: 20,
                    reps: 1,
                }),
                attention: None,
            }],
            embed: PackEmbed {
                file: "embed.bin".to_owned(),
                dtype: "F32".to_owned(),
                rows: 2,
                cols: 2,
            },
            lm_head: PackLmHead {
                file: "lm_head.gemv".to_owned(),
                seg_scales: None,
                input_scale: 0.5,
                output_scale: 0.25,
                reps: 1,
            },
            norms: vec![PackNorms {
                attention: vec![1.0; 2],
                post_attention: vec![1.0; 2],
                ffn: vec![1.0; 2],
                post_ffn: vec![1.0; 2],
            }],
            output_norm: vec![1.0; 2],
        }
    }

    #[test]
    fn manifest_round_trips_and_rebuilds_the_spec() {
        let pack = tiny_pack();
        let dir = std::env::temp_dir().join(format!("myelin-pack-{}", std::process::id()));
        pack.write(&dir).expect("write");
        let read = ModelPack::read(&dir).expect("read");
        assert_eq!(read, pack);
        let spec = read.spec();
        assert_eq!(spec.dims.layers, 1);
        assert_eq!(spec.layers, vec![LayerSpec::Conv { ksize: 7 }]);
        assert!(spec.tied_embeddings);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_rejects_a_layer_count_mismatch() {
        let mut pack = tiny_pack();
        pack.dims.layers = 2;
        match pack.check() {
            Err(ModelPackError::Layout(message)) => assert!(message.contains("layers")),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn manifest_rejects_short_norms() {
        let mut pack = tiny_pack();
        pack.norms[0].ffn.truncate(1);
        match pack.check() {
            Err(ModelPackError::Layout(message)) => assert!(message.contains("ffn_norm")),
            other => panic!("unexpected {other:?}"),
        }
    }
}
