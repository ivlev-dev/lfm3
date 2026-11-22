//! LFM3 family adapter.
//!
//! Every LFM3-specific number lives in this file. If a magic value escapes
//! into `spec.rs`, `engine.rs` or the FFI, the `no_lfm3_constants` test fails.

use crate::arch::Arch;
use crate::gguf::GgufMeta;
use crate::spec::{
    AttentionKind, ChatTemplateSpec, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec, Quant,
    RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
};
use crate::CoreError;

/// Marker type for the LFM3 adapter.
pub struct Lfm3;

const ARCH_ID: &str = "lfm3";
const EMBEDDING_TENSOR: &str = "token_embd.weight";
const OUTPUT_TENSOR: &str = "output.weight";
const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";
/// Fallback RMSNorm epsilon when the checkpoint does not carry one.
const DEFAULT_NORM_EPS: f32 = 1e-5;

/// Defaults and layer layout of the LFM3 1.7B checkpoint family.
struct Profile {
    hidden: usize,
    ffn: usize,
    layers: usize,
    head_count: usize,
    head_dim: usize,
    vocab: usize,
    conv_cache: usize,
    sliding_window: usize,
    full_rope_theta: f32,
    sliding_rope_theta: f32,
    /// Zero means a convolution layer, non-zero is the KV head count.
    kv_heads_by_layer: [u8; 24],
    sliding_by_layer: [bool; 24],
}

const PROFILE: Profile = Profile {
    hidden: 2048,
    ffn: 7680,
    layers: 24,
    head_count: 16,
    head_dim: 128,
    vocab: 128_000,
    conv_cache: 7,
    sliding_window: 256,
    full_rope_theta: 1_000_000.0,
    sliding_rope_theta: 10_000.0,
    // 13 convolution layers and 11 attention layers.
    kv_heads_by_layer: [
        0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0,
    ],
    // Sliding attention on layers 4, 8, 12, 16 and 20.
    sliding_by_layer: [
        false, false, false, false, true, false, false, false, true, false, false, false, true,
        false, false, false, true, false, false, false, true, false, false, false,
    ],
};

impl Arch for Lfm3 {
    fn detect(meta: &GgufMeta) -> Option<ModelFamily> {
        meta.architecture()
            .eq_ignore_ascii_case(ARCH_ID)
            .then_some(ModelFamily::Lfm3)
    }

    fn build_spec(meta: &GgufMeta) -> Result<ModelSpec, CoreError> {
        let layers = meta
            .get_arch_u32("block_count")
            .map_or(PROFILE.layers, |value| value as usize);
        let head_count = meta
            .get_arch_u32("attention.head_count")
            .map_or(PROFILE.head_count, |value| value as usize);
        let head_dim = meta
            .get_arch_u32("attention.key_length")
            .map_or(PROFILE.head_dim, |value| value as usize);
        let hidden = meta
            .get_arch_u32("embedding_length")
            .map_or(PROFILE.hidden, |value| value as usize);
        let ffn = meta
            .get_arch_u32("feed_forward_length")
            .map_or(PROFILE.ffn, |value| value as usize);
        let sliding_window = meta
            .get_arch_u32("attention.sliding_window")
            .map_or(PROFILE.sliding_window, |value| value as usize);
        let conv_kernel = meta
            .get_arch_u32("shortconv.l_cache")
            .map_or(PROFILE.conv_cache, |value| value as usize);
        let conv_cache = conv_kernel;

        let kv_by_layer = kv_heads_by_layer(meta, layers, head_count)?;
        let sliding_by_layer = sliding_by_layer(meta, layers)?;
        let kv_heads = kv_by_layer.iter().copied().max().unwrap_or(0);
        if kv_heads == 0 {
            return Err(invalid(
                "attention.head_count_kv",
                "no layer declares KV heads",
            ));
        }

        let vocab = vocab_size(meta, hidden)?;
        let (bos_token_id, eos_token_id) = (
            meta.get_u32("tokenizer.ggml.bos_token_id"),
            meta.get_u32("tokenizer.ggml.eos_token_id"),
        );
        let tokenizer = TokenizerSpec {
            model: meta
                .get_str("tokenizer.ggml.model")
                .unwrap_or("gpt2")
                .to_owned(),
            vocab_size: vocab,
            bos_token_id,
            eos_token_id,
        };
        let chat_template = ChatTemplateSpec {
            template: meta
                .get_str("tokenizer.chat_template")
                .unwrap_or_default()
                .to_owned(),
        };

        let full_rope_theta = meta
            .get_arch_f32("rope.freq_base")
            .unwrap_or(PROFILE.full_rope_theta);
        let sliding_rope_theta = meta
            .get_arch_f32("rope.freq_base_swa")
            .unwrap_or(PROFILE.sliding_rope_theta);
        let full_kind = if meta.has_tensor(ROPE_FREQS_TENSOR) {
            RopeKind::Partial {
                rotated_dim: head_dim / 2,
            }
        } else {
            RopeKind::Full
        };
        let rope = RopeConfig {
            full: RopeSpec {
                kind: full_kind,
                theta: full_rope_theta,
                dim: head_dim,
            },
            sliding: Some(RopeSpec {
                kind: RopeKind::Full,
                theta: sliding_rope_theta,
                dim: head_dim,
            }),
        };

        let mut layer_specs = Vec::with_capacity(layers);
        for (index, kv) in kv_by_layer.iter().copied().enumerate() {
            if kv == 0 {
                layer_specs.push(LayerSpec::Conv { ksize: conv_kernel });
            } else {
                let kind = if sliding_by_layer[index] {
                    AttentionKind::Sliding
                } else {
                    AttentionKind::Full
                };
                layer_specs.push(LayerSpec::Attention { kind });
            }
        }

        let kv_format = match meta.get_arch_str("attention.kv_format") {
            Some(value) => KvFormat {
                dtype: crate::DType::parse(value).ok_or_else(|| {
                    invalid("attention.kv_format", format!("unknown dtype {value:?}"))
                })?,
                group_size: None,
            },
            None => KvFormat::F16,
        };

        let file_type = meta.get_u32("general.file_type").unwrap_or(15);
        let weight_format = WeightFormat::from_quant(Quant::from_gguf_file_type(file_type));
        let norm_eps = meta
            .get_arch_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(DEFAULT_NORM_EPS);
        let tied_embeddings = !meta.has_tensor(OUTPUT_TENSOR);

        let spec = ModelSpec {
            family: ModelFamily::Lfm3,
            dims: Dims {
                hidden,
                ffn,
                layers,
                heads: head_count,
                kv_heads,
                head_dim,
                vocab,
                conv_cache,
                sliding_window,
            },
            rope,
            layers: layer_specs,
            kv_format,
            tokenizer,
            chat_template,
            weight_format,
            norm_eps,
            tied_embeddings,
        };
        spec.validate()?;
        Ok(spec)
    }
}

fn kv_heads_by_layer(
    meta: &GgufMeta,
    layers: usize,
    head_count: usize,
) -> Result<Vec<usize>, CoreError> {
    if let Some(values) = meta.get_arch_i32_array("attention.head_count_kv") {
        if values.len() != layers {
            return Err(invalid(
                "attention.head_count_kv",
                format!("expected {layers} entries, got {}", values.len()),
            ));
        }
        return values
            .iter()
            .map(|value| {
                let value = usize::try_from(*value)
                    .map_err(|_| invalid("attention.head_count_kv", value.to_string()))?;
                if value > head_count {
                    return Err(invalid(
                        "attention.head_count_kv",
                        format!("{value} exceeds {head_count} query heads"),
                    ));
                }
                Ok(value)
            })
            .collect();
    }
    if layers == PROFILE.layers {
        Ok(PROFILE
            .kv_heads_by_layer
            .iter()
            .map(|value| usize::from(*value))
            .collect())
    } else {
        Err(CoreError::MissingKey(
            meta.arch_key("attention.head_count_kv"),
        ))
    }
}

fn sliding_by_layer(meta: &GgufMeta, layers: usize) -> Result<Vec<bool>, CoreError> {
    if let Some(values) = meta.get_arch_bool_array("attention.sliding_window_pattern") {
        if values.len() != layers {
            return Err(invalid(
                "attention.sliding_window_pattern",
                format!("expected {layers} entries, got {}", values.len()),
            ));
        }
        return Ok(values.to_vec());
    }
    if layers == PROFILE.layers {
        Ok(PROFILE.sliding_by_layer.to_vec())
    } else {
        Err(CoreError::MissingKey(
            meta.arch_key("attention.sliding_window_pattern"),
        ))
    }
}

fn vocab_size(meta: &GgufMeta, hidden: usize) -> Result<usize, CoreError> {
    let Some(dims) = meta.tensor_dims(EMBEDDING_TENSOR) else {
        return Ok(PROFILE.vocab);
    };
    if dims.len() != 2 {
        return Err(invalid(
            EMBEDDING_TENSOR,
            format!("rank {} != 2", dims.len()),
        ));
    }
    if dims[0] as usize != hidden {
        return Err(invalid(
            EMBEDDING_TENSOR,
            format!("embedding dim {} != hidden {hidden}", dims[0]),
        ));
    }
    Ok(dims[1] as usize)
}

fn invalid(key: &str, value: impl ToString) -> CoreError {
    CoreError::InvalidValue {
        key: key.to_owned(),
        value: value.to_string(),
    }
}
