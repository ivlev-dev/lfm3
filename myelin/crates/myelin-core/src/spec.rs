//! Model-independent specification types.
//!
//! A [`ModelSpec`] is plain data: engine, FFI and future backends consume it
//! without matching on a model family. Family adapters in [`crate::arch`]
//! are the only code allowed to know concrete numbers.

use std::fmt;

use crate::CoreError;

/// Known model families.
///
/// Adding a family is a code change in two places: this enum and a new
/// adapter under `arch/`. `Unknown` is reserved for placeholder specs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    Lfm3,
    Llama,
    Qwen,
    Gemma,
    Mistral,
    Unknown,
}

impl ModelFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lfm3 => "lfm3",
            Self::Llama => "llama",
            Self::Qwen => "qwen",
            Self::Gemma => "gemma",
            Self::Mistral => "mistral",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Global shape of the transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    /// Maximum number of KV heads across layers. Convolution layers use zero.
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// Length of the short-convolution state cache.
    pub conv_cache: usize,
    /// Sliding-attention window in tokens.
    pub sliding_window: usize,
}

impl Dims {
    pub const fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub const fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Full or partial rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeKind {
    Full,
    Partial { rotated_dim: usize },
}

/// One rotary configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub kind: RopeKind,
    pub theta: f32,
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub full: RopeSpec,
    /// Some architectures use a different base for sliding-attention layers.
    pub sliding: Option<RopeSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    Sliding,
}

/// Per-layer structure. MoE is modelled but not executed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSpec {
    Attention { kind: AttentionKind },
    Conv { ksize: usize },
    Moe { experts: usize, top_k: usize },
}

impl LayerSpec {
    pub const fn is_attention(self) -> bool {
        matches!(self, Self::Attention { .. })
    }

    pub const fn is_conv(self) -> bool {
        matches!(self, Self::Conv { .. })
    }

    pub const fn is_moe(self) -> bool {
        matches!(self, Self::Moe { .. })
    }
}

/// Storage type of a tensor or cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    Bf16,
    Q8_0,
    Q4_0,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            _ => None,
        }
    }
}

/// KV-cache storage. The engine may override it with a runtime setting later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvFormat {
    pub dtype: DType,
    pub group_size: Option<u32>,
}

impl KvFormat {
    pub const F16: Self = Self {
        dtype: DType::F16,
        group_size: None,
    };
}

/// Quantization of model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    F16,
    Bf16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
    Other(u32),
}

impl Quant {
    /// Map a GGUF `general.file_type` value to a coarse quantization class.
    pub const fn from_gguf_file_type(file_type: u32) -> Self {
        match file_type {
            0 => Self::Other(0), // all F32
            1 => Self::F16,
            2 => Self::Q4_0,
            7 => Self::Q8_0,
            15 => Self::Q4K,
            18 => Self::Q6K,
            other => Self::Other(other),
        }
    }

    pub const fn group_size(self) -> Option<u32> {
        match self {
            Self::Q4_0 | Self::Q8_0 => Some(32),
            Self::Q4K | Self::Q6K => Some(256),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightFormat {
    pub quant: Quant,
    pub group_size: Option<u32>,
}

impl WeightFormat {
    pub const fn from_quant(quant: Quant) -> Self {
        Self {
            quant,
            group_size: quant.group_size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerSpec {
    /// GGUF tokenizer model name, for example `gpt2` or `llama`.
    pub model: String,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateSpec {
    /// Raw Jinja template. Empty when the metadata does not carry one.
    pub template: String,
}

impl ChatTemplateSpec {
    pub fn is_empty(&self) -> bool {
        self.template.trim().is_empty()
    }
}

/// Everything a backend needs to know about a model, with no family checks.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub family: ModelFamily,
    pub dims: Dims,
    pub rope: RopeConfig,
    pub layers: Vec<LayerSpec>,
    pub kv_format: KvFormat,
    pub tokenizer: TokenizerSpec,
    pub chat_template: ChatTemplateSpec,
    pub weight_format: WeightFormat,
}

impl ModelSpec {
    /// Tiny internally consistent spec used by the FFI before a real model is
    /// loaded. It deliberately carries no family-specific numbers.
    pub fn placeholder() -> Self {
        const HEAD_DIM: usize = 16;
        const HEADS: usize = 2;
        let dims = Dims {
            hidden: HEADS * HEAD_DIM,
            ffn: 64,
            layers: 2,
            heads: HEADS,
            kv_heads: HEADS,
            head_dim: HEAD_DIM,
            vocab: 128,
            conv_cache: 3,
            sliding_window: 8,
        };
        Self {
            family: ModelFamily::Unknown,
            dims,
            rope: RopeConfig {
                full: RopeSpec {
                    kind: RopeKind::Full,
                    theta: 10_000.0,
                    dim: HEAD_DIM,
                },
                sliding: None,
            },
            layers: vec![
                LayerSpec::Conv { ksize: 3 },
                LayerSpec::Attention {
                    kind: AttentionKind::Full,
                },
            ],
            kv_format: KvFormat::F16,
            tokenizer: TokenizerSpec {
                model: "stub".to_owned(),
                vocab_size: dims.vocab,
                bos_token_id: None,
                eos_token_id: None,
            },
            chat_template: ChatTemplateSpec {
                template: String::new(),
            },
            weight_format: WeightFormat::from_quant(Quant::F16),
        }
    }

    pub fn layer(&self, index: usize) -> Option<LayerSpec> {
        self.layers.get(index).copied()
    }

    pub fn conv_layers(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.layers
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, layer)| match layer {
                LayerSpec::Conv { ksize } => Some((i, ksize)),
                _ => None,
            })
    }

    pub fn attention_layers(&self) -> impl Iterator<Item = (usize, AttentionKind)> + '_ {
        self.layers
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, layer)| match layer {
                LayerSpec::Attention { kind } => Some((i, kind)),
                _ => None,
            })
    }

    /// Cross-check invariants before the spec reaches a backend.
    pub fn validate(&self) -> Result<(), CoreError> {
        let d = &self.dims;
        if d.layers == 0 {
            return Err(invalid("layer count is zero"));
        }
        if d.hidden == 0 || d.heads == 0 || d.head_dim == 0 {
            return Err(invalid(
                "hidden size, head count and head dim must be non-zero",
            ));
        }
        if d.hidden != d.q_dim() {
            return Err(invalid(format!(
                "hidden {} != heads {} * head_dim {}",
                d.hidden, d.heads, d.head_dim
            )));
        }
        if d.kv_heads == 0 || d.kv_heads > d.heads || d.heads % d.kv_heads != 0 {
            return Err(invalid(format!(
                "kv_heads {} is not a divisor of heads {}",
                d.kv_heads, d.heads
            )));
        }
        if d.ffn == 0 || d.vocab == 0 {
            return Err(invalid("ffn and vocab must be non-zero"));
        }
        if d.conv_cache == 0 {
            return Err(invalid("conv cache must be non-zero"));
        }
        if self.layers.len() != d.layers {
            return Err(invalid(format!(
                "layers vec has {} entries, dims say {}",
                self.layers.len(),
                d.layers
            )));
        }
        if self.rope.full.dim != d.head_dim {
            return Err(invalid(format!(
                "rope dim {} != head_dim {}",
                self.rope.full.dim, d.head_dim
            )));
        }
        validate_rope("rope.full", self.rope.full)?;
        if let Some(sliding) = self.rope.sliding {
            if sliding.dim != d.head_dim {
                return Err(invalid(format!(
                    "sliding rope dim {} != head_dim {}",
                    sliding.dim, d.head_dim
                )));
            }
            validate_rope("rope.sliding", sliding)?;
        }
        for (index, layer) in self.layers.iter().copied().enumerate() {
            match layer {
                LayerSpec::Attention {
                    kind: AttentionKind::Sliding,
                } if d.sliding_window == 0 => {
                    return Err(invalid(format!(
                        "layer {index} uses sliding attention but sliding_window is zero"
                    )));
                }
                LayerSpec::Conv { ksize } if ksize < 2 => {
                    return Err(invalid(format!("layer {index} has conv kernel {ksize}")));
                }
                LayerSpec::Moe { experts, top_k }
                    if experts == 0 || top_k == 0 || top_k > experts =>
                {
                    return Err(invalid(format!(
                        "layer {index} has invalid MoE experts={experts} top_k={top_k}"
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn validate_rope(name: &str, rope: RopeSpec) -> Result<(), CoreError> {
    if !rope.theta.is_finite() || rope.theta <= 1.0 {
        return Err(invalid(format!("{name} theta {} is invalid", rope.theta)));
    }
    if let RopeKind::Partial { rotated_dim } = rope.kind {
        if rotated_dim == 0 || rotated_dim > rope.dim {
            return Err(invalid(format!(
                "{name} rotated_dim {rotated_dim} outside 1..={}",
                rope.dim
            )));
        }
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidSpec(reason.into())
}
