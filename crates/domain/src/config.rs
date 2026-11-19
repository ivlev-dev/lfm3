use serde::{Deserialize, Serialize};
use thiserror::Error;

pub trait MetadataSource {
    fn get_str(&self, key: &str) -> Option<&str>;
    fn get_u32(&self, key: &str) -> Option<u32>;
    fn get_f32(&self, key: &str) -> Option<f32>;
    fn get_i32(&self, key: &str) -> Option<i32>;
    fn get_i32_array(&self, key: &str) -> Option<Vec<i32>>;
    fn get_bool_array(&self, key: &str) -> Option<Vec<bool>>;
    fn tensor_dims(&self, name: &str) -> Option<Vec<u64>>;
    fn has_tensor(&self, name: &str) -> bool;
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    #[error("missing metadata key: {0}")]
    MissingKey(&'static str),
    #[error("layer arrays have different lengths: layers={layers}, kv={kv}, swa={swa}")]
    LayerCount {
        layers: usize,
        kv: usize,
        swa: usize,
    },
    #[error("invalid value for {key}: {value}")]
    Invalid { key: &'static str, value: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Conv,
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    pub theta: f64,
    pub dim: usize,
    pub rotated_dim: usize,
}

impl RopeParams {
    pub fn inv_freq(&self) -> Vec<f32> {
        let half = self.dim / 2;
        let rotated = self.rotated_dim / 2;
        (0..half)
            .map(|i| {
                if i < rotated {
                    (1.0 / self.theta.powf((2.0 * i as f64) / self.dim as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub arch: String,
    pub hidden_size: usize,
    pub ffn_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kv_heads: Vec<usize>,
    pub layer_kinds: Vec<LayerKind>,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub conv_cache: usize,
    pub sliding_window: usize,
    pub rope_full: RopeParams,
    pub rope_sliding: RopeParams,
    pub tied_embeddings: bool,
}

impl ModelConfig {
    pub fn from_metadata(src: &impl MetadataSource) -> Result<Self, ConfigError> {
        let arch = src
            .get_str("general.architecture")
            .ok_or(ConfigError::MissingKey("general.architecture"))?;
        if arch != "lfm3" {
            return Err(ConfigError::UnsupportedArch(arch.to_owned()));
        }
        let num_layers = src
            .get_u32("lfm3.block_count")
            .ok_or(ConfigError::MissingKey("lfm3.block_count"))? as usize;
        let hidden_size =
            src.get_u32("lfm3.embedding_length")
                .ok_or(ConfigError::MissingKey("lfm3.embedding_length"))? as usize;
        let ffn_size =
            src.get_u32("lfm3.feed_forward_length")
                .ok_or(ConfigError::MissingKey("lfm3.feed_forward_length"))? as usize;
        let num_heads =
            src.get_u32("lfm3.attention.head_count")
                .ok_or(ConfigError::MissingKey("lfm3.attention.head_count"))? as usize;
        let head_dim =
            src.get_u32("lfm3.attention.key_length")
                .ok_or(ConfigError::MissingKey("lfm3.attention.key_length"))? as usize;
        let value_dim = src
            .get_u32("lfm3.attention.value_length")
            .ok_or(ConfigError::MissingKey("lfm3.attention.value_length"))?
            as usize;
        if value_dim != head_dim {
            return Err(ConfigError::Invalid {
                key: "lfm3.attention.value_length",
                value: format!("{value_dim} != {head_dim}"),
            });
        }
        let kv_heads: Vec<usize> = src
            .get_i32_array("lfm3.attention.head_count_kv")
            .ok_or(ConfigError::MissingKey("lfm3.attention.head_count_kv"))?
            .into_iter()
            .map(|v| {
                if v < 0 {
                    Err(ConfigError::Invalid {
                        key: "lfm3.attention.head_count_kv",
                        value: v.to_string(),
                    })
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<Result<Vec<usize>, ConfigError>>()?;
        let swa: Vec<bool> = src
            .get_bool_array("lfm3.attention.sliding_window_pattern")
            .ok_or(ConfigError::MissingKey(
                "lfm3.attention.sliding_window_pattern",
            ))?;
        if kv_heads.len() != num_layers || swa.len() != num_layers {
            return Err(ConfigError::LayerCount {
                layers: num_layers,
                kv: kv_heads.len(),
                swa: swa.len(),
            });
        }
        let layer_kinds = derive_layer_kinds(&kv_heads, &swa);
        let norm_eps =
            src.get_f32("lfm3.attention.layer_norm_rms_epsilon")
                .ok_or(ConfigError::MissingKey(
                    "lfm3.attention.layer_norm_rms_epsilon",
                ))?;
        let conv_cache =
            src.get_u32("lfm3.shortconv.l_cache")
                .ok_or(ConfigError::MissingKey("lfm3.shortconv.l_cache"))? as usize;
        let sliding_window = src
            .get_u32("lfm3.attention.sliding_window")
            .ok_or(ConfigError::MissingKey("lfm3.attention.sliding_window"))?
            as usize;
        let theta_full = src
            .get_f32("lfm3.rope.freq_base")
            .ok_or(ConfigError::MissingKey("lfm3.rope.freq_base"))? as f64;
        let theta_swa =
            src.get_f32("lfm3.rope.freq_base_swa")
                .ok_or(ConfigError::MissingKey("lfm3.rope.freq_base_swa"))? as f64;
        let emb_dims = src
            .tensor_dims("token_embd.weight")
            .ok_or(ConfigError::MissingKey("token_embd.weight"))?;
        if emb_dims.len() != 2 || emb_dims[0] as usize != hidden_size {
            return Err(ConfigError::Invalid {
                key: "token_embd.weight",
                value: format!("{emb_dims:?}"),
            });
        }
        let vocab_size = emb_dims[1] as usize;
        let full_rotated = if src.has_tensor("rope_freqs.weight") {
            head_dim / 2
        } else {
            head_dim
        };
        let config = Self {
            arch: arch.to_owned(),
            hidden_size,
            ffn_size,
            num_layers,
            num_heads,
            head_dim,
            kv_heads,
            layer_kinds,
            vocab_size,
            norm_eps,
            conv_cache,
            sliding_window,
            rope_full: RopeParams {
                theta: theta_full,
                dim: head_dim,
                rotated_dim: full_rotated,
            },
            rope_sliding: RopeParams {
                theta: theta_swa,
                dim: head_dim,
                rotated_dim: head_dim,
            },
            tied_embeddings: !src.has_tensor("output.weight"),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_layers == 0 || self.ffn_size == 0 || self.vocab_size == 0 {
            return Err(ConfigError::Invalid {
                key: "sizes",
                value: format!(
                    "layers={} ffn={} vocab={}",
                    self.num_layers, self.ffn_size, self.vocab_size
                ),
            });
        }
        if let Some((layer, kv)) = self.kv_heads.iter().enumerate().find(|(_, kv)| {
            **kv > self.num_heads || (**kv > 0 && !self.num_heads.is_multiple_of(**kv))
        }) {
            return Err(ConfigError::Invalid {
                key: "lfm3.attention.head_count_kv",
                value: format!(
                    "layer {layer}: {kv} heads vs {} query heads",
                    self.num_heads
                ),
            });
        }
        if !self.rope_full.theta.is_finite() || self.rope_full.theta <= 1.0 {
            return Err(ConfigError::Invalid {
                key: "lfm3.rope.freq_base",
                value: format!("{}", self.rope_full.theta),
            });
        }
        if !self.rope_sliding.theta.is_finite() || self.rope_sliding.theta <= 1.0 {
            return Err(ConfigError::Invalid {
                key: "lfm3.rope.freq_base_swa",
                value: format!("{}", self.rope_sliding.theta),
            });
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(ConfigError::Invalid {
                key: "lfm3.attention.layer_norm_rms_epsilon",
                value: format!("{}", self.norm_eps),
            });
        }
        if self.num_heads == 0 || self.head_dim == 0 || self.hidden_size == 0 {
            return Err(ConfigError::Invalid {
                key: "dims",
                value: format!("heads={} head_dim={}", self.num_heads, self.head_dim),
            });
        }
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(ConfigError::Invalid {
                key: "hidden_size",
                value: format!(
                    "{} != {} * {}",
                    self.hidden_size, self.num_heads, self.head_dim
                ),
            });
        }
        if self.rope_full.dim != self.head_dim || self.rope_sliding.dim != self.head_dim {
            return Err(ConfigError::Invalid {
                key: "rope.dim",
                value: format!("{}", self.head_dim),
            });
        }
        if self.conv_cache < 2 {
            return Err(ConfigError::Invalid {
                key: "lfm3.shortconv.l_cache",
                value: self.conv_cache.to_string(),
            });
        }
        if self.sliding_window == 0 {
            return Err(ConfigError::Invalid {
                key: "lfm3.attention.sliding_window",
                value: "0".to_owned(),
            });
        }
        Ok(())
    }

    pub fn kind(&self, layer: usize) -> LayerKind {
        self.layer_kinds[layer]
    }

    pub fn attention_layers(&self) -> impl Iterator<Item = (usize, LayerKind)> + '_ {
        self.layer_kinds
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, k)| *k != LayerKind::Conv)
    }

    pub fn conv_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layer_kinds
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, k)| *k == LayerKind::Conv)
            .map(|(i, _)| i)
    }

    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        (self.kv_heads.iter().copied().max().unwrap_or(0)) * self.head_dim
    }

    pub fn kv_heads_of(&self, layer: usize) -> usize {
        self.kv_heads[layer]
    }

    pub fn rope_for(&self, layer: usize) -> &RopeParams {
        match self.layer_kinds[layer] {
            LayerKind::SlidingAttention => &self.rope_sliding,
            _ => &self.rope_full,
        }
    }
}

fn derive_layer_kinds(kv_heads: &[usize], swa: &[bool]) -> Vec<LayerKind> {
    kv_heads
        .iter()
        .zip(swa)
        .map(|(kv, sliding)| {
            if *kv == 0 {
                LayerKind::Conv
            } else if *sliding {
                LayerKind::SlidingAttention
            } else {
                LayerKind::FullAttention
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_penalty_window: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_penalty_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            max_tokens: 512,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn lfm3_defaults(src: &impl MetadataSource) -> Self {
        let mut params = Self::default();
        if let Some(t) = src.get_f32("general.sampling.temp") {
            params.temperature = t;
        }
        if let Some(k) = src.get_u32("general.sampling.top_k").or_else(|| {
            src.get_i32("general.sampling.top_k")
                .map(|v| v.max(0) as u32)
        }) {
            params.top_k = k as usize;
        }
        if let Some(p) = src.get_f32("general.sampling.penalty_repeat") {
            params.repeat_penalty = p;
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct FakeMeta {
        strs: HashMap<String, String>,
        u32s: HashMap<String, u32>,
        f32s: HashMap<String, f32>,
        i32s: HashMap<String, Vec<i32>>,
        bools: HashMap<String, Vec<bool>>,
        dims: HashMap<String, Vec<u64>>,
    }

    impl MetadataSource for FakeMeta {
        fn get_str(&self, key: &str) -> Option<&str> {
            self.strs.get(key).map(|s| s.as_str())
        }
        fn get_u32(&self, key: &str) -> Option<u32> {
            self.u32s.get(key).copied()
        }
        fn get_f32(&self, key: &str) -> Option<f32> {
            self.f32s.get(key).copied()
        }
        fn get_i32(&self, key: &str) -> Option<i32> {
            self.i32s.get(key).and_then(|v| v.first().copied())
        }
        fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
            self.i32s.get(key).cloned()
        }
        fn get_bool_array(&self, key: &str) -> Option<Vec<bool>> {
            self.bools.get(key).cloned()
        }
        fn tensor_dims(&self, name: &str) -> Option<Vec<u64>> {
            self.dims.get(name).cloned()
        }
        fn has_tensor(&self, name: &str) -> bool {
            self.dims.contains_key(name) || name == "rope_freqs.weight"
        }
    }

    fn lfm3_meta() -> FakeMeta {
        let mut m = FakeMeta::default();
        m.strs.insert("general.architecture".into(), "lfm3".into());
        m.u32s.insert("lfm3.block_count".into(), 24);
        m.u32s.insert("lfm3.embedding_length".into(), 2048);
        m.u32s.insert("lfm3.feed_forward_length".into(), 7680);
        m.u32s.insert("lfm3.attention.head_count".into(), 16);
        m.u32s.insert("lfm3.attention.key_length".into(), 128);
        m.u32s.insert("lfm3.attention.value_length".into(), 128);
        m.u32s.insert("lfm3.attention.sliding_window".into(), 256);
        m.u32s.insert("lfm3.shortconv.l_cache".into(), 7);
        m.f32s
            .insert("lfm3.attention.layer_norm_rms_epsilon".into(), 1e-5);
        m.f32s.insert("lfm3.rope.freq_base".into(), 1e6);
        m.f32s.insert("lfm3.rope.freq_base_swa".into(), 1e4);
        m.f32s.insert("general.sampling.temp".into(), 0.1);
        m.u32s.insert("general.sampling.top_k".into(), 50);
        m.f32s.insert("general.sampling.penalty_repeat".into(), 1.1);
        m.i32s.insert(
            "lfm3.attention.head_count_kv".into(),
            vec![
                0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0,
            ],
        );
        let mut swa = vec![false; 24];
        for i in [4usize, 8, 12, 16, 20] {
            swa[i] = true;
        }
        m.bools
            .insert("lfm3.attention.sliding_window_pattern".into(), swa);
        m.dims
            .insert("token_embd.weight".into(), vec![2048, 128000]);
        m
    }

    #[test]
    fn parses_lfm3_config() {
        let cfg = ModelConfig::from_metadata(&lfm3_meta()).unwrap();
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.ffn_size, 7680);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 128000);
        assert_eq!(cfg.conv_cache, 7);
        assert_eq!(cfg.sliding_window, 256);
        assert!(cfg.tied_embeddings);
        assert_eq!(cfg.rope_full.rotated_dim, 64);
        assert_eq!(cfg.rope_sliding.rotated_dim, 128);
        assert_eq!(cfg.q_dim(), 2048);
        assert_eq!(cfg.kv_dim(), 512);
        assert_eq!(cfg.conv_layers().count(), 13);
        assert_eq!(
            cfg.attention_layers()
                .filter(|(_, k)| *k == LayerKind::FullAttention)
                .count(),
            6
        );
        assert_eq!(
            cfg.attention_layers()
                .filter(|(_, k)| *k == LayerKind::SlidingAttention)
                .count(),
            5
        );
        let params = SamplingParams::lfm3_defaults(&lfm3_meta());
        assert!((params.temperature - 0.1).abs() < 1e-6);
        assert_eq!(params.top_k, 50);
        assert!((params.repeat_penalty - 1.1).abs() < 1e-6);
    }

    #[test]
    fn inv_freq_zero_tail() {
        let rope = RopeParams {
            theta: 1e6,
            dim: 128,
            rotated_dim: 64,
        };
        let f = rope.inv_freq();
        assert_eq!(f.len(), 64);
        assert!((f[0] - 1.0).abs() < 1e-9);
        assert!(f[31] > 0.0);
        assert_eq!(f[32], 0.0);
        assert_eq!(f[63], 0.0);
        let full = RopeParams {
            theta: 1e4,
            dim: 128,
            rotated_dim: 128,
        };
        assert!(full.inv_freq().iter().all(|v| *v > 0.0));
        assert!(full.inv_freq()[0] > full.inv_freq()[63]);
    }

    #[test]
    fn rejects_wrong_arch() {
        let mut m = lfm3_meta();
        m.strs.insert("general.architecture".into(), "llama".into());
        assert!(matches!(
            ModelConfig::from_metadata(&m),
            Err(ConfigError::UnsupportedArch(_))
        ));
    }

    #[test]
    fn rejects_layer_mismatch() {
        let mut m = lfm3_meta();
        m.i32s
            .insert("lfm3.attention.head_count_kv".into(), vec![0, 4]);
        assert!(matches!(
            ModelConfig::from_metadata(&m),
            Err(ConfigError::LayerCount { .. })
        ));
    }
}
