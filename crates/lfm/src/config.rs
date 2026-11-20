use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const DEFAULT_H2_CANCEL_INTERVAL: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: ModelSection,
    pub backend: BackendSection,
    pub chat: ChatSection,
    pub server: ServerSection,
    pub ui: UiSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSection {
    pub path: PathBuf,
    pub tokenizer: PathBuf,
    pub template: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendMode {
    Single,
    Pipeline,
    Tp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendSection {
    pub kind: String,
    pub devices: Vec<usize>,
    pub mode: BackendMode,
    pub tiers: Vec<usize>,
    pub kv: String,
    pub ctx: usize,
    pub kv_pool: usize,
    pub max_seqs: usize,
    pub prefill_ntok: usize,
    pub pb: usize,
    pub splits: usize,
    pub cache: bool,
    pub fuse: bool,
    pub attn4: bool,
    pub attn2: bool,
    pub flash: bool,
    pub flash2: bool,
    pub b8p: bool,
    pub i8pf: bool,
    pub q4i8: bool,
    pub q6i8: bool,
    pub q4i8_ffn_only: bool,
    pub wg: usize,
    pub wg_b: usize,
    pub rgemv: usize,
    pub rgemv_b: usize,
    pub prepack: bool,
    pub prepack_threads: usize,
    pub program_cache: bool,
    pub gemm_force: Option<String>,
    pub profile: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatSection {
    pub system: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
    pub thinking: bool,
    pub tools: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerSection {
    pub addr: String,
    pub name: String,
    pub max_body: usize,
    pub prefill_budget: usize,
    pub prefill_in_decode: usize,
    pub h2_cancel_interval: usize,
    pub log_requests: bool,
    pub system: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiSection {
    pub stats: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelSection::default(),
            backend: BackendSection::default(),
            chat: ChatSection::default(),
            server: ServerSection::default(),
            ui: UiSection::default(),
        }
    }
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            path: PathBuf::from("models/LFM3-1.7B-Q4_K_M.gguf"),
            tokenizer: PathBuf::from("models/hf/tokenizer.json"),
            template: PathBuf::from("models/hf/chat_template.jinja"),
        }
    }
}

impl Default for BackendSection {
    fn default() -> Self {
        Self {
            kind: "opencl".to_string(),
            devices: vec![0],
            mode: BackendMode::Single,
            tiers: Vec::new(),
            kv: "f16".to_string(),
            ctx: 4096,
            kv_pool: 1,
            max_seqs: 8,
            prefill_ntok: 128,
            pb: 6,
            splits: 16,
            cache: true,
            fuse: true,
            attn4: true,
            attn2: true,
            flash: true,
            flash2: true,
            b8p: true,
            i8pf: true,
            q4i8: true,
            q6i8: true,
            q4i8_ffn_only: false,
            wg: 256,
            wg_b: 64,
            rgemv: 4,
            rgemv_b: 5,
            prepack: true,
            prepack_threads: 0,
            program_cache: true,
            gemm_force: None,
            profile: false,
        }
    }
}

impl Default for ChatSection {
    fn default() -> Self {
        Self {
            system: "You are a helpful assistant.".to_string(),
            max_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.05,
            seed: 0,
            thinking: false,
            tools: None,
        }
    }
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8000".to_string(),
            name: "lfm3".to_string(),
            max_body: 4 << 20,
            prefill_budget: 64,
            prefill_in_decode: 8,
            h2_cancel_interval: DEFAULT_H2_CANCEL_INTERVAL,
            log_requests: false,
            system: "You are a helpful assistant.".to_string(),
        }
    }
}

impl Default for UiSection {
    fn default() -> Self {
        Self { stats: true }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Self::from_toml(&text)
    }

    pub fn load_optional(path: Option<&Path>) -> Result<Self> {
        match path {
            Some(path) => Self::load(path),
            None => Ok(Self::default()),
        }
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let cfg: Self = toml::from_str(text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn example_toml() -> String {
        let mut cfg = Self::default();
        cfg.backend.mode = BackendMode::Pipeline;
        cfg.backend.devices = vec![0, 1];
        cfg.backend.tiers = vec![96, 4];
        cfg.backend.kv = "kvarn".to_string();
        cfg.backend.ctx = 32768;
        let body = cfg.to_toml().unwrap_or_default();
        format!(
            "# lfm.toml — конфигурация LFM3\n\
             # backend.devices: индексы OpenCL-устройств (0=GPU, 1=iGPU при RUSTICL_ENABLE=iris,radeonsi)\n\
             # backend.mode: single | pipeline | tp; tiers — проценты слоёв для pipeline\n\
             # backend.kv: f16 | q4 | kvarn | kvarn_k4v4 | k4v4\n\
             # Приоритет: CLI > этот файл > значения по умолчанию\n\n{body}"
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.backend.kind != "opencl" {
            bail!(
                "backend.kind = {:?} не поддерживается (доступно: opencl)",
                self.backend.kind
            );
        }
        if self.backend.devices.is_empty() {
            bail!("backend.devices пуст");
        }
        if self.backend.ctx == 0 || !self.backend.ctx.is_multiple_of(32) {
            bail!("backend.ctx должен быть положительным и кратным 32");
        }
        if !matches!(
            self.backend.kv.as_str(),
            "f16" | "q4" | "k4v4" | "kvarn" | "kvarn_k4v2" | "kvarn_k4v4"
        ) {
            bail!(
                "backend.kv = {:?} (доступно: f16, q4, k4v4, kvarn, kvarn_k4v2, kvarn_k4v4)",
                self.backend.kv
            );
        }
        if !(1..=16).contains(&self.backend.max_seqs) {
            bail!("backend.max_seqs должен быть в 1..=16");
        }
        if !(1..=16).contains(&self.backend.kv_pool) {
            bail!("backend.kv_pool должен быть в 1..=16");
        }
        if !(1..=128).contains(&self.backend.prefill_ntok) {
            bail!("backend.prefill_ntok должен быть в 1..=128");
        }
        if !(1..=12).contains(&self.backend.pb) {
            bail!("backend.pb должен быть в 1..=12");
        }
        if !(1..=16).contains(&self.backend.splits) {
            bail!("backend.splits должен быть в 1..=16");
        }
        match self.backend.mode {
            BackendMode::Single => {}
            BackendMode::Pipeline => {
                if self.backend.devices.len() < 2 {
                    bail!("backend.mode = pipeline требует минимум 2 устройства");
                }
                let tiers = self.backend.effective_tiers()?;
                if tiers.len() != self.backend.devices.len() {
                    bail!("backend.tiers должен совпадать по длине с devices");
                }
            }
            BackendMode::Tp => {
                if self.backend.kv != "f16" {
                    bail!("tensor parallel поддерживает только kv = f16");
                }
                if self.backend.devices.len() > 4 {
                    bail!("tensor parallel: не больше 4 устройств (4 KV-головы)");
                }
            }
        }
        if self.backend.tiers.iter().any(|p| *p > 100) {
            bail!("backend.tiers: проценты должны быть 0..=100");
        }
        if self.chat.max_tokens == 0 {
            bail!("chat.max_tokens должен быть положительным");
        }
        if self.server.max_body < 1024 {
            bail!("server.max_body должен быть не меньше 1024 байт");
        }
        if !(1..=128).contains(&self.server.prefill_budget) {
            bail!("server.prefill_budget должен быть в 1..=128");
        }
        if self.server.prefill_in_decode > 64 {
            bail!("server.prefill_in_decode должен быть в 0..=64");
        }
        if !(1..=1024).contains(&self.server.h2_cancel_interval) {
            bail!("server.h2_cancel_interval должен быть в 1..=1024");
        }
        Ok(())
    }
}

impl BackendSection {
    pub fn effective_tiers(&self) -> Result<Vec<usize>> {
        let n = self.devices.len();
        if n == 0 {
            bail!("backend.devices пуст");
        }
        if self.tiers.is_empty() {
            let base = 100 / n;
            let mut tiers = vec![base; n];
            tiers[n - 1] += 100 - base * n;
            Ok(tiers)
        } else {
            if self.tiers.len() != n {
                bail!(
                    "backend.tiers: {} значений на {} устройств",
                    self.tiers.len(),
                    n
                );
            }
            let sum: usize = self.tiers.iter().sum();
            if sum != 100 {
                bail!("backend.tiers: сумма {sum}, нужно 100");
            }
            Ok(self.tiers.clone())
        }
    }

    pub fn to_options(&self) -> lfm_gpu::Options {
        let mut o = lfm_gpu::Options::default();
        o.device = self.devices.first().copied().unwrap_or(0);
        o.profile = self.profile;
        o.kv = self.kv.clone();
        o.prepack = self.prepack;
        o.fuse = self.fuse;
        o.pb = self.pb;
        o.prefill_ntok = self.prefill_ntok;
        o.max_seqs = self.max_seqs;
        o.cache = self.cache;
        o.splits = self.splits;
        o.attn4 = self.attn4;
        o.attn2 = self.attn2;
        o.flash = self.flash;
        o.flash2 = self.flash2;
        o.b8p = self.b8p;
        o.i8pf = self.i8pf;
        o.q4i8 = self.q4i8;
        o.q6i8 = self.q6i8;
        o.q4i8_ffn_only = self.q4i8_ffn_only;
        o.wg = self.wg;
        o.wg_b = self.wg_b;
        o.rgemv = self.rgemv;
        o.rgemv_b = self.rgemv_b;
        o.prepack_threads = self.prepack_threads;
        o.program_cache = self.program_cache;
        o.kv_pool = self.kv_pool;
        o.gemm_force = self.gemm_force.clone();
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.backend.ctx, 4096);
        assert_eq!(cfg.backend.kv, "f16");
        assert_eq!(cfg.backend.mode, BackendMode::Single);
    }

    #[test]
    fn empty_tiers_split_evenly() {
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Pipeline;
        cfg.backend.devices = vec![0, 1];
        assert_eq!(cfg.backend.effective_tiers().unwrap(), vec![50, 50]);
        cfg.backend.devices = vec![0, 1, 2];
        assert_eq!(cfg.backend.effective_tiers().unwrap(), vec![33, 33, 34]);
    }

    #[test]
    fn bad_values_are_rejected() {
        let mut cfg = Config::default();
        cfg.backend.ctx = 100;
        assert!(cfg.validate().is_err());
        let mut cfg = Config::default();
        cfg.backend.kv = "q8".to_string();
        assert!(cfg.validate().is_err());
        let mut cfg = Config::default();
        cfg.backend.tiers = vec![60, 50];
        cfg.backend.devices = vec![0, 1];
        cfg.backend.mode = BackendMode::Pipeline;
        assert!(cfg.validate().is_err());
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Tp;
        cfg.backend.kv = "kvarn".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn toml_round_trip() {
        let cfg = Config::default();
        let text = cfg.to_toml().unwrap();
        let back = Config::from_toml(&text).unwrap();
        assert_eq!(back.backend.ctx, cfg.backend.ctx);
        assert_eq!(back.chat.system, cfg.chat.system);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(Config::from_toml("[backend]\nnope = 1\n").is_err());
    }
}
