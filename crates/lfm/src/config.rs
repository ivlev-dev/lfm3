use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const DEFAULT_H2_CANCEL_INTERVAL: usize = 16;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: ModelSection,
    pub backend: BackendSection,
    pub chat: ChatSection,
    pub server: ServerSection,
    pub ui: UiSection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiSection {
    pub stats: bool,
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
            max_tokens: 0,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.0,
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

/// Переопределения конфигурации из CLI.
///
/// Ядро приоритета `CLI > TOML > default`: `None` означает «флаг не передан»,
/// и значение остаётся таким, каким его дал TOML или `Default`.
/// Для булевых флагов CLI (`--thinking`, `--log-requests`) `Some(true)` —
/// «флаг передан», `None` — «не передан».
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub model: Option<PathBuf>,
    pub ctx: Option<usize>,
    pub kv: Option<String>,
    pub devices: Option<Vec<usize>>,
    pub mode: Option<BackendMode>,
    /// Строка CLI вида `96/4`.
    pub tiers: Option<String>,
    pub max_seqs: Option<usize>,
    pub addr: Option<String>,
    pub name: Option<String>,
    pub prefill_budget: Option<usize>,
    pub prefill_in_decode: Option<usize>,
    pub log_requests: Option<bool>,
    pub system: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub thinking: Option<bool>,
    pub tools: Option<PathBuf>,
}

/// Разбирает доли `tiers` из CLI: `96/4`.
pub fn parse_tiers(spec: &str) -> Result<Vec<usize>> {
    spec.split('/')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("доли tiers {spec:?}"))
        })
        .collect()
}

impl Config {
    /// Накладывает CLI поверх текущего конфига. Валидацию не выполняет —
    /// вызывающий обязан вызвать [`Config::validate`] после всех правок.
    pub fn apply_overrides(&mut self, ov: &Overrides) -> Result<()> {
        if let Some(v) = &ov.model {
            self.model.path = v.clone();
        }
        if let Some(v) = ov.ctx {
            self.backend.ctx = v;
        }
        if let Some(v) = &ov.kv {
            self.backend.kv = v.clone();
        }
        if let Some(v) = &ov.devices {
            self.backend.devices = v.clone();
        }
        if let Some(v) = ov.mode {
            self.backend.mode = v;
        }
        if let Some(v) = &ov.tiers {
            self.backend.tiers = parse_tiers(v)?;
        }
        if let Some(v) = ov.max_seqs {
            self.backend.max_seqs = v;
        }
        if let Some(v) = &ov.addr {
            self.server.addr = v.clone();
        }
        if let Some(v) = &ov.name {
            self.server.name = v.clone();
        }
        if let Some(v) = ov.prefill_budget {
            self.server.prefill_budget = v;
        }
        if let Some(v) = ov.prefill_in_decode {
            self.server.prefill_in_decode = v;
        }
        if let Some(v) = ov.log_requests {
            self.server.log_requests = v;
        }
        if let Some(v) = &ov.system {
            self.chat.system = v.clone();
        }
        if let Some(v) = ov.max_tokens {
            self.chat.max_tokens = v;
        }
        if let Some(v) = ov.temperature {
            self.chat.temperature = v;
        }
        if let Some(v) = ov.top_k {
            self.chat.top_k = v;
        }
        if let Some(v) = ov.top_p {
            self.chat.top_p = v;
        }
        if let Some(v) = ov.min_p {
            self.chat.min_p = v;
        }
        if let Some(v) = ov.repeat_penalty {
            self.chat.repeat_penalty = v;
        }
        if let Some(v) = ov.seed {
            self.chat.seed = v;
        }
        if let Some(v) = ov.thinking {
            self.chat.thinking = v;
        }
        if let Some(v) = &ov.tools {
            self.chat.tools = Some(v.clone());
        }
        Ok(())
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
        if !(1..=32).contains(&self.backend.max_seqs) {
            bail!("backend.max_seqs должен быть в 1..=32");
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

    /// Маппинг `[backend]` в `lfm_gpu::Options`.
    ///
    /// В `Options` не входят поля, которые потребляет не бэкенд, а вызывающий
    /// код: `kind` (проверяется в `validate`), `mode`, `tiers`, `ctx` (передаётся
    /// отдельным аргументом в `Runtime::open`), а из `devices` берётся только
    /// первое устройство (`device`) — остальные использует pipeline/tp.
    pub fn to_options(&self) -> lfm_gpu::Options {
        lfm_gpu::Options {
            device: self.devices.first().copied().unwrap_or(0),
            profile: self.profile,
            kv: self.kv.clone(),
            prepack: self.prepack,
            fuse: self.fuse,
            pb: self.pb,
            prefill_ntok: self.prefill_ntok,
            max_seqs: self.max_seqs,
            cache: self.cache,
            splits: self.splits,
            attn4: self.attn4,
            attn2: self.attn2,
            flash: self.flash,
            flash2: self.flash2,
            b8p: self.b8p,
            i8pf: self.i8pf,
            q4i8: self.q4i8,
            q6i8: self.q6i8,
            q4i8_ffn_only: self.q4i8_ffn_only,
            wg: self.wg,
            wg_b: self.wg_b,
            rgemv: self.rgemv,
            rgemv_b: self.rgemv_b,
            prepack_threads: self.prepack_threads,
            program_cache: self.program_cache,
            kv_pool: self.kv_pool,
            gemm_force: self.gemm_force.clone(),
        }
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
        assert_eq!(cfg.chat.max_tokens, 0, "0 = до EOS/контекста");
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

    fn err_of(cfg: &Config) -> String {
        cfg.validate()
            .expect_err("ожидалась ошибка валидации")
            .to_string()
    }

    fn assert_invalid(cfg: &Config, needle: &str) {
        let err = err_of(cfg);
        assert!(
            err.contains(needle),
            "ожидалось {needle:?} в ошибке, получено: {err}"
        );
    }

    #[test]
    fn overrides_precedence_cli_over_toml_over_default() {
        // default
        let mut cfg = Config::default();
        assert_eq!(cfg.backend.ctx, 4096);
        assert_eq!(cfg.backend.kv, "f16");

        // TOML поверх default
        cfg = Config::from_toml("[backend]\nctx = 8192\nkv = \"kvarn\"\n").unwrap();
        assert_eq!(cfg.backend.ctx, 8192);
        assert_eq!(cfg.backend.kv, "kvarn");

        // пустой набор оверрайдов не трогает TOML
        cfg.apply_overrides(&Overrides::default()).unwrap();
        assert_eq!(
            cfg.backend.ctx, 8192,
            "без CLI должно остаться значение TOML"
        );
        assert_eq!(cfg.backend.kv, "kvarn");
        assert_eq!(cfg.backend.max_seqs, 8, "не заданное нигде — default");

        // CLI поверх TOML
        let ov = Overrides {
            ctx: Some(2048),
            kv: Some("q4".to_string()),
            ..Overrides::default()
        };
        cfg.apply_overrides(&ov).unwrap();
        assert_eq!(cfg.backend.ctx, 2048);
        assert_eq!(cfg.backend.kv, "q4");
        cfg.validate().unwrap();
    }

    #[test]
    fn overrides_hit_their_own_sections() {
        let mut cfg = Config::default();
        let ov = Overrides {
            model: Some(PathBuf::from("/models/other.gguf")),
            ctx: Some(32768),
            kv: Some("kvarn".to_string()),
            devices: Some(vec![0, 1]),
            mode: Some(BackendMode::Pipeline),
            tiers: Some("96/4".to_string()),
            max_seqs: Some(3),
            addr: Some("0.0.0.0:9000".to_string()),
            name: Some("other-name".to_string()),
            prefill_budget: Some(32),
            prefill_in_decode: Some(0),
            log_requests: Some(true),
            system: Some("sys".to_string()),
            max_tokens: Some(64),
            temperature: Some(0.0),
            top_k: Some(7),
            top_p: Some(0.5),
            min_p: Some(0.01),
            repeat_penalty: Some(1.3),
            seed: Some(11),
            thinking: Some(true),
            tools: Some(PathBuf::from("/tmp/tools.json")),
        };
        cfg.apply_overrides(&ov).unwrap();
        cfg.validate().unwrap();

        // backend
        assert_eq!(cfg.backend.ctx, 32768);
        assert_eq!(cfg.backend.kv, "kvarn");
        assert_eq!(cfg.backend.devices, vec![0, 1]);
        assert_eq!(cfg.backend.mode, BackendMode::Pipeline);
        assert_eq!(cfg.backend.tiers, vec![96, 4]);
        assert_eq!(cfg.backend.effective_tiers().unwrap(), vec![96, 4]);
        assert_eq!(cfg.backend.max_seqs, 3);
        // server
        assert_eq!(cfg.server.addr, "0.0.0.0:9000");
        assert_eq!(cfg.server.name, "other-name");
        assert_eq!(cfg.server.prefill_budget, 32);
        assert_eq!(cfg.server.prefill_in_decode, 0);
        assert!(cfg.server.log_requests);
        // chat
        assert_eq!(cfg.chat.system, "sys");
        assert_eq!(cfg.chat.max_tokens, 64);
        assert_eq!(cfg.chat.temperature, 0.0);
        assert_eq!(cfg.chat.top_k, 7);
        assert_eq!(cfg.chat.top_p, 0.5);
        assert_eq!(cfg.chat.min_p, 0.01);
        assert_eq!(cfg.chat.repeat_penalty, 1.3);
        assert_eq!(cfg.chat.seed, 11);
        assert!(cfg.chat.thinking);
        assert_eq!(
            cfg.chat.tools.as_deref(),
            Some(Path::new("/tmp/tools.json"))
        );
        // model
        assert_eq!(cfg.model.path, PathBuf::from("/models/other.gguf"));
        // чужие поля не тронуты
        assert_eq!(cfg.server.max_body, 4 << 20);
        assert_eq!(cfg.server.h2_cancel_interval, 16);
        assert!(cfg.ui.stats);
    }

    #[test]
    fn overrides_bad_tiers_leave_config_untouched() {
        let mut cfg = Config::default();
        cfg.backend.tiers = vec![50, 50];
        let ov = Overrides {
            tiers: Some("96/x".to_string()),
            ..Overrides::default()
        };
        assert!(cfg.apply_overrides(&ov).is_err());
        assert_eq!(cfg.backend.tiers, vec![50, 50]);
    }

    #[test]
    fn tiers_spec_parsing() {
        assert_eq!(parse_tiers("96/4").unwrap(), vec![96, 4]);
        assert_eq!(parse_tiers(" 100 ").unwrap(), vec![100]);
        assert!(parse_tiers("96/x").is_err());
        assert!(parse_tiers("").is_err());
    }

    /// Все поля `[backend]`, которые бэкенд реально принимает, обязаны доехать
    /// до `Options` без потерь и без перестановок. Значения подобраны так, чтобы
    /// каждое отличалось от дефолта: перепутанное поле сразу ловится.
    #[test]
    fn to_options_maps_every_backend_field() {
        let backend = BackendSection {
            kind: "opencl".to_string(),
            devices: vec![3, 7],
            mode: BackendMode::Pipeline,
            tiers: vec![60, 40],
            kv: "kvarn_k4v4".to_string(),
            ctx: 8192,
            kv_pool: 4,
            max_seqs: 5,
            prefill_ntok: 32,
            pb: 9,
            splits: 7,
            cache: false,
            fuse: false,
            attn4: false,
            attn2: false,
            flash: false,
            flash2: false,
            b8p: false,
            i8pf: false,
            q4i8: false,
            q6i8: false,
            q4i8_ffn_only: true,
            wg: 128,
            wg_b: 32,
            rgemv: 8,
            rgemv_b: 2,
            prepack: false,
            prepack_threads: 12,
            program_cache: false,
            gemm_force: Some("vec".to_string()),
            profile: true,
        };
        let opts = backend.to_options();

        assert_eq!(opts.device, 3, "device = devices[0]");
        assert_eq!(opts.kv, "kvarn_k4v4");
        assert_eq!(opts.kv_pool, 4);
        assert_eq!(opts.max_seqs, 5);
        assert_eq!(opts.prefill_ntok, 32);
        assert_eq!(opts.pb, 9);
        assert_eq!(opts.splits, 7);
        assert_eq!(opts.wg, 128);
        assert_eq!(opts.wg_b, 32);
        assert_eq!(opts.rgemv, 8);
        assert_eq!(opts.rgemv_b, 2);
        assert_eq!(opts.prepack_threads, 12);
        assert_eq!(opts.gemm_force.as_deref(), Some("vec"));
        // булевы: все выставлены в противоположное дефолту значение
        assert!(opts.profile);
        assert!(!opts.prepack);
        assert!(!opts.fuse);
        assert!(!opts.cache);
        assert!(!opts.attn4);
        assert!(!opts.attn2);
        assert!(!opts.flash);
        assert!(!opts.flash2);
        assert!(!opts.b8p);
        assert!(!opts.i8pf);
        assert!(!opts.q4i8);
        assert!(!opts.q6i8);
        assert!(opts.q4i8_ffn_only);
        assert!(!opts.program_cache);

        // `Options` — 27 полей, все проверены выше. Не мапятся сознательно:
        // kind, mode, tiers, ctx (их читает Runtime/validate), devices[1..]
        // (pipeline/tp берут список устройств напрямую).
    }

    #[test]
    fn backend_defaults_match_options_defaults() {
        assert_eq!(
            BackendSection::default().to_options(),
            lfm_gpu::Options::default(),
            "дефолты конфига и бэкенда разъехались"
        );
    }

    #[test]
    fn validation_rejects_bad_devices_and_modes() {
        let mut cfg = Config::default();
        cfg.backend.devices.clear();
        assert_invalid(&cfg, "devices пуст");

        // pipeline требует минимум 2 устройства
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Pipeline;
        assert_invalid(&cfg, "минимум 2 устройства");

        // pipeline: tiers обязаны совпасть по длине с devices
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Pipeline;
        cfg.backend.devices = vec![0, 1];
        cfg.backend.tiers = vec![100];
        assert_invalid(&cfg, "значений на");

        // tp: не больше 4 устройств
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Tp;
        cfg.backend.devices = vec![0, 1, 2, 3, 4];
        assert_invalid(&cfg, "не больше 4 устройств");

        // tp: только f16
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Tp;
        cfg.backend.devices = vec![0, 1];
        cfg.backend.kv = "kvarn".to_string();
        assert_invalid(&cfg, "только kv = f16");

        // неизвестный бэкенд
        let mut cfg = Config::default();
        cfg.backend.kind = "cuda".to_string();
        assert_invalid(&cfg, "не поддерживается");

        // положительный контроль: валидный tp проходит
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Tp;
        cfg.backend.devices = vec![0, 1, 2, 3];
        cfg.validate().unwrap();
    }

    #[test]
    fn validation_rejects_bad_tiers_and_ctx() {
        // сумма tiers != 100
        let mut cfg = Config::default();
        cfg.backend.mode = BackendMode::Pipeline;
        cfg.backend.devices = vec![0, 1];
        cfg.backend.tiers = vec![60, 30];
        assert_invalid(&cfg, "нужно 100");

        // процент > 100
        let mut cfg = Config::default();
        cfg.backend.tiers = vec![101];
        assert_invalid(&cfg, "0..=100");

        // ctx = 0 и ctx не кратен 32
        let mut cfg = Config::default();
        cfg.backend.ctx = 0;
        assert_invalid(&cfg, "кратным 32");
        let mut cfg = Config::default();
        cfg.backend.ctx = 100;
        assert_invalid(&cfg, "кратным 32");

        // положительный контроль: 32 и 4096 проходят
        for ctx in [32, 4096] {
            let mut cfg = Config::default();
            cfg.backend.ctx = ctx;
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn validation_rejects_bad_backend_limits() {
        fn check(needle: &str, mutate: impl FnOnce(&mut Config)) {
            let mut cfg = Config::default();
            mutate(&mut cfg);
            let err = err_of(&cfg);
            assert!(err.contains(needle), "{needle}: неожиданная ошибка {err}");
        }
        check("max_seqs", |c| c.backend.max_seqs = 0);
        check("max_seqs", |c| c.backend.max_seqs = 33);
        check("kv_pool", |c| c.backend.kv_pool = 0);
        check("prefill_ntok", |c| c.backend.prefill_ntok = 129);
        check("pb", |c| c.backend.pb = 13);
        check("splits", |c| c.backend.splits = 0);
    }

    #[test]
    fn validation_rejects_bad_server_limits() {
        // max_body < 1024
        let mut cfg = Config::default();
        cfg.server.max_body = 1023;
        assert_invalid(&cfg, "не меньше 1024 байт");

        // prefill_in_decode > 64
        let mut cfg = Config::default();
        cfg.server.prefill_in_decode = 65;
        assert_invalid(&cfg, "0..=64");

        // prefill_budget вне 1..=128
        let mut cfg = Config::default();
        cfg.server.prefill_budget = 129;
        assert_invalid(&cfg, "1..=128");

        // h2_cancel_interval вне 1..=1024
        let mut cfg = Config::default();
        cfg.server.h2_cancel_interval = 0;
        assert_invalid(&cfg, "1..=1024");

        // положительные границы проходят
        let mut cfg = Config::default();
        cfg.server.max_body = 1024;
        cfg.server.prefill_in_decode = 64;
        cfg.server.prefill_budget = 128;
        cfg.server.h2_cancel_interval = 1024;
        cfg.validate().unwrap();
    }

    fn fully_custom_config() -> Config {
        let mut cfg = Config::default();
        cfg.model.path = PathBuf::from("/models/other.gguf");
        cfg.model.tokenizer = PathBuf::from("/models/tokenizer.json");
        cfg.model.template = PathBuf::from("/models/chat.jinja");
        cfg.backend.devices = vec![0, 1];
        cfg.backend.mode = BackendMode::Pipeline;
        cfg.backend.tiers = vec![96, 4];
        cfg.backend.kv = "kvarn_k4v4".to_string();
        cfg.backend.ctx = 8192;
        cfg.backend.kv_pool = 2;
        cfg.backend.max_seqs = 4;
        cfg.backend.prefill_ntok = 64;
        cfg.backend.pb = 8;
        cfg.backend.splits = 8;
        cfg.backend.cache = false;
        cfg.backend.fuse = false;
        cfg.backend.attn4 = false;
        cfg.backend.attn2 = false;
        cfg.backend.flash = false;
        cfg.backend.flash2 = false;
        cfg.backend.b8p = false;
        cfg.backend.i8pf = false;
        cfg.backend.q4i8 = false;
        cfg.backend.q6i8 = false;
        cfg.backend.q4i8_ffn_only = true;
        cfg.backend.wg = 128;
        cfg.backend.wg_b = 32;
        cfg.backend.rgemv = 2;
        cfg.backend.rgemv_b = 3;
        cfg.backend.prepack = false;
        cfg.backend.prepack_threads = 4;
        cfg.backend.program_cache = false;
        cfg.backend.gemm_force = Some("vec".to_string());
        cfg.backend.profile = true;
        cfg.chat.system = "sys".to_string();
        cfg.chat.max_tokens = 128;
        cfg.chat.temperature = 0.3;
        cfg.chat.top_k = 10;
        cfg.chat.top_p = 0.8;
        cfg.chat.min_p = 0.05;
        cfg.chat.repeat_penalty = 1.2;
        cfg.chat.seed = 42;
        cfg.chat.thinking = true;
        cfg.chat.tools = Some(PathBuf::from("/tmp/tools.json"));
        cfg.server.addr = "0.0.0.0:9090".to_string();
        cfg.server.name = "other".to_string();
        cfg.server.max_body = 1 << 20;
        cfg.server.prefill_budget = 32;
        cfg.server.prefill_in_decode = 0;
        cfg.server.h2_cancel_interval = 8;
        cfg.server.log_requests = true;
        cfg.server.system = "server sys".to_string();
        cfg.ui.stats = false;
        cfg
    }

    #[test]
    fn toml_round_trip_preserves_every_field() {
        let cfg = fully_custom_config();
        let text = cfg.to_toml().unwrap();
        // каждый непустой ключ реально попал в текст
        for key in [
            "gemm_force",
            "tiers",
            "prepack_threads",
            "h2_cancel_interval",
            "q4i8_ffn_only",
        ] {
            assert!(text.contains(key), "в TOML нет ключа {key}");
        }
        let back = Config::from_toml(&text).unwrap();
        assert_eq!(back, cfg, "round-trip потерял или исказил поля");
    }

    #[test]
    fn defaults_match_documented_values() {
        let cfg = Config::default();
        // [model]
        assert_eq!(
            cfg.model.path,
            PathBuf::from("models/LFM3-1.7B-Q4_K_M.gguf")
        );
        assert_eq!(
            cfg.model.tokenizer,
            PathBuf::from("models/hf/tokenizer.json")
        );
        assert_eq!(
            cfg.model.template,
            PathBuf::from("models/hf/chat_template.jinja")
        );
        // [backend]
        assert_eq!(cfg.backend.kind, "opencl");
        assert_eq!(cfg.backend.devices, vec![0]);
        assert_eq!(cfg.backend.mode, BackendMode::Single);
        assert!(cfg.backend.tiers.is_empty(), "пусто = поровну");
        assert_eq!(cfg.backend.kv, "f16");
        assert_eq!(cfg.backend.ctx, 4096);
        assert_eq!(cfg.backend.kv_pool, 1);
        assert_eq!(cfg.backend.max_seqs, 8);
        assert_eq!(cfg.backend.prefill_ntok, 128);
        assert_eq!(cfg.backend.pb, 6);
        assert_eq!(cfg.backend.splits, 16);
        assert!(cfg.backend.cache);
        assert!(cfg.backend.fuse);
        assert!(cfg.backend.attn4 && cfg.backend.attn2);
        assert!(cfg.backend.flash && cfg.backend.flash2);
        assert!(cfg.backend.b8p);
        assert!(cfg.backend.i8pf);
        assert!(cfg.backend.q4i8 && cfg.backend.q6i8);
        assert!(!cfg.backend.q4i8_ffn_only);
        assert_eq!((cfg.backend.wg, cfg.backend.wg_b), (256, 64));
        assert_eq!((cfg.backend.rgemv, cfg.backend.rgemv_b), (4, 5));
        assert!(cfg.backend.prepack);
        assert_eq!(cfg.backend.prepack_threads, 0);
        assert!(cfg.backend.program_cache);
        assert_eq!(cfg.backend.gemm_force, None);
        assert!(!cfg.backend.profile);
        // [chat]
        assert_eq!(cfg.chat.system, "You are a helpful assistant.");
        assert_eq!(cfg.chat.max_tokens, 0);
        assert_eq!(cfg.chat.temperature, 0.7);
        assert_eq!(cfg.chat.top_k, 40);
        assert_eq!(cfg.chat.top_p, 0.95);
        assert_eq!(cfg.chat.min_p, 0.0);
        assert_eq!(cfg.chat.repeat_penalty, 1.0);
        assert_eq!(cfg.chat.seed, 0);
        assert!(!cfg.chat.thinking);
        assert_eq!(cfg.chat.tools, None);
        // [server]
        assert_eq!(cfg.server.addr, "127.0.0.1:8000");
        assert_eq!(cfg.server.name, "lfm3");
        assert_eq!(cfg.server.max_body, 4194304);
        assert_eq!(cfg.server.prefill_budget, 64);
        assert_eq!(cfg.server.prefill_in_decode, 8);
        assert_eq!(cfg.server.h2_cancel_interval, 16);
        assert_eq!(cfg.server.h2_cancel_interval, DEFAULT_H2_CANCEL_INTERVAL);
        assert!(!cfg.server.log_requests);
        assert_eq!(cfg.server.system, "You are a helpful assistant.");
        // [ui]
        assert!(cfg.ui.stats);
    }
}
