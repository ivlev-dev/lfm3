use anyhow::{Context, Result, anyhow};
use lfm_gpu::Options;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const DEFAULT_H2_CANCEL_INTERVAL: usize = 16;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub template: Option<String>,
    pub addr: Option<String>,
    pub ctx: Option<usize>,
    pub system: Option<String>,
    pub name: Option<String>,
    pub log_requests: Option<bool>,
    pub max_body: Option<usize>,
    pub prefill_budget: Option<usize>,
    #[serde(default)]
    pub server: ServerFile,
    #[serde(default)]
    pub engine: EngineFile,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFile {
    pub h2_cancel_interval: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineFile {
    pub device: Option<usize>,
    pub profile: Option<bool>,
    pub kv: Option<String>,
    pub prepack: Option<bool>,
    pub prepack_threads: Option<usize>,
    pub program_cache: Option<bool>,
    pub fuse: Option<bool>,
    pub pb: Option<usize>,
    pub prefill_ntok: Option<usize>,
    pub max_seqs: Option<usize>,
    pub cache: Option<bool>,
    pub splits: Option<usize>,
    pub attn4: Option<bool>,
    pub attn2: Option<bool>,
    pub flash: Option<bool>,
    pub flash2: Option<bool>,
    pub b8p: Option<bool>,
    pub i8pf: Option<bool>,
    pub q4i8: Option<bool>,
    pub q6i8: Option<bool>,
    pub wg: Option<usize>,
    pub wg_b: Option<usize>,
    pub rgemv: Option<usize>,
    pub rgemv_b: Option<usize>,
    pub gemm_force: Option<String>,
}

pub fn load(path: &Path) -> Result<FileConfig> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))
}

impl FileConfig {
    pub fn engine(&self, base: Options) -> Options {
        let mut o = base;
        let e = &self.engine;
        if let Some(v) = e.device {
            o.device = v;
        }
        if let Some(v) = e.profile {
            o.profile = v;
        }
        if let Some(v) = &e.kv {
            o.kv = v.clone();
        }
        if let Some(v) = e.prepack {
            o.prepack = v;
        }
        if let Some(v) = e.prepack_threads {
            o.prepack_threads = v;
        }
        if let Some(v) = e.program_cache {
            o.program_cache = v;
        }
        if let Some(v) = e.fuse {
            o.fuse = v;
        }
        if let Some(v) = e.pb {
            o.pb = v;
        }
        if let Some(v) = e.prefill_ntok {
            o.prefill_ntok = v;
        }
        if let Some(v) = e.max_seqs {
            o.max_seqs = v;
        }
        if let Some(v) = e.cache {
            o.cache = v;
        }
        if let Some(v) = e.splits {
            o.splits = v;
        }
        if let Some(v) = e.attn4 {
            o.attn4 = v;
        }
        if let Some(v) = e.attn2 {
            o.attn2 = v;
        }
        if let Some(v) = e.flash {
            o.flash = v;
        }
        if let Some(v) = e.flash2 {
            o.flash2 = v;
        }
        if let Some(v) = e.b8p {
            o.b8p = v;
        }
        if let Some(v) = e.i8pf {
            o.i8pf = v;
        }
        if let Some(v) = e.q4i8 {
            o.q4i8 = v;
        }
        if let Some(v) = e.q6i8 {
            o.q6i8 = v;
        }
        if let Some(v) = e.wg {
            o.wg = v / 64 * 64;
        }
        if let Some(v) = e.wg_b {
            o.wg_b = v / 64 * 64;
        }
        if let Some(v) = e.rgemv {
            o.rgemv = v;
        }
        if let Some(v) = e.rgemv_b {
            o.rgemv_b = v;
        }
        if let Some(v) = &e.gemm_force {
            o.gemm_force = if v.is_empty() { None } else { Some(v.clone()) };
        }
        o
    }
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub template: PathBuf,
    pub addr: String,
    pub ctx: usize,
    pub system: String,
    pub name: String,
    pub log_requests: bool,
    pub max_body: usize,
    pub prefill_budget: usize,
    pub h2_cancel_interval: usize,
    pub engine: Options,
}

pub struct Overrides {
    pub model: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub template: Option<PathBuf>,
    pub addr: Option<String>,
    pub ctx: Option<usize>,
    pub system: Option<String>,
    pub name: Option<String>,
    pub log_requests: Option<bool>,
    pub max_body: Option<usize>,
    pub prefill_budget: Option<usize>,
    pub h2_cancel_interval: Option<usize>,
}

pub fn resolve(file: &FileConfig, cli: &Overrides, engine: Options) -> Result<Resolved> {
    let model = cli
        .model
        .clone()
        .or_else(|| file.model.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models/LFM3-1.7B-Q4_K_M.gguf"));
    let tokenizer = cli
        .tokenizer
        .clone()
        .or_else(|| file.tokenizer.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models/hf/tokenizer.json"));
    let template = cli
        .template
        .clone()
        .or_else(|| file.template.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models/hf/chat_template.jinja"));
    let addr = cli
        .addr
        .clone()
        .or_else(|| file.addr.clone())
        .unwrap_or_else(|| "127.0.0.1:8000".to_string());
    let ctx = cli.ctx.or(file.ctx).unwrap_or(4096);
    let system = cli
        .system
        .clone()
        .or_else(|| file.system.clone())
        .unwrap_or_default();
    let name = cli
        .name
        .clone()
        .or_else(|| file.name.clone())
        .unwrap_or_else(|| "lfm3".to_string());
    let log_requests = cli.log_requests.or(file.log_requests).unwrap_or(false);
    let max_body = cli.max_body.or(file.max_body).unwrap_or(4 << 20);
    let prefill_budget = cli
        .prefill_budget
        .or(file.prefill_budget)
        .unwrap_or(64)
        .clamp(1, 128);
    let h2_cancel_interval = cli
        .h2_cancel_interval
        .or(file.server.h2_cancel_interval)
        .unwrap_or(DEFAULT_H2_CANCEL_INTERVAL);
    if ctx == 0 || !ctx.is_multiple_of(32) {
        return Err(anyhow!("ctx must be a positive multiple of 32, got {ctx}"));
    }
    if max_body < 1024 {
        return Err(anyhow!(
            "max_body must be at least 1024 bytes, got {max_body}"
        ));
    }
    if !(1..=1024).contains(&h2_cancel_interval) {
        return Err(anyhow!(
            "h2_cancel_interval must be in 1..=1024, got {h2_cancel_interval}"
        ));
    }
    engine
        .validate()
        .map_err(|e| anyhow!("bad engine options: {e}"))?;
    Ok(Resolved {
        model,
        tokenizer,
        template,
        addr,
        ctx,
        system,
        name,
        log_requests,
        max_body,
        prefill_budget,
        h2_cancel_interval,
        engine,
    })
}

pub fn dump(cfg: &Resolved, backend: &str) -> String {
    let e = &cfg.engine;
    let mut out = String::new();
    out.push_str("# lfm-server: эффективная конфигурация (приоритет: CLI > файл > env > дефолт)\n");
    out.push_str(&format!("model = \"{}\"\n", cfg.model.display()));
    out.push_str(&format!("tokenizer = \"{}\"\n", cfg.tokenizer.display()));
    out.push_str(&format!("template = \"{}\"\n", cfg.template.display()));
    out.push_str(&format!("addr = \"{}\"\n", cfg.addr));
    out.push_str(&format!("ctx = {}\n", cfg.ctx));
    out.push_str(&format!("name = \"{}\"\n", cfg.name));
    out.push_str(&format!(
        "system = \"{}\"\n",
        cfg.system.replace('"', "\\\"")
    ));
    out.push_str(&format!("log_requests = {}\n", cfg.log_requests));
    out.push_str(&format!("max_body = {}\n", cfg.max_body));
    out.push_str(&format!("prefill_budget = {}\n", cfg.prefill_budget));
    out.push_str(&format!(
        "# backend = \"{backend}\" (для справки, не читается)\n"
    ));
    out.push_str("\n[server]\n");
    out.push_str(&format!(
        "h2_cancel_interval = {}\n",
        cfg.h2_cancel_interval
    ));
    out.push_str("\n[engine]\n");
    out.push_str(&format!("device = {}\n", e.device));
    out.push_str(&format!("profile = {}\n", e.profile));
    out.push_str(&format!("kv = \"{}\"\n", e.kv));
    out.push_str(&format!("prepack = {}\n", e.prepack));
    out.push_str(&format!("prepack_threads = {}\n", e.prepack_threads));
    out.push_str(&format!("program_cache = {}\n", e.program_cache));
    out.push_str(&format!("fuse = {}\n", e.fuse));
    out.push_str(&format!("pb = {}\n", e.pb));
    out.push_str(&format!("prefill_ntok = {}\n", e.prefill_ntok));
    out.push_str(&format!("max_seqs = {}\n", e.max_seqs));
    out.push_str(&format!("cache = {}\n", e.cache));
    out.push_str(&format!("splits = {}\n", e.splits));
    out.push_str(&format!("attn4 = {}\n", e.attn4));
    out.push_str(&format!("attn2 = {}\n", e.attn2));
    out.push_str(&format!("flash = {}\n", e.flash));
    out.push_str(&format!("flash2 = {}\n", e.flash2));
    out.push_str(&format!("b8p = {}\n", e.b8p));
    out.push_str(&format!("i8pf = {}\n", e.i8pf));
    out.push_str(&format!("q4i8 = {}\n", e.q4i8));
    out.push_str(&format!("q6i8 = {}\n", e.q6i8));
    out.push_str(&format!("wg = {}\n", e.wg));
    out.push_str(&format!("wg_b = {}\n", e.wg_b));
    out.push_str(&format!("rgemv = {}\n", e.rgemv));
    out.push_str(&format!("rgemv_b = {}\n", e.rgemv_b));
    out.push_str(&format!(
        "gemm_force = \"{}\"\n",
        e.gemm_force.clone().unwrap_or_default()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> FileConfig {
        FileConfig::default()
    }

    #[test]
    fn parses_engine_section() {
        let cfg: FileConfig = toml::from_str(
            r#"
model = "m.gguf"
ctx = 2048
[engine]
kv = "q4"
pb = 8
flash = false
wg = 128
"#,
        )
        .unwrap();
        assert_eq!(cfg.model.as_deref(), Some("m.gguf"));
        assert_eq!(cfg.ctx, Some(2048));
        let o = cfg.engine(Options::default());
        assert_eq!(o.kv, "q4");
        assert_eq!(o.pb, 8);
        assert!(!o.flash);
        assert_eq!(o.wg, 128);
    }

    #[test]
    fn rejects_unknown_keys() {
        let parsed: Result<FileConfig, _> = toml::from_str("[engine]\nnope = 1\n");
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_wins_over_file() {
        let mut cfg = empty();
        cfg.addr = Some("0.0.0.0:9000".to_string());
        cfg.ctx = Some(1024);
        let cli = Overrides {
            model: None,
            tokenizer: None,
            template: None,
            addr: Some("127.0.0.1:1".to_string()),
            ctx: None,
            system: None,
            name: None,
            log_requests: Some(true),
            max_body: None,
            prefill_budget: None,
            h2_cancel_interval: None,
        };
        let r = resolve(&cfg, &cli, Options::default()).unwrap();
        assert_eq!(r.addr, "127.0.0.1:1");
        assert_eq!(r.ctx, 1024);
        assert!(r.log_requests);
        assert_eq!(r.name, "lfm3");
    }

    #[test]
    fn rejects_bad_ctx() {
        let mut cfg = empty();
        cfg.ctx = Some(1000);
        let cli = Overrides {
            model: None,
            tokenizer: None,
            template: None,
            addr: None,
            ctx: None,
            system: None,
            name: None,
            log_requests: None,
            max_body: None,
            prefill_budget: None,
            h2_cancel_interval: None,
        };
        assert!(resolve(&cfg, &cli, Options::default()).is_err());
    }

    #[test]
    fn server_section_sets_h2_cancel_interval() {
        let cfg: FileConfig = toml::from_str("[server]\nh2_cancel_interval = 4\n").unwrap();
        assert_eq!(cfg.server.h2_cancel_interval, Some(4));
        let cli = Overrides {
            model: None,
            tokenizer: None,
            template: None,
            addr: None,
            ctx: None,
            system: None,
            name: None,
            log_requests: None,
            max_body: None,
            prefill_budget: None,
            h2_cancel_interval: None,
        };
        let r = resolve(&cfg, &cli, Options::default()).unwrap();
        assert_eq!(r.h2_cancel_interval, 4);
        let dump = dump(&r, "opencl");
        assert!(
            dump.contains("[server]\nh2_cancel_interval = 4\n"),
            "{dump}"
        );
        let round_trip: FileConfig = toml::from_str(&dump).unwrap();
        assert_eq!(round_trip.server.h2_cancel_interval, Some(4));
    }

    #[test]
    fn h2_cancel_interval_defaults_and_cli_priority() {
        let cfg = empty();
        let mut cli = Overrides {
            model: None,
            tokenizer: None,
            template: None,
            addr: None,
            ctx: None,
            system: None,
            name: None,
            log_requests: None,
            max_body: None,
            prefill_budget: None,
            h2_cancel_interval: None,
        };
        let r = resolve(&cfg, &cli, Options::default()).unwrap();
        assert_eq!(r.h2_cancel_interval, DEFAULT_H2_CANCEL_INTERVAL);
        cli.h2_cancel_interval = Some(8);
        let r = resolve(&cfg, &cli, Options::default()).unwrap();
        assert_eq!(r.h2_cancel_interval, 8);
        cli.h2_cancel_interval = Some(0);
        assert!(resolve(&cfg, &cli, Options::default()).is_err());
    }

    #[test]
    fn rejects_unknown_server_keys() {
        let parsed: Result<FileConfig, _> = toml::from_str("[server]\nnope = 1\n");
        assert!(parsed.is_err());
    }
}
