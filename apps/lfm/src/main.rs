mod markdown;
mod serve;
mod tui;

use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use lfm::config::{BackendMode, Config, Overrides};
use lfm_domain::SamplingParams;

#[derive(Parser)]
#[command(
    name = "lfm",
    version,
    about = "LFM3: локальный чат и OpenAI-совместимый сервер",
    args_conflicts_with_subcommands = false
)]
struct Cli {
    #[arg(
        long,
        global = true,
        help = "TOML-конфигурация [lfm.toml, если существует]"
    )]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "консольный чат (TUI) или один запрос с --once")]
    Chat(ChatArgs),
    #[command(about = "OpenAI-совместимый сервер")]
    Serve(ServeArgs),
    #[command(about = "работа с конфигурацией")]
    Config(ConfigArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    Single,
    Pipeline,
    Tp,
}

#[derive(Args, Clone, Default)]
struct CommonArgs {
    #[arg(long, help = "GGUF-модель")]
    model: Option<PathBuf>,
    #[arg(long, help = "контекст в токенах (кратен 32)")]
    ctx: Option<usize>,
    #[arg(long, help = "тип KV: f16, q4, kvarn, kvarn_k4v4, k4v4")]
    kv: Option<String>,
    #[arg(long, value_delimiter = ',', help = "устройства: 0 или 0,1")]
    devices: Option<Vec<usize>>,
    #[arg(long, value_enum, help = "режим: single, pipeline, tp")]
    mode: Option<ModeArg>,
    #[arg(long, help = "доли слоёв для pipeline: 96/4")]
    tiers: Option<String>,
}

#[derive(Args)]
struct ChatArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, help = "системный промпт")]
    system: Option<String>,
    #[arg(long, help = "максимум новых токенов")]
    max_tokens: Option<usize>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    min_p: Option<f32>,
    #[arg(long)]
    repeat_penalty: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, help = "включить режим рассуждений")]
    thinking: bool,
    #[arg(long, help = "не показывать счётчики")]
    no_stats: bool,
    #[arg(long, help = "JSON-файл с определениями инструментов (OpenAI tools)")]
    tools: Option<PathBuf>,
    #[arg(long, help = "один запрос без TUI ('-' — из stdin)")]
    once: Option<String>,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, help = "адрес прослушивания")]
    addr: Option<String>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long, help = "одновременных последовательностей, 1..=32 [8]")]
    max_seqs: Option<usize>,
    #[arg(long)]
    prefill_budget: Option<usize>,
    #[arg(long)]
    prefill_in_decode: Option<usize>,
    #[arg(long, help = "логировать запросы")]
    log_requests: bool,
}

#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommand,
}

#[derive(Subcommand)]
enum ConfigCommand {
    #[command(about = "напечатать эффективный конфиг")]
    Print,
    #[command(about = "создать пример lfm.toml")]
    Init {
        #[arg(long, default_value = "lfm.toml")]
        output: PathBuf,
        #[arg(long, help = "перезаписать существующий файл")]
        force: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Chat(args) => chat(load_config(cli.config.as_deref())?, args),
        Command::Serve(args) => serve(load_config(cli.config.as_deref())?, args),
        Command::Config(args) => match args.command {
            ConfigCommand::Init { output, force } => init_config(&output, force),
            ConfigCommand::Print => print_config(load_config(cli.config.as_deref())?),
        },
    }
}

fn load_config(path: Option<&Path>) -> Result<Config> {
    if let Some(path) = path {
        return Config::load(path);
    }
    let default = Path::new("lfm.toml");
    if default.exists() {
        return Config::load(default);
    }
    Ok(Config::default())
}

fn common_overrides(common: &CommonArgs) -> Overrides {
    Overrides {
        model: common.model.clone(),
        ctx: common.ctx,
        kv: common.kv.clone(),
        devices: common.devices.clone(),
        mode: common.mode.map(|mode| match mode {
            ModeArg::Single => BackendMode::Single,
            ModeArg::Pipeline => BackendMode::Pipeline,
            ModeArg::Tp => BackendMode::Tp,
        }),
        tiers: common.tiers.clone(),
        ..Overrides::default()
    }
}

fn params(cfg: &Config) -> SamplingParams {
    SamplingParams {
        temperature: cfg.chat.temperature,
        top_k: cfg.chat.top_k,
        top_p: cfg.chat.top_p,
        min_p: cfg.chat.min_p,
        repeat_penalty: cfg.chat.repeat_penalty,
        seed: cfg.chat.seed,
        max_tokens: cfg.chat.max_tokens,
        ..SamplingParams::default()
    }
}

fn chat(mut cfg: Config, args: ChatArgs) -> Result<()> {
    let mut ov = common_overrides(&args.common);
    ov.system = args.system.clone();
    ov.max_tokens = args.max_tokens;
    ov.temperature = args.temperature;
    ov.top_k = args.top_k;
    ov.top_p = args.top_p;
    ov.min_p = args.min_p;
    ov.repeat_penalty = args.repeat_penalty;
    ov.seed = args.seed;
    ov.thinking = args.thinking.then_some(true);
    ov.tools = args.tools.clone();
    cfg.apply_overrides(&ov)?;
    cfg.validate()?;
    let mut chat = lfm::Chat::open(&cfg)?;
    let params = params(&cfg);
    let show_stats = cfg.ui.stats && !args.no_stats;
    let name = cfg
        .model
        .path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "lfm".to_string());
    let mode = match cfg.backend.mode {
        lfm::BackendMode::Single => "single".to_string(),
        lfm::BackendMode::Pipeline => format!(
            "pipeline {}",
            cfg.backend
                .devices
                .iter()
                .zip(cfg.backend.effective_tiers().unwrap_or_default())
                .map(|(d, p)| format!("{d}@{p}"))
                .collect::<Vec<_>>()
                .join("/")
        ),
        lfm::BackendMode::Tp => "tp".to_string(),
    };
    let info = format!("{mode} · {} · ctx {}", cfg.backend.kv, cfg.backend.ctx);
    if let Some(text) = args.once {
        let text = if text == "-" {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        } else {
            text
        };
        return tui::run_once(&mut chat, &params, &text, show_stats);
    }
    if !std::io::stdout().is_terminal() {
        bail!("stdout не терминал: используйте `lfm chat --once \"...\"` или запустите в TTY");
    }
    tui::run_chat(&mut chat, &name, &info, &params, show_stats)
}

fn serve(mut cfg: Config, args: ServeArgs) -> Result<()> {
    let mut ov = common_overrides(&args.common);
    ov.addr = args.addr;
    ov.name = args.name;
    ov.max_seqs = args.max_seqs;
    ov.prefill_budget = args.prefill_budget;
    ov.prefill_in_decode = args.prefill_in_decode;
    ov.log_requests = args.log_requests.then_some(true);
    cfg.apply_overrides(&ov)?;
    cfg.validate()?;
    serve::run(&cfg)
}

fn print_config(cfg: Config) -> Result<()> {
    print!("{}", cfg.to_toml()?);
    Ok(())
}

fn init_config(output: &Path, force: bool) -> Result<()> {
    if output.exists() && !force {
        bail!(
            "{} уже существует (--force для перезаписи)",
            output.display()
        );
    }
    std::fs::write(output, example_config())
        .with_context(|| format!("запись {}", output.display()))?;
    println!("создан {}", output.display());
    Ok(())
}

fn example_config() -> String {
    let body = Config::default().to_toml().unwrap_or_default();
    format!(
        "# lfm.toml — конфигурация LFM3\n\
         # backend.devices: индексы OpenCL-устройств; на текущей машине одно — 0 (Vega 8)\n\
         # backend.mode: single | pipeline | tp\n\
         #   single   — одно устройство (значение по умолчанию, безопасно)\n\
         #   pipeline — слои по устройствам, нужно минимум два устройства:\n\
         #              # devices = [0, 1]\n\
         #              # mode = \"pipeline\"\n\
         #              # tiers = [96, 4]\n\
         #   tp       — тензорный параллелизм, нужно 2..=4 одинаковых быстрых GPU и kv = \"f16\"\n\
         # backend.kv: f16 | q4 | k4v4 | kvarn | kvarn_k4v2 | kvarn_k4v4\n\
         # Приоритет: CLI > этот файл > значения по умолчанию\n\n{body}"
    )
}
