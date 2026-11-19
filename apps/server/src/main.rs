mod config;
mod engine;
mod h2;
mod hpack;
mod hpack_tables;
mod http;
mod protocol;
mod ws;

use anyhow::Result;
use clap::Parser;
use engine::{ChatJob, Engine, Event, ServeConfig};
use http::{Request, end_sse, read_request, start_sse, write_event, write_json};
use protocol::{ChatRequest, CompletionMeta, CompletionRequest, metrics_json, models_json};
use serde_json::json;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "lfm-server", about = "OpenAI-совместимый сервер LFM3 (OpenCL)")]
struct Cli {
    #[arg(long, help = "TOML-конфигурация (см. docs/config.md)")]
    config: Option<PathBuf>,
    #[arg(long, help = "напечатать эффективную конфигурацию и выйти")]
    print_config: bool,
    #[arg(long, help = "GGUF-модель [models/LFM3-1.7B-Q4_K_M.gguf]")]
    model: Option<PathBuf>,
    #[arg(long, help = "tokenizer.json [models/hf/tokenizer.json]")]
    tokenizer: Option<PathBuf>,
    #[arg(long, help = "chat_template.jinja [models/hf/chat_template.jinja]")]
    template: Option<PathBuf>,
    #[arg(long, help = "адрес [127.0.0.1:8000]")]
    addr: Option<String>,
    #[arg(long, help = "контекст в токенах, кратный 32 [4096]")]
    ctx: Option<usize>,
    #[arg(long, help = "системный промпт по умолчанию []")]
    system: Option<String>,
    #[arg(long, help = "имя модели в API [lfm3]")]
    name: Option<String>,
    #[arg(long, help = "логировать запросы (access log + планировщик)")]
    log_requests: bool,
    #[arg(long, help = "максимальный размер тела запроса в байтах [4194304]")]
    max_body: Option<usize>,
    #[arg(long, help = "бюджет prefill-токенов на шаг, 1..=128 [64]")]
    prefill_budget: Option<usize>,
    #[arg(long, help = "индекс OpenCL-устройства [0]")]
    device: Option<usize>,
    #[arg(long, help = "тип KV-кэша: f16, q4, kvarn (=kvarn_k4v2), kvarn_k4v4 [f16]")]
    kv: Option<String>,
    #[arg(long, help = "порог батчевого GEMV, 1..=12 [6]")]
    pb: Option<usize>,
    #[arg(long, help = "размер чанка prefill, 1..=128 [128]")]
    prefill_ntok: Option<usize>,
    #[arg(long, help = "максимум одновременных последовательностей, 1..=16 [8]")]
    max_seqs: Option<usize>,
    #[arg(long, help = "максимум сплитов внимания, 1..=16 [16]")]
    splits: Option<usize>,
    #[arg(long, help = "отключить кэш префиксов")]
    no_cache: bool,
    #[arg(long, help = "собирать тайминги ядер (LFM_PROFILE)")]
    profile: bool,
}

fn backend_name() -> String {
    match std::env::var("LFM_TP") {
        Ok(v) if !v.is_empty() => format!("tp:{v}"),
        _ => match std::env::var("LFM_TIERS") {
            Ok(v) if !v.is_empty() => format!("pipeline:{v}"),
            _ => "opencl".to_string(),
        },
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let file = match &cli.config {
        Some(path) => config::load(path)?,
        None => config::FileConfig::default(),
    };
    let mut engine_opts = file.engine(lfm_gpu::Options::from_env());
    if let Some(v) = cli.device {
        engine_opts.device = v;
    }
    if let Some(v) = &cli.kv {
        engine_opts.kv = v.clone();
    }
    if let Some(v) = cli.pb {
        engine_opts.pb = v;
    }
    if let Some(v) = cli.prefill_ntok {
        engine_opts.prefill_ntok = v;
    }
    if let Some(v) = cli.max_seqs {
        engine_opts.max_seqs = v;
    }
    if let Some(v) = cli.splits {
        engine_opts.splits = v;
    }
    if cli.no_cache {
        engine_opts.cache = false;
    }
    if cli.profile {
        engine_opts.profile = true;
    }
    let overrides = config::Overrides {
        model: cli.model.clone(),
        tokenizer: cli.tokenizer.clone(),
        template: cli.template.clone(),
        addr: cli.addr.clone(),
        ctx: cli.ctx,
        system: cli.system.clone(),
        name: cli.name.clone(),
        log_requests: cli.log_requests.then_some(true),
        max_body: cli.max_body,
        prefill_budget: cli.prefill_budget,
    };
    let cfg = config::resolve(&file, &overrides, engine_opts)?;
    let backend_name = backend_name();
    if cli.print_config {
        print!("{}", config::dump(&cfg, &backend_name));
        return Ok(());
    }
    let opts = cfg.engine.clone();
    let backend: lfm_app::BackendLoader = match std::env::var("LFM_TP") {
        Ok(devices) if !devices.is_empty() => Box::new(move |path, ctx| {
            let devices: Vec<usize> = devices
                .split('/')
                .filter_map(|d| d.parse::<usize>().ok())
                .collect();
            Ok(Box::new(lfm_gpu::TpModel::load(path, ctx, &devices)?))
        }),
        _ => match std::env::var("LFM_TIERS") {
            Ok(tiers) if !tiers.is_empty() => Box::new(move |path, ctx| {
                let percents: Vec<usize> = tiers
                    .split('/')
                    .filter_map(|p| p.parse::<usize>().ok())
                    .collect();
                let plan: Vec<(usize, usize)> = percents
                    .iter()
                    .enumerate()
                    .map(|(device, &percent)| (device, percent))
                    .collect();
                Ok(Box::new(lfm_gpu::PipelineModel::load_tiers(
                    path, ctx, &plan,
                )?))
            }),
            _ => Box::new(move |path, ctx| {
                Ok(Box::new(lfm_gpu::GpuModel::load_with(path, ctx, &opts)?))
            }),
        },
    };
    let tokenizer: lfm_app::TokenizerLoader =
        Box::new(|path| Ok(Box::new(lfm_tokenizer::LfmTokenizer::from_file(path)?)));
    let engine = Engine::spawn(
        ServeConfig {
            model: cfg.model.clone(),
            tokenizer: cfg.tokenizer.clone(),
            template: cfg.template.clone(),
            ctx: cfg.ctx,
            system: cfg.system.clone(),
            log_requests: cfg.log_requests,
            pb: cfg.engine.pb,
            max_seqs: cfg.engine.max_seqs,
            prefill_budget: cfg.prefill_budget,
        },
        backend,
        tokenizer,
    )?;
    let info = Arc::new(ServerInfo {
        model: cfg.name.clone(),
        ctx: cfg.ctx,
        backend: backend_name,
        started: Instant::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    });
    let listener = TcpListener::bind(&cfg.addr)?;
    listener.set_nonblocking(true)?;
    println!(
        "lfm-server {}: http://{} (модель {}, ctx {}, бэкенд {})",
        info.version, cfg.addr, info.model, info.ctx, info.backend
    );
    if cfg.log_requests {
        println!("lfm-server: логирование запросов включено");
    }
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);
    let _ = ctrlc::set_handler(move || {
        flag.store(false, Ordering::SeqCst);
    });
    let active = Arc::new(AtomicUsize::new(0));
    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let engine = engine.clone();
                let info = Arc::clone(&info);
                let active = Arc::clone(&active);
                active.fetch_add(1, Ordering::SeqCst);
                let spawned = thread::Builder::new()
                    .name("conn".to_string())
                    .spawn(move || {
                        let guard = ActiveGuard(active);
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                handle(stream, &engine, &info, cfg.max_body)
                            }));
                        match outcome {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => eprintln!("conn: {e}"),
                            Err(_) => eprintln!("conn: panic"),
                        }
                        drop(guard);
                    });
                if let Err(e) = spawned {
                    eprintln!("spawn: {e}");
                    thread::sleep(Duration::from_millis(20));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                eprintln!("accept: {e}");
                thread::sleep(Duration::from_millis(20));
            }
        }
    }
    println!("lfm-server: остановка, ждём активные соединения");
    let deadline = Instant::now() + Duration::from_secs(10);
    while active.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    let left = active.load(Ordering::SeqCst);
    if left > 0 {
        eprintln!("lfm-server: соединений осталось {left}, останавливаем движок");
    }
    engine.shutdown();
    println!("lfm-server: остановлен");
    Ok(())
}

struct ActiveGuard(Arc<AtomicUsize>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct ServerInfo {
    pub model: String,
    pub ctx: usize,
    pub backend: String,
    pub started: Instant,
    pub version: String,
}

impl ServerInfo {
    pub fn health_json(&self) -> String {
        json!({
            "status": "ok",
            "model": self.model,
            "ctx": self.ctx,
            "backend": self.backend,
            "uptime_s": self.started.elapsed().as_secs(),
            "version": self.version,
        })
        .to_string()
    }
}

fn error_body(message: &str) -> String {
    json!({ "error": { "message": message } }).to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Health,
    Models,
    Chat,
    Completions,
    Metrics,
    Ws,
    NotFound,
}

fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/health") => Route::Health,
        ("GET", "/v1/models") => Route::Models,
        ("POST", "/v1/chat/completions") => Route::Chat,
        ("POST", "/v1/completions") => Route::Completions,
        ("GET", "/metrics") => Route::Metrics,
        ("GET", "/ws") => Route::Ws,
        _ => Route::NotFound,
    }
}

fn handle(
    mut stream: TcpStream,
    engine: &Engine,
    info: &ServerInfo,
    max_body: usize,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(60)))?;
    let mut peek = [0u8; 24];
    let seen = stream.peek(&mut peek).unwrap_or(0);
    if seen > 0 && h2::is_preface(&peek[..seen]) && seen < peek.len() {
        return h2::serve(stream, engine, info, max_body);
    }
    let started = Instant::now();
    let request = match read_request(&mut stream, max_body)? {
        Some(request) => request,
        None => return Ok(()),
    };
    let (method, path) = (request.method.clone(), request.path.clone());
    let result = match route(&method, &path) {
        Route::Health => {
            let code = if engine.alive() { 200 } else { 503 };
            write_json(&mut stream, code, &info.health_json())
        }
        Route::Models => write_json(&mut stream, 200, &models_json(&info.model, 0)),
        Route::Chat => chat(stream, engine, &info.model, &request),
        Route::Completions => completions(stream, engine, &info.model, &request),
        Route::Metrics => write_json(
            &mut stream,
            200,
            &metrics_json(&info.model, &engine.metrics()),
        ),
        Route::Ws => ws_session(stream, engine, &request),
        Route::NotFound => write_json(&mut stream, 404, &error_body("not found")),
    };
    if engine.log_requests() {
        match &result {
            Ok(()) => eprintln!(
                "req: {method} {path} ok {} ms",
                started.elapsed().as_millis()
            ),
            Err(e) => eprintln!("req: {method} {path} error {e}"),
        }
    }
    result
}

fn chat(mut stream: TcpStream, engine: &Engine, model: &str, request: &Request) -> Result<()> {
    let parsed: ChatRequest = match serde_json::from_slice(&request.body) {
        Ok(parsed) => parsed,
        Err(e) => return write_json(&mut stream, 400, &error_body(&format!("bad json: {e}"))),
    };
    let messages = match parsed.to_messages() {
        Ok(messages) => messages,
        Err(e) => return write_json(&mut stream, 400, &error_body(&format!("{e}"))),
    };
    if messages.is_empty() {
        return write_json(&mut stream, 400, &error_body("messages must not be empty"));
    }
    let params = parsed.to_params();
    let job = ChatJob::chat(messages, params, parsed.thinking(), parsed.to_tools());
    let _cancel = engine::CancelGuard::new(job.cancel_handle());
    let events = engine.submit(job)?;
    let name = parsed.model.as_deref().unwrap_or(model);
    let meta = CompletionMeta::new(name);
    if parsed.stream {
        start_sse(&mut stream)?;
        write_event(
            &mut stream,
            &meta.chunk(json!({ "role": "assistant", "content": "" }), None),
        )?;
        let mut tool_index = 0usize;
        let mut finish_reason = "stop";
        let mut finished = false;
        for event in events {
            match event {
                Event::Delta(text) => {
                    write_event(&mut stream, &meta.chunk(json!({ "content": text }), None))?;
                }
                Event::Thinking(text) => {
                    write_event(
                        &mut stream,
                        &meta.chunk(json!({ "reasoning_content": text }), None),
                    )?;
                }
                Event::ToolCall(tc) => {
                    write_event(&mut stream, &meta.tool_chunk(tool_index, &tc))?;
                    tool_index += 1;
                }
                Event::Final {
                    finish_reason: reason,
                    ..
                } => {
                    finish_reason = reason;
                    finished = true;
                }
                Event::Error(e) => {
                    eprintln!("engine error: {e}");
                    write_event(
                        &mut stream,
                        &meta.chunk(json!({ "content": "\n[error] generation failed" }), None),
                    )?;
                    finish_reason = "error";
                }
            }
        }
        if !finished {
            write_event(
                &mut stream,
                &json!({"error": {"message": "generation interrupted", "type": "server_error"}})
                    .to_string(),
            )?;
            return end_sse(&mut stream);
        }
        write_event(&mut stream, &meta.chunk(json!({}), Some(finish_reason)))?;
        write_event(&mut stream, "[DONE]")?;
        return end_sse(&mut stream);
    }
    let mut text = String::new();
    let mut thinking: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut finish_reason = "stop";
    let mut usage = (0usize, 0usize);
    for event in events {
        match event {
            Event::Final {
                text: t,
                thinking: th,
                tool_calls: calls,
                finish_reason: reason,
                prompt_tokens,
                completion_tokens,
            } => {
                text = t;
                thinking = th;
                tool_calls = calls;
                finish_reason = reason;
                usage = (prompt_tokens, completion_tokens);
            }
            Event::Error(e) => {
                eprintln!("engine error: {e}");
                return write_json(&mut stream, 500, &error_body("generation failed"));
            }
            Event::Delta(_) | Event::Thinking(_) | Event::ToolCall(_) => {}
        }
    }
    write_json(
        &mut stream,
        200,
        &meta.message_response(
            &text,
            thinking.as_deref(),
            &tool_calls,
            finish_reason,
            usage.0,
            usage.1,
        ),
    )
}

fn completions(
    mut stream: TcpStream,
    engine: &Engine,
    model: &str,
    request: &Request,
) -> Result<()> {
    let parsed: CompletionRequest = match serde_json::from_slice(&request.body) {
        Ok(parsed) => parsed,
        Err(e) => return write_json(&mut stream, 400, &error_body(&format!("bad json: {e}"))),
    };
    let prompts = match parsed.to_prompts() {
        Ok(prompts) => prompts,
        Err(e) => return write_json(&mut stream, 400, &error_body(&format!("{e}"))),
    };
    let params = parsed.to_params();
    let name = parsed.model.as_deref().unwrap_or(model);
    let meta = CompletionMeta::new_completion(name);
    if parsed.stream {
        if prompts.len() != 1 {
            return write_json(
                &mut stream,
                400,
                &error_body("streaming requires a single prompt"),
            );
        }
        let job = ChatJob::completion(prompts[0].clone(), params);
        let _cancel = engine::CancelGuard::new(job.cancel_handle());
        let events = engine.submit(job)?;
        start_sse(&mut stream)?;
        write_event(&mut stream, &meta.completion_chunk("", None))?;
        let mut finish_reason = "stop";
        let mut finished = false;
        for event in events {
            match event {
                Event::Delta(text) => {
                    write_event(&mut stream, &meta.completion_chunk(&text, None))?;
                }
                Event::Final {
                    finish_reason: reason,
                    ..
                } => {
                    finish_reason = reason;
                    finished = true;
                }
                Event::Error(e) => {
                    eprintln!("engine error: {e}");
                    finish_reason = "error";
                }
                Event::Thinking(_) | Event::ToolCall(_) => {}
            }
        }
        if !finished {
            write_event(
                &mut stream,
                &json!({"error": {"message": "generation interrupted", "type": "server_error"}})
                    .to_string(),
            )?;
            return end_sse(&mut stream);
        }
        write_event(&mut stream, &meta.completion_chunk("", Some(finish_reason)))?;
        write_event(&mut stream, "[DONE]")?;
        return end_sse(&mut stream);
    }
    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    for prompt in prompts {
        let job = ChatJob::completion(prompt, params.clone());
        let _cancel = engine::CancelGuard::new(job.cancel_handle());
        let events = engine.submit(job)?;
        let mut text = String::new();
        let mut finish_reason = "stop";
        for event in events {
            match event {
                Event::Final {
                    text: t,
                    finish_reason: reason,
                    prompt_tokens: p,
                    completion_tokens: c,
                    ..
                } => {
                    text = t;
                    finish_reason = reason;
                    prompt_tokens += p;
                    completion_tokens += c;
                }
                Event::Error(e) => {
                    eprintln!("engine error: {e}");
                    return write_json(&mut stream, 500, &error_body("generation failed"));
                }
                Event::Delta(_) | Event::Thinking(_) | Event::ToolCall(_) => {}
            }
        }
        choices.push((text, finish_reason.to_string()));
    }
    write_json(
        &mut stream,
        200,
        &meta.completion_response(&choices, prompt_tokens, completion_tokens),
    )
}

fn ws_session(mut stream: TcpStream, engine: &Engine, request: &Request) -> Result<()> {
    let key = request
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("sec-websocket-key"))
        .map(|(_, v)| v.clone());
    let Some(key) = key else {
        return write_json(&mut stream, 400, &error_body("missing sec-websocket-key"));
    };
    let header = |name: &str| {
        request
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .unwrap_or("")
    };
    let upgrade_ok = header("upgrade").eq_ignore_ascii_case("websocket");
    let connection_ok = header("connection")
        .split(',')
        .any(|part| part.trim().eq_ignore_ascii_case("upgrade"));
    let version_ok = header("sec-websocket-version") == "13";
    if !upgrade_ok || !connection_ok || !version_ok || key.len() < 16 {
        return write_json(&mut stream, 400, &error_body("bad websocket handshake"));
    }
    ws::handshake(&mut stream, &key)?;
    let mut conn = ws::WsConn::new(stream);
    loop {
        let Some(text) = conn.read_message()? else {
            break;
        };
        let parsed: ChatRequest = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(e) => {
                conn.send_text(
                    &json!({ "type": "error", "message": format!("bad json: {e}") }).to_string(),
                )?;
                continue;
            }
        };
        let messages = match parsed.to_messages() {
            Ok(messages) => messages,
            Err(e) => {
                conn.send_text(&json!({ "type": "error", "message": format!("{e}") }).to_string())?;
                continue;
            }
        };
        let params = parsed.to_params();
        let events = engine.submit(ChatJob::chat(
            messages,
            params,
            parsed.thinking(),
            parsed.to_tools(),
        ))?;
        for event in events {
            match event {
                Event::Delta(text) => {
                    conn.send_text(&json!({ "type": "delta", "content": text }).to_string())?;
                }
                Event::Thinking(text) => {
                    conn.send_text(&json!({ "type": "thinking", "content": text }).to_string())?;
                }
                Event::ToolCall(tc) => {
                    conn.send_text(
                        &json!({
                            "type": "tool_call",
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments
                        })
                        .to_string(),
                    )?;
                }
                Event::Final {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    ..
                } => {
                    conn.send_text(
                        &json!({
                            "type": "done",
                            "finish_reason": finish_reason,
                            "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens }
                        })
                        .to_string(),
                    )?;
                }
                Event::Error(e) => {
                    conn.send_text(&json!({ "type": "error", "message": e }).to_string())?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::EngineMetricsSnapshot;
    use std::io::Read;

    fn roundtrip(body: &str) -> (String, String) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let payload = body.to_string();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            write_json(&mut stream, 200, &payload).unwrap();
        });
        let mut client = TcpStream::connect(addr).unwrap();
        let mut raw = Vec::new();
        client.read_to_end(&mut raw).unwrap();
        server.join().unwrap();
        let text = String::from_utf8(raw).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        (head.to_string(), body.to_string())
    }

    fn sample_snapshot() -> EngineMetricsSnapshot {
        EngineMetricsSnapshot {
            ctx: 4096,
            active: 2,
            waiting: 1,
            steps: 7,
            avg_step_ms: 12.5,
            kv_used_pages: 3,
            kv_free_pages: 4,
            kv_cached_pages: 5,
            cache_hits: 6,
            reused_tokens: 8,
            uptime_s: 9.5,
        }
    }

    #[test]
    fn metrics_is_routed_for_get_only() {
        assert_eq!(route("GET", "/metrics"), Route::Metrics);
        assert_eq!(route("POST", "/metrics"), Route::NotFound);
        assert_eq!(route("GET", "/metrics/"), Route::NotFound);
        assert_eq!(route("GET", "/health"), Route::Health);
        assert_eq!(route("GET", "/v1/models"), Route::Models);
        assert_eq!(route("POST", "/v1/chat/completions"), Route::Chat);
        assert_eq!(route("POST", "/v1/completions"), Route::Completions);
        assert_eq!(route("GET", "/ws"), Route::Ws);
        assert_eq!(route("GET", "/nope"), Route::NotFound);
    }

    #[test]
    fn metrics_route_returns_200_json() {
        let (head, body) = roundtrip(&metrics_json("lfm3", &sample_snapshot()));
        assert!(head.starts_with("HTTP/1.1 200 OK\r\n"), "{head}");
        assert!(head.contains("Content-Type: application/json"), "{head}");
        assert!(head.contains(&format!("Content-Length: {}", body.len())));
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "lfm3");
        assert_eq!(v["ctx"], 4096);
        assert_eq!(v["active"], 2);
        assert_eq!(v["waiting"], 1);
        assert_eq!(v["steps"], 7);
        assert_eq!(v["avg_step_ms"].as_f64(), Some(12.5));
        assert_eq!(v["kv_used_pages"], 3);
        assert_eq!(v["kv_free_pages"], 4);
        assert_eq!(v["kv_cached_pages"], 5);
        assert_eq!(v["cache_hits"], 6);
        assert_eq!(v["reused_tokens"], 8);
        assert_eq!(v["uptime_s"].as_f64(), Some(9.5));
    }
}
