pub mod engine;
pub mod h2;
mod hpack;
mod hpack_tables;
pub mod http;
pub mod protocol;
mod ws;

use anyhow::Result;
use serde_json::json;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::{Duration, Instant};

use lfm::BackendMode;
use crate::serve::engine::{ChatJob, Engine, Event, ServeConfig};
use crate::serve::http::{Request, SseWriter, read_request, write_json};
use crate::serve::protocol::{ChatRequest, CompletionMeta, CompletionRequest, Delta, metrics_json, models_json};

const MAX_BATCH_EVENTS: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_body: usize,
    pub h2_cancel_interval: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pump {
    Continue,
    Stop,
}

pub(crate) fn pump_events(
    events: &Receiver<Event>,
    mut apply: impl FnMut(&[Event]) -> Result<Pump>,
) -> Result<()> {
    let mut batch: Vec<Event> = Vec::with_capacity(MAX_BATCH_EVENTS);
    loop {
        batch.clear();
        match events.recv() {
            Ok(event) => batch.push(event),
            Err(_) => return Ok(()),
        }
        while batch.len() < MAX_BATCH_EVENTS {
            match events.try_recv() {
                Ok(event) => batch.push(event),
                Err(_) => break,
            }
        }
        if apply(&batch)? == Pump::Stop {
            return Ok(());
        }
    }
}

pub fn run(cfg: &lfm::Config) -> Result<()> {
    cfg.validate()?;
    let opts = cfg.backend.to_options();
    let mode = cfg.backend.mode;
    let devices = cfg.backend.devices.clone();
    let plan: Vec<(usize, usize)> = devices
        .iter()
        .copied()
        .zip(cfg.backend.effective_tiers()?)
        .collect();
    let backend: lfm_app::BackendLoader = match mode {
        BackendMode::Single => Box::new(move |path, ctx| {
            Ok(Box::new(lfm_gpu::GpuModel::load_with(path, ctx, &opts)?))
        }),
        BackendMode::Pipeline => Box::new(move |path, ctx| {
            Ok(Box::new(lfm_gpu::PipelineModel::load_tiers_with(
                path, ctx, &plan, &opts,
            )?))
        }),
        BackendMode::Tp => Box::new(move |path, ctx| {
            Ok(Box::new(lfm_gpu::TpModel::load_with(
                path, ctx, &devices, &opts,
            )?))
        }),
    };
    let backend_name = match mode {
        BackendMode::Single => "opencl".to_string(),
        BackendMode::Pipeline => format!(
            "pipeline:{}",
            cfg.backend
                .devices
                .iter()
                .zip(cfg.backend.effective_tiers()?)
                .map(|(d, p)| format!("{d}@{p}"))
                .collect::<Vec<_>>()
                .join("/")
        ),
        BackendMode::Tp => format!(
            "tp:{}",
            cfg.backend
                .devices
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("/")
        ),
    };
    crate::serve::protocol::set_defaults(lfm_domain::SamplingParams {
        temperature: cfg.chat.temperature,
        top_k: cfg.chat.top_k,
        top_p: cfg.chat.top_p,
        min_p: cfg.chat.min_p,
        repeat_penalty: cfg.chat.repeat_penalty,
        max_tokens: cfg.chat.max_tokens,
        seed: cfg.chat.seed,
        ..lfm_domain::SamplingParams::default()
    });
    let tokenizer: lfm_app::TokenizerLoader = Box::new(|path| {
        Ok(Box::new(lfm_tokenizer::LfmTokenizer::from_file(path)?))
    });
    let engine = Engine::spawn(
        ServeConfig {
            model: cfg.model.path.clone(),
            tokenizer: cfg.model.tokenizer.clone(),
            template: cfg.model.template.clone(),
            ctx: cfg.backend.ctx,
            system: cfg.server.system.clone(),
            log_requests: cfg.server.log_requests,
            max_seqs: cfg.backend.max_seqs,
            prefill_budget: cfg.server.prefill_budget,
            prefill_in_decode: cfg.server.prefill_in_decode,
        },
        backend,
        tokenizer,
    )?;
    let info = Arc::new(ServerInfo {
        model: cfg.server.name.clone(),
        ctx: cfg.backend.ctx,
        backend: backend_name,
        started: Instant::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    });
    let listener = TcpListener::bind(&cfg.server.addr)?;
    listener.set_nonblocking(true)?;
    println!(
        "lfm {}: http://{} (модель {}, ctx {}, бэкенд {})",
        info.version, cfg.server.addr, info.model, info.ctx, info.backend
    );
    if cfg.server.log_requests {
        println!("lfm: логирование запросов включено");
    }
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);
    let _ = ctrlc::set_handler(move || {
        flag.store(false, Ordering::SeqCst);
    });
    let active = Arc::new(AtomicUsize::new(0));
    let limits = Limits {
        max_body: cfg.server.max_body,
        h2_cancel_interval: cfg.server.h2_cancel_interval,
    };
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
                                handle(stream, &engine, &info, limits)
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
    println!("lfm: остановка, ждём активные соединения");
    let deadline = Instant::now() + Duration::from_secs(10);
    while active.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    let left = active.load(Ordering::SeqCst);
    if left > 0 {
        eprintln!("lfm: соединений осталось {left}, останавливаем движок");
    }
    engine.shutdown();
    println!("lfm: остановлен");
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

fn handle(mut stream: TcpStream, engine: &Engine, info: &ServerInfo, limits: Limits) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(60)))?;
    let mut peek = [0u8; 24];
    let seen = stream.peek(&mut peek).unwrap_or(0);
    if seen > 0 && h2::is_preface(&peek[..seen]) && seen < peek.len() {
        return h2::serve(stream, engine, info, limits);
    }
    let started = Instant::now();
    let request = match read_request(&mut stream, limits.max_body)? {
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
        let mut out = SseWriter::start(&mut stream)?;
        out.json(|b| meta.write_chunk(b, Delta::opening("assistant"), None))?;
        out.flush()?;
        let mut tool_index = 0usize;
        let mut finish_reason = "stop";
        let mut finished = false;
        pump_events(&events, |batch| {
            for event in batch {
                match event {
                    Event::Delta(text) => {
                        out.json(|b| meta.write_chunk(b, Delta::content(text), None))?;
                    }
                    Event::Thinking(text) => {
                        out.json(|b| meta.write_chunk(b, Delta::reasoning(text), None))?;
                    }
                    Event::ToolCall(tc) => {
                        out.json(|b| meta.write_chunk(b, Delta::tool_call(tool_index, tc), None))?;
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
                        out.json(|b| {
                            meta.write_chunk(b, Delta::content("\n[error] generation failed"), None)
                        })?;
                        finish_reason = "error";
                    }
                }
            }
            out.flush()?;
            Ok(Pump::Continue)
        })?;
        if !finished {
            out.data(
                br#"{"error": {"message": "generation interrupted", "type": "server_error"}}"#,
            );
            return out.end();
        }
        out.json(|b| meta.write_chunk(b, Delta::empty(), Some(finish_reason)))?;
        out.data(b"[DONE]");
        return out.end();
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
        let mut out = SseWriter::start(&mut stream)?;
        out.json(|b| meta.write_completion_chunk(b, "", None))?;
        out.flush()?;
        let mut finish_reason = "stop";
        let mut finished = false;
        pump_events(&events, |batch| {
            for event in batch {
                match event {
                    Event::Delta(text) => {
                        out.json(|b| meta.write_completion_chunk(b, text, None))?;
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
            out.flush()?;
            Ok(Pump::Continue)
        })?;
        if !finished {
            out.data(
                br#"{"error": {"message": "generation interrupted", "type": "server_error"}}"#,
            );
            return out.end();
        }
        out.json(|b| meta.write_completion_chunk(b, "", Some(finish_reason)))?;
        out.data(b"[DONE]");
        return out.end();
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
        let job = ChatJob::chat(messages, params, parsed.thinking(), parsed.to_tools());
        let _cancel = engine::CancelGuard::new(job.cancel_handle());
        let events = engine.submit(job)?;
        pump_events(&events, |batch| {
            for event in batch {
                let message = match event {
                    Event::Delta(content) => ws::OutMessage::Delta { content },
                    Event::Thinking(content) => ws::OutMessage::Thinking { content },
                    Event::ToolCall(tc) => ws::OutMessage::ToolCall {
                        id: tc.id.as_deref(),
                        name: &tc.name,
                        arguments: &tc.arguments,
                    },
                    Event::Final {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        ..
                    } => ws::OutMessage::Done {
                        finish_reason: *finish_reason,
                        usage: ws::Usage {
                            prompt_tokens: *prompt_tokens,
                            completion_tokens: *completion_tokens,
                        },
                    },
                    Event::Error(message) => ws::OutMessage::Error { message },
                };
                conn.queue_json(&message)?;
            }
            conn.flush()?;
            Ok(Pump::Continue)
        })?;
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
            avg_gpu_us: 100,
            avg_logits_us: 200,
            avg_sample_us: 30,
            avg_detok_us: 40,
            rows_total: 11,
            rows_logits: 9,
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
