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
use http::{end_sse, read_request, start_sse, write_event, write_json, Request};
use protocol::{metrics_json, models_json, ChatRequest, CompletionMeta, CompletionRequest};
use serde_json::json;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
    model: PathBuf,
    #[arg(long, default_value = "models/hf/tokenizer.json")]
    tokenizer: PathBuf,
    #[arg(long, default_value = "models/hf/chat_template.jinja")]
    template: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8000")]
    addr: String,
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    #[arg(long, default_value = "")]
    system: String,
    #[arg(long, default_value = "lfm3")]
    name: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let backend: lfm_app::BackendLoader = match std::env::var("LFM_BACKEND") {
        Ok(name) if name == "cpu" => Box::new(|path, ctx| {
            Ok(Box::new(lfm_backend_cpu::CpuBackend::load(path, ctx)?))
        }),
        _ => match std::env::var("LFM_TP") {
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
            _ => Box::new(|path, ctx| Ok(Box::new(lfm_gpu::GpuModel::load(path, ctx)?))),
        },
        },
    };
    let tokenizer: lfm_app::TokenizerLoader = Box::new(|path| {
        Ok(Box::new(lfm_tokenizer::LfmTokenizer::from_file(path)?))
    });
    let engine = Engine::spawn(
        ServeConfig {
            model: cli.model.clone(),
            tokenizer: cli.tokenizer.clone(),
            template: cli.template.clone(),
            ctx: cli.ctx,
            system: cli.system.clone(),
        },
        backend,
        tokenizer,
    )?;
    let listener = TcpListener::bind(&cli.addr)?;
    println!("lfm-server: http://{} (модель {})", cli.addr, cli.name);
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let engine = engine.clone();
        let name = cli.name.clone();
        thread::spawn(move || {
            if let Err(e) = handle(stream, &engine, &name) {
                eprintln!("conn: {e}");
            }
        });
    }
    Ok(())
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

fn handle(mut stream: TcpStream, engine: &Engine, model: &str) -> Result<()> {
    let mut peek = [0u8; 24];
    let seen = stream.peek(&mut peek).unwrap_or(0);
    if h2::is_preface(&peek[..seen]) {
        return h2::serve(stream, engine, model);
    }
    let request = match read_request(&mut stream)? {
        Some(request) => request,
        None => return Ok(()),
    };
    match route(&request.method, &request.path) {
        Route::Health => write_json(&mut stream, 200, r#"{"status":"ok"}"#),
        Route::Models => write_json(&mut stream, 200, &models_json(model, 0)),
        Route::Chat => chat(stream, engine, model, &request),
        Route::Completions => completions(stream, engine, model, &request),
        Route::Metrics => write_json(&mut stream, 200, &metrics_json(model, &engine.metrics())),
        Route::Ws => ws_session(stream, engine, &request),
        Route::NotFound => write_json(&mut stream, 404, &error_body("not found")),
    }
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
    let params = parsed.to_params();
    let job = ChatJob::chat(messages, params, parsed.thinking(), parsed.to_tools());
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
                } => finish_reason = reason,
                Event::Error(e) => {
                    write_event(
                        &mut stream,
                        &meta.chunk(json!({ "content": format!("\n[error] {e}") }), None),
                    )?;
                }
            }
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
            Event::Error(e) => return write_json(&mut stream, 500, &error_body(&e)),
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

fn completions(mut stream: TcpStream, engine: &Engine, model: &str, request: &Request) -> Result<()> {
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
        let events = engine.submit(ChatJob::completion(prompts[0].clone(), params))?;
        start_sse(&mut stream)?;
        write_event(&mut stream, &meta.completion_chunk("", None))?;
        let mut finish_reason = "stop";
        for event in events {
            match event {
                Event::Delta(text) => {
                    write_event(&mut stream, &meta.completion_chunk(&text, None))?;
                }
                Event::Final {
                    finish_reason: reason,
                    ..
                } => finish_reason = reason,
                Event::Error(e) => {
                    write_event(
                        &mut stream,
                        &meta.completion_chunk(&format!("\n[error] {e}"), None),
                    )?;
                }
                Event::Thinking(_) | Event::ToolCall(_) => {}
            }
        }
        write_event(&mut stream, &meta.completion_chunk("", Some(finish_reason)))?;
        write_event(&mut stream, "[DONE]")?;
        return end_sse(&mut stream);
    }
    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    for prompt in prompts {
        let events = engine.submit(ChatJob::completion(prompt, params.clone()))?;
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
                Event::Error(e) => return write_json(&mut stream, 500, &error_body(&e)),
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
    ws::handshake(&mut stream, &key)?;
    let mut conn = ws::WsConn::new(stream);
    loop {
        let Some(text) = conn.read_message()? else {
            break;
        };
        let parsed: ChatRequest = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(e) => {
                conn.send_text(&json!({ "type": "error", "message": format!("bad json: {e}") }).to_string())?;
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
        assert!(
            head.contains("Content-Type: application/json"),
            "{head}"
        );
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
