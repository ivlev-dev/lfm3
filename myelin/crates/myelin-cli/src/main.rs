//! `myelin` — chat / server on the myelin CPU backend.
//!
//! ```text
//! myelin gen   --model M --prompt "..." [--tokens N]
//! myelin chat  --model M [--ctx 2048] [--thinking]
//! myelin serve --model M [--bind 127.0.0.1:8080]
//! ```
//!
//! Paths default to `models/...` next to the executable's working directory;
//! on the phone the deploy script passes them explicitly.

use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use lfm_chat::ChatTemplate;
use lfm_domain::{Message, Role};
use myelin_core::tokenizer::LfmTokenizer;
use myelin_core::{GenerateConfig, Model, SamplingParams, Session};

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    template: PathBuf,
    ctx: usize,
    max_tokens: usize,
    temperature: f32,
    thinking: bool,
    prompt: String,
    bind: String,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: PathBuf::from("models/LFM3-1.7B-Q4_K_M.gguf"),
            tokenizer: PathBuf::from("models/hf/tokenizer.json"),
            template: PathBuf::from("models/hf/chat_template.jinja"),
            ctx: 2048,
            max_tokens: 256,
            temperature: 0.0,
            thinking: false,
            prompt: String::new(),
            bind: "127.0.0.1:8080".to_owned(),
        }
    }
}

fn parse_args(cmd: &str, rest: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    if cmd == "serve" {
        a.temperature = 0.7;
    }
    let mut i = 0;
    while i < rest.len() {
        let key = rest[i].as_str();
        let mut value = || -> Result<String, String> {
            i += 1;
            rest.get(i)
                .cloned()
                .ok_or_else(|| format!("{key}: missing value"))
        };
        match key {
            "--model" | "-m" => a.model = PathBuf::from(value()?),
            "--tokenizer" => a.tokenizer = PathBuf::from(value()?),
            "--template" => a.template = PathBuf::from(value()?),
            "--ctx" => a.ctx = value()?.parse().map_err(|e| format!("ctx: {e}"))?,
            "--tokens" | "-n" => {
                a.max_tokens = value()?.parse().map_err(|e| format!("tokens: {e}"))?
            }
            "--temperature" | "-t" => {
                a.temperature = value()?.parse().map_err(|e| format!("temperature: {e}"))?
            }
            "--prompt" | "-p" => a.prompt = value()?,
            "--bind" => a.bind = value()?,
            "--thinking" => a.thinking = true,
            other => return Err(format!("unknown option `{other}`")),
        }
        i += 1;
    }
    Ok(a)
}

fn sampling(a: &Args) -> SamplingParams {
    SamplingParams {
        temperature: a.temperature,
        ..SamplingParams::default()
    }
}

fn load(a: &Args) -> Result<(Model, LfmTokenizer, ChatTemplate), String> {
    let model = Model::load_gguf(&a.model).map_err(|e| format!("model: {e}"))?;
    let tok = LfmTokenizer::from_file(&a.tokenizer).map_err(|e| format!("tokenizer: {e}"))?;
    let tpl = ChatTemplate::from_file(&a.template).map_err(|e| format!("template: {e}"))?;
    eprintln!(
        "[myelin] backend={} model={}",
        model.backend_name(),
        a.model.display()
    );
    Ok((model, tok, tpl))
}

fn config(a: &Args) -> GenerateConfig {
    GenerateConfig {
        max_tokens: a.max_tokens,
        sampling: sampling(a),
        eos: None,
        stop: Vec::new(),
    }
}

/// Generation with incremental detokenisation printed to `out`.
fn generate_print(
    session: &mut Session<'_>,
    tok: &LfmTokenizer,
    prompt: &[u32],
    cfg: &GenerateConfig,
    out: &mut dyn Write,
) -> Result<(usize, String), String> {
    let mut ids: Vec<u32> = Vec::new();
    let mut printed = String::new();
    let t0 = Instant::now();
    session
        .generate(prompt, cfg, |token| {
            ids.push(token);
            if let Ok(text) = tok.decode(&ids, true) {
                if text.len() > printed.len() && text.starts_with(&printed) {
                    let _ = out.write_all(text[printed.len()..].as_bytes());
                    let _ = out.flush();
                    printed = text;
                }
            }
        })
        .map_err(|e| format!("generate: {e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    if dt > 0.0 {
        eprintln!(
            "\n[myelin] {} tokens, {:.2} tok/s",
            ids.len(),
            ids.len() as f64 / dt
        );
    }
    Ok((ids.len(), printed))
}

fn run_gen(a: &Args) -> Result<(), String> {
    let (mut model, tok, _tpl) = load(a)?;
    let ids = tok
        .encode(&a.prompt, false)
        .map_err(|e| format!("encode: {e}"))?;
    let mut session = model
        .session(a.ctx)
        .map_err(|e| format!("session: {e}"))?;
    let mut out = std::io::stdout();
    generate_print(&mut session, &tok, &ids, &config(a), &mut out)?;
    println!();
    Ok(())
}

fn render(
    tpl: &ChatTemplate,
    messages: &[Message],
    thinking: bool,
) -> Result<String, String> {
    tpl.render(messages, &[], true, thinking, false)
        .map_err(|e| format!("template: {e}"))
}

fn run_chat(a: &Args) -> Result<(), String> {
    let (mut model, tok, tpl) = load(a)?;
    let mut messages: Vec<Message> = vec![Message {
        role: Role::System,
        content: Some("You are a helpful assistant.".to_owned()),
        thinking: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }];
    let stdin = std::io::stdin();
    println!("myelin chat — пустая строка = выход, /reset — сброс истории");
    loop {
        print!("you> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if line == "/reset" {
            messages.truncate(1);
            println!("[myelin] history reset");
            continue;
        }
        messages.push(Message {
            role: Role::User,
            content: Some(line.to_owned()),
            thinking: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
        let prompt = render(&tpl, &messages, a.thinking)?;
        let ids = tok.encode(&prompt, false).map_err(|e| format!("encode: {e}"))?;
        let mut session = model
            .session(a.ctx)
            .map_err(|e| format!("session: {e}"))?;
        print!("bot> ");
        std::io::stdout().flush().ok();
        let mut out = std::io::stdout();
        let (n, text) = generate_print(&mut session, &tok, &ids, &config(a), &mut out)?;
        println!();
        let _ = n;
        messages.push(Message {
            role: Role::Assistant,
            content: Some(text),
            thinking: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    Ok(())
}

fn run_serve(a: &Args) -> Result<(), String> {
    let (mut model, tok, tpl) = load(a)?;
    let listener = TcpListener::bind(&a.bind).map_err(|e| format!("bind: {e}"))?;
    eprintln!("[myelin] listening on http://{}", a.bind);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle(s, &mut model, &tok, &tpl, a) {
                    eprintln!("[myelin] request error: {e}");
                }
            }
            Err(e) => eprintln!("[myelin] accept: {e}"),
        }
    }
    Ok(())
}

fn handle(
    mut s: TcpStream,
    model: &mut Model,
    tok: &LfmTokenizer,
    tpl: &ChatTemplate,
    a: &Args,
) -> Result<(), String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let mut body_start = None;
    loop {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if body_start.is_none() {
            if let Some(p) = find(&buf, b"\r\n\r\n") {
                body_start = Some(p + 4);
            }
        }
        if let Some(start) = body_start {
            let head = String::from_utf8_lossy(&buf[..start]).to_ascii_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= start + len {
                break;
            }
        }
    }
    let start = body_start.ok_or("bad request: no headers")?;
    let head = String::from_utf8_lossy(&buf[..start]).to_string();
    let body = &buf[start..];
    if head.starts_with("GET /health") {
        return respond(&mut s, "text/plain", "ok");
    }
    if !head.starts_with("POST /v1/chat/completions") {
        return respond(&mut s, "text/plain", "myelin: POST /v1/chat/completions");
    }
    let req: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("json: {e}"))?;
    let messages: Vec<Message> = req
        .get("messages")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or("missing messages")?;
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(a.max_tokens);
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let prompt = render(tpl, &messages, a.thinking)?;
    let ids = tok.encode(&prompt, false).map_err(|e| format!("encode: {e}"))?;
    let cfg = GenerateConfig {
        max_tokens,
        sampling: sampling(a),
        eos: None,
        stop: Vec::new(),
    };

    if stream {
        let mut head_out = String::from(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        );
        s.write_all(head_out.as_bytes()).map_err(|e| e.to_string())?;
        head_out.clear();
        let mut session = model.session(a.ctx).map_err(|e| format!("session: {e}"))?;
        let mut generated: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut err = None;
        session
            .generate(&ids, &cfg, |token| {
                generated.push(token);
                if let Ok(text) = tok.decode(&generated, true) {
                    if text.len() > printed.len() && text.starts_with(&printed) {
                        let delta = &text[printed.len()..];
                        let chunk = serde_json::json!({ "delta": delta });
                        let line = format!("data: {chunk}\n\n");
                        if s.write_all(line.as_bytes()).is_err() {
                            err = Some("write".to_string());
                        }
                        printed = text;
                    }
                }
            })
            .map_err(|e| format!("generate: {e}"))?;
        if let Some(e) = err {
            return Err(e);
        }
        s.write_all(b"data: [DONE]\n\n").map_err(|e| e.to_string())?;
        return Ok(());
    }

    let mut session = model.session(a.ctx).map_err(|e| format!("session: {e}"))?;
    let mut generated: Vec<u32> = Vec::new();
    session
        .generate(&ids, &cfg, |t| generated.push(t))
        .map_err(|e| format!("generate: {e}"))?;
    let text = tok.decode(&generated, true).unwrap_or_default();
    let json = serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": text } }] });
    respond(&mut s, "application/json", &json.to_string())
}

fn respond(s: &mut TcpStream, ctype: &str, body: &str) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    s.write_all(body.as_bytes()).map_err(|e| e.to_string())
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn usage() {
    eprintln!(
        "usage: myelin <gen|chat|serve> [options]\n\
         \x20 --model PATH        GGUF (default models/LFM3-1.7B-Q4_K_M.gguf)\n\
         \x20 --tokenizer PATH    tokenizer.json\n\
         \x20 --template PATH     chat_template.jinja\n\
         \x20 --ctx N             context tokens (default 2048)\n\
         \x20 --tokens N          max new tokens\n\
         \x20 --temperature T     0 = greedy\n\
         \x20 --prompt TEXT       gen: prompt\n\
         \x20 --bind ADDR         serve: address (default 127.0.0.1:8080)\n\
         \x20 --thinking          enable the model's thinking block"
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    if cmd.is_empty() || cmd == "-h" || cmd == "--help" {
        usage();
        std::process::exit(if cmd.is_empty() { 2 } else { 0 });
    }
    let args = match parse_args(&cmd, &argv[1..]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("myelin: {e}");
            std::process::exit(2);
        }
    };
    let result = match cmd.as_str() {
        "gen" | "generate" => run_gen(&args),
        "chat" => run_chat(&args),
        "serve" => run_serve(&args),
        other => {
            eprintln!("myelin: unknown command `{other}`");
            usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("myelin: {e}");
        std::process::exit(1);
    }
}
