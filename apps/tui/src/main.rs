mod app;
mod http;
mod ui;

use anyhow::{Result, anyhow};
use clap::Parser;
use http::{Client, Metrics, StreamStep, Url};
use std::io::{Read, Write};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};
use ui::Key;

#[derive(Parser)]
#[command(name = "lfm-tui", about = "TUI-клиент для lfm-server")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    #[arg(long, default_value = "")]
    system: String,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
}

enum Event {
    Key(Key),
    Metrics(Metrics),
    MetricsDown,
}

static RESTORED: AtomicBool = AtomicBool::new(false);

struct Terminal;

fn stty(args: &[&str]) -> bool {
    std::process::Command::new("stty")
        .args(args)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

impl Terminal {
    fn enter() -> Result<Self> {
        RESTORED.store(false, Ordering::SeqCst);
        if !stty(&["raw", "-echo"]) {
            return Err(anyhow!("не удалось включить raw-режим: нужен терминал"));
        }
        let mut stdout = std::io::stdout();
        stdout.write_all(b"\x1b[?1049h\x1b[2J\x1b[?25l")?;
        stdout.flush()?;
        Ok(Self)
    }

    fn restore() {
        if RESTORED.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(b"\x1b[?25h\x1b[0m\x1b[?1049l");
        let _ = stdout.flush();
        stty(&["sane"]);
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        Self::restore();
    }
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        Terminal::restore();
        previous(info);
    }));
}

fn term_size() -> (usize, usize) {
    let from_env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    };
    let parsed = std::process::Command::new("stty")
        .arg("size")
        .stdin(Stdio::inherit())
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| {
            let text = String::from_utf8_lossy(&out.stdout).into_owned();
            let mut parts = text.split_whitespace();
            let rows = parts.next().and_then(|value| value.parse::<usize>().ok());
            let cols = parts.next().and_then(|value| value.parse::<usize>().ok());
            match (rows, cols) {
                (Some(rows), Some(cols)) if rows >= 3 && cols >= 20 => Some((rows, cols)),
                _ => None,
            }
        });
    parsed.unwrap_or_else(|| {
        (
            from_env("LINES", 24).max(3),
            from_env("COLUMNS", 80).max(20),
        )
    })
}

fn spawn_input(tx: Sender<Event>) {
    thread::spawn(move || {
        let mut stdin = std::io::stdin();
        let mut parser = ui::KeyParser::new();
        let mut buf = [0u8; 256];
        loop {
            match stdin.read(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(Event::Key(Key::Eof));
                    return;
                }
                Ok(read) => {
                    for key in parser.push(&buf[..read]) {
                        if tx.send(Event::Key(key)).is_err() {
                            return;
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    let _ = tx.send(Event::Key(Key::Eof));
                    return;
                }
            }
        }
    });
}

fn spawn_metrics(url: Url, tx: Sender<Event>, running: Arc<AtomicBool>) {
    thread::spawn(move || {
        let client = Client::new(url);
        while running.load(Ordering::SeqCst) {
            let event = match client.metrics() {
                Ok(metrics) => Event::Metrics(metrics),
                Err(_) => Event::MetricsDown,
            };
            if tx.send(event).is_err() {
                return;
            }
            for _ in 0..10 {
                if !running.load(Ordering::SeqCst) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    });
}

fn draw(app: &mut app::App) -> Result<()> {
    let (rows, cols) = term_size();
    let frame = app.render(rows, cols);
    let mut stdout = std::io::stdout();
    stdout.write_all(frame.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let url = Url::parse(&cli.url)?;
    install_panic_hook();
    let terminal = Terminal::enter()?;
    let result = run(cli, url);
    drop(terminal);
    result
}

fn run(cli: Cli, url: Url) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    spawn_input(tx.clone());
    let running = Arc::new(AtomicBool::new(true));
    spawn_metrics(url.clone(), tx.clone(), running.clone());
    let mut app = app::App::new(cli.system.clone());
    let result = ui_loop(&mut app, &cli, &url, &rx);
    running.store(false, Ordering::SeqCst);
    result
}

fn ui_loop(app: &mut app::App, cli: &Cli, url: &Url, rx: &Receiver<Event>) -> Result<()> {
    loop {
        draw(app)?;
        match rx.recv_timeout(Duration::from_millis(400)) {
            Ok(Event::Key(key)) => match app.handle_key(key) {
                app::IdleAction::None => {}
                app::IdleAction::Exit => return Ok(()),
                app::IdleAction::Submit => {
                    let line = std::mem::take(&mut app.input);
                    if submit(app, cli, url, rx, &line)? {
                        return Ok(());
                    }
                }
            },
            Ok(Event::Metrics(metrics)) => app.apply_metrics(&metrics),
            Ok(Event::MetricsDown) => app.metrics_down(),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
}

fn submit(
    app: &mut app::App,
    cli: &Cli,
    url: &Url,
    rx: &Receiver<Event>,
    line: &str,
) -> Result<bool> {
    match app::parse_command(line) {
        app::Command::None => Ok(false),
        app::Command::Exit => Ok(true),
        app::Command::Send(text) => {
            app.history.push(&text);
            app.scroll.reset_to_bottom();
            app.transcript.push_user(text);
            stream_reply(app, cli, url, rx)?;
            Ok(false)
        }
        app::Command::Reset => {
            app.reset();
            Ok(false)
        }
        app::Command::Save(path) => {
            app.save(path.as_deref());
            Ok(false)
        }
        app::Command::Load(path) => {
            app.load(path.as_deref());
            Ok(false)
        }
        app::Command::Unknown(name) => {
            app.notice(format!("неизвестная команда {name}"));
            Ok(false)
        }
    }
}

fn sampling(cli: &Cli) -> app::Sampling {
    app::Sampling {
        max_tokens: cli.max_tokens.max(1),
        temperature: cli.temperature,
        top_p: cli.top_p,
        seed: cli.seed,
    }
}

fn stream_reply(app: &mut app::App, cli: &Cli, url: &Url, rx: &Receiver<Event>) -> Result<()> {
    let client = Client::new(url.clone());
    let body = app::request_body(&app.transcript.request_messages(), &sampling(cli));
    let started = Instant::now();
    let mut stream = match client.chat_stream(body.as_bytes()) {
        Ok(stream) => stream,
        Err(e) => {
            app.notice(format!("{e}"));
            app.scroll.reset_to_bottom();
            draw(app)?;
            return Ok(());
        }
    };
    app.transcript.begin_assistant();
    app.scroll.reset_to_bottom();
    app.status.generating = true;
    app.status.tokens = 0;
    app.status.ttft_ms = None;
    app.status.tok_per_s = None;

    let mut first: Option<Instant> = None;
    let mut tokens = 0usize;
    let mut last_draw = Instant::now();
    let mut interrupted = false;
    let mut finish: Option<String> = None;
    let mut error: Option<String> = None;

    loop {
        loop {
            match rx.try_recv() {
                Ok(Event::Key(Key::Interrupt)) | Ok(Event::Key(Key::Eof)) => {
                    interrupted = true;
                    break;
                }
                Ok(Event::Key(key)) => app.handle_stream_key(key),
                Ok(Event::Metrics(metrics)) => app.apply_metrics(&metrics),
                Ok(Event::MetricsDown) => app.metrics_down(),
                Err(_) => break,
            }
        }
        if interrupted {
            break;
        }
        match stream.next_step() {
            Ok(StreamStep::Chunk(chunk)) => {
                let mut arrived = false;
                if let Some(text) = chunk.content.as_deref() {
                    app.transcript.append_content(text);
                    tokens += 1;
                    arrived = true;
                }
                if let Some(text) = chunk.thinking.as_deref() {
                    app.transcript.append_thinking(text);
                    arrived = true;
                }
                for tool in &chunk.tools {
                    app.transcript.append_tool(
                        tool.index,
                        tool.name.as_deref(),
                        tool.arguments.as_deref(),
                    );
                    arrived = true;
                }
                if let Some(reason) = chunk.finish {
                    finish = Some(reason);
                }
                let now = Instant::now();
                if arrived && first.is_none() {
                    first = Some(now);
                    app.status.ttft_ms = Some(now.duration_since(started).as_secs_f64() * 1000.0);
                }
                app.status.tokens = tokens;
                if let Some(first) = first {
                    let elapsed = now.duration_since(first).as_secs_f64();
                    if elapsed > 0.0 && tokens > 0 {
                        app.status.tok_per_s = Some(tokens as f64 / elapsed);
                    }
                }
                app.scroll.reset_to_bottom();
                if now.duration_since(last_draw) >= Duration::from_millis(50) {
                    draw(app)?;
                    last_draw = now;
                }
            }
            Ok(StreamStep::Tick) => {
                let now = Instant::now();
                if now.duration_since(last_draw) >= Duration::from_millis(50) {
                    draw(app)?;
                    last_draw = now;
                }
            }
            Ok(StreamStep::Done) | Ok(StreamStep::Eof) => break,
            Err(e) => {
                error = Some(e.to_string());
                break;
            }
        }
    }

    drop(stream);
    app.status.generating = false;
    if let Some(message) = error {
        if app.transcript.drop_empty_assistant() {
            app.notice(format!("ошибка: {message}"));
        } else {
            app.notice(format!("ошибка (ответ неполный): {message}"));
        }
    }
    if interrupted {
        if app.transcript.drop_empty_assistant() {
            app.notice("прервано");
        } else {
            app.notice("прервано (ответ неполный)");
        }
    }
    if finish.as_deref() == Some("length") {
        app.notice("ответ обрезан по max_tokens");
    }
    app.scroll.reset_to_bottom();
    draw(app)?;
    Ok(())
}
