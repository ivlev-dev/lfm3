use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use lfm::{Chat, ChatEvent, Stats};
use lfm_domain::SamplingParams;
use unicode_width::UnicodeWidthStr;

mod md {
    pub use crate::markdown::{plain, render};
}

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[38;5;245m";
const ACCENT: &str = "\x1b[38;5;114m";
const ALERT: &str = "\x1b[38;5;203m";
const TOOL: &str = "\x1b[38;5;221m";

const HELP: &str = "/help — справка · /reset — очистить · /thinking — показать/скрыть рассуждения · /exit — выход";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    You,
    Assistant,
    Thinking,
    Tool,
    System,
}

struct Block {
    kind: Kind,
    text: String,
}

struct TuiState {
    title: String,
    info: String,
    max_ctx: usize,
    history: Vec<Block>,
    assistant: String,
    thinking: String,
    tools: Vec<(String, String)>,
    input: String,
    input_history: InputHistory,
    scroll: usize,
    pending: Vec<u8>,
    show_thinking: bool,
    stats: Option<Stats>,
    live_started: Option<Instant>,
    live_tokens: usize,
    ttft: Option<f64>,
    note: Option<String>,
    last_render: Instant,
}

#[derive(Debug, Default, Clone)]
struct InputHistory {
    entries: Vec<String>,
    pos: usize,
    draft: String,
}

impl InputHistory {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, line: &str) {
        let line = line.trim();
        if !line.is_empty() && self.entries.last().map(|l| l.as_str()) != Some(line) {
            self.entries.push(line.to_string());
        }
        self.pos = self.entries.len();
        self.draft.clear();
    }

    fn up(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        if self.pos == self.entries.len() {
            self.draft = current.to_string();
        }
        if self.pos > 0 {
            self.pos -= 1;
        }
        self.entries.get(self.pos).cloned()
    }

    fn down(&mut self) -> Option<String> {
        if self.pos >= self.entries.len() {
            return None;
        }
        self.pos += 1;
        if self.pos == self.entries.len() {
            Some(std::mem::take(&mut self.draft))
        } else {
            self.entries.get(self.pos).cloned()
        }
    }
}

fn term_size() -> (usize, usize) {
    if let Ok(out) = std::process::Command::new("stty").arg("size").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut parts = text.split_whitespace();
        if let (Some(r), Some(c)) = (parts.next(), parts.next())
            && let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>())
            && r > 4
            && c > 20
        {
            return (r, c);
        }
    }
    let rows = std::env::var("LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let cols = std::env::var("COLUMNS").ok().and_then(|v| v.parse().ok()).unwrap_or(80);
    (rows, cols)
}

fn set_term_raw(on: bool) -> Result<()> {
    let mut cmd = std::process::Command::new("stty");
    if on {
        cmd.args(["raw", "-echo"]);
    } else {
        cmd.arg("sane");
    }
    cmd.status()?;
    Ok(())
}

fn set_term_timeout(on: bool) -> Result<()> {
    let mut cmd = std::process::Command::new("stty");
    if on {
        cmd.args(["min", "0", "time", "0"]);
    } else {
        cmd.args(["min", "1", "time", "0"]);
    }
    cmd.status()?;
    Ok(())
}

fn poll_interrupt(pending: &mut Vec<u8>) -> bool {
    let mut buf = [0u8; 64];
    let mut interrupt = false;
    match std::io::stdin().read(&mut buf) {
        Ok(n) => {
            for byte in &buf[..n] {
                if *byte == 3 {
                    interrupt = true;
                } else {
                    pending.push(*byte);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
        Err(_) => {}
    }
    interrupt
}

fn ansi_width(text: &str) -> usize {
    let mut used = 0;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        used += unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
    }
    used
}

fn fit(text: &str, cols: usize) -> String {
    if ansi_width(text) <= cols {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            out.push(ch);
            if chars.peek() == Some(&'[') {
                out.push(chars.next().unwrap());
                for c in chars.by_ref() {
                    out.push(c);
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w > cols {
            break;
        }
        out.push(ch);
        used += w;
    }
    if !out.is_empty() {
        out.push_str(RESET);
    }
    out
}

fn pad_or_fit(text: &str, cols: usize) -> String {
    let mut out = fit(text, cols);
    let w = ansi_width(&out);
    if w < cols {
        out.push_str(&" ".repeat(cols - w));
    }
    out
}

fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    let width = width.max(4);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut used = 0;
        for ch in raw.chars() {
            let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + w > width && used > 0 {
                out.push(std::mem::take(&mut line));
                used = 0;
            }
            line.push(ch);
            used += w;
        }
        out.push(line);
    }
    out
}

fn push_prefixed(lines: &mut Vec<String>, prefix: &str, text: &str, cols: usize) {
    let pad = ansi_width(prefix);
    let width = cols.saturating_sub(pad).max(4);
    for (i, l) in wrap_plain(text, width).into_iter().enumerate() {
        if i == 0 {
            lines.push(format!("{prefix}{l}"));
        } else {
            lines.push(format!("{}{l}", " ".repeat(pad)));
        }
    }
}

fn push_block(lines: &mut Vec<String>, block: &Block, cols: usize) {
    match block.kind {
        Kind::You => {
            push_prefixed(lines, &format!("{ACCENT}you ›{RESET} "), &block.text, cols);
            lines.push(String::new());
        }
        Kind::Assistant => {
            for l in md::render(&block.text, cols.saturating_sub(2)) {
                lines.push(format!("  {l}"));
            }
            lines.push(String::new());
        }
        Kind::Thinking => {
            let head = format!("{DIM}thinking ›{RESET}");
            let mut first = true;
            for l in md::plain(&block.text, cols.saturating_sub(13)) {
                if first {
                    lines.push(format!("{head} {l}{RESET}"));
                    first = false;
                } else {
                    lines.push(format!("           {l}{RESET}"));
                }
            }
            lines.push(String::new());
        }
        Kind::Tool => {
            let (name, args) = block.text.split_once('\n').unwrap_or((&block.text, ""));
            lines.push(format!(
                "{TOOL}⚙ {name}{RESET} {DIM}{}{RESET}",
                fit(args, cols.saturating_sub(name.width() + 4))
            ));
        }
        Kind::System => {
            for l in md::plain(&block.text, cols.saturating_sub(2)) {
                lines.push(format!("{l}"));
            }
            lines.push(String::new());
        }
    }
}

fn body_lines(state: &TuiState, cols: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for block in &state.history {
        push_block(&mut lines, block, cols);
    }
    if !state.thinking.is_empty() && state.show_thinking {
        push_block(
            &mut lines,
            &Block {
                kind: Kind::Thinking,
                text: state.thinking.clone(),
            },
            cols,
        );
    }
    for (name, args) in &state.tools {
        push_block(
            &mut lines,
            &Block {
                kind: Kind::Tool,
                text: format!("{name}\n{args}"),
            },
            cols,
        );
    }
    if !state.assistant.is_empty() {
        let text = state.assistant.clone();
        for l in md::render(&text, cols.saturating_sub(2)) {
            lines.push(format!("  {l}"));
        }
    }
    if let Some(note) = &state.note {
        for l in md::plain(note, cols.saturating_sub(2)) {
            lines.push(format!("{ALERT}{l}{RESET}"));
        }
    }
    lines
}

fn stats_line(state: &TuiState) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &state.stats {
        parts.push(format!("prefill {:.0} tok/s", s.prefill_tok_s()));
        parts.push(format!("decode {:.1} tok/s", s.decode_tok_s()));
        parts.push(format!("TTFT {:.2}s", s.ttft_s));
        parts.push(format!("{} tok", s.completion_tokens));
    } else if let Some(start) = state.live_started {
        let elapsed = start.elapsed().as_secs_f64();
        let tps = state.live_tokens as f64 / elapsed.max(1e-9);
        let ttft = state
            .ttft
            .map(|t| format!("{t:.2}s"))
            .unwrap_or_else(|| "…".to_string());
        parts.push(format!("{tps:.1} tok/s"));
        parts.push(format!("TTFT {ttft}"));
        parts.push(format!("{} tok", state.live_tokens));
        parts.push("Esc/Ctrl+C — стоп".to_string());
    } else {
        parts.push(format!("ctx 0/{}", state.max_ctx));
        parts.push(format!(
            "thinking: {}",
            if state.show_thinking { "on" } else { "off" }
        ));
    }
    if state.scroll > 0 {
        parts.push(format!("↑{}", state.scroll));
    }
    parts.join(" · ")
}

fn render(state: &mut TuiState) {
    let (rows, cols) = term_size();
    let cols = cols.max(24);
    let rows = rows.max(6);
    let body_rows = rows.saturating_sub(3).max(1);
    let lines = body_lines(state, cols);
    let total = lines.len();
    let max_scroll = total.saturating_sub(body_rows);
    let scroll = state.scroll.min(max_scroll);
    let start = max_scroll - scroll;
    let mut out = String::new();
    // header
    let title = format!(
        " lfm · {} · {} ",
        state.title,
        state.info
    );
    out.push_str(&format!("\x1b[1;1H\x1b[48;5;236m\x1b[38;5;252m{}\x1b[0m", pad_or_fit(&title, cols)));
    // body
    for i in 0..body_rows {
        out.push_str(&format!("\x1b[{};1H\x1b[2K", i + 2));
        if let Some(line) = lines.get(start + i) {
            out.push_str(&fit(line, cols));
        }
    }
    // stats
    let stats = stats_line(state);
    out.push_str(&format!(
        "\x1b[{};1H\x1b[2K{DIM}{}\x1b[0m",
        rows - 1,
        pad_or_fit(&format!(" {stats}"), cols)
    ));
    // input
    let prompt = format!("{ACCENT}you ›{RESET} ");
    out.push_str(&format!("\x1b[{};1H\x1b[2K{}\x1b[38;5;252m{}\x1b[0m", rows, prompt, fit(&state.input, cols.saturating_sub(6))));
    out.push_str("\x1b[?25h");
    print!("{out}");
    let _ = std::io::stdout().flush();
    state.last_render = Instant::now();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    Up,
    Down,
    PgUp,
    PgDn,
    Home,
    End,
    Other,
}

fn parse_escape(seq: &str) -> Key {
    match seq {
        "[A" | "OA" => Key::Up,
        "[B" | "OB" => Key::Down,
        "[5~" => Key::PgUp,
        "[6~" => Key::PgDn,
        "[H" | "[1~" | "[7~" | "OH" => Key::Home,
        "[F" | "[4~" | "[8~" | "OF" => Key::End,
        _ => Key::Other,
    }
}

fn read_escape(stdin: &mut std::io::Stdin) -> Option<Key> {
    let mut b = [0u8; 1];
    if stdin.read(&mut b).ok()? == 0 {
        return None;
    }
    if b[0] != b'[' && b[0] != b'O' {
        return Some(Key::Other);
    }
    let mut seq = String::new();
    seq.push(b[0] as char);
    loop {
        if stdin.read(&mut b).ok()? == 0 {
            return None;
        }
        seq.push(b[0] as char);
        if seq.len() > 8 {
            break;
        }
        if (0x40..=0x7e).contains(&b[0]) {
            break;
        }
    }
    Some(parse_escape(&seq))
}

fn scroll_by(state: &mut TuiState, delta: isize) {
    let (rows, cols) = term_size();
    let lines = body_lines(state, cols.max(24));
    let area = rows.saturating_sub(3).max(1);
    let max_scroll = lines.len().saturating_sub(area);
    let next = if delta >= 0 {
        state.scroll.saturating_add(delta as usize)
    } else {
        state.scroll.saturating_sub((-delta) as usize)
    };
    state.scroll = next.min(max_scroll);
}

fn handle_nav_key(state: &mut TuiState, key: Key) {
    let (rows, _) = term_size();
    let area = rows.saturating_sub(3).max(1);
    match key {
        Key::PgUp => scroll_by(state, area as isize),
        Key::PgDn => scroll_by(state, -(area as isize)),
        Key::Home => {
            let (rows, cols) = term_size();
            let lines = body_lines(state, cols.max(24));
            state.scroll = lines.len().saturating_sub(rows.saturating_sub(3).max(1));
        }
        Key::End => state.scroll = 0,
        Key::Up if state.input.is_empty() => scroll_by(state, 1),
        Key::Down if state.input.is_empty() => scroll_by(state, -1),
        Key::Up => {
            if let Some(line) = state.input_history.up(&state.input) {
                state.input = line;
            }
        }
        Key::Down => {
            if let Some(line) = state.input_history.down() {
                state.input = line;
            }
        }
        Key::Other => {}
    }
    render(state);
}

fn read_input_line(state: &mut TuiState) -> Option<String> {
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 1];
    let mut utf8: Vec<u8> = Vec::new();
    loop {
        let byte = if state.pending.is_empty() {
            match stdin.read(&mut buf) {
                Ok(0) => return None,
                Ok(_) => buf[0],
                Err(_) => return None,
            }
        } else {
            state.pending.remove(0)
        };
        match byte {
            3 | 4 => return None,
            0x14 => {
                state.show_thinking = !state.show_thinking;
                render(state);
            }
            b'\r' | b'\n' => {
                state.input_history.push(&state.input);
                return Some(std::mem::take(&mut state.input));
            }
            0x7f | 0x08 => {
                state.input.pop();
                utf8.clear();
                render(state);
            }
            0x1b => {
                let key = read_escape(&mut stdin)?;
                handle_nav_key(state, key);
            }
            b if b >= 0x20 => {
                utf8.push(b);
                if let Ok(text) = std::str::from_utf8(&utf8) {
                    state.input.push_str(text);
                    utf8.clear();
                }
                render(state);
            }
            _ => {}
        }
    }
}

pub fn run_chat(
    chat: &mut Chat,
    model_name: &str,
    info: &str,
    params: &SamplingParams,
    show_stats: bool,
) -> Result<()> {
    let mut state = TuiState {
        title: model_name.to_string(),
        info: info.to_string(),
        max_ctx: chat.runtime.max_ctx,
        history: Vec::new(),
        assistant: String::new(),
        thinking: String::new(),
        tools: Vec::new(),
        input: String::new(),
        input_history: InputHistory::new(),
        scroll: 0,
        pending: Vec::new(),
        show_thinking: true,
        stats: None,
        live_started: None,
        live_tokens: 0,
        ttft: None,
        note: None,
        last_render: Instant::now(),
    };
    set_term_raw(true)?;
    print!("\x1b[?1049h\x1b[2J");
    let _ = std::io::stdout().flush();
    let result = tui_loop(chat, params, show_stats, &mut state);
    print!("\x1b[?25h\x1b[?1049l");
    let _ = std::io::stdout().flush();
    set_term_raw(false)?;
    result
}

fn tui_loop(
    chat: &mut Chat,
    params: &SamplingParams,
    show_stats: bool,
    state: &mut TuiState,
) -> Result<()> {
    loop {
        render(state);
        let Some(line) = read_input_line(state) else {
            return Ok(());
        };
        let text = line.trim().to_string();
        if text.is_empty() {
            continue;
        }
        match text.as_str() {
            "/exit" | "/quit" => return Ok(()),
            "/help" => {
                state.history.push(Block {
                    kind: Kind::System,
                    text: HELP.to_string(),
                });
                continue;
            }
            "/reset" => {
                chat.reset();
                state.history.clear();
                state.stats = None;
                state.note = None;
                state.scroll = 0;
                continue;
            }
            "/thinking" => {
                state.show_thinking = !state.show_thinking;
                continue;
            }
            _ => {}
        }
        state.history.push(Block {
            kind: Kind::You,
            text: text.clone(),
        });
        state.assistant.clear();
        state.thinking.clear();
        state.tools.clear();
        state.note = None;
        state.stats = None;
        state.scroll = 0;
        state.live_started = Some(Instant::now());
        state.live_tokens = 0;
        state.ttft = None;
        let cancel = AtomicBool::new(false);
        set_term_timeout(true)?;
        let res = chat.send_stream(&text, params, &cancel, |event| {
            if poll_interrupt(&mut state.pending) {
                cancel.store(true, Ordering::Relaxed);
            }
            match event {
                ChatEvent::Token(_) => {
                    state.live_tokens += 1;
                    if state.ttft.is_none() {
                        state.ttft = state.live_started.map(|s| s.elapsed().as_secs_f64());
                    }
                }
                ChatEvent::Thinking(t) => state.thinking.push_str(&t),
                ChatEvent::Content(t) => state.assistant.push_str(&t),
                ChatEvent::ToolCall(call) => {
                    state.tools.push((call.name, call.arguments.to_string()));
                }
            }
            state.scroll = 0;
            if state.last_render.elapsed() >= Duration::from_millis(30) {
                render(state);
            }
        });
        set_term_timeout(false)?;
        state.live_started = None;
        match res {
            Ok(reply) => {
                let cancelled = cancel.load(Ordering::Relaxed);
                if !state.thinking.is_empty() && state.show_thinking {
                    let thinking = std::mem::take(&mut state.thinking);
                    state.history.push(Block {
                        kind: Kind::Thinking,
                        text: thinking,
                    });
                } else {
                    state.thinking.clear();
                }
                let tools = std::mem::take(&mut state.tools);
                for (name, args) in tools {
                    state.history.push(Block {
                        kind: Kind::Tool,
                        text: format!("{name}\n{args}"),
                    });
                }
                let content = std::mem::take(&mut state.assistant);
                if content.is_empty() {
                    state.history.pop();
                } else {
                    state.history.push(Block {
                        kind: Kind::Assistant,
                        text: content,
                    });
                }
                if cancelled {
                    state.note = Some("остановлено".to_string());
                }
                if show_stats {
                    state.stats = Some(reply.stats);
                }
            }
            Err(e) => {
                state.assistant.clear();
                state.thinking.clear();
                state.tools.clear();
                state.note = Some(format!("ошибка: {e}"));
            }
        }
        render(state);
    }
}

pub fn run_once(
    chat: &mut Chat,
    params: &SamplingParams,
    text: &str,
    show_stats: bool,
) -> Result<()> {
    let cancel = AtomicBool::new(false);
    let reply = chat.send_stream(text, params, &cancel, |event| match event {
        ChatEvent::Thinking(t) => {
            eprint!("{DIM}{t}{RESET}");
            let _ = std::io::stderr().flush();
        }
        ChatEvent::Content(t) => {
            print!("{t}");
            let _ = std::io::stdout().flush();
        }
        ChatEvent::ToolCall(call) => {
            println!("\n[tool] {} {}", call.name, call.arguments);
        }
        ChatEvent::Token(_) => {}
    })?;
    println!();
    if show_stats {
        let s = reply.stats;
        eprintln!(
            "[prefill {:.0} tok/s, decode {:.1} tok/s, TTFT {:.2}s, {} токенов]",
            s.prefill_tok_s(),
            s.decode_tok_s(),
            s.ttft_s,
            s.completion_tokens
        );
    }
    Ok(())
}
