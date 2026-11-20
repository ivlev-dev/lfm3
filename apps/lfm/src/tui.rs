use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use lfm::{Chat, ChatEvent};
use lfm_domain::SamplingParams;

const TOOL_ARGS_MAX: usize = 2000;

const TUI_HELP: &str = "/help — эта справка\n/reset — очистить историю\n/exit, /quit — выход\nPgUp/PgDn — листать историю\nHome/End — начало/конец\nUp/Down — история ввода (при пустой строке — прокрутка)\nCtrl+C — выход";

struct TuiState {
    history: Vec<(String, String)>,
    assistant: String,
    thinking: String,
    tools: Vec<(String, String)>,
    input: String,
    input_history: InputHistory,
    status: String,
    scroll: usize,
    pending: Vec<u8>,
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
        self.reset();
    }

    fn reset(&mut self) {
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

fn scroll_after_pgup(scroll: usize, area: usize) -> usize {
    scroll.saturating_add(area.max(1))
}

fn scroll_after_pgdn(scroll: usize, area: usize) -> usize {
    scroll.saturating_sub(area.max(1))
}

fn scroll_home(total: usize, area: usize) -> usize {
    total.saturating_sub(area.max(1))
}

fn scroll_end() -> usize {
    0
}

fn term_size() -> (usize, usize) {
    let out = std::process::Command::new("stty").arg("size").output();
    if let Ok(out) = out {
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
    (24, 80)
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

fn clip(text: &str, cols: usize) -> String {
    text.chars().take(cols).collect()
}

fn wrap(text: &str, cols: usize) -> Vec<String> {
    let width = cols.max(8);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut count = 0usize;
        for ch in raw.chars() {
            if count == width {
                out.push(std::mem::take(&mut line));
                count = 0;
            }
            line.push(ch);
            count += 1;
        }
        out.push(line);
    }
    out
}

fn history_prefix(who: &str) -> &'static str {
    match who {
        "you" => "you> ",
        "ai" => "ai>  ",
        _ => "[sys] ",
    }
}

fn format_tool_call(name: &str, args: &str, cols: usize) -> Vec<String> {
    let width = cols.max(8);
    let head = format!("[tool] {}", if name.is_empty() { "?" } else { name });
    let mut out = vec![clip(&head, cols)];
    let args = args.trim();
    if args.is_empty() {
        return out;
    }
    let total = args.chars().count();
    let mut body: String = args.chars().take(TOOL_ARGS_MAX).collect();
    if total > TOOL_ARGS_MAX {
        body.push('…');
    }
    for line in wrap(&body, width.saturating_sub(2)) {
        out.push(clip(&format!("  {line}"), cols));
    }
    out
}

fn push_block(lines: &mut Vec<String>, head: String, indent: usize, cols: usize) {
    let pad = " ".repeat(indent);
    for (i, l) in wrap(&head, cols).into_iter().enumerate() {
        lines.push(if i == 0 { l } else { format!("{pad}{l}") });
    }
}

fn transcript_lines(state: &TuiState, cols: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for (who, text) in &state.history {
        let prefix = history_prefix(who);
        push_block(
            &mut lines,
            format!("{prefix}{text}"),
            prefix.chars().count(),
            cols,
        );
        lines.push(String::new());
    }
    if !state.thinking.is_empty() {
        push_block(&mut lines, format!("[think] {}", state.thinking), 7, cols);
    }
    if !state.assistant.is_empty() {
        push_block(&mut lines, format!("ai>  {}", state.assistant), 5, cols);
    }
    for (name, args) in &state.tools {
        lines.extend(format_tool_call(name, args, cols));
    }
    lines
}

#[derive(Debug, Clone, Default)]
struct LinesView {
    body: Vec<String>,
    area: usize,
    total: usize,
    scroll: usize,
}

fn body_area(rows: usize) -> usize {
    rows.saturating_sub(3).max(1)
}

fn visible_lines(state: &TuiState, rows: usize, cols: usize) -> LinesView {
    let lines = transcript_lines(state, cols);
    let area = body_area(rows);
    let total = lines.len();
    let max_scroll = total.saturating_sub(area);
    let scroll = state.scroll.min(max_scroll);
    let start = max_scroll - scroll;
    let mut body: Vec<String> = Vec::with_capacity(area);
    for i in 0..area {
        body.push(lines.get(start + i).cloned().unwrap_or_default());
    }
    LinesView {
        body,
        area,
        total,
        scroll,
    }
}

fn tui_render(state: &TuiState, rows: usize, cols: usize) {
    let view = visible_lines(state, rows, cols);
    let mut out = String::from("\x1b[H");
    for line in &view.body {
        out.push_str("\x1b[2K");
        out.push_str(&clip(line, cols));
        out.push_str("\r\n");
    }
    let mut status = state.status.clone();
    if view.scroll > 0 {
        status.push_str(&format!(" | ↑{}", view.scroll));
    }
    out.push_str("\x1b[2K\x1b[7m");
    out.push_str(&clip(&status, cols));
    out.push_str("\x1b[0m\r\n\x1b[2Kyou> ");
    out.push_str(&clip(&state.input, cols.saturating_sub(5)));
    out.push_str("\x1b[?25h");
    print!("{out}");
    let _ = std::io::stdout().flush();
}

fn render_now(state: &TuiState) {
    let (rows, cols) = term_size();
    tui_render(state, rows, cols);
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

fn scroll_key(state: &mut TuiState, key: Key) {
    let (rows, cols) = term_size();
    let view = visible_lines(state, rows, cols);
    let max_scroll = view.total.saturating_sub(view.area);
    let area = view.area;
    let next = match key {
        Key::PgUp => scroll_after_pgup(state.scroll, area),
        Key::PgDn => scroll_after_pgdn(state.scroll, area),
        Key::Up => scroll_after_pgup(state.scroll, 1),
        Key::Down => scroll_after_pgdn(state.scroll, 1),
        Key::Home => scroll_home(view.total, area),
        Key::End | Key::Other => scroll_end(),
    };
    state.scroll = next.min(max_scroll);
    tui_render(state, rows, cols);
}

fn handle_nav_key(state: &mut TuiState, key: Key) {
    match key {
        Key::PgUp | Key::PgDn | Key::Home | Key::End => scroll_key(state, key),
        Key::Up if state.input.is_empty() => scroll_key(state, Key::Up),
        Key::Down if state.input.is_empty() => scroll_key(state, Key::Down),
        Key::Up => {
            if let Some(line) = state.input_history.up(&state.input) {
                state.input = line;
            }
            render_now(state);
        }
        Key::Down => {
            if let Some(line) = state.input_history.down() {
                state.input = line;
                render_now(state);
            }
        }
        Key::Other => {}
    }
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
            b'\r' | b'\n' => {
                state.input_history.reset();
                return Some(std::mem::take(&mut state.input));
            }
            0x7f | 0x08 => {
                state.input.pop();
                utf8.clear();
                render_now(state);
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
                render_now(state);
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]

pub fn run_chat(
    chat: &mut Chat,
    model_name: &str,
    params: &SamplingParams,
    show_stats: bool,
) -> Result<()> {
    let mut state = TuiState {
        history: Vec::new(),
        assistant: String::new(),
        thinking: String::new(),
        tools: Vec::new(),
        input: String::new(),
        input_history: InputHistory::new(),
        status: format!(
            "{model_name} | ctx {} | PgUp/PgDn — история, /help | Ctrl+C — выход",
            chat.runtime.max_ctx
        ),
        scroll: 0,
        pending: Vec::new(),
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
    let base_status = state.status.clone();
    loop {
        let (rows, cols) = term_size();
        tui_render(state, rows, cols);
        let Some(line) = read_input_line(state) else {
            return Ok(());
        };
        let text = line.trim().to_string();
        if text.is_empty() {
            continue;
        }
        if text == "/exit" || text == "/quit" {
            return Ok(());
        }
        state.input_history.push(&text);
        state.scroll = 0;
        if text == "/help" {
            state
                .history
                .push(("sys".to_string(), TUI_HELP.to_string()));
            continue;
        }
        if text == "/reset" {
            chat.reset();
            state.history.clear();
            state.tools.clear();
            state.status = base_status.clone();
            continue;
        }
        state.history.push(("you".to_string(), text.clone()));
        state.assistant.clear();
        state.thinking.clear();
        state.tools.clear();
        let started = Instant::now();
        let mut tokens = 0usize;
        let mut first: Option<f64> = None;
        let cancel = AtomicBool::new(false);
        set_term_timeout(true)?;
        let res = chat.send_stream(&text, params, &cancel, |event| {
            if poll_interrupt(&mut state.pending) {
                cancel.store(true, Ordering::Relaxed);
            }
            state.scroll = 0;
            match event {
                ChatEvent::Token(_) => tokens += 1,
                ChatEvent::Thinking(t) => state.thinking.push_str(&t),
                ChatEvent::Content(t) => state.assistant.push_str(&t),
                ChatEvent::ToolCall(tc) => {
                    state.tools.push((tc.name, tc.arguments.to_string()));
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            let ttft = *first.get_or_insert(elapsed);
            if show_stats {
                state.status = format!(
                    "{base_status} | TTFT {ttft:.2} c | {:.1} tok/s | {tokens} ток.",
                    tokens as f64 / elapsed.max(1e-9)
                );
            }
            let (rows, cols) = term_size();
            tui_render(state, rows, cols);
        });
        set_term_timeout(false)?;
        let cancelled = cancel.load(Ordering::Relaxed);
        match res {
            Ok(reply) => {
                if cancelled {
                    state.status = format!(
                        "прервано ({} ток.) | {base_status}",
                        reply.stats.completion_tokens
                    );
                } else if show_stats {
                    let stats = reply.stats;
                    state.status = format!(
                        "{base_status} | prefill {:.1} tok/s | decode {:.1} tok/s | TTFT {:.2} c | {} ток.",
                        stats.prefill_tok_s(),
                        stats.decode_tok_s(),
                        stats.ttft_s,
                        stats.completion_tokens
                    );
                }
                let (_, cols) = term_size();
                let mut answer = std::mem::take(&mut state.assistant);
                for (name, args) in state.tools.drain(..) {
                    if !answer.is_empty() && !answer.ends_with('\n') {
                        answer.push('\n');
                    }
                    answer.push_str(&format_tool_call(&name, &args, cols).join("\n"));
                }
                if answer.is_empty() {
                    state.history.pop();
                    continue;
                }
                state.history.push(("ai".to_string(), answer));
            }
            Err(e) => {
                state.status = format!("ошибка: {e} | {base_status}");
                state.history.pop();
            }
        }
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
            eprint!("{t}");
            let _ = std::io::stderr().flush();
        }
        ChatEvent::Content(t) => {
            print!("{t}");
            let _ = std::io::stdout().flush();
        }
        ChatEvent::ToolCall(tc) => {
            println!("\n[tool_call] {} {}", tc.name, tc.arguments);
        }
        ChatEvent::Token(_) => {}
    })?;
    println!();
    if show_stats {
        let stats = reply.stats;
        eprintln!(
            "[prefill {:.1} tok/s, decode {:.1} tok/s, TTFT {:.2} c, {} токенов]",
            stats.prefill_tok_s(),
            stats.decode_tok_s(),
            stats.ttft_s,
            stats.completion_tokens
        );
    }
    Ok(())
}
