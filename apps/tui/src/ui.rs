use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const HINT: &str =
    "Enter — отправить · PgUp/PgDn · Home/End · Ctrl+C стоп · Ctrl+D выход · /help — команды";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl ToolCall {
    pub fn append(&mut self, name: Option<&str>, arguments: Option<&str>) {
        if let Some(name) = name.filter(|_| self.name.is_empty()) {
            self.name = name.to_string();
        }
        if let Some(arguments) = arguments {
            self.arguments.push_str(arguments);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            thinking: String::new(),
            tools: Vec::new(),
        }
    }

    pub fn is_empty_assistant(&self) -> bool {
        self.role == Role::Assistant
            && self.content.trim().is_empty()
            && self.thinking.trim().is_empty()
            && self.tools.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    messages: Vec<Message>,
}

impl Transcript {
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn replace(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn set_system(&mut self, text: &str) {
        self.messages.retain(|message| message.role != Role::System);
        if !text.trim().is_empty() {
            self.messages.insert(0, Message::new(Role::System, text));
        }
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message::new(Role::User, text));
    }

    pub fn push_system(&mut self, text: impl Into<String>) {
        self.messages.push(Message::new(Role::System, text));
    }

    pub fn begin_assistant(&mut self) {
        if !matches!(self.messages.last(), Some(msg) if msg.role == Role::Assistant) {
            self.messages
                .push(Message::new(Role::Assistant, String::new()));
        }
    }

    fn last_assistant(&mut self) -> &mut Message {
        if !matches!(self.messages.last(), Some(msg) if msg.role == Role::Assistant) {
            self.messages
                .push(Message::new(Role::Assistant, String::new()));
        }
        self.messages.last_mut().expect("assistant just pushed")
    }

    pub fn append_content(&mut self, text: &str) {
        self.last_assistant().content.push_str(text);
    }

    pub fn append_thinking(&mut self, text: &str) {
        self.last_assistant().thinking.push_str(text);
    }

    pub fn append_tool(&mut self, index: usize, name: Option<&str>, arguments: Option<&str>) {
        let message = self.last_assistant();
        while message.tools.len() <= index {
            message.tools.push(ToolCall::default());
        }
        message.tools[index].append(name, arguments);
    }

    pub fn drop_empty_assistant(&mut self) -> bool {
        match self.messages.last() {
            Some(message) if message.is_empty_assistant() => {
                self.messages.pop();
                true
            }
            _ => false,
        }
    }

    pub fn request_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|message| match message.role {
                Role::Assistant => !message.content.trim().is_empty(),
                Role::System => !message.content.trim().is_empty(),
                Role::User => true,
            })
            .map(|message| Message::new(message.role, message.content.clone()))
            .collect()
    }
}

pub fn save_transcript(path: &Path, messages: &[Message]) -> Result<()> {
    let text = serde_json::to_string_pretty(messages).context("сериализация истории")?;
    std::fs::write(path, text).with_context(|| format!("запись {}", path.display()))
}

pub fn load_transcript(path: &Path) -> Result<Vec<Message>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("чтение {}", path.display()))?;
    let messages: Vec<Message> =
        serde_json::from_str(&text).map_err(|e| anyhow!("некорректный JSON истории: {e}"))?;
    Ok(messages
        .into_iter()
        .filter(|message| !message.is_empty_assistant())
        .collect())
}

pub fn char_count(text: &str) -> usize {
    text.chars().count()
}

pub fn truncate(text: &str, cols: usize) -> String {
    if char_count(text) <= cols {
        return text.to_string();
    }
    text.chars().take(cols).collect()
}

pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut count = 0usize;
        for ch in raw.chars() {
            if count >= width {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeKind {
    Info,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    pub kind: NoticeKind,
    pub text: String,
}

impl Notice {
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            kind: NoticeKind::Info,
            text: text.into(),
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: NoticeKind::Error,
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    System,
    User,
    Assistant,
    Thinking,
    Tool,
    Notice,
    Info,
    Blank,
}

impl LineKind {
    fn style(self) -> &'static str {
        match self {
            LineKind::Thinking => "\x1b[2m",
            LineKind::Tool => "\x1b[33m",
            LineKind::User => "\x1b[36m",
            LineKind::System => "\x1b[35m",
            LineKind::Notice => "\x1b[31m",
            LineKind::Info => "\x1b[32m",
            LineKind::Assistant | LineKind::Blank => "",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub kind: LineKind,
    pub text: String,
}

fn block(out: &mut Vec<Line>, prefix: &str, text: &str, kind: LineKind, cols: usize) {
    let lead = char_count(prefix);
    let width = cols.saturating_sub(lead).max(1);
    let pad = " ".repeat(lead);
    for (index, segment) in wrap_text(text, width).into_iter().enumerate() {
        let first = if index == 0 { prefix } else { pad.as_str() };
        out.push(Line {
            kind,
            text: format!("{first}{segment}"),
        });
    }
}

pub fn transcript_lines(messages: &[Message], notices: &[Notice], cols: usize) -> Vec<Line> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {
                if !message.content.trim().is_empty() {
                    block(&mut out, "[sys] ", &message.content, LineKind::System, cols);
                }
            }
            Role::User => {
                block(&mut out, "you> ", &message.content, LineKind::User, cols);
            }
            Role::Assistant => {
                if !message.thinking.is_empty() {
                    block(
                        &mut out,
                        "[think] ",
                        &message.thinking,
                        LineKind::Thinking,
                        cols,
                    );
                }
                if !message.content.is_empty() {
                    block(
                        &mut out,
                        "ai> ",
                        &message.content,
                        LineKind::Assistant,
                        cols,
                    );
                }
                for tool in &message.tools {
                    let text = format!("{}({})", tool.name, tool.arguments);
                    block(&mut out, "[tool] ", &text, LineKind::Tool, cols);
                }
            }
        }
        if !message.thinking.is_empty() || !message.content.is_empty() || !message.tools.is_empty()
        {
            out.push(Line {
                kind: LineKind::Blank,
                text: String::new(),
            });
        }
    }
    for notice in notices {
        match notice.kind {
            NoticeKind::Info => block(&mut out, "· ", &notice.text, LineKind::Info, cols),
            NoticeKind::Error => block(&mut out, "[!] ", &notice.text, LineKind::Notice, cols),
        }
    }
    out
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scroll {
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollView {
    pub start: usize,
    pub end: usize,
    pub offset: usize,
}

impl Scroll {
    pub fn reset_to_bottom(&mut self) {
        self.offset = 0;
    }

    pub fn top(&mut self, total: usize, area: usize) {
        self.offset = total.saturating_sub(area.max(1));
    }

    pub fn up(&mut self, lines: usize, total: usize, area: usize) {
        let max = total.saturating_sub(area.max(1));
        self.offset = (self.offset + lines).min(max);
    }

    pub fn down(&mut self, lines: usize) {
        self.offset = self.offset.saturating_sub(lines);
    }

    pub fn view(&self, total: usize, area: usize) -> ScrollView {
        let area = area.max(1);
        let max_offset = total.saturating_sub(area);
        let offset = self.offset.min(max_offset);
        let end = total - offset;
        let start = end.saturating_sub(area);
        ScrollView { start, end, offset }
    }
}

#[derive(Debug, Clone, Default)]
pub struct InputHistory {
    entries: Vec<String>,
    cursor: usize,
}

impl InputHistory {
    pub fn push(&mut self, line: &str) {
        let text = line.trim();
        if text.is_empty() {
            self.cursor = self.entries.len();
            return;
        }
        if self.entries.last().map(String::as_str) != Some(text) {
            self.entries.push(text.to_string());
        }
        self.cursor = self.entries.len();
    }

    pub fn up(&mut self) -> Option<String> {
        if self.entries.is_empty() || self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(self.entries[self.cursor].clone())
    }

    pub fn down(&mut self) -> Option<String> {
        if self.cursor >= self.entries.len() {
            return None;
        }
        self.cursor += 1;
        if self.cursor == self.entries.len() {
            Some(String::new())
        } else {
            Some(self.entries[self.cursor].clone())
        }
    }

    pub fn reset_cursor(&mut self) {
        self.cursor = self.entries.len();
    }
}

#[derive(Debug, Clone, Default)]
pub struct Status {
    pub model: String,
    pub ctx: usize,
    pub active: usize,
    pub waiting: usize,
    pub kv_used: usize,
    pub kv_free: usize,
    pub kv_cached: usize,
    pub cache_hits: usize,
    pub reused_tokens: usize,
    pub steps: usize,
    pub avg_step_ms: f64,
    pub uptime_s: f64,
    pub ttft_ms: Option<f64>,
    pub tok_per_s: Option<f64>,
    pub tokens: usize,
    pub history_tokens: usize,
    pub generating: bool,
    pub online: bool,
    pub scroll: usize,
}

fn seconds(ms: f64) -> String {
    if ms < 1000.0 {
        format!("{ms:.0}ms")
    } else {
        format!("{:.1}s", ms / 1000.0)
    }
}

impl Status {
    pub fn line(&self) -> String {
        let model = if self.model.is_empty() {
            "lfm3"
        } else {
            self.model.as_str()
        };
        let mut parts: Vec<String> = Vec::new();
        if self.scroll > 0 {
            parts.push(format!("↑ {}", self.scroll));
        }
        if self.ctx > 0 {
            parts.push(format!("{model} ctx {}", self.ctx));
        } else {
            parts.push(model.to_string());
        }
        parts.push(format!("active {}/{}", self.active, self.waiting));
        let mut kv = format!("kv {}/{}", self.kv_used, self.kv_free);
        if self.kv_cached > 0 {
            kv.push_str(&format!("+{}", self.kv_cached));
        }
        parts.push(kv);
        let mut cache = format!("cache {}", self.cache_hits);
        if self.reused_tokens > 0 {
            cache.push_str(&format!("/{}", self.reused_tokens));
        }
        parts.push(cache);
        if let Some(ttft) = self.ttft_ms {
            parts.push(format!("ttft {}", seconds(ttft)));
        }
        match self.tok_per_s {
            Some(rate) => parts.push(format!("{rate:.1} tok/s")),
            None => parts.push("— tok/s".to_string()),
        }
        parts.push(format!("{} tok", self.tokens));
        if self.history_tokens > 0 {
            parts.push(format!("hist {} tok", self.history_tokens));
        }
        if self.steps > 0 {
            parts.push(format!("steps {} {:.0}ms", self.steps, self.avg_step_ms));
        }
        if self.uptime_s > 0.0 {
            parts.push(format!("up {:.0}s", self.uptime_s));
        }
        if !self.online {
            parts.push("метрики: offline".to_string());
        }
        let marker = if self.generating { "▸ " } else { "○ " };
        format!("{marker}{}", parts.join(" · "))
    }
}

pub struct Frame<'a> {
    pub rows: usize,
    pub cols: usize,
    pub lines: &'a [Line],
    pub scroll: Scroll,
    pub input: &'a str,
    pub status: &'a Status,
}

fn push_row(out: &mut String, text: &str, cols: usize, style: &str) {
    out.push_str("\x1b[2K");
    out.push_str(style);
    out.push_str(&truncate(text, cols));
    if !style.is_empty() {
        out.push_str("\x1b[0m");
    }
}

fn tail(text: &str, limit: usize) -> String {
    let count = char_count(text);
    if count <= limit {
        return text.to_string();
    }
    text.chars().skip(count - limit).collect()
}

pub fn render(frame: &Frame) -> String {
    let cols = frame.cols.max(20);
    let rows = frame.rows.max(3);
    let area = rows - 3;
    let lines = frame.lines;
    let view = frame.scroll.view(lines.len(), area);
    let mut out = String::with_capacity((cols + 16) * rows);
    out.push_str("\x1b[H");

    let mut status = frame.status.clone();
    status.scroll = view.offset;
    let status_text = status.line();
    let padded = format!("{:<width$}", truncate(&status_text, cols), width = cols);
    push_row(&mut out, &padded, cols, "\x1b[7m");
    out.push_str("\r\n");

    for line in &lines[view.start..view.end] {
        push_row(&mut out, &line.text, cols, line.kind.style());
        out.push_str("\r\n");
    }
    for _ in (view.end - view.start)..area {
        push_row(&mut out, "", cols, "");
        out.push_str("\r\n");
    }

    let prefix = "you> ";
    let visible = tail(frame.input, cols.saturating_sub(prefix.len()));
    let mut input_line = String::from(prefix);
    input_line.push_str(&visible);
    push_row(&mut out, &input_line, cols, "");
    out.push_str("\r\n");
    push_row(&mut out, HINT, cols, "\x1b[2m");

    let col = prefix.len() + char_count(&visible) + 1;
    out.push_str(&format!("\x1b[{};{col}H\x1b[?25h", rows - 1));
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Enter,
    Backspace,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Interrupt,
    Eof,
    Ignore,
}

#[derive(Debug, Clone, Default)]
pub struct KeyParser {
    pending: Vec<u8>,
}

fn utf8_len(byte: u8) -> usize {
    if byte < 0x80 {
        1
    } else if byte >> 5 == 0b110 {
        2
    } else if byte >> 4 == 0b1110 {
        3
    } else if byte >> 3 == 0b11110 {
        4
    } else {
        0
    }
}

fn escape_key(sequence: &[u8]) -> Key {
    if sequence.len() < 3 {
        return Key::Ignore;
    }
    let final_byte = sequence[sequence.len() - 1];
    let params = &sequence[2..sequence.len() - 1];
    match final_byte {
        b'A' => Key::Up,
        b'B' => Key::Down,
        b'H' => Key::Home,
        b'F' => Key::End,
        b'~' => match params {
            b"1" | b"7" => Key::Home,
            b"4" | b"8" => Key::End,
            b"5" => Key::PageUp,
            b"6" => Key::PageDown,
            _ => Key::Ignore,
        },
        _ => Key::Ignore,
    }
}

impl KeyParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<Key> {
        self.pending.extend_from_slice(bytes);
        let mut keys = Vec::new();
        while let Some(&first) = self.pending.first() {
            match first {
                0x1b => {
                    if self.pending.len() < 2 {
                        break;
                    }
                    if matches!(self.pending[1], b'[' | b'O') {
                        let end = self
                            .pending
                            .iter()
                            .enumerate()
                            .skip(2)
                            .find(|(_, byte)| (0x40..=0x7e).contains(*byte))
                            .map(|(index, _)| index);
                        let Some(end) = end else {
                            if self.pending.len() > 8 {
                                self.pending.clear();
                            }
                            break;
                        };
                        let sequence: Vec<u8> = self.pending.drain(..=end).collect();
                        keys.push(escape_key(&sequence));
                    } else {
                        self.pending.drain(..1);
                        keys.push(Key::Ignore);
                    }
                }
                0x03 => {
                    self.pending.drain(..1);
                    keys.push(Key::Interrupt);
                }
                0x04 => {
                    self.pending.drain(..1);
                    keys.push(Key::Eof);
                }
                b'\r' | b'\n' => {
                    self.pending.drain(..1);
                    keys.push(Key::Enter);
                }
                0x7f | 0x08 => {
                    self.pending.drain(..1);
                    keys.push(Key::Backspace);
                }
                byte if byte < 0x20 => {
                    self.pending.drain(..1);
                    keys.push(Key::Ignore);
                }
                byte if byte < 0x80 => {
                    self.pending.drain(..1);
                    keys.push(Key::Char(byte as char));
                }
                byte => {
                    let len = utf8_len(byte);
                    if len == 0 {
                        self.pending.drain(..1);
                        keys.push(Key::Ignore);
                        continue;
                    }
                    if self.pending.len() < len {
                        break;
                    }
                    let candidate: Vec<u8> = self.pending[..len].to_vec();
                    match std::str::from_utf8(&candidate) {
                        Ok(text) => {
                            self.pending.drain(..len);
                            if let Some(ch) = text.chars().next() {
                                keys.push(if ch.is_control() {
                                    Key::Ignore
                                } else {
                                    Key::Char(ch)
                                });
                            }
                        }
                        Err(_) => {
                            self.pending.drain(..1);
                            keys.push(Key::Ignore);
                        }
                    }
                }
            }
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_hard_at_width() {
        assert_eq!(wrap_text("abcdef", 3), vec!["abc", "def"]);
        assert_eq!(wrap_text("ab", 8), vec!["ab"]);
        assert_eq!(wrap_text("a\nb", 8), vec!["a", "b"]);
        assert_eq!(wrap_text("", 8), vec![""]);
        assert_eq!(wrap_text("a\n\nb", 8), vec!["a", "", "b"]);
    }

    #[test]
    fn truncates_by_chars() {
        assert_eq!(truncate("привет", 3), "при");
        assert_eq!(truncate("ok", 5), "ok");
        assert_eq!(char_count("привет"), 6);
    }

    #[test]
    fn wraps_multibyte_text() {
        let wrapped = wrap_text("привет", 3);
        assert_eq!(wrapped, vec!["при", "вет"]);
    }

    fn assistant(content: &str) -> Message {
        Message::new(Role::Assistant, content)
    }

    #[test]
    fn transcript_renders_prefixes_and_indent() {
        let messages = vec![
            Message::new(Role::System, "rules"),
            Message::new(Role::User, "hello"),
            assistant("abcdefgh"),
        ];
        let lines = transcript_lines(&messages, &[], 10);
        assert_eq!(lines[0].kind, LineKind::System);
        assert_eq!(lines[0].text, "[sys] rule");
        assert_eq!(lines[1].text, "      s");
        assert_eq!(lines[2].kind, LineKind::Blank);
        assert_eq!(lines[3].kind, LineKind::User);
        assert_eq!(lines[3].text, "you> hello");
        let ai_index = lines
            .iter()
            .position(|line| line.kind == LineKind::Assistant)
            .expect("assistant");
        assert_eq!(lines[ai_index].text, "ai> abcdef");
        assert_eq!(lines[ai_index + 1].text, "    gh");
    }

    #[test]
    fn transcript_renders_thinking_and_tools() {
        let mut message = assistant("answer");
        message.thinking = "думаю".to_string();
        message.tools.push(ToolCall {
            name: "lookup".to_string(),
            arguments: "{\"q\":1}".to_string(),
        });
        let lines = transcript_lines(&[message], &[], 80);
        let kinds: Vec<LineKind> = lines.iter().map(|line| line.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LineKind::Thinking,
                LineKind::Assistant,
                LineKind::Tool,
                LineKind::Blank
            ]
        );
        assert_eq!(lines[0].text, "[think] думаю");
        assert_eq!(lines[2].text, "[tool] lookup({\"q\":1})");
    }

    #[test]
    fn transcript_renders_notices_after_messages() {
        let lines = transcript_lines(
            &[Message::new(Role::User, "hi")],
            &[Notice::error("ошибка")],
            80,
        );
        assert_eq!(lines.last().expect("notice").kind, LineKind::Notice);
        assert_eq!(lines.last().expect("notice").text, "[!] ошибка");
    }

    #[test]
    fn transcript_renders_info_notices() {
        let lines = transcript_lines(&[], &[Notice::info("готово")], 80);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].kind, LineKind::Info);
        assert_eq!(lines[0].text, "· готово");
    }

    #[test]
    fn transcript_appends_stream_pieces() {
        let mut transcript = Transcript::default();
        transcript.push_user("hi");
        transcript.begin_assistant();
        transcript.append_thinking("hm");
        transcript.append_content("hel");
        transcript.append_content("lo");
        transcript.append_tool(0, Some("f"), Some("{"));
        transcript.append_tool(0, None, Some("}"));
        let message = transcript.messages().last().expect("assistant");
        assert_eq!(message.content, "hello");
        assert_eq!(message.thinking, "hm");
        assert_eq!(message.tools[0].name, "f");
        assert_eq!(message.tools[0].arguments, "{}");
        transcript.append_tool(2, Some("g"), None);
        assert_eq!(
            transcript.messages().last().expect("assistant").tools.len(),
            3
        );
    }

    #[test]
    fn request_messages_skips_empty_assistant_and_empty_system() {
        let mut transcript = Transcript::default();
        transcript.push_system("");
        transcript.push_user("hi");
        transcript.begin_assistant();
        assert_eq!(transcript.request_messages().len(), 1);
        transcript.append_content("hey");
        transcript.push_system("rules");
        let request = transcript.request_messages();
        assert_eq!(request.len(), 3);
        assert_eq!(request[2].role, Role::System);
    }

    #[test]
    fn drops_empty_assistant() {
        let mut transcript = Transcript::default();
        transcript.push_user("hi");
        transcript.begin_assistant();
        assert!(transcript.drop_empty_assistant());
        assert_eq!(transcript.messages().len(), 1);
        transcript.begin_assistant();
        transcript.append_content("x");
        assert!(!transcript.drop_empty_assistant());
    }

    fn offset(scroll: &Scroll, total: usize, area: usize) -> usize {
        scroll.view(total, area).offset
    }

    #[test]
    fn scroll_window_at_bottom_and_top() {
        let mut scroll = Scroll::default();
        let view = scroll.view(100, 10);
        assert_eq!(view.start, 90);
        assert_eq!(view.end, 100);
        assert_eq!(view.offset, 0);

        scroll.up(10, 100, 10);
        let view = scroll.view(100, 10);
        assert_eq!(view.start, 80);
        assert_eq!(view.end, 90);
        assert_eq!(view.offset, 10);

        scroll.top(100, 10);
        let view = scroll.view(100, 10);
        assert_eq!(view.start, 0);
        assert_eq!(view.end, 10);
        assert_eq!(view.offset, 90);

        scroll.down(30);
        assert_eq!(offset(&scroll, 100, 10), 60);
        scroll.reset_to_bottom();
        assert_eq!(offset(&scroll, 100, 10), 0);
    }

    #[test]
    fn scroll_clamps_when_content_shrinks() {
        let mut scroll = Scroll::default();
        scroll.up(50, 20, 10);
        assert_eq!(offset(&scroll, 20, 10), 10);
        let view = scroll.view(5, 10);
        assert_eq!(view.start, 0);
        assert_eq!(view.end, 5);
        assert_eq!(view.offset, 0);
    }

    #[test]
    fn scroll_page_up_and_home() {
        let mut scroll = Scroll::default();
        scroll.up(8, 40, 8);
        scroll.up(8, 40, 8);
        assert_eq!(offset(&scroll, 40, 8), 16);
        scroll.top(40, 8);
        assert_eq!(offset(&scroll, 40, 8), 32);
        scroll.up(8, 40, 8);
        assert_eq!(offset(&scroll, 40, 8), 32);
        scroll.down(8);
        assert_eq!(offset(&scroll, 40, 8), 24);
    }

    #[test]
    fn status_line_contains_required_fields() {
        let status = Status {
            model: "lfm3".to_string(),
            ctx: 4096,
            active: 1,
            waiting: 2,
            kv_used: 3,
            kv_free: 50,
            cache_hits: 7,
            ttft_ms: Some(320.0),
            tok_per_s: Some(12.5),
            tokens: 64,
            generating: true,
            online: true,
            ..Status::default()
        };
        let line = status.line();
        assert!(line.contains("lfm3 ctx 4096"));
        assert!(line.contains("active 1/2"));
        assert!(line.contains("kv 3/50"));
        assert!(line.contains("cache 7"));
        assert!(line.contains("ttft 320ms"));
        assert!(line.contains("12.5 tok/s"));
        assert!(line.contains("64 tok"));
        assert!(line.starts_with("▸ "));
    }

    #[test]
    fn status_line_shows_history_tokens() {
        let status = Status {
            history_tokens: 42,
            ..Status::default()
        };
        assert!(status.line().contains("hist 42 tok"));
    }

    #[test]
    fn status_marks_offline_and_scroll_counter() {
        let status = Status {
            scroll: 12,
            ..Status::default()
        };
        let line = status.line();
        assert!(line.contains("метрики: offline"));
        assert!(line.contains("↑ 12"));
        assert!(line.starts_with("○ "));
    }

    #[test]
    fn render_has_one_row_per_terminal_line() {
        let messages = vec![Message::new(Role::User, "hi")];
        let status = Status::default();
        let lines = transcript_lines(&messages, &[], 40);
        let frame = Frame {
            rows: 10,
            cols: 40,
            lines: &lines,
            scroll: Scroll::default(),
            input: "abc",
            status: &status,
        };
        let out = render(&frame);
        assert_eq!(out.matches("\x1b[2K").count(), 10);
        assert!(out.contains("you> abc"));
        assert!(out.contains("Enter — отправить"));
        assert!(out.starts_with("\x1b[H"));
        assert!(out.contains("\x1b[9;9H"));
    }

    #[test]
    fn render_scrolls_to_top_with_home() {
        let messages: Vec<Message> = (0..40)
            .map(|i| Message::new(Role::User, format!("line {i}")))
            .collect();
        let status = Status::default();
        let lines = transcript_lines(&messages, &[], 80);
        let mut scroll = Scroll::default();
        scroll.top(lines.len(), 6);
        let frame = Frame {
            rows: 9,
            cols: 80,
            lines: &lines,
            scroll,
            input: "",
            status: &status,
        };
        let out = render(&frame);
        assert!(out.contains("you> line 0"));
        assert!(!out.contains("line 39"));
        assert!(out.contains("↑ "));
    }

    #[test]
    fn input_history_skips_consecutive_duplicates() {
        let mut history = InputHistory::default();
        history.push("first");
        history.push("first");
        history.push("  second  ");
        history.push("second");
        assert_eq!(history.up(), Some("second".to_string()));
        assert_eq!(history.up(), Some("first".to_string()));
        assert_eq!(history.up(), None);
        assert_eq!(history.down(), Some("second".to_string()));
        assert_eq!(history.down(), Some(String::new()));
        assert_eq!(history.down(), None);
    }

    #[test]
    fn input_history_ignores_empty_lines() {
        let mut history = InputHistory::default();
        history.push("   ");
        history.push("");
        assert_eq!(history.up(), None);
    }

    #[test]
    fn input_history_resets_cursor_on_submit() {
        let mut history = InputHistory::default();
        history.push("a");
        history.push("b");
        assert_eq!(history.up(), Some("b".to_string()));
        history.reset_cursor();
        assert_eq!(history.up(), Some("b".to_string()));
    }

    #[test]
    fn key_parser_reads_plain_utf8() {
        let mut parser = KeyParser::new();
        let mut keys = parser.push("ab".as_bytes());
        keys.extend(parser.push("при".as_bytes()));
        assert_eq!(
            keys,
            vec![
                Key::Char('a'),
                Key::Char('b'),
                Key::Char('п'),
                Key::Char('р'),
                Key::Char('и'),
            ]
        );
    }

    #[test]
    fn key_parser_reads_split_utf8() {
        let mut parser = KeyParser::new();
        let bytes = "ж".as_bytes();
        assert!(parser.push(&bytes[..1]).is_empty());
        assert_eq!(parser.push(&bytes[1..]), vec![Key::Char('ж')]);
    }

    #[test]
    fn key_parser_reads_control_keys() {
        let mut parser = KeyParser::new();
        assert_eq!(parser.push(b"\r"), vec![Key::Enter]);
        assert_eq!(parser.push(b"\x7f"), vec![Key::Backspace]);
        assert_eq!(parser.push(b"\x08"), vec![Key::Backspace]);
        assert_eq!(parser.push(b"\x03"), vec![Key::Interrupt]);
        assert_eq!(parser.push(b"\x04"), vec![Key::Eof]);
        assert_eq!(parser.push(b"\t"), vec![Key::Ignore]);
    }

    #[test]
    fn key_parser_reads_escape_sequences() {
        let mut parser = KeyParser::new();
        assert_eq!(parser.push(b"\x1b[A"), vec![Key::Up]);
        assert_eq!(parser.push(b"\x1b[B"), vec![Key::Down]);
        assert_eq!(parser.push(b"\x1b[5~"), vec![Key::PageUp]);
        assert_eq!(parser.push(b"\x1b[6~"), vec![Key::PageDown]);
        assert_eq!(parser.push(b"\x1b[H"), vec![Key::Home]);
        assert_eq!(parser.push(b"\x1b[F"), vec![Key::End]);
        assert_eq!(parser.push(b"\x1bOH"), vec![Key::Home]);
        assert_eq!(parser.push(b"\x1b[4~"), vec![Key::End]);
        assert_eq!(parser.push(b"\x1b[1~"), vec![Key::Home]);
        assert_eq!(parser.push(b"\x1b[Z"), vec![Key::Ignore]);
    }

    #[test]
    fn key_parser_waits_for_split_escape_sequence() {
        let mut parser = KeyParser::new();
        assert!(parser.push(b"\x1b").is_empty());
        assert!(parser.push(b"[").is_empty());
        assert_eq!(parser.push(b"5"), Vec::<Key>::new());
        assert_eq!(parser.push(b"~"), vec![Key::PageUp]);
    }

    #[test]
    fn key_parser_recovers_from_unknown_escape() {
        let mut parser = KeyParser::new();
        let keys = parser.push(b"\x1bx");
        assert_eq!(keys, vec![Key::Ignore, Key::Char('x')]);
    }

    #[test]
    fn transcript_roundtrip_json() {
        let mut transcript = Transcript::default();
        transcript.push_system("rules");
        transcript.push_user("hi");
        transcript.begin_assistant();
        transcript.append_thinking("hm");
        transcript.append_content("hello");
        transcript.append_tool(0, Some("f"), Some("{}"));
        let text = serde_json::to_string(transcript.messages()).expect("encode");
        let decoded: Vec<Message> = serde_json::from_str(&text).expect("decode");
        assert_eq!(decoded, transcript.messages());
    }

    #[test]
    fn save_and_load_transcript_roundtrip() {
        let dir = std::env::temp_dir().join(format!("lfm-tui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("history.json");
        let mut transcript = Transcript::default();
        transcript.push_user("hi");
        transcript.begin_assistant();
        transcript.append_content("hello");
        save_transcript(&path, transcript.messages()).expect("save");
        let loaded = load_transcript(&path).expect("load");
        assert_eq!(loaded, transcript.messages());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_broken_json_and_skips_empty_assistant() {
        let dir = std::env::temp_dir().join(format!("lfm-tui-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("broken.json");
        std::fs::write(&path, "{oops").expect("write");
        assert!(load_transcript(&path).is_err());
        let path = dir.join("empty.json");
        std::fs::write(
            &path,
            r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"  "}]"#,
        )
        .expect("write");
        let loaded = load_transcript(&path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, Role::User);
        std::fs::remove_dir_all(&dir).ok();
    }
}
