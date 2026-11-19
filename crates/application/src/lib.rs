pub mod generate;
pub mod ports;
pub mod sampling;
pub mod scheduler;

pub use generate::{FinishReason, Generate, GenerateOutcome};
pub use ports::{
    BackendLoader, CacheStats, InferenceBackend, SequenceStep, TokenizerLoader, TokenizerPort,
};
pub use sampling::{argmax, Sampler};
pub use scheduler::{
    pages_for, ActiveSeq, AdmitError, AdmitOutcome, Enqueued, JobId, Phase, Scheduler,
    SchedulerConfig, StepPlan, StepRow, StepToken,
};

use lfm_domain::{Message, Role, ToolCall};

#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    Thinking(String),
    Content(String),
    ToolCall(ToolCall),
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    Thinking,
    Tool,
}

pub struct ChatSession {
    messages: Vec<Message>,
    mode: Mode,
    buf: String,
    tool_acc: String,
    content_acc: String,
    thinking_acc: String,
    tool_calls_acc: Vec<ToolCall>,
    assistant: Option<Message>,
    pending_gt: bool,
}

impl ChatSession {
    const THINK_OPEN: &'static str = "<think";
    const THINK_CLOSE: &'static str = "</think";
    const TOOL_OPEN: &'static str = "<tool_call>";
    const TOOL_CLOSE: &'static str = "</tool_call>";

    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            mode: Mode::Normal,
            buf: String::new(),
            tool_acc: String::new(),
            content_acc: String::new(),
            thinking_acc: String::new(),
            tool_calls_acc: Vec::new(),
            assistant: None,
            pending_gt: false,
        }
    }

    pub fn push_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn begin_thinking(&mut self) {
        self.mode = Mode::Thinking;
        self.pending_gt = true;
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<SessionEvent> {
        self.buf.push_str(chunk);
        self.drain()
    }

    pub fn finish(&mut self) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        match self.mode {
            Mode::Normal => {
                if !self.buf.is_empty() {
                    let text = std::mem::take(&mut self.buf);
                    self.push_content(&mut events, text);
                }
            }
            Mode::Thinking => {
                if !self.buf.is_empty() {
                    let text = std::mem::take(&mut self.buf);
                    self.push_thinking(&mut events, text);
                }
            }
            Mode::Tool => {
                let mut raw = String::new();
                raw.push_str(Self::TOOL_OPEN);
                raw.push_str(&self.tool_acc);
                raw.push_str(&self.buf);
                self.push_content(&mut events, raw);
                self.tool_acc.clear();
                self.buf.clear();
            }
        }
        self.mode = Mode::Normal;
        self.pending_gt = false;

        let content = if self.content_acc.is_empty() {
            None
        } else {
            Some(self.content_acc.clone())
        };
        let thinking = if self.thinking_acc.is_empty() {
            None
        } else {
            Some(self.thinking_acc.clone())
        };
        let msg = Message {
            role: Role::Assistant,
            content,
            thinking,
            tool_calls: self.tool_calls_acc.clone(),
            tool_call_id: None,
        };
        self.messages.push(msg.clone());
        self.assistant = Some(msg);
        self.content_acc.clear();
        self.thinking_acc.clear();
        self.tool_calls_acc.clear();

        events.push(SessionEvent::Finished);
        events
    }

    pub fn take_assistant(&mut self) -> Option<Message> {
        self.assistant.take()
    }

    pub fn reset(&mut self) {
        self.mode = Mode::Normal;
        self.pending_gt = false;
        self.buf.clear();
        self.tool_acc.clear();
        self.content_acc.clear();
        self.thinking_acc.clear();
        self.tool_calls_acc.clear();
        self.assistant = None;
    }

    fn drain(&mut self) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        loop {
            if self.pending_gt {
                if self.buf.is_empty() {
                    break;
                }
                if self.buf.starts_with('>') {
                    self.buf.drain(..1);
                }
                self.pending_gt = false;
            }
            match self.mode {
                Mode::Normal => {
                    let think_pos = self.buf.find(Self::THINK_OPEN);
                    let tool_pos = self.buf.find(Self::TOOL_OPEN);
                    match (think_pos, tool_pos) {
                        (Some(a), Some(b)) if a <= b => {
                            if a > 0 {
                                let text = self.buf[..a].to_string();
                                self.push_content(&mut events, text);
                            }
                            self.buf = self.buf[a + Self::THINK_OPEN.len()..].to_string();
                            self.mode = Mode::Thinking;
                            self.pending_gt = true;
                        }
                        (Some(_), Some(b)) => {
                            if b > 0 {
                                let text = self.buf[..b].to_string();
                                self.push_content(&mut events, text);
                            }
                            self.buf = self.buf[b + Self::TOOL_OPEN.len()..].to_string();
                            self.mode = Mode::Tool;
                            self.tool_acc.clear();
                        }
                        (Some(a), None) => {
                            if a > 0 {
                                let text = self.buf[..a].to_string();
                                self.push_content(&mut events, text);
                            }
                            self.buf = self.buf[a + Self::THINK_OPEN.len()..].to_string();
                            self.mode = Mode::Thinking;
                            self.pending_gt = true;
                        }
                        (None, Some(b)) => {
                            if b > 0 {
                                let text = self.buf[..b].to_string();
                                self.push_content(&mut events, text);
                            }
                            self.buf = self.buf[b + Self::TOOL_OPEN.len()..].to_string();
                            self.mode = Mode::Tool;
                            self.tool_acc.clear();
                        }
                        (None, None) => {
                            let hold = longest_suffix_prefix(&self.buf, &[Self::THINK_OPEN, Self::TOOL_OPEN]);
                            let mut split = self.buf.len() - hold;
                            while split > 0 && !self.buf.is_char_boundary(split) {
                                split -= 1;
                            }
                            if split > 0 {
                                let text = self.buf[..split].to_string();
                                self.push_content(&mut events, text);
                            }
                            self.buf = self.buf[split..].to_string();
                            break;
                        }
                    }
                }
                Mode::Thinking => {
                    if let Some(p) = self.buf.find(Self::THINK_CLOSE) {
                        if p > 0 {
                            let text = self.buf[..p].to_string();
                            self.push_thinking(&mut events, text);
                        }
                        self.buf = self.buf[p + Self::THINK_CLOSE.len()..].to_string();
                        self.mode = Mode::Normal;
                        self.pending_gt = true;
                    } else {
                        let hold = longest_suffix_prefix(&self.buf, &[Self::THINK_CLOSE]);
                        let mut split = self.buf.len() - hold;
                        while split > 0 && !self.buf.is_char_boundary(split) {
                            split -= 1;
                        }
                        if split > 0 {
                            let text = self.buf[..split].to_string();
                            self.push_thinking(&mut events, text);
                        }
                        self.buf = self.buf[split..].to_string();
                        break;
                    }
                }
                Mode::Tool => {
                    if let Some(p) = self.buf.find(Self::TOOL_CLOSE) {
                        let before = self.buf[..p].to_string();
                        self.tool_acc.push_str(&before);
                        let raw = std::mem::take(&mut self.tool_acc);
                        if let Some(tc) = parse_tool_call(&raw) {
                            self.push_toolcall(&mut events, tc);
                        } else {
                            let reconstructed = format!("{}{}{}", Self::TOOL_OPEN, raw, Self::TOOL_CLOSE);
                            self.push_content(&mut events, reconstructed);
                        }
                        self.buf = self.buf[p + Self::TOOL_CLOSE.len()..].to_string();
                        self.mode = Mode::Normal;
                    } else {
                        let hold = longest_suffix_prefix(&self.buf, &[Self::TOOL_CLOSE]);
                        let mut split = self.buf.len() - hold;
                        while split > 0 && !self.buf.is_char_boundary(split) {
                            split -= 1;
                        }
                        let part = self.buf[..split].to_string();
                        self.tool_acc.push_str(&part);
                        self.buf = self.buf[split..].to_string();
                        break;
                    }
                }
            }
        }
        events
    }

    fn push_content(&mut self, events: &mut Vec<SessionEvent>, text: String) {
        if text.is_empty() {
            return;
        }
        self.content_acc.push_str(&text);
        events.push(SessionEvent::Content(text));
    }

    fn push_thinking(&mut self, events: &mut Vec<SessionEvent>, text: String) {
        if text.is_empty() {
            return;
        }
        self.thinking_acc.push_str(&text);
        events.push(SessionEvent::Thinking(text));
    }

    fn push_toolcall(&mut self, events: &mut Vec<SessionEvent>, tc: ToolCall) {
        self.tool_calls_acc.push(tc.clone());
        events.push(SessionEvent::ToolCall(tc));
    }
}

impl Default for ChatSession {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_tool_call(raw: &str) -> Option<ToolCall> {
    let func_marker = "<function=";
    let func_start = raw.find(func_marker)?;
    let after_func = &raw[func_start + func_marker.len()..];
    let name_end = after_func.find('>')?;
    let name = after_func[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let mut search = raw;
    while let Some(pstart) = search.find("<parameter=") {
        let after_key = &search[pstart + "<parameter=".len()..];
        let key_end = after_key.find('>')?;
        let key = after_key[..key_end].trim().to_string();
        let rest = &after_key[key_end + 1..];
        let close_idx = rest.find("</parameter>")?;
        let value_raw = rest[..close_idx].trim();
        let value = parse_value(value_raw);
        args.insert(key, value);
        search = &rest[close_idx + "</parameter>".len()..];
    }

    Some(ToolCall {
        id: None,
        name,
        arguments: serde_json::Value::Object(args),
    })
}

fn parse_value(s: &str) -> serde_json::Value {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
        return v;
    }
    serde_json::Value::String(s.trim().to_string())
}

fn longest_suffix_prefix(s: &str, markers: &[&str]) -> usize {
    let max_marker = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    let max_possible = max_marker.min(s.len());
    for k in (1..=max_possible).rev() {
        if !s.is_char_boundary(s.len() - k) {
            continue;
        }
        let suffix = &s[s.len() - k..];
        if markers.iter().any(|m| m.starts_with(suffix)) {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_thinking_streams_thinking_until_close() {
        let mut s = ChatSession::new();
        s.begin_thinking();
        let ev = s.feed("Let me think.");
        assert!(matches!(ev.as_slice(), [SessionEvent::Thinking(t)] if t == "Let me think."));
        let ev = s.feed("</think");
        assert!(ev.is_empty());
        let ev = s.feed("Answer");
        assert!(matches!(ev.as_slice(), [SessionEvent::Content(t)] if t == "Answer"));
    }
}
