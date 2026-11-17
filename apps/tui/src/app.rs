use crate::http::Metrics;
use crate::ui::{self, Frame, InputHistory, Key, Message, Role, Scroll, Status, Transcript};
use serde_json::{Map, Value, json};
use std::path::Path;

pub const MODEL: &str = "lfm3";
const MAX_NOTICES: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    None,
    Send(String),
    Exit,
    Reset,
    Save(Option<String>),
    Load(Option<String>),
    Unknown(String),
}

pub fn parse_command(line: &str) -> Command {
    let text = line.trim();
    if text.is_empty() {
        return Command::None;
    }
    if !text.starts_with('/') {
        return Command::Send(text.to_string());
    }
    let mut parts = text.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let rest = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match name {
        "/exit" | "/quit" => Command::Exit,
        "/reset" => Command::Reset,
        "/save" => Command::Save(rest.map(str::to_string)),
        "/load" => Command::Load(rest.map(str::to_string)),
        other => Command::Unknown(other.to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
}

pub fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

pub fn request_body(messages: &[Message], sampling: &Sampling) -> String {
    let messages: Vec<Value> = messages
        .iter()
        .map(|message| json!({ "role": role_name(message.role), "content": message.content }))
        .collect();
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(MODEL));
    payload.insert("messages".to_string(), Value::Array(messages));
    payload.insert("stream".to_string(), json!(true));
    payload.insert("max_tokens".to_string(), json!(sampling.max_tokens.max(1)));
    if let Some(temperature) = sampling.temperature {
        payload.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = sampling.top_p {
        payload.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(seed) = sampling.seed {
        payload.insert("seed".to_string(), json!(seed));
    }
    Value::Object(payload).to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleAction {
    None,
    Submit,
    Exit,
}

pub struct App {
    pub transcript: Transcript,
    pub notices: Vec<String>,
    pub input: String,
    pub history: InputHistory,
    pub scroll: Scroll,
    pub status: Status,
    pub area: usize,
    pub total: usize,
    system: String,
}

impl App {
    pub fn new(system: String) -> Self {
        let mut app = Self {
            transcript: Transcript::default(),
            notices: Vec::new(),
            input: String::new(),
            history: InputHistory::default(),
            scroll: Scroll::default(),
            status: Status::default(),
            area: 1,
            total: 0,
            system,
        };
        if !app.system.trim().is_empty() {
            app.transcript.push_system(app.system.clone());
        }
        app
    }

    pub fn reset(&mut self) {
        self.transcript.clear();
        if !self.system.trim().is_empty() {
            self.transcript.push_system(self.system.clone());
        }
        self.scroll.reset_to_bottom();
        self.notice("история очищена");
    }

    pub fn notice(&mut self, text: impl Into<String>) {
        self.notices.push(text.into());
        if self.notices.len() > MAX_NOTICES {
            let drop = self.notices.len() - MAX_NOTICES;
            self.notices.drain(..drop);
        }
        self.scroll.reset_to_bottom();
    }

    pub fn apply_metrics(&mut self, metrics: &Metrics) {
        self.status.online = true;
        if !metrics.model.is_empty() {
            self.status.model = metrics.model.clone();
        }
        self.status.ctx = metrics.ctx;
        self.status.active = metrics.active;
        self.status.waiting = metrics.waiting;
        self.status.kv_used = metrics.kv_used_pages;
        self.status.kv_free = metrics.kv_free_pages;
        self.status.kv_cached = metrics.kv_cached_pages;
        self.status.cache_hits = metrics.cache_hits;
        self.status.reused_tokens = metrics.reused_tokens;
        self.status.steps = metrics.steps;
        self.status.avg_step_ms = metrics.avg_step_ms;
        self.status.uptime_s = metrics.uptime_s;
    }

    pub fn metrics_down(&mut self) {
        self.status.online = false;
    }

    pub fn handle_key(&mut self, key: Key) -> IdleAction {
        match key {
            Key::Char(ch) => {
                self.input.push(ch);
                self.scroll.reset_to_bottom();
            }
            Key::Backspace => {
                self.input.pop();
            }
            Key::Enter => return IdleAction::Submit,
            Key::Up => {
                if let Some(text) = self.history.up() {
                    self.input = text;
                }
                return IdleAction::None;
            }
            Key::Down => {
                if let Some(text) = self.history.down() {
                    self.input = text;
                }
                return IdleAction::None;
            }
            Key::PageUp => self.scroll.up(self.area.max(1), self.total, self.area),
            Key::PageDown => self.scroll.down(self.area.max(1)),
            Key::Home => self.scroll.top(self.total, self.area),
            Key::End => self.scroll.reset_to_bottom(),
            Key::Interrupt | Key::Eof => return IdleAction::Exit,
            Key::Ignore => {}
        }
        self.history.reset_cursor();
        IdleAction::None
    }

    pub fn handle_stream_key(&mut self, key: Key) {
        match key {
            Key::PageUp => self.scroll.up(self.area.max(1), self.total, self.area),
            Key::PageDown => self.scroll.down(self.area.max(1)),
            Key::Home => self.scroll.top(self.total, self.area),
            Key::End => self.scroll.reset_to_bottom(),
            _ => {}
        }
    }

    pub fn save(&mut self, path: Option<&str>) {
        match path {
            Some(path) => match ui::save_transcript(Path::new(path), self.transcript.messages()) {
                Ok(()) => self.notice(format!("сохранено в {path}")),
                Err(e) => self.notice(format!("ошибка сохранения: {e}")),
            },
            None => self.notice("использование: /save <file>"),
        }
    }

    pub fn load(&mut self, path: Option<&str>) {
        let Some(path) = path else {
            self.notice("использование: /load <file>");
            return;
        };
        match ui::load_transcript(Path::new(path)) {
            Ok(messages) => {
                let count = messages.len();
                self.transcript.replace(messages);
                self.scroll.reset_to_bottom();
                self.notice(format!("загружено сообщений: {count}"));
            }
            Err(e) => self.notice(format!("ошибка загрузки: {e}")),
        }
    }

    pub fn render(&mut self, rows: usize, cols: usize) -> String {
        let lines = ui::transcript_lines(self.transcript.messages(), &self.notices, cols);
        self.total = lines.len();
        self.area = rows.saturating_sub(3).max(1);
        let frame = Frame {
            rows,
            cols,
            lines: &lines,
            scroll: self.scroll,
            input: &self.input,
            status: &self.status,
        };
        ui::render(&frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::Metrics;

    fn send(line: &str) -> String {
        match parse_command(line) {
            Command::Send(text) => text,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_plain_input_as_send() {
        assert_eq!(send("привет"), "привет");
        assert_eq!(send("  hi there  "), "hi there");
        assert_eq!(parse_command("   "), Command::None);
    }

    #[test]
    fn parses_slash_commands() {
        assert_eq!(parse_command("/exit"), Command::Exit);
        assert_eq!(parse_command("/quit"), Command::Exit);
        assert_eq!(parse_command("/reset"), Command::Reset);
        assert_eq!(
            parse_command("/save history.json"),
            Command::Save(Some("history.json".to_string()))
        );
        assert_eq!(parse_command("/save"), Command::Save(None));
        assert_eq!(
            parse_command("/load  a b.json "),
            Command::Load(Some("a b.json".to_string()))
        );
        assert_eq!(
            parse_command("/unknown"),
            Command::Unknown("/unknown".to_string())
        );
    }

    #[test]
    fn request_body_has_stream_and_messages() {
        let messages = vec![
            Message::new(Role::System, "rules"),
            Message::new(Role::User, "hi"),
        ];
        let sampling = Sampling {
            max_tokens: 128,
            temperature: Some(0.5),
            top_p: Some(0.5),
            seed: Some(7),
        };
        let body = request_body(&messages, &sampling);
        let value: Value = serde_json::from_str(&body).expect("json");
        assert_eq!(value["model"], MODEL);
        assert_eq!(value["stream"], true);
        assert_eq!(value["max_tokens"], 128);
        assert_eq!(value["temperature"], 0.5);
        assert_eq!(value["top_p"], 0.5);
        assert_eq!(value["seed"], 7);
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["content"], "hi");
    }

    #[test]
    fn request_body_omits_optional_sampling() {
        let messages = vec![Message::new(Role::User, "hi")];
        let sampling = Sampling {
            max_tokens: 4,
            temperature: None,
            top_p: None,
            seed: None,
        };
        let value: Value = serde_json::from_str(&request_body(&messages, &sampling)).expect("json");
        assert!(value.get("temperature").is_none());
        assert!(value.get("top_p").is_none());
        assert!(value.get("seed").is_none());
        assert_eq!(value["max_tokens"], 4);
    }

    #[test]
    fn app_starts_with_system_message() {
        let app = App::new("rules".to_string());
        assert_eq!(app.transcript.messages().len(), 1);
        assert_eq!(app.transcript.messages()[0].role, Role::System);
        assert_eq!(app.transcript.messages()[0].content, "rules");
    }

    #[test]
    fn reset_keeps_system_message() {
        let mut app = App::new("rules".to_string());
        app.transcript.push_user("hi");
        app.reset();
        assert_eq!(app.transcript.messages().len(), 1);
        assert_eq!(app.transcript.messages()[0].role, Role::System);
    }

    #[test]
    fn typing_and_enter_work() {
        let mut app = App::new(String::new());
        for ch in "hi".chars() {
            app.handle_key(Key::Char(ch));
        }
        app.handle_key(Key::Backspace);
        app.handle_key(Key::Char('y'));
        assert_eq!(app.input, "hy");
        assert_eq!(app.handle_key(Key::Enter), IdleAction::Submit);
        assert_eq!(app.handle_key(Key::Char('x')), IdleAction::None);
    }

    #[test]
    fn ctrl_c_and_ctrl_d_exit_when_idle() {
        let mut app = App::new(String::new());
        assert_eq!(app.handle_key(Key::Interrupt), IdleAction::Exit);
        assert_eq!(app.handle_key(Key::Eof), IdleAction::Exit);
    }

    #[test]
    fn up_down_walks_input_history() {
        let mut app = App::new(String::new());
        app.history.push("первый");
        app.history.push("второй");
        app.handle_key(Key::Up);
        assert_eq!(app.input, "второй");
        app.handle_key(Key::Up);
        assert_eq!(app.input, "первый");
        app.handle_key(Key::Up);
        assert_eq!(app.input, "первый");
        app.handle_key(Key::Down);
        assert_eq!(app.input, "второй");
        app.handle_key(Key::Down);
        assert_eq!(app.input, "");
    }

    #[test]
    fn scroll_keys_use_last_layout() {
        let mut app = App::new(String::new());
        for i in 0..30 {
            app.transcript.push_user(format!("строка {i}"));
        }
        let _ = app.render(10, 40);
        assert_eq!(app.total, 60);
        assert_eq!(app.area, 7);
        app.handle_key(Key::PageUp);
        assert_eq!(app.scroll.view(app.total, app.area).offset, 7);
        app.handle_key(Key::Home);
        assert_eq!(app.scroll.view(app.total, app.area).offset, 53);
        app.handle_key(Key::End);
        assert_eq!(app.scroll.view(app.total, app.area).offset, 0);
    }

    #[test]
    fn metrics_update_status_fields() {
        let mut app = App::new(String::new());
        app.apply_metrics(&Metrics {
            model: "lfm3".to_string(),
            ctx: 4096,
            active: 1,
            waiting: 0,
            steps: 5,
            avg_step_ms: 10.0,
            kv_used_pages: 2,
            kv_free_pages: 30,
            kv_cached_pages: 1,
            cache_hits: 3,
            reused_tokens: 12,
            uptime_s: 4.0,
        });
        assert!(app.status.online);
        assert_eq!(app.status.model, "lfm3");
        assert_eq!(app.status.ctx, 4096);
        assert_eq!(app.status.active, 1);
        assert_eq!(app.status.kv_used, 2);
        assert_eq!(app.status.cache_hits, 3);
        app.metrics_down();
        assert!(!app.status.online);
        assert_eq!(app.status.ctx, 4096);
    }

    #[test]
    fn notices_are_capped() {
        let mut app = App::new(String::new());
        for i in 0..(MAX_NOTICES + 5) {
            app.notice(format!("notice {i}"));
        }
        assert_eq!(app.notices.len(), MAX_NOTICES);
        assert_eq!(app.notices.last().map(String::as_str), Some("notice 24"));
    }

    #[test]
    fn render_uses_scroll_and_input() {
        let mut app = App::new(String::new());
        app.transcript.push_user("hi");
        app.input = "typed".to_string();
        let out = app.render(12, 50);
        assert!(out.contains("you> hi"));
        assert!(out.contains("you> typed"));
        assert_eq!(app.area, 9);
        assert_eq!(app.total, 2);
    }

    #[test]
    fn save_and_load_through_app() {
        let dir = std::env::temp_dir().join(format!("lfm-tui-app-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("history.json");
        let mut app = App::new(String::new());
        app.transcript.push_user("hi");
        app.transcript.begin_assistant();
        app.transcript.append_content("hello");
        app.save(Some(path.to_str().expect("path")));
        assert!(
            app.notices
                .iter()
                .any(|notice| notice.starts_with("сохранено"))
        );

        let mut restored = App::new(String::new());
        restored.load(Some(path.to_str().expect("path")));
        assert_eq!(restored.transcript.messages(), app.transcript.messages());
        restored.load(None);
        assert!(
            restored
                .notices
                .iter()
                .any(|notice| notice.contains("/load"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
