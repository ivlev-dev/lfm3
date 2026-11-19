use crate::http::Metrics;
use crate::ui::{
    self, Frame, InputHistory, Key, Message, Notice, Role, Scroll, Status, Transcript,
};
use serde_json::{Map, Value, json};
use std::path::Path;

pub const MODEL: &str = "lfm3";
const MAX_NOTICES: usize = 20;

pub const HELP: &[&str] = &[
    "/help — справка по командам",
    "/exit, /quit — выйти из клиента",
    "/reset — очистить историю, системный промпт сохраняется",
    "/save <file> — сохранить историю в JSON-файл",
    "/load <file> — загрузить историю из JSON-файла",
    "/metrics — показать метрики сервера",
    "/model — показать текущую модель и список доступных",
    "/model <name> — переключить модель запросов",
    "/system <текст> — задать системный промпт на лету",
    "/system — сбросить системный промпт",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    None,
    Send(String),
    Exit,
    Reset,
    Save(Option<String>),
    Load(Option<String>),
    Help,
    Metrics,
    Model(Option<String>),
    System(Option<String>),
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
        "/help" => Command::Help,
        "/metrics" => Command::Metrics,
        "/model" => Command::Model(rest.map(str::to_string)),
        "/system" => Command::System(rest.map(str::to_string)),
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

pub fn request_body(model: &str, messages: &[Message], sampling: &Sampling) -> String {
    let messages: Vec<Value> = messages
        .iter()
        .map(|message| json!({ "role": role_name(message.role), "content": message.content }))
        .collect();
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model));
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
    pub notices: Vec<Notice>,
    pub input: String,
    pub history: InputHistory,
    pub scroll: Scroll,
    pub status: Status,
    pub area: usize,
    pub total: usize,
    model: String,
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
            model: MODEL.to_string(),
            system,
        };
        app.status.model = app.model.clone();
        if !app.system.trim().is_empty() {
            app.transcript.push_system(app.system.clone());
        }
        app
    }

    pub fn model(&self) -> &str {
        &self.model
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
        self.push_notice(Notice::info(text));
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.push_notice(Notice::error(text));
    }

    fn push_notice(&mut self, notice: Notice) {
        self.notices.push(notice);
        if self.notices.len() > MAX_NOTICES {
            let drop = self.notices.len() - MAX_NOTICES;
            self.notices.drain(..drop);
        }
        self.scroll.reset_to_bottom();
    }

    pub fn apply_metrics(&mut self, metrics: &Metrics) {
        self.status.online = true;
        self.status.model = self.model.clone();
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

    pub fn show_metrics(&mut self, metrics: &Metrics) {
        self.apply_metrics(metrics);
        let model = if metrics.model.is_empty() {
            self.model.clone()
        } else {
            metrics.model.clone()
        };
        let lines = [
            format!(
                "метрики: модель {model} · ctx {} · uptime {:.1}s",
                metrics.ctx, metrics.uptime_s
            ),
            format!(
                "метрики: активные {} · очередь {}",
                metrics.active, metrics.waiting
            ),
            format!(
                "метрики: переиспользовано токенов {} · попаданий кэша {}",
                metrics.reused_tokens, metrics.cache_hits
            ),
            format!(
                "метрики: kv {}/{} · кэш-страниц {} · шаги {} · {:.1}ms/шаг",
                metrics.kv_used_pages,
                metrics.kv_free_pages,
                metrics.kv_cached_pages,
                metrics.steps,
                metrics.avg_step_ms
            ),
        ];
        for line in lines {
            self.notice(line);
        }
    }

    pub fn help(&mut self) {
        self.notice("команды:");
        for line in HELP {
            self.notice(*line);
        }
    }

    pub fn set_model(&mut self, model: &str) {
        let name = model.trim();
        if name.is_empty() {
            self.notice("использование: /model <name>");
            return;
        }
        self.model = name.to_string();
        self.status.model = name.to_string();
        self.notice(format!("модель переключена на {name}"));
    }

    pub fn show_models(&mut self, models: &[String]) {
        self.notice(format!("текущая модель: {}", self.model()));
        if models.is_empty() {
            self.notice("сервер не вернул ни одной модели");
            return;
        }
        self.notice(format!("доступные модели: {}", models.join(", ")));
        if !models.iter().any(|name| name == self.model()) {
            self.notice(format!("модели {} нет в списке сервера", self.model()));
        }
    }

    pub fn set_system(&mut self, text: Option<&str>) {
        match text.map(str::trim).filter(|value| !value.is_empty()) {
            Some(text) => {
                self.system = text.to_string();
                self.transcript.set_system(text);
                self.notice("системный промпт обновлён");
            }
            None => {
                self.system.clear();
                self.transcript.set_system("");
                self.notice("системный промпт сброшен");
            }
        }
    }

    pub fn history_tokens(&self) -> usize {
        self.transcript
            .request_messages()
            .iter()
            .map(|message| message.content.chars().count().div_ceil(4))
            .sum()
    }

    pub fn request_body(&self, sampling: &Sampling) -> String {
        request_body(self.model(), &self.transcript.request_messages(), sampling)
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
                Err(e) => self.error(format!("ошибка сохранения: {e}")),
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
            Err(e) => self.error(format!("ошибка загрузки: {e}")),
        }
    }

    pub fn render(&mut self, rows: usize, cols: usize) -> String {
        let lines = ui::transcript_lines(self.transcript.messages(), &self.notices, cols);
        self.total = lines.len();
        self.area = rows.saturating_sub(3).max(1);
        self.status.history_tokens = self.history_tokens();
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
        assert_eq!(parse_command("/help"), Command::Help);
        assert_eq!(parse_command("/metrics"), Command::Metrics);
        assert_eq!(parse_command("/model"), Command::Model(None));
        assert_eq!(
            parse_command("/model lfm3-small"),
            Command::Model(Some("lfm3-small".to_string()))
        );
        assert_eq!(parse_command("/system"), Command::System(None));
        assert_eq!(
            parse_command("/system  будь краток "),
            Command::System(Some("будь краток".to_string()))
        );
        assert_eq!(
            parse_command("/unknown"),
            Command::Unknown("/unknown".to_string())
        );
    }

    #[test]
    fn unknown_commands_keep_only_the_name() {
        assert_eq!(
            parse_command("/nope arg"),
            Command::Unknown("/nope".to_string())
        );
        assert_eq!(parse_command("/"), Command::Unknown("/".to_string()));
        assert_eq!(parse_command(" / "), Command::Unknown("/".to_string()));
    }

    #[test]
    fn command_names_ignore_extra_spaces() {
        assert_eq!(parse_command("  /exit  "), Command::Exit);
        assert_eq!(parse_command("/help now"), Command::Help);
        assert_eq!(parse_command("/metrics extra"), Command::Metrics);
        assert_eq!(
            parse_command("/model   lfm3   "),
            Command::Model(Some("lfm3".to_string()))
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
        let body = request_body(MODEL, &messages, &sampling);
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
    fn request_body_uses_passed_model() {
        let sampling = Sampling {
            max_tokens: 8,
            temperature: None,
            top_p: None,
            seed: None,
        };
        let value: Value =
            serde_json::from_str(&request_body("other", &[], &sampling)).expect("json");
        assert_eq!(value["model"], "other");
        assert_eq!(value["messages"], Value::Array(Vec::new()));
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
        let value: Value =
            serde_json::from_str(&request_body(MODEL, &messages, &sampling)).expect("json");
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
        assert_eq!(
            app.notices.last().map(|notice| notice.text.as_str()),
            Some("notice 24")
        );
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
                .any(|notice| notice.text.starts_with("сохранено"))
        );

        let mut restored = App::new(String::new());
        restored.load(Some(path.to_str().expect("path")));
        assert_eq!(restored.transcript.messages(), app.transcript.messages());
        restored.load(None);
        assert!(
            restored
                .notices
                .iter()
                .any(|notice| notice.text.contains("/load"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn help_lists_all_commands() {
        let mut app = App::new(String::new());
        app.help();
        let joined: Vec<&str> = app
            .notices
            .iter()
            .map(|notice| notice.text.as_str())
            .collect();
        assert!(joined.iter().any(|line| line.contains("/exit")));
        assert!(joined.iter().any(|line| line.contains("/reset")));
        assert!(joined.iter().any(|line| line.contains("/save")));
        assert!(joined.iter().any(|line| line.contains("/load")));
        assert!(joined.iter().any(|line| line.contains("/metrics")));
        assert!(joined.iter().any(|line| line.contains("/model")));
        assert!(joined.iter().any(|line| line.contains("/system")));
        assert_eq!(joined.len(), HELP.len() + 1);
    }

    #[test]
    fn metrics_panel_lists_core_counters() {
        let mut app = App::new(String::new());
        app.show_metrics(&Metrics {
            model: "lfm3".to_string(),
            ctx: 4096,
            active: 2,
            waiting: 5,
            steps: 10,
            avg_step_ms: 12.5,
            kv_used_pages: 3,
            kv_free_pages: 50,
            kv_cached_pages: 4,
            cache_hits: 7,
            reused_tokens: 128,
            uptime_s: 9.5,
        });
        let text: String = app
            .notices
            .iter()
            .map(|notice| notice.text.as_str())
            .collect::<Vec<&str>>()
            .join("\n");
        assert!(text.contains("активные 2"));
        assert!(text.contains("очередь 5"));
        assert!(text.contains("128"));
        assert!(text.contains("попаданий кэша 7"));
        assert_eq!(app.notices.len(), 4);
        assert!(app.status.online);
    }

    #[test]
    fn model_switch_updates_request_and_status() {
        let mut app = App::new(String::new());
        assert_eq!(app.model(), MODEL);
        app.set_model("  lfm3-mini ");
        assert_eq!(app.model(), "lfm3-mini");
        assert_eq!(app.status.model, "lfm3-mini");
        let sampling = Sampling {
            max_tokens: 4,
            temperature: None,
            top_p: None,
            seed: None,
        };
        let value: Value = serde_json::from_str(&app.request_body(&sampling)).expect("json");
        assert_eq!(value["model"], "lfm3-mini");
    }

    #[test]
    fn model_without_name_keeps_current_and_warns() {
        let mut app = App::new(String::new());
        app.set_model("   ");
        assert_eq!(app.model(), MODEL);
        assert!(
            app.notices
                .iter()
                .any(|notice| notice.text.contains("/model"))
        );
    }

    #[test]
    fn model_list_shows_current_and_flags_unknown() {
        let mut app = App::new(String::new());
        app.show_models(&["a".to_string(), "b".to_string()]);
        let text: String = app
            .notices
            .iter()
            .map(|notice| notice.text.clone())
            .collect::<Vec<String>>()
            .join("\n");
        assert!(text.contains("текущая модель: lfm3"));
        assert!(text.contains("a, b"));
        assert!(text.contains("нет в списке"));
    }

    #[test]
    fn system_prompt_can_be_set_and_reset() {
        let mut app = App::new("old".to_string());
        app.transcript.push_user("hi");
        app.set_system(Some("new rules"));
        assert_eq!(app.system, "new rules");
        let messages = app.transcript.messages();
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "new rules");
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.role == Role::System)
                .count(),
            1
        );
        let sampling = Sampling {
            max_tokens: 4,
            temperature: None,
            top_p: None,
            seed: None,
        };
        let value: Value = serde_json::from_str(&app.request_body(&sampling)).expect("json");
        assert_eq!(value["messages"][0]["content"], "new rules");

        app.set_system(None);
        assert_eq!(app.system, "");
        assert!(
            app.transcript
                .messages()
                .iter()
                .all(|message| message.role != Role::System)
        );
        app.reset();
        assert!(
            app.transcript
                .messages()
                .iter()
                .all(|message| message.role != Role::System)
        );
    }

    #[test]
    fn history_tokens_track_transcript_size() {
        let mut app = App::new(String::new());
        assert_eq!(app.history_tokens(), 0);
        app.transcript.push_user("12345678");
        assert_eq!(app.history_tokens(), 2);
        app.transcript.begin_assistant();
        app.transcript.append_content("1234");
        assert_eq!(app.history_tokens(), 3);
        let _ = app.render(10, 40);
        assert_eq!(app.status.history_tokens, 3);
    }
}
