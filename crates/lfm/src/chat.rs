use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail};
use lfm_app::{ChatSession, SessionEvent};
use lfm_domain::{Message, SamplingParams, ToolCall, ToolDef};

use crate::config::Config;
use crate::generate::{Stats, generate_stream};
use crate::runtime::Runtime;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    Thinking(String),
    Content(String),
    ToolCall(ToolCall),
    Token(u32),
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub stats: Stats,
}

pub struct Chat {
    pub runtime: Runtime,
    pub history: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub system: String,
    pub thinking: bool,
    pub max_tokens: usize,
}

pub fn load_tools(path: &Path) -> Result<Vec<ToolDef>> {
    #[derive(serde::Deserialize)]
    struct Wrapper {
        tools: Vec<ToolDef>,
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    if let Ok(list) = serde_json::from_str::<Vec<ToolDef>>(&text) {
        return Ok(list);
    }
    let wrapper: Wrapper = serde_json::from_str(&text)?;
    Ok(wrapper.tools)
}

impl Chat {
    pub fn open(cfg: &Config) -> Result<Self> {
        let runtime = Runtime::open(cfg)?;
        let tools = match &cfg.chat.tools {
            Some(path) => load_tools(path)?,
            None => Vec::new(),
        };
        Ok(Self {
            runtime,
            history: vec![Message::system(&cfg.chat.system)],
            tools,
            system: cfg.chat.system.clone(),
            thinking: cfg.chat.thinking,
            max_tokens: cfg.chat.max_tokens,
        })
    }

    pub fn set_tools(&mut self, tools: Vec<ToolDef>) {
        self.tools = tools;
    }

    pub fn set_system(&mut self, system: impl Into<String>) {
        self.system = system.into();
        self.reset();
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.history.push(Message::system(&self.system));
    }

    pub fn send(&mut self, user: &str, params: &SamplingParams) -> Result<Reply> {
        self.send_stream(user, params, &AtomicBool::new(false), |_| {})
    }

    pub fn send_stream(
        &mut self,
        user: &str,
        params: &SamplingParams,
        cancel: &AtomicBool,
        mut on_event: impl FnMut(ChatEvent),
    ) -> Result<Reply> {
        let mut messages = self.history.clone();
        messages.push(Message::user(user));
        let prompt = self.runtime.template.render(
            &messages,
            &self.tools,
            true,
            self.thinking,
            false,
        )?;
        let ids = self.runtime.tokenizer.encode(&prompt, false)?;
        if ids.is_empty() {
            bail!("промпт токенизировался в пустоту");
        }
        let avail = self.runtime.max_ctx;
        if ids.len() + 1 >= avail {
            bail!(
                "история не влезает в контекст: {} токенов при max_ctx {}",
                ids.len(),
                avail
            );
        }
        let room = avail - ids.len() - 1;
        let max_new = if params.max_tokens == 0 {
            room
        } else {
            params.max_tokens.min(room)
        };
        if max_new == 0 {
            bail!("в контексте не осталось места для ответа");
        }
        let mut session = ChatSession::new();
        if self.thinking {
            session.begin_thinking();
        }
        let mut tokens: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut thinking = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let stats = {
            let Runtime {
                backend,
                tokenizer,
                ..
            } = &mut self.runtime;
            generate_stream(
                &mut **backend,
                tokenizer,
                &ids,
                params,
                max_new,
                cancel,
                |next| {
                    on_event(ChatEvent::Token(next));
                    tokens.push(next);
                    let Ok(text) = tokenizer.decode(&tokens, true) else {
                        return;
                    };
                    if text.len() <= printed.len() || !text.starts_with(&printed) {
                        return;
                    }
                    let delta = text[printed.len()..].to_string();
                    printed = text;
                    for event in session.feed(&delta) {
                        match event {
                            SessionEvent::Thinking(t) => {
                                thinking.push_str(&t);
                                on_event(ChatEvent::Thinking(t));
                            }
                            SessionEvent::Content(t) => on_event(ChatEvent::Content(t)),
                            SessionEvent::ToolCall(tc) => {
                                tool_calls.push(tc.clone());
                                on_event(ChatEvent::ToolCall(tc));
                            }
                            SessionEvent::Finished => {}
                        }
                    }
                },
            )?
        };
        for event in session.finish() {
            if let SessionEvent::ToolCall(tc) = event {
                tool_calls.push(tc.clone());
                on_event(ChatEvent::ToolCall(tc));
            }
        }
        let assistant = session
            .take_assistant()
            .unwrap_or_else(|| Message::assistant(printed.clone()));
        let content = assistant.content.clone().unwrap_or(printed);
        self.history.push(Message::user(user));
        self.history.push(assistant);
        Ok(Reply {
            content,
            thinking,
            tool_calls,
            stats,
        })
    }
}
