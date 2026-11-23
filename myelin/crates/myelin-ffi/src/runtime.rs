//! Transport-neutral glue shared by the C ABI and UniFFI front ends.
//!
//! This module owns the stub generation flow: validate input, ask the core
//! engine for deterministic tokens, emit events with a small delay.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lfm_chat::ChatTemplate;
use myelin_core::tokenizer::LfmTokenizer;
use myelin_core::{
    Engine as CoreEngine, GenerateConfig, Model, ModelSpec, SamplingParams,
};
use myelin_npu::{MockBackend, NpuBackend};

/// A model loaded from disk plus everything needed to answer chat turns.
struct RealEngine {
    model: Model,
    tokenizer: LfmTokenizer,
    template: ChatTemplate,
    context: usize,
}

/// Number of fake tokens emitted by one stub generation.
pub const STUB_TOKEN_COUNT: u32 = 5;
/// Delay before each emitted token, so streaming is visible to callers.
pub const STUB_TOKEN_DELAY: Duration = Duration::from_millis(10);
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// One chat message crossing the FFI boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Sampling parameters. The stub ignores them except for validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            max_tokens: STUB_TOKEN_COUNT,
            seed: 0,
        }
    }
}

/// Events produced during generation.
///
/// `ToolCall` and `Error` are part of the stable event surface even though the
/// stub engine only emits deltas and the final `Done`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    Delta {
        index: u32,
        token: u32,
        text: String,
    },
    ToolCall {
        payload: String,
    },
    Done {
        total: u32,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidArgument(String),
    InvalidState(String),
    Io(String),
    Backend(String),
    Cancelled,
    Internal(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(reason) => write!(f, "invalid argument: {reason}"),
            Self::InvalidState(reason) => write!(f, "invalid state: {reason}"),
            Self::Io(reason) => write!(f, "I/O error: {reason}"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
            Self::Cancelled => f.write_str("generation cancelled"),
            Self::Internal(reason) => write!(f, "internal error: {reason}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Which transport the caller requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Pick the best compiled-in backend. Mock until FastRPC lands.
    Auto,
    /// Host-side deterministic backend.
    Mock,
    /// Reserved for the real DSP transport.
    Npu,
}

impl BackendKind {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Auto),
            1 => Ok(Self::Mock),
            2 => Ok(Self::Npu),
            other => Err(RuntimeError::InvalidArgument(format!(
                "unknown backend id {other}"
            ))),
        }
    }

    fn instantiate(self) -> Result<Box<dyn NpuBackend>> {
        match self {
            Self::Auto | Self::Mock => Ok(Box::new(MockBackend::default())),
            Self::Npu => Err(RuntimeError::Backend(
                "NPU transport is not implemented yet".to_owned(),
            )),
        }
    }
}

/// Shared engine state. Cheap to clone through `Arc`.
pub struct EngineHandle {
    core: Mutex<CoreEngine>,
    backend: Mutex<Box<dyn NpuBackend>>,
    real: Mutex<Option<RealEngine>>,
}

impl EngineHandle {
    pub fn new(kind: BackendKind) -> Result<Arc<Self>> {
        let spec = ModelSpec::placeholder();
        let mut backend = kind.instantiate()?;
        backend
            .prepare(&spec)
            .map_err(|error| RuntimeError::Backend(error.to_string()))?;
        let core = CoreEngine::load(spec).map_err(internal)?;
        Ok(Arc::new(Self {
            core: Mutex::new(core),
            backend: Mutex::new(backend),
            real: Mutex::new(None),
        }))
    }

    /// Validate that `path` looks like a GGUF file. Metadata parsing is the
    /// next stage; the spec stays the stub placeholder for now.
    pub fn load(&self, path: &str) -> Result<()> {
        if path.trim().is_empty() {
            return Err(RuntimeError::InvalidArgument("empty model path".to_owned()));
        }
        let mut file =
            File::open(path).map_err(|error| RuntimeError::Io(format!("{path}: {error}")))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|_| {
            RuntimeError::InvalidArgument(format!("{path}: shorter than GGUF magic"))
        })?;
        if &magic != GGUF_MAGIC {
            return Err(RuntimeError::InvalidArgument(format!(
                "{path}: bad GGUF magic"
            )));
        }

        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let tokenizer_path = dir.join("tokenizer.json");
        let template_path = dir.join("chat_template.jinja");
        let model = Model::load_gguf(path).map_err(runtime_err)?;
        let tokenizer = LfmTokenizer::from_file(&tokenizer_path).map_err(|error| {
            RuntimeError::Io(format!(
                "{}: {error} (tokenizer.json must sit next to the model)",
                tokenizer_path.display()
            ))
        })?;
        let template = ChatTemplate::from_file(&template_path).map_err(|error| {
            RuntimeError::Io(format!("{}: {error}", template_path.display()))
        })?;
        let spec = ModelSpec::placeholder();
        let mut core = self.core.lock().map_err(poisoned)?;
        *core = CoreEngine::load(spec.clone()).map_err(internal)?;
        let mut backend = self.backend.lock().map_err(poisoned)?;
        backend
            .prepare(&spec)
            .map_err(|error| RuntimeError::Backend(error.to_string()))?;
        drop(backend);
        drop(core);
        *self.real.lock().map_err(poisoned)? = Some(RealEngine {
            model,
            tokenizer,
            template,
            context: 2048,
        });
        Ok(())
    }

    pub fn session(self: &Arc<Self>) -> Arc<SessionHandle> {
        Arc::new(SessionHandle {
            engine: Arc::clone(self),
            cancelled: AtomicBool::new(false),
            busy: AtomicBool::new(false),
        })
    }
}

/// One generation stream. Cheap to clone through `Arc`.
pub struct SessionHandle {
    engine: Arc<EngineHandle>,
    cancelled: AtomicBool,
    busy: AtomicBool,
}

impl SessionHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Emit exactly [`STUB_TOKEN_COUNT`] fake tokens and a final `Done`.
    ///
    /// `on_event` returns `false` to abort the stream, which is reported as
    /// [`RuntimeError::Cancelled`].
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        sampling: Sampling,
        on_event: &mut dyn FnMut(StreamEvent) -> bool,
    ) -> Result<u32> {
        validate_messages(messages)?;
        validate_sampling(sampling)?;
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(RuntimeError::InvalidState(
                "session is already generating".to_owned(),
            ));
        }
        let _busy = BusyGuard(&self.busy);
        self.cancelled.store(false, Ordering::Release);

        {
            let mut guard = self.engine.real.lock().map_err(poisoned)?;
            if let Some(real) = guard.as_mut() {
                let mut dom: Vec<lfm_domain::Message> = Vec::with_capacity(messages.len());
                for message in messages {
                    let role = match message.role.as_str() {
                        "system" => lfm_domain::Role::System,
                        "assistant" => lfm_domain::Role::Assistant,
                        "tool" => lfm_domain::Role::Tool,
                        _ => lfm_domain::Role::User,
                    };
                    dom.push(lfm_domain::Message {
                        role,
                        content: Some(message.content.clone()),
                        thinking: None,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                }
                let prompt = real
                    .template
                    .render(&dom, &[], true, false, false)
                    .map_err(|error| RuntimeError::InvalidArgument(format!("template: {error}")))?;
                let ids = real
                    .tokenizer
                    .encode(&prompt, false)
                    .map_err(|error| RuntimeError::InvalidArgument(format!("tokenize: {error}")))?;
                let eos = real.tokenizer.eos_id();
                let ctx = real.context;
                let cfg = GenerateConfig {
                    max_tokens: sampling.max_tokens as usize,
                    sampling: SamplingParams {
                        temperature: sampling.temperature,
                        top_p: sampling.top_p,
                        top_k: sampling.top_k as usize,
                        seed: sampling.seed,
                    },
                    eos: Some(eos),
                    stop: Vec::new(),
                };
                let mut session = real.model.session(ctx).map_err(runtime_err)?;
                let mut generated: Vec<u32> = Vec::new();
                let mut printed = String::new();
                let mut total: u32 = 0;
                let mut stop = false;
                session
                    .generate(&ids, &cfg, |token| {
                        if stop {
                            return;
                        }
                        generated.push(token);
                        if let Ok(text) = real.tokenizer.decode(&generated, true) {
                            if text.len() > printed.len() && text.starts_with(&printed) {
                                let delta = text[printed.len()..].to_string();
                                printed = text;
                                let index = total;
                                total += 1;
                                if !on_event(StreamEvent::Delta { index, token, text: delta }) {
                                    stop = true;
                                }
                            }
                        }
                    })
                    .map_err(runtime_err)?;
                on_event(StreamEvent::Done { total });
                return Ok(total);
            }
        }
        for index in 0..STUB_TOKEN_COUNT {
            if self.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let token = {
                let mut core = self.engine.core.lock().map_err(poisoned)?;
                core.step(1).tokens.first().copied().unwrap_or(0)
            };
            std::thread::sleep(STUB_TOKEN_DELAY);
            if self.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let event = StreamEvent::Delta {
                index,
                token,
                text: format!("<{token}>"),
            };
            if !on_event(event) {
                return Err(RuntimeError::Cancelled);
            }
        }

        on_event(StreamEvent::Done {
            total: STUB_TOKEN_COUNT,
        });
        Ok(STUB_TOKEN_COUNT)
    }
}

struct BusyGuard<'a>(&'a AtomicBool);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn validate_messages(messages: &[ChatMessage]) -> Result<()> {
    if messages.is_empty() {
        return Err(RuntimeError::InvalidArgument(
            "at least one message is required".to_owned(),
        ));
    }
    for (index, message) in messages.iter().enumerate() {
        if message.role.trim().is_empty() {
            return Err(RuntimeError::InvalidArgument(format!(
                "message {index} has an empty role"
            )));
        }
    }
    Ok(())
}

fn validate_sampling(sampling: Sampling) -> Result<()> {
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return Err(RuntimeError::InvalidArgument(
            "temperature must be finite and non-negative".to_owned(),
        ));
    }
    if !sampling.top_p.is_finite() || !(0.0..=1.0).contains(&sampling.top_p) {
        return Err(RuntimeError::InvalidArgument(
            "top_p must be within 0..=1".to_owned(),
        ));
    }
    Ok(())
}

fn poisoned<T>(_: std::sync::PoisonError<MutexGuard<'_, T>>) -> RuntimeError {
    RuntimeError::Internal("engine lock poisoned".to_owned())
}

fn runtime_err(error: myelin_core::RuntimeError) -> RuntimeError {
    RuntimeError::Backend(error.to_string())
}

fn internal(error: myelin_core::CoreError) -> RuntimeError {
    RuntimeError::Internal(error.to_string())
}
