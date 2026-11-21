//! UniFFI surface: objects, dictionaries and errors described by `myelin.udl`.

use std::fmt;
use std::sync::Arc;

use crate::capi::MYELIN_ABI_VERSION;
use crate::runtime::{BackendKind, EngineHandle, RuntimeError, SessionHandle, StreamEvent};
use crate::{ChatMessage, Sampling};

/// Errors exposed to Kotlin/Swift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MyelinError {
    InvalidArgument,
    InvalidState,
    Io,
    Backend,
    Cancelled,
    Internal,
}

impl fmt::Display for MyelinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::InvalidArgument => "invalid argument",
            Self::InvalidState => "invalid state",
            Self::Io => "I/O error",
            Self::Backend => "backend error",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal error",
        };
        f.write_str(name)
    }
}

impl std::error::Error for MyelinError {}

impl From<RuntimeError> for MyelinError {
    fn from(error: RuntimeError) -> Self {
        match error {
            RuntimeError::InvalidArgument(_) => Self::InvalidArgument,
            RuntimeError::InvalidState(_) => Self::InvalidState,
            RuntimeError::Io(_) => Self::Io,
            RuntimeError::Backend(_) => Self::Backend,
            RuntimeError::Cancelled => Self::Cancelled,
            RuntimeError::Internal(_) => Self::Internal,
        }
    }
}

/// Streaming event delivered to [`TokenSink`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenEvent {
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

/// Callback implemented by Kotlin/Swift. Matches `callback interface TokenSink`
/// in `myelin.udl`; UniFFI generates the foreign-call plumbing.
pub trait TokenSink: Send + Sync {
    fn on_token(&self, event: TokenEvent);
}

/// Engine object. Holds a validated specification and the selected backend.
pub struct Engine {
    inner: Arc<EngineHandle>,
}

impl Engine {
    pub fn new(model_path: Option<String>) -> Result<Self, MyelinError> {
        let handle = EngineHandle::new(BackendKind::Auto)?;
        if let Some(path) = model_path {
            handle.load(&path)?;
        }
        Ok(Self { inner: handle })
    }

    pub fn load(&self, path: String) -> Result<(), MyelinError> {
        self.inner.load(&path)?;
        Ok(())
    }

    pub fn new_session(&self) -> Arc<Session> {
        Arc::new(Session {
            inner: self.inner.session(),
        })
    }
}

/// A single generation stream.
pub struct Session {
    inner: Arc<SessionHandle>,
}

impl Session {
    pub fn generate(
        &self,
        messages: Vec<ChatMessage>,
        sampling: Sampling,
        callback: Box<dyn TokenSink>,
    ) -> Result<(), MyelinError> {
        let mut emit = |event: StreamEvent| -> bool {
            callback.on_token(match event {
                StreamEvent::Delta { index, token, text } => {
                    TokenEvent::Delta { index, token, text }
                }
                StreamEvent::ToolCall { payload } => TokenEvent::ToolCall { payload },
                StreamEvent::Done { total } => TokenEvent::Done { total },
                StreamEvent::Error { message } => TokenEvent::Error { message },
            });
            true
        };
        self.inner.generate(&messages, sampling, &mut emit)?;
        Ok(())
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }
}

/// Exported as the `myelin` namespace function.
pub fn abi_version() -> u32 {
    MYELIN_ABI_VERSION
}
