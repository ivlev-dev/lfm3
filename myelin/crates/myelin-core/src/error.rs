use std::fmt;

use crate::backend::BackendError;

/// Errors produced while reading metadata and building a model specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// No architecture adapter claimed the metadata.
    UnsupportedArchitecture(String),
    /// A required metadata key is absent.
    MissingKey(String),
    /// A metadata value is present but unusable.
    InvalidValue { key: String, value: String },
    /// The assembled specification is internally inconsistent.
    InvalidSpec(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture(arch) => {
                write!(f, "unsupported architecture: {arch}")
            }
            Self::MissingKey(key) => write!(f, "missing metadata key: {key}"),
            Self::InvalidValue { key, value } => {
                write!(f, "invalid value for {key}: {value}")
            }
            Self::InvalidSpec(reason) => write!(f, "invalid model spec: {reason}"),
        }
    }
}

impl std::error::Error for CoreError {}

/// Errors produced by the host runtime: loading a checkpoint, running a step,
/// tokenizing or sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    /// Metadata or the assembled specification is unusable.
    Spec(CoreError),
    /// The GGUF container could not be opened or parsed.
    Gguf(String),
    /// A tensor the model needs is absent from the checkpoint.
    MissingTensor(String),
    /// A tensor is present with an unexpected shape.
    TensorShape {
        name: String,
        expected: Vec<usize>,
        found: Vec<u64>,
    },
    /// A tensor uses a storage type this backend cannot dequantize.
    UnsupportedTensorType { name: String, ggml_type: String },
    /// A compute backend refused an operation.
    Backend(BackendError),
    /// The sequence grew past the KV capacity it was created with.
    ContextOverflow { position: usize, capacity: usize },
    /// The caller passed an argument that cannot be honoured.
    InvalidArgument(String),
    /// Encoding or decoding text failed.
    Tokenizer(String),
    /// Sampling parameters are inconsistent.
    Sampling(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(error) => write!(f, "{error}"),
            Self::Gguf(message) => write!(f, "gguf: {message}"),
            Self::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Self::TensorShape {
                name,
                expected,
                found,
            } => write!(
                f,
                "tensor {name} has shape {found:?}, expected {expected:?}"
            ),
            Self::UnsupportedTensorType { name, ggml_type } => {
                write!(f, "tensor {name} uses unsupported storage type {ggml_type}")
            }
            Self::Backend(error) => write!(f, "{error}"),
            Self::ContextOverflow { position, capacity } => write!(
                f,
                "sequence position {position} exceeds KV capacity {capacity}"
            ),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Tokenizer(message) => write!(f, "tokenizer: {message}"),
            Self::Sampling(message) => write!(f, "sampling: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<CoreError> for RuntimeError {
    fn from(error: CoreError) -> Self {
        Self::Spec(error)
    }
}

impl From<BackendError> for RuntimeError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}
