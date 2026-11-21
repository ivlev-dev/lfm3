use std::fmt;

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
