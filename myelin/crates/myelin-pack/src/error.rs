//! Errors of the weight packer.
//!
//! Every fallible entry point returns [`Result`]; user input (file paths,
//! tensor names, blob contents) is validated instead of unwrapped.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, PackError>;

#[derive(Debug, Error)]
pub enum PackError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("gguf error: {0}")]
    Gguf(String),
    #[error("dequantization error: {0}")]
    Dequant(String),
    #[error("matrix is empty: k={k}, n={n}")]
    EmptyMatrix { k: usize, n: usize },
    #[error("matrix data length {actual} does not match k={k} * n={n} = {expected}")]
    MatrixShape {
        k: usize,
        n: usize,
        expected: usize,
        actual: usize,
    },
    #[error("tensor `{0}` not found in the gguf file")]
    TensorNotFound(String),
    #[error("tensor `{name}` is {dims}-dimensional, expected a 2d matrix")]
    TensorNotMatrix { name: String, dims: usize },
    #[error("tensor `{name}` uses unsupported ggml type {ggml_type}")]
    UnsupportedGgmlType { name: String, ggml_type: String },
    #[error("non-finite weight at row {row}, column {col}: {value}")]
    NonFiniteWeight { row: usize, col: usize, value: f32 },
    #[error("row {row} is empty")]
    EmptyRow { row: usize },
    #[error("scale {scale} for channel {channel} is not representable as fp16")]
    ScaleNotRepresentable { channel: usize, scale: f32 },
    #[error("blob too small: {actual} bytes, need at least {expected}")]
    BlobTooSmall { expected: usize, actual: usize },
    #[error("bad blob magic {magic:02x?}, expected {expected:02x?}")]
    BadMagic { magic: [u8; 4], expected: [u8; 4] },
    #[error("unsupported blob format version {0}")]
    UnsupportedBlobVersion(u16),
    #[error("unsupported blob dtype tag {0}")]
    UnsupportedBlobDType(u8),
    #[error("blob header is inconsistent: {0}")]
    InconsistentBlob(String),
    #[error("manifest does not describe this blob: {0}")]
    ManifestMismatch(String),
    #[error("quality measurement shape mismatch: {0}")]
    QualityShape(String),
    #[error("quality below threshold: {metric} = {actual:.6} < {threshold:.6}")]
    QualityBelowThreshold {
        metric: String,
        actual: f64,
        threshold: f64,
    },
    #[error("shortconv blob is invalid: {0}")]
    Shortconv(String),
    #[error("kv page image is invalid: {0}")]
    KvPage(String),
    #[error("kvarn page is invalid: {0}")]
    Kvarn(String),
    #[error("raw input file is invalid: {0}")]
    RawInput(String),
}
