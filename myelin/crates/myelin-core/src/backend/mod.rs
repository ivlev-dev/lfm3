//! Compute backend boundary.
//!
//! A backend owns model weights and executes *layer-level* operations. The
//! graph, the KV cache and sampling stay on the host, so the same trait serves
//! two very different implementations:
//!
//! * [`CpuBackend`] dequantizes a GGUF checkpoint to f32 and runs the reference
//!   math on the host;
//! * `DspBackend` (stub in [`dsp`]) will keep weights resident on the Hexagon
//!   and turn each trait method into one FastRPC command, mirroring the
//!   `NpuBackend` transport in `myelin-npu`.
//!
//! One call per layer is deliberate: the DSP stand is priced per RPC
//! (`dsp-kernels.md`), and norms are separate entries because the device keeps
//! the norm weights resident too.
//!
//! The host owns the sequence: position, chat template, sampling and the page
//! table. [`AttentionCall`] carries the token index and the resolved window so
//! the device never has to know about the scheduler.

pub mod cpu;
pub mod dsp;

use std::fmt;

pub use cpu::CpuBackend;
pub use dsp::DspBackend;

use crate::kv::KvCache;
use crate::spec::{AttentionKind, ModelSpec};

/// Failures a compute backend can report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The backend cannot execute this operation yet.
    Unsupported {
        backend: &'static str,
        operation: &'static str,
        reason: &'static str,
    },
    /// An activation buffer has the wrong length.
    Shape {
        operation: &'static str,
        expected: usize,
        got: usize,
    },
    /// The KV cache or its window is inconsistent.
    Kv(String),
    /// The model is not loaded as the spec claims.
    Model(String),
    /// Resident weights do not fit the device memory budget.
    Budget(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                backend,
                operation,
                reason,
            } => write!(f, "{backend} backend cannot run {operation}: {reason}"),
            Self::Shape {
                operation,
                expected,
                got,
            } => write!(f, "{operation} expected {expected} values, got {got}"),
            Self::Kv(message) => write!(f, "kv cache: {message}"),
            Self::Model(message) => write!(f, "backend model: {message}"),
            Self::Budget(message) => write!(f, "weight budget: {message}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Which learned norm a call refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormSlot {
    /// `blk.N.attn_norm.weight`
    Attention,
    /// `blk.N.post_attention_norm.weight`
    PostAttention,
    /// `blk.N.ffn_norm.weight`
    Ffn,
    /// `blk.N.post_ffw_norm.weight`
    PostFfn,
}

/// Weight address for one RMSNorm call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormWeights {
    /// One of the four per-layer norms.
    Layer { layer: usize, slot: NormSlot },
    /// Per-head query norm (`blk.N.attn_q_norm.weight`).
    Query { layer: usize },
    /// Per-head key norm (`blk.N.attn_k_norm.weight`).
    Key { layer: usize },
    /// Final `output_norm.weight`.
    Output,
}

/// Everything the backend needs for one attention layer call.
#[derive(Debug, Clone, Copy)]
pub struct AttentionCall<'a> {
    /// Layer index in the model.
    pub layer: usize,
    /// Absolute position of the token being appended.
    pub position: usize,
    /// First token index this layer attends to (window resolved by the host).
    pub start: usize,
    /// Full or sliding attention; selects the rope table on the device.
    pub kind: AttentionKind,
    /// Input activation, already normalized: `dims.hidden` values.
    pub input: &'a [f32],
}

/// Layer-level compute contract.
pub trait Backend {
    /// Short backend identifier used in errors and logs.
    fn name(&self) -> &'static str;

    /// Spec adopted at load time.
    fn spec(&self) -> &ModelSpec;

    /// Embedding lookup for one token id.
    fn embed(&mut self, token: u32, out: &mut [f32]) -> Result<(), BackendError>;

    /// Apply one learned RMSNorm.
    fn rms_norm(
        &mut self,
        weights: NormWeights,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError>;

    /// `down @ (silu(gate @ x) * (up @ x))`.
    fn ffn_layer(&mut self, layer: usize, x: &[f32], out: &mut [f32]) -> Result<(), BackendError>;

    /// q/k/v projections, query/key norms, rope, causal attention, head gate
    /// and the output projection. Appends the new key/value to `kv` and attends
    /// over `call.start ..= call.position`.
    fn attention_layer(
        &mut self,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError>;

    /// Short convolution over `state` (the previous `ksize - 1` inputs).
    fn conv_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError>;

    /// Vocabulary projection of the final hidden state.
    fn lm_head(&mut self, x: &[f32], logits: &mut [f32]) -> Result<(), BackendError>;

    /// Forget the sequence state the backend keeps on the device (CONV rings,
    /// counters). The default is a no-op for stateless backends.
    fn reset(&mut self) -> Result<(), BackendError> {
        Ok(())
    }
}
