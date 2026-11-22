//! DSP backend seam.
//!
//! This is the place a Hexagon implementation plugs in. It intentionally holds
//! only the adopted spec: weights will be staged through rpcmem by the
//! transport crate (`myelin-npu`), and every method below becomes one FastRPC
//! command. Until the transport is wired, every call fails with a typed
//! [`BackendError::Unsupported`] that names the missing piece, so the session
//! layer and the FFI can select the backend without silently degrading.

use super::{AttentionCall, Backend, BackendError, NormWeights};
use crate::kv::KvCache;
use crate::spec::ModelSpec;

/// Hexagon backend placeholder.
#[derive(Debug, Clone)]
pub struct DspBackend {
    spec: ModelSpec,
}

impl DspBackend {
    /// Adopt a validated spec. Weight staging is the transport's job.
    pub fn new(spec: ModelSpec) -> Self {
        Self { spec }
    }
}

const REASON: &str = "FastRPC transport and resident weight arenas are not wired yet";

fn unsupported(operation: &'static str) -> BackendError {
    BackendError::Unsupported {
        backend: "dsp",
        operation,
        reason: REASON,
    }
}

impl Backend for DspBackend {
    fn name(&self) -> &'static str {
        "dsp"
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn embed(&mut self, _token: u32, _out: &mut [f32]) -> Result<(), BackendError> {
        Err(unsupported("embed"))
    }

    fn rms_norm(
        &mut self,
        _weights: NormWeights,
        _x: &[f32],
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported("rms_norm"))
    }

    fn ffn_layer(
        &mut self,
        _layer: usize,
        _x: &[f32],
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported("ffn_layer"))
    }

    fn attention_layer(
        &mut self,
        _call: AttentionCall<'_>,
        _kv: &mut dyn KvCache,
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported("attention_layer"))
    }

    fn conv_layer(
        &mut self,
        _layer: usize,
        _x: &[f32],
        _state: &mut [f32],
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported("conv_layer"))
    }

    fn lm_head(&mut self, _x: &[f32], _logits: &mut [f32]) -> Result<(), BackendError> {
        Err(unsupported("lm_head"))
    }
}
