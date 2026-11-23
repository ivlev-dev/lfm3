//! NPU transport boundary for Myelin.
//!
//! Two layers live here:
//!
//! * the legacy [`NpuBackend`]/[`MockBackend`] step contract used by the FFI
//!   skeleton, kept so the Android build keeps working;
//! * the FastRPC transport ([`Transport`]) that `myelin-core`'s `DspBackend`
//!   drives. [`Cdsprpc`] is the Android implementation, [`MockTransport`] is
//!   the deterministic host implementation.
//!
//! [`HvxClient`] wraps one `lfmgemv` handle and exposes typed methods
//! (`ffn`, `attn`, `gemv`, ...); [`method`] owns the method IDs and scalar
//! layouts. [`weights`] uploads `myelin-pack` blobs into resident DSP buffers
//! under an explicit memory budget with LRU eviction.
//!
//! # Device deployment
//!
//! The handle ABI is the 32-bit `remote_handle_*` one used by the `qaic`
//! stub; `remote_handle64_*` returns `AEE_EBADPARM` on the XQ-BC72.
//!
//! `ADSP_LIBRARY_PATH` must name **exactly one** directory holding
//! `liblfmgemv_skel.so`. A colon-separated list is rejected by the driver with
//! `AEE_EUNABLETOLOAD`, even when every entry is valid.

use std::fmt;

use myelin_core::ModelSpec;

pub mod error;
pub mod executor;
pub mod gemv_weights;
pub mod hvx;
pub mod layer_fixture;
pub mod method;
pub mod mock;
pub mod model_pack;
pub mod segment;
pub mod transport;
pub mod weights;

#[cfg(target_os = "android")]
pub mod cdsprpc;

pub use error::{fastrpc_error_name, rpc_error, TransportError};
pub use executor::Executor;
pub use gemv_weights::{PackedGemv, GEMV_LOAD_CHUNK};
pub use hvx::HvxClient;
pub use method::{
    exec, AttnRequest, AttnResponse, AttnScalars, ExecRequest, ExecResponse, FfnRequest,
    FfnResponse, FfnScalars, GemvRequest, GemvResponse, GemvScalars,
};
pub use mock::{MockCall, MockReply, MockTransport, RecordedCall};
pub use segment::{
    SegPacked, SegmentError, Segments, RATIO_ONE, SEGMENT, SEG_HEADER_BYTES, SEG_MAGIC, SEG_VERSION,
};
pub use transport::{Call, Handle, SharedBuffer, SharedOwner, Transport};
pub use weights::{
    AttentionFullPlan, AttentionParams, AttentionPlan, ConvPlan, FfnPlan, LayerPlan, LmHeadPlan,
    LoadStrategy, PackedMatrix, PackedTensor, ResidentImage, ResidentWeights, StageKind,
    WeightBudget, WeightError, WeightPlan, WeightPlanBuilder, WeightStats, LOAD_CHUNK,
};

#[cfg(target_os = "android")]
pub use cdsprpc::{Cdsprpc, CdsprpcConfig, FASTRPC_ATTR_KEEP_MAP};

/// Transport-level failures of the legacy step contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpuError {
    NotPrepared,
    Unsupported(String),
    Transport(String),
}

impl fmt::Display for NpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPrepared => f.write_str("NPU backend is not prepared"),
            Self::Unsupported(reason) => write!(f, "unsupported NPU operation: {reason}"),
            Self::Transport(reason) => write!(f, "NPU transport error: {reason}"),
        }
    }
}

impl std::error::Error for NpuError {}

/// Backend contract. `step` is intentionally model-agnostic: the spec is
/// passed once through `prepare`, and per-token work uses plain buffers.
pub trait NpuBackend: Send {
    fn name(&self) -> &'static str;

    /// Adopt a validated model spec and allocate device-side state.
    fn prepare(&mut self, spec: &ModelSpec) -> Result<(), NpuError>;

    /// Run one step. Returns the number of output elements written.
    fn step(&mut self, input: &[u32], out: &mut [u32]) -> Result<usize, NpuError>;

    fn reset(&mut self) -> Result<(), NpuError>;
}

/// Deterministic host-side backend used by the skeleton.
#[derive(Debug, Default)]
pub struct MockBackend {
    prepared: bool,
    steps: u64,
}

impl MockBackend {
    pub const fn steps(&self) -> u64 {
        self.steps
    }
}

impl NpuBackend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn prepare(&mut self, _spec: &ModelSpec) -> Result<(), NpuError> {
        self.prepared = true;
        self.steps = 0;
        Ok(())
    }

    fn step(&mut self, input: &[u32], out: &mut [u32]) -> Result<usize, NpuError> {
        if !self.prepared {
            return Err(NpuError::NotPrepared);
        }
        let written = out.len().min(input.len());
        let offset = u32::try_from(self.steps).unwrap_or(u32::MAX);
        for (dst, src) in out.iter_mut().zip(input.iter()).take(written) {
            *dst = src.wrapping_add(offset);
        }
        self.steps = self.steps.saturating_add(1);
        Ok(written)
    }

    fn reset(&mut self) -> Result<(), NpuError> {
        self.steps = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_is_deterministic() {
        let spec = ModelSpec::placeholder();
        let mut backend = MockBackend::default();
        backend.prepare(&spec).expect("prepare");
        assert_eq!(backend.name(), "mock");

        let mut a = [0u32; 3];
        let mut b = [0u32; 3];
        assert_eq!(backend.step(&[1, 2, 3], &mut a), Ok(3));
        assert_eq!(backend.step(&[1, 2, 3], &mut b), Ok(3));
        assert_eq!(a, [1, 2, 3]);
        assert_eq!(b, [2, 3, 4]);

        backend.reset().expect("reset");
        let mut c = [0u32; 3];
        backend.step(&[1, 2, 3], &mut c).expect("step");
        assert_eq!(c, a);
    }

    #[test]
    fn unprepared_backend_rejects_step() {
        let mut backend = MockBackend::default();
        let mut out = [0u32; 1];
        assert_eq!(backend.step(&[1], &mut out), Err(NpuError::NotPrepared));
    }

    #[test]
    fn core_method_ids_match_the_table() {
        use myelin_core::backend::dsp::{
            HVX_METHOD_ATTN, HVX_METHOD_FFN, HVX_METHOD_KV_WRITE, HVX_METHOD_RESET_STATE,
        };
        assert_eq!(method::method_id::FFN, HVX_METHOD_FFN);
        assert_eq!(method::method_id::ATTN, HVX_METHOD_ATTN);
        assert_eq!(method::method_id::KV_WRITE, HVX_METHOD_KV_WRITE);
        assert_eq!(method::method_id::RESET_STATE, HVX_METHOD_RESET_STATE);
    }
}
