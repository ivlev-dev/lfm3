//! DSP backend seam.
//!
//! The transport itself lives in `myelin-npu`; this module owns the contract
//! between the model runtime and that transport, plus the layer-level executor
//! that turns [`Backend`] calls into FastRPC commands.
//!
//! The dependency direction is `myelin-npu` -> `myelin-core`, so the core
//! cannot name `myelin-npu` types. [`DspExecutor`] is therefore a narrow,
//! object-safe view of the transport: open/close a FastRPC handle and invoke
//! one method with scalar parameters and byte buffers. `myelin-npu` implements
//! it for its `Cdsprpc` and `MockTransport`.
//!
//! # Staged layers
//!
//! The HVX stand keeps quantized weights resident on the DSP. `DspBackend`
//! does not convert GGUF weights: a caller stages the resident handles (from
//! `alloc_buf`/`load_buf`/`set_scale`) and the activation/output scales through
//! [`DspBackend::stage_ffn`] and [`DspBackend::stage_attention`]. A layer with
//! no stage fails with a typed [`BackendError::Unsupported`] instead of
//! silently computing garbage.

use std::fmt;
use std::sync::Arc;

use super::host::HostWeights;
use super::{AttentionCall, Backend, BackendError, LmHeadStage, NormWeights};
use crate::kv::KvCache;
use crate::spec::{AttentionKind, ModelSpec};

/// Method ID of `lfmgemv.ffn` in the HVX FastRPC skeleton.
///
/// Kept in sync with `myelin-npu`'s method table by a test in that crate.
pub const HVX_METHOD_FFN: u32 = 6;
/// Method ID of `lfmgemv.attn` in the HVX FastRPC skeleton.
pub const HVX_METHOD_ATTN: u32 = 11;
/// Method ID of `lfmgemv.load_buf` in the HVX FastRPC skeleton.
pub const HVX_METHOD_LOAD_BUF: u32 = 2;
/// Method ID of `lfmgemv.kv_write` in the HVX FastRPC skeleton.
pub const HVX_METHOD_KV_WRITE: u32 = 22;
/// Method ID of `lfmgemv.reset_state` in the HVX FastRPC skeleton.
pub const HVX_METHOD_RESET_STATE: u32 = 23;

/// FastRPC handle. 32-bit `remote_handle`, as used by the `qaic` stub.
pub type DspHandle = i32;

/// One FastRPC invocation.
///
/// The wire order is the one `qaic` generates: `prim_in`, then input
/// sequences, then `prim_out` (scalar outputs), then output sequences. The
/// transport encodes the counts into `REMOTE_SCALARS_MAKEX`.
pub struct DspCall<'a, 'b> {
    /// Method index inside the `lfmgemv` interface.
    pub method: u32,
    /// Scalar input pack (`in long` arguments), little-endian.
    pub prim_in: &'a [u8],
    /// Scalar output pack (`rout long` arguments), little-endian.
    pub prim_out: &'b mut [u8],
    /// Input sequences (`in sequence<octet>` / `in sequence<long>`).
    pub in_bufs: &'a [&'a [u8]],
    /// Output sequences (`rout sequence<...>`).
    pub out_bufs: &'b mut [&'b mut [u8]],
}

/// Result of one `exec` invocation: the `out` sequence and the DSP counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    pub out: Vec<u8>,
    /// `{dsp_us, failed_command or -1, commands_run, team_fallbacks, ...}`.
    pub times: [i32; 8],
}

impl ExecResult {
    /// Index of the command that failed, or `None`.
    pub const fn failed_command(&self) -> Option<usize> {
        if self.times[1] < 0 {
            None
        } else {
            Some(self.times[1] as usize)
        }
    }
}

/// Transport-level failures surfaced by an executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DspError {
    /// `libcdsprpc.so` is absent or a required symbol is missing.
    Unavailable(String),
    /// `remote_handle_open` returned a non-zero FastRPC code.
    HandleOpen { code: i32 },
    /// A method returned a non-zero FastRPC code.
    Rpc {
        method: u32,
        code: i32,
        name: &'static str,
    },
    /// The RPC or the device did not answer in time.
    Timeout { method: u32, millis: u64 },
    /// A buffer handed to the transport was smaller than the method needs.
    BufferTooSmall { expected: usize, got: usize },
    /// A call was made on a handle that is no longer open.
    Closed,
    /// Anything the transport could describe but not classify above.
    Other(String),
}

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "FastRPC unavailable: {message}"),
            Self::HandleOpen { code } => {
                write!(f, "FastRPC handle open failed with 0x{code:08x}")
            }
            Self::Rpc { method, code, name } => write!(
                f,
                "FastRPC method {method} failed with 0x{code:08x} ({name})"
            ),
            Self::Timeout { method, millis } => {
                write!(f, "FastRPC method {method} timed out after {millis} ms")
            }
            Self::BufferTooSmall { expected, got } => {
                write!(f, "FastRPC buffer too small: need {expected}, got {got}")
            }
            Self::Closed => f.write_str("FastRPC handle is closed"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for DspError {}

/// Failure of an on-demand layer stage provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// The layer's resident weights do not fit the memory budget.
    Budget(String),
    /// The provider has no plan for this layer or stage kind.
    Unknown(String),
    /// The plan, manifest or blob is inconsistent.
    Layout(String),
    /// A FastRPC call failed while staging.
    Transport(DspError),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(message) => write!(f, "weight budget: {message}"),
            Self::Unknown(message) => write!(f, "no resident plan: {message}"),
            Self::Layout(message) => write!(f, "resident weight layout: {message}"),
            Self::Transport(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for StageError {}

/// Supplies resident layer stages on demand.
///
/// `DspBackend` asks the provider only when a layer has no explicitly staged
/// handles, so the planner owns the budget, the LRU order and the transport
/// buffers. The provider is a host-side object: all device work goes through
/// the executor and handle it was built with.
pub trait StageProvider: Send + Sync {
    /// Load (or reuse) the resident FFN weights of `layer`.
    fn ffn_stage(&mut self, layer: usize) -> Result<FfnStage, StageError>;
    /// Load (or reuse) the resident attention state of `layer`.
    fn attention_stage(&mut self, layer: usize) -> Result<AttentionStage, StageError>;
    /// Load (or reuse) a resident short-convolution stage of `layer`.
    ///
    /// The default rejects the request, so a planner that only knows FFN and
    /// paged-attention caches keeps working unchanged.
    fn conv_stage(&mut self, _layer: usize) -> Result<ConvStage, StageError> {
        Err(StageError::Unknown(
            "the weight provider has no short-convolution stage".to_owned(),
        ))
    }
    /// Load (or reuse) a fully wired attention layer (projections + caches).
    fn attention_full_stage(&mut self, _layer: usize) -> Result<AttentionFullStage, StageError> {
        Err(StageError::Unknown(
            "the weight provider has no full attention stage".to_owned(),
        ))
    }
    /// Resident vocabulary projection, when the provider stages one.
    fn lm_head_stage(&mut self) -> Result<Option<LmHeadStage>, StageError> {
        Ok(None)
    }
    /// Release every resident buffer. Must be idempotent.
    fn release_all(&mut self);
}

/// Object-safe transport view used by [`DspBackend`].
pub trait DspExecutor: Send + Sync {
    /// Open one FastRPC module (for example `"lfmgemv"`).
    fn open(&self, module: &str) -> Result<DspHandle, DspError>;
    /// Close a handle. Implementations must tolerate a handle already closed.
    fn close(&self, handle: DspHandle) -> Result<(), DspError>;
    /// Invoke one method on `handle`.
    fn invoke(&self, handle: DspHandle, call: DspCall<'_, '_>) -> Result<(), DspError>;

    /// Reserve the DSP-heap scratch arena used by parsed `exec` lists.
    ///
    /// `nbytes <= 0` frees it. Implementations that cannot allocate a shared
    /// command arena keep the default and reject [`DspExecutor::exec`].
    fn reserve_scratch(&self, _handle: DspHandle, _nbytes: i64) -> Result<(), DspError> {
        Err(DspError::Unavailable(
            "this executor does not implement `reserve_scratch`".to_owned(),
        ))
    }

    /// Run a command list in one FastRPC call.
    ///
    /// The list must be in [`exec::List::to_bytes`] form. Implementations own
    /// the host shared-memory arena: `myelin-npu`'s `Executor` keeps one
    /// mapped arena per distinct list (the list is immutable between tokens,
    /// so `map_arena`/`unmap_arena` leave the hot path); a one-shot
    /// implementation may map, call `exec` and unmap. `act` is copied by the
    /// DSP into scratch at offset 0; `rt` carries the per-token values
    /// referenced by `rt_base` and is therefore not part of the list.
    fn exec(&self, _handle: DspHandle, _request: ExecRun<'_>) -> Result<ExecResult, DspError> {
        Err(DspError::Unavailable(
            "this executor does not implement `exec` (no shared command arena)".to_owned(),
        ))
    }
}

/// Arguments of [`DspExecutor::exec`].
pub struct ExecRun<'a> {
    /// Encoded list (header + 96-byte commands).
    pub list: &'a [u8],
    /// Number of commands; must match the header.
    pub commands: usize,
    /// Scratch bytes the list addresses.
    pub scratch_bytes: u32,
    /// Initial activation, copied to DSP scratch offset 0.
    pub act: &'a [u8],
    /// Per-token values (`pos`, `seq`, `window`, `reps`) referenced by commands.
    pub rt: &'a [i32],
    /// Size of the `out` sequence.
    pub out_len: usize,
}

/// Resident FFN layer: gate/up `[ffn, hidden]` and down `[hidden, ffn]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FfnStage {
    /// Handle returned by `lfmgemv.alloc_buf` for the gate matrix.
    pub gate: i32,
    /// Handle for the up matrix.
    pub up: i32,
    /// Handle for the down matrix.
    pub down: i32,
    /// Multiplier applied to the f32 activation before int8 quantisation.
    pub act_scale: f32,
    /// Multiplier applied to the int8 result to recover f32 units.
    pub out_scale: f32,
    /// Internal repetitions executed inside one RPC.
    pub reps: i32,
}

impl FfnStage {
    fn validate(&self) -> Result<(), BackendError> {
        for (name, handle) in [("gate", self.gate), ("up", self.up), ("down", self.down)] {
            if handle < 0 {
                return Err(BackendError::Model(format!(
                    "ffn {name} handle is negative: {handle}"
                )));
            }
        }
        for (name, scale) in [("act_scale", self.act_scale), ("out_scale", self.out_scale)] {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(BackendError::Model(format!(
                    "ffn {name} must be finite and positive, got {scale}"
                )));
            }
        }
        if self.reps < 1 {
            return Err(BackendError::Model(format!(
                "ffn reps must be positive, got {}",
                self.reps
            )));
        }
        Ok(())
    }
}

/// Resident paged-attention state for one layer.
///
/// The HVX stand is fixed at 16 query heads, 4 KV heads and head_dim 128;
/// [`DspBackend::attention_layer`] rejects other geometries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionStage {
    /// `hk`: int4 key cache, `4 * slots * 64` bytes (token-major layout).
    pub keys: i32,
    /// `hkt`: per-token key scales, `4 * slots * 4` bytes.
    pub key_scales: i32,
    /// `hv`: int8 value cache, `4 * slots * 128` bytes.
    pub values: i32,
    /// `hvs`: value scales; `0` is accepted by the stand (unused today).
    pub value_scales: i32,
    /// Multiplier applied to the f32 query before int8 quantisation.
    pub q_scale: f32,
    /// Multiplier applied to the int32 Q12 attention result.
    pub out_scale: f32,
    /// Internal repetitions executed inside one RPC.
    pub reps: i32,
    /// Number of 32-token pages the resident caches can hold.
    pub cache_pages: usize,
    /// The caller has already applied q/k/v projections, q/k norm and rope,
    /// and wrote quantised K/V into the resident caches. The `attn` method
    /// itself only computes paged attention over an int8 q.
    pub preprojected: bool,
}

impl AttentionStage {
    fn validate(&self) -> Result<(), BackendError> {
        for (name, handle) in [
            ("hk", self.keys),
            ("hkt", self.key_scales),
            ("hv", self.values),
            ("hvs", self.value_scales),
        ] {
            if handle < 0 {
                return Err(BackendError::Model(format!(
                    "attention {name} handle is negative: {handle}"
                )));
            }
        }
        for (name, scale) in [("q_scale", self.q_scale), ("out_scale", self.out_scale)] {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(BackendError::Model(format!(
                    "attention {name} must be finite and positive, got {scale}"
                )));
            }
        }
        if self.reps < 1 {
            return Err(BackendError::Model(format!(
                "attention reps must be positive, got {}",
                self.reps
            )));
        }
        if self.cache_pages == 0 {
            return Err(BackendError::Model(
                "attention cache_pages must be positive".to_owned(),
            ));
        }
        if !self.preprojected {
            return Err(BackendError::Model(
                "attention stage must be preprojected: the HVX attn RPC takes a projected q"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Resident short-convolution stage for one conv layer.
///
/// `in_proj` maps `hidden -> 3 * hidden` (`b`, `c`, `x` blocks), the `CONV`
/// blob holds the rotated taps and `out_proj` maps back to `hidden`. All three
/// commands run in one `exec` list: int8 `bcx` stays in DSP scratch, so a
/// layer costs one RPC, not three.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvStage {
    /// int4 GEMV handle of `shortconv.in_proj` (`N = 3 * hidden`).
    pub in_proj: i32,
    /// `CONV` blob handle (`myelin-pack conv`).
    pub conv: i32,
    /// int4 GEMV handle of `shortconv.out_proj` (`N = hidden`).
    pub out_proj: i32,
    /// Shared right shift of the blob's `kq` table (`CONV` argument `a1`).
    pub ksh: i32,
    /// f32 input -> int8 scale.
    pub act_scale: f32,
    /// Scale of the `in_proj` int8 output (`bcx`).
    pub mix_scale: f32,
    /// Scale of the `CONV` int8 output (`y`).
    pub y_scale: f32,
    /// Scale of the final f32 output (dequantisation of the `out_proj` int8).
    pub out_scale: f32,
    /// `reps` for both GEMV commands.
    pub reps: i32,
    /// Scratch offset of the persistent ring state for this layer.
    pub state_off: i32,
}

impl ConvStage {
    fn validate(&self) -> Result<(), BackendError> {
        for (name, handle) in [
            ("in_proj", self.in_proj),
            ("conv", self.conv),
            ("out_proj", self.out_proj),
        ] {
            if handle < 0 {
                return Err(BackendError::Model(format!(
                    "conv {name} handle is negative: {handle}"
                )));
            }
        }
        for (name, scale) in [
            ("act_scale", self.act_scale),
            ("mix_scale", self.mix_scale),
            ("y_scale", self.y_scale),
            ("out_scale", self.out_scale),
        ] {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(BackendError::Model(format!(
                    "conv {name} must be finite and positive, got {scale}"
                )));
            }
        }
        if self.ksh < 1 || self.ksh > 40 {
            return Err(BackendError::Model(format!(
                "conv ksh must be 1..=40, got {}",
                self.ksh
            )));
        }
        if self.state_off < 0 {
            return Err(BackendError::Model(format!(
                "conv state_off is negative: {}",
                self.state_off
            )));
        }
        Ok(())
    }
}

/// Quantisation of the V cache: one real unit per int8 code.
#[derive(Debug, Clone, PartialEq)]
pub enum VQuant {
    /// One scale for the whole image.
    Global(f32),
    /// One scale per head and dimension (`kv_heads * 128` values).
    Channel(Vec<f32>),
}

impl VQuant {
    fn validate(&self, kv_dim: usize) -> Result<(), BackendError> {
        match self {
            Self::Global(scale) => {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(BackendError::Model(format!(
                        "v global scale must be finite and positive, got {scale}"
                    )));
                }
            }
            Self::Channel(scales) => {
                if scales.len() != kv_dim {
                    return Err(BackendError::Model(format!(
                        "v channel scales: {} values for kv_dim {kv_dim}",
                        scales.len()
                    )));
                }
                if scales
                    .iter()
                    .any(|scale| !scale.is_finite() || *scale <= 0.0)
                {
                    return Err(BackendError::Model(
                        "v channel scales must be finite and positive".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn scale(&self, dim: usize) -> f32 {
        match self {
            Self::Global(scale) => *scale,
            Self::Channel(scales) => scales[dim],
        }
    }
}

/// Full attention stage: resident caches, q/k/v/o projections and the host
/// post-processing contract (q/k norm, rope, gate, V scale).
///
/// The paged-attention kernel takes a projected `q` and pre-quantised K/V; the
/// backend therefore runs the projections on the DSP, applies q/k RMSNorm and
/// rope on the host exactly like [`crate::backend::CpuBackend`], writes one
/// token into the KV images and then issues `ATTN`. K/V are appended to the
/// host [`KvCache`] mirror as real f32 values, so the scheduler sees the same
/// sequence the device computes.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionFullStage {
    /// Resident K/KT/V handles and page count.
    pub caches: AttentionStage,
    /// int4 GEMV handle of `attn_q` (`N = q_dim`).
    pub query: i32,
    /// int4 GEMV handle of `attn_k` (`N = kv_dim`).
    pub key: i32,
    /// int4 GEMV handle of `attn_v` (`N = kv_dim`).
    pub value: i32,
    /// int4 GEMV handle of `attn_output` (`N = hidden`).
    pub output: i32,
    /// int4 GEMV handle of `attn_gate` (`N = heads`), if the model has one.
    pub gate: Option<i32>,
    /// Per-head query RMSNorm weights (`head_dim`).
    pub query_norm: Vec<f32>,
    /// Per-head key RMSNorm weights (`head_dim`).
    pub key_norm: Vec<f32>,
    /// f32 input -> int8 scale.
    pub act_scale: f32,
    /// Scale of the int8 q projection output.
    pub q_scale: f32,
    /// Scale of the int8 k projection output.
    pub k_scale: f32,
    /// Scale of the int8 v projection output.
    pub v_scale: f32,
    /// Scale of the int8 gate projection output.
    pub gate_scale: f32,
    /// f32 post-rope q -> int8 scale for `ATTN`.
    pub attn_q_scale: f32,
    /// `ATTN` int32 output is Q12: `real = q12 / 4096`.
    pub attn_out_scale: f32,
    /// f32 attended (after V scale and gate) -> int8 scale for `o_proj`.
    pub o_act_scale: f32,
    /// Scale of the final f32 output (dequantisation of the `o_proj` int8).
    pub out_scale: f32,
    /// Scale used to quantise V into the cache.
    pub v_quant: VQuant,
    /// `reps` for every GEMV command.
    pub reps: i32,
}

impl AttentionFullStage {
    fn validate(&self, dims: &crate::spec::Dims) -> Result<(), BackendError> {
        self.caches.validate()?;
        for (name, handle) in [
            ("query", self.query),
            ("key", self.key),
            ("value", self.value),
            ("output", self.output),
        ] {
            if handle < 0 {
                return Err(BackendError::Model(format!(
                    "attention {name} handle is negative: {handle}"
                )));
            }
        }
        if let Some(gate) = self.gate {
            if gate < 0 {
                return Err(BackendError::Model(format!(
                    "attention gate handle is negative: {gate}"
                )));
            }
        }
        for (name, scale) in [
            ("act_scale", self.act_scale),
            ("q_scale", self.q_scale),
            ("k_scale", self.k_scale),
            ("v_scale", self.v_scale),
            ("gate_scale", self.gate_scale),
            ("attn_q_scale", self.attn_q_scale),
            ("attn_out_scale", self.attn_out_scale),
            ("o_act_scale", self.o_act_scale),
            ("out_scale", self.out_scale),
        ] {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(BackendError::Model(format!(
                    "attention {name} must be finite and positive, got {scale}"
                )));
            }
        }
        if self.query_norm.len() != dims.head_dim || self.key_norm.len() != dims.head_dim {
            return Err(BackendError::Model(format!(
                "attention q/k norm lengths {} / {} != head_dim {}",
                self.query_norm.len(),
                self.key_norm.len(),
                dims.head_dim
            )));
        }
        self.v_quant.validate(dims.kv_dim())?;
        if self.reps < 1 {
            return Err(BackendError::Model(format!(
                "attention reps must be positive, got {}",
                self.reps
            )));
        }
        Ok(())
    }
}

/// Scratch layout shared by every layer list. The input activation is always
/// copied by `exec` to offset 0; the rest is fixed so a list can be reused.
const SCRATCH_ACT: i32 = 0;
const SCRATCH_BCX: i32 = 4096;
const SCRATCH_CY: i32 = 12288;
const CONV_STATE_BASE: i32 = 1 << 20;
const CONV_STATE_STRIDE: i32 = 1 << 18;

/// Scratch arena the backend reserves for `exec` (8 MiB).
///
/// The CONV ring of layer `N` lives at [`conv_state_offset`], `1 MiB + N *
/// 256 KiB`; the deepest LFM3 conv layer is 23, so the arena must reach
/// `1 MiB + 24 * 256 KiB = 7 MiB`. The attention block table and the
/// intermediate int8 buffers sit in the first 16 KiB.
pub const EXEC_SCRATCH_BYTES: u32 = 8 << 20;

/// Persistent `CONV` ring-state offset for `layer`.
///
/// Each layer owns 256 KiB (`8 slots × 2 * hidden * 8` bytes for a 2K hidden),
/// so the histories of all 13 conv layers survive between tokens.
pub const fn conv_state_offset(layer: usize) -> i32 {
    CONV_STATE_BASE + layer as i32 * CONV_STATE_STRIDE
}

///
/// `myelin-core` cannot depend on `myelin-npu`, so the 64-byte header and
/// 96-byte command are transcribed here from `lfm_exec_hdr` / `lfm_cmd` in
/// `myelin/dsp/hvx/lfmgemv_impl.c`. `myelin-npu` has a test that compares
/// this encoder with its own `method::exec` byte for byte.
pub mod exec {
    /// `LFM_EXEC_MAGIC`.
    pub const MAGIC: u32 = 0x454D_464C;
    /// `LFM_EXEC_VERSION`.
    pub const VERSION: u32 = 1;
    pub const HEADER_BYTES: usize = 64;
    pub const COMMAND_BYTES: usize = 96;
    pub const MAX_COMMANDS: usize = 256;

    /// `LFM_SRC_*`.
    pub mod src {
        pub const SCRATCH: u32 = 0;
        pub const OUT: u32 = 1;
    }

    /// `LFM_DST_*`.
    pub mod dst {
        pub const SCRATCH: u32 = 0;
        pub const OUT: u32 = 1;
        pub const ATTACHED: u32 = 2;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Op {
        Quant = 0,
        Gemv = 1,
        Ffn = 2,
        Attn = 3,
        Conv = 4,
        Dump = 5,
        Ffn16 = 6,
    }

    fn put_i32(out: &mut [u8], offset: usize, value: i32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(out: &mut [u8], offset: usize, value: u32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// One 96-byte command, defaults `-1` for handles and `rt_base`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Command {
        pub op: Op,
        pub src: u32,
        pub dst: u32,
        pub h: [i32; 4],
        pub dst_h: i32,
        pub src_off: i32,
        pub src_len: i32,
        pub dst_off: i32,
        pub dst_len: i32,
        pub tbl_h: i32,
        pub tbl_len: i32,
        pub tbl_off: i32,
        pub args: [i32; 4],
        pub rt_base: i32,
    }

    impl Command {
        const fn empty(op: Op) -> Self {
            Self {
                op,
                src: src::SCRATCH,
                dst: dst::SCRATCH,
                h: [-1; 4],
                dst_h: -1,
                src_off: 0,
                src_len: 0,
                dst_off: 0,
                dst_len: 0,
                tbl_h: -1,
                tbl_len: 0,
                tbl_off: 0,
                args: [0; 4],
                rt_base: -1,
            }
        }

        pub const fn quant(src_off: i32, dst_off: i32, n: i32, multiplier_q16: i32) -> Self {
            let mut command = Self::empty(Op::Quant);
            command.src_off = src_off;
            command.src_len = n * 4;
            command.dst_off = dst_off;
            command.dst_len = n;
            command.args[0] = multiplier_q16;
            command
        }

        /// FFN with a 16-bit `mid` buffer (two int8 planes inside the kernel).
        pub const fn ffn16(
            gate: i32,
            up: i32,
            down: i32,
            reps: i32,
            src_off: i32,
            src_len: i32,
            dst_off: i32,
            dst_len: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Ffn16);
            command.h = [gate, up, down, -1];
            command.args[0] = reps;
            command.src_off = src_off;
            command.src_len = src_len;
            command.dst_off = dst_off;
            command.dst_len = dst_len;
            command
        }

        /// Debug: copy `len` bytes from `scratch[src_off]` into the OUT
        /// sequence at `dst_off`. Only the `MYELIN_DUMP_CONV` path uses it.
        pub const fn dump(src_off: i32, dst_off: i32, len: i32) -> Self {
            let mut command = Self::empty(Op::Dump);
            command.src_off = src_off;
            command.src_len = len;
            command.dst_off = dst_off;
            command.dst_len = len;
            command
        }

        pub const fn gemv(
            h: i32,
            m: i32,
            reps: i32,
            src_off: i32,
            src_len: i32,
            dst_off: i32,
            dst_len: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Gemv);
            command.h[0] = h;
            command.args[0] = m;
            command.args[1] = reps;
            command.src_off = src_off;
            command.src_len = src_len;
            command.dst_off = dst_off;
            command.dst_len = dst_len;
            command
        }

        #[allow(clippy::too_many_arguments)]
        pub const fn ffn(
            gate: i32,
            up: i32,
            down: i32,
            reps: i32,
            src_off: i32,
            src_len: i32,
            dst_off: i32,
            dst_len: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Ffn);
            command.h = [gate, up, down, -1];
            command.args[0] = reps;
            command.src_off = src_off;
            command.src_len = src_len;
            command.dst_off = dst_off;
            command.dst_len = dst_len;
            command
        }

        #[allow(clippy::too_many_arguments)]
        pub const fn attn(
            keys: i32,
            key_scales: i32,
            values: i32,
            value_scales: i32,
            tbl_h: i32,
            tbl_len: i32,
            rt_base: i32,
            src_off: i32,
            dst_off: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Attn);
            command.h = [keys, key_scales, values, value_scales];
            command.tbl_h = tbl_h;
            command.tbl_len = tbl_len;
            command.rt_base = rt_base;
            command.src_off = src_off;
            command.src_len = 16 * 128;
            command.dst_off = dst_off;
            command.dst_len = 16 * 128 * 4;
            command
        }

        #[allow(clippy::too_many_arguments)]
        pub const fn conv(
            blob: i32,
            h: i32,
            ksh: i32,
            slot: i32,
            counter: i32,
            state_off: i32,
            src_off: i32,
            dst_off: i32,
        ) -> Self {
            let mut command = Self::empty(Op::Conv);
            command.h[0] = blob;
            command.args = [h, ksh, slot, counter];
            command.tbl_off = state_off;
            command.src_off = src_off;
            command.src_len = 3 * h;
            command.dst_off = dst_off;
            command.dst_len = h;
            command
        }

        pub const fn with_dst(mut self, kind: u32, offset: i32) -> Self {
            self.dst = kind;
            self.dst_off = offset;
            self
        }

        pub const fn with_dst_handle(mut self, handle: i32) -> Self {
            self.dst_h = handle;
            self
        }

        /// Read the per-token fields (ATTN pos/seq/window/reps, CONV counter)
        /// from `rt[rt_base..]` instead of the command itself.
        pub const fn with_rt_base(mut self, rt_base: i32) -> Self {
            self.rt_base = rt_base;
            self
        }

        pub const fn with_table_offset(mut self, offset: i32) -> Self {
            self.tbl_off = offset;
            self
        }

        pub const fn with_table_handle(mut self, handle: i32) -> Self {
            self.tbl_h = handle;
            self
        }

        pub fn to_bytes(&self) -> [u8; COMMAND_BYTES] {
            let mut out = [0u8; COMMAND_BYTES];
            put_u32(&mut out, 0, self.op as u32);
            put_u32(&mut out, 4, self.src);
            put_u32(&mut out, 8, self.dst);
            for (index, handle) in self.h.iter().enumerate() {
                put_i32(&mut out, 16 + 4 * index, *handle);
            }
            put_i32(&mut out, 32, self.dst_h);
            put_i32(&mut out, 36, self.src_off);
            put_i32(&mut out, 40, self.src_len);
            put_i32(&mut out, 44, self.dst_off);
            put_i32(&mut out, 48, self.dst_len);
            put_i32(&mut out, 52, self.tbl_h);
            put_i32(&mut out, 56, self.tbl_len);
            put_i32(&mut out, 60, self.tbl_off);
            for (index, arg) in self.args.iter().enumerate() {
                put_i32(&mut out, 64 + 4 * index, *arg);
            }
            put_i32(&mut out, 80, self.rt_base);
            out
        }
    }

    /// Header + commands of one `exec` list.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct List {
        pub scratch_bytes: u32,
        pub commands: Vec<Command>,
    }

    impl List {
        pub fn new(scratch_bytes: u32) -> Self {
            Self {
                scratch_bytes,
                commands: Vec::new(),
            }
        }

        pub fn push(&mut self, command: Command) {
            assert!(
                self.commands.len() < MAX_COMMANDS,
                "exec list holds at most {MAX_COMMANDS} commands"
            );
            self.commands.push(command);
        }

        pub fn len(&self) -> usize {
            HEADER_BYTES + self.commands.len() * COMMAND_BYTES
        }

        pub fn is_empty(&self) -> bool {
            self.commands.is_empty()
        }

        pub fn to_bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.len()];
            put_u32(&mut out, 0, MAGIC);
            put_u32(&mut out, 4, VERSION);
            put_u32(&mut out, 8, self.commands.len() as u32);
            put_u32(&mut out, 12, COMMAND_BYTES as u32);
            put_u32(&mut out, 16, self.len() as u32);
            put_u32(&mut out, 20, self.scratch_bytes);
            put_u32(&mut out, 28, HEADER_BYTES as u32);
            for (index, command) in self.commands.iter().enumerate() {
                let at = HEADER_BYTES + index * COMMAND_BYTES;
                out[at..at + COMMAND_BYTES].copy_from_slice(&command.to_bytes());
            }
            out
        }
    }
}

struct DspLink {
    executor: Arc<dyn DspExecutor>,
    handle: DspHandle,
    module: String,
}
impl Drop for DspLink {
    fn drop(&mut self) {
        // Closing is best effort: the process may already be tearing the
        // FastRPC session down.
        let _ = self.executor.close(self.handle);
    }
}

/// Hexagon backend: one FastRPC command per layer call.
pub struct DspBackend {
    spec: ModelSpec,
    link: Option<DspLink>,
    ffn: Vec<Option<FfnStage>>,
    attention: Vec<Option<AttentionStage>>,
    attention_full: Vec<Option<AttentionFullStage>>,
    conv: Vec<Option<ConvStage>>,
    conv_counters: Vec<i32>,
    conv_dumped: Vec<bool>,
    scratch_ready: bool,
    exec_micros: i64,
    exec_host_micros: i64,
    weights: Option<Box<dyn StageProvider>>,
    host: Option<HostWeights>,
    lm_head: Option<LmHeadStage>,
    last_error: Option<DspError>,
    last_exec_times: Option<[i32; 8]>,
}

const MISSING_TRANSPORT: &str =
    "FastRPC transport is not attached; use DspBackend::attach with a transport";
const MISSING_FFN_STAGE: &str =
    "FastRPC FFN weights are not staged; call DspBackend::stage_ffn first";
const MISSING_ATTENTION_STAGE: &str =
    "FastRPC attention caches are not staged; call DspBackend::stage_attention first";
const MISSING_LM_HEAD_STAGE: &str =
    "FastRPC vocabulary projection is not staged; attach a StageProvider with `lm_head_stage`";
const MISSING_HOST: &str = "host tensors are not attached; call DspBackend::attach_host first";

impl DspBackend {
    /// Adopt a validated spec without a transport.
    ///
    /// Useful for selecting the backend before Android initialises FastRPC;
    /// every layer call then fails with a typed error.
    pub fn new(spec: ModelSpec) -> Self {
        let layers = spec.dims.layers;
        Self {
            spec,
            link: None,
            ffn: vec![None; layers],
            attention: vec![None; layers],
            attention_full: vec![None; layers],
            conv: vec![None; layers],
            conv_counters: vec![0; layers],
            conv_dumped: vec![false; layers],
            scratch_ready: false,
            exec_micros: 0,
            exec_host_micros: 0,
            weights: None,
            host: None,
            lm_head: None,
            last_error: None,
            last_exec_times: None,
        }
    }

    /// Open the FastRPC module `module` and attach the executor.
    pub fn attach(
        spec: ModelSpec,
        executor: Arc<dyn DspExecutor>,
        module: &str,
    ) -> Result<Self, DspError> {
        let handle = executor.open(module)?;
        let mut backend = Self::new(spec);
        backend.link = Some(DspLink {
            executor,
            handle,
            module: module.to_owned(),
        });
        Ok(backend)
    }

    /// Adopt an already opened handle. The backend closes it on drop.
    pub fn with_handle(
        spec: ModelSpec,
        executor: Arc<dyn DspExecutor>,
        handle: DspHandle,
        module: &str,
    ) -> Self {
        let mut backend = Self::new(spec);
        backend.link = Some(DspLink {
            executor,
            handle,
            module: module.to_owned(),
        });
        backend
    }

    /// Module name the backend is attached to, if any.
    pub fn module(&self) -> Option<&str> {
        self.link.as_ref().map(|link| link.module.as_str())
    }

    /// FastRPC handle, once a transport is attached.
    ///
    /// A [`StageProvider`] built on the same executor and handle can be
    /// attached with [`DspBackend::attach_weights`].
    pub fn handle(&self) -> Option<DspHandle> {
        self.link.as_ref().map(|link| link.handle)
    }

    /// Executor the backend drives, once attached.
    pub fn executor(&self) -> Option<Arc<dyn DspExecutor>> {
        self.link.as_ref().map(|link| Arc::clone(&link.executor))
    }

    /// Attach an on-demand weight provider.
    ///
    /// Layers with an explicit [`DspBackend::stage_ffn`] or
    /// [`DspBackend::stage_attention`] keep that stage; otherwise the provider
    /// is asked on the first layer call and its answer is cached. The provider
    /// must use the same FastRPC handle as this backend (see
    /// [`DspBackend::handle`]).
    pub fn attach_weights(&mut self, provider: Box<dyn StageProvider>) {
        self.weights = Some(provider);
    }

    /// Whether an on-demand weight provider is attached.
    pub fn has_weight_provider(&self) -> bool {
        self.weights.is_some()
    }

    /// Attach the host-resident tensors: token embedding, RMSNorm weights and
    /// the material for the parity-mode vocabulary projection.
    ///
    /// `embed` and `rms_norm` need these; the DSP layer calls do not. A model
    /// file that only exercises the device graph can stay without them, in
    /// which case the two host operations fail with a typed error.
    pub fn attach_host(&mut self, host: HostWeights) -> Result<(), BackendError> {
        if host.layer_count() != self.spec.dims.layers {
            return Err(BackendError::Model(format!(
                "host weights describe {} layers, model has {}",
                host.layer_count(),
                self.spec.dims.layers
            )));
        }
        self.host = Some(host);
        Ok(())
    }

    /// Host tensors, when attached.
    pub fn host_weights(&self) -> Option<&HostWeights> {
        self.host.as_ref()
    }

    /// Mutable host tensors, for the head-mode switch.
    pub fn host_weights_mut(&mut self) -> Option<&mut HostWeights> {
        self.host.as_mut()
    }

    /// DSP microseconds spent in `exec` since the last call, summed over the
    /// command lists of one layer. Mapping the host arena is outside this
    /// counter, as are `load_buf` calls.
    pub fn take_exec_micros(&mut self) -> i64 {
        std::mem::take(&mut self.exec_micros)
    }

    /// Last transport failure, with the original FastRPC code when there is
    /// one. [`BackendError`] only carries static reasons, so this is the place
    /// to inspect the full diagnostic.
    pub fn last_error(&self) -> Option<&DspError> {
        self.last_error.as_ref()
    }

    /// DSP counters of the last successful `exec` call.
    ///
    /// `[0]` is the DSP microseconds spent in the commands, `[1]` the failing
    /// command index (`-1` on success), `[2]` the number of commands run.
    pub fn last_exec_times(&self) -> Option<&[i32; 8]> {
        self.last_exec_times.as_ref()
    }

    /// Reset the device session state: zero the CONV ring states in the DSP
    /// scratch, the DSP counters and the host-side token counters.
    ///
    /// The mapped command arenas and the resident weights stay in place, so
    /// the next token reuses the same mappings and starts from a fresh
    /// sequence. [`Session::reset`](crate::Session::reset) calls this through
    /// [`Backend::reset`].
    pub fn reset_session(&mut self) -> Result<(), BackendError> {
        self.conv_counters.fill(0);
        self.last_exec_times = None;
        let Some((executor, handle)) = self.link_parts() else {
            return Ok(());
        };
        self.ensure_scratch()?;
        // flags = 0: both the scratch (all CONV rings) and the counters.
        let prim = 0i32.to_le_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        let call = DspCall {
            method: HVX_METHOD_RESET_STATE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        };
        if let Err(error) = executor.invoke(handle, call) {
            return Err(self.fail("reset_state", error));
        }
        self.last_error = None;
        Ok(())
    }

    /// Stage resident FFN handles for `layer`.
    pub fn stage_ffn(&mut self, layer: usize, stage: FfnStage) -> Result<(), BackendError> {
        stage.validate()?;
        let layers = self.ffn.len();
        let slot = self.ffn.get_mut(layer).ok_or_else(|| {
            BackendError::Model(format!(
                "layer {layer} is outside the model ({layers} layers)"
            ))
        })?;
        *slot = Some(stage);
        Ok(())
    }

    /// Stage resident paged-attention caches for `layer`.
    pub fn stage_attention(
        &mut self,
        layer: usize,
        stage: AttentionStage,
    ) -> Result<(), BackendError> {
        stage.validate()?;
        let layers = self.attention.len();
        let slot = self.attention.get_mut(layer).ok_or_else(|| {
            BackendError::Model(format!(
                "layer {layer} is outside the model ({layers} layers)"
            ))
        })?;
        *slot = Some(stage);
        Ok(())
    }

    /// Stage a resident short-convolution stage for `layer`.
    pub fn stage_conv(&mut self, layer: usize, stage: ConvStage) -> Result<(), BackendError> {
        stage.validate()?;
        let layers = self.conv.len();
        let slot = self.conv.get_mut(layer).ok_or_else(|| {
            BackendError::Model(format!(
                "layer {layer} is outside the model ({layers} layers)"
            ))
        })?;
        *slot = Some(stage);
        Ok(())
    }

    /// Stage a fully wired attention layer (caches + projections + host-side
    /// norm/rope/gate parameters). The caches are also registered with the
    /// legacy [`DspBackend::stage_attention`] table.
    pub fn stage_attention_full(
        &mut self,
        layer: usize,
        stage: AttentionFullStage,
    ) -> Result<(), BackendError> {
        stage.validate(&self.spec.dims)?;
        let layers = self.attention_full.len();
        let slot = self.attention_full.get_mut(layer).ok_or_else(|| {
            BackendError::Model(format!(
                "layer {layer} is outside the model ({layers} layers)"
            ))
        })?;
        *slot = Some(stage.clone());
        self.attention[layer] = Some(stage.caches);
        Ok(())
    }

    fn attention_layer_preprojected(
        &mut self,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        if dims.heads != 16 || dims.head_dim != 128 {
            return Err(unsupported(
                "attention_layer",
                "FastRPC: the HVX attention stand is fixed at 16 heads x 128 dims",
            ));
        }
        if call.input.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp attention input",
                expected: dims.hidden,
                got: call.input.len(),
            });
        }
        if out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp attention output",
                expected: dims.hidden,
                got: out.len(),
            });
        }
        let Some((executor, handle)) = self.link_parts() else {
            return Err(unsupported("attention_layer", MISSING_TRANSPORT));
        };
        let stage = match self.attention.get(call.layer).copied().flatten() {
            Some(stage) => stage,
            None => self.load_attention_stage(call.layer)?,
        };
        self.last_error = None;
        if call.position >= kv.capacity() {
            return Err(BackendError::Kv(format!(
                "position {} exceeds KV capacity {}",
                call.position,
                kv.capacity()
            )));
        }
        if call.start > call.position {
            return Err(BackendError::Kv(format!(
                "attention window starts at {} after position {}",
                call.start, call.position
            )));
        }

        let seq = i32::try_from(call.position + 1)
            .map_err(|_| BackendError::Kv("position does not fit i32".to_owned()))?;
        let position = i32::try_from(call.position)
            .map_err(|_| BackendError::Kv("position does not fit i32".to_owned()))?;
        let window = match call.kind {
            AttentionKind::Full => 0,
            AttentionKind::Sliding => i32::try_from(call.position + 1 - call.start)
                .map_err(|_| BackendError::Kv("window does not fit i32".to_owned()))?,
        };
        if window < 0 {
            return Err(BackendError::Kv("attention window is negative".to_owned()));
        }
        let pages = (call.position / crate::kv::PAGE_TOKENS) + 1;
        if pages > stage.cache_pages {
            return Err(BackendError::Kv(format!(
                "attention needs {pages} pages, stage has {}",
                stage.cache_pages
            )));
        }
        let table: Vec<i32> = (0..pages)
            .map(|page| {
                i32::try_from(page)
                    .map_err(|_| BackendError::Kv("page id does not fit i32".to_owned()))
            })
            .collect::<Result<_, _>>()?;

        let q = quantize(call.input, stage.q_scale);
        let mut outp = vec![0u8; dims.q_dim() * 4];
        let mut times = [0i32; 8];

        // `_primIn[12]` of the qaic stub: handles, qLen, tblLen (long
        // elements), pos, seq, window, reps, outLen, timesLen.
        let mut prim = [0u8; 48];
        prim[0..4].copy_from_slice(&stage.keys.to_le_bytes());
        prim[4..8].copy_from_slice(&stage.key_scales.to_le_bytes());
        prim[8..12].copy_from_slice(&stage.values.to_le_bytes());
        prim[12..16].copy_from_slice(&stage.value_scales.to_le_bytes());
        prim[16..20].copy_from_slice(&id_len(q.len(), "dsp attn q")?.to_le_bytes());
        prim[20..24].copy_from_slice(&id_len(table.len(), "dsp attn tbl")?.to_le_bytes());
        prim[24..28].copy_from_slice(&position.to_le_bytes());
        prim[28..32].copy_from_slice(&seq.to_le_bytes());
        prim[32..36].copy_from_slice(&window.to_le_bytes());
        prim[36..40].copy_from_slice(&stage.reps.to_le_bytes());
        prim[40..44].copy_from_slice(&id_len(outp.len(), "dsp attn outp")?.to_le_bytes());
        prim[44..48].copy_from_slice(&id_len(times.len(), "dsp attn times")?.to_le_bytes());

        let table_bytes = i32_bytes(&table);
        let in_bufs = [q.as_slice(), table_bytes];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 2] = [outp.as_mut_slice(), i32_bytes_mut(&mut times)];
        let command = DspCall {
            method: HVX_METHOD_ATTN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &in_bufs,
            out_bufs: &mut out_bufs,
        };
        if let Err(error) = executor.invoke(handle, command) {
            return Err(self.fail("attention_layer", error));
        }

        for (index, dst) in out.iter_mut().enumerate() {
            let base = index * 4;
            let bits = [outp[base], outp[base + 1], outp[base + 2], outp[base + 3]];
            let value = i32::from_le_bytes(bits);
            *dst = value as f32 * stage.out_scale;
        }

        // The trait contract says the new token is appended by the backend.
        // Real K/V live in the resident DSP caches named by `AttentionStage`;
        // the host cache is a length/page mirror for the scheduler, so it is
        // advanced with neutral values when the caller has not done so.
        if kv.len() == call.position {
            let zeros = vec![0.0f32; kv.kv_dim()];
            kv.append(&zeros, &zeros)?;
        } else if kv.len() != call.position + 1 {
            return Err(BackendError::Kv(format!(
                "host KV length {} is not position {} or {}",
                kv.len(),
                call.position,
                call.position + 1
            )));
        }
        Ok(())
    }

    fn conv_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        if x.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp conv input",
                expected: dims.hidden,
                got: x.len(),
            });
        }
        if out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp conv output",
                expected: dims.hidden,
                got: out.len(),
            });
        }
        if dims.conv_cache > 1 {
            let expected = (dims.conv_cache - 1) * dims.hidden;
            if state.len() != expected {
                return Err(BackendError::Shape {
                    operation: "dsp conv state",
                    expected,
                    got: state.len(),
                });
            }
        }
        let Some(stage) = self.conv.get(layer).copied().flatten() else {
            let stage = self.load_conv_stage(layer)?;
            return self.conv_layer_with_stage(layer, stage, x, state, out);
        };
        self.conv_layer_with_stage(layer, stage, x, state, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_layer_with_stage(
        &mut self,
        layer: usize,
        stage: ConvStage,
        x: &[f32],
        _state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        self.last_error = None;
        let counter = *self
            .conv_counters
            .get(layer)
            .ok_or_else(|| BackendError::Model(format!("no conv counter for layer {layer}")))?;

        let hidden = dims.hidden as i32;
        let act = quantize(x, stage.act_scale);
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(exec::Command::gemv(
            stage.in_proj,
            1,
            stage.reps,
            SCRATCH_ACT,
            hidden,
            SCRATCH_BCX,
            3 * hidden,
        ));
        list.push(
            exec::Command::conv(
                stage.conv,
                hidden,
                stage.ksh,
                0,
                0,
                stage.state_off,
                SCRATCH_BCX,
                SCRATCH_CY,
            )
            .with_rt_base(0),
        );
        list.push(
            exec::Command::gemv(stage.out_proj, 1, stage.reps, SCRATCH_CY, hidden, 0, hidden)
                .with_dst(exec::dst::OUT, 0),
        );
        // The token counter is the only per-token value; keeping it out of the
        // command makes the list byte-identical between tokens, so the host
        // executor can keep one arena mapped for the whole generation.
        let result = self.run_exec(&list, &act, &[counter], dims.hidden)?;
        if result.out.len() != dims.hidden {
            return Err(unsupported(
                "conv_layer",
                "exec returned the wrong output length",
            ));
        }
        for (dst, &code) in out.iter_mut().zip(&result.out) {
            *dst = (code as i8) as f32 * stage.out_scale;
        }
        if std::env::var_os("MYELIN_DUMP_CONV").is_some() {
            let already = self.conv_dumped.get(layer).copied().unwrap_or(true);
            if !already {
                if let Some(flag) = self.conv_dumped.get_mut(layer) {
                    *flag = true;
                }
                let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
                list.push(exec::Command::dump(SCRATCH_BCX, 0, 3 * hidden));
                list.push(exec::Command::dump(SCRATCH_CY, 3 * hidden, hidden));
                match self.run_exec(&list, &[], &[], (4 * hidden) as usize) {
                  Ok(dump) => {
                    let split = (3 * hidden) as usize;
                    let _ = std::fs::write(
                        format!("/data/local/tmp/conv_bcx_{layer}.bin"),
                        &dump.out[..split],
                    );
                    let _ = std::fs::write(
                        format!("/data/local/tmp/conv_cy_{layer}.bin"),
                        &dump.out[split..],
                    );
                  }
                  Err(error) => eprintln!("conv dump failed: {error}"),
                }
            }
        }
        let counter = self
            .conv_counters
            .get_mut(layer)
            .ok_or_else(|| BackendError::Model(format!("no conv counter for layer {layer}")))?;
        *counter = counter.wrapping_add(1);
        Ok(())
    }

    fn attention_layer_full(
        &mut self,
        stage: &AttentionFullStage,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        if dims.heads != 16 || dims.head_dim != 128 || dims.kv_heads != 4 {
            return Err(unsupported(
                "attention_layer",
                "FastRPC: the HVX attention stand is fixed at 16 q-heads, 4 kv-heads x 128 dims",
            ));
        }
        if call.input.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp attention input",
                expected: dims.hidden,
                got: call.input.len(),
            });
        }
        if out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp attention output",
                expected: dims.hidden,
                got: out.len(),
            });
        }
        if call.position >= kv.capacity() {
            return Err(BackendError::Kv(format!(
                "position {} exceeds KV capacity {}",
                call.position,
                kv.capacity()
            )));
        }
        if call.start > call.position {
            return Err(BackendError::Kv(format!(
                "attention window starts at {} after position {}",
                call.start, call.position
            )));
        }
        if kv.len() != call.position {
            return Err(BackendError::Kv(format!(
                "host KV has {} tokens before position {}, expected {}",
                kv.len(),
                call.position,
                call.position
            )));
        }
        if kv.kv_dim() != dims.kv_dim() {
            return Err(BackendError::Kv(format!(
                "host KV width {} != model kv_dim {}",
                kv.kv_dim(),
                dims.kv_dim()
            )));
        }
        let pages = (call.position / crate::kv::PAGE_TOKENS) + 1;
        if pages > stage.caches.cache_pages {
            return Err(BackendError::Kv(format!(
                "attention needs {pages} pages, stage has {}",
                stage.caches.cache_pages
            )));
        }
        let Some((executor, handle)) = self.link_parts() else {
            return Err(unsupported("attention_layer", MISSING_TRANSPORT));
        };
        self.last_error = None;

        let hidden = dims.hidden;
        let kv_dim = dims.kv_dim();
        let head_dim = dims.head_dim;
        let q_dim = dims.q_dim();
        // int4 GEMV rows are padded to 32 by `myelin-pack gemv`; the kernel
        // writes `N_padded` codes, so the gate tail must be sized for that.
        let gate_rows = dims.heads.div_ceil(32) * 32;
        let gate_len = usize::from(stage.gate.is_some()) * gate_rows;

        // 1. q/k/v (and gate) projections in one list. Results travel back in
        // one `out` sequence: q | k | v | gate.
        let act = quantize(call.input, stage.act_scale);
        let q_out: usize = 0;
        let k_out = hidden;
        let v_out = hidden + kv_dim;
        let gate_out = hidden + 2 * kv_dim;
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(
            exec::Command::gemv(
                stage.query,
                1,
                stage.reps,
                SCRATCH_ACT,
                hidden as i32,
                q_out as i32,
                hidden as i32,
            )
            .with_dst(exec::dst::OUT, q_out as i32),
        );
        list.push(
            exec::Command::gemv(
                stage.key,
                1,
                stage.reps,
                SCRATCH_ACT,
                hidden as i32,
                k_out as i32,
                kv_dim as i32,
            )
            .with_dst(exec::dst::OUT, k_out as i32),
        );
        list.push(
            exec::Command::gemv(
                stage.value,
                1,
                stage.reps,
                SCRATCH_ACT,
                hidden as i32,
                v_out as i32,
                kv_dim as i32,
            )
            .with_dst(exec::dst::OUT, v_out as i32),
        );
        if let Some(gate) = stage.gate {
            list.push(
                exec::Command::gemv(
                    gate,
                    1,
                    stage.reps,
                    SCRATCH_ACT,
                    hidden as i32,
                    gate_out as i32,
                    gate_rows as i32,
                )
                .with_dst(exec::dst::OUT, gate_out as i32),
            );
        }
        let projected = self.run_exec(&list, &act, &[], hidden + 2 * kv_dim + gate_len)?;
        if projected.out.len() != hidden + 2 * kv_dim + gate_len {
            return Err(unsupported(
                "attention_layer",
                "exec returned the wrong projection length",
            ));
        }

        let mut query = dequantize(&projected.out[q_out..q_out + hidden], stage.q_scale);
        let mut key = dequantize(&projected.out[k_out..k_out + kv_dim], stage.k_scale);
        let value = dequantize(&projected.out[v_out..v_out + kv_dim], stage.v_scale);
        let gates: Option<Vec<f32>> = stage.gate.map(|_| {
            let raw = dequantize(
                &projected.out[gate_out..gate_out + dims.heads],
                stage.gate_scale,
            );
            raw.into_iter().map(crate::ops::sigmoid).collect()
        });

        // q/k RMSNorm per head, then rope, exactly like the CPU oracle.
        let rope_spec = match call.kind {
            AttentionKind::Sliding => self
                .spec
                .rope
                .sliding
                .as_ref()
                .unwrap_or(&self.spec.rope.full),
            AttentionKind::Full => &self.spec.rope.full,
        };
        let rope = crate::ops::RopeTable::new(rope_spec);
        for head in 0..dims.heads {
            let head_slice = &mut query[head * head_dim..(head + 1) * head_dim];
            norm_one_head(head_slice, &stage.query_norm, self.spec.norm_eps);
            rope.apply(head_slice, call.position);
        }
        for head in 0..dims.kv_heads {
            let head_slice = &mut key[head * head_dim..(head + 1) * head_dim];
            norm_one_head(head_slice, &stage.key_norm, self.spec.norm_eps);
            rope.apply(head_slice, call.position);
        }

        // The paged-attention kernel takes an int8 q and pre-quantised K/V.
        let q_int8 = quantize(&query, stage.attn_q_scale);
        self.write_kv_token(&executor, handle, stage, &key, &value, call.position, &dims)?;
        kv.append(&key, &value)?;

        // 2. paged attention. q and the block table travel in one activation
        // sequence: q at 0, table at `hidden`.
        let table: Vec<i32> = (0..pages)
            .map(|page| {
                i32::try_from(page)
                    .map_err(|_| BackendError::Kv("page id does not fit i32".to_owned()))
            })
            .collect::<Result<_, _>>()?;
        let table_bytes = i32_bytes(&table);
        let mut attention_act = Vec::with_capacity(q_int8.len() + table_bytes.len());
        attention_act.extend_from_slice(&q_int8);
        attention_act.extend_from_slice(table_bytes);
        let position = i32::try_from(call.position)
            .map_err(|_| BackendError::Kv("position does not fit i32".to_owned()))?;
        let seq = i32::try_from(call.position + 1)
            .map_err(|_| BackendError::Kv("position does not fit i32".to_owned()))?;
        let window = match call.kind {
            AttentionKind::Full => 0,
            AttentionKind::Sliding => i32::try_from(call.position + 1 - call.start)
                .map_err(|_| BackendError::Kv("window does not fit i32".to_owned()))?,
        };
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(
            exec::Command::attn(
                stage.caches.keys,
                stage.caches.key_scales,
                stage.caches.values,
                stage.caches.value_scales,
                -1,
                pages as i32,
                0,
                SCRATCH_ACT,
                0,
            )
            .with_dst(exec::dst::OUT, 0)
            .with_table_offset(hidden as i32),
        );
        let rt = [position, seq, window, stage.reps];
        let attended_q12 = self.run_exec(&list, &attention_act, &rt, q_dim * 4)?;
        if attended_q12.out.len() != q_dim * 4 {
            return Err(unsupported(
                "attention_layer",
                "exec returned the wrong ATTN length",
            ));
        }

        // 3. V scale, attention gate, then o_proj. The V scale is per channel
        // (128 multipliers per head), so this step is on the host.
        let mut attended = vec![0.0f32; q_dim];
        for (index, slot) in attended.iter_mut().enumerate() {
            let base = index * 4;
            let bits = [
                attended_q12.out[base],
                attended_q12.out[base + 1],
                attended_q12.out[base + 2],
                attended_q12.out[base + 3],
            ];
            let q12 = i32::from_le_bytes(bits);
            let qh = index / head_dim;
            let kh = qh / (dims.heads / dims.kv_heads);
            let dim = index % head_dim;
            let mut real =
                q12 as f32 * stage.attn_out_scale * stage.v_quant.scale(kh * head_dim + dim);
            if let Some(gates) = &gates {
                real *= gates[qh];
            }
            *slot = real;
        }
        let attended_int8 = quantize(&attended, stage.o_act_scale);
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(
            exec::Command::gemv(
                stage.output,
                1,
                stage.reps,
                SCRATCH_ACT,
                q_dim as i32,
                0,
                hidden as i32,
            )
            .with_dst(exec::dst::OUT, 0),
        );
        let result = self.run_exec(&list, &attended_int8, &[], hidden)?;
        if result.out.len() != hidden {
            return Err(unsupported(
                "attention_layer",
                "exec returned the wrong o_proj length",
            ));
        }
        for (dst, &code) in out.iter_mut().zip(&result.out) {
            *dst = (code as i8) as f32 * stage.out_scale;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_kv_token(
        &mut self,
        executor: &Arc<dyn DspExecutor>,
        handle: DspHandle,
        stage: &AttentionFullStage,
        key: &[f32],
        value: &[f32],
        position: usize,
        dims: &crate::spec::Dims,
    ) -> Result<(), BackendError> {
        let slots = stage.caches.cache_pages * crate::kv::PAGE_TOKENS;
        let slot = position % crate::kv::PAGE_TOKENS;
        let head_dim = dims.head_dim;
        let heads = dims.kv_heads;
        // One RPC carries every kv head: k (int4), kt (Q16) and v (int8) are
        // laid out head after head and scattered by the DSP at head*slots+slot.
        let mut k_bytes = Vec::with_capacity(heads * 64);
        let mut kt_bytes = Vec::with_capacity(heads * 4);
        let mut v_bytes = Vec::with_capacity(heads * 128);
        for head in 0..heads {
            let row = &key[head * head_dim..(head + 1) * head_dim];
            let mut max_abs = 0.0f32;
            for &value in row {
                max_abs = max_abs.max(value.abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
            // The kernel stores scores in Q8: `sc = dot(q8, code) * kt / 256`,
            // and the model's dot product is scaled by `1 / sqrt(head_dim)`
            // before the softmax. `q8 = round(q * attn_q_scale)`, so `kt` is
            // divided by both the q scale and `sqrt(head_dim)`; then `sc` is
            // Q8 of the exact CPU score.
            let kt =
                (scale * 65536.0 / (stage.attn_q_scale * (head_dim as f32).sqrt())).round() as i32;
            if kt > 4_194_303 {
                return Err(BackendError::Kv(format!(
                    "K scale {kt} exceeds the Q16 kernel limit (|k|/q_scale must stay below 448)"
                )));
            }
            let mut codes = [0u8; 64];
            for (dim, &value) in row.iter().enumerate() {
                let code = if scale > 0.0 {
                    (value / scale).round().clamp(-7.0, 7.0) as i32
                } else {
                    0
                };
                codes[dim / 2] |= (((code + 8) & 0x0f) as u8) << (4 * (dim % 2));
            }
            k_bytes.extend_from_slice(&codes);
            kt_bytes.extend_from_slice(&kt.to_le_bytes());

            let row = &value[head * head_dim..(head + 1) * head_dim];
            let mut v_codes = [0u8; 128];
            for (dim, &value) in row.iter().enumerate() {
                let scale = stage.v_quant.scale(head * head_dim + dim);
                let code = (value / scale).round().clamp(-127.0, 127.0) as i8;
                v_codes[dim] = code as u8;
            }
            v_bytes.extend_from_slice(&v_codes);
        }

        let slots = id_len(slots, "dsp kv slots")?;
        let slot = id_len(slot, "dsp kv slot")?;
        let mut prim = [0u8; 32];
        prim[0..4].copy_from_slice(&stage.caches.keys.to_le_bytes());
        prim[4..8].copy_from_slice(&stage.caches.key_scales.to_le_bytes());
        prim[8..12].copy_from_slice(&stage.caches.values.to_le_bytes());
        prim[12..16].copy_from_slice(&slots.to_le_bytes());
        prim[16..20].copy_from_slice(&slot.to_le_bytes());
        prim[20..24].copy_from_slice(&id_len(k_bytes.len(), "dsp kv k")?.to_le_bytes());
        prim[24..28].copy_from_slice(&id_len(kt_bytes.len(), "dsp kv kt")?.to_le_bytes());
        prim[28..32].copy_from_slice(&id_len(v_bytes.len(), "dsp kv v")?.to_le_bytes());

        let in_bufs = [k_bytes.as_slice(), kt_bytes.as_slice(), v_bytes.as_slice()];
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        let call = DspCall {
            method: HVX_METHOD_KV_WRITE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &in_bufs,
            out_bufs: &mut no_out,
        };
        if let Err(error) = executor.invoke(handle, call) {
            return Err(self.fail("kv_write", error));
        }
        Ok(())
    }

    fn ensure_scratch(&mut self) -> Result<(), BackendError> {
        if self.scratch_ready {
            return Ok(());
        }
        let Some((executor, handle)) = self.link_parts() else {
            return Err(unsupported("exec", MISSING_TRANSPORT));
        };
        match executor.reserve_scratch(handle, i64::from(EXEC_SCRATCH_BYTES)) {
            Ok(()) => {
                // A freshly reserved scratch is not guaranteed to be zeroed, and
                // the CONV ring lives in it: garbage taps on the first tokens
                // would corrupt the shortconv layers. Zero it once here.
                let prim = 0i32.to_le_bytes();
                let mut prim_out: [u8; 0] = [];
                let mut no_out: [&mut [u8]; 0] = [];
                let call = DspCall {
                    method: HVX_METHOD_RESET_STATE,
                    prim_in: &prim,
                    prim_out: &mut prim_out,
                    in_bufs: &[],
                    out_bufs: &mut no_out,
                };
                if let Err(error) = executor.invoke(handle, call) {
                    return Err(self.fail("reset_state", error));
                }
                self.scratch_ready = true;
                Ok(())
            }
            Err(error) => Err(self.fail("exec_scratch", error)),
        }
    }

    fn run_exec(
        &mut self,
        list: &exec::List,
        act: &[u8],
        rt: &[i32],
        out_len: usize,
    ) -> Result<ExecResult, BackendError> {
        self.ensure_scratch()?;
        let Some((executor, handle)) = self.link_parts() else {
            return Err(unsupported("exec", MISSING_TRANSPORT));
        };
        let encoded = list.to_bytes();
        let request = ExecRun {
            list: &encoded,
            commands: list.commands.len(),
            scratch_bytes: list.scratch_bytes,
            act,
            rt,
            out_len,
        };
        let host_started = std::time::Instant::now();
        let result = executor.exec(handle, request);
        self.exec_host_micros += host_started.elapsed().as_micros() as i64;
        match result {
            Ok(result) => {
                self.last_error = None;
                self.exec_micros += i64::from(result.times[0]);
                self.last_exec_times = Some(result.times);
                Ok(result)
            }
            Err(error) => Err(self.fail("exec", error)),
        }
    }

    fn link_parts(&self) -> Option<(Arc<dyn DspExecutor>, DspHandle)> {
        self.link
            .as_ref()
            .map(|link| (Arc::clone(&link.executor), link.handle))
    }

    fn fail(&mut self, operation: &'static str, error: DspError) -> BackendError {
        let reason = match &error {
            DspError::Unavailable(_) | DspError::Closed => "FastRPC transport is not available",
            DspError::HandleOpen { .. } => "FastRPC handle did not open",
            DspError::Rpc { .. } => "FastRPC call failed (see DspBackend::last_error)",
            DspError::Timeout { .. } => "FastRPC call timed out",
            DspError::BufferTooSmall { .. } => "FastRPC output buffer is too small",
            DspError::Other(_) => "FastRPC transport error",
        };
        self.last_error = Some(error);
        BackendError::Unsupported {
            backend: "dsp",
            operation,
            reason,
        }
    }

    fn stage_failure(&mut self, operation: &'static str, error: StageError) -> BackendError {
        match error {
            StageError::Transport(error) => self.fail(operation, error),
            StageError::Budget(message) => BackendError::Budget(message),
            StageError::Unknown(message) | StageError::Layout(message) => {
                BackendError::Model(message)
            }
        }
    }

    fn load_ffn_stage(&mut self, layer: usize) -> Result<FfnStage, BackendError> {
        let loaded = match self.weights.as_mut() {
            Some(provider) => provider.ffn_stage(layer),
            None => return Err(unsupported("ffn_layer", MISSING_FFN_STAGE)),
        };
        match loaded {
            Ok(stage) => {
                stage.validate()?;
                let layers = self.ffn.len();
                let slot = self.ffn.get_mut(layer).ok_or_else(|| {
                    BackendError::Model(format!(
                        "layer {layer} is outside the model ({layers} layers)"
                    ))
                })?;
                *slot = Some(stage);
                Ok(stage)
            }
            Err(error) => Err(self.stage_failure("ffn_layer", error)),
        }
    }

    fn load_attention_stage(&mut self, layer: usize) -> Result<AttentionStage, BackendError> {
        let loaded = match self.weights.as_mut() {
            Some(provider) => provider.attention_stage(layer),
            None => return Err(unsupported("attention_layer", MISSING_ATTENTION_STAGE)),
        };
        match loaded {
            Ok(stage) => {
                stage.validate()?;
                let layers = self.attention.len();
                let slot = self.attention.get_mut(layer).ok_or_else(|| {
                    BackendError::Model(format!(
                        "layer {layer} is outside the model ({layers} layers)"
                    ))
                })?;
                *slot = Some(stage);
                Ok(stage)
            }
            Err(error) => Err(self.stage_failure("attention_layer", error)),
        }
    }

    fn load_conv_stage(&mut self, layer: usize) -> Result<ConvStage, BackendError> {
        let loaded = match self.weights.as_mut() {
            Some(provider) => provider.conv_stage(layer),
            None => {
                return Err(unsupported(
                    "conv_layer",
                    "FastRPC conv weights are not staged; call DspBackend::stage_conv first",
                ));
            }
        };
        match loaded {
            Ok(stage) => {
                stage.validate()?;
                let layers = self.conv.len();
                let slot = self.conv.get_mut(layer).ok_or_else(|| {
                    BackendError::Model(format!(
                        "layer {layer} is outside the model ({layers} layers)"
                    ))
                })?;
                *slot = Some(stage);
                Ok(stage)
            }
            Err(error) => Err(self.stage_failure("conv_layer", error)),
        }
    }

    /// Ask the provider for a fully wired attention layer.
    ///
    /// `Ok(None)` means the provider does not stage projections, so the
    /// backend falls back to the pre-projected attention entry point.
    fn load_attention_full_stage(
        &mut self,
        layer: usize,
    ) -> Result<Option<AttentionFullStage>, BackendError> {
        let loaded = match self.weights.as_mut() {
            Some(provider) => provider.attention_full_stage(layer),
            None => return Ok(None),
        };
        match loaded {
            Ok(stage) => {
                stage.validate(&self.spec.dims)?;
                let layers = self.attention_full.len();
                let slot = self.attention_full.get_mut(layer).ok_or_else(|| {
                    BackendError::Model(format!(
                        "layer {layer} is outside the model ({layers} layers)"
                    ))
                })?;
                *slot = Some(stage.clone());
                self.attention[layer] = Some(stage.caches);
                Ok(Some(stage))
            }
            Err(StageError::Unknown(_)) => Ok(None),
            Err(error) => Err(self.stage_failure("attention_layer", error)),
        }
    }

    fn load_lm_head_stage(&mut self) -> Result<LmHeadStage, BackendError> {
        if let Some(stage) = self.lm_head {
            return Ok(stage);
        }
        let loaded = match self.weights.as_mut() {
            Some(provider) => provider.lm_head_stage(),
            None => return Err(unsupported("lm_head", MISSING_LM_HEAD_STAGE)),
        };
        match loaded {
            Ok(Some(stage)) => {
                stage.validate()?;
                self.lm_head = Some(stage);
                Ok(stage)
            }
            Ok(None) => Err(unsupported("lm_head", MISSING_LM_HEAD_STAGE)),
            Err(error) => Err(self.stage_failure("lm_head", error)),
        }
    }
}

impl Drop for DspBackend {
    fn drop(&mut self) {
        if std::env::var_os("MYELIN_TRACE").is_some() {
            eprintln!(
                "dsp exec totals: dsp {:.1} ms, host {:.1} ms",
                self.exec_micros as f64 / 1000.0,
                self.exec_host_micros as f64 / 1000.0
            );
        }
        // Resident buffers must go back before `DspLink` closes the handle.
        if let Some(provider) = self.weights.as_mut() {
            provider.release_all();
        }
    }
}

fn unsupported(operation: &'static str, reason: &'static str) -> BackendError {
    BackendError::Unsupported {
        backend: "dsp",
        operation,
        reason,
    }
}

/// Length of an IDL sequence (`long`/`octet`, both fit a 32-bit count).
fn id_len(count: usize, what: &'static str) -> Result<i32, BackendError> {
    i32::try_from(count).map_err(|_| BackendError::Shape {
        operation: what,
        expected: i32::MAX as usize,
        got: count,
    })
}

fn quantize(x: &[f32], scale: f32) -> Vec<u8> {
    x.iter()
        .map(|value| {
            let quantized = (value * scale).round().clamp(-127.0, 127.0);
            (quantized as i8) as u8
        })
        .collect()
}

fn dequantize(bytes: &[u8], scale: f32) -> Vec<f32> {
    bytes
        .iter()
        .map(|&byte| (byte as i8) as f32 * scale)
        .collect()
}

/// RMSNorm of one head vector, the same f64 accumulation as the CPU oracle.
fn norm_one_head(vector: &mut [f32], weight: &[f32], eps: f32) {
    let sum: f64 = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let mean = sum / vector.len() as f64;
    let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;
    for (value, weight) in vector.iter_mut().zip(weight) {
        *value = *value * scale * weight;
    }
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is plain old data and the slice keeps its lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn i32_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    let len = std::mem::size_of_val(values);
    // SAFETY: `i32` is plain old data and the slice keeps its lifetime.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len) }
}

impl Backend for DspBackend {
    fn name(&self) -> &'static str {
        "dsp"
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn embed(&mut self, token: u32, out: &mut [f32]) -> Result<(), BackendError> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| unsupported("embed", MISSING_HOST))?;
        host.embed(token, out)
    }

    fn rms_norm(
        &mut self,
        weights: NormWeights,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| unsupported("rms_norm", MISSING_HOST))?;
        host.norm(weights, x, out)
    }

    fn ffn_layer(&mut self, layer: usize, x: &[f32], out: &mut [f32]) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        if x.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp ffn input",
                expected: dims.hidden,
                got: x.len(),
            });
        }
        if out.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp ffn output",
                expected: dims.hidden,
                got: out.len(),
            });
        }
        let stage = match self.ffn.get(layer).copied().flatten() {
            Some(stage) => stage,
            None => self.load_ffn_stage(layer)?,
        };
        self.last_error = None;

        // The FFN command runs the same `ffn_impl` as the one-RPC `ffn`
        // method; only the transport differs. Same act, same int8 dst, so the
        // result is bit-identical to the old path (exec_test checks that
        // directly on the device).
        let act = quantize(x, stage.act_scale);
        let hidden = dims.hidden as i32;
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(
            exec::Command::ffn(
                stage.gate,
                stage.up,
                stage.down,
                stage.reps,
                SCRATCH_ACT,
                hidden,
                0,
                hidden,
            )
            .with_dst(exec::dst::OUT, 0),
        );
        let result = self.run_exec(&list, &act, &[], dims.hidden)?;
        if result.out.len() != dims.hidden {
            return Err(unsupported(
                "ffn_layer",
                "exec returned the wrong output length",
            ));
        }

        for (dst_value, quantized) in out.iter_mut().zip(&result.out) {
            *dst_value = (*quantized as i8) as f32 * stage.out_scale;
        }
        Ok(())
    }

    fn attention_layer(
        &mut self,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if let Some(stage) = self.attention_full.get(call.layer).cloned().flatten() {
            return self.attention_layer_full(&stage, call, kv, out);
        }
        if self.weights.is_some() {
            if let Some(stage) = self.load_attention_full_stage(call.layer)? {
                return self.attention_layer_full(&stage, call, kv, out);
            }
        }
        self.attention_layer_preprojected(call, kv, out)
    }

    fn conv_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        DspBackend::conv_layer(self, layer, x, state, out)
    }

    fn lm_head(&mut self, x: &[f32], logits: &mut [f32]) -> Result<(), BackendError> {
        let dims = self.spec.dims;
        if x.len() != dims.hidden {
            return Err(BackendError::Shape {
                operation: "dsp lm_head input",
                expected: dims.hidden,
                got: x.len(),
            });
        }
        if logits.len() != dims.vocab {
            return Err(BackendError::Shape {
                operation: "dsp lm_head logits",
                expected: dims.vocab,
                got: logits.len(),
            });
        }
        if self
            .host
            .as_ref()
            .is_some_and(|host| host.head_mode() == super::host::HeadMode::Host)
        {
            let host = self.host.as_ref().expect("host mode implies host weights");
            host.lm_head(x, logits)?;
            self.last_error = None;
            return Ok(());
        }
        let stage = self.load_lm_head_stage()?;
        self.last_error = None;
        let act = quantize(x, 1.0 / stage.input_scale);
        let mut list = exec::List::new(EXEC_SCRATCH_BYTES);
        list.push(
            exec::Command::gemv(
                stage.gemv,
                1,
                stage.reps,
                SCRATCH_ACT,
                dims.hidden as i32,
                0,
                stage.rows as i32,
            )
            .with_dst(exec::dst::OUT, 0),
        );
        let result = self.run_exec(&list, &act, &[], stage.rows)?;
        if result.out.len() != stage.rows {
            return Err(unsupported(
                "lm_head",
                "exec returned the wrong logits length",
            ));
        }
        for (dst, &code) in logits.iter_mut().zip(&result.out) {
            *dst = (code as i8) as f32 * stage.output_scale;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        self.reset_session()
    }

    fn diagnostic(&self) -> Option<String> {
        let error = self.last_error.as_ref()?;
        let times = self
            .last_exec_times
            .map(|times| format!("{times:?}"))
            .unwrap_or_else(|| "no times".to_owned());
        Some(format!("{error}; exec times {times}"))
    }
}

#[cfg(test)]
mod exec_encoder_tests {
    use super::exec::{Command, List, Op};

    #[test]
    fn ffn_command_packs_handles_and_lengths() {
        let bytes = Command::ffn(11, 12, 13, 2, 0, 2_048, 2_048, 2_048).to_bytes();
        assert_eq!(&bytes[0..4], &(Op::Ffn as u32).to_le_bytes());
        assert_eq!(&bytes[16..20], &11i32.to_le_bytes());
        assert_eq!(&bytes[20..24], &12i32.to_le_bytes());
        assert_eq!(&bytes[24..28], &13i32.to_le_bytes());
        assert_eq!(&bytes[28..32], &(-1i32).to_le_bytes());
        assert_eq!(&bytes[40..44], &2_048i32.to_le_bytes());
        assert_eq!(&bytes[48..52], &2_048i32.to_le_bytes());
        assert_eq!(&bytes[64..68], &2i32.to_le_bytes());
        assert_eq!(&bytes[80..84], &(-1i32).to_le_bytes());
    }

    #[test]
    fn conv_counter_travels_in_rt_to_keep_the_list_stable() {
        let first = Command::conv(9, 2_048, 40, 0, 0, 34816, 0, 8192).with_rt_base(0);
        let second = Command::conv(9, 2_048, 40, 0, 0, 34816, 0, 8192).with_rt_base(0);
        // The token counter is no longer a command field, so the bytes of the
        // whole list are identical between tokens.
        assert_eq!(first.to_bytes(), second.to_bytes());
        assert_eq!(&first.to_bytes()[76..80], &0i32.to_le_bytes());
        assert_eq!(&first.to_bytes()[80..84], &0i32.to_le_bytes());

        let mut list = List::new(4 << 20);
        list.push(first);
        assert_eq!(list.to_bytes(), list.to_bytes());
    }
}
