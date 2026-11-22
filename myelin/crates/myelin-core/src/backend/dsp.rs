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

use super::{AttentionCall, Backend, BackendError, NormWeights};
use crate::kv::KvCache;
use crate::spec::{AttentionKind, ModelSpec};

/// Method ID of `lfmgemv.ffn` in the HVX FastRPC skeleton.
///
/// Kept in sync with `myelin-npu`'s method table by a test in that crate.
pub const HVX_METHOD_FFN: u32 = 6;
/// Method ID of `lfmgemv.attn` in the HVX FastRPC skeleton.
pub const HVX_METHOD_ATTN: u32 = 11;

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
    weights: Option<Box<dyn StageProvider>>,
    last_error: Option<DspError>,
}

const MISSING_TRANSPORT: &str =
    "FastRPC transport is not attached; use DspBackend::attach with a transport";
const MISSING_FFN_STAGE: &str =
    "FastRPC FFN weights are not staged; call DspBackend::stage_ffn first";
const MISSING_ATTENTION_STAGE: &str =
    "FastRPC attention caches are not staged; call DspBackend::stage_attention first";

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
            weights: None,
            last_error: None,
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

    /// Last transport failure, with the original FastRPC code when there is
    /// one. [`BackendError`] only carries static reasons, so this is the place
    /// to inspect the full diagnostic.
    pub fn last_error(&self) -> Option<&DspError> {
        self.last_error.as_ref()
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
}

impl Drop for DspBackend {
    fn drop(&mut self) {
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

    fn embed(&mut self, _token: u32, _out: &mut [f32]) -> Result<(), BackendError> {
        Err(unsupported(
            "embed",
            "FastRPC: embedding lookup is not staged as a resident DSP buffer",
        ))
    }

    fn rms_norm(
        &mut self,
        _weights: NormWeights,
        _x: &[f32],
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported(
            "rms_norm",
            "FastRPC: resident RMSNorm weights are not staged on the DSP",
        ))
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
        let Some((executor, handle)) = self.link_parts() else {
            return Err(unsupported("ffn_layer", MISSING_TRANSPORT));
        };
        let stage = match self.ffn.get(layer).copied().flatten() {
            Some(stage) => stage,
            None => self.load_ffn_stage(layer)?,
        };
        self.last_error = None;

        let act = quantize(x, stage.act_scale);
        let mut dst = vec![0u8; dims.hidden];
        let mut mid = vec![0u8; dims.ffn];
        let mut times = [0i32; 8];

        // `_primIn[8]` of the qaic stub: handles, actLen, reps, dstLen,
        // midLen, timesLen. Sequence lengths are part of the scalar pack.
        let mut prim = [0u8; 32];
        prim[0..4].copy_from_slice(&stage.gate.to_le_bytes());
        prim[4..8].copy_from_slice(&stage.up.to_le_bytes());
        prim[8..12].copy_from_slice(&stage.down.to_le_bytes());
        prim[12..16].copy_from_slice(&id_len(act.len(), "dsp act")?.to_le_bytes());
        prim[16..20].copy_from_slice(&stage.reps.to_le_bytes());
        prim[20..24].copy_from_slice(&id_len(dst.len(), "dsp dst")?.to_le_bytes());
        prim[24..28].copy_from_slice(&id_len(mid.len(), "dsp mid")?.to_le_bytes());
        prim[28..32].copy_from_slice(&id_len(times.len(), "dsp times")?.to_le_bytes());

        let in_bufs = [act.as_slice()];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 3] = [
            dst.as_mut_slice(),
            mid.as_mut_slice(),
            i32_bytes_mut(&mut times),
        ];
        let call = DspCall {
            method: HVX_METHOD_FFN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &in_bufs,
            out_bufs: &mut out_bufs,
        };
        if let Err(error) = executor.invoke(handle, call) {
            return Err(self.fail("ffn_layer", error));
        }

        for (dst_value, quantized) in out.iter_mut().zip(&dst) {
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
        _layer: usize,
        _x: &[f32],
        _state: &mut [f32],
        _out: &mut [f32],
    ) -> Result<(), BackendError> {
        Err(unsupported(
            "conv_layer",
            "FastRPC: short-convolution kernels are not part of the HVX stand",
        ))
    }

    fn lm_head(&mut self, _x: &[f32], _logits: &mut [f32]) -> Result<(), BackendError> {
        Err(unsupported(
            "lm_head",
            "FastRPC: the vocabulary projection is not staged as a resident GEMV",
        ))
    }
}
