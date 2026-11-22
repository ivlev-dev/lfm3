//! Resident weight planner for the DSP heap.
//!
//! The planner owns the `alloc_buf`/`set_shape`/`load_buf`/`set_scale`/
//! `release_buf` sequence for weights packed by `myelin-pack`, keeps an exact
//! byte budget, and evicts whole layer stages in LRU order when the budget is
//! smaller than the model.
//!
//! # Call order
//!
//! The order is transcribed from the HVX stand, not invented:
//!
//! * `myelin/dsp/hvx/ffn_main.c:67-81` (`upload_matrix`): `alloc_buf`,
//!   `set_shape`, `load_buf` in 256 KiB chunks, then `set_scale`. The FFN stage
//!   repeats it for gate, up and down (`ffn_main.c:142-146`).
//! * `myelin/dsp/hvx/host_main.c:123-136` and `:201-211`: the same order for
//!   a single GEMV matrix.
//! * `myelin/dsp/hvx/attn_main.c:225-241`: three allocations (K, KT, V), then
//!   loads in K, V, KT order. Attention caches are raw images, not packed
//!   weight matrices, and the stand issues no `set_scale` for them.
//!
//! `alloc_buf` takes only a byte count in the IDL; there is no alignment
//! argument. The packed blob already starts its weight tiles at the 128-byte
//! `weight_base`, which is what the HMX core needs.
//!
//! # Budget
//!
//! One unit is one layer stage (`FFN` or `attention`). Its size is the sum of
//! the uploaded buffer lengths. A unit larger than the whole budget is
//! rejected before anything is allocated; otherwise least-recently-used units
//! are released until the new one fits. `Drop` (and `DspBackend`'s drop)
//! releases everything that is still resident.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use myelin_core::backend::dsp::{
    AttentionFullStage, AttentionStage, ConvStage, DspCall, DspError, DspExecutor, DspHandle,
    FfnStage, StageError, StageProvider, VQuant,
};
use myelin_core::backend::LmHeadStage;
use myelin_pack::{Manifest, PackResult};

use crate::method::{
    method_id, octet_len, AllocBufScalars, AllocInScalars, LoadBufWire, ReleaseBufScalars,
    SetScaleWire, SetShapeScalars,
};
use crate::transport::{SharedBuffer, Transport};

/// Default `load_buf` chunk size.
///
/// The stand used 256 KiB (183 RPCs per FFN layer, ~106 ms). On the device a
/// 4 MiB chunk takes ~6.8 ms, so the same layer needs ~12 calls (~79 ms); the
/// win comes only at >= 4 MiB, smaller chunks are dominated by per-RPC cost.
pub const LOAD_CHUNK: usize = 4 * 1024 * 1024;

/// How resident weights reach the DSP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadStrategy {
    /// `alloc_buf` + `load_buf` chunks + `set_scale`: the weights are copied
    /// into the DSP heap by the PD.
    #[default]
    Copy,
    /// Zero-copy: the weights stay in a host ION buffer that the DSP maps
    /// directly, so neither the host marshaller nor the PD copies them.
    ///
    /// One packed tensor becomes one `rpcmem` allocation mapped with
    /// `alloc_in`; the planner keeps the [`SharedBuffer`] alive until the
    /// handle is released. This is the strategy a full model uses: the DSP
    /// heap only holds scales and scratch, and 200-ish tensors stay inside the
    /// 128-byte-aligned ION buffers the kernel reads in place.
    Attach,
}

impl LoadStrategy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Attach => "attach",
        }
    }
}

impl fmt::Display for LoadStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// DSP heap budget for resident weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightBudget {
    bytes: usize,
}

impl WeightBudget {
    /// Budget of `bytes` resident bytes. Zero means "nothing fits".
    pub const fn new(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Budget in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

impl From<usize> for WeightBudget {
    fn from(bytes: usize) -> Self {
        Self::new(bytes)
    }
}

/// What kind of layer stage a unit holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKind {
    /// Gate, up and down FFN matrices.
    Ffn,
    /// Resident paged-attention caches.
    Attention,
    /// Short-convolution projections and the rotated tap blob.
    Conv,
    /// Attention projections plus the resident K/KT/V caches.
    AttentionFull,
    /// The tied vocabulary projection.
    LmHead,
}

impl StageKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::Attention => "attention",
            Self::Conv => "conv",
            Self::AttentionFull => "attention_full",
            Self::LmHead => "lm_head",
        }
    }
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Failure of the resident weight planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    /// The unit does not fit, even with every other unit evicted.
    Budget {
        layer: usize,
        kind: StageKind,
        bytes: usize,
        budget: usize,
        used: usize,
    },
    /// The plan has no unit of this kind for the layer.
    Missing { layer: usize, kind: StageKind },
    /// Manifest, blob or plan is inconsistent.
    Layout(String),
    /// A size does not fit the `long` fields of the IDL.
    TooLarge { what: &'static str, bytes: usize },
    /// A FastRPC call failed; the original code is preserved.
    Transport(DspError),
    /// The selected upload strategy is not implemented by the skeleton.
    UnsupportedStrategy {
        strategy: LoadStrategy,
        reason: &'static str,
    },
    /// The whole plan does not fit the ION budget.
    PlanOverBudget { bytes: usize, budget: usize },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget {
                layer,
                kind,
                bytes,
                budget,
                used,
            } => write!(
                f,
                "layer {layer} {kind} needs {bytes} bytes, budget {budget} bytes \
                 ({used} bytes already resident)"
            ),
            Self::Missing { layer, kind } => {
                write!(f, "layer {layer} has no {kind} plan")
            }
            Self::Layout(message) => f.write_str(message),
            Self::TooLarge { what, bytes } => {
                write!(
                    f,
                    "{what} is {bytes} bytes, does not fit a 32-bit IDL field"
                )
            }
            Self::Transport(error) => write!(f, "{error}"),
            Self::UnsupportedStrategy { strategy, reason } => {
                write!(f, "strategy `{strategy}` is unsupported: {reason}")
            }
            Self::PlanOverBudget { bytes, budget } => write!(
                f,
                "resident plan needs {bytes} bytes, ION budget is {budget} bytes"
            ),
        }
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WeightError> for StageError {
    fn from(error: WeightError) -> Self {
        match &error {
            WeightError::Transport(error) => Self::Transport(error.clone()),
            WeightError::Budget { .. } | WeightError::PlanOverBudget { .. } => {
                Self::Budget(error.to_string())
            }
            WeightError::Missing { .. } => Self::Unknown(error.to_string()),
            WeightError::Layout(_)
            | WeightError::TooLarge { .. }
            | WeightError::UnsupportedStrategy { .. } => Self::Layout(error.to_string()),
        }
    }
}

/// Snapshot of planner counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightStats {
    /// Number of units loaded from scratch.
    pub loads: u64,
    /// Number of units evicted to make room.
    pub evictions: u64,
    /// Number of units released (evictions plus explicit/drop releases).
    pub unloads: u64,
    /// Number of successful `load_buf` RPCs.
    pub load_calls: u64,
    /// Bytes handed to `load_buf`.
    pub load_bytes: u64,
    /// Host wall-clock time spent inside `load_buf` calls.
    pub load_micros: u64,
    /// Chunk size `load_buf` used.
    pub chunk: usize,
    /// Upload strategy.
    pub strategy: LoadStrategy,
    /// Units currently resident.
    pub resident_units: usize,
    /// Bytes currently resident.
    pub live_bytes: usize,
    /// Configured budget in bytes.
    pub budget: usize,
}

/// One raw resident buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentImage {
    label: String,
    bytes: Vec<u8>,
}

impl ResidentImage {
    /// Wrap `bytes` under a diagnostic label.
    pub fn new(label: impl Into<String>, bytes: Vec<u8>) -> Result<Self, WeightError> {
        if bytes.is_empty() {
            return Err(WeightError::Layout(format!(
                "resident image `{}` is empty",
                label.into()
            )));
        }
        Ok(Self {
            label: label.into(),
            bytes,
        })
    }

    /// Diagnostic label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Bytes uploaded by `load_buf`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Length of the resident allocation.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Always `false`: [`ResidentImage::new`] rejects empty images.
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// One matrix packed by `myelin-pack`.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    name: String,
    manifest: Manifest,
    blob: Vec<u8>,
}

impl PackedTensor {
    /// Adopt a manifest and its blob after checking that they agree.
    pub fn new(
        name: impl Into<String>,
        manifest: Manifest,
        blob: Vec<u8>,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        manifest
            .verify_blob(&blob)
            .map_err(|error| WeightError::Layout(format!("{name}: {error}")))?;
        Ok(Self {
            name,
            manifest,
            blob,
        })
    }

    /// Adopt a `myelin-pack` result, cloning its manifest and blob.
    pub fn from_pack(name: impl Into<String>, result: &PackResult) -> Result<Self, WeightError> {
        Self::new(name, result.manifest.clone(), result.blob.clone())
    }

    /// Tensor name used in diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Manifest that describes the blob.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Packed blob, uploaded unchanged.
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Bytes this matrix occupies on the DSP heap.
    pub fn len(&self) -> usize {
        self.blob.len()
    }

    /// Always `false`: a packed blob always carries a header.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Number of `load_buf` RPCs needed at [`LOAD_CHUNK`].
    pub fn load_calls(&self) -> usize {
        self.load_calls_with(LOAD_CHUNK)
    }

    /// Number of `load_buf` RPCs at `chunk` bytes per call.
    pub fn load_calls_with(&self, chunk: usize) -> usize {
        if chunk == 0 {
            return 0;
        }
        self.blob.len().div_ceil(chunk)
    }
}

/// One int4 matrix reduced to what the planner uploads.
///
/// [`PackedTensor`] keeps a `myelin-pack` crownstone blob, while production
/// int4 weights come from [`crate::gemv_weights::PackedGemv`]. Both answer the
/// same three questions — padded shape, weight bytes the kernel reads, and the
/// per-row f32 scales `set_scale` expects — so the planner works on this
/// normalised form.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedMatrix {
    name: String,
    n_padded: usize,
    k_padded: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
}

impl PackedMatrix {
    /// Adopt already separated weight bytes and per-row scales.
    pub fn new(
        name: impl Into<String>,
        n_padded: usize,
        k_padded: usize,
        weights: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        if n_padded == 0 || k_padded == 0 || weights.is_empty() {
            return Err(WeightError::Layout(format!(
                "{name}: empty matrix {n_padded}x{k_padded}"
            )));
        }
        if scales.len() != n_padded {
            return Err(WeightError::Layout(format!(
                "{name}: {} scales for {n_padded} padded rows",
                scales.len()
            )));
        }
        Ok(Self {
            name,
            n_padded,
            k_padded,
            weights,
            scales,
        })
    }

    /// Wrap a crownstone [`PackedTensor`] unchanged.
    pub fn from_tensor(tensor: PackedTensor) -> Result<Self, WeightError> {
        let manifest = tensor.manifest().clone();
        let name = tensor.name().to_owned();
        let scales = manifest.scales.scale_f32.clone();
        Self::new(
            name,
            manifest.n_padded,
            manifest.k_padded,
            tensor.blob().to_vec(),
            scales,
        )
    }

    /// Take the weight section of an int4 GEMV blob.
    pub fn from_gemv(name: impl Into<String>, packed: &crate::gemv_weights::PackedGemv) -> Self {
        let header = packed.header();
        let manifest = packed.manifest();
        let weights =
            packed.blob()[header.weight_base..header.weight_base + header.weight_bytes].to_vec();
        Self {
            name: name.into(),
            n_padded: manifest.n_padded,
            k_padded: manifest.k_padded,
            weights,
            scales: manifest.scales_f32.clone(),
        }
    }

    /// Diagnostic name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Padded output rows.
    pub const fn n_padded(&self) -> usize {
        self.n_padded
    }

    /// Padded input columns.
    pub const fn k_padded(&self) -> usize {
        self.k_padded
    }

    /// Weight bytes the GEMV kernel reads.
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    /// Exact per-row int4 scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Resident bytes this matrix needs.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Always `false`: a matrix has at least one row.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Number of `load_buf` RPCs needed at [`LOAD_CHUNK`].
    pub fn load_calls(&self) -> usize {
        self.load_calls_with(LOAD_CHUNK)
    }

    /// Number of `load_buf` RPCs at `chunk` bytes per call.
    pub fn load_calls_with(&self, chunk: usize) -> usize {
        if chunk == 0 {
            return 0;
        }
        self.weights.len().div_ceil(chunk)
    }
}

/// Resident FFN matrices and the three activation scales the DSP pipeline
/// needs.
///
/// The one-RPC FFN carries a scale protocol that is not obvious from the
/// matrix upload alone (`myelin/dsp/hvx/ffn_main.c`):
///
/// * gate/up are uploaded with `sa = s_in * 65536`, `so = 1`: the dequantiser
///   keeps the Q16 gate/up value and SiLU consumes it in Q16;
/// * down is uploaded with `sa = s_mid`, `so = s_out`, so its input is the
///   int8 mid buffer produced by `requant_mid`.
///
/// The host multiplies the layer input by `1 / s_in` before the RPC and
/// multiplies the int8 result by `s_out`.
#[derive(Debug, Clone)]
pub struct FfnPlan {
    gate: PackedMatrix,
    up: PackedMatrix,
    down: PackedMatrix,
    input_scale: f32,
    mid_scale: f32,
    out_scale: f32,
    reps: i32,
}

impl FfnPlan {
    /// Build a layer plan from the three packed matrices and their calibrated
    /// activation scales (`s_in`, `s_mid`, `s_out`, all in real units per
    /// int8 code).
    ///
    /// Shapes must compose: `down` is `[gate.n, gate.k]`.
    pub fn new(
        gate: PackedMatrix,
        up: PackedMatrix,
        down: PackedMatrix,
        input_scale: f32,
        mid_scale: f32,
        out_scale: f32,
        reps: i32,
    ) -> Result<Self, WeightError> {
        check_scale("ffn input_scale", input_scale)?;
        check_scale("ffn mid_scale", mid_scale)?;
        check_scale("ffn out_scale", out_scale)?;
        if reps < 1 {
            return Err(WeightError::Layout(format!(
                "ffn reps must be positive, got {reps}"
            )));
        }
        if gate.k_padded() != up.k_padded() || gate.n_padded() != up.n_padded() {
            return Err(WeightError::Layout(format!(
                "ffn gate is {}x{}, up is {}x{}",
                gate.k_padded(),
                gate.n_padded(),
                up.k_padded(),
                up.n_padded()
            )));
        }
        if down.k_padded() != gate.n_padded() || down.n_padded() != gate.k_padded() {
            return Err(WeightError::Layout(format!(
                "ffn down is {}x{}, expected {}x{}",
                down.k_padded(),
                down.n_padded(),
                gate.n_padded(),
                gate.k_padded()
            )));
        }
        Ok(Self {
            gate,
            up,
            down,
            input_scale,
            mid_scale,
            out_scale,
            reps,
        })
    }

    /// Gate matrix.
    pub fn gate(&self) -> &PackedMatrix {
        &self.gate
    }

    /// Up matrix.
    pub fn up(&self) -> &PackedMatrix {
        &self.up
    }

    /// Down matrix.
    pub fn down(&self) -> &PackedMatrix {
        &self.down
    }

    /// `s_in`: real units per int8 code of the layer input.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// `s_mid`: real units per int8 code of the SiLU(gate)*up buffer.
    pub fn mid_scale(&self) -> f32 {
        self.mid_scale
    }

    /// `s_out`: real units per int8 code of the layer output.
    pub fn out_scale(&self) -> f32 {
        self.out_scale
    }

    /// Host multiplier applied to the f32 activation before quantisation.
    pub fn act_multiplier(&self) -> f32 {
        1.0 / self.input_scale
    }

    /// Repetitions executed inside one RPC.
    pub fn reps(&self) -> i32 {
        self.reps
    }

    /// Bytes for all three matrices.
    pub fn bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

/// Attention geometry and quantisation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionParams {
    /// Multiplier applied to the f32 query before int8 quantisation.
    pub q_scale: f32,
    /// Multiplier applied to the int32 Q12 result.
    pub out_scale: f32,
    /// Repetitions executed inside one RPC.
    pub reps: i32,
    /// Number of 32-token pages the resident caches hold.
    pub cache_pages: usize,
    /// The caller has already projected q/k/v and filled the caches.
    pub preprojected: bool,
}

/// Resident paged-attention caches for one layer.
#[derive(Debug, Clone)]
pub struct AttentionPlan {
    keys: ResidentImage,
    key_scales: ResidentImage,
    values: ResidentImage,
    value_scales: Option<ResidentImage>,
    params: AttentionParams,
}

impl AttentionPlan {
    /// Build a layer plan from the three stand buffers and optional value
    /// scales. The stand passes handle `0` for `hvs`, so value scales are
    /// optional here too.
    pub fn new(
        keys: ResidentImage,
        key_scales: ResidentImage,
        values: ResidentImage,
        value_scales: Option<ResidentImage>,
        params: AttentionParams,
    ) -> Result<Self, WeightError> {
        check_scale("attention q_scale", params.q_scale)?;
        check_scale("attention out_scale", params.out_scale)?;
        if params.reps < 1 {
            return Err(WeightError::Layout(format!(
                "attention reps must be positive, got {}",
                params.reps
            )));
        }
        if params.cache_pages == 0 {
            return Err(WeightError::Layout(
                "attention cache_pages must be positive".to_owned(),
            ));
        }
        Ok(Self {
            keys,
            key_scales,
            values,
            value_scales,
            params,
        })
    }

    /// Key cache image (`hk`).
    pub fn keys(&self) -> &ResidentImage {
        &self.keys
    }

    /// Per-token key scales image (`hkt`).
    pub fn key_scales(&self) -> &ResidentImage {
        &self.key_scales
    }

    /// Value cache image (`hv`).
    pub fn values(&self) -> &ResidentImage {
        &self.values
    }

    /// Optional value scales image (`hvs`).
    pub fn value_scales(&self) -> Option<&ResidentImage> {
        self.value_scales.as_ref()
    }

    /// Attention parameters.
    pub const fn params(&self) -> AttentionParams {
        self.params
    }

    /// Bytes for every resident cache.
    pub fn bytes(&self) -> usize {
        self.keys.len()
            + self.key_scales.len()
            + self.values.len()
            + self.value_scales.as_ref().map_or(0, ResidentImage::len)
    }

    /// Number of resident cache images (3, or 4 with value scales).
    pub fn image_count(&self) -> usize {
        3 + usize::from(self.value_scales.is_some())
    }
}

/// Resident short-convolution stage of one layer.
///
/// `in_proj` maps `hidden -> 3 * hidden`, `conv` is the rotated-tap blob built
/// by `myelin-pack shortconv`, and `out_proj` maps back to `hidden`. The four
/// scales are the calibrated quantisation steps of the one-RPC chain:
/// `input_scale` (`s_in`), `mix_scale` (`s_bc`), `y_scale` (`s_y`) and
/// `out_scale` (`s_out`).
#[derive(Debug, Clone)]
pub struct ConvPlan {
    in_proj: PackedMatrix,
    conv: ResidentImage,
    out_proj: PackedMatrix,
    input_scale: f32,
    mix_scale: f32,
    y_scale: f32,
    out_scale: f32,
    ksh: i32,
    reps: i32,
    state_off: i32,
}

impl ConvPlan {
    /// Build a conv stage plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_proj: PackedMatrix,
        conv: ResidentImage,
        out_proj: PackedMatrix,
        input_scale: f32,
        mix_scale: f32,
        y_scale: f32,
        out_scale: f32,
        ksh: i32,
        reps: i32,
        state_off: i32,
    ) -> Result<Self, WeightError> {
        for (name, scale) in [
            ("input_scale", input_scale),
            ("mix_scale", mix_scale),
            ("y_scale", y_scale),
            ("out_scale", out_scale),
        ] {
            check_scale(&format!("conv {name}"), scale)?;
        }
        if !(1..=40).contains(&ksh) {
            return Err(WeightError::Layout(format!(
                "conv ksh must be 1..=40, got {ksh}"
            )));
        }
        if reps < 1 || state_off < 0 {
            return Err(WeightError::Layout(format!(
                "conv reps/state_off must be positive, got {reps}/{state_off}"
            )));
        }
        if in_proj.n_padded() % 3 != 0 {
            return Err(WeightError::Layout(format!(
                "conv in_proj has {} padded rows, not a multiple of 3",
                in_proj.n_padded()
            )));
        }
        if out_proj.n_padded() != in_proj.n_padded() / 3
            || out_proj.k_padded() != in_proj.k_padded()
        {
            return Err(WeightError::Layout(format!(
                "conv out_proj is {}x{}, expected {}x{}",
                out_proj.n_padded(),
                out_proj.k_padded(),
                in_proj.n_padded() / 3,
                in_proj.k_padded()
            )));
        }
        Ok(Self {
            in_proj,
            conv,
            out_proj,
            input_scale,
            mix_scale,
            y_scale,
            out_scale,
            ksh,
            reps,
            state_off,
        })
    }

    /// Bytes of the three resident buffers.
    pub fn bytes(&self) -> usize {
        self.in_proj.len() + self.conv.len() + self.out_proj.len()
    }
}

/// Resident attention layer: paged caches, five int4 projections and the
/// host-side post-processing parameters (q/k norm, rope is model-level, V
/// quantisation, head gate).
#[derive(Debug, Clone)]
pub struct AttentionFullPlan {
    caches: AttentionPlan,
    query: PackedMatrix,
    key: PackedMatrix,
    value: PackedMatrix,
    output: PackedMatrix,
    gate: Option<PackedMatrix>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    input_scale: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    gate_scale: f32,
    attn_q_scale: f32,
    attn_out_scale: f32,
    o_input_scale: f32,
    out_scale: f32,
    v_quant: VQuant,
    reps: i32,
}

impl AttentionFullPlan {
    /// Build a full attention plan. Shapes are checked against each other; the
    /// per-head dimensions are checked by [`AttentionFullStage::validate`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        caches: AttentionPlan,
        query: PackedMatrix,
        key: PackedMatrix,
        value: PackedMatrix,
        output: PackedMatrix,
        gate: Option<PackedMatrix>,
        query_norm: Vec<f32>,
        key_norm: Vec<f32>,
        input_scale: f32,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        gate_scale: f32,
        attn_q_scale: f32,
        attn_out_scale: f32,
        o_input_scale: f32,
        out_scale: f32,
        v_quant: VQuant,
        reps: i32,
    ) -> Result<Self, WeightError> {
        for (name, scale) in [
            ("input_scale", input_scale),
            ("q_scale", q_scale),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
            ("gate_scale", gate_scale),
            ("attn_q_scale", attn_q_scale),
            ("attn_out_scale", attn_out_scale),
            ("o_input_scale", o_input_scale),
            ("out_scale", out_scale),
        ] {
            check_scale(&format!("attention {name}"), scale)?;
        }
        if reps < 1 {
            return Err(WeightError::Layout(format!(
                "attention reps must be positive, got {reps}"
            )));
        }
        if query_norm.len() != key_norm.len() {
            return Err(WeightError::Layout(format!(
                "attention q/k norm lengths {} / {} differ",
                query_norm.len(),
                key_norm.len()
            )));
        }
        if query.k_padded() != key.k_padded()
            || query.k_padded() != value.k_padded()
            || query.n_padded() != output.k_padded()
        {
            return Err(WeightError::Layout(format!(
                "attention shapes do not compose: q {}x{}, k {}x{}, v {}x{}, o {}x{}",
                query.n_padded(),
                query.k_padded(),
                key.n_padded(),
                key.k_padded(),
                value.n_padded(),
                value.k_padded(),
                output.n_padded(),
                output.k_padded()
            )));
        }
        if let Some(gate) = &gate {
            if gate.k_padded() != query.k_padded() || gate.n_padded() < 16 {
                return Err(WeightError::Layout(format!(
                    "attention gate is {}x{}, expected {} rows of {}",
                    gate.n_padded(),
                    gate.k_padded(),
                    1,
                    query.k_padded()
                )));
            }
        }
        Ok(Self {
            caches,
            query,
            key,
            value,
            output,
            gate,
            query_norm,
            key_norm,
            input_scale,
            q_scale,
            k_scale,
            v_scale,
            gate_scale,
            attn_q_scale,
            attn_out_scale,
            o_input_scale,
            out_scale,
            v_quant,
            reps,
        })
    }

    /// Resident caches.
    pub const fn caches(&self) -> &AttentionPlan {
        &self.caches
    }

    /// Bytes of caches and projections.
    pub fn bytes(&self) -> usize {
        self.caches.bytes()
            + self.query.len()
            + self.key.len()
            + self.value.len()
            + self.output.len()
            + self.gate.as_ref().map_or(0, PackedMatrix::len)
    }
}

/// Resident vocabulary projection tied to the token embedding.
#[derive(Debug, Clone)]
pub struct LmHeadPlan {
    matrix: PackedMatrix,
    input_scale: f32,
    output_scale: f32,
    reps: i32,
}

impl LmHeadPlan {
    /// Build the head plan from the packed embedding matrix.
    pub fn new(
        matrix: PackedMatrix,
        input_scale: f32,
        output_scale: f32,
        reps: i32,
    ) -> Result<Self, WeightError> {
        check_scale("lm_head input_scale", input_scale)?;
        check_scale("lm_head output_scale", output_scale)?;
        if reps < 1 {
            return Err(WeightError::Layout(format!(
                "lm_head reps must be positive, got {reps}"
            )));
        }
        Ok(Self {
            matrix,
            input_scale,
            output_scale,
            reps,
        })
    }

    /// Resident bytes.
    pub fn bytes(&self) -> usize {
        self.matrix.len()
    }
}

/// Plan for one transformer layer.
#[derive(Debug, Clone, Default)]
pub struct LayerPlan {
    ffn: Option<FfnPlan>,
    attention: Option<AttentionPlan>,
    attention_full: Option<AttentionFullPlan>,
    conv: Option<ConvPlan>,
}

impl LayerPlan {
    /// FFN plan, if any.
    pub fn ffn(&self) -> Option<&FfnPlan> {
        self.ffn.as_ref()
    }

    /// Attention plan, if any.
    pub fn attention(&self) -> Option<&AttentionPlan> {
        self.attention.as_ref()
    }

    /// Full attention plan, if any.
    pub fn attention_full(&self) -> Option<&AttentionFullPlan> {
        self.attention_full.as_ref()
    }

    /// Short-convolution plan, if any.
    pub fn conv(&self) -> Option<&ConvPlan> {
        self.conv.as_ref()
    }

    fn bytes(&self) -> usize {
        self.ffn.as_ref().map_or(0, FfnPlan::bytes)
            + self.attention.as_ref().map_or(0, AttentionPlan::bytes)
            + self
                .attention_full
                .as_ref()
                .map_or(0, AttentionFullPlan::bytes)
            + self.conv.as_ref().map_or(0, ConvPlan::bytes)
    }
}

/// Full resident plan: budget plus per-layer stages.
#[derive(Debug, Clone)]
pub struct WeightPlan {
    budget: WeightBudget,
    layers: BTreeMap<usize, LayerPlan>,
    lm_head: Option<LmHeadPlan>,
}

impl WeightPlan {
    /// Start a plan with `budget` resident bytes.
    pub fn builder(budget: impl Into<WeightBudget>) -> WeightPlanBuilder {
        WeightPlanBuilder::new(budget)
    }

    /// Configured budget.
    pub const fn budget(&self) -> WeightBudget {
        self.budget
    }

    /// FFN plan for `layer`.
    pub fn ffn(&self, layer: usize) -> Option<&FfnPlan> {
        self.layers.get(&layer).and_then(LayerPlan::ffn)
    }

    /// Attention plan for `layer`.
    pub fn attention(&self, layer: usize) -> Option<&AttentionPlan> {
        self.layers.get(&layer).and_then(LayerPlan::attention)
    }

    /// Full attention plan for `layer`.
    pub fn attention_full(&self, layer: usize) -> Option<&AttentionFullPlan> {
        self.layers.get(&layer).and_then(LayerPlan::attention_full)
    }

    /// Short-convolution plan for `layer`.
    pub fn conv(&self, layer: usize) -> Option<&ConvPlan> {
        self.layers.get(&layer).and_then(LayerPlan::conv)
    }

    /// Vocabulary projection plan, if any.
    pub fn lm_head(&self) -> Option<&LmHeadPlan> {
        self.lm_head.as_ref()
    }

    /// Bytes of the FFN stage for `layer`.
    pub fn ffn_bytes(&self, layer: usize) -> Option<usize> {
        self.ffn(layer).map(FfnPlan::bytes)
    }

    /// Bytes of the attention stage for `layer`.
    pub fn attention_bytes(&self, layer: usize) -> Option<usize> {
        self.attention(layer).map(AttentionPlan::bytes)
    }

    /// Bytes of the full attention stage for `layer`.
    pub fn attention_full_bytes(&self, layer: usize) -> Option<usize> {
        self.attention_full(layer).map(AttentionFullPlan::bytes)
    }

    /// Bytes of the short-convolution stage for `layer`.
    pub fn conv_bytes(&self, layer: usize) -> Option<usize> {
        self.conv(layer).map(ConvPlan::bytes)
    }

    /// Sum of every planned stage.
    pub fn total_bytes(&self) -> usize {
        self.layers.values().map(LayerPlan::bytes).sum::<usize>()
            + self.lm_head.as_ref().map_or(0, LmHeadPlan::bytes)
    }

    /// Refuse a plan that cannot be resident in one go.
    ///
    /// This is the check a full-model plan runs before it allocates anything:
    /// 24 layers of int4 weights plus the tied head are all needed at once, so
    /// an LRU thrash is not an option and the caller gets a typed budget error
    /// instead of a half-loaded model.
    pub fn check_budget(&self) -> Result<(), WeightError> {
        let bytes = self.total_bytes();
        if bytes > self.budget.bytes() {
            return Err(WeightError::PlanOverBudget {
                bytes,
                budget: self.budget.bytes(),
            });
        }
        Ok(())
    }

    /// Layers that have at least one stage.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    fn unit_bytes(&self, key: UnitKey) -> Option<usize> {
        match key.kind {
            StageKind::Ffn => self.ffn_bytes(key.layer),
            StageKind::Attention => self.attention_bytes(key.layer),
            StageKind::Conv => self.conv_bytes(key.layer),
            StageKind::AttentionFull => self.attention_full_bytes(key.layer),
            StageKind::LmHead => self.lm_head.as_ref().map(LmHeadPlan::bytes),
        }
    }
}

/// Builds a [`WeightPlan`]. A second stage of the same kind replaces the first.
#[derive(Debug, Clone)]
pub struct WeightPlanBuilder {
    plan: WeightPlan,
}

impl WeightPlanBuilder {
    /// Start with `budget` resident bytes.
    pub fn new(budget: impl Into<WeightBudget>) -> Self {
        Self {
            plan: WeightPlan {
                budget: budget.into(),
                layers: BTreeMap::new(),
                lm_head: None,
            },
        }
    }

    /// Add or replace the FFN stage of `layer`.
    pub fn ffn(mut self, layer: usize, ffn: FfnPlan) -> Self {
        self.plan.layers.entry(layer).or_default().ffn = Some(ffn);
        self
    }

    /// Add or replace the attention stage of `layer`.
    pub fn attention(mut self, layer: usize, attention: AttentionPlan) -> Self {
        self.plan.layers.entry(layer).or_default().attention = Some(attention);
        self
    }

    /// Add or replace the short-convolution stage of `layer`.
    pub fn conv(mut self, layer: usize, conv: ConvPlan) -> Self {
        self.plan.layers.entry(layer).or_default().conv = Some(conv);
        self
    }

    /// Add or replace the full attention stage of `layer`.
    pub fn attention_full(mut self, layer: usize, attention: AttentionFullPlan) -> Self {
        let caches = attention.caches.clone();
        let entry = self.plan.layers.entry(layer).or_default();
        entry.attention_full = Some(attention);
        // The cache handles are the same buffers; keep the standalone plan too
        // so a pre-projected caller still finds them.
        entry.attention = Some(caches);
        self
    }

    /// Attach the tied vocabulary projection.
    pub fn lm_head(mut self, plan: LmHeadPlan) -> Self {
        self.plan.lm_head = Some(plan);
        self
    }

    /// Finish the plan.
    pub fn build(self) -> WeightPlan {
        self.plan
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct UnitKey {
    layer: usize,
    kind: StageKind,
}

struct LoadedImage {
    handle: i32,
    bytes: usize,
    /// ION buffer backing an `alloc_in` handle; unmapped by `release_buf` and
    /// freed when this drops.
    shared: Option<SharedBuffer>,
}

impl fmt::Debug for LoadedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedImage")
            .field("handle", &self.handle)
            .field("bytes", &self.bytes)
            .field("attached", &self.shared.is_some())
            .finish()
    }
}

#[derive(Debug)]
struct LoadedUnit {
    images: Vec<LoadedImage>,
    last_used: u64,
}

impl LoadedUnit {
    fn bytes(&self) -> usize {
        self.images.iter().map(|image| image.bytes).sum()
    }

    fn handles(&self) -> Vec<i32> {
        self.images.iter().map(|image| image.handle).collect()
    }
}

#[derive(Debug, Default)]
struct State {
    units: BTreeMap<UnitKey, LoadedUnit>,
    used: usize,
    clock: u64,
    loads: u64,
    evictions: u64,
    unloads: u64,
    load_calls: u64,
    load_bytes: u64,
    load_micros: u64,
}

/// Resident weights backed by a DSP heap or ION buffers plus an LRU budget.
pub struct ResidentWeights {
    executor: Arc<dyn DspExecutor>,
    handle: DspHandle,
    plan: WeightPlan,
    chunk: usize,
    strategy: LoadStrategy,
    transport: Option<Arc<dyn Transport>>,
    state: Mutex<State>,
}

impl ResidentWeights {
    /// Attach a plan to an open `lfmgemv` handle.
    ///
    /// The handle stays owned by the caller (normally `DspBackend`); the
    /// planner only invokes methods on it and releases its buffers in `Drop`.
    pub fn new(executor: Arc<dyn DspExecutor>, handle: DspHandle, plan: WeightPlan) -> Self {
        Self {
            executor,
            handle,
            plan,
            chunk: LOAD_CHUNK,
            strategy: LoadStrategy::Copy,
            transport: None,
            state: Mutex::new(State::default()),
        }
    }

    /// Attach a plan that stages weights as zero-copy ION views.
    ///
    /// The transport is only used for `alloc_shared`: every packed tensor gets
    /// its own 128-byte-padded `rpcmem` buffer, and `alloc_in` makes the DSP
    /// read it in place. This is the strategy a full model uses, because the
    /// DSP heap cannot hold 24 layers of int4 weights.
    pub fn with_ion(
        executor: Arc<dyn DspExecutor>,
        handle: DspHandle,
        plan: WeightPlan,
        transport: Arc<dyn Transport>,
    ) -> Self {
        Self {
            executor,
            handle,
            plan,
            chunk: LOAD_CHUNK,
            strategy: LoadStrategy::Attach,
            transport: Some(transport),
            state: Mutex::new(State::default()),
        }
    }

    /// Override the `load_buf` chunk size. Zero is clamped to one byte.
    pub fn set_chunk(&mut self, chunk: usize) {
        self.chunk = chunk.max(1);
    }

    /// Chunk size used by `load_buf`.
    pub const fn chunk(&self) -> usize {
        self.chunk
    }

    /// Select the upload strategy.
    pub fn set_strategy(&mut self, strategy: LoadStrategy) {
        self.strategy = strategy;
    }

    /// Selected upload strategy.
    pub const fn strategy(&self) -> LoadStrategy {
        self.strategy
    }

    /// Plan this planner follows.
    pub fn plan(&self) -> &WeightPlan {
        &self.plan
    }

    /// Counter snapshot.
    pub fn stats(&self) -> WeightStats {
        let state = self.lock();
        WeightStats {
            loads: state.loads,
            evictions: state.evictions,
            unloads: state.unloads,
            load_calls: state.load_calls,
            load_bytes: state.load_bytes,
            load_micros: state.load_micros,
            chunk: self.chunk,
            strategy: self.strategy,
            resident_units: state.units.len(),
            live_bytes: state.used,
            budget: self.plan.budget.bytes(),
        }
    }

    /// Units currently resident, in layer order.
    pub fn resident_units(&self) -> Vec<(usize, StageKind)> {
        let state = self.lock();
        state
            .units
            .keys()
            .map(|key| (key.layer, key.kind))
            .collect()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Make the FFN stage of `layer` resident and return its handles.
    pub fn ffn_stage(&self, layer: usize) -> Result<FfnStage, WeightError> {
        let (act_scale, out_scale, reps) = {
            let plan = self.plan.ffn(layer).ok_or(WeightError::Missing {
                layer,
                kind: StageKind::Ffn,
            })?;
            (plan.act_multiplier(), plan.out_scale(), plan.reps())
        };
        let handles = self.ensure(UnitKey {
            layer,
            kind: StageKind::Ffn,
        })?;
        let [gate, up, down] = handles.as_slice() else {
            return Err(WeightError::Layout(format!(
                "ffn layer {layer} has {} resident buffers, expected 3",
                handles.len()
            )));
        };
        Ok(FfnStage {
            gate: *gate,
            up: *up,
            down: *down,
            act_scale,
            out_scale,
            reps,
        })
    }

    /// Make the attention stage of `layer` resident and return its handles.
    pub fn attention_stage(&self, layer: usize) -> Result<AttentionStage, WeightError> {
        let (params, has_value_scales) = {
            let plan = self.plan.attention(layer).ok_or(WeightError::Missing {
                layer,
                kind: StageKind::Attention,
            })?;
            (plan.params, plan.value_scales.is_some())
        };
        let handles = self.ensure(UnitKey {
            layer,
            kind: StageKind::Attention,
        })?;
        if handles.len() < 3 {
            return Err(WeightError::Layout(format!(
                "attention layer {layer} has {} resident buffers, expected at least 3",
                handles.len()
            )));
        }
        if has_value_scales && handles.len() != 4 {
            return Err(WeightError::Layout(format!(
                "attention layer {layer} has {} resident buffers, expected 4",
                handles.len()
            )));
        }
        Ok(AttentionStage {
            keys: handles[0],
            key_scales: handles[1],
            values: handles[2],
            value_scales: handles.get(3).copied().unwrap_or(0),
            q_scale: params.q_scale,
            out_scale: params.out_scale,
            reps: params.reps,
            cache_pages: params.cache_pages,
            preprojected: params.preprojected,
        })
    }

    /// Make the short-convolution stage of `layer` resident.
    pub fn conv_stage(&self, layer: usize) -> Result<ConvStage, WeightError> {
        let plan = self.plan.conv(layer).ok_or(WeightError::Missing {
            layer,
            kind: StageKind::Conv,
        })?;
        let handles = self.ensure(UnitKey {
            layer,
            kind: StageKind::Conv,
        })?;
        let [in_proj, conv, out_proj] = handles.as_slice() else {
            return Err(WeightError::Layout(format!(
                "conv layer {layer} has {} resident buffers, expected 3",
                handles.len()
            )));
        };
        Ok(ConvStage {
            in_proj: *in_proj,
            conv: *conv,
            out_proj: *out_proj,
            ksh: plan.ksh,
            act_scale: 1.0 / plan.input_scale,
            mix_scale: plan.mix_scale,
            y_scale: plan.y_scale,
            out_scale: plan.out_scale,
            reps: plan.reps,
            state_off: plan.state_off,
        })
    }

    /// Make the full attention stage of `layer` resident.
    pub fn attention_full_stage(&self, layer: usize) -> Result<AttentionFullStage, WeightError> {
        let plan = self
            .plan
            .attention_full(layer)
            .ok_or(WeightError::Missing {
                layer,
                kind: StageKind::AttentionFull,
            })?;
        let handles = self.ensure(UnitKey {
            layer,
            kind: StageKind::AttentionFull,
        })?;
        let weights = &handles[plan.caches.image_count()..];
        let expected = 4 + usize::from(plan.gate.is_some());
        if weights.len() != expected {
            return Err(WeightError::Layout(format!(
                "attention layer {layer} has {} projection buffers, expected {expected}",
                weights.len()
            )));
        }
        let params = plan.caches.params;
        Ok(AttentionFullStage {
            caches: AttentionStage {
                keys: handles[0],
                key_scales: handles[1],
                values: handles[2],
                value_scales: handles.get(3).copied().unwrap_or(0),
                q_scale: params.q_scale,
                out_scale: params.out_scale,
                reps: params.reps,
                cache_pages: params.cache_pages,
                preprojected: params.preprojected,
            },
            query: weights[0],
            key: weights[1],
            value: weights[2],
            output: weights[3],
            gate: plan.gate.as_ref().map(|_| weights[4]),
            query_norm: plan.query_norm.clone(),
            key_norm: plan.key_norm.clone(),
            act_scale: 1.0 / plan.input_scale,
            q_scale: plan.q_scale,
            k_scale: plan.k_scale,
            v_scale: plan.v_scale,
            gate_scale: plan.gate_scale,
            attn_q_scale: plan.attn_q_scale,
            attn_out_scale: plan.attn_out_scale,
            o_act_scale: 1.0 / plan.o_input_scale,
            out_scale: plan.out_scale,
            v_quant: plan.v_quant.clone(),
            reps: plan.reps,
        })
    }

    /// Make the vocabulary projection resident.
    pub fn lm_head_stage(&self) -> Result<LmHeadStage, WeightError> {
        let plan = self.plan.lm_head().ok_or(WeightError::Missing {
            layer: 0,
            kind: StageKind::LmHead,
        })?;
        let handles = self.ensure(UnitKey {
            layer: 0,
            kind: StageKind::LmHead,
        })?;
        let [gemv] = handles.as_slice() else {
            return Err(WeightError::Layout(format!(
                "lm_head has {} resident buffers, expected 1",
                handles.len()
            )));
        };
        Ok(LmHeadStage {
            gemv: *gemv,
            input_scale: plan.input_scale,
            output_scale: plan.output_scale,
            rows: plan.matrix.n_padded(),
            reps: plan.reps,
        })
    }

    fn ensure(&self, key: UnitKey) -> Result<Vec<i32>, WeightError> {
        if self.strategy == LoadStrategy::Attach && self.transport.is_none() {
            return Err(WeightError::UnsupportedStrategy {
                strategy: self.strategy,
                reason: "zero-copy staging needs the FastRPC transport to allocate shared \
                         buffers; build the planner with ResidentWeights::with_ion",
            });
        }
        let bytes = self.plan.unit_bytes(key).ok_or(WeightError::Missing {
            layer: key.layer,
            kind: key.kind,
        })?;
        let budget = self.plan.budget.bytes();
        let mut state = self.lock();

        if state.units.contains_key(&key) {
            state.clock = state.clock.wrapping_add(1);
            let clock = state.clock;
            let unit = state.units.get_mut(&key).ok_or(WeightError::Missing {
                layer: key.layer,
                kind: key.kind,
            })?;
            unit.last_used = clock;
            return Ok(unit.handles());
        }

        if bytes > budget {
            return Err(WeightError::Budget {
                layer: key.layer,
                kind: key.kind,
                bytes,
                budget,
                used: state.used,
            });
        }

        while state.used + bytes > budget {
            let Some(victim) = state
                .units
                .iter()
                .min_by_key(|(_, unit)| unit.last_used)
                .map(|(key, _)| *key)
            else {
                return Err(WeightError::Budget {
                    layer: key.layer,
                    kind: key.kind,
                    bytes,
                    budget,
                    used: state.used,
                });
            };
            let unit = state.units.remove(&victim).ok_or(WeightError::Missing {
                layer: victim.layer,
                kind: victim.kind,
            })?;
            state.used = state.used.saturating_sub(unit.bytes());
            state.evictions += 1;
            state.unloads += 1;
            let release_error = unit
                .images
                .iter()
                .find_map(|image| self.release(image.handle).err());
            if let Some(error) = release_error {
                return Err(WeightError::Transport(error));
            }
        }

        let images = match key.kind {
            StageKind::Ffn => self.upload_ffn(key.layer, &mut state)?,
            StageKind::Attention => self.upload_attention(key.layer, &mut state)?,
            StageKind::Conv => self.upload_conv(key.layer, &mut state)?,
            StageKind::AttentionFull => self.upload_attention_full(key.layer, &mut state)?,
            StageKind::LmHead => self.upload_lm_head(&mut state)?,
        };
        let bytes: usize = images.iter().map(|image| image.bytes).sum();
        state.used += bytes;
        state.clock = state.clock.wrapping_add(1);
        let last_used = state.clock;
        state.loads += 1;
        let handles = images.iter().map(|image| image.handle).collect();
        state.units.insert(key, LoadedUnit { images, last_used });
        Ok(handles)
    }

    fn upload_ffn(&self, layer: usize, state: &mut State) -> Result<Vec<LoadedImage>, WeightError> {
        let plan = self.plan.ffn(layer).ok_or(WeightError::Missing {
            layer,
            kind: StageKind::Ffn,
        })?;
        // The FFN protocol: gate/up keep Q16 values (`sa = s_in * 65536`,
        // `so = 1`), down consumes the int8 mid buffer (`sa = s_mid`,
        // `so = s_out`). See FfnPlan and `ffn_main.c`.
        let gate_sa = plan.input_scale() * 65536.0;
        let mut images = Vec::with_capacity(3);
        for (matrix, sa, so) in [
            (plan.gate(), gate_sa, 1.0f32),
            (plan.up(), gate_sa, 1.0f32),
            (plan.down(), plan.mid_scale(), plan.out_scale()),
        ] {
            match self.upload_packed(matrix, sa, so, state) {
                Ok(image) => images.push(image),
                Err(error) => {
                    release_images(self, &images);
                    return Err(error);
                }
            }
        }
        Ok(images)
    }

    fn upload_attention(
        &self,
        layer: usize,
        state: &mut State,
    ) -> Result<Vec<LoadedImage>, WeightError> {
        let plan = self.plan.attention(layer).ok_or(WeightError::Missing {
            layer,
            kind: StageKind::Attention,
        })?;
        let sources: [Option<(&ResidentImage, &[u8])>; 4] = [
            Some((&plan.keys, plan.keys.bytes())),
            Some((&plan.key_scales, plan.key_scales.bytes())),
            Some((&plan.values, plan.values.bytes())),
            plan.value_scales
                .as_ref()
                .map(|image| (image, image.bytes())),
        ];

        if self.strategy == LoadStrategy::Attach {
            let mut images = Vec::with_capacity(4);
            for source in sources.into_iter().flatten() {
                match self.upload_image(source.0, state) {
                    Ok(image) => images.push(image),
                    Err(error) => {
                        release_images(self, &images);
                        return Err(error);
                    }
                }
            }
            return Ok(images);
        }

        // `attn_main.c` allocates K, KT, V (and value scales) before loading
        // anything, then loads K, V, KT.
        let mut images = Vec::with_capacity(4);
        for source in sources.iter().flatten() {
            match self.alloc(source.0.label(), source.0.len()) {
                Ok(handle) => images.push(LoadedImage {
                    handle,
                    bytes: source.0.len(),
                    shared: None,
                }),
                Err(error) => {
                    release_images(self, &images);
                    return Err(error);
                }
            }
        }
        let result = (|| -> Result<(), WeightError> {
            for index in [0usize, 2, 1, 3] {
                let Some((_, bytes)) = sources.get(index).copied().flatten() else {
                    continue;
                };
                let Some(image) = images.get(index) else {
                    continue;
                };
                self.load_all(image.handle, bytes, state)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            release_images(self, &images);
            return Err(error);
        }
        Ok(images)
    }

    fn upload_conv(
        &self,
        layer: usize,
        state: &mut State,
    ) -> Result<Vec<LoadedImage>, WeightError> {
        let plan = self.plan.conv(layer).ok_or(WeightError::Missing {
            layer,
            kind: StageKind::Conv,
        })?;
        let mut images = Vec::with_capacity(3);
        match self.upload_packed(&plan.in_proj, plan.input_scale, plan.mix_scale, state) {
            Ok(image) => images.push(image),
            Err(error) => {
                release_images(self, &images);
                return Err(error);
            }
        }
        match self.upload_image(&plan.conv, state) {
            Ok(image) => images.push(image),
            Err(error) => {
                release_images(self, &images);
                return Err(error);
            }
        }
        match self.upload_packed(&plan.out_proj, plan.y_scale, plan.out_scale, state) {
            Ok(image) => images.push(image),
            Err(error) => {
                release_images(self, &images);
                return Err(error);
            }
        }
        Ok(images)
    }

    fn upload_attention_full(
        &self,
        layer: usize,
        state: &mut State,
    ) -> Result<Vec<LoadedImage>, WeightError> {
        let plan = self
            .plan
            .attention_full(layer)
            .ok_or(WeightError::Missing {
                layer,
                kind: StageKind::AttentionFull,
            })?;

        // Caches first: attn_main.c allocates K, KT, V before the weights.
        let caches = &plan.caches;
        let mut images = Vec::with_capacity(8);
        let push_image = |uploaded: Result<LoadedImage, WeightError>,
                          images: &mut Vec<LoadedImage>|
         -> Result<(), WeightError> {
            match uploaded {
                Ok(image) => {
                    images.push(image);
                    Ok(())
                }
                Err(error) => {
                    release_images(self, images);
                    Err(error)
                }
            }
        };
        for image in [&caches.keys, &caches.key_scales, &caches.values] {
            push_image(self.upload_image(image, state), &mut images)?;
        }
        if let Some(scales) = &caches.value_scales {
            push_image(self.upload_image(scales, state), &mut images)?;
        }
        for (matrix, sa, so) in [
            (&plan.query, plan.input_scale, plan.q_scale),
            (&plan.key, plan.input_scale, plan.k_scale),
            (&plan.value, plan.input_scale, plan.v_scale),
            (&plan.output, plan.o_input_scale, plan.out_scale),
        ] {
            push_image(self.upload_packed(matrix, sa, so, state), &mut images)?;
        }
        if let Some(gate) = &plan.gate {
            push_image(
                self.upload_packed(gate, plan.input_scale, plan.gate_scale, state),
                &mut images,
            )?;
        }
        Ok(images)
    }

    fn upload_lm_head(&self, state: &mut State) -> Result<Vec<LoadedImage>, WeightError> {
        let plan = self.plan.lm_head().ok_or(WeightError::Missing {
            layer: 0,
            kind: StageKind::LmHead,
        })?;
        Ok(vec![self.upload_packed(
            &plan.matrix,
            plan.input_scale,
            plan.output_scale,
            state,
        )?])
    }

    /// Upload one packed matrix, in place (`Attach`) or into the DSP heap.
    fn upload_packed(
        &self,
        matrix: &PackedMatrix,
        input_scale: f32,
        output_scale: f32,
        state: &mut State,
    ) -> Result<LoadedImage, WeightError> {
        let payload = matrix_scale_payload(matrix, input_scale, output_scale)?;
        let (handle, shared) = if self.strategy == LoadStrategy::Attach {
            let image = self.attach_bytes(matrix.name(), matrix.weights(), state)?;
            (image.0, Some(image.1))
        } else {
            (self.alloc(matrix.name(), matrix.weights().len())?, None)
        };
        let image = LoadedImage {
            handle,
            bytes: matrix.weights().len(),
            shared,
        };
        let result = (|| -> Result<(), WeightError> {
            self.set_shape(handle, matrix.n_padded(), matrix.k_padded())?;
            if self.strategy == LoadStrategy::Copy {
                self.load_all(handle, matrix.weights(), state)?;
            }
            self.set_scale(handle, &payload)
        })();
        if let Err(error) = result {
            let _ = self.release(handle);
            return Err(error);
        }
        Ok(image)
    }

    /// Upload a raw image (KV cache, conv blob), zero-filled for `Attach`.
    fn upload_image(
        &self,
        image: &ResidentImage,
        state: &mut State,
    ) -> Result<LoadedImage, WeightError> {
        if self.strategy == LoadStrategy::Attach {
            let (handle, shared) = self.attach_bytes(image.label(), image.bytes(), state)?;
            Ok(LoadedImage {
                handle,
                bytes: image.len(),
                shared: Some(shared),
            })
        } else {
            let handle = self.alloc(image.label(), image.len())?;
            if let Err(error) = self.load_all(handle, image.bytes(), state) {
                let _ = self.release(handle);
                return Err(error);
            }
            Ok(LoadedImage {
                handle,
                bytes: image.len(),
                shared: None,
            })
        }
    }

    /// Copy `data` into a fresh `rpcmem` buffer and map it with `alloc_in`.
    fn attach_bytes(
        &self,
        label: &str,
        data: &[u8],
        state: &mut State,
    ) -> Result<(i32, SharedBuffer), WeightError> {
        let transport = self
            .transport
            .as_ref()
            .ok_or(WeightError::UnsupportedStrategy {
                strategy: LoadStrategy::Attach,
                reason: "no transport for `alloc_shared`",
            })?;
        let nbytes = to_i32(data.len().max(1).div_ceil(128) * 128, "alloc_in nbytes")?;
        let mut shared = transport
            .alloc_shared(nbytes as usize)
            .map_err(|error| WeightError::Transport(DspError::from(error)))?;
        shared.as_mut_slice()[..data.len()].copy_from_slice(data);
        // Flush the CPU writes: rpcmem is cached and the DSP maps the same
        // dma-buf, exactly like `attach_main.c` does before `attach_in`.
        crate::transport::sync_shared(&shared, true, true)
            .map_err(|error| WeightError::Transport(DspError::Other(error.to_string())))?;
        crate::transport::sync_shared(&shared, false, true)
            .map_err(|error| WeightError::Transport(DspError::Other(error.to_string())))?;
        let prim = AllocInScalars {
            fd: shared.fd(),
            nbytes,
        }
        .to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.executor
            .invoke(
                self.handle,
                DspCall {
                    method: method_id::ALLOC_IN,
                    prim_in: &prim,
                    prim_out: &mut out,
                    in_bufs: &[],
                    out_bufs: &mut no_out,
                },
            )
            .map_err(WeightError::Transport)?;
        let handle = i32::from_le_bytes(out);
        if handle < 0 {
            return Err(WeightError::Layout(format!(
                "{label}: alloc_in returned negative handle {handle}"
            )));
        }
        state.load_calls += 1;
        state.load_bytes += data.len() as u64;
        Ok((handle, shared))
    }

    fn alloc(&self, label: &str, bytes: usize) -> Result<i32, WeightError> {
        let nbytes = to_i32(bytes, "alloc_buf nbytes")?;
        let prim = AllocBufScalars { nbytes }.to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.executor
            .invoke(
                self.handle,
                DspCall {
                    method: method_id::ALLOC_BUF,
                    prim_in: &prim,
                    prim_out: &mut out,
                    in_bufs: &[],
                    out_bufs: &mut no_out,
                },
            )
            .map_err(WeightError::Transport)?;
        let handle = i32::from_le_bytes(out);
        if handle < 0 {
            return Err(WeightError::Layout(format!(
                "{label}: alloc_buf returned negative handle {handle}"
            )));
        }
        Ok(handle)
    }

    fn set_shape(&self, handle: i32, n: usize, k: usize) -> Result<(), WeightError> {
        let n = to_i32(n, "set_shape N")?;
        let k = to_i32(k, "set_shape K")?;
        let prim = SetShapeScalars {
            h: handle,
            n,
            k,
            with_scale: 1,
        }
        .to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.executor
            .invoke(
                self.handle,
                DspCall {
                    method: method_id::SET_SHAPE,
                    prim_in: &prim,
                    prim_out: &mut prim_out,
                    in_bufs: &[],
                    out_bufs: &mut no_out,
                },
            )
            .map_err(WeightError::Transport)
    }

    fn set_scale(&self, handle: i32, payload: &[u8]) -> Result<(), WeightError> {
        let data_len = octet_len(payload.len()).ok_or(WeightError::TooLarge {
            what: "set_scale data",
            bytes: payload.len(),
        })?;
        let prim = SetScaleWire {
            h: handle,
            data_len,
        }
        .to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.executor
            .invoke(
                self.handle,
                DspCall {
                    method: method_id::SET_SCALE,
                    prim_in: &prim,
                    prim_out: &mut prim_out,
                    in_bufs: &[payload],
                    out_bufs: &mut no_out,
                },
            )
            .map_err(WeightError::Transport)
    }

    fn load_all(&self, handle: i32, data: &[u8], state: &mut State) -> Result<(), WeightError> {
        for (index, chunk) in data.chunks(self.chunk).enumerate() {
            let offset = index * self.chunk;
            self.load_chunk(handle, offset, chunk, state)?;
        }
        Ok(())
    }

    fn load_chunk(
        &self,
        handle: i32,
        offset: usize,
        data: &[u8],
        state: &mut State,
    ) -> Result<(), WeightError> {
        let off = to_i32(offset, "load_buf offset")?;
        // The stub packs `dataLen` as the third scalar (`_primIn[3]`).
        let data_len = octet_len(data.len()).ok_or(WeightError::TooLarge {
            what: "load_buf data",
            bytes: data.len(),
        })?;
        let prim = LoadBufWire {
            h: handle,
            off,
            data_len,
        }
        .to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        let started = std::time::Instant::now();
        self.executor
            .invoke(
                self.handle,
                DspCall {
                    method: method_id::LOAD_BUF,
                    prim_in: &prim,
                    prim_out: &mut prim_out,
                    in_bufs: &[data],
                    out_bufs: &mut no_out,
                },
            )
            .map_err(WeightError::Transport)?;
        state.load_calls += 1;
        state.load_bytes += data.len() as u64;
        state.load_micros += started.elapsed().as_micros() as u64;
        Ok(())
    }

    fn release(&self, handle: i32) -> Result<(), DspError> {
        let prim = ReleaseBufScalars { h: handle }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.executor.invoke(
            self.handle,
            DspCall {
                method: method_id::RELEASE_BUF,
                prim_in: &prim,
                prim_out: &mut prim_out,
                in_bufs: &[],
                out_bufs: &mut no_out,
            },
        )
    }

    /// Release every resident buffer. Idempotent and safe to call from `Drop`.
    pub fn release_all(&self) {
        let mut state = self.lock();
        let units = std::mem::take(&mut state.units);
        state.used = 0;
        for (_, unit) in units {
            for image in &unit.images {
                let _ = self.release(image.handle);
            }
            state.unloads += 1;
        }
    }
}

impl StageProvider for ResidentWeights {
    fn ffn_stage(&mut self, layer: usize) -> Result<FfnStage, StageError> {
        ResidentWeights::ffn_stage(self, layer).map_err(StageError::from)
    }

    fn attention_stage(&mut self, layer: usize) -> Result<AttentionStage, StageError> {
        ResidentWeights::attention_stage(self, layer).map_err(StageError::from)
    }

    fn conv_stage(&mut self, layer: usize) -> Result<ConvStage, StageError> {
        ResidentWeights::conv_stage(self, layer).map_err(StageError::from)
    }

    fn attention_full_stage(&mut self, layer: usize) -> Result<AttentionFullStage, StageError> {
        ResidentWeights::attention_full_stage(self, layer).map_err(StageError::from)
    }

    fn lm_head_stage(&mut self) -> Result<Option<LmHeadStage>, StageError> {
        match self.plan.lm_head() {
            Some(_) => ResidentWeights::lm_head_stage(self)
                .map(Some)
                .map_err(StageError::from),
            None => Ok(None),
        }
    }

    fn release_all(&mut self) {
        ResidentWeights::release_all(self);
    }
}

impl Drop for ResidentWeights {
    fn drop(&mut self) {
        self.release_all();
    }
}

fn release_images(planner: &ResidentWeights, images: &[LoadedImage]) {
    for image in images {
        let _ = planner.release(image.handle);
    }
}

fn check_scale(what: &str, value: f32) -> Result<(), WeightError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(WeightError::Layout(format!(
            "{what} must be finite and positive, got {value}"
        )));
    }
    Ok(())
}

fn to_i32(value: usize, what: &'static str) -> Result<i32, WeightError> {
    i32::try_from(value).map_err(|_| WeightError::TooLarge { what, bytes: value })
}

/// `16 + 4 * n_padded` bytes, exactly the stand's `sbuf`
/// (`ffn_main.c:74-81`, `host_main.c:203-211`): `N`, `K`, the activation scale
/// (`s_in`, real units per int8 code), the output scale (`s_out`), then one
/// fp32 weight scale per padded output channel.
fn matrix_scale_payload(
    matrix: &PackedMatrix,
    input_scale: f32,
    output_scale: f32,
) -> Result<Vec<u8>, WeightError> {
    check_scale("set_scale input", input_scale)?;
    check_scale("set_scale output", output_scale)?;
    let n = to_i32(matrix.n_padded(), "set_scale N")?;
    let k = to_i32(matrix.k_padded(), "set_scale K")?;
    let mut payload = Vec::with_capacity(16 + 4 * matrix.n_padded());
    payload.extend_from_slice(&n.to_le_bytes());
    payload.extend_from_slice(&k.to_le_bytes());
    payload.extend_from_slice(&input_scale.to_le_bytes());
    payload.extend_from_slice(&output_scale.to_le_bytes());
    for scale in matrix.scales() {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    Ok(payload)
}
