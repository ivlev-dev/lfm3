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
    AttentionStage, DspCall, DspError, DspExecutor, DspHandle, FfnStage, StageError, StageProvider,
};
use myelin_pack::{Manifest, PackResult};

use crate::method::{
    method_id, octet_len, AllocBufScalars, LoadBufWire, ReleaseBufScalars, SetScaleWire,
    SetShapeScalars,
};

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
    /// This requires an `attach-in` style method on the skeleton; `attach_out`
    /// only maps *output* buffers (see [`crate::hvx::HvxClient::attach_out`]),
    /// so the current `lfmgemv` IDL rejects this strategy with
    /// [`WeightError::UnsupportedStrategy`] instead of falling back silently.
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
}

impl StageKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::Attention => "attention",
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
            WeightError::Budget { .. } => Self::Budget(error.to_string()),
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

/// Resident FFN matrices and their host-side quantisation scales.
#[derive(Debug, Clone)]
pub struct FfnPlan {
    gate: PackedTensor,
    up: PackedTensor,
    down: PackedTensor,
    act_scale: f32,
    out_scale: f32,
    reps: i32,
}

impl FfnPlan {
    /// Build a layer plan from the three packed matrices.
    ///
    /// Shapes must compose: `down` is `[gate.n, gate.k]`. `act_scale` and
    /// `out_scale` are the host activation scales used by [`FfnStage`]; the
    /// per-matrix `set_scale` header carries the same pair.
    pub fn new(
        gate: PackedTensor,
        up: PackedTensor,
        down: PackedTensor,
        act_scale: f32,
        out_scale: f32,
        reps: i32,
    ) -> Result<Self, WeightError> {
        check_scale("ffn act_scale", act_scale)?;
        check_scale("ffn out_scale", out_scale)?;
        if reps < 1 {
            return Err(WeightError::Layout(format!(
                "ffn reps must be positive, got {reps}"
            )));
        }
        let gate_shape = gate.manifest();
        let up_shape = up.manifest();
        let down_shape = down.manifest();
        if gate_shape.k_padded != up_shape.k_padded || gate_shape.n_padded != up_shape.n_padded {
            return Err(WeightError::Layout(format!(
                "ffn gate is {}x{}, up is {}x{}",
                gate_shape.k_padded, gate_shape.n_padded, up_shape.k_padded, up_shape.n_padded
            )));
        }
        if down_shape.k_padded != gate_shape.n_padded || down_shape.n_padded != gate_shape.k_padded
        {
            return Err(WeightError::Layout(format!(
                "ffn down is {}x{}, expected {}x{}",
                down_shape.k_padded, down_shape.n_padded, gate_shape.n_padded, gate_shape.k_padded
            )));
        }
        Ok(Self {
            gate,
            up,
            down,
            act_scale,
            out_scale,
            reps,
        })
    }

    /// Gate matrix.
    pub fn gate(&self) -> &PackedTensor {
        &self.gate
    }

    /// Up matrix.
    pub fn up(&self) -> &PackedTensor {
        &self.up
    }

    /// Down matrix.
    pub fn down(&self) -> &PackedTensor {
        &self.down
    }

    /// Host multiplier applied to the f32 activation before quantisation.
    pub fn act_scale(&self) -> f32 {
        self.act_scale
    }

    /// Host multiplier applied to the int8 result.
    pub fn out_scale(&self) -> f32 {
        self.out_scale
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
}

/// Plan for one transformer layer.
#[derive(Debug, Clone, Default)]
pub struct LayerPlan {
    ffn: Option<FfnPlan>,
    attention: Option<AttentionPlan>,
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
}

/// Full resident plan: budget plus per-layer stages.
#[derive(Debug, Clone)]
pub struct WeightPlan {
    budget: WeightBudget,
    layers: BTreeMap<usize, LayerPlan>,
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

    /// Bytes of the FFN stage for `layer`.
    pub fn ffn_bytes(&self, layer: usize) -> Option<usize> {
        self.ffn(layer).map(FfnPlan::bytes)
    }

    /// Bytes of the attention stage for `layer`.
    pub fn attention_bytes(&self, layer: usize) -> Option<usize> {
        self.attention(layer).map(AttentionPlan::bytes)
    }

    /// Sum of every planned stage.
    pub fn total_bytes(&self) -> usize {
        self.layers
            .values()
            .map(|layer| {
                layer.ffn.as_ref().map_or(0, FfnPlan::bytes)
                    + layer.attention.as_ref().map_or(0, AttentionPlan::bytes)
            })
            .sum()
    }

    /// Layers that have at least one stage.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    fn unit_bytes(&self, key: UnitKey) -> Option<usize> {
        match key.kind {
            StageKind::Ffn => self.ffn_bytes(key.layer),
            StageKind::Attention => self.attention_bytes(key.layer),
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

#[derive(Debug)]
struct LoadedImage {
    handle: i32,
    bytes: usize,
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

/// Resident weights backed by a DSP heap and an LRU budget.
pub struct ResidentWeights {
    executor: Arc<dyn DspExecutor>,
    handle: DspHandle,
    plan: WeightPlan,
    chunk: usize,
    strategy: LoadStrategy,
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
            (plan.act_scale, plan.out_scale, plan.reps)
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

    fn ensure(&self, key: UnitKey) -> Result<Vec<i32>, WeightError> {
        if self.strategy == LoadStrategy::Attach {
            return Err(WeightError::UnsupportedStrategy {
                strategy: self.strategy,
                reason: "`lfmgemv.attach_out` maps output buffers only; the skeleton has no \
                         attach-in method that would let the DSP read host ION weights in place",
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
        let mut images = Vec::with_capacity(3);
        for tensor in [&plan.gate, &plan.up, &plan.down] {
            match self.upload_packed(tensor, plan.act_scale, plan.out_scale, state) {
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

        // attn_main.c allocates K, KT, V before loading anything.
        let mut images = Vec::with_capacity(4);
        for image in [&plan.keys, &plan.key_scales, &plan.values] {
            match self.alloc(image.label(), image.len()) {
                Ok(handle) => images.push(LoadedImage {
                    handle,
                    bytes: image.len(),
                }),
                Err(error) => {
                    release_images(self, &images);
                    return Err(error);
                }
            }
        }
        if let Some(scales) = &plan.value_scales {
            match self.alloc(scales.label(), scales.len()) {
                Ok(handle) => images.push(LoadedImage {
                    handle,
                    bytes: scales.len(),
                }),
                Err(error) => {
                    release_images(self, &images);
                    return Err(error);
                }
            }
        }

        // attn_main.c then loads K, V, KT (value scales would follow).
        let load_order = [0usize, 2, 1, 3];
        let sources = [
            plan.keys.bytes(),
            plan.key_scales.bytes(),
            plan.values.bytes(),
            plan.value_scales
                .as_ref()
                .map_or(&[][..], ResidentImage::bytes),
        ];
        let result = (|| -> Result<(), WeightError> {
            for index in load_order {
                let Some(image) = images.get(index) else {
                    continue;
                };
                if sources[index].is_empty() {
                    continue;
                }
                self.load_all(image.handle, sources[index], state)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            release_images(self, &images);
            return Err(error);
        }
        Ok(images)
    }

    fn upload_packed(
        &self,
        tensor: &PackedTensor,
        act_scale: f32,
        out_scale: f32,
        state: &mut State,
    ) -> Result<LoadedImage, WeightError> {
        let bytes = tensor.len();
        let handle = self.alloc(tensor.name(), bytes)?;
        let result = (|| -> Result<(), WeightError> {
            let manifest = tensor.manifest();
            self.set_shape(handle, manifest.n_padded, manifest.k_padded)?;
            self.load_all(handle, tensor.blob(), state)?;
            let payload = scale_payload(manifest, act_scale, out_scale)?;
            self.set_scale(handle, &payload)
        })();
        if let Err(error) = result {
            let _ = self.release(handle);
            return Err(error);
        }
        Ok(LoadedImage { handle, bytes })
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
/// (`ffn_main.c:74-81`, `host_main.c:203-211`): `N`, `K`, activation scale,
/// output scale, then one fp32 weight scale per padded output channel.
fn scale_payload(
    manifest: &Manifest,
    act_scale: f32,
    out_scale: f32,
) -> Result<Vec<u8>, WeightError> {
    if manifest.scales.scale_f32.len() != manifest.n_padded {
        return Err(WeightError::Layout(format!(
            "{}: {} scales for {} padded channels",
            manifest.tensor,
            manifest.scales.scale_f32.len(),
            manifest.n_padded
        )));
    }
    let n = to_i32(manifest.n_padded, "set_scale N")?;
    let k = to_i32(manifest.k_padded, "set_scale K")?;
    let mut payload = Vec::with_capacity(16 + 4 * manifest.n_padded);
    payload.extend_from_slice(&n.to_le_bytes());
    payload.extend_from_slice(&k.to_le_bytes());
    payload.extend_from_slice(&act_scale.to_le_bytes());
    payload.extend_from_slice(&out_scale.to_le_bytes());
    for scale in &manifest.scales.scale_f32 {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    Ok(payload)
}
