//! Model handle and generation loop.
//!
//! [`Model`] pairs a validated [`ModelSpec`] with a compute [`Backend`].
//! [`Session`] owns the per-sequence state: the paged KV caches of the attention
//! layers, the short-convolution history of the conv layers and the current
//! position. The loop is deliberately backend-agnostic: it only issues one
//! layer-level call per layer, exactly the shape a DSP command stream needs.

use std::path::Path;

use lfm_gguf::GgufFile;

use crate::backend::{AttentionCall, Backend, NormSlot, NormWeights};
use crate::build_spec;
use crate::gguf::GgufMeta;
use crate::kv::PagedKvCache;
use crate::ops;
use crate::sampling::{Sampler, SamplingParams};
use crate::spec::{LayerSpec, ModelSpec};
use crate::RuntimeError;

/// A loaded checkpoint plus the backend that executes it.
pub struct Model {
    spec: ModelSpec,
    backend: Box<dyn Backend>,
}

impl Model {
    /// Open a GGUF checkpoint, build its spec and load the CPU backend.
    ///
    /// This is the host path: the DSP backend is selected through
    /// [`Model::from_backend`] once the transport is wired.
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let file =
            GgufFile::open(path.as_ref()).map_err(|error| RuntimeError::Gguf(error.to_string()))?;
        let meta = GgufMeta::from_gguf(&file);
        let spec = build_spec(&meta)?;
        let backend = crate::backend::CpuBackend::from_gguf(&file, &spec)?;
        Ok(Self {
            spec,
            backend: Box::new(backend),
        })
    }

    /// Pair an already built backend with its spec.
    pub fn from_backend(spec: ModelSpec, backend: Box<dyn Backend>) -> Result<Self, RuntimeError> {
        spec.validate()?;
        if backend.spec() != &spec {
            return Err(RuntimeError::InvalidArgument(
                "backend adopted a different model spec".to_owned(),
            ));
        }
        Ok(Self { spec, backend })
    }

    /// Adopted specification.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Backend identifier, for logs and diagnostics.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Backend-specific detail of the last failure, when there is one.
    pub fn backend_diagnostic(&self) -> Option<String> {
        self.backend.diagnostic()
    }

    /// Start a fresh sequence that can hold `max_context` tokens.
    pub fn session(&mut self, max_context: usize) -> Result<Session<'_>, RuntimeError> {
        Session::new(&mut *self.backend, self.spec.clone(), max_context)
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The token limit was reached.
    Length,
    /// The caller-provided EOS token was produced.
    Eos,
    /// One of the explicit stop tokens was produced.
    StopToken,
}

/// Result of one generation call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// Newly generated tokens, stop token excluded.
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
}

/// Per-call generation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop as soon as this token is produced. The token is not returned.
    pub eos: Option<u32>,
    /// Additional stop tokens (for example an end-of-turn marker).
    pub stop: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            sampling: SamplingParams::default(),
            eos: None,
            stop: Vec::new(),
        }
    }
}

/// One sequence: KV caches, convolution state and position.
pub struct Session<'a> {
    spec: ModelSpec,
    backend: &'a mut dyn Backend,
    kv: Vec<Option<PagedKvCache>>,
    conv: Vec<Option<Vec<f32>>>,
    position: usize,
    max_context: usize,
}

impl<'a> Session<'a> {
    fn new(
        backend: &'a mut dyn Backend,
        spec: ModelSpec,
        max_context: usize,
    ) -> Result<Self, RuntimeError> {
        if max_context == 0 {
            return Err(RuntimeError::InvalidArgument(
                "max_context must be positive".to_owned(),
            ));
        }
        spec.validate()?;
        let dims = &spec.dims;
        let kv_dim = dims.kv_dim();
        let mut kv = Vec::with_capacity(spec.layers.len());
        let mut conv = Vec::with_capacity(spec.layers.len());
        for layer in spec.layers.iter().copied() {
            match layer {
                LayerSpec::Attention { .. } => {
                    kv.push(Some(PagedKvCache::new(kv_dim, max_context)?));
                    conv.push(None);
                }
                LayerSpec::Conv { ksize } => {
                    kv.push(None);
                    conv.push(Some(vec![0.0; (ksize - 1) * dims.hidden]));
                }
                LayerSpec::Moe { .. } => {
                    return Err(RuntimeError::InvalidArgument(
                        "mixture-of-experts layers are not executable".to_owned(),
                    ));
                }
            }
        }
        Ok(Self {
            spec,
            backend,
            kv,
            conv,
            position: 0,
            max_context,
        })
    }

    /// Number of tokens consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Token capacity adopted at session creation.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Adopted specification.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Backend-specific detail of the last failure, when there is one.
    pub fn backend_diagnostic(&self) -> Option<String> {
        self.backend.diagnostic()
    }

    /// Pages used by one attention layer, useful for scheduler diagnostics.
    pub fn pages_used(&self, layer: usize) -> Option<usize> {
        self.kv.get(layer)?.as_ref().map(PagedKvCache::pages_used)
    }

    /// Forget the sequence: release every page, zero the conv history and
    /// reset the state the backend keeps on the device (CONV rings, counters).
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        self.backend.reset()?;
        let kv_dim = self.spec.dims.kv_dim();
        for cache in self.kv.iter_mut().flatten() {
            *cache = PagedKvCache::new(kv_dim, self.max_context)?;
        }
        for state in self.conv.iter_mut().flatten() {
            state.fill(0.0);
        }
        self.position = 0;
        Ok(())
    }

    /// Feed a prompt, returning the logits of its last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        self.prefill_with(tokens, &mut |_, _| {})
    }

    /// Feed a prompt, reporting the residual stream after every layer.
    ///
    /// The hook sees each layer's output in order; it exists so a device run
    /// can dump the same tensors the host oracle produces and bisect the first
    /// layer that diverges.
    pub fn prefill_with(
        &mut self,
        tokens: &[u32],
        on_layer: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>, RuntimeError> {
        if tokens.is_empty() {
            return Err(RuntimeError::InvalidArgument(
                "prompt must contain at least one token".to_owned(),
            ));
        }
        if self.position + tokens.len() > self.max_context {
            return Err(RuntimeError::ContextOverflow {
                position: self.position + tokens.len(),
                capacity: self.max_context,
            });
        }
        let mut logits = Vec::new();
        for token in tokens {
            logits = self.step_with(*token, on_layer)?;
        }
        Ok(logits)
    }

    /// Feed one decoded token, returning the next logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, RuntimeError> {
        self.step(token)
    }

    /// Feed one decoded token, reporting the residual stream after every layer.
    pub fn decode_with(
        &mut self,
        token: u32,
        on_layer: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>, RuntimeError> {
        self.step_with(token, on_layer)
    }

    /// Greedy or sampled generation from `prompt`.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        config: &GenerateConfig,
        on_token: impl FnMut(u32),
    ) -> Result<Generation, RuntimeError> {
        self.generate_with(prompt, config, on_token, &mut |_, _| {})
    }

    /// Generation with a per-layer hook, used by the device trace path.
    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        config: &GenerateConfig,
        mut on_token: impl FnMut(u32),
        on_layer: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Generation, RuntimeError> {
        let mut sampler = Sampler::new(config.sampling)?;
        let mut logits = self.prefill_with(prompt, on_layer)?;
        let mut tokens = Vec::with_capacity(config.max_tokens);
        let mut stop_reason = StopReason::Length;
        while tokens.len() < config.max_tokens {
            let token = sampler.sample(&mut logits)?;
            if config.eos == Some(token) {
                stop_reason = StopReason::Eos;
                break;
            }
            if config.stop.contains(&token) {
                stop_reason = StopReason::StopToken;
                break;
            }
            tokens.push(token);
            on_token(token);
            if tokens.len() == config.max_tokens {
                break;
            }
            logits = self.step_with(token, on_layer)?;
        }
        Ok(Generation {
            tokens,
            stop_reason,
        })
    }

/// Append one f32 vector to `path` when the environment variable is set.
fn dump_vector(variable: &str, values: &[f32]) {
    let Some(path) = std::env::var_os(variable) else {
        return;
    };
    use std::io::Write as _;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(&bytes);
    }
}

    fn step(&mut self, token: u32) -> Result<Vec<f32>, RuntimeError> {
        self.step_with(token, &mut |_, _| {})
    }

    fn step_with(
        &mut self,
        token: u32,
        on_layer: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>, RuntimeError> {
        if self.position >= self.max_context {
            return Err(RuntimeError::ContextOverflow {
                position: self.position,
                capacity: self.max_context,
            });
        }
        let Session {
            spec,
            backend,
            kv,
            conv,
            position,
            ..
        } = self;
        let dims = spec.dims;
        let hidden = dims.hidden;
        let position_now = *position;

        let mut x = vec![0.0f32; hidden];
        backend.embed(token, &mut x)?;
        let mut residual = vec![0.0f32; hidden];
        let mut normed = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];
        let mut merged = vec![0.0f32; hidden];

        for layer in 0..dims.layers {
            residual.copy_from_slice(&x);
            backend.rms_norm(
                NormWeights::Layer {
                    layer,
                    slot: NormSlot::Attention,
                },
                &x,
                &mut normed,
            )?;
            // `MYELIN_DUMP_NORMED=path` appends every pre-block normed input so
            // a device run can be compared with the host oracle layer by layer.
            if let Some(path) = std::env::var_os("MYELIN_DUMP_NORMED") {
                if position_now == 0 {
                    use std::io::Write as _;
                    let mut bytes = Vec::with_capacity(normed.len() * 4);
                    for value in &normed {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        let _ = file.write_all(&bytes);
                    }
                }
            }
            match spec.layers[layer] {
                LayerSpec::Attention { kind } => {
                    let start = ops::window_start(kind, position_now, dims.sliding_window);
                    let cache = kv[layer].as_mut().ok_or_else(|| {
                        RuntimeError::InvalidArgument(format!(
                            "layer {layer} is an attention layer without a KV cache"
                        ))
                    })?;
                    backend.attention_layer(
                        AttentionCall {
                            layer,
                            position: position_now,
                            start,
                            kind,
                            input: &normed,
                        },
                        cache,
                        &mut output,
                    )?;
                }
                LayerSpec::Conv { .. } => {
                    let state = conv[layer].as_mut().ok_or_else(|| {
                        RuntimeError::InvalidArgument(format!(
                            "layer {layer} is a conv layer without state"
                        ))
                    })?;
                    backend.conv_layer(layer, &normed, state, &mut output)?;
                }
                LayerSpec::Moe { .. } => {
                    return Err(RuntimeError::InvalidArgument(format!(
                        "layer {layer} is a mixture-of-experts layer"
                    )));
                }
            }
            if position_now == 0 {
                Self::dump_vector("MYELIN_DUMP_BLOCK1", &output);
            }
            backend.rms_norm(
                NormWeights::Layer {
                    layer,
                    slot: NormSlot::PostAttention,
                },
                &output,
                &mut merged,
            )?;
            for (merged, residual) in merged.iter_mut().zip(&residual) {
                *merged += residual;
            }
            backend.rms_norm(
                NormWeights::Layer {
                    layer,
                    slot: NormSlot::Ffn,
                },
                &merged,
                &mut normed,
            )?;
            backend.ffn_layer(layer, &normed, &mut output)?;
            if position_now == 0 {
                Self::dump_vector("MYELIN_DUMP_BLOCK2", &output);
            }
            backend.rms_norm(
                NormWeights::Layer {
                    layer,
                    slot: NormSlot::PostFfn,
                },
                &output,
                &mut normed,
            )?;
            for (merged, normalized) in merged.iter_mut().zip(&normed) {
                *merged += normalized;
            }
            std::mem::swap(&mut x, &mut merged);
            on_layer(layer, &x);
        }

        backend.rms_norm(NormWeights::Output, &x, &mut normed)?;
        let mut logits = vec![0.0f32; dims.vocab];
        backend.lm_head(&normed, &mut logits)?;
        *position += 1;
        Ok(logits)
    }
}
