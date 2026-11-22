//! Generation-loop tests against a scripted backend (no checkpoint needed):
//! stop conditions, KV growth and reset.

use myelin_core::backend::{AttentionCall, Backend, BackendError, NormWeights};
use myelin_core::kv::KvCache;
use myelin_core::spec::{
    AttentionKind, ChatTemplateSpec, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec, Quant,
    RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
};
use myelin_core::{GenerateConfig, Model, SamplingParams, Session, StopReason};

/// Backend that emits a fixed token sequence and records cache sizes.
struct ScriptedBackend {
    spec: ModelSpec,
    script: Vec<u32>,
    calls: usize,
    attention_lengths: Vec<usize>,
    conv_states: Vec<usize>,
}

impl ScriptedBackend {
    fn new(script: Vec<u32>) -> Self {
        let head_dim = 4;
        let dims = Dims {
            hidden: 8,
            ffn: 16,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim,
            vocab: 16,
            conv_cache: 3,
            sliding_window: 4,
        };
        let spec = ModelSpec {
            family: ModelFamily::Unknown,
            dims,
            rope: RopeConfig {
                full: RopeSpec {
                    kind: RopeKind::Full,
                    theta: 10_000.0,
                    dim: head_dim,
                },
                sliding: None,
            },
            layers: vec![
                LayerSpec::Conv { ksize: 3 },
                LayerSpec::Attention {
                    kind: AttentionKind::Full,
                },
            ],
            kv_format: KvFormat::F16,
            tokenizer: TokenizerSpec {
                model: "scripted".to_owned(),
                vocab_size: dims.vocab,
                bos_token_id: None,
                eos_token_id: None,
            },
            chat_template: ChatTemplateSpec {
                template: String::new(),
            },
            weight_format: WeightFormat::from_quant(Quant::F16),
            norm_eps: 1e-5,
            tied_embeddings: true,
        };
        Self {
            spec,
            script,
            calls: 0,
            attention_lengths: Vec::new(),
            conv_states: Vec::new(),
        }
    }
}

impl Backend for ScriptedBackend {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn embed(&mut self, _token: u32, out: &mut [f32]) -> Result<(), BackendError> {
        out.fill(0.0);
        Ok(())
    }

    fn rms_norm(
        &mut self,
        _weights: NormWeights,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        out.copy_from_slice(x);
        Ok(())
    }

    fn ffn_layer(
        &mut self,
        _layer: usize,
        _x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        out.fill(0.0);
        Ok(())
    }

    fn attention_layer(
        &mut self,
        call: AttentionCall<'_>,
        kv: &mut dyn KvCache,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        self.attention_lengths.push(kv.len());
        assert_eq!(kv.len(), call.position);
        kv.append(&vec![0.0; kv.kv_dim()], &vec![0.0; kv.kv_dim()])?;
        out.fill(0.0);
        Ok(())
    }

    fn conv_layer(
        &mut self,
        _layer: usize,
        _x: &[f32],
        state: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        self.conv_states.push(state.len());
        out.fill(0.0);
        Ok(())
    }

    fn lm_head(&mut self, _x: &[f32], logits: &mut [f32]) -> Result<(), BackendError> {
        let token = self
            .script
            .get(self.calls)
            .copied()
            .unwrap_or(0)
            .min(self.spec.dims.vocab as u32 - 1);
        self.calls += 1;
        logits.fill(0.0);
        logits[token as usize] = 10.0;
        Ok(())
    }
}

fn scripted_model(script: Vec<u32>) -> Model {
    let backend = ScriptedBackend::new(script);
    Model::from_backend(backend.spec.clone(), Box::new(backend)).expect("model")
}

#[test]
fn generation_stops_at_token_limit() {
    let mut model = scripted_model(vec![3; 8]);
    let mut session = model.session(16).expect("session");
    let config = GenerateConfig {
        max_tokens: 4,
        sampling: SamplingParams::greedy(),
        eos: None,
        stop: Vec::new(),
    };
    let generation = session.generate(&[1], &config, |_| {}).expect("generate");
    assert_eq!(generation.tokens, vec![3, 3, 3, 3]);
    assert_eq!(generation.stop_reason, StopReason::Length);
    assert_eq!(
        session.position(),
        4,
        "one prompt token + three decoded steps"
    );
}

#[test]
fn generation_stops_at_eos_without_emitting_it() {
    let mut model = scripted_model(vec![3, 7, 9]);
    let mut session = model.session(16).expect("session");
    let config = GenerateConfig {
        max_tokens: 8,
        sampling: SamplingParams::greedy(),
        eos: Some(7),
        stop: Vec::new(),
    };
    let generation = session.generate(&[1], &config, |_| {}).expect("generate");
    assert_eq!(generation.tokens, vec![3]);
    assert_eq!(generation.stop_reason, StopReason::Eos);
}

#[test]
fn generation_stops_at_explicit_stop_token() {
    let mut model = scripted_model(vec![5, 9, 3]);
    let mut session = model.session(16).expect("session");
    let config = GenerateConfig {
        max_tokens: 8,
        sampling: SamplingParams::greedy(),
        eos: None,
        stop: vec![9],
    };
    let generation = session.generate(&[1], &config, |_| {}).expect("generate");
    assert_eq!(generation.tokens, vec![5]);
    assert_eq!(generation.stop_reason, StopReason::StopToken);
}

#[test]
fn session_tracks_kv_pages_and_reset_releases_them() {
    let mut model = scripted_model(vec![3; 64]);
    let mut session: Session<'_> = model.session(64).expect("session");
    assert_eq!(session.pages_used(1), Some(0));
    let config = GenerateConfig {
        max_tokens: 40,
        sampling: SamplingParams::greedy(),
        eos: None,
        stop: Vec::new(),
    };
    session
        .generate(&[1, 2], &config, |_| {})
        .expect("generate");
    let pages = session.pages_used(1).expect("attention layer");
    assert_eq!(pages, (42usize).div_ceil(myelin_core::PAGE_TOKENS));
    session.reset().expect("reset");
    assert_eq!(session.position(), 0);
    assert_eq!(session.pages_used(1), Some(0));
}

#[test]
fn prefill_rejects_empty_prompt_and_context_overflow() {
    let mut model = scripted_model(vec![3; 8]);
    let mut session = model.session(4).expect("session");
    assert!(session.prefill(&[]).is_err());
    let long = vec![1u32; 5];
    assert!(session.prefill(&long).is_err());
}

#[test]
fn attention_call_sees_monotonic_positions() {
    let mut model = scripted_model(vec![3; 8]);
    let mut session = model.session(16).expect("session");
    session.prefill(&[1, 2, 3]).expect("prefill");
    session.decode(4).expect("decode");
    // The backend is owned by the model, so this test checks the observable
    // side: after four tokens the second layer holds four entries.
    assert_eq!(session.pages_used(1), Some(1));
    assert_eq!(session.position(), 4);
}

#[test]
fn dsp_backend_is_selectable_but_fails_typed() {
    let spec = ScriptedBackend::new(vec![0]).spec.clone();
    let mut model = Model::from_backend(spec.clone(), Box::new(myelin_core::DspBackend::new(spec)))
        .expect("model");
    let mut session = model.session(8).expect("session");
    let error = session.prefill(&[1]).expect_err("dsp cannot run");
    match error {
        myelin_core::RuntimeError::Backend(BackendError::Unsupported { backend, .. }) => {
            assert_eq!(backend, "dsp");
        }
        other => panic!("unexpected error {other:?}"),
    }
}
