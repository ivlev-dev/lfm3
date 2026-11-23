use std::fmt;

use crate::sampling::{Sampler, SamplingParams};
use crate::session::Model;
use crate::RuntimeError;

pub const DEFAULT_CHUNK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchEvent {
    Token { id: SeqId, token: u32 },
    Done { id: SeqId, tokens: Vec<u32> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    Session(RuntimeError),
    Empty,
    InvalidRequest(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(error) => write!(f, "session: {error}"),
            Self::Empty => write!(f, "prompt is empty"),
            Self::InvalidRequest(reason) => write!(f, "invalid request: {reason}"),
        }
    }
}

impl std::error::Error for BatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Session(error) => Some(error),
            Self::Empty | Self::InvalidRequest(_) => None,
        }
    }
}

impl From<RuntimeError> for BatchError {
    fn from(error: RuntimeError) -> Self {
        Self::Session(error)
    }
}

struct Sequence {
    id: SeqId,
    history: Vec<u32>,
    prompt_len: usize,
    generated: Vec<u32>,
    max_tokens: usize,
    capacity: usize,
    prefilled: usize,
    sampler: Sampler,
    finished: bool,
}

pub struct BatchScheduler<'m> {
    model: &'m mut Model,
    chunk: usize,
    sampling: SamplingParams,
    eos: Option<u32>,
    next_id: u64,
    sequences: Vec<Sequence>,
}

impl<'m> BatchScheduler<'m> {
    pub fn new(model: &'m mut Model) -> Self {
        let eos = model.spec().tokenizer.eos_token_id;
        Self {
            model,
            chunk: DEFAULT_CHUNK,
            sampling: SamplingParams::greedy(),
            eos,
            next_id: 0,
            sequences: Vec::new(),
        }
    }

    pub fn with_chunk(mut self, chunk: usize) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_eos(mut self, eos: Option<u32>) -> Self {
        self.eos = eos;
        self
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    pub fn active(&self) -> usize {
        self.sequences
            .iter()
            .filter(|sequence| !sequence.finished)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.active() == 0
    }

    pub fn submit(&mut self, request: SeqRequest) -> Result<SeqId, BatchError> {
        if request.prompt.is_empty() {
            return Err(BatchError::Empty);
        }
        if request.max_tokens == 0 {
            return Err(BatchError::InvalidRequest(
                "max_tokens must be positive".to_owned(),
            ));
        }
        let capacity = request
            .prompt
            .len()
            .checked_add(request.max_tokens)
            .ok_or_else(|| {
                BatchError::InvalidRequest("prompt and max_tokens overflow the context".to_owned())
            })?;
        let sampler = Sampler::new(self.sampling).map_err(BatchError::Session)?;
        let id = SeqId(self.next_id);
        self.next_id += 1;
        self.sequences.push(Sequence {
            id,
            prompt_len: request.prompt.len(),
            history: request.prompt,
            generated: Vec::new(),
            max_tokens: request.max_tokens,
            capacity,
            prefilled: 0,
            sampler,
            finished: false,
        });
        Ok(id)
    }

    pub fn cancel(&mut self, id: SeqId) -> bool {
        let before = self.sequences.len();
        self.sequences.retain(|sequence| sequence.id != id);
        self.sequences.len() != before
    }

    pub fn step(&mut self) -> Result<Vec<BatchEvent>, BatchError> {
        let Self {
            model,
            chunk,
            eos,
            sequences,
            ..
        } = self;
        let chunk = (*chunk).max(1);
        let eos = *eos;
        let mut events = Vec::new();
        for sequence in sequences.iter_mut() {
            if sequence.finished {
                continue;
            }
            if sequence.prefilled < sequence.prompt_len {
                let remaining = sequence.prompt_len - sequence.prefilled;
                sequence.prefilled += remaining.min(chunk);
                let fed = sequence.prefilled;
                let mut session = model
                    .session(sequence.capacity)
                    .map_err(BatchError::Session)?;
                for piece in sequence.history[..fed].chunks(chunk) {
                    session.prefill(piece).map_err(BatchError::Session)?;
                }
                continue;
            }
            let mut session = model
                .session(sequence.capacity)
                .map_err(BatchError::Session)?;
            let mut logits = Vec::new();
            for piece in sequence.history.chunks(chunk) {
                logits = session.prefill(piece).map_err(BatchError::Session)?;
            }
            let token = sequence
                .sampler
                .sample(&mut logits)
                .map_err(BatchError::Session)?;
            if eos == Some(token) {
                sequence.finished = true;
                events.push(BatchEvent::Done {
                    id: sequence.id,
                    tokens: std::mem::take(&mut sequence.generated),
                });
                continue;
            }
            sequence.generated.push(token);
            sequence.history.push(token);
            events.push(BatchEvent::Token {
                id: sequence.id,
                token,
            });
            if sequence.generated.len() >= sequence.max_tokens {
                sequence.finished = true;
                events.push(BatchEvent::Done {
                    id: sequence.id,
                    tokens: std::mem::take(&mut sequence.generated),
                });
            }
        }
        sequences.retain(|sequence| !sequence.finished);
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AttentionCall, Backend, BackendError, NormWeights};
    use crate::kv::KvCache;
    use crate::spec::{
        AttentionKind, ChatTemplateSpec, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec, Quant,
        RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
    };

    struct PositionBackend {
        spec: ModelSpec,
        position: usize,
    }

    impl PositionBackend {
        fn spec(eos: Option<u32>) -> ModelSpec {
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
            ModelSpec {
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
                    model: "position".to_owned(),
                    vocab_size: dims.vocab,
                    bos_token_id: None,
                    eos_token_id: eos,
                },
                chat_template: ChatTemplateSpec {
                    template: String::new(),
                },
                weight_format: WeightFormat::from_quant(Quant::F16),
                norm_eps: 1e-5,
                tied_embeddings: true,
            }
        }
    }

    impl Backend for PositionBackend {
        fn name(&self) -> &'static str {
            "position"
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
            if kv.len() != call.position {
                return Err(BackendError::Kv(format!(
                    "position {} does not match cache length {}",
                    call.position,
                    kv.len()
                )));
            }
            kv.append(&vec![0.0; kv.kv_dim()], &vec![0.0; kv.kv_dim()])?;
            self.position = call.position;
            out.fill(0.0);
            Ok(())
        }

        fn conv_layer(
            &mut self,
            _layer: usize,
            _x: &[f32],
            _state: &mut [f32],
            out: &mut [f32],
        ) -> Result<(), BackendError> {
            out.fill(0.0);
            Ok(())
        }

        fn lm_head(&mut self, _x: &[f32], logits: &mut [f32]) -> Result<(), BackendError> {
            let vocab = self.spec.dims.vocab;
            let token = ((self.position + 1) % vocab) as u32;
            logits.fill(0.0);
            logits[token as usize] = 10.0;
            Ok(())
        }
    }

    fn position_model(eos: Option<u32>) -> Model {
        let spec = PositionBackend::spec(eos);
        Model::from_backend(
            spec.clone(),
            Box::new(PositionBackend { spec, position: 0 }),
        )
        .expect("model")
    }

    fn run(requests: &[(Vec<u32>, usize)]) -> Vec<BatchEvent> {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model);
        for (prompt, max_tokens) in requests {
            scheduler
                .submit(SeqRequest {
                    prompt: prompt.clone(),
                    max_tokens: *max_tokens,
                })
                .expect("submit");
        }
        let mut events = Vec::new();
        for _ in 0..64 {
            if scheduler.is_empty() {
                break;
            }
            events.extend(scheduler.step().expect("step"));
        }
        events
    }

    #[test]
    fn empty_scheduler_steps_to_nothing() {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model);
        assert_eq!(scheduler.active(), 0);
        assert!(scheduler.is_empty());
        assert!(scheduler.step().expect("step").is_empty());
        assert!(scheduler.step().expect("step").is_empty());
    }

    #[test]
    fn two_sequences_of_different_length_finish_by_limit() {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model);
        let short = scheduler
            .submit(SeqRequest {
                prompt: vec![1, 2],
                max_tokens: 3,
            })
            .expect("submit");
        let long = scheduler
            .submit(SeqRequest {
                prompt: vec![1, 2, 3, 4, 5],
                max_tokens: 5,
            })
            .expect("submit");
        let mut short_tokens = Vec::new();
        let mut long_tokens = Vec::new();
        let mut short_done = None;
        let mut long_done = None;
        for _ in 0..32 {
            for event in scheduler.step().expect("step") {
                match event {
                    BatchEvent::Token { id, token } if id == short => short_tokens.push(token),
                    BatchEvent::Token { id, token } if id == long => long_tokens.push(token),
                    BatchEvent::Done { id, tokens } if id == short => short_done = Some(tokens),
                    BatchEvent::Done { id, tokens } if id == long => long_done = Some(tokens),
                    _ => {}
                }
            }
            if short_done.is_some() && long_done.is_some() {
                break;
            }
        }
        assert_eq!(short_tokens, vec![2, 3, 4]);
        assert_eq!(short_done, Some(vec![2, 3, 4]));
        assert_eq!(long_tokens, vec![5, 6, 7, 8, 9]);
        assert_eq!(long_done, Some(vec![5, 6, 7, 8, 9]));
        assert_eq!(scheduler.active(), 0);
    }

    #[test]
    fn cancelling_one_sequence_keeps_the_other_running() {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model);
        let removed = scheduler
            .submit(SeqRequest {
                prompt: vec![1, 2],
                max_tokens: 4,
            })
            .expect("submit");
        let kept = scheduler
            .submit(SeqRequest {
                prompt: vec![3],
                max_tokens: 2,
            })
            .expect("submit");
        scheduler.step().expect("prefill step");
        assert_eq!(scheduler.active(), 2);
        assert!(scheduler.cancel(removed));
        assert!(!scheduler.cancel(removed));
        assert_eq!(scheduler.active(), 1);
        let mut tokens = Vec::new();
        for _ in 0..8 {
            for event in scheduler.step().expect("step") {
                match event {
                    BatchEvent::Token { id, token } => {
                        assert_eq!(id, kept);
                        tokens.push(token);
                    }
                    BatchEvent::Done { id, tokens: done } => {
                        assert_eq!(id, kept);
                        assert_eq!(done, vec![1, 2]);
                    }
                }
            }
        }
        assert_eq!(tokens, vec![1, 2]);
        assert_eq!(scheduler.active(), 0);
    }

    #[test]
    fn identical_requests_produce_identical_events() {
        let requests = vec![(vec![1, 2, 3], 4), (vec![4, 5], 3)];
        assert_eq!(run(&requests), run(&requests));
        assert!(!run(&requests).is_empty());
    }

    #[test]
    fn chunked_prefill_delays_the_first_token() {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model).with_chunk(2);
        let id = scheduler
            .submit(SeqRequest {
                prompt: vec![1, 2, 3, 4, 5],
                max_tokens: 1,
            })
            .expect("submit");
        let mut tokens = Vec::new();
        let mut first_token_step = None;
        for step_index in 0..8 {
            for event in scheduler.step().expect("step") {
                if let BatchEvent::Token {
                    id: token_id,
                    token,
                } = event
                {
                    assert_eq!(token_id, id);
                    if first_token_step.is_none() {
                        first_token_step = Some(step_index);
                    }
                    tokens.push(token);
                }
            }
        }
        assert_eq!(first_token_step, Some(3));
        assert_eq!(tokens, vec![5]);
        assert_eq!(scheduler.active(), 0);
    }

    #[test]
    fn eos_from_spec_stops_generation() {
        let mut model = position_model(Some(3));
        let mut scheduler = BatchScheduler::new(&mut model);
        let id = scheduler
            .submit(SeqRequest {
                prompt: vec![1],
                max_tokens: 8,
            })
            .expect("submit");
        let mut tokens = Vec::new();
        let mut done = None;
        for _ in 0..8 {
            for event in scheduler.step().expect("step") {
                match event {
                    BatchEvent::Token {
                        id: token_id,
                        token,
                    } => {
                        assert_eq!(token_id, id);
                        tokens.push(token);
                    }
                    BatchEvent::Done {
                        id: token_id,
                        tokens: all,
                    } => {
                        assert_eq!(token_id, id);
                        done = Some(all);
                    }
                }
            }
            if done.is_some() {
                break;
            }
        }
        assert_eq!(tokens, vec![1, 2]);
        assert_eq!(done, Some(vec![1, 2]));
        assert_eq!(scheduler.active(), 0);
    }

    #[test]
    fn submit_rejects_empty_prompt_and_zero_limit() {
        let mut model = position_model(None);
        let mut scheduler = BatchScheduler::new(&mut model);
        assert_eq!(
            scheduler.submit(SeqRequest {
                prompt: Vec::new(),
                max_tokens: 4,
            }),
            Err(BatchError::Empty)
        );
        assert!(matches!(
            scheduler.submit(SeqRequest {
                prompt: vec![1],
                max_tokens: 0,
            }),
            Err(BatchError::InvalidRequest(_))
        ));
        assert_eq!(scheduler.active(), 0);
    }
}
