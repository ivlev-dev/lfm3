use std::cell::RefCell;
use std::collections::VecDeque;

use anyhow::{anyhow, Result};
use lfm_app::{
    CacheStats, FinishReason, Generate, InferenceBackend, SequenceStep, TokenizerPort,
};

struct FakeTokenizer {
    eos: u32,
}

impl TokenizerPort for FakeTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Ok(ids
            .iter()
            .map(|&id| char::from_u32(id).unwrap_or('?'))
            .collect())
    }

    fn eos_id(&self) -> u32 {
        self.eos
    }
}

struct FakeBackend {
    logits: RefCell<VecDeque<Vec<f32>>>,
    steps: Vec<SequenceStep>,
    prefilled: Vec<(usize, Vec<u32>)>,
    created: usize,
    freed: Vec<usize>,
    fail_step: bool,
}

impl FakeBackend {
    fn new(logits: Vec<Vec<f32>>) -> Self {
        Self {
            logits: RefCell::new(logits.into()),
            steps: Vec::new(),
            prefilled: Vec::new(),
            created: 0,
            freed: Vec::new(),
            fail_step: false,
        }
    }
}

impl InferenceBackend for FakeBackend {
    fn create_sequence(&mut self) -> Result<usize> {
        self.created += 1;
        Ok(9 + self.created)
    }

    fn free_sequence(&mut self, slot: usize) -> Result<()> {
        self.freed.push(slot);
        Ok(())
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], _need_logits: bool) -> Result<usize> {
        self.prefilled.push((slot, tokens.to_vec()));
        Ok(0)
    }

    fn prefill_begin(&mut self, _slot: usize, _tokens: &[u32]) -> Result<usize> {
        Ok(0)
    }

    fn prefill_finish(&mut self, _slot: usize) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, steps: &[SequenceStep], _need_logits: bool) -> Result<()> {
        if self.fail_step {
            return Err(anyhow!("step failed"));
        }
        self.steps.extend_from_slice(steps);
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.logits.borrow_mut().pop_front().unwrap_or_default())
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        assert_eq!(row, 0);
        Ok(self.logits.borrow_mut().pop_front().unwrap_or_default())
    }

    fn max_ctx(&self) -> usize {
        2048
    }

    fn free_pages(&self) -> usize {
        64
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

fn logit(token: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; len];
    out[token] = 1.0;
    out
}

#[test]
fn generates_greedy_tokens_and_frees_sequence() {
    let mut backend = FakeBackend::new(vec![logit(5, 12), logit(6, 12), logit(7, 12)]);
    let tokenizer = FakeTokenizer { eos: 11 };
    let outcome = Generate::new(&mut backend, &tokenizer)
        .run("hi", 3)
        .unwrap();
    assert_eq!(outcome.ids, vec![5, 6, 7]);
    assert_eq!(outcome.finish, FinishReason::Length);
    assert_eq!(outcome.text, "\u{5}\u{6}\u{7}");
    assert_eq!(backend.prefilled.len(), 1);
    assert_eq!(backend.prefilled[0].1, vec![u32::from(b'h'), u32::from(b'i')]);
    assert_eq!(
        backend.steps,
        vec![
            SequenceStep {
                slot: 10,
                token: 5,
                pos: 2
            },
            SequenceStep {
                slot: 10,
                token: 6,
                pos: 3
            }
        ]
    );
    assert_eq!(backend.freed, vec![10]);
}

#[test]
fn stops_on_eos_without_steps() {
    let mut backend = FakeBackend::new(vec![logit(11, 12)]);
    let tokenizer = FakeTokenizer { eos: 11 };
    let outcome = Generate::new(&mut backend, &tokenizer)
        .run("hi", 4)
        .unwrap();
    assert!(outcome.ids.is_empty());
    assert_eq!(outcome.finish, FinishReason::Stop);
    assert!(backend.steps.is_empty());
    assert_eq!(backend.freed, vec![10]);
}

#[test]
fn zero_max_tokens_returns_empty_length() {
    let mut backend = FakeBackend::new(vec![logit(5, 12)]);
    let tokenizer = FakeTokenizer { eos: 11 };
    let outcome = Generate::new(&mut backend, &tokenizer).run("hi", 0).unwrap();
    assert!(outcome.ids.is_empty());
    assert_eq!(outcome.finish, FinishReason::Length);
    assert!(backend.steps.is_empty());
    assert_eq!(backend.freed, vec![10]);
}

#[test]
fn backend_error_frees_sequence() {
    let mut backend = FakeBackend::new(vec![logit(5, 12), logit(6, 12)]);
    backend.fail_step = true;
    let tokenizer = FakeTokenizer { eos: 11 };
    let result = Generate::new(&mut backend, &tokenizer).run("hi", 4);
    assert!(result.is_err());
    assert_eq!(backend.freed, vec![10]);
}

#[test]
fn empty_prompt_fails_before_sequence() {
    let mut backend = FakeBackend::new(vec![logit(5, 12)]);
    let tokenizer = FakeTokenizer { eos: 11 };
    let result = Generate::new(&mut backend, &tokenizer).run("", 4);
    assert!(result.is_err());
    assert_eq!(backend.created, 0);
    assert!(backend.freed.is_empty());
}
