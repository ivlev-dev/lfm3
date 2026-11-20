use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use lfm_app::{InferenceBackend, Sampler, SequenceStep};
use lfm_domain::SamplingParams;
use lfm_tokenizer::LfmTokenizer;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub ttft_s: f64,
}

impl Stats {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_s.max(1e-9)
    }

    pub fn decode_tok_s(&self) -> f64 {
        self.completion_tokens.saturating_sub(1) as f64 / self.decode_s.max(1e-9)
    }

    pub fn total_s(&self) -> f64 {
        self.prefill_s + self.decode_s
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_stream(
    backend: &mut dyn InferenceBackend,
    tokenizer: &LfmTokenizer,
    ids: &[u32],
    params: &SamplingParams,
    max_new: usize,
    cancel: &AtomicBool,
    mut on_token: impl FnMut(u32),
) -> Result<Stats> {
    if max_new == 0 {
        bail!("max_tokens должен быть положительным");
    }
    let slot = backend.create_sequence()?;
    let mut stats = Stats {
        prompt_tokens: ids.len(),
        ..Stats::default()
    };
    let result = (|| -> Result<()> {
        let mut sampler = Sampler::new(params.clone());
        for &t in ids {
            sampler.push(t);
        }
        let started = Instant::now();
        backend.prefill(slot, ids, true)?;
        let mut logits = backend.logits()?;
        stats.prefill_s = started.elapsed().as_secs_f64();
        let mut pos = ids.len();
        let mut first: Option<Instant> = None;
        for step in 0..max_new {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let token = sampler.sample(&mut logits);
            if token == tokenizer.eos_id() {
                break;
            }
            sampler.push(token);
            if first.is_none() {
                first = Some(Instant::now());
                stats.ttft_s = started.elapsed().as_secs_f64();
            }
            stats.completion_tokens += 1;
            on_token(token);
            if step + 1 == max_new {
                break;
            }
            backend.step(&[SequenceStep { slot, token, pos }], true)?;
            logits = backend.logits_row(0)?;
            pos += 1;
        }
        if let Some(first) = first {
            stats.decode_s = first.elapsed().as_secs_f64();
        }
        Ok(())
    })();
    let freed = backend.free_sequence(slot);
    result.and(freed)?;
    Ok(stats)
}
