use anyhow::{Result, anyhow};

use crate::argmax;
use crate::ports::{InferenceBackend, SequenceStep, TokenizerPort};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutcome {
    pub ids: Vec<u32>,
    pub text: String,
    pub finish: FinishReason,
}

pub struct Generate<'a> {
    backend: &'a mut dyn InferenceBackend,
    tokenizer: &'a dyn TokenizerPort,
}

impl<'a> Generate<'a> {
    pub fn new(backend: &'a mut dyn InferenceBackend, tokenizer: &'a dyn TokenizerPort) -> Self {
        Self { backend, tokenizer }
    }

    pub fn run(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerateOutcome> {
        if prompt.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        let prompt_ids = self.tokenizer.encode(prompt, false)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("prompt tokenized to nothing"));
        }
        let slot = self.backend.create_sequence()?;
        let result = self.generate(slot, &prompt_ids, max_tokens);
        let freed = self.backend.free_sequence(slot);
        match (result, freed) {
            (Ok(outcome), Ok(())) => Ok(outcome),
            (Err(e), _) => Err(e),
            (Ok(_), Err(e)) => Err(e),
        }
    }

    fn generate(
        &mut self,
        slot: usize,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Result<GenerateOutcome> {
        self.backend.prefill(slot, prompt, true)?;
        let mut logits = self.backend.logits()?;
        let mut ids = Vec::with_capacity(max_tokens);
        let mut finish = FinishReason::Length;
        while ids.len() < max_tokens {
            let token = argmax(&logits);
            if token == self.tokenizer.eos_id() {
                finish = FinishReason::Stop;
                break;
            }
            ids.push(token);
            if ids.len() == max_tokens {
                break;
            }
            let pos = prompt.len() + ids.len() - 1;
            self.backend
                .step(&[SequenceStep { slot, token, pos }], true)?;
            logits = self.backend.logits_row(0)?;
        }
        let text = self.tokenizer.decode(&ids, true)?;
        Ok(GenerateOutcome { ids, text, finish })
    }
}
