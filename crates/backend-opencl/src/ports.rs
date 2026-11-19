use anyhow::Result;

use lfm_app::{CacheStats, InferenceBackend, SequenceStep};

use crate::{GpuModel, StepReq};

impl InferenceBackend for GpuModel {
    fn create_sequence(&mut self) -> Result<usize> {
        GpuModel::create_sequence(self)
    }

    fn free_sequence(&mut self, slot: usize) -> Result<()> {
        GpuModel::free_sequence(self, slot)
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], need_logits: bool) -> Result<usize> {
        GpuModel::prefill(self, slot, tokens, need_logits)
    }

    fn prefill_begin(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        GpuModel::prefill_begin(self, slot, tokens)
    }

    fn prefill_finish(&mut self, slot: usize) -> Result<()> {
        GpuModel::prefill_finish(self, slot)
    }

    fn prefill_batch_size(&self) -> usize {
        GpuModel::prefill_batch_size(self)
    }

    fn max_sequences(&self) -> usize {
        GpuModel::max_sequences(self)
    }

    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()> {
        // Один токен с логитами — это decode: идём быстрым путём (свои ядра
        // GEMV на один токен), а не через батчевый `step_mixed`.
        if need_logits && steps.len() == 1 {
            let s = steps[0];
            self.forward_seq(s.slot, s.token, s.pos)?;
            self.logits_to_batch_row()?;
            return Ok(());
        }
        let reqs: Vec<StepReq> = steps
            .iter()
            .map(|step| StepReq {
                slot: step.slot,
                token: step.token,
                pos: step.pos,
            })
            .collect();
        self.step_mixed(&reqs, need_logits)
    }

    fn logits(&self) -> Result<Vec<f32>> {
        GpuModel::logits(self)
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        GpuModel::logits_row(self, row)
    }

    fn max_ctx(&self) -> usize {
        GpuModel::max_ctx(self)
    }

    fn free_pages(&self) -> usize {
        GpuModel::free_pages(self)
    }

    fn cache_stats(&self) -> CacheStats {
        let (cached_pages, used_pages, hits) = GpuModel::cache_stats(self);
        CacheStats {
            cached_pages,
            used_pages,
            hits,
        }
    }
}
