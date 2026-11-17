use anyhow::Result;

use lfm_app::{InferenceBackend, SequenceStep};

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

    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()> {
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
}
