use std::path::Path;

use anyhow::{anyhow, Result};
use lfm_app::{CacheStats, InferenceBackend, SequenceStep};
use lfm_domain::ModelConfig;
use lfm_gguf::GgufFile;

use crate::{GpuModel, StepReq};

pub struct PipelineModel {
    stages: Vec<GpuModel>,
    hidden: usize,
}

impl PipelineModel {
    pub fn load_tiers(
        path: &Path,
        max_ctx: usize,
        tiers: &[(usize, usize)],
    ) -> Result<Self> {
        let total: usize = tiers.iter().map(|(_, percent)| *percent).sum();
        if total != 100 {
            return Err(anyhow!("tier percentages must sum to 100"));
        }
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let last_nonzero = tiers
            .iter()
            .rposition(|&(_, percent)| percent > 0)
            .unwrap_or(0);
        for (i, &(device, percent)) in tiers.iter().enumerate() {
            if percent == 0 {
                continue;
            }
            let end = if i == last_nonzero {
                cfg.num_layers
            } else {
                start + cfg.num_layers * percent / 100
            };
            ranges.push((device, start, end));
            start = end;
        }
        let mut stages = Vec::new();
        for &(device, from, to) in &ranges {
            if from >= to {
                continue;
            }
            stages.push(GpuModel::load_stage(path, max_ctx, Some(device), Some((from, to)), None)?);
        }
        if stages.is_empty() {
            return Err(anyhow!("no pipeline stages"));
        }
        Ok(Self {
            stages,
            hidden: cfg.hidden_size,
        })
    }

    pub fn stages(&self) -> &[GpuModel] {
        &self.stages
    }

    fn last(&self) -> &GpuModel {
        self.stages.last().expect("pipeline has stages")
    }

    fn last_mut(&mut self) -> &mut GpuModel {
        self.stages.last_mut().expect("pipeline has stages")
    }

    fn transfer(
        &mut self,
        steps: &[StepReq],
        mut hidden: Vec<f32>,
        need_logits: bool,
    ) -> Result<()> {
        let n = self.stages.len();
        for (i, stage) in self.stages.iter_mut().enumerate().skip(1) {
            let last = i + 1 == n;
            stage.step_mixed_input(steps, &hidden, need_logits && last)?;
            if !last {
                hidden = stage.stage_hidden_batch(steps.len())?;
            }
        }
        Ok(())
    }
}

impl InferenceBackend for PipelineModel {
    fn create_sequence(&mut self) -> Result<usize> {
        let mut slots = Vec::with_capacity(self.stages.len());
        for stage in self.stages.iter_mut() {
            match stage.create_sequence() {
                Ok(slot) => slots.push(slot),
                Err(e) => {
                    for (stage, slot) in self.stages.iter_mut().zip(slots.iter()) {
                        let _ = stage.free_sequence(*slot);
                    }
                    return Err(e);
                }
            }
        }
        if slots.windows(2).any(|w| w[0] != w[1]) {
            for (stage, slot) in self.stages.iter_mut().zip(slots.iter()) {
                let _ = stage.free_sequence(*slot);
            }
            return Err(anyhow!("pipeline stages allocated different slots"));
        }
        Ok(slots[0])
    }

    fn free_sequence(&mut self, slot: usize) -> Result<()> {
        let mut result = Ok(());
        for stage in self.stages.iter_mut() {
            if let Err(e) = stage.free_sequence(slot) {
                result = Err(e);
            }
        }
        result
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], need_logits: bool) -> Result<usize> {
        let mut reused = self.stages[0].prefill_begin(slot, tokens)?;
        for stage in self.stages.iter_mut().skip(1) {
            let r = stage.prefill_begin(slot, tokens)?;
            if r != reused {
                return Err(anyhow!("pipeline cache mismatch: {r} != {reused}"));
            }
        }
        reused = reused.min(tokens.len().saturating_sub(1));
        let pb = self.stages[0].batch_size();
        let mut pos = reused;
        let mut hidden: Vec<f32> = Vec::new();
        while pos < tokens.len() {
            let end = (pos + pb).min(tokens.len());
            let last_chunk = end == tokens.len();
            let n = self.stages.len();
            self.stages[0].forward_batch_seq(
                slot,
                &tokens[pos..end],
                pos,
                need_logits && n == 1 && last_chunk,
            )?;
            if n > 1 {
                hidden = self.stages[0].stage_hidden_batch(end - pos)?;
            }
            for (i, stage) in self.stages.iter_mut().enumerate().skip(1) {
                let last = i + 1 == n;
                stage.forward_batch_hidden(
                    slot,
                    &hidden,
                    pos,
                    need_logits && last && last_chunk,
                )?;
                if !last {
                    hidden = stage.stage_hidden_batch(end - pos)?;
                }
            }
            pos = end;
            if last_chunk {
                break;
            }
        }
        for stage in self.stages.iter_mut() {
            stage.prefill_finish(slot)?;
        }
        Ok(reused)
    }

    fn prefill_begin(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        let mut reused = self.stages[0].prefill_begin(slot, tokens)?;
        for stage in self.stages.iter_mut().skip(1) {
            let r = stage.prefill_begin(slot, tokens)?;
            if r != reused {
                return Err(anyhow!("pipeline cache mismatch: {r} != {reused}"));
            }
            reused = reused.min(r);
        }
        Ok(reused)
    }

    fn prefill_finish(&mut self, slot: usize) -> Result<()> {
        for stage in self.stages.iter_mut() {
            stage.prefill_finish(slot)?;
        }
        Ok(())
    }

    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()> {
        if steps.is_empty() {
            return Err(anyhow!("empty pipeline step"));
        }
        let reqs: Vec<StepReq> = steps
            .iter()
            .map(|step| StepReq {
                slot: step.slot,
                token: step.token,
                pos: step.pos,
            })
            .collect();
        if self.stages.len() == 1 {
            return self.stages[0].step_mixed(&reqs, need_logits);
        }
        self.stages[0].step_mixed(&reqs, false)?;
        let hidden = self.stages[0].stage_hidden_batch(steps.len())?;
        self.transfer(&reqs, hidden, need_logits)
    }

    fn logits(&self) -> Result<Vec<f32>> {
        self.last().logits()
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        self.last().logits_row(row)
    }

    fn max_ctx(&self) -> usize {
        self.stages[0].max_ctx()
    }

    fn free_pages(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.free_pages())
            .min()
            .unwrap_or(0)
    }

    fn cache_stats(&self) -> CacheStats {
        let (cached_pages, used_pages, hits) = self.stages[0].cache_stats();
        CacheStats {
            cached_pages,
            used_pages,
            hits,
        }
    }
}

impl PipelineModel {
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn last_stage_mut(&mut self) -> &mut GpuModel {
        self.last_mut()
    }
}
