use std::path::Path;

use anyhow::{anyhow, Result};
use lfm_app::{CacheStats, InferenceBackend, SequenceStep};
use lfm_domain::{LayerKind, ModelConfig};
use lfm_gguf::GgufFile;

use crate::{GpuModel, TpShard};

pub struct TpModel {
    stages: Vec<GpuModel>,
    cfg: ModelConfig,
    row_logits: Vec<Vec<f32>>,
    last_logits: Vec<f32>,
}

impl TpModel {
    pub fn load(path: &Path, max_ctx: usize, devices: &[usize]) -> Result<Self> {
        let world = devices.len();
        if world == 0 {
            return Err(anyhow!("no tp devices"));
        }
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let kv_heads = *cfg.kv_heads.iter().max().unwrap_or(&0);
        if cfg.num_heads % world != 0 || kv_heads % world != 0 {
            return Err(anyhow!("heads not divisible by tp world"));
        }
        let mut stages = Vec::with_capacity(world);
        for (rank, &device) in devices.iter().enumerate() {
            let shard = TpShard::full(
                rank,
                world,
                cfg.num_heads,
                kv_heads,
                cfg.head_dim,
                cfg.ffn_size,
                cfg.vocab_size,
            );
            stages.push(GpuModel::load_stage(
                path,
                max_ctx,
                Some(device),
                None,
                Some(shard),
            )?);
        }
        Ok(Self {
            stages,
            cfg,
            row_logits: Vec::new(),
            last_logits: Vec::new(),
        })
    }

    pub fn stages(&self) -> &[GpuModel] {
        &self.stages
    }

    fn sync_x(&mut self) -> Result<()> {
        if self.stages.len() <= 1 {
            return Ok(());
        }
        let (x, hq, hs) = self.stages[0].tp_norm_inputs()?;
        for stage in self.stages.iter_mut().skip(1) {
            stage.tp_set_norm_inputs(&x, &hq, &hs)?;
        }
        Ok(())
    }

    fn step_token(
        &mut self,
        slot: usize,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let h = self.cfg.hidden_size;
        self.stages[0].tp_embed(token)?;
        self.stages[0].tp_norm_attn(0)?;
        self.sync_x()?;
        for layer in 0..self.cfg.num_layers {
            let attn = if self.cfg.kind(layer) == LayerKind::Conv {
                self.stages[0].tp_conv(layer, slot, pos)?
            } else {
                let mut tmp = vec![0.0f32; h];
                for stage in self.stages.iter_mut() {
                    let part = stage.tp_attn(layer, slot, pos)?;
                    for (dst, src) in tmp.iter_mut().zip(part.iter()) {
                        *dst += *src;
                    }
                }
                tmp
            };
            for stage in self.stages.iter_mut().rev() {
                stage.tp_attn_norm(layer, &attn)?;
            }
            self.sync_x()?;
            let mut ffn = vec![0.0f32; h];
            for stage in self.stages.iter_mut() {
                let part = stage.tp_ffn_local(layer)?;
                for (dst, src) in ffn.iter_mut().zip(part.iter()) {
                    *dst += *src;
                }
            }
            for stage in self.stages.iter_mut().rev() {
                stage.tp_ffn_norm(layer, &ffn)?;
            }
            self.sync_x()?;
        }
        if want_logits {
            let mut logits = vec![0.0f32; self.cfg.vocab_size];
            for stage in self.stages.iter_mut() {
                let shard = stage
                    .tp_shard()
                    .ok_or_else(|| anyhow!("stage without tp shard"))?;
                let part = stage.tp_lmhead_partial()?;
                let end = (shard.vocab_start + part.len()).min(logits.len());
                logits[shard.vocab_start..end].copy_from_slice(&part[..end - shard.vocab_start]);
            }
            Ok(Some(logits))
        } else {
            Ok(None)
        }
    }
}

impl InferenceBackend for TpModel {
    fn create_sequence(&mut self) -> Result<usize> {
        let mut slots = Vec::with_capacity(self.stages.len());
        for stage in self.stages.iter_mut() {
            slots.push(stage.create_sequence()?);
        }
        if slots.windows(2).any(|w| w[0] != w[1]) {
            return Err(anyhow!("tp stages allocated different slots"));
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
        let reused = self.prefill_begin(slot, tokens)?;
        let mut last = Vec::new();
        for (i, &token) in tokens.iter().enumerate().skip(reused) {
            let want = need_logits && i + 1 == tokens.len();
            if let Some(logits) = self.step_token(slot, token, i, want)? {
                last = logits;
            }
        }
        self.prefill_finish(slot)?;
        self.last_logits = last;
        Ok(reused)
    }

    fn prefill_begin(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        let mut reused = self.stages[0].prefill_begin(slot, tokens)?;
        for stage in self.stages.iter_mut().skip(1) {
            let r = stage.prefill_begin(slot, tokens)?;
            if r != reused {
                return Err(anyhow!("tp cache mismatch: {r} != {reused}"));
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
        self.row_logits.clear();
        for step in steps {
            let logits = self.step_token(step.slot, step.token, step.pos, need_logits)?;
            if let Some(logits) = logits {
                self.last_logits = logits.clone();
                self.row_logits.push(logits);
            } else {
                self.row_logits.push(Vec::new());
            }
        }
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.last_logits.clone())
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        self.row_logits
            .get(row)
            .filter(|v| !v.is_empty())
            .cloned()
            .ok_or_else(|| anyhow!("no tp logits for row {row}"))
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
