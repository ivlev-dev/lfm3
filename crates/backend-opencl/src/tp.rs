use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
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

/// Группы подряд идущих шагов одного слота (непрерывные позиции).
pub fn group_runs(steps: &[SequenceStep]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < steps.len() {
        let mut j = i + 1;
        while j < steps.len()
            && steps[j].slot == steps[i].slot
            && steps[j].pos == steps[i].pos + (j - i)
        {
            j += 1;
        }
        runs.push((i, j));
        i = j;
    }
    runs
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
        if kv_heads == 0 || cfg.num_heads % kv_heads != 0 {
            return Err(anyhow!("num_heads must be a multiple of kv_heads"));
        }
        if world > kv_heads {
            return Err(anyhow!(
                "tp world {world} exceeds kv heads {kv_heads}: some stage would have no heads"
            ));
        }
        if cfg.ffn_size % world != 0 || (cfg.ffn_size / world) % 256 != 0 {
            return Err(anyhow!(
                "ffn {} cannot be split into {world} column shards (need multiple of 256)",
                cfg.ffn_size
            ));
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

    fn sync_x_b(&mut self, ntok: usize) -> Result<()> {
        if self.stages.len() <= 1 {
            return Ok(());
        }
        let (x, hq, hs) = self.stages[0].tp_norm_inputs_b(ntok)?;
        for stage in self.stages.iter_mut().skip(1) {
            stage.tp_set_norm_inputs_b(&x, &hq, &hs, ntok)?;
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
    fn prefill_chunk(
        &mut self,
        slot: usize,
        pos: usize,
        chunk: &[u32],
        need_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let h = self.cfg.hidden_size;
        let ntok = chunk.len();
        let mut want_logits: Option<Vec<f32>> = None;
        {
            let dbg = std::env::var("LFM_TP_DEBUG").is_ok();
            let (mut t_attn, mut t_ffn, mut t_sync, mut t_prep) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let t0 = Instant::now();
            for stage in self.stages.iter_mut() {
                stage.tp_prepare_batch(slot, pos, ntok)?;
            }
            t_prep += t0.elapsed().as_secs_f64();
            self.stages[0].tp_embed_batch(chunk)?;
            self.stages[0].tp_norm_attn_b(0, ntok)?;
            self.sync_x_b(ntok)?;
            for layer in 0..self.cfg.num_layers {
                let ta = Instant::now();
                let attn = if self.cfg.kind(layer) == LayerKind::Conv {
                    self.stages[0].tp_conv_b(layer, slot, pos, ntok)?
                } else {
                    let mut acc = vec![0.0f32; ntok * h];
                    for stage in self.stages.iter_mut() {
                        let part = stage.tp_attn_b(layer, slot, pos, ntok)?;
                        for (dst, src) in acc.iter_mut().zip(part.iter()) {
                            *dst += *src;
                        }
                    }
                    acc
                };
                t_attn += ta.elapsed().as_secs_f64();
                let ts = Instant::now();
                for stage in self.stages.iter_mut().rev() {
                    stage.tp_attn_norm_b(layer, &attn, ntok)?;
                }
                self.sync_x_b(ntok)?;
                t_sync += ts.elapsed().as_secs_f64();
                let tf = Instant::now();
                let mut ffn = vec![0.0f32; ntok * h];
                for stage in self.stages.iter_mut() {
                    let part = stage.tp_ffn_local_b(layer, ntok)?;
                    for (dst, src) in ffn.iter_mut().zip(part.iter()) {
                        *dst += *src;
                    }
                }
                t_ffn += tf.elapsed().as_secs_f64();
                let ts2 = Instant::now();
                for stage in self.stages.iter_mut().rev() {
                    stage.tp_ffn_norm_b(layer, &ffn, ntok)?;
                }
                self.sync_x_b(ntok)?;
                t_sync += ts2.elapsed().as_secs_f64();
            }
            if need_logits {
                let mut logits = vec![0.0f32; self.cfg.vocab_size];
                for stage in self.stages.iter_mut() {
                    let shard = stage
                        .tp_shard()
                        .ok_or_else(|| anyhow!("stage without tp shard"))?;
                    let part = stage.tp_lmhead_last(ntok)?;
                    let stop = (shard.vocab_start + part.len()).min(logits.len());
                    logits[shard.vocab_start..stop]
                        .copy_from_slice(&part[..stop - shard.vocab_start]);
                }
                want_logits = Some(logits);
            }
            if dbg {
                eprintln!(
                    "tp-debug chunk ntok={ntok}: prep {t_prep:.2}s attn {t_attn:.2}s ffn {t_ffn:.2}s norm+sync {t_sync:.2}s"
                );
            }
        }
        Ok(want_logits)
    }
}

impl InferenceBackend for TpModel {
    fn create_sequence(&mut self) -> Result<usize> {
        let mut slots = Vec::with_capacity(self.stages.len());
        let mut err = None;
        for stage in self.stages.iter_mut() {
            match stage.create_sequence() {
                Ok(slot) => slots.push(slot),
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        if err.is_none() && slots.windows(2).any(|w| w[0] != w[1]) {
            err = Some(anyhow!("tp stages allocated different slots"));
        }
        if let Some(e) = err {
            for slot in &slots {
                for stage in self.stages.iter_mut() {
                    let _ = stage.free_sequence(*slot);
                }
            }
            return Err(e);
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
        if std::env::var("LFM_TP_SERIAL").as_deref() == Ok("1") {
            let mut last = Vec::new();
            for (i, &token) in tokens.iter().enumerate().skip(reused) {
                let want = need_logits && i + 1 == tokens.len();
                if let Some(logits) = self.step_token(slot, token, i, want)? {
                    last = logits;
                }
            }
            self.prefill_finish(slot)?;
            self.last_logits = last;
            return Ok(reused);
        }
        let batch = self.prefill_batch_size().max(1);
        let mut pos = reused;
        let mut last_logits: Option<Vec<f32>> = None;
        while pos < tokens.len() {
            let end = (pos + batch).min(tokens.len());
            let chunk = &tokens[pos..end];
            let want = need_logits && end == tokens.len();
            if let Some(logits) = self.prefill_chunk(slot, pos, chunk, want)? {
                last_logits = Some(logits);
            }
            pos = end;
        }
        self.prefill_finish(slot)?;
        if let Some(logits) = last_logits {
            self.last_logits = logits;
        }
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
        let serial = std::env::var("LFM_TP_SERIAL").as_deref() == Ok("1");
        let runs: Vec<(usize, usize)> = if serial {
            (0..steps.len()).map(|i| (i, i + 1)).collect()
        } else {
            group_runs(steps)
        };
        for (i, j) in runs {
            if j - i > 1 {
                let chunk: Vec<u32> = steps[i..j].iter().map(|s| s.token).collect();
                let logits =
                    self.prefill_chunk(steps[i].slot, steps[i].pos, &chunk, need_logits)?;
                for k in i..j {
                    if k + 1 == j {
                        if let Some(l) = &logits {
                            self.last_logits = l.clone();
                            self.row_logits.push(l.clone());
                        } else {
                            self.row_logits.push(Vec::new());
                        }
                    } else {
                        self.row_logits.push(Vec::new());
                    }
                }
            } else {
                let logits =
                    self.step_token(steps[i].slot, steps[i].token, steps[i].pos, need_logits)?;
                if let Some(l) = logits {
                    self.last_logits = l.clone();
                    self.row_logits.push(l);
                } else {
                    self.row_logits.push(Vec::new());
                }
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

    fn max_sequences(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.max_sequences())
            .min()
            .unwrap_or(1)
    }

    fn prefill_batch_size(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.prefill_batch_size())
            .min()
            .unwrap_or(1)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn step(slot: usize, pos: usize) -> SequenceStep {
        SequenceStep {
            slot,
            token: 1,
            pos,
        }
    }

    #[test]
    fn groups_contiguous_runs_per_slot() {
        let steps = vec![
            step(0, 5),
            step(0, 6),
            step(0, 7),
            step(1, 3),
            step(1, 9),
            step(0, 20),
        ];
        assert_eq!(group_runs(&steps), vec![(0, 3), (3, 4), (4, 5), (5, 6)]);
    }

    #[test]
    fn groups_single_steps() {
        let steps = vec![step(0, 0), step(1, 0)];
        assert_eq!(group_runs(&steps), vec![(0, 1), (1, 2)]);
        assert!(group_runs(&[]).is_empty());
    }
}
