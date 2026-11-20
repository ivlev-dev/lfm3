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
    attn_acc: Vec<f32>,
    ffn_acc: Vec<f32>,
    sync_norms: bool,
    batch_dec: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TpBatch {
    Prefill(usize, usize),
    Decode(Vec<usize>),
}

pub(crate) fn plan_batches(
    steps: &[SequenceStep],
    cap: usize,
    serial: bool,
    batch_dec: bool,
) -> Vec<TpBatch> {
    let runs: Vec<(usize, usize)> = if serial {
        (0..steps.len()).map(|i| (i, i + 1)).collect()
    } else {
        group_runs(steps)
    };
    let batch_dec = batch_dec && !serial;
    let cap = cap.max(1);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < runs.len() {
        let (a, b) = runs[i];
        if b - a > 1 {
            out.push(TpBatch::Prefill(a, b));
            i += 1;
            continue;
        }
        if !batch_dec {
            out.push(TpBatch::Decode(vec![a]));
            i += 1;
            continue;
        }
        let mut grp = vec![a];
        let mut j = i + 1;
        while j < runs.len() && runs[j].1 - runs[j].0 == 1 && grp.len() < cap {
            let s = runs[j].0;
            if grp.iter().any(|&k| steps[k].slot == steps[s].slot) {
                break;
            }
            grp.push(s);
            j += 1;
        }
        out.push(TpBatch::Decode(grp));
        i = j;
    }
    out
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

fn env_is_one(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

fn env_flag_on(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

fn reset_acc(acc: &mut Vec<f32>, len: usize) {
    acc.clear();
    acc.resize(len, 0.0);
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn sync_x(stages: &mut [GpuModel]) -> Result<()> {
    if stages.len() <= 1 {
        return Ok(());
    }
    let (x, hq, hs) = stages[0].tp_norm_inputs()?;
    for stage in stages.iter_mut().skip(1) {
        stage.tp_set_norm_inputs(&x, &hq, &hs)?;
    }
    Ok(())
}

fn sync_x_b(stages: &mut [GpuModel], ntok: usize) -> Result<()> {
    if stages.len() <= 1 {
        return Ok(());
    }
    let (x, hq, hs) = stages[0].tp_norm_inputs_b(ntok)?;
    for stage in stages.iter_mut().skip(1) {
        stage.tp_set_norm_inputs_b(&x, &hq, &hs, ntok)?;
    }
    Ok(())
}

fn tp_weights(world: usize) -> Result<Option<Vec<u32>>> {
    let Ok(raw) = std::env::var("LFM_TP_WEIGHTS") else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let mut out = Vec::new();
    for part in raw.split('/') {
        out.push(
            part.trim()
                .parse::<u32>()
                .map_err(|_| anyhow!("bad LFM_TP_WEIGHTS {raw:?}: need N0/N1/..."))?,
        );
    }
    if out.len() != world || out.iter().all(|&w| w == 0) {
        return Err(anyhow!(
            "LFM_TP_WEIGHTS {raw:?} does not match {world} devices"
        ));
    }
    Ok(Some(out))
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
        let mut weights = tp_weights(world)?;
        if weights.is_none() && world == 2 {
            let kinds: Vec<String> = devices
                .iter()
                .map(|&d| {
                    lfm_opencl::Gpu::open(d)
                        .map(|g| g.caps.device_type)
                        .unwrap_or_default()
                })
                .collect();
            if kinds[0] == "gpu" && kinds[1] == "cpu" {
                weights = Some(vec![75, 25]);
            }
        }
        if weights.is_none() && (cfg.ffn_size % world != 0 || (cfg.ffn_size / world) % 256 != 0)
        {
            return Err(anyhow!(
                "ffn {} cannot be split into {world} column shards (need multiple of 256)",
                cfg.ffn_size
            ));
        }
        let mut stages = Vec::with_capacity(world);
        for (rank, &device) in devices.iter().enumerate() {
            let shard = match &weights {
                Some(w) => TpShard::full_weighted(
                    rank,
                    world,
                    cfg.num_heads,
                    kv_heads,
                    cfg.ffn_size,
                    cfg.vocab_size,
                    w,
                ),
                None => TpShard::full(
                    rank,
                    world,
                    cfg.num_heads,
                    kv_heads,
                    cfg.ffn_size,
                    cfg.vocab_size,
                ),
            };
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
            attn_acc: Vec::new(),
            ffn_acc: Vec::new(),
            sync_norms: env_is_one("LFM_TP_SYNC_NORMS"),
            batch_dec: env_flag_on("LFM_TP_BATCH_DEC"),
        })
    }

    pub fn stages(&self) -> &[GpuModel] {
        &self.stages
    }

    fn step_token(
        &mut self,
        slot: usize,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let Self {
            stages,
            cfg,
            attn_acc,
            ffn_acc,
            sync_norms,
            ..
        } = self;
        let h = cfg.hidden_size;
        stages[0].tp_embed(token)?;
        stages[0].tp_norm_attn(0)?;
        // Входную норму считает только rank0, поэтому hq/hs остальных стадий надо заполнить один раз.
        sync_x(stages)?;
        for layer in 0..cfg.num_layers {
            if cfg.kind(layer) == LayerKind::Conv {
                let part = stages[0].tp_conv(layer, slot, pos)?;
                attn_acc.clear();
                attn_acc.extend_from_slice(&part);
            } else {
                reset_acc(attn_acc, h);
                for stage in stages.iter_mut() {
                    let part = stage.tp_attn(layer, slot, pos)?;
                    add_into(attn_acc, &part);
                }
            }
            for stage in stages.iter_mut().rev() {
                stage.tp_attn_norm(layer, attn_acc.as_slice())?;
            }
            // Сумма attn/ffn одинакова на всех стадиях, веса норм реплицированы, вход x тоже
            // одинаков, а norm_add_quant_b пересчитывает x/xq/xs локально на каждой стадии —
            // рассылать результат норма-ядер обратно не нужно.
            if *sync_norms {
                sync_x(stages)?;
            }
            reset_acc(ffn_acc, h);
            for stage in stages.iter_mut() {
                let part = stage.tp_ffn_local(layer)?;
                add_into(ffn_acc, &part);
            }
            for stage in stages.iter_mut().rev() {
                stage.tp_ffn_norm(layer, ffn_acc.as_slice())?;
            }
            if *sync_norms {
                sync_x(stages)?;
            }
        }
        if want_logits {
            let mut logits = vec![0.0f32; cfg.vocab_size];
            for stage in stages.iter_mut() {
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

    fn layers_batch(&mut self, ntok: usize) -> Result<()> {
        let Self {
            stages,
            cfg,
            attn_acc,
            ffn_acc,
            sync_norms,
            ..
        } = self;
        let h = cfg.hidden_size;
        let dbg = std::env::var("LFM_TP_DEBUG").is_ok();
        let (mut t_attn, mut t_ffn, mut t_sync) = (0.0f64, 0.0f64, 0.0f64);
        for layer in 0..cfg.num_layers {
            let ta = Instant::now();
            if cfg.kind(layer) == LayerKind::Conv {
                let part = stages[0].tp_conv_b(layer, 0, 0, ntok)?;
                attn_acc.clear();
                attn_acc.extend_from_slice(&part);
            } else {
                reset_acc(attn_acc, ntok * h);
                for stage in stages.iter_mut() {
                    let part = stage.tp_attn_b(layer, 0, 0, ntok)?;
                    add_into(attn_acc, &part);
                }
            }
            t_attn += ta.elapsed().as_secs_f64();
            let ts = Instant::now();
            for stage in stages.iter_mut().rev() {
                stage.tp_attn_norm_b(layer, attn_acc.as_slice(), ntok)?;
            }
            if *sync_norms {
                sync_x_b(stages, ntok)?;
            }
            t_sync += ts.elapsed().as_secs_f64();
            let tf = Instant::now();
            reset_acc(ffn_acc, ntok * h);
            for stage in stages.iter_mut() {
                let part = stage.tp_ffn_local_b(layer, ntok)?;
                add_into(ffn_acc, &part);
            }
            t_ffn += tf.elapsed().as_secs_f64();
            let ts2 = Instant::now();
            for stage in stages.iter_mut().rev() {
                stage.tp_ffn_norm_b(layer, ffn_acc.as_slice(), ntok)?;
            }
            if *sync_norms {
                sync_x_b(stages, ntok)?;
            }
            t_sync += ts2.elapsed().as_secs_f64();
        }
        if dbg {
            eprintln!(
                "tp-debug batch ntok={ntok}: attn {t_attn:.2}s ffn {t_ffn:.2}s norm+sync {t_sync:.2}s"
            );
        }
        Ok(())
    }

    fn gather_last(&mut self, ntok: usize) -> Result<Vec<f32>> {
        let vocab = self.cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for stage in self.stages.iter_mut() {
            let shard = stage
                .tp_shard()
                .ok_or_else(|| anyhow!("stage without tp shard"))?;
            let part = stage.tp_lmhead_last(ntok)?;
            let stop = (shard.vocab_start + part.len()).min(vocab);
            logits[shard.vocab_start..stop].copy_from_slice(&part[..stop - shard.vocab_start]);
        }
        Ok(logits)
    }

    fn gather_logits(&mut self, ntok: usize, rows: &[usize]) -> Result<Vec<Vec<f32>>> {
        let vocab = self.cfg.vocab_size;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        for stage in self.stages.iter_mut() {
            let shard = stage
                .tp_shard()
                .ok_or_else(|| anyhow!("stage without tp shard"))?;
            let part = stage.tp_lmhead_rows(ntok, rows)?;
            if part.len() != rows.len() * shard.vocab_count {
                return Err(anyhow!(
                    "lm_head shard returned {} values for {} rows",
                    part.len(),
                    rows.len()
                ));
            }
            if out.is_empty() {
                out.resize_with(rows.len(), || vec![0.0f32; vocab]);
            }
            for (i, dst) in out.iter_mut().enumerate() {
                let seg = &part[i * shard.vocab_count..(i + 1) * shard.vocab_count];
                let stop = (shard.vocab_start + seg.len()).min(vocab);
                dst[shard.vocab_start..stop].copy_from_slice(&seg[..stop - shard.vocab_start]);
            }
        }
        Ok(out)
    }

    fn prefill_chunk(
        &mut self,
        slot: usize,
        pos: usize,
        chunk: &[u32],
        need_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let ntok = chunk.len();
        for stage in self.stages.iter_mut() {
            stage.tp_prepare_batch(slot, pos, ntok)?;
        }
        self.stages[0].tp_embed_batch(chunk)?;
        self.stages[0].tp_norm_attn_b(0, ntok)?;
        sync_x_b(&mut self.stages, ntok)?;
        self.layers_batch(ntok)?;
        if need_logits {
            Ok(Some(self.gather_last(ntok)?))
        } else {
            Ok(None)
        }
    }

    fn decode_batch(
        &mut self,
        steps: &[SequenceStep],
        idxs: &[usize],
        want: &[bool],
    ) -> Result<()> {
        let ntok = idxs.len();
        let rows: Vec<(usize, usize)> = idxs
            .iter()
            .map(|&i| (steps[i].slot, steps[i].pos))
            .collect();
        for stage in self.stages.iter_mut() {
            stage.tp_prepare_rows(&rows)?;
        }
        let tokens: Vec<u32> = idxs.iter().map(|&i| steps[i].token).collect();
        self.stages[0].tp_embed_batch(&tokens)?;
        self.stages[0].tp_norm_attn_b(0, ntok)?;
        sync_x_b(&mut self.stages, ntok)?;
        self.layers_batch(ntok)?;
        let sel: Vec<usize> = (0..ntok).filter(|&r| want[idxs[r]]).collect();
        if sel.is_empty() {
            return Ok(());
        }
        let logits = self.gather_logits(ntok, &sel)?;
        for (&row, l) in sel.iter().zip(logits) {
            self.last_logits = l.clone();
            self.row_logits[idxs[row]] = l;
        }
        Ok(())
    }

    fn run_steps(&mut self, steps: &[SequenceStep], want: &[bool]) -> Result<()> {
        self.row_logits.clear();
        self.row_logits.resize(steps.len(), Vec::new());
        let cap = self
            .stages
            .iter()
            .map(|stage| stage.batch_cap())
            .min()
            .unwrap_or(1)
            .max(1);
        let plan = plan_batches(steps, cap, env_is_one("LFM_TP_SERIAL"), self.batch_dec);
        for batch in plan {
            match batch {
                TpBatch::Prefill(a, b) => {
                    let mut pos = a;
                    while pos < b {
                        let end = (pos + cap).min(b);
                        let chunk: Vec<u32> = steps[pos..end].iter().map(|s| s.token).collect();
                        let logits = self.prefill_chunk(
                            steps[a].slot,
                            steps[pos].pos,
                            &chunk,
                            want[end - 1],
                        )?;
                        if let Some(l) = logits {
                            self.last_logits = l.clone();
                            self.row_logits[end - 1] = l;
                        }
                        pos = end;
                    }
                }
                TpBatch::Decode(idxs) if idxs.len() > 1 => {
                    self.decode_batch(steps, &idxs, want)?;
                }
                TpBatch::Decode(idxs) => {
                    let i = idxs[0];
                    let s = steps[i];
                    if let Some(l) = self.step_token(s.slot, s.token, s.pos, want[i])? {
                        self.last_logits = l.clone();
                        self.row_logits[i] = l;
                    }
                }
            }
        }
        Ok(())
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
        if env_is_one("LFM_TP_SERIAL") {
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
        let want = vec![need_logits; steps.len()];
        self.run_steps(steps, &want)
    }

    fn step_rows(&mut self, steps: &[SequenceStep], rows: &[usize]) -> Result<()> {
        let mut want = vec![false; steps.len()];
        for &row in rows {
            let slot = want
                .get_mut(row)
                .ok_or_else(|| anyhow!("logits row {row} outside batch of {}", steps.len()))?;
            *slot = true;
        }
        self.run_steps(steps, &want)
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

    #[test]
    fn plan_batches_decode_merges_slots() {
        let steps = vec![step(0, 5), step(1, 3), step(2, 9), step(3, 0)];
        let plan = plan_batches(&steps, 8, false, true);
        assert_eq!(plan, vec![TpBatch::Decode(vec![0, 1, 2, 3])]);
    }

    #[test]
    fn plan_batches_decode_never_repeats_slot() {
        let steps = vec![step(0, 5), step(1, 3), step(0, 6), step(2, 0)];
        let plan = plan_batches(&steps, 8, false, true);
        assert_eq!(
            plan,
            vec![TpBatch::Decode(vec![0, 1]), TpBatch::Decode(vec![2, 3]),]
        );
    }

    #[test]
    fn plan_batches_decode_respects_cap() {
        let steps = vec![step(0, 0), step(1, 0), step(2, 0), step(3, 0), step(4, 0)];
        let plan = plan_batches(&steps, 2, false, true);
        assert_eq!(
            plan,
            vec![
                TpBatch::Decode(vec![0, 1]),
                TpBatch::Decode(vec![2, 3]),
                TpBatch::Decode(vec![4]),
            ]
        );
    }

    #[test]
    fn plan_batches_keeps_prefill_runs_separate() {
        let steps = vec![
            step(0, 0),
            step(0, 1),
            step(0, 2),
            step(1, 7),
            step(1, 8),
            step(2, 4),
        ];
        let plan = plan_batches(&steps, 8, false, true);
        assert_eq!(
            plan,
            vec![
                TpBatch::Prefill(0, 3),
                TpBatch::Prefill(3, 5),
                TpBatch::Decode(vec![5]),
            ]
        );
    }

    #[test]
    fn plan_batches_disabled_is_one_row_per_batch() {
        let steps = vec![step(0, 5), step(1, 3), step(0, 6)];
        let plan = plan_batches(&steps, 8, false, false);
        assert_eq!(
            plan,
            vec![
                TpBatch::Decode(vec![0]),
                TpBatch::Decode(vec![1]),
                TpBatch::Decode(vec![2]),
            ]
        );
    }

    #[test]
    fn plan_batches_serial_splits_runs() {
        let steps = vec![step(0, 0), step(0, 1), step(1, 2)];
        let plan = plan_batches(&steps, 8, true, true);
        assert_eq!(
            plan,
            vec![
                TpBatch::Decode(vec![0]),
                TpBatch::Decode(vec![1]),
                TpBatch::Decode(vec![2]),
            ]
        );
    }

    #[test]
    fn plan_batches_empty() {
        assert!(plan_batches(&[], 8, false, true).is_empty());
    }

    #[test]
    fn env_flag_default_on_honours_off_values() {
        unsafe {
            std::env::set_var("LFM_TP_BATCH_DEC_TEST", "0");
        }
        assert!(!env_flag_on("LFM_TP_BATCH_DEC_TEST"));
        unsafe {
            std::env::set_var("LFM_TP_BATCH_DEC_TEST", "off");
        }
        assert!(!env_flag_on("LFM_TP_BATCH_DEC_TEST"));
        unsafe {
            std::env::set_var("LFM_TP_BATCH_DEC_TEST", "1");
        }
        assert!(env_flag_on("LFM_TP_BATCH_DEC_TEST"));
        unsafe {
            std::env::remove_var("LFM_TP_BATCH_DEC_TEST");
        }
        assert!(env_flag_on("LFM_TP_BATCH_DEC_TEST"));
    }

    #[test]
    fn acc_reset_keeps_capacity_and_zeroes() {
        let mut acc = vec![1.0f32; 4];
        reset_acc(&mut acc, 2);
        assert_eq!(acc, vec![0.0, 0.0]);
        assert!(acc.capacity() >= 4);
        reset_acc(&mut acc, 6);
        assert_eq!(acc.len(), 6);
        assert!(acc.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn add_into_sums_only_common_prefix() {
        let mut dst = vec![1.0f32, 2.0, 3.0];
        add_into(&mut dst, &[0.5, 0.5]);
        assert_eq!(dst, vec![1.5, 2.5, 3.0]);
    }

    #[test]
    fn env_flag_reads_only_exact_one() {
        unsafe {
            std::env::set_var("LFM_TP_TEST_FLAG", "1");
        }
        assert!(env_is_one("LFM_TP_TEST_FLAG"));
        unsafe {
            std::env::set_var("LFM_TP_TEST_FLAG", "true");
        }
        assert!(!env_is_one("LFM_TP_TEST_FLAG"));
        unsafe {
            std::env::remove_var("LFM_TP_TEST_FLAG");
        }
        assert!(!env_is_one("LFM_TP_TEST_FLAG"));
    }
}
