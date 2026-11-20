use std::path::Path;

use anyhow::{Result, anyhow};
use lfm_app::{CacheStats, InferenceBackend, SequenceStep};
use lfm_domain::ModelConfig;
use lfm_gguf::GgufFile;

use crate::{GpuModel, StepReq};

pub struct PipelineModel {
    stages: Vec<GpuModel>,
    hidden: usize,
}

/// Splits `num_layers` into contiguous layer ranges according to tier
/// percentages. Percentages must sum to exactly 100. Zero-percent tiers are
/// skipped entirely; a non-last tier that rounds down to zero layers yields an
/// empty `(start, start)` range (callers drop such ranges). The last non-zero
/// tier always ends at `num_layers`, which compensates for integer rounding.
fn layer_ranges(num_layers: usize, pcts: &[u32]) -> Result<Vec<(usize, usize)>> {
    let total: u64 = pcts.iter().map(|&pct| u64::from(pct)).sum();
    if total != 100 {
        return Err(anyhow!("tier percentages must sum to 100"));
    }
    let last_nonzero = pcts.iter().rposition(|&pct| pct > 0);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (i, &pct) in pcts.iter().enumerate() {
        if pct == 0 {
            continue;
        }
        let end = if last_nonzero == Some(i) {
            num_layers
        } else {
            start + num_layers * pct as usize / 100
        };
        ranges.push((start, end));
        start = end;
    }
    Ok(ranges)
}

/// Row index of the last request in the batch. Returns an error instead of
/// underflowing when the batch is empty.
fn last_row_index(steps: &[StepReq]) -> Result<usize> {
    steps
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow!("empty pipeline transfer batch"))
}

/// Minimal stage abstraction needed by [`PipelineModel::create_sequence`], so
/// slot allocation/cleanup can be tested without OpenCL devices.
trait SlotStage {
    fn alloc_slot(&mut self) -> Result<usize>;
    fn release_slot(&mut self, slot: usize) -> Result<()>;
}

impl SlotStage for GpuModel {
    fn alloc_slot(&mut self) -> Result<usize> {
        GpuModel::create_sequence(self)
    }

    fn release_slot(&mut self, slot: usize) -> Result<()> {
        GpuModel::free_sequence(self, slot)
    }
}

fn release_slots<T: SlotStage>(stages: &mut [T], slots: &[usize]) {
    for (stage, slot) in stages.iter_mut().zip(slots) {
        let _ = stage.release_slot(*slot);
    }
}

/// Allocates one sequence slot on every stage and verifies all stages chose the
/// same slot. On failure, already allocated slots are released on the stages
/// that own them.
fn allocate_common_slot<T: SlotStage>(stages: &mut [T]) -> Result<usize> {
    let mut slots = Vec::with_capacity(stages.len());
    for i in 0..stages.len() {
        match stages[i].alloc_slot() {
            Ok(slot) => slots.push(slot),
            Err(e) => {
                release_slots(&mut stages[..i], &slots);
                return Err(e);
            }
        }
    }
    if slots.windows(2).any(|w| w[0] != w[1]) {
        release_slots(stages, &slots);
        return Err(anyhow!("pipeline stages allocated different slots"));
    }
    slots
        .first()
        .copied()
        .ok_or_else(|| anyhow!("pipeline has no stages"))
}

/// Converts a panicking stage thread into an error that keeps the original
/// panic message.
fn stage_panic_error(index: usize, payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string());
    anyhow!("pipeline stage {index} panicked: {message}")
}

/// Splits a joined stage thread into its (possibly lost) model and an optional
/// error. A panic loses the model but is reported as an error.
fn join_outcome<T>(
    index: usize,
    joined: std::thread::Result<(T, Result<()>)>,
) -> (Option<T>, Option<anyhow::Error>) {
    match joined {
        Ok((stage, result)) => (Some(stage), result.err()),
        Err(payload) => (None, Some(stage_panic_error(index, payload))),
    }
}

/// Reassembles stage models, reporting an error instead of panicking if a stage
/// did not return one.
fn collect_stages<T>(results: Vec<Option<T>>) -> Result<Vec<T>> {
    results
        .into_iter()
        .map(|stage| stage.ok_or_else(|| anyhow!("pipeline stage did not return")))
        .collect()
}

impl PipelineModel {
    pub fn load_tiers(path: &Path, max_ctx: usize, tiers: &[(usize, usize)]) -> Result<Self> {
        Self::load_tiers_with(path, max_ctx, tiers, &crate::Options::from_env())
    }

    pub fn load_tiers_with(
        path: &Path,
        max_ctx: usize,
        tiers: &[(usize, usize)],
        opts: &crate::Options,
    ) -> Result<Self> {
        let pcts: Vec<u32> = tiers
            .iter()
            .map(|&(_, percent)| {
                u32::try_from(percent)
                    .map_err(|_| anyhow!("tier percentage {percent} is too large"))
            })
            .collect::<Result<Vec<_>>>()?;
        let total: u64 = pcts.iter().map(|&pct| u64::from(pct)).sum();
        if total != 100 {
            return Err(anyhow!("tier percentages must sum to 100"));
        }
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let ranges = layer_ranges(cfg.num_layers, &pcts)?;
        let devices: Vec<usize> = tiers
            .iter()
            .filter(|(_, percent)| *percent > 0)
            .map(|&(device, _)| device)
            .collect();
        let mut stages = Vec::new();
        for (&device, &(from, to)) in devices.iter().zip(ranges.iter()) {
            if from >= to {
                continue;
            }
            stages.push(GpuModel::load_opts(
                path,
                max_ctx,
                Some(device),
                Some((from, to)),
                None,
                opts,
            )?);
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
            if last && need_logits {
                let row = last_row_index(steps)?;
                stage.step_mixed_input_rows(steps, &hidden, &[row])?;
            } else {
                stage.step_mixed_input(steps, &hidden, false)?;
            }
            if !last {
                hidden = stage.stage_hidden_batch(steps.len())?;
            }
        }
        Ok(())
    }
    fn prefill_pipelined(
        &mut self,
        slot: usize,
        tokens: &[u32],
        need_logits: bool,
        chunks: &[(usize, usize)],
    ) -> Result<()> {
        let n = self.stages.len();
        let stages = std::mem::take(&mut self.stages);
        let mut slots: Vec<Option<GpuModel>> = stages.into_iter().map(Some).collect();
        let mut senders: Vec<Option<std::sync::mpsc::SyncSender<Vec<f32>>>> =
            (0..n).map(|_| None).collect();
        let mut receivers: Vec<Option<std::sync::mpsc::Receiver<Vec<f32>>>> =
            (0..n).map(|_| None).collect();
        for i in 1..n {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            senders[i] = Some(tx);
            receivers[i] = Some(rx);
        }
        let mut errors: Vec<Option<anyhow::Error>> = (0..n).map(|_| None).collect();
        let mut results: Vec<Option<GpuModel>> = (0..n).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n);
            for i in 0..n {
                let mut stage = slots[i].take().expect("stage");
                let rx = receivers[i].take();
                let tx = if i + 1 < n {
                    senders[i + 1].take()
                } else {
                    None
                };
                let handle = scope.spawn(move || -> (GpuModel, Result<()>) {
                    let mut result = Ok(());
                    let dbg = std::env::var("LFM_PIPE_DEBUG").is_ok();
                    for (ci, &(pos, end)) in chunks.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let last_chunk = ci + 1 == chunks.len();
                        let len = end - pos;
                        let input = if i == 0 {
                            None
                        } else {
                            match rx.as_ref().map(|r| r.recv()) {
                                Some(Ok(h)) => Some(h),
                                _ => break,
                            }
                        };
                        let step = if i == 0 {
                            stage.forward_batch_seq(slot, &tokens[pos..end], pos, false)
                        } else {
                            stage.forward_batch_hidden(
                                slot,
                                input.as_deref().unwrap_or(&[]),
                                pos,
                                need_logits && i + 1 == n && last_chunk,
                            )
                        };
                        if dbg {
                            eprintln!(
                                "stage {i} chunk {ci} pos {pos}..{end} = {:.3}s",
                                t0.elapsed().as_secs_f64()
                            );
                        }
                        if let Err(e) = step {
                            result = Err(e);
                            break;
                        }
                        if let Some(tx) = &tx {
                            match stage.stage_hidden_batch(len) {
                                Ok(h) => {
                                    if tx.send(h).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    result = Err(e);
                                    break;
                                }
                            }
                        }
                    }
                    (stage, result)
                });
                handles.push(handle);
            }
            for (i, handle) in handles.into_iter().enumerate() {
                let (stage, error) = join_outcome(i, handle.join());
                results[i] = stage;
                errors[i] = error;
            }
        });
        if let Some(e) = errors.into_iter().flatten().next() {
            // A panicking stage is lost, but whatever returned is kept.
            self.stages = results.into_iter().flatten().collect();
            return Err(e);
        }
        self.stages = collect_stages(results)?;
        Ok(())
    }
}

impl InferenceBackend for PipelineModel {
    fn create_sequence(&mut self) -> Result<usize> {
        allocate_common_slot(&mut self.stages)
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
        let pb = self.stages[0].prefill_batch_size().max(1);
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut pos = reused;
        while pos < tokens.len() {
            let end = (pos + pb).min(tokens.len());
            chunks.push((pos, end));
            pos = end;
        }
        let n = self.stages.len();
        if n == 1 {
            for &(pos, end) in &chunks {
                let last_chunk = end == tokens.len();
                self.stages[0].forward_batch_seq(
                    slot,
                    &tokens[pos..end],
                    pos,
                    need_logits && last_chunk,
                )?;
            }
        } else {
            self.prefill_pipelined(slot, tokens, need_logits, &chunks)?;
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

impl PipelineModel {
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn last_stage_mut(&mut self) -> &mut GpuModel {
        self.last_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    fn req(slot: usize, token: u32, pos: usize) -> StepReq {
        StepReq { slot, token, pos }
    }

    fn assert_covers_all_layers(name: &str, num_layers: usize, ranges: &[(usize, usize)]) {
        let mut covered = vec![0usize; num_layers];
        for &(from, to) in ranges {
            assert!(from <= to, "{name}: reversed range {from}..{to}");
            assert!(
                to <= num_layers,
                "{name}: range {from}..{to} exceeds {num_layers}"
            );
            for count in &mut covered[from..to] {
                *count += 1;
            }
        }
        assert!(
            covered.iter().all(|&count| count == 1),
            "{name}: layer coverage {covered:?} from {ranges:?}"
        );
    }

    #[test]
    fn layer_ranges_96_4_covers_every_layer_once() {
        let ranges = layer_ranges(24, &[96, 4]).expect("valid tiers");
        assert_eq!(ranges, vec![(0, 23), (23, 24)]);
        assert_eq!(ranges.iter().map(|(a, b)| b - a).sum::<usize>(), 24);
        assert_covers_all_layers("96/4", 24, &ranges);
    }

    #[test]
    fn layer_ranges_even_and_single_device_layouts() {
        let split = layer_ranges(24, &[50, 50]).expect("valid tiers");
        assert_eq!(split, vec![(0, 12), (12, 24)]);
        assert_covers_all_layers("50/50", 24, &split);

        let single = layer_ranges(24, &[100, 0]).expect("valid tiers");
        assert_eq!(single, vec![(0, 24)]);
        assert_eq!(single.len(), 1, "zero-percent tier must not create a stage");
        assert_covers_all_layers("100/0", 24, &single);
    }

    #[test]
    fn layer_ranges_three_tiers_and_trailing_zero() {
        let three = layer_ranges(24, &[33, 33, 34]).expect("valid tiers");
        assert_eq!(three, vec![(0, 7), (7, 14), (14, 24)]);
        assert_covers_all_layers("33/33/34", 24, &three);

        let trailing_zero = layer_ranges(24, &[20, 80, 0]).expect("valid tiers");
        assert_eq!(trailing_zero, vec![(0, 4), (4, 24)]);
        assert_covers_all_layers("20/80/0", 24, &trailing_zero);

        let leading_zeros = layer_ranges(24, &[0, 0, 100]).expect("valid tiers");
        assert_eq!(leading_zeros, vec![(0, 24)]);
        assert_covers_all_layers("0/0/100", 24, &leading_zeros);
    }

    #[test]
    fn layer_ranges_last_tier_absorbs_rounding() {
        // 7 * 50 / 100 = 3, so without the last_nonzero compensation layer 6
        // would be lost.
        let seven = layer_ranges(7, &[50, 50]).expect("valid tiers");
        assert_eq!(seven, vec![(0, 3), (3, 7)]);
        assert_covers_all_layers("7 layers 50/50", 7, &seven);

        // Tiny models round early tiers down to zero layers. The ranges are
        // reported as empty and the final tier covers everything; the loader
        // drops empty ranges, so no stage is created for them.
        let tiny = layer_ranges(2, &[33, 33, 34]).expect("valid tiers");
        assert_eq!(tiny, vec![(0, 0), (0, 0), (0, 2)]);
        assert_covers_all_layers("2 layers 33/33/34", 2, &tiny);
    }

    #[test]
    fn layer_ranges_requires_exact_hundred_percent() {
        for bad in [&[50, 49][..], &[50, 50, 1][..], &[99][..], &[][..]] {
            let err = layer_ranges(24, bad).expect_err("percentages must sum to 100");
            assert!(
                err.to_string().contains("sum to 100"),
                "unexpected error for {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn last_row_index_rejects_empty_batch() {
        assert_eq!(last_row_index(&[req(0, 1, 0)]).expect("one step"), 0);
        assert_eq!(
            last_row_index(&[req(0, 1, 0), req(1, 2, 1), req(0, 3, 2)]).expect("three steps"),
            2
        );
        let err = last_row_index(&[]).expect_err("empty batch must not underflow");
        assert!(err.to_string().contains("empty pipeline transfer"), "{err}");
    }

    struct FakeStage {
        slot: Option<usize>,
        freed: Vec<usize>,
    }

    impl FakeStage {
        fn new(slot: Option<usize>) -> Self {
            Self {
                slot,
                freed: Vec::new(),
            }
        }
    }

    impl SlotStage for FakeStage {
        fn alloc_slot(&mut self) -> Result<usize> {
            self.slot.ok_or_else(|| anyhow!("no free sequence slots"))
        }

        fn release_slot(&mut self, slot: usize) -> Result<()> {
            self.freed.push(slot);
            Ok(())
        }
    }

    #[test]
    fn allocate_common_slot_success() {
        let mut stages = vec![
            FakeStage::new(Some(2)),
            FakeStage::new(Some(2)),
            FakeStage::new(Some(2)),
        ];
        assert_eq!(allocate_common_slot(&mut stages).expect("same slot"), 2);
        assert!(stages.iter().all(|stage| stage.freed.is_empty()));

        let mut single = vec![FakeStage::new(Some(0))];
        assert_eq!(allocate_common_slot(&mut single).expect("single stage"), 0);
        assert!(single[0].freed.is_empty());
    }

    #[test]
    fn allocate_common_slot_releases_on_alloc_failure() {
        let mut stages = vec![FakeStage::new(Some(1)), FakeStage::new(None)];
        let err = allocate_common_slot(&mut stages).expect_err("second stage fails");
        assert!(err.to_string().contains("no free sequence slots"), "{err}");
        assert_eq!(stages[0].freed, vec![1], "already allocated slot is freed");
        assert!(
            stages[1].freed.is_empty(),
            "failed stage had nothing to free"
        );
    }

    #[test]
    fn allocate_common_slot_releases_on_slot_mismatch() {
        let mut stages = vec![
            FakeStage::new(Some(0)),
            FakeStage::new(Some(1)),
            FakeStage::new(Some(0)),
        ];
        let err = allocate_common_slot(&mut stages).expect_err("slots differ");
        assert!(
            err.to_string().contains("different slots"),
            "unexpected error: {err}"
        );
        assert_eq!(stages[0].freed, vec![0]);
        assert_eq!(stages[1].freed, vec![1]);
        assert_eq!(stages[2].freed, vec![0]);
    }

    #[test]
    fn allocate_common_slot_rejects_empty_pipeline() {
        let mut stages: Vec<FakeStage> = Vec::new();
        let err = allocate_common_slot(&mut stages).expect_err("no stages");
        assert!(err.to_string().contains("no stages"), "{err}");
    }

    #[test]
    fn panicking_stage_becomes_error_with_original_cause() {
        let joined = std::thread::scope(|scope| {
            let handle = scope.spawn(|| -> (u8, Result<()>) { panic!("boom from stage") });
            handle.join()
        });
        let (stage, error) = join_outcome(2, joined);
        assert!(stage.is_none(), "a panicked stage must not be resurrected");
        let error = error.expect("panic must become an error");
        let message = error.to_string();
        assert!(message.contains("stage 2"), "{message}");
        assert!(message.contains("boom from stage"), "{message}");
    }

    #[test]
    fn panicking_stage_keeps_formatted_panic_message() {
        let joined = std::thread::scope(|scope| {
            let handle = scope.spawn(|| -> (u8, Result<()>) {
                let detail = format!("stage {} exploded", 4);
                panic!("{detail}");
            });
            handle.join()
        });
        let (stage, error) = join_outcome(4, joined);
        assert!(stage.is_none());
        let message = error.expect("panic must become an error").to_string();
        assert!(message.contains("stage 4 exploded"), "{message}");
    }

    #[test]
    fn join_outcome_keeps_returned_stage_and_error() {
        let (stage, error) = join_outcome::<u8>(0, Ok((7, Ok(()))));
        assert_eq!(stage, Some(7));
        assert!(error.is_none());

        let (stage, error) = join_outcome::<u8>(1, Ok((8, Err(anyhow!("forward failed")))));
        assert_eq!(stage, Some(8));
        assert!(
            error
                .expect("stage error is preserved")
                .to_string()
                .contains("forward failed")
        );
    }

    #[test]
    fn collect_stages_reports_missing_stage_without_panicking() {
        assert_eq!(
            collect_stages(vec![Some(1u8), Some(2)]).expect("all stages returned"),
            vec![1, 2]
        );
        let err = collect_stages(vec![Some(1u8), None]).expect_err("lost stage");
        assert!(err.to_string().contains("did not return"), "{err}");
    }
}
