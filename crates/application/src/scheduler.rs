use std::collections::VecDeque;

use thiserror::Error;

pub const KV_PAGE_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub max_ctx: usize,
    pub max_active: usize,
    pub max_batch: usize,
    pub prefill_budget: usize,
    /// Сколько prefill-токенов подмешивать в шаг, где уже есть decode.
    pub prefill_in_decode: usize,
    /// Сколько decode-строк можно отдать за шаг (по умолчанию = max_batch).
    pub max_decode: usize,
    /// Предел всех строк за шаг (decode + prefill); usize::MAX — без предела.
    pub max_rows: usize,
}

impl SchedulerConfig {
    pub fn new(max_ctx: usize, max_active: usize, max_batch: usize, prefill_budget: usize) -> Self {
        let max_batch = max_batch.max(1);
        Self {
            max_ctx: max_ctx.max(1),
            max_active: max_active.clamp(1, 16),
            max_batch,
            prefill_budget: prefill_budget.max(1),
            prefill_in_decode: prefill_budget.max(1),
            max_decode: max_batch,
            max_rows: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AdmitError {
    #[error("empty prompt")]
    EmptyPrompt,
    #[error("prompt of {prompt_len} tokens does not fit context {max_ctx}")]
    ContextOverflow { prompt_len: usize, max_ctx: usize },
    #[error("request needs {needed} KV pages, only {available} free")]
    KvOverflow { needed: usize, available: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Enqueued {
    pub id: JobId,
    pub max_new: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitOutcome {
    Admitted(JobId),
    Rejected(JobId, AdmitError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveSeq {
    pub id: JobId,
    pub phase: Phase,
    pub prompt_len: usize,
    pub max_new: usize,
    pub prefilled: usize,
    pub generated: usize,
    pub reserved_pages: usize,
}

impl ActiveSeq {
    pub fn position(&self) -> usize {
        self.prefilled + self.generated.saturating_sub(1)
    }

    pub fn used_pages(&self) -> usize {
        pages_for(self.position())
    }

    pub fn future_pages(&self) -> usize {
        self.reserved_pages.saturating_sub(self.used_pages())
    }

    pub fn prompt_remaining(&self) -> usize {
        self.prompt_len.saturating_sub(self.prefilled)
    }

    pub fn is_prefill_done(&self) -> bool {
        self.prefilled >= self.prompt_len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitingJob {
    pub id: JobId,
    pub prompt_len: usize,
    pub max_new: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepToken {
    Prompt(usize),
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRow {
    pub id: JobId,
    pub pos: usize,
    pub token: StepToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepPlan {
    Idle,
    AtomicPrefill(JobId),
    Mixed { rows: Vec<StepRow> },
}

impl StepPlan {
    pub fn is_idle(&self) -> bool {
        matches!(self, StepPlan::Idle)
    }

    pub fn rows(&self) -> &[StepRow] {
        match self {
            StepPlan::Mixed { rows } => rows,
            _ => &[],
        }
    }
}

pub fn pages_for(tokens: usize) -> usize {
    tokens.div_ceil(KV_PAGE_SIZE)
}

fn pages_tokens(pages: usize) -> usize {
    pages * KV_PAGE_SIZE
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    next_id: u64,
    waiting: VecDeque<WaitingJob>,
    active: Vec<ActiveSeq>,
    cursor: usize,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Self {
        Self {
            cfg,
            next_id: 0,
            waiting: VecDeque::new(),
            cursor: 0,
            active: Vec::new(),
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    pub fn enqueue(
        &mut self,
        prompt_len: usize,
        requested_max_new: usize,
    ) -> Result<Enqueued, AdmitError> {
        if prompt_len == 0 {
            return Err(AdmitError::EmptyPrompt);
        }
        if prompt_len + 1 > self.cfg.max_ctx {
            return Err(AdmitError::ContextOverflow {
                prompt_len,
                max_ctx: self.cfg.max_ctx,
            });
        }
        let max_new = requested_max_new.max(1).min(self.cfg.max_ctx - prompt_len);
        let id = JobId(self.next_id);
        self.next_id += 1;
        self.waiting.push_back(WaitingJob {
            id,
            prompt_len,
            max_new,
        });
        Ok(Enqueued { id, max_new })
    }

    pub fn admit(&mut self, free_pages: usize) -> Vec<AdmitOutcome> {
        let mut out = Vec::new();
        while self.active.len() < self.cfg.max_active {
            let Some(front) = self.waiting.front().copied() else {
                break;
            };
            let available = self.available_for_new(free_pages);
            if available == 0 {
                break;
            }
            let needed = pages_for(front.prompt_len + front.max_new);
            let (needed, max_new) = if needed > available {
                let fit_tokens = pages_tokens(available);
                let max_new = fit_tokens.saturating_sub(front.prompt_len);
                if max_new == 0 {
                    self.waiting.pop_front();
                    out.push(AdmitOutcome::Rejected(
                        front.id,
                        AdmitError::KvOverflow { needed, available },
                    ));
                    continue;
                }
                (
                    pages_for(front.prompt_len + max_new).min(available),
                    max_new,
                )
            } else {
                (needed, front.max_new)
            };
            self.waiting.pop_front();
            self.active.push(ActiveSeq {
                id: front.id,
                phase: Phase::Prefill,
                prompt_len: front.prompt_len,
                max_new,
                prefilled: 0,
                generated: 0,
                reserved_pages: needed,
            });
            out.push(AdmitOutcome::Admitted(front.id));
        }
        out
    }

    pub fn available_for_new(&self, free_pages: usize) -> usize {
        let pending: usize = self.active.iter().map(ActiveSeq::future_pages).sum();
        free_pages.saturating_sub(pending)
    }

    pub fn plan_step(&mut self) -> StepPlan {
        let has_decode = self.active.iter().any(|seq| seq.phase == Phase::Decode);
        if !has_decode {
            let pending: Vec<&ActiveSeq> = self
                .active
                .iter()
                .filter(|seq| seq.phase == Phase::Prefill && seq.prompt_remaining() > 0)
                .collect();
            if pending.len() == 1 && self.waiting.is_empty() && pending[0].prefilled == 0 {
                return StepPlan::AtomicPrefill(pending[0].id);
            }
            if pending.is_empty() {
                return StepPlan::Idle;
            }
        }

        let n = self.active.len();
        let order: Vec<usize> = if n == 0 {
            Vec::new()
        } else {
            (0..n).map(|i| (self.cursor + i) % n).collect()
        };
        self.cursor = if n == 0 { 0 } else { (self.cursor + 1) % n };
        let mut rows: Vec<StepRow> = Vec::with_capacity(self.cfg.max_batch);
        for &i in &order {
            let seq = &self.active[i];
            if seq.phase != Phase::Decode {
                continue;
            }
            if rows.len() >= self.cfg.max_decode {
                break;
            }
            rows.push(StepRow {
                id: seq.id,
                pos: seq.position(),
                token: StepToken::Decode,
            });
        }

        let cap = if has_decode {
            self.cfg.prefill_in_decode
        } else {
            self.cfg.prefill_budget
        };
        let mut budget = cap.min(self.cfg.max_rows.saturating_sub(rows.len()));
        let prefs: Vec<&ActiveSeq> = order
            .iter()
            .map(|&i| &self.active[i])
            .filter(|seq| seq.phase == Phase::Prefill && seq.prompt_remaining() > 0)
            .collect();
        let mut taken = vec![0usize; prefs.len()];
        let mut progress = prefs.iter().any(|seq| seq.prompt_remaining() > 0);
        while budget > 0 && progress {
            progress = false;
            let per = (budget / prefs.len().max(1)).max(1);
            for (i, seq) in prefs.iter().enumerate() {
                if budget == 0 {
                    break;
                }
                let remaining = seq.prompt_remaining() - taken[i];
                if remaining == 0 {
                    continue;
                }
                let take = remaining.min(per).min(budget);
                for k in 0..take {
                    let index = seq.prefilled + taken[i] + k;
                    rows.push(StepRow {
                        id: seq.id,
                        pos: index,
                        token: StepToken::Prompt(index),
                    });
                }
                taken[i] += take;
                budget -= take;
                progress = true;
            }
        }
        if rows.is_empty() {
            StepPlan::Idle
        } else {
            StepPlan::Mixed { rows }
        }
    }

    pub fn advance_prefill(&mut self, id: JobId, tokens: usize) {
        if let Some(seq) = self.active.iter_mut().find(|seq| seq.id == id) {
            seq.prefilled = (seq.prefilled + tokens).min(seq.prompt_len);
        }
    }

    pub fn complete_prefill(&mut self, id: JobId) {
        if let Some(seq) = self.active.iter_mut().find(|seq| seq.id == id) {
            seq.prefilled = seq.prompt_len;
            seq.phase = Phase::Decode;
        }
    }

    pub fn record_token(&mut self, id: JobId) {
        if let Some(seq) = self.active.iter_mut().find(|seq| seq.id == id) {
            seq.generated += 1;
        }
    }

    pub fn finish(&mut self, id: JobId) -> Option<ActiveSeq> {
        if let Some(index) = self.waiting.iter().position(|job| job.id == id) {
            self.waiting.remove(index);
        }
        let index = self.active.iter().position(|seq| seq.id == id)?;
        Some(self.active.remove(index))
    }

    pub fn active(&self) -> &[ActiveSeq] {
        &self.active
    }

    pub fn active_seq(&self, id: JobId) -> Option<&ActiveSeq> {
        self.active.iter().find(|seq| seq.id == id)
    }

    pub fn waiting(&self) -> &VecDeque<WaitingJob> {
        &self.waiting
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn total_len(&self) -> usize {
        self.active.len() + self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty() && self.waiting.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SchedulerConfig {
        SchedulerConfig::new(4096, 8, 6, 12)
    }

    fn admit_all(s: &mut Scheduler, pages: usize) -> Vec<JobId> {
        let mut ids = Vec::new();
        for outcome in s.admit(pages) {
            match outcome {
                AdmitOutcome::Admitted(id) => ids.push(id),
                AdmitOutcome::Rejected(id, err) => panic!("unexpected rejection {id:?}: {err}"),
            }
        }
        ids
    }

    fn make_decode(s: &mut Scheduler, id: JobId) {
        if let Some(seq) = s.active_seq(id) {
            let remaining = seq.prompt_remaining();
            s.advance_prefill(id, remaining);
        }
        s.complete_prefill(id);
        s.record_token(id);
    }

    #[test]
    fn empty_scheduler_is_idle() {
        let mut s = Scheduler::new(cfg());
        assert!(s.is_empty());
        assert_eq!(s.total_len(), 0);
        assert_eq!(s.waiting_len(), 0);
        assert_eq!(s.active_len(), 0);
        assert_eq!(s.plan_step(), StepPlan::Idle);
        assert!(s.admit(128).is_empty());
        assert!(s.is_empty());
    }

    #[test]
    fn enqueue_rejects_empty_and_overflow() {
        let mut s = Scheduler::new(SchedulerConfig::new(16, 8, 6, 12));
        assert_eq!(s.enqueue(0, 4).unwrap_err(), AdmitError::EmptyPrompt);
        assert_eq!(
            s.enqueue(16, 4).unwrap_err(),
            AdmitError::ContextOverflow {
                prompt_len: 16,
                max_ctx: 16
            }
        );
        assert!(s.is_empty());
        let queued = s.enqueue(15, 4).unwrap();
        assert_eq!(queued.max_new, 1);
        assert_eq!(s.waiting_len(), 1);
    }

    #[test]
    fn max_new_is_clamped_to_context() {
        let mut s = Scheduler::new(SchedulerConfig::new(100, 8, 6, 12));
        let queued = s.enqueue(60, 1024).unwrap();
        assert_eq!(queued.max_new, 40);
    }

    #[test]
    fn single_request_admitted_then_atomic_prefill() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 128));
        let queued = s.enqueue(80, 20).unwrap();
        let ids = admit_all(&mut s, 128);
        assert_eq!(ids, vec![queued.id]);
        assert_eq!(s.active_len(), 1);
        let seq = s.active_seq(queued.id).unwrap();
        assert_eq!(seq.phase, Phase::Prefill);
        assert_eq!(seq.reserved_pages, pages_for(100));
        assert_eq!(s.plan_step(), StepPlan::AtomicPrefill(queued.id));
    }

    #[test]
    fn admission_clamps_max_new_to_free_pages() {
        let mut s = Scheduler::new(cfg());
        let a = s.enqueue(64, 64).unwrap();
        let b = s.enqueue(64, 64).unwrap();
        let ids = admit_all(&mut s, 7);
        assert_eq!(ids, vec![a.id, b.id]);
        assert_eq!(s.waiting_len(), 0);
        let sa = s.active_seq(a.id).unwrap();
        assert_eq!(sa.reserved_pages, pages_for(128));
        let sb = s.active_seq(b.id).unwrap();
        assert!(sb.max_new < 64, "b must be clamped, got {}", sb.max_new);
        assert_eq!(sb.reserved_pages, 7 - sa.reserved_pages);
    }

    #[test]
    fn lone_request_rejected_when_pages_do_not_fit() {
        let mut s = Scheduler::new(cfg());
        let queued = s.enqueue(128, 64).unwrap();
        let out = s.admit(3);
        assert_eq!(
            out,
            vec![AdmitOutcome::Rejected(
                queued.id,
                AdmitError::KvOverflow {
                    needed: 6,
                    available: 3
                }
            )]
        );
        assert!(s.is_empty());
    }

    #[test]
    fn finish_releases_reservation_and_promotes_next() {
        let mut s = Scheduler::new(cfg());
        let a = s.enqueue(60, 4).unwrap();
        let b = s.enqueue(30, 4).unwrap();
        assert_eq!(admit_all(&mut s, 2), vec![a.id]);
        assert_eq!(s.available_for_new(2), 0);
        let removed = s.finish(a.id).expect("finished sequence");
        assert_eq!(removed.id, a.id);
        assert_eq!(s.active_len(), 0);
        assert_eq!(s.finish(a.id), None);
        assert_eq!(admit_all(&mut s, 2), vec![b.id]);
    }

    #[test]
    fn mixed_step_forwards_one_token_per_decode_and_chunks_prefill() {
        let mut s = Scheduler::new(cfg());
        let a = s.enqueue(80, 20).unwrap();
        let b = s.enqueue(40, 10).unwrap();
        admit_all(&mut s, 128);
        make_decode(&mut s, a.id);

        let plan = s.plan_step();
        let rows = plan.rows();
        assert_eq!(rows.len(), 13);
        assert!(rows.len() > s.config().max_batch);
        assert!(matches!(
            rows[0],
            StepRow {
                id,
                pos: 80,
                token: StepToken::Decode
            } if id == a.id
        ));
        assert!(matches!(
            rows[1],
            StepRow {
                id,
                pos: 0,
                token: StepToken::Prompt(0)
            } if id == b.id
        ));
        assert!(matches!(
            rows[12],
            StepRow {
                id,
                pos: 11,
                token: StepToken::Prompt(11)
            } if id == b.id
        ));
    }

    #[test]
    fn prefill_budget_limits_chunk_size() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 2));
        let a = s.enqueue(10, 5).unwrap();
        let b = s.enqueue(100, 5).unwrap();
        admit_all(&mut s, 128);
        make_decode(&mut s, a.id);

        let plan = s.plan_step();
        let rows = plan.rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows.iter().filter(|row| row.id == b.id).count(), 2);
        assert!(matches!(rows[1].token, StepToken::Prompt(0)));
        assert!(matches!(rows[2].token, StepToken::Prompt(1)));
    }

    #[test]
    fn prefill_completion_lands_on_last_prompt_row() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 12));
        let a = s.enqueue(10, 5).unwrap();
        let b = s.enqueue(100, 5).unwrap();
        admit_all(&mut s, 128);
        make_decode(&mut s, a.id);
        s.advance_prefill(b.id, 97);

        let plan = s.plan_step();
        let rows = plan.rows();
        assert_eq!(rows.len(), 4);
        assert!(matches!(rows[1].token, StepToken::Prompt(97)));
        assert!(matches!(rows[3].token, StepToken::Prompt(99)));
        assert!(matches!(rows[3].token, StepToken::Prompt(index) if index + 1 == 100));
    }

    #[test]
    fn partial_prefill_continues_chunked_without_decodes() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 12));
        let a = s.enqueue(100, 5).unwrap();
        admit_all(&mut s, 128);
        s.advance_prefill(a.id, 10);

        let plan = s.plan_step();
        let rows = plan.rows();
        assert_eq!(rows.len(), 12);
        assert!(matches!(rows[0].token, StepToken::Prompt(10)));
        assert!(matches!(rows[11].token, StepToken::Prompt(index) if index + 1 < 100));
    }

    #[test]
    fn decode_batch_never_exceeds_limit() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 16, 4, 12));
        let mut ids = Vec::new();
        for _ in 0..8 {
            ids.push(s.enqueue(8, 5).unwrap().id);
        }
        let admitted = admit_all(&mut s, 512);
        assert_eq!(admitted.len(), 8);
        for id in &admitted {
            make_decode(&mut s, *id);
        }
        let plan = s.plan_step();
        assert_eq!(plan.rows().len(), 4);
        assert!(plan.rows().iter().all(|row| row.token == StepToken::Decode));
    }

    #[test]
    fn token_progress_tracks_position_and_completion() {
        let mut s = Scheduler::new(cfg());
        let a = s.enqueue(64, 8).unwrap();
        admit_all(&mut s, 64);
        s.advance_prefill(a.id, 64);
        s.complete_prefill(a.id);
        s.record_token(a.id);
        let seq = *s.active_seq(a.id).unwrap();
        assert_eq!(seq.phase, Phase::Decode);
        assert_eq!(seq.prefilled, 64);
        assert_eq!(seq.generated, 1);
        assert_eq!(seq.position(), 64);
        assert_eq!(seq.used_pages(), 2);
        s.record_token(a.id);
        let seq = *s.active_seq(a.id).unwrap();
        assert_eq!(seq.generated, 2);
        assert_eq!(seq.position(), 65);
    }

    #[test]
    fn decode_position_advances_after_each_forwarded_token() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 12));
        let a = s.enqueue(10, 5).unwrap();
        admit_all(&mut s, 64);
        s.complete_prefill(a.id);
        s.record_token(a.id);
        let plan = s.plan_step();
        assert!(matches!(
            plan.rows()[0],
            StepRow {
                id,
                pos: 10,
                token: StepToken::Decode
            } if id == a.id
        ));
        s.record_token(a.id);
        let plan = s.plan_step();
        assert!(matches!(
            plan.rows()[0],
            StepRow {
                id,
                pos: 11,
                token: StepToken::Decode
            } if id == a.id
        ));
    }

    #[test]
    fn cancelled_request_is_removed_from_active() {
        let mut s = Scheduler::new(cfg());
        let a = s.enqueue(32, 8).unwrap();
        admit_all(&mut s, 64);
        s.advance_prefill(a.id, 12);
        assert!(s.finish(a.id).is_some());
        assert_eq!(s.active_len(), 0);
        assert_eq!(s.plan_step(), StepPlan::Idle);
    }

    #[test]
    fn reservation_shrinks_as_tokens_are_processed() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 12));
        let a = s.enqueue(64, 32).unwrap();
        admit_all(&mut s, 3);
        let seq = *s.active_seq(a.id).unwrap();
        assert_eq!(seq.reserved_pages, 3);
        assert_eq!(seq.used_pages(), 0);
        assert_eq!(s.available_for_new(3), 0);
        s.advance_prefill(a.id, 32);
        let seq = *s.active_seq(a.id).unwrap();
        assert_eq!(seq.used_pages(), 1);
        assert_eq!(seq.future_pages(), 2);
        assert_eq!(s.available_for_new(3), 1);
    }

    #[test]
    fn prefill_in_decode_caps_prompt_rows_in_mixed_step() {
        let mut cfg = cfg();
        cfg.prefill_budget = 32;
        cfg.prefill_in_decode = 2;
        let mut s = Scheduler::new(cfg);
        let a = s.enqueue(80, 20).unwrap();
        let b = s.enqueue(40, 10).unwrap();
        admit_all(&mut s, 128);
        make_decode(&mut s, a.id);
        let plan = s.plan_step();
        let rows = plan.rows();
        let prompts = rows
            .iter()
            .filter(|r| matches!(r.token, StepToken::Prompt(_)))
            .count();
        assert_eq!(prompts, 2);
        assert!(rows.iter().any(|r| r.id == a.id));
        assert!(rows.iter().any(|r| r.id == b.id));
    }

    #[test]
    fn decode_rotation_covers_all_when_active_exceeds_batch() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 16, 4, 12));
        let mut ids = Vec::new();
        for _ in 0..8 {
            ids.push(s.enqueue(8, 5).unwrap().id);
        }
        admit_all(&mut s, 512);
        for id in &ids {
            make_decode(&mut s, *id);
        }
        let mut served = std::collections::HashSet::new();
        for _ in 0..8 {
            if let StepPlan::Mixed { rows, .. } = s.plan_step() {
                for row in rows {
                    served.insert(row.id);
                }
            }
        }
        assert_eq!(served.len(), 8);
    }

    #[test]
    fn prefill_order_rotates_between_plans() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 8, 2));
        let a = s.enqueue(40, 4).unwrap();
        let b = s.enqueue(40, 4).unwrap();
        admit_all(&mut s, 128);
        let first = |plan: StepPlan| plan.rows().first().map(|r| r.id);
        let p1 = first(s.plan_step());
        let p2 = first(s.plan_step());
        assert_ne!(p1, p2);
        assert!(p1 == Some(a.id) || p1 == Some(b.id));
        assert!(p2 == Some(a.id) || p2 == Some(b.id));
    }

    #[test]
    fn concurrent_prefills_are_chunked_not_atomic() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 16, 4));
        let a = s.enqueue(40, 4).unwrap();
        let b = s.enqueue(40, 4).unwrap();
        admit_all(&mut s, 128);
        match s.plan_step() {
            StepPlan::Mixed { rows, .. } => {
                assert!(rows.iter().any(|r| r.id == a.id));
                assert!(rows.iter().any(|r| r.id == b.id));
            }
            other => panic!("expected mixed prefills, got {other:?}"),
        }
    }

    #[test]
    fn single_prefill_stays_atomic() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 16, 64));
        let a = s.enqueue(40, 4).unwrap();
        admit_all(&mut s, 128);
        assert_eq!(s.plan_step(), StepPlan::AtomicPrefill(a.id));
    }

    #[test]
    fn long_single_prefill_is_atomic() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 16, 64));
        let a = s.enqueue(200, 4).unwrap();
        admit_all(&mut s, 128);
        assert_eq!(s.plan_step(), StepPlan::AtomicPrefill(a.id));
    }

    #[test]
    fn long_prefill_with_company_goes_chunked() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 16, 64));
        let a = s.enqueue(200, 4).unwrap();
        let b = s.enqueue(40, 4).unwrap();
        admit_all(&mut s, 128);
        match s.plan_step() {
            StepPlan::Mixed { rows, .. } => {
                assert!(!rows.is_empty());
                assert!(rows.len() <= 64, "chunk respects budget: {}", rows.len());
                assert!(rows.iter().all(|r| r.id == a.id || r.id == b.id));
            }
            other => panic!("expected chunked prefill, got {other:?}"),
        }
    }

    #[test]
    fn prefill_budget_is_shared_round_robin() {
        let mut s = Scheduler::new(SchedulerConfig::new(4096, 8, 6, 2));
        let a = s.enqueue(10, 5).unwrap();
        let b = s.enqueue(50, 5).unwrap();
        let c = s.enqueue(50, 5).unwrap();
        admit_all(&mut s, 128);
        make_decode(&mut s, a.id);

        let plan = s.plan_step();
        let rows = plan.rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].id, b.id);
        assert_eq!(rows[2].id, c.id);
        assert_eq!(rows[1].pos, 0);
        assert_eq!(rows[2].pos, 0);
    }
}
