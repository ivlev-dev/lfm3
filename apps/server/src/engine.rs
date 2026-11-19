use anyhow::{Result, anyhow};
use lfm_app::{
    ActiveSeq, BackendLoader, CacheStats, ChatSession, InferenceBackend, JobId, Sampler, Scheduler,
    SchedulerConfig, SequenceStep, SessionEvent, StepPlan, StepRow, StepToken, TokenizerLoader,
    TokenizerPort,
};
use lfm_chat::ChatTemplate;
use lfm_domain::{Message, SamplingParams, ToolCall, ToolDef};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct ServeConfig {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub template: PathBuf,
    pub ctx: usize,
    pub system: String,
    pub log_requests: bool,
    pub pb: usize,
    pub max_seqs: usize,
    pub prefill_budget: usize,
}

pub struct ChatJob {
    pub messages: Vec<Message>,
    pub params: SamplingParams,
    pub thinking: bool,
    pub tools: Vec<ToolDef>,
    pub raw_prompt: Option<String>,
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
}

impl ChatJob {
    pub fn cancel_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl ChatJob {
    pub fn chat(
        messages: Vec<Message>,
        params: SamplingParams,
        thinking: bool,
        tools: Vec<ToolDef>,
    ) -> Self {
        Self {
            messages,
            params,
            thinking,
            tools,
            raw_prompt: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn completion(prompt: String, params: SamplingParams) -> Self {
        Self {
            messages: Vec::new(),
            params,
            thinking: false,
            tools: Vec::new(),
            raw_prompt: Some(prompt),
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

pub struct CancelGuard(Arc<std::sync::atomic::AtomicBool>);

impl CancelGuard {
    pub fn new(flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self(flag)
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub enum Event {
    Delta(String),
    Thinking(String),
    ToolCall(ToolCall),
    Final {
        text: String,
        thinking: Option<String>,
        tool_calls: Vec<ToolCall>,
        finish_reason: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

struct Job {
    chat: ChatJob,
    tx: Sender<Event>,
}

#[derive(Clone)]
pub struct Engine {
    tx: Sender<Job>,
    metrics: Arc<EngineMetrics>,
    log_requests: bool,
    worker: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

pub struct EngineMetrics {
    started: Instant,
    alive: std::sync::atomic::AtomicBool,
    ctx: AtomicUsize,
    active: AtomicU32,
    waiting: AtomicU32,
    steps: AtomicU64,
    avg_step_us: AtomicU64,
    kv_used_pages: AtomicU64,
    kv_free_pages: AtomicU64,
    kv_cached_pages: AtomicU64,
    cache_hits: AtomicU64,
    reused_tokens: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MetricsUpdate {
    pub ctx: usize,
    pub active: u32,
    pub waiting: u32,
    pub steps: u64,
    pub avg_step_us: u64,
    pub kv_used_pages: u64,
    pub kv_free_pages: u64,
    pub kv_cached_pages: u64,
    pub cache_hits: u64,
    pub reused_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EngineMetricsSnapshot {
    pub ctx: usize,
    pub active: u32,
    pub waiting: u32,
    pub steps: u64,
    pub avg_step_ms: f64,
    pub kv_used_pages: u64,
    pub kv_free_pages: u64,
    pub kv_cached_pages: u64,
    pub cache_hits: u64,
    pub reused_tokens: u64,
    pub uptime_s: f64,
}

impl EngineMetrics {
    pub fn set_alive(&self, alive: bool) {
        self.alive.store(alive, Ordering::SeqCst);
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn new(ctx: usize) -> Self {
        Self {
            started: Instant::now(),
            alive: std::sync::atomic::AtomicBool::new(true),
            ctx: AtomicUsize::new(ctx),
            active: AtomicU32::new(0),
            waiting: AtomicU32::new(0),
            steps: AtomicU64::new(0),
            avg_step_us: AtomicU64::new(0),
            kv_used_pages: AtomicU64::new(0),
            kv_free_pages: AtomicU64::new(0),
            kv_cached_pages: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            reused_tokens: AtomicU64::new(0),
        }
    }

    fn apply(&self, update: MetricsUpdate) {
        self.ctx.store(update.ctx, Ordering::Relaxed);
        self.active.store(update.active, Ordering::Relaxed);
        self.waiting.store(update.waiting, Ordering::Relaxed);
        self.steps.store(update.steps, Ordering::Relaxed);
        self.avg_step_us
            .store(update.avg_step_us, Ordering::Relaxed);
        self.kv_used_pages
            .store(update.kv_used_pages, Ordering::Relaxed);
        self.kv_free_pages
            .store(update.kv_free_pages, Ordering::Relaxed);
        self.kv_cached_pages
            .store(update.kv_cached_pages, Ordering::Relaxed);
        self.cache_hits.store(update.cache_hits, Ordering::Relaxed);
        self.reused_tokens
            .store(update.reused_tokens, Ordering::Relaxed);
    }

    fn snapshot(&self) -> EngineMetricsSnapshot {
        EngineMetricsSnapshot {
            ctx: self.ctx.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            waiting: self.waiting.load(Ordering::Relaxed),
            steps: self.steps.load(Ordering::Relaxed),
            avg_step_ms: self.avg_step_us.load(Ordering::Relaxed) as f64 / 1000.0,
            kv_used_pages: self.kv_used_pages.load(Ordering::Relaxed),
            kv_free_pages: self.kv_free_pages.load(Ordering::Relaxed),
            kv_cached_pages: self.kv_cached_pages.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            uptime_s: self.started.elapsed().as_secs_f64(),
        }
    }
}

const IDLE_POLL: Duration = Duration::from_millis(20);
const BATCH_WINDOW: Duration = Duration::from_micros(2000);

impl Engine {
    pub fn spawn(
        cfg: ServeConfig,
        backend: BackendLoader,
        tokenizer: TokenizerLoader,
    ) -> Result<Self> {
        let log_requests = cfg.log_requests
            || std::env::var("LFM_SCHED_LOG")
                .map(|v| v != "0")
                .unwrap_or(false);
        let (job_tx, job_rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<()>>();
        let metrics = Arc::new(EngineMetrics::new(cfg.ctx));
        let worker_metrics = Arc::clone(&metrics);
        let alive = Arc::clone(&metrics);
        let worker = std::thread::Builder::new()
            .name("engine".to_string())
            .spawn(move || {
                let mut worker =
                    match Worker::load(cfg, log_requests, worker_metrics, backend, tokenizer) {
                        Ok(worker) => {
                            let _ = ready_tx.send(Ok(()));
                            worker
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker.run(job_rx);
                }));
                if outcome.is_err() {
                    eprintln!("engine: воркер упал");
                }
                alive.set_alive(false);
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx: job_tx,
                metrics,
                log_requests,
                worker: Arc::new(Mutex::new(Some(worker))),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!("engine thread stopped during load")),
        }
    }

    pub fn submit(&self, chat: ChatJob) -> Result<Receiver<Event>> {
        let (tx, rx) = channel();
        self.tx
            .send(Job { chat, tx })
            .map_err(|_| anyhow!("engine is not running"))?;
        Ok(rx)
    }

    pub fn log_requests(&self) -> bool {
        self.log_requests
    }

    pub fn alive(&self) -> bool {
        self.metrics.alive()
    }

    pub fn shutdown(self) {
        let Engine { tx, worker, .. } = self;
        drop(tx);
        let handle = match worker.lock() {
            Ok(mut slot) => slot.take(),
            Err(_) => None,
        };
        let Some(handle) = handle else {
            return;
        };
        let (done_tx, done_rx) = channel::<()>();
        std::thread::spawn(move || {
            let _ = handle.join();
            let _ = done_tx.send(());
        });
        if done_rx.recv_timeout(Duration::from_secs(10)).is_err() {
            eprintln!("lfm-server: движок не остановился за 10 с, выходим принудительно");
            std::process::exit(0);
        }
    }

    pub fn metrics(&self) -> EngineMetricsSnapshot {
        self.metrics.snapshot()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenFlow {
    Continue,
    Stop,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Commit {
    Continue,
    Stop,
    Eos,
    Limit,
    Cancelled,
}

struct RequestState {
    slot: usize,
    prompt: Vec<u32>,
    sampler: Sampler,
    session: Option<ChatSession>,
    raw: bool,
    tool_calls: Vec<ToolCall>,
    finish: &'static str,
    tx: Sender<Event>,
    stops: Vec<String>,
    hold: usize,
    max_new: usize,
    tokens: Vec<u32>,
    printed: String,
    content: String,
    thinking: String,
    sent: usize,
    stop_at: Option<usize>,
    next_token: Option<u32>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    eos: u32,
}

impl RequestState {
    fn push_text(&mut self, text: &str) -> TokenFlow {
        self.content.push_str(text);
        if let Some(pos) = first_stop(&self.content, &self.stops) {
            self.stop_at = Some(pos);
            return TokenFlow::Stop;
        }
        let safe = self.content.len().saturating_sub(self.hold);
        let mut cut = safe.max(self.sent);
        while cut > self.sent && !self.content.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut > self.sent {
            let chunk = self.content[self.sent..cut].to_string();
            if self.tx.send(Event::Delta(chunk)).is_err() {
                return TokenFlow::Cancelled;
            }
            self.sent = cut;
        }
        TokenFlow::Continue
    }

    fn process_token(&mut self, tok: &dyn TokenizerPort, token: u32) -> TokenFlow {
        self.tokens.push(token);
        let Ok(text) = tok.decode(&self.tokens, true) else {
            return TokenFlow::Continue;
        };
        let (delta, printed) = next_delta(&text, &self.printed);
        self.printed = printed;
        if delta.is_empty() {
            return TokenFlow::Continue;
        }
        if self.raw {
            return self.push_text(&delta);
        }
        let events = match self.session.as_mut() {
            Some(session) => session.feed(&delta),
            None => Vec::new(),
        };
        self.apply_session_events(events)
    }

    fn apply_session_events(&mut self, events: Vec<SessionEvent>) -> TokenFlow {
        for event in events {
            match event {
                SessionEvent::Thinking(text) => {
                    self.thinking.push_str(&text);
                    if self.tx.send(Event::Thinking(text)).is_err() {
                        return TokenFlow::Cancelled;
                    }
                }
                SessionEvent::Content(text) => match self.push_text(&text) {
                    TokenFlow::Continue => {}
                    other => return other,
                },
                SessionEvent::ToolCall(tc) => {
                    self.tool_calls.push(tc.clone());
                    if self.tx.send(Event::ToolCall(tc)).is_err() {
                        return TokenFlow::Cancelled;
                    }
                }
                SessionEvent::Finished => return TokenFlow::Stop,
            }
        }
        TokenFlow::Continue
    }

    fn commit_token(&mut self, token: u32, flow: TokenFlow) -> Commit {
        match flow {
            TokenFlow::Cancelled => {
                self.next_token = None;
                Commit::Cancelled
            }
            TokenFlow::Stop => {
                self.next_token = None;
                self.finish = "stop";
                Commit::Stop
            }
            TokenFlow::Continue if token == self.eos => {
                self.next_token = None;
                self.finish = "stop";
                Commit::Eos
            }
            TokenFlow::Continue if self.tokens.len() >= self.max_new => {
                self.next_token = None;
                self.finish = "length";
                Commit::Limit
            }
            TokenFlow::Continue => {
                self.next_token = Some(token);
                Commit::Continue
            }
        }
    }
}

struct Metrics {
    enabled: bool,
    steps: u64,
    step_us: u128,
    batch_max: usize,
    active_max: usize,
    reused_tokens: usize,
}

impl Metrics {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            steps: 0,
            step_us: 0,
            batch_max: 0,
            active_max: 0,
            reused_tokens: 0,
        }
    }

    fn record(&mut self, elapsed: Duration, batch: usize, active: usize) {
        self.steps += 1;
        self.step_us += elapsed.as_micros();
        self.batch_max = self.batch_max.max(batch);
        self.active_max = self.active_max.max(active);
    }

    fn reuse(&mut self, tokens: usize) {
        self.reused_tokens += tokens;
    }

    fn avg_step_us(&self) -> u64 {
        if self.steps == 0 {
            0
        } else {
            u64::try_from(self.step_us / u128::from(self.steps)).unwrap_or(u64::MAX)
        }
    }

    fn update(&self, sched: &Scheduler, gpu: &dyn InferenceBackend) -> MetricsUpdate {
        let CacheStats {
            cached_pages,
            used_pages,
            hits,
        } = gpu.cache_stats();
        MetricsUpdate {
            ctx: gpu.max_ctx(),
            active: saturating_u32(sched.active_len()),
            waiting: saturating_u32(sched.waiting_len()),
            steps: self.steps,
            avg_step_us: self.avg_step_us(),
            kv_used_pages: saturating_u64(used_pages),
            kv_free_pages: saturating_u64(gpu.free_pages()),
            kv_cached_pages: saturating_u64(cached_pages),
            cache_hits: hits,
            reused_tokens: saturating_u64(self.reused_tokens),
        }
    }

    fn log_final(&self, id: JobId, prompt: usize, completion: usize, active: usize) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "sched: id={} prompt_tokens={} completion_tokens={} active={} active_max={} batch_max={} steps={} avg_step_us={} reused_tokens={}",
            id.0,
            prompt,
            completion,
            active,
            self.active_max,
            self.batch_max,
            self.steps,
            self.avg_step_us(),
            self.reused_tokens
        );
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Текст, который можно отдавать клиенту: незавершённый UTF-8 (ByteLevel BPE
/// отдаёт U+FFFD на хвосте, если символ разбит между токенами) откладывается
/// до следующего токена.
fn stable_text(text: &str) -> &str {
    text.trim_end_matches('\u{FFFD}')
}

/// Прирост относительно уже отданного текста. При рассинхронизации (декодер
/// переписал ранее отданный префикс) возвращается общий хвост, чтобы не терять
/// вывод: дубликат лучше молчаливой обрезки.
fn next_delta(text: &str, printed: &str) -> (String, String) {
    let stable = stable_text(text);
    if let Some(rest) = stable.strip_prefix(printed) {
        return (rest.to_string(), stable.to_string());
    }
    let mut common = 0usize;
    for (a, b) in stable.chars().zip(printed.chars()) {
        if a != b {
            break;
        }
        common += a.len_utf8();
    }
    (stable[common..].to_string(), stable.to_string())
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

struct Worker {
    gpu: Box<dyn InferenceBackend>,
    tok: Box<dyn TokenizerPort>,
    chat: ChatTemplate,
    system: String,
    sched: Scheduler,
    reqs: HashMap<JobId, RequestState>,
    metrics: Metrics,
    shared: Arc<EngineMetrics>,
}

impl Worker {
    fn load(
        cfg: ServeConfig,
        log_requests: bool,
        shared: Arc<EngineMetrics>,
        backend: BackendLoader,
        tokenizer: TokenizerLoader,
    ) -> Result<Self> {
        let mut gpu = backend(&cfg.model, cfg.ctx)?;
        let _ = gpu.free_sequence(0);
        let max_batch = cfg.pb.clamp(1, 12);
        let max_active = cfg.max_seqs.clamp(1, 16).min(gpu.max_sequences());
        let prefill_budget = cfg.prefill_budget.clamp(1, 128);
        let sched = Scheduler::new(SchedulerConfig::new(
            gpu.max_ctx(),
            max_active,
            max_batch,
            prefill_budget,
        ));
        let tok = tokenizer(&cfg.tokenizer)?;
        let chat = ChatTemplate::from_file(&cfg.template)?;
        let worker = Self {
            gpu,
            tok,
            chat,
            system: cfg.system,
            sched,
            reqs: HashMap::new(),
            metrics: Metrics::new(log_requests),
            shared,
        };
        worker.publish();
        Ok(worker)
    }

    fn publish(&self) {
        self.shared
            .apply(self.metrics.update(&self.sched, self.gpu.as_ref()));
    }

    fn run(&mut self, job_rx: Receiver<Job>) {
        let mut closed = false;
        loop {
            if !closed {
                loop {
                    match job_rx.try_recv() {
                        Ok(job) => self.register(job),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            closed = true;
                            break;
                        }
                    }
                }
            }
            if self.sched.is_empty() {
                if closed {
                    break;
                }
                match job_rx.recv_timeout(IDLE_POLL) {
                    Ok(job) => {
                        self.register(job);
                        std::thread::sleep(BATCH_WINDOW);
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => closed = true,
                }
                continue;
            }
            self.admit();
            if self.sched.active_len() > 0 {
                self.step();
            }
            self.publish();
        }
        let pending: Vec<JobId> = self.reqs.keys().copied().collect();
        for id in pending {
            self.abort(id);
        }
        self.publish();
    }

    fn register(&mut self, job: Job) {
        let Job { chat, tx } = job;
        let ChatJob {
            messages,
            params,
            thinking,
            tools,
            raw_prompt,
            cancel,
        } = chat;
        let raw = raw_prompt.is_some();
        let prompt = if let Some(text) = raw_prompt {
            self.tok.encode(&text, false)
        } else {
            let mut all = Vec::with_capacity(messages.len() + 1);
            if !self.system.is_empty() {
                all.push(Message::system(self.system.clone()));
            }
            all.extend(messages);
            self.chat
                .render(&all, &tools, true, thinking, false)
                .and_then(|text| self.tok.encode(&text, false))
        };
        let prompt = match prompt {
            Ok(prompt) => prompt,
            Err(e) => {
                let _ = tx.send(Event::Error(format!("{e}")));
                return;
            }
        };
        let enqueued = match self.sched.enqueue(prompt.len(), params.max_tokens) {
            Ok(enqueued) => enqueued,
            Err(e) => {
                let _ = tx.send(Event::Error(format!("{e}")));
                return;
            }
        };
        let mut session = None;
        if !raw {
            let mut chat_session = ChatSession::new();
            if thinking {
                chat_session.begin_thinking();
            }
            session = Some(chat_session);
        }
        let mut sampler = Sampler::new(params.clone());
        for &token in &prompt {
            sampler.push(token);
        }
        let hold = params.stop.iter().map(|stop| stop.len()).max().unwrap_or(0);
        let eos = self.tok.eos_id();
        self.reqs.insert(
            enqueued.id,
            RequestState {
                slot: usize::MAX,
                prompt,
                sampler,
                session,
                raw,
                tool_calls: Vec::new(),
                finish: "stop",
                tx,
                stops: params.stop,
                hold,
                max_new: enqueued.max_new,
                tokens: Vec::new(),
                printed: String::new(),
                content: String::new(),
                thinking: String::new(),
                sent: 0,
                stop_at: None,
                next_token: None,
                cancel,
                eos,
            },
        );
        self.publish();
    }

    fn admit(&mut self) {
        let outcomes = self.sched.admit(self.gpu.free_pages());
        for outcome in outcomes {
            match outcome {
                lfm_app::AdmitOutcome::Admitted(id) => match self.gpu.create_sequence() {
                    Ok(slot) => {
                        if let Some(state) = self.reqs.get_mut(&id) {
                            state.slot = slot;
                        }
                        let prompt = self.reqs.get(&id).map(|s| s.prompt.clone());
                        let reused = match prompt {
                            Some(prompt) => match self.gpu.prefill_begin(slot, &prompt) {
                                Ok(reused) => reused,
                                Err(e) => {
                                    self.fail_request(id, format!("{e}"));
                                    continue;
                                }
                            },
                            None => 0,
                        };
                        if reused > 0 {
                            self.sched.advance_prefill(id, reused);
                            self.metrics.reuse(reused);
                        }
                    }
                    Err(e) => self.fail_request(id, format!("{e}")),
                },
                lfm_app::AdmitOutcome::Rejected(id, e) => self.fail_request(id, format!("{e}")),
            }
        }
    }

    fn drop_cancelled(&mut self) {
        let ids: Vec<JobId> = self
            .reqs
            .iter()
            .filter(|(_, state)| state.cancel.load(Ordering::Relaxed))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.fail_request(id, "cancelled by client".to_string());
        }
    }

    fn step(&mut self) {
        let started = Instant::now();
        self.drop_cancelled();
        let plan = self.sched.plan_step();
        let batch = plan.rows().len();
        match plan {
            StepPlan::Idle => return,
            StepPlan::AtomicPrefill(id) => self.atomic_prefill(id),
            StepPlan::Mixed { rows, need_logits } => self.mixed_step(&rows, need_logits),
        }
        self.metrics
            .record(started.elapsed(), batch, self.sched.active_len());
    }

    fn atomic_prefill(&mut self, id: JobId) {
        let Some(state) = self.reqs.get(&id) else {
            self.sched.finish(id);
            return;
        };
        if state.cancel.load(Ordering::Relaxed) {
            self.fail_request(id, "cancelled by client".to_string());
            return;
        }
        let slot = state.slot;
        let prompt = state.prompt.clone();
        let reused = match self.gpu.prefill(slot, &prompt, true) {
            Ok(reused) => reused,
            Err(e) => {
                self.fail_request(id, format!("{e}"));
                return;
            }
        };
        self.metrics.reuse(reused);
        let logits = match self.gpu.logits() {
            Ok(logits) => logits,
            Err(e) => {
                self.fail_request(id, format!("{e}"));
                return;
            }
        };
        self.sched.complete_prefill(id);
        self.accept_token(id, logits);
    }

    fn mixed_step(&mut self, rows: &[StepRow], _need_logits: bool) {
        let max_batch = self
            .sched
            .config()
            .max_batch
            .max(self.gpu.prefill_batch_size())
            .max(1);
        for slice in rows.chunks(max_batch) {
            if !self.mixed_slice(slice) {
                return;
            }
        }
    }

    fn mixed_slice(&mut self, rows: &[StepRow]) -> bool {
        let need_logits = rows.iter().any(|row| match row.token {
            StepToken::Decode => true,
            StepToken::Prompt(index) => self
                .reqs
                .get(&row.id)
                .map(|state| index + 1 >= state.prompt.len())
                .unwrap_or(false),
        });
        let mut reqs: Vec<SequenceStep> = Vec::with_capacity(rows.len());
        let mut failure: Option<(JobId, String)> = None;
        for row in rows {
            let Some(state) = self.reqs.get(&row.id) else {
                failure = Some((row.id, "scheduler lost sequence".to_string()));
                break;
            };
            let token = match row.token {
                StepToken::Prompt(index) => match state.prompt.get(index) {
                    Some(token) => *token,
                    None => {
                        failure = Some((row.id, "prompt index out of range".to_string()));
                        break;
                    }
                },
                StepToken::Decode => match state.next_token {
                    Some(token) => token,
                    None => {
                        failure = Some((row.id, "missing pending token".to_string()));
                        break;
                    }
                },
            };
            reqs.push(SequenceStep {
                slot: state.slot,
                token,
                pos: row.pos,
            });
        }
        if let Some((id, reason)) = failure {
            self.fail_request(id, reason);
            return false;
        }
        if let Err(e) = self.gpu.step(&reqs, need_logits) {
            let mut ids: Vec<JobId> = rows.iter().map(|row| row.id).collect();
            ids.sort();
            ids.dedup();
            for id in ids {
                self.fail_request(id, format!("{e}"));
            }
            return false;
        }
        for (row_index, row) in rows.iter().enumerate() {
            match row.token {
                StepToken::Prompt(_) => {
                    self.sched.advance_prefill(row.id, 1);
                    let done = self
                        .sched
                        .active_seq(row.id)
                        .map(ActiveSeq::is_prefill_done)
                        .unwrap_or(false);
                    if !done {
                        continue;
                    }
                    if !need_logits {
                        self.fail_request(row.id, "prefill completion without logits".to_string());
                        continue;
                    }
                    let logits = match self.gpu.logits_row(row_index) {
                        Ok(logits) => logits,
                        Err(e) => {
                            self.fail_request(row.id, format!("{e}"));
                            continue;
                        }
                    };
                    let slot = self.reqs.get(&row.id).map(|state| state.slot);
                    if let Some(slot) = slot
                        && let Err(e) = self.gpu.prefill_finish(slot)
                    {
                        self.fail_request(row.id, format!("{e}"));
                        continue;
                    }
                    self.sched.complete_prefill(row.id);
                    self.accept_token(row.id, logits);
                }
                StepToken::Decode => {
                    let logits = match self.gpu.logits_row(row_index) {
                        Ok(logits) => logits,
                        Err(e) => {
                            self.fail_request(row.id, format!("{e}"));
                            continue;
                        }
                    };
                    self.accept_token(row.id, logits);
                }
            }
        }
        true
    }

    fn accept_token(&mut self, id: JobId, mut logits: Vec<f32>) {
        let tok = &self.tok;
        let Some(state) = self.reqs.get_mut(&id) else {
            return;
        };
        let token = state.sampler.sample(&mut logits);
        state.sampler.push(token);
        let flow = state.process_token(tok.as_ref(), token);
        let commit = state.commit_token(token, flow);
        self.sched.record_token(id);
        match commit {
            Commit::Continue => {}
            Commit::Cancelled => self.abort(id),
            Commit::Stop | Commit::Eos | Commit::Limit => self.finish_ok(id),
        }
    }

    fn finish_ok(&mut self, id: JobId) {
        let Some(mut state) = self.reqs.remove(&id) else {
            self.sched.finish(id);
            return;
        };
        self.release_slot(&mut state);
        self.sched.finish(id);
        if let Some(mut session) = state.session.take() {
            let events = session.finish();
            let _ = state.apply_session_events(events);
        }
        let final_end = state.stop_at.unwrap_or(state.content.len());
        let mut cut = final_end.max(state.sent);
        while cut > state.sent && !state.content.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut > state.sent {
            let _ = state
                .tx
                .send(Event::Delta(state.content[state.sent..cut].to_string()));
        }
        let text = state.content[..final_end].to_string();
        let prompt_tokens = state.prompt.len();
        let completion_tokens = state.tokens.len();
        let finish_reason = if state.tool_calls.is_empty() {
            state.finish
        } else {
            "tool_calls"
        };
        let _ = state.tx.send(Event::Final {
            text,
            thinking: if state.thinking.is_empty() {
                None
            } else {
                Some(state.thinking.clone())
            },
            tool_calls: state.tool_calls.clone(),
            finish_reason,
            prompt_tokens,
            completion_tokens,
        });
        self.metrics.log_final(
            id,
            prompt_tokens,
            completion_tokens,
            self.sched.active_len(),
        );
    }

    fn fail_request(&mut self, id: JobId, message: String) {
        self.sched.finish(id);
        let Some(mut state) = self.reqs.remove(&id) else {
            return;
        };
        self.release_slot(&mut state);
        let _ = state.tx.send(Event::Error(message));
    }

    fn abort(&mut self, id: JobId) {
        self.sched.finish(id);
        if let Some(mut state) = self.reqs.remove(&id) {
            self.release_slot(&mut state);
        }
    }

    fn release_slot(&mut self, state: &mut RequestState) {
        if state.slot != usize::MAX {
            let _ = self.gpu.free_sequence(state.slot);
            state.slot = usize::MAX;
        }
    }
}

fn first_stop(content: &str, stops: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = content.find(stop) {
            best = Some(match best {
                Some(cur) => cur.min(pos),
                None => pos,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(max_new: usize, eos: u32) -> RequestState {
        let (tx, _rx) = channel();
        RequestState {
            slot: 0,
            prompt: vec![1, 2, 3],
            sampler: Sampler::new(SamplingParams::default()),
            session: Some(ChatSession::new()),
            raw: false,
            tool_calls: Vec::new(),
            finish: "stop",
            tx,
            stops: Vec::new(),
            hold: 0,
            max_new,
            tokens: Vec::new(),
            printed: String::new(),
            content: String::new(),
            thinking: String::new(),
            sent: 0,
            stop_at: None,
            next_token: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            eos,
        }
    }

    #[test]
    fn stop_scan_picks_earliest() {
        let stops = vec!["END".to_string(), "bye".to_string()];
        assert_eq!(first_stop("hello END bye", &stops), Some(6));
        assert_eq!(first_stop("hello bye END", &stops), Some(6));
        assert_eq!(first_stop("hello", &stops), None);
        assert_eq!(first_stop("hello", &[]), None);
        assert_eq!(first_stop("x", &["".to_string()]), None);
    }

    #[test]
    fn commit_token_sets_pending_token_after_prefill() {
        let mut state = test_state(8, 999);
        assert_eq!(state.next_token, None);
        let commit = state.commit_token(42, TokenFlow::Continue);
        assert_eq!(commit, Commit::Continue);
        assert_eq!(state.next_token, Some(42));
    }

    #[test]
    fn commit_token_replaces_pending_token_each_decode() {
        let mut state = test_state(8, 999);
        state.tokens.push(7);
        assert_eq!(state.commit_token(7, TokenFlow::Continue), Commit::Continue);
        assert_eq!(state.next_token, Some(7));
        state.tokens.push(8);
        assert_eq!(state.commit_token(8, TokenFlow::Continue), Commit::Continue);
        assert_eq!(state.next_token, Some(8));
        state.tokens.push(9);
        assert_eq!(state.commit_token(9, TokenFlow::Continue), Commit::Continue);
        assert_eq!(state.next_token, Some(9));
    }

    #[test]
    fn commit_token_clears_pending_on_terminal_outcomes() {
        let mut state = test_state(4, 2);
        state.tokens.push(7);
        assert_eq!(state.commit_token(7, TokenFlow::Stop), Commit::Stop);
        assert_eq!(state.next_token, None);

        state.tokens.push(8);
        assert_eq!(
            state.commit_token(8, TokenFlow::Cancelled),
            Commit::Cancelled
        );
        assert_eq!(state.next_token, None);

        assert_eq!(state.commit_token(2, TokenFlow::Continue), Commit::Eos);
        assert_eq!(state.next_token, None);

        state.tokens = vec![7, 8, 9, 10];
        assert_eq!(state.commit_token(11, TokenFlow::Continue), Commit::Limit);
        assert_eq!(state.next_token, None);
    }

    #[test]
    fn commit_token_keeps_pending_below_limit() {
        let mut state = test_state(4, 2);
        state.tokens = vec![7, 8];
        assert_eq!(state.commit_token(9, TokenFlow::Continue), Commit::Continue);
        assert_eq!(state.next_token, Some(9));
    }

    #[test]
    fn fresh_metrics_snapshot_is_zeroed() {
        let metrics = EngineMetrics::new(4096);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.ctx, 4096);
        assert_eq!(snapshot.active, 0);
        assert_eq!(snapshot.waiting, 0);
        assert_eq!(snapshot.steps, 0);
        assert_eq!(snapshot.avg_step_ms, 0.0);
        assert_eq!(snapshot.kv_used_pages, 0);
        assert_eq!(snapshot.kv_free_pages, 0);
        assert_eq!(snapshot.kv_cached_pages, 0);
        assert_eq!(snapshot.cache_hits, 0);
        assert_eq!(snapshot.reused_tokens, 0);
        assert!(snapshot.uptime_s >= 0.0);
    }

    #[test]
    fn apply_publishes_step_active_and_cache_counters() {
        let metrics = EngineMetrics::new(1024);
        metrics.apply(MetricsUpdate {
            ctx: 2048,
            active: 3,
            waiting: 2,
            steps: 11,
            avg_step_us: 2500,
            kv_used_pages: 7,
            kv_free_pages: 9,
            kv_cached_pages: 4,
            cache_hits: 12,
            reused_tokens: 128,
        });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.ctx, 2048);
        assert_eq!(snapshot.active, 3);
        assert_eq!(snapshot.waiting, 2);
        assert_eq!(snapshot.steps, 11);
        assert_eq!(snapshot.avg_step_ms, 2.5);
        assert_eq!(snapshot.kv_used_pages, 7);
        assert_eq!(snapshot.kv_free_pages, 9);
        assert_eq!(snapshot.kv_cached_pages, 4);
        assert_eq!(snapshot.cache_hits, 12);
        assert_eq!(snapshot.reused_tokens, 128);
    }

    #[test]
    fn metrics_shared_across_clones_see_updates() {
        let metrics = Arc::new(EngineMetrics::new(512));
        let reader = Arc::clone(&metrics);
        assert_eq!(reader.snapshot().steps, 0);
        metrics.apply(MetricsUpdate {
            ctx: 512,
            active: 1,
            waiting: 0,
            steps: 1,
            avg_step_us: 1000,
            kv_used_pages: 2,
            kv_free_pages: 30,
            kv_cached_pages: 1,
            cache_hits: 0,
            reused_tokens: 32,
        });
        let snapshot = reader.snapshot();
        assert_eq!(snapshot.steps, 1);
        assert_eq!(snapshot.active, 1);
        assert_eq!(snapshot.avg_step_ms, 1.0);
        assert_eq!(snapshot.kv_used_pages, 2);
        assert_eq!(snapshot.kv_free_pages, 30);
        assert_eq!(snapshot.kv_cached_pages, 1);
        assert_eq!(snapshot.reused_tokens, 32);
    }

    #[test]
    fn record_accumulates_steps_and_average() {
        let mut metrics = Metrics::new(false);
        assert_eq!(metrics.avg_step_us(), 0);
        metrics.record(Duration::from_micros(400), 3, 2);
        metrics.record(Duration::from_micros(600), 1, 4);
        assert_eq!(metrics.steps, 2);
        assert_eq!(metrics.avg_step_us(), 500);
        assert_eq!(metrics.batch_max, 3);
        assert_eq!(metrics.active_max, 4);
        metrics.reuse(64);
        metrics.reuse(36);
        assert_eq!(metrics.reused_tokens, 100);
    }

    #[test]
    fn saturating_helpers_clamp_at_limit() {
        assert_eq!(saturating_u32(7), 7);
        assert_eq!(saturating_u32(usize::MAX), u32::MAX);
        assert_eq!(saturating_u64(7), 7);
        assert_eq!(saturating_u64(usize::MAX), u64::MAX);
    }

    #[test]
    fn delta_handles_split_utf8() {
        let (d, printed) = next_delta("Привет", "");
        assert_eq!(d, "Привет");
        assert_eq!(printed, "Привет");
        let (d, printed) = next_delta("Привет, мир", "Привет");
        assert_eq!(d, ", мир");
        assert_eq!(printed, "Привет, мир");
        let (d, _) = next_delta("Привет, мир", "Привет, мир");
        assert!(d.is_empty());
    }

    #[test]
    fn delta_holds_back_incomplete_char() {
        let (d, printed) = next_delta("Привет\u{FFFD}", "Привет");
        assert!(d.is_empty());
        assert_eq!(printed, "Привет");
        let (d, printed) = next_delta("Привет!", "Привет");
        assert_eq!(d, "!");
        assert_eq!(printed, "Привет!");
    }

    #[test]
    fn delta_resyncs_instead_of_dropping() {
        let (d, printed) = next_delta("abcdef", "abcX");
        assert_eq!(d, "def");
        assert_eq!(printed, "abcdef");
        let (d, printed) = next_delta("ab", "abcd");
        assert!(d.is_empty());
        assert_eq!(printed, "ab");
    }
}
