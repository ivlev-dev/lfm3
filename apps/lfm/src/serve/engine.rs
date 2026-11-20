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
    pub max_seqs: usize,
    pub prefill_budget: usize,
    pub prefill_in_decode: usize,
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
    avg_gpu_us: AtomicU64,
    avg_logits_us: AtomicU64,
    avg_sample_us: AtomicU64,
    avg_detok_us: AtomicU64,
    rows_total: AtomicU64,
    rows_logits: AtomicU64,
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
    pub avg_gpu_us: u64,
    pub avg_logits_us: u64,
    pub avg_sample_us: u64,
    pub avg_detok_us: u64,
    pub rows_total: u64,
    pub rows_logits: u64,
    pub kv_used_pages: u64,
    pub kv_free_pages: u64,
    pub kv_cached_pages: u64,
    pub cache_hits: u64,
    pub reused_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub struct EngineStageMetrics {
    pub avg_gpu_us: u64,
    pub avg_logits_us: u64,
    pub avg_sample_us: u64,
    pub avg_detok_us: u64,
    pub rows_total: u64,
    pub rows_logits: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EngineMetricsSnapshot {
    pub ctx: usize,
    pub active: u32,
    pub waiting: u32,
    pub steps: u64,
    pub avg_step_ms: f64,
    pub avg_gpu_us: u64,
    pub avg_logits_us: u64,
    pub avg_sample_us: u64,
    pub avg_detok_us: u64,
    pub rows_total: u64,
    pub rows_logits: u64,
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
            avg_gpu_us: AtomicU64::new(0),
            avg_logits_us: AtomicU64::new(0),
            avg_sample_us: AtomicU64::new(0),
            avg_detok_us: AtomicU64::new(0),
            rows_total: AtomicU64::new(0),
            rows_logits: AtomicU64::new(0),
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
        self.avg_gpu_us.store(update.avg_gpu_us, Ordering::Relaxed);
        self.avg_logits_us
            .store(update.avg_logits_us, Ordering::Relaxed);
        self.avg_sample_us
            .store(update.avg_sample_us, Ordering::Relaxed);
        self.avg_detok_us
            .store(update.avg_detok_us, Ordering::Relaxed);
        self.rows_total.store(update.rows_total, Ordering::Relaxed);
        self.rows_logits
            .store(update.rows_logits, Ordering::Relaxed);
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
            rows_total: self.rows_total.load(Ordering::Relaxed),
            rows_logits: self.rows_logits.load(Ordering::Relaxed),
            avg_gpu_us: self.avg_gpu_us.load(Ordering::Relaxed),
            avg_logits_us: self.avg_logits_us.load(Ordering::Relaxed),
            avg_sample_us: self.avg_sample_us.load(Ordering::Relaxed),
            avg_detok_us: self.avg_detok_us.load(Ordering::Relaxed),
            kv_used_pages: self.kv_used_pages.load(Ordering::Relaxed),
            kv_free_pages: self.kv_free_pages.load(Ordering::Relaxed),
            kv_cached_pages: self.kv_cached_pages.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            uptime_s: self.started.elapsed().as_secs_f64(),
        }
    }

    #[allow(dead_code)]
    fn stage_snapshot(&self) -> EngineStageMetrics {
        EngineStageMetrics {
            avg_gpu_us: self.avg_gpu_us.load(Ordering::Relaxed),
            avg_logits_us: self.avg_logits_us.load(Ordering::Relaxed),
            avg_sample_us: self.avg_sample_us.load(Ordering::Relaxed),
            avg_detok_us: self.avg_detok_us.load(Ordering::Relaxed),
            rows_total: self.rows_total.load(Ordering::Relaxed),
            rows_logits: self.rows_logits.load(Ordering::Relaxed),
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
        let log_requests = cfg.log_requests;
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

    #[allow(dead_code)]
    pub fn stage_metrics(&self) -> EngineStageMetrics {
        self.metrics.stage_snapshot()
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
        if let Some(pos) = self.find_stop(text.len()) {
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

    fn find_stop(&self, appended: usize) -> Option<usize> {
        if self.stops.is_empty() {
            return None;
        }
        let start = stop_scan_start(self.content.len(), appended, self.hold);
        let mut start = start;
        while start > 0 && !self.content.is_char_boundary(start) {
            start -= 1;
        }
        first_stop(&self.content[start..], &self.stops).map(|pos| start + pos)
    }

    fn process_token(&mut self, tok: &dyn TokenizerPort, token: u32) -> TokenFlow {
        // EOS — служебный терминальный токен: он не является выданным текстом
        // и не входит в completion_tokens (семантика usage в OpenAI API).
        if token == self.eos {
            return TokenFlow::Continue;
        }
        self.tokens.push(token);
        let Some(delta) = decode_delta(tok, &self.tokens) else {
            return TokenFlow::Continue;
        };
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
    gpu_us: u128,
    gpu_calls: u64,
    logits_us: u128,
    logits_calls: u64,
    sample_us: u128,
    samples: u64,
    detok_us: u128,
    detoks: u64,
    rows_total: usize,
    rows_logits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Gpu,
    Logits,
    Sample,
    Detok,
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
            gpu_us: 0,
            gpu_calls: 0,
            logits_us: 0,
            logits_calls: 0,
            sample_us: 0,
            samples: 0,
            detok_us: 0,
            detoks: 0,
            rows_total: 0,
            rows_logits: 0,
        }
    }

    fn record(&mut self, elapsed: Duration, batch: usize, active: usize) {
        self.steps += 1;
        self.step_us += elapsed.as_micros();
        self.batch_max = self.batch_max.max(batch);
        self.active_max = self.active_max.max(active);
    }

    fn stage(&mut self, stage: Stage, elapsed: Duration) {
        let us = elapsed.as_micros();
        match stage {
            Stage::Gpu => {
                self.gpu_us += us;
                self.gpu_calls += 1;
            }
            Stage::Logits => {
                self.logits_us += us;
                self.logits_calls += 1;
            }
            Stage::Sample => {
                self.sample_us += us;
                self.samples += 1;
            }
            Stage::Detok => {
                self.detok_us += us;
                self.detoks += 1;
            }
        }
    }

    fn rows(&mut self, total: usize, logits: usize) {
        self.rows_total += total;
        self.rows_logits += logits;
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
            avg_gpu_us: average_us(self.gpu_us, self.gpu_calls),
            avg_logits_us: average_us(self.logits_us, self.logits_calls),
            avg_sample_us: average_us(self.sample_us, self.samples),
            avg_detok_us: average_us(self.detok_us, self.detoks),
            rows_total: saturating_u64(self.rows_total),
            rows_logits: saturating_u64(self.rows_logits),
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

fn greedy_token(ids: &[(usize, u32)], row: usize) -> Option<u32> {
    ids.iter()
        .find(|(index, _)| *index == row)
        .map(|(_, token)| *token)
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

const DETOK_CONTEXT: usize = 32;
const STOP_SCAN_MARGIN: usize = 4;

/// Декодируется окно из последних `DETOK_CONTEXT` токенов и предыдущее окно
/// без последнего токена: прирост считается по разнице, поэтому стоимость
/// токена не зависит от длины ответа. Общая для обоих окон граница (например,
/// разорванный UTF-8 символ) даёт одинаковый U+FFFD-префикс и не мешает.
fn decode_delta(tok: &dyn TokenizerPort, tokens: &[u32]) -> Option<String> {
    let end = tokens.len();
    if end == 0 {
        return Some(String::new());
    }
    let start = (end - 1).saturating_sub(DETOK_CONTEXT);
    let prev = tok.decode(&tokens[start..end - 1], true).ok()?;
    let cur = tok.decode(&tokens[start..end], true).ok()?;
    let (delta, _) = next_delta(&cur, stable_text(&prev));
    Some(delta)
}

/// Начало окна поиска стоп-строк: новый фрагмент плюс запас на самую длинную
/// стоп-строку, которая могла начаться раньше и закончиться в нём.
fn stop_scan_start(len: usize, appended: usize, max_stop_len: usize) -> usize {
    len.saturating_sub(
        appended
            .saturating_add(max_stop_len)
            .saturating_add(STOP_SCAN_MARGIN),
    )
}

fn saturating_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn average_us(total: u128, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    u64::try_from(total / u128::from(count)).unwrap_or(u64::MAX)
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
        let batch_cap = gpu.prefill_batch_size().max(1);
        let max_active = cfg.max_seqs.clamp(1, 32).min(gpu.max_sequences());
        let prefill_budget = cfg.prefill_budget.clamp(1, 128);
        let mut scfg = SchedulerConfig::new(gpu.max_ctx(), max_active, batch_cap, prefill_budget);
        scfg.max_decode = max_active.min(batch_cap);
        scfg.prefill_in_decode = cfg.prefill_in_decode.min(batch_cap);
        let sched = Scheduler::new(scfg);
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
        let requested = if params.max_tokens == 0 {
            self.sched.config().max_ctx.saturating_sub(prompt.len())
        } else {
            params.max_tokens
        };
        let enqueued = match self.sched.enqueue(prompt.len(), requested) {
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
        push_prompt_window(&mut sampler, &prompt, params.repeat_penalty_window);
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
            StepPlan::Mixed { rows } => self.mixed_step(&rows),
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
        let gpu_started = Instant::now();
        let prefill = self.gpu.prefill(slot, &prompt, true);
        self.metrics.stage(Stage::Gpu, gpu_started.elapsed());
        let reused = match prefill {
            Ok(reused) => reused,
            Err(e) => {
                self.fail_request(id, format!("{e}"));
                return;
            }
        };
        self.metrics.rows(prompt.len(), 1);
        self.metrics.reuse(reused);
        let logits = match self.logits_full() {
            Ok(logits) => logits,
            Err(e) => {
                self.fail_request(id, format!("{e}"));
                return;
            }
        };
        self.sched.complete_prefill(id);
        self.accept_token(id, logits);
    }

    fn mixed_step(&mut self, rows: &[StepRow]) {
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
        let mut reqs: Vec<SequenceStep> = Vec::with_capacity(rows.len());
        let mut rows_needed: Vec<usize> = Vec::with_capacity(rows.len());
        let mut failure: Option<(JobId, String)> = None;
        for (row_index, row) in rows.iter().enumerate() {
            let Some(state) = self.reqs.get(&row.id) else {
                failure = Some((row.id, "scheduler lost sequence".to_string()));
                break;
            };
            let token = match row.token {
                StepToken::Prompt(index) => match state.prompt.get(index) {
                    Some(token) => {
                        if index + 1 >= state.prompt.len() {
                            rows_needed.push(row_index);
                        }
                        *token
                    }
                    None => {
                        failure = Some((row.id, "prompt index out of range".to_string()));
                        break;
                    }
                },
                StepToken::Decode => {
                    rows_needed.push(row_index);
                    match state.next_token {
                        Some(token) => token,
                        None => {
                            failure = Some((row.id, "missing pending token".to_string()));
                            break;
                        }
                    }
                }
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
        let gpu_started = Instant::now();
        let stepped = if rows_needed.is_empty() {
            self.gpu.step(&reqs, false)
        } else {
            self.gpu.step_rows(&reqs, &rows_needed)
        };
        self.metrics.stage(Stage::Gpu, gpu_started.elapsed());
        if let Err(e) = stepped {
            let mut ids: Vec<JobId> = rows.iter().map(|row| row.id).collect();
            ids.sort();
            ids.dedup();
            for id in ids {
                self.fail_request(id, format!("{e}"));
            }
            return false;
        }
        self.metrics.rows(reqs.len(), rows_needed.len());
        let mut greedy: Vec<(usize, u32)> = Vec::new();
        if !rows_needed.is_empty() {
            let wanted: Vec<usize> = rows_needed
                .iter()
                .copied()
                .filter(|&index| {
                    rows.get(index)
                        .and_then(|row| self.reqs.get(&row.id))
                        .map(|state| state.sampler.is_greedy())
                        .unwrap_or(false)
                })
                .collect();
            if !wanted.is_empty() {
                let started = Instant::now();
                if let Ok(ids) = self.gpu.argmax_rows(&wanted) {
                    greedy = wanted
                        .into_iter()
                        .zip(ids.into_iter().map(|(id, _)| id))
                        .collect();
                }
                self.metrics.stage(Stage::Logits, started.elapsed());
            }
        }
        for (row_index, row) in rows.iter().enumerate() {
            let wanted = rows_needed.binary_search(&row_index).is_ok();
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
                    if !wanted {
                        self.fail_request(row.id, "prefill completion without logits".to_string());
                        continue;
                    }
                    let slot = match self.reqs.get(&row.id).map(|state| state.slot) {
                        Some(slot) => slot,
                        None => continue,
                    };
                    if let Some(token) = greedy_token(&greedy, row_index) {
                        if let Err(e) = self.gpu.prefill_finish(slot) {
                            self.fail_request(row.id, format!("{e}"));
                            continue;
                        }
                        self.sched.complete_prefill(row.id);
                        self.accept_sampled(row.id, token);
                        continue;
                    }
                    let logits = match self.logits_at(row_index) {
                        Ok(logits) => logits,
                        Err(e) => {
                            self.fail_request(row.id, format!("{e}"));
                            continue;
                        }
                    };
                    if let Err(e) = self.gpu.prefill_finish(slot) {
                        self.fail_request(row.id, format!("{e}"));
                        continue;
                    }
                    self.sched.complete_prefill(row.id);
                    self.accept_token(row.id, logits);
                }
                StepToken::Decode => {
                    if !wanted {
                        continue;
                    }
                    if let Some(token) = greedy_token(&greedy, row_index) {
                        self.accept_sampled(row.id, token);
                        continue;
                    }
                    let logits = match self.logits_at(row_index) {
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

    fn logits_at(&mut self, row: usize) -> Result<Vec<f32>> {
        let started = Instant::now();
        let logits = self.gpu.logits_row(row);
        self.metrics.stage(Stage::Logits, started.elapsed());
        logits
    }

    fn logits_full(&mut self) -> Result<Vec<f32>> {
        let started = Instant::now();
        let logits = self.gpu.logits();
        self.metrics.stage(Stage::Logits, started.elapsed());
        logits
    }

    fn accept_token(&mut self, id: JobId, mut logits: Vec<f32>) {
        let sample_started = Instant::now();
        let Some(state) = self.reqs.get_mut(&id) else {
            return;
        };
        let token = state.sampler.sample(&mut logits);
        self.metrics.stage(Stage::Sample, sample_started.elapsed());
        self.accept_sampled(id, token);
    }

    fn accept_sampled(&mut self, id: JobId, token: u32) {
        let tok = &self.tok;
        let Some(state) = self.reqs.get_mut(&id) else {
            return;
        };
        state.sampler.push(token);
        let detok_started = Instant::now();
        let flow = state.process_token(tok.as_ref(), token);
        self.metrics.stage(Stage::Detok, detok_started.elapsed());
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

fn push_prompt_window(sampler: &mut Sampler, prompt: &[u32], window: usize) {
    let keep = window.max(1);
    let skip = prompt.len().saturating_sub(keep);
    for &token in &prompt[skip..] {
        sampler.push(token);
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
        test_state_with_rx(max_new, eos).0
    }

    fn test_state_with_rx(max_new: usize, eos: u32) -> (RequestState, Receiver<Event>) {
        let (tx, rx) = channel();
        let state = RequestState {
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
            content: String::new(),
            thinking: String::new(),
            sent: 0,
            stop_at: None,
            next_token: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            eos,
        };
        (state, rx)
    }

    fn raw_state(stops: Vec<String>) -> (RequestState, Receiver<Event>) {
        let (mut state, rx) = test_state_with_rx(64, u32::MAX);
        state.raw = true;
        state.session = None;
        state.hold = stops.iter().map(|stop| stop.len()).max().unwrap_or(0);
        state.stops = stops;
        (state, rx)
    }

    fn deltas(rx: &Receiver<Event>) -> String {
        let mut out = String::new();
        while let Ok(event) = rx.try_recv() {
            if let Event::Delta(text) = event {
                out.push_str(&text);
            }
        }
        out
    }

    /// Байтовый ByteLevel-подобный токенизатор: id — байт, декодер режет
    /// неполные UTF-8 последовательности в U+FFFD. `eos` в декодированный
    /// текст не попадает, как при `skip_special_tokens = true`.
    struct ByteTokenizer {
        eos: u32,
    }

    impl ByteTokenizer {
        const fn new() -> Self {
            Self { eos: u32::MAX }
        }

        const fn with_eos(eos: u32) -> Self {
            Self { eos }
        }
    }

    impl TokenizerPort for ByteTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes: Vec<u8> = ids
                .iter()
                .filter(|&&id| id != self.eos)
                .map(|&id| u8::try_from(id).unwrap_or(b'?'))
                .collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn eos_id(&self) -> u32 {
            self.eos
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
        assert_eq!(metrics.stage_snapshot(), EngineStageMetrics::default());
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
            avg_gpu_us: 1700,
            avg_logits_us: 90,
            avg_sample_us: 12,
            avg_detok_us: 7,
            rows_total: 64,
            rows_logits: 5,
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
        let stages = metrics.stage_snapshot();
        assert_eq!(stages.avg_gpu_us, 1700);
        assert_eq!(stages.avg_logits_us, 90);
        assert_eq!(stages.avg_sample_us, 12);
        assert_eq!(stages.avg_detok_us, 7);
        assert_eq!(stages.rows_total, 64);
        assert_eq!(stages.rows_logits, 5);
    }

    #[test]
    fn stage_averages_are_per_call() {
        let mut metrics = Metrics::new(false);
        metrics.stage(Stage::Gpu, Duration::from_micros(900));
        metrics.stage(Stage::Gpu, Duration::from_micros(300));
        metrics.stage(Stage::Logits, Duration::from_micros(40));
        metrics.stage(Stage::Sample, Duration::from_micros(10));
        metrics.stage(Stage::Detok, Duration::from_micros(2));
        metrics.stage(Stage::Detok, Duration::from_micros(6));
        metrics.rows(8, 3);
        metrics.rows(4, 1);
        assert_eq!(average_us(metrics.gpu_us, metrics.gpu_calls), 600);
        assert_eq!(average_us(metrics.logits_us, metrics.logits_calls), 40);
        assert_eq!(average_us(metrics.sample_us, metrics.samples), 10);
        assert_eq!(average_us(metrics.detok_us, metrics.detoks), 4);
        assert_eq!(metrics.rows_total, 12);
        assert_eq!(metrics.rows_logits, 4);
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
            avg_gpu_us: 800,
            avg_logits_us: 30,
            avg_sample_us: 5,
            avg_detok_us: 3,
            rows_total: 16,
            rows_logits: 2,
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
        assert_eq!(reader.stage_snapshot().rows_logits, 2);
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

    #[test]
    fn incremental_detok_matches_full_decode() {
        let tok = ByteTokenizer::new();
        let text = "Привет, мир! 42 — тест";
        let ids = tok.encode(text, false).unwrap();
        let mut printed = String::new();
        let mut out = String::new();
        for end in 1..=ids.len() {
            let delta = decode_delta(&tok, &ids[..end]).expect("delta");
            let full = tok.decode(&ids[..end], true).unwrap();
            let (expected, next) = next_delta(&full, &printed);
            assert_eq!(delta, expected, "token {end}");
            printed = next;
            out.push_str(&delta);
        }
        assert_eq!(out, text);
    }

    #[test]
    fn incremental_detok_is_constant_window() {
        let tok = ByteTokenizer::new();
        let ids: Vec<u32> = std::iter::repeat_n(b'a', DETOK_CONTEXT * 4)
            .map(u32::from)
            .collect();
        let delta = decode_delta(&tok, &ids).expect("delta");
        assert_eq!(delta.len(), 1);
        assert_eq!(decode_delta(&tok, &[]), Some(String::new()));
    }

    #[test]
    fn process_token_streams_raw_text() {
        let tok = ByteTokenizer::new();
        let (mut state, rx) = test_state_with_rx(64, u32::MAX);
        state.raw = true;
        state.session = None;
        for id in tok.encode("Привет, мир", false).unwrap() {
            assert_eq!(state.process_token(&tok, id), TokenFlow::Continue);
        }
        assert_eq!(deltas(&rx), "Привет, мир");
        assert_eq!(state.content, "Привет, мир");
        assert_eq!(state.sent, state.content.len());
    }

    #[test]
    fn process_token_skips_eos_and_keeps_counter() {
        let eos = u32::from(b'!');
        let tok = ByteTokenizer::with_eos(eos);
        let (mut state, rx) = test_state_with_rx(64, eos);
        state.raw = true;
        state.session = None;
        let a = u32::from(b'a');
        assert_eq!(state.process_token(&tok, a), TokenFlow::Continue);
        assert_eq!(state.process_token(&tok, eos), TokenFlow::Continue);
        assert_eq!(state.tokens, vec![a], "EOS не должен попадать в счётчик");
        assert_eq!(deltas(&rx), "a");
        assert_eq!(state.commit_token(eos, TokenFlow::Continue), Commit::Eos);
        assert_eq!(state.finish, "stop");
    }

    #[test]
    fn push_text_holds_back_longest_stop_prefix() {
        let (mut state, rx) = test_state_with_rx(64, u32::MAX);
        state.stops = vec!["END".to_string()];
        state.hold = 3;
        assert_eq!(state.push_text("hello EN"), TokenFlow::Continue);
        assert_eq!(deltas(&rx), "hello");
        assert_eq!(state.push_text("D"), TokenFlow::Stop);
        assert_eq!(state.stop_at, Some(6));
        assert_eq!(deltas(&rx), "");
        assert_eq!(state.sent, 5);
    }

    #[test]
    fn stop_window_finds_stop_across_chunk_boundary() {
        let (mut state, _rx) = raw_state(vec!["<END>".to_string()]);
        assert_eq!(state.push_text("abc<EN"), TokenFlow::Continue);
        assert_eq!(state.push_text("D>tail"), TokenFlow::Stop);
        assert_eq!(state.stop_at, Some(3));
        assert_eq!(state.content, "abc<END>tail");
    }

    #[test]
    fn stop_window_matches_full_scan() {
        let stops = vec!["END".to_string(), "long-stop".to_string()];
        let prefix = "z".repeat(50);
        let scenarios: [&[&str]; 3] = [
            &["aaa", "EN", "D"],
            &[prefix.as_str(), "EN", "D"],
            &[prefix.as_str(), "long", "-st", "op"],
        ];
        for chunks in scenarios {
            let mut content = String::new();
            let mut stopped = false;
            for chunk in chunks {
                content.push_str(chunk);
                let expected = first_stop(&content, &stops);
                let start = stop_scan_start(content.len(), chunk.len(), 9);
                let mut start = start;
                while start > 0 && !content.is_char_boundary(start) {
                    start -= 1;
                }
                let scanned = first_stop(&content[start..], &stops).map(|pos| start + pos);
                assert_eq!(scanned, expected, "chunk {chunk}");
                if expected.is_some() {
                    stopped = true;
                    break;
                }
            }
            assert!(stopped, "scenario never stopped");
        }
    }

    #[test]
    fn stop_scan_start_keeps_margin() {
        assert_eq!(stop_scan_start(100, 4, 6), 100 - 4 - 6 - STOP_SCAN_MARGIN);
        assert_eq!(stop_scan_start(3, 10, 6), 0);
        assert_eq!(stop_scan_start(0, 0, 0), 0);
    }

    #[test]
    fn prompt_window_push_matches_full_push() {
        let params = SamplingParams {
            repeat_penalty: 2.0,
            repeat_penalty_window: 4,
            ..SamplingParams::default()
        };
        let prompt: Vec<u32> = (0..100).collect();
        let mut full = Sampler::new(params.clone());
        for &token in &prompt {
            full.push(token);
        }
        let mut windowed = Sampler::new(params.clone());
        push_prompt_window(&mut windowed, &prompt, params.repeat_penalty_window);
        let mut tail = Sampler::new(params.clone());
        for &token in &prompt[96..] {
            tail.push(token);
        }
        let logits: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let mut a = logits.clone();
        let mut b = logits.clone();
        let mut c = logits.clone();
        let token = full.sample(&mut a);
        assert_eq!(token, windowed.sample(&mut b));
        assert_eq!(token, tail.sample(&mut c));
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // --- Сквозные тесты жизненного цикла Worker на фейковом бэкенде ---

    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::Sender;

    const TEST_EOS: u32 = 0;
    const TEST_CTX: usize = 4096;
    const TEST_SLOTS: usize = 4;
    const TEST_TIMEOUT: Duration = Duration::from_secs(10);

    #[derive(Default)]
    struct FakeStats {
        /// Токены, которые «генерирует» бэкенд; после исчерпания отдаётся EOS.
        script: Vec<u32>,
        cursor: usize,
        active: BTreeSet<usize>,
        created: usize,
        freed: usize,
        step_calls: usize,
        cancel_after_steps: Option<usize>,
        cancel: Option<Arc<AtomicBool>>,
        fail_step_calls: usize,
        fail_prefill_calls: usize,
    }

    struct FakeBackend {
        stats: Arc<Mutex<FakeStats>>,
    }

    impl FakeBackend {
        fn lock(&self) -> std::sync::MutexGuard<'_, FakeStats> {
            self.stats.lock().expect("fake backend state")
        }

        fn next_logits(&self) -> Vec<f32> {
            let mut logits = vec![-10.0f32; 256];
            let token = {
                let mut stats = self.lock();
                let mut token = TEST_EOS;
                if let Some(&next) = stats.script.get(stats.cursor) {
                    token = next;
                    stats.cursor += 1;
                }
                token
            };
            let index = token as usize;
            assert!(index < logits.len(), "script token {token} out of logits");
            logits[index] = 10.0;
            logits
        }
    }

    impl InferenceBackend for FakeBackend {
        fn create_sequence(&mut self) -> Result<usize> {
            let mut stats = self.lock();
            let slot = (0..TEST_SLOTS)
                .find(|slot| !stats.active.contains(slot))
                .ok_or_else(|| anyhow!("no free slots"))?;
            stats.active.insert(slot);
            stats.created += 1;
            Ok(slot)
        }

        fn free_sequence(&mut self, slot: usize) -> Result<()> {
            let mut stats = self.lock();
            assert!(stats.active.remove(&slot), "double free of slot {slot}");
            stats.freed += 1;
            Ok(())
        }

        fn prefill(&mut self, _slot: usize, _tokens: &[u32], _need_logits: bool) -> Result<usize> {
            let mut stats = self.lock();
            if stats.fail_prefill_calls > 0 {
                stats.fail_prefill_calls -= 1;
                return Err(anyhow!("prefill exploded"));
            }
            Ok(0)
        }

        fn prefill_begin(&mut self, _slot: usize, _tokens: &[u32]) -> Result<usize> {
            Ok(0)
        }

        fn prefill_finish(&mut self, _slot: usize) -> Result<()> {
            Ok(())
        }

        fn step(&mut self, _steps: &[SequenceStep], _need_logits: bool) -> Result<()> {
            let mut stats = self.lock();
            stats.step_calls += 1;
            if stats.fail_step_calls > 0 {
                stats.fail_step_calls -= 1;
                return Err(anyhow!("step exploded"));
            }
            let trigger = stats
                .cancel_after_steps
                .is_some_and(|limit| stats.step_calls >= limit);
            let cancel = if trigger {
                stats.cancel_after_steps = None;
                stats.cancel.clone()
            } else {
                None
            };
            drop(stats);
            if let Some(flag) = cancel {
                flag.store(true, Ordering::Relaxed);
            }
            Ok(())
        }

        fn logits(&self) -> Result<Vec<f32>> {
            Ok(self.next_logits())
        }

        fn logits_row(&self, _row: usize) -> Result<Vec<f32>> {
            Ok(self.next_logits())
        }

        fn max_ctx(&self) -> usize {
            TEST_CTX
        }

        fn max_sequences(&self) -> usize {
            TEST_SLOTS
        }

        fn prefill_batch_size(&self) -> usize {
            8
        }

        fn free_pages(&self) -> usize {
            1024
        }

        fn cache_stats(&self) -> CacheStats {
            CacheStats::default()
        }
    }

    fn fake_backend(script: &str) -> (FakeBackend, Arc<Mutex<FakeStats>>) {
        let stats = Arc::new(Mutex::new(FakeStats {
            script: script.bytes().map(u32::from).collect(),
            ..FakeStats::default()
        }));
        (
            FakeBackend {
                stats: Arc::clone(&stats),
            },
            stats,
        )
    }

    fn greedy(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            max_tokens,
            ..SamplingParams::default()
        }
    }

    fn test_worker(backend: FakeBackend) -> Worker {
        Worker {
            gpu: Box::new(backend),
            tok: Box::new(ByteTokenizer::with_eos(TEST_EOS)),
            chat: ChatTemplate::from_str("chat", "{{ messages[0].content }}")
                .expect("test template"),
            system: String::new(),
            sched: Scheduler::new(SchedulerConfig::new(TEST_CTX, TEST_SLOTS, 8, 8)),
            reqs: HashMap::new(),
            metrics: Metrics::new(false),
            shared: Arc::new(EngineMetrics::new(TEST_CTX)),
        }
    }

    /// Гоняет настоящий `Worker::run` в отдельном потоке и возвращает воркер
    /// после остановки, чтобы проверить состояние планировщика и слотов.
    struct Harness {
        tx: Sender<Job>,
        join: std::thread::JoinHandle<Worker>,
    }

    impl Harness {
        fn start(backend: FakeBackend) -> Self {
            let (tx, rx) = channel::<Job>();
            let join = std::thread::spawn(move || {
                let mut worker = test_worker(backend);
                worker.run(rx);
                worker
            });
            Self { tx, join }
        }

        fn submit(&self, chat: ChatJob) -> Receiver<Event> {
            let (tx, rx) = channel();
            self.tx.send(Job { chat, tx }).expect("engine channel");
            rx
        }

        fn shutdown(self) -> Worker {
            drop(self.tx);
            self.join.join().expect("engine thread")
        }
    }

    fn drain_until_terminal(rx: &Receiver<Event>) -> Vec<Event> {
        let deadline = Instant::now() + TEST_TIMEOUT;
        let mut events = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(event) => {
                    let terminal = matches!(event, Event::Final { .. } | Event::Error(_));
                    events.push(event);
                    if terminal {
                        return events;
                    }
                }
                Err(e) => panic!("engine did not finish: {e}"),
            }
        }
    }

    fn streamed_text(events: &[Event]) -> String {
        let mut text = String::new();
        for event in events {
            if let Event::Delta(delta) = event {
                text.push_str(delta);
            }
        }
        text
    }

    fn expect_final(events: &[Event]) -> &Event {
        match events.last() {
            Some(event @ Event::Final { .. }) => event,
            Some(Event::Error(message)) => panic!("expected final event, got error: {message}"),
            _ => panic!("expected terminal final event"),
        }
    }

    fn expect_error(events: &[Event]) -> &str {
        match events.last() {
            Some(Event::Error(message)) => message,
            Some(event) => panic!("expected error event, got {event:?}"),
            None => panic!("expected error event, got none"),
        }
    }

    #[test]
    fn cancel_guard_sets_flag_on_drop() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _guard = CancelGuard::new(Arc::clone(&flag));
            assert!(!flag.load(Ordering::Relaxed));
        }
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn worker_success_path_streams_text_and_releases_slot() {
        let (backend, stats) = fake_backend("hi");
        let harness = Harness::start(backend);
        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("prompt".to_string(), greedy(8))),
        );

        assert_eq!(streamed_text(&events), "hi");
        let Event::Final {
            text,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            ..
        } = expect_final(&events)
        else {
            panic!("expected final event");
        };
        assert_eq!(text, "hi");
        assert_eq!(*finish_reason, "stop");
        assert_eq!(*prompt_tokens, "prompt".len());
        // EOS — служебный токен и в completion_tokens не входит: "h", "i".
        assert_eq!(*completion_tokens, 2);

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.created, 1);
        assert_eq!(stats.freed, 1);
        assert!(stats.active.is_empty());
    }

    #[test]
    fn cancelled_request_reports_error_and_frees_slot_for_next_request() {
        let (backend, stats) = fake_backend(&"x".repeat(64));
        let job = ChatJob::completion("prompt".to_string(), greedy(64));
        let flag = job.cancel_handle();
        {
            let mut state = stats.lock().unwrap();
            state.cancel_after_steps = Some(2);
            state.cancel = Some(Arc::clone(&flag));
        }
        let harness = Harness::start(backend);

        let events = drain_until_terminal(&harness.submit(job));
        assert_eq!(expect_error(&events), "cancelled by client");
        assert!(flag.load(Ordering::Relaxed));

        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("again".to_string(), greedy(1))),
        );
        let Event::Final { finish_reason, .. } = expect_final(&events) else {
            panic!("expected final event");
        };
        assert_eq!(*finish_reason, "length");

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.created, 2);
        assert_eq!(stats.freed, 2);
        assert!(stats.active.is_empty());
    }

    #[test]
    fn max_tokens_exhaustion_reports_length_and_releases_slot() {
        let (backend, stats) = fake_backend(&"x".repeat(32));
        let harness = Harness::start(backend);
        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("prompt".to_string(), greedy(5))),
        );

        assert_eq!(streamed_text(&events), "xxxxx");
        let Event::Final {
            text,
            finish_reason,
            completion_tokens,
            ..
        } = expect_final(&events)
        else {
            panic!("expected final event");
        };
        assert_eq!(text, "xxxxx");
        assert_eq!(*finish_reason, "length");
        assert_eq!(*completion_tokens, 5);

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.freed, 1);
        assert!(stats.active.is_empty());
    }

    #[test]
    fn backend_step_error_reports_error_and_worker_accepts_next_request() {
        let (backend, stats) = fake_backend(&"x".repeat(32));
        stats.lock().unwrap().fail_step_calls = 1;
        let harness = Harness::start(backend);

        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("prompt".to_string(), greedy(16))),
        );
        assert_eq!(expect_error(&events), "step exploded");

        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("again".to_string(), greedy(1))),
        );
        let Event::Final { finish_reason, .. } = expect_final(&events) else {
            panic!("expected final event");
        };
        assert_eq!(*finish_reason, "length");

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.created, 2);
        assert_eq!(stats.freed, 2);
        assert!(stats.active.is_empty());
    }

    #[test]
    fn backend_prefill_error_reports_error_and_worker_accepts_next_request() {
        let (backend, stats) = fake_backend(&"x".repeat(32));
        stats.lock().unwrap().fail_prefill_calls = 1;
        let harness = Harness::start(backend);

        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("prompt".to_string(), greedy(16))),
        );
        assert_eq!(expect_error(&events), "prefill exploded");

        let events = drain_until_terminal(
            &harness.submit(ChatJob::completion("again".to_string(), greedy(1))),
        );
        let Event::Final { finish_reason, .. } = expect_final(&events) else {
            panic!("expected final event");
        };
        assert_eq!(*finish_reason, "length");

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.created, 2);
        assert_eq!(stats.freed, 2);
        assert!(stats.active.is_empty());
    }

    #[test]
    fn tool_call_output_finishes_with_tool_calls_reason() {
        let script =
            "<tool_call><function=weather><parameter=city>Oslo</parameter></function></tool_call>";
        let (backend, stats) = fake_backend(script);
        let harness = Harness::start(backend);
        let chat = ChatJob::chat(vec![Message::user("ping")], greedy(128), false, Vec::new());
        let events = drain_until_terminal(&harness.submit(chat));

        let tool_events = events
            .iter()
            .filter(|event| matches!(event, Event::ToolCall(_)))
            .count();
        assert_eq!(tool_events, 1);
        let Event::Final {
            text,
            tool_calls,
            finish_reason,
            ..
        } = expect_final(&events)
        else {
            panic!("expected final event");
        };
        assert_eq!(text, "");
        assert_eq!(*finish_reason, "tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "weather");
        assert_eq!(tool_calls[0].arguments["city"], "Oslo");

        let worker = harness.shutdown();
        assert!(worker.sched.is_empty());
        assert!(worker.reqs.is_empty());
        let stats = stats.lock().unwrap();
        assert_eq!(stats.freed, 1);
        assert!(stats.active.is_empty());
    }
}
