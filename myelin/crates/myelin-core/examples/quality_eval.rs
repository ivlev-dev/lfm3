//! Quantization quality evaluation on the host CPU path.
//!
//! One process evaluates **one** weight format (the format is selected by the
//! `MYELIN_CPU_*` environment variables, which the backend reads once at load
//! time). It emits JSON Lines so a driver can compare formats:
//!
//! * one `prompt` record per prompt: greedy trajectory, decoded text and
//!   degeneration metrics (n-gram repetition, cyclic tail, token-run lengths,
//!   function-word runs, top-1 confidence/entropy);
//! * one `ppl` record per corpus text: teacher-forced cross-entropy (the
//!   trajectory-independent metric), next-token top-1/top-5 accuracy;
//! * one `summary` record with timings.
//!
//! ```text
//! cargo run --release -p myelin-core --features hf-tokenizer --example quality_eval -- \
//!   --prompts eval/prompts.jsonl --corpus eval/corpus.jsonl --max-tokens 40 \
//!   --format int4-per-row --out /tmp/perrow.jsonl
//! ```
//!
//! The harness never writes to stdout except the optional JSONL file; progress
//! and the weight statistics (`MYELIN_CPU_WSTATS=1`) go to stderr.

use std::collections::HashSet;
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use myelin_core::tokenizer::LfmTokenizer;
use myelin_core::Model;
use serde_json::{json, Value};

/// Function words (English + Russian) used by the "service token run" metric.
/// A degenerate int4 trajectory collapses into these tokens, so long runs of
/// them are a cheap symptom detector.
#[rustfmt::skip]
const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "of", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "to", "in", "on", "at", "for", "with", "by", "from", "as", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i", "his", "her", "their", "our", "your",
    "my", "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "there", "here", "what", "which", "who", "whom",
    "whose", "when", "where", "why", "how", "if", "then", "than", "so", "also", "very", "just",
    "about", "into", "over", "after", "before", "up", "down", "out", "off", "again", "more",
    "most", "some", "any", "all", "each", "every", "both", "one", "two", "three", "и", "в",
    "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его",
    "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну",
    "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут",
    "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот",
    "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту",
    "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
];

/// A prompt to run greedily.
#[derive(Debug, Clone)]
struct PromptCase {
    id: String,
    kind: String,
    lang: String,
    prompt: String,
    expect: Vec<String>,
}

/// A text the perplexity is measured on.
#[derive(Debug, Clone)]
struct TextCase {
    id: String,
    lang: String,
    text: String,
}

/// Per-step distribution statistics taken from the logits that produced a token.
#[derive(Debug, Clone, Copy)]
struct StepStats {
    top1: u32,
    top1_prob: f64,
    top2_prob: f64,
    /// Logit gap between top-1 and top-2, in nats.
    margin: f64,
    /// Shannon entropy of the full softmax, in nats.
    entropy: f64,
}

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    prompts: PathBuf,
    corpus: PathBuf,
    max_tokens: usize,
    format: String,
    seed: u64,
    out: Option<PathBuf>,
    run_generation: bool,
    run_perplexity: bool,
}

impl Default for Args {
    fn default() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        Self {
            model: root.join("models/LFM3-1.7B-Q4_K_M.gguf"),
            tokenizer: root.join("models/hf/tokenizer.json"),
            prompts: root.join("myelin/eval/prompts.jsonl"),
            corpus: root.join("myelin/eval/corpus.jsonl"),
            max_tokens: 40,
            format: "unnamed".to_owned(),
            seed: 0,
            out: None,
            run_generation: true,
            run_perplexity: true,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = PathBuf::from(value()?),
            "--tokenizer" => args.tokenizer = PathBuf::from(value()?),
            "--prompts" => args.prompts = PathBuf::from(value()?),
            "--corpus" => args.corpus = PathBuf::from(value()?),
            "--max-tokens" => {
                args.max_tokens = value()?
                    .parse()
                    .map_err(|_| "--max-tokens must be a positive integer".to_owned())?;
            }
            "--format" => args.format = value()?,
            "--seed" => {
                args.seed = value()?
                    .parse()
                    .map_err(|_| "--seed must be an integer".to_owned())?;
            }
            "--out" => args.out = Some(PathBuf::from(value()?)),
            "--no-gen" => args.run_generation = false,
            "--no-ppl" => args.run_perplexity = false,
            "--help" | "-h" => {
                println!(
                    "usage: quality_eval [--model PATH] [--tokenizer PATH] [--prompts PATH]\n\
                     \x20      [--corpus PATH] [--max-tokens N] [--format LABEL] [--seed S]\n\
                     \x20      [--out PATH] [--no-gen] [--no-ppl]\n\
                     \n\
                     \x20The weight format is selected with MYELIN_CPU_* env variables."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.max_tokens == 0 {
        return Err("--max-tokens must be positive".to_owned());
    }
    Ok(args)
}

fn read_jsonl(path: &PathBuf) -> Result<Vec<Value>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut records = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_no + 1))?;
        records.push(value);
    }
    Ok(records)
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("record without string field `{name}`"))
}

fn prompt_cases(path: &PathBuf) -> Result<Vec<PromptCase>, Box<dyn Error>> {
    let mut cases = Vec::new();
    for value in read_jsonl(path)? {
        let expect = value
            .get("expect")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        cases.push(PromptCase {
            id: field(&value, "id")?.to_owned(),
            kind: value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned(),
            lang: value
                .get("lang")
                .and_then(Value::as_str)
                .unwrap_or("en")
                .to_owned(),
            prompt: field(&value, "prompt")?.to_owned(),
            expect,
        });
    }
    Ok(cases)
}

fn text_cases(path: &PathBuf) -> Result<Vec<TextCase>, Box<dyn Error>> {
    let mut cases = Vec::new();
    for value in read_jsonl(path)? {
        cases.push(TextCase {
            id: field(&value, "id")?.to_owned(),
            lang: value
                .get("lang")
                .and_then(Value::as_str)
                .unwrap_or("en")
                .to_owned(),
            text: field(&value, "text")?.to_owned(),
        });
    }
    Ok(cases)
}

/// `log(sum(exp(logit)))`, computed in f64 to keep the tail stable.
fn logsumexp(logits: &[f32]) -> f64 {
    let max = logits
        .iter()
        .fold(f32::NEG_INFINITY, |best, value| best.max(*value));
    if !max.is_finite() {
        return f64::NAN;
    }
    let max = f64::from(max);
    let sum: f64 = logits
        .iter()
        .map(|value| (f64::from(*value) - max).exp())
        .sum();
    max + sum.ln()
}

/// Top-1/top-2, their probabilities, the logit margin and the full entropy.
fn step_stats(logits: &[f32]) -> StepStats {
    let mut first = 0usize;
    let mut first_value = f32::NEG_INFINITY;
    let mut second_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if *value > first_value {
            second_value = first_value;
            first_value = *value;
            first = index;
        } else if *value > second_value {
            second_value = *value;
        }
    }
    let lse = logsumexp(logits);
    let mut entropy = 0.0f64;
    for value in logits {
        let log_prob = f64::from(*value) - lse;
        if log_prob > -40.0 {
            let probability = log_prob.exp();
            entropy -= probability * log_prob;
        }
    }
    StepStats {
        top1: first as u32,
        top1_prob: (f64::from(first_value) - lse).exp(),
        top2_prob: (f64::from(second_value) - lse).exp(),
        margin: f64::from(first_value) - f64::from(second_value),
        entropy,
    }
}

/// Indices of the `k` largest logits, descending.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (index, value) in logits.iter().enumerate() {
        if best.len() == k && *value <= best[k - 1].1 {
            continue;
        }
        let position = best
            .iter()
            .position(|(_, candidate)| *value > *candidate)
            .unwrap_or(best.len());
        best.insert(position, (index, *value));
        best.truncate(k);
    }
    best.into_iter().map(|(index, _)| index).collect()
}

fn token_text(tokenizer: &LfmTokenizer, token: u32) -> String {
    tokenizer.decode(&[token], true).unwrap_or_default()
}

fn is_function_token(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed
        .chars()
        .all(|character| !character.is_alphanumeric())
    {
        return true;
    }
    let lower = trimmed.to_lowercase();
    FUNCTION_WORDS.contains(&lower.as_str())
}

/// Longest run of identical consecutive tokens.
fn max_token_run(tokens: &[u32]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    let mut previous = None;
    for token in tokens {
        if Some(*token) == previous {
            run += 1;
        } else {
            run = 1;
            previous = Some(*token);
        }
        best = best.max(run);
    }
    best
}

/// Fraction of repeated n-grams: `1 - distinct/total`.
fn repeat_ngram_fraction(tokens: &[u32], n: usize) -> f64 {
    if tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() - n + 1;
    let mut seen: HashSet<&[u32]> = HashSet::with_capacity(total);
    for window in tokens.windows(n) {
        seen.insert(window);
    }
    1.0 - seen.len() as f64 / total as f64
}

fn suffix_is_periodic(tokens: &[u32], period: usize, repeats: usize) -> bool {
    let start = tokens.len() - repeats * period;
    (1..repeats).all(|repeat| {
        (0..period).all(|offset| tokens[start + offset] == tokens[start + repeat * period + offset])
    })
}

/// Longest exact periodic tail: `(period, coverage)` where coverage is the
/// periodic suffix length divided by the sequence length. `(0, 0.0)` when no
/// period repeats at least twice.
fn cycle_metrics(tokens: &[u32]) -> (usize, f64) {
    let length = tokens.len() as f64;
    let mut best = (0usize, 0.0f64);
    for period in 1..=tokens.len() / 2 {
        if !suffix_is_periodic(tokens, period, 2) {
            continue;
        }
        let mut repeats = 2;
        while (repeats + 1) * period <= tokens.len()
            && suffix_is_periodic(tokens, period, repeats + 1)
        {
            repeats += 1;
        }
        let coverage = (repeats * period) as f64 / length;
        if coverage > best.1 {
            best = (period, coverage);
        }
    }
    best
}

fn longest_function_run(tokenizer: &LfmTokenizer, tokens: &[u32]) -> (usize, f64) {
    let mut best = 0usize;
    let mut run = 0usize;
    for token in tokens {
        if is_function_token(&token_text(tokenizer, *token)) {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    let fraction = if tokens.is_empty() {
        0.0
    } else {
        tokens
            .iter()
            .filter(|token| is_function_token(&token_text(tokenizer, **token)))
            .count() as f64
            / tokens.len() as f64
    };
    (best, fraction)
}

fn collect_logits_env() -> Value {
    let mut env = serde_json::Map::new();
    for name in [
        "MYELIN_CPU_Q4K",
        "MYELIN_CPU_INT4",
        "MYELIN_CPU_WBLOCK",
        "MYELIN_CPU_WASYM",
        "MYELIN_CPU_WHAD",
        "MYELIN_CPU_CONV16",
        "MYELIN_CPU_WHF32",
        "MYELIN_CPU_INT8_ACT",
        "MYELIN_CPU_ACT_GROUP",
        "MYELIN_CPU_KVK",
        "MYELIN_CPU_KVV",
    ] {
        if let Ok(value) = std::env::var(name) {
            env.insert(name.to_owned(), Value::String(value));
        }
    }
    Value::Object(env)
}

fn evaluate_prompt(
    model: &mut Model,
    tokenizer: &LfmTokenizer,
    case: &PromptCase,
    max_tokens: usize,
    eos: u32,
) -> Result<Value, Box<dyn Error>> {
    let prompt_ids = tokenizer.encode(&case.prompt, false)?;
    if prompt_ids.is_empty() {
        return Err(format!("prompt {} encoded to zero tokens", case.id).into());
    }
    let context = prompt_ids.len() + max_tokens + 8;
    let mut session = model.session(context)?;
    let mut logits = session.prefill(&prompt_ids)?;

    let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut steps: Vec<StepStats> = Vec::with_capacity(max_tokens);
    let mut stop = "length";
    while tokens.len() < max_tokens {
        let stats = step_stats(&logits);
        if stats.top1 == eos {
            stop = "eos";
            break;
        }
        tokens.push(stats.top1);
        steps.push(stats);
        if tokens.len() == max_tokens {
            break;
        }
        logits = session.decode(stats.top1)?;
    }

    let text = tokenizer.decode(&tokens, true)?;
    let lowered = text.to_lowercase();
    let hits: Vec<String> = case
        .expect
        .iter()
        .filter(|needle| lowered.contains(&needle.to_lowercase()))
        .cloned()
        .collect();
    let fact_hit = if case.expect.is_empty() {
        Value::Null
    } else {
        Value::Bool(hits.len() == case.expect.len())
    };

    let (cycle_period, cycle_coverage) = cycle_metrics(&tokens);
    let (max_function_run, function_fraction) = longest_function_run(tokenizer, &tokens);
    let mean = |values: &mut dyn Iterator<Item = f64>| -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in values {
            sum += value;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    };

    let top1_seen = steps
        .iter()
        .enumerate()
        .filter(|(index, stats)| tokens[..*index].contains(&stats.top1))
        .count() as f64
        / steps.len().max(1) as f64;
    let unique_tokens: HashSet<u32> = tokens.iter().copied().collect();
    let novel_token_fraction = unique_tokens.len() as f64 / tokens.len().max(1) as f64;
    let immediate_repeats = tokens.windows(2).filter(|pair| pair[0] == pair[1]).count();
    let immediate_repeat_fraction =
        immediate_repeats as f64 / tokens.len().saturating_sub(1).max(1) as f64;

    // One flag per prompt, documented in eval/README.md. The driver also
    // aggregates the raw metrics, so changing this heuristic only moves the
    // headline degradation rate.
    let degenerate = max_token_run(&tokens) >= 4
        || cycle_coverage >= 0.5
        || repeat_ngram_fraction(&tokens, 4) >= 0.35
        || max_function_run >= 6
        || function_fraction >= 0.65;

    Ok(json!({
        "kind": "prompt",
        "id": case.id,
        "type": case.kind,
        "lang": case.lang,
        "prompt": case.prompt,
        "prompt_tokens": prompt_ids.len(),
        "stop": stop,
        "text": text,
        "tokens": tokens,
        "metrics": {
            "n": tokens.len(),
            "max_token_run": max_token_run(&tokens),
            "repeat2": repeat_ngram_fraction(&tokens, 2),
            "repeat3": repeat_ngram_fraction(&tokens, 3),
            "repeat4": repeat_ngram_fraction(&tokens, 4),
            "cycle_period": cycle_period,
            "cycle_coverage": cycle_coverage,
            "novel_token_frac": novel_token_fraction,
            "immediate_repeat_frac": immediate_repeat_fraction,
            "max_function_run": max_function_run,
            "function_frac": function_fraction,
            "top1_seen_frac": top1_seen,
            "mean_top1_prob": mean(&mut steps.iter().map(|stats| stats.top1_prob)),
            "mean_top2_prob": mean(&mut steps.iter().map(|stats| stats.top2_prob)),
            "mean_margin": mean(&mut steps.iter().map(|stats| stats.margin)),
            "mean_entropy": mean(&mut steps.iter().map(|stats| stats.entropy)),
        },
        "expect": case.expect,
        "hits": hits,
        "fact_hit": fact_hit,
        "degenerate": degenerate,
    }))
}

fn evaluate_text(
    model: &mut Model,
    tokenizer: &LfmTokenizer,
    case: &TextCase,
) -> Result<Option<Value>, Box<dyn Error>> {
    let ids = tokenizer.encode(&case.text, false)?;
    if ids.len() < 2 {
        return Ok(None);
    }
    let context = ids.len() + 8;
    let mut session = model.session(context)?;
    let mut logits = session.prefill(&ids[..1])?;
    let mut nll_sum = 0.0f64;
    let mut top1_hits = 0usize;
    let mut top5_hits = 0usize;
    let mut counted = 0usize;
    for index in 1..ids.len() {
        let target = ids[index] as usize;
        if target >= logits.len() {
            return Err(format!(
                "target token {target} outside the vocabulary of {}",
                logits.len()
            )
            .into());
        }
        let lse = logsumexp(&logits);
        let nll = lse - f64::from(logits[target]);
        if !nll.is_finite() {
            return Err(format!("non-finite NLL at token {index} of {}", case.id).into());
        }
        nll_sum += nll;
        let best = top_k(&logits, 5);
        if best.first() == Some(&target) {
            top1_hits += 1;
        }
        if best.contains(&target) {
            top5_hits += 1;
        }
        counted += 1;
        if index + 1 < ids.len() {
            logits = session.decode(ids[index])?;
        }
    }
    let mean_nll = nll_sum / counted as f64;
    Ok(Some(json!({
        "kind": "ppl",
        "id": case.id,
        "lang": case.lang,
        "tokens": ids.len(),
        "predicted": counted,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": mean_nll.exp(),
        "bits_per_token": mean_nll / std::f64::consts::LN_2,
        "top1_acc": top1_hits as f64 / counted as f64,
        "top5_acc": top5_hits as f64 / counted as f64,
    })))
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = LfmTokenizer::from_file(&args.tokenizer)?;
    let eos = tokenizer.eos_id();
    let prompts = prompt_cases(&args.prompts)?;
    let corpus = text_cases(&args.corpus)?;

    let load_started = Instant::now();
    let mut model = Model::load_gguf(&args.model)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    eprintln!(
        "format {}: {} backend, {:?} spec, vocab {}, loaded in {:.1}s",
        args.format,
        model.backend_name(),
        model.spec().family,
        model.spec().dims.vocab,
        load_seconds
    );

    let mut records: Vec<Value> = Vec::new();
    records.push(json!({
        "kind": "meta",
        "format": args.format,
        "backend": model.backend_name(),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "model": args.model.display().to_string(),
        "env": collect_logits_env(),
        "load_seconds": load_seconds,
    }));

    let mut generated_tokens = 0usize;
    let mut generation_seconds = 0.0f64;
    if args.run_generation {
        for case in &prompts {
            let started = Instant::now();
            let record = evaluate_prompt(&mut model, &tokenizer, case, args.max_tokens, eos)?;
            let elapsed = started.elapsed().as_secs_f64();
            generation_seconds += elapsed;
            let count = record["metrics"]["n"].as_u64().unwrap_or(0) as usize;
            generated_tokens += count;
            eprintln!(
                "  [{}/{}] {:<22} {:>5.1}s {:>4} tok  {}",
                records.len(),
                prompts.len(),
                case.id,
                elapsed,
                count,
                record["text"]
                    .as_str()
                    .unwrap_or("")
                    .chars()
                    .take(70)
                    .collect::<String>()
                    .replace('\n', "\\n")
            );
            records.push(record);
        }
    }

    let mut ppl_tokens = 0usize;
    let mut ppl_seconds = 0.0f64;
    if args.run_perplexity {
        for case in &corpus {
            let started = Instant::now();
            match evaluate_text(&mut model, &tokenizer, case)? {
                Some(record) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    ppl_seconds += elapsed;
                    let count = record["predicted"].as_u64().unwrap_or(0) as usize;
                    ppl_tokens += count;
                    eprintln!(
                        "  ppl {:<16} {:>5.1}s {:>4} tok  ppl={:.3}",
                        case.id,
                        elapsed,
                        count,
                        record["ppl"].as_f64().unwrap_or(f64::NAN)
                    );
                    records.push(record);
                }
                None => eprintln!("  ppl {} skipped (too short)", case.id),
            }
        }
    }

    records.push(json!({
        "kind": "summary",
        "format": args.format,
        "prompts": if args.run_generation { prompts.len() } else { 0 },
        "generated_tokens": generated_tokens,
        "generation_seconds": generation_seconds,
        "ppl_tokens": ppl_tokens,
        "ppl_seconds": ppl_seconds,
        "tok_per_second": if generation_seconds > 0.0 {
            generated_tokens as f64 / generation_seconds
        } else { 0.0 },
    }));

    let body = records
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    match &args.out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, body + "\n")?;
            eprintln!("wrote {} records -> {}", records.len(), path.display());
        }
        None => println!("{body}"),
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_runs_are_counted() {
        assert_eq!(max_token_run(&[1, 1, 1, 2, 2, 3]), 3);
        assert_eq!(max_token_run(&[]), 0);
        assert_eq!(max_token_run(&[7]), 1);
    }

    #[test]
    fn repeated_ngrams_are_detected() {
        // 2-grams: (1,2) (2,1) (1,2) (2,1) (1,2) -> 5 windows, 2 distinct.
        let value = repeat_ngram_fraction(&[1, 2, 1, 2, 1, 2], 2);
        assert!((value - 0.6).abs() < 1e-9, "{value}");
        assert_eq!(repeat_ngram_fraction(&[1, 2, 3], 4), 0.0);
    }

    #[test]
    fn periodic_tail_is_detected() {
        let (period, coverage) = cycle_metrics(&[9, 1, 2, 3, 1, 2, 3]);
        assert_eq!(period, 3);
        assert!((coverage - 6.0 / 7.0).abs() < 1e-9);
        assert_eq!(cycle_metrics(&[1, 2, 3, 4, 5]), (0, 0.0));
    }

    #[test]
    fn function_tokens_classification() {
        assert!(is_function_token(" the"));
        assert!(is_function_token("."));
        assert!(is_function_token("\n"));
        assert!(is_function_token("и"));
        assert!(!is_function_token(" Paris"));
        assert!(!is_function_token("fibonacci"));
    }

    #[test]
    fn top_k_returns_indices_in_order() {
        assert_eq!(top_k(&[0.1, 5.0, 0.0, 2.0], 2), vec![1, 3]);
        assert_eq!(top_k(&[0.1, 0.2], 5), vec![1, 0]);
    }
}
