//! Parity of the myelin-core host path against the repository CPU reference.
//!
//! `lfm-cpu` (`Lfm3Ref`) is the older host implementation of the same LFM3
//! checkpoint: f32 dequantization, the same norm/rope/attention/FFN formulas.
//! This test loads it, records logits and greedy tokens, drops it (the f32
//! model is several gigabytes), then runs the myelin-core backend and compares.
//!
//! It is `#[ignore]`d because a full 1.7B checkpoint does not belong in the
//! unit-test loop:
//!
//! ```text
//! cargo test -p myelin-core --features hf-tokenizer --test parity -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use lfm_cpu::{Lfm3Ref, State};
use myelin_core::tokenizer::LfmTokenizer;
use myelin_core::{GenerateConfig, Model, SamplingParams, Tokenizer};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 4;

fn repository_file(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(relative)
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (left, right) in left.iter().zip(right) {
        dot += f64::from(*left) * f64::from(*right);
        left_norm += f64::from(*left) * f64::from(*left);
        right_norm += f64::from(*right) * f64::from(*right);
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k);
    indices
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best
}

#[test]
#[ignore = "loads the 1.7B checkpoint twice"]
fn greedy_generation_matches_cpu_reference() {
    let model_path = repository_file("models/LFM3-1.7B-Q4_K_M.gguf");
    let tokenizer_path = repository_file("models/hf/tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!(
            "skipping parity test: {} or {} is missing",
            model_path.display(),
            tokenizer_path.display()
        );
        return;
    }

    let tokenizer = LfmTokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|error| panic!("tokenizer: {error}"));
    let prompt = tokenizer
        .encode(PROMPT, false)
        .unwrap_or_else(|error| panic!("encode: {error}"));
    let eos = Tokenizer::eos_id(&tokenizer);
    println!("prompt {:?} -> {} tokens (eos {eos})", PROMPT, prompt.len());

    let reference = Lfm3Ref::load(Path::new(&model_path))
        .unwrap_or_else(|error| panic!("reference load: {error}"));
    let mut state = State::new(&reference.cfg);
    let mut reference_logits = Vec::new();
    for (position, token) in prompt.iter().enumerate() {
        reference_logits = reference
            .forward(&mut state, *token, position)
            .unwrap_or_else(|error| panic!("reference forward: {error}"));
    }
    let reference_tokens = reference
        .generate_greedy(&prompt, NEW_TOKENS, eos)
        .unwrap_or_else(|error| panic!("reference generation: {error}"));
    drop(state);
    drop(reference);

    let mut model =
        Model::load_gguf(&model_path).unwrap_or_else(|error| panic!("model load: {error}"));
    println!(
        "myelin backend {} on {:?}",
        model.backend_name(),
        model.spec().family
    );

    let mut session = model
        .session(prompt.len() + NEW_TOKENS + 8)
        .unwrap_or_else(|error| panic!("session: {error}"));
    let logits = session
        .prefill(&prompt)
        .unwrap_or_else(|error| panic!("prefill: {error}"));

    let similarity = cosine(&reference_logits, &logits);
    let max_abs_diff = reference_logits
        .iter()
        .zip(&logits)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let reference_argmax = argmax(&reference_logits);
    let myelin_argmax = argmax(&logits);
    let reference_top = top_k(&reference_logits, 5);
    let myelin_top = top_k(&logits, 5);
    let overlap = reference_top
        .iter()
        .filter(|index| myelin_top.contains(index))
        .count();
    println!(
        "last-prompt logits: vocab {}, cosine {similarity:.9}, max|diff| {max_abs_diff:.7}, \
         argmax {reference_argmax} vs {myelin_argmax}, top-5 overlap {overlap}/5",
        logits.len()
    );

    // Stepwise decode of the same tokens must reproduce the batched prefill
    // logits bit for bit: the graph and the caches are state-free otherwise.
    let mut stepwise = model
        .session(prompt.len() + NEW_TOKENS + 8)
        .unwrap_or_else(|error| panic!("session: {error}"));
    let mut stepped = Vec::new();
    for token in &prompt {
        stepped = stepwise
            .decode(*token)
            .unwrap_or_else(|error| panic!("decode: {error}"));
    }
    assert_eq!(
        stepped, logits,
        "stepwise decode diverged from batched prefill"
    );

    let config = GenerateConfig {
        max_tokens: NEW_TOKENS,
        sampling: SamplingParams::greedy(),
        eos: Some(eos),
        stop: Vec::new(),
    };
    let mut generation_session = model
        .session(prompt.len() + NEW_TOKENS + 8)
        .unwrap_or_else(|error| panic!("session: {error}"));
    let generation = generation_session
        .generate(&prompt, &config, |_| {})
        .unwrap_or_else(|error| panic!("generate: {error}"));
    let reference_text = tokenizer
        .decode(&reference_tokens, true)
        .unwrap_or_else(|error| panic!("reference decode: {error}"));
    let myelin_text = tokenizer
        .decode(&generation.tokens, true)
        .unwrap_or_else(|error| panic!("decode: {error}"));
    println!("reference tokens {reference_tokens:?} -> {reference_text:?}");
    println!(
        "myelin    tokens {:?} -> {myelin_text:?}",
        generation.tokens
    );

    assert!(
        similarity > 0.999_9,
        "logits cosine {similarity} is below 0.9999"
    );
    assert_eq!(
        reference_argmax, myelin_argmax,
        "argmax of the last prompt logits differs"
    );
    assert!(overlap >= 4, "top-5 overlap is {overlap}/5");
    assert_eq!(
        reference_tokens, generation.tokens,
        "greedy token stream differs from the CPU reference"
    );
}
