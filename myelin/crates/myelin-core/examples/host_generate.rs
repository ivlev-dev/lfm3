//! Host end-to-end run: GGUF -> tokenizer -> generation -> text.
//!
//! ```text
//! cargo run -p myelin-core --features hf-tokenizer --example host_generate -- \
//!   --prompt "The capital of France is" --max-tokens 16
//! ```
//!
//! Defaults point at `models/` in the repository root. Everything is printed to
//! stderr except the generated text, which goes to stdout.

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use myelin_core::tokenizer::LfmTokenizer;
use myelin_core::{GenerateConfig, Model, SamplingParams};

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_tokens: usize,
    context: usize,
    sampling: SamplingParams,
}

impl Default for Args {
    fn default() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        Self {
            model: root.join("models/LFM3-1.7B-Q4_K_M.gguf"),
            tokenizer: root.join("models/hf/tokenizer.json"),
            prompt: "The capital of France is".to_owned(),
            max_tokens: 16,
            context: 256,
            sampling: SamplingParams::greedy(),
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
            "--prompt" => args.prompt = value()?,
            "--max-tokens" => {
                args.max_tokens = value()?
                    .parse()
                    .map_err(|_| "--max-tokens must be a positive integer".to_owned())?;
            }
            "--context" => {
                args.context = value()?
                    .parse()
                    .map_err(|_| "--context must be a positive integer".to_owned())?;
            }
            "--temperature" => {
                args.sampling.temperature = value()?
                    .parse()
                    .map_err(|_| "--temperature must be a number".to_owned())?;
            }
            "--top-p" => {
                args.sampling.top_p = value()?
                    .parse()
                    .map_err(|_| "--top-p must be a number".to_owned())?;
            }
            "--top-k" => {
                args.sampling.top_k = value()?
                    .parse()
                    .map_err(|_| "--top-k must be an integer".to_owned())?;
            }
            "--seed" => {
                args.sampling.seed = value()?
                    .parse()
                    .map_err(|_| "--seed must be an integer".to_owned())?;
            }
            "--help" | "-h" => {
                println!(
                    "usage: host_generate [--model PATH] [--tokenizer PATH] [--prompt TEXT]\n\
                     \x20      [--max-tokens N] [--context N] [--temperature T] [--top-p P]\n\
                     \x20      [--top-k K] [--seed S]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.max_tokens == 0 {
        return Err("--max-tokens must be positive".to_owned());
    }
    if args.context < args.max_tokens {
        return Err("--context must be at least --max-tokens".to_owned());
    }
    Ok(args)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = LfmTokenizer::from_file(&args.tokenizer)?;
    let prompt = tokenizer.encode(&args.prompt, false)?;
    eprintln!("prompt: {} tokens from {:?}", prompt.len(), args.prompt);

    let load_started = Instant::now();
    let mut model = Model::load_gguf(&args.model)?;
    eprintln!(
        "model: {} backend, {:?} spec, loaded in {:.2}s",
        model.backend_name(),
        model.spec().family,
        load_started.elapsed().as_secs_f64()
    );

    let mut session = model.session(args.context)?;
    let config = GenerateConfig {
        max_tokens: args.max_tokens,
        sampling: args.sampling,
        eos: Some(tokenizer.eos_id()),
        stop: Vec::new(),
    };
    let started = Instant::now();
    let generation = session.generate(&prompt, &config, |_| {})?;
    let elapsed = started.elapsed().as_secs_f64();
    let text = tokenizer.decode(&generation.tokens, true)?;
    println!("{text}");
    eprintln!(
        "generated {} tokens in {:.2}s ({:.2} tok/s), stop {:?}, position {}",
        generation.tokens.len(),
        elapsed,
        generation.tokens.len() as f64 / elapsed.max(1e-9),
        generation.stop_reason,
        session.position()
    );
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
