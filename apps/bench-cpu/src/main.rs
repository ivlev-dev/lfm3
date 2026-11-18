use anyhow::Result;
use clap::{Parser, Subcommand};
use lfm_app::InferenceBackend;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Prefill {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 512)]
        ctx: usize,
    },
    Gen {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value_t = 512)]
        ctx: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Prefill {
            model,
            tokens,
            ctx,
        } => {
            let started = Instant::now();
            let mut backend = lfm_backend_cpu::CpuBackend::load(&model, ctx)?;
            eprintln!("load {:.2}s", started.elapsed().as_secs_f64());
            let ids: Vec<u32> = (0..tokens)
                .map(|i| ((i * 2654435761usize) % 60000) as u32 + 100)
                .collect();
            let slot = backend.create_sequence()?;
            let started = Instant::now();
            let _ = backend.prefill(slot, &ids, false)?;
            let elapsed = started.elapsed().as_secs_f64();
            println!(
                "cpu prefill {tokens} tok in {elapsed:.3}s ({:.1} tok/s)",
                tokens as f64 / elapsed
            );
        }
        Cmd::Gen {
            model,
            tokenizer,
            prompt,
            tokens,
            ctx,
        } => {
            let started = Instant::now();
            let mut backend = lfm_backend_cpu::CpuBackend::load(&model, ctx)?;
            eprintln!("load {:.2}s", started.elapsed().as_secs_f64());
            let tok = lfm_tokenizer::LfmTokenizer::from_file(&tokenizer)?;
            let started = Instant::now();
            let out = lfm_app::Generate::new(&mut backend, &tok).run(&prompt, tokens)?;
            let elapsed = started.elapsed().as_secs_f64();
            println!("text: {}", out.text);
            println!(
                "{} tok in {elapsed:.3}s ({:.2} tok/s)",
                out.ids.len(),
                out.ids.len() as f64 / elapsed.max(1e-9)
            );
        }
    }
    Ok(())
}
