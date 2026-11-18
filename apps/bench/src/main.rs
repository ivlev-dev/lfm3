use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use lfm_app::InferenceBackend;
use lfm_opencl::{Arg, Buffer, ClMem, Event, Gpu, Kernel, Program, Stats};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const BENCH_CL: &str = include_str!("../kernels/bench.cl");
const GEMV_CL: &str = include_str!("../kernels/gemv.cl");
const LFM_KERNELS: &str = concat!(
    include_str!("../../../crates/backend-opencl/kernels/common.cl"),
    "\n",
    include_str!("../../../crates/backend-opencl/kernels/decode.cl"),
    "\n",
    include_str!("../../../crates/backend-opencl/kernels/batch.cl"),
    "\n",
    include_str!("../../../crates/backend-opencl/kernels/kvq4.cl")
);
const GROUP: usize = 32;

#[derive(Parser)]
struct Cli {
    #[arg(long, global = true, default_value = "f16")]
    kv_cache: String,
    #[arg(long, global = true, default_value_t = true)]
    kv_nc: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Info,
    Bw {
        #[arg(long, default_value_t = 512)]
        mb: usize,
        #[arg(long, default_value_t = 10)]
        reps: usize,
    },
    Laten {
        #[arg(long, default_value_t = 2000)]
        iters: usize,
        #[arg(long, default_value_t = 262144)]
        grid: usize,
    },
    Peak {
        #[arg(long, default_value_t = 8192)]
        iters: u32,
    },
    Mixed {
        #[arg(long, default_value_t = 8192)]
        iters: u32,
    },
    Gemv {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "blk.2.attn_q.weight")]
        tensor: String,
        #[arg(long, default_value_t = 4)]
        check_rows: usize,
        #[arg(long, default_value_t = 4)]
        rgemv: u32,
        #[arg(long, default_value = "i8")]
        mode: String,
        #[arg(long)]
        onehot: Option<usize>,
    },
    Sweep {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 4)]
        rgemv: u32,
    },
    Dump {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "blk.2.attn_q.weight")]
        tensor: String,
    },
    RefGen {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        tokens: usize,
    },
    RefPpl {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The quick brown fox jumps over the lazy dog. In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.")]
        text: String,
    },
    GpuGen {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temp: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    GpuChat {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "models/hf/chat_template.jinja")]
        template: PathBuf,
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,
        #[arg(long, default_value = "Say hello in one short sentence.")]
        user: String,
        #[arg(long, default_value_t = 256)]
        tokens: usize,
        #[arg(long)]
        thinking: bool,
        #[arg(long, default_value_t = 0.3)]
        temp: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 1.05)]
        repeat_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        interactive: bool,
        #[arg(long)]
        tui: bool,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },
    GpuEmbed {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long)]
        id: u32,
    },
    GpuPort {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
    },
    SgProbe,
    GpuPp {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 4)]
        tokens: usize,
        #[arg(long, default_value = "12/12")]
        tiers: String,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
    },
    GpuTp {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "models/hf/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,
        #[arg(long, default_value_t = 2)]
        tokens: usize,
        #[arg(long, default_value = "0/1")]
        devices: String,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
    },
    GpuCheck {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
    },
    KernelBin {
        #[arg(long, default_value = "gemm_q4_k_i8_lds11")]
        kernel: String,
        #[arg(long, default_value = "/tmp/opencode/kernel.bin")]
        out: PathBuf,
    },
    DotRate {
        #[arg(long, default_value_t = 200000)]
        iters: u32,
    },
    GemmCmp {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "blk.2.ffn_up.weight")]
        tensor: String,
        #[arg(long, default_value_t = 8)]
        ntok: usize,
        #[arg(long, default_value = "lds17")]
        mode: String,
        #[arg(long, default_value_t = 4)]
        check_rows: usize,
    },
    Gemb {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value = "blk.2.attn_q.weight")]
        tensor: String,
        #[arg(long, default_value_t = 2)]
        pb: u32,
        #[arg(long, default_value_t = 4)]
        rgemv: u32,
        #[arg(long, default_value_t = 256)]
        threads: u32,
    },
    GpuMulti {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 3)]
        seqs: usize,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value_t = 6)]
        prompt: usize,
    },
    GpuMix {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 12)]
        tokens: usize,
    },
    GpuCache {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        #[arg(long)]
        reset: bool,
    },
    KernInfo {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
    },
    GpuPrefill {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        #[arg(long, default_value_t = 32)]
        decode: usize,
    },
    GpuTrace {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
    },
    GpuCmp {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        len: usize,
    },
    RefIds {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long, default_value_t = 16)]
        tokens: usize,
        #[arg(long, default_value_t = 124900)]
        eos: u32,
    },
    RefDebug {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Step {
        #[arg(long, default_value = "models/LFM3-1.7B-Q4_K_M.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 4)]
        rgemv: u32,
        #[arg(long, default_value = "i8")]
        mode: String,
        #[arg(long, default_value_t = 10)]
        reps: usize,
    },
}

struct Kernels {
    program: Program,
    q4: Kernel,
    q6: Kernel,
    quant: Kernel,
}

fn fmt_gbps(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / (ms / 1e3) / 1e9
}

fn describe(st: &Stats) -> String {
    format!(
        "median {:.3} ms, min {:.3} ms, mean {:.3} ms",
        st.median_ms, st.min_ms, st.mean_ms
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let kv = match cli.kv_cache.as_str() {
        "f16" => "f16",
        "k4v4" | "q4" => "q4",
        "kvarn_k4v2" | "kvarn" => "kvarn",
        "k4v3" => return Err(anyhow!("--kv-cache k4v3 is not implemented yet")),
        other => {
            return Err(anyhow!(
                "unknown --kv-cache {other}, expected f16|k4v4|k4v3|kvarn_k4v2"
            ))
        }
    };
    unsafe {
        std::env::set_var("LFM_KV", kv);
        std::env::set_var("LFM_KV_NC", if cli.kv_nc { "1" } else { "0" });
    }
    match cli.cmd {
        Cmd::Info => info(),
        Cmd::Bw { mb, reps } => bw(mb, reps),
        Cmd::Laten { iters, grid } => laten(iters, grid),
        Cmd::Peak { iters } => peak(iters),
        Cmd::Mixed { iters } => mixed(iters),
        Cmd::Gemv {
            model,
            tensor,
            check_rows,
            rgemv,
            mode,
            onehot,
        } => gemv(&model, &tensor, check_rows, rgemv, &mode, onehot),
        Cmd::Sweep { model, rgemv } => sweep(&model, rgemv),
        Cmd::Dump { model, tensor } => dump(&model, &tensor),
        Cmd::RefDebug { model, ids, out } => reference_debug(&model, &ids, out.as_ref()),
        Cmd::GpuGen {
            model,
            tokenizer,
            prompt,
            tokens,
            temp,
            top_k,
            top_p,
            repeat_penalty,
            seed,
        } => gpu_generate(
            &model,
            &tokenizer,
            &prompt,
            tokens,
            &sampling(temp, top_k, top_p, repeat_penalty, seed),
        ),
        Cmd::GpuChat {
            model,
            tokenizer,
            template,
            system,
            user,
            tokens,
            thinking,
            temp,
            top_k,
            top_p,
            repeat_penalty,
            seed,
            interactive,
            tui,
            ctx,
        } => gpu_chat(
            &model,
            &tokenizer,
            &template,
            &system,
            &user,
            tokens,
            thinking,
            &sampling(temp, top_k, top_p, repeat_penalty, seed),
            interactive,
            tui,
            ctx,
        ),
        Cmd::GpuCheck { model, ids } => gpu_check(&model, &ids),
        Cmd::GpuCmp {
            model,
            ids,
            batch,
            len,
        } => {
            let ids = if len > 0 {
                (0..len).map(|i| ((i * 2654435761usize) % 60000) as u32 + 100).collect()
            } else {
                ids
            };
            gpu_cmp(&model, &ids, batch)
        }
        Cmd::GpuTrace { model, ids } => gpu_trace(&model, &ids),
        Cmd::KernelBin { kernel, out } => kernel_bin(&kernel, &out),
        Cmd::DotRate { iters } => dot_rate(iters),
        Cmd::GemmCmp {
            model,
            tensor,
            ntok,
            mode,
            check_rows,
        } => gemm_cmp(&model, &tensor, ntok, &mode, check_rows),
        Cmd::Gemb {
            model,
            tensor,
            pb,
            rgemv,
            threads,
        } => gemb(&model, &tensor, pb, rgemv, threads),
        Cmd::GpuPrefill {
            model,
            tokens,
            decode,
        } => gpu_prefill(&model, tokens, decode),
        Cmd::GpuMulti {
            model,
            seqs,
            tokens,
            prompt,
        } => gpu_multi(&model, seqs, tokens, prompt),
        Cmd::GpuMix { model, tokens } => gpu_mix(&model, tokens),
        Cmd::GpuCache {
            model,
            tokens,
            reps,
            reset,
        } => gpu_cache(&model, tokens, reps, reset),
        Cmd::KernInfo { model } => {
            let gpu = lfm_gpu::GpuModel::load(&model, 1024)?;
            for (name, bytes) in gpu.kernel_stats() {
                println!("{name:<20} private/thread {bytes} bytes");
            }
            Ok(())
        }
        Cmd::GpuEmbed { model, id } => gpu_embed(&model, id),
        Cmd::GpuPort {
            model,
            tokenizer,
            prompt,
            tokens,
            ctx,
        } => gpu_port(&model, &tokenizer, &prompt, tokens, ctx),
        Cmd::SgProbe => sg_probe(),
        Cmd::GpuPp {
            model,
            tokenizer,
            prompt,
            tokens,
            tiers,
            ctx,
        } => gpu_pp(&model, &tokenizer, &prompt, tokens, &tiers, ctx),
        Cmd::GpuTp {
            model,
            tokenizer,
            prompt,
            tokens,
            devices,
            ctx,
        } => gpu_tp(&model, &tokenizer, &prompt, tokens, &devices, ctx),
        Cmd::RefPpl {
            model,
            tokenizer,
            text,
        } => reference_ppl(&model, &tokenizer, &text),
        Cmd::RefGen {
            model,
            tokenizer,
            prompt,
            tokens,
        } => reference_generate(&model, &tokenizer, &prompt, tokens),
        Cmd::RefIds {
            model,
            ids,
            tokens,
            eos,
        } => reference_ids(&model, &ids, tokens, eos),
        Cmd::Step {
            model,
            rgemv,
            mode,
            reps,
        } => step(&model, rgemv, &mode, reps),
    }
}

fn info() -> Result<()> {
    let gpu = Gpu::open(0)?;
    println!("name: {}", gpu.caps.name);
    println!("version: {}", gpu.caps.version);
    println!("driver: {}", gpu.caps.driver);
    println!("opencl_c: {}", gpu.caps.opencl_c);
    println!("compute_units: {}", gpu.caps.compute_units);
    println!("clock_mhz: {}", gpu.caps.clock_mhz);
    println!(
        "global_mem_gib: {:.2}",
        gpu.caps.global_mem_bytes as f64 / (1u64 << 30) as f64
    );
    println!("max_work_group_size: {}", gpu.caps.max_work_group_size);
    println!(
        "preferred_wg_multiple: {}",
        gpu.caps.preferred_wg_multiple
    );
    println!("sub_group_size: {}", gpu.caps.sub_group_size);
    println!("fp16: {}", gpu.caps.fp16);
    println!("integer_dot: {}", gpu.caps.integer_dot);
    println!("subgroups: {}", gpu.caps.subgroups);
    Ok(())
}

fn bw(mb: usize, reps: usize) -> Result<()> {
    let gpu = Gpu::open(0)?;
    let program = gpu.build(BENCH_CL, "-cl-std=CL3.0")?;
    let n_bytes = mb * 1024 * 1024;
    let n4 = n_bytes / 16;
    let zeros = vec![0u8; n_bytes];
    let src = gpu.upload_bytes(&zeros)?;
    drop(zeros);
    let sink = gpu.alloc::<f32>(64)?;
    let read = Kernel::create(&program, "bw_read")?;
    let global = 1usize << 20;
    let st = gpu.bench(2, reps, || -> Result<()> {
        unsafe {
            gpu.launch(
                &read,
                global,
                256,
                &[
                    Arg::Mem(src.get()),
                    Arg::Mem(sink.get()),
                    Arg::U32(n4 as u32),
                ],
            )?;
        }
        Ok(())
    })?;
    println!(
        "bw_read {} MiB: {} -> {:.1} GB/s",
        mb,
        describe(&st),
        fmt_gbps(n_bytes as u64, st.median_ms)
    );
    let dst = gpu.alloc::<u8>(n_bytes)?;
    let copy = Kernel::create(&program, "bw_copy")?;
    let st = gpu.bench(2, reps, || -> Result<()> {
        unsafe {
            gpu.launch(
                &copy,
                n4,
                256,
                &[Arg::Mem(src.get()), Arg::Mem(dst.get()), Arg::U32(n4 as u32)],
            )?;
        }
        Ok(())
    })?;
    println!(
        "bw_copy {} MiB: {} -> {:.1} GB/s (r+w)",
        mb,
        describe(&st),
        fmt_gbps(2 * n_bytes as u64, st.median_ms)
    );
    Ok(())
}

fn laten(iters: usize, grid: usize) -> Result<()> {
    for profiling in [true, false] {
        let gpu = Gpu::open_with(0, profiling)?;
        let program = gpu.build(BENCH_CL, "-cl-std=CL3.0")?;
        let sink = gpu.alloc::<u32>(grid)?;
        let nop = Kernel::create(&program, "nop")?;
        for _ in 0..50 {
            unsafe { gpu.launch(&nop, grid, 256, &[Arg::Mem(sink.get())]) }?.wait()?;
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { gpu.launch(&nop, grid, 256, &[Arg::Mem(sink.get())]) }?;
        }
        let enqueue_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t1 = Instant::now();
        gpu.queue.finish()?;
        let drain_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "grid {:>7} profiling={}: enqueue {:.2} us/kernel, drain {:.2} us/kernel, total {:.2} us/kernel",
            grid,
            profiling,
            enqueue_us,
            drain_us,
            enqueue_us + drain_us
        );
    }
    Ok(())
}

fn peak(iters: u32) -> Result<()> {
    let gpu = Gpu::open(0)?;
    let program = gpu.build(BENCH_CL, "-cl-std=CL3.0")?;
    let global = gpu.caps.compute_units as usize * 64 * 32;
    let sinkf = gpu.alloc::<f32>(global)?;
    let sinku = gpu.alloc::<u32>(global)?;
    let fp32 = Kernel::create(&program, "peak_fp32")?;
    let ops = iters as f64 * 8.0 * 2.0 * global as f64;
    let st = gpu.bench(3, 10, || -> Result<()> {
        unsafe {
            gpu.launch(&fp32, global, 256, &[Arg::Mem(sinkf.get()), Arg::U32(iters)])?;
        }
        Ok(())
    })?;
    println!(
        "fp32 fma: {} -> {:.2} TFLOPS",
        describe(&st),
        ops / (st.median_ms / 1e3) / 1e12
    );
    let fp16 = Kernel::create(&program, "peak_fp16")?;
    let st = gpu.bench(3, 10, || -> Result<()> {
        unsafe {
            gpu.launch(&fp16, global, 256, &[Arg::Mem(sinkf.get()), Arg::U32(iters)])?;
        }
        Ok(())
    })?;
    println!(
        "fp16x2 fma: {} -> {:.2} TFLOPS",
        describe(&st),
        ops * 2.0 / (st.median_ms / 1e3) / 1e12
    );
    let i8 = Kernel::create(&program, "peak_i8dot")?;
    let iops = iters as f64 * 8.0 * 4.0 * 2.0 * global as f64;
    let st = gpu.bench(3, 10, || -> Result<()> {
        unsafe {
            gpu.launch(&i8, global, 256, &[Arg::Mem(sinku.get()), Arg::U32(iters)])?;
        }
        Ok(())
    })?;
    println!(
        "int8 dot4: {} -> {:.2} TOPS",
        describe(&st),
        iops / (st.median_ms / 1e3) / 1e12
    );
    Ok(())
}

fn make_x(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.25 + ((i * 37) % 101) as f32 / 101.0)
        .collect()
}

fn quantize_ref(x: &[f32], group: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; x.len()];
    let mut s = vec![0f32; x.len() / group];
    for (g, chunk) in x.chunks_exact(group).enumerate() {
        let amax = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
        let sc = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        s[g] = sc;
        for (i, v) in chunk.iter().enumerate() {
            q[g * group + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

fn dot_f32(ty: lfm_gguf::GgmlType, bytes: &[u8], x: &[f32]) -> Result<f32> {
    match ty {
        lfm_gguf::GgmlType::Q4_K => Ok(lfm_quant::dot_row_q4_k(bytes, x)?),
        lfm_gguf::GgmlType::Q6_K => Ok(lfm_quant::dot_row_q6_k(bytes, x)?),
        other => Err(anyhow!("unsupported type {other:?}")),
    }
}

fn dot_i8(ty: lfm_gguf::GgmlType, bytes: &[u8], xq: &[i8], xs: &[f32]) -> Result<f32> {
    let w = match ty {
        lfm_gguf::GgmlType::Q4_K => lfm_quant::dequant_row_q4_k(bytes, xq.len())?,
        lfm_gguf::GgmlType::Q6_K => lfm_quant::dequant_row_q6_k(bytes, xq.len())?,
        other => return Err(anyhow!("unsupported type {other:?}")),
    };
    let mut acc = 0.0f32;
    for i in 0..xq.len() {
        acc += w[i] * (xq[i] as f32) * xs[i / GROUP];
    }
    Ok(acc)
}

fn build_kernels(gpu: &Gpu, rgemv: u32) -> Result<Kernels> {
    let opts = format!("-cl-std=CL3.0 -DRGEMV={rgemv} -DPB=2");
    let program = gpu.build(GEMV_CL, &opts)?;
    Ok(Kernels {
        q4: Kernel::create(&program, "gemv_q4_k_i8")?,
        q6: Kernel::create(&program, "gemv_q6_k_i8")?,
        quant: Kernel::create(&program, "quantize_q8")?,
        program,
    })
}

fn quantize_on_gpu(
    gpu: &Gpu,
    kernel: &Kernel,
    x: &Buffer<f32>,
    xq: &Buffer<i8>,
    xs: &Buffer<f32>,
    n: usize,
) -> Result<Event> {
    unsafe {
        gpu.launch(
            kernel,
            n / GROUP,
            0,
            &[
                Arg::Mem(x.get()),
                Arg::Mem(xq.get()),
                Arg::Mem(xs.get()),
                Arg::U32(n as u32),
                Arg::U32(GROUP as u32),
            ],
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    gpu: &Gpu,
    kernel: &Kernel,
    w: &Buffer<u8>,
    xq: &Buffer<i8>,
    xs: &Buffer<f32>,
    y: &Buffer<f32>,
    rows: usize,
    cols: usize,
    rgemv: usize,
) -> Result<Event> {
    let eff = (rows / (4 * 96)).max(1).min(rgemv);
    let per_wg = (4 * eff).max(1);
    let groups = rows.div_ceil(per_wg);
    unsafe {
        gpu.launch(
            kernel,
            groups * 256,
            256,
            &[
                Arg::Mem(w.get()),
                Arg::Mem(xq.get()),
                Arg::Mem(xs.get()),
                Arg::Mem(y.get()),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
            ],
        )
    }
}

fn gemv(
    path: &PathBuf,
    tensor: &str,
    check_rows: usize,
    rgemv: u32,
    mode: &str,
    onehot: Option<usize>,
) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(path)?;
    let info = gguf
        .tensor(tensor)
        .ok_or_else(|| anyhow!("tensor {tensor} not found"))?
        .clone();
    let cols = info.dims[0] as usize;
    let rows = info.dims[1] as usize;
    let bytes = gguf.tensor_bytes(tensor)?;
    let gpu = Gpu::open(0)?;
    let k = build_kernels(&gpu, rgemv)?;
    let x = if let Some(j) = onehot {
        let mut v = vec![0.0f32; cols];
        v[j] = 1.0;
        v
    } else {
        make_x(cols)
    };
    let w = gpu.upload_bytes(bytes)?;
    let xb = gpu.upload(&x)?;
    let (xq, xs) = if onehot.is_some() {
        let (xq_ref, xs_ref) = quantize_ref(&x, GROUP);
        (gpu.upload(&xq_ref)?, gpu.upload(&xs_ref)?)
    } else {
        let xq = gpu.alloc::<i8>(cols)?;
        let xs = gpu.alloc::<f32>(cols / GROUP)?;
        quantize_on_gpu(&gpu, &k.quant, &xb, &xq, &xs, cols)?.wait()?;
        (xq, xs)
    };
    let yb = gpu.alloc::<f32>(rows)?;
    let kernel = match info.ggml_type {
        lfm_gguf::GgmlType::Q4_K => &k.q4,
        lfm_gguf::GgmlType::Q6_K => &k.q6,
        other => return Err(anyhow!("unsupported type {other:?}")),
    };
    if mode == "sum" {
        let sumk = Kernel::create(&k.program, "gemv_q4_k_sum")?;
        let rpw = 256 / gpu.caps.sub_group_size.max(1);
        let groups = rows.div_ceil((rpw * rgemv as usize).max(1));
        let st = gpu.bench(2, 10, || -> Result<()> {
            unsafe {
                gpu.launch(
                    &sumk,
                    groups * 256,
                    256,
                    &[
                        Arg::Mem(w.get()),
                        Arg::Mem(yb.get()),
                        Arg::U32(rows as u32),
                        Arg::U32(cols as u32),
                    ],
                )?;
            }
            Ok(())
        })?;
        println!(
            "{} sum-only {}x{} rgemv={}: {} -> {:.1} GB/s",
            tensor,
            rows,
            cols,
            rgemv,
            describe(&st),
            fmt_gbps(bytes.len() as u64, st.median_ms)
        );
        return Ok(());
    }
    let st = gpu.bench(2, 10, || -> Result<()> {
        launch_gemv(&gpu, kernel, &w, &xq, &xs, &yb, rows, cols, rgemv as usize)?;
        Ok(())
    })?;
    let out = gpu.download(&yb, rows)?;
    println!(
        "{} {:?} {}x{} rgemv={}: {} -> {:.1} GB/s, {:.2} GFLOP/s",
        tensor,
        info.ggml_type,
        rows,
        cols,
        rgemv,
        describe(&st),
        fmt_gbps(bytes.len() as u64, st.median_ms),
        2.0 * rows as f64 * cols as f64 / (st.median_ms / 1e3) / 1e9
    );
    let (xq_ref, xs_ref) = quantize_ref(&x, GROUP);
    if onehot.is_none() {
        let gq = gpu.download(&xq, cols)?;
        let gs = gpu.download(&xs, cols / GROUP)?;
        let dq = gq.iter().zip(&xq_ref).filter(|(a, b)| a != b).count();
        let ds = gs.iter().zip(&xs_ref).filter(|(a, b)| a != b).count();
        println!("  quant: xq diff {dq}/{cols}, xs diff {ds}", );
    }
    let row_bytes = bytes.len() / rows;
    if let Some(j) = onehot {
        let deq = match info.ggml_type {
            lfm_gguf::GgmlType::Q4_K => lfm_quant::dequant_row_q4_k(&bytes[..row_bytes], cols)?,
            lfm_gguf::GgmlType::Q6_K => lfm_quant::dequant_row_q6_k(&bytes[..row_bytes], cols)?,
            _ => unreachable!(),
        };
        println!("  onehot j={j}: gpu {:.6} cpu {:.6}", out[0], deq[j]);
    }
    let n = check_rows.min(rows);
    let mut max_i8 = 0.0f32;
    let mut max_f32 = 0.0f32;
    for r in 0..n {
        let slice = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let ref_i8 = dot_i8(info.ggml_type, slice, &xq_ref, &xs_ref)?;
        let ref_f32 = dot_f32(info.ggml_type, slice, &x)?;
        max_i8 = max_i8.max((out[r] - ref_i8).abs() / ref_i8.abs().max(1e-6));
        max_f32 = max_f32.max((ref_i8 - ref_f32).abs() / ref_f32.abs().max(1e-6));
    }
    println!(
        "  rows {n}: kernel-vs-int8-ref max_rel {:.2e}, w4a8-vs-f32 max_rel {:.2e}",
        max_i8, max_f32
    );
    Ok(())
}

fn sweep(path: &PathBuf, rgemv: u32) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(path)?;
    let gpu = Gpu::open(0)?;
    let k = build_kernels(&gpu, rgemv)?;
    let mut total_ms = 0.0f64;
    let mut total_bytes = 0u64;
    let mut count = 0usize;
    let mut per_type: std::collections::BTreeMap<String, (f64, u64)> =
        std::collections::BTreeMap::new();
    for t in gguf.tensors() {
        if t.dims.len() != 2 {
            continue;
        }
        let kernel = match t.ggml_type {
            lfm_gguf::GgmlType::Q4_K => &k.q4,
            lfm_gguf::GgmlType::Q6_K => &k.q6,
            _ => continue,
        };
        let tyname = format!("{:?}", t.ggml_type);
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        if cols % 256 != 0 || cols == 0 || rows == 0 {
            continue;
        }
        let bytes = gguf.tensor_bytes(&t.name)?;
        let w = gpu.upload_bytes(bytes)?;
        let x = make_x(cols);
        let xb = gpu.upload(&x)?;
        let xq = gpu.alloc::<i8>(cols)?;
        let xs = gpu.alloc::<f32>(cols / GROUP)?;
        let yb = gpu.alloc::<f32>(rows)?;
        quantize_on_gpu(&gpu, &k.quant, &xb, &xq, &xs, cols)?.wait()?;
        let st = gpu.bench(1, 3, || -> Result<()> {
            launch_gemv(&gpu, kernel, &w, &xq, &xs, &yb, rows, cols, rgemv as usize)?;
            Ok(())
        })?;
        total_ms += st.median_ms;
        total_bytes += bytes.len() as u64;
        count += 1;
        let e = per_type.entry(tyname).or_insert((0.0, 0));
        e.0 += st.median_ms;
        e.1 += bytes.len() as u64;
        if rgemv == 8 && std::env::var("LFM_VERBOSE").is_ok() {
            println!(
                "{:<40} {:>6}x{:<6} {:>8.3} ms {:>7.1} GB/s",
                t.name,
                rows,
                cols,
                st.median_ms,
                fmt_gbps(bytes.len() as u64, st.median_ms)
            );
        }
    }
    println!(
        "sweep rgemv={rgemv}: {:.3} ms/token over {} tensors, {:.1} MiB -> {:.1} tok/s weights-only",
        total_ms,
        count,
        total_bytes as f64 / (1024.0 * 1024.0),
        1000.0 / total_ms
    );
    for (ty, (ms, b)) in per_type {
        println!(
            "  {ty}: {ms:.3} ms, {:.1} MiB, {:.1} GB/s",
            b as f64 / (1024.0 * 1024.0),
            fmt_gbps(b, ms)
        );
    }
    Ok(())
}

struct StepTensor {
    name: String,
    is_q4: bool,
    w: Buffer<u8>,
    xq: Buffer<i8>,
    xs: Buffer<f32>,
    y: Buffer<f32>,
    rows: usize,
    cols: usize,
    bytes: u64,
}

fn step(path: &PathBuf, rgemv: u32, mode: &str, reps: usize) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(path)?;
    let gpu = Gpu::open(0)?;
    let k = build_kernels(&gpu, rgemv)?;
    let sum4 = Kernel::create(&k.program, "gemv_q4_k_sum")?;
    let sum6 = Kernel::create(&k.program, "gemv_q6_k_sum")?;
    let want_sum = mode == "sum";
    if mode == "single" {
        let mut total = 0u64;
        for t in gguf.tensors() {
            if t.dims.len() == 2
                && matches!(t.ggml_type, lfm_gguf::GgmlType::Q4_K)
                && t.dims[0] % 256 == 0
                && t.dims[0] > 0
                && t.dims[1] > 0
            {
                total += t.byte_size;
            }
        }
        let row_bytes = 8u64 * 144;
        let rows = (total / row_bytes) as usize;
        let nbytes = rows as u64 * row_bytes;
        let data = vec![0x11u8; nbytes as usize];
        let w = gpu.upload_bytes(&data)?;
        drop(data);
        let y = gpu.alloc::<f32>(rows)?;
        let x = make_x(2048);
        let xb = gpu.upload(&x)?;
        let xq = gpu.alloc::<i8>(2048)?;
        let xs = gpu.alloc::<f32>(2048 / GROUP)?;
        quantize_on_gpu(&gpu, &k.quant, &xb, &xq, &xs, 2048)?.wait()?;
        let st = gpu.bench(1, reps, || -> Result<()> {
            launch_gemv(&gpu, &k.q4, &w, &xq, &xs, &y, rows, 2048, rgemv as usize)?;
            Ok(())
        })?;
        println!(
            "step single: {:.3} ms for {:.1} MiB, {} rows -> {:.1} GB/s, {:.1} tok/s eq",
            st.median_ms,
            nbytes as f64 / (1024.0 * 1024.0),
            rows,
            fmt_gbps(nbytes, st.median_ms),
            1000.0 / st.median_ms
        );
        return Ok(());
    }
    let mut tensors: Vec<StepTensor> = Vec::new();
    let mut total_bytes = 0u64;
    for t in gguf.tensors() {
        if t.dims.len() != 2 {
            continue;
        }
        let is_q4 = match t.ggml_type {
            lfm_gguf::GgmlType::Q4_K => true,
            lfm_gguf::GgmlType::Q6_K => false,
            _ => continue,
        };
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        if cols % 256 != 0 || cols == 0 || rows == 0 {
            continue;
        }
        let bytes = gguf.tensor_bytes(&t.name)?;
        let w = gpu.upload_bytes(bytes)?;
        let x = make_x(cols);
        let xb = gpu.upload(&x)?;
        let xq = gpu.alloc::<i8>(cols)?;
        let xs = gpu.alloc::<f32>(cols / GROUP)?;
        let y = gpu.alloc::<f32>(rows)?;
        if !want_sum {
            quantize_on_gpu(&gpu, &k.quant, &xb, &xq, &xs, cols)?.wait()?;
        }
        total_bytes += bytes.len() as u64;
        tensors.push(StepTensor {
            name: t.name.clone(),
            is_q4,
            w,
            xq,
            xs,
            y,
            rows,
            cols,
            bytes: bytes.len() as u64,
        });
    }
    if mode == "fused" {
        let mut groups: Vec<(Vec<usize>, Buffer<u8>, Buffer<f32>, usize, usize, bool)> = Vec::new();
        let mut used = vec![false; tensors.len()];
        let mut push_group = |idx: Vec<usize>, gpu: &Gpu| -> Result<()> {
            let rows: usize = idx.iter().map(|&i| tensors[i].rows).sum();
            let cols = tensors[idx[0]].cols;
            let is_q4 = tensors[idx[0]].is_q4;
            let total: u64 = idx.iter().map(|&i| tensors[i].bytes).sum();
            let mut w = gpu.alloc::<u8>(total as usize)?;
            let mut off = 0usize;
            for &i in &idx {
                unsafe {
                    gpu.queue.enqueue_copy_buffer(
                        &tensors[i].w,
                        &mut w,
                        0,
                        off,
                        tensors[i].bytes as usize,
                        &[],
                    )?;
                }
                off += tensors[i].bytes as usize;
            }
            let y = gpu.alloc::<f32>(rows)?;
            groups.push((idx, w, y, rows, cols, is_q4));
            Ok(())
        };
        for l in 0..24 {
            let attn: Vec<usize> = tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.name.starts_with(&format!("blk.{l}."))
                        && (t.name.contains("attn_q.")
                            || t.name.contains("attn_k.")
                            || t.name.contains("attn_v.")
                            || t.name.contains("attn_gate."))
                })
                .map(|(i, _)| i)
                .collect();
            if !attn.is_empty() && attn.iter().all(|&i| tensors[i].is_q4) {
                for &i in &attn {
                    used[i] = true;
                }
                push_group(attn, &gpu)?;
            }
            let ffn: Vec<usize> = tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.name.starts_with(&format!("blk.{l}."))
                        && (t.name.contains("ffn_up.") || t.name.contains("ffn_gate."))
                })
                .map(|(i, _)| i)
                .collect();
            if !ffn.is_empty() {
                for &i in &ffn {
                    used[i] = true;
                }
                push_group(ffn, &gpu)?;
            }
        }
        for (i, t) in tensors.iter().enumerate() {
            if !used[i] {
                let _ = t;
                push_group(vec![i], &gpu)?;
            }
        }
        let mut prepared: Vec<(Buffer<i8>, Buffer<f32>)> = Vec::with_capacity(groups.len());
        for (_, _, _, _, cols, _) in &groups {
            let x = make_x(*cols);
            let xb = gpu.upload(&x)?;
            let xq = gpu.alloc::<i8>(*cols)?;
            let xs = gpu.alloc::<f32>(*cols / GROUP)?;
            quantize_on_gpu(&gpu, &k.quant, &xb, &xq, &xs, *cols)?.wait()?;
            prepared.push((xq, xs));
        }
        let enq_fused = |groups: &Vec<(Vec<usize>, Buffer<u8>, Buffer<f32>, usize, usize, bool)>, prepared: &Vec<(Buffer<i8>, Buffer<f32>)>| -> Result<()> {
            for (gi, (_, w, y, rows, cols, is_q4)) in groups.iter().enumerate() {
                let kern = if *is_q4 { &k.q4 } else { &k.q6 };
                let (xq, xs) = &prepared[gi];
                launch_gemv(&gpu, kern, w, xq, xs, y, *rows, *cols, rgemv as usize)?;
            }
            Ok(())
        };
        enq_fused(&groups, &prepared)?;
        gpu.queue.finish()?;
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            enq_fused(&groups, &prepared)?;
            gpu.queue.finish()?;
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = samples[samples.len() / 2];
        println!(
            "step fused rgemv={}: median {:.3} ms, min {:.3} ms, {} groups -> {:.1} tok/s, {:.1} GB/s",
            rgemv,
            med,
            samples[0],
            groups.len(),
            1000.0 / med,
            fmt_gbps(total_bytes, med)
        );
        return Ok(());
    }
    let enqueue = |sum: bool| -> Result<()> {
        for t in &tensors {
            if sum {
                let kern = if t.is_q4 { &sum4 } else { &sum6 };
                let rpw = 256 / gpu.caps.sub_group_size.max(1);
                let per_wg = (rpw * rgemv as usize).max(1);
                let groups = t.rows.div_ceil(per_wg);
                unsafe {
                    gpu.launch(
                        kern,
                        groups * 256,
                        256,
                        &[
                            Arg::Mem(t.w.get()),
                            Arg::Mem(t.y.get()),
                            Arg::U32(t.rows as u32),
                            Arg::U32(t.cols as u32),
                        ],
                    )?;
                }
            } else {
                let kern = if t.is_q4 { &k.q4 } else { &k.q6 };
                launch_gemv(
                    &gpu,
                    kern,
                    &t.w,
                    &t.xq,
                    &t.xs,
                    &t.y,
                    t.rows,
                    t.cols,
                    rgemv as usize,
                )?;
            }
        }
        Ok(())
    };
    let t_upload = Instant::now();
    gpu.queue.finish()?;
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1e3;
    enqueue(want_sum)?;
    gpu.queue.finish()?;
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        enqueue(want_sum)?;
        gpu.queue.finish()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = samples[samples.len() / 2];
    let min = samples[0];
    println!(
        "step {} rgemv={}: median {:.3} ms, min {:.3} ms, {} tensors, {:.1} MiB -> {:.1} tok/s, {:.1} GB/s",
        if want_sum { "sum" } else { "w4a8" },
        rgemv,
        med,
        min,
        tensors.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
        1000.0 / med,
        fmt_gbps(total_bytes, med)
    );
    println!("  setup/finish before first step: {:.1} ms", upload_ms);
    Ok(())
}

fn dump(path: &PathBuf, name: &str) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(path)?;
    let t = gguf
        .tensor(name)
        .ok_or_else(|| anyhow!("tensor {name} not found"))?;
    let cols = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    let bytes = gguf.tensor_bytes(name)?;
    let row_bytes = bytes.len() / rows;
    let y = match t.ggml_type {
        lfm_gguf::GgmlType::Q4_K => lfm_quant::dequant_row_q4_k(&bytes[..row_bytes], cols)?,
        lfm_gguf::GgmlType::Q6_K => lfm_quant::dequant_row_q6_k(&bytes[..row_bytes], cols)?,
        other => return Err(anyhow!("unsupported type {other:?}")),
    };
    println!("first8 {:?}", &y[..8].iter().map(|v| format!("{v:.9}")).collect::<Vec<_>>());
    let x = make_x(cols);
    let dot = match t.ggml_type {
        lfm_gguf::GgmlType::Q4_K => lfm_quant::dot_row_q4_k(&bytes[..row_bytes], &x)?,
        lfm_gguf::GgmlType::Q6_K => lfm_quant::dot_row_q6_k(&bytes[..row_bytes], &x)?,
        other => return Err(anyhow!("unsupported type {other:?}")),
    };
    println!("dot_row0 {dot:.9}");
    Ok(())
}

fn mixed(iters: u32) -> Result<()> {
    let gpu = Gpu::open(0)?;
    let program = gpu.build(BENCH_CL, "-cl-std=CL3.0")?;
    let kernel = Kernel::create(&program, "peak_mixed")?;
    let global = gpu.caps.compute_units as usize * 64 * 32;
    let sinkf = gpu.alloc::<f32>(global)?;
    let sinku = gpu.alloc::<u32>(global)?;
    for ratio in [0u32, 25, 50, 75, 100] {
        let st = gpu.bench(3, 10, || -> Result<()> {
            unsafe {
                gpu.launch(
                    &kernel,
                    global,
                    256,
                    &[
                        Arg::Mem(sinkf.get()),
                        Arg::Mem(sinku.get()),
                        Arg::U32(iters),
                        Arg::U32(ratio),
                    ],
                )?;
            }
            Ok(())
        })?;
        let i8_macs = iters as f64 * 8.0 * 4.0 * global as f64 * ratio as f64 / 100.0;
        let f16_macs = iters as f64 * 8.0 * 2.0 * global as f64 * (100 - ratio) as f64 / 100.0;
        let tops = 2.0 * (i8_macs + f16_macs) / (st.median_ms / 1e3) / 1e12;
        println!(
            "int8 share {:>3}%: {:.3} ms -> {:.2} TOPS (optimistic fp16=2 mac/lane/instr)",
            ratio, st.median_ms, tops
        );
    }
    Ok(())
}

fn reference_ids(model: &PathBuf, ids: &[u32], max_new: usize, eos: u32) -> Result<()> {
    let started = Instant::now();
    let reference = lfm_cpu::Lfm3Ref::load(model)?;
    println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let out = reference.generate_greedy(ids, max_new, eos)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "generated {} ids in {:.2} s ({:.2} tok/s): {:?}",
        out.len(),
        elapsed,
        out.len() as f64 / elapsed,
        out
    );
    Ok(())
}

fn reference_debug(model: &PathBuf, ids: &[u32], out: Option<&PathBuf>) -> Result<()> {
    let reference = lfm_cpu::Lfm3Ref::load(model)?;
    let (logits, norms) = reference.forward_debug(ids)?;
    println!("hidden norms (embed + per layer + final):");
    for (i, n) in norms.iter().enumerate() {
        println!("  {i:>2}: {n:.4}");
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
    println!("top5:");
    for k in 0..5 {
        println!("  {} = {:.4}", idx[k], logits[idx[k]]);
    }
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let var = logits.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / logits.len() as f32;
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("logits: min {min:.4} max {max:.4} mean {mean:.4} std {:.4}", var.sqrt());
    println!("argmax {}", idx[0]);
    if let Some(path) = out {
        let mut bytes = Vec::with_capacity(logits.len() * 4);
        for v in &logits {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn reference_generate(
    model: &PathBuf,
    tokenizer: &PathBuf,
    prompt: &str,
    max_new: usize,
) -> Result<()> {
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let ids = tok.encode(prompt, false)?;
    println!("prompt ids: {ids:?}");
    let reference = lfm_cpu::Lfm3Ref::load(model)?;
    let eos = tok.eos_id();
    let started = Instant::now();
    let out = reference.generate_greedy(&ids, max_new, eos)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{} ids in {:.2} s ({:.2} tok/s)",
        out.len(),
        elapsed,
        out.len() as f64 / elapsed.max(1e-9)
    );
    println!("ids: {out:?}");
    println!("text: {}", tok.decode(&out, false)?);
    Ok(())
}

fn reference_ppl(model: &PathBuf, tokenizer: &PathBuf, text: &str) -> Result<()> {
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let ids = tok.encode(text, false)?;
    if ids.len() < 2 {
        return Err(anyhow!("text too short"));
    }
    let reference = lfm_cpu::Lfm3Ref::load(model)?;
    let mut state = lfm_cpu::State::new(&reference.cfg);
    let mut nll = 0.0f64;
    let mut count = 0usize;
    let started = Instant::now();
    for (pos, token) in ids.iter().enumerate() {
        let logits = reference.forward(&mut state, *token, pos)?;
        if pos + 1 < ids.len() {
            let next = ids[pos + 1] as usize;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for v in &logits {
                sum += ((*v - max) as f64).exp();
            }
            let logp = (logits[next] - max) as f64 - sum.ln();
            nll -= logp;
            count += 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "tokens {} scored {} ppl {:.4} in {:.1} s ({:.2} tok/s)",
        ids.len(),
        count,
        (nll / count as f64).exp(),
        elapsed,
        ids.len() as f64 / elapsed
    );
    Ok(())
}

fn sampling(
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    seed: u64,
) -> lfm_domain::SamplingParams {
    lfm_domain::SamplingParams {
        temperature: temp,
        top_k,
        top_p,
        repeat_penalty,
        seed,
        ..lfm_domain::SamplingParams::default()
    }
}

fn gpu_chat(
    model: &PathBuf,
    tokenizer: &PathBuf,
    template: &PathBuf,
    system: &str,
    user: &str,
    max_new: usize,
    thinking: bool,
    params: &lfm_domain::SamplingParams,
    interactive: bool,
    tui: bool,
    ctx: usize,
) -> Result<()> {
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let chat = lfm_chat::ChatTemplate::from_file(template)?;
    if tui {
        return tui_chat(model, tokenizer, template, system, max_new, thinking, params, ctx);
    }
    let mut gpu = lfm_gpu::GpuModel::load(model, ctx)?;
    let mut history = vec![lfm_domain::Message::system(system)];
    if !interactive {
        history.push(lfm_domain::Message::user(user));
        return answer(&mut gpu, &tok, &chat, &mut history, max_new, thinking, params);
    }
    println!(
        "интерактивный чат ({} токенов контекста). /exit — выход, /reset — очистить историю.",
        gpu.max_ctx()
    );
    let stdin = std::io::stdin();
    loop {
        print!("you> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/exit" || text == "/quit" {
            break;
        }
        if text == "/reset" {
            history.truncate(1);
            println!("история очищена");
            continue;
        }
        history.push(lfm_domain::Message::user(text));
        print!("assistant> ");
        std::io::stdout().flush()?;
        if let Err(e) = answer(&mut gpu, &tok, &chat, &mut history, max_new, thinking, params) {
            eprintln!("ошибка: {e}");
            history.pop();
        }
    }
    Ok(())
}

const TOOL_ARGS_MAX: usize = 2000;

const TUI_HELP: &str = "/help — эта справка\n/reset — очистить историю\n/exit, /quit — выход\nPgUp/PgDn — листать историю\nHome/End — начало/конец\nUp/Down — история ввода (при пустой строке — прокрутка)\nCtrl+C — выход";

struct TuiState {
    history: Vec<(String, String)>,
    assistant: String,
    thinking: String,
    tools: Vec<(String, String)>,
    input: String,
    input_history: InputHistory,
    status: String,
    scroll: usize,
    pending: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
struct InputHistory {
    entries: Vec<String>,
    pos: usize,
    draft: String,
}

impl InputHistory {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, line: &str) {
        let line = line.trim();
        if !line.is_empty() && self.entries.last().map(|l| l.as_str()) != Some(line) {
            self.entries.push(line.to_string());
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.pos = self.entries.len();
        self.draft.clear();
    }

    fn up(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        if self.pos == self.entries.len() {
            self.draft = current.to_string();
        }
        if self.pos > 0 {
            self.pos -= 1;
        }
        self.entries.get(self.pos).cloned()
    }

    fn down(&mut self) -> Option<String> {
        if self.pos >= self.entries.len() {
            return None;
        }
        self.pos += 1;
        if self.pos == self.entries.len() {
            Some(std::mem::take(&mut self.draft))
        } else {
            self.entries.get(self.pos).cloned()
        }
    }
}

fn scroll_after_pgup(scroll: usize, area: usize) -> usize {
    scroll.saturating_add(area.max(1))
}

fn scroll_after_pgdn(scroll: usize, area: usize) -> usize {
    scroll.saturating_sub(area.max(1))
}

fn scroll_home(total: usize, area: usize) -> usize {
    total.saturating_sub(area.max(1))
}

fn scroll_end() -> usize {
    0
}

fn term_size() -> (usize, usize) {
    let out = std::process::Command::new("stty").arg("size").output();
    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut parts = text.split_whitespace();
        if let (Some(r), Some(c)) = (parts.next(), parts.next()) {
            if let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>()) {
                if r > 4 && c > 20 {
                    return (r, c);
                }
            }
        }
    }
    (24, 80)
}

fn set_term_raw(on: bool) -> Result<()> {
    let mut cmd = std::process::Command::new("stty");
    if on {
        cmd.args(["raw", "-echo"]);
    } else {
        cmd.arg("sane");
    }
    cmd.status()?;
    Ok(())
}

fn set_term_timeout(on: bool) -> Result<()> {
    let mut cmd = std::process::Command::new("stty");
    if on {
        cmd.args(["min", "0", "time", "0"]);
    } else {
        cmd.args(["min", "1", "time", "0"]);
    }
    cmd.status()?;
    Ok(())
}

fn poll_interrupt(pending: &mut Vec<u8>) -> bool {
    let mut buf = [0u8; 64];
    let mut interrupt = false;
    match std::io::stdin().read(&mut buf) {
        Ok(n) => {
            for byte in &buf[..n] {
                if *byte == 3 {
                    interrupt = true;
                } else {
                    pending.push(*byte);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
        Err(_) => {}
    }
    interrupt
}

fn clip(text: &str, cols: usize) -> String {
    text.chars().take(cols).collect()
}

fn wrap(text: &str, cols: usize) -> Vec<String> {
    let width = cols.max(8);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut count = 0usize;
        for ch in raw.chars() {
            if count == width {
                out.push(std::mem::take(&mut line));
                count = 0;
            }
            line.push(ch);
            count += 1;
        }
        out.push(line);
    }
    out
}

fn history_prefix(who: &str) -> &'static str {
    match who {
        "you" => "you> ",
        "ai" => "ai>  ",
        _ => "[sys] ",
    }
}

fn format_tool_call(name: &str, args: &str, cols: usize) -> Vec<String> {
    let width = cols.max(8);
    let head = format!("[tool] {}", if name.is_empty() { "?" } else { name });
    let mut out = vec![clip(&head, cols)];
    let args = args.trim();
    if args.is_empty() {
        return out;
    }
    let total = args.chars().count();
    let mut body: String = args.chars().take(TOOL_ARGS_MAX).collect();
    if total > TOOL_ARGS_MAX {
        body.push('…');
    }
    for line in wrap(&body, width.saturating_sub(2)) {
        out.push(clip(&format!("  {line}"), cols));
    }
    out
}

fn push_block(lines: &mut Vec<String>, head: String, indent: usize, cols: usize) {
    let pad = " ".repeat(indent);
    for (i, l) in wrap(&head, cols).into_iter().enumerate() {
        lines.push(if i == 0 { l } else { format!("{pad}{l}") });
    }
}

fn transcript_lines(state: &TuiState, cols: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for (who, text) in &state.history {
        let prefix = history_prefix(who);
        push_block(
            &mut lines,
            format!("{prefix}{text}"),
            prefix.chars().count(),
            cols,
        );
        lines.push(String::new());
    }
    if !state.thinking.is_empty() {
        push_block(&mut lines, format!("[think] {}", state.thinking), 7, cols);
    }
    if !state.assistant.is_empty() {
        push_block(&mut lines, format!("ai>  {}", state.assistant), 5, cols);
    }
    for (name, args) in &state.tools {
        lines.extend(format_tool_call(name, args, cols));
    }
    lines
}

#[derive(Debug, Clone, Default)]
struct LinesView {
    body: Vec<String>,
    area: usize,
    total: usize,
    scroll: usize,
}

fn body_area(rows: usize) -> usize {
    rows.saturating_sub(3).max(1)
}

fn visible_lines(state: &TuiState, rows: usize, cols: usize) -> LinesView {
    let lines = transcript_lines(state, cols);
    let area = body_area(rows);
    let total = lines.len();
    let max_scroll = total.saturating_sub(area);
    let scroll = state.scroll.min(max_scroll);
    let start = max_scroll - scroll;
    let mut body: Vec<String> = Vec::with_capacity(area);
    for i in 0..area {
        body.push(lines.get(start + i).cloned().unwrap_or_default());
    }
    LinesView {
        body,
        area,
        total,
        scroll,
    }
}

fn tui_render(state: &TuiState, rows: usize, cols: usize) {
    let view = visible_lines(state, rows, cols);
    let mut out = String::from("\x1b[H");
    for line in &view.body {
        out.push_str("\x1b[2K");
        out.push_str(&clip(line, cols));
        out.push_str("\r\n");
    }
    let mut status = state.status.clone();
    if view.scroll > 0 {
        status.push_str(&format!(" | ↑{}", view.scroll));
    }
    out.push_str("\x1b[2K\x1b[7m");
    out.push_str(&clip(&status, cols));
    out.push_str("\x1b[0m\r\n\x1b[2Kyou> ");
    out.push_str(&clip(&state.input, cols.saturating_sub(5)));
    out.push_str("\x1b[?25h");
    print!("{out}");
    let _ = std::io::stdout().flush();
}

fn render_now(state: &TuiState) {
    let (rows, cols) = term_size();
    tui_render(state, rows, cols);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    Up,
    Down,
    PgUp,
    PgDn,
    Home,
    End,
    Other,
}

fn parse_escape(seq: &str) -> Key {
    match seq {
        "[A" | "OA" => Key::Up,
        "[B" | "OB" => Key::Down,
        "[5~" => Key::PgUp,
        "[6~" => Key::PgDn,
        "[H" | "[1~" | "[7~" | "OH" => Key::Home,
        "[F" | "[4~" | "[8~" | "OF" => Key::End,
        _ => Key::Other,
    }
}

fn read_escape(stdin: &mut std::io::Stdin) -> Option<Key> {
    let mut b = [0u8; 1];
    if stdin.read(&mut b).ok()? == 0 {
        return None;
    }
    if b[0] != b'[' && b[0] != b'O' {
        return Some(Key::Other);
    }
    let mut seq = String::new();
    seq.push(b[0] as char);
    loop {
        if stdin.read(&mut b).ok()? == 0 {
            return None;
        }
        seq.push(b[0] as char);
        if seq.len() > 8 {
            break;
        }
        if (0x40..=0x7e).contains(&b[0]) {
            break;
        }
    }
    Some(parse_escape(&seq))
}

fn scroll_key(state: &mut TuiState, key: Key) {
    let (rows, cols) = term_size();
    let view = visible_lines(state, rows, cols);
    let max_scroll = view.total.saturating_sub(view.area);
    let area = view.area;
    let next = match key {
        Key::PgUp => scroll_after_pgup(state.scroll, area),
        Key::PgDn => scroll_after_pgdn(state.scroll, area),
        Key::Up => scroll_after_pgup(state.scroll, 1),
        Key::Down => scroll_after_pgdn(state.scroll, 1),
        Key::Home => scroll_home(view.total, area),
        Key::End | Key::Other => scroll_end(),
    };
    state.scroll = next.min(max_scroll);
    tui_render(state, rows, cols);
}

fn handle_nav_key(state: &mut TuiState, key: Key) {
    match key {
        Key::PgUp | Key::PgDn | Key::Home | Key::End => scroll_key(state, key),
        Key::Up if state.input.is_empty() => scroll_key(state, Key::Up),
        Key::Down if state.input.is_empty() => scroll_key(state, Key::Down),
        Key::Up => {
            if let Some(line) = state.input_history.up(&state.input) {
                state.input = line;
            }
            render_now(state);
        }
        Key::Down => {
            if let Some(line) = state.input_history.down() {
                state.input = line;
                render_now(state);
            }
        }
        Key::Other => {}
    }
}

fn read_input_line(state: &mut TuiState) -> Option<String> {
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 1];
    let mut utf8: Vec<u8> = Vec::new();
    loop {
        let byte = if state.pending.is_empty() {
            match stdin.read(&mut buf) {
                Ok(0) => return None,
                Ok(_) => buf[0],
                Err(_) => return None,
            }
        } else {
            state.pending.remove(0)
        };
        match byte {
            3 | 4 => return None,
            b'\r' | b'\n' => {
                state.input_history.reset();
                return Some(std::mem::take(&mut state.input));
            }
            0x7f | 0x08 => {
                state.input.pop();
                utf8.clear();
                render_now(state);
            }
            0x1b => {
                let key = read_escape(&mut stdin)?;
                handle_nav_key(state, key);
            }
            b if b >= 0x20 => {
                utf8.push(b);
                if let Ok(text) = std::str::from_utf8(&utf8) {
                    state.input.push_str(text);
                    utf8.clear();
                }
                render_now(state);
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn tui_chat(
    model: &PathBuf,
    tokenizer: &PathBuf,
    template: &PathBuf,
    system: &str,
    max_new: usize,
    thinking: bool,
    params: &lfm_domain::SamplingParams,
    ctx: usize,
) -> Result<()> {
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let chat = lfm_chat::ChatTemplate::from_file(template)?;
    let mut gpu = lfm_gpu::GpuModel::load(model, ctx)?;
    let name = model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut state = TuiState {
        history: Vec::new(),
        assistant: String::new(),
        thinking: String::new(),
        tools: Vec::new(),
        input: String::new(),
        input_history: InputHistory::new(),
        status: format!(
            "{name} | ctx {} | PgUp/PgDn — история, /help | Ctrl+C — выход",
            gpu.max_ctx()
        ),
        scroll: 0,
        pending: Vec::new(),
    };
    set_term_raw(true)?;
    print!("\x1b[?1049h\x1b[2J");
    let _ = std::io::stdout().flush();
    let result = tui_loop(&mut gpu, &tok, &chat, system, max_new, thinking, params, &mut state);
    print!("\x1b[?25h\x1b[?1049l");
    let _ = std::io::stdout().flush();
    set_term_raw(false)?;
    result
}

#[allow(clippy::too_many_arguments)]
fn tui_loop(
    gpu: &mut lfm_gpu::GpuModel,
    tok: &lfm_tokenizer::LfmTokenizer,
    chat: &lfm_chat::ChatTemplate,
    system: &str,
    max_new: usize,
    thinking: bool,
    params: &lfm_domain::SamplingParams,
    state: &mut TuiState,
) -> Result<()> {
    let mut history = vec![lfm_domain::Message::system(system)];
    let base_status = state.status.clone();
    loop {
        let (rows, cols) = term_size();
        tui_render(state, rows, cols);
        let Some(line) = read_input_line(state) else {
            return Ok(());
        };
        let text = line.trim().to_string();
        if text.is_empty() {
            continue;
        }
        if text == "/exit" || text == "/quit" {
            return Ok(());
        }
        state.input_history.push(&text);
        state.scroll = 0;
        if text == "/help" {
            state.history.push(("sys".to_string(), TUI_HELP.to_string()));
            continue;
        }
        if text == "/reset" {
            history.truncate(1);
            state.history.clear();
            state.tools.clear();
            state.status = base_status.clone();
            continue;
        }
        history.push(lfm_domain::Message::user(&text));
        state.history.push(("you".to_string(), text));
        state.assistant.clear();
        state.thinking.clear();
        state.tools.clear();
        state.scroll = 0;
        let prompt = chat.render(&history, &[], true, thinking, false)?;
        let ids = tok.encode(&prompt, false)?;
        let avail = gpu.max_ctx();
        if ids.len() + 1 >= avail {
            state.status = format!("история не влезает: {} токенов при ctx {}", ids.len(), avail);
            history.pop();
            continue;
        }
        let max_new = max_new.min(avail - ids.len());
        let mut session = lfm_app::ChatSession::new();
        if thinking {
            session.begin_thinking();
        }
        let started = Instant::now();
        let mut tokens: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut first: Option<f64> = None;
        let cancel = AtomicBool::new(false);
        set_term_timeout(true)?;
        let res = {
            let prompt_tokens = ids.len();
            let _ = prompt_tokens;
            gpu.generate_stream_cancel(&ids, params, max_new, tok.eos_id(), &cancel, |next| {
                if poll_interrupt(&mut state.pending) {
                    cancel.store(true, Ordering::Relaxed);
                }
                tokens.push(next);
                state.scroll = 0;
                let Ok(text) = tok.decode(&tokens, true) else {
                    return;
                };
                if !(text.len() > printed.len() && text.starts_with(&printed)) {
                    return;
                }
                let delta = text[printed.len()..].to_string();
                printed = text;
                for ev in session.feed(&delta) {
                    match ev {
                        lfm_app::SessionEvent::Thinking(t) => state.thinking.push_str(&t),
                        lfm_app::SessionEvent::Content(t) => state.assistant.push_str(&t),
                        lfm_app::SessionEvent::ToolCall(tc) => {
                            state.tools.push((tc.name, tc.arguments.to_string()));
                        }
                        lfm_app::SessionEvent::Finished => {}
                    }
                }
                let elapsed = started.elapsed().as_secs_f64();
                let ttft = *first.get_or_insert(elapsed);
                state.status = format!(
                    "{base_status} | TTFT {ttft:.2} c | {:.1} tok/s | {} ток.",
                    tokens.len() as f64 / elapsed.max(1e-9),
                    tokens.len()
                );
                let (rows, cols) = term_size();
                tui_render(state, rows, cols);
            })
        };
        set_term_timeout(false)?;
        for ev in session.finish() {
            if let lfm_app::SessionEvent::ToolCall(tc) = ev {
                state.tools.push((tc.name, tc.arguments.to_string()));
            }
        }
        let cancelled = cancel.load(Ordering::Relaxed);
        if cancelled {
            state.status = format!("прервано ({} ток.) | {base_status}", tokens.len());
        }
        match res {
            Ok(_) => {
                let (_, cols) = term_size();
                let mut answer = std::mem::take(&mut state.assistant);
                for (name, args) in state.tools.drain(..) {
                    if !answer.is_empty() && !answer.ends_with('\n') {
                        answer.push('\n');
                    }
                    answer.push_str(&format_tool_call(&name, &args, cols).join("\n"));
                }
                if answer.is_empty() {
                    history.pop();
                    continue;
                }
                state.history.push(("ai".to_string(), answer.clone()));
                let msg = session
                    .take_assistant()
                    .unwrap_or_else(|| lfm_domain::Message::assistant(answer));
                history.push(msg);
            }
            Err(e) => {
                state.status = format!("ошибка: {e} | {base_status}");
                history.pop();
            }
        }
    }
}

fn answer(
    gpu: &mut lfm_gpu::GpuModel,
    tok: &lfm_tokenizer::LfmTokenizer,
    chat: &lfm_chat::ChatTemplate,
    history: &mut Vec<lfm_domain::Message>,
    max_new: usize,
    thinking: bool,
    params: &lfm_domain::SamplingParams,
) -> Result<()> {
    let prompt = chat.render(history, &[], true, thinking, false)?;
    let ids = tok.encode(&prompt, false)?;
    let avail = gpu.max_ctx();
    if ids.len() + 1 >= avail {
        return Err(anyhow!(
            "история не влезает в контекст: {} токенов при max_ctx {}",
            ids.len(),
            avail
        ));
    }
    let max_new = max_new.min(avail - ids.len());
    let mut session = lfm_app::ChatSession::new();
    if thinking {
        session.begin_thinking();
    }
    let started = Instant::now();
    let mut generated = Vec::new();
    let mut printed = String::new();
    gpu.generate_stream(&ids, params, max_new, tok.eos_id(), |next| {
        generated.push(next);
        let Ok(text) = tok.decode(&generated, true) else {
            return;
        };
        if text.len() > printed.len() && text.starts_with(&printed) {
            let delta = text[printed.len()..].to_string();
            for ev in session.feed(&delta) {
                match ev {
                    lfm_app::SessionEvent::Thinking(t) => print!("{t}"),
                    lfm_app::SessionEvent::Content(t) => print!("{t}"),
                    lfm_app::SessionEvent::ToolCall(tc) => {
                        println!("\n[tool_call] {} {}", tc.name, tc.arguments)
                    }
                    lfm_app::SessionEvent::Finished => {}
                }
            }
            printed = text;
        }
    })?;
    for ev in session.finish() {
        if let lfm_app::SessionEvent::ToolCall(tc) = ev {
            println!("\n[tool_call] {} {}", tc.name, tc.arguments);
        }
    }
    let msg = session
        .take_assistant()
        .unwrap_or_else(|| lfm_domain::Message::assistant(printed.clone()));
    history.push(msg);
    let elapsed = started.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "[{:.2} c, {:.1} tok/s, {} токенов]",
        elapsed,
        generated.len() as f64 / elapsed.max(1e-9),
        generated.len()
    );
    Ok(())
}

fn gpu_generate(
    model: &PathBuf,
    tokenizer: &PathBuf,
    prompt: &str,
    max_new: usize,
    params: &lfm_domain::SamplingParams,
) -> Result<()> {
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let ids = tok.encode(prompt, false)?;
    println!("prompt ids: {ids:?}");
    let started = Instant::now();
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let out = gpu.generate(&ids, params, max_new, tok.eos_id())?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{} ids in {:.2} s ({:.2} tok/s)",
        out.len(),
        elapsed,
        out.len() as f64 / elapsed.max(1e-9)
    );
    println!("ids: {out:?}");
    println!("text: {}", tok.decode(&out, false)?);
    Ok(())
}

fn gpu_mix(model: &PathBuf, tokens: usize) -> Result<()> {
    if tokens < 8 {
        return Err(anyhow!("need at least 8 tokens"));
    }
    let b_len = (tokens / 3).max(4);
    let mut gpu = lfm_gpu::GpuModel::load(model, 1024)?;
    let a_ids: Vec<u32> = (0..tokens)
        .map(|i| ((i * 2654435761usize) % 60000) as u32 + 100)
        .collect();
    let b_ids: Vec<u32> = (0..b_len)
        .map(|i| ((i * 7919 + 1234) % 60000) as u32 + 100)
        .collect();

    gpu.reset()?;
    let sa = gpu.create_sequence()?;
    let sb = gpu.create_sequence()?;
    let mut solo_a = Vec::with_capacity(tokens);
    for (k, token) in a_ids.iter().enumerate() {
        gpu.step_mixed(
            &[lfm_gpu::StepReq {
                slot: sa,
                token: *token,
                pos: k,
            }],
            true,
        )?;
        solo_a.push(gpu.logits_row(0)?);
    }
    gpu.prefill(sb, &b_ids, true)?;
    let mut solo_b: Vec<Vec<f32>> = vec![gpu.logits()?];
    for j in 1..tokens {
        let tok = lfm_gpu::argmax(solo_b.last().expect("logits"));
        gpu.step_mixed(
            &[lfm_gpu::StepReq {
                slot: sb,
                token: tok,
                pos: b_len + j - 1,
            }],
            true,
        )?;
        solo_b.push(gpu.logits_row(0)?);
    }

    gpu.reset()?;
    let ma = gpu.create_sequence()?;
    let mb = gpu.create_sequence()?;
    gpu.prefill(mb, &b_ids, true)?;
    let mut b_tok = lfm_gpu::argmax(&gpu.logits()?);
    let mut cos_a = Vec::with_capacity(tokens);
    let mut cos_b = Vec::with_capacity(tokens);
    let mut mismatch = 0usize;
    for k in 0..tokens {
        if b_tok != lfm_gpu::argmax(&solo_b[k]) {
            mismatch += 1;
        }
        let reqs = [
            lfm_gpu::StepReq {
                slot: ma,
                token: a_ids[k],
                pos: k,
            },
            lfm_gpu::StepReq {
                slot: mb,
                token: b_tok,
                pos: b_len + k,
            },
        ];
        gpu.step_mixed(&reqs, true)?;
        let la = gpu.logits_row(0)?;
        let lb = gpu.logits_row(1)?;
        cos_a.push(cosine(&solo_a[k], &la));
        if lfm_gpu::argmax(&la) != lfm_gpu::argmax(&solo_a[k]) {
            mismatch += 1;
        }
        if k + 1 < tokens {
            cos_b.push(cosine(&solo_b[k + 1], &lb));
            let next = lfm_gpu::argmax(&lb);
            if next != lfm_gpu::argmax(&solo_b[k + 1]) {
                mismatch += 1;
            }
            b_tok = next;
        }
    }
    let min_a = cos_a.iter().cloned().fold(1.0f64, f64::min);
    let mean_a = cos_a.iter().sum::<f64>() / cos_a.len() as f64;
    let min_b = cos_b.iter().cloned().fold(1.0f64, f64::min);
    let mean_b = cos_b.iter().sum::<f64>() / cos_b.len().max(1) as f64;
    println!(
        "mixed: {} шагов (A prefill {} ток., B decode из {}); cos A min {:.6} mean {:.6}; cos B min {:.6} mean {:.6}; расхождений {mismatch}",
        tokens, tokens, b_len, min_a, mean_a, min_b, mean_b
    );
    Ok(())
}

fn top_ids(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_by(|a, b| {
        v[*b as usize]
            .partial_cmp(&v[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

fn gpu_cache(model: &PathBuf, tokens: usize, reps: usize, reset: bool) -> Result<()> {
    if tokens < 64 {
        return Err(anyhow!("need at least 64 tokens"));
    }
    let ctx = (tokens + 64).max(1024);
    let mut gpu = lfm_gpu::GpuModel::load(model, ctx)?;
    let ids: Vec<u32> = (0..tokens)
        .map(|i| ((i * 2654435761usize) % 60000) as u32 + 100)
        .collect();
    let mut base: Option<Vec<f32>> = None;
    for rep in 0..reps {
        let slot = if reset {
            gpu.reset()?;
            0
        } else {
            gpu.free_sequence(0)?;
            gpu.create_sequence()?
        };
        gpu.finish()?;
        let started = Instant::now();
        let reused = if std::env::var("LFM_OLD").is_ok() {
            let pb = gpu.prefill_batch_size();
            let mut at = 0;
            while at < ids.len() {
                let end = (at + pb).min(ids.len());
                gpu.forward_batch(&ids[at..end], at, end == ids.len())?;
                at = end;
            }
            0
        } else {
            gpu.prefill(slot, &ids, true)?
        };
        let logits = gpu.logits()?;
        let elapsed = started.elapsed().as_secs_f64();
        let (cached, used, hits) = gpu.cache_stats();
        println!(
            "rep {rep}: {:.1} ms, {:.1} tok/s, reused {reused}/{} tok, cache: {cached} cached / {used} used pages, {hits} hits",
            elapsed * 1e3,
            ids.len() as f64 / elapsed.max(1e-9),
            ids.len()
        );
        match &base {
            Some(b) => println!(
                "  cos vs rep0 {:.8}, top5 {:?}",
                cosine(b, &logits),
                top_ids(&logits, 5)
            ),
            None => {
                println!("  top5 {:?}", top_ids(&logits, 5));
                base = Some(logits);
            }
        }
    }
    gpu.reset()?;
    let a = gpu.create_sequence()?;
    let b = gpu.create_sequence()?;
    let ra = gpu.prefill(a, &ids, true)?;
    let la = gpu.logits()?;
    let rb = gpu.prefill(b, &ids, true)?;
    let lb = gpu.logits()?;
    println!(
        "shared prefix: a reused {ra}, b reused {rb}, cos {:.8}, cache {:?}",
        cosine(&la, &lb),
        gpu.cache_stats()
    );
    gpu.free_sequence(a)?;
    gpu.free_sequence(b)?;
    let tail: Vec<u32> = (0..64).map(|i| 1000 + (i as u32) * 7).collect();
    let mut other = ids[..448].to_vec();
    other.extend_from_slice(&tail);
    let c = gpu.create_sequence()?;
    let rc = gpu.prefill(c, &other, true)?;
    println!(
        "partial prefix: reused {rc}/{} tok, cache {:?}",
        other.len(),
        gpu.cache_stats()
    );
    gpu.free_sequence(c)?;
    for i in 0..12usize {
        let p: Vec<u32> = (0..128)
            .map(|j| ((i * 131 + j * 17) % 50000) as u32 + 100)
            .collect();
        let s = gpu.create_sequence()?;
        gpu.prefill(s, &p, false)?;
        gpu.free_sequence(s)?;
    }
    let d = gpu.create_sequence()?;
    let rd = gpu.prefill(d, &ids, true)?;
    let ld = gpu.logits()?;
    let base = base.ok_or_else(|| anyhow!("missing baseline"))?;
    println!(
        "after churn: reused {rd}/{} tok, cos vs rep0 {:.8}, cache {:?}",
        ids.len(),
        cosine(&base, &ld),
        gpu.cache_stats()
    );
    Ok(())
}

fn gpu_multi(model: &PathBuf, nseq: usize, tokens: usize, prompt_len: usize) -> Result<()> {
    if nseq > 8 {
        return Err(anyhow!("at most 8 sequences"));
    }
    let prompts: Vec<Vec<u32>> = (0..nseq)
        .map(|i| {
            (0..prompt_len.max(1))
                .map(|j| ((i * 977 + j * 131 + 597) % 60000) as u32 + 100)
                .collect()
        })
        .collect();
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    gpu.free_sequence(0)?;
    let mut slots = Vec::new();
    for _ in 0..nseq {
        slots.push(gpu.create_sequence()?);
    }
    let mut pos: Vec<usize> = vec![0; nseq];
    let cap = gpu.prefill_batch_size().max(1);
    let total_prompt: usize = prompts.iter().map(|p| p.len()).sum();
    gpu.reset_profile();
    let prefill_started = Instant::now();
    let mut done = vec![0usize; nseq];
    loop {
        let mut reqs: Vec<lfm_gpu::StepReq> = Vec::new();
        let mut round = 0usize;
        while reqs.len() < cap {
            let mut pushed = false;
            for i in 0..nseq {
                let idx = done[i] + round;
                if idx < prompts[i].len() && reqs.len() < cap {
                    reqs.push(lfm_gpu::StepReq {
                        slot: slots[i],
                        token: prompts[i][idx],
                        pos: idx,
                    });
                    pushed = true;
                }
            }
            if !pushed {
                break;
            }
            round += 1;
        }
        if reqs.is_empty() {
            break;
        }
        for i in 0..nseq {
            let take = reqs.iter().filter(|r| r.slot == slots[i]).count();
            done[i] = (done[i] + take).min(prompts[i].len());
            pos[i] = done[i];
        }
        gpu.step_mixed(&reqs, false)?;
        if done.iter().zip(&prompts).all(|(d, p)| *d >= p.len()) {
            break;
        }
    }
    gpu.finish()?;
    let prefill_s = prefill_started.elapsed().as_secs_f64();
    if std::env::var("LFM_PROFILE").map(|v| v == "1").unwrap_or(false) {
        println!("  -- prefill kernels --");
        for (name, calls, avg_us, total_ms) in gpu.profile_report().iter().take(8) {
            println!("  {name:24} {calls} calls {avg_us:.1} us {total_ms:.1} ms");
        }
    }
    gpu.reset_profile();
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); nseq];
    let started = Instant::now();
    while outs.iter().any(|o| o.len() < tokens) {
        let mut reqs: Vec<lfm_gpu::StepReq> = Vec::new();
        let mut ids: Vec<usize> = Vec::new();
        for i in 0..nseq {
            if outs[i].len() >= tokens {
                continue;
            }
            let token = match outs[i].last() {
                Some(t) => *t,
                None => *prompts[i].last().unwrap(),
            };
            ids.push(i);
            reqs.push(lfm_gpu::StepReq {
                slot: slots[i],
                token,
                pos: pos[i],
            });
        }
        if reqs.is_empty() {
            break;
        }
        gpu.step_mixed(&reqs, true)?;
        for (row, i) in ids.iter().enumerate() {
            let logits = gpu.logits_row(row)?;
            outs[*i].push(lfm_gpu::argmax(&logits));
            pos[*i] += 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    gpu.finish()?;
    let decoded = (outs.iter().map(|o| o.len()).sum::<usize>()) as f64;
    println!(
        "seqs={nseq} prompt={prompt_len}: prefill {total_prompt} tok in {prefill_s:.2} s = {:.1} tok/s; decode {decoded:.0} tok in {elapsed:.2} s = {:.1} tok/s; total {:.1} tok/s",
        total_prompt as f64 / prefill_s.max(1e-9),
        decoded / elapsed.max(1e-9),
        (total_prompt as f64 + decoded) / (prefill_s + elapsed).max(1e-9),
    );
    Ok(())
}

fn gpu_check(model: &PathBuf, ids: &[u32]) -> Result<()> {
    let cpu = lfm_cpu::Lfm3Ref::load(model)?;
    let cpu_logits = cpu.logits(ids)?;
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    for (pos, token) in ids.iter().enumerate() {
        gpu.forward(*token, pos)?;
    }
    let gpu_logits = gpu.logits()?;
    let c: f64 = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = cpu_logits.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = gpu_logits.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    println!("cos(cpu,gpu) {:.6}", c / (na * nb + 1e-12));
    let mut ci: Vec<usize> = (0..cpu_logits.len()).collect();
    ci.sort_by(|a, b| cpu_logits[*b].partial_cmp(&cpu_logits[*a]).unwrap());
    let mut gi: Vec<usize> = (0..gpu_logits.len()).collect();
    gi.sort_by(|a, b| gpu_logits[*b].partial_cmp(&gpu_logits[*a]).unwrap());
    println!("cpu top5 {:?}", &ci[..5]);
    println!("gpu top5 {:?}", &gi[..5]);
    let max_d = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max|dlogit| {max_d:.4}");
    Ok(())
}

fn gemb(path: &PathBuf, tensor: &str, pb: u32, rgemv: u32, threads: u32) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(path)?;
    let info = gguf
        .tensor(tensor)
        .ok_or_else(|| anyhow!("tensor {tensor} not found"))?
        .clone();
    let cols = info.dims[0] as usize;
    let rows = info.dims[1] as usize;
    let bytes = gguf.tensor_bytes(tensor)?;
    let gpu = Gpu::open(0)?;
    let opts = format!("-cl-std=CL3.0 -DRGEMV={rgemv} -DPB={pb}");
    let program = gpu.build(GEMV_CL, &opts)?;
    let single = match info.ggml_type {
        lfm_gguf::GgmlType::Q4_K => Kernel::create(&program, "gemv_q4_k_i8")?,
        lfm_gguf::GgmlType::Q6_K => Kernel::create(&program, "gemv_q6_k_i8")?,
        other => return Err(anyhow!("unsupported {other:?}")),
    };
    let batched = match info.ggml_type {
        lfm_gguf::GgmlType::Q4_K => Kernel::create(&program, "gemv_q4_k_i8_b")?,
        lfm_gguf::GgmlType::Q6_K => Kernel::create(&program, "gemv_q6_k_i8_b")?,
        other => return Err(anyhow!("unsupported {other:?}")),
    };
    let quant = Kernel::create(&program, "quantize_q8")?;
    let w = gpu.upload_bytes(bytes)?;
    let pb = pb as usize;
    let row = make_x(cols);
    let xrow = gpu.upload(&row)?;
    let q0 = gpu.alloc::<i8>(cols)?;
    let s0 = gpu.alloc::<f32>(cols / GROUP)?;
    quantize_on_gpu(&gpu, &quant, &xrow, &q0, &s0, cols)?.wait()?;
    let mut xq = gpu.alloc::<i8>(pb * cols)?;
    let mut xs = gpu.alloc::<f32>(pb * cols / GROUP)?;
    let mut xb = gpu.alloc::<f32>(pb * cols)?;
    let q0mut = q0;
    let s0mut = s0;
    for t in 0..pb {
        unsafe {
            gpu.queue.enqueue_copy_buffer(
                &q0mut,
                &mut xq,
                0,
                t * cols,
                cols,
                &[],
            )?;
            gpu.queue.enqueue_copy_buffer(
                &s0mut,
                &mut xs,
                0,
                t * cols / GROUP * 4,
                cols / GROUP * 4,
                &[],
            )?;
            gpu.queue.enqueue_copy_buffer(
                &xrow,
                &mut xb,
                0,
                t * cols * 4,
                cols * 4,
                &[],
            )?;
        }
    }
    let y = gpu.alloc::<f32>(pb * rows)?;
    let y1 = gpu.alloc::<f32>(rows)?;
    let per_wg = (threads as usize / 64) * rgemv as usize;
    let grid = rows.div_ceil(per_wg) * threads as usize;
    let st = gpu.bench(2, 10, || -> Result<()> {
        unsafe {
            gpu.launch(
                &batched,
                grid,
                threads as usize,
                &[
                    Arg::Mem(w.get()),
                    Arg::Mem(xq.get()),
                    Arg::Mem(xs.get()),
                    Arg::Mem(y.get()),
                    Arg::U32(rows as u32),
                    Arg::U32(cols as u32),
                    Arg::U32(pb as u32),
                    Arg::U32(0),
                ],
            )?;
        }
        Ok(())
    })?;
    unsafe {
        gpu.launch(
            &single,
            grid,
            threads as usize,
            &[
                Arg::Mem(w.get()),
                Arg::Mem(xq.get()),
                Arg::Mem(xs.get()),
                Arg::Mem(y1.get()),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
            ],
        )?;
    }
    let out = gpu.download(&y, pb * rows)?;
    let out1 = gpu.download(&y1, rows)?;
    let max_d = out[..rows]
        .iter()
        .zip(&out1)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
        .fold(0.0f32, f32::max);
    let _ = xb;
    println!(
        "{} {:?} {}x{} pb={} rgemv={}: {} -> {:.1} GB/s effective, {:.1} GB/s raw",
        tensor,
        info.ggml_type,
        rows,
        cols,
        pb,
        rgemv,
        describe(&st),
        fmt_gbps(bytes.len() as u64 * pb as u64, st.median_ms),
        fmt_gbps(bytes.len() as u64, st.median_ms)
    );
    println!("  batch-vs-single token0 max_rel {max_d:.2e}");
    Ok(())
}

fn gpu_prefill(model: &PathBuf, tokens: usize, decode: usize) -> Result<()> {
    let ctx = (tokens + decode + 64).max(1024);
    let mut gpu = lfm_gpu::GpuModel::load(model, ctx)?;
    let ids: Vec<u32> = (0..tokens)
        .map(|i| ((i * 2654435761usize) % 60000) as u32 + 100)
        .collect();
    gpu.forward_batch(
        &ids[..gpu.prefill_batch_size().min(ids.len())],
        0,
        true,
    )?;
    gpu.reset()?;
    let started = Instant::now();
    let mut pos = 0;
    let mut chunks = Vec::new();
    while pos < ids.len() {
        let end = (pos + gpu.prefill_batch_size()).min(ids.len());
        chunks.push((pos, end));
        pos = end;
    }
    for (i, (start, end)) in chunks.iter().enumerate() {
        gpu.forward_batch(&ids[*start..*end], *start, i + 1 == chunks.len())?;
    }
    gpu.finish()?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "prefill {} tok in {:.3} s ({:.1} tok/s)",
        tokens,
        elapsed,
        tokens as f64 / elapsed.max(1e-9)
    );
    let report = gpu.profile_report();
    if !report.is_empty() {
        println!("  -- prefill, per chunk --");
        for (name, calls, avg_us, total_ms) in report {
            println!(
                "  {name:<22} {calls:>6} calls  {avg_us:>8.1} us  {total_ms:>7.2} ms total"
            );
        }
    }
    gpu.reset_profile();
    if decode > 0 {
        gpu.reset_profile();
        let started = Instant::now();
        for i in 0..decode {
            gpu.forward(((i * 7919) % 60000) as u32 + 100, tokens + i)?;
        }
        gpu.finish()?;
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "decode {} tok in {:.3} s ({:.1} tok/s)",
            decode,
            elapsed,
            decode as f64 / elapsed.max(1e-9)
        );
        for (name, calls, avg_us, total_ms) in gpu.profile_report() {
            println!(
                "  {name:<22} {calls:>6} calls  {avg_us:>8.1} us  {total_ms:>7.2} ms/tok"
            );
        }
    }
    Ok(())
}

fn gpu_trace(model: &PathBuf, ids: &[u32]) -> Result<()> {
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    gpu.trace = true;
    gpu.forward_batch(ids, 0, true)?;
    let bat = gpu.traced.clone();
    gpu.traced.clear();
    gpu.trace = false;
    gpu.reset()?;
    gpu.trace = true;
    for (pos, token) in ids.iter().enumerate() {
        gpu.traced.clear();
        gpu.forward(*token, pos)?;
        if pos + 1 == ids.len() {
            break;
        }
    }
    let seq = gpu.traced.clone();
    for (i, (a, b)) in bat.iter().zip(seq.iter()).enumerate() {
        println!("layer {i}: cos {:.6}", cosine(a, b));
    }
    Ok(())
}

fn gpu_cmp(model: &PathBuf, ids: &[u32], batch: usize) -> Result<()> {
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    for (pos, token) in ids.iter().enumerate() {
        gpu.forward(*token, pos)?;
    }
    let seq = gpu.logits()?;
    let last = ids.len().saturating_sub(1);
    let seq_state = gpu.debug_state(2, last)?;
    let seq_conv0 = gpu.debug_state(0, 0)?.0;
    let seq_x = gpu.debug_x(false, 0)?;
    gpu.reset()?;
    let mut pos = 0;
    while pos < ids.len() {
        let end = (pos + batch).min(ids.len());
        gpu.forward_batch(&ids[pos..end], pos, end == ids.len())?;
        pos = end;
    }
    let bat = gpu.logits()?;
    let bat_state = gpu.debug_state(2, last)?;
    let bat_conv0 = gpu.debug_state(0, 0)?.0;
    let btok = (ids.len() - 1) % batch.max(1);
    let bat_x = gpu.debug_x(true, btok)?;
    println!("x cos {:.6}", cosine(&seq_x, &bat_x));
    println!("x max|d| {:.4}", seq_x.iter().zip(&bat_x).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max));
    println!("kv0 cos {:.6}", cosine(&seq_state.1, &bat_state.1));
    println!("kv0 max|d| {:.4}", seq_state.1.iter().zip(&bat_state.1).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max));
    println!("conv0 cos {:.6}", cosine(&seq_conv0, &bat_conv0));
    println!("conv0 max|d| {:.4}", seq_conv0.iter().zip(&bat_conv0).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max));
    println!("batch cos {:.6}", cosine(&seq, &bat));
    let max_d = seq
        .iter()
        .zip(&bat)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("batch max|dlogit| {max_d:.4}");
    let mut si: Vec<usize> = (0..seq.len()).collect();
    si.sort_by(|a, b| seq[*b].partial_cmp(&seq[*a]).unwrap());
    let mut bi: Vec<usize> = (0..bat.len()).collect();
    bi.sort_by(|a, b| bat[*b].partial_cmp(&bat[*a]).unwrap());
    println!("seq top5 {:?}", &si[..5]);
    println!("bat top5 {:?}", &bi[..5]);
    Ok(())
}

fn sg_probe() -> Result<()> {
    let gpu = Gpu::open(0)?;
    println!("device: {}", gpu.caps.name);
    let src = r#"
__kernel void probe(__global uint *out) {
    uint sgs = get_sub_group_size();
    uint nsg = get_num_sub_groups();
    uint lid = get_local_id(0);
    uint sgid = get_sub_group_id();
    uint slane = get_sub_group_local_id();
    float s1 = sub_group_reduce_add(1.0f);
    float s2 = sub_group_reduce_add(get_global_id(0) == 1u ? 1.0f : 0.0f);
    if (get_global_id(0) == 0u) {
        out[0] = sgs;
        out[1] = nsg;
        out[2] = lid;
        out[3] = sgid;
        out[4] = slane;
        out[5] = (uint)s1;
        out[6] = (uint)s2;
    }
}
"#;
    let program = gpu.build(src, "-cl-std=CL3.0")?;
    let kernel = Kernel::create(&program, "probe")?;
    let out = gpu.alloc::<u32>(8)?;
    for wg in [64usize, 128, 256] {
        unsafe {
            gpu.launch(&kernel, wg, wg, &[Arg::Mem(out.get())])?;
        }
        gpu.queue.finish()?;
        let v = gpu.download(&out, 7)?;
        println!(
            "wg={wg} sgs={} nsg={} lid={} sgid={} slane={} sum1={} sum_lane1={}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6]
        );
    }
    Ok(())
}

fn gpu_tp(
    model: &PathBuf,
    tokenizer: &PathBuf,
    prompt: &str,
    tokens: usize,
    devices: &str,
    ctx: usize,
) -> Result<()> {
    let devices: Vec<usize> = devices
        .split('/')
        .filter_map(|d| d.parse::<usize>().ok())
        .collect();
    let mut tp = lfm_gpu::TpModel::load(model, ctx, &devices)?;
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let out = lfm_app::Generate::new(&mut tp, &tok).run(prompt, tokens)?;
    println!("tp     ids: {:?}", out.ids);
    println!("tp     text: {}", out.text);
    let mut single = lfm_gpu::GpuModel::load(model, ctx)?;
    let base = lfm_app::Generate::new(&mut single, &tok).run(prompt, tokens)?;
    println!("single ids: {:?}", base.ids);
    println!("equal: {}", out.ids == base.ids);
    let ids = tok.encode(prompt, false)?;
    let slot = tp.create_sequence()?;
    let _ = tp.prefill(slot, &ids, true)?;
    let tp_logits = tp.logits()?;
    tp.free_sequence(slot)?;
    let slot = single.create_sequence()?;
    let _ = single.prefill(slot, &ids, true)?;
    let one_logits = single.logits()?;
    single.free_sequence(slot)?;
    println!("logits cos {:.6}", cosine(&tp_logits, &one_logits));
    Ok(())
}

fn gpu_pp(
    model: &PathBuf,
    tokenizer: &PathBuf,
    prompt: &str,
    tokens: usize,
    tiers: &str,
    ctx: usize,
) -> Result<()> {
    let percents: Vec<usize> = tiers
        .split('/')
        .map(|p| p.parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow!("bad tiers {tiers}"))?;
    let plan: Vec<(usize, usize)> = percents
        .iter()
        .enumerate()
        .map(|(device, &percent)| (device, percent))
        .collect();
    let mut pipe = lfm_gpu::PipelineModel::load_tiers(model, ctx, &plan)?;
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let out = lfm_app::Generate::new(&mut pipe, &tok).run(prompt, tokens)?;
    println!("pp     ids: {:?}", out.ids);
    println!("pp     text: {}", out.text);
    let mut single = lfm_gpu::GpuModel::load(model, ctx)?;
    let base = lfm_app::Generate::new(&mut single, &tok).run(prompt, tokens)?;
    println!("single ids: {:?}", base.ids);
    println!("equal: {}", out.ids == base.ids);
    let ids = tok.encode(prompt, false)?;
    let slot = pipe.create_sequence()?;
    let _ = pipe.prefill(slot, &ids, true)?;
    let pp_logits = pipe.logits()?;
    pipe.free_sequence(slot)?;
    let slot = single.create_sequence()?;
    let _ = single.prefill(slot, &ids, true)?;
    let one_logits = single.logits()?;
    single.free_sequence(slot)?;
    println!("logits cos {:.6}", cosine(&pp_logits, &one_logits));
    let top = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap());
        idx[..5].to_vec()
    };
    println!(
        "top5 equal: {} {:?} {:?}",
        top(&pp_logits) == top(&one_logits),
        top(&pp_logits),
        top(&one_logits)
    );
    Ok(())
}

fn gpu_port(
    model: &PathBuf,
    tokenizer: &PathBuf,
    prompt: &str,
    tokens: usize,
    ctx: usize,
) -> Result<()> {
    let mut gpu = lfm_gpu::GpuModel::load(model, ctx)?;
    let tok = lfm_tokenizer::LfmTokenizer::from_file(tokenizer)?;
    let out = lfm_app::Generate::new(&mut gpu, &tok).run(prompt, tokens)?;
    println!("ids: {:?}", out.ids);
    println!("finish: {:?}", out.finish);
    println!("text: {}", out.text);
    Ok(())
}

fn gpu_embed(model: &PathBuf, id: u32) -> Result<()> {
    let cpu = lfm_cpu::Lfm3Ref::load(model)?;
    let mut gpu = lfm_gpu::GpuModel::load(model, 2048)?;
    let g = gpu.debug_embed(id)?;
    let c = cpu
        .tensors
        .get("token_embd.weight")
        .map(|t| t.row(id as usize).to_vec())
        .ok_or_else(|| anyhow!("no embedding tensor"))?;
    println!("embed cos {:.6}", cosine(&c, &g));
    println!("cpu[0..6] {:?}", &c[..6]);
    println!("gpu[0..6] {:?}", &g[..6]);
    let n = gpu.debug_norm("output_norm.weight")?;
    let w = cpu
        .tensors
        .get("output_norm.weight")
        .map(|t| t.data.clone())
        .ok_or_else(|| anyhow!("no output_norm"))?;
    let var = c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32;
    let scale = 1.0 / (var + cpu.cfg.norm_eps).sqrt();
    let cn: Vec<f32> = c.iter().zip(&w).map(|(x, wi)| x * scale * wi).collect();
    println!("norm cos {:.6}", cosine(&cn, &n));
    let ones = vec![1.0f32; cpu.cfg.hidden_size];
    let normed_ones = gpu.debug_norm_raw(&ones, "blk.0.attn_norm.weight")?;
    let w = cpu
        .tensors
        .get("blk.0.attn_norm.weight")
        .map(|t| t.data.clone())
        .ok_or_else(|| anyhow!("no norm weight"))?;
    println!("ones-norm cos {:.6}", cosine(&w, &normed_ones));
    println!("ones-norm[0..3] {:?} vs w {:?}", &normed_ones[..3], &w[..3]);
    let h_cpu = cpu.norm_named(&c, "blk.0.attn_norm.weight")?;
    let op_cpu = cpu.conv_op(0, &h_cpu)?;
    let h_gpu = gpu.debug_norm("blk.0.attn_norm.weight")?;
    let (bcx_gpu, op_gpu) = gpu.debug_conv(0, &h_gpu)?;
    let bcx_cpu = {
        let t = cpu
            .tensors
            .get("blk.0.shortconv.in_proj.weight")
            .ok_or_else(|| anyhow!("no in_proj"))?;
        (0..3 * cpu.cfg.hidden_size)
            .map(|o| {
                let row = t.row(o);
                row.iter().zip(&h_cpu).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect::<Vec<f32>>()
    };
    let cw_gpu = gpu.debug_weight_f32("blk.0.shortconv.conv.weight", 14)?;
    let cw_cpu = cpu
        .tensors
        .get("blk.0.shortconv.conv.weight")
        .map(|t| t.data[..14].to_vec())
        .ok_or_else(|| anyhow!("no conv weight"))?;
    println!("conv w cpu {:?}", &cw_cpu[..6]);
    println!("conv w gpu {:?}", &cw_gpu[..6]);
    println!("bcx |gpu| {:?}", &bcx_gpu[..3].iter().map(|v| v.abs()).collect::<Vec<f32>>());
    println!("bcx |cpu| {:?}", &bcx_cpu[..3].iter().map(|v| v.abs()).collect::<Vec<f32>>());
    println!("in_proj cos {:.6}", cosine(&bcx_cpu, &bcx_gpu));
    println!("cpu bcx[0..4] {:?}", &bcx_cpu[..4]);
    println!("gpu bcx[0..4] {:?}", &bcx_gpu[..4]);
    println!("h cos {:.6}", cosine(&h_cpu, &h_gpu));
    println!("h cpu[0..4] {:?}", &h_cpu[..4]);
    println!("h gpu[0..4] {:?}", &h_gpu[..4]);
    println!("conv op cos {:.6}", cosine(&op_cpu, &op_gpu));
    println!("cpu op[0..4] {:?}", &op_cpu[..4]);
    println!("gpu op[0..4] {:?}", &op_gpu[..4]);
    let (fh, fu, fa, fo) = cpu.ffn_debug(0, &c)?;
    let (gh, gu, ga, go) = gpu.debug_ffn(0, &c)?;
    println!("ffn h cos {:.6}", cosine(&fh, &gh));
    println!("ffn up cos {:.6}", cosine(&fu, &gu));
    let ref_act: Vec<f32> = fu
        .iter()
        .zip(&fa)
        .map(|(u, g)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    println!("ffn act cos {:.6}", cosine(&ref_act, &ga));
    println!("ffn out cos {:.6}", cosine(&fo, &go));
    println!("ffn up cpu[0..3] {:?}", &fu[..3]);
    println!("ffn up gpu[0..3] {:?}", &gu[..3]);
    let rows = 64usize;
    let gl = gpu.debug_lmhead(rows)?;
    let emb = cpu
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| anyhow!("no embedding"))?;
    let mut cl = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = emb.row(r);
        let dot: f32 = row.iter().zip(&n).map(|(a, b)| a * b).sum();
        cl.push(dot);
    }
    let xq = make_x(cpu.cfg.hidden_size);
    let (gq, gs) = gpu.debug_quant(&xq)?;
    let (rq, rs) = quantize_ref(&xq, GROUP);
    let dq = gq.iter().zip(&rq).filter(|(a, b)| a != b).count();
    let ds = gs.iter().zip(&rs).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("quant diff {dq}/{} xs max|d| {ds:.3e}", gq.len());
    println!("  gs[0..4] {:?}", &gs[..4]);
    println!("  rs[0..4] {:?}", &rs[..4]);
    let n = cpu.cfg.hidden_size;
    let op = make_x(n);
    let post = make_x(n);
    let res = make_x(n);
    let next_w = make_x(n);
    let (gx, gq, gs) = gpu.debug_norm_add_quant(&op, &post, &res, &next_w)?;
    let ss: f32 = op.iter().map(|v| v * v).sum();
    let sc = 1.0 / (ss / n as f32 + 1e-5).sqrt();
    let v: Vec<f32> = op.iter().zip(&post).zip(&res).map(|((a, b), c)| a * sc * b + c).collect();
    let ss2: f32 = v.iter().map(|x| x * x).sum();
    let sc2 = 1.0 / (ss2 / n as f32 + 1e-5).sqrt();
    let w: Vec<f32> = v.iter().zip(&next_w).map(|(a, b)| a * sc2 * b).collect();
    let (rq, rs) = quantize_ref(&w, GROUP);
    let dq = gq.iter().zip(&rq).filter(|(a, b)| a != b).count();
    let ds = gs.iter().zip(&rs).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("naq x cos {:.6}, quant diff {dq}/{}, xs max|d| {ds:.3e}", cosine(&v, &gx), gq.len());
    println!("ref scale {sc:.8} ss {ss:.4}");
    for i in 0..4 {
        let k = (gx[i] - res[i]) / (op[i] * post[i]);
        println!("  k[{i}] {k:.8} op {:.5} post {:.5} res {:.5}", op[i], post[i], res[i]);
    }
    let mut shown = 0;
    for i in 0..n {
        if (gx[i] - v[i]).abs() > 1e-6 * v[i].abs().max(1.0) {
            println!("  x[{i}] gpu {:.6} ref {:.6}", gx[i], v[i]);
            shown += 1;
            if shown == 6 {
                break;
            }
        }
    }
    let ffn = cpu.cfg.ffn_size;
    let ug = make_x(2 * ffn);
    let (sq, ss) = gpu.debug_silu_quant(&ug, ffn)?;
    let act: Vec<f32> = (0..ffn)
        .map(|i| {
            let g = ug[i];
            (g / (1.0 + (-g).exp())) * ug[ffn + i]
        })
        .collect();
    let (rq, rs) = quantize_ref(&act, GROUP);
    let dq = sq.iter().zip(&rq).filter(|(a, b)| a != b).count();
    let ds = ss.iter().zip(&rs).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("silu quant diff {dq}/{ffn}, xs max|d| {ds:.3e}");
    println!("lmhead cos {:.6}", cosine(&cl, &gl));
    println!("cpu dot[0..4] {:?}", &cl[..4]);
    println!("gpu dot[0..4] {:?}", &gl[..4]);
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_state() -> TuiState {
        TuiState {
            history: Vec::new(),
            assistant: String::new(),
            thinking: String::new(),
            tools: Vec::new(),
            input: String::new(),
            input_history: InputHistory::new(),
            status: String::new(),
            scroll: 0,
            pending: Vec::new(),
        }
    }

    fn chat_state(messages: usize) -> TuiState {
        let mut state = empty_state();
        for i in 0..messages {
            state
                .history
                .push(("you".to_string(), format!("m{i}")));
        }
        state
    }

    #[test]
    fn scroll_pgup_pgdn_respects_bounds() {
        assert_eq!(scroll_after_pgup(0, 5), 5);
        assert_eq!(scroll_after_pgup(3, 5), 8);
        assert_eq!(scroll_after_pgup(usize::MAX, 5), usize::MAX);
        assert_eq!(scroll_after_pgdn(8, 5), 3);
        assert_eq!(scroll_after_pgdn(2, 5), 0);
        assert_eq!(scroll_after_pgdn(0, 5), 0);
        assert_eq!(scroll_after_pgup(0, 0), 1);
        assert_eq!(scroll_after_pgdn(1, 0), 0);
    }

    #[test]
    fn scroll_home_end() {
        assert_eq!(scroll_home(40, 7), 33);
        assert_eq!(scroll_home(3, 7), 0);
        assert_eq!(scroll_home(0, 0), 0);
        assert_eq!(scroll_end(), 0);
    }

    #[test]
    fn input_history_empty_returns_none() {
        let mut hist = InputHistory::new();
        assert_eq!(hist.up("draft"), None);
        assert_eq!(hist.down(), None);
        assert_eq!(hist.entries.len(), 0);
    }

    #[test]
    fn input_history_up_down_walk() {
        let mut hist = InputHistory::new();
        hist.push("a");
        hist.push("b");
        assert_eq!(hist.pos, 2);
        assert_eq!(hist.up("draft"), Some("b".to_string()));
        assert_eq!(hist.up("draft"), Some("a".to_string()));
        assert_eq!(hist.up("draft"), Some("a".to_string()));
        assert_eq!(hist.down(), Some("b".to_string()));
        assert_eq!(hist.down(), Some("draft".to_string()));
        assert_eq!(hist.down(), None);
        assert_eq!(hist.down(), None);
    }

    #[test]
    fn input_history_skips_empty_and_consecutive_duplicates() {
        let mut hist = InputHistory::new();
        hist.push("a");
        hist.push("a");
        hist.push("  ");
        hist.push("b");
        hist.push("a");
        assert_eq!(hist.entries, vec!["a", "b", "a"]);
        assert_eq!(hist.pos, 3);
    }

    #[test]
    fn input_history_up_after_typing_new_line() {
        let mut hist = InputHistory::new();
        hist.push("one");
        hist.push("two");
        hist.reset();
        assert_eq!(hist.up(""), Some("two".to_string()));
        hist.push("three");
        assert_eq!(hist.pos, 3);
        assert_eq!(hist.up(""), Some("three".to_string()));
    }

    #[test]
    fn visible_lines_small_window_keeps_tail() {
        let mut state = empty_state();
        state
            .history
            .push(("ai".to_string(), "tail".to_string()));
        let view = visible_lines(&state, 3, 40);
        assert_eq!(view.area, 1);
        assert_eq!(view.body.len(), 1);
        assert_eq!(view.total, 2);
        assert_eq!(view.scroll, 0);
        assert!(view.body[0].is_empty());
    }

    #[test]
    fn visible_lines_large_window_pads() {
        let state = chat_state(3);
        let view = visible_lines(&state, 200, 80);
        assert_eq!(view.area, 197);
        assert_eq!(view.body.len(), 197);
        assert!(view.body[0].starts_with("you> m0"));
        assert_eq!(view.scroll, 0);
    }

    #[test]
    fn visible_lines_scrolls_up() {
        let mut state = chat_state(20);
        let last = visible_lines(&state, 10, 40);
        assert_eq!(last.area, 7);
        assert_eq!(last.total, 40);
        assert!(last.body.iter().any(|l| l.contains("m19")));
        state.scroll = 3;
        let scrolled = visible_lines(&state, 10, 40);
        assert_eq!(scrolled.scroll, 3);
        assert_eq!(scrolled.body[0], "you> m15");
        state.scroll = usize::MAX;
        let top = visible_lines(&state, 10, 40);
        assert_eq!(top.scroll, 33);
        assert_eq!(top.body[0], "you> m0");
    }

    #[test]
    fn visible_lines_renders_tools_after_answer() {
        let mut state = empty_state();
        state.assistant = "ans".to_string();
        state.tools.push(("calc".to_string(), "{\"a\":1}".to_string()));
        let view = visible_lines(&state, 12, 40);
        let text = view.body.join("\n");
        assert!(text.contains("ai>  ans"));
        assert!(text.contains("[tool] calc"));
        assert!(text.contains("{\"a\":1}"));
    }

    #[test]
    fn visible_lines_zero_size_does_not_panic() {
        let mut state = chat_state(5);
        state.scroll = 10;
        let view = visible_lines(&state, 0, 0);
        assert_eq!(view.body.len(), 1);
        assert_eq!(view.area, 1);
        tui_render(&state, 0, 0);
    }

    #[test]
    fn format_tool_call_wraps_long_args() {
        let args = "x".repeat(500);
        let lines = format_tool_call("fn", &args, 40);
        assert_eq!(lines[0], "[tool] fn");
        assert!(lines.len() > 2);
        assert!(lines.iter().all(|l| l.chars().count() <= 40));
        assert!(lines[1].starts_with("  "));
        let body_len: usize = lines[1..]
            .iter()
            .map(|l| l.trim_start().chars().count())
            .sum();
        assert_eq!(body_len, 500);
    }

    #[test]
    fn format_tool_call_truncates_huge_args() {
        let args = "y".repeat(TOOL_ARGS_MAX + 500);
        let lines = format_tool_call("fn", &args, 60);
        let joined: String = lines.join("");
        assert!(joined.ends_with('…'));
        assert!(joined.chars().count() <= TOOL_ARGS_MAX + 200);
    }

    #[test]
    fn format_tool_call_edge_sizes() {
        let lines = format_tool_call("", "", 0);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].is_empty());
        let lines = format_tool_call("fn", "   ", 10);
        assert_eq!(lines, vec!["[tool] fn".to_string()]);
        let very_long_name = "n".repeat(200);
        let lines = format_tool_call(&very_long_name, "", 20);
        assert_eq!(lines[0].chars().count(), 20);
    }

    #[test]
    fn parse_escape_recognizes_navigation_keys() {
        assert_eq!(parse_escape("[A"), Key::Up);
        assert_eq!(parse_escape("[B"), Key::Down);
        assert_eq!(parse_escape("[5~"), Key::PgUp);
        assert_eq!(parse_escape("[6~"), Key::PgDn);
        assert_eq!(parse_escape("[H"), Key::Home);
        assert_eq!(parse_escape("[1~"), Key::Home);
        assert_eq!(parse_escape("[F"), Key::End);
        assert_eq!(parse_escape("[4~"), Key::End);
        assert_eq!(parse_escape("OH"), Key::Home);
        assert_eq!(parse_escape("OF"), Key::End);
        assert_eq!(parse_escape("[C"), Key::Other);
        assert_eq!(parse_escape("[D"), Key::Other);
        assert_eq!(parse_escape(""), Key::Other);
    }

    #[test]
    fn status_scroll_indicator_appended_on_render() {
        let mut state = empty_state();
        state.status = "idle".to_string();
        state.scroll = 5;
        let view = visible_lines(&state, 10, 40);
        assert_eq!(view.scroll, 0);
        state.history = (0..30)
            .map(|i| ("you".to_string(), format!("m{i}")))
            .collect();
        let view = visible_lines(&state, 10, 40);
        assert_eq!(view.scroll, 5);
        tui_render(&state, 10, 40);
    }
}

fn prepack_q4_key(src: &[u8]) -> Vec<u8> {
    assert!(src.len() % 144 == 0);
    let blocks = src.len() / 144;
    let mut out = vec![0u8; blocks * 148];
    for b in 0..blocks {
        let s = &src[b * 144..(b + 1) * 144];
        let d = &mut out[b * 148..(b + 1) * 148];
        d[..4].copy_from_slice(&s[..4]);
        for j in 0..8 {
            let q = &s[4..16];
            let jm = j & 3;
            let h = j >> 2;
            let a = q[jm];
            let bq = q[jm + 4];
            let c = q[jm + 8];
            let (sc, m) = if h == 0 {
                (a & 63, bq & 63)
            } else {
                ((c & 15) | ((a >> 6) << 4), (c >> 4) | ((bq >> 6) << 4))
            };
            d[4 + 2 * j] = sc;
            d[4 + 2 * j + 1] = m;
        }
        d[20..148].copy_from_slice(&s[16..144]);
    }
    out
}

fn gemm_launch(
    gpu: &Gpu,
    program: &Program,
    mode: &str,
    rows: usize,
    cols: usize,
    ntok: usize,
    wbytes: &[u8],
    xq: &[i8],
    xs: &[f32],
) -> Result<(Vec<f32>, Stats)> {
    let nsg = (64 / gpu.caps.sub_group_size.max(1)).max(1);
    let (kernel, global, local) = match mode {
        "b8" => (
            Kernel::create(program, "gemm_q4_k_i8_b8")?,
            rows.div_ceil(64) * 64,
            64usize,
        ),
        "lds17" => (
            Kernel::create(program, "gemm_q4_k_i8_lds17")?,
            rows.div_ceil(64) * ntok.div_ceil(32) * 512,
            512usize,
        ),
        _ => (
            Kernel::create(program, "gemm_q6_k_i8")?,
            rows.div_ceil(2 * nsg) * 64,
            64usize,
        ),
    };
    let w = gpu.upload_bytes(wbytes)?;
    let xqb = gpu.upload(xq)?;
    let xsb = gpu.upload(xs)?;
    let yb = gpu.alloc::<f32>(rows * ntok)?;
    let launch = || -> Result<()> {
        unsafe {
            gpu.launch(
                &kernel,
                global,
                local,
                &[
                    Arg::Mem(w.get()),
                    Arg::Mem(xqb.get()),
                    Arg::Mem(xsb.get()),
                    Arg::Mem(yb.get()),
                    Arg::U32(rows as u32),
                    Arg::U32(cols as u32),
                    Arg::U32(ntok as u32),
                ],
            )?;
        }
        Ok(())
    };
    launch()?;
    let st = gpu.bench(2, 10, launch)?;
    Ok((gpu.download(&yb, rows * ntok)?, st))
}

fn gemm_one(
    gpu: &Gpu,
    program: &Program,
    name: &str,
    ty: lfm_gguf::GgmlType,
    rows: usize,
    cols: usize,
    bytes: &[u8],
    ntok: usize,
    mode: &str,
    check_rows: usize,
    verbose: bool,
) -> Result<f32> {
    let mut x = Vec::with_capacity(ntok * cols);
    for t in 0..ntok {
        for (i, v) in make_x(cols).into_iter().enumerate() {
            x.push(v * (1.0 + t as f32 * 0.03) * if (i + t) % 3 == 0 { -1.0 } else { 1.0 });
        }
    }
    let mut xq = Vec::with_capacity(ntok * cols);
    let mut xs = Vec::with_capacity(ntok * cols / GROUP);
    for t in 0..ntok {
        let (q, sc) = quantize_ref(&x[t * cols..(t + 1) * cols], GROUP);
        xq.extend_from_slice(&q);
        xs.extend_from_slice(&sc);
    }
    let wbytes = if ty == lfm_gguf::GgmlType::Q4_K {
        prepack_q4_key(bytes)
    } else {
        bytes.to_vec()
    };
    let (out, st) = gemm_launch(gpu, program, mode, rows, cols, ntok, &wbytes, &xq, &xs)?;
    let row_bytes = bytes.len() / rows;
    let mut max_rel = 0.0f32;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
    for r in 0..check_rows.min(rows) {
        let slice = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for t in 0..ntok {
            let g = out[t * rows + r];
            let refv = dot_i8(
                ty,
                slice,
                &xq[t * cols..(t + 1) * cols],
                &xs[t * cols / GROUP..(t + 1) * cols / GROUP],
            )?;
            let rel = (g - refv).abs() / refv.abs().max(1e-3);
            if rel > max_rel {
                max_rel = rel;
                worst = (r, t, g, refv);
            }
        }
    }
    if verbose {
        println!(
            "{mode} {name} {ty:?} {rows}x{cols} ntok={ntok}: {}",
            describe(&st)
        );
        println!(
            "  rows {} x ntok {ntok}: max_rel {:.2e} (worst row {} tok {} gpu {:.4} ref {:.4})",
            check_rows.min(rows), max_rel, worst.0, worst.1, worst.2, worst.3
        );
    }
    Ok(max_rel)
}

fn gemm_cmp(
    model: &PathBuf,
    tensor: &str,
    ntok: usize,
    mode: &str,
    check_rows: usize,
) -> Result<()> {
    let gguf = lfm_gguf::GgufFile::open(model)?;
    let gpu = Gpu::open(0)?;
    let program = gpu.build(LFM_KERNELS, "-cl-std=CL3.0 -DRGEMV=4 -DRGEMV_B=64 -DPB=6")?;
    if tensor != "*" {
        let info = gguf
            .tensor(tensor)
            .ok_or_else(|| anyhow!("tensor {tensor} not found"))?
            .clone();
        if mode != "q6" && info.ggml_type != lfm_gguf::GgmlType::Q4_K {
            return Err(anyhow!("mode {mode} needs Q4_K weights"));
        }
        if mode == "q6" && info.ggml_type != lfm_gguf::GgmlType::Q6_K {
            return Err(anyhow!("mode q6 needs Q6_K weights"));
        }
        let bytes = gguf.tensor_bytes(tensor)?;
        gemm_one(
            &gpu,
            &program,
            tensor,
            info.ggml_type,
            info.dims[1] as usize,
            info.dims[0] as usize,
            bytes,
            ntok,
            mode,
            check_rows,
            true,
        )?;
        return Ok(());
    }
    let mut worst = (String::new(), 0.0f32);
    for t in gguf.tensors() {
        if t.dims.len() != 2 {
            continue;
        }
        let m = match t.ggml_type {
            lfm_gguf::GgmlType::Q4_K => "lds17",
            lfm_gguf::GgmlType::Q6_K => "q6",
            _ => continue,
        };
        if mode != "all" && mode != m {
            continue;
        }
        let bytes = gguf.tensor_bytes(&t.name)?;
        let rel = gemm_one(
            &gpu,
            &program,
            &t.name,
            t.ggml_type,
            t.dims[1] as usize,
            t.dims[0] as usize,
            bytes,
            ntok,
            m,
            check_rows,
            false,
        )?;
        if rel > worst.1 {
            worst = (format!("{} {}({})", t.name, m, t.ggml_type == lfm_gguf::GgmlType::Q6_K), rel);
        }
    }
    println!("worst: {} max_rel {:.2e}", worst.0, worst.1);
    Ok(())
}

fn dot_rate(iters: u32) -> Result<()> {
    let gpu = Gpu::open(0)?;
    let src = r#"
__kernel void dot_rate(__global const char *src, __global int *out, uint iters) {
    uint gid = get_global_id(0);
    char4 a0 = vload4(0, src + gid * 16);
    char4 a1 = vload4(0, src + gid * 16 + 4);
    char4 b0 = vload4(0, src + gid * 16 + 8);
    char4 b1 = vload4(0, src + gid * 16 + 12);
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    for (uint i = 0; i < iters; ++i) {
        c0 += dot(a0, b0);
        c1 += dot(a1, b1);
        c2 += dot(a0, b1);
        c3 += dot(a1, b0);
        c4 += dot(a0, b0);
        c5 += dot(a1, b1);
        c6 += dot(a0, b1);
        c7 += dot(a1, b0);
        a0 = (char4)((c0 & 3), a0.y, a0.z, a0.w);
        b0 = (char4)(b0.x, (c1 & 3), b0.z, b0.w);
    }
    out[gid] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}
__kernel void f16_rate(__global const char *src, __global float *out, uint iters) {
    uint gid = get_global_id(0);
    half4 a0 = vload4(0, (__global const half *)(src + gid * 16));
    half4 b0 = vload4(0, (__global const half *)(src + gid * 16 + 8));
    half4 a1 = vload4(4, (__global const half *)(src + gid * 16));
    half4 b1 = vload4(4, (__global const half *)(src + gid * 16 + 8));
    half4 c0 = (half4)(0), c1 = (half4)(0), c2 = (half4)(0), c3 = (half4)(0);
    half4 c4 = (half4)(0), c5 = (half4)(0), c6 = (half4)(0), c7 = (half4)(0);
    for (uint i = 0; i < iters; ++i) {
        c0 = fma(a0, b0, c0);
        c1 = fma(a1, b1, c1);
        c2 = fma(a0, b1, c2);
        c3 = fma(a1, b0, c3);
        c4 = fma(a0, b0, c4);
        c5 = fma(a1, b1, c5);
        c6 = fma(a0, b1, c6);
        c7 = fma(a1, b0, c7);
    }
    float4 r = convert_float4(c0 + c1 + c2 + c3) + convert_float4(c4 + c5 + c6 + c7);
    out[gid] = r.x + r.y + r.z + r.w;
}
"#;
    let program = gpu.build(src, "-cl-std=CL3.0")?;
    let grid = 12 * 4 * 64 * 4;
    let bytes = vec![7u8; grid * 16];
    let sbuf = gpu.upload_bytes(&bytes)?;
    for name in ["dot_rate", "f16_rate"] {
        let k = Kernel::create(&program, name)?;
        let out_i = gpu.alloc::<i32>(grid)?;
        let out_f = gpu.alloc::<f32>(grid)?;
        let st = gpu.bench(2, 5, || -> Result<()> {
            unsafe {
                if name == "dot_rate" {
                    gpu.launch(&k, grid, 64, &[Arg::Mem(sbuf.get()), Arg::Mem(out_i.get()), Arg::U32(iters)])?;
                } else {
                    gpu.launch(&k, grid, 64, &[Arg::Mem(sbuf.get()), Arg::Mem(out_f.get()), Arg::U32(iters)])?;
                }
            }
            Ok(())
        })?;
        let macs = grid as f64 * iters as f64 * 8.0 * 4.0;
        println!(
            "{name}: {} -> {:.2} G MAC/s ({:.2} TFLOP/s)",
            describe(&st),
            macs / (st.median_ms / 1e3) / 1e9,
            macs * 2.0 / (st.median_ms / 1e3) / 1e12
        );
    }
    Ok(())
}

fn kernel_bin(name: &str, out: &std::path::Path) -> Result<()> {
    let gpu = Gpu::open(0)?;
    let opts = "-cl-std=CL3.0 -DRGEMV=5 -DRGEMV_B=4 -DPB=6";
    let program = gpu.build(LFM_KERNELS, opts)?;
    let _k = Kernel::create(&program, name)?;
    let sizes = program.get_binary_sizes()?;
    let bins = program.get_binaries()?;
    let mut wrote = 0usize;
    for (i, (size, bin)) in sizes.iter().zip(bins.iter()).enumerate() {
        if *size == 0 || bin.is_empty() {
            continue;
        }
        let mut body = bin.clone();
        body.truncate(*size);
        let path = if sizes.len() == 1 {
            out.to_path_buf()
        } else {
            out.with_extension(format!("bin{i}"))
        };
        std::fs::write(&path, &body)?;
        println!("wrote {} ({} bytes)", path.display(), body.len());
        wrote += 1;
    }
    if wrote == 0 {
        return Err(anyhow!("driver returned no program binaries"));
    }
    Ok(())
}
