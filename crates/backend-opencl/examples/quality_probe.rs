//! Greedy probe of the OpenCL backend, for A/B across weight formats.
//!
//! ```text
//! LFM_PREPACK=0 cargo run --release -p lfm-gpu --example quality_probe -- \
//!     models/LFM3-1.7B-Q4_K_M.gguf 597 5205 302 3980 355
//! ```
//!
//! Prints the greedy continuation so the same prompt can be compared between
//! native Q4_K weights, the int8/32 prepack, and the CPU reference.

use std::path::Path;

use lfm_gpu::GpuModel;

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("model path");
    let prompt: Vec<u32> = args.map(|value| value.parse().expect("token id")).collect();
    assert!(!prompt.is_empty(), "prompt token ids required");
    let steps: usize = std::env::var("PROBE_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let ctx: usize = std::env::var("PROBE_CTX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);

    let mut gpu = GpuModel::load(Path::new(&model), ctx).expect("load");
    let t0 = std::time::Instant::now();
    gpu.forward_batch(&prompt, 0, true).expect("prefill");
    let prefill = t0.elapsed().as_secs_f64();
    let mut logits = gpu.logits().expect("logits");
    let mut out = Vec::with_capacity(steps);
    let mut pos = prompt.len();
    for _ in 0..steps {
        let token = argmax(&logits);
        out.push(token);
        gpu.forward_batch(&[token], pos, true).expect("decode");
        logits = gpu.logits().expect("logits");
        pos += 1;
    }
    println!(
        "FIRST {}",
        out.first().copied().unwrap_or_default()
    );
    let decode = t0.elapsed().as_secs_f64() - prefill;
    println!(
        "TOKENS {}",
        out.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
    );
    println!(
        "TIME prefill {:.1} ms, decode {:.2} ms/token, {:.2} tok/s",
        prefill * 1e3,
        decode * 1e3 / steps as f64,
        steps as f64 / decode
    );
}
