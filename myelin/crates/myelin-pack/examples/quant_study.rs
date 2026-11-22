//! Compare 4-bit weight schemes on real LFM3 matrices.
//!
//! ```text
//! cargo run --release -p myelin-pack --example quant_study -- ../models/LFM3-1.7B-Q4_K_M.gguf
//! ```
//!
//! Answers a concrete question: does a Hadamard rotation along K lift the
//! per-row int4 cosine enough, or do we also need per-32 block scales?

use std::path::Path;

use myelin_pack::gguf_source;

fn wht(values: &mut [f32]) {
    let n = values.len();
    let mut len = 1;
    while len < n {
        let mut start = 0;
        while start < n {
            for i in start..start + len {
                let a = values[i];
                let b = values[i + len];
                values[i] = a + b;
                values[i + len] = a - b;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

/// Hadamard on the largest power-of-two chunk that divides `len` (2048 for the
/// model width, 512 for the 7680-wide FFN rows).
fn rotate(values: &mut [f32]) {
    let mut chunk = 1;
    while values.len() % (chunk * 2) == 0 && chunk < 2048 {
        chunk *= 2;
    }
    for block in values.chunks_mut(chunk) {
        wht(block);
    }
}

fn signs(k: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..k)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Per-column-block scales shared by every row: `s_b` is the max over the
/// whole 256-column block. The kernel is unchanged because the block scale is
/// folded into the activation instead of the accumulator.
fn quant_colblock(values: &[f32], n: usize, k: usize, block: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; values.len()];
    let mut scales = vec![0.0f32; k.div_ceil(block)];
    for (index, scale) in scales.iter_mut().enumerate() {
        let start = index * block;
        let end = (start + block).min(k);
        let mut max = 0.0f32;
        for row in 0..n {
            for value in &values[row * k + start..row * k + end] {
                max = max.max(value.abs());
            }
        }
        *scale = if max > 0.0 { max / 7.0 } else { 1.0 };
    }
    for row in 0..n {
        for kk in 0..k {
            let scale = scales[kk / block];
            out[row * k + kk] = (values[row * k + kk] / scale).round().clamp(-8.0, 7.0) * scale;
        }
    }
    (out, scales)
}

fn quant_row(row: &[f32], block: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; row.len()];
    for (dst_chunk, src_chunk) in out.chunks_mut(block).zip(row.chunks(block)) {
        let max = src_chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let step = if max > 0.0 { max / 7.0 } else { 1.0 };
        for (dst, src) in dst_chunk.iter_mut().zip(src_chunk) {
            *dst = (*src / step).round().clamp(-8.0, 7.0) * step;
        }
    }
    out
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let path = Path::new(&path);
    let tensors = [
        "blk.0.shortconv.in_proj.weight",
        "blk.0.shortconv.out_proj.weight",
        "blk.2.attn_q.weight",
        "blk.0.ffn_gate.weight",
    ];
    let block: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(32);
    println!("block = {block}");
    println!("{:40} {:>8} {:>8} {:>8} {:>8} {:>8}", "tensor", "perrow", "had+row", "block", "had+blk", "colblk");
    for name in tensors {
        let (matrix, _info) = gguf_source::load_matrix(path, name, None).expect("matrix");
        let (n, k) = (matrix.n(), matrix.k());
        let values = matrix.data();
        // deterministic probe vector
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut x: Vec<f32> = (0..k)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let s = signs(k);
        let mut xr = x.clone();
        for (value, sign) in xr.iter_mut().zip(&s) {
            *value *= *sign;
        }
        // reference product
        let mut reference = vec![0.0f32; n];
        for row in 0..n {
            let w = &values[row * k..(row + 1) * k];
            reference[row] = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
        let mut perrow = vec![0.0f32; n];
        let mut hadrow = vec![0.0f32; n];
        let mut blk32 = vec![0.0f32; n];
        let mut hadblk = vec![0.0f32; n];
        let inv = 1.0 / (k as f32).sqrt();
        for row in 0..n {
            let w = &values[row * k..(row + 1) * k];
            // per-row int4
            let q = quant_row(w, k);
            perrow[row] = q.iter().zip(&x).map(|(a, b)| a * b).sum();
            let qb = quant_row(w, block);
            blk32[row] = qb.iter().zip(&x).map(|(a, b)| a * b).sum();
            // rotated: W' = WHT(w * s) * inv^2? Keep the transform on the row
            // and the same on the activation so the product is preserved.
            let mut wr: Vec<f32> = w.iter().zip(&s).map(|(a, b)| a * b).collect();
            rotate(&mut wr);
            let mut xr2 = xr.clone();
            rotate(&mut xr2);
            let qr = quant_row(&wr, k);
            hadrow[row] = qr.iter().zip(&xr2).map(|(a, b)| a * b).sum::<f32>() * inv * inv;
            let qrb = quant_row(&wr, 32);
            hadblk[row] = qrb.iter().zip(&xr2).map(|(a, b)| a * b).sum::<f32>() * inv * inv;
        }
        // weight-space sanity check on the same row range
        let wq_row = quant_row(&values[..k], k);
        let wq_blk = quant_row(&values[..k], 32);
        eprintln!(
            "  weight-space cos: perrow {:.5} blk32 {:.5}",
            cos(&values[..k], &wq_row),
            cos(&values[..k], &wq_blk)
        );
        let (qcol, scales) = quant_colblock(&values, n, k, block);
        let mut xscaled = vec![0.0f32; k];
        for kk in 0..k {
            xscaled[kk] = x[kk] * scales[kk / block];
        }
        let mut colblk = vec![0.0f32; n];
        for row in 0..n {
            let w = &qcol[row * k..(row + 1) * k];
            colblk[row] = w.iter().zip(&xscaled).map(|(a, b)| a * b).sum::<f32>();
        }
        println!(
            "{:40} {:>8.5} {:>8.5} {:>8.5} {:>8.5} {:>8.5}",
            name.rsplit('.').nth(1).unwrap_or(name),
            cos(&reference, &perrow),
            cos(&reference, &hadrow),
            cos(&reference, &blk32),
            cos(&reference, &hadblk),
            cos(&reference, &colblk)
        );
    }
}
