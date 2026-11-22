//! Host replica of the DSP shortconv layer (int4 weights, int8 codes).
//!
//! ```text
//! cargo run --release -p myelin-npu --example layer_model -- \
//!     /tmp/opencode/model-pack3 0 /tmp/opencode/model-pack/input.bin
//! ```
//!
//! The point is to reproduce `bcx` and `y` bit-exactly, so a divergence from
//! the device dump points at one specific stage instead of "the layer".
//!
//! STATUS: not bit-exact yet. Against `conv_bcx_0.bin` from the device the
//! model gives cos 0.34 and rms 32.4 versus 17.1, so something in the
//! `in_proj` replica (the `qp` packing or the activation path) is still
//! wrong. Do not trust its output until that matches; it is checked in as a
//! starting point, not as a reference.

use std::path::{Path, PathBuf};

struct Gemv {
    k: usize,
    n: usize,
    k_padded: usize,
    n_padded: usize,
    row_bytes: usize,
    weights: Vec<u8>,
    sw: Vec<f32>,
    input_scale: f32,
    output_scale: f32,
}

impl Gemv {
    fn load(path: &Path, input_scale: f32, output_scale: f32) -> Self {
        let blob = std::fs::read(path).expect("blob");
        let read = |offset: usize| {
            u32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap()) as usize
        };
        let k = read(8);
        let n = read(12);
        let k_padded = read(16);
        let n_padded = read(20);
        let row_bytes = read(24);
        let weight_bytes = n_padded * row_bytes;
        let weights = blob[128..128 + weight_bytes].to_vec();
        let base = 128 + weight_bytes;
        let sw = (0..n_padded)
            .map(|i| {
                let at = base + 4 * i;
                f32::from_le_bytes(blob[at..at + 4].try_into().unwrap())
            })
            .collect();
        Self {
            k,
            n,
            k_padded,
            n_padded,
            row_bytes,
            weights,
            sw,
            input_scale,
            output_scale,
        }
    }

    /// `codes = round(x / input_scale)`, clamped like the DSP's `quantize`.
    fn quantize(&self, x: &[f32]) -> Vec<i8> {
        x.iter()
            .map(|v| (v / self.input_scale).round().clamp(-127.0, 127.0) as i8)
            .collect()
    }

    /// Exact int4 dot per row, then `pack_out_i8`.
    fn apply(&self, codes: &[i8]) -> Vec<i8> {
        let bias: i32 = -8 * codes.iter().map(|v| i32::from(*v)).sum::<i32>();
        let mut out = vec![0i8; self.n_padded];
        for row in 0..self.n_padded {
            let bytes = &self.weights[row * self.row_bytes..(row + 1) * self.row_bytes];
            let mut acc: i64 = i64::from(bias);
            for (index, byte) in bytes.iter().enumerate() {
                let k0 = 2 * index;
                if k0 >= codes.len() {
                    break;
                }
                let lo = i32::from(*byte & 0x0f) - 8;
                acc += i64::from(codes[k0]) * i64::from(lo);
                if k0 + 1 < codes.len() {
                    let hi = i32::from((*byte >> 4) & 0x0f) - 8;
                    acc += i64::from(codes[k0 + 1]) * i64::from(hi);
                }
            }
            // qp = clamp(sw*sa/so, 0.99997) in Q16, value = acc*m
            let m = (self.sw[row] * self.input_scale / self.output_scale).min(0.99997);
            let q = (m * 65536.0f32 + 0.5) as i64;
            let v = ((acc * q + 32768) >> 16).clamp(-127, 127);
            out[row] = v as i8;
        }
        out
    }
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
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("pack dir"));
    let layer: usize = args.next().expect("layer").parse().expect("layer id");
    let input_path = PathBuf::from(args.next().expect("input f32 file"));
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("pack.json")).expect("pack.json"))
            .expect("json");
    let conv = &json["layers"][layer]["conv"];
    let file = |key: &str| dir.join(conv[key]["file"].as_str().unwrap());
    let in_proj = Gemv::load(
        &file("in_proj"),
        conv["in_proj"]["input_scale"].as_f64().unwrap() as f32,
        conv["in_proj"]["output_scale"].as_f64().unwrap() as f32,
    );
    let out_proj = Gemv::load(
        &file("out_proj"),
        conv["out_proj"]["input_scale"].as_f64().unwrap() as f32,
        conv["out_proj"]["output_scale"].as_f64().unwrap() as f32,
    );
    let bytes = std::fs::read(&input_path).expect("input");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let codes = in_proj.quantize(&values);
    let bcx = in_proj.apply(&codes);
    let _ = (out_proj, layer);
    println!(
        "bcx: n={} rms={:.2} plain={:.4}",
        bcx.len(),
        (bcx.iter().map(|v| f64::from(*v as i32).powi(2)).sum::<f64>() / bcx.len() as f64).sqrt(),
        cos(
            &bcx.iter().map(|v| f32::from(*v)).collect::<Vec<_>>(),
            &bcx.iter().map(|v| f32::from(*v)).collect::<Vec<_>>()
        )
    );
}
