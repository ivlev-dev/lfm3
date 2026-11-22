#![allow(clippy::needless_range_loop)]
//! Offline study of the K/V formats on real post-RoPE activations.
//!
//! The DSP paged-attention kernel quantises K per token to int4 and V to int8
//! with a scale that cannot vary per token. This example runs the CPU oracle
//! through the first attention layer of LFM3, captures the real attention
//! input at every position, projects q/k/v with the real weights, applies
//! q/k RMSNorm and rope exactly like `CpuBackend`, and reports what different
//! scale layouts cost in `cos(score)` and `cos(PV)`.
//!
//! ```sh
//! cargo run -p myelin-npu --example analyze_kv -- ../models/LFM3-1.7B-Q4_K_M.gguf
//! ```

use lfm_gguf::{GgmlType, GgufFile};
use myelin_core::backend::{Backend, CpuBackend, NormSlot, NormWeights};
use myelin_core::kv::PagedKvCache;
use myelin_core::ops::RopeTable;
use myelin_core::spec::{AttentionKind, LayerSpec};
use myelin_core::{build_spec, GgufMeta};
use myelin_pack::gguf_source;
use myelin_pack::Matrix;

const TOKENS: usize = 8;

fn load_vector(file: &GgufFile, name: &str) -> Vec<f32> {
    let info = file.tensor(name).expect("tensor");
    let bytes = file.tensor_bytes(name).expect("bytes");
    match info.ggml_type {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        other => panic!("norm {name} has type {other:?}"),
    }
}

fn matvec(matrix: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.n()];
    myelin_core::ops::matvec(matrix.data(), matrix.n(), matrix.k(), x, &mut out);
    out
}

fn norm_head(vector: &mut [f32], weight: &[f32], eps: f32) {
    let sum: f64 = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let mean = sum / vector.len() as f64;
    let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;
    for (value, weight) in vector.iter_mut().zip(weight) {
        *value = *value * scale * weight;
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut ab = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (x, y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return 1.0;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// Per-token int4 with the current format's per-token scale.
fn quantize_k_token(row: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = row.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
    let codes = row
        .iter()
        .map(|&value| {
            if scale > 0.0 {
                (value / scale).round().clamp(-7.0, 7.0) as i8
            } else {
                0
            }
        })
        .collect();
    (codes, scale)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/LFM3-1.7B-Q4_K_M.gguf".to_string());
    let file = GgufFile::open(&path).expect("open gguf");
    let spec = build_spec(&GgufMeta::from_gguf(&file)).expect("spec");
    let dims = spec.dims;
    let mut backend = CpuBackend::from_gguf(&file, &spec).expect("cpu backend");

    // Layer 2 is the first attention layer in LFM3 (0 and 1 are conv).
    let layer = spec
        .layers
        .iter()
        .position(|layer| matches!(layer, LayerSpec::Attention { .. }))
        .expect("an attention layer");
    let prefix = format!("blk.{layer}");
    let head_dim = dims.head_dim;
    let heads = dims.heads;
    let kv_heads = dims.kv_heads;

    let q_weight = gguf_source::load_matrix(
        std::path::Path::new(&path),
        &format!("{prefix}.attn_q.weight"),
        None,
    )
    .expect("q")
    .0;
    let k_weight = gguf_source::load_matrix(
        std::path::Path::new(&path),
        &format!("{prefix}.attn_k.weight"),
        None,
    )
    .expect("k")
    .0;
    let v_weight = gguf_source::load_matrix(
        std::path::Path::new(&path),
        &format!("{prefix}.attn_v.weight"),
        None,
    )
    .expect("v")
    .0;
    let q_norm = load_vector(&file, &format!("{prefix}.attn_q_norm.weight"));
    let k_norm = load_vector(&file, &format!("{prefix}.attn_k_norm.weight"));

    // Drive the CPU oracle through the layers before the target and keep the
    // target layer's attention input at every position.
    let hidden = dims.hidden;
    let mut kv = vec![None; dims.layers];
    let mut conv = vec![None; dims.layers];
    for (index, layer_spec) in spec.layers.iter().enumerate() {
        match layer_spec {
            LayerSpec::Attention { .. } => {
                kv[index] = Some(PagedKvCache::new(dims.kv_dim(), TOKENS).expect("kv"));
            }
            LayerSpec::Conv { ksize } => {
                conv[index] = Some(vec![0.0; (ksize - 1) * hidden]);
            }
            LayerSpec::Moe { .. } => panic!("moe layer"),
        }
    }
    let tokens: Vec<u32> = (0..TOKENS).map(|index| 100 + index as u32 * 37).collect();
    let mut inputs = Vec::with_capacity(TOKENS);
    let mut x = vec![0.0f32; hidden];
    let mut residual = vec![0.0f32; hidden];
    let mut normed = vec![0.0f32; hidden];
    let mut output = vec![0.0f32; hidden];
    let mut merged = vec![0.0f32; hidden];
    for (position, token) in tokens.iter().enumerate() {
        backend.embed(*token, &mut x).expect("embed");
        for index in 0..=layer {
            residual.copy_from_slice(&x);
            backend
                .rms_norm(
                    NormWeights::Layer {
                        layer: index,
                        slot: NormSlot::Attention,
                    },
                    &x,
                    &mut normed,
                )
                .expect("norm");
            if index == layer {
                inputs.push(normed.clone());
                break;
            }
            match spec.layers[index] {
                LayerSpec::Attention { kind } => {
                    let start = myelin_core::ops::window_start(kind, position, dims.sliding_window);
                    backend
                        .attention_layer(
                            myelin_core::AttentionCall {
                                layer: index,
                                position,
                                start,
                                kind,
                                input: &normed,
                            },
                            kv[index].as_mut().unwrap(),
                            &mut output,
                        )
                        .expect("attention");
                }
                LayerSpec::Conv { .. } => {
                    backend
                        .conv_layer(index, &normed, conv[index].as_mut().unwrap(), &mut output)
                        .expect("conv");
                }
                LayerSpec::Moe { .. } => panic!("moe"),
            }
            backend
                .rms_norm(
                    NormWeights::Layer {
                        layer: index,
                        slot: NormSlot::PostAttention,
                    },
                    &output,
                    &mut merged,
                )
                .expect("post norm");
            for (value, residual) in merged.iter_mut().zip(&residual) {
                *value += residual;
            }
            backend
                .rms_norm(
                    NormWeights::Layer {
                        layer: index,
                        slot: NormSlot::Ffn,
                    },
                    &merged,
                    &mut normed,
                )
                .expect("ffn norm");
            backend.ffn_layer(index, &normed, &mut output).expect("ffn");
            backend
                .rms_norm(
                    NormWeights::Layer {
                        layer: index,
                        slot: NormSlot::PostFfn,
                    },
                    &output,
                    &mut normed,
                )
                .expect("post ffn norm");
            for (value, normalized) in merged.iter_mut().zip(&normed) {
                *value += normalized;
            }
            std::mem::swap(&mut x, &mut merged);
        }
    }

    let rope = match spec.layers[layer] {
        LayerSpec::Attention {
            kind: AttentionKind::Sliding,
        } => spec.rope.sliding.as_ref().unwrap_or(&spec.rope.full),
        _ => &spec.rope.full,
    };
    let rope = RopeTable::new(rope);
    let mut q = vec![vec![0.0f32; dims.q_dim()]; TOKENS];
    let mut k = vec![vec![0.0f32; dims.kv_dim()]; TOKENS];
    let mut v = vec![vec![0.0f32; dims.kv_dim()]; TOKENS];
    for position in 0..TOKENS {
        let qv = matvec(&q_weight, &inputs[position]);
        let kvv = matvec(&k_weight, &inputs[position]);
        let vv = matvec(&v_weight, &inputs[position]);
        for head in 0..heads {
            let slice = &mut q[position][head * head_dim..(head + 1) * head_dim];
            slice.copy_from_slice(&qv[head * head_dim..(head + 1) * head_dim]);
            norm_head(slice, &q_norm, spec.norm_eps);
            rope.apply(slice, position);
        }
        for head in 0..kv_heads {
            let slice = &mut k[position][head * head_dim..(head + 1) * head_dim];
            slice.copy_from_slice(&kvv[head * head_dim..(head + 1) * head_dim]);
            norm_head(slice, &k_norm, spec.norm_eps);
            rope.apply(slice, position);
        }
        v[position].copy_from_slice(&vv);
    }

    // q int8: one scalar over the layer, as `AttentionStage::q_scale` gives.
    let q_max = q
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let q_scale = 127.0 / q_max;
    let q8: Vec<Vec<i8>> = q
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| (value * q_scale).round().clamp(-127.0, 127.0) as i8)
                .collect()
        })
        .collect();

    // ---------- K: score cosine under four scale layouts ----------
    let mut exact = Vec::new();
    let mut per_token = Vec::new();
    let mut per_channel = Vec::new();
    let mut both = Vec::new();
    let mut channel_scale = vec![[0.0f32; 128]; kv_heads];
    for head in 0..kv_heads {
        for dim in 0..head_dim {
            let max = (0..TOKENS).fold(0.0f32, |acc, t| acc.max(k[t][head * head_dim + dim].abs()));
            channel_scale[head][dim] = if max > 0.0 { max / 7.0 } else { 1.0 };
        }
    }
    for position in 0..TOKENS {
        for qh in 0..heads {
            let kh = qh / (heads / kv_heads);
            let q_row = &q[position][qh * head_dim..(qh + 1) * head_dim];
            let q8_row = &q8[position][qh * head_dim..(qh + 1) * head_dim];
            for k_all in k.iter().take(TOKENS) {
                let k_row = &k_all[kh * head_dim..(kh + 1) * head_dim];
                let (codes_t, scale_t) = quantize_k_token(k_row);
                let codes_c: Vec<i8> = k_row
                    .iter()
                    .enumerate()
                    .map(|(dim, &value)| {
                        (value / channel_scale[kh][dim]).round().clamp(-7.0, 7.0) as i8
                    })
                    .collect();
                // Per-channel folded into q: q'[d] = q[d] * channel_scale[d],
                // quantised with one scalar (`qc_scale`). The K code is the
                // per-token/per-channel code; `scale_t` stays in `kt`.
                let mut q_channel = vec![0.0f32; head_dim];
                let mut q8_both = vec![0i8; head_dim];
                for dim in 0..head_dim {
                    q_channel[dim] = q_row[dim] * channel_scale[kh][dim];
                }
                let qc_max = q_channel.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                let qc_scale = if qc_max > 0.0 { 127.0 / qc_max } else { 1.0 };
                for dim in 0..head_dim {
                    q8_both[dim] = (q_channel[dim] * qc_scale).round().clamp(-127.0, 127.0) as i8;
                }
                let codes_both: Vec<i8> = k_row
                    .iter()
                    .enumerate()
                    .map(|(dim, &value)| {
                        (value / (channel_scale[kh][dim] * scale_t.max(1e-9)))
                            .round()
                            .clamp(-7.0, 7.0) as i8
                    })
                    .collect();

                let mut exact_score = 0.0f64;
                let mut token_score = 0.0f64;
                let mut channel_score = 0.0f64;
                let mut both_score = 0.0f64;
                for dim in 0..head_dim {
                    exact_score += q_row[dim] as f64 * k_row[dim] as f64;
                    // Current format: kt is divided by q_scale in the KV image.
                    token_score += q8_row[dim] as f64 * codes_t[dim] as f64 * scale_t as f64;
                    // Per-channel only: q scaled by channel_scale, unit kt.
                    channel_score += q8_both[dim] as f64 / qc_scale as f64 * codes_c[dim] as f64;
                    // Per-channel + per-token.
                    both_score += q8_both[dim] as f64 / qc_scale as f64
                        * codes_both[dim] as f64
                        * scale_t as f64;
                }
                exact.push(exact_score);
                per_token.push(token_score);
                per_channel.push(channel_score);
                both.push(both_score);
            }
        }
    }
    println!("layer {layer}, {TOKENS} positions, post-RoPE K, q8 scale {q_scale:.4}");
    println!("score cos: per-token {:.6}", cosine(&exact, &per_token));
    println!("score cos: per-channel {:.6}", cosine(&exact, &per_channel));
    println!("score cos: both {:.6}", cosine(&exact, &both));

    // ---------- V: one scale vs per channel ----------
    let v_max_global = v
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let v_global = v_max_global / 127.0;
    let mut v_channel = vec![0.0f32; dims.kv_dim()];
    for dim in 0..dims.kv_dim() {
        let max = (0..TOKENS).fold(0.0f32, |acc, t| acc.max(v[t][dim].abs()));
        v_channel[dim] = if max > 0.0 { max / 127.0 } else { 1.0 };
    }
    let mut flat = Vec::new();
    let mut global_flat = Vec::new();
    let mut channel_flat = Vec::new();
    for values in v.iter().take(TOKENS) {
        for &value in values {
            flat.push(value as f64);
            global_flat.push((value / v_global).round() as f64 * v_global as f64);
        }
        for (dim, &value) in values.iter().enumerate() {
            channel_flat.push((value / v_channel[dim]).round() as f64 * v_channel[dim] as f64);
        }
    }
    println!("V cos: global {:.6}", cosine(&flat, &global_flat));
    println!("V cos: channel {:.6}", cosine(&flat, &channel_flat));

    // PV with exact softmax and the true scores (temperature 1).
    let inv_sqrt = 1.0 / (head_dim as f64).sqrt();
    let mut pv_exact = Vec::new();
    let mut pv_global = Vec::new();
    let mut pv_channel = Vec::new();
    let mut idx = 0usize;
    for q_all in q.iter().take(TOKENS) {
        for qh in 0..heads {
            let kh = qh / (heads / kv_heads);
            let q_row = &q_all[qh * head_dim..(qh + 1) * head_dim];
            let mut scores = vec![0.0f64; TOKENS];
            for (t, k_all) in k.iter().take(TOKENS).enumerate() {
                let k_row = &k_all[kh * head_dim..(kh + 1) * head_dim];
                scores[t] = q_row
                    .iter()
                    .zip(k_row)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * inv_sqrt;
            }
            let max = scores.iter().copied().fold(f64::MIN, f64::max);
            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            for dim in 0..head_dim {
                let mut exact_v = 0.0;
                let mut global_v = 0.0;
                let mut channel_v = 0.0;
                for t in 0..TOKENS {
                    exact_v += scores[t] * v[t][kh * head_dim + dim] as f64;
                    global_v += scores[t]
                        * (v[t][kh * head_dim + dim] / v_global).round() as f64
                        * v_global as f64;
                    channel_v += scores[t]
                        * (v[t][kh * head_dim + dim] / v_channel[kh * head_dim + dim]).round()
                            as f64
                        * v_channel[kh * head_dim + dim] as f64;
                }
                pv_exact.push(exact_v / sum);
                pv_global.push(global_v / sum);
                pv_channel.push(channel_v / sum);
                idx += 1;
            }
        }
    }
    println!("PV cos: global {:.6}", cosine(&pv_exact, &pv_global));
    println!("PV cos: channel {:.6}", cosine(&pv_exact, &pv_channel));
    println!("PV samples: {idx}");
}
