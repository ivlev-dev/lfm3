//! Build the device-layer fixture from the real LFM3 checkpoint.
//!
//! ```sh
//! cargo run -p myelin-npu --example make_layer_fixture -- \
//!     ../models/LFM3-1.7B-Q4_K_M.gguf /tmp/opencode/layer-fixture
//! ```
//!
//! The fixture contains int4 GEMV blobs, one shortconv blob, the activation
//! inputs of conv layer 0 and attention layer 2, and the CPU-oracle outputs
//! for every position. `dsp-layer-check` runs the same layers through `exec`
//! on the phone and compares.

use std::path::Path;

use myelin_core::backend::{Backend, CpuBackend, NormSlot, NormWeights};
use myelin_core::kv::PagedKvCache;
use myelin_core::ops::{self, RopeTable};
use myelin_core::spec::{AttentionKind, LayerSpec, ModelSpec, RopeKind};
use myelin_core::KvCache;
use myelin_core::{build_spec, GgufMeta};
use myelin_npu::layer_fixture::{
    write_f32, AttentionFixture, ConvFixture, FixtureDims, FixtureRope, LayerFixture,
};
use myelin_pack::gemv::{self, GemvManifest};
use myelin_pack::gguf_source;
use myelin_pack::shortconv::{self, ActivationScales};
use myelin_pack::{Matrix, PackMeta};

const TOKENS: usize = 4;

fn matvec(matrix: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.n()];
    ops::matvec(matrix.data(), matrix.n(), matrix.k(), x, &mut out);
    out
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
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

fn load_vector(file: &lfm_gguf::GgufFile, name: &str) -> Vec<f32> {
    use lfm_gguf::GgmlType;
    let info = file.tensor(name).expect("tensor");
    let bytes = file.tensor_bytes(name).expect("tensor bytes");
    match info.ggml_type {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        other => panic!("norm {name} is {other:?}"),
    }
}

fn rope_of(spec: &myelin_core::RopeSpec) -> FixtureRope {
    FixtureRope {
        kind: match spec.kind {
            RopeKind::Full => "full".to_string(),
            RopeKind::Partial { .. } => "partial".to_string(),
        },
        theta: spec.theta,
        dim: spec.dim,
        rotated_dim: match spec.kind {
            RopeKind::Full => 0,
            RopeKind::Partial { rotated_dim } => rotated_dim,
        },
    }
}

fn pack_gemv(dir: &Path, name: &str, model: &Path, tensor: &str) -> Matrix {
    let (matrix, source) = gguf_source::load_matrix(model, tensor, None).expect("load matrix");
    let meta = PackMeta {
        tensor: tensor.to_string(),
        source: Some(source),
    };
    let packed = gemv::build(&matrix, &meta).expect("pack gemv");
    let manifest = GemvManifest::from_blob(&meta, &packed);
    std::fs::write(dir.join(format!("{name}.gemv")), &packed.blob).expect("write gemv");
    manifest
        .write(&dir.join(format!("{name}.json")))
        .expect("write manifest");
    println!(
        "{name}: {}x{} -> cos {:.6}, {} B",
        matrix.n(),
        matrix.k(),
        packed.quality.cos_total,
        packed.blob.len()
    );
    matrix
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/LFM3-1.7B-Q4_K_M.gguf".to_string());
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/opencode/layer-fixture".to_string());
    let model_path = Path::new(&model);
    let dir = Path::new(&out_dir);
    std::fs::create_dir_all(dir).expect("fixture dir");

    let file = lfm_gguf::GgufFile::open(model_path).expect("open gguf");
    let spec: ModelSpec = build_spec(&GgufMeta::from_gguf(&file)).expect("spec");
    let dims = spec.dims;
    let conv_layer = spec
        .layers
        .iter()
        .position(|layer| layer.is_conv())
        .expect("conv layer");
    let attention_layer = spec
        .layers
        .iter()
        .position(|layer| layer.is_attention())
        .expect("attention layer");
    assert_eq!(conv_layer, 0, "fixture assumes conv layer 0");
    assert_eq!(attention_layer, 2, "fixture assumes attention layer 2");
    let hidden = dims.hidden;

    let conv_prefix = format!("blk.{conv_layer}.shortconv");
    let in_proj =
        gguf_source::load_matrix(model_path, &format!("{conv_prefix}.in_proj.weight"), None)
            .expect("in_proj")
            .0;
    let out_proj =
        gguf_source::load_matrix(model_path, &format!("{conv_prefix}.out_proj.weight"), None)
            .expect("out_proj")
            .0;
    let template =
        gguf_source::load_matrix(model_path, &format!("{conv_prefix}.conv.weight"), None)
            .expect("conv taps")
            .0;
    let taps: Vec<f32> = template.data().to_vec();

    let attn_prefix = format!("blk.{attention_layer}");
    let q_weight =
        gguf_source::load_matrix(model_path, &format!("{attn_prefix}.attn_q.weight"), None)
            .expect("q")
            .0;
    let k_weight =
        gguf_source::load_matrix(model_path, &format!("{attn_prefix}.attn_k.weight"), None)
            .expect("k")
            .0;
    let v_weight =
        gguf_source::load_matrix(model_path, &format!("{attn_prefix}.attn_v.weight"), None)
            .expect("v")
            .0;
    let o_weight = gguf_source::load_matrix(
        model_path,
        &format!("{attn_prefix}.attn_output.weight"),
        None,
    )
    .expect("o")
    .0;
    let gate_weight =
        gguf_source::load_matrix(model_path, &format!("{attn_prefix}.attn_gate.weight"), None)
            .ok()
            .map(|(matrix, _)| matrix);
    let q_norm = load_vector(&file, &format!("{attn_prefix}.attn_q_norm.weight"));
    let k_norm = load_vector(&file, &format!("{attn_prefix}.attn_k_norm.weight"));

    // Run the CPU oracle through the first three layers, capturing the inputs
    // and outputs of the two layers under test.
    let mut backend = CpuBackend::from_gguf(&file, &spec).expect("cpu backend");
    let mut kv = vec![None; dims.layers];
    let mut conv_state = vec![None; dims.layers];
    for (index, layer) in spec.layers.iter().enumerate() {
        match layer {
            LayerSpec::Attention { .. } => {
                kv[index] = Some(PagedKvCache::new(dims.kv_dim(), TOKENS).expect("kv"));
            }
            LayerSpec::Conv { ksize } => {
                conv_state[index] = Some(vec![0.0f32; (ksize - 1) * hidden]);
            }
            LayerSpec::Moe { .. } => panic!("moe"),
        }
    }
    let tokens: Vec<u32> = vec![100; TOKENS];
    let mut conv_inputs = Vec::new();
    let mut conv_outputs = Vec::new();
    let mut attn_inputs = Vec::new();
    let mut attn_outputs = Vec::new();
    let mut x = vec![0.0f32; hidden];
    let mut residual = vec![0.0f32; hidden];
    let mut normed = vec![0.0f32; hidden];
    let mut output = vec![0.0f32; hidden];
    let mut merged = vec![0.0f32; hidden];
    for (position, token) in tokens.iter().enumerate() {
        backend.embed(*token, &mut x).expect("embed");
        for index in 0..=attention_layer {
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
            if index == conv_layer {
                conv_inputs.push(normed.clone());
            }
            if index == attention_layer {
                attn_inputs.push(normed.clone());
            }
            match spec.layers[index] {
                LayerSpec::Attention { kind } => {
                    let start = ops::window_start(kind, position, dims.sliding_window);
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
                        .conv_layer(
                            index,
                            &normed,
                            conv_state[index].as_mut().unwrap(),
                            &mut output,
                        )
                        .expect("conv");
                }
                LayerSpec::Moe { .. } => panic!("moe"),
            }
            if index == conv_layer {
                conv_outputs.push(output.clone());
            }
            if index == attention_layer {
                attn_outputs.push(output.clone());
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

    // ---- conv calibration and intermediates ----
    // `DspBackend` multiplies the f32 activation by `act_scale`, so the
    // field is the quantisation multiplier; `set_scale` gets its reciprocal.
    let s_in_conv = 110.0
        / max_abs(
            &conv_inputs
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
        );
    let mut conv_mix = Vec::new();
    let mut conv_y = Vec::new();
    let mut conv_out_real = Vec::new();
    let mut state = vec![0.0f32; 6 * hidden];
    let ksize = 7usize;
    for input in &conv_inputs {
        let mix = matvec(&in_proj, input);
        let b = &mix[..hidden];
        let c = &mix[hidden..2 * hidden];
        let xx = &mix[2 * hidden..];
        let mut conv_input = vec![0.0f32; hidden];
        for index in 0..hidden {
            conv_input[index] = b[index] * xx[index];
        }
        let mut y = vec![0.0f32; hidden];
        for channel in 0..hidden {
            let row = template.row(channel);
            let mut acc = row[ksize - 1] * conv_input[channel];
            for (lag, weight) in row.iter().take(ksize - 1).enumerate() {
                acc += weight * state[lag * hidden + channel];
            }
            y[channel] = c[channel] * acc;
        }
        if ksize > 2 {
            state.copy_within(hidden.., 0);
        }
        state[(ksize - 2) * hidden..].copy_from_slice(&conv_input);
        conv_out_real.push(matvec(&out_proj, &y));
        conv_mix.push(mix);
        conv_y.push(y);
    }
    for position in 0..TOKENS {
        let (cos, max) =
            myelin_npu::layer_fixture::compare(&conv_outputs[position], &conv_out_real[position]);
        println!("conv reimplementation vs CPU token {position}: cos {cos:.6} max {max:.3e}");
    }
    let s_bc = max_abs(
        &conv_mix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let s_y = max_abs(
        &conv_y
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let s_out_conv = max_abs(
        &conv_out_real
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let conv_blob =
        shortconv::build(&taps, hidden, ActivationScales::cli(s_bc, s_y)).expect("pack shortconv");
    std::fs::write(dir.join("conv.blob"), &conv_blob.blob).expect("write conv blob");
    println!(
        "conv: s_in={s_in_conv:.6} s_bc={s_bc:.6} s_y={s_y:.6} s_out={s_out_conv:.6} ksh={}",
        conv_blob.ksh
    );

    // ---- attention calibration ----
    let s_in_attn = 110.0
        / max_abs(
            &attn_inputs
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
        );
    let mut q_raw = Vec::new();
    let mut k_raw = Vec::new();
    let mut v_raw = Vec::new();
    let mut gate_raw = Vec::new();
    for input in &attn_inputs {
        q_raw.push(matvec(&q_weight, input));
        k_raw.push(matvec(&k_weight, input));
        v_raw.push(matvec(&v_weight, input));
        if let Some(gate) = &gate_weight {
            gate_raw.push(matvec(gate, input));
        }
    }
    let rope_spec = match spec.layers[attention_layer] {
        LayerSpec::Attention {
            kind: AttentionKind::Sliding,
        } => spec.rope.sliding.as_ref().unwrap_or(&spec.rope.full),
        _ => &spec.rope.full,
    };
    let rope = RopeTable::new(rope_spec);
    let mut q_post = Vec::new();
    let mut k_post = Vec::new();
    for position in 0..TOKENS {
        let mut q = q_raw[position].clone();
        for head in 0..dims.heads {
            let slice = &mut q[head * dims.head_dim..(head + 1) * dims.head_dim];
            norm_head(slice, &q_norm, spec.norm_eps);
            rope.apply(slice, position);
        }
        q_post.push(q);
        let mut k = k_raw[position].clone();
        for head in 0..dims.kv_heads {
            let slice = &mut k[head * dims.head_dim..(head + 1) * dims.head_dim];
            norm_head(slice, &k_norm, spec.norm_eps);
            rope.apply(slice, position);
        }
        k_post.push(k);
    }
    let q_scale = max_abs(
        &q_raw
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let k_scale = max_abs(
        &k_raw
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let v_scale = max_abs(
        &v_raw
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let gate_scale = if gate_raw.is_empty() {
        1.0
    } else {
        max_abs(
            &gate_raw
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
        ) / 110.0
    };
    let attn_q_scale = 127.0
        / max_abs(
            &q_post
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>(),
        );
    let mut v_channel = vec![0.0f32; dims.kv_dim()];
    for dim in 0..dims.kv_dim() {
        let max = (0..TOKENS).fold(0.0f32, |acc, t| acc.max(v_raw[t][dim].abs()));
        v_channel[dim] = if max > 0.0 { max / 127.0 } else { 1.0 };
    }
    // attended (with gate) and the final o_proj range
    let mut kv_ref = PagedKvCache::new(dims.kv_dim(), TOKENS).expect("kv ref");
    let mut attended_max = 0.0f32;
    let mut out_real = Vec::new();
    for position in 0..TOKENS {
        kv_ref
            .append(&k_post[position], &v_raw[position])
            .expect("append");
        let start = ops::window_start(
            match spec.layers[attention_layer] {
                LayerSpec::Attention { kind } => kind,
                _ => AttentionKind::Full,
            },
            position,
            dims.sliding_window,
        );
        let mut attended = vec![0.0f32; dims.q_dim()];
        ops::attention(
            &q_post[position],
            &kv_ref,
            start,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            &mut attended,
        )
        .expect("attention ref");
        if !gate_raw.is_empty() {
            for (head, gate) in gate_raw[position].iter().enumerate() {
                let factor = ops::sigmoid(*gate);
                let slice = &mut attended[head * dims.head_dim..(head + 1) * dims.head_dim];
                for value in slice {
                    *value *= factor;
                }
            }
        }
        attended_max = attended_max.max(max_abs(&attended));
        out_real.push(matvec(&o_weight, &attended));
    }
    let o_act_scale = 127.0 / attended_max;
    let out_scale_attn = max_abs(
        &out_real
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>(),
    ) / 110.0;
    let cache_pages = TOKENS.div_ceil(32).max(1);

    // ---- write blobs ----
    pack_gemv(
        dir,
        "conv_in_proj",
        model_path,
        &format!("{conv_prefix}.in_proj.weight"),
    );
    pack_gemv(
        dir,
        "conv_out_proj",
        model_path,
        &format!("{conv_prefix}.out_proj.weight"),
    );
    pack_gemv(
        dir,
        "attn_q",
        model_path,
        &format!("{attn_prefix}.attn_q.weight"),
    );
    pack_gemv(
        dir,
        "attn_k",
        model_path,
        &format!("{attn_prefix}.attn_k.weight"),
    );
    pack_gemv(
        dir,
        "attn_v",
        model_path,
        &format!("{attn_prefix}.attn_v.weight"),
    );
    pack_gemv(
        dir,
        "attn_o",
        model_path,
        &format!("{attn_prefix}.attn_output.weight"),
    );
    if gate_weight.is_some() {
        pack_gemv(
            dir,
            "attn_gate",
            model_path,
            &format!("{attn_prefix}.attn_gate.weight"),
        );
    }

    write_f32(&dir.join("conv_x.f32"), &conv_inputs.concat()).expect("conv x");
    write_f32(&dir.join("conv_y.f32"), &conv_outputs.concat()).expect("conv y");
    write_f32(&dir.join("attn_x.f32"), &attn_inputs.concat()).expect("attn x");
    write_f32(&dir.join("attn_y.f32"), &attn_outputs.concat()).expect("attn y");
    write_f32(&dir.join("attn_q_norm.f32"), &q_norm).expect("q norm");
    write_f32(&dir.join("attn_k_norm.f32"), &k_norm).expect("k norm");
    write_f32(&dir.join("attn_v_scales.f32"), &v_channel).expect("v scales");

    let layer_kinds = spec
        .layers
        .iter()
        .map(|layer| match layer {
            LayerSpec::Conv { .. } => "conv".to_string(),
            LayerSpec::Attention { .. } => "attention".to_string(),
            LayerSpec::Moe { .. } => "moe".to_string(),
        })
        .collect();
    let fixture = LayerFixture {
        model: model.clone(),
        conv_layer,
        attention_layer,
        dims: FixtureDims {
            hidden: dims.hidden,
            ffn: dims.ffn,
            layers: dims.layers,
            heads: dims.heads,
            kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            vocab: dims.vocab,
            conv_cache: dims.conv_cache,
            sliding_window: dims.sliding_window,
        },
        rope_full: rope_of(&spec.rope.full),
        rope_sliding: spec.rope.sliding.as_ref().map(rope_of),
        attention_kind: match spec.layers[attention_layer] {
            LayerSpec::Attention {
                kind: AttentionKind::Full,
            } => "full".to_string(),
            _ => "sliding".to_string(),
        },
        norm_eps: spec.norm_eps,
        layer_kinds,
        conv_tokens: TOKENS,
        attention_tokens: TOKENS,
        conv: ConvFixture {
            act_scale: s_in_conv,
            mix_scale: s_bc,
            y_scale: s_y,
            out_scale: s_out_conv,
            ksh: conv_blob.ksh,
            state_off: myelin_core::backend::dsp::conv_state_offset(conv_layer),
        },
        attention: AttentionFixture {
            act_scale: s_in_attn,
            q_scale,
            k_scale,
            v_scale,
            gate_scale,
            attn_q_scale,
            attn_out_scale: 1.0 / 4096.0,
            o_act_scale,
            out_scale: out_scale_attn,
            v_channel_scales: v_channel,
            has_gate: gate_weight.is_some(),
            cache_pages,
        },
    };
    fixture.write(dir).expect("write fixture");
    println!("fixture -> {}", dir.display());
}
