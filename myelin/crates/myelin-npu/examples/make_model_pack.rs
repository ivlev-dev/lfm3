//! Build a full LFM3 model pack for the DSP from the GGUF checkpoint.
//!
//! ```text
//! cargo run --release -p myelin-npu --example make_model_pack -- \
//!     ../models/LFM3-1.7B-Q4_K_M.gguf /tmp/opencode/model-pack \
//!     --tokens /tmp/opencode/prompt-tokens.txt --calib-tokens 8
//! ```
//!
//! The tool does two things:
//!
//! 1. **Calibration.** It runs the whole LFM3 graph in f32 on the host (the
//!    same formulas as `myelin_core::CpuBackend`) over the first
//!    `--calib-tokens` prompt tokens and records the absmax of every tensor a
//!    DSP kernel quantises: layer inputs, q/k/v/gate projections, post-rope q,
//!    the gated attention output, conv `bcx`/`y`, FFN `silu(gate)*up`, every
//!    layer output and the final logits.
//! 2. **Packing.** It quantises every transformer matrix to int4 with
//!    `myelin-pack gemv`, builds the quantised shortconv blobs with the
//!    calibrated `s_bc`/`s_y`, copies the Q6_K token embedding unchanged and
//!    packs the tied head.
//!
//! The pack is self-contained: the phone needs neither the GGUF nor a
//! tokenizer, only `pack.json`, the blobs and the prompt token ids.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lfm_gguf::{GgufFile, TensorInfo};
use myelin_core::dequant;
use myelin_core::kv::{KvCache, PagedKvCache};
use myelin_core::ops::{self, RopeTable};
use myelin_core::spec::{AttentionKind, LayerSpec, ModelSpec, RopeKind, RopeSpec};
use myelin_core::{build_spec, GgufMeta};
use myelin_npu::model_pack::{
    ModelPack, PackAttention, PackConv, PackDims, PackEmbed, PackFfn, PackGemv, PackLayer,
    PackLmHead, PackNorms, PackRope, PACK_VERSION,
};
use myelin_pack::gemv::{self, GemvManifest};
use myelin_pack::gguf_source;
use myelin_pack::shortconv::{self, ActivationScales};
use myelin_pack::{Matrix, PackMeta};

const ACT_TARGET: f32 = 110.0;

struct Args {
    model: PathBuf,
    out: PathBuf,
    tokens: PathBuf,
    calib_tokens: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(
        args.next()
            .ok_or("usage: make_model_pack MODEL OUT --tokens FILE")?,
    );
    let out = PathBuf::from(args.next().ok_or("missing OUT directory")?);
    let mut tokens = None;
    let mut calib_tokens = 8usize;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--tokens" => tokens = Some(PathBuf::from(value()?)),
            "--calib-tokens" => {
                calib_tokens = value()?
                    .parse()
                    .map_err(|_| "--calib-tokens must be a positive integer".to_owned())?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if calib_tokens == 0 {
        return Err("--calib-tokens must be positive".to_owned());
    }
    Ok(Args {
        model,
        out,
        tokens: tokens.ok_or("--tokens is required")?,
        calib_tokens,
    })
}

/// Running maximum of the tensors one layer quantises.
#[derive(Default, Debug, Clone, Copy)]
struct AbsMax {
    value: f32,
    samples: usize,
}

impl AbsMax {
    fn update(&mut self, values: &[f32]) {
        self.samples += 1;
        for value in values {
            self.value = self.value.max(value.abs());
        }
    }

    /// Real units per int8 code that map the observed range onto `target`
    /// of the 127 signed codes.
    fn step(&self, target: f32) -> f32 {
        if self.value > 0.0 && self.value.is_finite() {
            self.value / target
        } else {
            1.0
        }
    }

    /// Multiplier that maps the observed range onto `target` codes.
    fn multiplier(&self, target: f32) -> f32 {
        if self.value > 0.0 && self.value.is_finite() {
            target / self.value
        } else {
            1.0
        }
    }
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

fn matvec(matrix: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.n()];
    ops::matvec(matrix.data(), matrix.n(), matrix.k(), x, &mut out);
    out
}

fn load_vector(file: &GgufFile, name: &str) -> Vec<f32> {
    let info = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing {name}"));
    let bytes = file.tensor_bytes(name).expect("tensor bytes");
    let count: usize = info.dims.iter().product::<u64>() as usize;
    dequant::dequantize_row(info.ggml_type, name, bytes, count).expect("dequantise norm")
}

fn load_matrix(model: &Path, name: &str) -> Matrix {
    gguf_source::load_matrix(model, name, None)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .0
}

fn rope_of(spec: &RopeSpec) -> PackRope {
    PackRope {
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

struct FfnWeights {
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}

struct AttnWeights {
    query: Matrix,
    key: Matrix,
    value: Matrix,
    output: Matrix,
    gate: Option<Matrix>,
}

struct ConvWeights {
    in_proj: Matrix,
    taps: Matrix,
    out_proj: Matrix,
}

struct FfnAcc {
    input: AbsMax,
    mid: AbsMax,
    out: AbsMax,
    weights: FfnWeights,
}

struct AttnAcc {
    input: AbsMax,
    q: AbsMax,
    k: AbsMax,
    v: AbsMax,
    gate: AbsMax,
    q_post: AbsMax,
    attended: AbsMax,
    out: AbsMax,
    v_channel: Vec<f32>,
    rope: RopeTable,
    kind: AttentionKind,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    kv: PagedKvCache,
    weights: AttnWeights,
}

struct ConvAcc {
    input: AbsMax,
    mix: AbsMax,
    y: AbsMax,
    out: AbsMax,
    weights: ConvWeights,
    state: Vec<f32>,
    ksize: usize,
}

struct LayerAcc {
    norms: [Vec<f32>; 4],
    ffn: FfnAcc,
    attention: Option<AttnAcc>,
    conv: Option<ConvAcc>,
}

impl LayerAcc {
    fn weight_bytes(&self) -> usize {
        let ffn = self.ffn.weights.gate.data().len()
            + self.ffn.weights.up.data().len()
            + self.ffn.weights.down.data().len();
        let attention = self.attention.as_ref().map_or(0, |attention| {
            let weights = &attention.weights;
            weights.query.data().len()
                + weights.key.data().len()
                + weights.value.data().len()
                + weights.output.data().len()
                + weights.gate.as_ref().map_or(0, |gate| gate.data().len())
        });
        let conv = self.conv.as_ref().map_or(0, |conv| {
            conv.weights.in_proj.data().len()
                + conv.weights.taps.data().len()
                + conv.weights.out_proj.data().len()
        });
        (ffn + attention + conv) * 4
    }
}

fn load_layers(model: &Path, spec: &ModelSpec, context: usize) -> Vec<LayerAcc> {
    let dims = spec.dims;
    let file = GgufFile::open(model).expect("open gguf");
    let mut out = Vec::with_capacity(dims.layers);
    for (index, layer) in spec.layers.iter().copied().enumerate() {
        let prefix = format!("blk.{index}");
        let norms = [
            load_vector(&file, &format!("{prefix}.attn_norm.weight")),
            load_vector(&file, &format!("{prefix}.post_attention_norm.weight")),
            load_vector(&file, &format!("{prefix}.ffn_norm.weight")),
            load_vector(&file, &format!("{prefix}.post_ffw_norm.weight")),
        ];
        let mut acc = LayerAcc {
            norms,
            ffn: FfnAcc {
                input: AbsMax::default(),
                mid: AbsMax::default(),
                out: AbsMax::default(),
                weights: FfnWeights {
                    gate: load_matrix(model, &format!("{prefix}.ffn_gate.weight")),
                    up: load_matrix(model, &format!("{prefix}.ffn_up.weight")),
                    down: load_matrix(model, &format!("{prefix}.ffn_down.weight")),
                },
            },
            attention: None,
            conv: None,
        };
        match layer {
            LayerSpec::Attention { kind } => {
                let rope_spec = match kind {
                    AttentionKind::Sliding => spec.rope.sliding.as_ref().unwrap_or(&spec.rope.full),
                    AttentionKind::Full => &spec.rope.full,
                };
                acc.attention = Some(AttnAcc {
                    input: AbsMax::default(),
                    q: AbsMax::default(),
                    k: AbsMax::default(),
                    v: AbsMax::default(),
                    gate: AbsMax::default(),
                    q_post: AbsMax::default(),
                    attended: AbsMax::default(),
                    out: AbsMax::default(),
                    v_channel: vec![0.0f32; dims.kv_dim()],
                    rope: RopeTable::new(rope_spec),
                    kind,
                    query_norm: load_vector(&file, &format!("{prefix}.attn_q_norm.weight")),
                    key_norm: load_vector(&file, &format!("{prefix}.attn_k_norm.weight")),
                    kv: PagedKvCache::new(dims.kv_dim(), context).expect("calibration kv"),
                    weights: AttnWeights {
                        query: load_matrix(model, &format!("{prefix}.attn_q.weight")),
                        key: load_matrix(model, &format!("{prefix}.attn_k.weight")),
                        value: load_matrix(model, &format!("{prefix}.attn_v.weight")),
                        output: load_matrix(model, &format!("{prefix}.attn_output.weight")),
                        gate: gguf_source::load_matrix(
                            model,
                            &format!("{prefix}.attn_gate.weight"),
                            None,
                        )
                        .ok()
                        .map(|(matrix, _)| matrix),
                    },
                });
            }
            LayerSpec::Conv { ksize } => {
                acc.conv = Some(ConvAcc {
                    input: AbsMax::default(),
                    mix: AbsMax::default(),
                    y: AbsMax::default(),
                    out: AbsMax::default(),
                    weights: ConvWeights {
                        in_proj: load_matrix(model, &format!("{prefix}.shortconv.in_proj.weight")),
                        taps: load_matrix(model, &format!("{prefix}.shortconv.conv.weight")),
                        out_proj: load_matrix(
                            model,
                            &format!("{prefix}.shortconv.out_proj.weight"),
                        ),
                    },
                    state: vec![0.0f32; (ksize - 1) * dims.hidden],
                    ksize,
                });
            }
            LayerSpec::Moe { .. } => panic!("mixture-of-experts layers are not executable"),
        }
        out.push(acc);
        print!("\rloaded layer {}/{}", index + 1, dims.layers);
        let _ = std::io::stdout().flush();
    }
    println!();
    out
}

struct HeadAcc {
    input: AbsMax,
    logits: AbsMax,
}

fn calibrate(
    file: &GgufFile,
    spec: &ModelSpec,
    layers: &mut [LayerAcc],
    tokens: &[u32],
) -> HeadAcc {
    let dims = spec.dims;
    let hidden = dims.hidden;
    let info = file.tensor("token_embd.weight").expect("embedding info");
    let head_bytes = file.tensor_bytes("token_embd.weight").expect("embedding");
    let row_bytes = dequant::row_bytes(info.ggml_type, hidden).expect("embedding row");
    let output_norm = load_vector(file, "output_norm.weight");
    let mut head = HeadAcc {
        input: AbsMax::default(),
        logits: AbsMax::default(),
    };
    let mut x = vec![0.0f32; hidden];
    for (position, token) in tokens.iter().enumerate() {
        let start = *token as usize * row_bytes;
        let row = &head_bytes[start..start + row_bytes];
        x.copy_from_slice(
            &dequant::dequantize_row(info.ggml_type, "token_embd", row, hidden)
                .expect("embedding row"),
        );
        for layer in layers.iter_mut() {
            let residual = x.clone();
            let normed = rms_norm(&layer.norms[0], &x, spec.norm_eps);
            let output = match (&mut layer.conv, &mut layer.attention) {
                (Some(conv), None) => run_conv(conv, &normed),
                (None, Some(attention)) => run_attention(
                    attention,
                    &normed,
                    position,
                    dims.heads,
                    dims.head_dim,
                    spec,
                ),
                _ => panic!("layer has inconsistent stages"),
            };
            x = finish_layer(layer, &residual, &output, spec);
        }
        let normed = rms_norm(&output_norm, &x, spec.norm_eps);
        head.input.update(&normed);
        let logits = host_logits(info, head_bytes, row_bytes, &normed, dims.vocab);
        head.logits.update(&logits);
        println!(
            "  token {position} (id {token}): hidden absmax {:.3}, logits absmax {:.3}",
            max_abs(&x),
            max_abs(&logits)
        );
    }
    head
}

fn rms_norm(weight: &[f32], x: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    ops::rms_norm(x, weight, eps, &mut out);
    out
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
}

/// `merged = rms(post_attn, out) + residual`, then FFN and the second add.
fn finish_layer(
    layer: &mut LayerAcc,
    residual: &[f32],
    output: &[f32],
    spec: &ModelSpec,
) -> Vec<f32> {
    let mut merged = rms_norm(&layer.norms[1], output, spec.norm_eps);
    for (value, residual) in merged.iter_mut().zip(residual) {
        *value += residual;
    }
    let normed = rms_norm(&layer.norms[2], &merged, spec.norm_eps);
    let ffn = &mut layer.ffn;
    ffn.input.update(&normed);
    let gate = matvec(&ffn.weights.gate, &normed);
    let up = matvec(&ffn.weights.up, &normed);
    let mid: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| ops::silu(*gate) * up)
        .collect();
    ffn.mid.update(&mid);
    let down = matvec(&ffn.weights.down, &mid);
    ffn.out.update(&down);
    let normalized = rms_norm(&layer.norms[3], &down, spec.norm_eps);
    merged
        .iter()
        .zip(&normalized)
        .map(|(merged, normalized)| merged + normalized)
        .collect()
}

fn run_attention(
    attention: &mut AttnAcc,
    input: &[f32],
    position: usize,
    heads: usize,
    head_dim: usize,
    spec: &ModelSpec,
) -> Vec<f32> {
    attention.input.update(input);
    let mut query = matvec(&attention.weights.query, input);
    let mut key = matvec(&attention.weights.key, input);
    let value = matvec(&attention.weights.value, input);
    let gates = attention
        .weights
        .gate
        .as_ref()
        .map(|gate| matvec(gate, input));
    attention.q.update(&query);
    attention.k.update(&key);
    attention.v.update(&value);
    if let Some(gates) = &gates {
        attention.gate.update(gates);
    }
    for (dim, value) in value.iter().enumerate() {
        attention.v_channel[dim] = attention.v_channel[dim].max(value.abs());
    }
    for head in 0..heads {
        let slice = &mut query[head * head_dim..(head + 1) * head_dim];
        norm_head(slice, &attention.query_norm, spec.norm_eps);
        attention.rope.apply(slice, position);
    }
    attention.q_post.update(&query);
    let kv_heads = key.len() / head_dim;
    for head in 0..kv_heads {
        let slice = &mut key[head * head_dim..(head + 1) * head_dim];
        norm_head(slice, &attention.key_norm, spec.norm_eps);
        attention.rope.apply(slice, position);
    }
    attention.kv.append(&key, &value).expect("append");
    let start = ops::window_start(attention.kind, position, spec.dims.sliding_window);
    let mut attended = vec![0.0f32; query.len()];
    ops::attention(
        &query,
        &attention.kv,
        start,
        heads,
        kv_heads,
        head_dim,
        &mut attended,
    )
    .expect("attention");
    if let Some(gates) = &gates {
        for (head, gate) in gates.iter().enumerate() {
            let factor = ops::sigmoid(*gate);
            for value in &mut attended[head * head_dim..(head + 1) * head_dim] {
                *value *= factor;
            }
        }
    }
    attention.attended.update(&attended);
    let output = matvec(&attention.weights.output, &attended);
    attention.out.update(&output);
    output
}

fn run_conv(conv: &mut ConvAcc, input: &[f32]) -> Vec<f32> {
    let hidden = input.len();
    let ksize = conv.ksize;
    conv.input.update(input);
    let mix = matvec(&conv.weights.in_proj, input);
    conv.mix.update(&mix);
    let (b, c) = (&mix[..hidden], &mix[hidden..2 * hidden]);
    let xx = &mix[2 * hidden..];
    let conv_input: Vec<f32> = b.iter().zip(xx).map(|(b, x)| b * x).collect();
    let mut y = vec![0.0f32; hidden];
    for channel in 0..hidden {
        let template = conv.weights.taps.row(channel);
        let mut accumulator = template[ksize - 1] * conv_input[channel];
        for (lag, weight) in template.iter().take(ksize - 1).enumerate() {
            accumulator += weight * conv.state[lag * hidden + channel];
        }
        y[channel] = c[channel] * accumulator;
    }
    if ksize > 2 {
        conv.state.copy_within(hidden.., 0);
    }
    conv.state[(ksize - 2) * hidden..].copy_from_slice(&conv_input);
    conv.y.update(&y);
    let out = matvec(&conv.weights.out_proj, &y);
    conv.out.update(&out);
    out
}

fn host_logits(
    info: &TensorInfo,
    bytes: &[u8],
    row_bytes: usize,
    x: &[f32],
    vocab: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab];
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(vocab);
    let chunk = vocab.div_ceil(threads);
    let cols = x.len();
    std::thread::scope(|scope| {
        for (chunk_index, out) in logits.chunks_mut(chunk).enumerate() {
            let first = chunk_index * chunk;
            scope.spawn(move || {
                for (offset, dst) in out.iter_mut().enumerate() {
                    let start = (first + offset) * row_bytes;
                    let row = &bytes[start..start + row_bytes];
                    let values = dequant::dequantize_row(info.ggml_type, "token_embd", row, cols)
                        .expect("logits row");
                    *dst = ops::dot(x, &values);
                }
            });
        }
    });
    logits
}

fn main() {
    let args = parse_args().expect("arguments");
    std::fs::create_dir_all(&args.out).expect("pack directory");

    let text = std::fs::read_to_string(&args.tokens).expect("prompt token file");
    let prompt: Vec<u32> = text
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().expect("token id"))
        .collect();
    if prompt.len() < args.calib_tokens {
        panic!(
            "prompt has {} tokens, calibration needs {}",
            prompt.len(),
            args.calib_tokens
        );
    }

    let file = GgufFile::open(&args.model).expect("open gguf");
    let spec: ModelSpec = build_spec(&GgufMeta::from_gguf(&file)).expect("spec");
    let dims = spec.dims;
    println!(
        "model: {} layers, hidden {}, ffn {}, vocab {}; calibrating on {} tokens",
        dims.layers, dims.hidden, dims.ffn, dims.vocab, args.calib_tokens
    );

    let load_started = Instant::now();
    let mut layers = load_layers(&args.model, &spec, args.calib_tokens);
    println!(
        "weights in memory: {:.2} GiB in {:.1}s",
        layers.iter().map(LayerAcc::weight_bytes).sum::<usize>() as f64 / (1u64 << 30) as f64,
        load_started.elapsed().as_secs_f64()
    );

    let calib = &prompt[..args.calib_tokens];
    let started = Instant::now();
    let head = calibrate(&file, &spec, &mut layers, calib);
    println!(
        "calibration: {:.2}s, head input absmax {:.4}, logits absmax {:.4}",
        started.elapsed().as_secs_f64(),
        head.input.value,
        head.logits.value
    );

    let (pack, conv_blobs) = build_pack(&args, &spec, &layers, &head);
    pack.write(&args.out).expect("write pack.json");

    let started = Instant::now();
    let written = pack_blobs(&args, &spec, &layers, &pack, &conv_blobs);
    println!(
        "packed {:.2} GiB of blobs in {:.1}s -> {}",
        written as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
        args.out.display()
    );
}

fn build_pack(
    args: &Args,
    spec: &ModelSpec,
    layers: &[LayerAcc],
    head: &HeadAcc,
) -> (ModelPack, Vec<(String, Vec<u8>)>) {
    let dims = spec.dims;
    let mut pack_layers = Vec::with_capacity(layers.len());
    let mut conv_blobs = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        let ffn_input = layer.ffn.input.step(ACT_TARGET);
        let ffn = PackFfn {
            gate: pack_gemv(&format!("ffn_{index:02}_gate"), ffn_input, 1.0),
            up: pack_gemv(&format!("ffn_{index:02}_up"), ffn_input, 1.0),
            down: pack_gemv(
                &format!("ffn_{index:02}_down"),
                layer.ffn.mid.step(127.0),
                layer.ffn.out.step(ACT_TARGET),
            ),
            mid_scale: layer.ffn.mid.step(127.0),
            reps: 1,
        };
        let conv = layer.conv.as_ref().map(|conv| {
            let mix_scale = conv.mix.step(ACT_TARGET);
            let y_scale = conv.y.step(ACT_TARGET);
            let taps: Vec<f32> = conv.weights.taps.data().to_vec();
            let blob = shortconv::build(
                &taps,
                dims.hidden,
                ActivationScales::cli(mix_scale, y_scale),
            )
            .expect("shortconv");
            let file = format!("conv_{index:02}.blob");
            conv_blobs.push((file.clone(), blob.blob.clone()));
            PackConv {
                in_proj: pack_gemv(
                    &format!("conv_{index:02}_in"),
                    conv.input.step(ACT_TARGET),
                    mix_scale,
                ),
                conv_file: file,
                out_proj: pack_gemv(
                    &format!("conv_{index:02}_out"),
                    y_scale,
                    conv.out.step(ACT_TARGET),
                ),
                mix_scale,
                y_scale,
                ksh: blob.ksh,
                reps: 1,
            }
        });
        let attention = layer.attention.as_ref().map(|attention| {
            let input_scale = attention.input.step(ACT_TARGET);
            PackAttention {
                query: pack_gemv(
                    &format!("attn_{index:02}_q"),
                    input_scale,
                    attention.q.step(ACT_TARGET),
                ),
                key: pack_gemv(
                    &format!("attn_{index:02}_k"),
                    input_scale,
                    attention.k.step(ACT_TARGET),
                ),
                value: pack_gemv(
                    &format!("attn_{index:02}_v"),
                    input_scale,
                    attention.v.step(ACT_TARGET),
                ),
                output: pack_gemv(
                    &format!("attn_{index:02}_o"),
                    attention.attended.step(127.0),
                    attention.out.step(ACT_TARGET),
                ),
                gate: attention.weights.gate.as_ref().map(|_| {
                    pack_gemv(
                        &format!("attn_{index:02}_gate"),
                        input_scale,
                        attention.gate.step(ACT_TARGET),
                    )
                }),
                query_norm: attention.query_norm.clone(),
                key_norm: attention.key_norm.clone(),
                attn_q_scale: attention.q_post.multiplier(127.0),
                attn_out_scale: 1.0 / 4096.0,
                o_input_scale: attention.attended.step(127.0),
                v_channel: attention
                    .v_channel
                    .iter()
                    .map(|max| if *max > 0.0 { max / 127.0 } else { 1.0 })
                    .collect(),
                reps: 1,
            }
        });
        pack_layers.push(PackLayer {
            kind: if conv.is_some() {
                "conv".to_string()
            } else {
                "attention".to_string()
            },
            attention_kind: layer
                .attention
                .as_ref()
                .map(|attention| match attention.kind {
                    AttentionKind::Full => "full".to_string(),
                    AttentionKind::Sliding => "sliding".to_string(),
                }),
            ffn,
            conv,
            attention,
        });
        print!("\rplanned layer {}/{}", index + 1, dims.layers);
        let _ = std::io::stdout().flush();
    }
    println!();

    let norms = layers
        .iter()
        .map(|layer| PackNorms {
            attention: layer.norms[0].clone(),
            post_attention: layer.norms[1].clone(),
            post_ffn: layer.norms[3].clone(),
            ffn: layer.norms[2].clone(),
        })
        .collect();
    let embed_info = GgufFile::open(&args.model)
        .expect("gguf")
        .tensor("token_embd.weight")
        .expect("embedding")
        .ggml_type;
    let pack = ModelPack {
        version: PACK_VERSION,
        model: args.model.display().to_string(),
        dims: PackDims {
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
        norm_eps: spec.norm_eps,
        tied_embeddings: spec.tied_embeddings,
        rope_full: rope_of(&spec.rope.full),
        rope_sliding: spec.rope.sliding.as_ref().map(rope_of),
        layers: pack_layers,
        embed: PackEmbed {
            file: "embed.bin".to_string(),
            dtype: embed_type_name(embed_info),
            rows: dims.vocab,
            cols: dims.hidden,
        },
        lm_head: PackLmHead {
            file: "lm_head.gemv".to_string(),
            input_scale: head.input.step(ACT_TARGET),
            output_scale: head.logits.step(ACT_TARGET),
            reps: 1,
        },
        norms,
        output_norm: load_vector(
            &GgufFile::open(&args.model).expect("gguf"),
            "output_norm.weight",
        ),
    };
    (pack, conv_blobs)
}

fn pack_gemv(name: &str, input_scale: f32, output_scale: f32) -> PackGemv {
    PackGemv {
        file: format!("{name}.gemv"),
        input_scale,
        output_scale,
    }
}

fn embed_type_name(kind: lfm_gguf::GgmlType) -> String {
    match kind {
        lfm_gguf::GgmlType::F32 => "F32",
        lfm_gguf::GgmlType::F16 => "F16",
        lfm_gguf::GgmlType::BF16 => "BF16",
        lfm_gguf::GgmlType::Q4_K => "Q4_K",
        lfm_gguf::GgmlType::Q6_K => "Q6_K",
        lfm_gguf::GgmlType::Q8_0 => "Q8_0",
        other => panic!("embedding type {other:?} is not supported"),
    }
    .to_string()
}

fn pack_blobs(
    args: &Args,
    spec: &ModelSpec,
    layers: &[LayerAcc],
    pack: &ModelPack,
    conv_blobs: &[(String, Vec<u8>)],
) -> u64 {
    let _ = spec;
    let mut written = 0u64;
    for (index, layer) in layers.iter().enumerate() {
        let pack_layer = &pack.layers[index];
        for (matrix, gemv) in [
            (&layer.ffn.weights.gate, &pack_layer.ffn.gate),
            (&layer.ffn.weights.up, &pack_layer.ffn.up),
            (&layer.ffn.weights.down, &pack_layer.ffn.down),
        ] {
            written += write_gemv(&args.out, gemv, matrix, "ffn");
        }
        if let (Some(conv), Some(pack_conv)) = (&layer.conv, &pack_layer.conv) {
            written += write_gemv(&args.out, &pack_conv.in_proj, &conv.weights.in_proj, "conv");
            written += write_gemv(
                &args.out,
                &pack_conv.out_proj,
                &conv.weights.out_proj,
                "conv",
            );
        }
        if let (Some(attention), Some(pack_attention)) = (&layer.attention, &pack_layer.attention) {
            written += write_gemv(
                &args.out,
                &pack_attention.query,
                &attention.weights.query,
                "attn",
            );
            written += write_gemv(
                &args.out,
                &pack_attention.key,
                &attention.weights.key,
                "attn",
            );
            written += write_gemv(
                &args.out,
                &pack_attention.value,
                &attention.weights.value,
                "attn",
            );
            written += write_gemv(
                &args.out,
                &pack_attention.output,
                &attention.weights.output,
                "attn",
            );
            if let (Some(gate), Some(pack_gate)) = (&attention.weights.gate, &pack_attention.gate) {
                written += write_gemv(&args.out, pack_gate, gate, "attn");
            }
        }
        print!("\rpacked layer {}/{}", index + 1, layers.len());
        let _ = std::io::stdout().flush();
    }
    println!();

    for (file, blob) in conv_blobs {
        let path = args.out.join(file);
        std::fs::write(&path, blob).expect("write conv blob");
        written += blob.len() as u64;
    }

    let gguf = GgufFile::open(&args.model).expect("gguf");
    let head_bytes = gguf.tensor_bytes("token_embd.weight").expect("embedding");
    let embed_path = args.out.join(&pack.embed.file);
    std::fs::write(&embed_path, head_bytes).expect("write embed.bin");
    written += head_bytes.len() as u64;

    let embedding = load_matrix(&args.model, "token_embd.weight");
    let head = PackGemv {
        file: pack.lm_head.file.clone(),
        input_scale: pack.lm_head.input_scale,
        output_scale: pack.lm_head.output_scale,
    };
    written += write_gemv(&args.out, &head, &embedding, "lm_head");
    written
}

fn write_gemv(dir: &Path, packed: &PackGemv, matrix: &Matrix, kind: &str) -> u64 {
    let meta = PackMeta {
        tensor: format!("{kind}:{}", packed.file),
        source: None,
    };
    let blob = gemv::build(matrix, &meta).expect("pack gemv");
    let manifest = GemvManifest::from_blob(&meta, &blob);
    let path = dir.join(&packed.file);
    std::fs::write(&path, &blob.blob).expect("write gemv");
    manifest
        .write(&path.with_extension("json"))
        .expect("write gemv manifest");
    println!(
        "  {}: {}x{} cos {:.6}, {:.2} MiB",
        packed.file,
        matrix.n(),
        matrix.k(),
        blob.quality.cos_total,
        blob.blob.len() as f64 / (1u64 << 20) as f64
    );
    blob.blob.len() as u64
}
