//! On-device check of one conv layer and one attention layer through `exec`.
//!
//! The fixture is produced by `examples/make_layer_fixture.rs` on the host
//! from the real LFM3 checkpoint and the CPU oracle. This binary stages the
//! packed weights, runs the layers and compares each f32 output.
//!
//! ```text
//! adb push layer-fixture /data/local/tmp/layer-fixture
//! adb shell ADSP_LIBRARY_PATH=/data/local/tmp/lfmgemv \
//!     /data/local/tmp/lfmgemv/dsp-layer-check /data/local/tmp/layer-fixture
//! ```

#[cfg(target_os = "android")]
fn main() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    use myelin_core::backend::dsp::{
        AttentionFullStage, AttentionStage, ConvStage, DspBackend, VQuant,
    };
    use myelin_core::backend::AttentionCall;
    use myelin_core::kv::PagedKvCache;
    use myelin_core::spec::{
        AttentionKind, ChatTemplateSpec, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec, Quant,
        RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
    };
    use myelin_core::{ops, Backend};
    use myelin_npu::gemv_weights::{stage_blob, stage_zero, PackedGemv};
    use myelin_npu::layer_fixture::{compare, read_f32, FixtureRope, LayerFixture};
    use myelin_npu::{
        Cdsprpc, CdsprpcConfig, Executor, Handle, HvxClient, SharedBuffer, Transport,
        TransportError,
    };

    /// Delegating transport that counts FastRPC calls, so the check can print
    /// the real per-token RPC budget instead of estimating it.
    struct CountingTransport {
        inner: Arc<dyn Transport>,
        calls: AtomicUsize,
        maps: AtomicUsize,
    }

    impl CountingTransport {
        fn new(inner: Arc<dyn Transport>) -> Self {
            Self {
                inner,
                calls: AtomicUsize::new(0),
                maps: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }

        fn maps(&self) -> usize {
            self.maps.load(Ordering::Relaxed)
        }
    }

    impl Transport for CountingTransport {
        fn open(&self, module: &str) -> Result<Handle, TransportError> {
            self.inner.open(module)
        }

        fn close(&self, handle: Handle) -> Result<(), TransportError> {
            self.inner.close(handle)
        }

        fn invoke(
            &self,
            handle: Handle,
            call: myelin_npu::transport::Call<'_, '_>,
        ) -> Result<(), TransportError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if call.method == myelin_npu::method::method_id::MAP_ARENA {
                self.maps.fetch_add(1, Ordering::Relaxed);
            }
            self.inner.invoke(handle, call)
        }

        fn alloc_shared(&self, len: usize) -> Result<SharedBuffer, TransportError> {
            self.inner.alloc_shared(len)
        }
    }

    fn rope(spec: &FixtureRope) -> RopeSpec {
        RopeSpec {
            kind: if spec.kind == "partial" {
                RopeKind::Partial {
                    rotated_dim: spec.rotated_dim,
                }
            } else {
                RopeKind::Full
            },
            theta: spec.theta,
            dim: spec.dim,
        }
    }

    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/local/tmp/layer-fixture".to_string());
    let dir = std::path::Path::new(&dir);
    let fixture = LayerFixture::read(dir).expect("read fixture");
    let dims = Dims {
        hidden: fixture.dims.hidden,
        ffn: fixture.dims.ffn,
        layers: fixture.dims.layers,
        heads: fixture.dims.heads,
        kv_heads: fixture.dims.kv_heads,
        head_dim: fixture.dims.head_dim,
        vocab: fixture.dims.vocab,
        conv_cache: fixture.dims.conv_cache,
        sliding_window: fixture.dims.sliding_window,
    };
    let layers = fixture
        .layer_kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "conv" => LayerSpec::Conv { ksize: 7 },
            _ => LayerSpec::Attention {
                kind: if fixture.attention_kind == "sliding" {
                    AttentionKind::Sliding
                } else {
                    AttentionKind::Full
                },
            },
        })
        .collect();
    let spec = ModelSpec {
        family: ModelFamily::Lfm3,
        dims,
        rope: RopeConfig {
            full: rope(&fixture.rope_full),
            sliding: fixture.rope_sliding.as_ref().map(rope),
        },
        layers,
        kv_format: KvFormat::F16,
        tokenizer: TokenizerSpec {
            model: String::new(),
            vocab_size: fixture.dims.vocab,
            bos_token_id: None,
            eos_token_id: None,
        },
        chat_template: ChatTemplateSpec {
            template: String::new(),
        },
        weight_format: WeightFormat::from_quant(Quant::Q4K),
        norm_eps: fixture.norm_eps,
        tied_embeddings: true,
    };

    let transport: Arc<dyn Transport> =
        Arc::new(Cdsprpc::load(CdsprpcConfig::default()).expect("load libcdsprpc.so"));
    let counted = Arc::new(CountingTransport::new(Arc::clone(&transport)));
    let counted_dyn: Arc<dyn Transport> = counted.clone();
    let client = HvxClient::open(counted_dyn, "lfmgemv").expect("open lfmgemv");
    let hidden = dims.hidden;
    let kv_dim = dims.kv_dim();
    let slots = fixture.attention.cache_pages * 32;
    let chunk = 4 * 1024 * 1024;

    let stage = |name: &str, act: f32, out: f32| {
        let packed = PackedGemv::read(
            name.to_string(),
            dir.join(format!("{name}.json")),
            dir.join(format!("{name}.gemv")),
        )
        .expect("read gemv");
        let handle = packed
            .stage(&client, act, out, chunk)
            .unwrap_or_else(|error| panic!("stage {name}: {error}"));
        println!("stage {name:<14} handle {handle}");
        handle
    };
    let conv_in_proj = stage(
        "conv_in_proj",
        1.0 / fixture.conv.act_scale,
        fixture.conv.mix_scale,
    );
    let conv_out_proj = stage(
        "conv_out_proj",
        fixture.conv.y_scale,
        fixture.conv.out_scale,
    );
    let query = stage(
        "attn_q",
        1.0 / fixture.attention.act_scale,
        fixture.attention.q_scale,
    );
    let key = stage(
        "attn_k",
        1.0 / fixture.attention.act_scale,
        fixture.attention.k_scale,
    );
    let value = stage(
        "attn_v",
        1.0 / fixture.attention.act_scale,
        fixture.attention.v_scale,
    );
    let output = stage(
        "attn_o",
        1.0 / fixture.attention.o_act_scale,
        fixture.attention.out_scale,
    );
    let gate = if fixture.attention.has_gate {
        Some(stage(
            "attn_gate",
            1.0 / fixture.attention.act_scale,
            fixture.attention.gate_scale,
        ))
    } else {
        None
    };
    let conv_blob = stage_blob(
        &client,
        &std::fs::read(dir.join("conv.blob")).expect("conv blob"),
        chunk,
    )
    .expect("stage conv blob");
    let keys = stage_zero(&client, dims.kv_heads * slots * dims.head_dim, chunk).expect("keys");
    let key_scales = stage_zero(&client, dims.kv_heads * slots * 4, chunk).expect("key scales");
    let values = stage_zero(&client, dims.kv_heads * slots * dims.head_dim, chunk).expect("values");

    // The backend owns the handle from here; the client must not close it.
    let handle = client.handle().expect("handle");
    std::mem::forget(client);
    let executor_transport: Arc<dyn Transport> = counted.clone();
    let executor = Arc::new(Executor::new(executor_transport));
    let exec_dyn: Arc<dyn myelin_core::backend::dsp::DspExecutor> = executor.clone();
    let mut backend = DspBackend::with_handle(spec, exec_dyn, handle, "lfmgemv");

    backend
        .stage_conv(
            fixture.conv_layer,
            ConvStage {
                in_proj: conv_in_proj,
                conv: conv_blob,
                out_proj: conv_out_proj,
                ksh: fixture.conv.ksh,
                act_scale: fixture.conv.act_scale,
                mix_scale: fixture.conv.mix_scale,
                y_scale: fixture.conv.y_scale,
                out_scale: fixture.conv.out_scale,
                reps: 1,
                state_off: fixture.conv.state_off,
            },
        )
        .expect("stage conv");

    let q_norm = read_f32(&dir.join("attn_q_norm.f32"), dims.head_dim).expect("q norm");
    let k_norm = read_f32(&dir.join("attn_k_norm.f32"), dims.head_dim).expect("k norm");
    let v_scales = read_f32(&dir.join("attn_v_scales.f32"), dims.kv_dim()).expect("v scales");
    backend
        .stage_attention_full(
            fixture.attention_layer,
            AttentionFullStage {
                caches: AttentionStage {
                    keys,
                    key_scales,
                    values,
                    value_scales: 0,
                    q_scale: 1.0,
                    out_scale: 1.0,
                    reps: 1,
                    cache_pages: fixture.attention.cache_pages,
                    preprojected: true,
                },
                query,
                key,
                value,
                output,
                gate,
                query_norm: q_norm,
                key_norm: k_norm,
                act_scale: fixture.attention.act_scale,
                q_scale: fixture.attention.q_scale,
                k_scale: fixture.attention.k_scale,
                v_scale: fixture.attention.v_scale,
                gate_scale: fixture.attention.gate_scale,
                attn_q_scale: fixture.attention.attn_q_scale,
                attn_out_scale: fixture.attention.attn_out_scale,
                o_act_scale: fixture.attention.o_act_scale,
                out_scale: fixture.attention.out_scale,
                v_quant: VQuant::Channel(v_scales),
                reps: 1,
            },
        )
        .expect("stage attention");

    let conv_tokens = fixture.conv_tokens;
    let conv_x = read_f32(&dir.join("conv_x.f32"), conv_tokens * hidden).expect("conv x");
    let conv_y = read_f32(&dir.join("conv_y.f32"), conv_tokens * hidden).expect("conv y");
    let mut state = vec![0.0f32; (dims.conv_cache - 1) * hidden];
    let mut out = vec![0.0f32; hidden];
    let conv_calls0 = counted.calls();
    let conv_maps0 = counted.maps();
    let started = Instant::now();
    let mut conv_cos = Vec::new();
    let mut conv_max = 0.0f64;
    let mut conv_actual = Vec::with_capacity(conv_tokens * hidden);
    for position in 0..conv_tokens {
        let input = &conv_x[position * hidden..(position + 1) * hidden];
        let token_started = Instant::now();
        backend
            .conv_layer(fixture.conv_layer, input, &mut state, &mut out)
            .unwrap_or_else(|error| panic!("conv layer: {error} ({:?})", backend.last_error()));
        conv_actual.extend_from_slice(&out);
        println!(
            "  conv token {position}: dsp {} us wall {:.0} us",
            backend.take_exec_micros(),
            token_started.elapsed().as_secs_f64() * 1e6
        );
    }
    let conv_wall = started.elapsed();
    let conv_calls = counted.calls() - conv_calls0;
    let conv_maps = counted.maps() - conv_maps0;
    println!(
        "conv rpc: {conv_calls} calls for {conv_tokens} tokens ({conv_maps} map_arena), {:.2} per token",
        conv_calls as f64 / conv_tokens as f64
    );
    for position in 0..conv_tokens {
        let expected = &conv_y[position * hidden..(position + 1) * hidden];
        let actual = &conv_actual[position * hidden..(position + 1) * hidden];
        let (cos, max) = compare(expected, actual);
        conv_cos.push((position, cos));
        conv_max = conv_max.max(max);
    }

    // Reset must give a fresh DSP sequence: a second run of the same tokens
    // has to reproduce the first one bit for bit, otherwise the CONV ring
    // state leaks between sequences.
    let mut conv_reset = Vec::with_capacity(conv_tokens * hidden);
    backend.reset().expect("reset conv state");
    state.fill(0.0);
    for position in 0..conv_tokens {
        let input = &conv_x[position * hidden..(position + 1) * hidden];
        backend
            .conv_layer(fixture.conv_layer, input, &mut state, &mut out)
            .unwrap_or_else(|error| panic!("conv layer after reset: {error}"));
        conv_reset.extend_from_slice(&out);
    }
    let conv_reset_mismatches = conv_reset
        .iter()
        .zip(&conv_actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("conv reset identity: mismatches={conv_reset_mismatches}");

    let attn_tokens = fixture.attention_tokens;
    let attn_x = read_f32(&dir.join("attn_x.f32"), attn_tokens * hidden).expect("attn x");
    let attn_y = read_f32(&dir.join("attn_y.f32"), attn_tokens * hidden).expect("attn y");
    let kind = if fixture.attention_kind == "sliding" {
        AttentionKind::Sliding
    } else {
        AttentionKind::Full
    };
    let mut kv = PagedKvCache::new(kv_dim, attn_tokens).expect("kv");
    let mut attn_actual = Vec::with_capacity(attn_tokens * hidden);
    let attn_calls0 = counted.calls();
    let attn_maps0 = counted.maps();
    let started = Instant::now();
    for position in 0..attn_tokens {
        let input = &attn_x[position * hidden..(position + 1) * hidden];
        let start = ops::window_start(kind, position, dims.sliding_window);
        let token_started = Instant::now();
        backend
            .attention_layer(
                AttentionCall {
                    layer: fixture.attention_layer,
                    position,
                    start,
                    kind,
                    input,
                },
                &mut kv,
                &mut out,
            )
            .unwrap_or_else(|error| {
                panic!("attention layer: {error} ({:?})", backend.last_error())
            });
        attn_actual.extend_from_slice(&out);
        println!(
            "  attn token {position}: dsp {} us wall {:.0} us",
            backend.take_exec_micros(),
            token_started.elapsed().as_secs_f64() * 1e6
        );
    }
    let attn_wall = started.elapsed();
    let attn_calls = counted.calls() - attn_calls0;
    let attn_maps = counted.maps() - attn_maps0;
    println!(
        "attn rpc: {attn_calls} calls for {attn_tokens} tokens ({attn_maps} map_arena), {:.2} per token",
        attn_calls as f64 / attn_tokens as f64
    );
    let mut attn_cos = Vec::new();
    let mut attn_max = 0.0f64;
    for position in 0..attn_tokens {
        let expected = &attn_y[position * hidden..(position + 1) * hidden];
        let actual = &attn_actual[position * hidden..(position + 1) * hidden];
        let (cos, max) = compare(expected, actual);
        attn_cos.push((position, cos));
        attn_max = attn_max.max(max);
    }

    println!(
        "conv layer {}: {} tokens, wall {:.2} ms",
        fixture.conv_layer,
        conv_tokens,
        conv_wall.as_secs_f64() * 1e3
    );
    for (position, cos) in &conv_cos {
        println!("  token {position}: cos {cos:.9}");
    }
    let worst = conv_actual
        .iter()
        .zip(&conv_y)
        .enumerate()
        .max_by(|a, b| {
            (a.1 .0 - a.1 .1)
                .abs()
                .partial_cmp(&(b.1 .0 - b.1 .1).abs())
                .unwrap()
        })
        .unwrap();
    println!(
        "  max |delta| {conv_max:.6e} at {}: expected {:.6} actual {:.6}",
        worst.0, worst.1 .1, worst.1 .0
    );

    println!(
        "attention layer {}: {} tokens, wall {:.2} ms",
        fixture.attention_layer,
        attn_tokens,
        attn_wall.as_secs_f64() * 1e3
    );
    for (position, cos) in &attn_cos {
        println!("  token {position}: cos {cos:.9}");
    }
    let worst = attn_actual
        .iter()
        .zip(&attn_y)
        .enumerate()
        .max_by(|a, b| {
            (a.1 .0 - a.1 .1)
                .abs()
                .partial_cmp(&(b.1 .0 - b.1 .1).abs())
                .unwrap()
        })
        .unwrap();
    println!(
        "  max |delta| {attn_max:.6e} at {}: expected {:.6} actual {:.6}",
        worst.0, worst.1 .1, worst.1 .0
    );

    // The conv/attention output scales are global per matrix; the last token
    // is the generation step and the one the runtime produces, so the verdict
    // is on it. Earlier tokens' cos is reported for the dynamic-range picture.
    let conv_ok = conv_cos.last().is_some_and(|(_, cos)| *cos >= 0.95);
    let attn_ok = attn_cos.last().is_some_and(|(_, cos)| *cos >= 0.90);
    if conv_ok && attn_ok && conv_reset_mismatches == 0 {
        println!("VERDICT: PASS");
    } else {
        println!("VERDICT: FAIL");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("dsp-layer-check runs on Android only (needs libcdsprpc.so)");
}
