//! Host tests for the resident weight planner.
//!
//! Everything runs on [`MockTransport`]; the real LFM3 tensor test is
//! `#[ignore]` and reads `models/LFM3-1.7B-Q4_K_M.gguf` when it is present.

use std::path::PathBuf;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use myelin_core::backend::dsp::{DspBackend, DspError, DspExecutor, DspHandle};
use myelin_core::{Backend, BackendError, ModelSpec};
use myelin_npu::method::method_id;
use myelin_npu::weights::{
    AttentionFullPlan, AttentionParams, AttentionPlan, ConvPlan, FfnPlan, PackedMatrix,
    PackedTensor, ResidentImage, ResidentWeights, StageKind, WeightBudget, WeightError, WeightPlan,
};
use myelin_npu::{Executor, MockReply, MockTransport, Transport, TransportError};
use myelin_pack::manifest::PackMeta;
use myelin_pack::{pack, pack_gguf, DType, Matrix, PackOptions};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn matrix(k: usize, n: usize, seed: u64) -> Matrix {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(k * n);
    for _channel in 0..n {
        for _index in 0..k {
            data.push(rng.uniform() * 2.0 - 1.0);
        }
    }
    Matrix::new(k, n, data).expect("matrix")
}

fn options() -> PackOptions {
    PackOptions { dtype: DType::I8 }
}

fn packed(name: &str, k: usize, n: usize, seed: u64) -> PackedTensor {
    let weights = matrix(k, n, seed);
    let meta = PackMeta {
        tensor: name.to_owned(),
        source: None,
    };
    let result = pack(&weights, &meta, options()).expect("pack");
    PackedTensor::from_pack(name, &result).expect("packed tensor")
}

fn matrix_of(name: &str, k: usize, n: usize, seed: u64) -> PackedMatrix {
    PackedMatrix::from_tensor(packed(name, k, n, seed)).expect("packed matrix")
}

fn ffn(k: usize, n: usize, seed: u64) -> FfnPlan {
    FfnPlan::new(
        matrix_of("gate", k, n, seed),
        matrix_of("up", k, n, seed + 1),
        matrix_of("down", n, k, seed + 2),
        0.5,
        1.0,
        0.5,
        1,
    )
    .expect("ffn plan")
}

fn allocator(mock: &MockTransport, first: i32) -> Arc<AtomicI32> {
    let next = Arc::new(AtomicI32::new(first));
    let counter = Arc::clone(&next);
    mock.set_handler(method_id::ALLOC_BUF, move |call| {
        assert_eq!(call.prim_in.len(), 4);
        assert_eq!(call.prim_out_len, 4);
        let handle = counter.fetch_add(1, Ordering::SeqCst);
        Ok(MockReply {
            prim_out: handle.to_le_bytes().to_vec(),
            out_bufs: Vec::new(),
        })
    });
    next
}

struct Bench {
    mock: MockTransport,
    transport: Arc<dyn Transport>,
    executor: Arc<dyn DspExecutor>,
    handle: DspHandle,
}

fn bench() -> Bench {
    let mock = MockTransport::new();
    let transport: Arc<dyn Transport> = Arc::new(mock.clone());
    let executor: Arc<dyn DspExecutor> = Arc::new(Executor::new(Arc::clone(&transport)));
    let handle = executor.open("lfmgemv").expect("open");
    Bench {
        mock,
        transport,
        executor,
        handle,
    }
}

impl Bench {
    fn planner(&self, plan: WeightPlan) -> ResidentWeights {
        ResidentWeights::new(Arc::clone(&self.executor), self.handle, plan)
    }

    fn methods(&self) -> Vec<u32> {
        self.mock.calls().iter().map(|call| call.method).collect()
    }

    fn calls(&self, method: u32) -> Vec<myelin_npu::RecordedCall> {
        self.mock
            .calls()
            .into_iter()
            .filter(|call| call.method == method)
            .collect()
    }
}

fn read_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn expected_scale_payload(tensor: &PackedMatrix, act: f32, out: f32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(tensor.n_padded() as i32).to_le_bytes());
    payload.extend_from_slice(&(tensor.k_padded() as i32).to_le_bytes());
    payload.extend_from_slice(&act.to_le_bytes());
    payload.extend_from_slice(&out.to_le_bytes());
    for scale in tensor.scales() {
        payload.extend_from_slice(&scale.to_le_bytes());
    }
    payload
}

#[test]
fn ffn_upload_matches_the_stand_order() {
    let bench = bench();
    allocator(&bench.mock, 10);
    let layer = ffn(64, 32, 0x5eed);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(WeightBudget::new(bytes))
        .ffn(0, layer.clone())
        .build();
    let planner = bench.planner(plan);

    let stage = planner.ffn_stage(0).expect("ffn stage");
    assert_eq!((stage.gate, stage.up, stage.down), (10, 11, 12));
    assert_eq!(stage.act_scale, 2.0);
    assert_eq!(stage.out_scale, 0.5);
    assert_eq!(stage.reps, 1);

    let expected = [
        method_id::ALLOC_BUF,
        method_id::SET_SHAPE,
        method_id::LOAD_BUF,
        method_id::SET_SCALE,
    ];
    let mut sequence = Vec::new();
    for _ in 0..3 {
        sequence.extend_from_slice(&expected);
    }
    assert_eq!(bench.methods(), sequence);

    let allocs = bench.calls(method_id::ALLOC_BUF);
    let shapes = bench.calls(method_id::SET_SHAPE);
    let loads = bench.calls(method_id::LOAD_BUF);
    let scales = bench.calls(method_id::SET_SCALE);
    for (index, tensor) in [layer.gate(), layer.up(), layer.down()].iter().enumerate() {
        let handle = 10 + index as i32;
        assert_eq!(read_i32(&allocs[index].prim_in), tensor.len() as i32);
        assert_eq!(
            shapes[index].prim_in,
            [
                &handle.to_le_bytes()[..],
                &(tensor.n_padded() as i32).to_le_bytes()[..],
                &(tensor.k_padded() as i32).to_le_bytes()[..],
                &1i32.to_le_bytes()[..],
            ]
            .concat()
        );
        assert_eq!(read_i32(&loads[index].prim_in[0..4]), handle);
        assert_eq!(read_i32(&loads[index].prim_in[4..8]), 0);
        assert_eq!(
            read_i32(&loads[index].prim_in[8..12]) as usize,
            tensor.weights().len(),
            "load_buf packs dataLen as _primIn[3]"
        );
        assert_eq!(loads[index].in_bufs.len(), 1);
        assert_eq!(loads[index].in_bufs[0], tensor.weights());
        assert_eq!(read_i32(&scales[index].prim_in[0..4]), handle);
        assert_eq!(
            read_i32(&scales[index].prim_in[4..8]) as usize,
            scales[index].in_bufs[0].len(),
            "set_scale packs scalesLen as _primIn[2]"
        );
        assert_eq!(
            scales[index].in_bufs[0],
            expected_scale_payload(
                tensor,
                if index < 2 { 0.5 * 65536.0 } else { 1.0 },
                if index < 2 { 1.0 } else { 0.5 },
            )
        );
        assert_eq!(scales[index].in_bufs[0].len(), 16 + 4 * tensor.n_padded());
    }

    let stats = planner.stats();
    assert_eq!(stats.loads, 1);
    assert_eq!(stats.load_calls, 3);
    assert_eq!(stats.load_bytes, bytes as u64);
    assert_eq!(stats.chunk, myelin_npu::LOAD_CHUNK);
    assert_eq!(stats.strategy, myelin_npu::LoadStrategy::Copy);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.live_bytes, bytes);
    assert_eq!(stats.resident_units, 1);
}

#[test]
fn default_chunk_is_four_mebibytes() {
    assert_eq!(myelin_npu::LOAD_CHUNK, 4 * 1024 * 1024);
    let tensor = packed("gate", 64, 32, 5);
    assert_eq!(tensor.load_calls(), tensor.load_calls_with(4 * 1024 * 1024));
    assert_eq!(tensor.load_calls(), 1);
    assert_eq!(tensor.load_calls_with(tensor.len()), 1);
    assert_eq!(tensor.load_calls_with(tensor.len() - 1), 2);
}

#[test]
fn loads_follow_the_configured_chunking_and_offsets() {
    let bench = bench();
    allocator(&bench.mock, 1);
    let layer = ffn(512, 512, 0x1234);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(bytes).ffn(0, layer.clone()).build();
    let mut planner = bench.planner(plan);
    planner.set_chunk(64 * 1024);

    planner.ffn_stage(0).expect("ffn stage");

    let loads = bench.calls(method_id::LOAD_BUF);
    let gate = layer.gate();
    let expected_calls = gate.load_calls_with(64 * 1024);
    assert!(expected_calls > 1, "test must exercise chunking");
    assert_eq!(loads.len(), expected_calls * 3);
    for (index, tensor) in [layer.gate(), layer.up(), layer.down()].iter().enumerate() {
        assert_eq!(tensor.load_calls_with(64 * 1024), expected_calls);
        for chunk in 0..expected_calls {
            let call = &loads[index * expected_calls + chunk];
            assert_eq!(read_i32(&call.prim_in[4..8]) as usize, chunk * 64 * 1024);
            assert_eq!(
                read_i32(&call.prim_in[8..12]) as usize,
                call.in_bufs[0].len()
            );
            let start = chunk * 64 * 1024;
            let end = (start + 64 * 1024).min(tensor.len());
            assert_eq!(call.in_bufs[0], &tensor.weights()[start..end]);
        }
    }
    let stats = planner.stats();
    assert_eq!(stats.load_calls, (expected_calls * 3) as u64);
    assert_eq!(stats.load_bytes, bytes as u64);
    assert_eq!(stats.chunk, 64 * 1024);
    assert_eq!(stats.live_bytes, bytes);
}

#[test]
fn attach_strategy_is_rejected_explicitly() {
    let bench = bench();
    allocator(&bench.mock, 1);
    let layer = ffn(64, 32, 61);
    let plan = WeightPlan::builder(layer.bytes()).ffn(0, layer).build();
    let mut planner = bench.planner(plan);
    planner.set_strategy(myelin_npu::LoadStrategy::Attach);
    let error = planner.ffn_stage(0).expect_err("attach is unsupported");
    match error {
        WeightError::UnsupportedStrategy { strategy, reason } => {
            assert_eq!(strategy, myelin_npu::LoadStrategy::Attach);
            assert!(reason.contains("shared buffers"));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(bench.calls(method_id::ALLOC_BUF).is_empty());
    assert!(bench.calls(method_id::LOAD_BUF).is_empty());
}

#[test]
fn budget_is_exact_and_one_byte_short_is_rejected() {
    let exact_bench = bench();
    allocator(&exact_bench.mock, 1);
    let layer = ffn(64, 32, 7);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(bytes).ffn(0, layer.clone()).build();
    let planner = exact_bench.planner(plan);
    planner.ffn_stage(0).expect("exact budget fits");
    assert_eq!(planner.stats().live_bytes, bytes);
    drop(planner);

    let short_bench = bench();
    allocator(&short_bench.mock, 1);
    let plan = WeightPlan::builder(bytes - 1).ffn(0, layer).build();
    let planner = short_bench.planner(plan);
    let error = planner.ffn_stage(0).expect_err("one byte short");
    match error {
        WeightError::Budget {
            layer,
            kind,
            bytes: needed,
            budget,
            used,
        } => {
            assert_eq!(layer, 0);
            assert_eq!(kind, StageKind::Ffn);
            assert_eq!(needed, bytes);
            assert_eq!(budget, bytes - 1);
            assert_eq!(used, 0);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(short_bench.calls(method_id::ALLOC_BUF).is_empty());
    let stats = planner.stats();
    assert_eq!(stats.loads, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.live_bytes, 0);
}

#[test]
fn lru_evicts_the_least_recently_used_layer() {
    let bench = bench();
    allocator(&bench.mock, 1);
    let layer = ffn(64, 32, 11);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(2 * bytes)
        .ffn(0, layer.clone())
        .ffn(1, layer.clone())
        .ffn(2, layer)
        .build();
    let planner = bench.planner(plan);

    planner.ffn_stage(0).expect("layer 0");
    planner.ffn_stage(1).expect("layer 1");
    assert_eq!(
        planner.resident_units(),
        vec![(0, StageKind::Ffn), (1, StageKind::Ffn)]
    );
    assert_eq!(planner.stats().live_bytes, 2 * bytes);

    planner.ffn_stage(0).expect("touch layer 0");
    planner.ffn_stage(2).expect("layer 2 evicts layer 1");

    assert_eq!(
        planner.resident_units(),
        vec![(0, StageKind::Ffn), (2, StageKind::Ffn)]
    );
    let stats = planner.stats();
    assert_eq!(stats.loads, 3);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.unloads, 1);
    assert_eq!(stats.live_bytes, 2 * bytes);
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), 3);
}

#[test]
fn drop_releases_every_resident_buffer() {
    let bench = bench();
    allocator(&bench.mock, 1);
    let layer = ffn(64, 32, 21);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(bytes * 2)
        .ffn(0, layer.clone())
        .ffn(1, layer)
        .build();
    let planner = bench.planner(plan);

    planner.ffn_stage(0).expect("layer 0");
    planner.ffn_stage(1).expect("layer 1");
    assert_eq!(planner.stats().live_bytes, 2 * bytes);

    planner.release_all();
    assert_eq!(planner.stats().live_bytes, 0);
    assert_eq!(planner.stats().resident_units, 0);
    let released = bench.calls(method_id::RELEASE_BUF).len();
    assert_eq!(released, 6);
    planner.release_all();
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), released);

    planner.ffn_stage(0).expect("reload after release");
    drop(planner);
    let allocs = bench.calls(method_id::ALLOC_BUF).len();
    let releases = bench.calls(method_id::RELEASE_BUF).len();
    assert_eq!(allocs, releases, "every allocation is released");
}

#[test]
fn transport_errors_keep_the_code_and_release_partial_allocs() {
    let bench = bench();
    allocator(&bench.mock, 1);
    bench.mock.set_handler(method_id::LOAD_BUF, |_| {
        Err(TransportError::Rpc {
            method: method_id::LOAD_BUF,
            code: 0x8000_040e_u32 as i32,
            name: "AEE_EBADPARM",
        })
    });
    let layer = ffn(64, 32, 31);
    let plan = WeightPlan::builder(layer.bytes()).ffn(0, layer).build();
    let planner = bench.planner(plan);

    let error = planner.ffn_stage(0).expect_err("load must fail");
    match error {
        WeightError::Transport(DspError::Rpc { method, code, name }) => {
            assert_eq!(method, method_id::LOAD_BUF);
            assert_eq!(code, 0x8000_040e_u32 as i32);
            assert_eq!(name, "AEE_EBADPARM");
        }
        other => panic!("unexpected {other:?}"),
    }
    let stats = planner.stats();
    assert_eq!(stats.loads, 0);
    assert_eq!(stats.live_bytes, 0);
    assert_eq!(stats.resident_units, 0);
    assert_eq!(bench.calls(method_id::ALLOC_BUF).len(), 1);
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), 1);
}

#[test]
fn set_scale_errors_release_completed_and_partial_tensors() {
    let bench = bench();
    allocator(&bench.mock, 10);
    bench.mock.set_handler(method_id::SET_SCALE, |call| {
        if read_i32(&call.prim_in) == 11 {
            Err(TransportError::Rpc {
                method: method_id::SET_SCALE,
                code: 0x8000_040e_u32 as i32,
                name: "AEE_EBADPARM",
            })
        } else {
            Ok(MockReply::default())
        }
    });
    let layer = ffn(64, 32, 35);
    let plan = WeightPlan::builder(layer.bytes()).ffn(0, layer).build();
    let planner = bench.planner(plan);

    let error = planner.ffn_stage(0).expect_err("up scale upload fails");
    match error {
        WeightError::Transport(DspError::Rpc { method, code, name }) => {
            assert_eq!(method, method_id::SET_SCALE);
            assert_eq!(code, 0x8000_040e_u32 as i32);
            assert_eq!(name, "AEE_EBADPARM");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(bench.calls(method_id::ALLOC_BUF).len(), 2);
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), 2);
    assert_eq!(planner.stats().live_bytes, 0);
    assert_eq!(planner.stats().resident_units, 0);
}

#[test]
fn attention_allocates_then_loads_in_stand_order() {
    let bench = bench();
    allocator(&bench.mock, 10);
    let keys = ResidentImage::new("keys", vec![0x11; 128]).expect("keys");
    let key_scales = ResidentImage::new("key_scales", vec![0x22; 16]).expect("scales");
    let values = ResidentImage::new("values", vec![0x33; 64]).expect("values");
    let attention = AttentionPlan::new(
        keys,
        key_scales,
        values,
        None,
        AttentionParams {
            q_scale: 2.0,
            out_scale: 0.25,
            reps: 1,
            cache_pages: 4,
            preprojected: true,
        },
    )
    .expect("attention plan");
    let plan = WeightPlan::builder(attention.bytes())
        .attention(0, attention)
        .build();
    let planner = bench.planner(plan);

    let stage = planner.attention_stage(0).expect("attention stage");
    assert_eq!(
        (
            stage.keys,
            stage.key_scales,
            stage.values,
            stage.value_scales
        ),
        (10, 11, 12, 0)
    );
    assert_eq!(stage.cache_pages, 4);
    assert!(stage.preprojected);

    assert_eq!(
        bench.methods(),
        vec![
            method_id::ALLOC_BUF,
            method_id::ALLOC_BUF,
            method_id::ALLOC_BUF,
            method_id::LOAD_BUF,
            method_id::LOAD_BUF,
            method_id::LOAD_BUF,
        ]
    );
    let loads = bench.calls(method_id::LOAD_BUF);
    // attn_main.c loads K, V, KT.
    assert_eq!(
        loads
            .iter()
            .map(|call| read_i32(&call.prim_in[0..4]))
            .collect::<Vec<_>>(),
        vec![10, 12, 11]
    );
    assert_eq!(
        loads
            .iter()
            .map(|call| call.in_bufs[0].len())
            .collect::<Vec<_>>(),
        vec![128, 64, 16]
    );
    assert_eq!(planner.stats().live_bytes, 208);
}

#[test]
fn packed_tensor_rejects_a_mismatched_blob() {
    let weights = matrix(64, 32, 99);
    let meta = PackMeta {
        tensor: "t".to_owned(),
        source: None,
    };
    let mut result = pack(&weights, &meta, options()).expect("pack");
    result.blob[200] ^= 0xff;
    let error = PackedTensor::new("t", result.manifest.clone(), result.blob).expect_err("checksum");
    assert!(matches!(error, WeightError::Layout(_)));
}

#[test]
fn backend_loads_an_ffn_stage_lazily_and_caches_it() {
    let spec = ModelSpec::placeholder();
    let hidden = spec.dims.hidden;
    let bench = bench();
    allocator(&bench.mock, 1);
    bench.mock.set_handler(method_id::MAP_ARENA, |_| {
        Ok(MockReply {
            prim_out: 77i32.to_le_bytes().to_vec(),
            ..Default::default()
        })
    });
    bench.mock.set_handler(method_id::EXEC, move |call| {
        assert_eq!(call.out_lens, vec![hidden, 32]);
        Ok(MockReply {
            prim_out: Vec::new(),
            out_bufs: vec![vec![1u8; hidden], vec![0u8; 32]],
        })
    });
    let mut backend =
        DspBackend::attach(spec.clone(), Arc::clone(&bench.executor), "lfmgemv").expect("attach");
    let handle = backend.handle().expect("handle");
    let layer = ffn(64, 32, 41);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(bytes).ffn(0, layer).build();
    backend.attach_weights(Box::new(ResidentWeights::new(
        Arc::clone(&bench.executor),
        handle,
        plan,
    )));
    assert!(backend.has_weight_provider());

    let input = vec![1.0f32; hidden];
    let mut out = vec![0.0f32; hidden];
    backend.ffn_layer(0, &input, &mut out).expect("lazy ffn");
    assert_eq!(bench.calls(method_id::ALLOC_BUF).len(), 3);
    assert_eq!(bench.calls(method_id::EXEC).len(), 1);
    assert!(out.iter().all(|value| (*value - 0.5).abs() < 1e-6));

    backend.ffn_layer(0, &input, &mut out).expect("cached ffn");
    assert_eq!(bench.calls(method_id::ALLOC_BUF).len(), 3);
    assert_eq!(bench.calls(method_id::EXEC).len(), 2);
    assert_eq!(bench.calls(method_id::MAP_ARENA).len(), 1);

    drop(backend);
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), 3);
}

#[test]
fn backend_reports_a_budget_failure_without_calling_ffn() {
    let spec = ModelSpec::placeholder();
    let bench = bench();
    allocator(&bench.mock, 1);
    let mut backend =
        DspBackend::attach(spec.clone(), Arc::clone(&bench.executor), "lfmgemv").expect("attach");
    let handle = backend.handle().expect("handle");
    let layer = ffn(64, 32, 51);
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(bytes - 1).ffn(0, layer).build();
    backend.attach_weights(Box::new(ResidentWeights::new(
        Arc::clone(&bench.executor),
        handle,
        plan,
    )));

    let input = vec![1.0f32; spec.dims.hidden];
    let mut out = vec![0.0f32; spec.dims.hidden];
    let error = backend
        .ffn_layer(0, &input, &mut out)
        .expect_err("does not fit");
    match error {
        BackendError::Budget(message) => assert!(message.contains("budget")),
        other => panic!("unexpected {other:?}"),
    }
    assert!(bench.calls(method_id::ALLOC_BUF).is_empty());
    assert!(bench.calls(method_id::FFN).is_empty());
}

#[test]
#[ignore = "reads models/LFM3-1.7B-Q4_K_M.gguf"]
fn real_lfm3_ffn_gate_footprint() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/LFM3-1.7B-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }

    let (gate_result, gate_weights) =
        pack_gguf(&path, "blk.0.ffn_gate.weight", None, options()).expect("pack gate");
    assert_eq!((gate_weights.k(), gate_weights.n()), (2048, 7680));
    let (up_result, _) = pack_gguf(&path, "blk.0.ffn_up.weight", None, options()).expect("pack up");
    let (down_result, down_weights) =
        pack_gguf(&path, "blk.0.ffn_down.weight", None, options()).expect("pack down");
    assert_eq!((down_weights.k(), down_weights.n()), (7680, 2048));

    let gate = PackedTensor::from_pack("blk.0.ffn_gate.weight", &gate_result).expect("gate");
    let up = PackedTensor::from_pack("blk.0.ffn_up.weight", &up_result).expect("up");
    let down = PackedTensor::from_pack("blk.0.ffn_down.weight", &down_result).expect("down");
    let gate_bytes = gate.len();
    let layer_bytes = gate_bytes + up.len() + down.len();
    let chunks = gate.load_calls() + up.load_calls() + down.load_calls();
    let layer = FfnPlan::new(
        PackedMatrix::from_tensor(gate).expect("gate"),
        PackedMatrix::from_tensor(up).expect("up"),
        PackedMatrix::from_tensor(down).expect("down"),
        0.5,
        1.0,
        0.5,
        1,
    )
    .expect("ffn plan");

    let bench = bench();
    allocator(&bench.mock, 1);
    let plan = WeightPlan::builder(layer_bytes).ffn(0, layer).build();
    let planner = bench.planner(plan);
    planner.ffn_stage(0).expect("residency");

    let stats = planner.stats();
    println!(
        "blk.0.ffn_gate: blob {gate_bytes} B, weights {} B, {} tiles; \
         one FFN layer {layer_bytes} B; {chunks} load_buf per layer \
         ({} for gate); resident {} B in {} units; load_calls {}",
        gate_result.manifest.weight_bytes,
        gate_result.manifest.tiles.len(),
        gate_bytes.div_ceil(myelin_npu::LOAD_CHUNK),
        stats.live_bytes,
        stats.resident_units,
        stats.load_calls
    );
    assert_eq!(stats.live_bytes, layer_bytes);
    assert_eq!(stats.load_calls, chunks as u64);
    assert_eq!(stats.evictions, 0);
    assert_eq!(
        bench.calls(method_id::ALLOC_BUF).len(),
        3,
        "one alloc_buf per matrix"
    );
}

fn int4_matrix(name: &str, n_padded: usize, k_padded: usize, seed: u8) -> PackedMatrix {
    let weights = (0..n_padded * (k_padded / 2))
        .map(|index| (index as u8).wrapping_add(seed))
        .collect();
    let scales = (0..n_padded).map(|row| 0.001 * (row + 1) as f32).collect();
    PackedMatrix::new(name, n_padded, k_padded, weights, scales).expect("packed matrix")
}

#[test]
fn plan_over_budget_is_refused_before_any_allocation() {
    let gate = int4_matrix("gate", 32, 256, 1);
    let up = int4_matrix("up", 32, 256, 2);
    let down = int4_matrix("down", 256, 32, 3);
    let layer = FfnPlan::new(gate, up, down, 0.5, 1.0, 0.5, 1).expect("ffn plan");
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(WeightBudget::new(bytes - 1))
        .ffn(0, layer)
        .build();
    match plan.check_budget() {
        Err(WeightError::PlanOverBudget {
            bytes: need,
            budget,
        }) => {
            assert_eq!(need, bytes);
            assert_eq!(budget, bytes - 1);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn attach_strategy_maps_ion_buffers_and_frees_them() {
    let bench = bench();
    let next = Arc::new(AtomicI32::new(40));
    let counter = Arc::clone(&next);
    bench.mock.set_handler(method_id::ALLOC_IN, move |_| {
        Ok(MockReply {
            prim_out: counter
                .fetch_add(1, Ordering::SeqCst)
                .to_le_bytes()
                .to_vec(),
            ..Default::default()
        })
    });
    bench
        .mock
        .set_handler(method_id::SET_SHAPE, |_| Ok(MockReply::default()));
    bench
        .mock
        .set_handler(method_id::SET_SCALE, |_| Ok(MockReply::default()));
    bench
        .mock
        .set_handler(method_id::RELEASE_BUF, |_| Ok(MockReply::default()));

    let gate = int4_matrix("gate", 32, 256, 1);
    let up = int4_matrix("up", 32, 256, 2);
    let down = int4_matrix("down", 256, 32, 3);
    let layer = FfnPlan::new(gate, up, down, 0.5, 1.0, 0.5, 1).expect("ffn plan");
    let bytes = layer.bytes();
    let plan = WeightPlan::builder(WeightBudget::new(bytes))
        .ffn(0, layer)
        .build();
    let planner = ResidentWeights::with_ion(
        Arc::clone(&bench.executor),
        bench.handle,
        plan,
        Arc::clone(&bench.transport),
    );

    let stage = planner.ffn_stage(0).expect("ffn stage");
    assert_eq!((stage.gate, stage.up, stage.down), (40, 41, 42));
    assert!(bench.mock.live_bytes() >= bytes, "ION buffers stay alive");
    assert_eq!(bench.calls(method_id::ALLOC_IN).len(), 3);
    assert!(bench.calls(method_id::ALLOC_BUF).is_empty());
    assert!(bench.calls(method_id::LOAD_BUF).is_empty());
    assert_eq!(planner.stats().strategy, myelin_npu::LoadStrategy::Attach);

    // The gate/up payload carries the FFN Q16 protocol; down the mid->out one.
    let scales = bench.calls(method_id::SET_SCALE);
    let gate_sa = f32::from_le_bytes(scales[0].in_bufs[0][8..12].try_into().unwrap());
    let gate_so = f32::from_le_bytes(scales[0].in_bufs[0][12..16].try_into().unwrap());
    let down_sa = f32::from_le_bytes(scales[2].in_bufs[0][8..12].try_into().unwrap());
    let down_so = f32::from_le_bytes(scales[2].in_bufs[0][12..16].try_into().unwrap());
    assert_eq!(gate_sa, 0.5 * 65536.0);
    assert_eq!(gate_so, 1.0);
    assert_eq!(down_sa, 1.0);
    assert_eq!(down_so, 0.5);

    drop(planner);
    assert_eq!(bench.mock.live_bytes(), 0, "ION buffers are released");
    assert_eq!(bench.calls(method_id::RELEASE_BUF).len(), 3);
}

#[test]
fn conv_stage_exposes_the_planned_scales() {
    let bench = bench();
    allocator(&bench.mock, 60);
    let plan = ConvPlan::new(
        int4_matrix("in", 96, 256, 1),
        ResidentImage::new("conv", vec![0u8; 256]).expect("image"),
        int4_matrix("out", 32, 256, 2),
        2.0,
        0.25,
        0.5,
        0.125,
        20,
        1,
        4096,
    )
    .expect("conv plan");
    let bytes = plan.bytes();
    let plan = WeightPlan::builder(bytes).conv(3, plan).build();
    let planner = bench.planner(plan);
    let stage = planner.conv_stage(3).expect("conv stage");
    assert_eq!(stage.in_proj, 60);
    assert_eq!(stage.out_proj, 62);
    assert_eq!(stage.ksh, 20);
    assert_eq!(stage.act_scale, 0.5);
    assert_eq!(stage.mix_scale, 0.25);
    assert_eq!(stage.y_scale, 0.5);
    assert_eq!(stage.out_scale, 0.125);
    assert_eq!(stage.state_off, 4096);
}

#[test]
fn full_attention_stage_orders_caches_before_projections() {
    let bench = bench();
    allocator(&bench.mock, 70);
    let caches = AttentionPlan::new(
        ResidentImage::new("k", vec![0u8; 256]).expect("k"),
        ResidentImage::new("kt", vec![0u8; 128]).expect("kt"),
        ResidentImage::new("v", vec![0u8; 256]).expect("v"),
        None,
        AttentionParams {
            q_scale: 1.0,
            out_scale: 1.0 / 4096.0,
            reps: 1,
            cache_pages: 1,
            preprojected: true,
        },
    )
    .expect("caches");
    let plan = AttentionFullPlan::new(
        caches,
        int4_matrix("q", 32, 256, 1),
        int4_matrix("k", 32, 256, 2),
        int4_matrix("v", 32, 256, 3),
        int4_matrix("o", 256, 32, 4),
        None,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        2.0,
        0.5,
        0.25,
        0.125,
        1.0,
        16.0,
        1.0 / 4096.0,
        0.75,
        0.0625,
        myelin_core::backend::VQuant::Channel(vec![0.5, 0.5]),
        1,
    )
    .expect("attention plan");
    let bytes = plan.bytes();
    let plan = WeightPlan::builder(bytes).attention_full(2, plan).build();
    let planner = bench.planner(plan);
    let stage = planner.attention_full_stage(2).expect("attention stage");
    assert_eq!(stage.caches.keys, 70);
    assert_eq!(stage.caches.key_scales, 71);
    assert_eq!(stage.caches.values, 72);
    assert_eq!(stage.query, 73);
    assert_eq!(stage.key, 74);
    assert_eq!(stage.value, 75);
    assert_eq!(stage.output, 76);
    assert!(stage.gate.is_none());
    assert_eq!(stage.act_scale, 0.5);
    assert_eq!(stage.o_act_scale, 1.0 / 0.75);
    assert_eq!(stage.out_scale, 0.0625);
    assert_eq!(stage.attn_q_scale, 16.0);
    match &stage.v_quant {
        myelin_core::backend::VQuant::Channel(scales) => assert_eq!(scales, &[0.5, 0.5]),
        other => panic!("unexpected V quantisation {other:?}"),
    }
}
