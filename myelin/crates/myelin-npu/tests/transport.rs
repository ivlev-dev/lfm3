//! Host tests for the FastRPC transport surface and the DSP backend contract.
//!
//! Everything runs against [`MockTransport`]; the real-device test is in
//! `real_rpc.rs` behind `#[ignore]`.

use std::sync::Arc;

use myelin_core::backend::dsp::{AttentionStage, DspBackend, FfnStage};
use myelin_core::kv::{KvCache, PagedKvCache};
use myelin_core::{AttentionCall, AttentionKind, Backend, BackendError, ModelSpec};
use myelin_npu::method::method_id;
use myelin_npu::{
    AttnRequest, AttnScalars, Executor, FfnRequest, FfnScalars, GemvRequest, GemvScalars,
    HvxClient, MockCall, MockReply, MockTransport, Transport, TransportError,
};

fn reply(prim_out: Vec<u8>, out_bufs: Vec<Vec<u8>>) -> MockReply {
    MockReply { prim_out, out_bufs }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn alloc_buf_serialises_scalars_and_scalar_output() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::ALLOC_BUF, |call| {
        assert_eq!(call.prim_in.len(), 4);
        assert_eq!(read_i32(&call.prim_in), 4096);
        assert_eq!(call.prim_out_len, 4);
        assert!(call.out_lens.is_empty());
        Ok(reply(7i32.to_le_bytes().to_vec(), Vec::new()))
    });
    let client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    assert_eq!(client.alloc_buf(4096), Ok(7));

    let call = mock.last_call().expect("recorded");
    assert_eq!(call.method, method_id::ALLOC_BUF);
    assert_eq!(call.scalars, 0x0001_0100);
    assert_eq!(call.prim_in, 4096i32.to_le_bytes());
    assert!(call.out_lens.is_empty());
}

#[test]
fn set_shape_packs_four_longs() {
    let mock = MockTransport::new();
    let client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    client.set_shape(3, 7680, 2048, 1).expect("set_shape rpc");

    let call = mock.last_call().expect("recorded");
    assert_eq!(call.method, method_id::SET_SHAPE);
    assert_eq!(call.prim_in, i32_bytes(&[3, 7680, 2048, 1]));
    assert_eq!(call.scalars, 0x0101_0000);
    assert_eq!(call.out_lens, Vec::<usize>::new());
}

#[test]
fn ffn_round_trips_typed_buffers() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::FFN, |call| {
        assert_eq!(call.in_bufs.len(), 1);
        assert_eq!(call.in_bufs[0], vec![1u8, 2, 3, 4]);
        assert_eq!(call.out_lens.len(), 3);
        assert_eq!(call.out_lens[0], 4);
        assert_eq!(call.out_lens[1], 8);
        assert_eq!(call.out_lens[2], 32);
        Ok(reply(
            Vec::new(),
            vec![
                vec![9u8; 4],
                vec![5u8; 8],
                i32_bytes(&[11, 12, 13, 14, 0, 0, 0, 0]),
            ],
        ))
    });
    let client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    let response = client
        .ffn(FfnRequest {
            scalars: FfnScalars {
                gate: 1,
                up: 2,
                down: 3,
                reps: 1,
            },
            act: &[1, 2, 3, 4],
            dst_len: 4,
            mid_len: 8,
        })
        .expect("ffn rpc");
    assert_eq!(response.dst, vec![9u8; 4]);
    assert_eq!(response.mid, vec![5u8; 8]);
    assert_eq!(response.times[0], 11);
    assert_eq!(response.times[3], 14);

    let call = mock.last_call().expect("recorded");
    assert_eq!(call.scalars, 0x0602_0300);
    // _primIn[8] from the stub: three handles, actLen, reps, dstLen, midLen,
    // timesLen (sequence<long> elements).
    assert_eq!(call.prim_in, i32_bytes(&[1, 2, 3, 4, 1, 4, 8, 8]));
}

#[test]
fn attn_packs_eight_scalars_and_block_table() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::ATTN, move |call| {
        assert_eq!(call.in_bufs.len(), 2);
        assert_eq!(call.in_bufs[0], vec![7u8; 4]);
        assert_eq!(call.in_bufs[1], i32_bytes(&[0, 1, 2]));
        assert_eq!(call.out_lens, vec![8, 32]);
        Ok(reply(Vec::new(), vec![vec![0u8; 8], i32_bytes(&[1; 8])]))
    });
    let client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    let response = client
        .attn(AttnRequest {
            scalars: AttnScalars {
                keys: 1,
                key_scales: 2,
                values: 3,
                value_scales: 0,
                position: 63,
                seq: 64,
                window: 32,
                reps: 2,
            },
            q: &[7u8; 4],
            table: &[0, 1, 2],
            out_len: 8,
        })
        .expect("attn rpc");
    assert_eq!(response.outp, vec![0u8; 8]);
    assert_eq!(response.times, [1; 8]);

    let call = mock.last_call().expect("recorded");
    assert_eq!(call.scalars, 0x0b03_0200);
    // _primIn[12]: handles, qLen, tblLen (long elements), pos, seq, window,
    // reps, outLen, timesLen.
    assert_eq!(
        call.prim_in,
        i32_bytes(&[1, 2, 3, 0, 4, 3, 63, 64, 32, 2, 8, 8])
    );
}

#[test]
fn gemv_returns_declared_dst_length() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::GEMV, |call| {
        assert_eq!(call.out_lens, vec![6, 32]);
        Ok(reply(Vec::new(), vec![vec![4u8; 6], i32_bytes(&[0; 8])]))
    });
    let client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    let response = client
        .gemv(GemvRequest {
            scalars: GemvScalars {
                h: 4,
                m: 1,
                reps: 1,
            },
            act: &[1, 2, 3],
            dst_len: 6,
        })
        .expect("gemv rpc");
    assert_eq!(response.dst.len(), 6);
    // _primIn[6]: h, M, actLen, reps, dstLen, timesLen.
    assert_eq!(
        mock.last_call().expect("call").prim_in,
        i32_bytes(&[4, 1, 3, 1, 6, 8])
    );
}

#[test]
fn rpc_errors_keep_the_code_and_name() {
    let mock = MockTransport::new();
    mock.fail_invoke(TransportError::Rpc {
        method: method_id::FFN,
        code: 0x8000_040e_u32 as i32,
        name: "AEE_EBADPARM",
    });
    let client = HvxClient::open(Arc::new(mock), "lfmgemv").expect("open");
    let error = client.load_buf(1, 0, &[0u8; 4]).expect_err("rpc must fail");
    assert!(error.to_string().contains("AEE_EBADPARM"));
    assert_eq!(error.method(), Some(method_id::FFN));
}

#[test]
fn timeout_is_typed() {
    let mock = MockTransport::new();
    mock.fail_invoke(TransportError::Timeout {
        method: method_id::ATTN,
        millis: 250,
    });
    let client = HvxClient::open(Arc::new(mock), "lfmgemv").expect("open");
    let error = client.load_buf(1, 0, &[0u8; 4]).expect_err("rpc must fail");
    match error {
        TransportError::Timeout { method, millis } => {
            assert_eq!(method, method_id::ATTN);
            assert_eq!(millis, 250);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn handle_is_reused_and_closed_once() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::TEAM_INFO, |_call| {
        Ok(reply(
            Vec::new(),
            vec![i32_bytes(&[1, 2, 3, 4, 5, 6, 7, 8])],
        ))
    });
    let mut client = HvxClient::open(Arc::new(mock.clone()), "lfmgemv").expect("open");
    let handle = client.handle().expect("handle");
    for _ in 0..3 {
        assert_eq!(client.team_info(8).expect("team_info").len(), 8);
    }
    assert_eq!(client.handle(), Some(handle));
    assert_eq!(mock.calls().len(), 3);
    assert!(mock.calls().iter().all(|call| call.handle == handle));
    assert_eq!(mock.live_handles(), vec![handle]);

    client.close().expect("close");
    client.close().expect("close is idempotent");
    assert_eq!(mock.closed(), vec![handle]);
    assert!(mock.live_handles().is_empty());
}

#[test]
fn shared_buffers_are_freed_by_drop() {
    let mock = MockTransport::new();
    {
        let first = mock.alloc_shared(1024).expect("alloc");
        let mut second = mock.alloc_shared(2048).expect("alloc");
        assert_eq!(first.fd(), 1000);
        assert_eq!(second.fd(), 1001);
        assert!(first.physical_address().is_some());
        second.as_mut_slice()[0] = 0x5a;
        assert_eq!(second.as_slice()[0], 0x5a);
        assert_eq!(mock.allocations(), 2);
        assert_eq!(mock.live_bytes(), 3072);
        assert_eq!(mock.frees(), 0);
    }
    assert_eq!(mock.frees(), 2);
    assert_eq!(mock.live_bytes(), 0);
}

#[test]
fn too_small_output_is_reported() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::GEMV, |_| {
        Ok(reply(Vec::new(), vec![vec![0u8; 16], i32_bytes(&[0; 8])]))
    });
    let client = HvxClient::open(Arc::new(mock), "lfmgemv").expect("open");
    let error = client
        .gemv(GemvRequest {
            scalars: GemvScalars {
                h: 1,
                m: 1,
                reps: 1,
            },
            act: &[1],
            dst_len: 8,
        })
        .expect_err("must reject");
    match error {
        TransportError::BufferTooSmall {
            buffer,
            expected,
            got,
        } => {
            assert_eq!(buffer, "out_buf");
            assert_eq!((expected, got), (16, 8));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn open_failure_is_typed() {
    let mock = MockTransport::new();
    mock.fail_open(TransportError::HandleOpen {
        module: "lfmgemv".to_owned(),
        code: 0x8000_042c_u32 as i32,
    });
    let error = match HvxClient::open(Arc::new(mock), "lfmgemv") {
        Ok(_) => panic!("open must fail"),
        Err(error) => error,
    };
    match error {
        TransportError::HandleOpen { module, code } => {
            assert_eq!(module, "lfmgemv");
            assert_eq!(code, 0x8000_042c_u32 as i32);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn backend_without_transport_reports_fastrpc_reason() {
    let spec = ModelSpec::placeholder();
    let mut backend = DspBackend::new(spec.clone());
    let mut out = vec![0.0f32; spec.dims.hidden];
    let error = backend
        .ffn_layer(0, &vec![0.0f32; spec.dims.hidden], &mut out)
        .expect_err("no transport");
    match error {
        BackendError::Unsupported { reason, .. } => assert!(reason.contains("FastRPC")),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn backend_without_stage_reports_stage_reason() {
    let spec = ModelSpec::placeholder();
    let mock = Arc::new(MockTransport::new());
    let executor: Arc<dyn myelin_core::backend::dsp::DspExecutor> = Arc::new(Executor::new(mock));
    let mut backend = DspBackend::attach(spec.clone(), executor, "lfmgemv").expect("attach");
    let mut out = vec![0.0f32; spec.dims.hidden];
    let error = backend
        .ffn_layer(0, &vec![0.0f32; spec.dims.hidden], &mut out)
        .expect_err("not staged");
    match error {
        BackendError::Unsupported { reason, .. } => assert!(reason.contains("stage")),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn backend_ffn_quantises_calls_and_dequantises() {
    let spec = ModelSpec::placeholder();
    let hidden = spec.dims.hidden;
    let ffn = spec.dims.ffn;
    let mock = MockTransport::new();
    mock.set_handler(method_id::FFN, move |call| {
        assert_eq!(
            call.prim_in,
            i32_bytes(&[1, 2, 3, hidden as i32, 1, hidden as i32, ffn as i32, 8])
        );
        assert_eq!(call.in_bufs[0].len(), hidden);
        assert_eq!(call.in_bufs[0][0], 2); // 1.0 * act_scale 2.0
        Ok(reply(
            Vec::new(),
            vec![vec![10u8; hidden], vec![0u8; ffn], i32_bytes(&[0; 8])],
        ))
    });
    let executor: Arc<dyn myelin_core::backend::dsp::DspExecutor> =
        Arc::new(Executor::new(Arc::new(mock.clone())));
    let mut backend = DspBackend::attach(spec.clone(), executor, "lfmgemv").expect("attach");
    backend
        .stage_ffn(
            0,
            FfnStage {
                gate: 1,
                up: 2,
                down: 3,
                act_scale: 2.0,
                out_scale: 0.5,
                reps: 1,
            },
        )
        .expect("stage");

    let mut out = vec![0.0f32; hidden];
    backend
        .ffn_layer(0, &vec![1.0f32; hidden], &mut out)
        .expect("ffn");
    assert!(out.iter().all(|value| (*value - 5.0).abs() < 1e-6));
    let call = mock.last_call().expect("recorded");
    assert_eq!(call.method, method_id::FFN);
    assert_eq!(call.scalars, 0x0602_0300);
}

#[test]
fn backend_ffn_rpc_error_is_remembered() {
    let spec = ModelSpec::placeholder();
    let mock = MockTransport::new();
    mock.fail_invoke(TransportError::Rpc {
        method: method_id::FFN,
        code: 0x8000_040e_u32 as i32,
        name: "AEE_EBADPARM",
    });
    let executor: Arc<dyn myelin_core::backend::dsp::DspExecutor> =
        Arc::new(Executor::new(Arc::new(mock)));
    let mut backend = DspBackend::attach(spec.clone(), executor, "lfmgemv").expect("attach");
    backend
        .stage_ffn(
            0,
            FfnStage {
                gate: 1,
                up: 2,
                down: 3,
                act_scale: 1.0,
                out_scale: 1.0,
                reps: 1,
            },
        )
        .expect("stage");
    let mut out = vec![0.0f32; spec.dims.hidden];
    assert!(backend
        .ffn_layer(0, &vec![0.0f32; spec.dims.hidden], &mut out)
        .is_err());
    match backend.last_error() {
        Some(myelin_core::backend::dsp::DspError::Rpc { method, code, name }) => {
            assert_eq!(*method, method_id::FFN);
            assert_eq!(*code, 0x8000_040e_u32 as i32);
            assert_eq!(*name, "AEE_EBADPARM");
        }
        other => panic!("unexpected last error {other:?}"),
    }
}

#[test]
fn backend_attention_calls_attn_and_mirrors_host_kv() {
    let mut spec = ModelSpec::placeholder();
    spec.dims.hidden = 2048;
    spec.dims.heads = 16;
    spec.dims.head_dim = 128;
    spec.dims.kv_heads = 4;
    spec.dims.ffn = 64;
    let q_dim = spec.dims.q_dim();
    let mock = MockTransport::new();
    mock.set_handler(method_id::ATTN, move |call| {
        assert_eq!(
            call.prim_in,
            i32_bytes(&[
                1,
                2,
                3,
                0,
                q_dim as i32,
                1,
                0,
                1,
                0,
                1,
                (q_dim * 4) as i32,
                8
            ])
        );
        assert_eq!(call.in_bufs[0].len(), q_dim);
        assert_eq!(call.in_bufs[1], i32_bytes(&[0]));
        let mut outp = Vec::with_capacity(q_dim * 4);
        for _ in 0..q_dim {
            outp.extend_from_slice(&4096i32.to_le_bytes());
        }
        Ok(reply(Vec::new(), vec![outp, i32_bytes(&[0; 8])]))
    });
    let executor: Arc<dyn myelin_core::backend::dsp::DspExecutor> =
        Arc::new(Executor::new(Arc::new(mock.clone())));
    let mut backend = DspBackend::attach(spec.clone(), executor, "lfmgemv").expect("attach");
    backend
        .stage_attention(
            0,
            AttentionStage {
                keys: 1,
                key_scales: 2,
                values: 3,
                value_scales: 0,
                q_scale: 1.0,
                out_scale: 1.0 / 4096.0,
                reps: 1,
                cache_pages: 8,
                preprojected: true,
            },
        )
        .expect("stage");

    let mut kv = PagedKvCache::new(spec.dims.kv_dim(), 64).expect("kv");
    let input = vec![1.0f32; q_dim];
    let mut out = vec![0.0f32; q_dim];
    backend
        .attention_layer(
            AttentionCall {
                layer: 0,
                position: 0,
                start: 0,
                kind: AttentionKind::Full,
                input: &input,
            },
            &mut kv,
            &mut out,
        )
        .expect("attention");
    assert!(out.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    assert_eq!(kv.len(), 1);
    assert_eq!(kv.capacity(), 64);
}

#[test]
fn backend_rejects_non_stand_attention_geometry() {
    let spec = ModelSpec::placeholder();
    let mock = Arc::new(MockTransport::new());
    let executor: Arc<dyn myelin_core::backend::dsp::DspExecutor> = Arc::new(Executor::new(mock));
    let mut backend = DspBackend::attach(spec.clone(), executor, "lfmgemv").expect("attach");
    let mut kv = PagedKvCache::new(spec.dims.kv_dim(), 8).expect("kv");
    let input = vec![0.0f32; spec.dims.hidden];
    let mut out = vec![0.0f32; spec.dims.hidden];
    let error = backend
        .attention_layer(
            AttentionCall {
                layer: 0,
                position: 0,
                start: 0,
                kind: AttentionKind::Full,
                input: &input,
            },
            &mut kv,
            &mut out,
        )
        .expect_err("geometry");
    match error {
        BackendError::Unsupported { reason, .. } => assert!(reason.contains("16 heads")),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn stage_validation_rejects_bad_scales_and_layers() {
    let spec = ModelSpec::placeholder();
    let mut backend = DspBackend::new(spec);
    let bad_scale = FfnStage {
        gate: 1,
        up: 2,
        down: 3,
        act_scale: 0.0,
        out_scale: 1.0,
        reps: 1,
    };
    assert!(backend.stage_ffn(0, bad_scale).is_err());
    let stage = FfnStage {
        gate: 1,
        up: 2,
        down: 3,
        act_scale: 1.0,
        out_scale: 1.0,
        reps: 1,
    };
    assert!(backend.stage_ffn(99, stage).is_err());
    let attention = AttentionStage {
        keys: 1,
        key_scales: 2,
        values: 3,
        value_scales: 0,
        q_scale: 1.0,
        out_scale: 1.0,
        reps: 1,
        cache_pages: 1,
        preprojected: false,
    };
    assert!(backend.stage_attention(0, attention).is_err());
}

#[test]
fn mock_handler_sees_exact_call_shape() {
    let mock = MockTransport::new();
    mock.set_handler(method_id::LOAD_BUF, |call: &MockCall| {
        assert_eq!(call.method, method_id::LOAD_BUF);
        // `dataLen` is part of the scalar pack in the qaic stub.
        assert_eq!(call.prim_in, i32_bytes(&[3, 1024, 4]));
        assert_eq!(call.in_bufs, vec![vec![0xab; 4]]);
        Ok(MockReply::default())
    });
    let transport: Arc<dyn Transport> = Arc::new(mock.clone());
    let client = HvxClient::open(transport, "lfmgemv").expect("open");
    client.load_buf(3, 1024, &[0xab; 4]).expect("load");
}
