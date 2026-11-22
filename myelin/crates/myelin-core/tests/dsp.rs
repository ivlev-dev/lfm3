//! The DSP seam must fail loudly instead of silently returning garbage.

use myelin_core::kv::PagedKvCache;
use myelin_core::{AttentionCall, AttentionKind, Backend, BackendError, DspBackend, ModelSpec};

#[test]
fn dsp_backend_reports_missing_transport() {
    let spec = ModelSpec::placeholder();
    let mut backend = DspBackend::new(spec.clone());
    assert_eq!(backend.name(), "dsp");
    assert_eq!(backend.spec(), &spec);

    let hidden = spec.dims.hidden;
    let mut out = vec![0.0f32; hidden];
    let error = backend.embed(0, &mut out).expect_err("embed must fail");
    match error {
        BackendError::Unsupported {
            backend,
            operation,
            reason,
        } => {
            assert_eq!(backend, "dsp");
            assert_eq!(operation, "embed");
            assert!(reason.contains("host tensors"));
        }
        other => panic!("unexpected error {other:?}"),
    }

    let mut kv = PagedKvCache::new(spec.dims.kv_dim(), 1).expect("cache");
    let call = AttentionCall {
        layer: 0,
        position: 0,
        start: 0,
        kind: AttentionKind::Full,
        input: &vec![0.0f32; hidden],
    };
    assert!(backend.attention_layer(call, &mut kv, &mut out).is_err());
    assert!(backend
        .ffn_layer(0, &vec![0.0f32; hidden], &mut out)
        .is_err());
    assert!(backend
        .rms_norm(
            myelin_core::NormWeights::Output,
            &vec![0.0f32; hidden],
            &mut out
        )
        .is_err());
}
