use myelin_core::spec::{AttentionKind, LayerSpec, ModelFamily, RopeConfig, RopeKind, RopeSpec};
use myelin_core::{build_spec, CoreError, Dims, GgufMeta, MetaValue, ModelSpec};

fn minimal_spec() -> ModelSpec {
    let head_dim = 8;
    ModelSpec {
        family: ModelFamily::Llama,
        dims: Dims {
            hidden: 2 * head_dim,
            ffn: 16,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim,
            vocab: 32,
            conv_cache: 4,
            sliding_window: 4,
        },
        rope: RopeConfig {
            full: RopeSpec {
                kind: RopeKind::Full,
                theta: 10_000.0,
                dim: head_dim,
            },
            sliding: None,
        },
        layers: vec![
            LayerSpec::Attention {
                kind: AttentionKind::Full,
            },
            LayerSpec::Attention {
                kind: AttentionKind::Sliding,
            },
        ],
        kv_format: myelin_core::KvFormat::F16,
        tokenizer: myelin_core::TokenizerSpec {
            model: "gpt2".to_owned(),
            vocab_size: 32,
            bos_token_id: None,
            eos_token_id: None,
        },
        chat_template: myelin_core::ChatTemplateSpec {
            template: String::new(),
        },
        weight_format: myelin_core::WeightFormat::from_quant(myelin_core::Quant::F16),
        norm_eps: 1e-5,
        tied_embeddings: true,
    }
}

#[test]
fn placeholder_is_valid() {
    let spec = ModelSpec::placeholder();
    spec.validate().expect("placeholder must be valid");
    assert_eq!(spec.family, ModelFamily::Unknown);
}

#[test]
fn rejects_inconsistent_spec() {
    let mut spec = minimal_spec();
    spec.dims.hidden += 1;
    assert!(matches!(spec.validate(), Err(CoreError::InvalidSpec(_))));

    let mut spec = minimal_spec();
    spec.layers.pop();
    assert!(matches!(spec.validate(), Err(CoreError::InvalidSpec(_))));

    let mut spec = minimal_spec();
    spec.layers[0] = LayerSpec::Conv { ksize: 1 };
    assert!(matches!(spec.validate(), Err(CoreError::InvalidSpec(_))));

    let mut spec = minimal_spec();
    spec.layers[0] = LayerSpec::Moe {
        experts: 2,
        top_k: 3,
    };
    assert!(matches!(spec.validate(), Err(CoreError::InvalidSpec(_))));
}

#[test]
fn lfm3_defaults_fill_missing_optional_keys() {
    let meta = GgufMeta::new("lfm3");
    let spec = build_spec(&meta).expect("defaults must build a spec");
    assert_eq!(spec.family, ModelFamily::Lfm3);
    assert_eq!(spec.dims.layers, 24);
    assert_eq!(spec.conv_layers().count(), 13);
    assert_eq!(spec.attention_layers().count(), 11);
    assert_eq!(spec.dims.ffn, 7680);
    assert_eq!(spec.dims.kv_heads, 4);
    assert_eq!(spec.dims.heads, 16);
    assert_eq!(spec.dims.sliding_window, 256);
    assert_eq!(spec.dims.vocab, 128_000);
}

#[test]
fn rejects_unknown_architecture() {
    let meta = GgufMeta::new("not-a-model");
    let error = build_spec(&meta).expect_err("must reject");
    assert_eq!(
        error,
        CoreError::UnsupportedArchitecture("not-a-model".to_owned())
    );
}

#[test]
fn rejects_kv_heads_above_query_heads() {
    let mut meta = GgufMeta::new("lfm3");
    meta.set(
        "lfm3.attention.head_count_kv",
        MetaValue::I32Array(vec![0; 24]),
    );
    let error = build_spec(&meta).expect_err("must reject");
    assert!(matches!(error, CoreError::InvalidValue { .. }));
}

#[test]
fn rejects_wrong_layer_pattern_length() {
    let mut meta = GgufMeta::new("lfm3");
    meta.set(
        "lfm3.attention.head_count_kv",
        MetaValue::I32Array(vec![0, 0]),
    );
    let error = build_spec(&meta).expect_err("must reject");
    assert!(matches!(error, CoreError::InvalidValue { .. }));
}
