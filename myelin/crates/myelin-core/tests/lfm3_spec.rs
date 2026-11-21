use myelin_core::spec::{AttentionKind, ModelFamily, RopeKind};
use myelin_core::{arch, build_spec, detect, GgufMeta, MetaValue};

fn lfm3_meta() -> GgufMeta {
    let mut meta = GgufMeta::new("lfm3");
    meta.set(
        "general.file_type",
        MetaValue::U32(15), // Q4_K_M
    )
    .set("lfm3.block_count", MetaValue::U32(24))
    .set("lfm3.embedding_length", MetaValue::U32(2048))
    .set("lfm3.feed_forward_length", MetaValue::U32(7680))
    .set("lfm3.attention.head_count", MetaValue::U32(16))
    .set("lfm3.attention.key_length", MetaValue::U32(128))
    .set("lfm3.attention.value_length", MetaValue::U32(128))
    .set("lfm3.attention.sliding_window", MetaValue::U32(256))
    .set("lfm3.shortconv.l_cache", MetaValue::U32(7))
    .set(
        "lfm3.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    )
    .set("lfm3.rope.freq_base", MetaValue::F32(1_000_000.0))
    .set("lfm3.rope.freq_base_swa", MetaValue::F32(10_000.0))
    .set(
        "lfm3.attention.kv_format",
        MetaValue::Str("q8_0".to_owned()),
    )
    .set(
        "lfm3.attention.head_count_kv",
        MetaValue::I32Array(vec![
            0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0,
        ]),
    )
    .set(
        "lfm3.attention.sliding_window_pattern",
        MetaValue::BoolArray(vec![
            false, false, false, false, true, false, false, false, true, false, false, false, true,
            false, false, false, true, false, false, false, true, false, false, false,
        ]),
    )
    .set("tokenizer.ggml.model", MetaValue::Str("gpt2".to_owned()))
    .set("tokenizer.ggml.bos_token_id", MetaValue::U32(1))
    .set("tokenizer.ggml.eos_token_id", MetaValue::U32(2))
    .set(
        "tokenizer.chat_template",
        MetaValue::Str("{{ messages }}".to_owned()),
    )
    .set_tensor_dims("token_embd.weight", vec![2048, 128_000])
    .set_tensor_dims("rope_freqs.weight", vec![64]);
    meta
}

#[test]
fn detects_and_builds_lfm3() {
    let meta = lfm3_meta();
    assert_eq!(detect(&meta), Some(ModelFamily::Lfm3));
    assert_eq!(arch::detect(&meta), Some(ModelFamily::Lfm3));
    let spec = build_spec(&meta).expect("spec");
    assert_eq!(spec.family, ModelFamily::Lfm3);

    assert_eq!(spec.dims.layers, 24);
    assert_eq!(spec.dims.hidden, 2048);
    assert_eq!(spec.dims.ffn, 7680);
    assert_eq!(spec.dims.heads, 16);
    assert_eq!(spec.dims.kv_heads, 4);
    assert_eq!(spec.dims.head_dim, 128);
    assert_eq!(spec.dims.vocab, 128_000);
    assert_eq!(spec.dims.conv_cache, 7);
    assert_eq!(spec.dims.sliding_window, 256);

    assert_eq!(spec.layers.len(), 24);
    assert_eq!(spec.conv_layers().count(), 13);
    assert_eq!(spec.attention_layers().count(), 11);
    assert_eq!(
        spec.attention_layers()
            .filter(|(_, kind)| *kind == AttentionKind::Sliding)
            .map(|(index, _)| index)
            .collect::<Vec<_>>(),
        vec![4, 8, 12, 16, 20]
    );
    assert_eq!(
        spec.attention_layers()
            .filter(|(_, kind)| *kind == AttentionKind::Full)
            .count(),
        6
    );
    for (_, ksize) in spec.conv_layers() {
        assert_eq!(ksize, 7);
    }
    assert!(spec.layers.iter().all(|layer| !layer.is_moe()));

    assert_eq!(spec.rope.full.theta, 1_000_000.0);
    assert_eq!(spec.rope.full.kind, RopeKind::Partial { rotated_dim: 64 });
    assert_eq!(spec.rope.full.dim, 128);
    let sliding = spec.rope.sliding.expect("sliding rope");
    assert_eq!(sliding.theta, 10_000.0);
    assert_eq!(sliding.kind, RopeKind::Full);
    assert_eq!(sliding.dim, 128);

    assert_eq!(spec.kv_format.dtype, myelin_core::DType::Q8_0);
    assert_eq!(spec.tokenizer.model, "gpt2");
    assert_eq!(spec.tokenizer.vocab_size, 128_000);
    assert_eq!(spec.tokenizer.bos_token_id, Some(1));
    assert_eq!(spec.tokenizer.eos_token_id, Some(2));
    assert_eq!(spec.chat_template.template, "{{ messages }}");
    assert_eq!(spec.weight_format.quant, myelin_core::Quant::Q4K);
}
