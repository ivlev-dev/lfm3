use lfm_domain::{LayerKind, ModelConfig, SamplingParams};
use lfm_gguf::GgufFile;
use std::path::PathBuf;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/LFM3-1.7B-Q4_K_M.gguf")
}

#[test]
fn config_from_real_gguf() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let f = GgufFile::open(&path).unwrap();
    let cfg = ModelConfig::from_metadata(&f).unwrap();
    assert_eq!(cfg.arch, "lfm3");
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.ffn_size, 7680);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 128000);
    assert_eq!(cfg.conv_cache, 7);
    assert_eq!(cfg.sliding_window, 256);
    assert!(cfg.tied_embeddings);
    assert_eq!(cfg.rope_full.theta, 1e6);
    assert_eq!(cfg.rope_sliding.theta, 1e4);
    assert_eq!(cfg.rope_full.rotated_dim, 64);
    assert_eq!(cfg.rope_sliding.rotated_dim, 128);
    assert_eq!(cfg.conv_layers().count(), 13);
    assert_eq!(cfg.attention_layers().count(), 11);
    assert_eq!(cfg.kv_dim(), 512);
    assert_eq!(cfg.kind(0), LayerKind::Conv);
    assert_eq!(cfg.kind(2), LayerKind::FullAttention);
    assert_eq!(cfg.kind(4), LayerKind::SlidingAttention);
    assert_eq!(cfg.kv_heads_of(2), 4);
    assert_eq!(cfg.kv_heads_of(0), 0);
    let params = SamplingParams::lfm3_defaults(&f);
    assert_eq!(params.top_k, 50);
    assert!((params.temperature - 0.1).abs() < 1e-6);
}
