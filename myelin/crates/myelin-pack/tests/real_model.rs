//! Round-trip quality on real LFM3 tensors.
//!
//! Skipped when the GGUF is absent (the repo intentionally does not ship it).

use std::path::PathBuf;

use myelin_pack::manifest::PackMeta;
use myelin_pack::{pack_gguf, unpack, DType, PackOptions};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/LFM3-1.7B-Q4_K_M.gguf")
}

fn options() -> PackOptions {
    PackOptions { dtype: DType::I8 }
}

fn meta(name: &str) -> PackMeta {
    PackMeta {
        tensor: name.to_string(),
        source: None,
    }
}

#[test]
fn q4_k_tensor_with_padded_channels() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }

    // K = 2048, N = 512; read 45 channels so N padding is exercised too.
    let (result, weights) = pack_gguf(&path, "blk.2.attn_k.weight", Some(45), options()).unwrap();
    assert_eq!(weights.k(), 2048);
    assert_eq!(weights.n(), 45);
    assert_eq!(result.manifest.k_padded, 2048);
    assert_eq!(result.manifest.n_padded, 64);
    assert_eq!(result.manifest.k_tiles, 64);
    assert_eq!(result.manifest.n_tiles, 2);
    assert_eq!(result.manifest.tiles.len(), 128);

    assert!(
        result.quality.cos_total >= 0.999,
        "cos_total = {}",
        result.quality.cos_total
    );
    assert!(
        result.quality.cos_min_channel >= 0.999,
        "cos_min_channel = {}",
        result.quality.cos_min_channel
    );
    assert!(result.quality.max_abs_err_over_scale <= 0.5001);
    assert!(result.quality.saturated > 0);
    for channel in &result.quantized {
        assert_eq!(channel.clamped, 0);
    }

    let decoded = unpack(&result.blob).unwrap();
    let quality = decoded.quality(&weights).unwrap();
    assert_eq!(quality.cos_total, result.quality.cos_total);
    assert_eq!(quality.max_abs_err, result.quality.max_abs_err);

    // Padding channels carry scale 0 and zero codes.
    for channel in 45..decoded.n_padded() {
        assert_eq!(decoded.scale_bits()[channel], 0);
        let row =
            &decoded.codes()[channel * decoded.k_padded()..(channel + 1) * decoded.k_padded()];
        assert!(row.iter().all(|value| *value == 0));
    }

    // Manifest survives a JSON round trip and still describes the blob.
    let json = result.manifest.to_json().unwrap();
    let parsed: myelin_pack::Manifest = serde_json::from_str(&json).unwrap();
    parsed.verify_blob(&result.blob).unwrap();
    assert_eq!(parsed.tensor, "blk.2.attn_k.weight");
    assert_eq!(parsed.source.as_ref().unwrap().ggml_type, "Q4_K");
}

#[test]
fn q6_k_tensor_round_trip() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }

    // Q6_K source, K = 7680 (FFN down projection), first 33 output channels.
    let (result, weights) = pack_gguf(&path, "blk.0.ffn_down.weight", Some(33), options()).unwrap();
    assert_eq!(weights.k(), 7680);
    assert_eq!(weights.n(), 33);
    assert_eq!(result.manifest.k_padded, 7680);
    assert_eq!(result.manifest.n_padded, 64);
    assert_eq!(result.manifest.k_tiles, 240);

    assert!(
        result.quality.cos_total >= 0.999,
        "cos_total = {}",
        result.quality.cos_total
    );
    assert!(
        result.quality.cos_min_channel >= 0.999,
        "cos_min_channel = {}",
        result.quality.cos_min_channel
    );
    assert!(result.quality.max_abs_err_over_scale <= 0.5001);
    assert!(result.manifest.source.as_ref().unwrap().ggml_type == "Q6_K");
}

#[test]
fn error_on_unknown_tensor_and_1d_tensor() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    assert!(pack_gguf(&path, "blk.99.missing.weight", None, options()).is_err());
    assert!(pack_gguf(&path, "blk.0.attn_norm.weight", None, options()).is_err());
}

#[test]
fn synthetic_manifest_meta_is_stable() {
    // Guards the `PackMeta` shape used by the CLI without touching the model.
    let matrix = myelin_pack::Matrix::new(32, 32, vec![0.5; 1024]).unwrap();
    let result = myelin_pack::pack(&matrix, &meta("synthetic"), options()).unwrap();
    assert_eq!(result.manifest.tensor, "synthetic");
    assert!(result.manifest.source.is_none());
}
