use lfm_gguf::{GgmlType, GgufFile, MetadataValue};
use std::io::Write;
use std::path::PathBuf;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/LFM3-1.7B-Q4_K_M.gguf")
}

#[test]
fn real_model_layout() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.get_str("general.architecture"), Some("lfm3"));
    assert_eq!(f.get_u32("lfm3.block_count"), Some(24));
    assert_eq!(f.get_u32("lfm3.embedding_length"), Some(2048));
    assert_eq!(f.get_u32("lfm3.feed_forward_length"), Some(7680));
    assert_eq!(f.get_u32("lfm3.attention.sliding_window"), Some(256));
    assert_eq!(f.tensors().len(), 287);

    let kv = f.get_i32_array("lfm3.attention.head_count_kv").unwrap();
    assert_eq!(kv.len(), 24);
    assert_eq!(kv[0], 0);
    assert_eq!(kv[2], 4);
    let sliding = match f.metadata().get("lfm3.attention.sliding_window_pattern") {
        Some(MetadataValue::Array(a)) => a
            .values
            .iter()
            .map(|v| match v {
                MetadataValue::Bool(b) => *b,
                MetadataValue::U32(x) => *x == 1,
                MetadataValue::I32(x) => *x == 1,
                _ => false,
            })
            .collect::<Vec<_>>(),
        other => panic!("unexpected sliding pattern {other:?}"),
    };
    assert_eq!(sliding.len(), 24);
    for (i, v) in sliding.iter().enumerate() {
        assert_eq!(*v, matches!(i, 4 | 8 | 12 | 16 | 20));
    }

    let tokens = match f.metadata().get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(a)) => a.values.len(),
        other => panic!("unexpected tokenizer entry {other:?}"),
    };
    assert_eq!(tokens, 128000);

    let q = f.tensor("blk.2.attn_q.weight").unwrap();
    assert_eq!(q.ggml_type, GgmlType::Q4_K);
    assert_eq!(q.dims, vec![2048, 2048]);
    assert_eq!(q.byte_size, 2359296);
    assert_eq!(
        f.tensor_bytes("blk.2.attn_q.weight").unwrap().len(),
        2359296
    );

    let emb = f.tensor("token_embd.weight").unwrap();
    assert_eq!(emb.ggml_type, GgmlType::Q6_K);
    assert_eq!(emb.dims, vec![2048, 128000]);
    assert_eq!(emb.byte_size, 215040000);
    assert_eq!(
        f.tensor_bytes("token_embd.weight").unwrap().len(),
        215040000
    );

    let down = f.tensor("blk.0.ffn_down.weight").unwrap();
    assert_eq!(down.ggml_type, GgmlType::Q6_K);
    assert_eq!(down.dims, vec![7680, 2048]);
    assert_eq!(down.byte_size, 12902400);

    let mut max_end = 0u64;
    for t in f.tensors() {
        max_end = max_end.max(t.offset + t.byte_size);
    }
    assert!(max_end <= f.raw().len() as u64);
    assert!(f.tensor_bytes("nope.weight").is_err());
}

fn build_gguf(alignment: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());

    let key = b"general.alignment";
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&alignment.to_le_bytes());

    let key = b"test.note";
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&3u64.to_le_bytes());
    out.extend_from_slice(b"abc");

    let name = b"t0";
    out.extend_from_slice(&(name.len() as u64).to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&4u64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());

    let pad = (alignment as usize - out.len() % alignment as usize) % alignment as usize;
    out.extend(std::iter::repeat_n(0u8, pad));
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn synthetic_file() {
    let bytes = build_gguf(32);
    let path = std::env::temp_dir().join("lfm_gguf_synthetic.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&bytes).unwrap();
    drop(file);

    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.get_u32("general.alignment"), Some(32));
    assert_eq!(f.get_str("test.note"), Some("abc"));
    let t = f.tensor("t0").unwrap();
    assert_eq!(t.ggml_type, GgmlType::F32);
    assert_eq!(t.dims, vec![4]);
    let data = f.tensor_bytes("t0").unwrap();
    assert_eq!(data.len(), 16);
    assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), 2.0);
    let _ = std::fs::remove_file(path);
}

#[test]
fn rejects_broken_files() {
    let bad = std::env::temp_dir().join("lfm_gguf_bad.gguf");
    std::fs::write(
        &bad,
        b"GGUF\x03\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00",
    )
    .unwrap();
    assert!(GgufFile::open(&bad).is_err());

    let bad_magic = std::env::temp_dir().join("lfm_gguf_bad_magic.gguf");
    std::fs::write(
        &bad_magic,
        b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    )
    .unwrap();
    assert!(GgufFile::open(&bad_magic).is_err());
    let _ = std::fs::remove_file(bad);
    let _ = std::fs::remove_file(bad_magic);
}

fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn rejects_zero_dims() {
    for (label, n_dims, dim_value) in [("zero_dims", 0u32, 4u64), ("zero_dim_value", 1u32, 0u64)] {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let name = b"t0";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&n_dims.to_le_bytes());
        for _ in 0..n_dims {
            out.extend_from_slice(&dim_value.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let path = write_temp(&format!("lfm_gguf_{label}.gguf"), &out);
        assert!(GgufFile::open(&path).is_err(), "{label} must be rejected");
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn rejects_huge_array_count() {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    let key = b"test.arr";
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&u64::MAX.to_le_bytes());
    let path = write_temp("lfm_gguf_huge_array.gguf", &out);
    assert!(GgufFile::open(&path).is_err());
    let _ = std::fs::remove_file(path);
}
