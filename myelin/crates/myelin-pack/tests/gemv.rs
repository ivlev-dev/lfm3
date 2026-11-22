//! int4 GEMV format: layout invariants, padding, quality, CLI, and a
//! byte-for-byte comparison with the stand packer `ffn_pack_i4_row()`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use myelin_pack::gemv::{self, GemvHeader, GemvManifest, K_ALIGN, N_ALIGN};
use myelin_pack::{gguf_source, Matrix, PackMeta};

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

    fn signed(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("myelin-gemv-{tag}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn model_path() -> PathBuf {
    repo_root().join("models/LFM3-1.7B-Q4_K_M.gguf")
}

fn matrix(k: usize, n: usize, data: Vec<f32>) -> Matrix {
    Matrix::new(k, n, data).expect("matrix")
}

fn meta(tensor: &str) -> PackMeta {
    PackMeta {
        tensor: tensor.to_string(),
        source: None,
    }
}

fn write_f32(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).expect("write f32");
}

fn read_f32(path: &Path) -> Vec<f32> {
    fs::read(path)
        .expect("read f32")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn header_roundtrip_and_padding() {
    for (k, n) in [(256, 32), (300, 33), (7, 1), (2048, 6144), (1, 31)] {
        let header = GemvHeader::new(k, n).expect("header");
        assert_eq!(header.k_padded, k.div_ceil(K_ALIGN) * K_ALIGN);
        assert_eq!(header.n_padded, n.div_ceil(N_ALIGN) * N_ALIGN);
        assert_eq!(header.row_bytes, header.k_padded / 2);
        assert_eq!(header.weight_bytes, header.n_padded * header.row_bytes);
        let bytes = header.encode();
        let parsed = GemvHeader::parse(&{
            let mut blob = vec![0u8; header.total_bytes()];
            blob[..128].copy_from_slice(&bytes);
            blob
        })
        .expect("parse");
        assert_eq!(parsed, header);
        assert_eq!(parsed.build_id, gemv::format_build_id());
    }
}

#[test]
fn roundtrip_padding_and_quality() {
    let k = 100usize;
    let n = 7usize;
    let mut data = Vec::with_capacity(k * n);
    let mut rng = Rng::new(0x51ee);
    for row in 0..n {
        for col in 0..k {
            let value = if row == 0 {
                0.0
            } else if row == 1 {
                (col as f32) * 1e-6
            } else {
                rng.signed() * (1.0 + row as f32)
            };
            data.push(value);
        }
    }
    let source = matrix(k, n, data);
    let packed = gemv::build(&source, &meta("synthetic")).expect("build");
    assert_eq!(packed.header.k, k);
    assert_eq!(packed.header.n, n);
    assert_eq!(packed.header.k_padded, 256);
    assert_eq!(packed.header.n_padded, 32);
    assert_eq!(packed.blob.len(), packed.header.total_bytes());

    // Padding rows and columns are zero, padding scales are zero.
    let header = &packed.header;
    let weights = header.weights(&packed.blob);
    for row in n..header.n_padded {
        assert!(
            weights[row * header.row_bytes..(row + 1) * header.row_bytes]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(packed.header.scales(&packed.blob)[row], 0.0);
    }
    for row in 0..n {
        let row_bytes = &weights[row * header.row_bytes..(row + 1) * header.row_bytes];
        for byte in &row_bytes[k / 2..] {
            assert_eq!(*byte, 0, "padding column is not zero");
        }
    }

    // The decoded real rows match the source to within half an int4 step.
    assert!(packed.quality.cos_total > 0.95);
    assert!(packed.quality.max_abs_err_over_scale <= 0.5 + 1e-6);
    for row in 0..n {
        let decoded = packed.row(row).expect("row");
        for (a, b) in source.row(row).iter().zip(&decoded) {
            let scale = packed.scale_f32[row];
            assert!((a - b).abs() <= 0.5 * scale + 1e-6);
        }
    }

    // Manifest round trip and verification.
    let manifest = GemvManifest::from_blob(&meta("synthetic"), &packed);
    let json = manifest.to_json().expect("json");
    let parsed: GemvManifest = serde_json::from_str(&json).expect("parse json");
    parsed.verify_blob(&packed.blob).expect("verify");
    assert_eq!(parsed.scales_f32.len(), header.n_padded);
    assert_eq!(parsed.scales_q16.len(), header.n_padded);

    // Determinism.
    let again = gemv::build(&source, &meta("synthetic")).expect("build again");
    assert_eq!(packed.blob, again.blob);
}

#[test]
fn rejects_non_finite_and_bad_blobs() {
    let source = matrix(256, 32, vec![f32::NAN; 256 * 32]);
    assert!(gemv::build(&source, &meta("nan")).is_err());

    let good = matrix(256, 32, (0..256 * 32).map(|i| i as f32 * 0.01).collect());
    let packed = gemv::build(&good, &meta("good")).expect("build");
    let mut blob = packed.blob.clone();
    blob[0] ^= 0xff;
    assert!(GemvHeader::parse(&blob).is_err());
    let blob = packed.blob.clone();
    assert!(GemvHeader::parse(&blob[..64]).is_err());
    let mut blob = packed.blob.clone();
    blob[16..20].copy_from_slice(&(9999u32).to_le_bytes());
    assert!(GemvHeader::parse(&blob).is_err());
}

#[test]
fn quantize_row_uses_ties_to_even_and_clamps() {
    // `lrintf` rounds half to even: 0.5 -> 0, 1.5 -> 2, -1.5 -> -2.
    let row = vec![0.5f32, 1.5, -0.5, -1.5, 7.0, -7.0];
    let (scale, codes) = gemv::quantize_row(&row);
    assert_eq!(scale, 1.0);
    assert_eq!(codes, vec![0, 2, 0, -2, 7, -7]);

    // All-zero row: scale is 1.0, exactly like `ffn_pack_i4_row`.
    let (scale, codes) = gemv::quantize_row(&[0.0, -0.0, 0.0]);
    assert_eq!(scale, 1.0);
    assert_eq!(codes, vec![0, 0, 0]);

    // The clamp is reachable through float rounding of the extreme value.
    let row = vec![7.0f32, 7.0 - f32::EPSILON * 4.0, -7.0];
    let (_, codes) = gemv::quantize_row(&row);
    assert!(codes.iter().all(|code| (-8..=7).contains(code)));
}

static REFERENCE: OnceLock<Option<PathBuf>> = OnceLock::new();

fn reference_binary() -> Option<&'static Path> {
    REFERENCE
        .get_or_init(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let include = repo_root().join("myelin/dsp/hvx");
            if !include.join("ffn_ref.h").exists() {
                eprintln!(
                    "skip: {} not found (dsp sources unavailable)",
                    include.display()
                );
                return None;
            }
            let source = manifest_dir.join("tests/c/gemv_ref.c");
            let dir = temp_dir("ref");
            let binary = dir.join("gemv_ref");
            let status = Command::new("cc")
                .args(["-std=c99", "-O2", "-I"])
                .arg(&include)
                .arg(&source)
                .arg("-o")
                .arg(&binary)
                .arg("-lm")
                .status();
            match status {
                Ok(status) if status.success() => Some(binary),
                Ok(status) => {
                    eprintln!("skip: cc exited with {status}");
                    None
                }
                Err(error) => {
                    eprintln!("skip: cannot run cc: {error}");
                    None
                }
            }
        })
        .as_deref()
}

struct Reference {
    packed: Vec<u8>,
    scales: Vec<f32>,
}

fn run_reference(n: usize, k: usize, source: &Matrix) -> Option<Reference> {
    let binary = reference_binary()?;
    assert_eq!(source.n(), n);
    assert_eq!(source.k(), k);
    let dir = temp_dir("ref-run");
    let weights = dir.join("weights.f32");
    let out = dir.join("weights.i4");
    let scales = dir.join("scales.f32");
    write_f32(&weights, source.data());
    let status = Command::new(binary)
        .arg(n.to_string())
        .arg(k.to_string())
        .arg(&weights)
        .arg(&out)
        .arg(&scales)
        .status()
        .expect("run reference");
    assert!(status.success(), "reference failed");
    Some(Reference {
        packed: fs::read(out).expect("read i4"),
        scales: read_f32(&scales),
    })
}

fn compare_with_c(label: &str, source: &Matrix) {
    let n = source.n();
    let k = source.k();
    let Some(reference) = run_reference(n, k, source) else {
        return;
    };
    let packed = gemv::build(source, &meta(label)).expect("build");
    let header = &packed.header;
    let weights = header.weights(&packed.blob);
    let mut mismatches = 0usize;
    for row in 0..n {
        let ours = &weights[row * header.row_bytes..row * header.row_bytes + k / 2];
        let theirs = &reference.packed[row * (k / 2)..(row + 1) * (k / 2)];
        if ours != theirs {
            mismatches += 1;
        }
        assert_eq!(
            packed.scale_f32[row].to_bits(),
            reference.scales[row].to_bits(),
            "{label}: row {row} scale differs"
        );
    }
    assert_eq!(mismatches, 0, "{label}: {mismatches} rows differ from C");
}

#[test]
fn matches_c_reference_on_synthetic() {
    // K not a multiple of 32, a zero row, an extreme row.
    let k = 100usize;
    let n = 5usize;
    let mut data = Vec::with_capacity(k * n);
    let mut rng = Rng::new(0xabc);
    for row in 0..n {
        for col in 0..k {
            let value = match row {
                0 => 0.0,
                1 => rng.signed() * 1e-3,
                2 => {
                    if col % 17 == 0 {
                        -7.0
                    } else {
                        0.25
                    }
                }
                _ => rng.signed() * 3.0,
            };
            data.push(value);
        }
    }
    compare_with_c("synthetic", &matrix(k, n, data));
}

#[test]
fn matches_c_reference_on_real_tensors() {
    let model = model_path();
    if !model.exists() {
        eprintln!("skip: {} not found", model.display());
        return;
    }
    // Conv layer 0 and the first attention layer (2), the shapes the DSP graph
    // will actually see.
    for (layer, role) in [
        (0usize, "in_proj"),
        (0, "out_proj"),
        (2, "attn_q"),
        (2, "attn_k"),
        (2, "attn_v"),
        (2, "attn_o"),
    ] {
        let name = gemv::tensor_for_role(layer, role).expect("role");
        // attn_k is compared in full; the rest keep the test fast.
        let rows = if role == "attn_k" { None } else { Some(8) };
        let (source, _) = gguf_source::load_matrix(&model, &name, rows).expect("load");
        compare_with_c(&name, &source);
    }
}

#[test]
fn cli_pack_verify_unpack() {
    let model = model_path();
    if !model.exists() {
        eprintln!("skip: {} not found", model.display());
        return;
    }
    let dir = temp_dir("cli");
    let blob = dir.join("attn_k.gemv");
    let manifest = dir.join("attn_k.json");
    let decoded = dir.join("attn_k.f32");
    let binary = env!("CARGO_BIN_EXE_myelin-pack");

    let status = Command::new(binary)
        .args(["gemv", "--model"])
        .arg(&model)
        .args(["--layer", "2", "--role", "attn_k", "--rows", "16"])
        .arg("--out")
        .arg(&blob)
        .arg("--manifest")
        .arg(&manifest)
        .status()
        .expect("run gemv");
    assert!(status.success());
    assert!(blob.exists() && manifest.exists());

    let status = Command::new(binary)
        .args(["gemv-verify", "--blob"])
        .arg(&blob)
        .arg("--manifest")
        .arg(&manifest)
        .arg("--model")
        .arg(&model)
        .status()
        .expect("run gemv-verify");
    assert!(status.success());

    let status = Command::new(binary)
        .args(["gemv-unpack", "--blob"])
        .arg(&blob)
        .arg("--manifest")
        .arg(&manifest)
        .arg("--out")
        .arg(&decoded)
        .status()
        .expect("run gemv-unpack");
    assert!(status.success());
    assert_eq!(read_f32(&decoded).len(), 16 * 2048);
}
