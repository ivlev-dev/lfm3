//! Shortconv blob: layout invariants, quality, validation and the byte-for-byte
//! comparison with the C builder the DSP stand uses.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use myelin_pack::gguf_source;
use myelin_pack::shortconv::{
    self, blob_bytes, kq_base, ActivationScales, ConvManifest, ConvQuality,
};
use myelin_pack::PackError;

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

    fn gaussian(&mut self) -> f32 {
        (0..6).map(|_| self.signed()).sum::<f32>() / 3.0
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("myelin-pack-{tag}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn model_path() -> PathBuf {
    repo_root().join("models/LFM3-1.7B-Q4_K_M.gguf")
}

fn write_f32(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).expect("write f32 file");
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("read f32 file");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

static REFERENCE: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Compile `tests/c/conv_blob_ref.c` against the real `conv_math.h`.
fn reference_binary() -> Option<&'static Path> {
    REFERENCE
        .get_or_init(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let include = repo_root().join("myelin/dsp/conv");
            if !include.join("conv_math.h").exists() {
                eprintln!(
                    "skip: {} not found (dsp sources unavailable)",
                    include.display()
                );
                return None;
            }
            let source = manifest_dir.join("tests/c/conv_blob_ref.c");
            let dir = temp_dir("conv-ref");
            let binary = dir.join("conv_blob_ref");
            let status = Command::new("cc")
                .args(["-std=c99", "-O2", "-I"])
                .arg(&include)
                .arg(&source)
                .arg("-o")
                .arg(&binary)
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
    blob: Vec<u8>,
    ksh: i32,
}

fn run_reference(binary: &Path, h: usize, s_bc: f32, s_y: f32, taps: &[f32]) -> Reference {
    let dir = temp_dir("conv-case");
    let taps_path = dir.join("taps.f32");
    let out_path = dir.join("blob.bin");
    write_f32(&taps_path, taps);
    let output = Command::new(binary)
        .arg(h.to_string())
        .arg(s_bc.to_string())
        .arg(s_y.to_string())
        .arg(&taps_path)
        .arg(&out_path)
        .output()
        .expect("run C reference");
    assert!(
        output.status.success(),
        "C reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let ksh = stdout
        .lines()
        .find_map(|line| line.strip_prefix("ksh "))
        .map(|value| value.trim().parse::<i32>().expect("ksh"))
        .expect("C reference prints ksh");
    Reference {
        blob: fs::read(&out_path).expect("C blob"),
        ksh,
    }
}

fn random_taps(rng: &mut Rng, h: usize) -> Vec<f32> {
    let mut taps = Vec::with_capacity(h * shortconv::TAPS);
    for channel in 0..h {
        for tap in 0..shortconv::TAPS {
            let value = if channel % 97 == 0 && tap == 3 {
                // occasional heavy tail, like the real filters
                rng.signed() * 40.0
            } else {
                rng.gaussian() * 0.05
            };
            taps.push(value);
        }
    }
    taps
}

fn assert_quality(quality: &ConvQuality, h: usize) {
    assert!(
        quality.cos_total >= 0.999,
        "cos_total = {}",
        quality.cos_total
    );
    assert!(
        quality.cos_min_channel >= 0.99,
        "cos_min_channel = {}",
        quality.cos_min_channel
    );
    assert!(
        quality.max_abs_err_over_scale <= 0.5001,
        "max |d| / scale = {}",
        quality.max_abs_err_over_scale
    );
    assert!(quality.samples == h * shortconv::TAPS);
    // one tap per channel is the maximum itself; ties can add a second one
    assert!(
        quality.saturated <= 2 * h,
        "saturated {}",
        quality.saturated
    );
}

#[test]
fn blob_matches_c_reference_on_synthetic_taps() {
    let Some(binary) = reference_binary() else {
        return;
    };
    let mut rng = Rng::new(0x5eed_1234);
    let cases = [
        (16usize, 0.01f32, 0.05f32),
        (32, 0.0031, 0.0217),
        (48, 1.0, 0.5),
        (2048, 0.01, 0.05),
        (2048, 0.0173, 0.0411),
    ];
    for (h, s_bc, s_y) in cases {
        let taps = random_taps(&mut rng, h);
        let scales = ActivationScales::cli(s_bc, s_y);
        let blob = shortconv::build(&taps, h, scales).expect("build");
        let reference = run_reference(binary, h, s_bc, s_y, &taps);
        assert_eq!(blob.ksh, reference.ksh, "H={h} s_bc={s_bc} s_y={s_y}");
        assert_eq!(blob.blob.len(), reference.blob.len(), "H={h}: blob size");
        if let Some(index) = blob
            .blob
            .iter()
            .zip(reference.blob.iter())
            .position(|(a, b)| a != b)
        {
            let channel = index / 8;
            panic!(
                "H={h} s_bc={s_bc} s_y={s_y}: first mismatch at byte {index} \
                 (phase block / channel {channel}): rust {:02x} != c {:02x}",
                blob.blob[index], reference.blob[index]
            );
        }
    }
}

#[test]
fn blob_matches_c_reference_on_edge_taps() {
    let Some(binary) = reference_binary() else {
        return;
    };
    let h = 64;
    let mut taps = vec![0.0f32; h * shortconv::TAPS];
    // channel 0: all zero, channel 1: only the last tap, channel 2: +-max,
    // channel 3: values just below/above the half-step rounding boundary.
    taps[7] = -1.0;
    for tap in 0..shortconv::TAPS {
        taps[2 * shortconv::TAPS + tap] = if tap % 2 == 0 { 1.0 } else { -1.0 };
        taps[3 * shortconv::TAPS + tap] =
            0.5 / 127.0 * 127.0 * if tap % 2 == 0 { 1.0 } else { -1.0 };
    }
    for channel in 4..h {
        taps[channel * shortconv::TAPS] = 1.0e-6;
        taps[channel * shortconv::TAPS + 6] = -1.0e-6;
    }
    for (s_bc, s_y) in [(0.01f32, 0.05f32), (0.9, 0.001)] {
        let blob = shortconv::build(&taps, h, ActivationScales::cli(s_bc, s_y)).unwrap();
        let reference = run_reference(binary, h, s_bc, s_y, &taps);
        assert_eq!(blob.ksh, reference.ksh);
        assert_eq!(blob.blob, reference.blob);
    }
}

#[test]
fn blob_matches_c_reference_on_real_layer() {
    let Some(binary) = reference_binary() else {
        return;
    };
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    for layer in [0usize, 5, 23] {
        let name = format!("blk.{layer}.shortconv.conv.weight");
        let (matrix, _) = gguf_source::load_matrix(&path, &name, None).expect("layer taps");
        assert_eq!(matrix.k(), shortconv::TAPS);
        let taps = matrix.data();
        let h = matrix.n();
        for (s_bc, s_y) in [(0.01f32, 0.05f32), (0.0142, 0.0389)] {
            let blob = shortconv::build(taps, h, ActivationScales::cli(s_bc, s_y)).unwrap();
            let reference = run_reference(binary, h, s_bc, s_y, taps);
            assert_eq!(blob.ksh, reference.ksh, "{name} {s_bc}/{s_y}");
            assert_eq!(blob.blob, reference.blob, "{name} {s_bc}/{s_y}");
        }
    }
}

#[test]
fn real_layer_quality_metrics() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    for layer in [0usize, 5, 23] {
        let name = format!("blk.{layer}.shortconv.conv.weight");
        let (matrix, _) = gguf_source::load_matrix(&path, &name, None).expect("layer taps");
        let h = matrix.n();
        let blob = shortconv::build(matrix.data(), h, ActivationScales::stand_default()).unwrap();
        let quality = shortconv::measure(matrix.data(), h, &blob.blob).unwrap();
        println!(
            "layer {layer}: cos={:.6} min_channel={:.6} max|d|/s={:.4} saturated={}",
            quality.cos_total,
            quality.cos_min_channel,
            quality.max_abs_err_over_scale,
            quality.saturated
        );
        assert!(quality.cos_total >= 0.999, "{name}");
        assert!(quality.cos_min_channel >= 0.99, "{name}");
        assert!(quality.max_abs_err_over_scale <= 0.5001, "{name}");
        assert!(blob.ksh >= 1 && blob.ksh <= 40);
        shortconv::verify_structure(&blob.blob, h).unwrap();
    }
}

#[test]
fn phase_rotation_layout_is_consistent() {
    let h = 64;
    let mut rng = Rng::new(7);
    let taps = random_taps(&mut rng, h);
    let blob = shortconv::build(&taps, h, ActivationScales::stand_default()).unwrap();

    let codes = shortconv::decode_tap_codes(&blob.blob, h).unwrap();
    assert_eq!(codes, blob.tap_codes);
    for channel in 0..h {
        for tap in 0..shortconv::TAPS {
            // tap 0 is the oldest sample; the DSP reads it from slot (tap+2)&7
            // at phase 0 and from the rotating slot in every other phase.
            let slot = (tap + 2) & 7;
            for phase in 0..shortconv::RING {
                let slot_in_phase = (slot + phase) & 7;
                assert_eq!(shortconv::slot_tap(phase, slot_in_phase), tap);
                let base = (phase * (h / shortconv::GROUP) + channel / shortconv::GROUP) * 128
                    + (channel % shortconv::GROUP) * 8;
                assert_eq!(
                    blob.blob[base + slot_in_phase],
                    codes[channel * shortconv::TAPS + tap] as u8
                );
            }
        }
        let base = (channel / shortconv::GROUP) * 128 + (channel % shortconv::GROUP) * 8;
        assert_eq!(blob.blob[base + 1], 0, "unused ring slot must stay zero");
    }
    assert_eq!(
        shortconv::decode_kq(&blob.blob, h).unwrap(),
        blob.kq,
        "kq words live at 8*H*8"
    );
    assert_eq!(kq_base(h), 8 * h * 8);
    assert_eq!(blob.blob.len(), blob_bytes(h));
    shortconv::verify_structure(&blob.blob, h).unwrap();
}

#[test]
fn quality_metrics_of_random_and_zero_taps() {
    let mut rng = Rng::new(99);
    let h = 128;
    let taps = random_taps(&mut rng, h);
    let blob = shortconv::build(&taps, h, ActivationScales::stand_default()).unwrap();
    let quality = shortconv::measure(&taps, h, &blob.blob).unwrap();
    assert_quality(&quality, h);
    assert_eq!(quality.per_channel.len(), h);

    let zeros = vec![0.0f32; h * shortconv::TAPS];
    let blob = shortconv::build(&zeros, h, ActivationScales::stand_default()).unwrap();
    let quality = shortconv::measure(&zeros, h, &blob.blob).unwrap();
    assert_eq!(quality.cos_total, 1.0);
    assert_eq!(quality.max_abs_err, 0.0);
    assert_eq!(quality.saturated, 0);
    assert!(blob.blob[..8 * h * 8].iter().all(|byte| *byte == 0));
}

#[test]
fn rejects_bad_geometry_and_values() {
    let taps = vec![0.0f32; 16 * shortconv::TAPS];
    assert!(matches!(
        shortconv::build(&taps, 0, ActivationScales::stand_default()),
        Err(PackError::Shortconv(_))
    ));
    assert!(matches!(
        shortconv::build(&taps, 15, ActivationScales::stand_default()),
        Err(PackError::Shortconv(_))
    ));
    assert!(matches!(
        shortconv::build(&taps, 17, ActivationScales::stand_default()),
        Err(PackError::Shortconv(_))
    ));
    assert!(matches!(
        shortconv::build(&taps, 16 * 1024, ActivationScales::stand_default()),
        Err(PackError::Shortconv(_))
    ));
    assert!(matches!(
        shortconv::build(&taps[..10], 16, ActivationScales::stand_default()),
        Err(PackError::Shortconv(_))
    ));
    let mut bad = taps.clone();
    bad[3] = f32::NAN;
    assert!(matches!(
        shortconv::build(&bad, 16, ActivationScales::stand_default()),
        Err(PackError::NonFiniteWeight { .. })
    ));
    for (s_bc, s_y) in [
        (0.0f32, 0.05f32),
        (0.01, 0.0),
        (-1.0, 0.05),
        (0.01, f32::INFINITY),
    ] {
        assert!(matches!(
            shortconv::build(&taps, 16, ActivationScales::cli(s_bc, s_y)),
            Err(PackError::Shortconv(_))
        ));
    }
}

#[test]
fn manifest_round_trip_and_tamper_detection() {
    let mut rng = Rng::new(1234);
    let h = 32;
    let taps = random_taps(&mut rng, h);
    let blob = shortconv::build(&taps, h, ActivationScales::cli(0.01, 0.05)).unwrap();
    let manifest = ConvManifest::new(&blob, "synthetic".to_string(), None);
    manifest.verify_blob(&blob.blob).unwrap();

    let dir = temp_dir("conv-manifest");
    let path = dir.join("conv.json");
    manifest.write(&path).unwrap();
    let read = ConvManifest::read(&path).unwrap();
    read.verify_blob(&blob.blob).unwrap();
    assert_eq!(read.h, h);
    assert_eq!(read.ksh, blob.ksh);
    assert_eq!(read.kq_base, kq_base(h));
    assert_eq!(read.command.src_len, 3 * h);
    assert_eq!(read.command.dst_len, h);
    assert_eq!(read.command.ring_state_bytes_per_slot, 2 * h * 8);

    let mut broken = blob.blob.clone();
    broken[5] ^= 0x40;
    assert!(matches!(
        read.verify_blob(&broken),
        Err(PackError::ManifestMismatch(_))
    ));

    let mut kq_broken = blob.blob.clone();
    let word = kq_base(h);
    kq_broken[word] ^= 1;
    assert!(matches!(
        read.verify_blob(&kq_broken),
        Err(PackError::ManifestMismatch(_))
    ));

    let mut wrong_scales = read.clone();
    wrong_scales.activation_scales.s_y = 0.0;
    assert!(matches!(
        wrong_scales.verify_blob(&blob.blob),
        Err(PackError::Shortconv(_))
    ));

    let mut wrong_len = blob.blob.clone();
    wrong_len.pop();
    assert!(read.verify_blob(&wrong_len).is_err());
}

#[test]
fn cli_conv_matches_c_reference() {
    let Some(binary) = reference_binary() else {
        return;
    };
    let mut rng = Rng::new(4242);
    let h = 32;
    let taps = random_taps(&mut rng, h);
    let dir = temp_dir("conv-cli");
    let taps_path = dir.join("taps.f32");
    let out_path = dir.join("conv.blob");
    let manifest_path = dir.join("conv.json");
    let taps_out = dir.join("taps.out.f32");
    write_f32(&taps_path, &taps);

    let bin = env!("CARGO_BIN_EXE_myelin-pack");
    let output = Command::new(bin)
        .args(["conv", "--taps"])
        .arg(&taps_path)
        .arg("--out")
        .arg(&out_path)
        .arg("--manifest")
        .arg(&manifest_path)
        .args(["--s-bc", "0.01", "--s-y", "0.05", "--taps-out"])
        .arg(&taps_out)
        .output()
        .expect("run myelin-pack conv");
    assert!(
        output.status.success(),
        "conv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("cos total"), "stdout: {stdout}");

    let written_taps = read_f32(&taps_out);
    assert_eq!(written_taps, taps);
    let reference = run_reference(binary, h, 0.01, 0.05, &written_taps);
    let blob = fs::read(&out_path).expect("blob");
    assert_eq!(blob, reference.blob);

    let output = Command::new(bin)
        .args(["conv-verify", "--blob"])
        .arg(&out_path)
        .arg("--manifest")
        .arg(&manifest_path)
        .arg("--taps")
        .arg(&taps_path)
        .output()
        .expect("run myelin-pack conv-verify");
    assert!(
        output.status.success(),
        "conv-verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bit-exact"), "stdout: {stdout}");
}
