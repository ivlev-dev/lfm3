//! KV page images: both DSP layouts, round-trip quality, padding, boundaries,
//! manifest checks, and a bit-exact comparison with the `kvarn` reference codec.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use half::f16;
use lfm_npu_ref::KvFormat as RefKvFormat;
use lfm_npu_ref::KvarnCodec as RefKvarnCodec;

use myelin_pack::gguf_source;
use myelin_pack::kvarn::{self, KvarnFormat, KvarnManifest};
use myelin_pack::kvpage::{self, KvLayout, KvManifest, KvPackParams, KvQuality, VScaleMode};
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

fn params(heads: usize, pages: usize, tokens: usize, layout: KvLayout) -> KvPackParams {
    KvPackParams {
        heads,
        pages,
        tokens_in_last_page: tokens,
        layout,
        v_scale_mode: VScaleMode::Channel,
    }
}

fn values(rng: &mut Rng, heads: usize, pages: usize, tokens: usize) -> Vec<f32> {
    let count = heads * ((pages - 1) * kvpage::PAGE_TOKENS + tokens) * kvpage::HEAD_DIM;
    (0..count).map(|_| rng.gaussian() * 0.1).collect()
}

fn write_f32(path: &std::path::Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).expect("write f32");
}

fn assert_kv_quality(quality: &KvQuality) {
    assert!(
        quality.k_cos_total >= 0.99,
        "K cos total {}",
        quality.k_cos_total
    );
    assert!(
        quality.k_cos_min_token >= 0.95,
        "K cos min token {}",
        quality.k_cos_min_token
    );
    assert!(quality.k_max_abs_err_over_scale <= 0.5001);
    assert!(
        quality.v_cos_total >= 0.999,
        "V cos total {}",
        quality.v_cos_total
    );
    assert!(quality.v_max_abs_err_over_scale <= 0.5001);
}

#[test]
fn row_and_page_layouts_decode_identically() {
    let mut rng = Rng::new(11);
    let input = values(&mut rng, 2, 2, 32);
    let row = kvpage::build(&input, params(2, 2, 32, KvLayout::Row)).unwrap();
    let page = kvpage::build(&input, params(2, 2, 32, KvLayout::Page)).unwrap();

    assert_eq!(row.k.len(), 2 * 2 * kvpage::PAGE_TOKENS * kvpage::HEAD_DIM);
    assert_eq!(page.k.len(), 2 * 2 * 16 * 128);
    assert_eq!(page.v.len(), 2 * 2 * 4096);
    let (row_k, row_v) = row.decode();
    let (page_k, page_v) = page.decode();
    assert_eq!(row_k, page_k);
    assert_eq!(row_v, page_v);
    row.verify_structure().unwrap();
    page.verify_structure().unwrap();
    let row_manifest = KvManifest::new(&row, None);
    assert_eq!(row_manifest.k.view_bytes, row.k.len());
    assert_eq!(row_manifest.k.page_stride, 0);
}

#[test]
fn k_and_v_round_trip_quality() {
    let mut rng = Rng::new(22);
    for layout in [KvLayout::Row, KvLayout::Page] {
        let input = values(&mut rng, 4, 2, 32);
        let image = kvpage::build(&input, params(4, 2, 32, layout)).unwrap();
        let quality = kvpage::measure(&input, &image).unwrap();
        println!(
            "{layout:?}: K cos {:.6} (min {:.6}), V cos {:.6}, K sat {}",
            quality.k_cos_total, quality.k_cos_min_token, quality.v_cos_total, quality.k_saturated
        );
        assert_kv_quality(&quality);
        assert_eq!(quality.k_samples, input.len());
    }
}

#[test]
fn global_v_scale_works_too() {
    let mut rng = Rng::new(33);
    let input = values(&mut rng, 2, 1, 32);
    let mut p = params(2, 1, 32, KvLayout::Row);
    p.v_scale_mode = VScaleMode::Global;
    let image = kvpage::build(&input, p).unwrap();
    assert_eq!(image.v_scales.len(), 1);
    let quality = kvpage::measure(&input, &image).unwrap();
    assert_kv_quality(&quality);
}

#[test]
fn partial_last_page_is_zero_padded() {
    let mut rng = Rng::new(44);
    let input = values(&mut rng, 2, 2, 17);
    for layout in [KvLayout::Row, KvLayout::Page] {
        let image = kvpage::build(&input, params(2, 2, 17, layout)).unwrap();
        let slots = image.params.slots();
        let valid = image.params.valid_tokens();
        for head in 0..2 {
            for slot in valid..slots {
                assert_eq!(image.k_scales_q16[head * slots + slot], 0);
                // zero bytes: int4 decodes to -8, int8 to 0; both scale to 0.
                assert_eq!(image.decode_k_codes(head, slot), [-8i8; kvpage::HEAD_DIM]);
                assert_eq!(image.decode_v_codes(head, slot), [0i8; kvpage::HEAD_DIM]);
            }
        }
        let quality = kvpage::measure(&input, &image).unwrap();
        assert_kv_quality(&quality);
        image.verify_structure().unwrap();
    }
}

#[test]
fn zeros_stay_zero_and_finite() {
    let input = vec![0.0f32; 2 * kvpage::PAGE_TOKENS * kvpage::HEAD_DIM];
    let image = kvpage::build(&input, params(2, 1, 32, KvLayout::Page)).unwrap();
    assert!(image.k_scales_q16.iter().all(|kt| *kt == 0));
    assert!(image.v_scales.iter().all(|scale| *scale == 0.0));
    for head in 0..2 {
        for slot in 0..kvpage::PAGE_TOKENS {
            assert_eq!(image.decode_k_codes(head, slot), [0i8; kvpage::HEAD_DIM]);
        }
    }
    assert!(image.v.iter().all(|byte| *byte == 0));
    let (keys, vals) = image.decode();
    assert!(keys.iter().all(|value| *value == 0.0));
    assert!(vals.iter().all(|value| *value == 0.0));
    let quality = kvpage::measure(&input, &image).unwrap();
    assert_eq!(quality.k_cos_total, 1.0);
    assert_eq!(quality.v_cos_total, 1.0);
}

#[test]
fn k_scale_must_fit_the_q16_kernel_multiplier() {
    let mut ok = vec![0.0f32; kvpage::PAGE_TOKENS * kvpage::HEAD_DIM];
    ok[0] = 400.0;
    let image = kvpage::build(&ok, params(1, 1, 32, KvLayout::Row)).unwrap();
    assert!(image.k_scales_q16[0] > 0);

    let mut too_big = vec![0.0f32; kvpage::PAGE_TOKENS * kvpage::HEAD_DIM];
    too_big[0] = 500.0;
    let error = kvpage::build(&too_big, params(1, 1, 32, KvLayout::Row)).unwrap_err();
    assert!(matches!(error, PackError::KvPage(_)), "{error}");
}

#[test]
fn rejects_bad_input() {
    let good = vec![0.0f32; kvpage::PAGE_TOKENS * kvpage::HEAD_DIM];
    for bad in [0usize, 33] {
        assert!(matches!(
            kvpage::build(&good, params(1, 1, bad, KvLayout::Row)),
            Err(PackError::KvPage(_))
        ));
    }
    assert!(matches!(
        kvpage::build(&good, params(0, 1, 32, KvLayout::Row)),
        Err(PackError::KvPage(_))
    ));
    assert!(matches!(
        kvpage::build(&good, params(1, 0, 32, KvLayout::Row)),
        Err(PackError::KvPage(_))
    ));
    assert!(matches!(
        kvpage::build(&good[..10], params(1, 1, 32, KvLayout::Row)),
        Err(PackError::KvPage(_))
    ));
    let mut nan = good.clone();
    nan[7] = f32::NAN;
    assert!(matches!(
        kvpage::build(&nan, params(1, 1, 32, KvLayout::Row)),
        Err(PackError::NonFiniteWeight { .. })
    ));
    let mut inf = good;
    inf[7] = f32::INFINITY;
    assert!(matches!(
        kvpage::build(&inf, params(1, 1, 32, KvLayout::Row)),
        Err(PackError::NonFiniteWeight { .. })
    ));
}

#[test]
fn manifest_round_trip_and_tamper_detection() {
    let mut rng = Rng::new(55);
    let input = values(&mut rng, 4, 2, 32);
    let image = kvpage::build(&input, params(4, 2, 32, KvLayout::Page)).unwrap();
    let manifest = KvManifest::new(&image, None);
    manifest
        .verify_images(&image.k, &image.kt, &image.v)
        .unwrap();
    assert_eq!(manifest.k.head_stride, manifest.k.bytes_per_head);
    assert_eq!(manifest.k.page_stride, 16 * 128);
    assert_eq!(manifest.v.page_stride, 4096);
    // token-major K needs a view of heads * slots * 128 bytes for the kernel.
    assert_eq!(manifest.k.view_bytes, 4 * 2 * kvpage::PAGE_TOKENS * 128);
    assert!(manifest.k.view_bytes > manifest.k.bytes);
    assert_eq!(manifest.v.view_bytes, manifest.v.bytes);
    assert_eq!(manifest.kt.view_bytes, manifest.kt.bytes);
    assert_eq!(manifest.kt.token_stride, 4);
    assert_eq!(manifest.command.kv_heads, 4);
    assert_eq!(manifest.command.head_dim, 128);

    let dir = temp_dir("kv-manifest");
    let path = dir.join("kv.json");
    manifest.write(&path).unwrap();
    let read = KvManifest::read(&path).unwrap();
    read.verify_images(&image.k, &image.kt, &image.v).unwrap();
    assert_eq!(read.params().unwrap(), image.params);

    let mut broken_k = image.k.clone();
    broken_k[3] ^= 0x10;
    assert!(matches!(
        read.verify_images(&broken_k, &image.kt, &image.v),
        Err(PackError::ManifestMismatch(_))
    ));
    let mut broken_kt = image.kt.clone();
    broken_kt[4] ^= 1;
    assert!(matches!(
        read.verify_images(&image.k, &broken_kt, &image.v),
        Err(PackError::ManifestMismatch(_))
    ));
    let mut broken_v = image.v.clone();
    broken_v[0] ^= 1;
    assert!(matches!(
        read.verify_images(&image.k, &image.kt, &broken_v),
        Err(PackError::ManifestMismatch(_))
    ));
    let mut short = image.v.clone();
    short.pop();
    assert!(matches!(
        read.verify_images(&image.k, &image.kt, &short),
        Err(PackError::ManifestMismatch(_))
    ));
}

#[test]
fn kvarn_matches_reference_bit_for_bit() {
    for seed in 0..4u64 {
        let mut rng = Rng::new(1000 + seed);
        let mut bits = Vec::with_capacity(kvarn::PAGE_TOKENS * kvarn::HEAD_DIM);
        for _ in 0..kvarn::PAGE_TOKENS * kvarn::HEAD_DIM {
            bits.push(f16::from_f32(rng.signed()));
        }

        for (ours, reference) in [
            (KvarnFormat::K4V2, RefKvFormat::KvarnK4V2),
            (KvarnFormat::K4V4, RefKvFormat::KvarnK4V4),
        ] {
            let mut our_rotated = bits.clone();
            kvarn::rot_fwd_page(&mut our_rotated).unwrap();
            let mut ref_rotated = bits.clone();
            for row in ref_rotated.chunks_exact_mut(kvarn::HEAD_DIM) {
                let row: &mut [f16; kvarn::HEAD_DIM] = row.try_into().expect("row");
                lfm_npu_ref::kvarn::rot_fwd_f16(row);
            }
            let our_bits: Vec<u16> = our_rotated.iter().map(|x| x.to_bits()).collect();
            let ref_bits: Vec<u16> = ref_rotated.iter().map(|x| x.to_bits()).collect();
            assert_eq!(our_bits, ref_bits, "rotation differs (seed {seed})");

            let page = kvarn::commit(ours, &our_rotated).unwrap();
            let reference_page = RefKvarnCodec::new(reference)
                .unwrap()
                .commit(&ref_rotated)
                .unwrap();

            assert_eq!(page.kcode, reference_page.kcode, "kcode seed {seed}");
            assert_eq!(page.vcode, reference_page.vcode, "vcode seed {seed}");
            for (name, a, b) in [
                ("ka", &page.ka, &reference_page.ka),
                ("kc", &page.kc, &reference_page.kc),
                ("vb", &page.vb, &reference_page.vb),
                ("vbz", &page.vbz, &reference_page.vbz),
                ("vc", &page.vc, &reference_page.vc),
            ] {
                let a: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
                let b: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
                assert_eq!(a, b, "{name} seed {seed}");
            }
            assert_eq!(page.encoded_len(), reference_page.encoded_len());
            assert_eq!(page.encoded_len(), ours.page_bytes());

            for token in 0..kvarn::PAGE_TOKENS {
                let ours_k: Vec<u32> = page.decode_k(token).iter().map(|x| x.to_bits()).collect();
                let ref_k: Vec<u32> = reference_page
                    .decode_k(token)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                assert_eq!(ours_k, ref_k, "decode_k token {token} seed {seed}");
                let ours_v: Vec<u32> = page.decode_v(token).iter().map(|x| x.to_bits()).collect();
                let ref_v: Vec<u32> = reference_page
                    .decode_v(token)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                assert_eq!(ours_v, ref_v, "decode_v token {token} seed {seed}");
            }

            let encoded = page.encode();
            let parsed = kvarn::KvarnPage::decode(ours, &encoded).unwrap();
            assert_eq!(parsed, page);
        }
    }
}

#[test]
fn kvarn_rotation_round_trips() {
    let mut rng = Rng::new(77);
    let original: Vec<f32> = (0..kvarn::PAGE_TOKENS * kvarn::HEAD_DIM)
        .map(|_| rng.signed())
        .collect();
    let mut rotated = original.clone();
    for row in rotated.chunks_exact_mut(kvarn::HEAD_DIM) {
        let row: &mut [f32; kvarn::HEAD_DIM] = row.try_into().expect("row");
        kvarn::rot_fwd(row);
    }
    for row in rotated.chunks_exact_mut(kvarn::HEAD_DIM) {
        let row: &mut [f32; kvarn::HEAD_DIM] = row.try_into().expect("row");
        kvarn::rot_inv(row);
    }
    for (a, b) in original.iter().zip(rotated.iter()) {
        assert!((a - b).abs() < 1e-3, "{a} != {b}");
    }
}

#[test]
fn kvarn_page_sizes_match_layout() {
    assert_eq!(KvarnFormat::K4V2.page_bytes(), 3776);
    assert_eq!(KvarnFormat::K4V4.page_bytes(), 4800);
    for (format, reference) in [
        (KvarnFormat::K4V2, RefKvFormat::KvarnK4V2),
        (KvarnFormat::K4V4, RefKvFormat::KvarnK4V4),
    ] {
        assert_eq!(
            format.page_bytes() as u64,
            lfm_npu_ref::page_bytes(reference, 1, 128)
        );
    }
}

fn real_kv_page() -> Option<Vec<f32>> {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return None;
    }
    // 4 KV heads x 32 tokens x 128 dims = 8 rows of blk.12.attn_k.weight.
    let count = 4 * kvpage::PAGE_TOKENS * kvpage::HEAD_DIM;
    let (values, _) =
        gguf_source::load_f32_prefix(&path, "blk.12.attn_k.weight", count).expect("attn_k prefix");
    Some(values)
}

#[test]
fn real_layer_page_metrics() {
    let Some(input) = real_kv_page() else {
        return;
    };
    for layout in [KvLayout::Row, KvLayout::Page] {
        let image = kvpage::build(&input, params(4, 1, 32, layout)).unwrap();
        let quality = kvpage::measure(&input, &image).unwrap();
        println!(
            "{layout:?}: K cos {:.6} (min {:.6}), V cos {:.6} (min {:.6}), \
             K max|d|/s {:.4}, V max|d|/s {:.4}, K sat {}",
            quality.k_cos_total,
            quality.k_cos_min_token,
            quality.v_cos_total,
            quality.v_cos_min_token,
            quality.k_max_abs_err_over_scale,
            quality.v_max_abs_err_over_scale,
            quality.k_saturated
        );
        assert!(quality.k_cos_total >= 0.99);
        assert!(quality.v_cos_total >= 0.999);
        assert!(quality.k_max_abs_err_over_scale <= 0.5001);
        assert!(quality.v_max_abs_err_over_scale <= 0.5001);
    }
}

#[test]
fn real_layer_kvarn_metrics() {
    let Some(input) = real_kv_page() else {
        return;
    };
    let mut rotated = vec![f16::ZERO; kvarn::PAGE_TOKENS * kvarn::HEAD_DIM];
    for (dst, src) in rotated.iter_mut().zip(input.iter()) {
        *dst = f16::from_f32(*src);
    }
    kvarn::rot_fwd_page(&mut rotated).unwrap();
    for format in [KvarnFormat::K4V2, KvarnFormat::K4V4] {
        let page = kvarn::commit(format, &rotated).unwrap();
        let mut k_cos = Vec::new();
        let mut v_cos = Vec::new();
        for token in 0..kvarn::PAGE_TOKENS {
            let orig = &input[token * kvarn::HEAD_DIM..(token + 1) * kvarn::HEAD_DIM];
            let mut k = page.decode_k(token);
            let mut v = page.decode_v(token);
            kvarn::rot_inv(&mut k);
            kvarn::rot_inv(&mut v);
            k_cos.push(cosine(orig, &k));
            v_cos.push(cosine(orig, &v));
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let min = |v: &[f32]| v.iter().copied().fold(1.0f32, f32::min);
        println!(
            "{}: K cos mean {:.6} min {:.6}, V cos mean {:.6} min {:.6}",
            format.name(),
            mean(&k_cos),
            min(&k_cos),
            mean(&v_cos),
            min(&v_cos)
        );
        // 2-bit V is limited by construction: four levels on heavy-tailed
        // weight rows give ~0.89; the reference codec reports the same.
        assert!(mean(&k_cos) >= 0.98, "{} K", format.name());
        assert!(mean(&v_cos) >= 0.85, "{} V", format.name());
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

#[test]
fn cli_kv_round_trip() {
    let mut rng = Rng::new(88);
    let input = values(&mut rng, 2, 2, 20);
    let dir = temp_dir("kv-cli");
    let input_path = dir.join("page.f32");
    write_f32(&input_path, &input);
    let manifest = dir.join("kv.json");
    let k = dir.join("k.bin");
    let kt = dir.join("kt.bin");
    let v = dir.join("v.bin");
    let decoded = dir.join("decoded.f32");
    let bin = env!("CARGO_BIN_EXE_myelin-pack");

    let output = Command::new(bin)
        .arg("kv")
        .arg("--in")
        .arg(&input_path)
        .args(["--heads", "2", "--pages", "2", "--tokens", "20"])
        .args(["--layout", "page", "--v-scale", "channel"])
        .arg("--out-k")
        .arg(&k)
        .arg("--out-kt")
        .arg(&kt)
        .arg("--out-v")
        .arg(&v)
        .arg("--manifest")
        .arg(&manifest)
        .output()
        .expect("run kv");
    assert!(
        output.status.success(),
        "kv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("K cos"), "stdout: {stdout}");
    assert!(stdout.contains("V cos"), "stdout: {stdout}");

    let output = Command::new(bin)
        .arg("kv-verify")
        .arg("--manifest")
        .arg(&manifest)
        .arg("--k")
        .arg(&k)
        .arg("--kt")
        .arg(&kt)
        .arg("--v")
        .arg(&v)
        .arg("--in")
        .arg(&input_path)
        .output()
        .expect("run kv-verify");
    assert!(
        output.status.success(),
        "kv-verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("bit-exact"));

    let output = Command::new(bin)
        .arg("kv-unpack")
        .arg("--manifest")
        .arg(&manifest)
        .arg("--k")
        .arg(&k)
        .arg("--kt")
        .arg(&kt)
        .arg("--v")
        .arg(&v)
        .arg("--out")
        .arg(&decoded)
        .output()
        .expect("run kv-unpack");
    assert!(
        output.status.success(),
        "kv-unpack failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(&decoded).expect("decoded");
    assert_eq!(bytes.len(), 2 * input.len() * 4);
}

#[test]
fn cli_kvarn_round_trip() {
    let mut rng = Rng::new(99);
    let input = values(&mut rng, 2, 1, 32);
    let dir = temp_dir("kvarn-cli");
    let input_path = dir.join("page.f32");
    write_f32(&input_path, &input);
    let manifest = dir.join("kvarn.json");
    let page = dir.join("page.kvarn");
    let decoded = dir.join("decoded.f32");
    let bin = env!("CARGO_BIN_EXE_myelin-pack");

    for format in ["kvarn-k4v2", "kvarn-k4v4"] {
        let output = Command::new(bin)
            .arg("kv")
            .arg("--in")
            .arg(&input_path)
            .args(["--heads", "2", "--format", format])
            .arg("--out")
            .arg(&page)
            .arg("--manifest")
            .arg(&manifest)
            .output()
            .expect("run kv kvarn");
        assert!(
            output.status.success(),
            "kv {format} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let read = KvarnManifest::read(&manifest).unwrap();
        assert_eq!(read.variant, format);
        assert!(read.rotate);

        let output = Command::new(bin)
            .arg("kv-verify")
            .arg("--manifest")
            .arg(&manifest)
            .arg("--page")
            .arg(&page)
            .arg("--in")
            .arg(&input_path)
            .output()
            .expect("run kv-verify kvarn");
        assert!(
            output.status.success(),
            "kv-verify {format} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("bit-exact"));

        let output = Command::new(bin)
            .arg("kv-unpack")
            .arg("--manifest")
            .arg(&manifest)
            .arg("--page")
            .arg(&page)
            .arg("--out")
            .arg(&decoded)
            .output()
            .expect("run kv-unpack kvarn");
        assert!(
            output.status.success(),
            "kv-unpack {format} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = fs::read(&decoded).expect("decoded");
        assert_eq!(bytes.len(), 2 * input.len() * 4);
    }
}
