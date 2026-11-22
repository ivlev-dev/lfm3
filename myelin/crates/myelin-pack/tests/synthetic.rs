//! Synthetic round-trip tests: quantization, crownstone addressing, padding,
//! manifest checks and input validation.

use myelin_pack::crownstone::{BlobHeader, TILE_BYTES};
use myelin_pack::manifest::PackMeta;
use myelin_pack::quant::{decode_scale, encode_scale};
use myelin_pack::{pack, quantize_row, unpack, DType, Matrix, PackError, PackOptions};

fn options() -> PackOptions {
    PackOptions { dtype: DType::I8 }
}

fn meta(name: &str) -> PackMeta {
    PackMeta {
        tensor: name.to_string(),
        source: None,
    }
}

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

    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn matrix_from_fn(k: usize, n: usize, mut f: impl FnMut(usize, usize) -> f32) -> Matrix {
    let mut data = Vec::with_capacity(k * n);
    for channel in 0..n {
        for index in 0..k {
            data.push(f(channel, index));
        }
    }
    Matrix::new(k, n, data).unwrap()
}

#[test]
fn scale_word_matches_report18() {
    // 0x6000 == 512.0 in fp16, i.e. multiplier 1.0 on the device.
    assert_eq!(encode_scale(1.0), Some(0x6000));
    assert_eq!(decode_scale(0x6000), 1.0);
    assert_eq!(decode_scale(encode_scale(0.25).unwrap()), 0.25);
}

#[test]
fn random_matrix_round_trip() {
    let k = 37; // not a multiple of 32 on purpose
    let n = 45; // neither is N
    let mut rng = Rng::new(0x5eed);
    let weights = matrix_from_fn(k, n, |_, _| rng.uniform() * 2.0 - 1.0);

    let result = pack(&weights, &meta("random"), options()).unwrap();
    let decoded = unpack(&result.blob).unwrap();

    assert_eq!(decoded.k(), k);
    assert_eq!(decoded.n(), n);
    assert_eq!(decoded.k_padded(), 64);
    assert_eq!(decoded.n_padded(), 64);
    assert_eq!(
        result.blob.len(),
        BlobHeader::new(k, n).unwrap().total_bytes()
    );

    assert!(result.quality.cos_total >= 0.999, "{:?}", result.quality);
    assert!(result.quality.cos_min_channel >= 0.999);
    assert!(result.quality.max_abs_err_over_scale <= 0.5001);
    assert!(result.quality.saturated >= n);

    // Padded channels must be flat zero with a zero scale.
    let padded = decoded.dequantized().unwrap();
    assert_eq!(padded.k(), k);
    for channel in n..decoded.n_padded() {
        assert_eq!(decoded.scales()[channel], 0.0);
    }
    // Padded K slices of real channels are absent from the cropped matrix:
    // check the raw codes instead.
    for channel in 0..n {
        let row =
            &decoded.codes()[channel * decoded.k_padded()..(channel + 1) * decoded.k_padded()];
        assert!(row[k..].iter().all(|value| *value == 0));
    }
    for channel in n..decoded.n_padded() {
        let row =
            &decoded.codes()[channel * decoded.k_padded()..(channel + 1) * decoded.k_padded()];
        assert!(row.iter().all(|value| *value == 0));
        assert_eq!(decoded.scale_bits()[channel], 0);
    }
}

#[test]
fn all_zero_matrix_stays_zero() {
    let weights = matrix_from_fn(64, 32, |_, _| 0.0);
    let result = pack(&weights, &meta("zeros"), options()).unwrap();
    assert_eq!(result.quality.cos_total, 1.0);
    assert_eq!(result.quality.max_abs_err, 0.0);
    assert!(result.blob[128..].iter().all(|byte| *byte == 0));

    let decoded = unpack(&result.blob).unwrap();
    assert!(decoded.scales().iter().all(|scale| *scale == 0.0));
    assert!(decoded.codes().iter().all(|value| *value == 0));
}

#[test]
fn all_ones_is_exact() {
    // max_abs = 1 -> scale = 1/127 -> q = 127 -> dequantized exactly 1.0.
    let weights = matrix_from_fn(32, 32, |_, _| 1.0);
    let result = pack(&weights, &meta("ones"), options()).unwrap();
    assert_eq!(result.quality.cos_total, 1.0);
    // The scale is stored as fp16, so a 1.0 weight can come back a fraction
    // of 2^-11 away; the cosine is still exactly 1.
    assert!(result.quality.max_abs_err < 1e-3);
    assert_eq!(result.quality.saturated, 32 * 32);
    assert!(result.quantized.iter().all(|row| row.clamped == 0));
}

#[test]
fn alternating_extremes_are_exact() {
    let weights = matrix_from_fn(64, 32, |_, index| if index % 2 == 0 { 1.5 } else { -1.5 });
    let result = pack(&weights, &meta("extremes"), options()).unwrap();
    assert_eq!(result.quality.cos_total, 1.0);
    assert!(result.quality.max_abs_err < 2e-3);
    for channel in &result.quantized {
        assert_eq!(channel.clamped, 0);
        assert!(channel.q.iter().all(|value| value.abs() == 127));
    }
}

#[test]
fn single_spike_round_trips() {
    let weights = matrix_from_fn(32, 32, |channel, index| {
        if channel == 7 && index == 19 {
            0.75
        } else {
            0.0
        }
    });
    let result = pack(&weights, &meta("spike"), options()).unwrap();
    assert_eq!(result.quality.cos_total, 1.0);
    assert!(result.quality.max_abs_err < 1e-3);
    let decoded = unpack(&result.blob).unwrap();
    let row = &decoded.codes()[7 * 32..8 * 32];
    assert_eq!(row[19], 127);
    assert!(row
        .iter()
        .enumerate()
        .all(|(index, value)| index == 19 || *value == 0));
}

#[test]
fn tiny_channel_underflows_to_zero() {
    // 512 * (1e-12 / 127) is below the smallest fp16 subnormal.
    let weights = matrix_from_fn(32, 32, |_, index| if index == 0 { 1e-12 } else { 0.0 });
    let result = pack(&weights, &meta("tiny"), options()).unwrap();
    assert_eq!(result.quantized[0].scale, 0.0);
    assert_eq!(
        result.quantized[0]
            .q
            .iter()
            .filter(|value| **value != 0)
            .count(),
        0
    );
    assert_eq!(result.quality.saturated, 0);
    assert_eq!(result.quality.cos_total, 0.0); // original non-zero, decoded zero
    assert!(result.quality.max_abs_err <= 1e-12);
}

#[test]
fn packed_codes_match_quantization() {
    let mut rng = Rng::new(11);
    let weights = matrix_from_fn(96, 40, |_, _| rng.uniform() - 0.5);
    let result = pack(&weights, &meta("codes"), options()).unwrap();
    let decoded = unpack(&result.blob).unwrap();

    for channel in 0..weights.n() {
        let expected = &result.quantized[channel].q;
        let offset = channel * decoded.k_padded();
        assert_eq!(
            &decoded.codes()[offset..offset + weights.k()],
            expected.as_slice()
        );
        assert_eq!(decoded.scales()[channel], result.quantized[channel].scale);
    }
}

#[test]
fn tile_order_is_k_major_with_1024_stride() {
    let weights = matrix_from_fn(96, 64, |_, _| 0.5);
    let result = pack(&weights, &meta("tiles"), options()).unwrap();
    let manifest = &result.manifest;
    assert_eq!(manifest.k_tiles, 3);
    assert_eq!(manifest.n_tiles, 2);
    assert_eq!(manifest.tiles.len(), 6);
    assert_eq!(manifest.tile_order, "k_tile_major");

    for (index, tile) in manifest.tiles.iter().enumerate() {
        let expected_k = index / manifest.n_tiles;
        let expected_n = index % manifest.n_tiles;
        assert_eq!(tile[0], expected_k);
        assert_eq!(tile[1], expected_n);
        assert_eq!(tile[2], manifest.weight_base + index * TILE_BYTES);
        assert_eq!(tile[3], manifest.bias_base + expected_n * 256);
    }

    // The 2nd N tile of one K chunk sits at +1024, the next K chunk at +2048:
    // the layout used by `i8_pack_wgt(..., wl=4)` on the stand.
    assert_eq!(manifest.tiles[1][2] - manifest.tiles[0][2], 1024);
    assert_eq!(manifest.tiles[2][2] - manifest.tiles[1][2], 1024);
}

#[test]
fn manifest_round_trips_and_verifies() {
    let weights = matrix_from_fn(40, 33, |channel, index| {
        ((channel * 31 + index * 17) % 23) as f32 / 23.0 - 0.5
    });
    let mut result = pack(&weights, &meta("manifest"), options()).unwrap();
    result.manifest.source = Some(myelin_pack::SourceInfo {
        file: "synthetic.gguf".to_string(),
        ggml_type: "F32".to_string(),
    });

    let json = result.manifest.to_json().unwrap();
    let parsed: myelin_pack::Manifest = serde_json::from_str(&json).unwrap();
    parsed.verify_blob(&result.blob).unwrap();
    assert_eq!(parsed.tensor, "manifest");
    assert_eq!(parsed.k_padded, 64);
    assert_eq!(parsed.n_padded, 64);
    assert_eq!(parsed.scales.scale_f32.len(), 64);
    assert_eq!(parsed.tiles.len(), 4);

    // Truncated blob is rejected.
    assert!(matches!(
        parsed.verify_blob(&result.blob[..result.blob.len() - 1]),
        Err(PackError::BlobTooSmall { .. })
    ));
    // Flip one byte: checksum catches it.
    let mut tampered = result.blob.clone();
    let offset = manifest_checksum_offset(&tampered);
    tampered[offset] ^= 0x01;
    assert!(matches!(
        parsed.verify_blob(&tampered),
        Err(PackError::ManifestMismatch(_))
    ));
}

fn manifest_checksum_offset(blob: &[u8]) -> usize {
    // First byte after the header: inside the first weight tile.
    blob.len().min(200)
}

#[test]
fn packing_is_deterministic() {
    let mut rng = Rng::new(3);
    let weights = matrix_from_fn(64, 32, |_, _| rng.uniform() * 4.0 - 2.0);
    let first = pack(&weights, &meta("deterministic"), options()).unwrap();
    let second = pack(&weights, &meta("deterministic"), options()).unwrap();
    assert_eq!(first.blob, second.blob);
    assert_eq!(first.manifest.blob_fnv1a64, second.manifest.blob_fnv1a64);
}

#[test]
fn non_finite_weights_are_rejected() {
    let weights = matrix_from_fn(32, 32, |_, index| if index == 5 { f32::NAN } else { 0.0 });
    assert!(matches!(
        pack(&weights, &meta("nan"), options()),
        Err(PackError::NonFiniteWeight { col: 5, .. })
    ));
}

#[test]
fn empty_rows_are_rejected() {
    assert!(matches!(
        quantize_row(&[]),
        Err(PackError::EmptyRow { row: 0 })
    ));
}

#[test]
fn corrupted_blob_is_rejected() {
    let weights = matrix_from_fn(32, 32, |_, _| 0.25);
    let result = pack(&weights, &meta("corrupt"), options()).unwrap();

    let mut bad_magic = result.blob.clone();
    bad_magic[0] = b'Z';
    assert!(matches!(
        unpack(&bad_magic),
        Err(PackError::BadMagic { .. })
    ));

    let mut bad_version = result.blob.clone();
    bad_version[4] = 9;
    assert!(matches!(
        unpack(&bad_version),
        Err(PackError::UnsupportedBlobVersion(9))
    ));

    let mut bad_dtype = result.blob.clone();
    bad_dtype[6] = 7;
    assert!(matches!(
        unpack(&bad_dtype),
        Err(PackError::UnsupportedBlobDType(7))
    ));

    let mut bad_shape = result.blob.clone();
    bad_shape[8..12].copy_from_slice(&33u32.to_le_bytes());
    assert!(matches!(
        unpack(&bad_shape),
        Err(PackError::InconsistentBlob(_))
    ));

    assert!(matches!(
        unpack(&result.blob[..64]),
        Err(PackError::BlobTooSmall { .. })
    ));
}

#[test]
fn matrix_shape_is_validated() {
    assert!(matches!(
        Matrix::new(4, 4, vec![0.0; 15]),
        Err(PackError::MatrixShape {
            expected: 16,
            actual: 15,
            ..
        })
    ));
    assert!(matches!(
        Matrix::new(0, 4, vec![]),
        Err(PackError::EmptyMatrix { k: 0, n: 4 })
    ));
}
