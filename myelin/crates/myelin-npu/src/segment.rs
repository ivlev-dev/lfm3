//! Per-256 int4 weight scales for the `lfmgemv` int4 path.
//!
//! # Why
//!
//! `myelin-pack gemv` stores one `absmax/7` scale per row (`K` = 2048..7680
//! inputs). A single outlier anywhere in the row then sets the step for the
//! whole row, and the int4 round trip of the remaining 2000-odd weights is
//! coarse: `myelin/docs/quality-int4-cpu.md` measures relRMSE 0.184 and a
//! degenerate text loop at that granularity, 0.139 and coherent text at one
//! scale per 256 inputs (+2.7 % bytes).
//!
//! This module keeps the blob layout untouched (row-major `[N][K/2]`, two
//! nibbles per byte, offset-8) and adds a second, finer table next to it:
//!
//! * the blob still carries the per-row scale `max|w_row|/7` in its f32/Q16
//!   tables, so `myelin-pack` can parse and verify it unchanged;
//! * one `u8` ratio per (row, 256-input segment) says how much smaller that
//!   segment's scale is: `seg = row_scale * (ratio + 1) / 128`. The byte 127
//!   is exactly the row scale, so a row without outliers costs 8-30 bytes
//!   instead of 32-120;
//! * the ratios travel in a small sidecar file (`<blob>.seg`) next to the
//!   blob and are named by `pack.json`; old packs simply do not have one.
//!
//! The ratio costs 1/256 of the row scale (0.39 %) in the worst case, which
//! is one thirtieth of the int4 weight error (relRMSE 0.14) and below the int8
//! output LSB (1/127) for segments within 3x of the row scale. Storing f32
//! segment scales instead would cost 4x the bytes; the DSP harness measures
//! the actual deviation on a real layer (`dsp/hvx/seg_main.c`).
//!
//! # Rounding
//!
//! Codes are rounded with `round_ties_even`, like [`myelin_pack::gemv`] and
//! the HVX stand's reference `ffn_pack_i4_row()` (`lrintf`). The diagnostic
//! CPU control in `myelin-core/src/backend/cpu.rs` uses `round()` (half away
//! from zero); the two differ only when `w/step` is exactly `x.5`, which
//! changes no measured metric (`quality-int4-cpu.md`).

use std::fmt;

use myelin_pack::gemv::{GemvHeader, HEADER_BYTES, SCALE_Q16_ONE};
use myelin_pack::{GemvBlob, GemvQuality, Matrix};

/// Inputs that share one int4 scale.
pub const SEGMENT: usize = 256;

/// Stored ratio of a segment whose scale equals the row scale (`q == 128`).
pub const RATIO_ONE: u8 = 127;

/// Ratio levels: `q = ratio + 1` is in `1..=RATIO_LEVELS`, 128 = row scale.
pub const RATIO_LEVELS: u32 = 128;

/// Sidecar magic (`MSQ2`: Myelin Segment Q8, format family 2).
pub const SEG_MAGIC: [u8; 4] = *b"MSQ2";

/// Sidecar header: magic, version, `nseg`, reserved, `n_padded`, `k_padded`.
pub const SEG_HEADER_BYTES: usize = 16;

/// Sidecar format version understood by this reader.
pub const SEG_VERSION: u8 = 1;

/// Failure while packing, reading or applying a segment table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentError(pub String);

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SegmentError {}

/// One `u8` ratio per (row, 256-input segment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segments {
    n_padded: usize,
    nseg: usize,
    ratios: Vec<u8>,
}

impl Segments {
    /// Adopt a ratio table after checking its shape.
    pub fn new(
        n_padded: usize,
        k_padded: usize,
        ratios: Vec<u8>,
    ) -> Result<Self, SegmentError> {
        if n_padded == 0 || k_padded == 0 || k_padded % SEGMENT != 0 {
            return Err(SegmentError(format!(
                "segment table needs a positive {SEGMENT}-multiple K, got {k_padded}"
            )));
        }
        let nseg = k_padded / SEGMENT;
        if ratios.len() != n_padded * nseg {
            return Err(SegmentError(format!(
                "segment table has {} ratios for {n_padded} rows x {nseg} segments",
                ratios.len()
            )));
        }
        Ok(Self {
            n_padded,
            nseg,
            ratios,
        })
    }

    /// Segments per row (`K / 256`).
    pub const fn nseg(&self) -> usize {
        self.nseg
    }

    /// Rows the table covers (padded output channels).
    pub const fn n_padded(&self) -> usize {
        self.n_padded
    }

    /// Raw Q8 ratios, `n_padded * nseg` values.
    pub fn ratios(&self) -> &[u8] {
        &self.ratios
    }

    /// Ratios of one row.
    pub fn row(&self, row: usize) -> &[u8] {
        &self.ratios[row * self.nseg..(row + 1) * self.nseg]
    }

    /// `row_scale * (ratio + 1) / 128`: the scale the kernel uses.
    pub fn scale(&self, row_scale: f32, row: usize, segment: usize) -> f32 {
        row_scale * f32::from(self.ratios[row * self.nseg + segment] + 1)
            / RATIO_LEVELS as f32
    }

    /// Sidecar bytes: header, then the ratios.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SEG_HEADER_BYTES + self.ratios.len());
        out.extend_from_slice(&SEG_MAGIC);
        out.push(SEG_VERSION);
        out.push(self.nseg as u8);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(self.n_padded as u32).to_le_bytes());
        out.extend_from_slice(&((self.nseg * SEGMENT) as u32).to_le_bytes());
        out.extend_from_slice(&self.ratios);
        out
    }

    /// Parse sidecar bytes written by [`Segments::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SegmentError> {
        if bytes.len() < SEG_HEADER_BYTES {
            return Err(SegmentError(format!(
                "segment table is {} bytes, header is {SEG_HEADER_BYTES}",
                bytes.len()
            )));
        }
        if bytes[0..4] != SEG_MAGIC {
            return Err(SegmentError("segment table magic is not MSQ2".to_owned()));
        }
        if bytes[4] != SEG_VERSION {
            return Err(SegmentError(format!(
                "segment table version {} is not {SEG_VERSION}",
                bytes[4]
            )));
        }
        let nseg = bytes[5] as usize;
        let n_padded = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
        let k_padded = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes")) as usize;
        if nseg == 0 || nseg * SEGMENT != k_padded {
            return Err(SegmentError(format!(
                "segment table says {nseg} segments, K says {k_padded}"
            )));
        }
        let expected = SEG_HEADER_BYTES + n_padded * nseg;
        if bytes.len() != expected {
            return Err(SegmentError(format!(
                "segment table is {} bytes, header says {expected}",
                bytes.len()
            )));
        }
        Self::new(n_padded, k_padded, bytes[SEG_HEADER_BYTES..].to_vec())
    }

    /// Bytes this table appends to the `set_scale` payload after the per-row
    /// f32 scales: the magic, the segment count and the ratios.
    pub fn payload_tail(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.ratios.len());
        out.extend_from_slice(&SEG_MAGIC);
        out.extend_from_slice(&(self.nseg as u32).to_le_bytes());
        out.extend_from_slice(&self.ratios);
        out
    }
}

/// A blob packed with per-256 scales plus the table that describes it.
#[derive(Debug, Clone)]
pub struct SegPacked {
    /// Complete `myelin-pack` v1 gemv blob (weights + per-row scale tables).
    pub blob: Vec<u8>,
    /// Ratio table the kernel needs.
    pub segments: Segments,
    /// `n_padded` per-row scales, exactly the blob's f32 table.
    pub row_scales: Vec<f32>,
    /// Quality of the per-256 round trip.
    pub quality: GemvQuality,
    /// `||w - ŵ|| / ||w||` over the per-256 round trip.
    pub rel_rmse: f64,
}

impl SegPacked {
    /// Wrap as a [`GemvBlob`] so `GemvManifest::from_blob` can describe it.
    pub fn gemv_blob(&self) -> GemvBlob {
        let header = GemvHeader::parse(&self.blob).expect("SegPacked wrote its own header");
        let scale_q16: Vec<i32> = self
            .row_scales
            .iter()
            .map(|scale| (scale * SCALE_Q16_ONE as f32).round() as i32)
            .collect();
        GemvBlob {
            header,
            blob: self.blob.clone(),
            scale_f32: self.row_scales[..self.real_rows()].to_vec(),
            scale_q16: scale_q16[..self.real_rows()].to_vec(),
            max_abs: self
                .row_scales
                .iter()
                .map(|scale| scale * 7.0)
                .collect::<Vec<f32>>()[..self.real_rows()]
                .to_vec(),
            quality: self.quality.clone(),
        }
    }

    fn real_rows(&self) -> usize {
        GemvHeader::parse(&self.blob)
            .expect("SegPacked wrote its own header")
            .n
    }
}

/// Quantise `matrix` (`k x n`) with one int4 scale per 256 inputs.
///
/// The blob itself stays a valid `myelin-pack` v1 int4 GEMV blob: its header,
/// weight section and per-row `max|w|/7` tables are byte-for-byte what the old
/// packer would write for the same *row* scale. Only the nibbles differ (each
/// 256-block uses its own step) and the ratio sidecar describes that.
pub fn build(matrix: &Matrix) -> Result<SegPacked, SegmentError> {
    let (k, n) = (matrix.k(), matrix.n());
    let header = GemvHeader::new(k, n).map_err(|error| SegmentError(error.to_string()))?;
    if k % SEGMENT != 0 {
        return Err(SegmentError(format!(
            "K = {k} is not a multiple of {SEGMENT}; pad the matrix before packing"
        )));
    }
    let nseg = k / SEGMENT;
    let mut blob = vec![0u8; header.total_bytes()];
    blob[..HEADER_BYTES].copy_from_slice(&header.encode());

    let mut row_scales = vec![0.0f32; header.n_padded];
    let mut ratios = vec![RATIO_ONE; header.n_padded * nseg];
    let mut row_stats = Vec::with_capacity(n);
    let mut dot_all = 0.0f64;
    let mut orig_all = 0.0f64;
    let mut dec_all = 0.0f64;
    let mut err_all = 0.0f64;
    let mut max_err_over_scale = 0.0f64;
    let mut saturated = 0usize;

    for row in 0..n {
        let values = matrix.row(row);
        for (col, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SegmentError(format!(
                    "{row},{col}: non-finite weight {value}"
                )));
            }
        }
        let mut row_max = 0.0f32;
        for &value in values {
            row_max = row_max.max(value.abs());
        }
        // Same convention as `myelin-pack`: an all-zero row gets 1.0 so the
        // multiplier stays positive; every code is 0 and the output is 0.
        let row_scale = if row_max > 0.0 { row_max / 7.0 } else { 1.0 };
        row_scales[row] = row_scale;

        let base = header.weight_base + row * header.row_bytes;
        let mut row_dot = 0.0f64;
        let mut row_orig = 0.0f64;
        let mut row_dec = 0.0f64;
        let mut row_err = 0.0f64;
        let mut row_max_err = 0.0f64;
        let mut row_sat = 0usize;
        for (segment, block) in values.chunks(SEGMENT).enumerate() {
            let seg_max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let seg_scale = if seg_max > 0.0 { seg_max / 7.0 } else { row_scale };
            let levels = (seg_scale / row_scale * RATIO_LEVELS as f32)
                .round_ties_even()
                .clamp(1.0, RATIO_LEVELS as f32);
            let ratio_u8 = (levels as u32 - 1) as u8;
            ratios[row * nseg + segment] = ratio_u8;
            // Exactly what the DSP will decode: the 7-bit ratio, not the exact
            // segment scale, is the step.
            let step = row_scale * (f32::from(ratio_u8) + 1.0) / RATIO_LEVELS as f32;
            for (offset, &value) in block.iter().enumerate() {
                let rounded = (value / step).round_ties_even().clamp(-8.0, 7.0);
                let code = rounded as i8;
                let k_index = segment * SEGMENT + offset;
                let nibble = ((code + 8) & 0x0f) as u8;
                let byte = &mut blob[base + k_index / 2];
                if k_index % 2 == 0 {
                    *byte |= nibble;
                } else {
                    *byte |= nibble << 4;
                }
                let decoded = f64::from(code) * f64::from(step);
                let original = f64::from(value);
                row_dot += original * decoded;
                row_orig += original * original;
                row_dec += decoded * decoded;
                row_err += (original - decoded) * (original - decoded);
                let relative = (original - decoded).abs() / f64::from(step).max(1e-30);
                row_max_err = row_max_err.max(relative);
                if code == -8 || code == 7 {
                    row_sat += 1;
                }
            }
        }
        let cos = cosine(row_dot, row_orig, row_dec);
        dot_all += row_dot;
        orig_all += row_orig;
        dec_all += row_dec;
        err_all += row_err;
        max_err_over_scale = max_err_over_scale.max(row_max_err);
        saturated += row_sat;
        row_stats.push(myelin_pack::gemv::GemvChannelStats {
            row,
            scale: row_scale,
            max_abs: row_max,
            cos,
            max_abs_err_over_scale: row_max_err,
            saturated: row_sat,
            saturated_fraction: row_sat as f64 / k as f64,
        });
    }

    for (row, &scale) in row_scales.iter().enumerate() {
        let offset = header.scale_base + row * 4;
        blob[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
        let q16 = (scale * SCALE_Q16_ONE as f32).round() as i32;
        let q_offset = header.scale_q16_base + row * 4;
        blob[q_offset..q_offset + 4].copy_from_slice(&q16.to_le_bytes());
    }

    let samples = k * n;
    let cos_min_row = row_stats.iter().map(|row| row.cos).fold(1.0f64, f64::min);
    let cos_mean_row = if row_stats.is_empty() {
        1.0
    } else {
        row_stats.iter().map(|row| row.cos).sum::<f64>() / row_stats.len() as f64
    };
    let quality = GemvQuality {
        cos_total: cosine(dot_all, orig_all, dec_all),
        cos_min_row,
        cos_mean_row,
        max_abs_err_over_scale: max_err_over_scale,
        saturated,
        saturated_fraction: if samples == 0 {
            0.0
        } else {
            saturated as f64 / samples as f64
        },
        samples,
        per_row: row_stats,
    };
    let rel_rmse = if orig_all > 0.0 {
        (err_all / orig_all).sqrt()
    } else {
        0.0
    };
    Ok(SegPacked {
        blob,
        segments: Segments::new(header.n_padded, header.k_padded, ratios)?,
        row_scales,
        quality,
        rel_rmse,
    })
}

fn cosine(dot: f64, a_sq: f64, b_sq: f64) -> f64 {
    if a_sq == 0.0 && b_sq == 0.0 {
        return 1.0;
    }
    if a_sq == 0.0 || b_sq == 0.0 {
        return 0.0;
    }
    dot / (a_sq.sqrt() * b_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(k: usize, n: usize, seed: u64) -> Matrix {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut data = Vec::with_capacity(k * n);
        for _row in 0..n {
            for col in 0..k {
                // One outlier per 256-block: the case per-row scales cannot
                // represent and per-256 can.
                let mut value = next() * 2.0 - 1.0;
                // One outlier per 1024 inputs: 3 of 4 segments are clean,
                // which is the case a per-row scale cannot represent.
                if col % 1024 == 0 {
                    value *= 6.0;
                }
                data.push(value);
            }
        }
        Matrix::new(k, n, data).expect("matrix")
    }

    #[test]
    fn sidecar_round_trips() {
        let segments = Segments::new(4, 512, (0..8).map(|i| i as u8 * 15 + 3).collect()).unwrap();
        let bytes = segments.encode();
        assert_eq!(bytes.len(), SEG_HEADER_BYTES + 8);
        assert_eq!(&bytes[0..4], b"MSQ2");
        assert_eq!(Segments::decode(&bytes).unwrap(), segments);
    }

    #[test]
    fn sidecar_rejects_wrong_length_and_magic() {
        let segments = Segments::new(2, 512, vec![127; 4]).unwrap();
        let mut bytes = segments.encode();
        bytes.pop();
        assert!(Segments::decode(&bytes).is_err());
        let mut bytes = segments.encode();
        bytes[0] = b'X';
        assert!(Segments::decode(&bytes).is_err());
    }

    #[test]
    fn ratio_table_is_smaller_than_f32_and_keeps_the_row_scale() {
        let segments = Segments::new(3, 768, vec![127, 64, 32, 127, 16, 1, 100, 50, 7]).unwrap();
        assert_eq!(segments.nseg(), 3);
        assert_eq!(segments.row(1), &[127, 16, 1]);
        assert_eq!(segments.scale(2.0, 1, 0), 2.0);
        assert!((segments.scale(2.0, 1, 1) - 2.0 * 17.0 / 128.0).abs() < 1e-6);
        assert_eq!(&segments.payload_tail()[0..4], b"MSQ2");
        assert_eq!(
            u32::from_le_bytes(segments.payload_tail()[4..8].try_into().unwrap()),
            3
        );
    }

    #[test]
    fn per_256_beats_per_row_on_block_outliers() {
        let matrix = matrix(1024, 16, 0x51ed);
        let packed = build(&matrix).expect("pack");
        assert_eq!(packed.segments.nseg(), 4);
        assert_eq!(packed.blob.len(), GemvHeader::new(1024, 16).unwrap().total_bytes());
        // The per-row packer of the same matrix, for the comparison.
        let per_row = myelin_pack::gemv::build(
            &matrix,
            &myelin_pack::PackMeta {
                tensor: "synthetic".to_owned(),
                source: None,
            },
        )
        .expect("per-row pack");
        let decoded = myelin_pack::gemv::unpack(&per_row.blob).expect("unpack");
        let mut err = 0.0f64;
        let mut orig = 0.0f64;
        for (value, restored) in matrix.data().iter().zip(&decoded) {
            let delta = f64::from(*value) - f64::from(*restored);
            err += delta * delta;
            orig += f64::from(*value) * f64::from(*value);
        }
        let row_rel = (err / orig).sqrt();
        assert!(
            packed.rel_rmse < 0.7 * row_rel,
            "per-256 relRMSE {} is not well below the per-row {row_rel}",
            packed.rel_rmse
        );
        assert!(
            packed.quality.cos_total > per_row.quality.cos_total,
            "per-256 cos {} <= per-row cos {}",
            packed.quality.cos_total,
            per_row.quality.cos_total
        );
    }

    #[test]
    fn packed_nibbles_decode_with_the_ratio_step() {
        let matrix = matrix(512, 3, 7);
        let packed = build(&matrix).expect("pack");
        let header = GemvHeader::parse(&packed.blob).unwrap();
        let row = packed.segments.row(1);
        for (segment, ratio) in row.iter().enumerate() {
            let step =
                packed.row_scales[1] * (f32::from(*ratio) + 1.0) / RATIO_LEVELS as f32;
            for offset in 0..SEGMENT {
                let k = segment * SEGMENT + offset;
                let byte = packed.blob[header.weight_base + header.row_bytes + k / 2];
                let code = if k % 2 == 0 {
                    (byte & 0x0f) as i32 - 8
                } else {
                    (byte >> 4) as i32 - 8
                };
                let decoded = code as f32 * step;
                let original = matrix.row(1)[k];
                assert!((decoded - original).abs() <= step * 0.5001 + 1e-6);
            }
        }
    }

    #[test]
    fn all_zero_row_keeps_the_multiplier_positive() {
        let matrix = Matrix::new(256, 1, vec![0.0; 256]).unwrap();
        let packed = build(&matrix).expect("pack");
        assert_eq!(packed.row_scales[0], 1.0);
        assert!(packed.segments.row(0).iter().all(|r| *r == RATIO_ONE));
        assert_eq!(packed.quality.saturated, 0);
    }
}
