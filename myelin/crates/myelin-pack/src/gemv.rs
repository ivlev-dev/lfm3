//! int4 GEMV weights for the HVX `gemv` kernel.
//!
//! The DSP kernel (`gemv_rows_m1_*` in `myelin/dsp/hvx/lfmgemv_impl.c`) reads
//! a row-major `[N][K/2]` matrix, two signed int4 per byte:
//!
//! ```text
//! byte = lo | (hi << 4)      lo -> even k, hi -> odd k
//! stored nibble = w_signed + 8   (offset-8, 0..15)
//! ```
//!
//! The dot is corrected once per row with `-8 * sum(activation)`.  The host
//! reference that defines the quantisation is `ffn_pack_i4_row()` in
//! `myelin/dsp/hvx/ffn_ref.h`: per-row `sw = max|w| / 7`, codes rounded with
//! `lrintf` (ties to even), clamped to `[-8, 7]`.
//!
//! `lfmgemv_set_shape()` rejects `K % 256 != 0`, so the packer pads `K` to
//! [`K_ALIGN`] (`= 256`).  Rows are padded with zero weights and a zero scale,
//! which makes the extra output channels exactly zero.  `N` is padded to
//! [`N_ALIGN`] (`= 32`) for the same reason (aligned output, no special case at
//! the end of the matrix); the kernel itself accepts any `N`.
//!
//! Blob layout, all integers little endian:
//!
//! ```text
//! [0   .. 128)        header
//! [128 .. ...)        weights [n_padded][k_padded / 2], row-major
//! [... .. ...)        sw f32   [n_padded]   (exact C per-row scale)
//! [... .. ...)        sw q16 i32[n_padded]  (round(sw * 65536))
//! ```
//!
//! `set_scale()` consumes the f32 table (the kernel computes
//! `m = sw * sa / so` itself); the Q16 table is the canonical integer form for
//! manifests, comparison and host-side dequantisation.

use serde::{Deserialize, Serialize};

use crate::crownstone::fnv1a64;
use crate::manifest::{PackMeta, SourceInfo};
use crate::{Matrix, PackError, Result};

/// File magic of the binary blob.
pub const BLOB_MAGIC: [u8; 4] = *b"MPG1";
/// Stable name written into the manifest.
pub const FORMAT_NAME: &str = "myelin-dsp-gemv-int4";
/// Blob format version understood by this crate.
pub const FORMAT_VERSION: u16 = 1;
/// Header size; also the weight base (HVX loads want 128-byte alignment).
pub const HEADER_BYTES: usize = 128;
/// `K` is padded to this multiple: `lfmgemv_set_shape()` requires `K % 256 == 0`.
pub const K_ALIGN: usize = 256;
/// `N` is padded to this multiple (aligned output rows, no tail case).
pub const N_ALIGN: usize = 32;
/// int4 weights never carry their own scale; the kernel gets it via `set_scale`.
pub const SCALE_Q16_ONE: i32 = 65_536;

const FLAG_K_PADDED: u8 = 1 << 0;
const FLAG_N_PADDED: u8 = 1 << 1;

/// Format identity that does not depend on git or on the tensor shape.
///
/// Two packers with different row order, nibble order, scale convention or
/// alignment must not accept each other's blobs, so the identity covers
/// exactly those choices.
pub fn format_build_id() -> u64 {
    fnv1a64(
        b"myelin-dsp-gemv-int4|v1|row_major_nk|i4_offset8_lo_even_hi_odd|k_align=256|n_align=32|scale=f32+q16",
    )
}

pub fn format_build_id_hex() -> String {
    format!("{:016x}", format_build_id())
}

fn round_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// Parsed blob header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemvHeader {
    pub version: u16,
    pub dtype_tag: u8,
    pub flags: u8,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub row_bytes: usize,
    pub header_bytes: usize,
    pub weight_base: usize,
    pub weight_bytes: usize,
    pub scale_base: usize,
    pub scale_bytes: usize,
    pub scale_q16_base: usize,
    pub scale_q16_bytes: usize,
    pub build_id: u64,
}

impl GemvHeader {
    pub fn new(k: usize, n: usize) -> Result<Self> {
        if k == 0 || n == 0 {
            return Err(PackError::EmptyMatrix { k, n });
        }
        let k_padded = round_up(k, K_ALIGN);
        let n_padded = round_up(n, N_ALIGN);
        let row_bytes = k_padded / 2;
        let weight_bytes = n_padded
            .checked_mul(row_bytes)
            .ok_or_else(|| PackError::InconsistentBlob("gemv weight size overflow".to_string()))?;
        let scale_bytes = n_padded
            .checked_mul(4)
            .ok_or_else(|| PackError::InconsistentBlob("gemv scale size overflow".to_string()))?;
        let weight_base = HEADER_BYTES;
        let scale_base = weight_base + weight_bytes;
        let scale_q16_base = scale_base + scale_bytes;
        let mut flags = 0u8;
        if k_padded != k {
            flags |= FLAG_K_PADDED;
        }
        if n_padded != n {
            flags |= FLAG_N_PADDED;
        }
        Ok(Self {
            version: FORMAT_VERSION,
            dtype_tag: 4,
            flags,
            k,
            n,
            k_padded,
            n_padded,
            row_bytes,
            header_bytes: HEADER_BYTES,
            weight_base,
            weight_bytes,
            scale_base,
            scale_bytes,
            scale_q16_base,
            scale_q16_bytes: scale_bytes,
            build_id: format_build_id(),
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.scale_q16_base + self.scale_q16_bytes
    }

    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..4].copy_from_slice(&BLOB_MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6] = self.dtype_tag;
        out[7] = self.flags;
        out[8..12].copy_from_slice(&(self.k as u32).to_le_bytes());
        out[12..16].copy_from_slice(&(self.n as u32).to_le_bytes());
        out[16..20].copy_from_slice(&(self.k_padded as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(self.n_padded as u32).to_le_bytes());
        out[24..28].copy_from_slice(&(self.row_bytes as u32).to_le_bytes());
        out[28..32].copy_from_slice(&(self.header_bytes as u32).to_le_bytes());
        out[32..40].copy_from_slice(&(self.weight_base as u64).to_le_bytes());
        out[40..48].copy_from_slice(&(self.weight_bytes as u64).to_le_bytes());
        out[48..56].copy_from_slice(&(self.scale_base as u64).to_le_bytes());
        out[56..64].copy_from_slice(&(self.scale_bytes as u64).to_le_bytes());
        out[64..72].copy_from_slice(&(self.scale_q16_base as u64).to_le_bytes());
        out[72..80].copy_from_slice(&(self.scale_q16_bytes as u64).to_le_bytes());
        out[80..88].copy_from_slice(&self.build_id.to_le_bytes());
        out
    }

    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() < HEADER_BYTES {
            return Err(PackError::BlobTooSmall {
                expected: HEADER_BYTES,
                actual: blob.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&blob[0..4]);
        if magic != BLOB_MAGIC {
            return Err(PackError::BadMagic {
                magic,
                expected: BLOB_MAGIC,
            });
        }
        let version = u16::from_le_bytes([blob[4], blob[5]]);
        if version != FORMAT_VERSION {
            return Err(PackError::UnsupportedBlobVersion(version));
        }
        let dtype_tag = blob[6];
        if dtype_tag != 4 {
            return Err(PackError::UnsupportedBlobDType(dtype_tag));
        }
        let header = Self {
            version,
            dtype_tag,
            flags: blob[7],
            k: read_u32(blob, 8)? as usize,
            n: read_u32(blob, 12)? as usize,
            k_padded: read_u32(blob, 16)? as usize,
            n_padded: read_u32(blob, 20)? as usize,
            row_bytes: read_u32(blob, 24)? as usize,
            header_bytes: read_u32(blob, 28)? as usize,
            weight_base: read_u64(blob, 32)? as usize,
            weight_bytes: read_u64(blob, 40)? as usize,
            scale_base: read_u64(blob, 48)? as usize,
            scale_bytes: read_u64(blob, 56)? as usize,
            scale_q16_base: read_u64(blob, 64)? as usize,
            scale_q16_bytes: read_u64(blob, 72)? as usize,
            build_id: read_u64(blob, 80)?,
        };
        header.validate(blob.len())?;
        Ok(header)
    }

    fn validate(&self, blob_len: usize) -> Result<()> {
        if self.k == 0 || self.n == 0 {
            return Err(PackError::InconsistentBlob(format!(
                "empty gemv shape {}x{}",
                self.k, self.n
            )));
        }
        let expected = Self::new(self.k, self.n)?;
        for (name, actual, wanted) in [
            ("k_padded", self.k_padded, expected.k_padded),
            ("n_padded", self.n_padded, expected.n_padded),
            ("row_bytes", self.row_bytes, expected.row_bytes),
            ("header_bytes", self.header_bytes, expected.header_bytes),
            ("weight_base", self.weight_base, expected.weight_base),
            ("weight_bytes", self.weight_bytes, expected.weight_bytes),
            ("scale_base", self.scale_base, expected.scale_base),
            ("scale_bytes", self.scale_bytes, expected.scale_bytes),
            (
                "scale_q16_base",
                self.scale_q16_base,
                expected.scale_q16_base,
            ),
            (
                "scale_q16_bytes",
                self.scale_q16_bytes,
                expected.scale_q16_bytes,
            ),
        ] {
            if actual != wanted {
                return Err(PackError::InconsistentBlob(format!(
                    "gemv {name}: header {actual} != layout {wanted}"
                )));
            }
        }
        if self.build_id != format_build_id() {
            return Err(PackError::InconsistentBlob(format!(
                "gemv build id {:016x} != {:016x}",
                self.build_id,
                format_build_id()
            )));
        }
        if self.total_bytes() != blob_len {
            return Err(PackError::InconsistentBlob(format!(
                "gemv blob is {blob_len} bytes, header says {}",
                self.total_bytes()
            )));
        }
        Ok(())
    }

    pub fn weights<'a>(&self, blob: &'a [u8]) -> &'a [u8] {
        &blob[self.weight_base..self.weight_base + self.weight_bytes]
    }

    pub fn scales(&self, blob: &[u8]) -> Vec<f32> {
        blob[self.scale_base..self.scale_base + self.scale_bytes]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    pub fn scales_q16(&self, blob: &[u8]) -> Vec<i32> {
        blob[self.scale_q16_base..self.scale_q16_base + self.scale_q16_bytes]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

fn read_u32(blob: &[u8], offset: usize) -> Result<u32> {
    let bytes = blob
        .get(offset..offset + 4)
        .ok_or_else(|| PackError::InconsistentBlob(format!("gemv header truncated at {offset}")))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(blob: &[u8], offset: usize) -> Result<u64> {
    let bytes = blob
        .get(offset..offset + 8)
        .ok_or_else(|| PackError::InconsistentBlob(format!("gemv header truncated at {offset}")))?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Per-row quality of the int4 round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvChannelStats {
    pub row: usize,
    /// `max |w| / 7`; `1.0` for an all-zero row, exactly like `ffn_pack_i4_row`.
    pub scale: f32,
    pub max_abs: f32,
    pub cos: f64,
    /// `max |w - decode(code)| / scale`; 0.5 is the int4 rounding limit.
    pub max_abs_err_over_scale: f64,
    /// Codes at the clamp boundary (`-8` or `+7`).
    pub saturated: usize,
    pub saturated_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvQuality {
    pub cos_total: f64,
    pub cos_min_row: f64,
    pub cos_mean_row: f64,
    pub max_abs_err_over_scale: f64,
    pub saturated: usize,
    pub saturated_fraction: f64,
    pub samples: usize,
    pub per_row: Vec<GemvChannelStats>,
}

impl GemvQuality {
    pub fn check(&self, min_cos: f64, min_row_cos: Option<f64>) -> Result<()> {
        if self.cos_total < min_cos {
            return Err(PackError::QualityBelowThreshold {
                metric: "cos_total".to_string(),
                actual: self.cos_total,
                threshold: min_cos,
            });
        }
        if let Some(threshold) = min_row_cos {
            if self.cos_min_row < threshold {
                return Err(PackError::QualityBelowThreshold {
                    metric: "cos_min_row".to_string(),
                    actual: self.cos_min_row,
                    threshold,
                });
            }
        }
        Ok(())
    }
}

/// A packed int4 GEMV matrix.
#[derive(Debug, Clone)]
pub struct GemvBlob {
    pub header: GemvHeader,
    pub blob: Vec<u8>,
    /// Per-row scale for real rows only (`n` values).
    pub scale_f32: Vec<f32>,
    pub scale_q16: Vec<i32>,
    pub max_abs: Vec<f32>,
    pub quality: GemvQuality,
}

impl GemvBlob {
    /// Source `[k][n]` matrix and the blob described by its header.
    pub fn new(header: GemvHeader, blob: Vec<u8>, source: &Matrix) -> Result<Self> {
        let all_scales = header.scales(&blob);
        let all_q16 = header.scales_q16(&blob);
        let mut scale_f32 = Vec::with_capacity(header.n);
        let mut scale_q16 = Vec::with_capacity(header.n);
        let mut max_abs = Vec::with_capacity(header.n);
        let mut quality_rows = Vec::with_capacity(header.n);
        let weights = header.weights(&blob);
        let mut dot = 0.0f64;
        let mut orig_sq = 0.0f64;
        let mut dec_sq = 0.0f64;
        let mut saturated = 0usize;
        let mut max_err_over_scale = 0.0f64;
        for row in 0..header.n {
            let sw = all_scales[row];
            let original = source.row(row);
            let row_bytes = &weights[row * header.row_bytes..(row + 1) * header.row_bytes];
            let mut row_dot = 0.0f64;
            let mut row_orig = 0.0f64;
            let mut row_dec = 0.0f64;
            let mut row_max_err = 0.0f64;
            let mut row_sat = 0usize;
            let mut row_max_abs = 0.0f32;
            for (k, &value) in original.iter().enumerate() {
                let byte = row_bytes[k / 2];
                let code = if k % 2 == 0 {
                    (byte & 0x0f) as i32 - 8
                } else {
                    (byte >> 4) as i32 - 8
                };
                let decoded = (code as f64) * (sw as f64);
                let a = value as f64;
                row_dot += a * decoded;
                row_orig += a * a;
                row_dec += decoded * decoded;
                let err = (a - decoded).abs() / (sw.max(f32::MIN_POSITIVE) as f64);
                if err > row_max_err {
                    row_max_err = err;
                }
                if code == -8 || code == 7 {
                    row_sat += 1;
                }
                let abs = value.abs();
                if abs > row_max_abs {
                    row_max_abs = abs;
                }
            }
            let cos = cosine(row_dot, row_orig, row_dec);
            dot += row_dot;
            orig_sq += row_orig;
            dec_sq += row_dec;
            saturated += row_sat;
            if row_max_err > max_err_over_scale {
                max_err_over_scale = row_max_err;
            }
            scale_f32.push(sw);
            scale_q16.push(all_q16[row]);
            max_abs.push(row_max_abs);
            quality_rows.push(GemvChannelStats {
                row,
                scale: sw,
                max_abs: row_max_abs,
                cos,
                max_abs_err_over_scale: row_max_err,
                saturated: row_sat,
                saturated_fraction: row_sat as f64 / header.k as f64,
            });
        }
        let samples = header.k * header.n;
        let cos_min_row = quality_rows
            .iter()
            .map(|row| row.cos)
            .fold(1.0f64, f64::min);
        let cos_mean_row = if quality_rows.is_empty() {
            1.0
        } else {
            quality_rows.iter().map(|row| row.cos).sum::<f64>() / quality_rows.len() as f64
        };
        Ok(Self {
            quality: GemvQuality {
                cos_total: cosine(dot, orig_sq, dec_sq),
                cos_min_row,
                cos_mean_row,
                max_abs_err_over_scale: max_err_over_scale,
                saturated,
                saturated_fraction: saturated as f64 / samples as f64,
                samples,
                per_row: quality_rows,
            },
            header,
            blob,
            scale_f32,
            scale_q16,
            max_abs,
        })
    }

    /// Real (unpadded) row plus its scale, decoded back to f32.
    pub fn row(&self, row: usize) -> Option<Vec<f32>> {
        if row >= self.header.n {
            return None;
        }
        let weigths = self.header.weights(&self.blob);
        let row_bytes = &weigths[row * self.header.row_bytes..(row + 1) * self.header.row_bytes];
        let sw = self.scale_f32[row];
        let mut out = Vec::with_capacity(self.header.k);
        for k in 0..self.header.k {
            let byte = row_bytes[k / 2];
            let code = if k % 2 == 0 {
                (byte & 0x0f) as i32 - 8
            } else {
                (byte >> 4) as i32 - 8
            };
            out.push(code as f32 * sw);
        }
        Some(out)
    }
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

/// Quantise one row exactly like `ffn_pack_i4_row()` in `ffn_ref.h`.
///
/// Returns `(scale, codes)`: `scale` is the f32 `max|w|/7` (or `1.0` for an
/// all-zero row) and `codes` are signed int4 values in `[-8, 7]`.
pub fn quantize_row(row: &[f32]) -> (f32, Vec<i8>) {
    let mut max_abs = 0.0f32;
    for &value in row {
        let abs = if value < 0.0 { -value } else { value };
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let codes = row
        .iter()
        .map(|&value| {
            // `lrintf` is round-half-to-even; `round()` is half-away-from-zero.
            let rounded = (value / scale).round_ties_even();
            rounded.clamp(-8.0, 7.0) as i8
        })
        .collect();
    (scale, codes)
}

/// Pack `matrix` (`k x n`) into the int4 GEMV blob.
pub fn build(matrix: &Matrix, _meta: &PackMeta) -> Result<GemvBlob> {
    let (k, n) = (matrix.k(), matrix.n());
    let header = GemvHeader::new(k, n)?;
    for row in 0..n {
        for (col, &value) in matrix.row(row).iter().enumerate() {
            if !value.is_finite() {
                return Err(PackError::NonFiniteWeight { row, col, value });
            }
        }
    }
    let mut blob = vec![0u8; header.total_bytes()];
    blob[..HEADER_BYTES].copy_from_slice(&header.encode());
    let row_bytes = header.row_bytes;
    let mut scales = vec![0.0f32; header.n_padded];
    let mut scales_q16 = vec![0i32; header.n_padded];
    for row in 0..n {
        let (scale, codes) = quantize_row(matrix.row(row));
        let base = header.weight_base + row * row_bytes;
        for (k_index, &code) in codes.iter().enumerate() {
            let nibble = ((code + 8) & 0x0f) as u8;
            let byte = &mut blob[base + k_index / 2];
            if k_index % 2 == 0 {
                *byte |= nibble;
            } else {
                *byte |= nibble << 4;
            }
        }
        scales[row] = scale;
        scales_q16[row] = (scale * SCALE_Q16_ONE as f32).round() as i32;
    }
    for (row, (&scale, &q16)) in scales.iter().zip(&scales_q16).enumerate() {
        let offset = header.scale_base + row * 4;
        blob[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
        let q_offset = header.scale_q16_base + row * 4;
        blob[q_offset..q_offset + 4].copy_from_slice(&q16.to_le_bytes());
    }
    GemvBlob::new(header, blob, matrix)
}

/// A manifest written next to a GEMV blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvManifest {
    pub format: String,
    pub format_version: u32,
    pub build_id: String,
    pub dtype: String,
    pub layout: String,
    pub encoding: String,
    pub tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub k_align: usize,
    pub n_align: usize,
    pub row_bytes: usize,
    pub header_bytes: usize,
    pub weight_base: usize,
    pub weight_bytes: usize,
    pub scale_base: usize,
    pub scale_bytes: usize,
    pub scale_q16_base: usize,
    pub scale_q16_bytes: usize,
    pub blob_bytes: usize,
    pub blob_fnv1a64: String,
    /// Exact per-row `max|w|/7`, `n_padded` values (zeros for padding rows).
    pub scales_f32: Vec<f32>,
    /// `round(scales_f32 * 65536)`, `n_padded` values.
    pub scales_q16: Vec<i32>,
    /// Largest absolute f32 weight per real row.
    pub max_abs: Vec<f32>,
    pub command: GemvCommand,
    pub quality: GemvQuality,
}

/// The `GEMV` command fields this blob is built for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvCommand {
    pub op: String,
    /// `set_shape(with_scale)`: 1 = per-row dequant tables allocated.
    pub with_scale: i32,
    /// `n_padded`, the `N` passed to `set_shape`.
    pub n: usize,
    /// `k_padded`, the `K` passed to `set_shape`.
    pub k: usize,
    /// Activation sequence length in int8 values (`K`).
    pub act_len: usize,
    /// Output length: `n` int8 values (sum of the per-row multiplier).
    pub dst_len: usize,
}

impl GemvManifest {
    pub fn from_blob(meta: &PackMeta, packed: &GemvBlob) -> Self {
        let header = &packed.header;
        Self {
            format: FORMAT_NAME.to_string(),
            format_version: u32::from(FORMAT_VERSION),
            build_id: format_build_id_hex(),
            dtype: "i4".to_string(),
            layout: "row_major_nk".to_string(),
            encoding: "offset8_lo_even_hi_odd".to_string(),
            tensor: meta.tensor.clone(),
            source: meta.source.clone(),
            k: header.k,
            n: header.n,
            k_padded: header.k_padded,
            n_padded: header.n_padded,
            k_align: K_ALIGN,
            n_align: N_ALIGN,
            row_bytes: header.row_bytes,
            header_bytes: header.header_bytes,
            weight_base: header.weight_base,
            weight_bytes: header.weight_bytes,
            scale_base: header.scale_base,
            scale_bytes: header.scale_bytes,
            scale_q16_base: header.scale_q16_base,
            scale_q16_bytes: header.scale_q16_bytes,
            blob_bytes: packed.blob.len(),
            blob_fnv1a64: format!("{:016x}", fnv1a64(&packed.blob)),
            scales_f32: {
                let mut values = packed.scale_f32.clone();
                values.resize(header.n_padded, 0.0);
                values
            },
            scales_q16: {
                let mut values = packed.scale_q16.clone();
                values.resize(header.n_padded, 0);
                values
            },
            max_abs: {
                let mut values = packed.max_abs.clone();
                values.resize(header.n_padded, 0.0);
                values
            },
            command: GemvCommand {
                op: "GEMV".to_string(),
                with_scale: 1,
                n: header.n_padded,
                k: header.k_padded,
                act_len: header.k_padded,
                dst_len: header.n_padded,
            },
            quality: packed.quality.clone(),
        }
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Check that this manifest really describes `blob`.
    pub fn verify_blob(&self, blob: &[u8]) -> Result<GemvHeader> {
        if self.format != FORMAT_NAME {
            return Err(PackError::ManifestMismatch(format!(
                "format `{}` != `{FORMAT_NAME}`",
                self.format
            )));
        }
        if self.format_version != u32::from(FORMAT_VERSION) {
            return Err(PackError::ManifestMismatch(format!(
                "format_version {} != {FORMAT_VERSION}",
                self.format_version
            )));
        }
        if self.build_id != format_build_id_hex() {
            return Err(PackError::ManifestMismatch(format!(
                "build_id `{}` != `{}`",
                self.build_id,
                format_build_id_hex()
            )));
        }
        let header = GemvHeader::parse(blob)?;
        for (name, manifest_value, header_value) in [
            ("k", self.k, header.k),
            ("n", self.n, header.n),
            ("k_padded", self.k_padded, header.k_padded),
            ("n_padded", self.n_padded, header.n_padded),
            ("row_bytes", self.row_bytes, header.row_bytes),
            ("header_bytes", self.header_bytes, header.header_bytes),
            ("weight_base", self.weight_base, header.weight_base),
            ("weight_bytes", self.weight_bytes, header.weight_bytes),
            ("scale_base", self.scale_base, header.scale_base),
            ("scale_bytes", self.scale_bytes, header.scale_bytes),
            ("scale_q16_base", self.scale_q16_base, header.scale_q16_base),
            (
                "scale_q16_bytes",
                self.scale_q16_bytes,
                header.scale_q16_bytes,
            ),
            ("blob_bytes", self.blob_bytes, blob.len()),
        ] {
            if manifest_value != header_value {
                return Err(PackError::ManifestMismatch(format!(
                    "{name}: manifest {manifest_value} != blob {header_value}"
                )));
            }
        }
        let checksum = format!("{:016x}", fnv1a64(blob));
        if self.blob_fnv1a64 != checksum {
            return Err(PackError::ManifestMismatch(format!(
                "blob checksum {} != {}",
                self.blob_fnv1a64, checksum
            )));
        }
        if self.scales_f32.len() != header.n_padded
            || self.scales_q16.len() != header.n_padded
            || self.max_abs.len() != header.n_padded
        {
            return Err(PackError::ManifestMismatch(
                "scale table length does not match n_padded".to_string(),
            ));
        }
        if header.scales(blob) != self.scales_f32 || header.scales_q16(blob) != self.scales_q16 {
            return Err(PackError::ManifestMismatch(
                "scale tables do not match the blob".to_string(),
            ));
        }
        Ok(header)
    }
}

/// Decode a blob back to `[n][k]` f32 (real rows only).
pub fn unpack(blob: &[u8]) -> Result<Vec<f32>> {
    let header = GemvHeader::parse(blob)?;
    let weights = header.weights(blob);
    let scales = header.scales(blob);
    let mut out = Vec::with_capacity(header.k * header.n);
    for row in 0..header.n {
        let sw = scales[row];
        let row_bytes = &weights[row * header.row_bytes..(row + 1) * header.row_bytes];
        for k in 0..header.k {
            let byte = row_bytes[k / 2];
            let code = if k % 2 == 0 {
                (byte & 0x0f) as i32 - 8
            } else {
                (byte >> 4) as i32 - 8
            };
            out.push(code as f32 * sw);
        }
    }
    Ok(out)
}

/// Resolve a GGUF tensor name from a layer index and a symbolic role.
pub fn tensor_for_role(layer: usize, role: &str) -> Result<String> {
    let name = match role {
        "in_proj" | "shortconv_in_proj" => format!("blk.{layer}.shortconv.in_proj.weight"),
        "out_proj" | "shortconv_out_proj" => format!("blk.{layer}.shortconv.out_proj.weight"),
        "attn_q" | "q" => format!("blk.{layer}.attn_q.weight"),
        "attn_k" | "k" => format!("blk.{layer}.attn_k.weight"),
        "attn_v" | "v" => format!("blk.{layer}.attn_v.weight"),
        "attn_o" | "o" | "attn_output" => format!("blk.{layer}.attn_output.weight"),
        "attn_gate" | "gate" => format!("blk.{layer}.attn_gate.weight"),
        other => {
            return Err(PackError::RawInput(format!(
                "unknown GEMV role `{other}` (in_proj, out_proj, attn_q, attn_k, attn_v, attn_o, attn_gate)"
            )))
        }
    };
    Ok(name)
}
