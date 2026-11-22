//! HMX int8 crownstone layout: 1 KiB weight tiles and 256 B bias/scale
//! blocks.
//!
//! Empirical layout (REPORT18 §3, verified bit-exactly on SD888 / HMX v68):
//!
//! ```text
//! weight tile 32(K) x 32(N), 1 KiB:
//!     addr(k, n) = (k >> 2) * 128 + 4 * n + (k & 3)
//! bias/scale block, 32 channels x 8 B = 256 B:
//!     low words  (32 x u32, offset +0):   [15:0] scale fp16, [31:16] out_bias
//!     high words (32 x u32, offset +128): [3:0] scale mantissa, [7:4] out_bias
//!                                         mantissa, [10:8] shape, [31:11] in_bias
//! ```
//!
//! The int8 convert applies an implicit `1/512` factor, so the stored fp16
//! word is `512 * s_w[n]` and the device multiplier equals `s_w[n]`.
//!
//! Blob layout (all integers little endian):
//!
//! ```text
//! [0 .. 128)        header
//! [128 .. ...)      weight tiles, ordered by n tile, then by k tile,
//!                   k tile stride 1024 bytes (matches `weight.b:deep`)
//! [... .. ...)      bias/scale blocks, one 256 B block per n tile
//! ```

use crate::manifest::{self, Manifest, PackMeta};
use crate::metrics::{self, QualityStats};
use crate::quant::{self, ChannelQuant, DType, PackOptions, QMAX};
use crate::{Matrix, PackError, Result};

/// File magic of the binary blob.
pub const BLOB_MAGIC: [u8; 4] = *b"MPK1";
/// Stable name written into the manifest.
pub const FORMAT_NAME: &str = "myelin-hmx-crownstone";
/// Blob format version understood by this crate.
pub const FORMAT_VERSION: u16 = 1;
/// Header size; also the (128 B aligned) weight base required by HMX.
pub const HEADER_BYTES: usize = 128;
pub const WEIGHT_BASE: usize = HEADER_BYTES;
pub const TILE_K: usize = 32;
pub const TILE_N: usize = 32;
pub const TILE_BYTES: usize = 1024;
pub const SCALE_BLOCK_BYTES: usize = 256;
pub const SCALE_BLOCK_CHANNELS: usize = 32;

const FLAG_K_PADDED: u8 = 1 << 0;
const FLAG_N_PADDED: u8 = 1 << 1;

/// Parsed blob header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobHeader {
    pub version: u16,
    pub dtype: DType,
    pub flags: u8,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub k_tiles: usize,
    pub n_tiles: usize,
    pub header_bytes: usize,
    pub weight_base: usize,
    pub weight_bytes: usize,
    pub bias_base: usize,
    pub bias_bytes: usize,
}

impl BlobHeader {
    pub fn new(k: usize, n: usize) -> Result<Self> {
        if k == 0 || n == 0 {
            return Err(PackError::EmptyMatrix { k, n });
        }
        if k > u32::MAX as usize || n > u32::MAX as usize {
            return Err(PackError::InconsistentBlob(format!(
                "shape {k}x{n} does not fit the 32-bit header fields"
            )));
        }
        let k_padded = round_up(k, TILE_K);
        let n_padded = round_up(n, TILE_N);
        let k_tiles = k_padded / TILE_K;
        let n_tiles = n_padded / TILE_N;
        let weight_bytes = k_tiles
            .checked_mul(n_tiles)
            .and_then(|tiles| tiles.checked_mul(TILE_BYTES))
            .ok_or_else(|| PackError::InconsistentBlob("weight size overflow".to_string()))?;
        let bias_bytes = n_tiles
            .checked_mul(SCALE_BLOCK_BYTES)
            .ok_or_else(|| PackError::InconsistentBlob("bias size overflow".to_string()))?;
        let weight_base = WEIGHT_BASE;
        let bias_base = weight_base
            .checked_add(weight_bytes)
            .ok_or_else(|| PackError::InconsistentBlob("blob size overflow".to_string()))?;
        let mut flags = 0u8;
        if k_padded != k {
            flags |= FLAG_K_PADDED;
        }
        if n_padded != n {
            flags |= FLAG_N_PADDED;
        }
        Ok(Self {
            version: FORMAT_VERSION,
            dtype: DType::I8,
            flags,
            k,
            n,
            k_padded,
            n_padded,
            k_tiles,
            n_tiles,
            header_bytes: HEADER_BYTES,
            weight_base,
            weight_bytes,
            bias_base,
            bias_bytes,
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.bias_base + self.bias_bytes
    }

    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..4].copy_from_slice(&BLOB_MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6] = self.dtype.tag();
        out[7] = self.flags;
        out[8..12].copy_from_slice(&(self.k as u32).to_le_bytes());
        out[12..16].copy_from_slice(&(self.n as u32).to_le_bytes());
        out[16..20].copy_from_slice(&(self.k_padded as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(self.n_padded as u32).to_le_bytes());
        out[24..28].copy_from_slice(&(self.k_tiles as u32).to_le_bytes());
        out[28..32].copy_from_slice(&(self.n_tiles as u32).to_le_bytes());
        out[32..36].copy_from_slice(&(self.header_bytes as u32).to_le_bytes());
        out[36..40].copy_from_slice(&(self.weight_base as u32).to_le_bytes());
        out[40..48].copy_from_slice(&(self.weight_bytes as u64).to_le_bytes());
        out[48..56].copy_from_slice(&(self.bias_base as u64).to_le_bytes());
        out[56..64].copy_from_slice(&(self.bias_bytes as u64).to_le_bytes());
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
        let dtype = DType::from_tag(blob[6])?;
        let flags = blob[7];
        let k = read_u32(blob, 8)? as usize;
        let n = read_u32(blob, 12)? as usize;
        let k_padded = read_u32(blob, 16)? as usize;
        let n_padded = read_u32(blob, 20)? as usize;
        let k_tiles = read_u32(blob, 24)? as usize;
        let n_tiles = read_u32(blob, 28)? as usize;
        let header_bytes = read_u32(blob, 32)? as usize;
        let weight_base = read_u32(blob, 36)? as usize;
        let weight_bytes_u64 = read_u64(blob, 40)?;
        let bias_base_u64 = read_u64(blob, 48)?;
        let bias_bytes_u64 = read_u64(blob, 56)?;

        let header = Self {
            version,
            dtype,
            flags,
            k,
            n,
            k_padded,
            n_padded,
            k_tiles,
            n_tiles,
            header_bytes,
            weight_base,
            weight_bytes: usize_from_u64(weight_bytes_u64, "weight_bytes")?,
            bias_base: usize_from_u64(bias_base_u64, "bias_base")?,
            bias_bytes: usize_from_u64(bias_bytes_u64, "bias_bytes")?,
        };
        header.validate(blob.len())?;
        Ok(header)
    }

    fn validate(&self, blob_len: usize) -> Result<()> {
        if self.k == 0 || self.n == 0 {
            return Err(PackError::InconsistentBlob(format!(
                "empty tensor {}x{}",
                self.k, self.n
            )));
        }
        if self.header_bytes != HEADER_BYTES {
            return Err(PackError::InconsistentBlob(format!(
                "header_bytes {} != {}",
                self.header_bytes, HEADER_BYTES
            )));
        }
        if self.weight_base != HEADER_BYTES {
            return Err(PackError::InconsistentBlob(format!(
                "weight_base {} != {} (weight tiles must be 128 B aligned)",
                self.weight_base, HEADER_BYTES
            )));
        }
        if self.k_padded != round_up(self.k, TILE_K) {
            return Err(PackError::InconsistentBlob(format!(
                "k_padded {} does not match k {} rounded to {}",
                self.k_padded, self.k, TILE_K
            )));
        }
        if self.n_padded != round_up(self.n, TILE_N) {
            return Err(PackError::InconsistentBlob(format!(
                "n_padded {} does not match n {} rounded to {}",
                self.n_padded, self.n, TILE_N
            )));
        }
        if self.k_tiles != self.k_padded / TILE_K || self.n_tiles != self.n_padded / TILE_N {
            return Err(PackError::InconsistentBlob(
                "tile counts do not match padded shape".to_string(),
            ));
        }
        let expected_weight = self
            .k_tiles
            .checked_mul(self.n_tiles)
            .and_then(|tiles| tiles.checked_mul(TILE_BYTES))
            .ok_or_else(|| PackError::InconsistentBlob("weight size overflow".to_string()))?;
        if self.weight_bytes != expected_weight {
            return Err(PackError::InconsistentBlob(format!(
                "weight_bytes {} != {}",
                self.weight_bytes, expected_weight
            )));
        }
        let expected_bias = self
            .n_tiles
            .checked_mul(SCALE_BLOCK_BYTES)
            .ok_or_else(|| PackError::InconsistentBlob("bias size overflow".to_string()))?;
        if self.bias_bytes != expected_bias {
            return Err(PackError::InconsistentBlob(format!(
                "bias_bytes {} != {}",
                self.bias_bytes, expected_bias
            )));
        }
        if self.bias_base != self.weight_base + self.weight_bytes {
            return Err(PackError::InconsistentBlob(
                "bias block does not follow the weight tiles".to_string(),
            ));
        }
        if blob_len < self.total_bytes() {
            return Err(PackError::BlobTooSmall {
                expected: self.total_bytes(),
                actual: blob_len,
            });
        }
        let mut expected_flags = 0u8;
        if self.k_padded != self.k {
            expected_flags |= FLAG_K_PADDED;
        }
        if self.n_padded != self.n {
            expected_flags |= FLAG_N_PADDED;
        }
        if self.flags & (FLAG_K_PADDED | FLAG_N_PADDED) != expected_flags {
            return Err(PackError::InconsistentBlob(
                "padding flags disagree with the header shape".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decoded content of a blob.
#[derive(Debug, Clone)]
pub struct Decoded {
    header: BlobHeader,
    scale_bits: Vec<u16>,
    scales: Vec<f32>,
    codes: Vec<i8>,
}

impl Decoded {
    pub fn header(&self) -> &BlobHeader {
        &self.header
    }

    pub fn k(&self) -> usize {
        self.header.k
    }

    pub fn n(&self) -> usize {
        self.header.n
    }

    pub fn k_padded(&self) -> usize {
        self.header.k_padded
    }

    pub fn n_padded(&self) -> usize {
        self.header.n_padded
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn scale_bits(&self) -> &[u16] {
        &self.scale_bits
    }

    /// `n_padded x k_padded` int8 codes, row-major by output channel.
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// Restore the real `n x k` weights (padding cropped).
    pub fn dequantized(&self) -> Result<Matrix> {
        let mut data = Vec::with_capacity(self.header.k * self.header.n);
        for channel in 0..self.header.n {
            let scale = self.scales[channel];
            let offset = channel * self.header.k_padded;
            for index in 0..self.header.k {
                data.push(self.codes[offset + index] as f32 * scale);
            }
        }
        Matrix::new(self.header.k, self.header.n, data)
    }

    /// Compare the stored weights with the original f32 tensor.
    pub fn quality(&self, original: &Matrix) -> Result<QualityStats> {
        if original.k() != self.header.k || original.n() != self.header.n {
            return Err(PackError::QualityShape(format!(
                "original matrix is {}x{}, blob is {}x{}",
                original.n(),
                original.k(),
                self.header.n,
                self.header.k
            )));
        }
        let decoded = self.dequantized()?;
        metrics::measure(
            original,
            &decoded,
            &self.scales,
            &self.codes,
            self.header.k_padded,
        )
    }

    /// Share of stored codes equal to +-127 (saturation of the int8 range).
    pub fn saturated(&self) -> usize {
        self.codes
            .iter()
            .filter(|value| value.unsigned_abs() == QMAX as u8)
            .count()
    }
}

/// Result of packing one tensor.
#[derive(Debug)]
pub struct PackResult {
    pub blob: Vec<u8>,
    pub manifest: Manifest,
    pub quality: QualityStats,
    pub quantized: Vec<ChannelQuant>,
}

/// Quantize and pack `matrix` into an int8 crownstone blob.
pub fn pack(matrix: &Matrix, meta: &PackMeta, options: PackOptions) -> Result<PackResult> {
    match options.dtype {
        DType::I8 => {}
    }
    let header = BlobHeader::new(matrix.k(), matrix.n())?;
    let mut quantized = Vec::with_capacity(matrix.n());
    for channel in 0..matrix.n() {
        quantized.push(quant::quantize_row_at(channel, matrix.row(channel))?);
    }

    let mut blob = vec![0u8; header.total_bytes()];
    blob[..HEADER_BYTES].copy_from_slice(&header.encode());

    for k_tile in 0..header.k_tiles {
        for n_tile in 0..header.n_tiles {
            let tile = k_tile * header.n_tiles + n_tile;
            let base = header.weight_base + tile * TILE_BYTES;
            for n_local in 0..TILE_N {
                let channel = n_tile * TILE_N + n_local;
                if channel >= header.n {
                    break;
                }
                let codes = &quantized[channel].q;
                for k_local in 0..TILE_K {
                    let k = k_tile * TILE_K + k_local;
                    if k >= header.k {
                        break;
                    }
                    let address = weight_address(k_local, n_local);
                    blob[base + address] = codes[k] as u8;
                }
            }
        }
    }

    for n_tile in 0..header.n_tiles {
        let block = header.bias_base + n_tile * SCALE_BLOCK_BYTES;
        for channel in 0..SCALE_BLOCK_CHANNELS {
            let global = n_tile * SCALE_BLOCK_CHANNELS + channel;
            let bits = if global < header.n {
                quantized[global].scale_bits
            } else {
                0
            };
            // Low word: scale fp16 in [15:0], out_bias fp16 in [31:16] (zero).
            let low = u32::from(bits);
            blob[block + channel * 4..block + channel * 4 + 4].copy_from_slice(&low.to_le_bytes());
            // High word: extra mantissa bits, shape and in_bias stay zero.
        }
    }

    let decoded = unpack(&blob)?;
    let quality = decoded.quality(matrix)?;

    let scale_bits: Vec<u16> = decoded.scale_bits().to_vec();
    let scales: Vec<f32> = decoded.scales().to_vec();
    let max_abs: Vec<f32> = quantized.iter().map(|channel| channel.max_abs).collect();
    let manifest = manifest::build(meta, &header, &scales, &scale_bits, &max_abs, &blob);

    Ok(PackResult {
        blob,
        manifest,
        quality,
        quantized,
    })
}

/// Read back a blob into int8 codes and per-channel scales.
pub fn unpack(blob: &[u8]) -> Result<Decoded> {
    let header = BlobHeader::parse(blob)?;
    let mut codes = vec![0i8; header.n_padded * header.k_padded];

    for k_tile in 0..header.k_tiles {
        for n_tile in 0..header.n_tiles {
            let tile = k_tile * header.n_tiles + n_tile;
            let base = header.weight_base + tile * TILE_BYTES;
            for n_local in 0..TILE_N {
                let channel = n_tile * TILE_N + n_local;
                if channel >= header.n_padded {
                    break;
                }
                for k_local in 0..TILE_K {
                    let k = k_tile * TILE_K + k_local;
                    if k >= header.k_padded {
                        break;
                    }
                    let address = weight_address(k_local, n_local);
                    codes[channel * header.k_padded + k] = blob[base + address] as i8;
                }
            }
        }
    }

    let mut scale_bits = Vec::with_capacity(header.n_padded);
    let mut scales = Vec::with_capacity(header.n_padded);
    for n_tile in 0..header.n_tiles {
        let block = header.bias_base + n_tile * SCALE_BLOCK_BYTES;
        for channel in 0..SCALE_BLOCK_CHANNELS {
            let word = read_u32(blob, block + channel * 4)?;
            let bits = (word & 0xFFFF) as u16;
            scale_bits.push(bits);
            scales.push(quant::decode_scale(bits));
        }
    }

    Ok(Decoded {
        header,
        scale_bits,
        scales,
        codes,
    })
}

/// Byte address of `(k, n)` inside a 32 x 32 int8 weight tile.
pub const fn weight_address(k: usize, n: usize) -> usize {
    (k >> 2) * 128 + 4 * n + (k & 3)
}

/// Small FNV-1a checksum used to tie the manifest to the blob.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn read_u32(blob: &[u8], offset: usize) -> Result<u32> {
    let bytes = blob
        .get(offset..offset + 4)
        .ok_or(PackError::BlobTooSmall {
            expected: offset + 4,
            actual: blob.len(),
        })?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(blob: &[u8], offset: usize) -> Result<u64> {
    let bytes = blob
        .get(offset..offset + 8)
        .ok_or(PackError::BlobTooSmall {
            expected: offset + 8,
            actual: blob.len(),
        })?;
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    Ok(u64::from_le_bytes(array))
}

fn usize_from_u64(value: u64, field: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        PackError::InconsistentBlob(format!("{field} = {value} does not fit in usize"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_address_is_a_bijection() {
        let mut seen = [false; TILE_BYTES];
        for k in 0..TILE_K {
            for n in 0..TILE_N {
                let address = weight_address(k, n);
                assert!(address < TILE_BYTES);
                assert!(!seen[address], "address {address} used twice");
                seen[address] = true;
            }
        }
        assert!(seen.iter().all(|used| *used));
    }

    #[test]
    fn weight_address_matches_report18_examples() {
        assert_eq!(weight_address(0, 0), 0);
        assert_eq!(weight_address(1, 0), 1);
        assert_eq!(weight_address(4, 0), 128);
        assert_eq!(weight_address(31, 31), 7 * 128 + 4 * 31 + 3);
        assert_eq!(weight_address(31, 31), 1023);
    }

    #[test]
    fn header_round_trip() {
        let header = BlobHeader::new(100, 45).unwrap();
        assert_eq!(header.k_padded, 128);
        assert_eq!(header.n_padded, 64);
        assert_eq!(header.k_tiles, 4);
        assert_eq!(header.n_tiles, 2);
        assert_eq!(header.flags, FLAG_K_PADDED | FLAG_N_PADDED);
        assert_eq!(header.weight_bytes, 8 * TILE_BYTES);
        assert_eq!(header.bias_bytes, 2 * SCALE_BLOCK_BYTES);

        let header_bytes = header.encode();
        let mut blob = vec![0u8; header.total_bytes()];
        blob[..HEADER_BYTES].copy_from_slice(&header_bytes);
        let parsed = BlobHeader::parse(&blob).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let header = BlobHeader::new(32, 32).unwrap();
        let mut blob = header.encode();
        blob[0] = b'X';
        assert!(matches!(
            BlobHeader::parse(&blob),
            Err(PackError::BadMagic { .. })
        ));
    }

    #[test]
    fn header_rejects_truncated_blob() {
        let header = BlobHeader::new(32, 32).unwrap();
        let blob = header.encode();
        assert!(matches!(
            BlobHeader::parse(&blob),
            Err(PackError::BlobTooSmall { .. })
        ));
    }

    #[test]
    fn fnv_checksum_is_stable() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
