//! Machine-readable description of a blob.
//!
//! The manifest is written next to the binary blob. It carries everything the
//! runtime (and a human) needs to use the blob without re-reading the GGUF:
//! shape, padding, tile offsets, per-channel scales and a checksum.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::crownstone::{self, BlobHeader, SCALE_BLOCK_BYTES, TILE_BYTES, TILE_K, TILE_N};
use crate::quant::DType;
use crate::{PackError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceInfo {
    pub file: String,
    pub ggml_type: String,
}

/// Caller-provided identity of the packed tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackMeta {
    pub tensor: String,
    pub source: Option<SourceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleTable {
    pub layout: String,
    pub device_multiplier: String,
    pub quantization: String,
    /// Device multiplier per output channel (`n_padded` values, zeros for
    /// padding channels).
    pub scale_f32: Vec<f32>,
    /// fp16 word stored in the bias block, `fp16(512 * scale_f32)`.
    pub scale_f16_bits: Vec<u16>,
    /// Largest absolute f32 weight per real output channel.
    pub max_abs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    pub dtype: String,
    pub tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub k: usize,
    pub n: usize,
    pub k_padded: usize,
    pub n_padded: usize,
    pub k_tiles: usize,
    pub n_tiles: usize,
    pub tile_k: usize,
    pub tile_n: usize,
    pub tile_bytes: usize,
    /// `"k_tile_major"`: inside one k chunk all n tiles are contiguous
    /// (n stride 1024, k stride `n_tiles * 1024`), matching `weight.b:deep`.
    pub tile_order: String,
    pub header_bytes: usize,
    pub weight_base: usize,
    pub weight_bytes: usize,
    pub bias_base: usize,
    pub bias_bytes: usize,
    pub scale_block_bytes: usize,
    pub scale_block_layout: String,
    pub blob_bytes: usize,
    /// FNV-1a 64 of the blob, hex without prefix.
    pub blob_fnv1a64: String,
    pub scales: ScaleTable,
    /// `[k_tile, n_tile, weight_offset, bias_offset]`, in blob order.
    pub tiles: Vec<[usize; 4]>,
}

pub(crate) fn build(
    meta: &PackMeta,
    header: &BlobHeader,
    scales: &[f32],
    scale_bits: &[u16],
    max_abs: &[f32],
    blob: &[u8],
) -> Manifest {
    let mut tiles = Vec::with_capacity(header.k_tiles * header.n_tiles);
    for k_tile in 0..header.k_tiles {
        for n_tile in 0..header.n_tiles {
            let index = k_tile * header.n_tiles + n_tile;
            tiles.push([
                k_tile,
                n_tile,
                header.weight_base + index * TILE_BYTES,
                header.bias_base + n_tile * SCALE_BLOCK_BYTES,
            ]);
        }
    }

    let mut padded_scales = scales.to_vec();
    padded_scales.resize(header.n_padded, 0.0);
    let mut padded_bits = scale_bits.to_vec();
    padded_bits.resize(header.n_padded, 0);
    let mut padded_max_abs = max_abs.to_vec();
    padded_max_abs.resize(header.n_padded, 0.0);

    Manifest {
        format: crownstone::FORMAT_NAME.to_string(),
        format_version: u32::from(crownstone::FORMAT_VERSION),
        dtype: header.dtype.name().to_string(),
        tensor: meta.tensor.clone(),
        source: meta.source.clone(),
        k: header.k,
        n: header.n,
        k_padded: header.k_padded,
        n_padded: header.n_padded,
        k_tiles: header.k_tiles,
        n_tiles: header.n_tiles,
        tile_k: TILE_K,
        tile_n: TILE_N,
        tile_bytes: TILE_BYTES,
        tile_order: "k_tile_major".to_string(),
        header_bytes: header.header_bytes,
        weight_base: header.weight_base,
        weight_bytes: header.weight_bytes,
        bias_base: header.bias_base,
        bias_bytes: header.bias_bytes,
        scale_block_bytes: SCALE_BLOCK_BYTES,
        scale_block_layout: "32 channels x u64: low u32 words then high u32 words".to_string(),
        blob_bytes: blob.len(),
        blob_fnv1a64: format!("{:016x}", crownstone::fnv1a64(blob)),
        scales: ScaleTable {
            layout: "per_output_channel".to_string(),
            device_multiplier: "fp16_bits / 512".to_string(),
            quantization: "symmetric int8, s = max_abs / 127, zero point 0".to_string(),
            scale_f32: padded_scales,
            scale_f16_bits: padded_bits,
            max_abs: padded_max_abs,
        },
        tiles,
    }
}

impl Manifest {
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Check that the manifest really describes `blob` (shape, size, checksum).
    pub fn verify_blob(&self, blob: &[u8]) -> Result<()> {
        let header = BlobHeader::parse(blob)?;
        if self.format != crownstone::FORMAT_NAME {
            return Err(PackError::ManifestMismatch(format!(
                "format `{}` != `{}`",
                self.format,
                crownstone::FORMAT_NAME
            )));
        }
        if self.format_version != u32::from(crownstone::FORMAT_VERSION) {
            return Err(PackError::ManifestMismatch(format!(
                "format_version {} != {}",
                self.format_version,
                crownstone::FORMAT_VERSION
            )));
        }
        let dtype = DType::from_tag(match self.dtype.as_str() {
            "i8" => 0,
            other => {
                return Err(PackError::ManifestMismatch(format!(
                    "unknown dtype `{other}`"
                )))
            }
        })?;
        if dtype != header.dtype {
            return Err(PackError::ManifestMismatch(format!(
                "dtype `{}` != `{}`",
                self.dtype,
                header.dtype.name()
            )));
        }
        for (name, manifest_value, header_value) in [
            ("k", self.k, header.k),
            ("n", self.n, header.n),
            ("k_padded", self.k_padded, header.k_padded),
            ("n_padded", self.n_padded, header.n_padded),
            ("k_tiles", self.k_tiles, header.k_tiles),
            ("n_tiles", self.n_tiles, header.n_tiles),
            ("tile_bytes", self.tile_bytes, TILE_BYTES),
            ("header_bytes", self.header_bytes, header.header_bytes),
            ("weight_base", self.weight_base, header.weight_base),
            ("weight_bytes", self.weight_bytes, header.weight_bytes),
            ("bias_base", self.bias_base, header.bias_base),
            ("bias_bytes", self.bias_bytes, header.bias_bytes),
            ("blob_bytes", self.blob_bytes, blob.len()),
        ] {
            if manifest_value != header_value {
                return Err(PackError::ManifestMismatch(format!(
                    "{name}: manifest {manifest_value} != blob {header_value}"
                )));
            }
        }
        let checksum = format!("{:016x}", crownstone::fnv1a64(blob));
        if self.blob_fnv1a64 != checksum {
            return Err(PackError::ManifestMismatch(format!(
                "blob checksum {} != {}",
                self.blob_fnv1a64, checksum
            )));
        }
        if self.scales.scale_f32.len() != header.n_padded
            || self.scales.scale_f16_bits.len() != header.n_padded
            || self.scales.max_abs.len() != header.n_padded
        {
            return Err(PackError::ManifestMismatch(
                "scale table length does not match n_padded".to_string(),
            ));
        }
        let expected_tiles = header.k_tiles * header.n_tiles;
        if self.tiles.len() != expected_tiles {
            return Err(PackError::ManifestMismatch(format!(
                "{} tiles described, blob has {expected_tiles}",
                self.tiles.len()
            )));
        }
        let decoded = crownstone::unpack(blob)?;
        if decoded.scale_bits() != self.scales.scale_f16_bits.as_slice() {
            return Err(PackError::ManifestMismatch(
                "scale table does not match the blob bias blocks".to_string(),
            ));
        }
        Ok(())
    }
}
