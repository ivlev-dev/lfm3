//! GGUF -> int8 HMX crownstone weight packer with built-in quality control.
//!
//! The packer turns one GGUF tensor (`dims = [K, N]`, `N` output channels) into
//! the tiled int8 layout used by the HMX `weight.b` path plus the 256-byte
//! bias/scale blocks the int8 convert path requires:
//!
//! * symmetric int8 quantization with per-output-channel scales;
//! * 32 (K) x 32 (N) tiles, 1 KiB each, `addr(k, n) = (k >> 2) * 128 + 4 * n + (k & 3)`;
//! * zero padding to whole tiles with `K`, `N` (and their padded values) kept
//!   in the header and manifest;
//! * a JSON manifest with tile offsets, fp16 scale words and a checksum;
//! * an unpack + compare pass that reports cosine similarity, `max |d|`
//!   relative to the channel scale and the saturation share.
//!
//! The empirical layout comes from `myelin/dsp/hmx/research/REPORT18.md`
//! (int8 section) and `myelin/dsp/qnn-reverse/REPORT.md` (bias/scale block).
//!
//! ```no_run
//! use myelin_pack::{pack_gguf, DType, PackOptions};
//!
//! let (result, _weights) = pack_gguf(
//!     std::path::Path::new("models/LFM3-1.7B-Q4_K_M.gguf"),
//!     "blk.2.attn_k.weight",
//!     None,
//!     PackOptions { dtype: DType::I8 },
//! )?;
//! println!("cos = {:.6}", result.quality.cos_total);
//! # Ok::<(), myelin_pack::PackError>(())
//! ```

pub mod crownstone;
pub mod gguf_source;
pub mod manifest;
pub mod matrix;
pub mod metrics;
pub mod quant;

mod error;

use std::path::Path;

pub use crownstone::{pack, unpack, Decoded, PackResult};
pub use error::{PackError, Result};
pub use manifest::{Manifest, PackMeta, ScaleTable, SourceInfo};
pub use matrix::Matrix;
pub use metrics::{ChannelStats, QualityStats};
pub use quant::{dequantize_row, quantize_row, ChannelQuant, DType, PackOptions};

/// Load one tensor from a GGUF file and pack it.
///
/// `max_rows` reads only the first output channels of the tensor, which is
/// useful for quick checks on large matrices.
pub fn pack_gguf(
    path: &Path,
    tensor: &str,
    max_rows: Option<usize>,
    options: PackOptions,
) -> Result<(PackResult, Matrix)> {
    let (matrix, source) = gguf_source::load_matrix(path, tensor, max_rows)?;
    let meta = PackMeta {
        tensor: tensor.to_string(),
        source: Some(source),
    };
    let result = pack(&matrix, &meta, options)?;
    Ok((result, matrix))
}
