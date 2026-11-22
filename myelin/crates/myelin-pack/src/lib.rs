//! Weight packer for the DSP/HMX paths, with built-in quality control.
//!
//! Three wire formats live here, each transcribed from the code that consumes
//! it rather than from a specification:
//!
//! * [`crownstone`] — int8 HMX `weight.b` tiles + bias/scale blocks
//!   (GGUF `[K, N]` tensor -> 32x32 tiles);
//! * [`shortconv`] — the quantised LFM3 depthwise convolution blob read by the
//!   DSP `CONV` command (8 rotated int8 tap phases + per-channel `kq`);
//! * [`kvpage`] — K/KT/V images read by the DSP paged-attention kernel
//!   (int4 offset-8 K with per-token Q16 scales, int8 V), and [`kvarn`] — the
//!   reference K4/V2 and K4/V4 page codec, kept for comparison and not read by
//!   the DSP.
//!
//! Every packer unpacks its own output and reports cosine similarity,
//! `max |d|` relative to the quantisation scale and the saturation share.
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
pub mod kvarn;
pub mod kvpage;
pub mod manifest;
pub mod matrix;
pub mod metrics;
pub mod quant;
pub mod shortconv;

mod error;

use std::path::Path;

pub use crownstone::{pack, unpack, Decoded, PackResult};
pub use error::{PackError, Result};
pub use kvpage::{KvImage, KvLayout, KvManifest, KvPackParams, KvQuality, VScaleMode};
pub use manifest::{Manifest, PackMeta, ScaleTable, SourceInfo};
pub use matrix::Matrix;
pub use metrics::{ChannelStats, QualityStats};
pub use quant::{dequantize_row, quantize_row, ChannelQuant, DType, PackOptions};
pub use shortconv::{ActivationScales, ConvBlob, ConvManifest, ConvQuality};

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
