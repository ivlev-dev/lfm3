//! Adapter from the existing `lfm-gguf` reader / `lfm-quant` dequantizers to
//! the packer's [`Matrix`] form.
//!
//! No second GGUF parser lives here: the header is read by `lfm-gguf`, the
//! quantized rows are expanded by `lfm-quant`.

use std::path::Path;

use lfm_gguf::{GgmlType, GgufFile};

use crate::manifest::SourceInfo;
use crate::{Matrix, PackError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub ggml_type: String,
}

/// List every tensor of a GGUF file.
pub fn list_tensors(path: &Path) -> Result<Vec<GgufTensor>> {
    let file = GgufFile::open(path).map_err(gguf_error)?;
    let mut tensors = Vec::with_capacity(file.tensors().len());
    for tensor in file.tensors() {
        let mut dims = Vec::with_capacity(tensor.dims.len());
        for dim in &tensor.dims {
            dims.push(usize::try_from(*dim).map_err(|_| {
                PackError::Gguf(format!(
                    "tensor `{}` dimension {dim} overflows usize",
                    tensor.name
                ))
            })?);
        }
        tensors.push(GgufTensor {
            name: tensor.name.clone(),
            dims,
            ggml_type: format!("{:?}", tensor.ggml_type),
        });
    }
    Ok(tensors)
}

/// Load a 2D GGUF tensor as f32 weights `n x k` (`dims = [k, n]`).
///
/// `max_rows` limits how many output channels are read, which keeps real-model
/// tests and spot checks cheap.
pub fn load_matrix(
    path: &Path,
    tensor: &str,
    max_rows: Option<usize>,
) -> Result<(Matrix, SourceInfo)> {
    let file = GgufFile::open(path).map_err(gguf_error)?;
    let info = file
        .tensor(tensor)
        .ok_or_else(|| PackError::TensorNotFound(tensor.to_string()))?;
    if info.dims.len() != 2 {
        return Err(PackError::TensorNotMatrix {
            name: tensor.to_string(),
            dims: info.dims.len(),
        });
    }
    let k = usize::try_from(info.dims[0])
        .map_err(|_| PackError::Gguf(format!("tensor `{tensor}` k overflows usize")))?;
    let n_total = usize::try_from(info.dims[1])
        .map_err(|_| PackError::Gguf(format!("tensor `{tensor}` n overflows usize")))?;
    if k == 0 || n_total == 0 {
        return Err(PackError::EmptyMatrix { k, n: n_total });
    }
    let n = match max_rows {
        Some(limit) => limit.min(n_total),
        None => n_total,
    };
    if n == 0 {
        return Err(PackError::EmptyMatrix { k, n });
    }

    let bytes = file.tensor_bytes(tensor).map_err(gguf_error)?;
    if bytes.len() % n_total != 0 {
        return Err(PackError::Gguf(format!(
            "tensor `{tensor}` byte size {} is not divisible by {n_total} rows",
            bytes.len()
        )));
    }
    let row_bytes = bytes.len() / n_total;

    let mut data = Vec::with_capacity(k * n);
    for row in 0..n {
        let slice = &bytes[row * row_bytes..(row + 1) * row_bytes];
        data.extend_from_slice(&dequant_row(info.ggml_type, slice, k, tensor)?);
    }

    let matrix = Matrix::new(k, n, data)?;
    let source = SourceInfo {
        file: path.display().to_string(),
        ggml_type: format!("{:?}", info.ggml_type),
    };
    Ok((matrix, source))
}

fn dequant_row(ggml_type: GgmlType, bytes: &[u8], k: usize, tensor: &str) -> Result<Vec<f32>> {
    let expected = |row_bytes: usize| -> Result<()> {
        if bytes.len() != row_bytes {
            return Err(PackError::Dequant(format!(
                "tensor `{tensor}`: row has {} bytes, expected {row_bytes} for k={k}",
                bytes.len()
            )));
        }
        Ok(())
    };
    match ggml_type {
        GgmlType::F32 => {
            expected(k * 4)?;
            let mut out = Vec::with_capacity(k);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        GgmlType::F16 => {
            expected(k * 2)?;
            Ok(lfm_quant::f16s_to_f32(bytes))
        }
        GgmlType::Q4_K => {
            expected(k.div_ceil(256) * 144)?;
            lfm_quant::dequant_row_q4_k(bytes, k)
                .map_err(|error| PackError::Dequant(format!("tensor `{tensor}` Q4_K: {error}")))
        }
        GgmlType::Q6_K => {
            expected(k.div_ceil(256) * 210)?;
            lfm_quant::dequant_row_q6_k(bytes, k)
                .map_err(|error| PackError::Dequant(format!("tensor `{tensor}` Q6_K: {error}")))
        }
        GgmlType::Q8_0 => {
            expected(k.div_ceil(32) * 34)?;
            let mut out = Vec::with_capacity(k);
            for block in bytes.chunks_exact(34) {
                let mut array = [0u8; 34];
                array.copy_from_slice(block);
                out.extend_from_slice(&lfm_quant::dequant_q8_0(&array));
            }
            Ok(out)
        }
        other => Err(PackError::UnsupportedGgmlType {
            name: tensor.to_string(),
            ggml_type: format!("{other:?}"),
        }),
    }
}

fn gguf_error(error: impl std::fmt::Display) -> PackError {
    PackError::Gguf(error.to_string())
}

/// Load the first `count` f32 values of a tensor, row-major after dequantisation.
///
/// Only the rows that cover `count` values are read, so a large FFN tensor can
/// supply a small KV page without materialising the whole matrix.
pub fn load_f32_prefix(path: &Path, tensor: &str, count: usize) -> Result<(Vec<f32>, SourceInfo)> {
    if count == 0 {
        return Err(PackError::EmptyMatrix { k: 0, n: 0 });
    }
    let file = GgufFile::open(path).map_err(gguf_error)?;
    let info = file
        .tensor(tensor)
        .ok_or_else(|| PackError::TensorNotFound(tensor.to_string()))?;
    if info.dims.len() != 2 {
        return Err(PackError::TensorNotMatrix {
            name: tensor.to_string(),
            dims: info.dims.len(),
        });
    }
    let k = usize::try_from(info.dims[0])
        .map_err(|_| PackError::Gguf(format!("tensor `{tensor}` k overflows usize")))?;
    if k == 0 {
        return Err(PackError::EmptyMatrix { k, n: 0 });
    }
    drop(file);
    let rows = count.div_ceil(k);
    let (matrix, source) = load_matrix(path, tensor, Some(rows))?;
    if matrix.data().len() < count {
        return Err(PackError::Gguf(format!(
            "tensor `{tensor}` has {} f32 values, need {count}",
            matrix.data().len()
        )));
    }
    let mut data = matrix.data().to_vec();
    data.truncate(count);
    Ok((data, source))
}
