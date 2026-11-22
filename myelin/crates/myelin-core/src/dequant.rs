//! GGUF tensor dequantisation shared by the host backends.
//!
//! Two consumers need the same value semantics: [`crate::CpuBackend`] loads
//! whole tensors, while [`crate::DspBackend`] keeps the token embedding in its
//! checkpoint storage and dequantises exactly one row per token. Both go
//! through this module so the numbers stay identical by construction.

use lfm_gguf::GgmlType;
use lfm_quant;

/// Storage type name for diagnostics.
pub fn type_name(kind: GgmlType) -> String {
    match kind {
        GgmlType::Unknown(raw) => format!("type({raw})"),
        other => format!("{other:?}"),
    }
}

/// Bytes of one row of `cols` values stored as `kind`.
pub fn row_bytes(kind: GgmlType, cols: usize) -> Result<usize, String> {
    let (block, size) = kind.type_size().ok_or_else(|| {
        format!(
            "row dequantisation does not support {} tensors",
            type_name(kind)
        )
    })?;
    let block = usize::try_from(block).map_err(|_| "block size overflows usize".to_owned())?;
    let size = usize::try_from(size).map_err(|_| "block bytes overflow usize".to_owned())?;
    if block == 0 || cols % block != 0 {
        return Err(format!(
            "{cols} values are not a multiple of the {} block of {}",
            block,
            type_name(kind)
        ));
    }
    Ok(cols / block * size)
}

/// Dequantise `count` values from a row stored as `kind`.
pub fn dequantize_row(
    kind: GgmlType,
    name: &str,
    bytes: &[u8],
    count: usize,
) -> Result<Vec<f32>, String> {
    let expected = row_bytes(kind, count)?;
    if bytes.len() != expected {
        return Err(format!(
            "{name}: {} dequantise needs {expected} bytes, got {}",
            type_name(kind),
            bytes.len()
        ));
    }
    let values = match kind {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>(),
        GgmlType::Q4_K => {
            lfm_quant::dequant_row_q4_k(bytes, count).map_err(|error| format!("{name}: {error}"))?
        }
        GgmlType::Q6_K => {
            lfm_quant::dequant_row_q6_k(bytes, count).map_err(|error| format!("{name}: {error}"))?
        }
        GgmlType::Q8_0 => {
            let mut out = Vec::with_capacity(count);
            for block in bytes.chunks_exact(34) {
                let mut values = [0u8; 34];
                values.copy_from_slice(block);
                out.extend_from_slice(&lfm_quant::dequant_q8_0(&values));
            }
            out
        }
        other => {
            return Err(format!(
                "{name}: dequantisation of {} is not implemented",
                type_name(other)
            ));
        }
    };
    if values.len() != count {
        return Err(format!(
            "{name}: dequantised {} values, expected {count}",
            values.len()
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_rows_round_trip() {
        let values = [1.5f32, -2.0, 0.25, 8.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let decoded = dequantize_row(GgmlType::F32, "test", &bytes, 4).expect("row");
        assert_eq!(decoded, values);
        assert_eq!(row_bytes(GgmlType::F32, 4), Ok(16));
    }

    #[test]
    fn q6_k_rows_need_the_block_size() {
        // Eight blocks of 210 bytes in one row.
        let cols = 8 * 256;
        assert_eq!(row_bytes(GgmlType::Q6_K, cols), Ok(8 * 210));
        assert!(row_bytes(GgmlType::Q6_K, 100).is_err());
    }
}
