//! Symmetric int8 quantization with per-output-channel scales.
//!
//! `s_w[n] = max_k |w[k][n]| / 127`; the reciprocal multiplier is stored in
//! the HMX bias block as an fp16 value with an implicit `1/512` factor, so the
//! packer writes `fp16(512 * s_w[n])` and the device multiplies the integer
//! accumulator by `s_w[n]`. Quantization uses the fp16-rounded scale (not the
//! f32 one), so the stored integers are exactly the ones the device will
//! multiply by the stored scale.

use half::f16;

use crate::{PackError, Result};

/// Symmetric int8 range used for weights (HMX `weight.b`).
pub const QMAX: i32 = 127;

/// Implicit factor of the int8 convert path (`scale = 0x6000` means 1.0).
pub const DEVICE_SCALE_FACTOR: f32 = 512.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    I8,
}

impl DType {
    pub const fn tag(self) -> u8 {
        match self {
            Self::I8 => 0,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::I8),
            other => Err(PackError::UnsupportedBlobDType(other)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PackOptions {
    pub dtype: DType,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self { dtype: DType::I8 }
    }
}

/// One quantized output channel.
#[derive(Debug, Clone)]
pub struct ChannelQuant {
    /// Multiplier the device applies (== fp16-rounded `max_abs / 127`).
    pub scale: f32,
    /// fp16 bits of `512 * scale`, as stored in the bias block.
    pub scale_bits: u16,
    /// Largest absolute f32 weight of the channel.
    pub max_abs: f32,
    /// Quantized weights, `k` values.
    pub q: Vec<i8>,
    /// Values whose rounded magnitude reached 127 (the per-channel maximum).
    pub saturated: usize,
    /// Values that had to be clamped beyond +-127 (should stay zero).
    pub clamped: usize,
}

/// Encode a device multiplier as `fp16(512 * scale)`.
pub fn encode_scale(scale: f32) -> Option<u16> {
    let encoded = f16::from_f32(scale * DEVICE_SCALE_FACTOR);
    if encoded.is_finite() {
        Some(encoded.to_bits())
    } else {
        None
    }
}

/// Decode the fp16 word from the bias block back into the device multiplier.
pub fn decode_scale(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32() / DEVICE_SCALE_FACTOR
}

/// Quantize one channel. `channel` is only used for error reporting.
pub fn quantize_row_at(channel: usize, w: &[f32]) -> Result<ChannelQuant> {
    if w.is_empty() {
        return Err(PackError::EmptyRow { row: channel });
    }
    let mut max_abs = 0.0f32;
    for (col, value) in w.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(PackError::NonFiniteWeight {
                row: channel,
                col,
                value,
            });
        }
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0.0 {
        return Ok(ChannelQuant {
            scale: 0.0,
            scale_bits: 0,
            max_abs,
            q: vec![0; w.len()],
            saturated: 0,
            clamped: 0,
        });
    }

    let raw = max_abs / QMAX as f32;
    let bits = encode_scale(raw).ok_or(PackError::ScaleNotRepresentable {
        channel,
        scale: raw,
    })?;
    let scale = decode_scale(bits);
    if scale == 0.0 {
        // fp16 underflow: the channel is below the smallest representable
        // multiplier, so all weights quantize to zero by construction.
        return Ok(ChannelQuant {
            scale: 0.0,
            scale_bits: 0,
            max_abs,
            q: vec![0; w.len()],
            saturated: 0,
            clamped: 0,
        });
    }

    let mut q = Vec::with_capacity(w.len());
    let mut saturated = 0usize;
    let mut clamped = 0usize;
    for value in w.iter().copied() {
        let scaled = value / scale;
        let rounded = scaled.round();
        if rounded.abs() > QMAX as f32 {
            clamped += 1;
        }
        let quantized = rounded.clamp(-(QMAX as f32), QMAX as f32) as i8;
        if quantized.unsigned_abs() == QMAX as u8 {
            saturated += 1;
        }
        q.push(quantized);
    }

    Ok(ChannelQuant {
        scale,
        scale_bits: bits,
        max_abs,
        q,
        saturated,
        clamped,
    })
}

/// Quantize one channel outside a matrix context.
pub fn quantize_row(w: &[f32]) -> Result<ChannelQuant> {
    quantize_row_at(0, w)
}

/// Restore f32 values from int8 codes and the device multiplier.
pub fn dequantize_row(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|value| *value as f32 * scale).collect()
}
