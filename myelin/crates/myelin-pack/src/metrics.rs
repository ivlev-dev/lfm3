//! Accuracy of the quantize -> pack -> unpack round trip.
//!
//! Reported per output channel: cosine similarity, the largest absolute error
//! measured in units of the channel scale, and the fraction of values that
//! saturate the int8 range. A well-conditioned int8 weight channel must stay
//! below half a scale step (`max_abs_err <= 0.5 * scale`) and reach
//! `cos >= 0.999`.

use serde::Serialize;

use crate::quant::QMAX;
use crate::{Matrix, PackError, Result};

#[derive(Debug, Clone, Serialize)]
pub struct ChannelStats {
    pub channel: usize,
    pub scale: f32,
    pub max_abs: f32,
    pub cos: f64,
    /// Root mean square of the f32 channel; `max_abs / rms` flags heavy tails.
    pub rms: f64,
    pub peak_over_rms: f64,
    pub max_abs_err: f64,
    /// `max_abs_err / scale`; `0.0` for an all-zero channel.
    pub max_abs_err_over_scale: f64,
    /// Number of stored values equal to +-127.
    pub saturated: usize,
    /// Saturation share of the channel (`saturated / k`).
    pub saturated_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityStats {
    /// Cosine similarity over the whole tensor.
    pub cos_total: f64,
    pub cos_min_channel: f64,
    pub cos_mean_channel: f64,
    pub max_abs_err: f64,
    /// Largest per-channel `max_abs_err / scale`.
    pub max_abs_err_over_scale: f64,
    pub saturated: usize,
    pub saturated_fraction: f64,
    pub samples: usize,
    pub per_channel: Vec<ChannelStats>,
}

/// Compare `original` (f32 weights, `n x k`) with `decoded` (round-tripped
/// int8 weights) using `scales` (per output channel) and `codes`
/// (`n_padded x k_padded` int8, used for the saturation count).
pub fn measure(
    original: &Matrix,
    decoded: &Matrix,
    scales: &[f32],
    codes: &[i8],
    k_padded: usize,
) -> Result<QualityStats> {
    let k = original.k();
    let n = original.n();
    if decoded.k() != k || decoded.n() != n {
        return Err(PackError::QualityShape(format!(
            "decoded matrix is {}x{}, original is {}x{}",
            decoded.n(),
            decoded.k(),
            n,
            k
        )));
    }
    if scales.len() < n {
        return Err(PackError::QualityShape(format!(
            "{} scales for {} channels",
            scales.len(),
            n
        )));
    }
    let n_padded = if n == 0 {
        0
    } else {
        codes.len().div_ceil(k_padded)
    };
    if n_padded < n {
        return Err(PackError::QualityShape(format!(
            "{} code rows for {} channels",
            n_padded, n
        )));
    }

    let mut per_channel = Vec::with_capacity(n);
    let mut global_dot = 0.0f64;
    let mut global_original_sq = 0.0f64;
    let mut global_decoded_sq = 0.0f64;
    let mut max_abs_err = 0.0f64;
    let mut max_abs_err_over_scale = 0.0f64;
    let mut saturated = 0usize;

    for channel in 0..n {
        let scale = scales[channel];
        let mut dot = 0.0f64;
        let mut original_sq = 0.0f64;
        let mut decoded_sq = 0.0f64;
        let mut channel_err = 0.0f64;
        let mut channel_saturated = 0usize;

        let original_row = original.row(channel);
        let decoded_row = decoded.row(channel);
        let code_row = &codes[channel * k_padded..channel * k_padded + k];

        for (index, (&a, &b)) in original_row.iter().zip(decoded_row.iter()).enumerate() {
            let a = a as f64;
            let b = b as f64;
            dot += a * b;
            original_sq += a * a;
            decoded_sq += b * b;
            let err = (a - b).abs();
            if err > channel_err {
                channel_err = err;
            }
            if code_row[index].unsigned_abs() == QMAX as u8 {
                channel_saturated += 1;
            }
        }

        let cos = cosine(dot, original_sq, decoded_sq);
        let err_over_scale = if scale > 0.0 {
            channel_err / scale as f64
        } else {
            0.0
        };
        if err_over_scale > max_abs_err_over_scale {
            max_abs_err_over_scale = err_over_scale;
        }
        if channel_err > max_abs_err {
            max_abs_err = channel_err;
        }
        saturated += channel_saturated;

        global_dot += dot;
        global_original_sq += original_sq;
        global_decoded_sq += decoded_sq;

        let max_abs = original_row
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        let rms = (original_sq / k as f64).sqrt();
        let peak_over_rms = if rms > 0.0 { max_abs as f64 / rms } else { 0.0 };

        per_channel.push(ChannelStats {
            channel,
            scale,
            max_abs,
            cos,
            rms,
            peak_over_rms,
            max_abs_err: channel_err,
            max_abs_err_over_scale: err_over_scale,
            saturated: channel_saturated,
            saturated_fraction: channel_saturated as f64 / k as f64,
        });
    }

    let samples = k * n;
    let cos_min_channel = per_channel
        .iter()
        .map(|channel| channel.cos)
        .fold(1.0f64, f64::min);
    let cos_mean_channel = if per_channel.is_empty() {
        1.0
    } else {
        per_channel.iter().map(|channel| channel.cos).sum::<f64>() / per_channel.len() as f64
    };

    Ok(QualityStats {
        cos_total: cosine(global_dot, global_original_sq, global_decoded_sq),
        cos_min_channel,
        cos_mean_channel,
        max_abs_err,
        max_abs_err_over_scale,
        saturated,
        saturated_fraction: if samples == 0 {
            0.0
        } else {
            saturated as f64 / samples as f64
        },
        samples,
        per_channel,
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

/// Enforce the acceptance thresholds. The whole-tensor cosine is checked by
/// default; a per-channel floor is optional because a channel whose energy is
/// concentrated in a few outliers can drop below the tensor average while the
/// tensor as a whole is still far above it.
pub fn check_threshold(
    stats: &QualityStats,
    min_cos: f64,
    min_channel_cos: Option<f64>,
) -> Result<()> {
    if stats.cos_total < min_cos {
        return Err(PackError::QualityBelowThreshold {
            metric: "cos_total".to_string(),
            actual: stats.cos_total,
            threshold: min_cos,
        });
    }
    if let Some(threshold) = min_channel_cos {
        if stats.cos_min_channel < threshold {
            return Err(PackError::QualityBelowThreshold {
                metric: "cos_min_channel".to_string(),
                actual: stats.cos_min_channel,
                threshold,
            });
        }
    }
    Ok(())
}
