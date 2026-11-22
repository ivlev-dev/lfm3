//! Scalar reference kernels shared by the CPU backend and its tests.
//!
//! Everything here is plain f32 math over slices. The CPU backend is the host
//! oracle: a DSP implementation must reproduce these results within its own
//! numeric tolerance, so the formulas are kept explicit and boring.

use crate::backend::BackendError;
use crate::kv::KvCache;
use crate::spec::{AttentionKind, RopeSpec};

/// Matrix-vector products below this work size stay on the calling thread.
/// The threshold is small enough that unit tests never spawn threads.
const PARALLEL_THRESHOLD: usize = 1 << 18;

/// Dot product. Four independent accumulators keep the loop vectorizable
/// without changing the summation order between runs.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        acc[0] += a[base] * b[base];
        acc[1] += a[base + 1] * b[base + 1];
        acc[2] += a[base + 2] * b[base + 2];
        acc[3] += a[base + 3] * b[base + 3];
    }
    let mut sum = acc.iter().sum::<f32>();
    for index in chunks * 4..a.len() {
        sum += a[index] * b[index];
    }
    sum
}

/// `out = weights @ x` where `weights` is row-major `rows x cols`.
///
/// Rows are independent, so the work is split across host threads for large
/// matrices. Every output element is computed by exactly one dot product, which
/// makes the result independent of the thread count.
pub fn matvec(weights: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    if rows == 0 || cols == 0 {
        out.fill(0.0);
        return;
    }
    let work = rows.saturating_mul(cols);
    let threads = host_threads().min(rows);
    if work < PARALLEL_THRESHOLD || threads <= 1 {
        for (row, dst) in out.iter_mut().enumerate() {
            *dst = dot(&weights[row * cols..(row + 1) * cols], x);
        }
        return;
    }
    let chunk_rows = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk, out_chunk) in out.chunks_mut(chunk_rows).enumerate() {
            let first_row = chunk * chunk_rows;
            let weights = &weights[first_row * cols..];
            let x = &*x;
            scope.spawn(move || {
                for (row, dst) in out_chunk.iter_mut().enumerate() {
                    let start = row * cols;
                    *dst = dot(&weights[start..start + cols], x);
                }
            });
        }
    });
}

fn host_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// RMSNorm with the same epsilon placement and f64 accumulation as the host
/// reference path.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let sum: f64 = x
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum();
    let mean = sum / x.len() as f64;
    let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;
    for ((dst, value), weight) in out.iter_mut().zip(x).zip(weight) {
        *dst = value * scale * weight;
    }
}

/// SiLU: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place softmax with max subtraction.
pub fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        values.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

/// Index of the largest value, first index wins on ties.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}

/// First token index a layer attends to at `position`.
///
/// Full attention reads the whole history; sliding attention reads at most
/// `sliding_window` tokens ending at `position` (inclusive).
pub fn window_start(kind: AttentionKind, position: usize, sliding_window: usize) -> usize {
    match kind {
        AttentionKind::Full => 0,
        AttentionKind::Sliding => {
            let window = sliding_window.max(1);
            (position + 1).saturating_sub(window)
        }
    }
}

/// Rotary table for one rope configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeTable {
    inv_freq: Vec<f32>,
}

impl RopeTable {
    /// Precompute inverse frequencies, matching the host reference layout:
    /// pair `i` rotates dimensions `i` and `i + dim/2`, and pairs beyond the
    /// rotated half stay untouched.
    pub fn new(rope: &RopeSpec) -> Self {
        let half = rope.dim / 2;
        let rotated_pairs = match rope.kind {
            crate::spec::RopeKind::Full => half,
            crate::spec::RopeKind::Partial { rotated_dim } => rotated_dim / 2,
        };
        let theta = f64::from(rope.theta);
        let inv_freq = (0..half)
            .map(|index| {
                if index < rotated_pairs {
                    (1.0 / theta.powf((2.0 * index as f64) / rope.dim as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect();
        Self { inv_freq }
    }

    /// Inverse frequencies, zero for pairs outside the rotated part.
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Rotate one head vector in place.
    pub fn apply(&self, vector: &mut [f32], position: usize) {
        let half = self.inv_freq.len();
        debug_assert!(vector.len() >= half * 2);
        for (index, inv_freq) in self.inv_freq.iter().copied().enumerate() {
            if inv_freq == 0.0 {
                continue;
            }
            let angle = position as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let first = vector[index];
            let second = vector[index + half];
            vector[index] = first * cos - second * sin;
            vector[index + half] = second * cos + first * sin;
        }
    }
}

/// Causal attention of one new token over a KV cache.
///
/// `query` holds `heads * head_dim` values; `out` gets the same shape. The
/// cache must already contain the new token at `position`. `start` is the first
/// token of the window (see [`window_start`]).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    query: &[f32],
    kv: &dyn KvCache,
    start: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<(), BackendError> {
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(BackendError::Shape {
            operation: "attention",
            expected: 1,
            got: 0,
        });
    }
    if heads % kv_heads != 0 {
        return Err(BackendError::Kv(format!(
            "{heads} query heads are not a multiple of {kv_heads} KV heads"
        )));
    }
    let total = kv.len();
    if total == 0 || start >= total {
        return Err(BackendError::Kv(format!(
            "attention window {start}..{total} is empty"
        )));
    }
    if query.len() != heads * head_dim || out.len() != heads * head_dim {
        return Err(BackendError::Shape {
            operation: "attention",
            expected: heads * head_dim,
            got: query.len(),
        });
    }
    let group = heads / kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; total - start];
    out.fill(0.0);
    for head in 0..heads {
        let kv_head = head / group;
        let query = &query[head * head_dim..(head + 1) * head_dim];
        scores.fill(0.0);
        for (slot, position) in (start..total).enumerate() {
            let key = kv
                .key(position)
                .ok_or_else(|| BackendError::Kv(format!("cache has no key at token {position}")))?;
            let key = &key[kv_head * head_dim..(kv_head + 1) * head_dim];
            scores[slot] = dot(query, key) * scale;
        }
        softmax(&mut scores);
        let out = &mut out[head * head_dim..(head + 1) * head_dim];
        for (slot, position) in (start..total).enumerate() {
            let value = kv.value(position).ok_or_else(|| {
                BackendError::Kv(format!("cache has no value at token {position}"))
            })?;
            let value = &value[kv_head * head_dim..(kv_head + 1) * head_dim];
            let weight = scores[slot];
            for (dst, value) in out.iter_mut().zip(value) {
                *dst += weight * value;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_manual_rows() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        matvec(&weights, 2, 3, &[1.0, 1.0, 1.0], &mut out);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut values = [1.0, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
