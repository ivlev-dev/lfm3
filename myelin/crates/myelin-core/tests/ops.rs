//! Reference-kernel tests: RoPE, RMSNorm, SiLU and attention against direct
//! naive implementations on small shapes.

use myelin_core::backend::BackendError;
use myelin_core::kv::KvCache;
use myelin_core::ops::{self, RopeTable};
use myelin_core::spec::{AttentionKind, RopeKind, RopeSpec};

/// Minimal in-memory KV cache used as a test double for the backend trait.
struct MemoryKv {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    dim: usize,
}

impl MemoryKv {
    fn new(dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            dim,
        }
    }
}

impl KvCache for MemoryKv {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn kv_dim(&self) -> usize {
        self.dim
    }

    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), BackendError> {
        assert_eq!(key.len(), self.dim);
        assert_eq!(value.len(), self.dim);
        self.keys.push(key.to_vec());
        self.values.push(value.to_vec());
        Ok(())
    }

    fn key(&self, index: usize) -> Option<&[f32]> {
        self.keys.get(index).map(Vec::as_slice)
    }

    fn value(&self, index: usize) -> Option<&[f32]> {
        self.values.get(index).map(Vec::as_slice)
    }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn naive_rope(vector: &[f32], position: usize, rope: &RopeSpec) -> Vec<f32> {
    let half = rope.dim / 2;
    let rotated_pairs = match rope.kind {
        RopeKind::Full => half,
        RopeKind::Partial { rotated_dim } => rotated_dim / 2,
    };
    let mut out = vector.to_vec();
    for index in 0..rotated_pairs {
        let angle = position as f64
            * (1.0 / f64::from(rope.theta).powf(2.0 * index as f64 / rope.dim as f64));
        let (sin, cos) = angle.sin_cos();
        let first = vector[index];
        let second = vector[index + half];
        out[index] = (f64::from(first) * cos - f64::from(second) * sin) as f32;
        out[index + half] = (f64::from(second) * cos + f64::from(first) * sin) as f32;
    }
    out
}

fn naive_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
    let scale = (1.0 / (mean + f64::from(eps)).sqrt()) as f32;
    x.iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

fn naive_attention(
    query: &[f32],
    kv: &MemoryKv,
    start: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = heads / kv_heads;
    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let mut out = vec![0.0f32; heads * head_dim];
    for head in 0..heads {
        let kv_head = head / group;
        let query = &query[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::new();
        for position in start..kv.len() {
            let key = &kv.keys[position][kv_head * head_dim..(kv_head + 1) * head_dim];
            let dot: f64 = query
                .iter()
                .zip(key)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum();
            scores.push(dot * scale);
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for (slot, position) in (start..kv.len()).enumerate() {
            let value = &kv.values[position][kv_head * head_dim..(kv_head + 1) * head_dim];
            for (dst, value) in out[head * head_dim..(head + 1) * head_dim]
                .iter_mut()
                .zip(value)
            {
                *dst += (exps[slot] / sum) as f32 * value;
            }
        }
    }
    out
}

#[test]
fn rope_matches_naive_full_rotation() {
    let rope = RopeSpec {
        kind: RopeKind::Full,
        theta: 10_000.0,
        dim: 8,
    };
    let table = RopeTable::new(&rope);
    let mut seed = 7u64;
    let original: Vec<f32> = (0..8).map(|_| lcg(&mut seed)).collect();
    let mut rotated = original.clone();
    table.apply(&mut rotated, 5);
    let expected = naive_rope(&original, 5, &rope);
    for (got, expected) in rotated.iter().zip(&expected) {
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }
    let energy: f32 = rotated.iter().map(|value| value * value).sum();
    let original_energy: f32 = original.iter().map(|value| value * value).sum();
    assert!((energy - original_energy).abs() < 1e-5);
}

#[test]
fn rope_partial_leaves_tail_untouched() {
    let rope = RopeSpec {
        kind: RopeKind::Partial { rotated_dim: 4 },
        theta: 1_000_000.0,
        dim: 8,
    };
    let table = RopeTable::new(&rope);
    let original: Vec<f32> = (0..8).map(|index| index as f32 + 0.25).collect();
    let mut rotated = original.clone();
    table.apply(&mut rotated, 3);
    // Pairs 0 and 1 rotate, pairs 2 and 3 stay exactly as they were.
    assert_ne!(rotated[0], original[0]);
    assert_ne!(rotated[4], original[4]);
    for index in 2..4 {
        assert_eq!(rotated[index], original[index]);
        assert_eq!(rotated[index + 4], original[index + 4]);
    }
    let expected = naive_rope(&original, 3, &rope);
    for (got, expected) in rotated.iter().zip(&expected) {
        assert!((got - expected).abs() < 1e-6);
    }
}

#[test]
fn rope_zero_position_is_identity() {
    let rope = RopeSpec {
        kind: RopeKind::Full,
        theta: 10_000.0,
        dim: 16,
    };
    let table = RopeTable::new(&rope);
    let original: Vec<f32> = (0..16).map(|index| index as f32).collect();
    let mut rotated = original.clone();
    table.apply(&mut rotated, 0);
    assert_eq!(rotated, original);
}

#[test]
fn rms_norm_matches_naive() {
    let mut seed = 11u64;
    let x: Vec<f32> = (0..12).map(|_| lcg(&mut seed)).collect();
    let weight: Vec<f32> = (0..12).map(|_| 0.5 + lcg(&mut seed).abs()).collect();
    let mut out = vec![0.0f32; x.len()];
    ops::rms_norm(&x, &weight, 1e-5, &mut out);
    let expected = naive_rms_norm(&x, &weight, 1e-5);
    for (got, expected) in out.iter().zip(&expected) {
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }
}

#[test]
fn silu_matches_definition() {
    for value in [-3.0f32, -1.0, 0.0, 0.5, 2.0, 8.0] {
        let expected = (value as f64) / (1.0 + (-(value as f64)).exp());
        assert!((f64::from(ops::silu(value)) - expected).abs() < 1e-6);
    }
}

#[test]
fn attention_matches_naive_with_gqa() {
    let heads = 4;
    let kv_heads = 2;
    let head_dim = 3;
    let positions = 6;
    let mut seed = 21u64;
    let mut kv = MemoryKv::new(kv_heads * head_dim);
    for _ in 0..positions {
        let key: Vec<f32> = (0..kv_heads * head_dim).map(|_| lcg(&mut seed)).collect();
        let value: Vec<f32> = (0..kv_heads * head_dim).map(|_| lcg(&mut seed)).collect();
        kv.append(&key, &value).expect("append");
    }
    let query: Vec<f32> = (0..heads * head_dim).map(|_| lcg(&mut seed)).collect();
    let mut out = vec![0.0f32; heads * head_dim];
    ops::attention(&query, &kv, 1, heads, kv_heads, head_dim, &mut out).expect("attention");
    let expected = naive_attention(&query, &kv, 1, heads, kv_heads, head_dim);
    for (got, expected) in out.iter().zip(&expected) {
        assert!(
            (got - expected).abs() < 1e-5,
            "got {got}, expected {expected}"
        );
    }
}

#[test]
fn sliding_window_ignores_tokens_before_start() {
    let heads = 2;
    let kv_heads = 1;
    let head_dim = 2;
    let positions = 8;
    let mut seed = 5u64;
    let mut kv = MemoryKv::new(kv_heads * head_dim);
    for _ in 0..positions {
        let key: Vec<f32> = (0..kv_heads * head_dim).map(|_| lcg(&mut seed)).collect();
        let value: Vec<f32> = (0..kv_heads * head_dim).map(|_| lcg(&mut seed)).collect();
        kv.append(&key, &value).expect("append");
    }
    let query: Vec<f32> = (0..heads * head_dim).map(|_| lcg(&mut seed)).collect();
    let start = 5;
    let mut windowed = vec![0.0f32; heads * head_dim];
    ops::attention(&query, &kv, start, heads, kv_heads, head_dim, &mut windowed)
        .expect("attention");

    // Poison everything outside the window: the result must not change.
    for position in 0..start {
        for value in kv.keys[position].iter_mut() {
            *value = 1e6;
        }
        for value in kv.values[position].iter_mut() {
            *value = -1e6;
        }
    }
    let mut after_poison = vec![0.0f32; heads * head_dim];
    ops::attention(
        &query,
        &kv,
        start,
        heads,
        kv_heads,
        head_dim,
        &mut after_poison,
    )
    .expect("attention");
    assert_eq!(windowed, after_poison);

    let expected = naive_attention(&query, &kv, start, heads, kv_heads, head_dim);
    for (got, expected) in after_poison.iter().zip(&expected) {
        assert!((got - expected).abs() < 1e-5);
    }
}

#[test]
fn window_start_respects_sliding_window() {
    assert_eq!(ops::window_start(AttentionKind::Full, 100, 8), 0);
    assert_eq!(ops::window_start(AttentionKind::Sliding, 100, 8), 93);
    assert_eq!(ops::window_start(AttentionKind::Sliding, 3, 8), 0);
    assert_eq!(ops::window_start(AttentionKind::Sliding, 0, 8), 0);
    assert_eq!(ops::window_start(AttentionKind::Sliding, 5, 0), 5);
}

#[test]
fn attention_rejects_empty_window() {
    let mut kv = MemoryKv::new(2);
    kv.append(&[1.0, 0.0], &[1.0, 0.0]).expect("append");
    let mut out = vec![0.0f32; 2];
    let error = ops::attention(&[1.0, 0.0], &kv, 1, 1, 1, 2, &mut out).expect_err("empty window");
    assert!(matches!(error, BackendError::Kv(_)));
}
