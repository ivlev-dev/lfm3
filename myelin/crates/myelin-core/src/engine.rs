//! Deterministic fake engine.
//!
//! This stage has no DSP, no NPU transport and no tokenizer. `Engine` turns a
//! validated [`ModelSpec`] into a reproducible token stream so the whole path
//! (Kotlin -> FFI -> Rust) can be exercised end to end.

use crate::{CoreError, ModelSpec};

/// One decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOut {
    pub tokens: Vec<u32>,
    /// Always false for the stub; the caller decides when to stop.
    pub done: bool,
}

/// Stub engine producing deterministic fake tokens.
#[derive(Debug, Clone)]
pub struct Engine {
    spec: ModelSpec,
    seed: u64,
    state: u64,
    steps: u64,
}

impl Engine {
    /// Validate and adopt a specification.
    pub fn load(spec: ModelSpec) -> Result<Self, CoreError> {
        spec.validate()?;
        let seed = seed_of(&spec);
        Ok(Self {
            spec,
            seed,
            state: seed,
            steps: 0,
        })
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn reset(&mut self) {
        self.state = self.seed;
        self.steps = 0;
    }

    /// Produce `rows` fake tokens. Same input state always yields the same
    /// sequence; no allocation beyond the returned vector.
    pub fn step(&mut self, rows: usize) -> StepOut {
        let vocab = self.spec.dims.vocab.max(1) as u64;
        let mut tokens = Vec::with_capacity(rows);
        for _ in 0..rows {
            self.state = self.state.wrapping_add(GOLDEN_GAMMA);
            let value = splitmix64(self.state);
            tokens.push((value % vocab) as u32);
        }
        self.steps = self.steps.saturating_add(1);
        StepOut {
            tokens,
            done: false,
        }
    }
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn seed_of(spec: &ModelSpec) -> u64 {
    let d = &spec.dims;
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    let words = [
        spec.family.as_str().len() as u64,
        d.hidden as u64,
        d.ffn as u64,
        d.layers as u64,
        d.heads as u64,
        d.kv_heads as u64,
        d.head_dim as u64,
        d.vocab as u64,
        d.conv_cache as u64,
        d.sliding_window as u64,
        spec.layers.len() as u64,
    ];
    for word in words {
        hash ^= word;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}
