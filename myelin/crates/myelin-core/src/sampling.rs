//! Token sampling: greedy, temperature and nucleus (top-p).
//!
//! Greedy is the default and is fully deterministic. The stochastic path uses a
//! splitmix64 stream seeded by the caller, so a run is reproducible without
//! pulling a random-number crate into the core.

use crate::ops;
use crate::RuntimeError;

/// Sampling controls for one generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `0.0` means greedy decoding; any positive value enables sampling.
    pub temperature: f32,
    /// Nucleus probability mass, in `(0, 1]`. `1.0` disables the cut.
    pub top_p: f32,
    /// Keep only the `top_k` most likely tokens. `0` disables the cut.
    pub top_k: usize,
    /// Seed of the deterministic random stream.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding.
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Check that the parameters can be honoured.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RuntimeError::Sampling(format!(
                "temperature {} must be a finite non-negative number",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(RuntimeError::Sampling(format!(
                "top_p {} must be in (0, 1]",
                self.top_p
            )));
        }
        Ok(())
    }
}

/// Stateful sampler: owns the random stream for one generation.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    state: u64,
}

impl Sampler {
    /// Validate parameters and seed the stream.
    pub fn new(params: SamplingParams) -> Result<Self, RuntimeError> {
        params.validate()?;
        let state = params.seed ^ GOLDEN_GAMMA;
        Ok(Self { params, state })
    }

    /// Parameters in use.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Pick one token id from `logits`, truncating them in place.
    pub fn sample(&mut self, logits: &mut [f32]) -> Result<u32, RuntimeError> {
        if logits.is_empty() {
            return Err(RuntimeError::Sampling("logit vector is empty".to_owned()));
        }
        if self.params.temperature == 0.0 {
            return Ok(ops::argmax(logits) as u32);
        }
        let temperature = f64::from(self.params.temperature);
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|left, right| {
            logits[*right]
                .total_cmp(&logits[*left])
                .then_with(|| left.cmp(right))
        });
        let keep = if self.params.top_k == 0 {
            order.len()
        } else {
            self.params.top_k.min(order.len())
        };
        let max = f64::from(logits[order[0]]);
        let mut weights: Vec<f64> = order
            .iter()
            .take(keep)
            .map(|index| ((f64::from(logits[*index]) - max) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Ok(order[0] as u32);
        }
        for weight in weights.iter_mut() {
            *weight /= total;
        }
        let mut kept = weights.len();
        if self.params.top_p < 1.0 {
            let mut cumulative = 0.0f64;
            for (index, weight) in weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= f64::from(self.params.top_p) {
                    kept = index + 1;
                    break;
                }
            }
        }
        let kept_total: f64 = weights[..kept].iter().sum();
        if !(kept_total.is_finite() && kept_total > 0.0) {
            return Ok(order[0] as u32);
        }
        let mut target = f64::from(self.next_unit()) * kept_total;
        for (index, weight) in weights[..kept].iter().enumerate() {
            target -= weight;
            if target <= 0.0 {
                return Ok(order[index] as u32);
            }
        }
        Ok(order[kept - 1] as u32)
    }

    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let value = splitmix64(self.state);
        ((value >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
