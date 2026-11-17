use lfm_domain::SamplingParams;

#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: u64,
    recent: Vec<u32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = if params.seed == 0 {
            0x9E3779B97F4A7C15
        } else {
            params.seed
        };
        Self {
            params,
            rng: seed,
            recent: Vec::new(),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    pub fn reset(&mut self) {
        self.recent.clear();
    }

    pub fn push(&mut self, token: u32) {
        self.recent.push(token);
        let keep = self.params.repeat_penalty_window.max(1);
        if self.recent.len() > keep {
            let excess = self.recent.len() - keep;
            self.recent.drain(0..excess);
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        for v in logits.iter_mut() {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }
        if self.params.repeat_penalty != 1.0 {
            let penalty = self.params.repeat_penalty;
            for &t in &self.recent {
                let i = t as usize;
                if i < logits.len() {
                    let v = logits[i];
                    logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
                }
            }
        }
        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }
        let scale = 1.0 / self.params.temperature;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let k = self.params.top_k;
        let mut candidates: Vec<(u32, f32)> =
            Vec::with_capacity(if k > 0 { k.min(logits.len()) } else { logits.len() });
        if k > 0 && k < logits.len() {
            let mut min_pos = 0usize;
            let mut filled = 0usize;
            for (i, v) in logits.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let s = (*v - max) * scale;
                if filled < k {
                    candidates.push((i as u32, s));
                    filled += 1;
                    if filled < k {
                        continue;
                    }
                    min_pos = 0;
                    for (j, c) in candidates.iter().enumerate() {
                        if c.1 < candidates[min_pos].1 {
                            min_pos = j;
                        }
                    }
                } else if s > candidates[min_pos].1 {
                    candidates[min_pos] = (i as u32, s);
                    min_pos = 0;
                    for (j, c) in candidates.iter().enumerate() {
                        if c.1 < candidates[min_pos].1 {
                            min_pos = j;
                        }
                    }
                }
            }
        } else {
            for (i, v) in logits.iter().enumerate() {
                if v.is_finite() {
                    candidates.push((i as u32, (*v - max) * scale));
                }
            }
        }
        if candidates.is_empty() {
            return argmax(logits);
        }
        for c in candidates.iter_mut() {
            c.1 = c.1.exp();
        }
        if self.params.top_p < 1.0 || self.params.top_k > 0 {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let total: f32 = candidates.iter().map(|(_, p)| p).sum();
            let mut acc = 0.0;
            let mut keep = 0;
            for (_, p) in &candidates {
                acc += p / total;
                keep += 1;
                if acc >= self.params.top_p {
                    break;
                }
            }
            candidates.truncate(keep.max(1));
        }
        if self.params.min_p > 0.0 {
            let max_p = candidates
                .iter()
                .map(|(_, p)| *p)
                .fold(0.0f32, f32::max);
            let cutoff = max_p * self.params.min_p;
            candidates.retain(|(_, p)| *p >= cutoff);
            if candidates.is_empty() {
                return argmax(logits);
            }
        }
        if self.params.top_k == 1 || candidates.len() == 1 {
            return candidates[0].0;
        }
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        if total <= 0.0 || !total.is_finite() {
            return argmax(logits);
        }
        let mut r = self.next_f32() * total;
        for (id, p) in &candidates {
            if r < *p {
                return *id;
            }
            r -= *p;
        }
        candidates.last().map(|(id, _)| *id).unwrap_or(0)
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn greedy_on_zero_temperature() {
        let mut p = params();
        p.temperature = 0.0;
        let mut s = Sampler::new(p);
        let mut logits = vec![0.1, 0.9, 0.5];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut p = params();
        p.temperature = 1.0;
        p.top_k = 1;
        let mut s = Sampler::new(p);
        let mut logits = vec![0.1, 0.9, 0.5];
        for _ in 0..16 {
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn top_k_matches_sorted_reference() {
        let mut p = params();
        p.temperature = 0.7;
        p.top_k = 5;
        p.seed = 7;
        let logits: Vec<f32> = (0..4096)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0) - ((i % 17) as f32) * 0.05)
            .collect();
        let mut sorted: Vec<usize> = (0..logits.len()).collect();
        sorted.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
        let allowed: std::collections::HashSet<usize> = sorted[..5].iter().copied().collect();
        for _ in 0..64 {
            let mut s = Sampler::new(p.clone());
            let mut row = logits.clone();
            let picked = s.sample(&mut row) as usize;
            assert!(allowed.contains(&picked), "sampled {picked} outside top-5");
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut p = params();
        p.temperature = 0.8;
        p.seed = 42;
        let mut a = Sampler::new(p.clone());
        let mut b = Sampler::new(p);
        let mut la = vec![1.0, 1.2, 1.4, 0.9];
        let mut lb = la.clone();
        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        for _ in 0..32 {
            out_a.push(a.sample(&mut la));
            out_b.push(b.sample(&mut lb));
        }
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn top_p_restricts_candidates() {
        let mut p = params();
        p.temperature = 1.0;
        p.top_p = 0.01;
        let mut s = Sampler::new(p);
        let mut logits = vec![0.0, 20.0, 0.0];
        for _ in 0..16 {
            assert_eq!(s.sample(&mut logits), 1);
        }
    }

    #[test]
    fn repetition_penalty_lowers_repeated_token() {
        let mut p = params();
        p.temperature = 0.0;
        p.repeat_penalty = 10.0;
        p.repeat_penalty_window = 8;
        let mut s = Sampler::new(p);
        s.push(1);
        let mut logits = vec![0.0, 5.0, 4.9];
        assert_eq!(s.sample(&mut logits), 2);
    }

    #[test]
    fn non_finite_logits_are_ignored() {
        let mut p = params();
        p.temperature = 0.0;
        let mut s = Sampler::new(p);
        let mut logits = vec![f32::NAN, 1.0, f32::INFINITY];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn empty_logits_return_zero() {
        let mut s = Sampler::new(params());
        let mut logits: Vec<f32> = Vec::new();
        assert_eq!(s.sample(&mut logits), 0);
    }
}
