//! Sampler behaviour: greedy determinism, temperature, top-k and top-p.

use myelin_core::{Sampler, SamplingParams};

#[test]
fn greedy_picks_argmax() {
    let params = SamplingParams::default();
    let mut sampler = Sampler::new(params).expect("sampler");
    let mut logits = [0.1, 0.5, 0.2, -1.0];
    assert_eq!(sampler.sample(&mut logits).expect("sample"), 1);
    // Greedy does not consume the random stream: repeated calls agree.
    assert_eq!(sampler.sample(&mut logits).expect("sample"), 1);
}

#[test]
fn top_k_one_equals_greedy() {
    for seed in [0u64, 1, 42, 987_654_321] {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed,
        };
        let mut sampler = Sampler::new(params).expect("sampler");
        let mut logits = [0.3f32, 2.0, 1.5, -4.0, 2.5];
        assert_eq!(sampler.sample(&mut logits).expect("sample"), 4);
    }
}

#[test]
fn nucleus_keeps_only_the_confident_token() {
    // After temperature 1 the top token holds ~73% of the mass, so top_p = 0.5
    // truncates the distribution to that single token.
    for seed in 0..16u64 {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            top_k: 0,
            seed,
        };
        let mut sampler = Sampler::new(params).expect("sampler");
        let mut logits = [10.0f32, 9.0, 0.0];
        assert_eq!(sampler.sample(&mut logits).expect("sample"), 0);
    }
}

#[test]
fn sampling_is_reproducible_for_a_seed() {
    let draw = |seed: u64| {
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            seed,
        };
        let mut sampler = Sampler::new(params).expect("sampler");
        let mut tokens = Vec::new();
        for _ in 0..32 {
            let mut logits = vec![0.0f32; 128];
            for (index, logit) in logits.iter_mut().enumerate() {
                *logit = ((index * 37 % 128) as f32) * 0.05;
            }
            tokens.push(sampler.sample(&mut logits).expect("sample"));
        }
        tokens
    };
    assert_eq!(draw(1234), draw(1234));
    assert_ne!(draw(1234), draw(4321));
}

#[test]
fn invalid_parameters_are_rejected() {
    assert!(Sampler::new(SamplingParams {
        temperature: -1.0,
        ..SamplingParams::default()
    })
    .is_err());
    assert!(Sampler::new(SamplingParams {
        temperature: f32::NAN,
        ..SamplingParams::default()
    })
    .is_err());
    assert!(Sampler::new(SamplingParams {
        top_p: 0.0,
        ..SamplingParams::default()
    })
    .is_err());
    assert!(Sampler::new(SamplingParams {
        top_p: 1.5,
        ..SamplingParams::default()
    })
    .is_err());
}

#[test]
fn empty_logits_are_rejected() {
    let mut sampler = Sampler::new(SamplingParams::default()).expect("sampler");
    let mut logits: [f32; 0] = [];
    assert!(sampler.sample(&mut logits).is_err());
}
