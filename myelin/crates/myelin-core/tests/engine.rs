use myelin_core::{Engine, ModelSpec};

#[test]
fn fake_tokens_are_deterministic() {
    let spec = ModelSpec::placeholder();
    let mut a = Engine::load(spec.clone()).expect("load");
    let mut b = Engine::load(spec).expect("load");

    for _ in 0..8 {
        assert_eq!(a.step(3).tokens, b.step(3).tokens);
    }
    assert_eq!(a.steps(), 8);
    assert_eq!(b.steps(), 8);
}

#[test]
fn step_returns_requested_row_count() {
    let mut engine = Engine::load(ModelSpec::placeholder()).expect("load");
    assert!(engine.step(0).tokens.is_empty());
    let out = engine.step(5);
    assert_eq!(out.tokens.len(), 5);
    assert!(!out.done);
    for token in out.tokens {
        assert!(usize::try_from(token).unwrap() < engine.spec().dims.vocab);
    }
}

#[test]
fn reset_restarts_the_sequence() {
    let mut engine = Engine::load(ModelSpec::placeholder()).expect("load");
    let first = engine.step(4).tokens;
    engine.step(2);
    engine.reset();
    let after_reset = engine.step(4).tokens;
    assert_eq!(first, after_reset);
    assert_eq!(engine.steps(), 1);
}

#[test]
fn rejects_invalid_spec() {
    let mut spec = ModelSpec::placeholder();
    spec.dims.vocab = 0;
    let error = Engine::load(spec).expect_err("must reject");
    assert!(matches!(error, myelin_core::CoreError::InvalidSpec(_)));
}
