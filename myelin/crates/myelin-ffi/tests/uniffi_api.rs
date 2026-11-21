use std::sync::{Arc, Mutex};

use myelin::{ChatMessage, Engine, MyelinError, Sampling, TokenEvent, TokenSink, STUB_TOKEN_COUNT};

#[derive(Default)]
struct Sink {
    events: Arc<Mutex<Vec<TokenEvent>>>,
}

impl TokenSink for Sink {
    fn on_token(&self, event: TokenEvent) {
        self.events.lock().unwrap().push(event);
    }
}

fn user_message() -> ChatMessage {
    ChatMessage {
        role: "user".to_owned(),
        content: "hello".to_owned(),
    }
}

#[test]
fn uniffi_objects_stream_events() {
    let engine = Engine::new(None).expect("engine");
    let session = engine.new_session();
    let sink = Sink::default();
    let events = Arc::clone(&sink.events);

    session
        .generate(vec![user_message()], Sampling::default(), Box::new(sink))
        .expect("generate");

    let events = events.lock().unwrap();
    assert_eq!(events.len(), STUB_TOKEN_COUNT as usize + 1);
    let deltas = events
        .iter()
        .filter(|event| matches!(event, TokenEvent::Delta { .. }))
        .count();
    assert_eq!(deltas, STUB_TOKEN_COUNT as usize);
    assert!(events
        .iter()
        .any(|event| matches!(event, TokenEvent::Done { total } if *total == STUB_TOKEN_COUNT)));
}

#[test]
fn loading_a_missing_file_reports_io() {
    let engine = Engine::new(None).expect("engine");
    let error = engine
        .load("/definitely/missing/myelin.gguf".to_owned())
        .expect_err("must fail");
    assert_eq!(error, MyelinError::Io);
    assert_eq!(
        Engine::new(Some("/definitely/missing/myelin.gguf".to_owned())).map(|_| ()),
        Err(MyelinError::Io)
    );
}

#[test]
fn invalid_messages_are_rejected() {
    let engine = Engine::new(None).expect("engine");
    let session = engine.new_session();
    let error = session
        .generate(Vec::new(), Sampling::default(), Box::new(Sink::default()))
        .expect_err("must fail");
    assert_eq!(error, MyelinError::InvalidArgument);
}

#[test]
fn abi_version_is_exported() {
    assert_eq!(myelin::abi_version(), myelin::MYELIN_ABI_VERSION);
    assert!(myelin::abi_version() >= 1);
}
