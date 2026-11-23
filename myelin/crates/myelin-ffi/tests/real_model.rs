//! End-to-end FFI test on the real checkpoint.
//!
//! Ignored by default (needs `models/LFM3-1.7B-Q4_K_M.gguf` plus
//! `tokenizer.json` and `chat_template.jinja` in the same folder):
//!
//! ```text
//! cargo test -p myelin-ffi --test real_model -- --ignored --nocapture
//! ```

use std::sync::{Arc, Mutex};

use myelin::{ChatMessage, Engine, Sampling, TokenEvent, TokenSink};

#[derive(Clone, Default)]
struct Collect(Arc<Mutex<String>>);

impl TokenSink for Collect {
    fn on_token(&self, event: TokenEvent) {
        if let TokenEvent::Delta { text, .. } = event {
            self.0.lock().unwrap().push_str(&text);
        }
    }
}

fn model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}

#[test]
#[ignore = "reads models/LFM3-1.7B-Q4_K_M.gguf"]
fn real_model_answers_over_the_ffi() {
    let path = model_dir().join("LFM3-1.7B-Q4_K_M.gguf");
    let engine = Engine::new(Some(path.to_string_lossy().into_owned())).expect("engine");
    let session = engine.new_session();

    let sink = Collect::default();
    session
        .generate(
            vec![ChatMessage {
                role: "user".to_owned(),
                content: "Say hello in one word.".to_owned(),
            }],
            Sampling {
                temperature: 0.0,
                max_tokens: 6,
                ..Sampling::default()
            },
            Box::new(sink.clone()),
        )
        .expect("generate");
    let text = sink.0.lock().unwrap().clone();
    assert!(!text.trim().is_empty(), "empty answer");
    println!("answer: {text}");
}
