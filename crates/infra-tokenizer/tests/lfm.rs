use lfm_tokenizer::LfmTokenizer;
use std::path::PathBuf;

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tokenizer.json")
}

fn load() -> Option<LfmTokenizer> {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return None;
    }
    Some(LfmTokenizer::from_file(&path).unwrap())
}

#[test]
fn vocab_size_is_128000() {
    let Some(tok) = load() else {
        return;
    };
    assert_eq!(tok.vocab_size(), 128000);
}

#[test]
fn special_tokens_roundtrip() {
    let path = tokenizer_path();
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let tok = LfmTokenizer::from_file(&path).unwrap();
    let json = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let added = value.get("added_tokens").unwrap().as_array().unwrap();

    let find_content = |id: u32| -> String {
        added
            .iter()
            .find(|t| t.get("id").and_then(|v| v.as_u64()) == Some(id as u64))
            .unwrap()
            .get("content")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    };

    let bos_str = find_content(124894);
    assert_eq!(tok.token_to_id(&bos_str), Some(tok.bos_id()));
    assert!(tok.id_to_token(tok.bos_id()).is_some());

    let eos_str = find_content(124900);
    assert_eq!(tok.token_to_id(&eos_str), Some(tok.eos_id()));

    let pad_str = find_content(124893);
    assert_eq!(tok.token_to_id(&pad_str), Some(tok.pad_id()));
}

#[test]
fn roundtrip_strings() {
    let Some(tok) = load() else {
        return;
    };
    for s in [
        "Hello, world!",
        "Привет, мир!",
        "def f(x):\n    return x*2",
        "<think>reasoning</think>",
    ] {
        let ids = tok.encode(s, false).unwrap();
        let dec = tok.decode(&ids, false).unwrap();
        assert_eq!(dec.trim(), s.trim(), "mismatch for {s}");
    }
}

#[test]
fn encode_hello_nonempty_and_deterministic() {
    let Some(tok) = load() else {
        return;
    };
    let ids = tok.encode("hello", false).unwrap();
    assert!(!ids.is_empty());
    assert!(ids.iter().all(|&id| id < 128000));
    let ids2 = tok.encode("hello", false).unwrap();
    assert_eq!(ids, ids2);
}

#[test]
fn decode_skip_special_removes_special() {
    let Some(tok) = load() else {
        return;
    };
    let mut ids = vec![tok.bos_id()];
    ids.extend(tok.encode("hi", false).unwrap());
    let dec = tok.decode(&ids, true).unwrap();
    assert!(!dec.contains("<|"));
}

#[test]
fn from_json_corrupted_returns_err() {
    assert!(LfmTokenizer::from_json("not json").is_err());
}