use lfm_chat::ChatTemplate;
use lfm_domain::{Message, ToolCall, ToolDef};
use std::path::Path;

fn template_path() -> String {
    format!(
        "{}/../../models/hf/chat_template.jinja",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn load() -> Option<ChatTemplate> {
    let p = template_path();
    if !Path::new(&p).exists() {
        return None;
    }
    Some(ChatTemplate::from_file(&p).expect("load template"))
}

#[test]
fn t1_system_user() {
    let Some(ct) = load() else {
        return;
    };
    let msgs = vec![Message::system("sys"), Message::user("hi")];
    let out = ct.render(&msgs, &[], true, true, false).unwrap();
    assert!(out.starts_with("<|im_start|>system\n"));
    assert!(out.contains("<|im_start|>user\n"));
    assert!(out.ends_with("<|im_start|>assistant\n<think>"));
}

#[test]
fn t2_enable_thinking_false() {
    let Some(ct) = load() else {
        return;
    };
    let msgs = vec![Message::system("sys"), Message::user("hi")];
    let out = ct.render(&msgs, &[], true, false, false).unwrap();
    assert!(out.ends_with("<|im_start|>assistant\n<think></think>"));
}

#[test]
fn t3_tools() {
    let Some(ct) = load() else {
        return;
    };
    let msgs = vec![Message::user("hi")];
    let tool = ToolDef {
        name: "search".into(),
        description: Some("search tool".into()),
        parameters: serde_json::json!({"type":"object"}),
    };
    let out = ct.render(&msgs, &[tool], true, true, false).unwrap();
    assert!(out.contains("<tools>\n"));
    assert!(out.contains("\"name\": \"search\""));
}

#[test]
fn t4_assistant_thinking_tool_calls() {
    let Some(ct) = load() else {
        return;
    };
    let msg = Message::assistant("answer")
        .with_thinking("abc")
        .with_tool_call(ToolCall {
            id: Some("call_1".into()),
            name: "search".into(),
            arguments: serde_json::json!({"q": "rust"}),
        });
    let msgs = vec![msg];
    let out = ct.render(&msgs, &[], true, true, false).unwrap();
    assert!(out.contains("<think>abc</think>"));
    assert!(out.contains("<tool_call>\n<function=search>"));
    assert!(out.contains("</tool_call>"));
}

#[test]
fn t5_tool_result() {
    let Some(ct) = load() else {
        return;
    };
    let msgs = vec![Message::tool_result("call_1", "ok")];
    let out = ct.render(&msgs, &[], true, true, false).unwrap();
    assert!(out.contains("<|im_start|>tool\n<tool_response>\nok\n</tool_response>"));
}

#[test]
fn t6_bad_template() {
    let res = ChatTemplate::from_str("bad", "{% if %}");
    assert!(res.is_err());
}
