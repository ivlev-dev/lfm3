use lfm_app::{ChatSession, SessionEvent};
use lfm_domain::{Message, Role};

fn collect_content(events: &[SessionEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::Content(c) => Some(c.as_str()),
            _ => None,
        })
        .collect()
}

fn collect_thinking(events: &[SessionEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::Thinking(c) => Some(c.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn plain_text_chunked() {
    let mut s = ChatSession::new();
    let text = "hello world, this is a plain answer";
    let mut events = Vec::new();
    for chunk in text.as_bytes().chunks(5) {
        events.extend(s.feed(std::str::from_utf8(chunk).unwrap()));
    }
    events.extend(s.finish());
    assert_eq!(collect_content(&events), text);
    assert!(events.contains(&SessionEvent::Finished));
    let msg = s.take_assistant().expect("assistant message");
    assert_eq!(msg.role, Role::Assistant);
    assert_eq!(msg.content.as_deref(), Some(text));
}

#[test]
fn thinking_block() {
    let mut s = ChatSession::new();
    let mut events = s.feed("abc<thinkreason</thinkdef");
    events.extend(s.finish());
    assert_eq!(collect_thinking(&events), "reason");
    assert_eq!(collect_content(&events), "abcdef");
    let first_think = events
        .iter()
        .position(|e| matches!(e, SessionEvent::Thinking(_)))
        .unwrap();
    let first_content = events
        .iter()
        .position(|e| matches!(e, SessionEvent::Content(_)))
        .unwrap();
    assert!(first_content < first_think);
    let msg = s.take_assistant().unwrap();
    assert_eq!(msg.thinking.as_deref(), Some("reason"));
    assert_eq!(msg.content.as_deref(), Some("abcdef"));
}

#[test]
fn split_tag() {
    let mut s = ChatSession::new();
    let mut events = s.feed("<thi");
    events.extend(s.feed("nk"));
    events.extend(s.feed("xyz</thi"));
    events.extend(s.feed("nk"));
    events.extend(s.feed("end"));
    events.extend(s.finish());
    assert_eq!(collect_thinking(&events), "xyz");
    assert_eq!(collect_content(&events), "end");
}

#[test]
fn tool_call_parsed() {
    let mut s = ChatSession::new();
    let raw = "<tool_call>\n<function=search>\n<parameter=q>\nrust\n</parameter>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>";
    let mut events = s.feed(raw);
    events.extend(s.finish());
    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].arguments["q"], "rust");
    assert_eq!(calls[0].arguments["n"], 5);
    assert!(collect_content(&events).is_empty());
}

#[test]
fn tool_call_split_chunks() {
    let raw = "<tool_call>\n<function=search>\n<parameter=q>\nrust</parameter>\n</function>\n</tool_call>";
    let mut s = ChatSession::new();
    let mut events = Vec::new();
    let step = 7;
    let mut i = 0;
    while i < raw.len() {
        let end = (i + step).min(raw.len());
        let mut e = end;
        while !raw.is_char_boundary(e) {
            e -= 1;
        }
        events.extend(s.feed(&raw[i..e]));
        i = e;
    }
    events.extend(s.finish());
    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].arguments["q"], "rust");
}

#[test]
fn two_tool_calls() {
    let mut s = ChatSession::new();
    let raw = "<tool_call>\n<function=a>\n</function>\n</tool_call><tool_call>\n<function=b>\n<parameter=x>\ntrue\n</parameter>\n</function>\n</tool_call>";
    let mut events = s.feed(raw);
    events.extend(s.finish());
    let calls: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            SessionEvent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "a");
    assert_eq!(calls[1].name, "b");
    assert_eq!(calls[1].arguments["x"], true);
}

#[test]
fn unclosed_tool_call_becomes_content() {
    let mut s = ChatSession::new();
    let mut events = s.feed("<tool_call>\n<function=search>\n<parameter=q>rus");
    events.extend(s.finish());
    let content = collect_content(&events);
    assert!(content.contains("<tool_call>"));
    assert!(content.contains("rus"));
    let calls = events
        .iter()
        .filter(|e| matches!(e, SessionEvent::ToolCall(_)))
        .count();
    assert_eq!(calls, 0);
}

#[test]
fn take_assistant_once() {
    let mut s = ChatSession::new();
    s.feed("hi");
    s.finish();
    assert!(s.take_assistant().is_some());
    assert!(s.take_assistant().is_none());
}

#[test]
fn messages_and_finish() {
    let mut s = ChatSession::new();
    s.push_message(Message::user("q"));
    assert_eq!(s.messages().len(), 1);
    s.feed("a");
    assert_eq!(s.messages().len(), 1);
    s.finish();
    assert_eq!(s.messages().len(), 2);
    assert_eq!(s.messages()[1].role, Role::Assistant);
}

#[test]
fn stray_close_tag_is_content() {
    let mut s = ChatSession::new();
    let mut events = s.feed("abc</think def");
    events.extend(s.finish());
    assert_eq!(collect_content(&events), "abc</think def");
    assert!(collect_thinking(&events).is_empty());
}

#[test]
fn reset_clears_stream_state() {
    let mut s = ChatSession::new();
    s.feed("abc");
    s.reset();
    let events = s.finish();
    assert_eq!(collect_content(&events), "");
}

#[test]
fn think_close_swallows_angle_bracket() {
    let mut s = ChatSession::new();
    s.begin_thinking();
    let mut events = s.feed("Let me think");
    events.extend(s.feed("</think"));
    events.extend(s.feed(">Answer"));
    events.extend(s.finish());
    assert_eq!(collect_thinking(&events), "Let me think");
    assert_eq!(collect_content(&events), "Answer");
}

#[test]
fn think_close_without_bracket_keeps_text() {
    let mut s = ChatSession::new();
    s.begin_thinking();
    let mut events = s.feed("hmm</think");
    events.extend(s.feed(" Answer"));
    events.extend(s.finish());
    assert_eq!(collect_thinking(&events), "hmm");
    assert_eq!(collect_content(&events), " Answer");
}

#[test]
fn begin_thinking_swallows_leading_bracket() {
    let mut s = ChatSession::new();
    s.begin_thinking();
    let events = s.feed(">hi");
    assert_eq!(collect_thinking(&events), "hi");
    let events = s.finish();
    assert!(collect_content(&events).is_empty());
}

#[test]
fn inline_think_open_swallows_bracket() {
    let mut s = ChatSession::new();
    let mut events = s.feed("a<think");
    events.extend(s.feed(">x"));
    events.extend(s.feed("</think>"));
    events.extend(s.finish());
    assert_eq!(collect_content(&events), "a");
    assert_eq!(collect_thinking(&events), "x");
}
