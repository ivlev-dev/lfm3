use anyhow::{anyhow, Result};
use lfm_domain::{Message, Role, SamplingParams, ToolCall, ToolDef};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::EngineMetricsSnapshot;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<WireMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Stop>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(default)]
    pub tools: Option<Vec<WireTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct WireTool {
    #[serde(default)]
    pub function: WireFunction,
}

#[derive(Debug, Default, Deserialize)]
pub struct WireFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WireToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: WireFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct WireFunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl ChatRequest {
    pub fn to_messages(&self) -> Result<Vec<Message>> {
        let mut out = Vec::with_capacity(self.messages.len());
        for wire in &self.messages {
            let role = match wire.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                other => return Err(anyhow!("unsupported role {other}")),
            };
            let mut msg = Message::text(role, wire_text(wire.content.as_ref())?);
            msg.tool_call_id = wire.tool_call_id.clone();
            msg.tool_calls = wire
                .tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: wire_arguments(&tc.function.arguments),
                })
                .collect();
            out.push(msg);
        }
        Ok(out)
    }

    pub fn to_tools(&self) -> Vec<ToolDef> {
        if matches!(&self.tool_choice, Some(Value::String(s)) if s == "none") {
            return Vec::new();
        }
        self.tools
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter(|t| !t.function.name.is_empty())
            .map(|t| ToolDef {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: t.function.parameters.clone().unwrap_or_else(
                    || json!({"type": "object", "properties": {}}),
                ),
            })
            .collect()
    }

    pub fn to_params(&self) -> SamplingParams {
        let mut params = SamplingParams {
            temperature: self.temperature.unwrap_or(0.7),
            top_k: self.top_k.unwrap_or(40),
            top_p: self.top_p.unwrap_or(0.95),
            ..SamplingParams::default()
        };
        if let Some(stop) = &self.stop {
            match stop {
                Stop::One(s) => params.stop = vec![s.clone()],
                Stop::Many(v) => params.stop = v.clone(),
            }
        }
        if let Some(seed) = self.seed {
            params.seed = seed;
        }
        params.max_tokens = self.max_tokens.unwrap_or(256).max(1);
        params
    }

    pub fn thinking(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|v| v.get("enable_thinking"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

fn wire_text(content: Option<&Value>) -> Result<String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            Ok(out)
        }
        Some(_) => Err(anyhow!("content must be a string or an array of parts")),
    }
}

fn wire_arguments(value: &Value) -> Value {
    match value {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        other => other.clone(),
    }
}

fn tool_arguments_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn openai_tool_call(tc: &ToolCall, index: usize) -> Value {
    let id = tc.id.clone().unwrap_or_else(|| format!("call_{index}"));
    json!({
        "id": id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": tool_arguments_text(&tc.arguments)
        }
    })
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: Prompt,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Stop>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub echo: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
}

impl CompletionRequest {
    pub fn to_prompts(&self) -> Result<Vec<String>> {
        if self.n.unwrap_or(1) > 1 {
            return Err(anyhow!("n > 1 is not supported"));
        }
        if self.echo.unwrap_or(false) {
            return Err(anyhow!("echo is not supported"));
        }
        let prompts = match &self.prompt {
            Prompt::Text(s) => vec![s.clone()],
            Prompt::Texts(v) => v.clone(),
            Prompt::Tokens(v) => {
                return Err(anyhow!(
                    "token array prompts are not supported ({} tokens)",
                    v.len()
                ))
            }
        };
        if prompts.is_empty() || prompts.iter().any(|p| p.is_empty()) {
            return Err(anyhow!("prompt must not be empty"));
        }
        Ok(prompts)
    }

    pub fn to_params(&self) -> SamplingParams {
        let mut params = SamplingParams {
            temperature: self.temperature.unwrap_or(0.7),
            top_k: self.top_k.unwrap_or(40),
            top_p: self.top_p.unwrap_or(0.95),
            ..SamplingParams::default()
        };
        if let Some(stop) = &self.stop {
            match stop {
                Stop::One(s) => params.stop = vec![s.clone()],
                Stop::Many(v) => params.stop = v.clone(),
            }
        }
        if let Some(seed) = self.seed {
            params.seed = seed;
        }
        params.max_tokens = self.max_tokens.unwrap_or(256).max(1);
        params
    }
}

pub struct CompletionMeta {
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl CompletionMeta {
    pub fn new(model: &str) -> Self {
        Self::with_prefix("chatcmpl-", model)
    }

    pub fn new_completion(model: &str) -> Self {
        Self::with_prefix("cmpl-", model)
    }

    fn with_prefix(prefix: &str, model: &str) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let id = format!("{prefix}{created}");
        Self {
            id,
            created,
            model: model.to_string(),
        }
    }

    pub fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
        })
        .to_string()
    }

    pub fn tool_chunk(&self, index: usize, tc: &ToolCall) -> String {
        let delta = json!({ "tool_calls": [{
            "index": index,
            "id": tc.id.clone().unwrap_or_else(|| format!("call_{index}")),
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tool_arguments_text(&tc.arguments)
            }
        }]});
        self.chunk(delta, None)
    }

    pub fn message_response(
        &self,
        text: &str,
        thinking: Option<&str>,
        tool_calls: &[ToolCall],
        finish_reason: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> String {
        let content = if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(text.to_string())
        };
        let mut message = json!({ "role": "assistant", "content": content });
        if let Some(thinking) = thinking.filter(|t| !t.is_empty()) {
            message["reasoning_content"] = Value::String(thinking.to_string());
        }
        if !tool_calls.is_empty() {
            let calls: Vec<Value> = tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| openai_tool_call(tc, i))
                .collect();
            message["tool_calls"] = Value::Array(calls);
        }
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })
        .to_string()
    }

    pub fn completion_chunk(&self, text: &str, finish: Option<&str>) -> String {
        json!({
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "text": text, "logprobs": Value::Null, "finish_reason": finish }]
        })
        .to_string()
    }

    pub fn completion_response(
        &self,
        choices: &[(String, String)],
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> String {
        let items: Vec<Value> = choices
            .iter()
            .enumerate()
            .map(|(i, (text, finish))| {
                json!({ "index": i, "text": text, "logprobs": Value::Null, "finish_reason": finish })
            })
            .collect();
        json!({
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": items,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })
        .to_string()
    }
}

pub fn models_json(model: &str, created: u64) -> String {
    json!({
        "object": "list",
        "data": [{ "id": model, "object": "model", "created": created, "owned_by": "local" }]
    })
    .to_string()
}

#[derive(Serialize)]
struct MetricsBody<'a> {
    model: &'a str,
    ctx: usize,
    active: u32,
    waiting: u32,
    steps: u64,
    avg_step_ms: f64,
    kv_used_pages: u64,
    kv_free_pages: u64,
    kv_cached_pages: u64,
    cache_hits: u64,
    reused_tokens: u64,
    uptime_s: f64,
}

pub fn metrics_json(model: &str, snapshot: &EngineMetricsSnapshot) -> String {
    let body = MetricsBody {
        model,
        ctx: snapshot.ctx,
        active: snapshot.active,
        waiting: snapshot.waiting,
        steps: snapshot.steps,
        avg_step_ms: snapshot.avg_step_ms,
        kv_used_pages: snapshot.kv_used_pages,
        kv_free_pages: snapshot.kv_free_pages,
        kv_cached_pages: snapshot.kv_cached_pages,
        cache_hits: snapshot.cache_hits,
        reused_tokens: snapshot.reused_tokens,
        uptime_s: snapshot.uptime_s,
    };
    serde_json::to_string(&body).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> ChatRequest {
        serde_json::from_str(body).expect("request")
    }

    #[test]
    fn parses_basic_request() {
        let req = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#);
        assert!(req.stream);
        let msgs = req.to_messages().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content.as_deref(), Some("hi"));
    }

    #[test]
    fn parses_stop_string_and_array() {
        let one = parse(r#"{"messages":[],"stop":"\n\n"}"#);
        assert_eq!(one.to_params().stop, vec!["\n\n".to_string()]);
        let many = parse(r#"{"messages":[],"stop":["a","b"]}"#);
        assert_eq!(many.to_params().stop, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parses_content_parts() {
        let req = parse(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        );
        let msgs = req.to_messages().unwrap();
        assert_eq!(msgs[0].content.as_deref(), Some("ab"));
    }

    #[test]
    fn params_defaults_and_override() {
        let req = parse(r#"{"messages":[],"max_tokens":7,"temperature":0.0}"#);
        let p = req.to_params();
        assert_eq!(p.max_tokens, 7);
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, 40);
    }

    #[test]
    fn thinking_from_kwargs() {
        let req = parse(r#"{"messages":[],"chat_template_kwargs":{"enable_thinking":true}}"#);
        assert!(req.thinking());
        let req = parse(r#"{"messages":[]}"#);
        assert!(!req.thinking());
    }

    #[test]
    fn rejecting_unknown_role() {
        let req = parse(r#"{"messages":[{"role":"wizard","content":"x"}]}"#);
        assert!(req.to_messages().is_err());
    }

    #[test]
    fn chunk_and_response_shapes() {
        let meta = CompletionMeta::new("lfm3");
        let chunk = meta.chunk(json!({"content":"hi"}), None);
        let v: Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["content"], "hi");
        assert!(v["choices"][0]["finish_reason"].is_null());

        let resp = meta.message_response("hello", None, &[], "stop", 3, 2);
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["usage"]["total_tokens"], 5);
    }

    #[test]
    fn models_list() {
        let v: Value = serde_json::from_str(&models_json("lfm3", 1)).unwrap();
        assert_eq!(v["data"][0]["id"], "lfm3");
    }

    #[test]
    fn metrics_json_matches_schema_with_zeros() {
        let body = metrics_json("lfm3", &EngineMetricsSnapshot::default());
        assert_eq!(
            body,
            r#"{"model":"lfm3","ctx":0,"active":0,"waiting":0,"steps":0,"avg_step_ms":0.0,"kv_used_pages":0,"kv_free_pages":0,"kv_cached_pages":0,"cache_hits":0,"reused_tokens":0,"uptime_s":0.0}"#
        );
        let v: Value = serde_json::from_str(&body).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        let expected = [
            "model",
            "ctx",
            "active",
            "waiting",
            "steps",
            "avg_step_ms",
            "kv_used_pages",
            "kv_free_pages",
            "kv_cached_pages",
            "cache_hits",
            "reused_tokens",
            "uptime_s",
        ];
        assert_eq!(keys.len(), expected.len());
        for key in expected {
            assert!(v.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(v["model"], "lfm3");
        assert!(v["ctx"].is_u64());
        assert!(v["active"].is_u64());
        assert!(v["waiting"].is_u64());
        assert!(v["steps"].is_u64());
        assert!(v["avg_step_ms"].is_f64());
        assert!(v["kv_used_pages"].is_u64());
        assert!(v["kv_free_pages"].is_u64());
        assert!(v["kv_cached_pages"].is_u64());
        assert!(v["cache_hits"].is_u64());
        assert!(v["reused_tokens"].is_u64());
        assert!(v["uptime_s"].is_f64());
        assert_eq!(v["ctx"], 0);
        assert_eq!(v["active"], 0);
        assert_eq!(v["waiting"], 0);
        assert_eq!(v["steps"], 0);
        assert_eq!(v["avg_step_ms"].as_f64(), Some(0.0));
        assert_eq!(v["uptime_s"].as_f64(), Some(0.0));
    }

    #[test]
    fn metrics_json_emits_float_precision() {
        let snapshot = EngineMetricsSnapshot {
            ctx: 4096,
            active: 2,
            waiting: 1,
            steps: 7,
            avg_step_ms: 12.5,
            kv_used_pages: 3,
            kv_free_pages: 4,
            kv_cached_pages: 5,
            cache_hits: 6,
            reused_tokens: 8,
            uptime_s: 9.25,
        };
        let body = metrics_json("m", &snapshot);
        assert_eq!(
            body,
            r#"{"model":"m","ctx":4096,"active":2,"waiting":1,"steps":7,"avg_step_ms":12.5,"kv_used_pages":3,"kv_free_pages":4,"kv_cached_pages":5,"cache_hits":6,"reused_tokens":8,"uptime_s":9.25}"#
        );
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["ctx"], 4096);
        assert_eq!(v["active"], 2);
        assert_eq!(v["waiting"], 1);
        assert_eq!(v["steps"], 7);
        assert_eq!(v["avg_step_ms"].as_f64(), Some(12.5));
        assert_eq!(v["kv_used_pages"], 3);
        assert_eq!(v["kv_free_pages"], 4);
        assert_eq!(v["kv_cached_pages"], 5);
        assert_eq!(v["cache_hits"], 6);
        assert_eq!(v["reused_tokens"], 8);
        assert_eq!(v["uptime_s"].as_f64(), Some(9.25));
    }

    #[test]
    fn parses_tools_and_tool_choice() {
        let req = parse(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"weather","description":"get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_choice":"auto"}"#,
        );
        let tools = req.to_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "weather");
        assert_eq!(tools[0].description.as_deref(), Some("get weather"));
        assert_eq!(tools[0].parameters["properties"]["city"]["type"], "string");

        let none = parse(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"weather"}}],"tool_choice":"none"}"#,
        );
        assert!(none.to_tools().is_empty());
    }

    #[test]
    fn parses_tool_call_history() {
        let req = parse(
            r#"{"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"sunny"}
            ]}"#,
        );
        let msgs = req.to_messages().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].tool_calls.len(), 1);
        assert_eq!(msgs[1].tool_calls[0].name, "weather");
        assert_eq!(msgs[1].tool_calls[0].arguments["city"], "Paris");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn message_response_carries_tool_calls() {
        let meta = CompletionMeta::new("lfm3");
        let calls = vec![ToolCall {
            id: Some("call_1".into()),
            name: "weather".into(),
            arguments: json!({"city": "Paris"}),
        }];
        let resp = meta.message_response("", None, &calls, "tool_calls", 5, 7);
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert!(v["choices"][0]["message"]["content"].is_null());
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[test]
    fn tool_delta_chunk_shape() {
        let meta = CompletionMeta::new("lfm3");
        let tc = ToolCall {
            id: None,
            name: "weather".into(),
            arguments: json!({"city": "Paris"}),
        };
        let chunk = meta.tool_chunk(2, &tc);
        let v: Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["index"], 2);
        assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["id"], "call_2");
        assert_eq!(
            v["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
    }

    #[test]
    fn parses_completion_prompts() {
        let one = parse_completion(r#"{"prompt":"Hello","max_tokens":4}"#);
        assert_eq!(one.to_prompts().unwrap(), vec!["Hello".to_string()]);
        assert_eq!(one.to_params().max_tokens, 4);

        let many = parse_completion(r#"{"prompt":["a","b"]}"#);
        assert_eq!(many.to_prompts().unwrap(), vec!["a".to_string(), "b".to_string()]);

        let tokens = parse_completion(r#"{"prompt":[1,2,3]}"#);
        assert!(tokens.to_prompts().is_err());

        let n = parse_completion(r#"{"prompt":"a","n":2}"#);
        assert!(n.to_prompts().is_err());

        let echo = parse_completion(r#"{"prompt":"a","echo":true}"#);
        assert!(echo.to_prompts().is_err());

        let empty = parse_completion(r#"{"prompt":""}"#);
        assert!(empty.to_prompts().is_err());
    }

    #[test]
    fn completion_response_and_stream_shapes() {
        let meta = CompletionMeta::new_completion("lfm3");
        let resp = meta.completion_response(&[("hello".to_string(), "stop".to_string())], 3, 2);
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 5);

        let chunk = meta.completion_chunk("hi", None);
        let v: Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["choices"][0]["text"], "hi");
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    fn parse_completion(body: &str) -> CompletionRequest {
        serde_json::from_str(body).expect("completion request")
    }
}
