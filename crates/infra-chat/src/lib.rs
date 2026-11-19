use std::path::Path;

use anyhow::Context;
use lfm_domain::{Message, ToolDef};
use minijinja::Environment;
use serde_json::Value as JsonValue;

pub struct ChatTemplate {
    pub env: Environment<'static>,
}

impl ChatTemplate {
    pub fn from_str(name: &str, src: &str) -> anyhow::Result<Self> {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);
        env.add_filter("endswith", endswith_filter);

        let mut src = src.to_owned();
        src = strip_generation_tags(&src);
        let src_static = Box::leak(src.into_boxed_str());
        let name_static = Box::leak(name.to_owned().into_boxed_str());

        env.add_template("chat", src_static)
            .context("failed to add chat template")?;

        if name_static != "chat" {
            let _ = env.add_template(name_static, src_static);
        }

        Ok(Self { env })
    }

    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let src = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read template file: {}", path.as_ref().display()))?;
        Self::from_str("chat", &src)
    }

    pub fn render(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        add_generation_prompt: bool,
        enable_thinking: bool,
        preserve_thinking: bool,
    ) -> anyhow::Result<String> {
        self.render_with_bos(
            "",
            messages,
            tools,
            add_generation_prompt,
            enable_thinking,
            preserve_thinking,
        )
    }

    pub fn render_with_bos(
        &self,
        bos_token: &str,
        messages: &[Message],
        tools: &[ToolDef],
        add_generation_prompt: bool,
        enable_thinking: bool,
        preserve_thinking: bool,
    ) -> anyhow::Result<String> {
        let messages_json: Vec<JsonValue> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    lfm_domain::Role::System => "system",
                    lfm_domain::Role::User => "user",
                    lfm_domain::Role::Assistant => "assistant",
                    lfm_domain::Role::Tool => "tool",
                };
                let tool_calls: Vec<JsonValue> = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        })
                    })
                    .collect();
                serde_json::json!({
                    "role": role,
                    "content": m.content,
                    "thinking": m.thinking,
                    "tool_calls": tool_calls,
                    "tool_call_id": m.tool_call_id
                })
            })
            .collect();

        let tools_json: Vec<JsonValue> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();

        let ctx = serde_json::json!({
            "messages": messages_json,
            "tools": tools_json,
            "bos_token": bos_token,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking
        });

        let value = json_to_value(&ctx);
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(anyhow::Error::from)?;
        let out = tmpl.render(value).map_err(anyhow::Error::from)?;
        Ok(out)
    }

    pub fn render_json(&self, ctx: JsonValue) -> anyhow::Result<String> {
        let value = json_to_value(&ctx);
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(anyhow::Error::from)?;
        let out = tmpl.render(value).map_err(anyhow::Error::from)?;
        Ok(out)
    }
}


fn json_to_value(value: &JsonValue) -> minijinja::Value {
    match value {
        JsonValue::Null => minijinja::Value::from(()),
        JsonValue::Bool(b) => minijinja::Value::from(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                minijinja::Value::from(i)
            } else if let Some(u) = n.as_u64() {
                minijinja::Value::from(u)
            } else if let Some(f) = n.as_f64() {
                minijinja::Value::from(f)
            } else {
                minijinja::Value::from(())
            }
        }
        JsonValue::String(s) => minijinja::Value::from(s.clone()),
        JsonValue::Array(arr) => {
            let vals: Vec<minijinja::Value> = arr.iter().map(json_to_value).collect();
            minijinja::Value::from(vals)
        }
        JsonValue::Object(obj) => {
            let pairs: Vec<(String, minijinja::Value)> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            minijinja::Value::from_pairs(pairs)
        }
    }
}


fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed == "{% generation %}"
            || trimmed == "{%- generation -%}"
            || trimmed == "{% generation -%}"
            || trimmed == "{%- generation %}"
            || trimmed == "{% endgeneration %}"
            || trimmed == "{%- endgeneration -%}"
            || trimmed == "{% endgeneration -%}"
            || trimmed == "{%- endgeneration %}"
        {
            continue;
        }
        out.push_str(line);
    }
    let out = out.replace("{% generation %}", "");
    let out = out.replace("{%- generation -%}", "");
    let out = out.replace("{% generation -%}", "");
    let out = out.replace("{%- generation %}", "");
    let out = out.replace("{% endgeneration %}", "");
    let out = out.replace("{%- endgeneration -%}", "");
    let out = out.replace("{% endgeneration -%}", "");
    let out = out.replace("{%- endgeneration %}", "");
    let out = out.replace(".get(\"additional_tools\")", "[\"additional_tools\"]");
    let out = out.replace(".get(\"arguments\")", "[\"arguments\"]");
    let out = out.replace(".get(\"content\")", "[\"content\"]");
    let out = out.replace(".get(\"text\")", "[\"text\"]");
    let out = out.replace(".get(\"tool_call_id\")", "[\"tool_call_id\"]");
    let out = out.replace(".get(\"type\")", "[\"type\"]");
    let out = out.replace(".items()", " | items");
    let out = out.replace(".endswith(", " | endswith(");
    let out = out.replace(".split(", " | split(");
    
    out.replace("content | split(\"</think>\")[-1]", "(content | split(\"</think>\"))[-1]")
}


fn endswith_filter(value: minijinja::Value, suffix: minijinja::Value) -> bool {
    let s = value.as_str().unwrap_or("");
    let suf = suffix.as_str().unwrap_or("");
    s.ends_with(suf)
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}
