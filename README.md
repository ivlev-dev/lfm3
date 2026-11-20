# lfm

Локальный инференс **LFM3-1.7B** на Rust + OpenCL. Один бинарь `lfm`
(консольный чат и OpenAI-совместимый сервер) и библиотека `lfm` для встраивания.

## Возможности

- **OpenCL-бэкенд** (Rusticl): отдельные пути ядер под AMD (подгруппа 64) и
  Intel iGPU (16, без fp16), KV `f16` / `q4` / `kvarn` / `kvarn_k4v4`.
- **Multi-device**: `single`, `pipeline` (слои по устройствам, конвейер по
  чанкам) и `tp` (тензорный параллелизм, взвешенный шардинг).
- **Сервинг**: continuous batching, paged KV, radix prefix cache, пул KV
  (`kv_pool`), адресные логиты, GPU-argmax для greedy.
- **Чат**: streaming, reasoning (`thinking`), tool calling, TTFT и счётчики
  prefill/decode tok/s.
- **Одна конфигурация**: TOML + CLI, без пользовательских env-переменных.

## Сборка

```bash
cargo build --release
cargo test --workspace
```

Модель и файлы HF (по умолчанию ищутся в `models/`):

```bash
mkdir -p models/hf
curl -L -o models/LFM3-1.7B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf
for f in tokenizer.json chat_template.jinja config.json; do
  curl -L -o models/hf/$f \
    https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/$f
done
```

## Быстрый старт

```bash
# консольный чат (TUI) на одном GPU
./target/release/lfm chat

# один запрос без TUI (удобно для скриптов и CI)
./target/release/lfm chat --once "Say hi." --max-tokens 32

# OpenAI-совместимый сервер
./target/release/lfm serve
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"lfm3","messages":[{"role":"user","content":"Say hi."}],"max_tokens":32
}'
```

## Конфигурация

```bash
./target/release/lfm config init       # создать lfm.toml с примером (GPU+iGPU)
./target/release/lfm config print      # эффективный конфиг в TOML
./target/release/lfm --config my.toml chat
```

Приоритет: **CLI > `--config`/`lfm.toml` > значения по умолчанию**.
Полное описание ключей — `docs/config.md`.

Пример: GPU + iGPU, сжатый KV, контекст 32k:

```toml
[backend]
devices = [0, 1]
mode = "pipeline"
tiers = [96, 4]
kv = "kvarn"
ctx = 32000
```

```bash
RUSTICL_ENABLE=iris,radeonsi ./target/release/lfm chat
```

## Устройства и KV

| режим | как включить | когда |
|---|---|---|
| single | `backend.devices = [0]` | одна карта (быстрее всего) |
| pipeline | `mode = "pipeline"`, `devices = [0,1]`, `tiers = [96,4]` | не хватает VRAM, есть iGPU/CPU |
| tp | `mode = "tp"`, `devices = [0,1]` | одинаковые быстрые GPU; только `kv = "f16"` |

Индексы устройств зависят от `RUSTICL_ENABLE`: без переменных `0` — дискретный
GPU, `1` — CPU; с `RUSTICL_ENABLE=iris,radeonsi` — `0` GPU, `1` iGPU, `2` CPU.
Замеры, ограничения и выбор раскладки — `docs/backends.md`.

## Библиотека

```rust
use lfm::{Chat, Config};
use lfm_domain::SamplingParams;

let cfg = Config::load(std::path::Path::new("lfm.toml"))?;
let mut chat = Chat::open(&cfg)?;
let params = SamplingParams { max_tokens: 64, ..SamplingParams::default() };
let reply = chat.send("What is the capital of France?", &params)?;
println!("{}", reply.content);
println!(
    "prefill {:.1} tok/s, decode {:.1} tok/s",
    reply.stats.prefill_tok_s(),
    reply.stats.decode_tok_s()
);
# Ok::<(), anyhow::Error>(())
```

`Chat::send_stream` отдаёт события `Thinking` / `Content` / `ToolCall` / `Token`
и поддерживает отмену через `AtomicBool`. Низкоуровневые порты
(`InferenceBackend`, `Sampler`, `Scheduler`) — в крейте `lfm-app`.

## Проверки

```bash
cargo test --workspace
tools/bench_check.sh amd            # корректность на RX 6400
tools/bench_check.sh igpu           # Intel UHD 630
tools/bench_check.sh amd --server   # плюс смоук сервера
```

`lfm-bench` (`tools/bench/`) — диагностический стенд: профили ядер, сравнение
батчей, замеры prefill/decode. В поставку не входит.

## Документация

- `docs/usage.md` — чат, сервер, конфиг, устройства, устранение проблем.
- `docs/config.md` — все ключи TOML и соответствие флагам CLI.
- `docs/backends.md` — ядра, KV, TP/pipeline, замеры.
- `docs/architecture.md` — устройство кода.
- `benchmarks/machine.md` — история замеров и методика.
