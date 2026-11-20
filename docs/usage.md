# Использование

## Установка

```bash
cargo build --release      # бинарь target/release/lfm
```

Модель и файлы HF — см. `README.md` («Сборка»). Пути по умолчанию:
`models/LFM3-1.7B-Q4_K_M.gguf`, `models/hf/{tokenizer.json,chat_template.jinja}`.

## Чат

```bash
lfm chat                                  # TUI на одном GPU
lfm chat --ctx 32768 --kv kvarn           # оверрайды на запуск
lfm chat --once "Say hi." --max-tokens 32 # один запрос, без TUI
echo "..." | lfm chat --once -            # промпт из stdin
```

В TUI: `Enter` — отправить, `PgUp/PgDn` — история, `Ctrl+C` — выход,
`/help`, `/reset`, `/thinking` (или `Ctrl+T`) — показать/скрыть рассуждения,
`/exit`. Снизу постоянная строка статистики: во время ответа — tok/s, TTFT и
число токенов, после — итоговые prefill/decode tok/s, TTFT и токены
(`--no-stats` выключает). Ответы рендерятся как markdown: заголовки, списки,
**жирный**/*курсив*, `код`, блоки кода и ссылки; рассуждения показываются
серым блоком `thinking ›`.

Флаги сэмплинга: `--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repeat-penalty`, `--seed`. Режим рассуждений — `--thinking`
(рассуждения идут отдельным потоком и в TUI показываются как `[thinking]`).

Инструменты (tool calling) включаются JSON-файлом:

```bash
lfm chat --tools examples/tools.json
```

Формат — массив OpenAI tool-объектов (`name`, `description`, `parameters`);
допускается и обёртка `{"tools": [...]}`. Модель вызывает инструмент в
формате шаблона, приложение показывает `[tool_call] name {...}`.

## Сервер

```bash
lfm serve --addr 127.0.0.1:8000 --ctx 8192
```

OpenAI-совместимые эндпоинты:

| метод | путь | смысл |
|---|---|---|
| GET | `/health` | статус, ctx, бэкенд, uptime |
| GET | `/v1/models` | список моделей |
| POST | `/v1/chat/completions` | чат (stream и non-stream) |
| POST | `/v1/completions` | текстовые комплишены |
| GET | `/metrics` | active/waiting, avg_step_ms, стадии, KV-страницы |
| GET | `/ws` | WebSocket-стриминг |

`temperature: 0` включает greedy-путь (argmax на GPU). `tools` в запросе
поддерживаются как в OpenAI API. `stream: true` отдаёт SSE; клиент может
оборвать соединение — сервер отменит генерацию.

Метрики включают `avg_gpu_us`, `avg_logits_us`, `avg_sample_us`,
`avg_detok_us`, `rows_total`, `rows_logits` — по ним видно, что lm_head
считается только для нужных строк.

## Конфигурация

`lfm.toml` в текущем каталоге подхватывается автоматически. Явно:
`lfm --config prod.toml ...`. Приоритет CLI > файл > дефолты. Все ключи —
`docs/config.md`.

```bash
lfm config init            # пример: GPU+iGPU, kvarn, ctx 32768
lfm config print           # эффективный конфиг
```

## Устройства

```bash
# один GPU
lfm chat

# GPU + iGPU (pipeline): 23 слоя на GPU, 1 на iGPU
RUSTICL_ENABLE=iris,radeonsi lfm chat \
  --devices 0,1 --mode pipeline --tiers 96/4 --kv kvarn --ctx 32000

# tensor parallel на двух одинаковых GPU (только kv=f16)
lfm serve --devices 0,1 --mode tp
```

Индексы: без `RUSTICL_ENABLE` — `0` дискретный GPU, `1` CPU; с
`RUSTICL_ENABLE=iris,radeonsi` — `0` GPU, `1` iGPU, `2` CPU.

Ориентиры (промпт ~400 токенов, RX 6400): один GPU — ~0.6 с prefill; с iGPU
в pipeline `96/4` — 2.2 с prefill и ~5 tok/s decode (iGPU в 30–80 раз
медленнее на слой). iGPU имеет смысл для запаса VRAM, а не скорости.

## KV-кэш

| `kv` | память (ctx 8192, 8 seq) | особенность |
|---|---|---|
| `f16` | ≈185 МБ | базовый |
| `kvarn` | ≈63 МБ | K 4 бита, V 2 бита; prefill −10…−31% |
| `kvarn_k4v4` | ≈88 МБ | V 4 бита, качество ближе к f16 |
| `q4` | ≈? | только на устройстве с fp16 |

`ctx` — на одну последовательность; суммарный пул — `ctx × kv_pool`.
Для 16 длинных запросов увеличивайте `kv_pool`.

## Библиотека

```rust
use lfm::{Chat, ChatEvent, Config};
use lfm_domain::SamplingParams;

let cfg = Config::load("lfm.toml".as_ref())?;
let mut chat = Chat::open(&cfg)?;          // модель, токенайзер, шаблон
let params = SamplingParams { max_tokens: 128, ..SamplingParams::default() };
let cancel = std::sync::atomic::AtomicBool::new(false);
let reply = chat.send_stream("hi", &params, &cancel, |event| match event {
    ChatEvent::Content(text) => print!("{text}"),
    ChatEvent::Thinking(text) => eprint!("{text}"),
    ChatEvent::ToolCall(call) => println!("tool {} {:?}", call.name, call.arguments),
    ChatEvent::Token(_) => {}
})?;
println!("{} tok/s", reply.stats.decode_tok_s());
# Ok::<(), anyhow::Error>(())
```

`reply.stats` — `prompt_tokens`, `completion_tokens`, `prefill_s`,
`decode_s`, `ttft_s` и производные `prefill_tok_s()`/`decode_tok_s()`.
`Chat::set_tools`, `Chat::reset`, `Chat::set_system` — управление сессией;
`lfm::load_tools(path)` читает JSON с инструментами.

## Устранение проблем

- `RUSTICL_ENABLE=iris` не видит GPU — проверьте `lfm-bench info`
  (нужен Rusticl; intel-compute-runtime устройство не отдаёт).
- `CL_INVALID_WORK_GROUP_SIZE` — устаревший бинарь; пересоберите
  (сетка зануления округляется до рабочей группы).
- Не хватает VRAM — `kv = "kvarn"`, `kv_pool` меньше, `q4i8_ffn_only = true`,
  `cache = false` (без снапшотов conv) или больше слоёв на iGPU.
- Медленный decode на GPU+iGPU — уменьшите долю iGPU (`tiers = [96,4]`);
  один GPU быстрее.
- Долгий первый запуск — компиляция ядер; кэш в
  `$XDG_CACHE_HOME/lfm3/programs` ускоряет повторные старты.
