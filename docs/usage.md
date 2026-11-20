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

В TUI: `Enter` — отправить, `PgUp/PgDn` — история, `Ctrl+C` во время
генерации — прервать ответ, на пустом вводе — выйти; `/exit` (или `/quit`) —
выход, `/help`, `/reset`, `/thinking` (или `Ctrl+T`) — показать/скрыть
рассуждения. Снизу постоянная строка статистики: во время ответа — tok/s, TTFT и
число токенов, после — итоговые prefill/decode tok/s, TTFT и токены
(`--no-stats` выключает). Ответы рендерятся как markdown: заголовки, списки,
**жирный**/*курсив*, `код`, блоки кода и ссылки; рассуждения показываются
серым блоком `thinking ›`.

Флаги сэмплинга: `--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repeat-penalty`, `--seed`. Режим рассуждений — `--thinking` (рассуждения идут
отдельным потоком: в TUI — блок `thinking ›`, в `--once` — в stderr без метки).

Инструменты (tool calling) включаются JSON-файлом:

```bash
lfm chat --tools examples/tools.json
```

Формат — массив OpenAI tool-объектов (`name`, `description`, `parameters`);
допускается и обёртка `{"tools": [...]}`. Модель вызывает инструмент в
формате шаблона; вызов показывается как `⚙ name args` в TUI и как
`[tool] name args` в `--once`.

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

`temperature: 0` при `repeat_penalty = 1.0` включает greedy-путь (argmax на GPU);
при `repeat_penalty ≠ 1.0` штраф применяется на CPU, и GPU-argmax не
используется. `tools` в запросе поддерживаются как в OpenAI API, но
`tool_choice` понимается только как `"none"` (выбор конкретной функции не
поддержан). `stream: true` отдаёт SSE; клиент может оборвать соединение —
сервер отменит генерацию.

Метрики включают `avg_gpu_us`, `avg_logits_us`, `avg_sample_us`,
`avg_detok_us`, `rows_total`, `rows_logits` — по ним видно, что lm_head
считается только для нужных строк.

## Конфигурация

`lfm.toml` в текущем каталоге подхватывается автоматически. Явно:
`lfm --config prod.toml ...`. Приоритет CLI > файл > дефолты. Все ключи —
`docs/config.md`.

```bash
lfm config init            # пример: single, devices=[0], kv=f16, ctx 4096
lfm config print           # эффективный конфиг
```

## Устройства

```bash
# одно устройство (текущая машина — Vega 8, индекс 0)
lfm chat

# GPU + iGPU (pipeline) — только на машине с двумя устройствами
lfm chat --devices 0,1 --mode pipeline --tiers 96/4 --kv kvarn --ctx 32000

# tensor parallel на двух одинаковых GPU (только kv=f16)
lfm serve --devices 0,1 --mode tp
```

На текущей машине OpenCL-устройство одно: `0` — интегрированный AMD Radeon
Vega 8 (8 CU, подгруппа 64, fp16, память общая с CPU). Дискретной карты и Intel
iGPU в системе нет; `RUSTICL_ENABLE` для Vega 8 не нужен. PoCL не установлен
(в `/etc/OpenCL/vendors` только rusticl), поэтому второго OpenCL-устройства нет
и `pipeline`/`tp` здесь недоступны. Правила режимов — `docs/config.md`, раздел
«Режимы устройств».

Ориентиры по производительности: замеры — `benchmarks/machine.md` (раздел
«Машина 2: Vega 8» заполняется — чисел для Vega 8 пока нет). Числа для
RX 6400/iGPU в старых разделах — история.

## KV-кэш

| `kv` | память (ctx 8192, пул ×1) | особенность |
|---|---|---|
| `f16` | ≈185 МБ (расчёт: 11 attention-слоёв × 512 dim × K/V × 2 байта × 8192) | базовый |
| `kvarn` | ≈66 МБ (≈63 МиБ) | K 4 бита, V 2 бита; prefill −10…−31% |
| `kvarn_k4v4` | ≈77 МБ (≈74 МиБ, расчёт по коду) | V 4 бита, качество ближе к f16 |
| `q4` | ≈48 МБ (расчёт по коду) | только на устройстве с fp16 |

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

- Не видно OpenCL-устройств — проверьте `lfm-bench info` (нужен Rusticl;
  на текущей машине это radeonsi/Vega 8). `RUSTICL_ENABLE=iris` относится к
  Intel iGPU, которого в системе нет.
- `CL_INVALID_WORK_GROUP_SIZE` — устаревший бинарь; пересоберите
  (сетка зануления округляется до рабочей группы).
- Не хватает памяти (у Vega 8 она общая с CPU, UMA) — `kv = "kvarn"`,
  `kv_pool` меньше, `q4i8_ffn_only = true`, `cache = false` (без снапшотов conv).
- Медленный decode в `pipeline` с CPU-стадией — уменьшите её долю (`tiers`);
  на текущей машине второго устройства нет, `pipeline` здесь не запускается.
- Долгий первый запуск — компиляция ядер; кэш в
  `$XDG_CACHE_HOME/lfm3/programs` ускоряет повторные старты.
