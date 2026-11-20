# Архитектура

## Принцип

Зависимости направлены внутрь. `domain` — нижний слой, над ним `application`
(порты и use-cases). Инфраструктура делится на две группы: `infra-gguf`,
`infra-cpu` и `infra-chat` зависят только от `domain` (+ `gguf`/`quant`), а
`backend-opencl` и `infra-tokenizer` реализуют порты и поэтому зависят ещё и от
`application`. `domain` не знает про I/O, OpenCL, GGUF и HTTP.

```
apps/lfm (бинарь `lfm`: chat + serve)          композиция, UX
    │
    ├── crates/lfm (библиотека): Config, Runtime, Chat, Generate
    │        ├── backend-opencl → application (порты) + domain
    │        │                    + infra-gguf/quant/opencl
    │        ├── infra-tokenizer → application (TokenizerPort)
    │        ├── infra-chat → domain
    │        └── infra-gguf, infra-cpu → domain (+ gguf/quant);
    │            infra-opencl, infra-hf, infra-quant — без зависимостей на lfm-*

tools/bench (`lfm-bench`) — отдельный пакет, диагностический стенд, не входит в поставку
```

Конфигурация (TOML + CLI, приоритет и все ключи) — `docs/config.md`.

## Domain (`crates/domain`)

- `ModelConfig` — топология LFM3, извлекается из `MetadataSource` (trait), без знания
  о GGUF. Слои: `Conv | FullAttention | SlidingAttention`, rope-параметры с нулевым
  хвостом, `SamplingParams` (дефолты из метаданных), `Message/ToolCall/ToolDef`.
- Правило: домен не зависит от `lfm-gguf`; адаптер `GgufFile → MetadataSource` живёт
  в `infra-gguf`.

## Application (`crates/application`)

- `ChatSession` — стриминговый парсер ответа: `<think…</think`, `<tool_call>…</tool_call>`,
  буферизация разрезанных тегов, сборка `Message`.
- `Sampler` — temperature/top-k/top-p/min-p/repeat-penalty, seed-детерминизм.
  Выбор top-k идёт bounded-проходом по логитам (без аллокации на весь словарь),
  `exp` считается только для выживших кандидатов; на GPU-декоде это ~1 мс/токен
  при словаре 128k.
- `Scheduler` (`scheduler.rs`) — чистая логика continuous batching: FIFO-очередь
  и активные последовательности, admission по свободным KV-страницам (`ceil(len/32)`
  с учётом резерва под `max_new` активных), план шага (атомарный `prefill`, когда
  декодов нет, иначе `step_mixed`: по 1 токену на декод + prefill-чанки в бюджете
  `server.prefill_budget` (ключ TOML, флаг `--prefill-budget`)), проверка контекста, отмена, позиции декода. Без I/O и GPU —
  покрыт юнит-тестами.
- Сервер (`apps/lfm/src/serve`) работает только через порты: движок не знает
  про OpenCL и HTTP; транспорт (HTTP/1.1, h2c, WS, SSE) — отдельные модули.
- Библиотека `lfm`: `Config` (TOML), `Runtime` (выбор бэкенда и устройств),
  `Chat` (история, reasoning, tools, стриминг), `generate_stream` со счётчиками.

## Инфраструктура

| Крейт | Ответственность |
|---|---|
| `infra-gguf` | mmap-парсер GGUF v2/v3, метаданные, срезы тензоров |
| `infra-quant` | эталонная деквантизация Q4_K/Q6_K/Q8_0, dot для валидации |
| `infra-cpu` | CPU-эталон forward (f32) с KV/conv-состоянием; oracle для ядер |
| `infra-opencl` | устройства, контекст, буферы, диспатч; профилирующая очередь — только для диагностики |
| `infra-hf` | клиент HF: кэш, докачка Range, ETag, sha256 |
| `infra-tokenizer` | BPE из tokenizer.json |
| `infra-chat` | рендер chat_template.jinja (minijinja), thinking/tools |

## Ключевые проектные решения движка

### Память весов
- Формат GGUF остаётся нативным: 4.5/6.5 бит на вес, 0% накладных на трафик.
- Раскладка при загрузке (prepack) может переставлять байты внутри блока; для
  Q4_K блок растёт со 144 до 148 байт (+4 байта: пары скейл/мин распакованы в
  байты рядом с заголовком), raw-копия при этом не хранится. Decode упирается в
  ПСП.
- План (не реализовано): три тира памяти — VRAM → RAM (pinned/mmap) → SSD
  (mmap + префетч) с политикой residency.

### Ядра (W4A8)
- Активации квантуются в int8 по группам 32 (один раз на стадию слоя).
- Веса распаковываются в `char4` на лету, `dot(char4,char4)` — 4 MAC/инструкцию.
- Редукции по подгруппе: при фактической подгруппе 64 (`LFM_SG64`) —
  `sub_group_reduce_add`, без LDS-барьеров; для подгрупп меньше 64 остаётся
  LDS-склад (fallback). Multi-row `RGEMV` и адаптивный размер.
- Fusion на загрузке: ffn up+gate в один буфер (`ffn_ug`), один GEMV на слой.
- Слитые ядра: `rms_quant`, `norm_add_quant`, `silu_quant`, `rms_rope_qk`,
  gate внутри `attention`, KV-store по формату KV — `kv_store_h`,
  `kv_store_b_h`, `kv_store_q`, `kv_store_b_q`, `kv_store_h_rot`,
  `kv_store_b_h_rot`. Тяжёлые поэлементные ядра (`norm_add_quant`,
  `quantize_q8`) работают по 4 потока на группу из 32 и векторизованы
  `float4`/`char4` — иначе упираются в латентность при 64 потоках.
- Профилирующая очередь по умолчанию выключена (+80 мкс/диспатч); включается
  `backend.profile = true` — только для диагностики.
- Размер подгруппы брать из фактического запуска, а не из пробы rusticl
  (наблюдали 32 вместо 64): от него зависит выбор `LFM_SG64`-редукции.
- Скейлы квантованного блока привязаны к строке: при multi-row GEMV читать их
  для каждой строки, а не один раз на блок.
- Состояние shortconv и KV обязаны обнуляться при загрузке модели: буферы
  OpenCL не гарантируют нули, иначе первый токен недетерминирован.

### Prefill
- `prefill` идёт чанками по `backend.prefill_ntok` (по умолчанию 128 токенов);
  один чанк считает `forward_batch(tokens, pos0, need_logits)`. `backend.pb`
  (по умолчанию 6) — порог выбора ядра: при `2..=pb` токенов батч идёт через
  батчевый GEMV (`b8p`), при `ntok > pb` — через GEMM (`lds17`/int8-GEMM).
  `need_logits=false` пропускает lm_head (215 МБ) на промежуточных чанках —
  экономит ~20% трафика на чанк.
- Батчевый GEMV переиспользует распакованные веса на все токены чанка: x-вектор
  каждого токена держится в регистрах (на RGEMV строк), аккумуляторы — `PB×RGEMV`.
- Conv-шаг идёт последовательно по токенам внутри одного ядра (состояние в
  регистрах), внимание — каждый токен по своему префиксу, KV пишется пачкой.
- Длинные префиксы в prefill считаются тем же split-K, что и в decode
  (`attention_split_b` + `attention_combine_b`), чанк 512 позиций, до 16 частей.
- Увеличение `pb` выше 6 (больше токенов идёт через батчевый GEMV вместо GEMM)
  упирается в регистры (спиллы) и замедляет шаг; Q4-часть батчевого GEMV
  ограничена ALU распаковки, Q6 — ПСП (~92 ГБ/с).
- `prefill(slot, prompt, need_logits)` — вход с prefix cache: находит общий
  закэшированный префикс, восстанавливает conv-состояние из снапшота последней
  общей страницы, досчитывает остаток чанками `backend.prefill_ntok` и публикует
  новые блоки. Возвращает число переиспользованных токенов.

### Attention
- Реализовано: GQA, online-softmax, один head на workgroup (64 потока), две
  позиции за итерацию (вдвое меньше барьеров), partial RoPE, gate-ветка.
- KV — страницами по 32 токена: `block_table` на последовательность, свободный
  список страниц, адресация `block_table[pos >> 5] << 5 | (pos & 31)`; у каждой
  последовательности свой conv-стейт и таблица, токены разных запросов можно
  считать одним проходом по весам (`step_mixed`).
- K/V хранятся в fp16 (`half`) в той же раскладке; лейн читает пару соседних dim
  через `vload_half2`, что вдвое режет трафик внимания и VRAM под кэш. Партиалы
  split-K остаются f32. Тип KV-кэша задаётся `backend.kv` (ключ TOML, флаг
  `--kv`; `f16` по умолчанию).
  (`KvBuf` в `LayerBufs`, набор ядер `attention_h`/`attention_b_h`/`kv_store_h`).
  Тайлинг 8 позиций в регистрах/LDS замерен и откачен: не быстрее простого варианта.
- Sliding-слои (окно 256) — окно по позициям; conv-слои — состояние
  `(l_cache-1)×hidden` на последовательность.
- Split-K реализован: при ctx > 512 позиции делятся на чанки по 512 (до 16),
  каждый чанк считает свой partial-софтмакс, отдельное ядро собирает результат.
  Дальше: тайловый prefill-attention (K/V блоками в LDS). KV int8/int4 проверены
  и откачены (на 4096 decode −11%, см. `docs/plan.md`, «Проверено и откачено»);
  возвращаться только под 16k+ и с более дешёвым форматом (fp8 native).

### Порты и use-cases
- `lfm-app::ports`: `InferenceBackend` (`create/free sequence`, `prefill`, `step`,
  `logits`, `logits_row`) и `TokenizerPort` (`encode/decode/eos_id`) — только типы
  application, без зависимостей на инфру.
- `lfm-app::Generate` — greedy prompt→ids/text с гарантированным освобождением слота.
  Адаптеры: `GpuModel` (backend-opencl) и `LfmTokenizer` (infra-tokenizer); фейки в
  `crates/application/tests/generate.rs`, смоук через `lfm-bench gpu-port`.
- Серверный движок (`apps/lfm/src/serve/engine.rs`) работает только через порты:
  `Engine::spawn` принимает `BackendLoader`/`TokenizerLoader`, метрики KV/кэша идут
  через `InferenceBackend::cache_stats/free_pages/max_ctx`. Конкретные реализации
  подставляет `apps/lfm/src/serve/mod.rs` (composition root).

### Планировщик
- Реализован (`crates/application/src/scheduler.rs` + `apps/lfm/src/serve/engine.rs`):
  один поток-движок, `step_mixed` на все активные декоды, chunked prefill в бюджете,
  admission по свободным страницам, отмена по разрыву соединения, метрики шага.
- Радикс префикс-кэш: хэш-цепочка блоков по 32 токена, общие страницы KV с
  refcount, снапшот conv-состояния на страницу, LRU-вытеснение; вход —
  `prefill_begin`/`prefill_finish` (атомарный `prefill` использует их же), который
  возвращает число переиспользованных токенов. Последняя страница промпта всегда
  пересчитывается (нужны логиты).
- Chunked prefill тоже работает с кэшем: `prefill_begin` забирает префикс ещё на
  admission (планировщик сдвигает `prefilled`), `forward_impl` сам пишет снапшоты
  conv на границах страниц, а по завершении чанков `prefill_finish` публикует хэши.
  Проверено: при активном декоде промпт на 191 токен публикует 5 страниц, следующий
  такой же запрос переиспользует 160 токенов, текст ответа совпадает.

### Транспорт и телеметрия
- `lfm serve` (`apps/lfm/src/serve`): HTTP/1.1 и HTTP/2 h2c (свой HPACK + flow control отправки, чтение
  SETTINGS/PING/WINDOW_UPDATE во время ожидания окна), SSE, WebSocket.
- Pipelined HEADERS/DATA, пришедшие во время ожидания окна, буферизуются в соединении
  (до 64 кадров / 1 МиБ) и обрабатываются главным циклом после отправки; окна
  принимающей стороны обновляются один раз.
- `GET /metrics`: атомарный снимок из потока-движка (`active/waiting`, `steps`,
  `avg_step_ms`, KV-страницы, `cache_hits`, `reused_tokens`, `uptime_s`), без
  блокировки генерации; один и тот же JSON по HTTP/1.1 и h2.
- `lfm chat` (`apps/lfm/src/tui.rs`): локальный TUI без сервера — рендер/ввод/
  история, стриминг через `lfm::Chat`, статус с TTFT и prefill/decode tok/s;
  `--once` — один запрос для скриптов.

### Устройства
- TP для decode: all-reduce через host staging (P2P в rusticl нет); PP по слоям —
  `PipelineModel`.
- Устройство выбирается индексом OpenCL; CPU через PoCL подключался как обычное
  OpenCL-устройство, но в текущей системе PoCL не установлен.
- План (не реализовано): абстракция `DeviceSet`, отдельный порт `NpuBackend`
  (Hexagon v68) без влияния на ядро.

## Тестовая стратегия

1. `domain`/`application` — юнит-тесты на фейках, без сети и GPU.
2. `infra-quant` — голден-блоки и сверка с независимой реализацией (gguf-py).
3. `infra-cpu` — эталон; сверка логитов с llama.cpp на усечённых моделях
   (послойная бисекция через зануление тензоров).
4. Ядра — сверка с CPU-эталоном на тех же int8-активациях + замеры ГБ/с и tok/s.
5. Сервер/TUI — контрактные тесты транспорта и планировщика, нагрузочный
   `tools/loadtest.py` и pty-смоук `lfm chat`; отдельного браузерного e2e нет.
