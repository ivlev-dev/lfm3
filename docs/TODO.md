# TODO — продолжить с пустой сессии

Состояние на 17.11.2026 (конец пятой сессии подряд). Всё собирается и тестируется:

```bash
cd /home/ivlev98/Проекты/PC/lfm3
cargo build --release --workspace          # без предупреждений
cargo test --workspace                     # 245 тестов
```

`main` = `origin/main`, рабочее дерево чистое, незакоммиченных экспериментов нет.

## Что сделано (сессия 1)

| Блок | Итог |
|---|---|
| KV fp16 | K/V хранятся как `half`, ядра `attention_h`/`attention_split_h`/`attention_b_h`/`attention_split_b_h`, `kv_store_h`, `fill_zero_h`; VRAM под KV −50% (201 МБ на ctx 8192), A/B на 8192 +4–10%, `LFM_KV=f32` — старый путь |
| P2.1 lm_head по строкам | `GpuModel::step_mixed_rows(reqs, logits_rows)`; 1 строка из 6 — 2.08 мс против 4.93; бит-в-бит |
| P2.2 слияние q/k/gate (+v) | prepack-конкатенация на загрузке + `gemv_q4_k_i8_4p_b`; −27.5 запуска/шаг, prefill 512 +2.4% (3/3 пары); `LFM_FUSE=0` |
| P2.3 PB 8–10 | замерено и **откачено**: PB=8 −31%, PB=10 −54% (спиллы 128/256 Б); RGEMV 2/8 и вынос min-коррекции — 0 или хуже |
| P3 планировщик | `crates/application/src/scheduler.rs` + переписанный `apps/server/src/engine.rs`: общий `step_mixed`, chunked prefill, admission по страницам, отмена по разрыву, `LFM_PREFILL_BUDGET`, `LFM_SCHED_LOG` |
| P3 `/metrics` | атомарный снимок из воркера, строгая JSON-схема, HTTP/1.1 + h2 |
| P3 flow control h2 | окна соединения/стримов, SETTINGS с отрицательной дельтой, ожидание WINDOW_UPDATE с чтением входящих фреймов; SSE 124 647 Б по h2c |
| TUI in-process | `gpu-chat --tui`: скролл PgUp/PgDn/Home/End, Up/Down-история с черновиком, resize в стриме, тулколлы, `/help`, Ctrl+C-стоп (`generate_stream_cancel` + `stty min 0 time 0`) |
| TUI клиент | новый крейт `apps/tui` (`lfm-tui`): HTTP/1.1 + chunked/SSE, статус из `/metrics`, скролл/история/resize, Ctrl+C рвёт сокет, `/save`/`/load` |
| Нагрузка | `tools/loadtest.py`: 6 соединений, 0 ошибок, второй запрос стартует до конца первого |

## Сессия 2 (продолжение, 17.11): отрицательные результаты

| Блок | Итог |
|---|---|
| Порядок | веток для слияния/удаления не было (`main` = `origin/main`); удалён случайно закоммиченный `tools/__pycache__`, добавлены `__pycache__/` и `*.pyc` в `.gitignore` |
| SoA Q4_K | реализовано (160 Б/блок: 128 Б кванты + 8 пар f16 `d·scale`/`dmin·min`; ядра `_s`/`_s_b`/`_4s_b`), замерено и **откачено**: −4–5% prefill 512, −5% decode 2048 (по 3 пары); ядро `q4_s_b` 307 против 280 мкс ровно на +8% байт — 144/148 Б и так идут непрерывным потоком, выравнивание секторов ничего не экономит |
| Фьюжн q/k/v/g в decode | замерено и **откачено**: батчевое 4p-ядро в decode −30–40% (регистры PB=6), отдельное ядро без PB — +0.7% (512), −1.4% медиана (2048), +4.6/−0.3% (4096); запуск ~4–5 мкс, 33 запуска ≈ 1% |

## Сессия 3 (P0, тайлинг prefill-attention)

| Блок | Итог |
|---|---|
| `attention_split_b_h4` | один workgroup на (токен, kv-голову, сплит) обслуживает 4 q-головы с общим K/V (GQA 16/4); 8192 — 3/3 пары +0.9…+1.7% (медиана +1.7%), ядро 1627→1418 мкс (−13%), cos/top-5 без изменений; `LFM_ATTN4=0` |
| `attention_b_h2` | 2 позиции за итерацию в sliding/full-short ветке; 512 — медиана +3.3%, 8192 — +1.2…+3.5%; ядро 213→135 мкс (−37%), cos идентичен; `LFM_ATTN2=0` |
| Вместе | 8192 — 3/3 пары +2.1…+3.6% (медиана +3.6%, среднее +2.8%) |
| 4 позиции | **откачено**: −4…−8% (3/3), регистры/LDS съели экономию барьеров |
| WG батчевых GEMV | 256→64: на 8192 128 vs 256 +2.9…+5.8% (3/3), 128→64 ещё +0.1…+2.3% (9/9 на 512/2048/8192); decode оставлен на 256 (`LFM_WG_B`) |
| RGEMV batch | 4→5: 8192 +6.2…+9.0% (3/3, медиана +6.8%), 512 +3.0…+8.1%; decode — 4; найдена и исправлена потеря хвостовых строк при `rows % RGEMV != 0` (`LFM_RGEMV_B`) |
| Итого сессии | против исходной конфигурации: prefill 8192 +14.9%, 512 +14.9%, агрегат 6 стримов +12% (3/3 пары во всех режимах) |

## Сессия 4 (P3: tools и completions)

| Блок | Итог |
|---|---|
| tools в chat | `tools`/`tool_choice` в запросе, `tool_calls`/`tool_call_id` в истории, рендер шаблона с tools, `finish_reason: tool_calls`, стриминговые дельты `tool_calls` по индексам; `/ws` тоже отдаёт `tool_call` |
| `/v1/completions` | raw-режим движка (без шаблона и разбора тегов), строка или массив промптов, stream (один промпт), `usage`, HTTP/1.1 и h2c |
| E2E | `openai-python` 3.52.0: chat, stream, tools (обычный/стрим), completions (обычный/стрим), многоходовой tool-ответ; curl по h2c |
| Chunked prefill + кэш | `prefill_begin`/`prefill_finish`, `PendingPrefill` в `Sequence`, авто-снапшоты conv на границах страниц; admission сдвигает `prefilled`, завершение публикует хэши. Проверка: промпт 191 токен при активном декоде → 5 страниц, повтор reused 160, ответ совпадает; `gpu-cache` 480/512 |
| Хвост сессии | `ChatSession::finish()` вызывается на EOS/лимите (thinking/маркер больше не теряются), не-stream отдаёт `reasoning_content` |
| h2 pipelined HEADERS | `wait_for_window` буферизует HEADERS/CONTINUATION/DATA (тело и окна не теряются, лимит 64 кадра/1 МиБ), главный цикл обрабатывает отложенные кадры первыми; +1 тест |
| Тесты | 240 зелёных (+7: protocol×6, h2 defer) |

## Сессия 5 (M1: token-parity)

| Блок | Итог |
|---|---|
| llama.cpp oracle | собран CPU-only в `/tmp/llama.cpp`; новый `tools/oracle/greedy.cpp` (greedy-ids через llama API) и `tools/oracle/parity.py` (5 промптов, exit code при расхождении) |
| Parity | F16 GGUF: **5/5 промптов × 64 greedy-токена** совпадают с llama.cpp; логиты `cosine 1.000000`, `max|dlogit| 0.0055` |
| Q4 | расхождение принципиально (llama.cpp квантует активации int8), точный oracle только F16/BF16 |
| Порты и Generate | `InferenceBackend`/`TokenizerPort` в `lfm-app`, use-case `Generate` (greedy, гарантированный `free_sequence`), адаптеры для `GpuModel` и `LfmTokenizer`, CLI `gpu-port`; 5 тестов с фейками |
| Движок на портах | `Engine::spawn` принимает `BackendLoader`/`TokenizerLoader`; в `engine.rs` нет импортов `lfm-gpu`/`lfm-tokenizer`, метрики KV/кэша — через `InferenceBackend`; смоук сервера (tools, completions, loadtest) пройден |
| Тесты | 245 зелёных (+5 Generate) |

## Проверки, которые уже прогонялись (и чем повторять)

```bash
# корректность
./target/release/lfm-bench gpu-gen --tokens 64            # ids/текст « Paris, and the capital of Japan is Tokyo…»
./target/release/lfm-bench gpu-cmp --len 601 --batch 6    # cos 0.999937
./target/release/lfm-bench gpu-cmp --len 2048             # cos 0.999592
./target/release/lfm-bench gpu-multi --seqs 6 --tokens 48 # 0/6 расхождений
./target/release/lfm-bench gpu-cache --tokens 512 --reps 2# повтор бит-в-бит, reused 480/512
./target/release/lfm-bench gpu-mix --tokens 24            # cos 1.0
python3 tools/oracle/parity.py --tokens 64                # 5/5 против llama.cpp (F16)

# скорость (машина шумит ±10%, мерить парами/сериями)
./target/release/lfm-bench bw --mb 512                    # ПСП: 112–118 ГБ/с сейчас (в M0 было 125)LFM_PROFILE=1 ./target/release/lfm-bench gpu-prefill --tokens 8192 --decode 8   # по ядрам
./target/release/lfm-bench gpu-prefill --tokens 8192 --decode 24
./target/release/lfm-bench gpu-prefill --tokens 512 --decode 32
./target/release/lfm-bench gpu-multi --seqs 6 --tokens 48
```

Текущие числа (17.11, машина медленнее записанной в M0: int8 8.8 против 12.3 TOPS):

- decode 8192 — 49–57 tok/s, prefill 8192 — 152, prefill 512 — 178–197, агрегат 6 стримов — 148–155;
- перемер в начале следующей сессии: bw 116 ГБ/с, prefill 8192 — 159, prefill 512 — 182,
  decode @8192 — 64, агрегат 6 — 124 (разброс машины между сессиями 10–20%);
- сервер: `tools/loadtest.py --conns 6 --tokens 24` — 0 ошибок, ~45 tok/s агрегата;
- префикс-кэш через сервер: второй одинаковый промпт — reused 512 токенов, hits 16;
- prefill 8192 по ядрам (`LFM_PROFILE=1`): `q4p_b` 31.1 с (**61%**), attention 13.9 с (**27%**),
  `q6_b` 5.0 с (10%); батчевый Q4-GEMV — ~24 ГБ/с (21% ПСП) и ~11–14% int8-пика,
  то есть упирается не в DRAM и не в ALU, а в латентность/занятость;
- после тайлинга attention: prefill 8192 — 150–153 tok/s на этой машине, доля attention
  в профиле ~23% (`attention_split_b_h4` 1418 мкс, `attention_b_h2` 135 мкс);
- после аудита GEMV (WG=64, RGEMV=5): prefill 8192 — 171–177, 512 — 204–214 tok/s;
  cos 601 — 0.999937, 2048 — 0.999592 (как до изменений);
- суммарный A/B сессии на 8192: 148.5 → 170.6 tok/s (**+14.9%**), 512: 179.0 → 205.7
  (+14.9%), агрегат 6 стримов: 140 → 159 (+12%), 3/3 пары везде.

## Приоритеты дальше

Рекомендация: тайлинг attention и аудит WG/RGEMV сделаны; следующий шаг —
x-вектор в LDS и схема загрузок Q4-GEMV. P3 — если нужна функциональность,
P1 — только после перемера на простое.

### 1. P2 — агрегат 6 стримов ≥ 200 (сейчас 159, после сессии)

- Цель недостижима без переработки батчевого Q4-GEMV (см. P0): остальные рычаги закрыты.
- **SoA-раскладка Q4_K замерена и откачена**: +12 Б/блок дороже снятой ALU, выравнивание
  не экономит трафик на непрерывном потоке блоков.
- **Меньше запусков — замерено, не рычаг**: слияние q/k/v/g в decode (−33 запуска
  из ~274) дало ~1% (запуск ~4–5 мкс, а не 28). Оставшиеся кандидаты (`quantize`
  в эпилоге внимания, `rope+kv_store`) дадут ещё ~1–2% суммарно; за цель 200 не
  тянут. Профиль prefill 8192: `q4p_b` 61%, attention 23%, `q6_b` 10%.
- PB 8–10 не повторять: узкое место — загрузки x на группу строк, а не ALU; SoA
  проверена и тоже не помогла.

### 2. P0 — prefill 8192 ≥ 300 (сейчас 171–177)

- **Тайлинг prefill-attention — сделан** (сессия 3): +2.8% в среднем на 8192,
  attention в профиле 27% → 23%.
- **Батчевый Q4-GEMV** (61%): WG=64 и RGEMV=5 дали +6.2…+9.0% на 8192, но цель
  всё ещё далеко. Остались идеи: x-вектор в LDS (сейчас 3:1 к весам), схема загрузок,
  ILP. SoA, PB 8–10 и фьюжны запусков уже проверены и не помогли.
- Смешанный батч prefill+decode — для latency/агрегата, не для потолка prefill.

### 3. P1 — decode 8192 ≥ 70 (при простое 63.9, после долгих прогонов 49–57)

- Первым делом перемерить на простое (`bw` 116 ГБ/с, int8 8.8 TOPS в этой сессии);
  цель упирается в GEMV (~14 мс из ~19).
- Идеи: KV-тайлинг в decode не помог (откачен), KV int8/int4 — только если 16k ctx
  станет актуальным; fp16 уже сделан.

### 4. P3 — остаток M5

- **Сделано (сессия 4)**: `tools`/`tool_calls`, `/v1/completions`, e2e openai-python,
  радикс-кэш в chunked prefill, flush сессии и `reasoning_content`, pipelined HEADERS
  в h2 `wait_for_window`.
- Осталось только TLS (h2c по prior knowledge) — по требованию; для OpenAI-клиентов
  не критично.

### 5. Выбор устройства / CPU

- `Gpu::open_with(0, ...)` всегда берёт устройство 0. Для CPU через PoCL нужен
  `--device`/`LFM_DEVICE` (или скрыть rusticl). CPU-эталон `ref-*` живёт отдельно
  (`infra-cpu`, ~1.5 tok/s), к серверу/TUI не подключён.

### 6. P6 — M1 и M6/M7

- **M1 — готов**: token-parity по F16 (5×64, cos 1.000000), порты
  `InferenceBackend`/`TokenizerPort`, `Generate` с адаптерами/фейками и серверный
  движок на портах (composition root — `main.rs`).
- M6: 3-tier + TP/PP; M7: NPU Hexagon, fuzz парсеров, спека протокола.

## Подводные камни

- **Машина шумит**: абсолютные tok/s между сессиями различаются на 10–20%. Всегда
  A/B парами (чередование), проверять `bw --mb 512` перед замерами.
- **`LFM_PROFILE=1`** замедляет диспатч и искажает wall-time; абсолютные времена ядер
  тоже завышены (сумма по ядрам больше wall) — сравнивать только относительно, в одном
  режиме профилирования.
- **Стоимость запуска ~4–5 мкс**, а не 28: проверено слиянием 33 запусков (≈1%).
  Фьюжны запусков как рычаг производительности закрыты.
- **`pkill -f "lfm-server ..."`** убивает и собственный shell, если шаблон в командной
  строке — использовать `pkill -f lfm-server` аккуратно или `kill <pid>`.
- **Долгие прогоны** (prefill 8192 ~55 с) греют GPU и занижают decode сразу после —
  для decode-замеров делать паузу или мерить на коротком контексте.
- `models/` (4.4 ГБ) в git не попадает — качать по README.
- `LFM_PB` дублируется в `GpuModel` и в движке (`LFM_PB`, `LFM_SEQS`) — менять синхронно.
- **RGEMV и хвосты**: одиночные batch-GEMV при `rows % RGEMV != 0` пересчитывают
  последнюю группу с перекрытием (иначе терялись строки); 4p-ядро `q4x4p_b` оставлено
  на RGEMV=4 — его группы не должны пересекать границы сегментов q/k/v/g (qd/kvg/heads
  делятся на 4, но не на 5).
- **`/v1/completions` — raw-режим**: без чат-шаблона и без BOS (`encode(..., false)`),
  поэтому для instruct-модели это чистое продолжение текста; для диалогов нужен
  `/v1/chat/completions`. Стрим поддерживает только один промпт.

## Запуск

```bash
# TUI на своём процессе (без сервера)
./target/release/lfm-bench gpu-chat --tui

# сервер + TUI-клиент
./target/release/lfm-server --ctx 4096 &
./target/release/lfm-tui --url http://127.0.0.1:8000

# нагрузка и метрики
python3 tools/loadtest.py --conns 6 --tokens 24
curl -s http://127.0.0.1:8000/metrics
curl -sN --http2-prior-knowledge -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"max_tokens":64,"messages":[{"role":"user","content":"Hi"}]}'
```

## Файлы, изменённые в сессии 1

- `crates/backend-opencl/src/lib.rs`, `kernels/decode.cl`, `kernels/batch.cl` — KV fp16,
  `step_mixed_rows`, слияние q/k/gate, `generate_stream_cancel`, `free_pages`.
- `crates/application/src/scheduler.rs` (новый), `crates/application/src/lib.rs`.
- `apps/server/src/engine.rs`, `protocol.rs`, `main.rs`, `h2.rs` — планировщик,
  `/metrics`, flow control.
- `apps/bench/src/main.rs` — in-process TUI, Ctrl+C-стоп, `pending`-буфер ввода.
- `apps/tui/**` (новый крейт), корневой `Cargo.toml`.
- `tools/loadtest.py` (новый).
- `README.md`, `docs/plan.md`, `docs/architecture.md`, этот файл.

## Файлы, изменённые в сессии 5

- `crates/application/src/ports.rs` (новый), `src/generate.rs` (новый),
  `tests/generate.rs` (новый, 5 тестов), экспорты в `lib.rs`.
- `crates/backend-opencl/src/ports.rs` — `InferenceBackend` для `GpuModel`
  (включая `prefill_begin/finish`, `max_ctx`, `free_pages`, `cache_stats`);
  `crates/infra-tokenizer/src/ports.rs` + зависимость `lfm-app` — `TokenizerPort`.
- `apps/server/src/engine.rs`, `main.rs` — движок на портах, загрузчики в composition root.
- `apps/bench/src/main.rs` — команда `gpu-port`.
- `tools/oracle/greedy.cpp` (новый) — greedy-ids через llama API для сверки.
- `tools/oracle/parity.py` (новый) — parity-скрипт на 5 промптах.
- `tools/oracle/README.md` — сборка `greedy.cpp`, запуск parity, результаты F16.
- `README.md`, `docs/plan.md`, `docs/architecture.md`, этот файл — статус M1.

## Файлы, изменённые в сессии 4

- `apps/server/src/protocol.rs` — tools/tool_calls, tool_choice, finish_reason
  `tool_calls`, `CompletionRequest`, формы ответов/дельт и 6 новых тестов.
- `apps/server/src/engine.rs` — tools в шаблоне, `Event::ToolCall`, `Final` с
  `tool_calls`/`finish_reason`/`thinking`, raw-режим, `ChatJob::chat/completion`,
  admission с `prefill_begin` и публикация на завершении, flush сессии.
- `apps/server/src/main.rs`, `h2.rs` — маршрут `/v1/completions`, tools и tool-дельты
  в HTTP/1.1, h2c и WebSocket.
- `crates/backend-opencl/src/lib.rs` — `prefill_begin`/`prefill_finish`,
  `PendingPrefill`, авто-снапшоты conv в `forward_impl`.
- `README.md`, `docs/plan.md`, `docs/architecture.md`, этот файл.

## Файлы, изменённые в сессии 3

- `crates/backend-opencl/kernels/batch.cl` — `attention_split_b_h4` (GQA-шеринг K/V),
  `attention_b_h2` (пары позиций), RGEMV_B и корректный хвост batch-GEMV;
  4-позиционный вариант удалён после замера.
- `crates/backend-opencl/src/lib.rs` — ядра, выбор веток, WG/RGEMV для decode и batch,
  флаги `LFM_ATTN4`/`LFM_ATTN2`/`LFM_WG`/`LFM_WG_B`/`LFM_RGEMV`/`LFM_RGEMV_B`,
  `kern-info` показывает prepacked-ядра.
- `README.md`, `docs/plan.md`, этот файл — замеры и новые флаги.

## Файлы, изменённые в сессии 2

- `.gitignore` — `__pycache__/`, `*.pyc`; удалён случайно закоммиченный `tools/__pycache__`.
- `README.md`, `docs/plan.md`, этот файл — отрицательные результаты SoA Q4_K и слияний
  запусков, профиль prefill 8192, обновлённые приоритеты.
- Код движка не изменился: SoA-раскладка и decode-фьюжн реализованы, замерены и
  полностью откачены (рабочее дерево чистое).
