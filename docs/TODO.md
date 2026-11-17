# TODO — продолжить с пустой сессии

Состояние на 17.11.2026 (конец второй сессии подряд). Всё собирается и тестируется:

```bash
cd /home/ivlev98/Проекты/PC/lfm3
cargo build --release --workspace          # без предупреждений
cargo test --workspace                     # 233 теста
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

## Проверки, которые уже прогонялись (и чем повторять)

```bash
# корректность
./target/release/lfm-bench gpu-gen --tokens 64            # ids/текст « Paris, and the capital of Japan is Tokyo…»
./target/release/lfm-bench gpu-cmp --len 601 --batch 6    # cos 0.999937
./target/release/lfm-bench gpu-cmp --len 2048             # cos 0.999592
./target/release/lfm-bench gpu-multi --seqs 6 --tokens 48 # 0/6 расхождений
./target/release/lfm-bench gpu-cache --tokens 512 --reps 2# повтор бит-в-бит, reused 480/512
./target/release/lfm-bench gpu-mix --tokens 24            # cos 1.0

# скорость (машина шумит ±10%, мерить парами/сериями)
./target/release/lfm-bench bw --mb 512                    # ПСП: 112–118 ГБ/с сейчас (в M0 было 125)
LFM_PROFILE=1 ./target/release/lfm-bench gpu-prefill --tokens 8192 --decode 8   # по ядрам
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
  то есть упирается не в DRAM и не в ALU, а в латентность/занятость.

## Приоритеты дальше

Рекомендация: начинать с P0-тайлинга (ограниченный риск), затем аудит батчевого
Q4-GEMV; P3 — если нужна функциональность, P1 — только после перемера на простое.

### 1. P2 — агрегат 6 стримов ≥ 200 (сейчас 148–155)

- Цель недостижима без переработки батчевого Q4-GEMV (см. P0): остальные рычаги закрыты.
- **SoA-раскладка Q4_K замерена и откачена**: +12 Б/блок дороже снятой ALU, выравнивание
  не экономит трафик на непрерывном потоке блоков.
- **Меньше запусков — замерено, не рычаг**: слияние q/k/v/g в decode (−33 запуска
  из ~274) дало ~1% (запуск ~4–5 мкс, а не 28). Оставшиеся кандидаты (`quantize`
  в эпилоге внимания, `rope+kv_store`) дадут ещё ~1–2% суммарно; за цель 200 не
  тянут. Профиль prefill 8192: `q4p_b` 61%, attention 27%, `q6_b` 10%.
- PB 8–10 не повторять: узкое место — загрузки x на группу строк, а не ALU; SoA
  проверена и тоже не помогла.

### 2. P0 — prefill 8192 ≥ 300 (сейчас 152–159)

- **Тайловый prefill-attention** (K/V блоками в LDS, 4–8 позиций за итерацию): закрывает
  27% времени, ожидание +8–15% (потолок ~185 tok/s). В decode тайлинг не дал
  (930 против 835 мкс, откачен), но в `attention_b` другой профиль. Риск качества низкий.
- **Батчевый Q4-GEMV — главный резерв** (61%): идёт на ~24 ГБ/с и ~11–14% int8-пика,
  значит латентность/занятость, а не DRAM/ALU. SoA и PB 8–10 уже проверены и не помогли.
  Нужен аудит: регистры/спиллы, wg на строку, RGEMV, порядок загрузок, ILP; результат
  не гарантирован, но это единственный путь к цели.
- Смешанный батч prefill+decode — для latency/агрегата, не для потолка prefill.

### 3. P1 — decode 8192 ≥ 70 (при простое 63.9, после долгих прогонов 49–57)

- Первым делом перемерить на простое (`bw` 116 ГБ/с, int8 8.8 TOPS в этой сессии);
  цель упирается в GEMV (~14 мс из ~19).
- Идеи: KV-тайлинг в decode не помог (откачен), KV int8/int4 — только если 16k ctx
  станет актуальным; fp16 уже сделан.

### 4. P3 — остаток M5

- `tools`/`tool_calls` в запросе и ответе OpenAI API, `/v1/completions`.
- e2e openai-python (обычный ответ, стриминг, thinking) — пока проверено curl/loadtest.
- Chunked prefill при активных декодах **не публикует** страницы в радикс-кэш: нужен
  `GpuModel::prefill_chunk(slot, tokens, pos0, need_logits)` с публикацией блоков.
- h2: `wait_for_window` игнорирует pipelined HEADERS во время ожидания окна —
  буферизовать или отвечать ошибкой (для OpenAI-клиентов не критично).
- TLS не поддерживается (h2c по prior knowledge) — по требованию.

### 5. Выбор устройства / CPU

- `Gpu::open_with(0, ...)` всегда берёт устройство 0. Для CPU через PoCL нужен
  `--device`/`LFM_DEVICE` (или скрыть rusticl). CPU-эталон `ref-*` живёт отдельно
  (`infra-cpu`, ~1.5 tok/s), к серверу/TUI не подключён.

### 6. P6 — M1 и M6/M7

- Token-parity с llama.cpp на 5 промптах по F16 GGUF (точный oracle для f32-математики).
- Порты `InferenceBackend`/`TokenizerPort`, use-case `Generate` с фейками.
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

## Файлы, изменённые в сессии 2

- `.gitignore` — `__pycache__/`, `*.pyc`; удалён случайно закоммиченный `tools/__pycache__`.
- `README.md`, `docs/plan.md`, этот файл — отрицательные результаты SoA Q4_K и слияний
  запусков, профиль prefill 8192, обновлённые приоритеты.
- Код движка не изменился: SoA-раскладка и decode-фьюжн реализованы, замерены и
  полностью откачены (рабочее дерево чистое).
