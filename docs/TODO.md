# TODO — продолжить с пустой сессии

Актуальное состояние (конец сессии 14): `main` = `origin/main`, дерево чистое,
`cargo build --release --workspace` без предупреждений, `cargo test --workspace`
зелёный. Ядра не оптимизируем: результат зафиксирован в `benchmarks/machine.md` и
`benchmarks/llama-cpp.md`. Итоги последней сессии — раздел «Сессия 14» ниже.

Дополнительно с сессии 12: работает второй бэкенд-путь без fp16 (Intel UHD 630
через Rusticl, `RUSTICL_ENABLE=iris`) — KV во float, 64-полосные ядра и
`gemv_q6_k_i8` заменены, см. `docs/backends.md`.

Разделы «Сессия 1…9» — исторические логи: в них упоминаются удалённые пути
(`LFM_KV=f32`, ядро `lds11`) и старые цифры, актуальными они не являются.

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

## Сессия 6 (GPU-эксперименты, всё откачено)

| Блок | Итог |
|---|---|
| KV int8 + Hadamard | реализовано (K: 4 скейла на группу 32 + поворот, V: скейл на позицию, q поворачивается в attention): 512 +8%, 2048 ≈0, **4096 −11%**, ids расходятся с f16 на 49/64; найден баг `reset()` (заливал i8 как f32 → OOB) — исправлен в эксперименте. Код удалён, fp16 лучше |
| Marlin-микротрюки | `#pragma unroll 2` — −1.5% (8192), `-cl-mad-enable -cl-fast-relaxed-math` — −3.4% (512): удалено |
| Q6/PB тюнинг | Q6-RGEMV 2/3/8 против 5 — 5 быстрее; PB=7 — 145 против 232 tok/s (спиллы); оставлено 5/6 |
| FlashInfer-fusion | отклонено: `kv_store` и attention текущего токена требуют глобального барьера между workgroup; page-table lookups — шум |
| M6/PoCL | CPU-тир PoCL заработал: причины были в хосте (`per_wg` считался как `wg/64`, а PoCL даёт подгруппу = всему workgroup) и в гонках `norm_add_quant` (чтение `red[0]` без барьера перед перезаписью). Теперь `gpu-check` cos 0.9993, `gpu-gen` ids совпадают с AMD, `gpu-cmp` batch cos 0.9994. Intel runtime удалён (iGPU в системе нет) |
| M6/PP | `GpuModel::load_stage` (диапазон слоёв, embed/head), `PipelineModel` (host-handoff, синхронные prefill/кэш), `LFM_DEVICE`/`LFM_TIERS`, CLI `gpu-pp`. 20/80/0 и 50/50 дают ids как single GPU (16 токенов), logits cos 0.999824, top-5; сервер с `LFM_TIERS` переиспользует кэш на обеих ступенях (reused 192) |
| Бэкенды | новый `backend-cpu` (W4A8, AVX2): Q4/Q6 предпак в u8+скейлы, активации i8 по 32, runtime AVX2/FMA + scalar fallback, Rayon; `parse/forward/attention/conv` pure Rust; `InferenceBackend` + `LFM_BACKEND=cpu` + `cpu-port`. ids = GPU Q4 (8 токенов), ~6.9 tok/s; архитектура — `docs/backends.md` |
| M6/TP | full TP: головы q/k/v/g/gate и `attn_output` по столбцам, FFN `up/gate` по строкам + `down` по столбцам, lm_head по словарю с host-gather; `TpShard` режет Q4/Q6 по строкам/блокам, `x/hq/hs` синхронизируются после фаз. `LFM_TP=0/1`/`gpu-tp`: ids как single (8/8), logits cos 0.999745; сервер отвечает |

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
- Идеи: KV-тайлинг в decode не помог (откачен), KV int8 с Hadamard-ротацией
  замерен и откачен (512 +8%, 2048 ≈0, 4096 −11%, ids расходятся с f16 на 49/64);
  fp16 остаётся лучшим вариантом.

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

### 6. Бэкенды и обгон llama.cpp

- **Бейзлайн** (`benchmarks/llama-cpp.md`): llama.cpp Vulkan pp512 1393 / tg128 91,
  pp8192 1236 / tg64 94; CPU pp512 142 / tg128 23.
- **Наши**: OpenCL ~597 pp512 / 525 pp2048 / 402 pp8192 / 64 tg.
- Цель: на каждом бэкенде prefill и decode ≥ llama.cpp. Главный разрыв — prefill
  (GEMM/батчевый Q4) и Q6-GEMM; на CPU — батчевый AVX2-GEMM и 4-битная упаковка.
- **GEMM готов (LDS-тайлинг, сессия 8).** Итоговое ядро `gemm_q4_k_i8_lds11`:
  варп = 4 токена, дорожка = строка (32 строки), x и транспонированные веса в LDS,
  суммы активаций по 32-колоночным группам считаются при загрузке тайла. Дефолт
  `LFM_PREFILL_NTOK=128`. Результат: prefill 512 — 502 tok/s, 8192 — 345 (было 205/159).
  Отрицательные шаги, которые к этому привели: регистровый тайл 2×16 (150 tok/s) и
  LDS-тайл 32×32 с потоком на токен (225 tok/s, банковские конфликты LDS).
- **Q6-GEMM готов**: `prepack_q6_i8` (при загрузке Q6_K → int8 по 32-колоночным
  скейлам, раскладка `[группа][строка][32]`) + ядро `gemm_q6_i8_pp`. Q6-GEMM
  536 → 173 мс (3.1×), prefill 512: 502 → 585 tok/s. Точность cos 0.999886.
  Замечание: LDS-порт без предпака (`gemm_q6_k_i8_lds`) корректен, но не быстрее —
  распаковка 6-битных квантов съедает выигрыш, поэтому нужен именно int8-предпак.
- **Attention на 8192 — главный резерв** (≈40% времени). Проверено и не помогло:
  тайлинг по 2 запросам (трафик K/V не является узким местом — он живёт в L2) и
  subgroup-редукция вместо LDS. Помогло: 4 позиции за итерацию
  (`attention_split_b_h4_p4`) — 380 → 402 tok/s. Ядро latency-bound (~2% ALU-пика);
  дальше — 8 позиций на итерацию, f16x2-математика, warp-shuffle-редукция.
- **Проверка ядер**: `lfm-bench gemm-cmp --tensor blk.2.ffn_up.weight --ntok 64
  --mode lds11` сверяет GEMM с CPU-эталоном (max_rel ~1e-5) и печатает время.


- **CPU готов**: W4A8 AVX2, `LFM_BACKEND=cpu`; ARMv7 — NEON-ядра (`arm_neon.c`).
- **Android (Xperia Z3, MSM8974)**: llama.cpp CPU pp32 3.78 / tg16 2.28 tok/s; наш
  движок — 1.1–1.3 prefill / 0.50 decode (было 0.5 / — на скалярах). OpenCL на Adreno 330
  нецелесообразен (драйвер 1.2 без `cl_khr_integer_dot_product` и подгрупп CL3.0).
  Детали и план — `benchmarks/android.md`.
- **CPU-оптимизация (сессия 9)**: батчевый prefill (тайл 8 токенов) + векторные
  аккумуляторы AVX2 + ниббл-упаковка Q4 + векторизованный `matvec`:
  prefill 10.0 → 54.5 tok/s, decode 8.2 → 15.7 (llama.cpp CPU: 141.5 / 23.3).
  Prefill остаётся ALU-bound (~8 ГБ/s трафика), decode — у полосы памяти.
  Дальше: prefetch/стриминг весов, f16-KV, NEON-ветка.
- **Vulkan/ROCm исключены** — только CPU и OpenCL.
- Архитектура и DoD — `docs/backends.md`.

### 7. P6 — M1 и M6/M7

- **M1 — готов**: token-parity по F16 (5×64, cos 1.000000), порты
  `InferenceBackend`/`TokenizerPort`, `Generate` с адаптерами/фейками и серверный
  движок на портах (composition root — `main.rs`).
- **M6 PP+TP — готовы**: `PipelineModel` (`LFM_TIERS=20/80/0`) и `TpModel`
  (`LFM_TP=0/1`, головы/FFN/lm_head), ids/logits совпадают с single GPU,
  кэш работает на обеих ступенях.
- M7: NPU Hexagon, fuzz парсеров, спека протокола.

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
- **`LFM_DEVICE`**: 0 — rusticl GPU, 1 — PoCL CPU (при обоих ICD). Проверка:
  `LFM_DEVICE=1 ./target/release/lfm-bench gpu-gen --tokens 4`.
- **`LFM_TIERS`** (PP): проценты слоёв по устройствам, например `20/80/0`
  (GPU 4 слоя + PoCL CPU 20). Работает в сервере и `gpu-pp`; CPU-ступень медленная,
  поэтому для серверных тестов брать короткие ответы.
- **`LFM_BACKEND=cpu`**: pure-Rust CPU-бэкенд; веса предпакованы, загрузка ~0.7 с,
  память ~1.6 ГБ, AVX2 определяется в рантайме. Для бенча — `cpu-port`.
- **`LFM_TP`** (TP): список устройств через `/`, например `0/1`; full TP
  (головы/FFN/lm_head), работает в сервере и `gpu-tp`. На GPU+CPU это корректность,
  а не скорость: каждый токен синхронизируется через host.
- **PoCL CPU**: установлен из репозитория, CPU-девайс работает. Ключевые отличия:
  `get_sub_group_size()` может равняться всему workgroup (nsg=1), поэтому сетку
  GEMV нужно считать как `wg / subgroup`, а не `wg / 64`; в `norm_add_quant` нужен
  барьер после чтения `red[0]` перед перезаписью `red` (на GPU-лейнах гонка не
  проявлялась). Проверка: `gpu-check` cos 0.9993, `gpu-gen` ids = AMD.
  Intel NEO удалён (iGPU отсутствует).
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

## Файлы, изменённые в сессии 7

- `crates/backend-cpu/**` (новый крейт): W4A8-веса, AVX2-ядра, forward, `CpuBackend`.
- `Cargo.toml`/`Cargo.lock` — член workspace, зависимости (`rayon`, `lfm-backend-cpu`).
- `apps/server/src/main.rs` — `LFM_BACKEND=cpu`; `apps/bench/src/main.rs` — `cpu-port`.
- `docs/backends.md` (новый), `README.md`, `docs/plan.md`, этот файл.

## Файлы, изменённые в сессии 6

- `crates/backend-opencl/src/lib.rs` — `per_wg` по фактическому `sub_group_size`
  (PoCL: подгруппа = рабочая группа), исправлен `debug_conv` (7 аргументов),
  `load_stage`/hidden-входы/выход для PP.
- `crates/backend-opencl/kernels/decode.cl`, `batch.cl` — барьеры после чтения
  `red[0]` в `norm_add_quant`/`_b`; Q6-GEMV считает блоки по `get_sub_group_size`.
- `crates/backend-opencl/src/pipeline.rs` (новый) — `PipelineModel` (PP по слоям,
  синхронные prefill/кэш, копирование hidden через host).
- `apps/bench/src/main.rs` — `gpu-pp` и диагностика `sg-probe`.
- `crates/backend-opencl/src/tp.rs` (новый) — full TP: `TpShard` (головы/FFN/словарь),
  `TpModel` (all-reduce/gather через host, синхронизация `x/hq/hs`).
- `apps/server/src/main.rs` — загрузчик по `LFM_TIERS`/`LFM_TP` (composition root).
- `apps/bench/src/main.rs` — `gpu-tp`, `cpu-port`, `cpu-prefill`.
- KV int8 (+Hadamard), unroll/fast-math, Q6-RGEMV и PB-варианты реализованы,
  замерены и полностью откачены.
- `docs/plan.md`, этот файл — результаты экспериментов и PoCL-совместимости.

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

## Сессия 14 (контроль AMD, замеры пределов iGPU)

| Блок | Итог |
|---|---|
| Инструменты | `bench.cl` собирается без fp16 (`LFM_F32_KV`): заработали `laten`/`bw`/`peak` на iGPU |
| Задержка запуска | enqueue 0.74 мкс + drain 15.2 мкс на ядро; 274 запуска на токен ≈ 4.4 мс из 167 (2.6%) — слияние ядер не окупается |
| Полоса | линейное чтение 24.7 ГБ/с, шаблон decode-GEMV (16 Б с шагом 1184 Б) — 9.8, фактически 6.6–7 |
| Отвергнуто | LDS-тайлинг decode-GEMV: `ffn_ug` 13.0 против 3.1 мс (фазы не перекрываются); построчная int8-копия и планарная Q4 — тоже (см. machine.md) |
| Гейтация | Q6 через int8 включается по `subgroup < 32`, а не по fp16; в `Caps` добавлены vendor/device_type |
| AMD контроль | f16: pp512 833–866, pp4096 789, pp8192 681–698, decode 63 (512)/28.8 (8192), multi 216–225; kvarn: 676/440/327, decode 50/28.4, cos 0.999205 на 601 |
| Вывод | decode-GEMV на iGPU упирается в перекрытие загрузки/счёт и латентность dot4; kvarn меняет память (403 → 143 МБ) на скорость prefill (−22…−53%) |

## Сессия 13 (kvarn: staging по последовательностям)

| Блок | Итог |
|---|---|
| kvarn без fp16 | `kv=kvarn` разрешён на iGPU: staging и скейлы (`ka`/`kc`/`vb`/`vbz`/`vc`) в f32 (`KV_T`), `kv=q4` по-прежнему требует fp16 |
| Staging | вместо полного f16-контекста — кольцо `KV_STAGE = 128` позиций на последовательность (`seq*128 + pos&127`); размер `max_seqs*128*kvd`; в batch добавлен `seq_b`, commit и attention получают `seq`; в ядрах кламп `seq >= max_seqs` |
| Batch-внимание | `attention_b_h_rot`/`attention_split_b_h_rot` получают коды/скейлы и читают закоммиченные страницы через `lfm_kvarn_k/v`; из staging — только `pos >= full_end`; граница страницы обрабатывается по каждой позиции пары отдельно |
| Память | ctx 8192: 497 МБ → **143 МБ** (f16 403 МБ); ctx 2048: 101 → 74 МБ |
| Проверки | Intel: kvarn `gpu-check` 0.998127, `gpu-cmp` len100/batch40 0.992241, gen совпадает с AMD; AMD: 0.998543, 0.994927, `gpu-multi` 8×6 — 135 tok/s при незафиксированных частотах |
| Ограничение | батч ≤ 128 токенов на последовательность (иначе ошибка при загрузке), tensor parallel с kvarn несовместим |

Побочно найден и исправлен баг batch-пути без fp16: выбор одной строки логитов
шёл через `gemv_q6_k_i8_b` (подгруппа 16 → мусор), теперь Q6 идёт через
`gemm_i8_gz`; после этого генерация на iGPU совпала с AMD.

Оптимизация под iGPU: Q6-decode через `gemm_q6_i8_b8p` вместо `gemm_i8_gz`
(298 → 49 мс/токен), `rgemv=8` при подгруппе ≤16 (+11%), Q6 в батче при
`ntok<=8` тоже через b8p (`gpu-multi` 8×6: 6.6 → 10.7 tok/s); decode
1.9 → 4.1 tok/s, prefill 10.9 → 11.6 (на подгруппе ≤16 отключён префетч весов
int8-GEMM: не хватает регистров). Отдельным заходом: Q4-decode-GEMV на подгруппе
≤16 запускается локальным размером `wg`=512 (было 64 при grid на 256) — decode
4.1 → **6.0** tok/s; clamp строк заменён на `rlim` (без OOB при `rows < RGEMV`).
`rgemv` больше 8 в изоляции быстрее, но в модели хуже (для `attn_q` остаётся
2 рабочие группы на 3 CU), оставлено 8. Проверено и отвергнуто: слияние q/k/v/gate
(3.5 против 4.1),
int8-decode-GEMV (3.4), int8-GEMM тайлом 32×32/WG256 (prefill 7.5 против 11.1).
Замер шаблона доступа: 16-байтные чтения с шагом 1184 Б дают на Intel 9.8 ГБ/с
(AMD 145+), выравнивание ни при чём — потолок шаблона для `attn_q` ~0.25 мс
против ~1.5 мс у ядра, то есть предел в зависимых цепочках dot4 на подгруппе 16.
Профилировщик `LFM_PROFILE=1` переведён на стенное время (timestamps Rusticl
не работают). Резерв: ~360 запусков ядер на токен и тайлинг int8-GEMM под 3 CU.

## Сессия 12 (порт OpenCL на устройство без fp16: Intel UHD 630)

| Блок | Итог |
|---|---|
| Запуск iGPU | Rusticl отдаёт Intel Gen9 только с `RUSTICL_ENABLE=iris`; `lfm-bench info`: OpenCL 3.1, 3 CU, WG 1024, подгруппа 16, `fp16: false`, `integer_dot: true` |
| KV во float | `KV_T`/`KV_V16` в `common.cl`, `-DLFM_F32_KV` включается по `caps.fp16`; `lfm_h2f` (битовая конверсия f16→f32), `lfm_scale`/`lfm_scale_local` для заголовков блоков; буфер KV на f32-устройстве вдвое больше по `u16`; `kv=q4`/`kvarn` отвергаются при загрузке |
| Ядра | `attention_b_h2`/`attention_b_fa`/`attention_split_b_fa*` и attn4 только при `subgroup == 64`; Q6-decode на f32-устройстве идёт через `gemm_i8_gz` (у `gemv_q6_k_i8` неверный результат при подгруппе 16) |
| Отладка | найден баг каста: `(ushort *)blk + 208` давал смещение в элементах — исправлено на `(ushort *)(blk + 208)`, embed cos 1.000000 |
| Точность | `gpu-check` cos 0.998703, top-4 совпадает с CPU; `gpu-trace` все 24 слоя ≥ 0.987; `gemm-cmp` lds17 1.65e-5, q6b8 1.31e-2 |
| Скорость | Intel: prefill 128 — 10.9 tok/s, decode — 1.9 tok/s, `gpu-multi` 2×6 — 3.5 tok/s; AMD не регрессировал (cos 0.998427, prefill 871, decode 82) |
| Тесты | `cargo test --workspace` зелёный, сборка без предупреждений |
| Документация | `docs/config.md` (Intel iGPU), `docs/backends.md` (что меняется без fp16), `benchmarks/machine.md` (цифры и потолки) |

Проверено и откачено: ручной `LFM_F32_KV=1` на AMD — GPU hang (context lost,
hard recovery), поэтому режим включается только по `caps.fp16`.

## Сессия 11 (аудит: исправление остатка)

| Блок | Итог |
|---|---|
| Отмена запроса | `ChatJob.cancel` (Arc<AtomicBool>), `CancelGuard` в HTTP/1.1 и h2: при обрыве клиента флаг ставится и воркер освобождает слот на следующем шаге. Проверено: битый SSE-стрим → слот свободен, следующий запрос за 193 мс |
| Планировщик | `StepPlan::AtomicPrefill` только для промптов ≤ `prefill_budget` (длинные идут чанками, воркер не блокируется на весь промпт) |
| h2c | `MAX_CONCURRENT_STREAMS=8` с RST(REFUSED_STREAM); `WINDOW_UPDATE` отдаётся при накоплении ≥ половины окна (было на каждый DATA → переполнение окна отправителя после ~2 ГБ) |
| TP/pipeline | `max_sequences`/`prefill_batch_size` = минимум по стадиям; rollback слотов в `TpModel::create_sequence` |
| kvarn | предупреждение при загрузке: staging держит полный f16, память не уменьшается (поведение задокументировано) |
| Тесты | эталонный тест раскладки Q4_K (d=1, dmin=0, sc=1 → значение = ниббл, ручная проверка порядка групп), нулевой блок, тесты отмены/чанков |
| clippy | 114 → 13 (остались `too_many_arguments`/`type_complexity` у запусков ядер, `# Safety` добавлен) |

Известное ограничение (не баг кода): вырожденный повторяющийся промпт заставляет
модель выдавать только пустые/спец-токены — ответ приходит с пустым `content` и
`finish_reason: "length"`. `SessionEvent::Finished` теперь завершает генерацию
(раньше игнорировался), но этот случай он не покрывает.

## Сессия 10 (обвязка: конфигурация, сервер, TUI)

Оптимизация ядер остановлена на зафиксированном результате (см.
`benchmarks/machine.md`, `benchmarks/llama-cpp.md`); дальше — обвязка.

| Блок | Итог |
|---|---|
| Конфигурация | `lfm_gpu::Options` (`from_env`, `validate`) вместо чтения env внутри `load_stage`; сервер: `--config <toml>` + `--print-config` (round-trip dump→load→dump), `docs/config.md` со всеми ключами и env; планировщик берёт `pb`/`max_seqs`/`prefill_budget` из конфига, а не из env |
| Сервер | graceful shutdown по SIGINT/SIGTERM (дренаж активных соединений до 10 с, затем принудительный выход), `/health` с моделью/ctx/бэкендом/uptime/версией, `--log-requests` (access log + `sched:`), `--max-body`, проверка `ctx` (кратен 32), h2c отдаёт тот же `/health` |
| TUI | `/help`, `/metrics` (панель из GET /metrics), `/model [name]` (список из /v1/models, переключение модели в запросах), `/system <текст>`, `hist N tok` в статус-строке, разделение уведомлений info/error |
| Обвязка | 0 предупреждений сборки (было 13: 11 «unnecessary unsafe», 2 мёртвых метода), `cargo clippy --fix` (−76), README «Быстрый старт», `justfile` (build/test/clippy/fmt/serve/tui/bench/power) |
| Тесты | 272 (было 251): +4 config, +3 options, +14 TUI |

Проверено вживую: `/health` (`model`/`ctx`/`backend`/`uptime_s`/`version`),
access log, конфиг round-trip, SIGINT во время SSE-стрима — стрим доходит до
`[DONE]`, сервер печатает «остановлен» и выходит. Замеры не изменились
(pp512 871–883, decode 70 при шумной машине).
