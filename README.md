# lfm3-engine

Движок инференса для **LiquidAI LFM3-1.7B** на Rust + OpenCL (rusticl). Цель — vLLM-подобный
сервинг (paged attention, continuous batching, radix prefix cache) с максимальной
производительностью на потребительских GPU.

## Статус

- **M0 (спайк ядер) — готов.** Замеры на RX 6400: `benchmarks/m0-rx6400.md`.
- **M2 (GPU decode) — работает, цель на 8192 не взята.** Полный граф LFM3 на OpenCL в
  `crates/backend-opencl`: текст совпадает с llama.cpp, **82.7 tok/s** на коротком
  промпте, 74.8 на ctx 512, 71.4 на ctx 4096, 65.5 на ctx 8192 (цель ≥ 70).
- **M3 (paged KV + батч шага) — база готова.** KV живёт страницами по 32 токена
  (`block_table` на последовательность), у каждой последовательности свой conv-стейт
  и своя таблица; шаг может считать до 6 запросов за один проход по весам
  (`step_mixed`). Замер: 4 стрима — 128 tok/s, 6 стримов — **143 tok/s** агрегат,
  8 стримов — 125 tok/s (один стрим — 75), расхождений с одиночным прогоном нет
  (цель 200 — упирается в ALU батчевого GEMV и lm_head).
- **KV fp16 — готово.** K и V лежат как half в той же раскладке страниц (`vload_half2`
  на пару соседних dim), ядра `attention_h`/`attention_split_h`/`attention_b_h`/
  `attention_split_b_h` + `kv_store_h`. VRAM под KV вдвое меньше (402 → 201 МБ при
  ctx 8192), A/B на 8192 (три пары чередующихся прогонов) — **+4–10% decode**;
  `gpu-gen` даёт те же ids/текст, `gpu-cmp --len 601 --batch 6` cos 0.999937,
  `gpu-cache` повтор бит-в-бит. Отключение — `LFM_KV=f32`. Тайлинг 8 позиций на
  тайл (регистры/LDS) замерен и откачен: 930 против 835 мкс на вызов
  `attention_split` при 8192 — ядро упирается не в трафик и не в барьеры.
- **M4 (radix prefix cache) — готов.** `GpuModel::prefill` находит самый длинный
  закэшированный префикс, делит страницы KV между последовательностями по refcount,
  восстанавливает conv-состояние из снапшота страницы и вытесняет блоки по LRU.
  Замер: повтор промпта 512 токенов — **193 мс против 2773 мс** (480 из 512 токенов
  из кэша), логиты бит-в-бит как на холодном прогоне; две последовательности с общим
  префиксом делят страницы без расхождений. Отключение — `LFM_CACHE=0`.
- **Prefill — работает батчами.** Чанки по 6 токенов, результат совпадает с
  последовательным проходом. Слияние q/k/gate (+v, где Q4) GEMV на загрузке дало
  +2.4% на 512 токенах (`LFM_FUSE=0` — A/B); lm_head в батче считается только для
  нужных строк (`step_mixed_rows`, экономит ~2.9 мс/шаг, когда логиты нужны 1 из 6).
  Замер (текущее состояние машины): **178–197 tok/s** на 512, 152 на 8192
  (цель 300 на чанке ≥ 4096; PB 8–10 упирается в регистры и замерен как −31/−54%).
  Тайлинг prefill-attention (общий K/V на 4 q-головы в split-ядре + 2 позиции
  за итерацию в sliding) дал **+2.8–3.6%** на 8192 (3/3 пары, логиты без изменений);
  отключение — `LFM_ATTN4=0`, `LFM_ATTN2=0`. Аудит батчевых GEMV: workgroup 256→64
  и RGEMV 4→5 дали ещё **+6.8%** на 8192 (3/3 пары); хвост строк при RGEMV, не
  делящем `rows`, теперь пересчитывается с перекрытием, а не теряется.
  Суммарно изменения сессии (attention + GEMV) — **+14.9%** на 8192 и 512
  (3/3 пары: 148→171 и 179→205 tok/s), агрегат 6 стримов 140→159.
- **Сэмплинг и чат работают.** `gpu-gen` принимает temperature/top-k/top-p/repeat-penalty/seed,
  `gpu-chat` рендерит `chat_template.jinja`, стримит ответ и разбирает `<think`/`<tool_call>`
  через `ChatSession`. Замер: 28-токенный промпт + 45 токенов ответа — prefill 0.16 с,
  decode 78 tok/s.
- **M6 (pipeline parallel) — база готова.** `GpuModel::load_stage` грузит диапазон
  слоёв (embed только на первой, lm_head только на последней), `PipelineModel`
  гоняет PP по слоям через host-handoff; устройство задаётся `LFM_DEVICE`, раскладка —
  `LFM_TIERS=20/80/0` (GPU/PoCL-CPU/заглушка). Проверка: `gpu-pp --tiers 20/80/0`
  даёт те же greedy-ids, что одиночный GPU (16 токенов), `logits cos 0.999824`,
  top-5 совпадает; сервер с `LFM_TIERS` отдаёт ответы и переиспользует радикс-кэш
  на обеих ступенях (повтор — reused 192 токена).
- **Модульные бэкенды: CPU-бэкенд готов.** Новый крейт `backend-cpu` (pure Rust):
  веса Q4_K/Q6_K предпакованы в u8+скейлы (1.6 ГБ вместо 6.8 ГБ f32), активации
  W4A8 по группам 32, AVX2 (`maddubs`/`fmadd`, runtime-детект, scalar fallback),
  Rayon по строкам. `cpu-port` даёт те же greedy-ids, что GPU Q4 (8 токенов),
  ~6.9 tok/s на i5-10400; сервер запускается с `LFM_BACKEND=cpu`. Архитектура и
  архитектура бэкендов — `docs/backends.md`.
- **M6 (tensor parallel) — готов.** Full TP на двух устройствах: q/k/v/g и gate —
  по головам, `attn_output` — по столбцам, FFN — `up/gate` по строкам и `down`
  по столбцам, lm_head — по словарю с host-gather. `LFM_TP=0/1`/`gpu-tp`: ids
  совпадают с одиночным GPU (8/8), `logits cos 0.999745`; на GPU+CPU это
  корректность, не ускорение.
- **M1 (эталон/токенизация) — parity достигнут.** CPU-эталон LFM3 (f32) по F16 GGUF
  совпадает с llama.cpp по greedy-ids: **5 промптов × 64 токена**, логиты `cos 1.000000`,
  `max|dlogit| 0.0055`; найден и исправлен сдвиг conv-ядра в GGUF. Инструменты —
  `tools/oracle/` (`parity.py`, `compare_prompt.py`, `greedy.cpp`). На Q4 расхождение
  остаётся из-за квантования активаций — точный oracle только F16. Порты
  `InferenceBackend`/`TokenizerPort` и use-case `Generate` готовы (адаптеры для
  `GpuModel`/`LfmTokenizer`, фейки в тестах, CLI `gpu-port`); серверный движок
  работает только через порты.
- **M5 (сервер, планировщик и TUI) — готов.** `lfm-server` отдаёт
  `/v1/models`, `/v1/chat/completions` (обычный ответ и SSE, stop, usage, thinking,
  `tools`/`tool_calls` с `finish_reason: tool_calls`), `/v1/completions` (строка или
  массив промптов, стрим), `/metrics` (телеметрия движка), `/ws` и HTTP/2 h2c
  с flow control. История с assistant-`tool_calls` и `role: tool` проходит через
  шаблон; e2e проверен `openai-python` (chat, stream, tools, completions).
  Планировщик continuous batching (`crates/application/src/scheduler.rs`):
  общий `step_mixed` на все активные декоды, chunked prefill, admission по свободным
  KV-страницам, отмена по разрыву соединения; нагрузка 6×24 — 45 tok/s агрегата,
  0 ошибок, второй запрос стартует до конца первого. Chunked prefill тоже публикует
  страницы в радикс-кэш и восстанавливает conv-состояние: промпт 191 токен,
  посчитанный при активном декоде, даёт повтору reused 160 токенов с тем же ответом.
  Два TUI: `gpu-chat --tui` (in-process: скролл, история ввода, resize, Ctrl+C-стоп)
  и `lfm-tui` (`apps/tui`, клиент к серверу: SSE, `/metrics` в статусе, `/save`/`/load`).
- **Тесты:** 248 зелёных (`cargo test --workspace`), сборка без предупреждений.

## Что дальше

Полный план с DoD — в [`docs/plan.md`](docs/plan.md) («План дальше»), чек-лист для
продолжения с пустой сессии (команды, подводные камни, приоритеты) — в
[`docs/TODO.md`](docs/TODO.md). Кратко:

1. **P0. Prefill 8192 → 300 tok/s** (сейчас 171–177): тайловый prefill-attention уже
   сделан (+2.8–3.6% на 8192), аудит батчевых GEMV дал ещё +6.8% (WG=64, RGEMV=5).
   Дальше — x-вектор в LDS и новые схемы загрузок Q4; цель пока не достигнута.
2. **P1. Decode 8192 → 70 tok/s**: KV fp16 уже даёт −50% VRAM под кэш и +4–10%
   к decode; KV int8 с Hadamard-ротацией замерен и откачен (4096 −11%, ids расходятся
   с f16); остаток упирается в GEMV (int8-пик 8.8 TOPS в текущем состоянии машины).
3. **P2. Агрегат 6 стримов → 200 tok/s** (сейчас 148, после сессии — 159): lm_head по строкам и слияние
   q/k/gate уже в коде (+2.4% prefill, бит-в-бит). PB 8–10 замерен и откачен
   (−31/−54%, спиллы). SoA-раскладка Q4_K замерена и откачена (−4–5% prefill,
   −5% decode 2048). Слияния запусков замерены и откачены (~1%: запуск ~4–5 мкс,
   а не 28). Остаток — ALU батчевого GEMV.
4. **P3. Остаток M5**: e2e openai-python уже покрыт (chat/stream/tools/completions),
   chunked prefill публикует страницы в радикс-кэш, pipelined HEADERS в h2 буферизуются;
   остался только TLS по требованию.
5. **Цель — обогнать llama.cpp** (`benchmarks/llama-cpp.md`) на CPU и OpenCL:
   prefill и decode. Бейзлайн: CPU 142 pp512 / 23 tg128, Vulkan 1393 / 91.
   Главный разрыв — prefill (GEMM): CPU ×50, OpenCL ×2.3 (512) / ×3.1 (8192);
   decode OpenCL ×1.4. Готовы LDS-GEMM для Q4 (`gemm_q4_k_i8_lds11`), int8-предпак
   для Q6 (`gemm_q6_i8_pp`) и attention по 4 позиции за итерацию:
   prefill 512 — 597 tok/s, 2048 — 525, 8192 — 402. Дальше — attention (40% времени
   на 8192) и эффективность Q4-GEMM.
   Vulkan/ROCm исключены из плана.

## Железо и целевые показатели

Тестовая машина: AMD Radeon RX 6400 (navi24, 12 CU, 4 ГиБ, ~128 ГБ/с), Mesa rusticl.

| Режим | Цель |
|---|---|
| decode, 1 стрим, ctx 8192 | ≥ 70 tok/s |
| decode, 4–6 стримов | ≥ 200 tok/s агрегат |
| prefill, промпт 8192, чанк ≥ 4096 | ≥ 300 tok/s (1 стрим), ≥ 800 агрегат |
| модель целиком в VRAM, Q4_K_M | 1.09 ГБ весов |

Измерено: GEMV-часть токена — 10.3 мс (105 ГБ/с при «полу» памяти ~10.0 мс),
остальное (внимание, нормы, квантование, диспатч) — ~1.7 мс. При ctx 8192
split-K внимание добавляет ~1.8 мс (трафик KV).

## Архитектура

Зависимости направлены внутрь: `apps → infra → application → domain`.

```
crates/
  domain/           модель, конфиг, чат/тулколлы, сэмплинг (без I/O)
  application/      порты и use-cases: сессия чата, планировщик, движок,
                    `InferenceBackend`/`TokenizerPort`, `Generate`
  backend-cpu/      pure Rust CPU: W4A8, AVX2 (NEON — позже), Rayon
  backend-opencl/   GPU через OpenCL (rusticl/PoCL)
  infra-gguf/       парсер GGUF + адаптер метаданных
  infra-quant/      эталонная деквантизация Q4_K/Q6_K/Q8_0
  infra-cpu/        CPU-эталон forward (f32)
  infra-opencl/     устройства, контекст, очереди, буферы, диспатч
  infra-hf/         клиент HuggingFace Hub (кэш, докачка, sha256)
  infra-tokenizer/  BPE-обёртка над tokenizer.json
  infra-chat/       рендер chat_template.jinja (minijinja)
apps/bench/         бенчмарки и инструменты (GEMV, bandwidth, oracle-сверка)
```

## Быстрый старт

```bash
# 1. Модель (Q4_K_M, ~1.1 ГБ)
mkdir -p models && curl -L -o models/LFM3-1.7B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf

# 2. Токенизатор и шаблон чата
mkdir -p models/hf
for f in tokenizer.json chat_template.jinja config.json; do
  curl -L -o models/hf/$f https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/$f
done

# 3. Сборка и тесты
cargo build --release
cargo test --workspace

# 4. Бенчмарки ядер
cargo run --release -p lfm-bench -- info
cargo run --release -p lfm-bench -- bw --mb 512
cargo run --release -p lfm-bench -- peak
cargo run --release -p lfm-bench -- step --mode fused --rgemv 4
cargo run --release -p lfm-bench -- sweep

# 5. Проверка корректности и производительности GPU-графа
cargo run --release -p lfm-bench -- gpu-gen --tokens 64      # текст + tok/s
cargo run --release -p lfm-bench -- gpu-port --tokens 24     # через порты + Generate
cargo run --release -p lfm-bench -- gpu-pp --tokens 8 --tiers 20/80/0  # PP: GPU+PoCL
cargo run --release -p lfm-bench -- gpu-tp --tokens 8 --devices 0/1      # full TP
cargo run --release -p lfm-bench -- cpu-port --tokens 8                   # CPU W4A8/AVX2
cargo run --release -p lfm-bench -- gpu-gen --tokens 64 --temp 0.7 --top-k 40 --top-p 0.95
cargo run --release -p lfm-bench -- gpu-chat --user "What is the capital of Japan?"
cargo run --release -p lfm-bench -- gpu-multi --seqs 6 --tokens 48   # 6 стримов
cargo run --release -p lfm-bench -- gpu-cmp --ids 597,5205,302 --batch 5  # batch == seq
cargo run --release -p lfm-bench -- gpu-prefill --tokens 512 --decode 32
cargo run --release -p lfm-bench -- gpu-cache --tokens 512 --reps 3  # prefix cache
cargo run --release -p lfm-bench -- gpu-mix --tokens 24        # prefill+decode одним проходом
cargo run --release -p lfm-bench -- gpu-chat --tui             # in-process TUI-чат
cargo run --release -p lfm-server -- --ctx 4096                # OpenAI-совместимый сервер
cargo run --release -p lfm-tui                                 # TUI-клиент к серверу
# HTTP/1.1, HTTP/2 (h2c), SSE и WebSocket на одном порту:
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics
curl -s --http2-prior-knowledge http://127.0.0.1:8000/health
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"Hi"}]}'
# tool calling и /v1/completions:
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
curl -s -X POST http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16}'
python3 tools/loadtest.py --conns 6                            # нагрузка на сервер
cargo run --release -p lfm-bench -- gpu-embed --id 597       # сверка отдельных ядер
```

Переменные окружения: `LFM_PB` (размер prefill-чанка, по умолчанию 6),
`LFM_PROFILE=1` (сбор событий OpenCL по ядрам, замедляет диспатч),
`LFM_CACHE=0` (выключить радикс-кэш префиксов и снапшоты conv-состояния),
`LFM_PREPACK=0` (не строить prepacked-копию Q4-весов для батчевого prefill),
`LFM_KV=f32` (хранить KV во float вместо half — для A/B и отладки),
`LFM_FUSE=0` (не сливать q/k/gate GEMV), `LFM_ATTN4=0` (не шарить K/V на 4 q-головы
в split prefill-attention), `LFM_ATTN2=0` (не обрабатывать пары позиций в sliding
prefill-attention), `LFM_WG` / `LFM_WG_B` (workgroup decode/batch GEMV, по умолчанию
256/64), `LFM_RGEMV` / `LFM_RGEMV_B` (строк на подгруппу decode/batch, 4/5),
`LFM_DEVICE` (индекс OpenCL-устройства: 0 — rusticl GPU, 1 — PoCL CPU),
`LFM_TIERS=20/80/0` (PP-раскладка по устройствам), `LFM_TP=0/1` (full TP по устройствам),
`LFM_BACKEND=cpu` (CPU-бэкенд вместо OpenCL),
`LFM_PREFILL_BUDGET` (prefill-токенов
на шаг планировщика, по умолчанию 12), `LFM_SCHED_LOG=1` (лог шагов планировщика).

## Ключевые решения (M0)

- **W4A8 на `dot(char4,char4)`**: int8-активации по группам 32, распаковка весов «на лету».
  Замерено 12.3 TOPS int8-dot против 2.18 TFLOPS fp16 — fp16-путь в rusticl бесполезен.
- **Subgroup-reduce вместо LDS-барьеров**, multi-row (`RGEMV`), адаптивный размер группы.
- **Load-time fusion**: ffn up+gate в один буфер (`ffn_ug`), один GEMV на слой.
- **Слияние ядер**: RMSNorm+квант, SiLU+квант, QK-norm+RoPE, gate внутри внимания.
- **Split-K внимание**: при контексте > 512 позиции делятся на чанки по 512
  (до 16), партиал-софтмакс затем складывается отдельным ядром; то же для prefill.
- **Paged KV**: страницы по 32 токена, `block_table` на последовательность,
  свободный список страниц, пер-последовательные conv-стейт и таблица; ядра KV и
  внимания адресуют `block_table[pos / 32] * 32 + pos % 32`.
- **Q4_K prepacked-копия** (`LFM_PREPACK=0` отключает): второй набор Q4-весов на
  +2.8% памяти, блок 148 байт с распакованными байтами скейлов/мин, ядро без
  `gsm_k4` и без ветвления; математика бит-в-бит, +5% на prefill 512. Compact-копия
  остаётся для decode-ядер.
- **Экономия на prefill**: logits (lm_head, 215 МБ) считаются только для последнего
  чанка промпта; батчевый GEMV держит x-векторы токенов в регистрах (PB=6 — оптимум).
- **Никогда не включать profiling-очередь** в рантайме: +80 мкс на диспатч.
- Формат весов остаётся GGUF-нативным (0% накладных на трафик).

## Лицензия

Код движка — по решению автора проекта. Веса модели — по лицензии LFM (см. LICENSE модели).
