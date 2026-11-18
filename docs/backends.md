# Модульные бэкенды

Цель — один движок и несколько взаимозаменяемых исполнителей (CPU, OpenCL),
с одинаковым поведением и общими проверками. Vulkan/ROCm исключены.

## Слои и зависимости

```
apps/*  (composition root: выбор бэкенда по CLI/env)
  └── application (порты, Generate, планировщик)
        └── domain (модель, сэмплинг, чат)
бэкенды реализуют lfm_app::InferenceBackend / TokenizerPort
  backend-cpu    pure Rust, AVX2 → NEON
  backend-opencl rusticl/PoCL (текущий)
общие крейты: domain, application, infra-gguf, infra-quant, infra-tokenizer
```

Правило: `application`/`domain` не знают про инфру; конкретный бэкенд подставляет
`main.rs` (`BackendLoader`), поэтому сервер и TUI зависят только от портов.

## Контракт

`InferenceBackend` (см. `crates/application/src/ports.rs`):
`create/free_sequence`, `prefill`, `prefill_begin/finish`, `step`, `logits`,
`logits_row`, `max_ctx`, `free_pages`, `cache_stats`.

`TokenizerPort`: `encode`, `decode`, `eos_id`.

Требования к каждому бэкенду:
- `Generate` даёт те же greedy-ids, что эталон, на фиксированных промптах;
- `step` поддерживает несколько последовательностей (mixed batch);
- `prefill_begin/finish` — крючки для префикс-кэша; бэкенд без кэша возвращает 0;
- `free_pages` используется admission-контролем планировщика.

## Выбор бэкенда

Env/CLI (composition root):
- `LFM_BACKEND=cpu|opencl|vulkan|rocm` (default opencl);
- `LFM_DEVICE` — индекс устройства у OpenCL;
- `LFM_TIERS` — PP по слоям (OpenCL), `LFM_TP` — full TP (OpenCL);
- фичи Cargo: `backend-cpu` (default), `backend-opencl` (default); Vulkan/ROCm
  исключены.

## backend-cpu (готов, W4A8 + AVX2)

- Веса Q4_K/Q6_K **предпаковываются при загрузке**:
  - Q4: нибблы `u8 q[128]` (значение `i` — младший ниббл байта `i`, `i+128` — старший)
    + 8 пар `d*scale`, `dmin*min` (f32). 192 Б на 256 значений;
  - Q6: `u8 q[256]` (0..63) + 16 `d*scale` (f32). 320 Б на 256 значений.
  Память ≈ 1.2 ГБ на 1.7B Q4.
- Активации квантуются в i8 по группам 32 (`q`, `s`, суммы по 32 и 16).
- Ядра: `_mm256_maddubs_epi16` + `_mm256_madd_epi16`, распаковка нибблов
  `and`/`srli`; аккумулятор — 8 f32-доржек на строку, горизонтальная редукция
  **один раз на строку**; скалярные `mn*sum` собраны в отдельный аккумулятор
  (в prefill — векторный по 8 группам). runtime-детект `avx2/fma` кешируется,
  есть скалярный fallback (`LFM_CPU_SCALAR=1`).
- **Prefill батчевый**: `forward_batch` обрабатывает 8 токенов за проход
  (`matmul_batch`), веса читаются раз на тайл; Q6-ветка считает по 32 значения
  за итерацию (по 16-групповым скейлам). attention/conv/нормы — по токену, KV общий.
- Decode (`matvec`) — та же векторизация, Rayon по строкам; attention/conv/нормы f32,
  KV непрерывный (без страниц), радикс-кэша нет (`prefill_begin → 0`).
- Результат на i5-10400: prefill 512 ≈ 54.5 tok/s, decode ≈ 15.7 tok/s (llama.cpp
  CPU: 141.5 / 23.3); `cpu-port` совпадает с GPU Q4 по greedy-ids на 8 токенах.
- Дальше по CPU: prefetch/стриминг весов, f16-KV, отказ от аллокаций в forward,
  NEON-ветка (aarch64).

## Vulkan/ROCm

Исключены решением от 18.11.2026: остаются CPU и OpenCL, цель — обогнать
llama.cpp по prefill и decode.

## Проверки и производительность

Единый набор для любого бэкенда:
- `*-gen --tokens N` — greedy-ids/текст;
- `*-cmp --len N --batch B` — батч == последовательный (cos/top-5);
- `*-multi --seqs N` — mixed batch;
- `Generate` через порты (`gpu-port`/`cpu-port`);
- сервер: `/v1/chat/completions`, `tools/loadtest.py`, `/metrics`.

Перф-метрики: decode/prefill tok/s на ctx 512/2048/8192, агрегат 4–6 стримов,
заполнение VRAM, время загрузки и сборки ядер.
