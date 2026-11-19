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
- бэкенд один — OpenCL;
- `LFM_DEVICE` — индекс устройства у OpenCL;
- `LFM_TIERS` — PP по слоям (OpenCL), `LFM_TP` — full TP (OpenCL);
- Cargo-фич для выбора бэкенда нет: `infra-cpu` — эталонная CPU-реализация для
  сверки (`lfm-bench gpu-check`, `ref-gen`), серверный инференс идёт через
  `backend-opencl`; Vulkan/ROCm
  исключены.

## CPU и ARMv7

Вынесены в отдельные проекты (в этом репозитории остаётся только OpenCL):
CPU-бэкенд W4A8/AVX2 и ARMv7 (NEON + OpenCL 1.1 для Adreno 330).

## Vulkan/ROCm

Исключены решением от 18.11.2026: остаются CPU и OpenCL, цель — обогнать
llama.cpp по prefill и decode.

## Устройства без `cl_khr_fp16` (Intel iGPU)

Бэкенд работает и на устройстве без fp16 (проверено на Intel UHD 630 через
Rusticl). Отличия включаются автоматически по `caps.fp16`, отдельных флагов нет:

- KV-кэш хранится во `float` (`KV_T`/`-DLFM_F32_KV`), заголовки блоков
  (`d`/`dmin`) читаются из байтов и конвертируются вручную (`lfm_h2f`).
  Тип `kv=f16` в конфиге остаётся, меняется только представление в памяти;
  `kv=q4`/`kvarn` на таком устройстве отвергаются при загрузке.
- Ядра, рассчитанные на подгруппу 64 (`attention_b_h2`, `attention_b_fa`,
  `attention_split_b_fa*`, attn4/flash), не включаются: у Intel подгруппа 16,
  а `reqd_sub_group_size` Rusticl игнорирует. Остаётся `attention_b_h` /
  `attention_split_b_h`.
- Q6-decode идёт через int8-ядро `gemm_i8_gz` по `prepacked_i8`: `gemv_q6_k_i8`
  написан под подгруппу ≥ 32 и на 16 полосах считает неверно. Q4-decode
  (`gemv_q4_k_i8*`) корректен.

Intel Gen9 не виден в OpenCL по умолчанию: нужен Rusticl с `RUSTICL_ENABLE=iris`
(intel-compute-runtime и PoCL/Level Zero устройство не отдают).

## Проверки и производительность

Единый набор для любого бэкенда:
- `*-gen --tokens N` — greedy-ids/текст;
- `*-cmp --len N --batch B` — батч == последовательный (cos/top-5);
- `*-multi --seqs N` — mixed batch;
- `Generate` через порты (`gpu-port`);
- сервер: `/v1/chat/completions`, `tools/loadtest.py`, `/metrics`.

Перф-метрики: decode/prefill tok/s на ctx 512/2048/8192, агрегат 4–6 стримов,
заполнение VRAM, время загрузки и сборки ядер.
