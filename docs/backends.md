# Модульные бэкенды

Цель — один движок и несколько взаимозаменяемых исполнителей (CPU, OpenCL, Vulkan,
ROCm), с одинаковым поведением и общими проверками.

## Слои и зависимости

```
apps/*  (composition root: выбор бэкенда по CLI/env)
  └── application (порты, Generate, планировщик)
        └── domain (модель, сэмплинг, чат)
бэкенды реализуют lfm_app::InferenceBackend / TokenizerPort
  backend-cpu    pure Rust, AVX2 → NEON
  backend-opencl rusticl/PoCL (текущий)
  backend-vulkan compute shaders (SPIR-V)
  backend-rocm   HIP/ROCm
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
- `LFM_DEVICE` — индекс устройства у OpenCL/Vulkan/ROCm;
- `LFM_TIERS` — PP по слоям (OpenCL), `LFM_TP` — full TP (OpenCL);
- фичи Cargo: `backend-cpu` (default), `backend-opencl` (default), `backend-vulkan`,
  `backend-rocm` — чтобы сборка под ARM/NPU не тянула лишние зависимости.

## backend-cpu (готов, W4A8 + AVX2)

- Веса Q4_K/Q6_K **предпаковываются при загрузке**:
  - Q4: `u8 q[256]` (0..15) + 8 пар `d*scale`, `dmin*min` (f32);
  - Q6: `u8 q[256]` (0..63) + 16 `d*scale` (f32).
  Память ≈ 1.6 ГБ на 1.7B Q4 вместо 6.8 ГБ f32.
- Активации квантуются в i8 по группам 32 (`q`, `s`, суммы по 32 и 16).
- Ядра: `_mm256_maddubs_epi16` + `_mm256_madd_epi16` для Q4 (32 значения) и f32
  `_mm256_fmadd_ps`; runtime-детект `avx2/fma`, скалярный fallback (ARM — NEON позже).
- Матвек — Rayon по строкам; attention/conv/нормы — f32, KV непрерывный (без страниц),
  поэтому `free_pages` — оценка, радикс-кэша пока нет (`prefill_begin → 0`).
- Результат: `cpu-port` совпадает с GPU Q4 по greedy-ids на 8 токенах,
  ~6.9 tok/s на i5-10400 (memory-bound, 1.6 ГБ/токен).
- Дальше по CPU: AVX2 для Q6, блочный prefetch, f16-KV, отказ от аллокаций в forward,
  NEON-ветка (aarch64).

## backend-vulkan (каркас готов)

- `ash`, инстанс/физическое устройство (RADV), compute-очередь, host-visible
  SSBO-буферы, descriptor sets, push-constants.
- Шейдеры — GLSL, сборка в SPIR-V через `glslc` (build.rs); первое ядро
  `gemv_q4.comp` (Q4_K, W4A8) уже даёт **бит-в-бит** результат с CPU на реальном
  тензоре (`vulkan-gemv`, max_rel ≤ 1.6e-6), synthetic/real тесты в крейте.
- CLI: `vulkan-info`, `vulkan-gemv --rows N`.
- Дальше: Q6_K-GEMV, нормы/silu/quant, attention/KV, lm_head, сборка полного
  forward; затем переход с host-visible на device-local + staging, subgroup-редукции,
  несколько строк на workgroup и pipeline cache. Проверки — тот же parity-набор.

## backend-rocm

- HIP через `hip-sys`/`rocm-sys`; ядра — порт OpenCL-ядер (HIP C++) или SPIR-V.
- Память: hipMalloc/hipMemcpy, стримы; без P2P, обмен через host как в TP.
- Цель — сравнить с OpenCL/Vulkan на одной машине, выбрать быстрейший путь.

## Проверки и производительность

Единый набор для любого бэкенда:
- `*-gen --tokens N` — greedy-ids/текст;
- `*-cmp --len N --batch B` — батч == последовательный (cos/top-5);
- `*-multi --seqs N` — mixed batch;
- `Generate` через порты (`gpu-port`/`cpu-port`);
- сервер: `/v1/chat/completions`, `tools/loadtest.py`, `/metrics`.

Перф-метрики: decode/prefill tok/s на ctx 512/2048/8192, агрегат 4–6 стримов,
заполнение VRAM, время загрузки и сборки ядер.
