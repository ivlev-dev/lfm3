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
  `kv=q4` на таком устройстве отвергается при загрузке, `kvarn` работает
  (staging и скейлы — тоже float).
- Ядра, рассчитанные на подгруппу 64 (`attention_b_h2`, `attention_b_fa`,
  `attention_split_b_fa*`, attn4/flash), не включаются: у Intel подгруппа 16,
  а `reqd_sub_group_size` Rusticl игнорирует. Остаётся `attention_b_h` /
  `attention_split_b_h`.
- Q6-decode идёт через int8-ядра по `prepacked_i8`: `gemv_q6_k_i8` написан под
  подгруппу ≥ 32 и на 16 полосах считает неверно. Признак — `sub_group_size < 32`
  (а не отсутствие fp16: Intel Xe с fp16 и подгруппой 16 тоже требует обхода).
  Q4-decode (`gemv_q4_k_i8*`) корректен при любой подгруппе.

Intel Gen9 не виден в OpenCL по умолчанию: нужен Rusticl с `RUSTICL_ENABLE=iris`
(intel-compute-runtime и PoCL/Level Zero устройство не отдают).

## kvarn

`kv=kvarn` держит в staging только неполную страницу: по кольцу из 128 позиций
(`KV_STAGE`) на каждую последовательность, а не весь контекст. Закоммиченные
страницы читаются из кодов (4 бита на K, 2 бита на V) в `attention_*_qk` и в
batch-ядрах `attention_b_h_rot`/`attention_split_b_h_rot`; из staging читается
только хвост `pos >= full_end`. Батч не должен превышать 128 токенов на
последовательность (это ограничение `prefill_ntok`), иначе загрузка вернёт
ошибку.

Варианты `kvarn`: `kvarn` (=`kvarn_k4v2`, по умолчанию: K 4 бита, V 2 бита) и
`kvarn_k4v4` (V 4 бита — 16 уровней с нулевой точкой). `k4v4` — опция, не
замена: точность 0.998151 на 1024/64 и 0.998632 на 2048/128 против 0.991101 и
0.995183 у k4v2 (f16: 0.999375/0.998781), цена — +25 МБ на ctx 8192 и −2…−4%
prefill.

Память KV (24 слоя, ctx 8192, 4 KV-головы): `f16` — 403 МБ, `kvarn` — 143 МБ
при `prefill_ntok=128` (staging 50, коды 76, скейлы 17). Staging — кольцо на
`prefill_ntok` позиций, поэтому память зависит от размера чанка: 64 → 118 МБ,
32 → **106 МБ** (prefill при этом −7…−11%), decode не меняется. На ctx 2048 —
101 и 74 МБ. Цена — скорость
prefill: коды декодируются в batch-внимании; после переноса flash-структуры
(`attention_split_b_fa_qk`, 4 головы × 16 полос, вращение в регистрах) на AMD
это −10% на 512 и −31% на 8192; decode на длинном контексте −5%. Остаток
разрыва — стоимость декодирования кодов K/V (1 загрузка + 3 ALU на канал
против одной векторной загрузки f16). `kvarn` не поддерживается вместе с tensor parallel.

## Проверки и производительность

Единый набор для любого бэкенда:
- `*-gen --tokens N` — greedy-ids/текст;
- `*-cmp --len N --batch B` — батч == последовательный (cos/top-5);
- `*-multi --seqs N` — mixed batch;
- `Generate` через порты (`gpu-port`);
- сервер: `/v1/chat/completions`, `tools/loadtest.py`, `/metrics`.

Перф-метрики: decode/prefill tok/s на ctx 512/2048/8192, агрегат 4–6 стримов,
заполнение VRAM, время загрузки и сборки ядер.

Регрессионный прогон корректности — `tools/bench_check.sh` (`just bench-check`,
`bench-check-igpu`, `bench-check-server`). Пороги cos зафиксированы по замерам и
проверяются именно split-пути (`gpu-cmp` при len > 512), где ошибки нормировки
kvarn не видны на коротких контекстах:

| проверка | порог | AMD | Intel |
|---|---|---|---|
| `gpu-check` f16 | 0.9975 | 0.998532 | 0.998242 |
| `gpu-check` kvarn | 0.9965 | 0.998435 | 0.998372 |
| `gpu-cmp` f16 1024/64 | 0.9980 | 0.999375 | — |
| `gpu-cmp` kvarn 1024/64 | 0.9880 | 0.990922 | — |
| `gpu-cmp` f16 2048/128 | 0.9970 | 0.998781 | — |
| `gpu-cmp` kvarn 2048/128 | 0.9920 | 0.995045 | — |
| `gpu-gen` (f16/kvarn) | текст «Paris…» | ok | ok |

Для iGPU split-пути проверяются отдельным медленным профилем
`bench-check-igpu-full` (единичный проход 1024 токенов занимает минуты).
