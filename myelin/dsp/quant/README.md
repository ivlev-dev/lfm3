# `dsp/quant` — активации, zero-point, Hadamard

Общий модуль квантования для DSP-ядер (HVX и HMX). Ничего не линкует сам:
`quant.h` — header-only, компилируется и `hexagon-clang` (DSP), и системным
`cc` (хостовые утилиты/исследования).

## Что внутри

| Блок | Функция | Назначение |
|---|---|---|
| Эталон активаций | `lfm_quant_i8_group32` | per-group-32 absmax, симметрично `q ∈ [-127,127]`, `scale = amax/127` f32, округление **RTE**, клиппинга нет, нулевая группа → `sc=0,q=0`. Точно как `quantize_q8` в `crates/backend-opencl/kernels/decode.cl`. |
| Zero-point | `lfm_q_to_u8`, `lfm_dot_u8_zp` | `u = q + 128` для HMX `activation.ub`; тождество `dot(w,u) - 128·Σw == dot(w,q)` — ровно то, что делает int32-слово bias в HMX. |
| Ротация | `lfm_rot_fwd_f32`, `lfm_rot_inv_f32`, `lfm_rot_blocks_f32` | `R = (1/√d)·H_d·D`, `d = 32` или `128`; `D` — те же знаки, что `LFM_TQ_SIGN` в `kvq4.cl`. `R^T R = I`. |
| Метаданные пака | `lfm_rot_meta`, `lfm_rot_meta_of` | какой `R` запечён в веса (`size`, `group`, `scale`, `sign0`). |
| int4/32 (HVX) | `lfm_pack_i4_row` | строка f32 → int4 `[-7,7]`, `scale=amax/7`, half-away-from-zero, ниббл `q+8`, low = чётный k. |

## Ключевые инварианты

1. **HVX активации знаковые.** `vrmpy(Vub, Vb)` = беззнаковый вес × знаковый
   активатор; int4-веса уже в offset-8, компенсация `-8·Σa` на строку. Поэтому
   на HVX zero-point **не нужен** (`lfmgemv_impl.c`, `gemv_rows_m1*`).
2. **HMX активации беззнаковые.** `activation.ub` читает 0..255, поэтому
   `u = q + 128`, а `-128·Σw[n]` кладётся в **верхнее int32-слово** bias-блока
   (`bias = mxmem2(...)`, 32 младших слова — scale, 32 старших — input bias).
   Компенсация точная и бесплатная.
3. **Ротация**: веса и активации поворачиваются одним и тем же `R` (офлайн —
   веса, на лету — активации); скалярное произведение не меняется, но per-32
   квантование видит гауссизированный блок. Формат — **int4/32**, Q4_K не
   используется (отменён).

## Утилиты

```
host/rotstudy.c   — качество int4/32 + int8/32: без ротации / H32 / H128
host/mkcase.c     — GGUF → (поворот) → int4/32 HMX-блоб + метаданные + эталон
hmxsim/probe_bias.c  — семантика bias-слов (int32 input bias)
hmxsim/probe_round.c — округление/насыщение int8-сторa
hmxsim/hmxtest.c     — полный HMX GEMM с uint8(zp) активациями
```

Сборка/запуск — в `host` (cc) и `hmxsim` (`./build.sh && ./run.sh ...`).
Результаты и методика — `myelin/research/REPORT25.md`.
