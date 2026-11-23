# Контракт ядер LFM3: int4 + Hadamard, HVX и HMX

Новый слой ядер `myelin/dsp`. Источник семантики — OpenCL-эталон
(`crates/backend-opencl`: `kernels/decode.cl`, `kernels/batch.cl`,
`kernels/kvarn.cl`, `kernels/kvq4.cl`, `src/prepack.rs`) и Rust-эталоны
`crates/lfm-npu-ref` (`abi.rs`, `layout.rs`, `kvarn.rs`) и
`myelin/crates/myelin-pack/src/kvarn.rs`.

Старый слой (`hvx/`, `hmx/`, `conv/`, `quant/`) удалён целиком. Ни один файл
оттуда не используется как зависимость; сохранён только стенд `dsp/sim/`.

## 1. Форматы

### 1.1 Веса: int4 / блок 32

Матрица `N x K`, `K % 32 == 0`. Образ:

```
[ scales: N * G * 2 байта     ][ pad до 128 ][ planes: N * K/2 байт ]
G = K / 32
```

* `scales[n*G + g]` — f16, шаг блока (`amax / 7`).
* `planes[n*K/2 + k/2]` — нибблы `q + 8 ∈ [0,15]`; младший ниббл — чётный
  `k`, старший — нечётный.
* `q ∈ [-7,7]`, округление half-away-from-zero.
* Плоскости выровнены на 128 байт, шаг строки `K/2` (для `K % 256 == 0`
  выравнивание сохраняется).

Перед квантованием каждая строка весов поворачивается блоками по 128:
`W' = W R`, `R = H128 D / sqrt(128)`. Если `K` не кратно 128, хвост
квантуется без поворота, метаданные пака это фиксируют.

### 1.2 Активации: int8 / блок 32

Ровно `quantize_q8` из `decode.cl:78-121`:

```
amax  = max|x| по 32 каналам
scale = amax / 127
q     = rte(x * 127 / amax), q ∈ [-127,127]
нулевая группа -> scale = 0, q = 0
```

Активация поворачивается `R` поблочно по 128 **до** квантования.
Скалярное произведение не меняется: `<W R, R x> = <W, x>`.

### 1.3 HMX zero-point

`activation.ub` читает беззнаковые байты:

```
u = q + 128
dot(w, q) = dot(w, u) - 128 * sum(w)
```

`128 * sum(w)` кладётся в int32-слово input-bias (32 младших слова — scale,
32 старших — bias). Компенсация точная.

### 1.4 KV: kvarn K4/V2 и K4/V4

Побитово как `myelin/crates/myelin-pack/src/kvarn.rs`. Страница — 32 токена
x 128 измерений на голову, little endian:

```
kcode 32*64 B   нибблы (q+8) | ((q+8) << 4), младший = чётное измерение
ka    128 f16   per-channel K scale
kc     32 f16   per-token K scale
vcode 32*32 B (k4v2) или 32*64 B (k4v4)
vb     32 f16   per-token V step
vbz    32 f16   per-token V zero point
vc    128 f16   per-channel V scale
```

Размеры страницы на голову: 15 104 Б (k4v2), 19 200 Б (k4v4).
`K` и `V` пишутся уже повёрнутыми `R`; декод возвращает повёрнутый базис,
обратный поворот делается один раз на выходе attention.

## 2. Ротация

`R = (1/sqrt(d)) H_d D`, `d = 128` для весов, активаций и KV (H32 — только
для диагностики). `D` — знаки из `LFM_TQ_SIGN`:

```
0x9E3779B9 0x85EBCA6B 0xC2B2AE35 0x27D4EB2F
```

`R^T R = I`. Веса поворачиваются офлайн, активации — на лету в начале слоя,
KV — при записи и чтении. Один и тот же `R` на все три места.

Attention считается в повёрнутом базисе: `Q` поворачивается до раскладки по
головам, `K`/`V` пишутся в kvarn уже повёрнутыми, выход attention
поворачивается обратно (`lfm_rot_inv_blocks_f32`) один раз перед выходной
проекцией. Это убирает обратный поворот из горячего цикла декода.

`lfm_kvarn_commit_work(..., float *scratch)` требует scratch на
`32 * 128` f32 (16 КиБ); обёртка `lfm_kvarn_commit` берёт его из кучи и
предназначена для хоста. На DSP scratch выделяется один раз на PD.

## 3. Paged KV

* Страница = 32 токена (`PAGE_TOKENS`).
* `max_pages = ceil(ctx / 32) * kv_pool`.
* Таблица страниц: `max_seqs * pt_stride * 4` байт, `pt_stride = max_pages`.
* Логический адрес: `page = pos >> 5`, `slot = (tbl[page] << 5) | (pos & 31)`.
* KV-арена: `kv_layers * max_pages * page_bytes(format)`.
* Окно sliding = 256 токенов; `window = 0` — полный контекст.

## 4. ABI

Структуры `#[repr(C)]`, little endian, смещения в элементах арен:

```c
typedef struct { uint32_t token, seq, pos, ntok, kv_off, tbl_off; } lfm_row;
typedef struct {
    uint32_t n_rows, ntok, window, flags, max_rows, reserved;
} lfm_step;
typedef struct { uint32_t n_rows, ntok, logits_off, status; } lfm_out;
```

`FLAG_PREFILL = 1`. `kv_off = seq * kv_stage` (токены KV),
`tbl_off = seq * pt_stride` (записи `u32`).

Методы DSP (один IDL `lfmgemv.idl`): `init`, `load_buf`, `set_scale`,
`exec`, `kv_commit`, `kv_read`, `reset_state`, `probe`.

`exec` принимает список команд на слой и массив строк, исполняет
continuous batching: строки decode (ntok == 1) идут GEMV, строки prefill
(ntok > 1) идут HMX GEMM по чанкам.

## 5. Разделение HVX / HMX

| путь | M | блок | ядро |
|---|---|---|---|
| decode | 1..8 | HVX 128 B | `hvx/gemv_i4.c` |
| prefill | 32/64/128 | HMX `weight.n` | `hmx/gemm_i4.c` |
| точный fallback | 32/64/128 | HMX int8 | `hmx/gemm_i8.c` |

HVX: `vrmpyacc(Vub, Vb)` — беззнаковый вес, знаковый активатор; компенсация
`-8 * sum(a)` считается один раз на вектор активации. `l2fetch` с окном
8 КБ, два HVX-контекста на PD.

HMX: `weight.n` 512 Б на 32 K x 32 N, `activation.ub`, bias-блок
`scale[32] | input_bias[32]`.

## 6. Гейты

1. GEMV: cos >= 0.9999 против f32-оракула, 0 расхождений с CPU-int8.
   Измерено на host: `max_abs = 0`, `cos_dq = 1.0`, `cos_int4` 0.9934–0.9963.
2. HMX int8: `err = 0` побитово; int4/32 + H128: cos >= 0.995 на случайных
   гауссовых данных (предел 4 бит), реальные веса — гейт качества текста.
   Измерено на host-эмуляции: int8 `err = 0` (M=32/64/128), int4
   `cos = 0.995229` (M=32, N=256, K=2048), prefill == gemm.
3. kvarn: байт-в-байт с `myelin-pack/src/kvarn.rs`. Выполнено для K4V2/K4V4
   (encode и decode, `tests/test_common.c`).
4. Полоса HVX >= 22 ГБ/с; HMX compute ~5 TMAC/s; пустой RPC <= 0.25 мс.
   На host переносимый путь даёт 0.71–0.79 ГБ/с; на устройстве не измерялось.
5. Стенды: `qemu-hexagon` (HVX `exec`), `hexagon-sim` (HMX, `--mhmx 1`).
   Текущее состояние стендов и известные зависания — в [STATUS.md](STATUS.md).
6. Качество текста LFM3-1.7B на реальных весах — отдельный гейт следующей
   волны (интеграция), не ядра.
