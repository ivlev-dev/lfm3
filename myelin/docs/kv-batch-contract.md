# Контракт KV-кэша, радикс-кэша и батчера (v1)

Документ фиксирует границу между **хостом** (Rust: сессии, планировщик,
radix, аллокация страниц) и **DSP** (HVX/HMX: paged attention, запись KV) так,
чтобы DSP-ядра и Rust-планировщик можно было писать параллельно и сверять
независимо.

Источники истины, с которыми документ обязан согласовываться:

* [kernel-contract.md](kernel-contract.md) — что реально делает эталон
  `crates/backend-opencl` (формулы, инварианты, §5.2 — форматы K/V);
* [paged-kv.md](paged-kv.md) — страницы, staging, radix, инварианты арен;
* [architecture.md](architecture.md) §4 — разделение host/DSP;
* [decode-plan.md](decode-plan.md) — бюджет RPC и команд на токен;
* [measurements.md](measurements.md) §4, §8 — замеры attention и RPC.

Метки: **[код]** — следует из исходников (можно пересчитать), **[замер]** —
цифра из `measurements.md`/`decode-plan.md`/`benchmarks/machine.md`,
**[оценка]** — вывод, требует подтверждения на устройстве.

---

## 0. Термины и область

| термин | значение |
|---|---|
| страница (page) | 32 токена KV одной последовательности; единица аллокации и шаринга |
| слот (slot) | один токен внутри страницы, `0..31` |
| физическая страница | индекс страницы в общем пуле; именно он лежит в block table |
| block table | массив `u32`, по записи на страницу; `tbl[pos >> 5] = phys` |
| `full_end` | число токенов строки, уже закодированных в страницы; хвост `[full_end, len)` читается из staging |
| строка (row) | одна последовательность в одном шаге; в decode `ntok = 1` |
| TQ | «turbo-quant»-схема эталона: Hadamard + знаки + кодбук (`q4`, алиас `k4v4`) |
| kvarn | формат эталона: K 4 бита, V 2/4 бита, масштабы на канал и токен, тоже с Hadamard |
| stand | текущий DSP-стенд `myelin/dsp/hvx` + `myelin-pack kvpage` (свой формат, не из эталона) |

Область: KV, paged attention, радикс, батчер. Веса, FFN, conv, токенизатор и
сэмплинг сюда не входят, кроме стыков.

---

## 1. Что есть в Myelin сейчас

### 1.1 Хост: страницы и таблица

`myelin-core` держит контракт [`KvCache`](../crates/myelin-core/src/kv.rs:15) и
реализацию [`PagedKvCache`](../crates/myelin-core/src/kv.rs:128):

* страница — **32 токена** ([`PAGE_TOKENS`](../crates/myelin-core/src/kv.rs:12));
* `PagePool` — два плоских массива `f32` (`keys`, `values`) и free-list
  ([kv.rs:44](../crates/myelin-core/src/kv.rs:44));
* `PagedKvCache::new(kv_dim, capacity)` создаёт **свой** пул на
  `ceil(capacity/32)` страниц ([kv.rs:138](../crates/myelin-core/src/kv.rs:138));
* адрес токена: `page = table[pos/32]`, `offset = pos % 32`,
  `start = page*kv_dim*32 + offset*kv_dim` ([kv.rs:181](../crates/myelin-core/src/kv.rs:181)).

`Session` создаёт по одному кэшу **на каждый attention-слой**
([session.rs:147](../crates/myelin-core/src/session.rs:147)): пул приватный для
(последовательность, слой), кросс-последовательного шаринга нет.

### 1.2 Резидентные образы DSP (формат стенда)

`AttentionStage` ([dsp.rs:290](../crates/myelin-core/src/backend/dsp.rs:290)):

| поле | смысл | размер на слой |
|---|---|---|
| `keys` (`hk`) | K: int4 offset-8, **128 Б на токен** (64 Б кодов + 64 Б нулей) | `4 · slots · 128` Б |
| `key_scales` (`hkt`) | `kt`: один Q16 `i32` на (голова, токен) | `4 · slots · 4` Б |
| `values` (`hv`) | V: int8, 128 Б на токен | `4 · slots · 128` Б |
| `value_scales` (`hvs`) | не используется, передаётся `0` | 0 |

`slots = cache_pages · 32`, `cache_pages = ceil(ctx/32)`
([model_pack.rs:439](../crates/myelin-npu/src/model_pack.rs:439)); раскладка —
**head-major**: голова владеет непрерывным диапазоном `slot ∈ [0, slots)`.

Точный формат — [`myelin-pack/kvpage.rs`](../crates/myelin-pack/src/kvpage.rs:1)
(`KvLayout::Row`):

```text
K:  byte[ (head*slots + slot)*128 + lane ] = (q_even+8) | ((q_odd+8) << 4)
    lane = dim/2, q = round(k / s_k) ∈ [-7, 7], s_k = max_d|k_d| / 7
    байты 64..127 строки — нули (паддинг под 128-Б вектор)
V:  byte[ (head*slots + slot)*128 + dim ] = clamp(round(v / s_v[d]), -127, 127)
    s_v — статический поканальный калибровочный масштаб слоя (не по данным страницы)
kt: i32[ head*slots + slot ] = round(s_k · 65536 / (q_scale · sqrt(128)))
```

Это **не** формат эталона: нет вращения, нет кодбука, V с одним статическим
масштабом на канал, K с per-token масштабом. В таблице форматов
[kernel-contract.md §5.2](kernel-contract.md) его нет (там только `f16`, `q4`,
`kvarn_*`); формат стенда описан здесь и в
[`myelin-pack/kvpage.rs`](../crates/myelin-pack/src/kvpage.rs:1).

Размеры **[код]**:

| секция на голову-страницу | байт |
|---|---|
| K (32 × 128) | 4096 |
| kt (32 × 4) | 128 |
| V (32 × 128) | 4096 |
| **итого** | **8320** |

Страница (4 KV-головы) — **33 280 Б**, 8.125 бит/элемент; на токен на слой —
1040 Б. При ctx 2048 и 11 слоях — 23.4 МБ; с окнами (6 full + 5×256) — 14.1 МБ.

### 1.3 Как пишется: `kv_write`

Хост считает q/k/v на DSP, затем **на хосте** делает q/k RMSNorm, rope,
квантует K/V и шлёт один RPC ([dsp.rs:1813](../crates/myelin-core/src/backend/dsp.rs:1813)):

```text
page = position / 32
slot = table[page] * 32 + position % 32             // глобальный слот, dsp.rs:1838
row  = head * slots + slot                            // impl.c:2396
kt   = round(s_k · 65536 / (q_scale · sqrt(128)))     // dsp.rs:1845
K[row*128 .. +64]   = nibbles(q + 8)                  // impl.c:2400
K[row*128+64..+128] = 0                               // impl.c:2401
kt[head*slots+slot] = Q16                             // impl.c:2402
V[row*128 .. +128]  = int8(v / s_v[d])                // impl.c:2403
```

`table` — та же identity-таблица `0..pages`, что уходит в `ATTN` (§2.1), поэтому
`kv_write` и `attn_core` адресуют один и тот же слот. Слот вычисляет
[`kv_slot`](../crates/myelin-core/src/backend/dsp.rs:2177), а не `position % 32`
(см. §9).

Ограничения C-функции ([impl.c:2389-2394](../dsp/hvx/lfmgemv_impl.c:2389)):
`1 ≤ slots`, `0 ≤ slot < slots`; в **старом** скелете дополнительно
`slot < 32`, из-за чего глобальный слот ≥ 32 отклоняется
(`AEE_EBADPARM`) — это единственное, что осталось снять в DSP (§9);
`kLen % 64 == 0`, `ktLen % 4 == 0`, `vLen % 128 == 0`;
`heads = kLen/64 = ktLen/4 = vLen/128`.

Масштабы V (512 значений `kv_heads·head_dim`) живут на хосте в
`VQuant::Channel` ([dsp.rs:426-471](../crates/myelin-core/src/backend/dsp.rs:426))
и применяются **после** attention.

### 1.4 Как читается: attention стенда

`attn_core` ([impl.c:2138](../dsp/hvx/lfmgemv_impl.c:2138)): страйды выводятся
из размеров буферов ([impl.c:2161-2165](../dsp/hvx/lfmgemv_impl.c:2161)):

```text
sk = k->wbytes/4     (128 Б на слот),  sv = v->wbytes/4,  skt = kt->wbytes/4
slot = (tbl[tok >> 5] << 5) | (tok & 31)              // impl.c:1890
dot  = Σ_d q8[d]·(code[d] − 8)                        // 2 vrmpyacc + reduce
sc_q8 = (dot << 8) · kt_q16 >> 16                     // mulq16, impl.c:1936
```

Softmax **двухпроходный и целочисленный**: scores материализуются в
`int32[16][n]` ([impl.c:2180](../dsp/hvx/lfmgemv_impl.c:2180)), затем
`mx = max`, `p_t = exp_q7(sc−mx) ∈ [0,128]`, `L = Σp`
([impl.c:1972-1974](../dsp/hvx/lfmgemv_impl.c:1972)). PV — int32-аккумулятор
`acc[d] = Σ p_t·v8[t][d]`, `|acc| ≤ n·127·127` ([impl.c:1976-1987](../dsp/hvx/lfmgemv_impl.c:1976)).
Нормализация: `inv = 2^28/L` (int64, раз на голову),
`out_q12[d] = (acc·inv) >> 16` ([impl.c:2255-2265](../dsp/hvx/lfmgemv_impl.c:2255)).
Хост: `real = q12 · attn_out_scale · s_v[kh·128+d] · gate[qh]`
([dsp.rs:1766-1786](../crates/myelin-core/src/backend/dsp.rs:1766)).

Следствие: `s_v` **обязан быть постоянным по токенам**, иначе его нельзя
вынести за сумму по `t`. Это ограничение стенда (эталон так не делает, §3).

### 1.5 Хост-зеркало (сейчас — с данными)

`attention_layer_full` после `kv_write` делает `kv.append(&key, &value)`
([dsp.rs:1716](../crates/myelin-core/src/backend/dsp.rs:1716)) и хранит f32
K/V: `11 · 512 · 4 = 22.5 КБ/токен` (46 МБ при ctx 2048), ×M при батче.
Планировщику нужна только длина (§6.3 п. 5).

---

## 2. Paged attention: таблица, varlen, ограничения

### 2.1 Block table

Формат (`lfm-npu-ref`, эталон): `u32`, запись на страницу,
`tbl[pos>>5] = phys`; срез последовательности
`tbl_all[seq·pt_stride .. (seq+1)·pt_stride]`, `pt_stride = pool_pages`
([abi.rs:18-33](../../crates/lfm-npu-ref/src/abi.rs:18)).

В Myelin таблица **фиктивная**: и legacy-, и полный путь строят `[0,1,2,…]` на
`pages = pos/32 + 1` ([dsp.rs:1330](../crates/myelin-core/src/backend/dsp.rs:1330),
[dsp.rs:1720](../crates/myelin-core/src/backend/dsp.rs:1720)). Ядро умеет
произвольные `phys` (`tbl[tok>>5]`), но ни запись, ни таблица это не используют.
`tbl_len = pages` меняется каждые 32 токена — источник дрейфа `exec`-списков
([decode-plan.md §5/O2](decode-plan.md)).

### 2.2 Ограничения ядра стенда

| ограничение | значение | где |
|---|---|---|
| страница | 32 токена, `pos>>5` | `ATTN_PAGE`, impl.c:598 |
| q-головы | ровно 16 | impl.c:595 |
| KV-головы | ровно 4 (GQA 4:1) | impl.c:596 |
| head_dim | ровно 128 | impl.c:597 |
| K/V на токен | 128 Б / 128 Б | model_pack.rs:452,454 |
| q | `16·128` int8 | impl.c:2151 |
| выход | `16·128` int32 Q12 | impl.c:2152 |
| `pos`, `seq` | `0 ≤ pos < seq` | impl.c:2153 |
| `tbl_len` | `1..2^20`, `(pos>>5) < tbl_len` | impl.c:2154, 3509 |
| `slots`/`slot` | `slot` — глобальный, `0 ≤ slot < slots`; старый скелет ещё требует `slot < 32` | impl.c:2389 |
| команды в списке | 256 | `LFM_CMD_MAX`, dsp.rs:618 |
| буферы / арены | 256 / 160 | impl.c:40,43 |
| `map_arena` | кратен 128 Б | executor.rs:205 |

Team: 4 KV-головы делятся между контекстами (`g_threads = 2` замерено) —
полезная параллельность attention ≤ 4 задач; при M строках — `4·M`.

### 2.3 Что сегодня означают `pos/seq/window`

* `pos` — абсолютная позиция токена;
* `seq` — **не** id: хост передаёт `pos + 1`
  ([dsp.rs:1311](../crates/myelin-core/src/backend/dsp.rs:1311),
  [dsp.rs:1732](../crates/myelin-core/src/backend/dsp.rs:1732)), ядро
  проверяет `pos < seq`. Фактически это длина контекста;
* `window = 0` — полное окно, иначе `tok0 = max(0, pos+1-window)`
  ([impl.c:2169-2171](../dsp/hvx/lfmgemv_impl.c:2169)).

### 2.4 Varlen уже есть в эталоне

Батчевые ядра эталона принимают строки именно так, как нужно контракту
([kvq4.cl:399-421](../../crates/backend-opencl/kernels/kvq4.cl:399),
[kvarn.cl:559-587](../../crates/backend-opencl/kernels/kvarn.cl:559)):

```text
t = get_group_id(0)/hg          — номер строки
pos = pos_b[t]                  — абсолютная позиция токена строки
seq = seq_b[t]                  — слот последовательности (для staging-кольца)
toff = tbl_off[t]               — срез таблицы (в u32-записях)
len = pos + 1;  start = max(0, len − window)
full_end = (len/32)*32          — всё, что < full_end, читается из кодов
slot = (tbl[toff + (pos>>5)] << 5) | (pos & 31)
```

То есть varlen-контракт **не надо изобретать**: `Row {token, seq, pos, ntok,
kv_off, tbl_off}` и `StepDesc` из [abi.rs:18-68](../../crates/lfm-npu-ref/src/abi.rs:18)
плюс `pos_b/seq_b/tbl_off_b` — это уже работающая схема (8–32 строк на шаг,
`max_seqs` до 32 в эталоне, [options.rs:114-119](../../crates/backend-opencl/src/options.rs:114)).

Чего нет в стенде и что нужно добавить (список изменений для DSP-ядра):

1. **офсет таблицы на строку** (`tbl_off[t]`) вместо одной таблицы на вызов;
2. **`pos`/`window`/`full_end` на строку** (сейчас всё в `rt[4]`, общее);
3. **`seq` как id слота staging** (сейчас `seq = pos+1` и ни на что не влияет);
4. **несколько q/out на вызов** (сейчас q ровно `16·128`, out ровно `16·128·4`);
5. **проверка `(pos>>5) < tbl_len`** на строку;
6. фиксированный `tbl_len = pt_stride` (снимает дрейф списков) и таблица
   данными (scratch/`act`), а не в байтах команды.

Инварианты varlen-шага:

```
rows.len() ≤ max_rows (≤ 64 по контракту; эталон держит 8–32)
Σ rows[i].ntok ≤ max_ntok            (= M в decode)
∀ i: pos_i + ntok_i ≤ ctx
∀ i: ceil((pos_i+ntok_i)/32) ≤ pt_stride
∀ i: 0 ≤ tbl[toff_i + (pos_i>>5)] < pool_pages
full_end_i = floor((pos_i+1)/32)·32   (для kvarn; для q4/stand не нужен)
```

---

## 3. Форматы KV эталона (`backend-opencl`)

### 3.0 Общая механика квантованных форматов

**Ротация.** Все квантованные форматы (`q4`, `kvarn_*`) хранят K и V в
повёрнутом базисе

```text
R = (1/√128) · H · D,   D = diag(±1) по LFM_TQ_SIGN[4] u32
R·Rᵀ = I  (ортогонально: s²·128 = 1)
```

* знаки `LFM_TQ_SIGN = {0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F}`
  ([kvq4.cl:10](../../crates/backend-opencl/kernels/kvq4.cl:10)),
  `lfm_tq_sign(d) = (bit d) ? −1 : +1` ([kvq4.cl:12-14](../../crates/backend-opencl/kernels/kvq4.cl:12));
* Адамар 128 ([kvq4.cl:16-28](../../crates/backend-opencl/kernels/kvq4.cl:16)),
  нормировка `0.0883883476 = 1/√128` ([kvq4.cl:35](../../crates/backend-opencl/kernels/kvq4.cl:35));
* `rot_fwd = s·H·D` ([kvq4.cl:30-38](../../crates/backend-opencl/kernels/kvq4.cl:30)),
  `rot_inv = s·D·H` ([kvq4.cl:40-46](../../crates/backend-opencl/kernels/kvq4.cl:40));
* применяется: к K и V при записи (`kv_store_*_rot`,
  [kvarn.cl:6-29](../../crates/backend-opencl/kernels/kvarn.cl:6)), к `q` перед
  QK ([kvq4.cl:197-202](../../crates/backend-opencl/kernels/kvq4.cl:197)) и к
  выходу перед гейтом ([kvq4.cl:289](../../crates/backend-opencl/kernels/kvq4.cl:289)).
  Для `f16` ротации нет (raw store, [decode.cl:609-616](../../crates/backend-opencl/kernels/decode.cl:609)).

**Online softmax.** Во всех квантованных путях softmax — flash-style в f32, без
материализации scores: 2 позиции за итерацию,
`mn = max(m, s0, s1)`, `corr = exp(m−mn)`, `w = exp(s−mn)`,
`l = l·corr + w0 + w1`, `acc = acc·corr + w0·v0 + w1·v1`
([kvq4.cl:249-258](../../crates/backend-opencl/kernels/kvq4.cl:249),
[kvarn.cl:352-372](../../crates/backend-opencl/kernels/kvarn.cl:352)).
Масштаб `1/√d` — **после** редукции dot: `sc = dot · 0.0883883476`
([kvq4.cl:247](../../crates/backend-opencl/kernels/kvq4.cl:247)); в kvarn то же
число, потому что `q` уже повёрнут с тем же `1/√128`.

Почему именно online, а не двухпроходно, как у стенда: у kvarn/q4 V-масштаб
зависит от токена (`vb/vbz`), поэтому PV обязан считаться в f32; а f32-acc
дешевле перемасштабировать на `corr`, чем хранить все scores. У стенда
аккумулятор целочисленный, `corr` в него не вписать без потери точности —
отсюда двухпроходная схема (§1.4).

**Разбиение по длине.** `splits = clamp(span/512, 1, 16)`
([lib.rs:47-49](../../crates/backend-opencl/src/lib.rs:47)) — у каждой части свой
`(m, l, acc)`, затем `attention_combine` перенормирует
([decode.cl:583-607](../../crates/backend-opencl/kernels/decode.cl:583)).

### 3.1 `f16` (дефолт эталона)

* K и V — f16, без ротации, без масштабов
  ([decode.cl:609-616](../../crates/backend-opencl/kernels/decode.cl:609),
  [common.cl:61-73](../../crates/backend-opencl/kernels/common.cl:61));
* раскладка: `kv[slot·kv_heads·d + kvh·d + dim]` — токен-major, голова внутри
  токена;
* страница: `32·4·128·2·2 = 65 536` Б, **16 бит/элемент**;
* декуант в attention: загрузка `half2` + конверсия в f32
  ([decode.cl:655-657](../../crates/backend-opencl/kernels/decode.cl:655));
* softmax/PV — f32 online;
* качество **[замер]**: cos 0.999375 (1024/64) и 0.998781 (2048/128)
  ([backends.md:327-330](../../docs/backends.md:327));
* роль: эталон качества и отладка; в Myelin — `KvFormat::F16`
  ([spec.rs:153](../crates/myelin-core/src/spec.rs:153), `model_pack.rs:369`).

### 3.2 `q4` (алиас `k4v4`) — TQ, кодбук

Макрос `-DLFM_KV4` у `q4` **не** включается (он только у `kvarn_k4v4`);
`kv_q4 = ("q4" | "k4v4")` ([lib.rs:999](../../crates/backend-opencl/src/lib.rs:999)),
`kv4 = ("kvarn_k4v4")` ([lib.rs:1001](../../crates/backend-opencl/src/lib.rs:1001)).
То есть **`k4v4` — алиас `q4`, а не kvarn**: частая ошибка, см. §3.3.

* **K**: нормируется на L2-норму вектора (128 измерений), поворачивается `R`,
  квантуется в **кодбук из 16 значений** `LFM_TQ_CB`
  ([kvq4.cl:3-8](../../crates/backend-opencl/kernels/kvq4.cl:3)):

```text
LFM_TQ_CB = [−0.237629839, −0.180795565, −0.141763277, −0.110247883,
             −0.082792693, −0.057744624, −0.034133885, −0.011296587,
              0.011296085,  0.034133404,  0.057743630,  0.082791700,
              0.110246892,  0.141762679,  0.180795257,  0.237628942]
k_d = CB[code_d] · kn,    kn = ||k||₂  (f16)
```

  ([kvq4.cl:62-89](../../crates/backend-opencl/kernels/kvq4.cl:62): норма → `rot_fwd`
  → ближайший уровень; `kn` пишется один раз на токен);
* **V**: поворачивается `R`, квантуется **равномерно** в 4 бита,
  `q = clamp(round(x/sc), −7, 7)`, код `q+7`, `sc = max|x|/7` на токен
  ([kvq4.cl:91-122](../../crates/backend-opencl/kernels/kvq4.cl:91)); нулевой точки
  нет;
* раскладка: `code[(slot·kv_heads + h)·64 + lane]`, ниббл: младший — чётное
  измерение; масштабы `kn[slot·kv_heads + h]`, `vs[...]` — f16 на (токен, голову)
  ([kvq4.cl:124-145](../../crates/backend-opencl/kernels/kvq4.cl:124),
  [lib.rs:5774-5778](../../crates/backend-opencl/src/lib.rs:5774));
* декуант: `k = CB[code]·kn`; `v = (code − 7)·vs`
  ([kvq4.cl:175-195](../../crates/backend-opencl/kernels/kvq4.cl:175));
* страница: `4 · (2048 K + 64 kn + 2048 V + 64 vs) = 16 896` Б,
  **4.125 бит/элемент** **[код]** ([config.md:92](../../docs/config.md:92));
* качество: прямого `gpu-cmp` для `q4` в документах нет; в
  [kernel-contract.md §5.2](kernel-contract.md) стоит **~0.998** как оценка
  формата — это не замер, при приёмке DSP нужен свой;
* важно: масштабы **на токен**, поэтому страницу можно писать по токену, без
  staging и commit (в отличие от kvarn, §3.7); в эталоне так и сделано
  ([kvq4.cl:124-145](../../crates/backend-opencl/kernels/kvq4.cl:124) без ring).

### 3.3 `kvarn_k4v2` (алиас `kvarn`) — то, что в ТЗ названо `k4v2`

**Подтверждение соответствия:** `kv_kvarn = ("kvarn"|"kvarn_k4v2"|"kvarn_k4v4")`,
`kv4` включается только для `kvarn_k4v4` ([lib.rs:1000-1001](../../crates/backend-opencl/src/lib.rs:1000)).
Значит `k4v2` из ТЗ = `kvarn_k4v2` = алиас `kvarn`. Это дефолт по ADR-003
([decisions.md:61-79](decisions.md)).

* **K**: 4 бита, **равномерно**, коды в повёрнутом базисе; масштаб на канал
  `ka[c]` и на токен `kc[t]`, оба f16; коды `q+8`
  ([kvarn.cl:154-177](../../crates/backend-opencl/kernels/kvarn.cl:154));
  декуант ([kvarn.cl:263-272](../../crates/backend-opencl/kernels/kvarn.cl:263)):

```text
k_d = (code_d − 8) · ka[c] · kc_t
```

* **V**: 2 бита, `q ∈ [0,3]`, 4 измерения в байте (`sh = 2·(d%4)`);
  шаг `vb[t]`, нуль `vbz[t]` (f16, на токен), масштаб `vc[c]` (f16, на канал)
  ([kvarn.cl:274-293](../../crates/backend-opencl/kernels/kvarn.cl:274)):

```text
v_d = (q_d · vb_t + vbz_t) · vc[c]
```

* **как считаются масштабы** ([kvarn.cl:105-261](../../crates/backend-opencl/kernels/kvarn.cl:105)):
  три чередующиеся итерации по дисперсии в лог-домене
  (`lc += ½·ln(var)` , `lr += ½·ln(var)`, clamp `[−0.3, 10]`), затем
  `ka[c] = (max_t |k·ic·ir| / 7) · e^{lr[c]}`,
  `kc[t] = e^{lc[t]}`, `vb[t] = step·e^{lc}`, `vbz[t] = min·e^{lc}`,
  `vc[c] = e^{lr[c]}`; всё считается **по странице целиком** (32 токена);
* раскладка: коды `kcode[((phys<<5|tok)·kv_heads + h)·64 + lane]`,
  `vcode[...·32 + (lane>>1)]` — это ровно `(slot·kv_heads+h)` при
  `slot = phys·32+tok`, как и у `q4`, поэтому коды обоих форматов
  адресуются одинаково; масштабы же лежат **по странице**:
  `ka[(phys·kv_heads+h)·128 + c]`, `kc/vb/vbz[(phys·kv_heads+h)·32 + t]`,
  `vc[...·128 + c]` ([kvarn.cl:155-166,227-231](../../crates/backend-opencl/kernels/kvarn.cl:155));
* страница: `4 · (2048 + 256 + 64 + 1024 + 64 + 64 + 256) = 15 104` Б,
  **3.69 бит/элемент** **[код]** ([config.md:93](../../docs/config.md:93));
* качество **[замер]** ([machine.md:332-333](../../benchmarks/machine.md:332),
  повторный прогон — [backends.md:328-330](../../docs/backends.md:328)):
  cos **0.991101** (1024/64) и **0.995183** (2048/128); на Vega 8 порог в
  `bench_check.sh` опущен до 0.9860; `gpu-check` целиком — 0.998543 против
  0.998427 у f16;
* цена **[замер, AMD]**: decode −14 % на ctx 512 и −5 % на ctx 8192;
  prefill −10 % (512) … −31 % (8192) — из-за декодирования кодов и
  обратного Hadamard ([backends.md:140-145](../../docs/backends.md:140),
  [machine.md:340-346](../../benchmarks/machine.md:340)).

### 3.4 `kvarn_k4v4`

То же, что §3.3, но V — 4 бита, `q ∈ [0,15]`, нибблы без смещения
(`q | q<<4`), шаг `(max−min)/15`, нуль `min` ([kvarn.cl:217-243](../../crates/backend-opencl/kernels/kvarn.cl:217)).
Страница **19 200 Б**, 4.69 бит/элемент. Качество **[замер]**:
0.998151 / 0.998632, строгий прогон — 0.998341 / 0.998720
([machine.md:870-873](../../benchmarks/machine.md:870)). Цена: +11 МиБ на ctx 8192 и
−2…−4 % prefill ([backends.md:109-114](../../docs/backends.md:109)).

### 3.5 Сводная таблица

| формат | K, бит/эл. | V, бит/эл. | бит/эл. (итого) | страница 32 т. | масштабы | качество cos 1024/64 · 2048/128 | staging |
|---|---|---|---|---|---|---|---|
| `f16` | 16 | 16 | 16 | 65 536 | — | 0.999375 · 0.998781 | нет |
| `q4` / `k4v4` | 4 (кодбук 16) | 4 (offset-7) | 4.125 | 16 896 | L2-норма токена · max/7 токена | ~0.998 **[оценка]** | нет |
| `kvarn` / `kvarn_k4v2` | 4 | 2 | 3.69 | 15 104 | ka канал + kc токен · vb/vbz токен + vc канал | 0.991101 · 0.995183 | да |
| `kvarn_k4v4` | 4 | 4 | 4.69 | 19 200 | то же | 0.998151 · 0.998632 | да |
| stand `k4v8` | 4 (+64 Б паддинг) | 8 | 8.125 | 33 280 | kt токен · s_v канал (статич.) | K cos ≥ 0.99, V cos ≥ 0.999 ([kvpage tests](../crates/myelin-pack/tests/kvpage.rs:85)) | нет |

Память KV на ctx 8192 (11 слоёв, 185 МБ у f16): k4v2 — 42.5 МБ (коды+масштабы)
(≈63 МиБ с staging при `prefill_ntok=128`), k4v4 — 54 МБ, `q4` — 47.5 МБ,
stand — 93.7 МБ **[код]** ([config.md:89-94](../../docs/config.md:89),
[backends.md:131-140](../../docs/backends.md:131)).

Тип softmax/аккумулятора: `f16`, `q4`, `kvarn_*` — online **f32** (`m,l,acc` —
float, §3.0); stand — двухпроходный **int** (scores Q8, `p` Q7, `acc` int32,
§1.4). Это главное, что нельзя «смешать»: формат и тип аккумулятора выбираются
вместе.

### 3.6 Стоимость декванта в attention

Байт на токен на голову (то, что реально читает ядро):

| формат | K | V | всего |
|---|---|---|---|
| `f16` | 256 | 256 | 512 |
| `q4` | 64 + 2 | 64 + 2 | 132 |
| `kvarn_k4v2` | 64 + 8 (ka/32) + 2 | 32 + 2 + 2 + 8 (vc/32) | 118 |
| `kvarn_k4v4` | 64 + 8 + 2 | 64 + 2 + 2 + 8 | 150 |
| stand | 128 + 4 | 128 | 260 |

ALU на элемент (K — на путь QK, V — на путь PV):

| формат | K | V |
|---|---|---|
| `f16` | 0 (load+convert) | 0 |
| `q4` | распаковка ниббла + **lookup кодбука** | распаковка + `(q−7)·s` |
| `kvarn` | распаковка + `(code−8)·ka·kc` (2 mul) | распаковка 2/4 б + `q·vb + vbz`, затем `·vc` |
| stand | `vand`+`vlsr`, `vrmpyacc` (4 MAC/инстр.), `−8·Σq` | `vrmpyacc` int8 (4 MAC/инстр.) |

Ключевое различие: у эталона QK/PV — **f32** (ровно 1 MAC на lane за
инструкцию), у стенда — int8/int4 `vrmpyacc` (4 MAC на lane). На AMD это
окупается памятью (decode −5…−14 %), потому что decode там упирается в полосу
весов. На HVX decode attention **считает**, а не читает: 878 мкс на 11 слоёв при
ctx 2048 **[замер]**, то есть ~80 мкс/слой; f32-путь даст примерно 4–6× больше
инструкций **[оценка]** → 3.5–5 мс/токен при 2048 и 14–21 мс при 8192. Это
главный риск переноса kvarn «как есть» (§3.8, §8 вопрос №1).

### 3.7 Staging и commit: почему kvarn требует, а `q4` — нет

* `kvarn`: `ka` и `vc` — **на канал по всей странице** (32 токена). Пока не
  известны все 32 токена, страницу закодировать нельзя. Отсюда staging-кольцо
  на `max(64, next_pow2(prefill_ntok))` токенов
  ([lib.rs:56-65](../../crates/backend-opencl/src/lib.rs:56)), инвариант
  `tile ≤ stage − 32` ([lib.rs:74-76](../../crates/backend-opencl/src/lib.rs:74)),
  и `kvarn_commit` при `(pos+1) % 32 == 0`
  ([lib.rs:2368-2395](../../crates/backend-opencl/src/lib.rs:2368)). Внимание
  читает `pos < full_end` из кодов, `pos ≥ full_end` — из staging
  ([kvarn.cl:319-328](../../crates/backend-opencl/kernels/kvarn.cl:319)).
  В staging лежат **уже повёрнутые** f16 K/V ([kvarn.cl:6-29](../../crates/backend-opencl/kernels/kvarn.cl:6)).
* `q4`: масштабы на токен (норма K, max/7 у V) → каждый токен кодируется
  независимо, страницы нет, staging и commit не нужны
  ([kvq4.cl:124-145](../../crates/backend-opencl/kernels/kvq4.cl:124)).
* `f16`: пишется напрямую ([decode.cl:609-616](../../crates/backend-opencl/kernels/decode.cl:609)).
* stand: масштабы тоже на токен (K) и статические (V) → пишется напрямую.

Следствие для Myelin: **выбор формата определяет, нужен ли staging**.
`q4`-подобная схема (per-token масштабы) позволяет писать KV токен-в-токен и
снимает `kv_stage`, `max_rows ≤ kv_stage−32` и `full_end`. kvarn этого не
позволяет, пока масштабы остаются на канал/страницу (§7, п. 2).

### 3.8 Что переносится на HVX/HMX, а что завязано на OpenCL

| механизм эталона | где | переносится? |
|---|---|---|
| биты кодов, уровни (`+8`, `+7`, nibble order), формулы декуанта | `kvq4.cl`, `kvarn.cl` | **да, один-в-один** — это контракт качества |
| кодбук `LFM_TQ_CB`, `LFM_TQ_SIGN`, `1/√128` | `kvq4.cl:3-46` | **да** (таблица и знаки — константы) |
| `__local` + `barrier()` в Hadamard | `kvq4.cl:16-28` | **нет**: на HVX барьеров нет; 128-точечный Адамар — 7 стадий бабочек на регистрах, кросс-lane стадии через `vror` (аналог `shuffle_xor`); образец — `lfm_tq_rot_fwd8/rot_inv8` ([kvarn.cl:793-835](../../crates/backend-opencl/kernels/kvarn.cl:793)) |
| `sub_group_reduce_add` | `LFM_SG64`, `kvq4.cl:330` | **нет**: на HVX — дерево `VSUM5` ([impl.c:1664-1671](../dsp/hvx/lfmgemv_impl.c:1664)) |
| 64 lane × 2 канала (f32) | все ядра | **нет**: HVX-вектор 128 Б = 32 lane × 4 Б (int) или 32 f32; раскладка «2 канала на lane» не переносится |
| `uchar4`/`float8` + `vload8` | `kvarn.cl:838-879` | **нет 1:1**: на HVX — маски `vand`/`vlsr`, как в `dot_q8_k4` ([impl.c:1650-1662](../dsp/hvx/lfmgemv_impl.c:1650)) |
| `KV_T` = half/float, авто-конверсия | `common.cl:26-74` | **частично**: f16 надо конвертировать в f32 на загрузке; f16-векторной арифметики в стенде нет |
| online softmax, `(m,l,acc)`, splits | `decode.cl:583-607`, `kvq4.cl:249-258` | **да** (алгоритм), но реализация другая: нет `exp` как инструкции — нужна LUT/полином (у стенда уже есть Q7-таблица, [impl.c:1598-1643](../dsp/hvx/lfmgemv_impl.c:1598)) |
| 4 позиции за итерацию, кэш `ka`/`vc` в регистрах | `kvarn.cl:926-984` | **да** (идея) |
| сплиты по длине `clamp(span/512,1,16)` | `lib.rs:47-49` | **да** (на HVX — team из 2 контекстов) |
| `__builtin_amdgcn_sdot4` / `dot(char4)` | `common.cl:8-24` | **нет**: у HVX `vrmpyacc` |
| 2 независимых набора буферов K и V (`make_kv` ×2) | `lib.rs:1199-1224`, `5747-5782` | **нет**: у эталона вторая копия кодов просто не используется; в DSP держим один набор |

Отдельно: **`q4` K требует lookup кодбука на каждый элемент**. У HVX нет
gather по 16-элементной f32-таблице; варианты — байтовый `vlut32` (4 lookup на
f32), монотонная аппроксимация кодбука или равномерный int4 в повёрнутом базисе
(теряется Gaussian-shaping; качество надо мерить). Это отдельное решение для
DSP-команды (вопрос №2).

### 3.9 Рекомендуемая последовательность для DSP

1. **`f16`** — для сверки с эталоном и как эталон качества (уже есть
   `KvFormat::F16`); на HVX даёт ту же f32-цену, что и kvarn, но без кода.
2. **stand `k4v8`** — оставить как «быстрый integer-путь» и эталон паритета:
   он единственный сегодня даёт 0.88 мс attention на 11 слоёв.
3. **`q4`/TQ или kvarn** — только после решения вопроса №1 (f32 attention) и
   №2 (кодбук). Если нужен integer-путь с вращением и per-token масштабами —
   это будет **новый формат**, которого в эталоне нет; его надо явно
   благословить, а не выдать за kvarn.

---

## 4. Радикс-кэш

### 4.1 Структура

Trie по блокам 32 токена; образец — [radix.rs](../../crates/backend-opencl/src/radix.rs):

```text
block_hash(parent, tokens) = FNV(tokens) ^ mix(parent),  результат | 1
hash_chain(prompt) = [h1, h2, …],  hi зависит от всех предыдущих токенов
```

Ключ — префикс, а не «набор токенов» ([radix.rs:5-23](../../crates/backend-opencl/src/radix.rs:5)).
Эталон уже использует это: `reuse_cap(prompt) = (len−1)/32` — последняя
страница промпта не переиспользуется ([lib.rs:78-80](../../crates/backend-opencl/src/lib.rs:78)).

### 4.2 Единица шаринга — страница (KV + conv-снапшот)

* страница 32 токена, `phys` общий для нескольких таблиц, данные не копируются;
* вместе с KV-страницей шарится **снапшот conv-состояния на её конце**:
  `(conv_cache−1)·hidden·4 = 49 152` Б на слой × 13 conv-слоёв =
  **0.624 МБ на страницу** ([paged-kv.md §4](paged-kv.md)); в эталоне —
  `snap` на слой, `kv_blocks·(conv_cache−1)·h` f32
  ([lib.rs:1188-1190](../../crates/backend-opencl/src/lib.rs:1188)).

### 4.3 Refcount

| операция | эффект |
|---|---|
| `publish(hash)` | повесить хэш после досчёта страницы |
| `take(hash)` | `refs += 1`, вернуть `phys` |
| `release(phys)` | `refs -= 1`; при 0 — в free-list, хэш остаётся |
| `alloc()` | свободная страница с минимальным `used`, старый хэш снимается |

`refs > 0` ⇒ страницу нельзя вытеснять и перезаписывать
([radix.rs:97-185](../../crates/backend-opencl/src/radix.rs:97)).

### 4.4 Вытеснение

LRU по свободным страницам. Это эквивалентно «LRU по листьям»: `refs = 0`
⇒ страница не входит ни в одну живую таблицу, а живой потомок невозможен без
живого предка (таблица потомка содержит предка). Отдельного бюджета нет —
пул страниц и есть бюджет ([paged-kv.md §4](paged-kv.md)).

### 4.5 Block table и API ядра

* префикс-хит: `phys`-ids копируются в таблицу последовательности, `take` на
  каждую; KV не копируется;
* ядро **не знает про radix**: видит только `u32 tbl[pt_stride]` и `pos`;
* единственное требование — произвольные `phys` (у эталона и у `attn_core`
  выполнено) и запись по `phys`, а не по `pos%32` (§6.3 п. 1).

### 4.6 Инварианты

```
hash_chain — только по полным 32-токенным блокам
последняя страница промпта не публикуется          (reuse_cap = (len−1)/32)
refs = 0 ⇒ страница не в живой таблице
alloc вытесняет только refs = 0
∀ записей таблицы: 0 ≤ phys < pool_pages
таблица строки: ceil((pos+ntok)/32) записей
Σ refs = число записей во всех живых таблицах (диагностика)
```

---

## 5. Continuous batching

### 5.1 API планировщика (хост, Rust)

```rust
pub struct SeqId(u32);

pub struct StepBudget {
    pub max_rows: usize,   // M: одновременных последовательностей
    pub max_ntok: usize,   // сумма токенов (decode: = max_rows)
    pub max_pages: usize,  // сколько страниц разрешено добрать
}

pub enum Admit { Ok { prefix_pages: usize }, Full, TooLong, Rejected }

pub trait BatchPlanner {
    fn add(&mut self, seq: SeqId, prompt: &[u32], cfg: &SeqConfig) -> Admit;
    fn step(&mut self, budget: StepBudget) -> StepBatch;
    fn finish(&mut self, seq: SeqId, reason: Finish);
    fn truncate(&mut self, seq: SeqId, pos: u32);      // откат (spec decode)
    fn evict(&mut self, need_pages: usize) -> usize;   // radix LRU
    fn stats(&self) -> PlannerStats;
}

pub struct StepBatch { pub rows: Vec<RowDesc>, pub ntok: usize }
```

`add` делает префикс-хит, резервирует страницы, кладёт строку в очередь;
`step` выбирает готовые строки, проверяет бюджеты и **сам** собирает таблицы;
`evict` вызывается, когда `alloc` не нашёл страницу.

### 5.2 Дескриптор для DSP

Принимается ABI `Row`/`StepDesc`/`StepOut`
([abi.rs:18-68](../../crates/lfm-npu-ref/src/abi.rs:18)) плюс на строку:

| поле | смысл |
|---|---|
| `q_off`, `out_off` | адреса активаций в scratch (16·128 int8 · int32) |
| `pos`, `ntok` | позиция первого токена строки и число токенов |
| `tbl_off`, `tbl_len` | срез таблицы (в `u32`) и её длина (= `pt_stride`) |
| `full_end` | граница коды/staging (нужно только kvarn) |
| `window`, `flags` | окно и флаги (prefill, последняя строка) |

Это ровно то, что уже принимают батчевые ядра эталона (§2.4), плюс `q_off/out_off`,
которых там нет (в эталоне q/out — плотные массивы `ntok × heads × 128`).

### 5.3 Инварианты шага

см. §2.4; дополнительно:

```
Σ новых страниц ≤ StepBudget.max_pages
rows.len() ≤ min(64, StepBudget.max_rows)
команд в списке ≤ LFM_CMD_MAX (иначе — микро-батчи)
```

### 5.4 Шаги

* **decode**: M строк по 1 токену; один `exec` на микро-батч; на строку —
  qkv, norm/rope, `KVSTORE`, `ATTN`, `ATTN_SCALE`, o_proj;
* **prefill**: строки с `ntok > 1` (эталон так и делает: `pos_b[t]`, `ntok` до
  `batch_cap`), либо последовательный проход по токенам, как сейчас
  ([session.rs:237](../crates/myelin-core/src/session.rs:237));
* **finish** — `release` страниц, опубликованные префиксы остаются в trie;
* **truncate** — откат `pos`, `release` страниц выше `ceil(pos/32)`, пересчёт
  хвоста (staging) при kvarn.

### 5.5 Предельное M

Ограничение — память KV, не арифметика: `ceil(ctx/32)·page_bytes·kv_layers` на
последовательность.

| формат | ctx | МБ на посл. | M при 256 МБ | M при 512 МБ |
|---|---|---|---|---|
| kvarn_k4v2 | 2048 | 10.6 | 24 | 48 |
| kvarn_k4v2 | 8192 | 42.5 | 6 | 12 |
| kvarn_k4v2 (6 full + 5 sliding) | 8192 | 23.8 | 10 | 21 |
| q4 | 2048 | 11.9 | 21 | 43 |
| f16 | 8192 | 185 | 1 | 2 |
| stand k4v8 | 8192 | 93.7 | 2 | 5 |

Плюс conv-снапшоты радикса (§4.2). Практический потолок: **M = 4…10 при ctx 8k,
M = 16…32 при ctx ≤ 2k**; формальный предел контракта — 64, но при `M > 8`
один `exec` на токен не собирается по числу команд (§5.6).

### 5.6 Команды и HVX/HMX

Бюджет на токен при `M = 1` — 200–236 команд из 256
([decode-plan.md §3.2](decode-plan.md)); на строку +7 команд (qkv 4,
`KVSTORE`, `ATTN`, `ATTN_SCALE`):

| M | attention-команд (8·M·11) | влезает в 256? |
|---|---|---|
| 1 | ~88–99 | да (200–236) |
| 2 | +88 | нет |
| 8 | +616 | нет |

Отсюда: `M_exec = 1…2` при `LFM_CMD_MAX = 256`; `512…1024` (список 64–98 КБ
< 1 МиБ) → `M_exec ≈ 8–12`; при `M_exec = 8` надо мерить wall и кэш арен
(`LIST_CACHE = 128`, `MAXARENA = 160`, [executor.rs:130](../crates/myelin-npu/src/executor.rs:130)).

Железо: **HVX** (decode) — работа = `M · 4` групп (строка × KV-голова);
**HMX** (prefill) — spatial M = 64 строк тайла, это **64 токена**, а не 64
последовательности ([hmx.md §7.2](hmx.md)); attention на HMX не считается.

---

## 6. Граница DSP/host

### 6.1 Кто что делает

| работа | host | DSP |
|---|---|---|
| токенизация, chat template, сэмплинг | да | — |
| admission, M, порядок строк | да | — |
| block table, radix, refcount, LRU | да | — |
| `pos/window/full_end/tbl_off` на строку | да | читает |
| q/k/v проекции | — | да |
| q/k RMSNorm, rope | сейчас хост | **цель: да** (O5) |
| вращение `R` (q, K, V) | — | **да** (эталон делает на устройстве) |
| квантование K/V и запись в страницу | — | да |
| attention (QK/softmax/PV/norm) | — | да |
| гейт, o_proj, применение `vc`/`s_v` | хост | **цель: да** |
| логиты / argmax | хост | да |
| conv-состояние и снапшоты | копия для отладки | да |

### 6.2 Что обязан уметь DSP-интерфейс

1. `KVSTORE(phys, slot, …)` — запись по **физической** странице; формат — §3.
2. `ATTN(rows[], nrows)` — varlen по §2.4 (свой `q/out/tbl/pos/ntok/window/
   full_end` на строку, произвольные `phys`).
3. `ATTN_SCALE(vc[], gate[])` — применение поканального масштаба V и гейта без
   хоста (в эталоне это уже внутри ядра, [kvq4.cl:290-292](../../crates/backend-opencl/kernels/kvq4.cl:290)).
4. `ROT`/`ROT_INV` — Hadamard; либо `q` вращается на хосте, а обратное
   вращение **свёрнуто в веса o_proj** при упаковке (гейт — скаляр на голову,
   с вращением коммутирует; o_proj придётся перекалибровать в пакере, потому
   что матрица меняется на `W_o · blockdiag(R⁻¹)`).
5. `COMMIT(phys)` — только если K/V-масштабы остаются на страницу (kvarn).
6. `NORMQ`/`ROPE`/`RESIDUAL` — чтобы один `exec` на токен (O5).
7. Типизированные ошибки (`Ctl::Invalid/Oom` с `need/have`), без паники:
   `pos ≥ ctx`, `phys ≥ pool_pages`, несоответствие `tbl_len`, `ntok = 0`.

### 6.3 Что из текущего кода этому противоречит

1. **`kv_write` писал не в ту страницу (исправлено на хосте, ждёт скелет).**
   Хост передавал `slot = position % 32`
   ([dsp.rs:1825](../crates/myelin-core/src/backend/dsp.rs:1825)), C считает
   `row = head*slots + slot` ([impl.c:2396](../dsp/hvx/lfmgemv_impl.c:2396)) и
   требует `slot < 32` ([impl.c:2389](../dsp/hvx/lfmgemv_impl.c:2389)).
   При `pos ≥ 32` запись уходила в **первую** страницу головы, а чтение — по
   `(tbl[pos>>5]<<5)|(pos&31)` ([impl.c:1890](../dsp/hvx/lfmgemv_impl.c:1890)).
   Хост исправлен: `kv_slot` считает глобальный
   `table[pos/32]·32 + pos%32` ([dsp.rs:2177](../crates/myelin-core/src/backend/dsp.rs:2177),
   см. §9). Старый скелет теперь честно отвечает `AEE_EBADPARM` на `slot ≥ 32`;
   чтобы включить контекст > 32, нужно снять проверку `slot < 32` в C (§9).
2. **Таблица identity, `tbl_len = pages`.** [dsp.rs:1330](../crates/myelin-core/src/backend/dsp.rs:1330),
   [dsp.rs:1720](../crates/myelin-core/src/backend/dsp.rs:1720): шаринг и радикс
   невозможны; `tbl_len` растёт каждые 32 токена → новый список → `map_arena`
   2.2 мс и вытеснение стабильных списков ([decode-plan.md §5/O2](decode-plan.md)).
3. **`seq` = длина, а не id.** [dsp.rs:1311](../crates/myelin-core/src/backend/dsp.rs:1311),
   [dsp.rs:1732](../crates/myelin-core/src/backend/dsp.rs:1732); в ABI `Row.seq`
   — слот последовательности (он же индекс кольца: `slot = (seq << rbits) +
   (pos & rmask)`, [kvarn.cl:13](../../crates/backend-opencl/kernels/kvarn.cl:13)).
4. **Пул приватный, per-layer per-session.** [session.rs:147](../crates/myelin-core/src/session.rs:147),
   [kv.rs:44-100](../crates/myelin-core/src/kv.rs:44): радикс требует один пул
   на модель и общие `phys`.
5. **Хост-зеркало хранит f32 K/V.** [dsp.rs:1716](../crates/myelin-core/src/backend/dsp.rs:1716):
   22.5 КБ/токен, 46 МБ при ctx 2048, ×M.
6. **Страйды из размеров буферов.** `sk/sv/skt = wbytes/heads`
   ([impl.c:2161-2165](../dsp/hvx/lfmgemv_impl.c:2161)); проверки
   `wbytes == heads·slots·{128,128,4}` нет.
7. **Нет батча в `Backend`.** `attention_layer` принимает одну
   `kv: &mut dyn KvCache` ([mod.rs:196](../crates/myelin-core/src/backend/mod.rs:196)),
   `Session` хранит `Vec<Option<PagedKvCache>>`
   ([session.rs:122](../crates/myelin-core/src/session.rs:122)), шаг — один токен.
8. **embed и norm/rope на хосте.** [session.rs:350](../crates/myelin-core/src/session.rs:350),
   [dsp.rs:2203](../crates/myelin-core/src/backend/dsp.rs:2203) — один `exec`
   на токен невозможен.
9. **Нет вращения и нет online softmax.** Стенд: raw K/V, двухпроходный int
   softmax (§1.4); эталон: `R` + online f32 (§3.0). Это не «оптимизация», а
   разная арифметика: перенос kvarn/q4 требует переписать ядро attention.
10. **`hvs` не используется** ([dsp.rs:297](../crates/myelin-core/src/backend/dsp.rs:297)),
    `attn_core` игнорирует `h3`; для kvarn нужны `vb/vbz/vc`, для q4 — `vs`
    на токен: место в ABI есть, семантики нет.
11. **Нет staging/`full_end`.** Нужны, если K/V-масштабы остаются на страницу
    (kvarn); при q4/stand не нужны.

---

## 7. Расхождения с paged-kv.md

Ни одно не отменяет ADR-002/003/004: страница 32, `u32`-ids,
`kv_pool = max_seqs`, radix с refcount и LRU, kvarn_k4v2 по умолчанию.

| # | paged-kv.md | здесь | причина |
|---|---|---|---|
| 1 | kvarn: `ka` на канал + `kc` на токен; staging обязателен | сохраняем как есть; но фиксируем, что **kvarn требует f32-attention** (§3.6) и что для integer-пути нужен другой формат | это следует из кода эталона: `ka` нельзя свернуть в int8 `q` (свой на страницу), PV с `vb/vbz/vc` не факторизуется; в эталоне поэтому всё в f32 |
| 2 | staging-кольцо, `max_rows ≤ kv_stage−32`, чтение хвоста | сохраняем для kvarn; для `q4` и `f16` не нужно | у q4 масштабы на токен ([kvq4.cl:62-122](../../crates/backend-opencl/kernels/kvq4.cl:62)), страница кодируется по токену |
| 3 | раскладка страницы: OpenCL — token-major с головой внутри токена, DSP — head-major; «выбрать одну» | **кодек — бит-в-бит как в эталоне, порядок арены — приватный для DSP**; проверка не побайтовая, а round-trip decode | иначе нельзя ни сравнивать с `lfm-npu-ref`, ни использовать page-transposed PV стенда; paged-kv сам оставил выбор открытым |
| 4 | `Row.seq` — слот последовательности | сохраняем; текущий DSP использует поле как длину | баг кода, не дизайна |
| 5 | conv-снапшот на каждую страницу, `snap_slots ≤ max_pages` | снапшот на чекпоинт каждые N страниц (N = 8…32) | 0.624 МБ/страница против 0.166 МБ KV: при 1540 страницах ~1 ГБ снапшотов на 256 МБ KV (вопрос №6) |
| 6 | форматы: f16, kvarn_k4v2, kvarn_k4v4 | добавлены `q4`/`k4v4` (TQ) и явно назван stand `k4v8` | `q4` есть в эталоне ([options.rs:182](../../crates/backend-opencl/src/options.rs:182)), но не описан в paged-kv; stand-формат — то, что реально читает HVX сегодня |
| 7 | `kv_stage` — степень двойки ≥ 64 | остаётся только для kvarn | следствие п. 2 |

Отдельно: paged-kv.md §2 называет «kvarn k4v2 = 15 104, k4v4 = 19 200» —
совпадает с кодом ([config.md:93-94](../../docs/config.md:93)); но там же сказано
«DSP-ядро attention приняло head-major раскладку» — это про **стенд**, не про
kvarn; для kvarn/Q4 такой раскладки в эталоне нет (§7 п. 3).

---

## 8. Открытые вопросы к Оркестратору

1. **Формат KV для DSP v1 и цена точного kvarn.** Дефолт ADR-003 —
   `kvarn_k4v2`, но его перенос на HVX означает f32 QK/PV:
   **[оценка]** 3.5–5 мс/токен при ctx 2048 (сейчас 0.88 мс) и 14–21 мс при
   8192, то есть −8…−15 % decode на целевом бюджете 40 мс. Варианты:
   (а) принять f32 и качество 0.995; (б) integer-путь с per-token масштабами
   (byte-layout kvarn, `ka=1`, `vc` статический) — требует замера качества и
   формально расходится с ADR-003; (в) `q4`-путь (per-token, кодбук) — нужен
   lookup; (г) f16 для v1. **Нужно решение до начала порта attention.**
2. **Кодбук `q4` на HVX.** 16-уровневый `LFM_TQ_CB` требует gather; варианты:
   `vlut32` (4 lookup на f32), монотонная аппроксимация, равномерный int4 в
   повёрнутом базисе. Что из этого приемлемо по качеству и кто мерит?
3. **Риск качества k4v2.** Замеры эталона: 0.991101 (1024/64) и 0.995183
   (2048/128) против 0.999375/0.998781 у f16; на Vega 8 порог уже опущен до
   0.9860. Для LFM3-1.7B это осознанный выбор пользователя; подтверждаем его
   как приёмку для DSP (cos логитов ≥ 0.99 на 2048) или требуем k4v4/`q4` на
   длинном контексте?
4. **Varlen-команда.** Эталон уже принимает `pos_b/seq_b/tbl_off_b` (§2.4).
   Делаем `ATTN` с массивом строк сразу (перенос эталонной схемы) или первый
   релиз — M копий команд в одном списке? Рекомендация: сразу строки, это
   дешевле, чем дважды менять ABI.
5. **Целевой M и бюджет KV.** При ctx 8k и 256 МБ M ≈ 6–10 (§5.5). Целимся в
   M = 4 (полный ctx) или M = 16 (короткие диалоги) с отказом при исчерпании?
6. **Conv-снапшоты для радикса.** 0.624 МБ на страницу; варианты: чекпоинт
   каждые 8/32 страницы, только для опубликованных листьев, отказ от шаринга
   conv-состояния с пересчётом префикса.
7. **`kv_pool` при батче.** ADR-004 фиксирует `kv_pool = max_seqs`; при M = 16
   и ctx 8k это 680 МБ кодов kvarn. Оставляем отказ при исчерпании или вводим
   общий пул с preemption?
8. **Порядок работ.** (1) починить `kv_write` (физ. страница) и сделать
   таблицу настоящей; (2) varlen-`ATTN` по эталонной схеме; (3) f16 как
   эталон качества DSP; (4) формат из вопроса №1; (5) радикс. Подтвердить.

---

## 9. Найденный баг: `kv_write` писал в слот 0..31 вместо глобального

**Симптом.** При `position ≥ 32` внимание перестаёт видеть новые токены:
запись уходит в первую страницу головы (`slot = position % 32`), чтение идёт
по глобальному слоту (`(tbl[pos>>5]<<5) | (pos&31)`, при identity-таблице —
`pos`). Первые 32 слота перезаписываются каждые 32 токена, страницы 1..N-1
остаются нулями.

**Причина.** Контракт `kv_write` — `row = head*slots + slot`, где `slots` —
полное число слотов (`cache_pages·32`), то есть `slot` обязан быть глобальным.
Хост передавал page-local `position % 32`. C-скелет это принимал, потому что
дополнительно проверял `slot < 32` — та же page-local семантика, но в
`attn_core` чтение уже глобальное.

**Правка.**
* хост: `kv_slot(table, position) = table[position/32]·32 + position%32`,
  проверка `slot < slots` до `id_len`; таблица считается один раз и уходит и в
  `kv_write`, и в `ATTN` ([dsp.rs:1724](../crates/myelin-core/src/backend/dsp.rs:1724),
  [dsp.rs:2177](../crates/myelin-core/src/backend/dsp.rs:2177));
* DSP (нужно согласовать, файл не менялся): в
  [impl.c:2389](../dsp/hvx/lfmgemv_impl.c:2389) снять `slot >= 32`:

```c
- if (slots < 1 || slot < 0 || slot >= 32 || slot >= slots) return AEE_EBADPARM;
+ if (slots < 1 || slot < 0 || slot >= slots) return AEE_EBADPARM;
```

Границы строк уже проверяются ниже (`ko + 128 > bk->wbytes` и т.д.), а
`slot < slots` гарантирует `row = head*slots + slot < heads*slots`.

**Доказательство (телефон, реальный слой attention 2, фикстура на 48
позиций, CPU-эталон `attn_y.f32`).**

```text
старый хост + старый скелет:  token 0..31  cos ≈ 0.9671..0.9678
                              token 32     cos   0.7931
                              token 47     cos   0.8571   VERDICT: FAIL
новый хост + скелет без `slot >= 32`:
                              token 0..31  cos ≈ 0.9671..0.9678 (те же цифры)
                              token 32     cos   0.9680
                              token 47     cos   0.9686   VERDICT: PASS
```

End-to-end `dsp-generate` (prompt 5 токенов, 64 сгенерированных): с новым
хостом на старом скелете `kv_write` падает на позиции 32 с
`AEE_EBADPARM` (0x8000040e); на исправленном скелете прогон доходит до
позиции 68. Токены на позициях < 32 совпадают со старым прогоном
(`383 278 2690 2891 302 278 5201 …`), то есть короткий контекст не изменён.

**Остаточный риск.** Текст `dsp-generate` и до, и после правки расходится с
CPU уже на первом токене (`383` против `4741` у CPU), а после позиции 32
остаётся периодичным. Это отдельный дефект/эффект int4-пути: слой attention 2
на первом токене даёт cos 0.968 в обоих прогонах и правкой слота не меняется.
KV-баг объясняет провал качества именно после 32 токенов, но не начальное
расхождение с CPU.

