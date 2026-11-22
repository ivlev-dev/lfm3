# HMX int8 на Hexagon v68 — геометрия кроутонов, control-слова, семантика

Дата: 2026-11-22. Все артефакты в `/tmp/opencode/hmx5/`.
Репозиторий `lfm3` не трогался.

## 0. TL;DR (что меняется относительно прошлых попыток)

1. Формула fp16 `addr(s,c) = (s>>1)*128 + (s&1)*64 + (c>>4)*32 + (c&15)*2`
   **неверна**: она тождественно равна `64*s + 2*c` (плоский spatial-major).
   Проверенная (v68, скалярный эталон) формула:
   `addr16(s,c) = (s>>1)*128 + 4*c + 2*(s&1)`.
2. Формула int8 (A8W8, обычная `activation.ub`, без `:cm`):
   `addr8(s,c) = (s>>2)*128 + 4*c + (s&3)`, s = 0..63, c = 0..31 (2 КБ).
3. Один multiply-pair на v68 int8 = **64 spatial × 32 N × 32 K = 65536 MAC
   (131072 ops)**, а не 16 MAC. 32 «канала» (N) — это штатный размер.
4. `weight.b` = signed int8 (1 КБ на 32×32), `activation.ub` = unsigned int8
   (2 КБ на 64×32), выход `sat.ub` = saturating unsigned byte (2 КБ).
5. `:deep` на активации = накопление по K (dC+1 кроутонов, до 32 → K до 1024).
   `:deep` на весе = 2 КБ весов = 64 выходных канала и 2 банка аккумулятора,
   два store'а. Это два независимых механизма.
6. На v68 **нет инструкций `cvt.*`** (assembler: «requires -mv73 or higher»).
   Вместо них — слитный `mxmem(Rs,Rt):after.<fmt> = acc`.
7. У QNN int8-ядра (`bbb`) на v68 **нет deep вообще**: `hmx_v73_convbbb1x1deep_stride1`
   в v68-скелете — заглушка `r0=-1`. Deep для int8 есть только у hbh/hnh
   (16-битный выход).
8. 24-канальный «потолок», скорее всего, следствие п.1/п.2 (раскладка) и/или
   Rt веса; сама геометрия такой потолок не накладывает.

---

## 1. Геометрия кроутонов

### 1.1. Как читается `ChunkedMemoryLayout`

`/tmp/opencode/qairt/include/QNN/HTP/core/memory_layout.h:109-224`.
Шаблон `ChunkedMemoryLayout<Rank, Dim0,Chunk0, Dim1,Chunk1, ...>`:

* пары идут **от старшей размерности к младшей** (первая пара — самая внешняя);
* внутри пары `(Dim, ChunkSize)`: координата делится на «внутренний» размер
  (`Smaller::ChunkSizes[Dim]`), берётся `% ChunkSize`, умножается на число
  элементов внутреннего чанка (`Smaller::chunk_total`), затем рекурсия;
* `ChunkSize == 0` — размерность не чанкуется (просто проходит насквозь);
* `ChunkSizes[]` — произведение всех ChunkSize по данной размерности,
  `chunk_total` — размер одного кроутона в элементах;
* для `ChunkedMemoryLayout` (IndirRanks = Rank) `linear_offset = chunk_index*chunk_total + chunk_offset`,
  т.е. для тензора из одного кроутона (dims == ChunkSizes) получается чистая
  внутрикроутонная раскладка.

Проверка метода: программка `/tmp/opencode/hmx5/layout_test2.cpp`
(собирается g++, подключает настоящий `memory_layout.h`) печатает offset'ы
для `R4CroutonLayout`, `R4Crouton2Layout`, `R4H1WideCrouton2Layout`,
`R4WideCroutonLayout`. Результаты совпали с формулами ниже до последнего байта.

### 1.2. int8: какие раскладки бывают

В QNN 8-битные тензоры — это `Ldefs::Crouton_8` = `R4CroutonLayout`
(`tens/tensor_definitions.h:163`, `memory_layout.h:309`):

```
R4CroutonLayout = ChunkedMemoryLayout<4, 0,0, 1,0, 2,0, 3,0, 1,8, 2,8, 3,32>
=> ChunkSizes = [1,8,8,32], chunk_total = 2048 (байт для u8)
addr8_qnn(h,w,c) = h*256 + w*32 + c          // = s*32 + c, s = h*8+w, s=0..63
```

`Ldefs::WideCrouton_8` (`R4WideCroutonLayout`, `memory_layout.h:311`):
`[1,2,32,32]`, 2 КБ = 2 × 1 КБ:
`addr8_wide(y,w,c) = y*1024 + w*32 + c`.

Это **не** то, что читает `activation.ub` без модификатора. Аппаратная
раскладка обычной (не `:cm`) 8-битной активации — продолжение fp16-геометрии
PRM: в 128-байтном векторе 32 канала с шагом 4 байта, а младшие биты spatial
лежат в младших битах адреса:

```
addr8_hw(s,c) = (s>>2)*128 + 4*c + (s&3)     // s=0..63, c=0..31, 2048 B
```

Отсюда «4 spatial на 128-байтный вектор». Именно так пакует данные
`cdsp_peak` (см. §3), и на реальном v68 (8cx Gen3, QCS6490) его скалярный
эталон проходит по всем 64×32 элементам.

`activation.ub:cm` / `...:after:cm...` — «channel-major» вариант: каналы
лежат подряд внутри spatial-ячейки, т.е. ровно QNN-раскладка `Crouton_8`
(§2). Поэтому QNN-ядра `bbb` и используют `:cm` вместе с `QUint8Crouton`.

### 1.3. fp16: проверка метода и исправление эталона

16-битные кроутоны QNN — `Crouton_16` = `R4Crouton2Layout`
(`tens/tensor_definitions.h:164`, `memory_layout.h:323`), 8×4×32 = 1024 элемента = 2 КБ;
`H1WideCrouton_16` (`memory_layout.h:333`) — 1×32×32, 1024 элемента = 2 КБ.

Обе дают одну и ту же формулу при плоском spatial `s`:

```
Crouton_16:   addr16(y,x,c) = y*256 + (x>>1)*128 + 4*c + 2*(x&1)
              s = y*4+x  =>  addr16(s,c) = (s>>1)*128 + 4*c + 2*(s&1)
H1Wide_16:    addr16(s,c) = (s>>1)*128 + 4*c + 2*(s&1)     (s = w)
```

Совпадает с PRM V81, Figure 4 («2 activation spatials per VTCM vector»,
адреса 0,4,…,124 / 2,6,…,126 для каналов) и с упаковкой `cdsp_peak`
(`(r/2)*128 + k*4 + (r&1)*2`, см. §3).

Старая формула `(s>>1)*128 + (s&1)*64 + (c>>4)*32 + (c&15)*2`
упрощается до `64*s + 2*c` — это плоский spatial-major. Она совпадает с
*намерением* кода `qhblas_hmx_transfer_row_ahf` (байты 0,1 = a0; 64,65 = a32;
2,3 = a1 …), но этот сэмпл писан с ошибками типов (`uint8_t*` вместо `__fp16*`)
и с off-by-one в `act_ctrl` (см. §2.6), так что как эталон он не годится.

### 1.4. Веса (все типы) — по PRM, Figure 6/7

```
addr_wgt_bytes = (ic>>k)*128 + 4*oc + (ic & ((1<<k)-1))
k=1 для 16-бит (2 ic на 128B-вектор), k=2 для 8-бит (4 ic), k=3 для 4-бит (8 ic)
```
т.е. int8-вес 32(K)×32(N) = 1 КБ, 4-битный — 512 Б, 2-битный — 256 Б.
Адрес веса должен быть выровнен на 128 Б; `Rt = nbytes - 1`
(= `(dW<<7)|0x7F`), `Rs[5]` — инверсия знака веса.

---

## 2. Что реально делает QNN v68 (дизассемблер)

Инструменты: `hexagon-llvm-objdump -d --mattr=+hmx`
(`.../HEXAGON_Tools/21.0.04/Tools/bin`). Дампы:
`/tmp/opencode/hmx5/bbb_hbh.dis`, `hnh.dis`, `f16.dis`, `params_stride1.dis`.

### 2.1. Семейства ядер и набор инструкций

| семейство | activation | weight | store | deep |
|---|---|---|---|---|
| `hmx_convbbb*` (u8→u8) | `activation.ub:cm` | `weight.b` | `mxmem:after:cm:sat.ub = acc` | нет (v73+) |
| `hmx_convhbh*` (→u16) | `activation.ub` | `weight.b`, `weight.b:deep`, `:dilate` | `mxmem:after:sat.uh = acc:2x1` | да |
| `hmx_convhnh*` (→u16, 4-бит вес) | `activation.ub` | `weight.n` (`:dilate`, `:deep`) | `mxmem:after:sat.uh = acc:2x1` | да |
| `hmx_convf16*` | `activation.hf` | `weight.hf` | `mxmem:after.hf = acc` | (hf: `:deep`) |

Статический подсчёт сигнатур по `.text` (скрипт в `/tmp/opencode/hmx5`,
маски по полям Rs[12:8]/Rt[20:16], parse-биты [15:14] исключены):

```
activation.ub:cm      25   только bbb-семейство
activation.ub         76   hbh/hnh (все), 5x5, NxN
activation.ub:deep     2   hbh1x1deep (0x3720c8), hnh1x1deep (0x3743c8)
activation.hf         47   f16-семейство
weight.b              28   bbb + один остаточный путь hbh deep
weight.b:deep          4   hbh1x1deep (0x371f98…)
weight.n               1   hnh1x1deep (0x3743cc)
weight.n:dilate       41   hnh-семейство
weight.c:deep          0   ← на v68 не используется вообще
weight.hf             56   f16
```

### 2.2. `hmx_convbbb1x1_stride1` (0x36fdc0) — эталон int8 A8W8

Ключевые места (`bbb_hbh.dis`):

```
36fdf0  mxclracc                                  ; целочисленный аккумулятор
36fe70  r24 = or(r7, #0x1c)                       ; Rt активации = params[0xc] | 0x1C
36fe74  r25 = #0x3ff                              ; Rt веса по умолчанию = 1023 (1 КБ-1)
36fe8c  bias = mxmem2(r3)
36feb8  activation.ub = mxmem(r6,r24):cm
36febc  weight.b     = mxmem(r8,r25)
36fec0  r8 += r25+1                               ; +1 КБ на следующий weight-тайл
36fec4  activation.ub = mxmem(r23,r24):cm         ; вторая активация
36fec8  weight.b     = mxmem(r8,r25)
36feb0  r8 = add(r8,#0x400)                       ; (в другой ветке — +1 КБ)
36fef8  mxmem(r10,r11):after:cm:sat.ub = acc      ; конверт+store, 2 КБ
36ff40  r3 = add(r3,#0x100)                       ; следующий 256-байтный bias-блок
```

Факты:

* int8-аккумулятор: `mxclracc` (без `.hf`); вес **signed** `weight.b`;
  активация **unsigned** `activation.ub`; выход `sat.ub` (0…255).
* один weight-тайл = **1024 байта** (32 K × 32 N), по умолчанию `Rt=0x3FF`,
  для «хвостового» тайла берётся `params[0x18]`.
* активация читается двумя инструкциями (два указателя `r1+0`/`r1+4`,
  шаг таблицы 8 Б) на каждый weight-тайл — характерно для `:cm`-пути.
* выход пишется одной `sat.ub`-инструкцией; указатель выхода `r10` двигается
  на `m0 = 4*params[4]` (типично 2048 = 64×32 байт).
* bias грузится через `mxmem2` по 256 Б на группу 32 каналов, `r3 += 0x100`.
* **deep нет**. `hmx_v73_convbbb1x1deep_stride1` (0x370300) в v68-скелете:
  `r0 = #-0x1; jumpr r31` — заглушка.

Control-слова берутся из структуры `hmx_params`, их готовит
`set_hmx_params_conv1x1` (0x2d8900) / `set_hmx_params_stride1` (0x2d83a0):

* `[p+0x04]` — Rt выхода (store), `[p+0x08]` — вспомогательный (act low),
  `[p+0x0c]` — Rt активации, `[p+0x18]` — Rt веса,
  `[p+0x1c]` — второй Rt веса (для deep/2 тайлов; у w4b по умолчанию 0x3FF).
* в `conv1x1`: `p[0xc] = (arg<<5) | ((K/32-1)<<2) | 3` в промежуточном
  кодировании, затем перекодируется в PRM-формат:
  `rt = (x & ~0x7F) | (0x7C & (x<<2)) | ((x>>5)&3)` — т.е. ch_stop
  кладётся в биты [6:2], а 2 бита маски — в [1:0]; dY/dC (биты ≥11)
  сохраняются как есть. Это подтверждает PRM-поля (ch_stop [6:2],
  spatial mask [10:7]+[1], dY/dC [31:11]).
* Для fp16 в том же конвейере: `r22 = or(r7,#0x70)`, дефолт `Rt веса = 0x7FF`
  (2 КБ-1), `mxclracc.hf`, `bias = mxmem(r3)` (не `mxmem2`), store `after.hf`.

### 2.3. Формулы control-слов (PRM V81, Table 1/2 + проверка по QNN)

```
act_rt = (dY_or_dC << 11) | (((sm>>1)&0xF) << 7) | ((ch_stop & 0x1F) << 2) | ((sm&1) << 1)
act_rs = addr(2KB) | (((so>>1)&0xF) << 7) | ((ch_start & 0x1F) << 2) | ((so&1) << 1)
wgt_rt = (dW << 7) | 0x7F = nbytes-1   (nbytes кратно 128)
wgt_rs = addr(128B) | (neg << 5)
store_rt = только spatial mask (те же биты [10:7],[1])
```

* `dC = (#кроутонов - 1)` для `activation.*:deep` (PRM §4.2.1, max 31 → 32 кроутона
  = 1024 входных канала; на v68 FastRPC-капабилити `hmx_depth=32`).
* `ch_stop` — 5 бит, максимум **31** (32 входных канала на инструкцию);
  `ch_start` тоже 5 бит, кратен 8.
* `dY` — смещение до второго кроутона для `:single` (может быть отрицательным).
* `so` — spatial offset (для `:single`/unaligned).
* `sm` (spatial mask) — 5 бит: 0 = X, 1 = Y; младший бит — самый быстрый
  spatial. Пример PRM: `11100` = YYYXX = 8(Y)×4(X). Для плоского GEMM маска
  роли не играет, можно 0.

Числа, которые реально использует/подтверждает практика:

| что | значение | источник |
|---|---|---|
| fp16 act Rt (32 spatial, плоский) | `0x7C` (sm=0, ch_stop=31) | cdsp `hmx_fp16_accumulate_hf` |
| int8 act Rt (64 spatial, плоский) | `0x3FF` | cdsp `hmx_int8_tile_ub` |
| int8 weight Rt (1 КБ) | `1023` | cdsp, QNN дефолт |
| int8 weight Rt (2 КБ, deep) | `2047` | cdsp wdeep |
| fp16 weight Rt (2 КБ) | `2047` | cdsp/QNN `0x7FF` |
| 4-бит weight Rt (512 Б) | `511` | QNN `hnh` дефолт `r23=#0x1ff` |
| store Rt | `0` | cdsp (для плоского выхода) |
| bias-блок | 256 Б / 32 канала | PRM §2.3.1 |
| bias set index | `addr[1:0]` | PRM Figure 14 |

Интересно, что у QNN int8-ядер маска в Rt не используется: `bbb` делает
`or(r7,#0x1C)` (это гарантирует младшие 3 бита ch_stop = 111, т.е. stop
кратен 8 минус 1), `hbh/hnh` — `or(r7,#0x70)`. Это не маска, а «доводка»
ch_stop/резервных битов; фактические значения приходят из `hmx_params`.

### 2.4. `hbh` / `hnh` и deep

`hmx_convhbh1x1deep_stride1` (0x371ec0):

```
371f98  activation.ub = mxmem(r6,r22)
371f98  weight.b     = mxmem(r8,r9):deep      ; r9 = params[0x18] (для N=64 cdsp: 2047 = 2 КБ)
371fc8  bias = mxmem2(r3+0x100)
371fcc  mxmem(r10,r11):after:sat.uh = acc:2x1  ; банк 0
371fd4  bias = mxmem2(r3+0x100)
371fe0  mxmem(r10,r11):after:sat.uh = acc:2x1  ; банк 1 (после clear+swap)
```

и «вторая ветка» того же ядра — deep по активации:

```
3720c8  activation.ub = mxmem(r6,r7):deep     ; dC в Rt, до 32 кроутонов
3720cc  weight.b      = mxmem(r8,r9)          ; диапазон веса = весь K
```

`hnh` (0x373b80) — то же, но `weight.n` (4-бит) и дефолт `Rt=0x1FF` (512 Б),
поэтому это **A8W4**, а не A8W2 (см. §3).

`mxmem(...):after:... = acc` — слитный convert+store; на v68 только он.
`cvt.hf = acc(Rs)` / `cvt.ub = acc(Rs)` / `mxmem = cvt` ассемблер отвергает:
«instruction requires: -mv73 or higher» (проверено `hexagon-clang -mv68`).

`mxswapacc` в 1x1/GEMM-путях **не используется** (0 вхождений), банки там
переключает сам `after`-store (clear+swap, PRM §2.3). Зато `mxswapacc`
встречается в 5x5-свёртках: 4 раза в `hbh_5x5`/`hnh_5x5` и 2 раза
`mxswapacc.hf` в `convf16_5x5` — там он нужен, чтобы досчитывать окна.
`mxclracc` — 32 вхождения, `mxclracc.hf` — 19, `mxshl(acc,#16)` — 0.

### 2.5. Bias / scale (int8)

PRM §2.3.1, Figure 13/15: 32 канала × 64 бита = 256 Б на «set»; в VTCM
сначала идут младшие 32 бита всех 32 каналов (128 Б), затем старшие (128 Б).
Поля одного 64-битного регистра:

```
[63:43] input bias   (FP16 + 5 доп. бит мантиссы)
[42:40] shape        (0:x, 1:min(x,0), 2:max(x,0), 3:|x|, 4..7 — с минусом)
[31:16] output bias  (FP16), [39:36] +4 бита мантиссы
[15:0]  scale        (FP16), [35:32] +4 бита мантиссы
```

convert: `cvt = round( shape(ACC + input_bias) * scale + output_bias )`.
Для int8 у cdsp: `scale = 512.0 (0x6000)` даёт тождество — т.е. целочисленный
конверт имеет неявный множитель **1/512**; реальный коэффициент квантования
надо домножать на 512. Это же место — для zero-point: `input_bias = -zp*sum(w)`
на канал.

QNN int8: `bias = mxmem2(addr)`, 256 Б на 32 канала, `addr += 0x100` на
каждую группу; low 2 бита адреса = индекс set'а, QNN всегда использует set 0.
QNN fp16: `bias = mxmem(addr)` (не `mxmem2`), 128 Б на 32 канала (только
scale+output bias, без input bias) — на v68 это отдельный «старый» bias-регистр.

### 2.6. Прочие находки

* `hmx_convhbh1x1_stride1_unaligned` использует `:single` (0x371d38,
  `activation.ub = mxmem(r6,r7):single`) и store `:after:retain` /
  `:before` — это чистая «склейка» двух кроутонов по Y (PRM §4.4).
* `:above` встречается в `bbb`-dilate (0x370ef8) — активация «на строку выше».
* `qhblas_hmx_matrix_matrix_mpy_hf_sm.S`: `act_ctrl=(K>>5)<<11|0x7C` —
  **off-by-one**: по PRM `dC = K/32 - 1`. Плюс `wgt_ctrl=(K>>5)<<11 - 1`
  корректен (nbytes-1). Это объясняет, почему fp16-сэмпл «работал не так».
* `activations`/`weights` строки: `hmx_extra_8b_acc`, `hmx_bias_comp`,
  `native_hmx_a16w4`, `Qu16_h8w4d32` — это graph-опции/имена типов,
  отдельных HMX-функций под ними в скелете нет (только строки).
* `set_hmx_params_convw4b*` / `convw2b*` существуют: 4-бит тайл = 512 Б,
  2-бит = 256 Б (по дефолтным Rt 0x1FF / 0xFF в билдерах).

---

## 3. Внешние подтверждения

1. **PRM V81 HMX** `80-N2040-62 Rev AA` (скачан в `/tmp/opencode/hmx5/hmx_v81_prm.pdf`,
   текст `hmx_v81_prm.txt`):
   * Table 1 — поля активации; Table 2 — поля веса (`Rt=nbytes-1`, `Rs[5]=neg`);
   * Figure 4/7 — VTCM-порядок (канал с шагом 4 Б, spatial в младших битах);
   * §2.3.1/Figure 15 — 64-битный bias, 4 set'а, 256 Б;
   * §4.2 — `:deep` по активации = dC+1 кроутонов (до 32);
   * §4.4 — `:single`/spatial offset.
   URL: https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf

2. **strongtz/cdsp_peak** (github) — рабочий v68 int8-код с проверкой по
   скалярному эталону:
   * `dsp/cdsp_peak_hmx.c`: `HMX_TILE_U8_BYTES=1024`,
     `activation.ub = mxmem(act,1023)`, `weight.b = mxmem(w,1023)`,
     `mxmem(out,0):after:sat.ub=acc`, wdeep: `weight.b:deep` c Rt=2047 и
     два store'а (+2048), adeep: `activation.ub:deep` c Rt=32*1024-1;
   * `host/compute_validation.h`: точные раскладки
     `activation[(r/4)*128 + k*4 + (r&3)]`,
     `weight[(c/32)*1024 + (k/4)*128 + (c%32)*4 + (k&3)]`,
     `output[(c/32)*2048 + (r/4)*128 + (c%32)*4 + (r&3)]`
     и формула fp16 `(r/2)*128 + k*4 + (r&1)*2`;
   * `results/old/v68.md`: реальные прогоны на Snapdragon 8cx Gen3 и QCS6490,
     баннер `hmx_depth=32 hmx_spatial=64`; строки `hmx-int8-ub`,
     `...-full-x16/x32/x64`, `...-wdeep-full-x64` отработали (checksum есть),
     т.е. **64 spatial, накопление по K парами и N=64 через weight:deep
     работают на v68**. Оговорка: README репозитория просит перепроверять
     старые results (могли быть собраны прежним бинарём), поэтому это
     сильное, но не абсолютное подтверждение; сама упаковка в
     `host/compute_validation.h` актуальна.
   URL: https://github.com/strongtz/cdsp_peak

3. **LLVM** `34f2fb6` «Hexagon hmx» (12.11.2026):
   * добавлены только fp16-инструкции + `-mhmx`, `+hmx`;
   * кодировки: activation `0x…E4`, weight `0x…EF`, `mxclr_acc` и т.д.;
   * `cvt.hf`/`mxmem=cvt` помечены `Requires<[HasV73,UseHMX]>` — прямое
     подтверждение, что на v68 cvt нет.
   URL: https://github.com/llvm/llvm-project/commit/34f2fb6b56d03b62885900c86f5c83e4c9d63641
   Позже был PR #243572 (интринсики/builtins), в main `.td` пока только hf.

4. **cdsp_peak README**: «The INT4 probe is A8W4 (`activation.ub` with
   `weight.n:2x`), not A4W4» → `weight.n` = 4-бит (nibble).
   В v68-скелете есть строки `native_hmx_a16w4`, `Qu16_h8w4d32` — тоже про
   A8/A16 + W4.
   `weight.c` в v68-скелете не встречается ни разу (0 сигнатур); ядра с `c`
   появляются только в v81 (`hmx_v81_convhch...`). По именам (n=nibble,
   c=crumb) `c` = 2-бит, но это **гипотеза**: прямого подтверждения для v68 нет.

5. **llama.cpp HMX** (`/tmp/opencode/docs/hmx-llamacpp.diff`): fp16-only,
   использует слитный `mxmem(...):after.hf = acc` и `bias = mxmem2`,
   `activation.hf:deep` для K — согласуется с §2.

---

## 4. Итоговая таблица int8 (v68)

| параметр | значение |
|---|---|
| активация | `activation.ub` = **uint8**, 2 КБ = 64 spatial × 32 K |
| раскладка активации (обычная) | `(s>>2)*128 + 4*k + (s&3)` |
| раскладка активации (`:cm`) | `s*32 + k` (QNN `Crouton_8`, 8×8×32) |
| Rt активации | `(dC<<11) | mask | (ch_stop<<2)`; плоско: `0x3FF` |
| вес | `weight.b` = **int8**, 1 КБ = 32 K × 32 N, `Rt=1023` |
| раскладка веса | `(k>>2)*128 + 4*c + (k&3)` |
| элементов в 2 КБ | активация 2048 (64×32), вес 1024 (32×32), выход 2048 (64×32) |
| max `ch_stop` | 31 (5 бит), т.е. 32 вх. канала на инструкцию |
| N (каналы) за проход | 32; 64 — через `weight.b:deep` (2 КБ) + 2 store'а |
| `:deep` на активации | dC+1 кроутонов по K (до 32 → K≤1024), вес должен покрывать тот же K |
| `:deep` на весе | 2 КБ = 64 вых. канала, 2 банка аккумулятора |
| накопление K>32 | (а) просто повторять пары до store'а — валидировано; (б) `activation.ub:deep` |
| аккумулятор | целочисленный (`mxclracc`, без `.hf`), store сам делает clear+swap |
| выход | `mxmem:after:sat.ub = acc` (2 КБ) или `:after:sat.uh = acc:2x1` (16-бит) |
| bias/scale | 256 Б на 32 канала (`bias = mxmem2`), 64-бит/канал, layout: low-слова 128 Б, затем high-слова; scale=[15:0]+[35:32], out_bias=[31:16]+[39:36], in_bias=[63:43], shape=[42:40] |
| неявный масштаб int8 | 1/512 (scale=0x6000 → 1.0) |
| `cvt` на v68 | нет; только слитный `mxmem:after` |
| идеальный порядок весов для kvarn | packed int8 32×32, 1 КБ/тайл, `(k>>2)*128+4*c+(k&3)` |

### Готовый список экспериментов для следующего агента

1. **База**: A(64×32 u8, 2 КБ, формула §1.2) × W(32×32 i8, 1 КБ, §1.4),
   `act_rt=0x3FF`, `wgt_rt=1023`, bias[32] = `0x6000` (в low-словах),
   `mxclracc; bias=mxmem2; activation.ub; weight.b; mxmem:after:sat.ub`.
   Ожидание: `out = clamp(dot,0,255)` поэлементно на всех 2048 байтах.
   Паттерны: all-ones, identity (r%32==k), impulse (r=63,k=31), отриц. веса,
   k=0/1 значения 200/100 (проверка знака/переноса).
2. Если тест 1 даёт не все 32 канала — проверить `act_rt` (0x3FF vs 0x7C vs
   0x77C) и `store_rt` (0 vs 0x70/0x7C); impulse по r=0,1,2,3,63 покажет,
   какой бит spatial куда лёг.
3. **K=64**: две пары (act+2048, weight+1024) до store'а — ожидание сумма
   двух dot'ов. Затем K=128/512 (8/32 пары).
4. **K=64 одной инструкцией**: `activation.ub = mxmem(a,dC=1):deep` +
   `weight.b = mxmem(w,2047)`; сравнить с (3). Проверить dC=0..31.
5. **N=64**: `weight.b = mxmem(w,2047):deep`, затем два
   `mxmem(out,0):after:sat.ub` (второй на out+2048), bias перезагружать
   (+0x100) перед каждым; эталоны с ненулевыми только в c<32 и только в c≥32
   (у cdsp тесты 10/11).
6. **`:cm`**: упаковать активацию как `s*32+k`, `act_rt = 0x1C|...`
   (как QNN bbb) и store `:after:cm:sat.ub`; impulse'ом определить, сколько
   spatial реально в одном `:cm`-кроутоне (1 КБ или 2 КБ).
7. **bias/zero-point**: scale=0x6000 даёт 1.0; out_bias в [31:16];
   input_bias=[63:43] проверить на `-zp*sum(w)`.
8. **unaligned/`:single`** (если нужен conv): `activation.ub:single`,
   `Rs[offset]`, `Rt[dY]=2048`, store `:after:retain`/`:before`.

---

## 5. Неудачи / что не удалось

* Полностью разобрать тайлинг QNN `bbb` (сколько spatial/каналов на один
  store в `:cm`-режиме) не удалось: цикл нетривиальный (loop0 вложен в loop1,
  два указателя активации, `r20/m0` из params). Достоверно: типы, Rt,
  размеры тайлов (1 КБ вес, 2 КБ выход), наличие/отсутствие deep.
* Точная семантика `:cm` в PRM не описана (PRM покрывает только fp16).
  Вывод «`:cm` = раскладка `Crouton_8` (s*32+c)» — сильная, но косвенная
  реконструкция (QNN использует `QUint8Crouton` + `:cm`; cdsp показывает,
  что обычный `ub` — другая раскладка).
* `weight.c` = 2-бит — только гипотеза по именам; на v68 таких ядер нет.
* `hmx_extra_8b_acc` / `hmx_bias_comp` — только строки graph-опций, кода
  в скелете не нашли; влияние на точность не проверено.
* В `hbh1x1deep` вес в `:deep`-паре двигается на `(params[0x18]+1)/2`, а не
  на `params[0x18]+1`; какое именно значение лежит в `params[0x18]` (1 КБ или
  2 КБ) — не дотрассировал. Для реального кода берите эмпирику cdsp:
  `Rt=2047` на 2 КБ весов при N=64.
* V81 PRM не описывает int8/4-бит; расшифровка полей Rt для int8 сделана по
  аналогии с fp16 + PRM-формулам + дизассемблеру QNN, и подтверждена
  эмпирикой cdsp (v68).
