# QNN на Hexagon v68: что выяснено

Заметки по QNN (QAIRT 2.52) на Snapdragon 888 / HTP V68: как строится граф,
что не работает, что даёт по скорости и что видно в дизассемблере скела.
Разбор `qnn-study/REPORT.md` (22.11.2026) — карта символов, `kernels.tsv` и
дизассемблер отдельных функций; раздел 6 ниже опирается на него.

Стенд: `/tmp/opencode/android/qnn*.c` (загружают `libQnnHtp.so` через
`dlopen`, строят граф через `QnnInterface`), SDK — `v2.52.0.261104.zip` в корне
lfm3 (не коммитится). Скел: `libQnnHtpV68Skel.so` (unsigned).

## 1. Как строится граф

Последовательность API (так же в `qnnlayer.c`, `qnnblk.c`, `qnnarg.c`):

1. `QnnInterface_getProviders` → выбрать интерфейс HTP.
2. Создать backend (`QnnBackend_create`) и device (`QnnDevice_create`).
3. Создать context (`QnnContext_create`).
4. Создать граф (`QnnGraph_create(ctx, name, ...)`).
5. Добавить тензоры (`tensorCreateGraphTensor`) и узлы (`QnnGraph_addNode`);
   типы тензоров: `APP_WRITE`, `APP_READ`, `NATIVE`, `STATIC` (веса).
6. `finalize` (`QnnGraph_finalize`) — компиляция под HTP; время печатается.
7. `execute` (`QnnGraph_execute`) с `Qnn_Tensor_t[]` in/out; замер — вокруг
   execute, после прогрева.

Квантование задаётся на тензоре:

```c
t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
t.v1.quantizeParams.quantizationEncoding  = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
t.v1.quantizeParams.scaleOffsetEncoding.scale  = sc;
t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
```

Для весов — `AXIS_SCALE_OFFSET` по оси 0. Для fp16 — `QUANTIZATION_ENCODING_UNDEFINED`.

## 2. Ограничения V68, на которые наткнулись

| что | результат | обход |
|---|---|---|
| `QNN_OP_GROUP_QUERY_ATTENTION` (GQA) | на V68 не проходит: граф не финализируется / опера не зарегистрирована для этого arch | собирать attention из примитивов: `MAT_MUL` (QK^T) → `SOFTMAX` → `MAT_MUL` (PV) |
| depthwise conv (`CONV_2D` с `group = C`, k=7) | падает на финализации | замер слоя снимался **без** depthwise conv (`qnnblk.c`, режим `conv`: «graph_macs … без depthwise conv») |
| два и более `MAT_MUL` в одном графе | граф не собирается | разбивать на отдельные графы/шаги |
| Argmax на NPU | **работает**: `QNN_OP_ARGMAX`, выход int32 | использовать, чтобы не выгружать 128k логитов |
| `cvt`/`f8`-пути HMX | на v68 отсутствуют по ISA (см. [hmx.md](hmx.md)) | слитный `:after:sat.ub = acc` + обязательный bias/scale-блок (hmx.md, §4.4) |

Оговорка: «не зарегистрирована» здесь означает наблюдаемый отказ сборки графа
на V68, а не отсутствие строки в бинарнике — строка `GroupQueryAttention` в
скеле есть (`strings_skel.txt`), но рабочей реализации для V68 нет.

## 3. Context binary сжат

`QnnContext_getBinarySize` / `QnnContext_getBinary` работают. Для крошечного
int8 FC 4×8×15:

* `binary size = 49152`, `written = 49152` — выгружается;
* шапка `v2.52.0.261104115025`, дальше `PICKLE/256/g`;
* веса лежат в скомпилированном HTP-блобе **сжатыми** (DLBC / sparsity-упаковка);
* поиск «120 последовательных различных байт» (stride 1 и 2) и значений
  `129..248` по всему буферу — ничего; zlib-потоков нет.

Вывод: восстановить перестановку весов из context binary нельзя без реверса
DLBC. Инструмент оставлен как рабочий для сверки **размеров** тайлов
(`/tmp/opencode/android/qnndump.c`).

## 4. Что QNN даёт по скорости

| блок | время | условия |
|---|---|---|
| int8 FC 7680×2048 (GEMV) | **1.37 мс** (11.5 ГБ/с) | V68, unsigned, один вызов |
| FFN-слой (gate+up+SiLU+down) | ~4.11 мс (3 × 1.37) | тот же блок трижды |
| полный слой LFM3 (attn или conv) | **1.75 мс** | `qnnlayer`, один граф на слой, M=1 |
| prefill 128×7680×2048 | **2.44 мс** ≈ 0.83 TMAC/s | заявлено QNN; паддинг N до 8192 даёт ~2.4 TMAC/s |

Это опорные цифры для наших ядер: int4 GEMV на HVX — 0.63–0.71 мс (быстрее
QNN int8), FFN-слой на HVX с 2 контекстами — **1.90 мс** (быстрее 4.11 мс;
REPORT20). Prefill на HMX по compute быстрее (376.8 мкс на 128×7680×2048), но
со стейджингом 15.7 МБ весов — **3180–3219 мкс**, т.е. **в 1.31 раза**
медленнее QNN 2.44 мс (REPORT20; [hmx.md](hmx.md) §9.1). Прежние «7319 мкс и
втрое» включали `copycheck` (~4.2 мс), это ошибка замера.

Разрыв с QNN — это разница скорости копирования, а не арифметики: наш
стейджинг идёт на 5.35 ГБ/с, QNN нужно ~6.4 ГБ/с (достижимо на грани: лучший
кратковременный замер 6.5 ГБ/с). Гипотеза «QNN держит веса резидентными»
**невозможна**: 15.7 МБ не влезают в 4 МБ VTCM, а кэш context binary — это DDR.
В рантайм не интегрирован, поэтому пока prefill остаётся за QNN.

## 5. Практические выводы

* QNN годится как **prefill-путь** и как эталон скорости, но не как основа
  рантайма: граф на слой (1.75 мс) плюс RPC/синхронизация не оставляют запаса
  на decode, а ограничения V68 (GQA, depthwise conv, 2× MatMul) вынуждают
  дробить граф.
* Квантование задаётся вручную (`QNN_DEFINITION_DEFINED` + scale/offset или
  axis-scale), автоматический калибратор не используется.
* Для отладки раскладок QNN полезен как «источник истины»: кодировки инструкций
  HMX совпали побайтово, а `:deep`-рецепт найден именно в скеле (раздел 6).

## 6. Дизассемблер скела

Артефакты `/tmp/opencode/qnn-study` (22.11.2026): `hmx_syms.txt` (41
HMX-символ), `groups.md` (карта по категориям: hmx, softmax, norm, requant,
matmul/conv/pool), `kernels.tsv` (op → исходный файл QNN → адреса реализации),
дизассемблер `.s` для `hmx_convhbh_stride1`, `hmx_convf16_1x1_stride1`,
`set_hmx_params_stride1`, `sm_h_crouton`, `ln_*`, `rope`, `mm_q8` и др.
Полный разбор — `REPORT.md` в том же каталоге.

### int8 1×1 = GEMM: `hmx_convhbh1x1deep_stride1` (`0x371ec0`)

```
mxclracc
loop:
  activation.ub = mxmem(r6, r22)          # адрес активации из таблицы
  weight.b      = mxmem(r8, r9):deep      # ← :deep на ВЕСАХ
  r8 += r23                               # r23 = 0x800 = 2048 Б на кроутон
bias = mxmem2(r3)
mxmem(r10, r11):after:sat.uh = acc:2x1    # store 2 КБ
```

Это и есть рабочая форма накопления 32 каналов (проверено у нас: `A=1,B=1` →
32.0). QNN **не** использует `:deep` на активации для K > 16.

### int8 in/out: `hmx_convbbb1x1_stride1` (`0x36fdc0`)

```
activation.ub = mxmem(r6, r24):cm
weight.b      = mxmem(r8, r25)
mxmem(r10, r11):after:cm:sat.ub = acc     # 2 store на итерацию
```

`sat.ub` (8-битный выход) у нас на v68 писал нули; QNN использует его вместе с
`:cm` и двумя store — форма требует отдельного разбора.

### Прочие формы

* `hmx_convhbh_stride1` (`0x372e40`): `activation.ub = mxmem(...):above`,
  `weight.b = mxmem(...):dilate`, store `:after:sat.uh = acc:2x1`, а также
  `:retain` и `:before` — то есть у QNN есть несколько вариантов store под
  разные случаи; для нашего тайла достаточно `:after` без `:before/:retain`.
* `hmx_v73_*` (`0x370000+`) — заглушки `r0 = -1`; на v68 не используются.
* `set_hmx_params_stride1` (`0x2d83a0`) считает `hmx_params` (страйды,
  кроутоны). **Не декодирован** — следующий шаг для точной раскладки активации.
* `hmx_convf16_*` (`0x37d100+`) — fp16-путь, включая `1x1`, `dilate`,
  `stride2`, `5x5`, `NxN`.

Что это даёт: подтверждён `:deep` на весах и порядок
`mxclracc → пара multiply → bias → store`; подтверждено, что QNN ходит по тем
же MMU-адресам и использует `compute_resource_*`, а не `qurt_hmx_*`. Что не
даёт: готовую формулу раскладки активации — её надо либо вытащить из
`set_hmx_params_stride1`/`tile_write`, либо снять дампом VTCM работающего графа
(см. «что осталось» в [hmx.md](hmx.md)).

## 7. Нет данных

* Полный `qnn-study/REPORT.md` — в работе.
* Время GQA/attention через QNN на V68 — граф не собирается, цифры нет.
* Время depthwise conv через QNN — падает, цифры нет.
* Сравнение QNN-слоя 1.75 мс с нашим полным слоем — наш слой целиком (attention
  + FFN + нормы) на устройстве не собран.
