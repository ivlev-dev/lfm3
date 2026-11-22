# REPORT22 — VTCM и HMX в `hexagon-sim` (Hexagon SDK 6.7, v68)

Дата: 2026-11-22.
Тулчейн: `/opt/hexagon-sdk-6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools`
Стенд: `myelin/dsp/hmx/sim/` (`hmx_vtcm_probe.c`, `build.sh`, `run.sh`).

---

## 1. Итог (TL;DR)

1. **VTCM в симуляторе включён по умолчанию** и находится по адресу
   **`0xd8400000 … 0xd87fffff` (4 МБ)**. Включать ничего не нужно, `--mhmx`,
   `--tcm:*`, `--revid`, HAP и QuRT не требуются.
   Ошибка `0x26 Coprocessor VMEM address error` была не «VTCM выключен»,
   а «адрес вне окна VTCM»: все испробованные ранее адреса (`0x08000000`,
   `0x0a000000`, `0x10000000`, …) лежат вне `0xd8400000…0xd87fffff`.
2. **HMX в симуляторе работает**: int8 (`.ub`/`.b`) и int4 (`.ub`/`.n`)
   считаются корректно, K = 32 / 64 / 128 набирается полностью.
3. Найден **обязательный шаг, которого не хватало в `probe.c`**: загрузка
   блока bias/scale через `bias = mxmem2(addr)` с `scale = 0x6000` (identity).
   Без неё MAC считает правильно (в аккумуляторе лежат верные числа), но
   конвертер `mxmem():after:sat.ub = acc` пишет **нули**, потому что умножает
   на нулевой «scale». Именно это выглядело как «HMX не считает».
4. **По вопросу K (п.3 задачи): симулятор НЕ ограничивает K до `ceil(ncr/2)`.**
   При рецепте «одна пара `mxmem`-инструкций на кроутон» набирается полный
   K = 32·ncr и для int8, и для int4. Потеря K на железе — следствие рецепта
   (`:deep` + короткий weight-контрол, который работает как *range clamp*),
   а не архитектурного ограничения.

---

## 2. Как получить/включить VTCM

### 2.1. Окно по умолчанию

```
vtcm_low = 0xd8400000
vtcm_hi  = 0xd87fffff     (4 МБ)
```

Самый простой способ увидеть окно — попросить симулятор поругаться на «чужой»
адрес (это печатает границы):

```sh
hexagon-sim -mv68 --mhmx 1 --verbose ./hmx_probe -- 0x08000000
```
```
WARNING: in_space_warning .../arch_options_calc.c:358:
  Paddr not in vtcm space: pa: 0000000008000000
  vtcm_hi: 00000000d87fffff vtcm_low: 00000000d8400000
WARNING: register_error_exception ...:
  Error exception detected, tnum=0 code=0x26 pc=... badva0=0x8000000 ...
I think the exception was: 0x26, Coprocessor VMEM address error
```

### 2.2. Откуда берётся адрес (дизассемблер, подтверждение)

`in_vtcm_space` (`0x575530` в `lib/iss/libhexagonissv68.so`):

```asm
57553b: mov 0x1a38(%rdi),%rax        ; rax = processor->config
575542: cmpl $0x0,0xf38(%rax)        ; VTCM present?
575549: je   ...                     ; нет -> вернуть 0
575559: movslq 0xdf8(%rax),%rbx      ; offset_kb
575560: shl $0xa,%rbx                ; << 10  -> байты
575564: add 0x308(%rcx),%rbx         ; + memory_base
57556b: mov 0x32dc(%rax),%ecx        ; size (или -1 -> вычисляется)
```

Поля конфига отображены на archstring-опции по строкам, которые печатает
`set_option`:

| смещение | archstring-опция                    | значение по умолчанию |
|----------|-------------------------------------|------------------------|
| `+0xf38` | `QDSP6_VX_PRESENT`                  | 1 (VTCM «есть»)        |
| `+0xdf8` | `QDSP6_TCM_REGION_VTCM_OFFSET` (КБ) | 4096 КБ → `+0x400000`  |
| `+0xdf4` | `QDSP6_TCM_REGION_SZ`               | —                      |

То есть база VTCM = `TCM_region_base (0xd8000000) + offset_kb << 10`.
Проверено:

```sh
hexagon-sim -mv68 --mhmx 1 --verbose \
  --archstring "--QDSP6_TCM_REGION_VTCM_OFFSET 8192" ./hmx_probe -- 0xd8800000
```
```
ARCHSTRING set QDSP6_TCM_REGION_VTCM_OFFSET = 8192
Paddr not in vtcm space: ... vtcm_hi: d8bfffff vtcm_low: d8800000
```
(окно переехало, как и ожидалось).

**Важно:** переносить VTCM этим способом в данной сборке **нельзя** — окно
декодирования меняется, но фактическое хранилище VTCM остаётся на
`0xd8400000`, поэтому и `0xd8400000`, и `0xd8800000` после переноса дают
`0x26`. Рабочая конфигурация — дефолтная.

`QDSP6_VX_PRESENT 0` выключает VTCM совсем (`in_vtcm_space` сразу возвращает 0).

### 2.3. Что НЕ требуется (проверено)

* `--mhmx 0` / `1` / `2` — HMX-инструкции исполняются одинаково; `--mhmx`
  задаёт версию HMX, а не «включение». С `--mhmx 0` и выставленным SSR.XE2
  стенд проходит (`RESULT: OK`).
* `--l2tcm_base`, `--tcm:lowaddr/highaddr`, `--revid` — на окно VTCM не влияют
  (проверены 0xd800/0xd840/0xd900/0xe000 для `--l2tcm_base` — окно неизменно).
* HAP/QuRT (`HAP_compute_res_*`, `runelf.pbn`) — не нужны.
* Секция `.vtcm` в ELF / `-Wl,--section-start` — не нужны: VTCM доступен
  обычными указателями по физическому адресу (v2p-трансляция по умолчанию
  выключена), что и доказывает стенд.

---

## 3. Обязательные условия для непустого `mxmem`-результата

Проверено по одному, каждое ломает результат по-своему:

1. **SSR.XE2 = бит 26**
   ```c
   unsigned s; asm("r0 = ssr\n\t%0 = r0":"=r"(s)::"r0");
   s |= 1u<<26; asm volatile("ssr = %0"::"r"(s):"memory");
   ```
   Без него — `exception code = 0x18` на первой же HMX-инструкции.
2. **Адрес в окне `0xd8400000…0xd87fffff`** и выравнивание на 2 КБ
   (адресное поле `mxmem` — это `Rs[31:11]`). Иначе — `0x26`.
3. **Блок bias/scale, загруженный ДО store-инструкции:**
   ```c
   asm volatile("{ bias = mxmem2(%0) }" :: "r"(bias_ptr) : "memory");
   ```
   256 байт: 32 слова `{out_bias[31:16], scale_fp16[15:0]}`, затем 32 слова
   нулей. Для int8/int4 identity-scale = **`0x00006000`**.
   **Без этой загрузки store пишет нули**, хотя аккумулятор полон правильных
   чисел. Это подтверждено дампом памяти процессора (см. §6): при нулевом
   bias в acc лежало 2048 слов `32`, а `out[]` был полностью нулевой;
   после `bias = mxmem2` с `0x6000` в `out[0]` появилось `32`.
4. `mxclracc` перед серией (иначе acc содержит мусор от предыдущей операции).

---

## 4. Стенд и результаты

`myelin/dsp/hmx/sim/hmx_vtcm_probe.c` — самодостаточная программа:

* раскладывает act/wgt/out/bias в VTCM `0xd8400000` (все слоты 2 КБ aligned);
* данные — все единицы, поэтому результат = **число MAC = число каналов K**;
* проверяет int8 и int4 для K = 32, 64, 128 тремя рецептами.

```sh
cd myelin/dsp/hmx/sim
./build.sh
./run.sh
```

Фактический вывод:

```
VTCM base=0xd8400000  (SSR.XE2 set)
int8 loop per-crouton   K= 32 ->  32  PASS
int8 :deep full-range    K= 32 ->  64  (informational)
int8 :deep short-weight K= 32 ->  64  (weight range clamps the block count)
int4 loop per-crouton   K= 32 ->  32  PASS
int4 :deep full-range    K= 32 ->  32  PASS
int4 :deep short-weight K= 32 ->  32  (weight range clamps the block count)

int8 loop per-crouton   K= 64 ->  64  PASS
int8 :deep full-range    K= 64 ->  96  (informational)
int8 :deep short-weight K= 64 ->  64  (weight range clamps the block count)
int4 loop per-crouton   K= 64 ->  64  PASS
int4 :deep full-range    K= 64 ->  64  PASS
int4 :deep short-weight K= 64 ->  32  (weight range clamps the block count)

int8 loop per-crouton   K=128 -> 128  PASS
int8 :deep full-range    K=128 -> 255  (informational, sat.ub)
int8 :deep short-weight K=128 ->  64  (weight range clamps the block count)
int4 loop per-crouton   K=128 -> 128  PASS
int4 :deep full-range    K=128 -> 128  PASS
int4 :deep short-weight K=128 ->  32  (weight range clamps the block count)

RESULT: OK (VTCM + HMX functional)
```

Контрольные отрицательные прогоны:

```sh
./run.sh 0xd8400000 0          # без SSR.XE2
#   I think the exception was: Don't know, exception code = 0x18
hexagon-sim -mv68 --mhmx 1 ./hmx_probe -- 0x08000000
#   I think the exception was: 0x26, Coprocessor VMEM address error
```

Управляющие слова (совпадают с `mx4_impl.c`):

* `ACT_CTRL = (ncr << 11) | 0x7C` — `ncr` = число 2-КБ блоков, `0x7C` = spatial
  mask 32 строки (`0x3C` = 16 строк);
* `WGT_CTRL` для int8 = «длина в байтах − 1» (2047 на кроутон);
  для int4 = 511 (512 байт на кроутон);
* store: `mxmem(addr, 0):after:sat.ub = acc` (выход 32×64 ub = 2 КБ).

---

## 5. Про K = 32 / 64 / 128 и «ceil(ncr/2)» на железе

Измерения (§4) дают однозначную картину:

| рецепт | int8 | int4 |
|---|---|---|
| цикл: пара `mxmem` на каждый кроутон (`ACT_CTRL(1)` на вызов) | **32 / 64 / 128** | **32 / 64 / 128** |
| один пакет `:deep`, `WGT_CTRL` покрывает все кроутоны | 64 / 96 / 255(sat) — «ncr+1 блок» | 32 / 64 / 128 (ровно K) |
| один пакет `:deep`, `WGT_CTRL` = один кроутон (2047 / 511) | 64 / 64 / 64 | 32 / 32 / 32 |

Выводы:

1. **Архитектурного потолка K нет.** Симулятор честно суммирует и 32, и 64,
   и 128 канала — и для `.b`, и для `.n`. Значит K = 64/128 не «не
   поддерживается»: проблема в рецепте.
2. **`WGT_CTRL` работает как ограничитель диапазона.** Если контрольная
   длина веса не покрывает все блоки (например, константа `WGT_CTRL = 1023`
   при ncr = 2/4), симулятор молча отбрасывает блоки за пределами диапазона:
   int4 даёт ровно один кроутон (32) при любом ncr. Это ровно тот механизм,
   который на железе выглядит как «просуммировались не все кроутоны».
   На железе наблюдалось `ceil(ncr/2)`; в симуляторе при коротком диапазоне
   остаётся 1 кроутон (int4) — то есть симулятор режет даже сильнее, но
   природа явления та же: длина/диапазон веса, а не лимит K.
3. **`:deep` для int8 в этой сборке ведёт себя странно** (`ncr+1` блок: 64 при
   ncr=1, 96 при ncr=2). Судя по семантике из ISS
   (`deep_block_end = deep_mode ? 2 : 1`, `current_acc ^= 1`), `:deep`
   переключает два аккумулятора, а не «удваивает K»; на `mxmem():after`
   читается только текущий. Для int4 `:deep` даёт ровно K.

**Практическая рекомендация:** для K-накопления использовать **отдельную пару
`mxmem`-инструкций на каждый кроутон** с `ACT_CTRL(1)` и весом, покрывающим
ровно этот кроутон (`WGT_CTRL` = 2047 для int8 / 511 для int4). Этот рецепт
детерминирован в симуляторе и даёт полный K. Рецепт «один `:deep`-пакет на
весь K» — источник потери каналов.

---

## 6. Инструменты отладки HMX/ISS (пригодится дальше)

1. **Трассировка HMX-операций (главная находка):**
   ```sh
   hexagon-sim -mv68 --mhmx 1 --archstring "--hmxdebugfile stdout" ./prog
   ```
   Печатает по каждой `mxmem`:
   ```
   MXMEM Debug FXP Mult: Start=d8400000 (PA=d8400000) Range=7c dY=0 block_Type=BLOCK
       Tile: x_mask: 0x780 x_inc: 0x80 y_mask: 0x60 y_inc: 0x20
       Block count in loop 1 Channel start=0 stop=32 y start=0 y_stop = 0
       mac_cycle_limit: 512
   MXMEM Debug Weights: Start=d8408000 Range=3ff Type=NORMAL max pa=d84083ff
   MXMEM Debug FXP CVT: Start=d8410000 ... Current Accumulator: 0x0
   ```
   `RANGE` — это и есть Rt; `max pa` и `block count` сразу показывают, сколько
   блоков реально участвует.
2. `--archstring "--coprocdebugfile stdout"`, `--ihist FILE`
   (гистограмма инструкций: видно, что `M8_mxmem_*` реально исполняются).
3. **gdb**: исполнение идёт через `libhexagonissv68.so` (MCD — это только
   remote-debug протокол). Точки останова на `hmx_exec_M8_mxcvta_sat_uh`
   срабатывают, а на `hmx_exec_M8_mxmema_wei_b` — нет (MAC диспатчится иначе),
   поэтому надёжнее смотреть память:
   ```
   acc = *(unsigned long *)(*(unsigned long *)(state + 0x1998) + 0x10050) + 0x60
   ```
   `state` в gdb — `rdi` на входе любого `hmx_exec_*`; аккумулятор — 32 КБ
   (`0x8000`) по этому адресу (fp16/fxp акки лежат рядом, `current_acc` —
   индекс). Именно так было доказано, что MAC считает верно, а нули
   появляются на конвертере из-за нулевого scale.

---

## 7. Что проверено и оказалось тупиком

* `--tcm:lowaddr/--tcm:highaddr`, `--l2tcm_base`, `--revid`, `--mhmx 0/1/2` —
  на доступность VTCM/HMX не влияют.
* `--archstring "--QDSP6_VX_VTCMBANKS 8"`, `"QDSP6_VX_VTCM_8MB_HACK 1"`,
  `"QDSP6_MBIST_VTCM_2X 1"`, `"QDSP6_VX_VTCMPHYBANKS 8"` — окно/размер не меняют.
* Перенос VTCM через `QDSP6_TCM_REGION_VTCM_OFFSET` — меняет только окно
  декодирования; обращения по новому адресу всё равно дают `0x26`.
* HAP/QuRT-путь (`HAP_compute_res_*`, `runelf.pbn`) — не нужен и в симуляторе
  не работает (возвращает NULL-символы).
* ELF-секция `.vtcm` / `--section-start` — не нужна, VTCM доступен как обычная
  физическая память.

---

## 8. Файлы

| файл | назначение |
|---|---|
| `myelin/dsp/hmx/sim/hmx_vtcm_probe.c` | стенд: VTCM + int8/int4 HMX, K=32/64/128 |
| `myelin/dsp/hmx/sim/build.sh` | сборка `hexagon-clang -mv68 -mhmx -mhvx ...` |
| `myelin/dsp/hmx/sim/run.sh` | запуск `hexagon-sim -mv68 --mhmx 1` |

Существующие `mx4_impl.c`, `host.c` и каталоги `myelin/dsp/{hvx,sim}` не
изменялись.
