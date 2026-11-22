# Локальный стенд exec-каркаса (Hexagon v68 + QuRT под QEMU)

Запускает DSP-ядра из `../hvx/lfmgemv_impl.c` как обычный qexe под
`qemu-system-hexagon` (`-M V68N_1024`): без телефона, без FastRPC, без
Android NDK. Ядра линкуются в образ напрямую, тесты вызывают entry points
`lfmgemv_*` в том же процессе. Плоская память QEMU означает, что адрес DSP
совпадает с указателем, поэтому VTCM/ION для вычислений не нужны.

## Требования

* Hexagon SDK 6.7.0.0 (или совместимый) с QuRT v68:
  `$HEXAGON_SDK_ROOT/rtos/qurt/computev68`.
* Тулчейн `HEXAGON_Tools/21.0.04` (`hexagon-clang`).
* Собранный `qemu-system-hexagon` с машиной `V68N_1024`, поддерживающей HVX.
  По умолчанию ожидается `/tmp/opencode/qemu/build/qemu-system-hexagon`.

## Сборка и запуск

```sh
cd myelin/dsp/sim
./build.sh            # -> build/lfmsim (qexe)
./runner.sh           # build.sh + qemu + проверка результата
```

Переменные (все необязательные):

```sh
HEXAGON_SDK_ROOT=/opt/hexagon-sdk-6.7.0.0
HEXAGON_TOOLS_VERSION=21.0.04
QEMU_BIN=/tmp/opencode/qemu/build/qemu-system-hexagon
OUT=build
```

`runner.sh` печатает только строки стенда (`SIM ...`), проверяет маркер
`SIM RESULT: PASS` и возвращает 0/1. Код возврата самого QEMU для успеха
равен 0; при ненулевом `main()` sdksim-runelf печатает `User Exit Code` и
падает в teardown (qemu rc=255), поэтому авторитетный признак успеха —
маркер PASS, а не код QEMU.

## Что проверяется

| Группа | Данные | Эталон | Проверка |
|---|---|---|---|
| `gemv` | int4 `[7680][2048]`, M=1/2/4, int32 и int8-выход с масштабами | скалярное произведение на тех же данных (как `host_main.c`) | 0 расхождений, max_abs_err=0 |
| `ffn` | gate/up `[7680][2048]`, down `[2048][7680]`, 2 токена | целочисленная модель `ffn_ref.h` (+ cos к f32-истине) | mid/out бит-в-бит, cos 0.9994 |
| `exec` | одна команда-лист: QUANT → GEMV `2048x2048`, FFN, ATTN, CONV `H=2048`, DUMP | прямые вызовы ядер, формула QUANT, host ring-модель CONV | 0 расхождений; счётчик CONV через `rt[]`; `reset_state` возвращает тот же токен |
| `team` | GEMV `[7680][2048]`, M=1 | 1 поток vs 4 потока | строки бит-в-бит, 3 воркера созданы, fallbacks=0 |

Дополнительно `ffn` сравнивает результат 1 и 4 потоков (бит-в-бит) и
показывает `team_info`: `state`, `created`, `thr_rc`, `hvx_units`.

Фактический вывод последнего прогона:

```
SIM [gemv] int32 M=1: elems=7680 mismatches=0 max_abs_err=0 kernel_min=71299 cycles
SIM [gemv] int32 M=2: elems=15360 mismatches=0 max_abs_err=0 kernel_min=77818 cycles
SIM [gemv] int32 M=4: elems=30720 mismatches=0 max_abs_err=0 kernel_min=172758 cycles
SIM [gemv] int8+scales: mismatches=0/7680 max_err=0
SIM gemv     PASS
SIM [ffn] tok0 single-thread vs integer model: mid mismatches=0/7680 out mismatches=0/2048
SIM [ffn] tok0 cos(out vs f32 truth)=0.999434
SIM [team] after 4-thread FFN: state=1 hvx_rc=32343 mode=1 created=3 threads=4 thr_rc=0 hvx_units=4 rows=2
SIM [ffn] tok0 4-thread vs 1-thread: mid_diff=0 out_diff=0 workers=3 fallbacks=0
SIM [ffn] tok1 4-thread vs integer model: mid mismatches=0/7680 out mismatches=0/2048
SIM ffn      PASS
SIM [exec] exec(token0): dsp=152076 us cmds=6 fallbacks=0
SIM [exec] QUANT vs host formula: mismatches=0/2048
SIM [exec] GEMV(quant-chain) vs direct call: mismatches=0/2048
SIM [exec] FFN vs direct call: mismatches=0/2048
SIM [exec] ATTN vs direct call: mismatches=0/2048
SIM [exec] CONV token0 vs host ring model: mismatches=0/2048
SIM [exec] CONV token1 via rt[] + persistent ring: mismatches=0/2048; other outputs changed=0
SIM [exec] reset_state identity: changed bytes=0/16384
SIM exec     PASS
SIM [team] 4-thread vs 1-thread rows: mismatches=0/7680 workers=3 fallbacks=0 kernel_min=10899 cycles
SIM team     PASS
SIM RESULT: PASS (0/4 groups failed)
```

## Рецепт qexe

Компиляция объекта (HVX обязателен для `lfmgemv_impl.c`):

```sh
$CC -mv68 -mhvx -mhvx-length=128b -G0 -O2 -Wall \
    -I$S/incs -I$S/incs/stddef -I../hvx -I../conv \
    -I$QU -I$QU/qurt -I$QU/posix -c file.c -o file.o
```

`-G0` важен: qexe грузится sdksim-runelf без GP-relative small data.

Линковка:

```sh
$CC -mv68 -g -nodefaultlibs -nostdlib \
    -Wl,--section-start,.interp=0x23000000 \
    -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
    -Wl,--whole-archive \
    $G0/init.o file1.o file2.o \
    $QL/crt1.o $QL/debugmon.o $QL/libqurt.a \
    $G0/libc.a $G0/libqcc.a $G0/libhexagon.a $G0/libm.a \
    $QL/libqurtcfs.a $QL/libtimer_island.a $QL/libtimer_main.a $QL/libposix.a \
    -Wl,--no-whole-archive $G0/fini.o -o app
```

где `QU=$S/rtos/qurt/computev68/include`,
`QL=$S/rtos/qurt/computev68/lib`,
`G0=$S/tools/HEXAGON_Tools/21.0.04/Tools/target/hexagon/lib/v68/G0`.
Предупреждение про undefined weak `__dl_cxa_refcount` — нормальное.

Запуск:

```sh
$QEMU_BIN -M V68N_1024 -display none \
    -kernel $S/rtos/qurt/computev68/sdksim_bin/runelf.pbn \
    -append "./build/lfmsim"
```

## Ограничения стенда и QEMU

* **HMX под QEMU не работает**: ядро падает на HMX-инструкциях
  (exception 0x18, `SSR.XE2`, в окружении отсутствует `hmx_ctx`). Стенд
  проверяет HVX-путь; HMX-ядра (`../hmx`) сюда не входят.
* **HVX работает** (в т.ч. `qurt_hvx_get_units()` возвращает 4 контекста,
  `qurt_hvx_lock/unlock`, `qurt_mem_syncht`).
* **QuRT heap по умолчанию отсутствует**: `sbrk` сразу возвращает
  `out of memory`, поэтому `sim_main.c` объявляет
  `QURT_DECLARE_STATIC_HEAP(64 MiB)`. 96 MiB на этой машине не стартует
  (`failed to allocate virtual memory with MEM_MAPPING_NONE`), 64 MiB —
  максимум. Пиковое потребление тестов ~46 MiB (FFN: модель + резидентные
  веса).
* **Потоки требуют явного стека**: без
  `qurt_thread_attr_set_stack_addr/size` `qurt_thread_create` возвращает -1.
  `lfmgemv_impl.c` это уже делает (96 KiB на воркер).
* **FastRPC/ION отсутствуют**: fd-API (`map_arena`, `alloc_view`,
  `attach_out`, `attach_in`, `alloc_in`) эмулируется в `shim_hap.c`:
  тест регистрирует обычный буфер и получает фейковый fd, а
  `apps_mem_share_map()` возвращает тот же указатель. Семантика
  host-кэша/DMA, которую на устройстве обеспечивает драйвер, здесь не
  моделируется — она и не нужна, потому что память общая.
* **HAP**: `HAP_power_set` — no-op с успехом, `HAP_power_get`/
  `HAP_power_get_max_speed` возвращают константы. `HAP_perf_*` не
  шиммируются: в `HAP_perf.h` это `static inline` (`C31:30`/`C15:14`).
  `lfmgemv_impl.c` не вызывает `FARF`, поэтому лог-шим не нужен.
* **Время**: счётчики — гостевые циклы/тики QEMU, не производительность
  телефона; `dsp=... us` в exec — сумма времени команд внутри эмуляции.
* **ATTN в exec-тесте**: ctx=256 (эмуляция 2048 токенов слишком медленная).
  Проверка — exec против прямого вызова того же ядра; независимого
  CPU-эталона attention в этот стенд не переносили (он есть в `attn_main.c`
  для устройства).
* `team`-тест запускается после `ffn`, поэтому команда уже существует;
  сравнение 1 vs 4 потока всё равно честное: при `threads=1` range-split
  вырождается в один поток, при `threads=4` работают 3 воркера
  (`created=3`, `fallbacks=0`).

## Файлы

* `sim_main.c` — entry point и все группы тестов; статический heap.
* `shim_hap.c` — эмуляция `apps_mem_share_map/unmap` и HAP power.
* `sim_shim.h` — регистрация фейкового fd.
* `build.sh` — компиляция/линковка qexe.
* `runner.sh` — сборка + QEMU + проверка `SIM RESULT: PASS`.
* `build/` — артефакты (игнорируется git).

## Если понадобится hexagon-sim

Стенд рассчитан на QEMU, но тот же qexe запускается в `hexagon-sim` с
cosim-конфигами QuRT:

```sh
printf '%s\n' "$QL/../debugger/lnx64/qurt_model.so" > osam.cfg
printf '%s\n' "$TD/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > q6ss.cfg
printf '%s\n' "$TD/lib/iss/l2vic.so 32 0xFC910000" >> q6ss.cfg
hexagon-sim -mv68 --simulated_returnval --usefs . \
    --cosim_file q6ss.cfg --l2tcm_base 0xd800 --rtos osam.cfg \
    $QL/../sdksim_bin/runelf.pbn -- ./build/lfmsim
```

QEMU-путь проверен и в этом сценарии не нужен; рецепт приведён для
локализации расхождений.
