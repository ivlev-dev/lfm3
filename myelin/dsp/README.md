# DSP-слой LFM3

Слой целочисленных DSP-ядер LFM3: int4-веса с поворотом Hadamard, int8-активации,
paged KV, HVX-путь для decode и HMX-путь для prefill. Ядра собираются как
обычные C11-файлы: на хосте системным `cc`, на Hexagon — `hexagon-clang` из
SDK.

Текущее состояние проверок, известные зависания стендов и следующие шаги —
в [STATUS.md](STATUS.md).

## Структура

```
myelin/dsp/
  CONTRACT.md         контракт форматов, ABI, ротации и гейтов
  common/             header-only примитивы: fp16, q4, quant, rot, kvarn, page
  hvx/                HVX-ядра (gemv/gemm/ffn/attn/kvcache/exec) и стенды *_main.c
  hmx/                HMX-ядра (gemm i4/i8, prefill) и стенд gemm_main.c
  tests/              test_common.c, эталон kvarn_case.h
  sim/                общий стенд: shim QuRT/HVX, sim_main.c, build.sh, runner.sh
  run_tests.sh        единая точка прогона
  Makefile            цели test, host, dsp, build, clean
```

Форматы данных, раскладка весов и KV, ABI команд и числовые гейты описаны в
[CONTRACT.md](CONTRACT.md). Кратко: веса `int4` блоками по 32 со `scale` f16,
активации `quantize_q8` блоками по 32, поворот `R = H128 D / sqrt(128)` для
весов, активаций и KV, страница KV — 32 токена x 128 измерений на голову.

## Сборка и запуск

```sh
cd myelin/dsp

./run_tests.sh        # собрать и прогнать всё, сводка PASS/SKIP/FAIL
make test             # то же
make host             # только хостовые стенды
make dsp              # только DSP-стенды (hexagon-sim/QEMU)
make clean            # удалить sim/build
```

Раздельно:

```sh
./sim/build.sh        # собрать всё, что нашлось, в sim/build/
./sim/runner.sh       # build.sh + прогон + сводка
SKIP_BUILD=1 ./sim/runner.sh
```

`build.sh` сам находит существующие стенды:

* `tests/test_common.c` — хостовый, всегда;
* `hvx/*_main.c` — HVX, требует Hexagon (собирается только для DSP);
* `hmx/*_main.c` — HMX, требует Hexagon/hexagon-sim.

Если файла нет или символа стенда нет в образе, строка получает `SKIP`.
Ненулевой код возврата, падение, таймаут или маркер `SIM RESULT: FAIL`
дают `FAIL`; лог пишется в `sim/build/log/` и `sim/build/run/`.

### Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `HOST_CC` | `cc` | компилятор хостовых стендов |
| `HEXAGON_SDK_ROOT` | первый `/opt/hexagon-sdk-*` | SDK с `rtos/qurt/computev68` |
| `HEXAGON_TOOLS_VERSION` | найденная | версия `HEXAGON_Tools` |
| `LFM_SIM_MODES` | `host,dsp` | какие режимы собирать и запускать |
| `LFM_SIM_DSP_ALL` | `0` | `1` — собирать DSP-вариант и для хостовых стендов |
| `LFM_SIM_TIMEOUT` | `300` | таймаут одного стенда, секунды |
| `LFM_SIM_USE_QEMU` | `0` | `1` — предпочесть `qemu-system-hexagon` вместо `hexagon-sim` |
| `QEMU_BIN` | поиск в PATH | путь к `qemu-system-hexagon` |
| `SKIP_BUILD` | `0` | `1` — не пересобирать перед прогоном |

## Какие стенды требуют устройства

* `tests/test_common.c` проверяет `common/` (fp16, rot, kvarn, q4, page) и
  проходит на хосте обычным `cc`.
* `hvx/*` и `hmx/*` используют HVX/HMX-интринсики и QuRT. На хосте они
  помечаются `SKIP (device-only)`, на DSP запускаются через `hexagon-sim`
  (по умолчанию) или `qemu-system-hexagon -M V68N_1024` (если включён
  `LFM_SIM_USE_QEMU=1` и QEMU поддерживает эту машину).
* HMX под `qemu-system-hexagon` не работает; HMX-стенды рассчитаны на
  `hexagon-sim`. Если симулятора нет, DSP-строки получают `SKIP`.

`runner.sh` копирует DSP-образ и конфиги симулятора в ASCII-каталог
`${TMPDIR:-/tmp}/lfm3-dsp-sim-<pid>`: `runelf.pbn` из SDK не открывает образы
по путям с не-ASCII символами.

## Устройство sim/

* `sim_main.c` — точка входа DSP-образа. Держит слабую extern-таблицу
  стендов (`lfm_case_common`, `lfm_case_gemv`, ...): если символ не слинкован,
  печатается `SKIP`, процесс не падает. Каждый стенд запускается в отдельном
  потоке со стеком 1 МиБ; там же QuRT/HVX-shim (`sim_hvx_lock/unlock/units`,
  потоки, таймер), статическая куча QuRT (56 МиБ) и `sim_ion_register`.
* `sim_shim.h` — интерфейс shim и объявления слабых точек входа.
* `shim_hap.c` — заглушки HAP power и `apps_mem_share_map/unmap` для QuRT.
* `build.sh` — хостовые бинарники (`cc`) и DSP qexe (`hexagon-clang` + QuRT):
  стенд компилируется с `-Dmain=lfm_case_<slug>`, ядра группы (`hvx/` или
  `hmx/`) собираются в `libkernels.a` и линкуются только по нужным символам.
* `runner.sh` — прогон manifest из `build/`, проверка кода возврата/маркера,
  сводка и ненулевой код при `FAIL`.

Чтобы добавить новый стенд, достаточно положить `<name>_main.c` в `hvx/`,
`hmx/` или `tests/`; для нового имени нужно добавить slug в `slug_for()` в
`sim/build.sh` и слабое объявление/строку таблицы в `sim/sim_main.c`.
