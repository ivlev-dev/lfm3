# Стенд myelin/dsp

Общая инфраструктура прогона DSP-ядер LFM3 без телефона и FastRPC:
хостовые стенды собираются системным `cc`, DSP-варианты — `hexagon-clang`
из Hexagon SDK и запускаются в `hexagon-sim` или `qemu-system-hexagon`.

Полное описание слоя, форматов и переменных окружения — в
[../README.md](../README.md), контракт ядер — в [../CONTRACT.md](../CONTRACT.md).

## Быстрый старт

```sh
cd myelin/dsp
./run_tests.sh          # единая точка: сборка + прогон + PASS/SKIP/FAIL
make test               # то же
```

Раздельно:

```sh
./sim/build.sh          # build/manifest.txt, build/host/, build/dsp/
./sim/runner.sh         # build.sh + прогон manifest
SKIP_BUILD=1 ./sim/runner.sh
```

## Файлы

* `sim_main.c` — точка входа DSP-образа, слабая extern-таблица стендов,
  QuRT/HVX-shim, потоки, таймер, статическая куча.
* `sim_shim.h` — интерфейс shim, `sim_ion_register`, слабые символы
  `lfm_case_*`.
* `shim_hap.c` — заглушки HAP power и `apps_mem_share_map/unmap`.
* `build.sh` — сборка найденных стендов (`tests/test_common.c`, `hvx/*_main.c`,
  `hmx/*_main.c`), отсутствующие — `SKIP`.
* `runner.sh` — прогон, проверка маркера `SIM RESULT: PASS` и кода возврата,
  сводка, ненулевой код при `FAIL`.
* `build/` — артефакты и логи, в git не входит.

## Результаты

* `PASS` — стенд собран и завершился успешно.
* `SKIP` — файла нет, стенд требует устройства без доступного симулятора,
  либо символ стенда не слинкован.
* `FAIL` — ошибка сборки/линковки, ненулевой код возврата, таймаут или
  `SIM RESULT: FAIL`.

Логи: `build/log/*.log` (сборка) и `build/run/*.log` (прогон).
