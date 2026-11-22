# hmx — стенд HMX (Hexagon v68, FastRPC + unsigned PD)

Проверка HMX-инструкций (`mxmem`, `mxclracc`) на реальном CDSP: fp16-матмул
через `qhl_hmx` и int8-матмул по рецепту обратного инжиниринга QNN.

## Файлы

| файл | что |
|---|---|
| `mx4.idl` | FastRPC-интерфейс (`info`, `href`, `i8`, `clk`, `refdump`, …) |
| `mx4_impl.c` | DSP-сторона: загрузка/паковка VTCM, fp16- и int8-пути, замеры |
| `host.c` | host-драйвер: команды, эталон, сравнение |
| `build.sh` | `qaic` + hexagon-clang (v68, `-mhmx`) + NDK host-линк |
| `deploy.sh` | пуш и запуск на телефоне (`MADB`, `MDEV`) |
| `research/` | `REPORT18.md` (итог), `REPORT17.md` (неудачная серия), `FINAL6.md` (ранний `:deep`-тупик) |

`research/FINAL6.md` описывает состояние **до** REPORT18 и оставлен как история:
вывод «HMX не готов» устарел, рабочий рецепт — в `REPORT18.md`.

## Сборка

```bash
HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.7.0.0 ./build.sh
```

Скрипт генерирует `mx4.h`/`mx4_stub.c`/`mx4_skel.c` через `qaic`, собирает
`libmx4_skel.so` и `mx4_test`. Исходники `qhblas_hmx` берутся напрямую из
`$HEXAGON_SDK_ROOT/libs/qhl_hmx/src/qhblas_hmx` — локальная копия `ref/` не
нужна. `libcdsprpc.so` линкуется из
`$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64`.

## Запуск

```bash
export MADB=<adb-серийник>
./deploy.sh push
./deploy.sh clk                     # частота CDSP, pcycles/µs
./deploy.sh info
./deploy.sh hrefall 0 200 1         # fp16 эталон vs HMX
./deploy.sh i8 32 2048 3 3 0x7C 0 0 0 1 1 0 0 0 1000
```

Драйвер сам включает unsigned PD (`DSPRPC_CONTROL_UNSIGNED_MODULE` →
`fastrpc_shell_unsigned_3`). В деплое выставляются `LD_LIBRARY_PATH=.` и
`ADSP_LIBRARY_PATH`.

## Подтверждено (REPORT18)

fp16 32×K × K×N, K=32/64/128/256, N=32/64 — cos=1.000000, err≤0.0075.
int8 32×K × K×N, K=32…2048, N=32/64 — **побитово** (err=0, cos=1.0).

Рабочая последовательность: `qurt_hvx_lock` → `transfer_row` →
`transfer_column` → `qurt_hvx_unlock`, затем `HAP_power_set(HAP_power_set_HMX, 1)`
до первой HMX-инструкции и `compute_resource_hmx_lock`. Контрольные слова
(`ncr = K>>5`): `act_rt = (ncr<<11)|0x7C`, `wgt_rt = (ncr<<11)-1`, `store_rt = 0`;
`:deep` — только на активации, вес без `:deep`.

Темп: 19 тактов на кроутон (65536 MAC) ≈ 3.9 TMAC/s «сырых», 1.95 TMAC/s
полезных при 32 spatial. Prefill FFN 128×7680×2048 — 774 мкс (2.6 TMAC/s,
в 3.1 раза быстрее QNN 2.44 мс). Для M=1 HMX невыгоден — декод на HVX.

## Грабли (не наши)

1. `qhblas_hmx_transfer_column_ahf` при `N<=32` копирует `K/32` векторов вместо
   `K*32` элементов — вес получается мусорным (cos≈0.145). Паковать `N=64`.
2. `l2fetch` в том же коде игнорирует шаг строки: для K=224/256 уходит на ~7 КБ
   за буфер; без guard-хвоста 64 КБ — TLBMISS и падение PD.
3. `pcycles` не тикает, пока ядро ждёт HMX: одиночный замер врёт (170 тактов vs
   честные 1217 на K=2048), мерить в цикле (`reps`).
