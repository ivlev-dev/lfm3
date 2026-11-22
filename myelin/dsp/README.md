# DSP-стенды: HMX, HVX, conv (Hexagon v68)

Отдельные C-стенды для проверки ядер на реальном CDSP (Snapdragon 888, v68).
В cargo workspace не входят и на сборку Rust не влияют — собираются своими
`build.sh` через Hexagon SDK и Android NDK.

```
hmx/          HMX-инструкции: fp16 и int8 матмул, замеры, research/
hvx/          HVX-ядра: int4/int8 GEMV, SiLU/FFN, paged attention, PV
conv/         depthwise shortconv k=7 + офлайн-симулятор (без устройства)
qnn-reverse/  разбор QNN: геометрия int8-кроутонов, control-слова
```

## Что нужно

| компонент | версия | зачем |
|---|---|---|
| Hexagon SDK | 6.7.0.0 | `hexagon-clang` 21.0.04, `qaic`, qurt, `qhl_hmx`/`qhl_hvx` |
| Android NDK | r27c | host-бинарники, `aarch64-linux-android31-clang` |
| adb | — | запуск на телефоне (для сборки не нужен) |

Пути задаются переменными, дефолты — из исследовательской среды:

```bash
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.7.0.0   # дефолт /tmp/opencode/sdk/6.7.0.0
export ANDROID_NDK_ROOT=/path/to/android-ndk-r27c      # дефолт /tmp/opencode/ndk/android-ndk-r27c
# опционально: HEXAGON_TOOLS_VERSION=21.0.04, ANDROID_API=31
```

Проприетарные части SDK **не вендорятся**: `qhl_hmx`/`qhl_hvx` подключаются
заголовками/исходниками прямо из `$HEXAGON_SDK_ROOT`, `libcdsprpc.so` берётся
из `$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64`, генераты
`qaic` (`*_skel.c`, `*_stub.c`, `*.h`) создаются в процессе сборки и лежат в
`myelin/.gitignore`.

## Сборка

```bash
cd myelin/dsp/hmx  && ./build.sh   # libmx4_skel.so + mx4_test
cd myelin/dsp/hvx  && ./build.sh   # liblfmgemv_skel.so + lfmgemv_test/ab_test/ffn_test/attn_test
cd myelin/dsp/conv && ./build.sh   # liblfmgemv_skel.so + conv_sim (conv_test — если есть conv_main.c)
```

Все три собраны на SDK 6.7.0.0 + NDK r27c; устройство не требуется.

## Запуск на телефоне

```bash
adb devices                                   # взять серийник
export MADB=<серийник>                        # иначе берётся первое device из adb devices
cd myelin/dsp/hmx && ./deploy.sh push && ./deploy.sh clk
cd myelin/dsp/hvx && ./run.sh push && ./run.sh attn_test 64 0 2
```

Тесты сами запрашивают **unsigned PD** (`remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`,
на устройстве это `fastrpc_shell_unsigned_3`). Скелет ищется на DSP через
`ADSP_LIBRARY_PATH`, поэтому `deploy.sh`/`run.sh` выставляют его в каталог
`/data/local/tmp/...`; `LD_LIBRARY_PATH=.` нужен hmx-драйверу.
Root не требуется, но нужен рабочий `libcdsprpc.so` в системе и
разрешённый unsigned PD (стандартно на userdebug/инженерных сборках; на
production-прошивке запрос может вернуть ошибку — тогда только подписанный PD).

## Статус

| стенд | состояние | цифры |
|---|---|---|
| HMX fp16 | работает, cos=1.0 | K=32…256, N=32/64, err≤0.0075 |
| HMX int8 | работает побитово (err=0) | 32×2048×32 — 1.075 мкс, 1.95 TMAC/s; 32×2048×64 — 1.612 мкс, 2.60 TMAC/s |
| HVX GEMV int4 | работает | 7680×2048: 0.63 мс, 12.5 GB/s; 2 контекста — 347 мкс |
| HVX FFN | работает, bit-exact | **1.45 мс** на слой (после 2 контекстов) |
| HVX attention | работает | **1178 мкс** @ ctx 2048 |
| conv | симулятор **FAIL по калибровке** (известное состояние) | ring/phase vs shift-register: 0 расхождений; cos(out) до −0.17 |

Подробности и история — в `hmx/research/`, `hvx/README.md`, `conv/README.md`,
`qnn-reverse/REPORT.md`. Пути `/tmp/opencode/...` внутри отчётов относятся к
исходной исследовательской среде, а не к этому репозиторию.
