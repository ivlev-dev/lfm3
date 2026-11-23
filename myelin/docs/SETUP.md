# Установка окружения Myelin с нуля

Документ рассчитан на нового разработчика: что скачать, куда положить, какие
переменные выставить и как проверить, что всё живо. Все пути ниже — пример для
Linux x86_64; на другой машине меняются только префиксы.

Проверено на Arch Linux, ядро 6.x, 32 ГБ RAM. Минимум: 16 ГБ RAM,
~40 ГБ диска (Rust target + SDK + модель + эмуляторы).

---

## 0. Что понадобится

| Компонент | Версия | Зачем | Откуда |
|---|---|---|---|
| Rust | stable ≥ 1.85 (edition 2024) | весь хост-код | https://rustup.rs |
| Android NDK | **r30** (подойдёт r27+) | host-тесты под телефон, `libmyelin.so` | https://developer.android.com/ndk/downloads |
| Android SDK + JDK | JDK 17, SDK 35 | сборка APK | Android Studio или `sdkmanager` |
| Hexagon SDK | **6.7.0.0** | DSP-ядра, `hexagon-clang` 21.0.04, `hexagon-sim`, QuRT | Qualcomm Software Center (см. §2) |
| QEMU | ветка master ≥ 11.2.0-rc0 | Hexagon v68 + QuRT локально | исходники, см. §4 |
| adb | любой свежий | телефон | `android-tools` / `platform-tools` |
| Модель | `LFM3-1.7B-Q4_K_M.gguf` | веса | Hugging Face, см. §3 |

QAIRT/QNN SDK **не нужен** для сборки и работы: он лежит в репозитории только
как справочный материал для реверса (архив `v2.52.0.261104.zip` в корне можно
не распаковывать).

---

## 1. Переменные окружения

Добавьте в `~/.bashrc` (или `~/.config/environment.d/`):

```bash
export ANDROID_NDK_ROOT=/opt/android-ndk
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_API=31                 # DSP-стенды; для myelin-ffi достаточно 24
export HEXAGON_SDK_ROOT=/opt/hexagon-sdk-6.7.0.0
export HEXAGON_TOOLS_VERSION=21.0.04  # каталог внутри SDK
export QEMU_BIN=/opt/qemu/bin/qemu-system-hexagon
export MADB=                          # серийник adb; пусто = первое устройство
```

`MADB` нужен только при нескольких подключённых устройствах:

```bash
adb devices
export MADB=<серийник из вывода>
```

---

## 2. Hexagon SDK 6.7

SDK распространяется через **Qualcomm Software Center**. Прямых ссылок нет —
каталог отдаёт файл только авторизованному клиенту.

### Вариант A: `qsc-cli` (консоль, работает на Arch)

```bash
# клиент качается без авторизации
curl -L -o qsc.deb \
  https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Software_Center/Linux/Debian/latest.deb
mkdir -p qsc && bsdtar -xf qsc.deb -C qsc && bsdtar -xf qsc/data.tar.* -C qsc
sudo cp -r qsc/opt/qcom/softwarecenter /opt/softwarecenter

export PATH=/opt/softwarecenter/bin/qsc-cli:$PATH
qsc-cli login            # интерактивно: выдаст URL + код, подтвердить в браузере
qsc-cli info product     # список доступных продуктов
```

`qsc-cli` показывает только chip-software (BSP). **SDK качается через QPM3**
(`qpm-cli` в том же пакете) либо через веб-каталог:

- каталог: https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK
- версия: **6.7.0.0**, файл `Hexagon_SDK_lnx.zip` (~1.5 ГБ)

После скачивания:

```bash
sudo mkdir -p /opt/hexagon-sdk-6.7.0.0
sudo bsdtar -xf Hexagon_SDK_lnx.zip -C /opt/hexagon-sdk-6.7.0.0 --strip-components=1
# проверка
ls $HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools/bin/hexagon-clang
ls $HEXAGON_SDK_ROOT/rtos/qurt/computev68/sdksim_bin/runelf.pbn
```

### Вариант B: GitHub-зеркало

Для 6.x есть зеркала (`Hexagon_SDK` в открытых репозиториях). Версия должна
быть ≥ 6.6: в 6.7 есть `libs/qhl_hmx` (HMX-хелперы) и `qhl_hvx`.
SDK 5.x **не нужен** (он требовался только для QNN QEMU, от которого отказались).

### Важно про путаницу путей

В make-системе SDK переменная `HEXAGON_TOOLS_ROOT` указывает на каталог
**без** `/Tools`, а в наших скриптах `TD=$S/tools/HEXAGON_Tools/$VER/Tools`.
Если собираете SDK-примеры, выставляйте:

```bash
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION
```

---

## 3. Модель

```bash
cd <корень репозитория>
mkdir -p models/hf
curl -L -o models/LFM3-1.7B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf
for f in tokenizer.json chat_template.jinja config.json; do
  curl -L -o models/hf/$f https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/$f
done
```

Размер GGUF ≈ 1.1 ГБ. Путь по умолчанию — `models/`, переопределяется в
конфиге/CLI приложения `lfm`.

---

## 4. QEMU для Hexagon v68

Системный QEMU Hexagon v68 не знает; нужна сборка из master (≥ 11.2.0-rc0),
где есть машина `V68N_1024`. Собираем только hexagon-таргет.

```bash
sudo pacman -S --needed git ninja meson python glib2 pixman        # Arch
# Ubuntu: sudo apt install git ninja-build meson python3 python3-venv libglib2.0-dev libpixman-1-dev

git clone --depth 1 https://gitlab.com/qemu-project/qemu.git /tmp/qemu
cd /tmp/qemu
./configure --target-list=hexagon-softmmu \
            --disable-werror --disable-docs --disable-tools \
            --without-default-features --enable-tcg \
            --prefix=/opt/qemu
make -j$(nproc)
sudo make install
qemu-system-hexagon -M help | grep V68N_1024
```

Ожидаемый вывод: `V68N_1024  Hexagon V68N_1024`.

Ограничение: **HMX в QEMU не эмулируется** (нет `hmx_ctx`, исключение 0x18).
QEMU используем для QuRT/HVX/exec; HMX — в `hexagon-sim` или на телефоне.

---

## 5. Сборка хоста

```bash
# приложение lfm (OpenCL-эталон) — из корня
cargo build --release
cargo test --workspace

# рантайм Myelin (Android/DSP)
cd myelin
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Генерация Kotlin-биндингов и `libmyelin.so`:

```bash
cd myelin
ANDROID_NDK_HOME=$ANDROID_NDK_ROOT ./build-android.sh
```

Сборка APK:

```bash
cd myelin/android
cat > local.properties <<EOF
sdk.dir=$ANDROID_SDK_ROOT
EOF
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## 6. DSP-стенды

Все стенды — отдельные мини-проекты, SDK-зависимые. Собираются без телефона.

### 6.1 HVX/exec на телефоне

```bash
cd myelin/dsp/hvx
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT ./build.sh
# появляются: liblfmgemv_skel.so, lfmgemv_test, exec_test, attn_test, ffn_test, ...
adb push liblfmgemv_skel.so exec_test attn_test /data/local/tmp/lfmgemv/
adb shell "cd /data/local/tmp/lfmgemv && ADSP_LIBRARY_PATH=\$PWD ./exec_test"
```

### 6.2 HVX/exec в QEMU (без телефона)

```bash
cd myelin/dsp/sim
QEMU_BIN=$QEMU_BIN ./runner.sh        # ~25 с, проверяет маркер SIM RESULT: PASS
```

### 6.3 HMX на телефоне

```bash
cd myelin/dsp/hmx
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT ./build.sh
export MADB=<серийник>
./deploy.sh push
./deploy.sh info
./deploy.sh i4 32 32 2048 4 0x6000 -1 0 200     # int4 K=2048, ожидается MATCH
```

### 6.4 HMX в hexagon-sim (без телефона)

```bash
cd myelin/dsp/hmx/sim
./build.sh && ./run.sh
# ожидается: RESULT: OK (VTCM + HMX functional)
```

Три вещи, без которых HMX в симуляторе не работает:

1. `--mhmx 1` у `hexagon-sim`;
2. бит **SSR.XE2 = 26** в программе (`ssr = rN`);
3. адреса буферов в окне VTCM **`0xd8400000…0xd87fffff`** и обязательный
   `bias = mxmem2()` с 32 словами `0x6000`, иначе конвертер пишет нули.

---

## 7. Телефон

Требуется **root** (unsigned PD `fastrpc_shell_unsigned_3`).

```bash
adb devices                       # либо adb connect <ip>:5555, либо mDNS
adb shell getprop ro.product.model
```

Особенности, проверенные на Snapdragon 888 (Xperia 1 III):

- в unsigned PD доступно до **512 МиБ** одним `memalign`, суммарно ~1.5 ГиБ;
- **несколько `remote_handle_open` в одном процессе = одна PD** (session 0);
  для нескольких PD нужны разные процессы;
- `remote_register_buf_attr` с `KEEP_MAP` **портит данные** на этой
  `libcdsprpc.so` — не использовать;
- если `.so` и хост разошлись по версии протокола, `exec` вернёт
  `AEE_EBADPARM` — пересоберите и залейте оба.

---

## 8. Проверка «всё готово»

```bash
# 1. хост
cd myelin && cargo test --workspace

# 2. SDK
ls $HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04/Tools/bin/hexagon-clang

# 3. QEMU
$QEMU_BIN -M help | grep V68N_1024

# 4. HVX без телефона
cd myelin/dsp/sim && ./runner.sh | tail -1     # SIM RESULT: PASS

# 5. HMX без телефона
cd myelin/dsp/hmx/sim && ./run.sh | tail -1    # RESULT: OK

# 6. телефон
adb devices
cd myelin/dsp/hmx && ./deploy.sh push && ./deploy.sh info
```

---

## 9. Частые проблемы

| Симптом | Причина | Решение |
|---|---|---|
| `exception 0x18` (HMX) | не выставлен SSR.XE2 | `ssr = rN`, бит 26 (в симуляторе — до первой `mxmem`) |
| `exception 0x26` | адрес вне VTCM | окно `0xd8400000…0xd87fffff` |
| HMX пишет нули | не загружен bias/scale | `mxmem2()` с 32 словами `0x6000` |
| `qurt_thread_create rc=-1` | не задан стек | `qurt_thread_attr_set_stack_addr/size` |
| `argv` под `runelf.pbn` мусорный | sdksim не передаёт argv | не полагаться на argv, отдельные бинари |
| `AEE_EBADPARM` на 32-м токене | старый скелет без глобального KV-слота | пересобрать `liblfmgemv_skel.so` (см. `docs/kv-batch-contract.md`) |
| Текст вырождается в повтор | per-row масштаб int4 | продуктовый формат — **Q4_K**, см. `docs/HANDOVER.md` |
| `remote_register_buf_attr` + `KEEP_MAP` | драйвер не инвалидирует кэш | не использовать, дефолт — без регистрации |
| `cargo test` не находит пакет | запуск из корня, а workspace в `myelin/` | `cd myelin && cargo test --workspace` |

---

## 10. Что читать дальше

- `myelin/docs/HANDOVER.md` — состояние проекта, решения, открытые задачи.
- `myelin/docs/README.md` — индекс всей документации.
- `myelin/docs/architecture.md` — слои и поток одного шага.
- `myelin/dsp/*/research/REPORT*.md` — журнал экспериментов с числами.
