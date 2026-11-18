# Android / ARMv7 (Sony Xperia Z3)

> Код ARMv7 (CPU NEON + OpenCL 1.1) вынесен в отдельный репозиторий `lfm-arm-opencl`.
> Здесь сохранены замеры платформы и выводы.

Устройство: **Sony Xperia Z3** (LineageOS 11, Android 11, userdebug + root).

| параметр | значение |
|---|---|
| SoC | Snapdragon 801 (MSM8974), 4×Krait 400 @ 2.26 ГГц, ARMv7-A + NEON |
| GPU | Adreno 330 (драйвер OpenCL 2013 г.) |
| RAM | 2.7 ГБ видимых |
| ABI | armeabi-v7a (32 бита) |
| OpenCL | `/vendor/lib/libOpenCL.so` присутствует |

Модель: `LFM3-1.7B-Q4_K_M.gguf` (1.02 ГиБ).

## Сборка

```bash
# Rust-цель
rustup target add armv7-linux-androideabi

# .cargo/config.toml
[target.armv7-linux-androideabi]
linker = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang"
rustflags = ["-C", "target-feature=+neon,+vfp4"]

# env для C-зависимостей (onig/tokenizers) и NEON-ядер
export CC_armv7_linux_androideabi=.../armv7a-linux-androideabi24-clang
export AR_armv7_linux_androideabi=.../llvm-ar
cargo build --release -p lfm-bench-cpu --target armv7-linux-androideabi
adb push target/armv7-linux-androideabi/release/lfm-bench-cpu /data/local/tmp/lfm-cpu

# llama.cpp
cmake -B build-armv7 -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-24 \
  -DGGML_NATIVE=OFF -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
```

## Замеры (CPU, 4 потока)

| движок | prefill 32 | decode |
|---|---|---|
| llama.cpp `ad9d9e8` (Q4_K_M, NEON) | **3.78 tok/s** | **2.28 tok/s** |
| наш (скалярный Rust) | 0.5 tok/s | — |
| наш (NEON, C++, 2 потока Rayon) | 1.1–1.4 tok/s | 0.51–0.54 tok/s |
| наш (NEON, C++, **4 потока**) | **2.0–2.1 tok/s** | **0.85–0.91 tok/s** |
| разрыв | ×1.8 | ×2.6 |

Загрузка модели: ~9 с (GGUF 1.1 ГБ mmap + предпак 1.2 ГБ в RAM).

## Что сделано под ARM

- `crates/backend-cpu/src/arm_neon.c` + `build.rs` (cc): NEON-ядра `lfm_q4_rows`/`lfm_q6_rows`
  на `vmlal.s16`, распаковка нибблов `vand`/`vshr`, до 8 токенов на проход (prefill) и
  1 токен (decode), `__builtin_prefetch` на 2 блока вперёд.
- Rust stable на armv7 не даёт NEON-интринсики (`stdarch_arm_neon_intrinsics` — nightly),
  поэтому горячие циклы на C; x86-путь (AVX2) не затронут.
- Активации и раскладка весов общие с x86 (Q4 — нибблы, Q6 — 1 байт/значение).

## Почему всё ещё ×3–5

1. **Трафик весов**: 192 Б на 256 значений (Q4) против 144 Б у GGUF (+33%), и предпак
   занимает 1.2 ГБ RAM при 2.7 ГБ у устройства (llama.cpp читает mmap-GGUF 1.02 ГБ).
2. **MLP/латентность**: 4 ядра Krait без SMT, узкие 32-битные загрузки; каждое чтение
   блока весов — зависимая нагрузка. llama.cpp использует cpuid-подобранный путь и
   лучше перекрывает латентность.
3. **Скалярные вспомогательные пути**: attention (f32 dot по окну), rms_norm, quantize,
   silu остались скалярными; на длинном контексте attention заметен.
4. **NEON-плотность**: `vmlal.s16` даёт 4 MAC на инструкцию (у AVX2 `maddubs`+`madd` —
   32 MAC на 2 инструкции), поэтому потолок armv7 ниже.

## mmap-режим (`LFM_MMAP`)

Опция в `backend-cpu`: веса не предпаковываются, а читаются напрямую из mmap-GGUF
(Q4_K 144 Б/блок, Q6_K 210 Б/блок). Включена по умолчанию на ARM; на x86 — по `LFM_MMAP=1`
(там пока скалярный raw-путь, только для корректности).

Эффект на Z3:

| | предпак | mmap |
|---|---|---|
| загрузка модели | 9.0 с | **0.38 с** |
| RSS процесса | 1.50 ГБ | **167 МБ** |
| major page faults | ~2300/с (трэшинг) | **0** |
| prefill 32 | 1.0–1.3 | 1.1–1.4 |
| decode | 0.48 | 0.51–0.54 |

То есть RAM-давление убрано полностью; узким местом остаётся само ядро.

## OpenCL на Adreno 330 — РАБОТАЕТ (но не для int8)

Драйвер исправен: `libOpenCL.so` сам инициализирует `libCB.so`/`libgsl.so`/`/dev/kgsl-3d0`.
Прежний SIGSEGV был **багом нашего пробника**: `cl_device_type` — 64-битный, а `devices`
передавался как NULL (драйвер пишет по указателю без проверки). После исправления:

- Platform `OpenCL 1.2 QUALCOMM #cde337a (04/04/16)`, устройство `OpenCL 1.1 Adreno(TM) 330`,
  OpenCL C 1.1, 4 CU, WG ≤ 512, local 8 КБ, unified memory.
- Расширения: `cl_khr_fp16`, `byte_addressable_store`, `gl_sharing`, атомики int32.
  **Нет** `cl_khr_subgroups` (`get_sub_group_size()` не компилируется), **нет** dot-product,
  нет fp64, нет раздельной компиляции.
- Замеры: fp32 `float4` — 43.8 GFLOPS; `int4 mad24` — 22.4 GMAC/s; скалярный `fma()` —
  программная эмуляция (0.1 GFLOPS, использовать нельзя); копирование — 5.4 / 5.2 GB/s;
  int8 GEMV через `convert_int4` — **0.72 GMAC/s ≈ одно ядро CPU**.

Вывод: для наших W4A8-ядер (int8/dot) GPU бесполезен; имеет смысл только fp32/int32-путь,
а он упирается в 5.2 GB/s unified memory (1 ГБ весов на токен → ≥0.2 с/токен).
Поэтому приоритет — NEON, а CL1.1/1.2-бэкенд — как отдельная ветка с fp32-ядрами.

## DSP (Hexagon v5) — невозможно

- Подпись ELF проверяется прошивкой ADSP (`sigverify.c`, `SigVerify_AuthenticateELF`,
  `sigverify failed and no valid testsig found`).
- «Unsigned PD» на msm8974 отсутствует (нет ни в `libadsprpc.so`, ни в ядре/прошивке).
- На устройстве нет ни одной skel-библиотеки (`find / -name '*skel*'` пусто).
- Публичного тулчейна под v5 нет (SDK целится в v60+ с HVX).
- Плюс HVX в v5 отсутствует: ~1–2 GOPS int16 — медленнее NEON.

PoC: RPC-канал работает (`remote_handle_open("securitytests")` → 0), но загрузка своего
модуля упирается в подпись.

## Главная находка: Rayon брал 2 потока из 4

`/sys/devices/system/cpu/online` на Android отдаёт `0` (ядра хотплагятся, когда простаивают),
поэтому `available_parallelism`/пул Rayon по умолчанию = **2**. Исправлено: размер пула
берётся из `/sys/devices/system/cpu/present` (0-3) с override `LFM_THREADS`.

Эффект: prefill 1.2 → **2.1**, decode 0.54 → **0.85** tok/s (×1.75). Это самое дешёвое
и самое результативное изменение за всю сессию по CPU.

Дальше пробовали (эффекта не дало):
- int-эпилог как у llama.cpp (per-256 скейл активаций + целочисленные скейлы весов):
  ARM 2.1 → 2.0 prefill, 0.85 → 0.89 decode (в пределах шума); x86 57.0 → 58.5 tok/s.
- chunking строк по 16 (локальность prefetch): 0.
- PMU: IPC 0.3, L1-промахов почти нет, 12% branch-miss → упор в латентность NEON-цепочек,
  а не в память и не в кэш.

## Что переносится куда (замеры)

| оптимизация | Adreno/ARMv7 CPU | Desktop x86 CPU | Desktop GPU | Adreno GPU |
|---|---|---|---|---|
| пул потоков по `present` | **+75%** | 0 (уже 12) | — | — |
| per-256 активации + int-эпилог | 0 (шум) | 0 (57.0→58.5) | 0 | 0 |
| mmap весов | **load 9→0.38 с, RSS 1.5 ГБ→167 МБ** | load 0.74→0.03 с, но −92% скорости (нет AVX2-raw) | +3% VRAM, быстрый load | быстрый load |
| int8 NEON (`vmull_s8`+`vpadalq`) | 0 (эпилог/латентность) | n/a | — | — |

Вывод: **на десктопе эти приёмы не дают ничего** — там уже 12 потоков, AVX2-эпилог
векторизован, а mmap без AVX2-raw только вредит (нужны raw-ядра для AVX2, чтобы получить
−25% трафика и −1.6 ГБ RAM).

## Почему ядро всё ещё медленное (разбор)

Замеры на устройстве: 4 ядра @2.46 ГГц (не троттлит), NEON int8-потолок ≈ 10 GMAC/s на ядро
(микробенч субагента: 13.6–15.2 GOPS при 1.5 ГГц), а мы получаем ~0.5 GMAC/s на ядро.
Сделано и не помогло: int8-путь `vmull_s8`+`vpadalq` с двумя аккумуляторами, prefetch,
снижение трафика через mmap. Профиль simpleperf: 90% времени — в самих ядрах.

Гипотеза: (а) скалярный float-эпилог на каждую 32-группу (2 загрузки + cvt + 4 mul/add)
съедает столько же, сколько dot; (б) зависимая цепочка `vpadalq`/`vmlal` при 1 NEON
инструкции/такт; (в) нет переиспользования `x` между строками. Дальше нужен аккуратный
разбор по счётчикам (циклы на группу) — это следующий шаг.

## Следующие шаги под ARM

1. Разбор цикла по счётчикам: сколько тактов на 32-группу, где stalls (simpleperf + PMU).
2. Убрать скалярный эпилог: считать 4 группы сразу в 4 дорожках int32 и применять скейлы
   векторно (смена раскладки весов/активаций под 4-дорожечный аккумулятор).
3. NEON для attention/rms_norm/quantize.
4. OpenCL 1.1/1.2-бэкенд — отдельная ветка с fp32-ядрами (int8 на Adreno 330 бесполезен).
