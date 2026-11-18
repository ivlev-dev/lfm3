# Android / ARMv7 (Sony Xperia Z3)

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
| наш (NEON, `arm_neon.c`) | **1.1–1.3 tok/s** | **0.50 tok/s** |
| разрыв | ×3.3 | ×4.6 |

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

## OpenCL на Adreno 330 — нецелесообразно

Драйвер `libOpenCL.so` — 2013 год (Adreno 330, OpenCL 1.2-эпоха). Наши ядра написаны под
OpenCL 3.0: `cl_khr_integer_dot_product` (`dot(char4…)`), подгруппы
(`get_sub_group_id`, `sub_group_reduce_add`), fp16-вектор. В драйвере есть только
`clGetKernelSubGroupInfoKHR`; `cl_khr_integer_dot_product` отсутствует. Порт потребовал бы
эмуляции dot через `vmlal`-последовательности внутри OpenCL C и замены subgroup-редукций
на LDS-вариант — то есть фактически второго набора ядер. Плюс известная нестабильность
драйверов Adreno 3xx. Приоритет — CPU-путь (llama.cpp на этом устройстве тоже CPU-only).

## Следующие шаги под ARM

1. Хранить веса в формате GGUF (144 Б/блок) и распаковывать на лету из mmap — минус 33%
   трафика и 1.2 ГБ RAM.
2. NEON для attention (`vmulq_f32`/`vaddq_f32`, блокировка по 4 позиции) и норм/квантайза.
3. Упаковка Q6 (320 → 256 Б) и предзагрузка строк (prefetch на следующую строку).
4. При желании — OpenCL 1.2 fallback-набор ядер (эмуляция dot, LDS-редукции).
