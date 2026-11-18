# Бейзлайн llama.cpp (18.11.2026)

Машина: AMD RX 6400 (RADV NAVI24), Intel i5-10400 (12 потоков), 31 ГБ RAM.
Модель: `LFM3-1.7B-Q4_K_M.gguf` (1.02 GiB).
llama.cpp: master `ad9d9e8` (18.11.2026), CPU и Vulkan сборки.

```bash
# CPU
cmake -B build -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12 --target llama-bench

# Vulkan
cmake -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j12 --target llama-bench
```

Замеры (`llama-bench`, reps 2, threads 12, ngl 99 для Vulkan):

| backend | тест | tok/s |
|---|---|---|
| CPU | pp512 | 141.53 ± 51.12 |
| CPU | tg128 | 23.30 ± 0.26 |
| Vulkan | pp512 | **1393.39 ± 34.59** |
| Vulkan | tg128 | **91.35 ± 1.17** |
| Vulkan | pp8192 | **1235.93** |
| Vulkan | tg64 (ctx 8192) | **94.47** |

## Наши текущие числа (для сравнения)

| бэкенд | prefill | decode |
|---|---|---|
| OpenCL (Q4_K_M, GPU) | 512: ~205 tok/s, 8192: ~159 tok/s | 8192: ~64 tok/s |
| CPU (W4A8/AVX2) | 512: 10.0 tok/s | 6.9 tok/s |
| ROCm | нет (ROCm не установлен) | — |

## Разрывы и цели

| бэкенд | prefill | decode | цель |
|---|---|---|---|
| CPU | 14× медленнее | 3.4× медленнее | ≥ 142 pp512, ≥ 24 tg |
| OpenCL | 6.8× медленнее (512), 7.8× (8192) | 1.4× медленнее | ≥ 1400 pp, ≥ 95 tg |
| ROCm | нужно поднять ROCm + forward | — | ≥ 1400 pp, ≥ 95 tg |

Вывод: decode у OpenCL уже близко к llama.cpp Vulkan; главный разрыв — **prefill**
(GEMM/батчевый Q4) на всех бэкендах. На CPU нужен батчевый AVX2-GEMM и лучшая
упаковка весов; на GPU — GEMM-тайлинг и subgroup-редукции, как в llama.cpp Vulkan.
