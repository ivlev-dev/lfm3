# Мультибэкендный план (SPIR-V / Vulkan / ROCm)

Проверка осуществимости (18.11.2026, RX 6400 / RADV NAVI24 / Rusticl):

| путь | статус | факт |
|---|---|---|
| **Свой SPIR-V через OpenCL** (`clCreateProgramWithIL`) | **заблокировано** | Rusticl не экспортирует `clCreateProgramWithIL` (0 вхождений в `libRusticlOpenCL.so`). Загрузить SPIR-V некуда. Альтернативы: проприетарный AMD OpenCL (нет), PoCL (только CPU). |
| **Vulkan** | **доступно** | `vulkaninfo` → `AMD Radeon RX 6400 (RADV NAVI24)`, `glslc`/`glslangValidator` есть, заголовки есть. Каркас бэкенда восстановлен из истории (`crates/backend-vulkan`, `ash` 0.38, GLSL→SPIR-V через `build.rs`), Q4_K-GEMV считается бит-в-бит с CPU (max_rel 1.4e-6), `vulkan-info`/`vulkan-gemv` работают. |
| **ROCm** | **не установлен** | `/dev/kfd` есть, но нет `/opt/rocm`, `hipcc`, `rocminfo`. gfx1034 (Navi 24) официально не поддерживается — понадобится `HSA_OVERRIDE_GFX_VERSION=10.3.x` и установка ROCm (несколько ГБ). |

## Почему вообще SPIR-V/Vulkan

Компилятор Rusticl (LLVM, путь OpenCL C → SPIR-V → LLVM IR → AMDGPU) генерирует для наших
GEMM-ядер код с 15k инструкций на 1040 dp4a и спилами (256 VGPR, private 176–352 Б), тогда как
llama.cpp Vulkan (glslc → SPIR-V → RADV/ACO) на той же карте выдаёт 31% dp4a, 0 спилов и
**2.7 T MAC/s против наших 1.5 T**. Отсюда гипотеза: разрыв в кодогенерации, а не в алгоритме.

Первый дешёвый тест: поднять в Vulkan **Q4-GEMM** (порт `lds17`: тайл 64×32, WG 512,
4 аккумулятора, int8/dp4a через `GL_EXT_shader_explicit_arithmetic_types_int8` +
`dotPacked4x8EXT`/эмуляцию) и сравнить с OpenCL на тех же формах.

## Что уже сделано

- `crates/backend-vulkan` восстановлен (`ash` 0.38, build.rs с `glslc`), сборка проходит.
- `lfm-bench vulkan-info` / `vulkan-gemv` возвращены в CLI.
- `vulkan-gemv` проверяет Q4_K-GEMV против CPU-декванта: max_rel 1.4e-6.
- Известная проблема каркаса: повторные запуски падают в `ERROR_OUT_OF_POOL_MEMORY`
  (descriptor pool не сбрасывается) — нужно пересоздавать pool на запуск.

## Следующие шаги

1. Починить descriptor pool в Vulkan-каркасе, добавить замер GEMV (decode-путь).
2. Порт Q4-GEMM (`lds17`) в Vulkan + A/B против OpenCL на `ffn_up`/`attn_q` (512/8192).
3. Если гипотеза подтвердится — портировать остальные ядра (Q6-GEMM, attention, нормы,
   silu, quant, conv, lm_head) и получить полный Vulkan-форвард.
4. ROCm — отдельной веткой, требует установки ROCm и неподдерживаемой gfx1034;
   начинать только после решения по Vulkan (тот же порт ядер на HIP).
5. SPIR-V-путь закрыть до появления OpenCL-драйвера с `clCreateProgramWithIL`.
