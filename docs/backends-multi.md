# Мультибэкендный план — закрыт

Проверены три пути. Все три работают технически, но **ни один не быстрее** текущего OpenCL C
на Rusticl, поэтому отдельные бэкенды удалены из репозитория (история — в git).

| путь | что выяснилось |
|---|---|
| **Свой SPIR-V через OpenCL** | Rusticl принимает `clCreateProgramWithIL` (SPIR-V 1.0–1.6) — прежний вывод «заблокировано» был неверным. Целочисленный `dot` clang эмитит как невалидный `OpDot`; после патча в `OpSDot`/`PackedVectorFormat4x8Bit` ядра грузятся и считают верно, но **в 1.3–2.2× медленнее** того же OpenCL C. Флаги `-O2/-O3/-cl-fast-relaxed-math` не помогают. |
| **Vulkan** | Дескрипторный пул починен, Q4-GEMM портирован (64×64, WG 256, `dotPacked4x8EXT`), считается верно, но **в 6× медленнее**. Плюс на этой карте Vulkan-путь llama.cpp использует f16-дот (`v_dot2_f32_f16`), а не dp4a. |
| **ROCm (amdocl64)** | `gfx1034` виден только при `OCL_ICD_VENDORS=/etc/OpenCL/vendors/amdocl64.icd`. OpenCL 2.0, **WG ≤ 256**, нет `cl_khr_integer_dot_product`, но есть `__builtin_amdgcn_sdot4`. По скорости паритет: @64 быстрее (1.20 против 1.29 мс), @512 медленнее (9.2 против 7.3 мс). |

## Решение

Оставляем **один путь — OpenCL C на Rusticl** — и работаем над структурой ядра и
кодогенерацией. Порт архитектуры `mul_mm` из llama.cpp в Rusticl (int8, 64×64, 128 потоков,
32 накопителя) сделан и измерен: он **в 1.6× медленнее** нашего `lds17` из-за спилов
(см. `benchmarks/rusticl-llamacpp-kernel.md`).

ROCm из системы удалён. Крейты `backend-vulkan`, `backend-rocm`, `backend-spirv` удалены,
вместе с ними убраны `LFM_PLATFORM`, `Gpu::build_il`, `ProgramBuilder` и ветка `LFM_AMD`.
