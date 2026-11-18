# Мультибэкендный план (SPIR-V / Vulkan / ROCm)

Состояние на 18.11.2026, RX 6400 (RADV NAVI24 / Rusticl / ROCm 7.16 / gfx1034).

## Итог одной таблицей

| путь | статус | результат |
|---|---|---|
| **Свой SPIR-V через OpenCL** | **работает**, но медленнее | Rusticl принимает `clCreateProgramWithIL` (SPIR-V 1.0–1.6). Наши ядра грузятся и считают верно, но **в 1.4–2.2× медленнее** того же OpenCL C. |
| **Vulkan** | **работает** | Descriptor pool починен, Q4-GEMM портирован (64×64, 256 потоков, `dotPacked4x8EXT`), считается верно, но **в 6× медленнее** OpenCL — нужна ручная доводка под ACO. |
| **ROCm (amdocl64)** | **работает через изоляцию ICD** | `gfx1034` виден только при `OCL_ICD_VENDORS=/etc/OpenCL/vendors/amdocl64.icd`. OpenCL 2.0, **WG ≤ 256**, нет `cl_khr_integer_dot_product`, но есть `__builtin_amdgcn_sdot4`. По скорости: @64 быстрее Rusticl (1.20 vs 1.29 мс), @512 медленнее (9.2 vs 7.3 мс). |

## 1. SPIR-V через OpenCL

Первоначальный вывод «заблокировано» был **неверным**: искали символ `clCreateProgramWithIL` в
`libRusticlOpenCL.so`, а его предоставляет ICD-загрузчик, и платформа объявляет
`cl_khr_il_program`. Проверка: `clGetDeviceInfo(CL_DEVICE_IL_VERSION)` →
`SPIR-V_1.0 … SPIR-V_1.6`, загрузка и запуск тривиального ядра работают.

Что понадобилось:

1. **Компиляция**. `clang --target=spirv64 -x cl -cl-std=CL3.0 -Xclang -finclude-default-header -O3`.
2. **Патч целочисленного dot**. clang для SPIR-V эмитит `OpDot` на целочисленных векторах — это
   невалидный SPIR-V, Mesa падает с `Failed to specialize SPIR-V`. Патчим бинарник:
   `OpDot` (148) → `OpSDot` (4450), добавляем capabilities `DotProductInputAll` (6016),
   `DotProductInput4x8BitPacked` (6018), `DotProduct` (6019), версию 1.6. Если оба операнда —
   `OpBitcast` из `uint`, операнды заменяются на исходные `uint` и ставится
   `PackedVectorFormat4x8Bit` — так получается честный `v_dot4_i32_i8`.
3. **Загрузка**. `Program::create_from_il` (opencl3 0.12) → `build` → `Kernel::create`.

Замеры (`ffn_up` 7680×2048, `gemm-cmp`):

| вариант | ntok=64 | против OpenCL C |
|---|---|---|
| OpenCL C `lds17` (Rusticl) | 1.15 мс | — |
| SPIR-V `lds17` + packed `OpSDot` | 2.53 мс | ×2.2 хуже |
| SPIR-V `lds21` (packed-uint операнды) | 1.54 мс | ×1.3 хуже |
| SPIR-V «wide» 64×64/16 акк | 1.97 мс | ×1.7 хуже |

Флаги `-O2/-O3`, `-cl-fast-relaxed-math`, `-cl-mad-enable` результат не меняют. Вывод: Rusticl
компилирует OpenCL C в тот же NIR/ACO и делает это лучше, чем clang→SPIR-V; сам по себе
SPIR-V-путь выигрыша не даёт. Оставлен как лабораторный инструмент (`gemm-cmp --spv`,
`Gpu::build_il`).

## 2. ROCm / amdocl64

ROCm 7.16 установлен (`rocminfo` видит `gfx1034`, HIP работает). ROCm-OpenCL-платформа
(`AMD Accelerated Parallel Processing`) при обычном запуске отдаёт **0 устройств**, потому что
ICD-загрузчик смешивает её с Rusticl. При изоляции:

```sh
OCL_ICD_VENDORS=/etc/OpenCL/vendors/amdocl64.icd lfm-bench info
```

видно `gfx1034`, OpenCL 2.0, **max_work_group_size 256** (lds17 с WG 512 не запускается).

Особенности компилятора:

- `cl_khr_integer_dot_product` **нет** → `dot(char4,char4)` не компилируется;
- `__builtin_amdgcn_sdot4` **есть** → добавили ветку `LFM_AMD` в `common.cl`
  (`lfm_dot4`/`lfm_sum4` через `sdot4`), сборка `-cl-std=CL2.0 -DLFM_AMD`;
- сборка задаётся через `LFM_BUILD_OPTS`, ядро выбирается как обычно.

Замеры `ffn_up`:

| ядро | Rusticl @64 | AMD @64 | Rusticl @512 | AMD @512 |
|---|---|---|---|---|
| `lds11` | 1.29 мс | **1.20 мс** | 7.32 мс | 9.20 мс |
| `lds10` | 1.34 мс | **1.13 мс** | 8.09 мс | 8.90 мс |

Компилятор AMD лучше на малом батче и хуже на большом; принципиального выигрыша нет.
Плюс жёсткое ограничение WG ≤ 256 ломает наш лучший `lds17`.

## 3. Vulkan

- Descriptor pool: не хватало флага `FREE_DESCRIPTOR_SET` — повторные запуски падали в
  `ERROR_OUT_OF_POOL_MEMORY`. Исправлено.
- Портирован Q4-GEMM (`shaders/gemm_q4.comp`): тайл 64×64, WG 256, 16 аккумуляторов,
  `dotPacked4x8EXT` (SPIR-V `OpSDot` + `PackedVectorFormat4x8Bit`), общая память под веса и x.
- Проверка: `vulkan-gemm` совпадает с CPU-эталоном (max_rel 2·10⁻⁵).
- Скорость: **6.9 мс** @64 и **47.8 мс** @512 против 1.15 / 6.5 мс у OpenCL. То есть простой
  порт структуры не помогает: нужна оптимизация под ACO (f16-LDS-стейджинг, 64 потока,
  8 аккумуляторов, контроль развёртки), как в `mul_mm.comp` llama.cpp.

## Вывод

Ни SPIR-V-через-OpenCL, ни ROCm-OpenCL, ни наивный Vulkan-порт не обгоняют текущий OpenCL C
на Rusticl. Разрыв с llama.cpp (1.5 против 2.7 T MAC/s) — не в «драйвере/фронтенде», а в
конкретной структуре GEMM-ядра и качественной ручной доводке.

Разумный порядок дальнейших работ:

1. Довести Vulkan-GEMM до структуры llama.cpp (64 потока, f16-LDS, 8 аккумуляторов) — там
   ACO и полный контроль над кодогенерацией; это единственный путь, где уже есть доказанный
   результат 2.7 T MAC/s на этой же карте.
2. Параллельно убрать decode-оверхеды (argmax/сэмплинг на GPU, кэш `tbl_off`/`pos_b`).
3. SPIR-V-путь и ROCm-OpenCL оставить как инструменты (IL-загрузка и `LFM_AMD` уже в коде),
   не вкладываясь в них до появления выигрыша.
