# Источники

Внешние источники, на которые опирается документация, и что именно из них
взято. Локальные пути — на машине разработки; в репозиторий они не входят.

| Источник | Ссылка / путь | Что берём | Ограничения |
|---|---|---|---|
| **Hexagon V68 Programmer's Reference Manual**, 80-N2040-46 Rev B, 2020 | документ Qualcomm (локальной копии нет) | скалярное ядро, регистры, модель памяти, интринсики | закрытый документ; версия только для v68 |
| **Hexagon V68 HVX Programmer's Reference Manual**, 80-N2040-47 Rev E, 2021 | `/tmp/opencode/docs/hvx-v68.pdf` (299 с.) | ISA HVX: `vrmpy`, `vlsr`, `vror`, `vshuffe/vshuffo`, `l2fetch`, выравнивание 128 Б, маски | описывает ISA, а не конкретные грабли прошивки; поведение `qurt_*` — вне него |
| **Hexagon HMX PRM**, 80-N2040-62 | `/tmp/opencode/hmx.pdf`, `/tmp/opencode/hmx.txt` | кроутоны, `:deep`, `dC`, store `:after`, bias-блок, геометрия тайла | **Rev AA — это V81**: `cvt`/`:2x2` на v68 отсутствуют, пример §3.2 не собирается; раскладка int8 в PRM не документирована |
| **Qualcomm QAIRT / QNN docs** | docs.qualcomm.com (QNN SDK, HTP backend) | модель графа (backend/device/context/graph/finalize/execute), поддерживаемые ops, квантование, context binary | онлайн-документация привязана к версии SDK; список ops для V68 неполный |
| **llama.cpp, Hexagon backend + HMX PR #20693** | `/tmp/opencode/docs/hmx-llamacpp.diff` | открытая реализация HVX/HMX-пути, флаг `GGML_HEXAGON_USE_HMX`, структура вызовов | HMX-путь только для **v73+**; код в diff не является эталоном для v68 |
| **htp-ops-lib** | github.com/haozixu/htp-ops-lib | примеры HMX/HVX-операторов для LLM (GEMM, attention) на Hexagon | ориентирован на v73+; лицензия/качество — проверять перед заимствованием |
| **hexagon-mlir** | github.com/qualcomm/hexagon-mlir | компилятор Qualcomm для Hexagon NPU, представление операций | не используется рантаймом; полезен как источник семантики, не как зависимость |
| **Android: 16 KB page size** | developer.android.com/guide/practices/page-sizes | требование 16-КБ выравнивания `.so`, `-Wl,-z,max-page-size`, `useLegacyPackaging` | относится к Android 15+; для старых устройств не требуется |
| **Android: AGP 9 / SDK 37** | developer.android.com, release notes AGP 9 | `compileSdk 37`, `compileSdkMinor`, Build-Tools 37.0.0, Gradle 9.9 | версии привязаны к Compose BOM 2026.11.01; обновление BOM может поднять требования |
| **QNN SDK v2.52.0.261104** | `lfm3/v2.52.0.261104.zip` (корень репозитория) | `libQnnHtpV68Skel.so` для дизассемблера, заголовки `QNN/*.h`, `libQnnHtp.so` для стендов | 3 ГБ, **не коммитится** (`*.zip` в `.gitignore`); проприетарный, распространять нельзя |
| **lfm3: `docs/backends.md`, `benchmarks/machine.md`, `crates/backend-opencl`** | репозиторий `lfm3` | эталон семантики paged KV, radix-кэша, kvarn-раскладки, батчинга; цифры APU для сравнения | это OpenCL-бэкенд, не Hexagon; цифры APU несопоставимы с DSP-ядрами напрямую |
| **`crates/lfm-npu-ref`** | репозиторий `lfm3` | ABI (`abi.rs`), формулы арен (`layout.rs`), скалярный kvarn-кодек (`kvarn.rs`) | host-эталон, не DSP-порт; раскладка страницы в OpenCL и в DSP-ядре различается (см. `paged-kv.md`) |

## Локальные артефакты замеров

Не источники, а первичные данные, на которые ссылаются цифры:

| Путь | Что внутри |
|---|---|
| `/tmp/opencode/gemv/` | int4 GEMV, FFN, attention, `README.md`, `*.c` |
| `/tmp/opencode/ffn_raw.txt` | сырые прогоны FFN (разброс времени) |
| `/tmp/opencode/conv/` | shortconv: `lfmconv_impl.c`, `conv_sim`, `conv_math.h` |
| `/tmp/opencode/hmx4/` | HMX: `mx4_impl.c`, `REPORT18.md` (тайл), `REPORT19.md` (M=64, честный prefill, decode, полоса DDR), `FINAL*.md` |
| `/tmp/opencode/qnn-study/` | дизассемблер скела: `hmx_syms.txt`, `groups.md`, `kernels.tsv`, `*.s`, `REPORT.md` |
| `/tmp/opencode/android/qnn*.c` | стенды QNN: `qnnlayer.c`, `qnnblk.c`, `qnnarg.c`, `qnndump.c` |

## Чего в источниках нет

* Документированной раскладки **int8**-кроутонов HMX для v68 (PRM покрывает
  только fp16) — снимали экспериментально; S1 закрыт 22.11.2026
  (`REPORT18.md`: int8 побитово), M=64 и честный prefill — в `REPORT19.md`.
* int4 (`weight.n`) на v68 — ни документации, ни собственного замера:
  в REPORT19 ядро собрано, но на устройстве не запускалось.
* Официального описания `qurt_hvx_get_units`/поведения `qurt_thread_create` с
  HVX на потребительских прошивках — выяснено на стенде.
* Публичных цифр QNN для LFM3 на V68 — все числа сняты нами.
