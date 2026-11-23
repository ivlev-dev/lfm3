# Статус слоя ядер `myelin/dsp`

Дата: 23.11.2026. Зафиксировано после big-bang удаления старых ядер
(`hvx/`, `hmx/`, `conv/`, `quant/`) и написания нового слоя int4 + Hadamard.
Документ описывает **что проверено, где и чем**, и **что заведомо не работает**.
Контракт форматов и ABI — в [CONTRACT.md](CONTRACT.md).

## 1. Что сделано

| Файл/каталог | Содержимое |
|---|---|
| `common/` | `fp16.h`, `rot.h` (H32/H128 + Rademacher), `quant.h` (int8/32, RTE, zp), `q4.h` (int4/32 image), `kvarn.h` (K4/V2, K4/V4), `page.h` (paged KV, ABI) |
| `hvx/` | `gemv_i4.c`, `gemm_i4.c` (+`i4_core.h`), `ffn_i4.c`, `attn_flash.c`, `kvcache.c`, `exec.c`, `hvx_ops.h`, `lfmgemv.idl` |
| `hmx/` | `gemm_i4.c`, `gemm_i8.c`, `prefill.c`, `hmx_util.h`, `hmx_ops.h`, `mx4.idl`, `mx4_impl.c` |
| `tests/` | `test_common.c`, `kvarn_case.h` (генерируется из Rust) |
| `sim/` | QuRT/HVX shim, стенд, `build.sh`, `runner.sh` |
| `run_tests.sh`, `Makefile` | единая точка прогона |

Форматы: веса int4/блок 32 с f16-шагом и H128-поворотом; активации int8/блок
32 с H128 на лету; KV — kvarn K4/V2 и K4/V4 с H128; HMX — `u = q + 128` и
компенсация `-128·Σw` в int32-bias. Q4_K не используется.

## 2. Проверено

### 2.1 x86 host (`cc -O2`, переносимый путь)

`cd myelin/dsp && LFM_SIM_MODES=host ./run_tests.sh` — **6/6 PASS**:
common, gemv, gemm, attn, exec, hmx_gemm.

| Проверка | Результат |
|---|---|
| fp16 roundtrip | бит-в-бит |
| поворот страницы 32x128 | бит-в-бит с Rust `kvarn::rot_fwd_page` |
| kvarn K4/V2, K4/V4 encode | **байт-в-байт** с `myelin-pack/src/kvarn.rs` |
| kvarn decode K/V | совпадает с Rust-эталоном |
| HVX GEMV/GEMM (скалярный путь) | `max_abs = 0` против `lfm_q4_gemv_i8`; `cos_dq = 1.0`; `cos_int4` 0.9934–0.9963 |
| int4/32 + H128 | cos 0.9952 (M=32, N=256, K=2048) |
| HMX int8 | `err = 0` побитово, M=32/64/128 |
| HMX prefill vs gemm | `max_abs_err = 0` |
| attention (paged, kvarn) | `cos_min = 1.0` против CPU-декода; окно 64/128, GQA, 2 последовательности, 192 токена, f16 |
| FFN | cos 1.0 против оракула на тех же int4/int8; 0.9878 против истинных f32 (предел формата) |
| exec / continuous batching | joint-шаг == отдельные шаги бит-в-бит; перестановка строк инвариантна; decode == прямой gemv |

### 2.2 `hexagon-sim` (CDSP v68, QuRT)

| Стенд | Результат |
|---|---|
| `common` | PASS |
| `gemv` | PASS (после фикса HVX-лока, см. §3.1) |
| `gemm` | PASS до подключения настоящих ядер; после фиксов не перепроверен |
| `exec` | PASS до подключения настоящих ядер; после фиксов не перепроверен |
| `attn` | commit-layout K4V2/K4V4 проходит, затем зависает до финального PASS |
| `hmx_gemm` | зависает после `SIM case hmx_gemm RUN`, до первого вывода |

### 2.3 Телефон

Xperia 1 III (XQ-BC72, SD888, `lahaina`, Android 17) виден в `adb devices`.
**Ни один стенд на телефоне не запускался:** старые `deploy.sh`/FastRPC-обвязка
удалены, новые stub/skel и Android-хост не собраны.

## 3. Известные проблемы

### 3.1 HVX под QuRT требует `qurt_hvx_lock`

`hexagon-sim` в режиме QuRT (`runelf.pbn`) не даёт выполнять HVX без лока.
Симптом — зависание в первом же ядре. Исправлено: в `sim_main.c` контекст
берётся на время кейса (`qurt_hvx_lock(QURT_HVX_MODE_128B)` /
`qurt_hvx_unlock()`), как требует [docs/hvx.md](../docs/hvx.md).

### 3.2 Weak-заглушки перекрывали настоящие ядра

`exec_stub.c` содержит weak-версии `lfm_hvx_gemv_i4`/`gemm`/`attn`/`kvcache`.
При линковке из архива weak-символ satisfying-ссылки не давал подтянуть
настоящий объект, и стенд молча гонял скалярную заглушку. Исправлено:
`*_stub.c`, `*_skel.c`, `mx4_impl.c` исключены из DSP-архива. Прежние
«PASS» для gemv/attn в QuRT-стенде относились к заглушкам, а не к HVX.

### 3.3 Attention в `hexagon-sim`

`test_commit_layout` для K4V2 и K4V4 печатает `committed`, после чего процесс
не доходит до финального `PASS` (таймаут). На x86 тот же код проходит
полностью. Точка зависания не локализована; подозреваются `free()`/`malloc`
в потоке QuRT либо прямой вызов `lfm_kvarn_commit_work` с heap-scratch в
тесте. Численные attention-кейсы под `__HEXAGON__` отключены, чтобы стенд
не висел на них.

### 3.4 HMX в `hexagon-sim`

`--mhmx 1` включает HMX (см. [docs/hmx.md](../docs/hmx.md) §8.1), VTCM видна.
Стенд `hmx_gemm` зависает сразу после `SIM case hmx_gemm RUN`, до первого
`printf`; то же без `--mhmx`. Как и в §3.3, это не подтверждённый дефект ядра:
хостовая эмуляция проходит (`int8 err=0`, `int4 cos=0.9952`). Нужен разбор
harness/HVX-контекста и прогон на телефоне.

### 3.5 Внешняя загрузка машины

На x86 работает чужой процесс `auth_hardening` (~450% CPU, load average 9–10).
Его не убивали по указанию. Тайминги и таймауты `hexagon-sim` в этих условиях
недостоверны: часть прогонов падала по `timeout`, а не по ошибке ядра.

### 3.6 Продуктового пути нет

- Нет паковщика int4/32 + H128 в `myelin-pack`.
- `exec` гоняет синтетическую модель (hidden=64, 2 слоя), а не LFM3-1.7B.
- Нет FastRPC-сборки и Android-хоста для телефона.
- Качество текста int4+H128 на реальных весах не проверялось; прошлые замеры
  int4 без ротации давали циклы, с ротацией — не проверено.
- Полоса/латентность на устройстве не измерялись.

## 4. Команды

```sh
cd myelin/dsp
LFM_SIM_MODES=host ./run_tests.sh   # только x86
./run_tests.sh                      # x86 + hexagon-sim
make host                           # то же через Makefile
```

Ручной прогон одного DSP-образа:

```sh
hexagon-sim -mv68 --mhmx 1 --simulated_returnval --usefs <run_root> \
  --cosim_file <run_root>/cfg/q6ss.cfg --l2tcm_base 0xd800 \
  --rtos <run_root>/cfg/osam.cfg \
  $HEXAGON_SDK_ROOT/rtos/qurt/computev68/sdksim_bin/runelf.pbn -- <app>
```

`kvarn_case.h` генерируется автоматически из Rust:
`cargo run -q -p myelin-pack --example kvarn_dump -- dsp/tests/kvarn_case.h`.

## 5. Следующие шаги

1. Чистый прогон `hexagon-sim` при свободной машине; локализовать §3.3/§3.4.
2. Собрать FastRPC-модуль (`qaic` + stub/skel) и прогнать gemv/gemm/attn/hmx
   на подключённом телефоне.
3. Паковщик весов int4/32 + H128 в `myelin-pack`.
4. Интеграция `exec` в `myelin-npu`/`myelin-core`: полный forward LFM3-1.7B
   (conv-слои, rope, lm_head, токенизация), paged KV, continuous batching.
5. Гейт качества текста int4+H128 на реальных весах и замеры на устройстве.
