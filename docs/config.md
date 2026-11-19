# Конфигурация

Приоритет источников: **CLI > файл `--config` > переменные окружения > дефолт**.
`lfm-server --print-config` печатает эффективные значения в формате TOML — этот
вывод можно сохранить в файл и передать обратно через `--config` (round-trip
проверен тестом вручную: `dump` → `--config` → `dump` совпадает).

```bash
./target/release/lfm-server --print-config > server.toml
./target/release/lfm-server --config server.toml
```

Неизвестные ключи в файле отвергаются (`deny_unknown_fields`), опечатка не
проглотится молча.

## Сервер

| ключ TOML | флаг CLI | дефолт | смысл |
|---|---|---|---|
| `model` | `--model` | `models/LFM3-1.7B-Q4_K_M.gguf` | GGUF-модель |
| `tokenizer` | `--tokenizer` | `models/hf/tokenizer.json` | tokenizer.json |
| `template` | `--template` | `models/hf/chat_template.jinja` | chat-шаблон |
| `addr` | `--addr` | `127.0.0.1:8000` | адрес прослушивания |
| `ctx` | `--ctx` | `4096` | контекст в токенах, кратный 32 |
| `system` | `--system` | `""` | системный промпт по умолчанию |
| `name` | `--name` | `lfm3` | имя модели в API |
| `log_requests` | `--log-requests` | `false` | access log + лог планировщика (`sched: …`) |
| `max_body` | `--max-body` | `4194304` | максимальный размер тела запроса, байт |
| `prefill_budget` | `--prefill-budget` | `64` | бюджет prefill-токенов на шаг, 1..=128 |

## Движок (`[engine]`)

Доступно только через файл, кроме отмеченных флагами. Те же значения читаются
из переменных окружения (для `lfm-bench` и обратной совместимости).

| ключ TOML | env | флаг CLI | дефолт | смысл |
|---|---|---|---|---|
| `device` | `LFM_DEVICE` | `--device` | `0` | индекс OpenCL-устройства |
| `profile` | `LFM_PROFILE` | `--profile` | `false` | тайминги ядер (`lfm-bench gpu-prefill` печатает профиль) |
| `kv` | `LFM_KV` | `--kv` | `f16` | тип KV-кэша: `f16`, `q4`, `k4v4`, `kvarn`, `kvarn_k4v2`; на устройстве без fp16 `f16` хранится во float, `kvarn` работает на float-скейлах, `q4` недоступен |
| `prepack` | `LFM_PREPACK` | — | `true` | 148-байтная раскладка Q4_K и int8-копии; `false` оставляет только decode (prefill требует prepack) |
| `fuse` | `LFM_FUSE` | — | `true` | слияние `q‖k‖g` и `gate‖up` |
| `pb` | `LFM_PB` | `--pb` | `6` | порог батчевого GEMV, 1..=12 |
| `prefill_ntok` | `LFM_PREFILL_NTOK` | `--prefill-ntok` | `128` | размер чанка prefill, 1..=128 |
| `max_seqs` | `LFM_SEQS` | `--max-seqs` | `8` | одновременных последовательностей, 1..=16 |
| `cache` | `LFM_CACHE` | `--no-cache` | `true` | кэш префиксов (radix) |
| `splits` | `LFM_SPLITS` | `--splits` | `16` | максимум сплитов внимания, 1..=16 |
| `attn4` | `LFM_ATTN4` | — | `true` | GQA-шеринг K/V (4 головы на группу) |
| `attn2` | `LFM_ATTN2` | — | `true` | пары позиций в коротком внимании |
| `flash` | `LFM_FLASH` | — | `true` | flash-внимание: 4 головы × 16 полос |
| `flash2` | `LFM_FLASH2` | — | `true` | два токена на волну в split-внимании |
| `b8p` | `LFM_B8P` | — | `true` | префетч x в b8-ядрах decode |
| `i8pf` | `LFM_I8PF` | — | `true` | префетч весов группы в int8-GEMM |
| `q4i8` | `LFM_Q4I8` | — | `true` | int8-перепаковка Q4_K для prefill |
| `q6i8` | `LFM_Q6I8` | — | `true` | Q6-prefill на int8-ядре |
| `wg` | `LFM_WG` | — | `256` | размер рабочей группы decode-GEMV (кратен 64) |
| `wg_b` | `LFM_WG_B` | — | `64` | рабочая группа батчевого GEMV |
| `rgemv` | `LFM_RGEMV` | — | `4` | строк на подгруппу в decode-GEMV, 1..=8; на устройстве с подгруппой ≤16 поднимается до 8 |
| `rgemv_b` | `LFM_RGEMV_B` | — | `5` | строк на подгруппу в батчевом GEMV, 1..=8 |
| `gemm_force` | `LFM_GEMM_FORCE` | — | `""` | принудительный GEMM по подстроке имени тензора |

## Переменные, которых нет в файле

| env | где | смысл |
|---|---|---|
| `LFM_TP` | сервер | tensor parallel, устройства через `/`: `LFM_TP=0/1` |
| `LFM_TIERS` | сервер | pipeline parallel, проценты слоёв: `LFM_TIERS=60/40` |
| `LFM_SCHED_LOG` | сервер | алиас `log_requests` |
| `LFM_PLATFORM` | infra-opencl | индекс OpenCL-платформы |
| `LFM_LAUNCH_DEBUG` | infra-opencl | печать параметров запуска ядер |
| `LFM_CPU_W4A8` | infra-cpu | W4A8-веса на CPU |
| `LFM_VERBOSE`, `LFM_OLD` | bench | диагностика бенчмарка |

Важно: при `LFM_TP`/`LFM_TIERS` настройки `[engine]` из файла
не применяются — эти пути читают окружение сами (см. `docs/backends.md`).
`lfm-bench` файл конфигурации не читает, только окружение.

## Intel iGPU

Встроенная Intel (Gen9/UHD 630) появляется в OpenCL только через Rusticl с
`RUSTICL_ENABLE=iris`; без переменной устройств не будет вовсе:

```bash
RUSTICL_ENABLE=iris ./target/release/lfm-bench info
RUSTICL_ENABLE=iris ./target/release/lfm-server --config server.toml
```

Индекс устройства — как обычно (`LFM_DEVICE`/`device`). На таком устройстве нет
`cl_khr_fp16`: KV-кэш `f16` и staging/scales у `kv = "kvarn"` хранятся во float,
`kv = "q4"` возвращает ошибку при загрузке. Подробности — `docs/backends.md`.

## Пример

```toml
model = "models/LFM3-1.7B-Q4_K_M.gguf"
addr = "0.0.0.0:8000"
ctx = 8192
name = "lfm3"
log_requests = true
prefill_budget = 96

[engine]
kv = "f16"
pb = 6
prefill_ntok = 128
max_seqs = 8
cache = true
splits = 16
flash = true
flash2 = true
q4i8 = true
q6i8 = true
```

Проверка: `lfm-server --config server.toml --print-config`.
