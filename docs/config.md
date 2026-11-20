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
| `prefill_in_decode` | `--prefill-in-decode` | `8` | сколько prefill-токенов подмешивать в шаг, где уже есть decode, 0..=64 |
| `h2_cancel_interval` | `--h2-cancel-interval` | `16` | как часто (в батчах ответа) HTTP/2 проверяет обрыв клиента, 1..=1024 |

## Движок (`[engine]`)

Все ключи можно задать флагом сервера (приоритет CLI > файл > env > дефолт).
Булевы ключи со значением `true` по умолчанию выключаются отрицательными
флагами: `--no-prepack`, `--no-fuse`, `--no-attn2`, `--no-attn4`,
`--no-flash`, `--no-flash2`, `--no-b8p`, `--no-i8pf`, `--no-q4i8`,
`--no-q6i8`, `--no-cache`. Те же значения читаются из переменных окружения
(для `lfm-bench` и обратной совместимости).

| ключ TOML | env | флаг CLI | дефолт | смысл |
|---|---|---|---|---|
| `device` | `LFM_DEVICE` | `--device` | `0` | индекс OpenCL-устройства |
| `profile` | `LFM_PROFILE` | `--profile` | `false` | тайминги ядер (`lfm-bench gpu-prefill` печатает профиль) |
| `kv` | `LFM_KV` | `--kv` | `f16` | тип KV-кэша: `f16`, `q4`, `k4v4`, `kvarn` (=`kvarn_k4v2`, K 4 бита + V 2 бита), `kvarn_k4v4` (K 4 бита + V 4 бита, память +25 МБ, качество ближе к f16); на устройстве без fp16 `f16` хранится во float, `kvarn*` работают на float-скейлах, `q4` недоступен |
| `prepack` | `LFM_PREPACK` | `--no-prepack` | `true` | 148-байтная раскладка Q4_K и int8-копии; `false` оставляет только decode (prefill требует prepack) |
| `prepack_threads` | `LFM_PREPACK_THREADS` | `--prepack-threads` | `0` | потоков на CPU-prepack, 0 = авто (rayon), 1 = последовательно; 0..=256 |
| `program_cache` | `LFM_PROGRAM_CACHE` | `--no-program-cache` | `true` | кэш скомпилированных ядер OpenCL в `$XDG_CACHE_HOME/lfm3/programs` (ключ: исходники + опции + устройство + драйвер) |
| `fuse` | `LFM_FUSE` | `--no-fuse` | `true` | слияние `q‖k‖g` и `gate‖up` |
| `pb` | `LFM_PB` | `--pb` | `6` | порог батчевого GEMV, 1..=12 |
| `kv_pool` | `LFM_KV_POOL` | `--kv-pool` | `1` | во сколько раз пул KV больше `ctx` одной последовательности, 1..=16; больше пул — больше одновременных длинных запросов (память KV растёт линейно) |
| `prefill_ntok` | `LFM_PREFILL_NTOK` | `--prefill-ntok` | `128` | размер чанка prefill, 1..=128; у `kv=kvarn` задаёт ещё и размер staging-кольца (память KV при ctx 8192 и 8 seq: ≈65 МБ на 128, ≈59 на 64, ≈56 на 32) |
| `max_seqs` | `LFM_SEQS` | `--max-seqs` | `8` | одновременных последовательностей, 1..=16 |
| `cache` | `LFM_CACHE` | `--no-cache` | `true` | кэш префиксов (radix) |
| `splits` | `LFM_SPLITS` | `--splits` | `16` | максимум сплитов внимания, 1..=16 |
| `attn4` | `LFM_ATTN4` | `--no-attn4` | `true` | GQA-шеринг K/V (4 головы на группу) |
| `attn2` | `LFM_ATTN2` | `--no-attn2` | `true` | пары позиций в коротком внимании |
| `flash` | `LFM_FLASH` | `--no-flash` | `true` | flash-внимание: 4 головы × 16 полос |
| `flash2` | `LFM_FLASH2` | `--no-flash2` | `true` | два токена на волну в split-внимании |
| `b8p` | `LFM_B8P` | `--no-b8p` | `true` | префетч x в b8-ядрах decode |
| `i8pf` | `LFM_I8PF` | `--no-i8pf` | `true` | префетч весов группы в int8-GEMM; на устройстве с подгруппой ≤16 выключается (не хватает регистров) |
| `q4i8` | `LFM_Q4I8` | `--no-q4i8` | `true` | int8-перепаковка Q4_K для prefill |
| `q6i8` | `LFM_Q6I8` | `--no-q6i8` | `true` | Q6-prefill на int8-ядре |
| `wg` | `LFM_WG` | `--wg` | `256` | размер рабочей группы decode-GEMV (кратен 64) |
| `wg_b` | `LFM_WG_B` | `--wg-b` | `64` | рабочая группа батчевого GEMV |
| `rgemv` | `LFM_RGEMV` | `--rgemv` | `4` | строк на подгруппу в decode-GEMV, 1..=16; на устройстве с подгруппой ≤16 поднимается до 16 |
| `rgemv_b` | `LFM_RGEMV_B` | `--rgemv-b` | `5` | строк на подгруппу в батчевом GEMV, 1..=8 |
| `gemm_force` | `LFM_GEMM_FORCE` | `--gemm-force` | `""` | принудительный GEMM по подстроке имени тензора |

## Переменные, которых нет в файле

| env | где | смысл |
|---|---|---|
| `LFM_TP` | сервер | tensor parallel, устройства через `/`: `LFM_TP=0/1`; головы (16/4) должны делиться на число устройств — миры 1, 2, 4, 8, 16, три устройства невозможны |
| `LFM_TIERS` | сервер | pipeline parallel, проценты слоёв: `LFM_TIERS=60/40`; поддерживает любое число стадий (24 слоя → 3×33/33/34) и батчевый prefill |
| `RUSTICL_ENABLE` | окружение драйвера | какие GPU видит Rusticl: `iris` — только Intel, `iris,radeonsi` — Intel и AMD; PoCL CPU виден всегда отдельной платформой |
| `LFM_SCHED_LOG` | сервер | алиас `log_requests` |
| `LFM_TP_WEIGHTS` | tp.rs | доли работы по стадиям TP, например `75/25`; головы KV округляются целыми, FFN — до 256; по умолчанию поровну, но для пары GPU+CPU автоматически 75/25 |
| `LFM_TP_BATCH_DEC` | tp.rs | `0` — построчный decode в TP (по умолчанию строки разных слотов считаются одним батчем) |
| `LFM_TP_SYNC_NORMS` | tp.rs | `1` — вернуть синхронизацию x/hq/hs после каждой нормы (A/B, по умолчанию стадии считают нормы сами) |
| `LFM_ARGMAX` | сервер | `0` — отключить GPU-argmax для greedy (A/B; по умолчанию включён) |
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
