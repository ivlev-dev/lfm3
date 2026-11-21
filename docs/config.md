# Конфигурация `lfm`

Один источник настроек — TOML-файл и флаги CLI. Приоритет:
**CLI > `--config` (или `lfm.toml` в текущем каталоге) > значения по умолчанию**.

```bash
lfm config init                  # создать lfm.toml с примером
lfm config print                 # эффективный конфиг (TOML)
lfm --config prod.toml serve
lfm chat --ctx 32768 --kv kvarn # оверрайды только для этого запуска
```

Неизвестные ключи отвергаются (`deny_unknown_fields`): опечатка не проходит молча.

## `[model]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `path` | `--model` | `models/LFM3-1.7B-Q4_K_M.gguf` | GGUF-модель |
| `tokenizer` | — | `models/hf/tokenizer.json` | tokenizer.json |
| `template` | — | `models/hf/chat_template.jinja` | chat-шаблон |

## `[backend]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `kind` | — | `opencl` | вычислительный бэкенд (сейчас только OpenCL) |
| `devices` | `--devices 0,1` | `[0]` | индексы OpenCL-устройств |
| `mode` | `--mode` | `single` | `single`, `pipeline`, `tp` |
| `tiers` | `--tiers 96/4` | `[]` (поровну) | проценты слоёв для `pipeline`, сумма 100 |
| `kv` | `--kv` | `f16` | `f16`, `q4` (алиас `k4v4` — тот же формат), `kvarn` (=`kvarn_k4v2`), `kvarn_k4v4` (отдельный формат: V 4 бита) |
| `ctx` | `--ctx` | `4096` | контекст одной последовательности, кратен 32 |
| `kv_pool` | — | `0` (auto) | `0` — auto: пул = `max_seqs × ctx` (каждой последовательности весь `ctx`); `1..=32` — явный множитель `ctx` |
| `max_seqs` | `--max-seqs` | `8` | одновременных последовательностей, 1..=32 |
| `prefill_ntok` | — | `128` | размер чанка prefill, 1..=128 |
| `pb` | — | `6` | порог батчевого GEMV: `ntok` 2..=`pb` — батчевый GEMV, больше — GEMM, 1..=12 |
| `splits` | — | `16` | максимум сплитов внимания, 1..=16 |
| `cache` | — | `true` | radix prefix cache (снапшоты conv входят в память) |
| `fuse` | — | `true` | слияние q‖k‖g и gate‖up |
| `attn4` / `attn2` | — | `true` | GQA-шеринг K/V и пары позиций в коротком внимании |
| `flash` / `flash2` | — | `true` | flash-внимание и два токена на волну в split |
| `b8p` | — | `true` | префетч x в b8-ядрах |
| `i8pf` | — | `true` | префетч весов в int8-GEMM (на подгруппе ≤16 выключается) |
| `q4i8` / `q6i8` | — | `true` | int8-копии Q4/Q6 для prefill |
| `q4i8_ffn_only` | — | `false` | int8-копии Q4_K только для FFN; внимание — по 148-байтной раскладке (−370 МБ памяти, prefill −6%) |
| `wg` / `wg_b` | — | `256` / `64` | рабочие группы GEMV (см. пояснение ниже) |
| `rgemv` / `rgemv_b` | — | `4` / `5` | строк на подгруппу |
| `prepack` | — | `true` | 148-байтная раскладка и int8-копии |
| `prepack_threads` | — | `0` | потоки CPU-prepack (0 = авто) |
| `program_cache` | — | `true` | кэш скомпилированных ядер в `$XDG_CACHE_HOME/lfm3/programs` |
| `gemm_force` | — | — | принудительный GEMM по подстроке имени (диагностика) |
| `profile` | — | `false` | тайминги ядер |

Прочерк в колонке CLI означает «только TOML»: флагов нет у большинства ключей
(например, `kv_pool`, `prefill_ntok`, `pb`, `splits`, `cache`, `fuse`,
`q4i8_ffn_only`, `profile`); CLI-флаги есть только там, где они указаны в
колонке.

На AMD (подгруппа 64) Q4 decode-GEMV с prepack запускается с local size =
`wg_b`, а `wg` задаёт шаг сетки. Q6 и lm_head идут общим GEMV-путём, где `wg`
задаёт и local size, и шаг сетки.

### KV-пул и слоты

`ctx` — контекст одной последовательности, страница KV = 32 токена. При
`kv_pool = 0` (auto, дефолт) эффективный пул равен `max_seqs`, то есть
`max_seqs × ctx` токенов: каждая последовательность может занять весь `ctx`.
Явное `kv_pool = N` (1..=32) задаёт `N × ctx` токенов независимо от
`max_seqs`; auto его не переопределяет. `lfm config print` печатает уже
разрешённое число и строкой-комментарием `# auto (max_seqs=N)`, а `config init`
сохраняет `0`, чтобы auto не «запёкся» в файл.

Разрешение auto живёт в одном месте — `BackendSection::effective_kv_pool()`;
её используют `kv_pages()`, предупреждения, `to_options()` и `config print`,
поэтому OpenCL (и будущий NPU) получают готовое число страниц.

Планировщик выдаёт страницы по мере роста запросов и при нехватке урезает или
придерживает их, поэтому конфиг не отвергается — но при старте выводится
предупреждение, если страниц меньше, чем слотов, или на слот приходится меньше
`ctx/8`. Для явного пула: `max_seqs = 16` требует `kv_pool ≥ 2`
(`16 × ctx/8 = 2 × ctx`); при `kv_pool = 1` запросы будут заметно урезаться.

### Память KV-пула

Оценка в `warnings()`: `kv_bytes = страницы × 11 attention-слоёв × байт_на_страницу_слоя`.
Страница — 32 токена, `kv_dim = 512`, в LFM3 1.7B 24 слоя, из них 11
attention (13 conv KV не хранят). Байт на страницу одного слоя:

| `kv` | байт/страница/слой | при `ctx=4096`, `max_seqs=8` (auto) |
|---|---:|---:|
| `f16` | 65 536 (32×512×2×2) | 738 МБ |
| `q4` / `k4v4` | 16 896 | 190 МБ |
| `kvarn` / `kvarn_k4v2` | 15 104 | 170 МБ |
| `kvarn_k4v4` | 19 200 | 216 МБ |

Формула без staging-кольца `kvarn` (оно зависит от `max_seqs × prefill_ntok`
и добавляет ≈22 МиБ на дефолте). Если оценка превышает 1.5 ГБ, при старте
выводится предупреждение с советом уменьшить `kv_pool`/`max_seqs`/`ctx` или
включить `kvarn`. Число слоёв берётся из констант модели: конфиг грузится до
GGUF, а `ModelConfig` доступен только после чтения файла модели. Для другой
архитектуры оценку нужно переносить в бэкенд.

Пример: `ctx = 32768`, `max_seqs = 8` (auto), `f16` → 8192 страницы × 11 ×
65 536 ≈ 5.9 ГБ (предупреждение); тот же конфиг с `kv = "kvarn"` → ≈1.36 ГБ
(тишина). Предел `kv_pool` у бэкенда — 32, столько же, сколько максимум
`max_seqs`, поэтому auto всегда проходит валидацию; отдельного предупреждения
о превышении лимита бэкенда нет.

## `[chat]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `system` | `--system` | `You are a helpful assistant.` | системный промпт |
| `max_tokens` | `--max-tokens` | `0` | максимум новых токенов; `0` — до EOS или заполнения контекста |
| `temperature` | `--temperature` | `0.7` | температура (0 = greedy при `repeat_penalty = 1.0`) |
| `top_k` | `--top-k` | `40` | top-k |
| `top_p` | `--top-p` | `0.95` | top-p |
| `min_p` | `--min-p` | `0.0` | min-p |
| `repeat_penalty` | `--repeat-penalty` | `1.0` | штраф за повторы (`1.0` = выключен); greedy-путь включается только при `temperature = 0` **и** `repeat_penalty = 1.0` |
| `seed` | `--seed` | `0` | 0 — фиксированное зерно по умолчанию |
| `thinking` | `--thinking` | `false` | режим рассуждений |
| `tools` | `--tools` | — | JSON с определениями инструментов (OpenAI tools) |

## `[server]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `addr` | `--addr` | `127.0.0.1:8000` | адрес сервера |
| `name` | `--name` | `lfm3` | имя модели в API |
| `max_body` | — | `4194304` | максимальный размер тела, байт |
| `prefill_budget` | `--prefill-budget` | `64` | prefill-токенов на шаг, 1..=128 |
| `prefill_in_decode` | `--prefill-in-decode` | `8` | prefill-токенов в шаге с decode, 0..=64 |
| `h2_cancel_interval` | — | `16` | проверка обрыва h2 раз в N батчей |
| `log_requests` | `--log-requests` | `false` | access log и лог планировщика |
| `system` | — | `You are a helpful assistant.` | системный промпт сервера |

## `[ui]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `stats` | `--no-stats` (у chat) | `true` | TTFT и prefill/decode tok/s в статусе и после ответа |

## Режимы устройств

Режим проверяется при загрузке конфига: `single` допускает ровно одно
устройство, `pipeline` — минимум два (и `tiers` обязаны совпадать по длине с
`devices`), `tp` — от двух до четырёх и только с `kv = "f16"`.
Неверная комбинация — ошибка загрузки, а не тихое вырождение режима.

## Примеры

Один GPU, сжатый KV, контекст 32k:

```toml
[backend]
kv = "kvarn"
ctx = 32000
kv_pool = 2
```

GPU + iGPU (pipeline), машина с двумя устройствами (на текущей машине недоступно):

```toml
[backend]
devices = [0, 1]
mode = "pipeline"
tiers = [96, 4]
kv = "kvarn"
ctx = 32000
```

Tensor parallel на двух одинаковых GPU (только `kv = "f16"`):

```toml
[backend]
devices = [0, 1]
mode = "tp"
```

## Переменные окружения

Пользовательской конфигурации через env нет: все настройки — TOML/CLI.
Переменные ниже читаются кодом, но это драйвер, кэш и диагностика (а не способ
настроить приложение).

| env | где | смысл |
|---|---|---|
| `RUSTICL_ENABLE` | драйвер Mesa | какие устройства видит Rusticl. На текущей машине не требуется: Vega 8 виден по умолчанию; `iris` относится к Intel iGPU, которого в системе нет |
| `HF_TOKEN` | `infra-hf` | токен для скачивания файлов HF (не настройка инференса) |
| `XDG_CACHE_HOME` | `infra-opencl` | каталог кэша программ: `$XDG_CACHE_HOME/lfm3/programs` |
| `LFM_PROGRAM_CACHE` | `infra-opencl` | `0`/`false` отключает кэш скомпилированных программ |
| `LFM_PLATFORM` | `infra-opencl` | фильтр OpenCL-платформ (диагностика) |
| `LFM_LAUNCH_DEBUG` | `infra-opencl` | печать запусков ядер |
| `LFM_PIPE_DEBUG` | `backend-opencl` | время чанков по стадиям pipeline |
| `LFM_TP_DEBUG`, `LFM_TP_WEIGHTS`, `LFM_TP_SERIAL`, `LFM_TP_SYNC_NORMS`, `LFM_TP_BATCH_DEC` | `backend-opencl` | диагностика и A/B-переключатели TP |
| `LFM_DEVICE`, `LFM_KV`, `LFM_PB`, `LFM_SEQS`, `LFM_KV_POOL`, `LFM_PREFILL_NTOK`, `LFM_PROFILE` и остальные `LFM_*` из `Options::from_env` | стенд `lfm-bench` | настройки диагностического стенда, не приложения |

`lfm-bench` (`tools/bench/`) — основной потребитель `Options::from_env`;
приложение получает настройки из `Config` (TOML + CLI) и идёт через
`load_with(&Options)`, а не через env.
