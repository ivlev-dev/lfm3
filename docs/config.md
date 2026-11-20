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
| `kv` | `--kv` | `f16` | `f16`, `q4`, `k4v4`, `kvarn` (=`kvarn_k4v2`), `kvarn_k4v4` |
| `ctx` | `--ctx` | `4096` | контекст одной последовательности, кратен 32 |
| `kv_pool` | — | `1` | во сколько раз пул KV больше `ctx`, 1..=16 |
| `max_seqs` | `--max-seqs` | `8` | одновременных последовательностей, 1..=32 |
| `prefill_ntok` | — | `128` | размер чанка prefill, 1..=128 |
| `pb` | — | `6` | порог батчевого GEMV, 1..=12 |
| `splits` | — | `16` | максимум сплитов внимания, 1..=16 |
| `cache` | — | `true` | radix prefix cache (снапшоты conv входят в память) |
| `fuse` | — | `true` | слияние q‖k‖g и gate‖up |
| `attn4` / `attn2` | — | `true` | GQA-шеринг K/V и пары позиций в коротком внимании |
| `flash` / `flash2` | — | `true` | flash-внимание и два токена на волну в split |
| `b8p` | — | `true` | префетч x в b8-ядрах |
| `i8pf` | — | `true` | префетч весов в int8-GEMM (на подгруппе ≤16 выключается) |
| `q4i8` / `q6i8` | — | `true` | int8-копии Q4/Q6 для prefill |
| `q4i8_ffn_only` | — | `false` | int8 только для FFN (−370 МБ VRAM, prefill −6%) |
| `wg` / `wg_b` | — | `256` / `64` | рабочие группы GEMV |
| `rgemv` / `rgemv_b` | — | `4` / `5` | строк на подгруппу |
| `prepack` | — | `true` | 148-байтная раскладка и int8-копии |
| `prepack_threads` | — | `0` | потоки CPU-prepack (0 = авто) |
| `program_cache` | — | `true` | кэш скомпилированных ядер в `$XDG_CACHE_HOME/lfm3/programs` |
| `gemm_force` | — | — | принудительный GEMM по подстроке имени (диагностика) |
| `profile` | — | `false` | тайминги ядер |

## `[chat]`

| ключ | CLI | дефолт | смысл |
|---|---|---|---|
| `system` | `--system` | `You are a helpful assistant.` | системный промпт |
| `max_tokens` | `--max-tokens` | `0` | максимум новых токенов; `0` — до EOS или заполнения контекста |
| `temperature` | `--temperature` | `0.7` | температура (0 = greedy) |
| `top_k` | `--top-k` | `40` | top-k |
| `top_p` | `--top-p` | `0.95` | top-p |
| `min_p` | `--min-p` | `0.0` | min-p |
| `repeat_penalty` | `--repeat-penalty` | `1.0` | штраф за повторы (`1.0` = выключен, включает greedy-путь) |
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

## Примеры

Один GPU, сжатый KV, контекст 32k:

```toml
[backend]
kv = "kvarn"
ctx = 32000
kv_pool = 2
```

GPU + iGPU (pipeline), 23 слоя на GPU и 1 на iGPU:

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

Пользовательские настройки — только TOML/CLI. Env остались для драйвера и
диагностики:

| env | где | смысл |
|---|---|---|
| `RUSTICL_ENABLE` | драйвер | какие устройства видит Rusticl: `iris` (Intel), `iris,radeonsi` (Intel+AMD) |
| `LFM_LAUNCH_DEBUG` | `lfm-opencl` | печать запусков ядер |
| `LFM_PROFILE` | `lfm-bench` | тайминги ядер в стенде |
| `LFM_PIPE_DEBUG` | `lfm-gpu` | время чанков по стадиям pipeline |
| `LFM_TP_DEBUG` | `lfm-gpu` | фазы TP-prefill |
| `LFM_TP_WEIGHTS` | `lfm-gpu` | доли работы TP (`75/25`); для пары GPU+CPU выбирается автоматически |

`lfm-bench` (стенд) по-прежнему читает `LFM_*` — это диагностический
инструмент, не часть приложения.
