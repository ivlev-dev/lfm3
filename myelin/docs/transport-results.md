# Транспортные оптимизации O1/O2/O4: замеры

Документ фиксирует числа по каждому шагу `decode-plan.md` §5. Телефон —
тот же XQ-BC72 (QV7807HL8N), стенд `/data/local/tmp/lfmgemv`, модель
`/data/local/tmp/model-pack` (LFM3, 24 слоя, 831.7 MiB резидентных весов),
`ctx 256`, greedy decode, экран включён.

Команда замера (32 токена — 5 prompt + 32 decode, позиции 0…36):

```sh
adb shell 'cd /data/local/tmp/lfmgemv && \
  ADSP_LIBRARY_PATH=$PWD MYELIN_TRACE=1 ./dsp-generate /data/local/tmp/model-pack \
  --tokens /data/local/tmp/model-pack/prompt.txt --max-tokens 32 --context 256'
```

## 0. Итог

| шаг | до | после | вывод |
|---|---|---|---|
| baseline (HEAD до правок) | 112.67 мс/токен, 8.88 tok/s, 1.36 мс/RPC | — | TOKENS `383 278 2690 …` |
| O1: переиспользуемые rpcmem-буферы | 112.67 | 113–115 | **выигрыша нет**; изолированный Rust `exec` уже стоит 150–190 мкс, как C |
| O1: `remote_register_buf_attr` + `KEEP_MAP` | — | — | **ломает результат** (stale-чтение), в дефолт не включён |
| O1: `remote_register_buf_attr` + `COHERENT` | 112.67 | 111.6–113.7 | корректно, но **выигрыша нет** |
| O1: убрать пустой `rt` и `times` из горячего пути | — | (включено по умолчанию) | по `rpc_test` ≈ 20 мкс/последовательность → ожидание ≤2–3 мс; отдельно не измерено (устройство ушло) |
| O2: identity-таблица, `LIST_CACHE=160`, хэш | — | — | юнит-тесты; длинный ctx ждёт DSP-скелет |
| O4: `KVSTORE`+`ATTN_SCALE`, 82→60 RPC | — | — | хост + контракт готовы; нужна DSP-часть |

**Главный вывод O1:** гипотеза «42–46 мс — это pin/маршалинг четырёх
последовательностей на вызов» на этом устройстве/`libcdsprpc` **не
подтвердилась**. C-стенд `exec_test` сегодня: 1 exec 48.9 мс / 85 exec
69.2 мс → 242 мкс накладных на вызов. Rust в изоляции (`npu-bench`,
1 команда) — 150–190 мкс. Разница возникает **в модели**, а не в
транспорте: 0.6–0.8 мс на вызов при 0.7 мс DSP, и она растёт не от числа
последовательностей, а от числа команд/времени DSP (см. §3). Поэтому
единственный подтверждённый рычаг — **уменьшать число вызовов** (O4/O5),
а не цену вызова.

## 1. Baseline (HEAD, до правок)

```
plan: 831.7 MiB resident (budget 1024.0 MiB), 24 layers, context 256 (8 pages)
model ready in 7.66s (backend dsp), prompt 5 tokens
FIRST_TOKEN 383
PREFILL 2119.1 ms (5 tokens, 423.8 ms/token, mean layer 17.59 ms)
TOKENS 383 278 2690 2891 302 278 5201 20 309 278 5201 355 267 6229 11526 302 278 5201 4341 20 513 267 6229 11526 302 278 5201 22 440 5201 355 267
DECODE 3605.3 ms for 32 generated tokens (112.67 ms/token, 8.88 tok/s)
dsp exec totals: dsp 2185.9 ms, host 4154.3 ms
exec transport: 2982 calls, arena 105.6 ms, invoke 4044.0 ms (1.36 ms/call)
RESET_CHECK PASS (first 383, after reset 383)
```

`PREFILL` включает одноразовые `map_arena` всех 71 списка (≈2.2 мс × 71 на
скорости того момента), поэтому prefill на коротком промпте дороже decode.

## 2. O1: что сделано

`myelin-npu`:

* `transport.rs`: `Transport::register_shared` / `unregister_shared`
  (по умолчанию no-op).
* `cdsprpc.rs`: резолвит `remote_register_buf_attr` / `remote_register_buf`
  опционально; регистрация best-effort, дерегистрация — `fd = -1`
  (`remote_deregister_buf` в этой `libcdsprpc.so` нет).
* `executor.rs`: `ExecBuffers` — один rpcmem-буфер на `act`, `rt`, `out`
  на handle, рост по необходимости; `out` намеренно **не** регистрируется.
  `times` остаётся на стеке (32 Б) и передаётся только под `MYELIN_TRACE`.
  Пустой `rt` не передаётся вовсе (FFN, lm_head, qkv).

Откат без пересборки:

| env | что возвращает |
|---|---|
| `MYELIN_EXEC_HEAP=1` | старые per-call `Vec`/стек вместо rpcmem |
| `MYELIN_FULL_SEQS=1` | пустой `rt` и `times` в каждом вызове (старая форма) |
| `MYELIN_REG_ATTR=N` | **включает** регистрацию с атрибутом `N`: 4 = `COHERENT`, 8 = `KEEP_MAP`; по умолчанию регистрации нет |
| `MYELIN_NO_REGISTER=1` | не вызывать `remote_register_buf*` даже при заданном `MYELIN_REG_ATTR` |

### 2.1 Почему `KEEP_MAP` выключен

С `FASTRPC_ATTR_KEEP_MAP` (8) и зарегистрированными `act`/`rt` **или**
`act`/`rt`/`out` первый же токен меняется (`FIRST_TOKEN 20` вместо `383`),
а на 32 токенах `kv_write` падает `AEE_EBADPARM`: драйвер не инвалидирует
кэш после вызова, и DSP читает/пишет stale-данные. `remote_register_buf_attr`
возвращает `void`, ошибку увидеть нельзя. С `FASTRPC_ATTR_COHERENT` (4)
результат корректный, но цена не меняется. Поэтому дефолт — без
регистрации (только переиспользуемые rpcmem-буферы).

### 2.2 A/B (26 токенов, позиции < 32)

Три прогона подряд, чередование конфигураций; телефон шумит (см. §4),
поэтому median по повторам:

| конфигурация | мс/токен |
|---|---|
| A `MYELIN_EXEC_HEAP=1` (старый транспорт) | 109.4 / 203.9 (разброс из-за фона) |
| B rpcmem, без регистрации | 114.0 / 113.6 |
| C rpcmem + `COHERENT` | 113.7 / 111.6 |

Все три — в пределах шума baseline (112–115). Вывод: ни rpcmem, ни
регистрация на этом устройстве цену вызова не меняют.

## 3. Изолированная цена `exec` (E2)

`npu-bench` теперь меряет `exec` через тот же `Executor`, что и модель,
на стабильном списке из 1/8/32 `QUANT` (`--iterations 100`):

| форма | wall median | DSP `times[0]` | накладные |
|---|---|---|---|
| `exec QUANT` ×1 | 150–190 мкс | 3 мкс | 147–190 мкс |
| `exec QUANT` ×8 | 184–276 мкс | 17 мкс | 167–259 мкс |
| `exec QUANT` ×32 | 299–427 мкс | 67–68 мкс | 232–359 мкс |

Для сравнения, C-стенды сегодня:

| стенд | что | цена |
|---|---|---|
| `rpc_test` | пустой вызов | 128 мкс |
| `rpc_test` | `ffn` 1 in / 3 out, 256×256 | 187 мкс |
| `exec_test 5` | 1 exec (85 команд) / 85 exec | 48.9 / 69.2 мс → **242 мкс/вызов** |
| `exec_test 5` | один FFN | wall 1802 мкс, DSP 1438 мкс |

То есть **Rust-транспорт в изоляции не хуже C**. Дополнительные
0.4–0.6 мс/вызов появляются только в модели; они растут с числом команд
(≈2.8 мкс на команду — это `HAP_perf_get_time_us` в цикле `lfmgemv_exec`,
не входящий в `times[0]`) и с временем DSP, но не с числом
последовательностей. Отсюда приоритет O4/O5 (меньше вызовов), а не
микрооптимизации транспорта.

## 4. Шум на телефоне

Во время работы в дереве параллельно шёл другой агент (`lfmgemv_impl.c`,
`dsp.rs`). Наблюдалось:

* `DECODE_STEPS` 103–130 мс/токен в спокойном состоянии;
* отдельные прогоны 200–930 мс/токен и `arena` 525 мс вместо 70 мс;
* после пересборки скелета другим агентом (07:54) прогон на 32 токена
  сначала дал 475 мс/токен, затем `npu-bench` — нормальные 150 мкс.

Правило: сравнивать только прогоны подряд, на одном состоянии прошивки,
и фиксировать `DECODE_STEPS`; выбросы 2× и более — фон/троттлинг, а не
код.

## 5. O2: статичные списки

Код:

* `ATTN` получает `tbl_len = cache_pages` и identity-таблицу `0..cache_pages`
  в `act` рядом с q (+до 256 Б). `attn_core` использует `tblLen` только в
  проверке `(pos >> 5) >= tblLen` и индексирует `tbl[pos >> 5]`, поэтому
  список больше не зависит от `pages`.
* `LIST_CACHE` 128 → 160 (как `MAXARENA` в скелете).
* Поиск по FNV-1a хэшу + длине, затем побайтовое сравнение (хэш — только
  префильтр, коллизия не ломает корректность).
* `attention_exec_list()` — единственный конструктор `ATTN`-списка, что
  закреплено тестом `attention_list_is_stable_across_context_growth`:
  `tbl_len = cache_pages`, `rt_base = 0`, байты одинаковы на позициях
  0 и 2047.

На ctx 256 (8 страниц) до правки было до 8 вариантов `ATTN`-списка на слой
(до 88 сверх 71 стабильного при `LIST_CACHE = 128`), на ctx 2048 — до 704.
Теперь вариант один на слой. Замер на длинном ctx требует скелета с
исправленным `kv_write` (см. §4): старый `.so` отвергает `slot >= 32`.

## 6. O4: attention 4 RPC → 2

Хостовая часть (`MYELIN_ATTN_FUSED=1`, по умолчанию выключено):

* команды 9/10 в обоих энкодерах (`KVSTORE`, `ATTN_SCALE`), байт-в-байт;
* `pack_kv_rows()` — общий код `kv_write` и fused-пути;
* `DspBackend::attention_fused` собирает
  `KVSTORE + ATTN + ATTN_SCALE + o_proj` в один список;
* resident-таблица V-масштабов (512 f32) в `model_pack::weight_plan`
  (`blk.N.vscale`), чтобы `ATTN_SCALE` читал её по handle;
* при `AEE_EBADPARM` — сообщение `fused attention unsupported, falling
  back: …`, флаг снимается до конца сессии, старый путь продолжает работу
  (повторный `KVSTORE` того же слота идемпотентен).

DSP-контракт — `myelin/docs/attn-fused-contract.md`. Числа появятся после
реализации команд в `lfmgemv_impl.c`; ожидание при текущей цене вызова
0.6–0.8 мс: 22 снятых RPC × 0.7 ≈ **15 мс/токен** (109 → ~94), это
`decode-plan.md` O4 «14–15.5 tok/s» при более быстром DSP.

## 7. Статус гейтов

| шаг | гейт | статус |
|---|---|---|
| O1 | 32 токена, те же `TOKENS`, `RESET_CHECK PASS` | **пройден** на устройстве (с rpcmem-буферами; `TOKENS` совпал с baseline) |
| O1 | `ms/call` ≤ 0.35 | **не достигнут и не является свойством кода**: изолированный Rust `exec` = 150–190 мкс, но в модели 0.6–0.8 мс (см. §3) |
| O2 | юнит: `tbl_len`/identity/байты списка стабильны | **пройден** |
| O2 | 128 токенов, ctx 512, нет ремапов, те же токены | **ожидает устройство** (телефон ушёл из сети; нужен скелет с исправленным `kv_write`) |
| O4 | хост: `qkv + один fused exec`, форма `act`/`rt` | **пройден** (mock-тест `fused_attention_costs_one_exec_after_qkv`) |
| O4 | устройство: `MYELIN_ATTN_FUSED=1`, те же токены, 82→60 RPC | **ожидает DSP-команды 20/21** (`attn-fused-contract.md`); до неё флаг автоматически откатывается |

## 8. Как воспроизвести

```sh
# сборка (NDK r27c, API 31)
export NDK=/path/to/android-ndk
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
export CC_aarch64_linux_android=$CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER
export AR_aarch64_linux_android=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar
cargo build --release --target aarch64-linux-android -p myelin-npu --bin dsp-generate --bin npu-bench
adb push target/aarch64-linux-android/release/{dsp-generate,npu-bench} /data/local/tmp/lfmgemv/

# изоляция транспорта
adb shell 'cd /data/local/tmp/lfmgemv && ADSP_LIBRARY_PATH=$PWD MYELIN_TRACE=1 ./npu-bench -n 100'

# модель: трассировка (times включён), затем «продакшн» без трассы
adb shell 'cd /data/local/tmp/lfmgemv && ADSP_LIBRARY_PATH=$PWD MYELIN_TRACE=1 ./dsp-generate … --max-tokens 32'
adb shell 'cd /data/local/tmp/lfmgemv && ADSP_LIBRARY_PATH=$PWD ./dsp-generate … --max-tokens 32'
```
