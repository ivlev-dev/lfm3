# Архитектура Myelin

Документ описывает фактическое устройство репозитория `myelin/` на 22.11.2026:
слои, крейты, поток данных одного шага, разделение работы host/DSP и правила
добавления новой модели. Там, где реализация ещё заглушка, это сказано прямо.

## 1. Слои

```
Kotlin (Compose demo, android/app)
   │  generated UniFFI bindings, package uniffi.myelin
   ▼
myelin-ffi ── UniFFI (myelin.udl) ──┐
   │                                ├── C ABI (include/myelin.h)
   │                                └── Kotlin через JNA (libmyelin.so)
   ├── runtime.rs      валидация, генерация, события, отмена
   ├── uniffi_api.rs   Engine / Session / TokenEvent / MyelinError
   └── capi.rs         opaque-хэндлы, версионированные структуры, коды ошибок
   ▼
myelin-core
   ├── spec.rs         ModelSpec, Dims, RopeConfig, LayerSpec — чистые данные
   ├── gguf.rs         GgufMeta — типизированная карта метаданных (парсера нет)
   ├── arch.rs         таблица адаптеров Arch
   ├── arch/lfm3.rs    единственное место с числами LFM3
   └── engine.rs       детерминированный движок-заглушка (splitmix64)
   ▼
myelin-npu
   └── NpuBackend trait + MockBackend; FastRPC-клиент в крейте (FFI пока mock)

── граница процесса/домена (план) ──────────────────────────────
   FastRPC (qaic stub/skel, libcdsprpc.so)
   ▼
DSP: Hexagon CDSP v68, unsigned PD
   ├── int4/int8 GEMV, FFN, attention, shortconv   (/tmp/opencode/gemv, conv)
   ├── HMX prefill (рецепт и тайл измерены; интеграция — в P2)  (/tmp/opencode/hmx4)
   │     compute-only 376.8 мкс, со стейджингом весов 3180–3219 мкс (REPORT20)
   └── staging KV, conv-состояние, kvarn commit
```

Крейт `lfm-npu-ref` (в репозитории `lfm3`, не в workspace `myelin`) —
host-эталон контрактов будущего DSP-бэкенда: побайтовый ABI, формулы арен,
скалярный кодек `kvarn`. Он не входит в `myelin`, но задаёт семантику, которую
DSP-порт обязан повторить.

## 2. Крейты

| Крейт | Роль | Что внутри | Статус |
|---|---|---|---|
| `myelin-core` | модель как данные | `ModelSpec`, `GgufMeta`, `Arch`, `Engine` | спецификация и валидация готовы; GGUF-парсер и математика — заглушки |
| `myelin-npu` | транспорт к NPU | `NpuBackend { prepare, step, reset }`, `MockBackend`, FastRPC-клиент, загрузчик весов | FFI использует mock; FastRPC-клиент имеет два бага: `remote_handle64_*` → `AEE_EBADPARM` (нужны 32-битные) и `load_buf` → `AEE_EBADPARM` (REPORT20 §3.2) |
| `myelin-ffi` | два фронта | `runtime.rs`, `uniffi_api.rs`, `capi.rs`, `myelin.udl`, `include/myelin.h` | работает на заглушке |
| `lfm-npu-ref` (lfm3) | оракул контрактов | `abi.rs`, `layout.rs`, `kvarn.rs` | готов, покрыт тестами |
| DSP-артефакты (`/tmp/opencode`) | ядра | `lfmgemv`, `mx4`, `lfmconv` | измерены отдельно, в рантайм не вкомпонованы |

`myelin-core` не знает про Android и FastRPC; `myelin-ffi` не знает про
конкретную модель; `myelin-npu` не знает про UI. Числа LFM3 (2048, 7680,
128 000, 24 слоя, окно 256) живут только в `arch/lfm3.rs`; это проверяет
интеграционный тест `no_lfm3_constants`.

## 3. Модель как данные: `ModelSpec` и `Arch`

Адаптер семьи делает ровно две вещи:

```
GgufMeta ──Arch::detect──▶ ModelFamily
         ──Arch::build_spec──▶ ModelSpec (провалидированный)
```

Ключевые типы (`crates/myelin-core/src/spec.rs`):

* `ModelSpec { family, dims, rope, layers, kv_format, tokenizer, chat_template,
  weight_format }`.
* `Dims { hidden, ffn, layers, heads, kv_heads, head_dim, vocab, conv_cache,
  sliding_window }`; `q_dim() = heads*head_dim`, `kv_dim() = kv_heads*head_dim`.
* `LayerSpec::Attention { kind: Full | Sliding } | Conv { ksize } |
  Moe { experts, top_k }`; MoE смоделирован, но не исполняется.
* `RopeConfig { full, sliding: Option<RopeSpec> }`, `RopeKind::Full |
  Partial { rotated_dim }`.
* `KvFormat` (dtype + group_size), `Quant`/`WeightFormat`, `TokenizerSpec`,
  `ChatTemplateSpec` — plain data, едут через FFI.

`ModelSpec::validate()` отсекает несогласованные формы до бэкенда:
`hidden == heads*head_dim`, `kv_heads` делит `heads`, длина `layers` совпадает
с `dims.layers`, размеры rope совпадают с `head_dim`, у sliding-слоя есть окно,
`top_k <= experts`, `conv ksize >= 2`.

LFM3-профиль (`arch/lfm3.rs`): hidden 2048, ffn 7680, 24 слоя, 16 голов,
head_dim 128, vocab 128 000, `conv_cache` 7, окно 256, 11 attention-слоёв
(4 KV-головы, GQA 16/4) и 13 conv-слоёв; sliding — слои 4, 8, 12, 16, 20;
rope: full θ=1e6, sliding θ=1e4. Значения по умолчанию переопределяются
ключами GGUF (`lfm3.block_count`, `lfm3.attention.head_count`,
`lfm3.attention.head_count_kv`, `lfm3.attention.sliding_window_pattern` и т.д.).

### Как добавить модель

1. Вариант в `ModelFamily` (`spec.rs`).
2. Файл `arch/<family>.rs` с типом, реализующим `Arch`:
   `detect` по `general.architecture`; `build_spec` читает только
   `get_arch_*`/`require_arch_*`; все дефолты семьи — здесь.
3. Строка в `ARCHES` (`arch.rs`).
4. Синтетический `GgufMeta`-тест в `crates/myelin-core/tests/`.
5. `cargo test`: тест-страж констант упадёт, если числа утекли из адаптера.

Никаких `match family` в `engine.rs`, `runtime.rs`, `capi.rs` быть не должно.
Если новая семья требует новое поле в `ModelSpec` — это изменение контракта,
а не ветка в движке.

## 4. Поток данных одного шага

### Сейчас (заглушка)

```
Kotlin Session.generate(messages, sampling, sink)
  → runtime.rs: валидация сообщений/сэмплинга, busy-флаг
  → myelin-core::Engine::step(1): splitmix64 → один u32-токен
  → sleep 10 мс → TokenEvent::Delta{index, token, "<token>"}
  → 5 раз → TokenEvent::Done{total=5}
```

Сеть, файл модели, токенизатор и DSP не участвуют. Это позволяет проверять
ABI, UniFFI, JNA, упаковку `.so` и lifecycle на телефоне без модели.

### Целевой (после подключения FastRPC)

1. Хост принимает сообщения, применяет chat template и токенизатор (пока нет).
2. Планировщик формирует строки шага: `Row { token, seq, pos, ntok, kv_off,
   tbl_off }` и `StepDesc { n_rows, ntok, window, flags, max_rows }`.
   `flags & FLAG_PREFILL` — prefill-чанк; иначе decode.
3. FastRPC-вызов уходит в DSP (одна команда на шаг, не на слой).
4. DSP проходит слои: conv или attention (с paged KV и kvarn) → FFN (один RPC
   на слой в текущем стенде) → логиты; коммитит готовые KV-страницы.
5. Возвращается `StepOut { n_rows, ntok, logits_off, status }`; сами логиты
   лежат в общей арене (ION/rpcmem), а не копируются в ответе.
6. Хост сэмплирует, отдаёт `Delta`, обновляет таблицу страниц и повторяет.

### Где что живёт

| Работа | Host (Rust/Android) | DSP (Hexagon) |
|---|---|---|
| Токенизация, chat template | да | — |
| Планировщик, батчинг, `max_seqs` | да | — |
| Таблица страниц, radix-кэш, LRU | да | — |
| Выбор страниц, публикация хэшей | да | — |
| Загрузка/преппак весов, rpcmem | да (инициирует) | принимает, держит резидентно |
| GEMV/GEMM, attention, conv | — | да |
| RMSNorm, RoPE, SiLU, residual | — (эталон) | да |
| kvarn commit (кодирование страницы) | эталон в `lfm-npu-ref` | да |
| staging-кольцо, conv-состояние | копия для отладки | да |
| Сэмплинг, детокенизация, UI | да | — |

Важно: `lfm-npu-ref` на хосте повторяет **семантику** раскладки и кодека, но не
является быстрым путём. Расхождение host-эталона и DSP-порта — баг, а не
оптимизация.

Цена границы процесса (REPORT20 §3, §7): пустой FastRPC-вызов — 118–165 мкс
(min/median), 1 МиБ — 479–861 мкс; `load_buf` 256 КиБ / 1 МиБ / 4 МиБ —
0.36 / 0.45 / 0.62 ГБ/с. Для слоя из 183 блоков по 256 КиБ это **106.6 мс**
против 1.9 мс счёта, поэтому загрузчик весов должен идти через zero-copy
(`attach_out`, ION) или блоками ≥4 МиБ, а не через копию в DSP-heap.

## 5. C ABI

`crates/myelin-ffi/include/myelin.h` — источник истины для C-потребителей.
Версия: `MYELIN_ABI_VERSION = 1`, `myelin_version()`,
`myelin_version_string()`.

Соглашения:

* `myelin_engine` / `myelin_session` — непрозрачные указатели, память выделяет
  и освобождает библиотека.
* Каждая входная структура начинается с `size`; вызовы отвергают размер
  меньше текущего и принимают больше — поля можно добавлять в конец.
* Строки — NUL-terminated UTF-8, заимствованные на время вызова.
* `0` — успех; ошибка заполняет `myelin_error { size, code, message[512] }`.
* Стрим: `myelin_session_generate(session, messages, n, sampling, callback,
  user, err)`; callback возвращает non-zero, чтобы отменить.

| Код | Значение |
|---|---|
| 0 | OK |
| 1 | invalid argument |
| 2 | invalid state (например, сессия уже генерирует) |
| 3 | I/O error |
| 4 | backend error / unsupported backend |
| 5 | cancelled |
| 6 | panic пойман на границе |
| 7 | internal error |

`MYELIN_BACKEND_AUTO=0`, `MOCK=1`, `NPU=2`; сейчас `AUTO` и `MOCK` дают
`MockBackend`, `NPU` возвращает `MYELIN_ERR_BACKEND`. Паники ловятся во всех
`extern "C"`-точках.

## 6. UniFFI

`crates/myelin-ffi/src/myelin.udl` объявляет:

* объекты `Engine` (constructor с необязательным путём, `load`, `new_session`)
  и `Session` (`generate`, `cancel`);
* словари `ChatMessage`, `Sampling`;
* enum `TokenEvent { Delta, ToolCall, Done, Error }`;
* callback `TokenSink { on_token(event) }`;
* плоскую ошибку `MyelinError` (`InvalidArgument`, `InvalidState`, `Io`,
  `Backend`, `Cancelled`, `Internal`).

Kotlin генерируется `uniffi-bindgen` из того же крейта в
`android/myelin-sdk/src/main/kotlin/uniffi/myelin/`. Сгенерированный класс
ошибки в Kotlin называется `MyelinException`.

`Session.generate` блокирующий: события приходят в `TokenSink`. Демо оборачивает
его в `callbackFlow` на `Dispatchers.IO`; `cancel()` останавливает Rust-цикл на
ближайшей границе токена.

## 7. Android

`build-android.sh`:

1. берёт NDK из `ANDROID_NDK_HOME` (по умолчанию `/tmp/opencode/ndk/android-ndk-r27c`);
2. собирает `myelin-ffi` под `aarch64-linux-android`, API 24;
3. линкует с `-Wl,-z,max-page-size=16384` и проверяет **каждый** `LOAD`-сегмент
   через `readelf -l`;
4. копирует `libmyelin.so` в `android/myelin-sdk/src/main/jniLibs/arm64-v8a/`;
5. перегенерирует Kotlin-биндинги.

Gradle: `compileSdk = 37`, `compileSdkMinor = 1`, `buildTools 37.0.0`,
`minSdk 26`, `targetSdk 37`, JDK/Gradle toolchain 17, AGP 9.5.2, Gradle 9.9,
только `arm64-v8a`. `packaging.jniLibs.useLegacyPackaging = false` — AGP пакует
`.so` без сжатия и с 16-КБ выравниванием. Демо — in-process, без демона и без
разрешений: `Engine(modelPath = null)` выбирает mock.

Имя библиотеки: биндинги ищут `libuniffi_myelin.so`, ядро лежит как
`libmyelin.so`, поэтому `MyelinApp.onCreate()` выставляет
`uniffi.component.myelin.libraryOverride=myelin` до первого обращения к UniFFI.

## 8. Что ещё заглушка

* GGUF: `Engine::load` проверяет только magic `GGUF`; парсер метаданных — следующий этап.
* Математика движка: `splitmix64`, без attention/conv/FFN.
* Сэмплинг: валидируется и игнорируется, всегда 5 токенов.
* NPU: FFI отдаёт `MYELIN_ERR_BACKEND`; в крейте `myelin-npu` FastRPC-клиент уже
  есть, но к FFI не подключён и упирается в `remote_handle64_*` и `load_buf`
  (`AEE_EBADPARM`, REPORT20 §3.2).
* Токенизатор и chat template: едут в спеке, не применяются.
* MoE: вариант есть, исполнения нет.
* Публикация AAR/Maven: вне этапа.

## 9. Тесты

```
cd myelin
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Набор детерминированный: без сети, GPU и телефона. C ABI-тесты вызывают
экспортированные символы из Rust (включая независимое `extern "C"`-объявление);
UniFFI-тесты дергают тот же Rust API, что и сгенерированный Kotlin. Отдельный
тест `no_lfm3_constants` сканирует `myelin-core/src` и падает, если числа LFM3
появились вне `arch/lfm3.rs`.
