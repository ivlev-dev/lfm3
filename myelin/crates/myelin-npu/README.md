# myelin-npu

Транспорт «Rust-рантайм → DSP» для Myelin: FastRPC-клиент над
`libcdsprpc.so`, типизированные обёртки методов `lfmgemv` (HVX-стенд) и
`MockTransport` для хостовых тестов.

Крейт зависит от `myelin-core`, а не наоборот. Поэтому `DspBackend` в
`myelin-core` не может называть типы этого крейта: в
`myelin_core::backend::dsp` есть узкий object-safe трейт `DspExecutor`
(`open`/`close`/`invoke`), а этот крейт реализует его для `Cdsprpc`,
`MockTransport` и универсального `Executor`-адаптера.

## Контракт

```rust
pub trait Transport: Send + Sync {
    fn open(&self, module: &str) -> Result<Handle, TransportError>;
    fn close(&self, handle: Handle) -> Result<(), TransportError>;
    fn invoke(&self, handle: Handle, call: Call<'_, '_>) -> Result<(), TransportError>;
    fn alloc_shared(&self, len: usize) -> Result<SharedBuffer, TransportError>;
}
```

`Call` — это ровно то, что `qaic` кладёт в `remote_arg[]`:

1. `prim_in` — скалярные `in long`-параметры, little-endian;
2. входные последовательности (`in sequence<octet>` / `in sequence<long>`);
3. `prim_out` — скалярные `rout long`-параметры;
4. выходные последовательности (`rout sequence<...>`).

Число входов/выходов кодируется в `REMOTE_SCALARS_MAKEX(0, method, in, out,
0, 0)`. Имена и ID методов — в `src/method.rs`; структуры `*Scalars` дают
`to_bytes()` с точной раскладкой qaic.

`SharedBuffer` — RAII: `alloc_shared` выделяет rpcmem/ION (на устройстве) или
кучу (в моке), а `Drop` вызывает `rpcmem_free` / учитывает освобождение.
На Android доступен `fd()` (для `attach_out`); `physical_address()` там
возвращает `None`, потому что rpcmem не публикует физический адрес — DSP
мапит DMA-buf по fd.

### Ошибки

`TransportError` различает:

| вариант | когда |
|---|---|
| `LibraryNotFound` | `dlopen("libcdsprpc.so")` не удался (в том числе на хосте) |
| `SymbolMissing` | в библиотеке нет `remote_handle64_*`/`rpcmem_*` |
| `SessionControl` | не включился unsigned PD |
| `HandleOpen` | `remote_handle64_open` вернул код |
| `Rpc` | метод вернул ненулевой FastRPC-код, с расшифровкой `AEE_*` |
| `Timeout` | `AEE_EEXPIRED` (или таймаут мока) |
| `BufferTooSmall` | выход мока/метода не влез в буфер вызывающего |
| `SharedMemory` | `rpcmem_alloc`/`rpcmem_to_fd` не удались |
| `Closed` | вызов на уже закрытом handle |

`fastrpc_error_name(code)` понимает и host-offset `0`, и DSP-offset
`0x80000400` (сравнение по младшим 16 битам).

### unsigned PD

`CdsprpcConfig` по умолчанию:

```rust
library: "libcdsprpc.so",
domain: 3,                 // CDSP_DOMAIN_ID
unsigned: true,            // DSPRPC_CONTROL_UNSIGNED_MODULE
shell: "fastrpc_shell_unsigned_3",
map_attributes: 0x8,       // FASTRPC_ATTR_KEEP_MAP
rpcmem_heap: 25,           // RPCMEM_HEAP_ID_SYSTEM
rpcmem_flags: 1,           // RPCMEM_DEFAULT_FLAGS
```

Перед первым `open` клиент вызывает
`remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` с `domain` и
`enable = 1`, как стенды. Имя shell — диагностический контекст: shell выбирает
драйвер, а `open` получает имя модуля (`"lfmgemv"`).

## Таблица методов

`src/method.rs` содержит все 14 методов IDL `myelin/dsp/hvx/lfmgemv.idl` в
порядке объявления. Значения выписаны из сгенерированного qaic
`lfmgemv_stub.c` (SDK 6.7.0.0):

| ID | метод | ID | метод |
|---:|---|---:|---|
| 0 | `alloc_buf` | 7 | `attach_out` |
| 1 | `set_shape` | 8 | `detach_out` |
| 2 | `load_buf` | 9 | `gemv_resident` |
| 3 | `set_scale` | 10 | `ffn_resident` |
| 4 | `gemv` | 11 | `attn` |
| 5 | `probe` | 12 | `team_info` |
| 6 | `ffn` | 13 | `release_buf` |

`build.rs` при каждой сборке:

* парсит `myelin/dsp/hvx/lfmgemv.idl` и проверяет, что объявления идут в том
  же порядке, что константы в `src/method.rs`;
* если рядом есть сгенерированный `myelin/dsp/hvx/lfmgemv_stub.c`, проверяет
  каждый `uint32_t _mid = N;` против константы.

Расхождение — ошибка сборки. `myelin/dsp/**` не изменяется.

## `DspBackend`

`myelin_core::backend::dsp::DspBackend` — исполнитель слоёв:

* `DspBackend::attach(spec, executor, "lfmgemv")` открывает handle;
* `stage_ffn(layer, FfnStage { gate, up, down, act_scale, out_scale, reps })`
  и `stage_attention(layer, AttentionStage { ... })` сообщают handle'ы
  резидентных буферов и масштабы квантования;
* `ffn_layer` квантует активацию в int8, делает один RPC `ffn`, деквантует
  результат в f32;
* `attention_layer` квантует запрос в int8, строит block table
  (`0..ceil((position+1)/32)`), делает один RPC `attn` и деквантует int32 Q12
  в f32. Стенд фиксирован на 16 q-heads × 128, поэтому другая геометрия
  отклоняется типизированной ошибкой. `AttentionStage::preprojected` должен
  быть выставлен: `attn` принимает уже спроецированный q, а q/k/v-проекции,
  q/k-norm и rope в этом крейте ещё не реализованы.

`embed`, `rms_norm`, `conv_layer`, `lm_head` пока возвращают
`BackendError::Unsupported` с конкретной причиной (нет резидентных весов /
ядра вне HVX-стенда), а не `unimplemented!()`. Полный диагностический
`DspError` (с кодом FastRPC) доступен через `DspBackend::last_error()`.

## Проверка на хосте

```bash
cd myelin
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Тесты `crates/myelin-npu/tests/transport.rs` прогоняют на `MockTransport`:

* раскладку скаляров и буферов для `alloc_buf`/`set_shape`/`gemv`/`ffn`/`attn`;
* коды ошибок FastRPC и их имена;
* повторное использование handle и ровно один `close` на drop;
* освобождение `SharedBuffer` (счётчики alloc/free, `live_bytes == 0`);
* `DspBackend`: квантование входа, деквантование выхода, типизированные
  ошибки «нет транспорта»/«нет стадии»/RPC.

### Сборка под Android

NDK r27c, API 31:

```bash
export NDK=/tmp/opencode/ndk/android-ndk-r27c
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
export CC_aarch64_linux_android=$CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER
export AR_aarch64_linux_android=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar
cargo build --target aarch64-linux-android -p myelin-npu
```

`libcdsprpc.so` не линкуется: она грузится через `dlopen`, поэтому сборка не
требует SDK Qualcomm, а на хосте отсутствие библиотеки — типизированная
ошибка.

### Запуск на телефоне

Скелет HVX собирается отдельно (`myelin/dsp/hvx/build.sh`, SDK 6.7.0.0):

```bash
cd myelin/dsp/hvx && ./build.sh
adb push liblfmgemv_skel.so /data/local/tmp/myelin/
# host-бинарь npu-bench из target/aarch64-linux-android/debug/
adb push ../target/aarch64-linux-android/debug/npu-bench /data/local/tmp/myelin/
adb shell ADSP_LIBRARY_PATH=/data/local/tmp/myelin \
    /data/local/tmp/myelin/npu-bench --iterations 31
```

`npu-bench` печатает min/median/p95 для пустого RPC (`team_info`) и
`load_buf` с 1 КиБ и 1 МиБ. На хосте он честно пишет, что меряет
`MockTransport`, а не FastRPC.

Настоящий smoke-тест помечен `#[ignore]`:

```bash
cargo test -p myelin-npu --target aarch64-linux-android -- --ignored
```

Он требует `libcdsprpc.so` в системе, unsigned PD и
`liblfmgemv_skel.so` в `ADSP_LIBRARY_PATH`.

## Известные ограничения

* Нет загрузчика весов GGUF → int4/int8. `DspBackend` принимает уже
  загруженные резидентные handle'ы; конвертация и `set_scale` — следующий
  этап интеграции.
* `rms_norm`, `embed`, `conv_layer`, `lm_head` не реализованы на DSP и
  возвращают типизированную ошибку с причиной.
* `attention_layer` считает, что q уже спроецирован, а K/V уже лежат в
  резидентных буферах `AttentionStage`; host-KV обновляется нейтральными
  значениями только чтобы держать инвариант длины `KvCache`. Проекции q/k/v,
  rope и заливка K/V в DSP-кэш (int4/int2 + per-token scales) — отдельная
  задача.
* Block table для `attn` строится как последовательные страницы `0..N`;
  совместное использование страниц между последовательностями не поддержано.
* `rpcmem` не даёт физический адрес на Android; наружу отдаётся fd.
* Стенд `attn` жёстко зашит на 16 q-heads / 4 kv-heads / 128 dim.
* `remote_register_buf_attr` с `FASTRPC_ATTR_KEEP_MAP` пока только в
  конфиге: `attach_out` использует fd напрямую.
