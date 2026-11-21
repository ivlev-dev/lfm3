# Myelin для Android

Gradle-проект вокруг уже собранного нативного ядра Myelin:

```
myelin/android/
├── settings.gradle.kts          # :app + :myelin-sdk
├── build.gradle.kts             # AGP 9.5.2, Kotlin 2.4.21 (built-in в AGP), Compose-плагин
├── gradle.properties
├── myelin-sdk/                  # AAR-модуль: JNA + libmyelin.so + UniFFI-биндинги
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── jniLibs/arm64-v8a/libmyelin.so      # собран build-android.sh
│       └── kotlin/uniffi/myelin/myelin.kt      # сгенерированные биндинги
└── app/                         # demo-приложение на Compose
    ├── build.gradle.kts
    ├── proguard-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── kotlin/dev/myelin/demo/
        │   ├── MainActivity.kt      # Compose UI
        │   ├── ChatViewModel.kt     # StateFlow + viewModelScope
        │   └── MyelinClient.kt      # Engine/Session + callbackFlow-мост
        └── res/values{,-night}/…
```

## Что нужно для сборки

| Компонент | Версия | Зачем |
|---|---|---|
| JDK | **17** | `jvmToolchain(17)`, AGP 9.5 |
| Android SDK Platform | **37.1** (`compileSdk = 37`, `compileSdkMinor = 1`) | Compose BOM 2026.11.01 требует API 37.1 |
| Android Build-Tools | **37.0.0** | aapt2/zipalign |
| Gradle | **9.9** (wrapper в репозитории) | требование AGP 9.5 |
| Android Studio | Ladybug+ | не обязательна, но проще всего |

Для пересборки нативной части дополнительно: Rust (stable), `cargo-ndk` не нужен,
NDK **r27c** и target `aarch64-linux-android` (см. `../build-android.sh`).

## Сборка

```bash
# 1) указать SDK (или открыть проект в Android Studio — она создаст local.properties)
cd myelin/android
echo "sdk.dir=$HOME/Android/Sdk" > local.properties      # путь подставьте свой

# 2) wrapper уже лежит в репозитории (Gradle 9.9), генерировать не нужно
#    при желании: gradle wrapper --gradle-version 9.9

# 3) debug APK
./gradlew :app:assembleDebug
# → app/build/outputs/apk/debug/app-debug.apk

# 4) установка на подключённое устройство (arm64)
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`ANDROID_HOME`/`ANDROID_SDK_ROOT` тоже подходят вместо `local.properties`.

### Про Gradle wrapper

В репозитории лежит wrapper на Gradle **9.9** (`gradlew`, `gradlew.bat`,
`gradle/wrapper/*`). Пересоздать его можно командой:

```bash
gradle wrapper --gradle-version 9.9
```

## Пересборка `.so` и биндингов

Из корня `myelin/` (не из `android/`):

```bash
ANDROID_NDK_HOME=/path/to/android-ndk-r27c ./build-android.sh
```

Скрипт:

1. собирает `myelin-ffi` под `aarch64-linux-android` с
   `-Wl,-z,max-page-size=16384`;
2. копирует `libmyelin.so` в `android/myelin-sdk/src/main/jniLibs/arm64-v8a/`;
3. проверяет, что все LOAD-сегменты выровнены на 16 КБ;
4. заново генерирует Kotlin-биндинги в `android/myelin-sdk/src/main/kotlin/`.

После этого достаточно повторить `./gradlew :app:assembleDebug` — новые `.so` и
`myelin.kt` подхватятся без правок Gradle-файлов.

## 16 КБ выравнивание (Android 15+)

Android 15 требует, чтобы нативные библиотеки были выровнены на 16 КБ, иначе
установка/запуск падают. Здесь это обеспечено с двух сторон:

* линковка: `-Wl,-z,max-page-size=16384` в `build-android.sh`;
* упаковка: `packaging { jniLibs { useLegacyPackaging = false } }` в обоих модулях —
  AGP 9 пакует `.so` без сжатия и выравнивает их `zipalign -P 16`.
  **Не включайте** `useLegacyPackaging = true` и не добавляйте `extractNativeLibs`
  в манифест — это ломает выравнивание.

Проверка готового APK:

```bash
$ANDROID_HOME/build-tools/37.0.0/zipalign -c -P 16 -v 4 app/build/outputs/apk/debug/app-debug.apk
readelf -lW myelin-sdk/src/main/jniLibs/arm64-v8a/libmyelin.so | grep LOAD
```

## Как устроен demo

* `MyelinClient` создаёт `Engine(modelPath = null)` — это Auto-backend, который
  выбирает встроенную mock-реализацию. Файл модели, сеть и разрешения не нужны.
* `Engine.newSession()` даёт `Session`; `.so` грузится JNA при первом обращении к
  `uniffi.myelin` (поэтому в `myelin-sdk` подключён `net.java.dev.jna:jna:5.19.1@aar`).
* Имя библиотеки: сгенерированные биндинги по умолчанию ищут `libuniffi_myelin.so`,
  а ядро пакуется как `libmyelin.so`. `MyelinApp.onCreate()` выставляет
  `uniffi.component.myelin.libraryOverride=myelin` до первого обращения к UniFFI-классам.
* `Session.generate(...)` — **блокирующий** вызов, который отдаёт `TokenEvent` в
  `TokenSink.onToken`. Он обёрнут в `callbackFlow` и запускается на `Dispatchers.IO`:

```kotlin
fun stream(messages: List<ChatMessage>, sampling: Sampling = Sampling()): Flow<TokenEvent> =
    callbackFlow {
        val sink = object : TokenSink {
            override fun onToken(event: TokenEvent) {
                trySendBlocking(event) // backpressure, не теряем токены
            }
        }
        val worker = launch(Dispatchers.IO) {
            try {
                session.generate(messages, sampling, sink)
                close()
            } catch (cancelled: MyelinException.Cancelled) {
                close()
            } catch (error: Throwable) {
                close(error)
            }
        }
        awaitClose {
            session.cancel()   // разблокирует Rust-цикл на ближайшем токене
            worker.cancel()
        }
    }
```

* `ChatViewModel` собирает `TokenEvent.Delta` в последнее сообщение-ассистента и
  публикует `StateFlow<ChatUiState>`; UI читает его через
  `collectAsStateWithLifecycle()`.
* Stop → `Session.cancel()`; `generate` возвращает `MyelinException.Cancelled`, поток
  закрывается штатно, без ошибки в UI. `onCleared()` отменяет job и закрывает
  `Session`/`Engine` (у сгенерированных классов есть `AutoCloseable`).
* Заглушка эмитит ровно 5 токенов `<token>` с паузой ~10 мс (`STUB_TOKEN_COUNT` в
  `crates/myelin-ffi/src/runtime.rs`), затем `TokenEvent.Done(total = 5)`.

### Ошибки

Сгенерированный тип ошибки называется **`MyelinException`** (в Rust — `MyelinError`):
`InvalidArgument`, `InvalidState`, `Io`, `Backend`, `Cancelled`, `Internal`.
`ChatViewModel` маппит их в текст через `Throwable.toUserMessage()`, а `TokenEvent.Error`
показывается тем же баннером над полем ввода.

## Ограничения demo

* Только `arm64-v8a`: на x86_64-эмуляторе APK не запустится (нет `.so`).
* Нет `INTERNET`, доступа к файлам и модели — backend-заглушка.
* `minSdk 26`, `targetSdk 37`, Compose Material 3, тема — динамические цвета на
  Android 12+.
* AAR-публикация, version catalog и Maven-репозиторий сознательно не настраивались.
