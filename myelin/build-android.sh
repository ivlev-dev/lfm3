#!/usr/bin/env bash
# Build libmyelin.so for Android arm64, generate Kotlin UniFFI bindings and
# verify 16 KB page alignment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-/tmp/opencode/ndk/android-ndk-r27c}}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
API="${ANDROID_API_LEVEL:-24}"
TARGET="aarch64-linux-android"
ABI="arm64-v8a"
MIN_PAGE=16384

CLANG="$TOOLCHAIN/bin/${TARGET}${API}-clang"
if [[ ! -x "$CLANG" ]]; then
    echo "error: NDK clang not found: $CLANG" >&2
    echo "set ANDROID_NDK_HOME to an NDK r27 installation" >&2
    exit 1
fi

export CC_aarch64_linux_android="$CLANG"
export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG"
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-z,max-page-size=${MIN_PAGE}"

echo "==> toolchain: $CLANG"
echo "==> cargo build -p myelin-ffi --release --target $TARGET"
cargo build -p myelin-ffi --release --target "$TARGET"

SO="target/$TARGET/release/libmyelin.so"
if [[ ! -f "$SO" ]]; then
    echo "error: $SO was not produced" >&2
    exit 1
fi

JNI_DIR="android/myelin-sdk/src/main/jniLibs/$ABI"
mkdir -p "$JNI_DIR"
cp "$SO" "$JNI_DIR/libmyelin.so"

echo "==> artifact: $JNI_DIR/libmyelin.so"
file "$JNI_DIR/libmyelin.so"

if ! file "$JNI_DIR/libmyelin.so" | grep -q 'aarch64'; then
    echo "error: artifact is not an aarch64 shared object" >&2
    exit 1
fi

echo "==> LOAD alignment"
mapfile -t ALIGNMENTS < <(
    readelf -lW "$JNI_DIR/libmyelin.so" |
        awk '$1 == "LOAD" { print $NF }' |
        sort -u
)
if [[ ${#ALIGNMENTS[@]} -eq 0 ]]; then
    echo "error: no LOAD segments found" >&2
    exit 1
fi
for align in "${ALIGNMENTS[@]}"; do
    value=$((16#${align#0x}))
    printf '    %s\n' "$align"
    if (( value < MIN_PAGE )); then
        echo "error: LOAD segment alignment $align is below 16 KB" >&2
        exit 1
    fi
done
echo "==> 16 KB alignment: OK"

KOTLIN_DIR="android/myelin-sdk/src/main/kotlin"
mkdir -p "$KOTLIN_DIR"

echo "==> generating Kotlin bindings into $KOTLIN_DIR"
env -u RUSTFLAGS cargo run -q -p myelin-ffi --bin uniffi-bindgen -- \
    generate crates/myelin-ffi/src/myelin.udl \
    --language kotlin \
    --no-format \
    --out-dir "$KOTLIN_DIR"

mapfile -t KOTLIN_FILES < <(find "$KOTLIN_DIR" -name '*.kt' -type f | sort)
if [[ ${#KOTLIN_FILES[@]} -eq 0 ]]; then
    echo "error: no Kotlin files generated" >&2
    exit 1
fi
printf '    %s\n' "${KOTLIN_FILES[@]}"

echo "==> done"
