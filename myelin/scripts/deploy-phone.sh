#!/bin/bash
# Build myelin for aarch64 Android and run it on the phone over adb.
#
#   ./scripts/deploy-phone.sh build          # cross-compile only
#   ./scripts/deploy-phone.sh push           # binary + tokenizer + template
#   ./scripts/deploy-phone.sh push-model     # + GGUF (1.1 GB, use USB)
#   ./scripts/deploy-phone.sh chat           # interactive chat (adb shell -t)
#   ./scripts/deploy-phone.sh gen "prompt"   # one-shot
#   ./scripts/deploy-phone.sh serve          # server + adb forward 8080
#
# Environment: MADB (serial), D (device dir), MODEL, CTX, TOKENS.
set -e
cd "$(dirname "$0")/.."

MADB=${MADB:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}
D=${D:-/data/local/tmp/myelin}
MODEL=${MODEL:-models/LFM3-1.7B-Q4_K_M.gguf}
CTX=${CTX:-2048}
TOKENS=${TOKENS:-256}
NDKBIN=/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin
BIN=target/aarch64-linux-android/release/myelin

[ -n "$MADB" ] || { echo "нет adb-устройства: задайте MADB=<серийник>" >&2; exit 1; }

build() {
    export CC_aarch64_linux_android=$NDKBIN/aarch64-linux-android31-clang
    export AR_aarch64_linux_android=$NDKBIN/llvm-ar
    export CXX_aarch64_linux_android=$NDKBIN/aarch64-linux-android31-clang++
    cargo build --release --target aarch64-linux-android -p myelin-cli
    ls -lh "$BIN"
}

push() {
    adb -s "$MADB" shell mkdir -p "$D"
    adb -s "$MADB" push "$BIN" "$D/myelin" >/dev/null
    adb -s "$MADB" push models/hf/tokenizer.json models/hf/chat_template.jinja "$D/" >/dev/null
    adb -s "$MADB" shell chmod 755 "$D/myelin"
    echo "pushed binary + tokenizer -> $D"
}

push_model() {
    echo "uploading $MODEL (это долго по Wi-Fi; USB быстрее)"
    adb -s "$MADB" push "$MODEL" "$D/$(basename "$MODEL")" >/dev/null
    echo "model -> $D/$(basename "$MODEL")"
}

shell_cmd() {
    adb -s "$MADB" shell "cd $D && ./myelin $*"
}

case "${1:-}" in
build) build ;;
push) build; push ;;
push-model) push_model ;;
chat)
    shift
    adb -s "$MADB" shell -t "cd $D && ./myelin chat --model $(basename "$MODEL") --tokenizer tokenizer.json --template chat_template.jinja --ctx $CTX --tokens $TOKENS $*"
    ;;
gen)
    shift
    shell_cmd "gen --model $(basename "$MODEL") --tokenizer tokenizer.json --template chat_template.jinja --ctx $CTX --tokens $TOKENS --prompt \"$*\""
    ;;
serve)
    shift
    adb -s "$MADB" forward tcp:8080 tcp:8080
    echo "сервер на телефоне; локально http://127.0.0.1:8080/v1/chat/completions"
    adb -s "$MADB" shell -t "cd $D && ./myelin serve --model $(basename "$MODEL") --tokenizer tokenizer.json --template chat_template.jinja --ctx $CTX --bind 127.0.0.1:8080 $*"
    ;;
*)
    sed -n '2,14p' "$0"
    ;;
esac
