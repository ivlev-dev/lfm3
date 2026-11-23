#!/bin/bash
# Deploy the NPU chain (dsp-generate + model pack) to the phone and run it.
#
#   ./scripts/deploy-npu.sh build          # cross-compile dsp-generate
#   ./scripts/deploy-npu.sh pack           # build the model pack (host)
#   ./scripts/deploy-npu.sh push           # binary, skel and pack -> device
#   ./scripts/deploy-npu.sh run [tokens]   # one generation on the DSP
#
# Env: MADB, D, PACK, CTX, TOKENS, MODEL.
set -e
cd "$(dirname "$0")/.."

MADB=${MADB:-$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')}
D=${D:-/data/local/tmp/myelin}
PACK=${PACK:-/tmp/opencode/model-pack}
MODEL=${MODEL:-../models/LFM3-1.7B-Q4_K_M.gguf}
CTX=${CTX:-512}
TOKENS=${TOKENS:-16}
NDKBIN=/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin
BIN=target/aarch64-linux-android/release/dsp-generate

build() {
    export CC_aarch64_linux_android=$NDKBIN/aarch64-linux-android31-clang
    export AR_aarch64_linux_android=$NDKBIN/llvm-ar
    export CXX_aarch64_linux_android=$NDKBIN/aarch64-linux-android31-clang++
    cargo build --release --target aarch64-linux-android -p myelin-npu --bin dsp-generate
    ls -lh "$BIN"
}

pack() {
    [ -f /tmp/opencode/prompt-tokens.txt ] || {
        cargo run --release -q -p myelin-core --features hf-tokenizer \
            --example host_generate -- --prompt "The capital of France is" \
            --max-tokens 1 --prompt-tokens-out /tmp/opencode/prompt-tokens.txt >/dev/null
    }
    cargo run --release -q -p myelin-npu --example make_model_pack -- \
        "$MODEL" "$PACK" --tokens /tmp/opencode/prompt-tokens.txt --calib-tokens 5
}

push() {
    [ -n "$MADB" ] || { echo "нет adb-устройства: задайте MADB=<серийник>" >&2; exit 1; }
    # DSP skeleton built by dsp/hvx/build.sh
    local skel=dsp/hvx/liblfmgemv_skel.so
    [ -f "$skel" ] || { echo "нет $skel: соберите dsp/hvx/build.sh" >&2; exit 1; }
    adb -s "$MADB" shell mkdir -p "$D"
    adb -s "$MADB" push "$BIN" "$D/dsp-generate" >/dev/null
    adb -s "$MADB" push "$skel" "$D/liblfmgemv_skel.so" >/dev/null
    adb -s "$MADB" push "$PACK/." "$D/model-pack/" >/dev/null
    adb -s "$MADB" shell chmod 755 "$D/dsp-generate"
    echo "pushed -> $D (pack $(du -sh "$PACK" | cut -f1))"
}

run() {
    [ -n "$1" ] && TOKENS=$1
    adb -s "$MADB" shell -t \
        "cd $D && ADSP_LIBRARY_PATH=$D ./dsp-generate model-pack --tokens model-pack/prompt.txt --max-tokens $TOKENS --context $CTX"
}

case "${1:-}" in
build) build ;;
pack) pack ;;
push) push ;;
run) shift; run "$@" ;;
*) sed -n '2,12p' "$0" ;;
esac
