#!/bin/bash
# Build the multi-PD host driver.  Does not touch the existing skel/stub/IDL:
# it compiles multi_main.c against ../lfmgemv.h + ../lfmgemv_stub.c and uses
# the already built ../liblfmgemv_skel.so.
#
#   HEXAGON_SDK_ROOT=/opt/hexagon-sdk-6.7.0.0 ./build.sh
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/android-ndk}
ANDROID_API=${ANDROID_API:-31}
S=$HEXAGON_SDK_ROOT
HCC=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API}-clang
CDSPLIB=$S/ipc/fastrpc/remote/ship/android_aarch64

[ -x "$HCC" ] || { echo "no NDK clang: $HCC" >&2; exit 1; }
[ -f ../liblfmgemv_skel.so ] || { echo "build ../build.sh first (skel missing)" >&2; exit 1; }
[ -f ../lfmgemv_stub.c ] || { echo "../lfmgemv_stub.c missing" >&2; exit 1; }

$HCC -O2 -Wall -I"$S/incs" -I"$S/incs/stddef" -I.. -o multi_test \
    multi_main.c ../lfmgemv_stub.c \
    -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm
echo "MULTI BUILD OK"
ls -la multi_test
