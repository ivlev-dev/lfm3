#!/bin/bash
# Сборка HVX-стенда: int4/int8 GEMV, FFN, paged attention.
#
# Устройство не требуется: собираются DSP-скелет (Hexagon v68, -mhvx)
# и host-тесты (Android aarch64). Запуск — run.sh на телефоне.
#
#   HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.7.0.0 ./build.sh
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/tmp/opencode/sdk/6.7.0.0}
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/tmp/opencode/ndk/android-ndk-r27c}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
ANDROID_API=${ANDROID_API:-31}

S=$HEXAGON_SDK_ROOT
TD=$S/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
CC=$TD/bin/hexagon-clang
HCC=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API}-clang
QAIC=$S/ipc/fastrpc/qaic/bin/qaic
QU=$S/rtos/qurt/computev68/include
CDSPLIB=$S/ipc/fastrpc/remote/ship/android_aarch64

for f in "$CC" "$HCC" "$QAIC"; do
    [ -x "$f" ] || { echo "не найден $f — проверьте HEXAGON_SDK_ROOT / ANDROID_NDK_ROOT" >&2; exit 1; }
done

# IDL -> lfmgemv.h / lfmgemv_stub.c / lfmgemv_skel.c
"$QAIC" -I$S/incs -I$S/incs/stddef lfmgemv.idl

# DSP-скелет: GEMV + SiLU + FFN + attention + PV
$CC -mv68 -mhvx -mhvx-length=128b -O3 -fPIC -shared -g -Wall \
    -I$S/incs -I$S/incs/stddef -I. -I$QU -I$QU/qurt \
    -o liblfmgemv_skel.so lfmgemv_skel.c lfmgemv_impl.c \
    -Wl,-Bsymbolic -Wl,--start-group -lc -lgcc -Wl,--end-group

# Host-тесты
$HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o lfmgemv_test \
    host_main.c lfmgemv_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm
echo "GEMV OK"

$HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o ab_test \
    ab_main.c lfmgemv_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl
echo "AB OK"

$HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o ffn_test \
    ffn_main.c lfmgemv_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm
echo "FFN OK"

$HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o attn_test \
    attn_main.c lfmgemv_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm
echo "ATTN OK"

echo "BUILD OK"
ls -la liblfmgemv_skel.so lfmgemv_test ab_test ffn_test attn_test
