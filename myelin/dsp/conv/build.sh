#!/bin/bash
# Сборка стенда shortconv (depthwise k=7 на HVX).
#
#   liblfmgemv_skel.so — DSP-скелет (v68, -mhvx): gemv + conv-слой
#   conv_sim           — офлайновый симулятор на хосте (x86/aarch64, без устройства)
#   conv_test          — host-драйвер, ЕСЛИ рядом лежит conv_main.c
#                        (в репозиторий он не переносился: см. README)
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

# Офлайн-симулятор: чистая математика, никакого SDK не нужно.
${HOSTCC:-cc} -O2 -Wall -o conv_sim conv_sim.c -lm
echo "HOST SIM OK"

# IDL -> lfmgemv.h / lfmgemv_stub.c / lfmgemv_skel.c
"$QAIC" -I$S/incs -I$S/incs/stddef lfmgemv.idl

$CC -mv68 -mhvx -mhvx-length=128b -O3 -fPIC -shared -g -Wall \
    -I$S/incs -I$S/incs/stddef -I. -I$QU -I$QU/qurt \
    -o liblfmgemv_skel.so lfmgemv_skel.c lfmgemv_impl.c lfmconv_impl.c \
    -Wl,-Bsymbolic -Wl,--start-group -lc -lgcc -Wl,--end-group
echo "SKEL OK"

if [ -f conv_main.c ]; then
    $HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o conv_test \
        conv_main.c lfmgemv_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm
    echo "CONV TEST OK"
else
    echo "conv_test пропущен: нет conv_main.c (host-драйвер не сохранился, см. README)"
fi

echo "BUILD OK"
ls -la liblfmgemv_skel.so conv_sim
