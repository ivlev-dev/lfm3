#!/bin/bash
# Сборка стенда HMX (fp16 + int8) для Hexagon v68.
#
# Устройство не требуется: на выходе
#   libmx4_skel.so — DSP-скелет (Hexagon, v68, -mhmx)
#   mx4_test       — host-драйвер (Android aarch64)
#
# SDK Qualcomm в репозитории не лежит. Укажите путь к нему:
#   HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.7.0.0 ./build.sh
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/android-ndk}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
ANDROID_API=${ANDROID_API:-31}

S=$HEXAGON_SDK_ROOT
TD=$S/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
CC=$TD/bin/hexagon-clang
HCC=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API}-clang
QAIC=$S/ipc/fastrpc/qaic/bin/qaic
QH=$S/libs/qhl_hmx/src/qhblas_hmx
CDSPLIB=$S/ipc/fastrpc/remote/ship/android_aarch64

for f in "$CC" "$HCC" "$QAIC"; do
    [ -x "$f" ] || { echo "не найден $f — проверьте HEXAGON_SDK_ROOT / ANDROID_NDK_ROOT" >&2; exit 1; }
done
[ -d "$QH" ] || { echo "нет $QH — нужен Hexagon SDK 6.7 с qhl_hmx (src)" >&2; exit 1; }

INC="-I$S/incs -I$S/incs/stddef -I. \
     -I$S/libs/qhl_hmx/inc/qhblas_hmx -I$S/libs/qhl_hvx/inc/internal \
     -I$S/rtos/qurt/computev68/include \
     -I$S/rtos/qurt/computev68/include/qurt"

# IDL -> mx4.h / mx4_stub.c / mx4_skel.c (генерируются при сборке, в git не лежат)
"$QAIC" -I$S/incs -I$S/incs/stddef mx4.idl

# DSP-скелет. Исходники qhblas_hmx берутся прямо из SDK (ref/ не нужен).
# -mhvx нужен для HVX-копий VTCM (pref/bw) и для паковки.
$CC -mv68 -mhmx -mhvx -mhvx-length=128b -O2 -fPIC -shared -g -Wall $INC \
    -o libmx4_skel.so mx4_skel.c mx4_impl.c \
    "$QH/qhblas_hmx_matrix_matrix_mpy_ahf.c" \
    "$QH/qhblas_hmx_matrix_matrix_mpy_hf_sm.S" \
    "$QH/qhblas_hmx_transfer_row_ahf.c" \
    "$QH/qhblas_hmx_transfer_column_ahf.c" \
    "$S/rtos/qurt/computev68/lib/pic/libqurt.a" \
    -Wl,-Bsymbolic -Wl,--start-group -lc -lgcc -Wl,--end-group

# Host-драйвер. libcdsprpc.so линкуется из SDK, а не копируется в репозиторий.
$HCC -O2 -Wall -I$S/incs -I$S/incs/stddef -I. -o mx4_test \
    host.c mx4_stub.c -L"$CDSPLIB" -lcdsprpc -llog -ldl -lm \
    -Wl,-rpath,'$ORIGIN'

# mkblob — утилита dev-машины (GGUF -> int4/int8 блобы для i4pref).
# Системный cc: на телефоне она не нужна, SDK ей не требуется.
cc -O2 -Wall -Wno-unused-function -o mkblob mkblob.c -lm

echo "BUILD OK"
ls -la libmx4_skel.so mx4_test mkblob
