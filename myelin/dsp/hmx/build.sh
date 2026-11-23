#!/bin/sh
set -eu
cd "$(dirname "$0")"

CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -O2 -Wall -Wextra -ffp-contract=off -fno-fast-math}

$CC $CFLAGS -I. -I../common -o gemm_test \
    gemm_main.c gemm_i4.c gemm_i8.c prefill.c -lm

SDK=${HEXAGON_SDK_ROOT:-}
if [ -z "$SDK" ] && [ -d /opt/hexagon-sdk-6.7.0.0 ]; then
    SDK=/opt/hexagon-sdk-6.7.0.0
fi
TOOLS=${HEXAGON_TOOLS_VERSION:-21.0.04}
HCC=${HEXAGON_CC:-$SDK/tools/HEXAGON_Tools/$TOOLS/Tools/bin/hexagon-clang}
QAIC=${QAIC:-$SDK/ipc/fastrpc/qaic/bin/qaic}

if [ -n "$SDK" ] && [ -x "$HCC" ] && [ -x "$QAIC" ]; then
    "$QAIC" -I"$SDK/incs" -I"$SDK/incs/stddef" mx4.idl
    INC="-I$SDK/incs -I$SDK/incs/stddef -I. -I../common \
         -I$SDK/rtos/qurt/computev68/include \
         -I$SDK/rtos/qurt/computev68/include/qurt"
    "$HCC" -mv68 -mhmx -mhvx -mhvx-length=128b -std=c11 -O2 -fPIC -shared \
        -Wall -Wextra -ffp-contract=off -fno-fast-math $INC \
        -o libmx4_skel.so mx4_skel.c mx4_impl.c gemm_i4.c gemm_i8.c prefill.c \
        "$SDK/rtos/qurt/computev68/lib/pic/libqurt.a" \
        -Wl,-Bsymbolic -Wl,--start-group -lc -lgcc -Wl,--end-group
else
    echo "hexagon SDK не найден: собран только хостовый тест"
fi
