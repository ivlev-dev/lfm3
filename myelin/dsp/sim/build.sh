#!/bin/bash
# Build the lfmgemv exec stand as a QuRT qexe for the Hexagon v68 QEMU
# machine.  No phone, no FastRPC, no Android NDK.
#
#   HEXAGON_SDK_ROOT=/opt/hexagon-sdk-6.7.0.0 ./build.sh
#   QEMU_BIN=/path/to/qemu-system-hexagon ./build.sh
#
# Output: build/lfmsim  (run it with ./runner.sh)
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
QEMU_BIN=${QEMU_BIN:-/tmp/opencode/qemu/build/qemu-system-hexagon}
OUT=${OUT:-build}

S=$HEXAGON_SDK_ROOT
TD=$S/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
CC=$TD/bin/hexagon-clang
QU=$S/rtos/qurt/computev68/include
QL=$S/rtos/qurt/computev68/lib
G0=$TD/target/hexagon/lib/v68/G0
HVX=../hvx
CONV=../conv

if [ ! -x "$CC" ]; then
    echo "hexagon-clang не найден: $CC" >&2
    echo "проверьте HEXAGON_SDK_ROOT и HEXAGON_TOOLS_VERSION" >&2
    exit 1
fi
if [ ! -f "$QL/crt1.o" ] || [ ! -f "$QL/libqurt.a" ]; then
    echo "QuRT v68 не найден в $QL" >&2
    echo "проверьте HEXAGON_SDK_ROOT (нужен rtos/qurt/computev68)" >&2
    exit 1
fi
if [ ! -f "$G0/libc.a" ]; then
    echo "тулчейн-библиотеки не найдены в $G0" >&2
    exit 1
fi
if [ ! -x "$QEMU_BIN" ]; then
    echo "QEMU не найден: $QEMU_BIN" >&2
    echo "соберите qemu-system-hexagon или укажите QEMU_BIN=... " >&2
    exit 1
fi

mkdir -p "$OUT"

CFLAGS="-mv68 -mhvx -mhvx-length=128b -G0 -O2 -Wall \
        -I$S/incs -I$S/incs/stddef -I$HVX -I$CONV -I$QU -I$QU/qurt -I$QU/posix"

echo "CC=$CC"
$CC $CFLAGS -c "$HVX/lfmgemv_impl.c" -o "$OUT/lfmgemv_impl.o"
$CC $CFLAGS -c sim_main.c -o "$OUT/sim_main.o"
$CC $CFLAGS -c shim_hap.c -o "$OUT/shim_hap.o"

# qexe link recipe (QuRT v68, sdksim runelf in QEMU):
#   .interp is the fixed user image base, -force-dynamic keeps the ELF layout
#   the runelf loader expects, --whole-archive pulls the QuRT/crt objects.
$CC -mv68 -g -nodefaultlibs -nostdlib \
    -Wl,--section-start,.interp=0x23000000 \
    -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
    -Wl,--whole-archive \
    "$G0/init.o" "$OUT/lfmgemv_impl.o" "$OUT/shim_hap.o" "$OUT/sim_main.o" \
    "$QL/crt1.o" "$QL/debugmon.o" "$QL/libqurt.a" \
    "$G0/libc.a" "$G0/libqcc.a" "$G0/libhexagon.a" "$G0/libm.a" \
    "$QL/libqurtcfs.a" "$QL/libtimer_island.a" "$QL/libtimer_main.a" \
    "$QL/libposix.a" \
    -Wl,--no-whole-archive "$G0/fini.o" -o "$OUT/lfmsim"

echo "BUILD OK: $PWD/$OUT/lfmsim"
