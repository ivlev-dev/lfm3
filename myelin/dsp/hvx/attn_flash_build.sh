#!/bin/bash
# Build and run the fused attention (flash) stand under hexagon-sim:
#   attn_flash.c + attn_flash_main.c -> build/attn_flash_main
#   prints cos / maxabs for ctx 64/256/512 and pcycles, then
#   "ATTN FLASH RESULT: PASS".
#
#   ./attn_flash_build.sh
#   OUT=/tmp/af ./attn_flash_build.sh
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
SIM_BIN=${SIM_BIN:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools/bin/hexagon-sim}
OUT=${OUT:-build}

S=$HEXAGON_SDK_ROOT
TD=$S/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
CC=$TD/bin/hexagon-clang
QU=$S/rtos/qurt/computev68/include
QL=$S/rtos/qurt/computev68/lib
G0=$TD/target/hexagon/lib/v68/G0
HVX=.

mkdir -p "$OUT"

echo "[1/2] qexe"
if [ ! -x "$CC" ]; then
    echo "hexagon-clang not found: $CC (set HEXAGON_SDK_ROOT)" >&2
    exit 1
fi
CFLAGS="-mv68 -mhvx -mhvx-length=128b -G0 -O2 -Wall \
        -I$S/incs -I$S/incs/stddef -I$TD/target/hexagon/include \
        -I$HVX -I$QU -I$QU/qurt -I$QU/posix"
$CC $CFLAGS -c attn_flash.c -o "$OUT/attn_flash.o"
$CC $CFLAGS -c attn_flash_main.c -o "$OUT/attn_flash_main.o"

$CC -mv68 -g -nodefaultlibs -nostdlib \
    -Wl,--section-start,.interp=0x23000000 \
    -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
    -Wl,--whole-archive \
    "$G0/init.o" "$OUT/attn_flash.o" "$OUT/attn_flash_main.o" \
    "$QL/crt1.o" "$QL/debugmon.o" "$QL/libqurt.a" \
    "$G0/libc.a" "$G0/libqcc.a" "$G0/libhexagon.a" "$G0/libm.a" \
    "$QL/libqurtcfs.a" "$QL/libtimer_island.a" "$QL/libtimer_main.a" \
    "$QL/libposix.a" \
    -Wl,--no-whole-archive "$G0/fini.o" -o "$OUT/attn_flash_main"
echo "      $OUT/attn_flash_main"

echo "[2/2] hexagon-sim"
RUNELF=$S/rtos/qurt/computev68/sdksim_bin/runelf.pbn
LOG=$OUT/sim_attn_flash.log
printf '%s\n' "$QL/../debugger/lnx64/qurt_model.so" > "$OUT/osam.cfg"
printf '%s\n' "$TD/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$OUT/q6ss.cfg"
printf '%s\n' "$TD/lib/iss/l2vic.so 32 0xFC910000" >> "$OUT/q6ss.cfg"
set +e
"$SIM_BIN" -mv68 --simulated_returnval --usefs . \
    --cosim_file "$OUT/q6ss.cfg" --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" \
    "$RUNELF" -- "./$OUT/attn_flash_main" > "$LOG" 2>&1
QRC=$?
set -e
grep -E '^AF |^ATTN FLASH ' "$LOG" || true
if ! grep -q '^ATTN FLASH RESULT: PASS' "$LOG"; then
    echo "---- last sim lines ----" >&2
    tail -25 "$LOG" >&2
    echo "FAIL: no 'ATTN FLASH RESULT: PASS' (sim rc=$QRC, log $LOG)" >&2
    exit 1
fi
echo "PASS (sim rc=$QRC, log $LOG)"
