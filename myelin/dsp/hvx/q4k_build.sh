#!/bin/bash
# Build and run the Q4_K stand:
#   1. host tool  -> build/q4k_host          (generation + case extraction)
#   2. case header-> build/q4k_case.h        (real weights + real activation)
#   3. DSP qexe   -> build/q4k_main          (HVX kernel under QEMU)
#   4. run it     -> prints "Q4K RESULT: PASS/FAIL"
#
#   ./q4k_build.sh              # full build + run
#   SKIP_CASE=1 ./q4k_build.sh  # reuse build/q4k_case.h
#   HEXAGON_SDK_ROOT=/opt/hexagon-sdk-6.7.0.0 QEMU_BIN=... ./q4k_build.sh
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
QEMU_BIN=${QEMU_BIN:-/tmp/opencode/qemu/build/qemu-system-hexagon}
SIM_BIN=${SIM_BIN:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools/bin/hexagon-sim}
OUT=${OUT:-build}
MODEL=${MODEL:-../../../models/LFM3-1.7B-Q4_K_M.gguf}
CASE_ROWS=${CASE_ROWS:-512}

S=$HEXAGON_SDK_ROOT
TD=$S/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
CC=$TD/bin/hexagon-clang
QU=$S/rtos/qurt/computev68/include
QL=$S/rtos/qurt/computev68/lib
G0=$TD/target/hexagon/lib/v68/G0
HOSTCC=${HOSTCC:-cc}
HVX=.
QUANT=../quant
SIM=../sim

mkdir -p "$OUT"

# ---- 1. host tool -------------------------------------------------------
echo "[1/4] host tool"
$HOSTCC -O2 -march=native -fopenmp -I$QUANT -o "$OUT/q4k_host" \
    q4k_host.c q4k_impl.c -lm
echo "      $OUT/q4k_host"

# ---- 2. case header ----------------------------------------------------
if [ "${SKIP_CASE:-0}" != "1" ]; then
    echo "[2/4] case header (real blk.0.ffn_gate.weight rows + layer-0 activation)"
    "$OUT/q4k_host" --model "$MODEL" --emit-case "$OUT/q4k_case.h" \
        --case-rows "$CASE_ROWS"
else
    echo "[2/4] SKIP_CASE: reusing $OUT/q4k_case.h"
fi

# ---- 3. qexe -----------------------------------------------------------
echo "[3/4] qexe"
if [ ! -x "$CC" ]; then
    echo "hexagon-clang not found: $CC (set HEXAGON_SDK_ROOT)" >&2
    exit 1
fi
CFLAGS="-mv68 -mhvx -mhvx-length=128b -G0 -O2 -Wall \
        -I$S/incs -I$S/incs/stddef -I$TD/target/hexagon/include \
        -I$HVX -I$QUANT -I$SIM -I$OUT -I$QU -I$QU/qurt -I$QU/posix"
$CC $CFLAGS -c q4k_impl.c -o "$OUT/q4k_impl.o"
$CC $CFLAGS -c q4k_main.c -o "$OUT/q4k_main.o"

$CC -mv68 -g -nodefaultlibs -nostdlib \
    -Wl,--section-start,.interp=0x23000000 \
    -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
    -Wl,--whole-archive \
    "$G0/init.o" "$OUT/q4k_impl.o" "$OUT/q4k_main.o" \
    "$QL/crt1.o" "$QL/debugmon.o" "$QL/libqurt.a" \
    "$G0/libc.a" "$G0/libqcc.a" "$G0/libhexagon.a" "$G0/libm.a" \
    "$QL/libqurtcfs.a" "$QL/libtimer_island.a" "$QL/libtimer_main.a" \
    "$QL/libposix.a" \
    -Wl,--no-whole-archive "$G0/fini.o" -o "$OUT/q4k_main"
echo "      $OUT/q4k_main"

# ---- 4. run -----------------------------------------------------------
echo "[4/4] hexagon-sim"
RUNELF=$S/rtos/qurt/computev68/sdksim_bin/runelf.pbn
LOG=$OUT/sim_q4k.log
QL=$S/rtos/qurt/computev68/lib
printf '%s\n' "$QL/../debugger/lnx64/qurt_model.so" > "$OUT/osam.cfg"
printf '%s\n' "$TD/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$OUT/q6ss.cfg"
printf '%s\n' "$TD/lib/iss/l2vic.so 32 0xFC910000" >> "$OUT/q6ss.cfg"
set +e
"$SIM_BIN" -mv68 --simulated_returnval --usefs . \
    --cosim_file "$OUT/q6ss.cfg" --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" \
    "$RUNELF" -- "./$OUT/q4k_main" > "$LOG" 2>&1
QRC=$?
set -e
grep -E '^Q4K ' "$LOG" || true
if ! grep -q '^Q4K RESULT: PASS' "$LOG"; then
    echo "---- last qemu lines ----" >&2
    tail -25 "$LOG" >&2
    echo "FAIL: no 'Q4K RESULT: PASS' (sim rc=$QRC, log $LOG)" >&2
    exit 1
fi
echo "PASS (sim rc=$QRC, log $LOG)"
