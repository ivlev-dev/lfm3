#!/bin/bash
# Build (if needed) and run the stand under hexagon-sim (QuRT cosim).
#
#   ./runner.sh              # build + run under hexagon-sim (QuRT cosim)
#   USE_QEMU=1 ./runner.sh   # старый путь через qemu-system-hexagon
#   SKIP_BUILD=1 ./runner.sh # run the existing build/lfmsim
#
# Exit code: 0 = every check passed, 1 = test/build/emulator failure.
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
QEMU_BIN=${QEMU_BIN:-/tmp/opencode/qemu/build/qemu-system-hexagon}
OUT=${OUT:-build}
RUNELF=$HEXAGON_SDK_ROOT/rtos/qurt/computev68/sdksim_bin/runelf.pbn
LOG=$OUT/sim.log

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    ./build.sh
fi

if [ ! -x "$OUT/lfmsim" ]; then
    echo "нет $OUT/lfmsim — сначала выполните ./build.sh" >&2
    exit 1
fi
TD=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
SIM=$TD/bin/hexagon-sim
if [ "${USE_QEMU:-0}" != "1" ] && [ ! -x "$SIM" ]; then
    SIM=$(command -v hexagon-sim || true)
fi
if [ "${USE_QEMU:-0}" = "1" ] && [ ! -x "$QEMU_BIN" ]; then
    echo "QEMU не найден: $QEMU_BIN" >&2
    exit 1
fi
if [ "${USE_QEMU:-0}" != "1" ] && [ ! -x "$SIM" ]; then
    echo "hexagon-sim не найден: $SIM (проверьте HEXAGON_SDK_ROOT)" >&2
    exit 1
fi
if [ ! -f "$RUNELF" ]; then
    echo "runelf.pbn не найден: $RUNELF" >&2
    exit 1
fi

if [ "${USE_QEMU:-0}" = "1" ]; then
    echo "RUN: $QEMU_BIN -M V68N_1024 -display none -kernel .../runelf.pbn -append ./$OUT/lfmsim"
    set +e
    "$QEMU_BIN" -M V68N_1024 -display none \
        -kernel "$RUNELF" -append "./$OUT/lfmsim" > "$LOG" 2>&1
    QRC=$?
    set -e
else
    QL=$HEXAGON_SDK_ROOT/rtos/qurt/computev68/lib
    printf '%s\n' "$QL/../debugger/lnx64/qurt_model.so" > "$OUT/osam.cfg"
    printf '%s\n' "$TD/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$OUT/q6ss.cfg"
    printf '%s\n' "$TD/lib/iss/l2vic.so 32 0xFC910000" >> "$OUT/q6ss.cfg"
    echo "RUN: $SIM -mv68 --rtos .../runelf.pbn -- ./$OUT/lfmsim"
    set +e
    "$SIM" -mv68 --simulated_returnval --usefs . \
        --cosim_file "$OUT/q6ss.cfg" --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" \
        "$RUNELF" -- "./$OUT/lfmsim" > "$LOG" 2>&1
    QRC=$?
    set -e
fi

# QuRT boot log is noisy; the stand prefixes every line with "SIM ".
grep -E '^SIM ' "$LOG" || true

if ! grep -q '^SIM RESULT: PASS' "$LOG"; then
    echo "---- последние строки hexagon-sim ----" >&2
    tail -25 "$LOG" >&2
    echo "FAIL: стенд не дошёл до SIM RESULT: PASS (sim rc=$QRC, лог $LOG)" >&2
    exit 1
fi

# Non-zero main() makes sdksim's runelf print "User Exit Code" and then take
# an exception during teardown (sim rc=255), so the PASS marker is the
# authoritative success signal and the emulator code is reported, not used.
echo "PASS (sim rc=$QRC, лог $LOG)"
