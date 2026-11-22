#!/bin/bash
# Build (if needed) and run the stand under qemu-system-hexagon.
#
#   ./runner.sh              # ./build.sh, then run
#   SKIP_BUILD=1 ./runner.sh # run the existing build/lfmsim
#
# Exit code: 0 = every check passed, 1 = test/build/emulator failure.
set -e
cd "$(dirname "$0")"

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
QEMU_BIN=${QEMU_BIN:-/tmp/opencode/qemu/build/qemu-system-hexagon}
OUT=${OUT:-build}
RUNELF=$HEXAGON_SDK_ROOT/rtos/qurt/computev68/sdksim_bin/runelf.pbn
LOG=$OUT/qemu.log

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    ./build.sh
fi

if [ ! -x "$OUT/lfmsim" ]; then
    echo "нет $OUT/lfmsim — сначала выполните ./build.sh" >&2
    exit 1
fi
if [ ! -x "$QEMU_BIN" ]; then
    echo "QEMU не найден: $QEMU_BIN" >&2
    echo "соберите qemu-system-hexagon или укажите QEMU_BIN=..." >&2
    exit 1
fi
if [ ! -f "$RUNELF" ]; then
    echo "runelf.pbn не найден: $RUNELF" >&2
    exit 1
fi

echo "RUN: $QEMU_BIN -M V68N_1024 -display none -kernel .../runelf.pbn -append ./$OUT/lfmsim"

set +e
"$QEMU_BIN" -M V68N_1024 -display none \
    -kernel "$RUNELF" -append "./$OUT/lfmsim" > "$LOG" 2>&1
QRC=$?
set -e

# QuRT boot log is noisy; the stand prefixes every line with "SIM ".
grep -E '^SIM ' "$LOG" || true

if ! grep -q '^SIM RESULT: PASS' "$LOG"; then
    echo "---- последние строки qemu ----" >&2
    tail -25 "$LOG" >&2
    echo "FAIL: стенд не дошёл до SIM RESULT: PASS (qemu rc=$QRC, лог $LOG)" >&2
    exit 1
fi

# Non-zero main() makes sdksim's runelf print "User Exit Code" and then take
# an exception during teardown (qemu rc=255), so the PASS marker is the
# authoritative success signal and the emulator code is reported, not used.
echo "PASS (qemu rc=$QRC, лог $LOG)"
