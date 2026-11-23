#!/bin/bash
# Build/run the v2 int4 GEMV stand (host check + hexagon-sim).
set -e
cd "$(dirname "$0")"
S=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}
TV=${HEXAGON_TOOLS_VERSION:-21.0.04}
TD=$S/tools/HEXAGON_Tools/$TV/Tools
CC=$TD/bin/hexagon-clang
SIM=$TD/bin/hexagon-sim
QU=$S/rtos/qurt/computev68/include
QL=$S/rtos/qurt/computev68/lib
G0=$TD/target/hexagon/lib/v68/G0
OUT=build
mkdir -p "$OUT"
echo "[1/2] host check"
${HOSTCC:-cc} -O2 -march=native -DGEMV2_HOST -I. -o "$OUT/gemv2_host" gemv2_main.c gemv2.c -lm
"$OUT/gemv2_host"
echo "[2/2] qexe"
CFLAGS="-mv68 -mhvx -mhvx-length=128b -G0 -O3 -Wall -I$S/incs -I$S/incs/stddef -I$TD/target/hexagon/include -I. -I$QU -I$QU/qurt -I$QU/posix"
$CC $CFLAGS -c gemv2.c -o "$OUT/gemv2.o"
$CC $CFLAGS -c gemv2_main.c -o "$OUT/gemv2_main.o"
$CC -mv68 -g -nodefaultlibs -nostdlib -Wl,--section-start,.interp=0x23000000 \
    -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
    -Wl,--whole-archive "$G0/init.o" "$OUT/gemv2.o" "$OUT/gemv2_main.o" \
    "$QL/crt1.o" "$QL/debugmon.o" "$QL/libqurt.a" "$G0/libc.a" "$G0/libqcc.a" \
    "$G0/libhexagon.a" "$G0/libm.a" "$QL/libqurtcfs.a" "$QL/libtimer_island.a" \
    "$QL/libtimer_main.a" "$QL/libposix.a" \
    -Wl,--no-whole-archive "$G0/fini.o" -o "$OUT/gemv2_main"
RUNELF=$S/rtos/qurt/computev68/sdksim_bin/runelf.pbn
LOG=$OUT/sim_gemv2.log
printf '%s\n' "$QL/../debugger/lnx64/qurt_model.so" > "$OUT/osam.cfg"
printf '%s\n' "$TD/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$OUT/q6ss.cfg"
printf '%s\n' "$TD/lib/iss/l2vic.so 32 0xFC910000" >> "$OUT/q6ss.cfg"
set +e
"$SIM" -mv68 --simulated_returnval --usefs . --cosim_file "$OUT/q6ss.cfg" \
    --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" "$RUNELF" -- "./$OUT/gemv2_main" > "$LOG" 2>&1
RC=$?
set -e
grep -E '^GEMV2 ' "$LOG" || true
grep -q '^GEMV2 RESULT: PASS' "$LOG" || { tail -20 "$LOG" >&2; echo "FAIL (rc=$RC)" >&2; exit 1; }
echo "PASS (rc=$RC)"
