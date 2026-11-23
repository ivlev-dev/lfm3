#!/bin/bash
# REPORT23: матрица замеров int4 vs int8 prefill на реальных весах FFN из GGUF.
#
#   export MADB=<adb-серийник>
#   ./deploy.sh weights                                   # gate.i4/.i8 (+ .meta)
#   TENSOR=blk.0.ffn_down.weight WBLOB=down ./deploy.sh weights
#   ./research/run23.sh [out.log]
#
# mode для i4pref: 1 = стейджинг DDR->VTCM, 16 = двойная буферизация,
# 64 = вытеснить L2 перед замером (холодный DDR), 2 = только стейджинг,
# 8 = собрать C, 256 = sparse-активации, 12..15 = глубина стейджинга-1.
set -e
cd "$(dirname "$0")/.."
LOG=${1:-research/run23.txt}
: > "$LOG"

run() { echo "\$ mx4_test $*" | tee -a "$LOG"; ./deploy.sh "$@" 2>&1 | grep -vE "^unsigned PD" | tee -a "$LOG"; }
runn() { echo "\$ WNAME=$WNAME mx4_test $*" | tee -a "$LOG"; WNAME=$WNAME ./deploy.sh "$@" 2>&1 | grep -vE "^unsigned PD" | tee -a "$LOG"; }

echo "=== int4 gate 7680x2048, холодный DDR (L2 evict), M-скейлинг ===" | tee -a "$LOG"
for r in 1 2 3; do
    for m in 32 64 128; do run i4pref $m 7680 2048 81 1; done
done
echo "=== int4 gate, глубина стейджинга 4 и 16 (M=128) ===" | tee -a "$LOG"
for r in 1 2 3; do run i4pref 128 7680 2048 12369 1; done
for r in 1 2 3; do run i4pref 128 7680 2048 61521 1; done
echo "=== int4 gate, reps=3 (L2 тёплый) ===" | tee -a "$LOG"
for m in 32 64 128; do run i4pref $m 7680 2048 17 3; done
echo "=== int4 gate: только стейджинг / только счёт ===" | tee -a "$LOG"
for r in 1 2 3; do run i4pref 128 7680 2048 83 1; done
for m in 32 64 128; do run i4pref $m 7680 2048 0 3; done
echo "=== int4 down 2048x7680 (второй тип матрицы FFN) ===" | tee -a "$LOG"
WNAME=down
for r in 1 2 3; do runn i4pref 128 2048 7680 81 1; done
runn i4pref 128 2048 7680 83 1
runn i4pref 128 2048 7680 0 3
WNAME=gate
echo "=== int8 (pref8) на тех же весах, тот же протокол ===" | tee -a "$LOG"
for r in 1 2 3; do for m in 32 64 128; do run pref8 $m 7680 2048 1 1; done; done
for m in 32 64 128; do run pref8 $m 7680 2048 1 3; done
for m in 32 64 128; do run pref8 $m 7680 2048 0 3; done
echo "=== int8 down ===" | tee -a "$LOG"
WNAME=down
runn pref8 128 2048 7680 1 1
runn pref8 128 2048 7680 0 3
WNAME=gate
echo "=== корректность на реальных весах (sparse, сбор C) ===" | tee -a "$LOG"
run i4pref 128 7680 2048 265 1
run i4pref 32 7680 2048 265 1
run pref8 128 7680 2048 9 1
echo "=== decode N=1, реальная строка 0 ===" | tee -a "$LOG"
run dec 2048 2000

echo "written: $LOG"
