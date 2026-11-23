#!/bin/sh
# Reproduce REPORT26.md.
#   1) dump real layer-0 FFN inputs (4 tokens, ~16 s)
#   2) run the rate-distortion study over 512 rows, 3 rotations (~4 min)
#   3) stability check on a different row range
set -e
cd "$(dirname "$0")"
M=${1:-../../../models/LFM3-1.7B-Q4_K_M.gguf}

./dump_act "$M" act0.bin 0,1,2 4
./tcq "$M" 512 act0.bin > results_final.txt
TQ_ROW0=4096 ./tcq "$M" 256 act0.bin > results_row4096.txt
echo "wrote results_final.txt results_row4096.txt"
