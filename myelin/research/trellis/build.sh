#!/bin/sh
# Build the TCQ rate-distortion study (offline, dev machine only).
set -e
cd "$(dirname "$0")"
cc -O2 -I ../../dsp/hvx -o dump_act dump_act.c -lm
cc -O2 -I ../../dsp/hvx -o tcq tcq.c -lm
echo "ok: ./dump_act ./tcq"
