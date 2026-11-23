#!/bin/bash
set -e
cd "$(dirname "$0")"

CC=${CC:-cc}
OUT=${OUT:-/tmp/opencode/lfm_dsp_hvx_test}
SRC="exec_main.c exec.c ffn_i4.c"

if [ "${SKIP_REAL:-0}" != "1" ]; then
    for f in gemv_i4.c gemm_i4.c attn_flash.c kvcache.c; do
        if [ -f "$f" ]; then
            SRC="$SRC $f"
        fi
    done
fi
SRC="$SRC exec_stub.c"

mkdir -p "$(dirname "$OUT")"
$CC -std=c11 -O2 -Wall -Wextra $SRC -lm -o "$OUT"
"$OUT"
