#!/bin/bash
set -eu
cd "$(dirname "$0")"

CC=${CC:-cc}
OUT=${OUT:-build}
CFLAGS=${CFLAGS:--O2 -std=c11 -Wall -Wextra}

mkdir -p "$OUT"

$CC $CFLAGS -I. -I../common -c kvcache.c -o "$OUT/kvcache.o"
$CC $CFLAGS -I. -I../common -c attn_flash.c -o "$OUT/attn_flash.o"
$CC $CFLAGS -I. -I../common attn_main.c "$OUT/kvcache.o" "$OUT/attn_flash.o" -lm \
    -o "$OUT/attn_main"

"$OUT/attn_main"

if [ -n "${HEXAGON_SDK_ROOT:-}" ]; then
    HEXAGON_TOOLS_VERSION=${HEXAGON_TOOLS_VERSION:-21.0.04}
    TD=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
    HCC=$TD/bin/hexagon-clang
    if [ ! -x "$HCC" ]; then
        echo "hexagon-clang не найден: $HCC" >&2
        exit 1
    fi
    DOUT=$OUT/dsp
    mkdir -p "$DOUT"
    HFLAGS="-mv68 -mhvx -mhvx-length=128b -G0 -O2 -std=c11 -Wall -Wextra
        -I$HEXAGON_SDK_ROOT/incs -I$HEXAGON_SDK_ROOT/incs/stddef -I. -I../common"
    $HCC $HFLAGS -c kvcache.c -o "$DOUT/kvcache.o"
    $HCC $HFLAGS -c attn_flash.c -o "$DOUT/attn_flash.o"
    echo "DSP BUILD OK: $PWD/$DOUT"
fi
