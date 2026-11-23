#!/bin/sh
set -e
cd "$(dirname "$0")"

OUT=${OUT:-build}
CC=${CC:-cc}
CFLAGS_HOST=${CFLAGS_HOST:-"-O2 -std=c11 -Wall -Wextra"}

mkdir -p "$OUT"

echo "== host: $CC $CFLAGS_HOST"
$CC $CFLAGS_HOST -o "$OUT/gemv_host" gemv_main.c gemv_i4.c -lm
$CC $CFLAGS_HOST -o "$OUT/gemm_host" gemm_main.c gemm_i4.c gemv_i4.c -lm
echo "host ok: $OUT/gemv_host $OUT/gemm_host"

find_hexagon_clang()
{
    for root in "$HEXAGON_SDK_ROOT" /opt/hexagon-sdk-6.7.0.0 /opt/hexagon-sdk; do
        [ -n "$root" ] || continue
        [ -d "$root/tools/HEXAGON_Tools" ] || continue
        for c in "$root"/tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang; do
            if [ -x "$c" ]; then
                echo "$c"
                return 0
            fi
        done
    done
    return 1
}

HCC=$(find_hexagon_clang || true)

if [ -z "$HCC" ]; then
    echo "== dsp: hexagon-clang не найден (HEXAGON_SDK_ROOT), DSP-вариант пропущен"
    exit 0
fi

HFLAGS="-mv68 -mhvx -mhvx-length=128b -O2 -Wall -Wextra"
echo "== dsp: $HCC $HFLAGS"
$HCC $HFLAGS -static -o "$OUT/gemv_sim" gemv_main.c gemv_i4.c -lm
$HCC $HFLAGS -static -o "$OUT/gemm_sim" gemm_main.c gemm_i4.c gemv_i4.c -lm
echo "dsp ok: $OUT/gemv_sim $OUT/gemm_sim"
