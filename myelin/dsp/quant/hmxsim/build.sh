#!/bin/sh
# Build the HMX zero-point tools for hexagon-sim (SDK 6.7, v68).
#   probe_bias    - bias register semantics (int32 input-bias word, scale)
#   probe_round   - int8 store truncation / saturation
#   hmxtest       - full int4/32 GEMM with uint8(zp) activations
set -e
cd "$(dirname "$0")"
TD=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}/tools/HEXAGON_Tools/21.0.04/Tools
CC=$TD/bin/hexagon-clang
[ -x "$CC" ] || CC=$(command -v hexagon-clang)
FLAGS="-mv68 -mhmx -mhvx -mhvx-length=128b -O2 -Wall"
"$CC" $FLAGS -o probe_bias probe_bias.c
"$CC" $FLAGS -o probe_round probe_round.c
"$CC" $FLAGS -o hmxtest hmxtest.c
echo "built: probe_bias probe_round hmxtest"
