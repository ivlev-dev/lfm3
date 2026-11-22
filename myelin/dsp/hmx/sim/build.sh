#!/bin/sh
# Build the HMX/VTCM probe for hexagon-sim (SDK 6.7, v68).
set -e
TD=${HEXAGON_TOOLS:-/opt/hexagon-sdk-6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools}
CC="$TD/bin/hexagon-clang"
[ -x "$CC" ] || CC=$(command -v hexagon-clang)

"$CC" -mv68 -mhmx -mhvx -mhvx-length=128b -O2 -Wall -o hmx_probe hmx_vtcm_probe.c
echo "built: $(pwd)/hmx_probe"
