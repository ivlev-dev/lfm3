#!/bin/sh
# Run the HMX/VTCM probe under hexagon-sim.
#   ./run.sh            # default VTCM base 0xd8400000
#   ./run.sh 0xd8400000 1     # 2nd arg 0 => leave SSR.XE2 alone (shows 0x18)
set -e
TD=${HEXAGON_TOOLS:-/opt/hexagon-sdk-6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools}
SIM="$TD/bin/hexagon-sim"
[ -x "$SIM" ] || SIM=$(command -v hexagon-sim)

BASE=${1:-0xd8400000}
XE2=${2:-1}

exec "$SIM" -mv68 --mhmx 1 ./hmx_probe -- "$BASE" "$XE2"
