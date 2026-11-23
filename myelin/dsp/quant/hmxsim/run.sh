#!/bin/sh
# Run a probe/HMX test under hexagon-sim. Files are opened via --usefs, so
# run from the directory that holds them (absolute paths also work there).
#   ./run.sh probe_bias
#   ./run.sh probe_round
#   ./run.sh hmxtest CASE.bin FIXED.bin RAW.bin
set -e
cd "$(dirname "$0")"
TD=${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk-6.7.0.0}/tools/HEXAGON_Tools/21.0.04/Tools
SIM=$TD/bin/hexagon-sim
[ -x "$SIM" ] || SIM=$(command -v hexagon-sim)
BIN=$1; shift || true
exec "$SIM" -mv68 --mhmx 1 --usefs . "./$BIN" -- "$@"
