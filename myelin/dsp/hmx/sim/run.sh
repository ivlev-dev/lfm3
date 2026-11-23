#!/bin/sh
# Run one REPORT24 probe under hexagon-sim.
#   ./run.sh            # default VTCM base 0xd8400000, hmx_probe
#   ./run.sh blk        # store/scale semantics      (hmx_blk_probe)
#   ./run.sh ctrl       # control word + bias words   (hmx_ctrl_probe)
#   ./run.sh u16        # .uh spatial map             (hmx_u16map)
#   ./run.sh scale      # end-to-end block-scale recipe (hmx_blkscale_sim)
#   ./run.sh 0xd8400000 1   -> run hmx_probe with an explicit base / XE2 flag
set -e
cd "$(dirname "$0")"
TD=${HEXAGON_TOOLS:-/opt/hexagon-sdk-6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools}
SIM="$TD/bin/hexagon-sim"
[ -x "$SIM" ] || SIM=$(command -v hexagon-sim)

case "${1:-}" in
blk)   BIN=hmx_blk_probe ;;
ctrl)  BIN=hmx_ctrl_probe ;;
part)  BIN=hmx_part_probe ;;
part2) BIN=hmx_part_probe2 ;;
part3) BIN=hmx_part_probe3 ;;
u16)   BIN=hmx_u16map ;;
u16b)  BIN=hmx_u16map2 ;;
scale) BIN=hmx_blkscale_sim ;;
*)     BIN=hmx_probe ;;
esac

if [ "$BIN" = hmx_probe ]; then
    BASE=${1:-0xd8400000}
    XE2=${2:-1}
    exec "$SIM" -mv68 --mhmx 1 ./hmx_probe -- "$BASE" "$XE2"
fi
exec "$SIM" -mv68 --mhmx 1 "./$BIN"
