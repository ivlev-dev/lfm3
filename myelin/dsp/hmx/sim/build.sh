#!/bin/sh
# Build the REPORT24 probes for hexagon-sim (SDK 6.7, Hexagon v68).
#
#   ./build.sh          # hmx_probe (REPORT22) + every REPORT24 probe
#   ./build.sh blk      # only hmx_blk_probe  (store/scale semantics)
set -e
cd "$(dirname "$0")"
TD=${HEXAGON_TOOLS:-/opt/hexagon-sdk-6.7.0.0/tools/HEXAGON_Tools/21.0.04/Tools}
CC="$TD/bin/hexagon-clang"
[ -x "$CC" ] || CC=$(command -v hexagon-clang)
FLAGS="-mv68 -mhmx -mhvx -mhvx-length=128b -O2 -Wall -Wno-unused-function"

one() { "$CC" $FLAGS -o "$1" "$2"; echo "built: $(pwd)/$1"; }

case "${1:-all}" in
blk)      one hmx_blk_probe hmx_blk_probe.c ;;
ctrl)     one hmx_ctrl_probe hmx_ctrl_probe.c ;;
part)     one hmx_part_probe hmx_part_probe.c ;;
part2)    one hmx_part_probe2 hmx_part_probe2.c ;;
part3)    one hmx_part_probe3 hmx_part_probe3.c ;;
u16)      one hmx_u16map hmx_u16map.c ;;
u16b)     one hmx_u16map2 hmx_u16map2.c ;;
scale)    one hmx_blkscale_sim hmx_blkscale_sim.c ;;
all)
    one hmx_probe hmx_vtcm_probe.c
    one hmx_blk_probe hmx_blk_probe.c
    one hmx_ctrl_probe hmx_ctrl_probe.c
    one hmx_part_probe hmx_part_probe.c
    one hmx_part_probe2 hmx_part_probe2.c
    one hmx_part_probe3 hmx_part_probe3.c
    one hmx_u16map hmx_u16map.c
    one hmx_u16map2 hmx_u16map2.c
    one hmx_blkscale_sim hmx_blkscale_sim.c
    ;;
*) echo "usage: $0 [all|blk|ctrl|part|part2|part3|u16|u16b|scale]" >&2; exit 2 ;;
esac
