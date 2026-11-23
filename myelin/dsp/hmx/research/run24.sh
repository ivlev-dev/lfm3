#!/bin/bash
# REPORT24: block scales on the int4 prefill path.
#
# Host-only part (works offline, no device):
#   ./research/run24.sh host      -> research/run24-host.txt
# Device part (needs MADB; the phone is offline today -> "pending"):
#   export MADB=<serial>
#   G=1 ./deploy.sh weights; G=2 ...; G=4 ...; G=8 ...
#   ./research/run24.sh device    -> research/run24-device.txt
#
# mode for blkpref: 1 = staging DDR->VTCM (double buffered), 2 = staging only,
# 64 = evict L2 (cold DDR), 8 = write C (correctness/combine), 4 = keep locks.
set -e
cd "$(dirname "$0")/.."

case "${1:-host}" in
host)
LOG=research/run24-host.txt
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG" | grep -vE "^unsigned PD"; }

say "=== Q4_K-free check: our int4/32 format, block size sweep (mkblob fidelity) ==="
say "REPORT23 reference: block-32 cos=0.995341, per-row cos=0.988286 (f32 activations)"
MODEL=${MODEL:-../../../models/LFM3-1.7B-Q4_K_M.gguf}
[ -x ./mkblob ] || ./build.sh >/dev/null
for g in 1 2 4 8 16; do
    TENSOR=blk.0.ffn_gate.weight ./mkblob "$MODEL" blk.0.ffn_gate.weight \
        /tmp/opencode/r24_g$g $g 2>&1 | grep -E "tensor|block-|per-row scale :" | tee -a "$LOG"
done

say ""
say "=== part 2: accuracy/cost model of the DSP path (blkstat, real weights) ==="
say "cos16 = exact 16-bit partial; cos8(unifKc) = 8-bit with one global store"
say "multiplier; cos8(perGrp) = 8-bit normalised per (n,group) -- the recipe"
say "mx4_blkpref implements.  Criterion: cos8(perGrp) >= 0.9995."
./blkstat "$MODEL" blk.0.ffn_gate.weight 7680 8 2>&1 | tee -a "$LOG"

say ""
say "=== part 3: second FFN shape (down: N=2048 K=7680) ==="
./blkstat "$MODEL" blk.0.ffn_down.weight 2048 8 2>&1 | tee -a "$LOG"

say ""
say "=== part 4: hexagon-sim functional check of the full recipe ==="
say "(offset packet + 8-bit normalised store + per-group combine, CPU reference)"
( cd sim && ./build.sh scale >/dev/null && ./run.sh scale ) 2>&1 | tee -a "$LOG"

say ""
say "=== part 5: HMX store semantics that forced this design ==="
( cd sim && ./run.sh u16 && ./run.sh ctrl ) 2>&1 \
    | grep -E "scale-word|word=|spatial|live|rows matching|acc=|out_bias|uh :|ub :" \
    | tee -a "$LOG"
say ""
say "written: $LOG"
;;

device)
LOG=research/run24-device.txt
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
run() { echo "\$ mx4_test $*" | tee -a "$LOG"; ./deploy.sh "$@" 2>&1 \
        | grep -vE "^unsigned PD" | tee -a "$LOG"; }

say "=== blkpref, per G: full path, cold DDR, M=128, gate 7680x2048 ==="
for g in 1 2 4 8 16; do
    say "--- G=$g (block = $((32*g)) K).  Weights must be regenerated:"
    say "    G=$g ./deploy.sh weights"
    for r in 1 2 3; do run blkpref 128 7680 2048 $g 65 1; done
    say "--- G=$g: staging only (copy cost, mode 2) and compute+combine resident"
    run blkpref 128 7680 2048 $g 67 1      # 64|2|1: cold + staging only
    run blkpref 128 7680 2048 $g 4 2       # resident weights, locked
    say "--- G=$g: correctness (mode 8|64|1 = 73), 8 rows x N=7680 + 4x64 slice"
    run blkpref 128 7680 2048 $g 73 1
done

say ""
say "=== baseline for comparison: i4pref (no block scales), same protocol ==="
run i4pref 128 7680 2048 81 1
run i4pref 128 7680 2048 83 1

say ""
say "=== staged as staged: which combine dominates?  mode 8 alone ==="
say "(compare blkpref 73 vs i4pref 83: the delta is the whole price of the"
say " block scales: offset packets + partial stores + combine)"
say "written: $LOG"
;;
*) echo "usage: $0 [host|device]" >&2; exit 2 ;;
esac
