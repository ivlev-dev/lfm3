#!/bin/bash
# Push + run the multi-PD series on the phone.
#
#   ./run_multi.sh push
#   ./run_multi.sh base2        # two N=1 baselines 30 s apart (noise check)
#   ./run_multi.sh series       # N=1..4, slice + full, 2 runs each
#   ./run_multi.sh probe        # heap ceiling + session limit
#   ./run_multi.sh raw <args>   # pass args straight to multi_test
#
# Device dir: MDEV (default /data/local/tmp/lfmgemv_multi).
set -e
cd "$(dirname "$0")"

D=${MDEV:-/data/local/tmp/lfmgemv_multi}
if [ -z "$MADB" ]; then
    MADB=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
[ -n "$MADB" ] || { echo "no adb device: set MADB=<serial>" >&2; exit 1; }
REPS=${REPS:-20}
ROUNDS=${ROUNDS:-5}

push() {
    adb -s "$MADB" shell mkdir -p "$D"
    adb -s "$MADB" push multi_test ../liblfmgemv_skel.so "$D/" >/dev/null
    adb -s "$MADB" shell chmod 755 "$D/multi_test" "$D/liblfmgemv_skel.so"
    echo "pushed -> $D"
}

run() {
    adb -s "$MADB" shell "cd $D && ADSP_LIBRARY_PATH=\$PWD ./multi_test $*"
}

case "${1:-series}" in
push)  push ;;
base2) run "mode=base2 reps=$REPS rounds=$ROUNDS gap=${GAP:-30}" ;;
probe) run "mode=heap"; run "mode=sessions" ;;
series)
    for n in 1 2 3 4; do
        run "mode=main n=$n reps=$REPS rounds=$ROUNDS"
        run "mode=main n=$n reps=$REPS rounds=$ROUNDS"
    done
    ;;
raw) shift; run "$*" ;;
*) echo "usage: $0 [push|base2|probe|series|raw <args>]" >&2; exit 1 ;;
esac
