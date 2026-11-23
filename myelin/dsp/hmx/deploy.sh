#!/bin/bash
# Пуш и запуск стенда HMX на телефоне.
#
#   ./deploy.sh push           — залить libmx4_skel.so и mx4_test
#   ./deploy.sh weights        — сгенерировать блобы весов из GGUF и залить их
#                                (G=<croutons per block scale> для blkpref)
#   ./deploy.sh <args...>      — выполнить mx4_test <args> на устройстве
#
# Серийник adb: переменная MADB (по умолчанию — первое устройство из `adb devices`).
# Каталог на устройстве: переменная MDEV (по умолчанию /data/local/tmp/hmx4).
set -e
cd "$(dirname "$0")"

D=${MDEV:-/data/local/tmp/hmx4}
if [ -z "$MADB" ]; then
    MADB=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
[ -n "$MADB" ] || { echo "нет adb-устройства: задайте MADB=<серийник>" >&2; exit 1; }

adb -s "$MADB" shell mkdir -p "$D"
if [ "$1" = "push" ]; then
    adb -s "$MADB" push libmx4_skel.so mx4_test "$D/" >/dev/null
    adb -s "$MADB" shell chmod 777 "$D/libmx4_skel.so" "$D/mx4_test"
    echo "pushed -> $D"
    exit 0
fi
if [ "$1" = "weights" ]; then
    # GGUF читается здесь, на dev-машине; на телефон уезжают только блобы.
    MODEL=${MODEL:-../../../models/LFM3-1.7B-Q4_K_M.gguf}
    TENSOR=${TENSOR:-blk.0.ffn_gate.weight}
    WOUT=${WOUT:-/tmp/opencode/hmxw}
    NAME=${WBLOB:-gate}
    mkdir -p "$WOUT"
    [ -x ./mkblob ] || ./build.sh >/dev/null
    ./mkblob "$MODEL" "$TENSOR" "$WOUT/$NAME" "${G:-1}"
    adb -s "$MADB" push "$WOUT/$NAME.i4" "$WOUT/$NAME.i4s" "$WOUT/$NAME.i8" \
        "$WOUT/$NAME.meta" "$D/" >/dev/null
    echo "weights -> $D/$NAME.{i4,i4s,i8} (from $WOUT, G=${G:-1}, B=$((32*${G:-1})))"
    exit 0
fi
adb -s "$MADB" shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=\$PWD WNAME=${WNAME:-gate} ./mx4_test $*"
