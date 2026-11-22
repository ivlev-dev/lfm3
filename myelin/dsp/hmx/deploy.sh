#!/bin/bash
# Пуш и запуск стенда HMX на телефоне.
#
#   ./deploy.sh push           — залить libmx4_skel.so и mx4_test
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
adb -s "$MADB" shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=\$PWD ./mx4_test $*"
