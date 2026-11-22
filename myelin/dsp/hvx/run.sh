#!/bin/bash
# Пуш и запуск HVX-тестов на телефоне.
#
#   ./run.sh push                     — залить скелет и все тесты
#   ./run.sh <binary> [args...]        — запустить тест (lfmgemv_test|ab_test|ffn_test|attn_test)
#   ./run.sh attn_test 64 0 2          — пример: 64 страницы (ctx 2048), полное окно
#
# Серийник adb: переменная MADB (по умолчанию — первое устройство из `adb devices`).
# Каталог на устройстве: переменная MDEV (по умолчанию /data/local/tmp/lfmgemv).
set -e
cd "$(dirname "$0")"

BINS="liblfmgemv_skel.so lfmgemv_test ab_test ffn_test attn_test"
D=${MDEV:-/data/local/tmp/lfmgemv}
if [ -z "$MADB" ]; then
    MADB=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
fi
[ -n "$MADB" ] || { echo "нет adb-устройства: задайте MADB=<серийник>" >&2; exit 1; }

adb -s "$MADB" shell mkdir -p "$D"
if [ "$1" = "push" ]; then
    adb -s "$MADB" push $BINS "$D/" >/dev/null
    adb -s "$MADB" shell chmod 755 "$D"/liblfmgemv_skel.so "$D"/*_test
    echo "pushed -> $D"
    exit 0
fi

BIN=${1:-lfmgemv_test}
shift || true
adb -s "$MADB" shell "cd $D && ADSP_LIBRARY_PATH=\$PWD ./$BIN $*"
