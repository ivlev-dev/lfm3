#!/bin/sh
set -e
cd "$(dirname "$0")"

OUT=${OUT:-build}
SIM_TIMEOUT=${SIM_TIMEOUT:-1200}

./build.sh

rc=0
echo "== gemv (host)"
./"$OUT"/gemv_host || rc=1
echo "== gemm (host)"
./"$OUT"/gemm_host || rc=1

find_hexagon_sim()
{
    for root in "$HEXAGON_SDK_ROOT" /opt/hexagon-sdk-6.7.0.0 /opt/hexagon-sdk; do
        [ -n "$root" ] || continue
        [ -d "$root/tools/HEXAGON_Tools" ] || continue
        for s in "$root"/tools/HEXAGON_Tools/*/Tools/bin/hexagon-sim; do
            if [ -x "$s" ]; then
                echo "$s"
                return 0
            fi
        done
    done
    return 1
}

HSIM=$(find_hexagon_sim || true)

if [ -z "$HSIM" ] || [ ! -x "$OUT/gemv_sim" ]; then
    echo "== dsp: hexagon-sim или DSP-бинарники отсутствуют, прогон пропущен"
    exit $rc
fi

for t in gemv gemm; do
    echo "== $t (hexagon-sim)"
    if timeout "$SIM_TIMEOUT" "$HSIM" -mv68 --simulated_returnval ./"$OUT"/"$t"_sim; then
        :
    else
        echo "FAIL $t: hexagon-sim"
        rc=1
    fi
    echo "== $t bench (hexagon-sim)"
    if timeout "$SIM_TIMEOUT" "$HSIM" -mv68 --simulated_returnval ./"$OUT"/"$t"_sim -- bench; then
        :
    else
        echo "FAIL $t bench: hexagon-sim"
        rc=1
    fi
done

exit $rc
