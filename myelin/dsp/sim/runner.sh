#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BUILD=${BUILD_DIR:-$HERE/build}
MANIFEST=$BUILD/manifest.txt
TIMEOUT=${LFM_SIM_TIMEOUT:-300}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    set +e
    "$HERE/build.sh"
    BUILD_RC=$?
    set -e
    if [ "$BUILD_RC" != "0" ] && [ ! -f "$MANIFEST" ]; then
        echo "FAIL: build.sh завершился с кодом $BUILD_RC без manifest" >&2
        exit 1
    fi
fi

if [ ! -f "$MANIFEST" ]; then
    echo "FAIL: нет $MANIFEST, сначала выполните ./build.sh" >&2
    exit 1
fi

find_qemu() {
    local c
    if [ -n "${QEMU_BIN:-}" ] && [ -x "$QEMU_BIN" ]; then
        printf '%s\n' "$QEMU_BIN"
        return 0
    fi
    for c in /tmp/opencode/qemu/build/qemu-system-hexagon \
             /usr/local/bin/qemu-system-hexagon; do
        if [ -x "$c" ] && "$c" -M help 2>/dev/null | grep -q V68N_1024; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    if c=$(command -v qemu-system-hexagon 2>/dev/null); then
        if "$c" -M help 2>/dev/null | grep -q V68N_1024; then
            printf '%s\n' "$c"
            return 0
        fi
    fi
    return 1
}

detect_sdk() {
    if [ -n "${HEXAGON_SDK_ROOT:-}" ] &&
       [ -d "$HEXAGON_SDK_ROOT/rtos/qurt/computev68" ]; then
        printf '%s\n' "$HEXAGON_SDK_ROOT"
        return 0
    fi
    for d in /opt/hexagon-sdk-*; do
        if [ -d "$d/rtos/qurt/computev68" ]; then
            printf '%s\n' "$d"
            return 0
        fi
    done
    return 1
}

find_sim() {
    local sdk=$1 dir
    if [ -n "${HEXAGON_TOOLS_VERSION:-}" ]; then
        dir=$sdk/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
        if [ -x "$dir/bin/hexagon-sim" ]; then
            printf '%s\n' "$dir/bin/hexagon-sim"
            return 0
        fi
    fi
    for dir in "$sdk"/tools/HEXAGON_Tools/*/Tools; do
        if [ -x "$dir/bin/hexagon-sim" ]; then
            printf '%s\n' "$dir/bin/hexagon-sim"
            return 0
        fi
    done
    return 1
}

SDK=$(detect_sdk || true)
QEMU=$(find_qemu || true)
SIM_BIN=""
RUNELF=""
if [ -n "$SDK" ]; then
    SIM_BIN=$(find_sim "$SDK" || true)
    RUNELF=$SDK/rtos/qurt/computev68/sdksim_bin/runelf.pbn
fi

RUN_ROOT=${LFM_SIM_RUN_ROOT:-${TMPDIR:-/tmp}/lfm3-dsp-sim-$$}
mkdir -p "$RUN_ROOT/cfg"

run_host() {
    local slug=$1 bin=$2 log=$3 rc=0

    if command -v timeout >/dev/null 2>&1; then
        if timeout "$TIMEOUT" "$bin" >"$log" 2>&1; then rc=0; else rc=$?; fi
    else
        if "$bin" >"$log" 2>&1; then rc=0; else rc=$?; fi
    fi
    if [ "$rc" = "0" ]; then
        echo "TEST $slug host PASS"
        return 0
    fi
    echo "TEST $slug host FAIL (rc=$rc)" >&2
    tail -20 "$log" >&2 || true
    return 1
}

run_dsp_qemu() {
    local app=$1 log=$2 rc=0

    if command -v timeout >/dev/null 2>&1; then
        if timeout "$TIMEOUT" "$QEMU" -M V68N_1024 -display none \
            -kernel "$RUNELF" -append "$app" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    else
        if "$QEMU" -M V68N_1024 -display none \
            -kernel "$RUNELF" -append "$app" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi
    return $rc
}

run_dsp_sim() {
    local app=$1 log=$2 rc=0 cfg=$RUN_ROOT/cfg
    local ql=$SDK/rtos/qurt/computev68/lib
    local td
    td=$(dirname "$(dirname "$SIM_BIN")")

    mkdir -p "$cfg"
    printf '%s\n' "$ql/../debugger/lnx64/qurt_model.so" >"$cfg/osam.cfg"
    printf '%s\n' "$td/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" >"$cfg/q6ss.cfg"
    printf '%s\n' "$td/lib/iss/l2vic.so 32 0xFC910000" >>"$cfg/q6ss.cfg"

    if command -v timeout >/dev/null 2>&1; then
        if timeout "$TIMEOUT" "$SIM_BIN" -mv68 --mhmx 1 --simulated_returnval --usefs "$RUN_ROOT" \
            --cosim_file "$cfg/q6ss.cfg" --l2tcm_base 0xd800 \
            --rtos "$cfg/osam.cfg" "$RUNELF" -- "$app" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    else
        if "$SIM_BIN" -mv68 --mhmx 1 --simulated_returnval --usefs "$RUN_ROOT" \
            --cosim_file "$cfg/q6ss.cfg" --l2tcm_base 0xd800 \
            --rtos "$cfg/osam.cfg" "$RUNELF" -- "$app" >"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi
    return $rc
}

run_dsp() {
    local slug=$1 app=$2 log=$3 rc=0 stage=$RUN_ROOT/$1

    if [ -z "$SIM_BIN" ] && [ -z "$QEMU" ]; then
        echo "TEST $slug dsp SKIP (no hexagon simulator)"
        return 2
    fi
    if [ -z "$RUNELF" ] || [ ! -f "$RUNELF" ]; then
        echo "TEST $slug dsp SKIP (no runelf.pbn)"
        return 2
    fi

    rm -rf "$stage"
    mkdir -p "$stage"
    if ! cp "$app" "$stage/app"; then
        echo "TEST $slug dsp FAIL (stage copy failed)" >&2
        return 1
    fi
    app=$stage/app

    if [ "${LFM_SIM_USE_QEMU:-0}" = "1" ] && [ -n "$QEMU" ]; then
        if run_dsp_qemu "$app" "$log"; then rc=0; else rc=$?; fi
    elif [ -n "$SIM_BIN" ] && [ -x "$SIM_BIN" ]; then
        if run_dsp_sim "$app" "$log"; then rc=0; else rc=$?; fi
    elif [ -n "$QEMU" ]; then
        if run_dsp_qemu "$app" "$log"; then rc=0; else rc=$?; fi
    fi

    if grep -q '^SIM RESULT: PASS' "$log"; then
        sed -n 's/^SIM /  SIM /p' "$log" || true
        echo "TEST $slug dsp PASS"
        return 0
    fi
    if [ "$rc" = "0" ]; then
        echo "TEST $slug dsp PASS (rc=0, без маркера)"
        return 0
    fi
    echo "TEST $slug dsp FAIL (rc=$rc)" >&2
    tail -20 "$log" >&2 || true
    return 1
}

declare -A CASE_STATE=()

mark_case() {
    local slug=$1 state=$2 cur=${CASE_STATE[$1]:-}

    case "$state" in
        fail) CASE_STATE[$slug]=fail ;;
        pass) [ "$cur" = "fail" ] || CASE_STATE[$slug]=pass ;;
        skip) [ -n "$cur" ] || CASE_STATE[$slug]=skip ;;
    esac
}

LOG_DIR_RUN=$BUILD/run
mkdir -p "$LOG_DIR_RUN"

while IFS='|' read -r slug mode state rel artifact log reason; do
    [ -n "$slug" ] || continue
    case "$state" in
        skip)
            echo "TEST $slug $mode SKIP ($reason)"
            mark_case "$slug" skip
            ;;
        fail)
            echo "TEST $slug $mode FAIL ($reason)" >&2
            if [ -n "$log" ] && [ -f "$log" ]; then
                tail -20 "$log" >&2 || true
            fi
            mark_case "$slug" fail
            ;;
        ready)
            if [ ! -f "$artifact" ]; then
                echo "TEST $slug $mode FAIL (missing artifact)" >&2
                mark_case "$slug" fail
                continue
            fi
            rlog=$LOG_DIR_RUN/$mode-$slug.log
            if [ "$mode" = "host" ]; then
                if run_host "$slug" "$artifact" "$rlog"; then
                    mark_case "$slug" pass
                else
                    mark_case "$slug" fail
                fi
            else
                if run_dsp "$slug" "$artifact" "$rlog"; then
                    rc=0
                else
                    rc=$?
                fi
                case "$rc" in
                    0) mark_case "$slug" pass ;;
                    2) mark_case "$slug" skip ;;
                    *) mark_case "$slug" fail ;;
                esac
            fi
            ;;
        *)
            echo "TEST $slug $mode FAIL (bad state $state)" >&2
            mark_case "$slug" fail
            ;;
    esac
done < "$MANIFEST"

pass=0
skip=0
fail=0
for slug in "${!CASE_STATE[@]}"; do
    case "${CASE_STATE[$slug]}" in
        pass) pass=$((pass + 1)) ;;
        skip) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
done

for state in PASS SKIP FAIL; do
    names=""
    while IFS= read -r slug; do
        [ -n "$slug" ] || continue
        case "$state:${CASE_STATE[$slug]}" in
            PASS:pass|SKIP:skip|FAIL:fail) names="$names $slug" ;;
        esac
    done < <(printf '%s\n' "${!CASE_STATE[@]}" | sort)
    if [ -n "$names" ]; then
        echo "$state:$names"
    fi
done

echo "SUMMARY: pass=$pass skip=$skip fail=$fail"
if [ "$fail" != "0" ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: PASS"
