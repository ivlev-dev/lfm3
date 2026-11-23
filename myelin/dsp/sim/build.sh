#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BUILD=${BUILD_DIR:-$HERE/build}
HOST_CC=${HOST_CC:-cc}
MODES=${LFM_SIM_MODES:-host,dsp}
DSP_ALL=${LFM_SIM_DSP_ALL:-0}

MANIFEST=$BUILD/manifest.txt
HOST_DIR=$BUILD/host
DSP_DIR=$BUILD/dsp
LOG_DIR=$BUILD/log

mkdir -p "$HOST_DIR" "$DSP_DIR" "$LOG_DIR"
: > "$MANIFEST"

record() {
    printf '%s|%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$MANIFEST"
}

slug_for() {
    case "$1" in
        tests/test_common.c) echo common ;;
        hvx/gemv_main.c) echo gemv ;;
        hvx/gemm_main.c) echo gemm ;;
        hvx/ffn_main.c) echo ffn ;;
        hvx/attn_main.c) echo attn ;;
        hvx/kvcache_main.c) echo kvcache ;;
        hvx/exec_main.c) echo exec ;;
        hvx/quant_main.c) echo quant ;;
        hmx/gemm_main.c) echo hmx_gemm ;;
        hmx/gemm_i8_main.c) echo hmx_gemm_i8 ;;
        *) basename "$1" | sed 's/_main\.c$//; s/[^A-Za-z0-9_]/_/g' ;;
    esac
}

list_cases() {
    printf '%s\n' \
        tests/test_common.c \
        hvx/gemv_main.c \
        hvx/gemm_main.c \
        hvx/attn_main.c \
        hvx/exec_main.c \
        hmx/gemm_main.c
    for d in tests hvx hmx; do
        if [ -d "$ROOT/$d" ]; then
            find "$ROOT/$d" -maxdepth 1 -name '*_main.c' -print 2>/dev/null |
                sed "s|^$ROOT/||" || true
        fi
    done
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

detect_tools() {
    local sdk=$1 dir
    if [ -n "${HEXAGON_TOOLS_VERSION:-}" ]; then
        dir=$sdk/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools
        if [ -x "$dir/bin/hexagon-clang" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    fi
    for dir in "$sdk"/tools/HEXAGON_Tools/*/Tools; do
        if [ -x "$dir/bin/hexagon-clang" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done
    return 1
}

mode_enabled() {
    case ",$MODES," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

host_extra() {
    case "$1" in
        gemv) printf '%s\n' hvx/gemv_i4.c ;;
        gemm) printf '%s\n' hvx/gemm_i4.c hvx/gemv_i4.c ;;
        attn) printf '%s\n' hvx/attn_flash.c hvx/kvcache.c ;;
        kvcache) printf '%s\n' hvx/kvcache.c ;;
        ffn) printf '%s\n' hvx/ffn_i4.c hvx/gemv_i4.c ;;
        exec)
            printf '%s\n' hvx/exec.c hvx/ffn_i4.c hvx/gemv_i4.c hvx/gemm_i4.c \
                hvx/attn_flash.c hvx/kvcache.c hvx/exec_stub.c
            ;;
        hmx_gemm|hmx_gemm_i8) printf '%s\n' hmx/gemm_i4.c hmx/gemm_i8.c hmx/prefill.c ;;
        *) ;;
    esac
}

build_host() {
    local rel=$1 slug=$2 src=$ROOT/$1
    local bin=$HOST_DIR/$slug
    local log=$LOG_DIR/host-$slug.log
    local -a extra=()
    local e

    if [ ! -f "$src" ]; then
        record "$slug" host skip "$rel" "" "" "missing-source"
        return 0
    fi
    while IFS= read -r e; do
        [ -n "$e" ] || continue
        [ -f "$ROOT/$e" ] && extra+=("$ROOT/$e")
    done < <(host_extra "$slug")
    if "$HOST_CC" -std=c11 -O2 -Wall -Wextra \
        -I"$ROOT" -I"$ROOT/common" -I"$ROOT/tests" \
        -I"$ROOT/hvx" -I"$ROOT/hmx" -I"$HERE" \
        -o "$bin" "$src" "${extra[@]}" -lm >"$log" 2>&1; then
        record "$slug" host ready "$rel" "$bin" "$log" ""
        return 0
    fi
    record "$slug" host fail "$rel" "" "$log" "host-build-failed"
}

DSP_CC=""
DSP_TD=""
DSP_QL=""
DSP_QU=""
DSP_G0=""

detect_dsp() {
    local sdk
    if ! sdk=$(detect_sdk); then
        return 1
    fi
    if ! DSP_TD=$(detect_tools "$sdk"); then
        return 1
    fi
    DSP_QL=$sdk/rtos/qurt/computev68/lib
    DSP_QU=$sdk/rtos/qurt/computev68/include
    DSP_G0=$DSP_TD/target/hexagon/lib/v68/G0
    DSP_CC=$DSP_TD/bin/hexagon-clang
    if [ ! -x "$DSP_CC" ] || [ ! -f "$DSP_QL/libqurt.a" ] ||
       [ ! -f "$DSP_G0/libc.a" ]; then
        return 1
    fi
    return 0
}

compile_kernels() {
    local out=$1 rel=$2 dir src obj log rc=0

    case "$rel" in
        hvx/*) dir=$ROOT/hvx ;;
        hmx/*) dir=$ROOT/hmx ;;
        *) return 0 ;;
    esac
    [ -d "$dir" ] || return 0
    for src in "$dir"/*.c; do
        [ -f "$src" ] || continue
        case "$(basename "$src")" in
            *_main.c|*_stub.c|*_skel.c|mx4_impl.c) continue ;;
        esac
        obj=$out/$(basename "$src" .c).o
        log=$LOG_DIR/dsp-kernel-$(basename "$src" .c).log
        if ! "$DSP_CC" $DSP_CFLAGS -c "$src" -o "$obj" >"$log" 2>&1; then
            rc=1
        fi
    done
    return $rc
}

DSP_CFLAGS=""

build_dsp() {
    local rel=$1 slug=$2 src=$ROOT/$1
    local out=$DSP_DIR/$slug
    local log=$LOG_DIR/dsp-$slug.log
    local obj heap_flag case_obj sim_obj shim_obj
    local -a objs

    if [ ! -f "$src" ]; then
        record "$slug" dsp skip "$rel" "" "" "missing-source"
        return 0
    fi
    if [ -z "$DSP_CC" ]; then
        record "$slug" dsp skip "$rel" "" "" "no-hexagon-clang"
        return 0
    fi
    case " common gemv gemm ffn attn kvcache exec quant hmx_gemm hmx_gemm_i8 " in
        *" $slug "*) ;;
        *)
            record "$slug" dsp skip "$rel" "" "" "no-slot-in-sim-main"
            return 0
            ;;
    esac

    mkdir -p "$out"
    rm -f "$out"/*.o "$out"/app

    if grep -q 'QURT_DECLARE_STATIC_HEAP' "$src"; then
        heap_flag=-DLFM_SIM_NO_STATIC_HEAP=1
    else
        heap_flag=
    fi

    sim_obj=$out/sim_main.o
    case_obj=$out/case.o
    shim_obj=$out/shim_hap.o
    objs=()

    if ! "$DSP_CC" $DSP_CFLAGS $heap_flag \
        "-DLFM_SIM_ONLY_SLUG=\"$slug\"" \
        -c "$HERE/sim_main.c" -o "$sim_obj" >"$log" 2>&1; then
        record "$slug" dsp fail "$rel" "" "$log" "sim-main-compile-failed"
        return 0
    fi
    objs+=("$sim_obj")

    if ! "$DSP_CC" $DSP_CFLAGS "-Dmain=lfm_case_$slug" \
        -c "$src" -o "$case_obj" >>"$log" 2>&1; then
        record "$slug" dsp fail "$rel" "" "$log" "case-compile-failed"
        return 0
    fi
    objs+=("$case_obj")

    if [ -f "$HERE/shim_hap.c" ]; then
        if "$DSP_CC" $DSP_CFLAGS -c "$HERE/shim_hap.c" -o "$shim_obj" \
            >>"$log" 2>&1; then
            objs+=("$shim_obj")
        fi
    fi

    local -a kern_objs=()
    local archive=""

    compile_kernels "$out" "$rel" || true
    for obj in "$out"/*.o; do
        [ -f "$obj" ] || continue
        case "$obj" in
            "$sim_obj"|"$case_obj"|"$shim_obj") continue ;;
        esac
        kern_objs+=("$obj")
    done
    if [ "${#kern_objs[@]}" -gt 0 ] && [ -x "$DSP_TD/bin/hexagon-ar" ]; then
        archive=$out/libkernels.a
        rm -f "$archive"
        if ! "$DSP_TD/bin/hexagon-ar" rcs "$archive" "${kern_objs[@]}" \
            >>"$log" 2>&1; then
            archive=""
        fi
    fi

    if ! "$DSP_CC" -mv68 -g -nodefaultlibs -nostdlib \
        -Wl,--section-start,.interp=0x23000000 \
        -Wl,--dynamic-linker= -Wl,--force-dynamic -Wl,-E -Wl,-z,muldefs \
        -Wl,--whole-archive \
        "$DSP_G0/init.o" "${objs[@]}" \
        "$DSP_QL/crt1.o" "$DSP_QL/debugmon.o" "$DSP_QL/libqurt.a" \
        "$DSP_G0/libc.a" "$DSP_G0/libqcc.a" "$DSP_G0/libhexagon.a" \
        "$DSP_G0/libm.a" "$DSP_QL/libqurtcfs.a" "$DSP_QL/libtimer_island.a" \
        "$DSP_QL/libtimer_main.a" "$DSP_QL/libposix.a" \
        -Wl,--no-whole-archive ${archive:+"$archive"} "$DSP_G0/fini.o" \
        -o "$out/app" >>"$log" 2>&1; then
        record "$slug" dsp fail "$rel" "" "$log" "link-failed"
        return 0
    fi

    record "$slug" dsp ready "$rel" "$out/app" "$log" ""
}

if detect_dsp; then
    SDK_ROOT=$(detect_sdk)
    DSP_CFLAGS="-mv68 -mhvx -mhmx -mhvx-length=128b -G0 -O2 -Wall \
        -I$ROOT -I$ROOT/common -I$ROOT/hvx -I$ROOT/hmx -I$HERE \
        -I$SDK_ROOT/incs -I$SDK_ROOT/incs/stddef \
        -I$DSP_QU -I$DSP_QU/qurt -I$DSP_QU/posix"
    echo "DSP: $DSP_CC"
else
    echo "DSP: hexagon-clang не найден, DSP-варианты будут SKIP"
fi

mapfile -t CASES < <(list_cases | awk '!seen[$0]++')

for rel in "${CASES[@]}"; do
    slug=$(slug_for "$rel")
    if mode_enabled host; then
        build_host "$rel" "$slug"
    fi
    if mode_enabled dsp; then
        case "$rel" in
            hvx/*|hmx/*) build_dsp "$rel" "$slug" ;;
            *)
                if [ "$DSP_ALL" = "1" ]; then
                    build_dsp "$rel" "$slug"
                fi
                ;;
        esac
    fi
done

awk -F'|' '
    { n[$3]++ }
    END {
        printf "BUILD: ready=%d skip=%d fail=%d\n",
            n["ready"] + 0, n["skip"] + 0, n["fail"] + 0
    }
' "$MANIFEST"

if awk -F'|' '$3 == "fail" { found = 1 } END { exit found ? 0 : 1 }' "$MANIFEST"; then
    exit 1
fi
