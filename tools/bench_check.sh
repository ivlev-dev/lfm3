#!/usr/bin/env bash
# Регрессионный прогон корректности: пороги cos для f16, kvarn и kvarn_k4v4.
#
# Использование:
#   tools/bench_check.sh amd            # AMD (Rusticl без переменных; Vega 8)
#   tools/bench_check.sh apu            # то же, что amd (синоним для Vega 8)
#   tools/bench_check.sh igpu           # Intel UHD 630 (RUSTICL_ENABLE=iris)
#   tools/bench_check.sh amd --server   # плюс смоук `lfm serve`
#   tools/bench_check.sh amd --long     # плюс длинный prefill 4096 (вотчдог amdgpu)
#
# Пороги зафиксированы по замерам сессий 13–15 (RX 6400, см. benchmarks/machine.md)
# и уточнены по замерам Vega 8 (раздел «Машина 2»). Порог kvarn на 2048 зависит
# от устройства: 2-битный V усиливает рассогласование путей, поэтому на Vega 8
# уровень ниже, чем на RX 6400 (0.9889 против 0.9952), но воспроизводится
# бит-в-бит. Скрипт проверяет именно split-пути (len > 512): ошибки нормировки в
# kvarn на коротких контекстах не видны.
set -uo pipefail

BIN=${BIN:-./target/release/lfm-bench}
SRV=${SRV:-./target/release/lfm}
APP=${APP:-./target/release/lfm}
MODE=${1:-amd}
SERVER=0
LONG=0
TP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER=1 ;;
        --long) LONG=1 ;;
        --tp) shift; TP=${1:-} ;;
    esac
    shift
done

case "$MODE" in
    igpu) export RUSTICL_ENABLE=iris ;;
    igpu-full) export RUSTICL_ENABLE=iris ;;
    amd) ;;
    # apu — синоним amd для текущей машины (Vega 8): radeonsi виден без переменных.
    apu) ;;
    *) echo "usage: $0 amd|apu|igpu|igpu-full [--server]" >&2; exit 2 ;;
esac

FAIL=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }

check_min() { # имя значение порог
    local name=$1 got=$2 min=$3
    if [ -z "$got" ]; then bad "$name: нет значения"; return; fi
    local okv
    okv=$(awk -v g="$got" -v m="$min" 'BEGIN { print (g + 0 >= m + 0) ? 1 : 0 }')
    if [ "$okv" = "1" ]; then ok "$name = $got (>= $min)"; else bad "$name = $got (< $min)"; fi
}

cos_cmp() { # len batch kv
    local len=$1 batch=$2 kv=$3
    timeout 2400 "$BIN" gpu-cmp --len "$len" --batch "$batch" --kv-cache "$kv" 2>/dev/null \
        | awk '/^batch cos/ { print $3 }'
}

cos_check() { # ids kv
    local ids=$1 kv=$2
    timeout 900 "$BIN" gpu-check --ids "$ids" --kv-cache "$kv" 2>/dev/null \
        | awk '/^cos\(cpu,gpu\)/ { print $2 }'
}

echo "== $MODE =="

if "$APP" config print >/tmp/opencode/bench_check_cfg.toml 2>/tmp/opencode/bench_check_cfg.err; then
    ok "lfm config print"
else
    bad "lfm config print"
fi

check_min "gpu-check f16"          "$(cos_check 1,2,3 f16)"   0.9975
check_min "gpu-check kvarn"        "$(cos_check 1,2,3 kvarn)" 0.9965

if [ "$MODE" = "amd" ] || [ "$MODE" = "apu" ]; then
    check_min "gpu-cmp f16 1024/64"    "$(cos_cmp 1024 64 f16)"   0.9980
    check_min "gpu-cmp kvarn 1024/64"  "$(cos_cmp 1024 64 kvarn)" 0.9880
    check_min "gpu-cmp kvarn_k4v4 1024/64" "$(cos_cmp 1024 64 kvarn_k4v4)" 0.9960
    check_min "gpu-cmp f16 2048/128"   "$(cos_cmp 2048 128 f16)"  0.9970
    # kvarn (K 4 бита, V 2 бита) на 2048: 0.9952 на RX 6400, 0.9889 на Vega 8.
    # Разница — свойство устройства (2-битный V усиливает рассогласование путей),
    # а не регрессия: на Vega 8 значение воспроизводится бит-в-бит.
    check_min "gpu-cmp kvarn 2048/128" "$(cos_cmp 2048 128 kvarn)" 0.9860
    # kvarn_k4v4 — тот же код (сплиты/staging/commit), но почти без квантового
    # шума (V 4 бита): строгий тест структуры батч/сплит/staging, ловит баги,
    # которые 2-битный kvarn маскирует.
    check_min "gpu-cmp kvarn_k4v4 2048/128" "$(cos_cmp 2048 128 kvarn_k4v4)" 0.9970
elif [ "$MODE" = "igpu-full" ]; then
    # медленно (единичный проход 1024 токенов на iGPU ~3 мин), поэтому по запросу
    check_min "gpu-cmp f16 1024/32"    "$(cos_cmp 1024 32 f16)"   0.9960
    check_min "gpu-cmp kvarn 1024/32"  "$(cos_cmp 1024 32 kvarn)" 0.9850
fi

for kv in f16 kvarn; do
    text=$(timeout 900 "$BIN" gpu-gen --prompt "The capital of France is" --tokens 8 --kv-cache "$kv" 2>/dev/null \
        | awk -F'text: ' '/^text:/ { print $2 }')
    case "$text" in
        *Paris*) ok "gpu-gen $kv: $text" ;;
        *) bad "gpu-gen $kv: '${text}'" ;;
    esac
done

timeout 300 "$BIN" --kv-cache kvarn gpu-chat --ctx 32000 --tui </dev/null >/tmp/opencode/bench_check_ctx.log 2>&1
if [ $? -eq 0 ]; then ok "kvarn ctx 32000 загружается"; else bad "kvarn ctx 32000 (см. /tmp/opencode/bench_check_ctx.log)"; fi

pps=$(timeout 900 "$BIN" gpu-prefill --tokens 512 --decode 16 --kv-cache kvarn 2>/dev/null \
    | awk -F'[()]' '/^decode/ { print $2 }' | awk '{ print $1 }')
check_min "kvarn decode tok/s" "$pps" 1.0

if [ -n "$TP" ]; then
    port=8178
    tp_devices=$(awk -F/ '{ print NF }' <<<"$TP")
    tp_env="LFM_TP=$TP"
    if [ "$tp_devices" -ge 3 ] && [ "$MODE" != "igpu" ] && [ "$MODE" != "igpu-full" ]; then
        tp_env="RUSTICL_ENABLE=iris,radeonsi LFM_TP=$TP"
    fi
    ( env $tp_env "$SRV" serve --mode tp --devices "${TP//\//,}" --addr 127.0.0.1:$port --ctx 1024 >/tmp/opencode/bench_check_tp.log 2>&1 ) &
    srv_pid=$!
    for _ in $(seq 1 60); do
        curl -sf --max-time 1 http://127.0.0.1:$port/health >/dev/null 2>&1 && break
        sleep 2
    done
    if curl -sf --max-time 3 http://127.0.0.1:$port/health >/dev/null 2>&1; then
        ok "tp:$TP /health"
        body='{"model":"lfm3","messages":[{"role":"user","content":"Count from 1 to 8, one number per line."}],"max_tokens":8,"stream":false}'
        content=$(curl -s --max-time 600 http://127.0.0.1:$port/v1/chat/completions \
            -H 'content-type: application/json' -d "$body" \
            | sed -n 's/.*"content":"\([^"]*\)".*/\1/p')
        if [ -n "$content" ]; then ok "tp:$TP chat: $content"; else bad "tp:$TP chat: пустой ответ"; fi
    else
        bad "tp:$TP /health"
    fi
    kill "$srv_pid" 2>/dev/null
    wait "$srv_pid" 2>/dev/null
fi

if [ "$SERVER" = "1" ]; then
    port=8177
    "$SRV" serve --addr 127.0.0.1:$port --ctx 1024 --kv kvarn >/tmp/opencode/bench_check_srv.log 2>&1 &
    srv_pid=$!
    for _ in $(seq 1 40); do
        curl -sf --max-time 1 http://127.0.0.1:$port/health >/dev/null 2>&1 && break
        sleep 1
    done
    if curl -sf --max-time 3 http://127.0.0.1:$port/health >/dev/null 2>&1; then
        ok "server /health"
        body='{"model":"lfm3","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8,"stream":false}'
        content=$(curl -s --max-time 120 http://127.0.0.1:$port/v1/chat/completions \
            -H 'content-type: application/json' -d "$body" \
            | sed -n 's/.*"content":"\([^"]*\)".*/\1/p')
        if [ -n "$content" ]; then ok "server chat: $content"; else bad "server chat: пустой ответ"; fi
    else
        bad "server /health"
    fi
    kill "$srv_pid" 2>/dev/null
    wait "$srv_pid" 2>/dev/null
fi

if [ "$LONG" = "1" ]; then
    # Регресс на вотчдог amdgpu: `lockup_timeout` по умолчанию 2 с, и если
    # prefill-чанк уходит в драйвер одним пакетом дольше, контекст теряется
    # (`ring gfx timeout` / `The CS has cancelled because the context is lost`).
    # CPU-эталон такой баг не ловит: там длинный prefill просто медленный,
    # контекст никуда не теряется. Проверяем именно длину контекста (4096),
    # а не скорость; 600 с с запасом хватает даже на слабой iGPU.
    if timeout 600 "$BIN" gpu-prefill --tokens 4096 --decode 8 \
        >/tmp/opencode/bench_check_long.log 2>&1; then
        ok "длинный prefill 4096 (без ring timeout)"
    else
        bad "длинный prefill 4096 (см. /tmp/opencode/bench_check_long.log)"
    fi
fi

if [ "$FAIL" -eq 0 ]; then
    printf '\n\033[32mвсе проверки пройдены\033[0m\n'
else
    printf '\n\033[31mпровалено проверок: %d\033[0m\n' "$FAIL"
fi
exit "$FAIL"
