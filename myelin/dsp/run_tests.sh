#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MYELIN=$(cd "$HERE/.." && pwd)

if [ ! -f "$HERE/tests/kvarn_case.h" ]; then
    echo "=== генерация kvarn_case.h из Rust-эталона ==="
    (cd "$MYELIN" && cargo run -q -p myelin-pack --example kvarn_dump -- dsp/tests/kvarn_case.h)
fi

echo "=== myelin/dsp: сборка и прогон стендов ==="
"$HERE/sim/runner.sh"
