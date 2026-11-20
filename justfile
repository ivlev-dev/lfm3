# Рецепты для сборки, тестов, сервера и бенчмарков lfm3-engine.
# Модель и файлы HF по умолчанию лежат в models/ (см. README, «Быстрый старт»).

model := "models/LFM3-1.7B-Q4_K_M.gguf"
tokenizer := "models/hf/tokenizer.json"
template := "models/hf/chat_template.jinja"
addr := "127.0.0.1:8000"
ctx := "4096"

default:
    @just --list

build:
    cargo build --release --workspace

test:
    cargo test --workspace

clippy:
    cargo clippy --workspace --all-targets

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

serve:
    ./target/release/lfm serve --model {{model}} --addr {{addr}} --ctx {{ctx}}

chat:
    ./target/release/lfm chat --model {{model}}

config-init:
    ./target/release/lfm config init

config-print:
    ./target/release/lfm config print

bench-prefill:
    ./target/release/lfm-bench gpu-prefill --tokens 512 --decode 128

bench-multi:
    ./target/release/lfm-bench gpu-multi --seqs 8 --prompt 128 --tokens 64

check:
    ./target/release/lfm-bench gpu-check --ids 1,2,3,4,5,6,7,8,9,10

# Регрессия корректности. `amd` — профиль для AMD; на текущей машине это Vega 8.
bench-check:
    tools/bench_check.sh amd

# То же на текущей APU (Vega 8): отдельного профиля в скрипте нет, `amd` уже
# не выставляет RUSTICL_ENABLE. Рецепты `igpu*` относились к Intel UHD 630 —
# устройства в системе больше нет.
bench-check-apu:
    tools/bench_check.sh amd

bench-check-server:
    tools/bench_check.sh amd --server
