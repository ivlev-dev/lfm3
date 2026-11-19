# Рецепты для сборки, тестов, сервера и бенчмарков lfm3-engine.
# Модель и файлы HF по умолчанию лежат в models/ (см. README, «Быстрый старт»).

model := "models/LFM3-1.7B-Q4_K_M.gguf"
tokenizer := "models/hf/tokenizer.json"
template := "models/hf/chat_template.jinja"
addr := "127.0.0.1:8000"
ctx := "4096"
power := "/sys/class/drm/card1/device/power_dpm_force_performance_level"

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
    ./target/release/lfm-server --model {{model}} --tokenizer {{tokenizer}} --template {{template}} --addr {{addr}} --ctx {{ctx}}

tui:
    ./target/release/lfm-tui --url http://{{addr}}

bench-prefill:
    ./target/release/lfm-bench gpu-prefill --tokens 512 --decode 128

bench-multi:
    ./target/release/lfm-bench gpu-multi --seqs 8 --prompt 128 --tokens 64

bench-check:
    ./target/release/lfm-bench gpu-check --ids 1,2,3,4,5,6,7,8,9,10

power-high:
    echo high | sudo tee {{power}}

power-auto:
    echo auto | sudo tee {{power}}
