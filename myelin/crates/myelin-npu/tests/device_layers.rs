//! Device check: one conv layer and one attention layer through `exec`.
//!
//! Ignored by default: it needs the phone, the HVX skeleton and a fixture made
//! from the real checkpoint by `examples/make_layer_fixture.rs`.
//!
//! ```text
//! cd myelin
//! cargo run -p myelin-npu --example make_layer_fixture -- \
//!     ../models/LFM3-1.7B-Q4_K_M.gguf /tmp/opencode/layer-fixture
//!
//! NDK=/tmp/opencode/ndk/android-ndk-r27c
//! CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang \
//! cargo build --release --target aarch64-linux-android -p myelin-npu --bin dsp-layer-check
//!
//! MYELIN_LAYER_FIXTURE=/tmp/opencode/layer-fixture \
//! cargo test -p myelin-npu --test device_layers -- --ignored --nocapture
//! ```
//!
//! The test pushes the skeleton, the checker and the fixture, then asserts the
//! `VERDICT: PASS` line. The checker compares every position of conv layer 0
//! and attention layer 2 against the CPU oracle that produced the fixture.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn adb_available() -> bool {
    Command::new("adb")
        .arg("devices")
        .output()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .skip(1)
                .any(|line| line.split_whitespace().nth(1) == Some("device"))
        })
        .unwrap_or(false)
}

fn adb(args: &[&str]) -> std::process::Output {
    Command::new("adb").args(args).output().expect("run adb")
}

fn push(local: &Path, remote: &str) {
    let output = adb(&["push", &local.display().to_string(), remote]);
    assert!(
        output.status.success(),
        "adb push {} failed: {}",
        local.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires a phone, liblfmgemv_skel.so and a generated layer fixture"]
fn conv_and_attention_layers_match_cpu() {
    if !adb_available() {
        eprintln!("skip: no adb device");
        return;
    }
    let fixture = std::env::var("MYELIN_LAYER_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/opencode/layer-fixture"));
    if !fixture.join("fixture.json").exists() {
        eprintln!(
            "skip: {} has no fixture.json (run the make_layer_fixture example)",
            fixture.display()
        );
        return;
    }
    let checker = repo_root().join("myelin/target/aarch64-linux-android/release/dsp-layer-check");
    let skeleton = repo_root().join("myelin/dsp/hvx/liblfmgemv_skel.so");
    for path in [&checker, &skeleton] {
        if !path.exists() {
            eprintln!("skip: {} is missing", path.display());
            return;
        }
    }

    let _ = adb(&[
        "shell",
        "mkdir -p /data/local/tmp/lfmgemv /data/local/tmp/layer-fixture",
    ]);
    push(&skeleton, "/data/local/tmp/lfmgemv/");
    push(&checker, "/data/local/tmp/lfmgemv/");
    let fixture_arg = format!("{}/.", fixture.display());
    push(Path::new(&fixture_arg), "/data/local/tmp/layer-fixture/");
    let _ = adb(&["shell", "chmod 755 /data/local/tmp/lfmgemv/dsp-layer-check"]);

    let output = adb(&[
        "shell",
        "cd /data/local/tmp/lfmgemv && ADSP_LIBRARY_PATH=$PWD ./dsp-layer-check /data/local/tmp/layer-fixture",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stdout}");
    eprintln!("{stderr}");
    assert!(
        stdout.contains("VERDICT: PASS"),
        "device layer check did not pass"
    );
}
