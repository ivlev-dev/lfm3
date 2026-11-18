use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader = PathBuf::from("shaders/gemv_q4.comp");
    println!("cargo:rerun-if-changed={}", shader.display());
    let spv = out.join("gemv_q4.spv");
    let status = Command::new("glslc")
        .args([
            "-fshader-stage=compute",
            "--target-env=vulkan1.2",
            "-O",
        ])
        .arg(&shader)
        .arg("-o")
        .arg(&spv)
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => panic!("glslc failed; install shaderc/glslc"),
    }
}
