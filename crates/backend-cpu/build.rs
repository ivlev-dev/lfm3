fn main() {
    println!("cargo:rerun-if-changed=src/arm_neon.cpp");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "arm" || arch == "aarch64" {
        cc::Build::new()
            .cpp(true)
            .file("src/arm_neon.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-mfpu=neon")
            .flag_if_supported("-mfloat-abi=softfp")
            .opt_level(3)
            .compile("lfmcpu_neon");
        println!("cargo:rustc-link-lib=static=lfmcpu_neon");
        if arch == "arm" {
            println!("cargo:rustc-link-lib=m");
        }
    }
}
