//! Generates Kotlin bindings from `src/myelin.udl`.
//!
//! Usage: `cargo run -p myelin-ffi --bin uniffi-bindgen -- generate ...`

fn main() {
    uniffi::uniffi_bindgen_main();
}
