fn main() {
    println!("cargo:rerun-if-changed=src/myelin.udl");
    uniffi::generate_scaffolding("src/myelin.udl").expect("generate UniFFI scaffolding");
}
