//! Verify the HVX method table against the IDL and the generated qaic stub.
//!
//! `src/method.rs` is the single source of truth for the Rust constants. This
//! script re-derives the expected order from `myelin/dsp/hvx/lfmgemv.idl`
//! (tracked) and, when the generated `lfmgemv_stub.c` exists, checks every
//! `_mid` in it. A mismatch fails the build, so the Rust side cannot silently
//! drift from the DSP skeleton.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let method_rs = manifest.join("src/method.rs");
    let idl = manifest.join("../../dsp/hvx/lfmgemv.idl");
    let stub = manifest.join("../../dsp/hvx/lfmgemv_stub.c");

    println!("cargo:rerun-if-changed=src/method.rs");
    println!("cargo:rerun-if-changed={}", idl.display());
    println!("cargo:rerun-if-changed={}", stub.display());

    let constants = parse_constants(&read(&method_rs));
    if !idl.exists() {
        println!(
            "cargo:warning={} is absent; skipping the HVX method-table check",
            idl.display()
        );
        return;
    }
    let names = parse_idl(&read(&idl));

    if names.is_empty() {
        panic!("no methods parsed from {}", idl.display());
    }
    if names.len() != constants.len() {
        panic!(
            "method table has {} constants but {} declares {} methods",
            constants.len(),
            idl.display(),
            names.len()
        );
    }
    for (index, name) in names.iter().enumerate() {
        let constant = name.to_ascii_uppercase();
        match constants.get(&constant) {
            Some(value) if *value == index as u32 => {}
            Some(value) => panic!(
                "{constant} = {value} in src/method.rs, but {name} is method {index} in {}",
                idl.display()
            ),
            None => panic!(
                "src/method.rs has no {constant} for {name} (method {index} in {})",
                idl.display()
            ),
        }
    }
    if stub.exists() {
        let mids = parse_stub(&read(&stub));
        if mids.is_empty() {
            panic!("no method IDs parsed from {}", stub.display());
        }
        for (name, mid) in &mids {
            let constant = name.to_ascii_uppercase();
            match constants.get(&constant) {
                Some(value) if value == mid => {}
                Some(value) => panic!(
                    "{constant} = {value} in src/method.rs, but qaic assigns {mid} to {name}"
                ),
                None => panic!("src/method.rs has no constant for {name}"),
            }
        }
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// `pub const NAME: u32 = N;` inside the `method_id` module.
fn parse_constants(source: &str) -> BTreeMap<String, u32> {
    let mut constants = BTreeMap::new();
    for line in source.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const ") else {
            continue;
        };
        let Some((name, rest)) = rest.split_once(':') else {
            continue;
        };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix("u32") else {
            continue;
        };
        let Some(rest) = rest.trim().strip_prefix('=') else {
            continue;
        };
        let value = rest.trim().trim_end_matches(';').trim();
        let Ok(value) = value.parse::<u32>() else {
            continue;
        };
        constants.insert(name.trim().to_owned(), value);
    }
    constants
}

/// Method names in declaration order. The IDL uses one method per `long
/// name(...)` declaration; parameters are `in`/`rout` and comments start with
/// `/*`.
fn parse_idl(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_interface = false;
    for raw in source.lines() {
        let line = raw.trim();
        if line.starts_with("interface ") {
            in_interface = true;
            continue;
        }
        if !in_interface {
            continue;
        }
        if line == "};" {
            break;
        }
        if let Some(rest) = line.strip_prefix("long ") {
            let name = rest.split('(').next().unwrap_or("").trim();
            if !name.is_empty() {
                names.push(name.to_owned());
            }
        }
    }
    names
}

/// `__QAIC_STUB(lfmgemv_name)` followed by `uint32_t _mid = N;`.
fn parse_stub(source: &str) -> BTreeMap<String, u32> {
    let mut mids = BTreeMap::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if let Some(start) = line.find("__QAIC_STUB(lfmgemv_") {
            let rest = &line[start + "__QAIC_STUB(lfmgemv_".len()..];
            if let Some(name) = rest.split(')').next() {
                let mut cursor = index + 1;
                while cursor < lines.len() {
                    let candidate = lines[cursor].trim();
                    if let Some(value) = candidate.strip_prefix("uint32_t _mid = ") {
                        let value = value.trim_end_matches(';').trim();
                        if let Ok(mid) = value.parse::<u32>() {
                            mids.insert(name.to_owned(), mid);
                        }
                        break;
                    }
                    if candidate.starts_with("__QAIC_STUB_EXPORT") {
                        break;
                    }
                    cursor += 1;
                }
            }
        }
        index += 1;
    }
    mids
}
