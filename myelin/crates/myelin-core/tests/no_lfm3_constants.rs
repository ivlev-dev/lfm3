//! Guards the architecture boundary: no LFM3 numbers outside the adapter.

use std::fs;
use std::path::{Path, PathBuf};

const ALLOWED: &str = "arch/lfm3.rs";
const FORBIDDEN: &[&str] = &["7680", "128_000", "1_000_000", "2048"];

fn rust_sources(dir: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, output);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn lfm3_constants_stay_in_the_adapter() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);

    let mut checked = 0;
    for path in files {
        let relative = path
            .strip_prefix(&src)
            .expect("path under src")
            .to_string_lossy()
            .replace('\\', "/");
        if relative == ALLOWED {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source");
        for needle in FORBIDDEN {
            assert!(
                !text.contains(needle),
                "{relative} contains model-specific constant {needle}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 5, "source scan did not see enough files");

    let adapter = fs::read_to_string(src.join(ALLOWED)).expect("read adapter");
    for needle in ["7680", "128_000", "1_000_000", "shortconv.l_cache"] {
        assert!(
            adapter.contains(needle),
            "adapter lost expected value {needle}"
        );
    }
}
