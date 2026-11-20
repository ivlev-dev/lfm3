use std::process::Command;

fn lfm() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lfm"))
}

#[test]
fn config_print_is_valid_toml() {
    let out = lfm().args(["config", "print"]).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert_eq!(value["backend"]["ctx"].as_integer(), Some(4096));
    assert_eq!(value["backend"]["mode"].as_str(), Some("single"));
}

#[test]
fn config_init_writes_loadable_file() {
    let dir = std::env::temp_dir().join(format!("lfm-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lfm.toml");
    let _ = std::fs::remove_file(&path);
    let out = lfm()
        .args(["config", "init", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert_eq!(value["backend"]["mode"].as_str(), Some("pipeline"));
    let again = lfm()
        .args(["config", "init", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        !again.status.success(),
        "без --force файл перезаписываться не должен"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn config_file_overrides_defaults() {
    let dir = std::env::temp_dir().join(format!("lfm-cli-override-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("prod.toml");
    std::fs::write(
        &path,
        "[backend]\nctx = 8192\nkv = \"kvarn\"\nmax_seqs = 2\n[server]\nprefill_budget = 16\n",
    )
    .unwrap();
    let out = lfm()
        .arg("--config")
        .arg(&path)
        .args(["config", "print"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert_eq!(value["backend"]["ctx"].as_integer(), Some(8192));
    assert_eq!(value["backend"]["kv"].as_str(), Some("kvarn"));
    assert_eq!(value["backend"]["max_seqs"].as_integer(), Some(2));
    assert_eq!(value["server"]["prefill_budget"].as_integer(), Some(16));
    // не переопределённое остаётся дефолтом
    assert_eq!(value["backend"]["prefill_ntok"].as_integer(), Some(128));
    assert_eq!(value["server"]["prefill_in_decode"].as_integer(), Some(8));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn unknown_config_key_is_rejected() {
    let dir = std::env::temp_dir().join(format!("lfm-cli-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[backend]\nnope = 1\n").unwrap();
    let out = lfm()
        .arg("--config")
        .arg(&path)
        .args(["config", "print"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
