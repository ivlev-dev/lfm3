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
    assert_eq!(value["backend"]["mode"].as_str(), Some("single"));
    assert_eq!(
        value["backend"]["devices"].as_array().map(|a| a.len()),
        Some(1),
        "пример init должен быть безопасен для одного устройства"
    );
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
fn config_init_output_passes_config_print() {
    let dir = std::env::temp_dir().join(format!("lfm-cli-init-print-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lfm.toml");
    let _ = std::fs::remove_file(&path);

    let init = lfm()
        .args(["config", "init", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    // пример из init обязан загружаться и проходить validate()
    let printed = lfm()
        .arg("--config")
        .arg(&path)
        .args(["config", "print"])
        .output()
        .unwrap();
    assert!(
        printed.status.success(),
        "init → print упал: {}",
        String::from_utf8_lossy(&printed.stderr)
    );
    let effective: toml::Value =
        toml::from_str(&String::from_utf8(printed.stdout).unwrap()).unwrap();
    assert_eq!(effective["backend"]["mode"].as_str(), Some("single"));
    assert_eq!(
        effective["backend"]["devices"].as_array().map(|a| a.len()),
        Some(1)
    );

    // подсказки для двух устройств — только комментарии, без чужих железяк
    let example = std::fs::read_to_string(&path).unwrap();
    assert!(example.contains("# devices = [0, 1]"));
    assert!(example.contains("# mode = \"pipeline\""));
    assert!(example.contains("# tiers = [96, 4]"));
    for stale in ["RX 6400", "RUSTICL_ENABLE", "iris"] {
        assert!(!example.contains(stale), "в примере осталось {stale:?}");
    }

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
