use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

use lfm_hf::HubClient;

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join(format!("lfm_hf_test_{}", std::process::id()))
}

fn _lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap()
}

#[test]
fn download_and_cache() {
    let _g = _lock();
    let dir = cache_dir();
    let _ = fs::remove_dir_all(&dir);
    let client = HubClient::new(&dir);
    let path = client
        .ensure_file("LiquidAI/LFM3-1.7B", "config.json", "main")
        .unwrap();
    assert!(path.exists());
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert!(value.get("architectures").is_some());

    let before = fs::metadata(&path).unwrap().modified().unwrap();
    let path2 = client
        .ensure_file("LiquidAI/LFM3-1.7B", "config.json", "main")
        .unwrap();
    assert_eq!(path, path2);
    let after = fs::metadata(&path).unwrap().modified().unwrap();
    assert_eq!(before, after);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn file_url_expected() {
    let _g = _lock();
    let dir = cache_dir();
    let client = HubClient::new(&dir);
    assert_eq!(
        client.file_url("repo", "file.txt", "main"),
        "https://huggingface.co/repo/resolve/main/file.txt"
    );
}

#[test]
fn sha256_known() {
    let _g = _lock();
    let dir = cache_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("abc.txt");
    fs::write(&path, b"abc").unwrap();
    assert_eq!(
        HubClient::sha256_file(&path).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let _ = fs::remove_file(&path);
}

#[test]
fn missing_repo_errors() {
    let _g = _lock();
    let dir = cache_dir();
    let _ = fs::remove_dir_all(&dir);
    let client = HubClient::new(&dir);
    let res = client.ensure_file("LiquidAI/definitely-not-a-repo-xyz", "config.json", "main");
    assert!(res.is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn offline_returns_cached() {
    let _g = _lock();
    let dir = cache_dir();
    let _ = fs::remove_dir_all(&dir);
    let client = HubClient::new(&dir);
    let path = client
        .ensure_file("LiquidAI/LFM3-1.7B", "config.json", "main")
        .unwrap();
    assert!(path.exists());

    let offline = HubClient::new(&dir).with_endpoint("http://127.0.0.1:1");
    let path2 = offline
        .ensure_file("LiquidAI/LFM3-1.7B", "config.json", "main")
        .unwrap();
    assert_eq!(path, path2);
    let _ = fs::remove_dir_all(&dir);
}
