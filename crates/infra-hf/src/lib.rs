use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum HubError {
    #[error("http status {status} for {url}")]
    HttpStatus { status: u16, url: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sha256 mismatch: expected {expected}, got {actual}")]
    ShaMismatch { expected: String, actual: String },
    #[error("reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
}

pub struct HubClient {
    endpoint: String,
    cache_dir: PathBuf,
    token: Option<String>,
    client: Client,
}

impl HubClient {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            endpoint: "https://huggingface.co".to_string(),
            cache_dir: cache_dir.into(),
            token,
            client,
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let mut e = endpoint.into();
        while e.ends_with('/') {
            e.pop();
        }
        self.endpoint = e;
        self
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn file_url(&self, repo: &str, file: &str, revision: &str) -> String {
        format!("{}/{}/resolve/{}/{}", self.endpoint, repo, revision, file)
    }

    fn cache_path(&self, repo: &str, file: &str, revision: &str) -> PathBuf {
        let repo_dir = repo.replace('/', "--");
        let mut path = self.cache_dir.clone();
        path.push("hub");
        path.push(format!("models--{}", repo_dir));
        path.push("snapshots");
        path.push(revision);
        path.push(file);
        path
    }

    pub fn ensure_file(&self, repo: &str, file: &str, revision: &str) -> anyhow::Result<PathBuf> {
        let target = self.cache_path(repo, file, revision);
        if target.exists() {
            let url = self.file_url(repo, file, revision);
            let etag_path = etag_path_for(&target);
            let mut head_req = self.client.head(&url);
            if let Some(t) = &self.token {
                head_req = head_req.header("Authorization", format!("Bearer {}", t));
            }
            match head_req.send() {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let remote_size = resp
                            .headers()
                            .get(reqwest::header::CONTENT_LENGTH)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok());
                        let remote_etag = resp
                            .headers()
                            .get("etag")
                            .or_else(|| resp.headers().get("x-linked-etag"))
                            .and_then(|v| v.to_str().ok())
                            .map(|s| s.trim().to_string());
                        let local_size = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
                        let stored_etag = fs::read_to_string(&etag_path)
                            .ok()
                            .map(|s| s.trim().to_string());

                        let size_ok = remote_size.is_none_or(|s| s == local_size);
                        let mut etag_ok = true;
                        match (&remote_etag, &stored_etag) {
                            (Some(r), Some(s)) => {
                                etag_ok = r == s;
                            }
                            (Some(r), None) => {
                                let _ = fs::write(&etag_path, r);
                            }
                            _ => {}
                        }
                        if size_ok && etag_ok {
                            return Ok(target);
                        }
                    }
                }
                Err(_) => {
                    return Ok(target);
                }
            }
        }
        self.download_with_progress(repo, file, revision, &mut |_, _| {})
    }

    pub fn download_with_progress(
        &self,
        repo: &str,
        file: &str,
        revision: &str,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> anyhow::Result<PathBuf> {
        let url = self.file_url(repo, file, revision);
        let target = self.cache_path(repo, file, revision);
        let parent = target
            .parent()
            .ok_or_else(|| anyhow::Error::from(HubError::Io(std::io::Error::other("no parent"))))?;
        fs::create_dir_all(parent)?;
        let part = part_path_for(&target);
        let etag_path = etag_path_for(&target);

        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 1u64..=3 {
            match self.download_once(&url, &target, &part, &etag_path, progress) {
                Ok(p) => return Ok(p),
                Err(e) => {
                    if attempt < 3 && is_retryable(&e) {
                        last_error = Some(e);
                        thread::sleep(Duration::from_millis(250 * attempt));
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            anyhow::Error::from(HubError::Io(std::io::Error::other("download failed")))
        }))
    }

    fn download_once(
        &self,
        url: &str,
        target: &Path,
        part: &Path,
        etag_path: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> anyhow::Result<PathBuf> {
        let mut downloaded = if part.exists() {
            fs::metadata(part).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let mut req = if downloaded > 0 {
            self.client
                .get(url)
                .header("Range", format!("bytes={}-", downloaded))
        } else {
            self.client.get(url)
        };
        if let Some(t) = &self.token {
            req = req.header("Authorization", format!("Bearer {}", t));
        }

        let mut resp = req.send()?;
        if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let _ = fs::remove_file(part);
            downloaded = 0;
            let mut r2 = self.client.get(url);
            if let Some(t) = &self.token {
                r2 = r2.header("Authorization", format!("Bearer {}", t));
            }
            resp = r2.send()?;
        }

        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(HubError::HttpStatus {
                status: status.as_u16(),
                url: url.to_string(),
            }
            .into());
        }

        let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
        if !resumed && downloaded > 0 {
            let _ = fs::remove_file(part);
            downloaded = 0;
        }

        let total_size = match resp.content_length() {
            Some(cl) if resumed => Some(downloaded + cl),
            Some(cl) => Some(cl),
            None => None,
        };

        let remote_etag = resp
            .headers()
            .get("etag")
            .or_else(|| resp.headers().get("x-linked-etag"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string());
        let x_linked = resp
            .headers()
            .get("x-linked-etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string());

        let mut file = if resumed && downloaded > 0 {
            OpenOptions::new().append(true).open(part)?
        } else {
            File::create(part)?
        };

        let mut buf = [0u8; 65536];
        let mut current = downloaded;
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            current += n as u64;
            progress(current, total_size);
        }
        file.flush()?;
        drop(file);

        if let Some(sha) = x_linked
            && is_hex64(&sha)
        {
            let actual = Self::sha256_file(part)?;
            if actual != sha.to_lowercase() {
                let _ = fs::remove_file(part);
                return Err(HubError::ShaMismatch {
                    expected: sha,
                    actual,
                }
                .into());
            }
        }

        if let Some(e) = remote_etag {
            let _ = fs::write(etag_path, e);
        }

        let _ = fs::remove_file(target);
        fs::rename(part, target)?;
        Ok(target.to_path_buf())
    }

    pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let result = hasher.finalize();
        let mut out = String::with_capacity(64);
        for b in result.iter() {
            out.push_str(&format!("{:02x}", b));
        }
        Ok(out)
    }
}

fn etag_path_for(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    target
        .parent()
        .map(|p| p.join(format!("{}.etag", name)))
        .unwrap_or_else(|| PathBuf::from(format!("{}.etag", name)))
}

fn part_path_for(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    target
        .parent()
        .map(|p| p.join(format!("{}.part", name)))
        .unwrap_or_else(|| PathBuf::from(format!("{}.part", name)))
}

fn is_retryable(err: &anyhow::Error) -> bool {
    if let Some(h) = err.downcast_ref::<HubError>() {
        match h {
            HubError::HttpStatus { status, .. } => *status >= 500,
            HubError::Reqwest(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            HubError::Io(_) => true,
            _ => false,
        }
    } else {
        false
    }
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}
