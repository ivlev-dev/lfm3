use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const HEAD_TIMEOUT: Duration = Duration::from_secs(15);
const BODY_TIMEOUT: Duration = Duration::from_millis(120);
const BODY_DEADLINE: Duration = Duration::from_secs(10);
const MAX_HEAD: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    host: String,
    port: u16,
    base: String,
}

impl Url {
    pub fn parse(input: &str) -> Result<Self> {
        let text = input.trim();
        let rest = text
            .strip_prefix("http://")
            .ok_or_else(|| anyhow!("поддерживается только http://, получено {input:?}"))?;
        if rest.is_empty() {
            return Err(anyhow!("в URL не указан хост"));
        }
        let (authority, base) = match rest.find('/') {
            Some(pos) => (&rest[..pos], rest[pos..].trim_end_matches('/')),
            None => (rest, ""),
        };
        if authority.is_empty() {
            return Err(anyhow!("в URL не указан хост"));
        }
        let (host, port) = split_authority(authority)?;
        Ok(Self {
            host,
            port,
            base: base.to_string(),
        })
    }

    pub fn host_header(&self) -> String {
        if self.port == 80 {
            host_for_header(&self.host)
        } else {
            format!("{}:{}", host_for_header(&self.host), self.port)
        }
    }

    pub fn path(&self, suffix: &str) -> String {
        let mut path = self.base.clone();
        if !suffix.starts_with('/') {
            path.push('/');
        }
        path.push_str(suffix);
        if path.is_empty() {
            path.push('/');
        }
        path
    }
}

fn host_for_header(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    }
}

fn split_authority(authority: &str) -> Result<(String, u16)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, tail) = rest
            .split_once(']')
            .ok_or_else(|| anyhow!("некорректный IPv6-адрес в URL"))?;
        let port = match tail.strip_prefix(':') {
            Some(value) => value
                .parse::<u16>()
                .with_context(|| format!("некорректный порт {value:?}"))?,
            None => 80,
        };
        return Ok((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, value)) => {
            if host.is_empty() {
                return Err(anyhow!("в URL не указан хост"));
            }
            let port = value
                .parse::<u16>()
                .with_context(|| format!("некорректный порт {value:?}"))?;
            Ok((host.to_string(), port))
        }
        None => Ok((authority.to_string(), 80)),
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

pub fn decode_line(bytes: &[u8]) -> String {
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    match std::str::from_utf8(&bytes[..end]) {
        Ok(text) => text.to_string(),
        Err(_) => String::from_utf8_lossy(&bytes[..end]).into_owned(),
    }
}

pub fn take_line(pending: &mut Vec<u8>) -> Option<String> {
    let pos = pending.iter().position(|b| *b == b'\n')?;
    let line: Vec<u8> = pending.drain(..=pos).collect();
    Some(decode_line(&line))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    Size,
    Data(usize),
    DataEnd,
    Trailer,
    Done,
}

#[derive(Debug, Clone)]
pub struct ChunkedDecoder {
    state: ChunkState,
    raw: Vec<u8>,
}

impl Default for ChunkedDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedDecoder {
    pub fn new() -> Self {
        Self {
            state: ChunkState::Size,
            raw: Vec::new(),
        }
    }

    pub fn is_done(&self) -> bool {
        self.state == ChunkState::Done
    }

    pub fn feed(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        self.raw.extend_from_slice(data);
        let mut out = Vec::new();
        loop {
            match self.state {
                ChunkState::Size => {
                    let Some(pos) = find_bytes(&self.raw, b"\r\n") else {
                        break;
                    };
                    let line = String::from_utf8_lossy(&self.raw[..pos]).into_owned();
                    self.raw.drain(..pos + 2);
                    let size_text = line.split(';').next().unwrap_or("").trim();
                    if size_text.is_empty() {
                        return Err(anyhow!("chunked: пустой размер чанка"));
                    }
                    let size = usize::from_str_radix(size_text, 16)
                        .with_context(|| format!("chunked: некорректный размер {size_text:?}"))?;
                    self.state = if size == 0 {
                        ChunkState::Trailer
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(remaining) => {
                    if self.raw.is_empty() {
                        break;
                    }
                    let take = remaining.min(self.raw.len());
                    out.extend(self.raw.drain(..take));
                    let left = remaining - take;
                    self.state = if left == 0 {
                        ChunkState::DataEnd
                    } else {
                        ChunkState::Data(left)
                    };
                }
                ChunkState::DataEnd => {
                    if self.raw.len() < 2 {
                        break;
                    }
                    if &self.raw[..2] != b"\r\n" {
                        return Err(anyhow!("chunked: ожидался CRLF после данных чанка"));
                    }
                    self.raw.drain(..2);
                    self.state = ChunkState::Size;
                }
                ChunkState::Trailer => {
                    let Some(pos) = find_bytes(&self.raw, b"\r\n") else {
                        break;
                    };
                    let empty = pos == 0;
                    self.raw.drain(..pos + 2);
                    if empty {
                        self.state = ChunkState::Done;
                    }
                }
                ChunkState::Done => break,
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineRead {
    Line(String),
    Tick,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fill {
    Data,
    Timeout,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Chunked,
    Length(usize),
    UntilClose,
}

pub struct BodyReader {
    status: u16,
    stream: TcpStream,
    mode: Mode,
    decoder: ChunkedDecoder,
    pending: Vec<u8>,
    finished: bool,
}

impl BodyReader {
    fn new(
        stream: TcpStream,
        status: u16,
        headers: Vec<(String, String)>,
        leftover: Vec<u8>,
    ) -> Result<Self> {
        let chunked = header_value(&headers, "transfer-encoding")
            .map(|value| value.to_ascii_lowercase().contains("chunked"))
            .unwrap_or(false);
        let length = header_value(&headers, "content-length")
            .and_then(|value| value.trim().parse::<usize>().ok());
        let mode = if chunked {
            Mode::Chunked
        } else if let Some(len) = length {
            Mode::Length(len)
        } else {
            Mode::UntilClose
        };
        let mut reader = Self {
            status,
            stream,
            mode,
            decoder: ChunkedDecoder::new(),
            pending: Vec::new(),
            finished: matches!(mode, Mode::Length(0)),
        };
        if !leftover.is_empty() {
            reader.feed_raw(&leftover)?;
        }
        Ok(reader)
    }

    fn feed_raw(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        match self.mode {
            Mode::Chunked => {
                let decoded = self.decoder.feed(data)?;
                self.pending.extend_from_slice(&decoded);
                if self.decoder.is_done() {
                    self.finished = true;
                }
            }
            Mode::Length(ref mut remaining) => {
                let take = (*remaining).min(data.len());
                self.pending.extend_from_slice(&data[..take]);
                *remaining -= take;
                if *remaining == 0 {
                    self.finished = true;
                }
            }
            Mode::UntilClose => self.pending.extend_from_slice(data),
        }
        Ok(())
    }

    fn fill(&mut self) -> Result<Fill> {
        if self.finished {
            return Ok(Fill::Eof);
        }
        let mut tmp = [0u8; 8192];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => {
                    self.finished = true;
                    return Ok(Fill::Eof);
                }
                Ok(n) => {
                    self.feed_raw(&tmp[..n])?;
                    return Ok(Fill::Data);
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(Fill::Timeout);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(anyhow!("чтение ответа: {e}")),
            }
        }
    }

    pub fn read_line(&mut self) -> Result<LineRead> {
        loop {
            if let Some(line) = take_line(&mut self.pending) {
                return Ok(LineRead::Line(line));
            }
            if self.finished {
                if self.pending.is_empty() {
                    return Ok(LineRead::Eof);
                }
                let rest = std::mem::take(&mut self.pending);
                return Ok(LineRead::Line(decode_line(&rest)));
            }
            match self.fill()? {
                Fill::Data | Fill::Eof => {}
                Fill::Timeout => return Ok(LineRead::Tick),
            }
        }
    }

    pub fn read_all(&mut self) -> Result<Vec<u8>> {
        let deadline = Instant::now() + BODY_DEADLINE;
        let mut out = Vec::new();
        loop {
            out.append(&mut self.pending);
            if self.finished {
                break;
            }
            match self.fill()? {
                Fill::Data | Fill::Eof => {}
                Fill::Timeout => {
                    if Instant::now() >= deadline {
                        return Err(anyhow!("таймаут чтения тела ответа"));
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn read_all_text(&mut self) -> Result<String> {
        let bytes = self.read_all()?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

type Head = (u16, Vec<(String, String)>, Vec<u8>);

fn read_head(stream: &mut TcpStream) -> Result<Head> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let end = loop {
        match stream.read(&mut tmp) {
            Ok(0) => {
                return Err(anyhow!("соединение закрыто до получения заголовков"));
            }
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Err(anyhow!("сервер не ответил за {} с", HEAD_TIMEOUT.as_secs()));
            }
            Err(e) => return Err(anyhow!("чтение заголовков ответа: {e}")),
        }
        if let Some(pos) = find_bytes(&buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > MAX_HEAD {
            return Err(anyhow!("слишком большой заголовок ответа"));
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(anyhow!("неожиданная строка статуса: {status_line:?}"));
    }
    let code = parts
        .next()
        .ok_or_else(|| anyhow!("в ответе нет кода статуса"))?;
    let status: u16 = code
        .parse()
        .with_context(|| format!("некорректный код статуса {code:?}"))?;
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("некорректный заголовок ответа {line:?}"))?;
        headers.push((name.trim().to_string(), value.trim().to_string()));
    }
    Ok((status, headers, buf[end + 4..].to_vec()))
}

pub struct Client {
    url: Url,
}

impl Client {
    pub fn new(url: Url) -> Self {
        Self { url }
    }

    fn open(
        &self,
        method: &str,
        suffix: &str,
        body: Option<&[u8]>,
        accept: &str,
    ) -> Result<BodyReader> {
        let mut stream = connect(&self.url)?;
        stream
            .set_write_timeout(Some(HEAD_TIMEOUT))
            .context("не удалось задать таймаут записи")?;
        let path = self.url.path(suffix);
        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: lfm-tui\r\nAccept: {accept}\r\nConnection: close\r\n",
            self.url.host_header()
        );
        if let Some(payload) = body {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", payload.len()));
        }
        request.push_str("\r\n");
        stream
            .write_all(request.as_bytes())
            .and_then(|_| match body {
                Some(payload) => stream.write_all(payload),
                None => Ok(()),
            })
            .and_then(|_| stream.flush())
            .context("отправка запроса на сервер")?;
        stream
            .set_read_timeout(Some(HEAD_TIMEOUT))
            .context("не удалось задать таймаут чтения")?;
        let (status, headers, leftover) = read_head(&mut stream)?;
        stream
            .set_read_timeout(Some(BODY_TIMEOUT))
            .context("не удалось задать таймаут чтения тела")?;
        BodyReader::new(stream, status, headers, leftover)
    }

    pub fn chat_stream(&self, body: &[u8]) -> Result<ChatStream> {
        let reader = self.open(
            "POST",
            "/v1/chat/completions",
            Some(body),
            "text/event-stream",
        )?;
        if !(200..300).contains(&reader.status) {
            let mut reader = reader;
            let text = reader.read_all_text().unwrap_or_default();
            return Err(anyhow!(
                "сервер ответил HTTP {}: {}",
                reader.status,
                snippet(&text)
            ));
        }
        Ok(ChatStream::new(reader))
    }

    pub fn metrics(&self) -> Result<Metrics> {
        let mut reader = self.open("GET", "/metrics", None, "application/json")?;
        let status = reader.status;
        let text = reader.read_all_text()?;
        if !(200..300).contains(&status) {
            return Err(anyhow!(
                "GET /metrics вернул HTTP {status}: {}",
                snippet(&text)
            ));
        }
        serde_json::from_str(&text)
            .with_context(|| format!("не удалось разобрать /metrics: {}", snippet(&text)))
    }
}

fn connect(url: &Url) -> Result<TcpStream> {
    let addrs = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .with_context(|| format!("не удалось разрешить адрес {}:{}", url.host, url.port))?;
    let mut last: Option<anyhow::Error> = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                return Ok(stream);
            }
            Err(e) => last = Some(anyhow!("{addr}: {e}")),
        }
    }
    Err(last
        .map(|e| anyhow!("не удалось подключиться к {}:{} — {e}", url.host, url.port))
        .unwrap_or_else(|| anyhow!("не удалось подключиться к {}:{}", url.host, url.port)))
}

fn snippet(text: &str) -> String {
    let text = text.trim().replace('\n', " ");
    let taken: String = text.chars().take(200).collect();
    taken
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolDelta {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SseChunk {
    pub content: Option<String>,
    pub thinking: Option<String>,
    pub tools: Vec<ToolDelta>,
    pub finish: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SseLine {
    Chunk(SseChunk),
    Done,
    Skip,
}

pub fn parse_sse_line(line: &str) -> SseLine {
    let line = line.trim_end();
    let Some(rest) = line.strip_prefix("data:") else {
        return SseLine::Skip;
    };
    let payload = rest.trim();
    if payload.is_empty() {
        return SseLine::Skip;
    }
    if payload == "[DONE]" {
        return SseLine::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return SseLine::Skip;
    };
    let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
        return SseLine::Skip;
    };
    let mut chunk = SseChunk::default();
    if let Some(delta) = choice.get("delta") {
        if let Some(text) = delta
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|text| !text.is_empty())
        {
            chunk.content = Some(text.to_string());
        }
        if let Some(text) = delta
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .filter(|text| !text.is_empty())
        {
            chunk.thinking = Some(text.to_string());
        }
        if let Some(tools) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tool in tools {
                let index = tool.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let function = tool.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
                let arguments = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
                if name.is_some() || arguments.is_some() {
                    chunk.tools.push(ToolDelta {
                        index,
                        name,
                        arguments,
                    });
                }
            }
        }
    }
    if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
        chunk.finish = Some(reason.to_string());
    }
    if chunk.content.is_none()
        && chunk.thinking.is_none()
        && chunk.tools.is_empty()
        && chunk.finish.is_none()
    {
        SseLine::Skip
    } else {
        SseLine::Chunk(chunk)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamStep {
    Chunk(SseChunk),
    Done,
    Tick,
    Eof,
}

pub struct ChatStream {
    reader: BodyReader,
    done: bool,
}

impl ChatStream {
    fn new(reader: BodyReader) -> Self {
        Self {
            reader,
            done: false,
        }
    }

    pub fn next_step(&mut self) -> Result<StreamStep> {
        if self.done {
            return Ok(StreamStep::Eof);
        }
        match self.reader.read_line()? {
            LineRead::Tick => Ok(StreamStep::Tick),
            LineRead::Eof => {
                self.done = true;
                Ok(StreamStep::Eof)
            }
            LineRead::Line(line) => match parse_sse_line(&line) {
                SseLine::Skip => Ok(StreamStep::Tick),
                SseLine::Done => {
                    self.done = true;
                    Ok(StreamStep::Done)
                }
                SseLine::Chunk(chunk) => Ok(StreamStep::Chunk(chunk)),
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Metrics {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub ctx: usize,
    #[serde(default)]
    pub active: usize,
    #[serde(default)]
    pub waiting: usize,
    #[serde(default)]
    pub steps: usize,
    #[serde(default)]
    pub avg_step_ms: f64,
    #[serde(default)]
    pub kv_used_pages: usize,
    #[serde(default)]
    pub kv_free_pages: usize,
    #[serde(default)]
    pub kv_cached_pages: usize,
    #[serde(default)]
    pub cache_hits: usize,
    #[serde(default)]
    pub reused_tokens: usize,
    #[serde(default)]
    pub uptime_s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    fn chunk(data: &[u8]) -> Vec<u8> {
        let mut out = format!("{:x}\r\n", data.len()).into_bytes();
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
        out
    }

    struct FakeServer {
        url: Url,
        captured: Arc<Mutex<Option<String>>>,
    }

    impl FakeServer {
        fn start(response: Vec<u8>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let port = listener.local_addr().expect("addr").port();
            let captured = Arc::new(Mutex::new(None));
            let slot = captured.clone();
            thread::spawn(move || {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let head_end = loop {
                    match stream.read(&mut tmp) {
                        Ok(0) => return,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        Err(_) => return,
                    }
                    if let Some(pos) = find_bytes(&buf, b"\r\n\r\n") {
                        break pos;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
                let length = head
                    .split("\r\n")
                    .filter_map(|line| line.split_once(':'))
                    .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while buf.len() < head_end + 4 + length {
                    match stream.read(&mut tmp) {
                        Ok(0) => break,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        Err(_) => break,
                    }
                }
                *slot.lock().expect("lock") = Some(String::from_utf8_lossy(&buf).into_owned());
                let _ = stream.write_all(&response);
                let _ = stream.flush();
            });
            Self {
                url: Url {
                    host: "127.0.0.1".to_string(),
                    port,
                    base: String::new(),
                },
                captured,
            }
        }

        fn request(&self) -> String {
            self.captured
                .lock()
                .expect("lock")
                .clone()
                .unwrap_or_default()
        }
    }

    #[test]
    fn parses_url_with_port_and_path() {
        let url = Url::parse("http://127.0.0.1:8000").expect("url");
        assert_eq!(url.host, "127.0.0.1");
        assert_eq!(url.port, 8000);
        assert_eq!(url.path("/v1/chat/completions"), "/v1/chat/completions");
        assert_eq!(url.host_header(), "127.0.0.1:8000");

        let url = Url::parse("http://localhost").expect("url");
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 80);
        assert_eq!(url.host_header(), "localhost");

        let url = Url::parse("http://example.org:9000/api/").expect("url");
        assert_eq!(url.path("/metrics"), "/api/metrics");
    }

    #[test]
    fn rejects_bad_urls() {
        assert!(Url::parse("https://127.0.0.1:8000").is_err());
        assert!(Url::parse("127.0.0.1:8000").is_err());
        assert!(Url::parse("http://").is_err());
        assert!(Url::parse("http://host:notaport").is_err());
        assert!(Url::parse("http://:8000").is_err());
    }

    #[test]
    fn chunked_decodes_hex_sizes_and_terminator() {
        let mut decoder = ChunkedDecoder::new();
        let mut raw = Vec::new();
        raw.extend(chunk(b"hello "));
        raw.extend_from_slice(b"5\r\nWorld\r\n");
        raw.extend_from_slice(b"0\r\n\r\n");
        raw.extend_from_slice(b"ignored");
        let out = decoder.feed(&raw).expect("decode");
        assert_eq!(out, b"hello World");
        assert!(decoder.is_done());
        let again = decoder.feed(b"more").expect("decode");
        assert!(again.is_empty());
    }

    #[test]
    fn chunked_handles_split_feeds_and_utf8() {
        let text = "data: {\"x\":\"Привет\"}\n\n";
        let mut raw = chunk(text.as_bytes());
        raw.extend_from_slice(b"0\r\n\r\n");
        let mut decoder = ChunkedDecoder::new();
        let mut out = Vec::new();
        for byte in &raw {
            let piece = decoder.feed(&[*byte]).expect("decode");
            out.extend_from_slice(&piece);
        }
        assert!(decoder.is_done());
        assert_eq!(String::from_utf8(out).expect("utf8"), text);
    }

    #[test]
    fn chunked_accepts_extensions_and_reports_bad_size() {
        let mut decoder = ChunkedDecoder::new();
        let out = decoder
            .feed(b"2;name=value\r\nhi\r\n0\r\n\r\n")
            .expect("decode");
        assert_eq!(out, b"hi");
        assert!(decoder.is_done());

        let mut decoder = ChunkedDecoder::new();
        assert!(decoder.feed(b"zz\r\n").is_err());
        let mut decoder = ChunkedDecoder::new();
        assert!(decoder.feed(b"3\r\nabcX\r\n").is_err());
    }

    #[test]
    fn chunked_split_at_every_boundary() {
        let mut src = Vec::new();
        src.extend(chunk(b"alpha"));
        src.extend(chunk("бета".as_bytes()));
        src.extend_from_slice(b"0\r\n\r\n");
        for cut in 1..src.len() {
            let mut decoder = ChunkedDecoder::new();
            let mut out = Vec::new();
            out.extend(decoder.feed(&src[..cut]).expect("first"));
            out.extend(decoder.feed(&src[cut..]).expect("second"));
            let expected = decoder.is_done();
            let text = String::from_utf8(out).expect("utf8");
            if expected {
                assert_eq!(text, "alphaбета", "cut {cut}");
            }
        }
    }

    #[test]
    fn sse_parses_content_and_thinking() {
        let line = r#"data: {"choices":[{"delta":{"content":"При","reasoning_content":"думаю"},"finish_reason":null}]}"#;
        match parse_sse_line(line) {
            SseLine::Chunk(chunk) => {
                assert_eq!(chunk.content.as_deref(), Some("При"));
                assert_eq!(chunk.thinking.as_deref(), Some("думаю"));
                assert!(chunk.finish.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn sse_ignores_done_comments_and_bad_json() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseLine::Done);
        assert_eq!(parse_sse_line("data: {oops"), SseLine::Skip);
        assert_eq!(parse_sse_line(": keep-alive"), SseLine::Skip);
        assert_eq!(parse_sse_line(""), SseLine::Skip);
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{},"finish_reason":null}]}"#),
            SseLine::Skip
        );
    }

    #[test]
    fn sse_parses_finish_and_tool_calls() {
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":"tool_calls"}]}"#;
        match parse_sse_line(line) {
            SseLine::Chunk(chunk) => {
                assert_eq!(chunk.finish.as_deref(), Some("tool_calls"));
                assert_eq!(chunk.tools.len(), 1);
                assert_eq!(chunk.tools[0].index, 0);
                assert_eq!(chunk.tools[0].name.as_deref(), Some("lookup"));
                assert_eq!(chunk.tools[0].arguments.as_deref(), Some("{\"q\":"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn chat_stream_joins_deltas_across_chunks() {
        let first = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
        let second = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"думаю\"},\"finish_reason\":null}]}\n\n";
        let third =
            "data: {\"choices\":[{\"delta\":{\"content\":\"При\"},\"finish_reason\":null}]}\n\n";
        let fourth =
            "data: {\"choices\":[{\"delta\":{\"content\":\"вет\"},\"finish_reason\":null}]}\n\n";
        let broken = "data: {not json}\n\n";
        let finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        let done = "data: [DONE]\n\n";
        let body = format!("{first}{second}{third}{fourth}{broken}{finish}{done}");

        let bytes = body.as_bytes();
        let cut = body.find("При").expect("marker") + 1;
        let mut response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        response.extend(chunk(&bytes[..cut]));
        response.extend(chunk(&bytes[cut..]));
        response.extend_from_slice(b"0\r\n\r\n");

        let server = FakeServer::start(response);
        let client = Client::new(server.url.clone());
        let mut stream = client.chat_stream(b"{}").expect("stream");
        assert_eq!(stream.reader.status, 200);
        let mut content = String::new();
        let mut thinking = String::new();
        let mut finish_reason = None;
        let mut saw_done = false;
        loop {
            match stream.next_step().expect("step") {
                StreamStep::Chunk(chunk) => {
                    if let Some(text) = chunk.content {
                        content.push_str(&text);
                    }
                    if let Some(text) = chunk.thinking {
                        thinking.push_str(&text);
                    }
                    if let Some(reason) = chunk.finish {
                        finish_reason = Some(reason);
                    }
                }
                StreamStep::Done => {
                    saw_done = true;
                    break;
                }
                StreamStep::Eof => break,
                StreamStep::Tick => {}
            }
        }
        assert!(saw_done);
        assert_eq!(content, "Привет");
        assert_eq!(thinking, "думаю");
        assert_eq!(finish_reason.as_deref(), Some("stop"));

        let request = server.request();
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
        assert!(request.contains("Connection: close"));
        assert!(request.contains("Content-Length: 2"));
        assert!(request.ends_with("{}"));
    }

    #[test]
    fn metrics_get_parses_schema() {
        let body = r#"{"model":"lfm3","ctx":4096,"active":1,"waiting":2,"steps":10,"avg_step_ms":12.5,"kv_used_pages":3,"kv_free_pages":50,"kv_cached_pages":4,"cache_hits":7,"reused_tokens":128,"uptime_s":9.5}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        let server = FakeServer::start(response);
        let client = Client::new(server.url.clone());
        let metrics = client.metrics().expect("metrics");
        assert_eq!(
            metrics,
            Metrics {
                model: "lfm3".to_string(),
                ctx: 4096,
                active: 1,
                waiting: 2,
                steps: 10,
                avg_step_ms: 12.5,
                kv_used_pages: 3,
                kv_free_pages: 50,
                kv_cached_pages: 4,
                cache_hits: 7,
                reused_tokens: 128,
                uptime_s: 9.5,
            }
        );
        let request = server.request();
        assert!(request.starts_with("GET /metrics HTTP/1.1\r\n"));
        assert!(!request.contains("Content-Length"));
    }

    #[test]
    fn metrics_accepts_partial_json() {
        let metrics: Metrics =
            serde_json::from_str(r#"{"model":"lfm3","ctx":4096}"#).expect("parse");
        assert_eq!(metrics.model, "lfm3");
        assert_eq!(metrics.ctx, 4096);
        assert_eq!(metrics.active, 0);
        assert_eq!(metrics.avg_step_ms, 0.0);
    }

    #[test]
    fn metrics_reports_error_status() {
        let body = r#"{"error":{"message":"nope"}}"#;
        let response = format!(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        let server = FakeServer::start(response);
        let client = Client::new(server.url.clone());
        let error = client.metrics().expect_err("must fail");
        assert!(error.to_string().contains("503"));
    }

    #[test]
    fn chat_stream_reports_error_body() {
        let body = r#"{"error":{"message":"bad json"}}"#;
        let response = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        let server = FakeServer::start(response);
        let client = Client::new(server.url.clone());
        let error = match client.chat_stream(b"{}") {
            Ok(_) => panic!("must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("400"));
        assert!(error.to_string().contains("bad json"));
    }

    #[test]
    fn content_length_body_without_chunking() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_vec();
        let server = FakeServer::start(response);
        let client = Client::new(server.url.clone());
        let mut reader = client
            .open("GET", "/health", None, "application/json")
            .expect("open");
        assert_eq!(reader.status, 200);
        assert_eq!(reader.read_all_text().expect("body"), "hello");
    }

    #[test]
    fn read_line_keeps_partial_utf8_until_newline() {
        let mut pending = Vec::new();
        pending.extend_from_slice("При".as_bytes());
        assert!(take_line(&mut pending).is_none());
        pending.extend_from_slice(b"vet\n");
        let line = take_line(&mut pending).expect("line");
        assert_eq!(line, "Приvet");
        assert!(pending.is_empty());
    }

    #[test]
    fn read_line_strips_crlf() {
        let mut pending = b"data: x\r\nrest".to_vec();
        assert_eq!(take_line(&mut pending).as_deref(), Some("data: x"));
        assert_eq!(take_line(&mut pending), None);
    }
}
