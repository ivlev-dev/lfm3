use anyhow::{Result, anyhow};
use std::io::{Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub fn parse_head(head: &str) -> Result<(String, String, Vec<(String, String)>)> {
    let mut lines = head.split("\r\n");
    let first = lines.next().ok_or_else(|| anyhow!("empty request"))?;
    let mut parts = first.split_whitespace();
    let method = parts.next().ok_or_else(|| anyhow!("missing method"))?;
    let path = parts.next().ok_or_else(|| anyhow!("missing path"))?;
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("bad header {line}"))?;
        headers.push((name.trim().to_string(), value.trim().to_string()));
    }
    Ok((method.to_string(), path.to_string(), headers))
}

fn early_error(stream: &mut TcpStream, code: u16, reason: &str, message: &str) -> Result<()> {
    let body = format!(
        "{{\"error\":{{\"message\":\"{}\",\"type\":\"invalid_request_error\"}}}}",
        message.replace('"', "'")
    );
    let mut out = Vec::with_capacity(body.len() + 160);
    write!(
        out,
        "HTTP/1.1 {code} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    out.extend_from_slice(body.as_bytes());
    stream.write_all(&out)?;
    stream.flush()?;
    Ok(())
}

fn head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub fn read_request(stream: &mut TcpStream, max_body: usize) -> Result<Option<Request>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let end = loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = head_end(&buf) {
            break pos;
        }
        if buf.len() > 64 * 1024 {
            return Err(anyhow!("request head too large"));
        }
    };
    let head = std::str::from_utf8(&buf[..end])?;
    let (method, path, headers) = parse_head(head)?;
    if headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("transfer-encoding"))
    {
        early_error(
            stream,
            400,
            "Bad Request",
            "transfer-encoding is not supported",
        )?;
        return Ok(None);
    }
    let mut len: Option<usize> = None;
    for (k, v) in &headers {
        if k.eq_ignore_ascii_case("content-length") {
            let parsed = v
                .parse::<usize>()
                .map_err(|_| anyhow!("bad content-length {v:?}"))?;
            if let Some(prev) = len
                && prev != parsed
            {
                early_error(stream, 400, "Bad Request", "conflicting content-length")?;
                return Ok(None);
            }
            len = Some(parsed);
        }
    }
    let len = len.unwrap_or(0);
    if len > max_body {
        early_error(stream, 413, "Payload Too Large", "request body too large")?;
        return Ok(None);
    }
    let mut body = buf[end + 4..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    if body.len() != len {
        return Err(anyhow!("short body: got {} of {} bytes", body.len(), len));
    }
    Ok(Some(Request {
        method,
        path,
        headers,
        body,
    }))
}

pub fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut out = Vec::with_capacity(body.len() + 160);
    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    out.extend_from_slice(body.as_bytes());
    stream.write_all(&out)?;
    stream.flush()?;
    Ok(())
}

const SSE_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";

pub struct SseWriter<'a> {
    stream: &'a mut TcpStream,
    payload: Vec<u8>,
    batch: Vec<u8>,
}

impl<'a> SseWriter<'a> {
    pub fn start(stream: &'a mut TcpStream) -> Result<Self> {
        stream.write_all(SSE_HEAD)?;
        stream.flush()?;
        Ok(Self {
            stream,
            payload: Vec::with_capacity(256),
            batch: Vec::with_capacity(4096),
        })
    }

    pub fn json<F>(&mut self, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Vec<u8>) -> serde_json::Result<()>,
    {
        let Self { payload, batch, .. } = self;
        payload.clear();
        payload.extend_from_slice(b"data: ");
        encode(payload)?;
        payload.extend_from_slice(b"\n\n");
        push_chunk(batch, payload);
        Ok(())
    }

    pub fn data(&mut self, data: &[u8]) {
        let Self { payload, batch, .. } = self;
        payload.clear();
        payload.extend_from_slice(b"data: ");
        payload.extend_from_slice(data);
        payload.extend_from_slice(b"\n\n");
        push_chunk(batch, payload);
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.batch.is_empty() {
            self.stream.write_all(&self.batch)?;
            self.stream.flush()?;
            self.batch.clear();
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn pending_bytes(&self) -> usize {
        self.batch.len()
    }

    pub fn end(mut self) -> Result<()> {
        self.batch.extend_from_slice(b"0\r\n\r\n");
        self.flush()
    }
}

fn push_chunk(batch: &mut Vec<u8>, payload: &[u8]) {
    push_hex(batch, payload.len());
    batch.extend_from_slice(payload);
    batch.extend_from_slice(b"\r\n");
}

fn push_hex(out: &mut Vec<u8>, mut value: usize) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 16];
    let mut pos = buf.len();
    loop {
        pos -= 1;
        buf[pos] = HEX[value & 0xf];
        value >>= 4;
        if value == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[pos..]);
    out.extend_from_slice(b"\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn parses_request_line_and_headers() {
        let (m, p, h) =
            parse_head("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 12")
                .unwrap();
        assert_eq!(m, "POST");
        assert_eq!(p, "/v1/chat/completions");
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].0, "Content-Length");
        assert_eq!(h[1].1, "12");
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let (_, _, h) = parse_head("GET / HTTP/1.0\r\ncontent-length: 3").unwrap();
        assert!(
            h.iter()
                .any(|(k, v)| k.eq_ignore_ascii_case("content-length") && v == "3")
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_head("GET / HTTP/1.1\r\nbad-header").is_err());
        assert!(parse_head("").is_err());
    }

    #[test]
    fn finds_head_boundary() {
        assert_eq!(head_end(b"GET / HTTP/1.1\r\n\r\nbody"), Some(14));
        assert_eq!(head_end(b"GET / HTTP/1.1\r\n"), None);
    }

    #[test]
    fn chunk_sizes_are_hex_encoded() {
        let mut out = Vec::new();
        push_hex(&mut out, 0);
        assert_eq!(out, b"0\r\n");
        let mut out = Vec::new();
        push_hex(&mut out, 0x1f);
        assert_eq!(out, b"1f\r\n");
        let mut out = Vec::new();
        push_hex(&mut out, 4096);
        assert_eq!(out, b"1000\r\n");
    }

    #[test]
    fn sse_writer_batches_events_into_single_chunks() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let mut raw = Vec::new();
            stream.read_to_end(&mut raw).unwrap();
            raw
        });
        let (mut server, _) = listener.accept().unwrap();
        {
            let mut writer = SseWriter::start(&mut server).unwrap();
            writer
                .json(|b| serde_json::to_writer(b, &serde_json::json!({"content": "a"})))
                .unwrap();
            writer
                .json(|b| serde_json::to_writer(b, &serde_json::json!({"content": "b"})))
                .unwrap();
            assert!(writer.pending_bytes() > 0);
            writer.flush().unwrap();
            assert_eq!(writer.pending_bytes(), 0);
            writer.data(b"[DONE]");
            writer.end().unwrap();
        }
        drop(server);
        let raw = client.join().unwrap();
        let text = String::from_utf8(raw).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(head.contains("Transfer-Encoding: chunked"));
        assert_eq!(
            decode_chunked(body),
            "data: {\"content\":\"a\"}\n\ndata: {\"content\":\"b\"}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn sse_writer_flushes_accumulated_batch_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let mut raw = Vec::new();
            stream.read_to_end(&mut raw).unwrap();
            raw
        });
        let (mut server, _) = listener.accept().unwrap();
        {
            let mut writer = SseWriter::start(&mut server).unwrap();
            for _ in 0..32 {
                writer
                    .json(|b| serde_json::to_writer(b, &serde_json::json!({"content": "x"})))
                    .unwrap();
            }
            assert!(writer.pending_bytes() > 512);
            writer.flush().unwrap();
            assert_eq!(writer.pending_bytes(), 0);
        }
        drop(server);
        let raw = client.join().unwrap();
        let text = String::from_utf8(raw).unwrap();
        let (_, body) = text.split_once("\r\n\r\n").unwrap();
        let expected = "data: {\"content\":\"x\"}\n\n".repeat(32);
        assert_eq!(decode_chunked(body), expected);
        assert_eq!(body.matches("17\r\n").count(), 32);
        assert!(!body.contains("0\r\n\r\n"));
    }

    fn decode_chunked(body: &str) -> String {
        let mut out = String::new();
        let mut rest = body;
        loop {
            let Some((size, tail)) = rest.split_once("\r\n") else {
                break;
            };
            let size = usize::from_str_radix(size, 16).unwrap();
            if size == 0 {
                break;
            }
            out.push_str(&tail[..size]);
            rest = &tail[size + 2..];
        }
        out
    }
}
