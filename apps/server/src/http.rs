use anyhow::{anyhow, Result};
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
    let len = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    if len > max_body {
        return Err(anyhow!("request body {} bytes exceeds limit {}", len, max_body));
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
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

pub fn start_sse(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n",
    )?;
    stream.flush()?;
    Ok(())
}

pub fn write_event(stream: &mut TcpStream, data: &str) -> Result<()> {
    let payload = format!("data: {data}\n\n");
    write!(stream, "{:x}\r\n", payload.len())?;
    stream.write_all(payload.as_bytes())?;
    stream.write_all(b"\r\n")?;
    stream.flush()?;
    Ok(())
}

pub fn end_sse(stream: &mut TcpStream) -> Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_line_and_headers() {
        let (m, p, h) = parse_head("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 12").unwrap();
        assert_eq!(m, "POST");
        assert_eq!(p, "/v1/chat/completions");
        assert_eq!(h.len(), 2);
        assert_eq!(h[1].0, "Content-Length");
        assert_eq!(h[1].1, "12");
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let (_, _, h) = parse_head("GET / HTTP/1.0\r\ncontent-length: 3").unwrap();
        assert!(h.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-length") && v == "3"));
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
}
