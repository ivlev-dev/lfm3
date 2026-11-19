use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::net::TcpStream;

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(T[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(T[n as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn accept_key(client_key: &str) -> String {
    base64(&sha1(format!("{client_key}{GUID}").as_bytes()))
}

pub fn handshake(stream: &mut TcpStream, client_key: &str) -> Result<()> {
    let accept = accept_key(client_key);
    let head = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()?;
    Ok(())
}

pub struct WsConn {
    stream: TcpStream,
}

impl WsConn {
    pub fn new(stream: TcpStream) -> Self {
        Self { stream }
    }

    pub fn send_text(&mut self, text: &str) -> Result<()> {
        self.send_frame(0x1, text.as_bytes())
    }

    pub fn send_close(&mut self) -> Result<()> {
        self.send_frame(0x8, &[])
    }

    fn send_frame(&mut self, opcode: u8, payload: &[u8]) -> Result<()> {
        let mut head = Vec::with_capacity(10);
        head.push(0x80 | opcode);
        let len = payload.len();
        if len < 126 {
            head.push(len as u8);
        } else if len <= 65535 {
            head.push(126);
            head.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            head.push(127);
            head.extend_from_slice(&(len as u64).to_be_bytes());
        }
        self.stream.write_all(&head)?;
        self.stream.write_all(payload)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_frame(&mut self) -> Result<(u8, bool, Vec<u8>)> {
        let mut b = [0u8; 2];
        self.stream.read_exact(&mut b)?;
        let fin = b[0] & 0x80 != 0;
        let opcode = b[0] & 0x0f;
        let masked = b[1] & 0x80 != 0;
        let mut len = (b[1] & 0x7f) as usize;
        if len == 126 {
            let mut x = [0u8; 2];
            self.stream.read_exact(&mut x)?;
            len = u16::from_be_bytes(x) as usize;
        } else if len == 127 {
            let mut x = [0u8; 8];
            self.stream.read_exact(&mut x)?;
            len = u64::from_be_bytes(x) as usize;
        }
        if len > 8 * 1024 * 1024 {
            return Err(anyhow!("ws frame too large: {len}"));
        }
        let mut mask = [0u8; 4];
        if masked {
            self.stream.read_exact(&mut mask)?;
        }
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload)?;
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i & 3];
            }
        }
        Ok((opcode, fin, payload))
    }

    pub fn read_message(&mut self) -> Result<Option<String>> {
        let mut acc: Vec<u8> = Vec::new();
        let mut started = false;
        loop {
            let (opcode, fin, payload) = self.read_frame()?;
            match opcode {
                0x1 => {
                    if started {
                        return Err(anyhow!("new ws message before continuation"));
                    }
                    started = true;
                    acc = payload;
                }
                0x2 => return Err(anyhow!("binary ws messages are not supported")),
                0x0 => {
                    if !started {
                        return Err(anyhow!("continuation without message"));
                    }
                    acc.extend_from_slice(&payload);
                }
                0x8 => {
                    let _ = self.send_close();
                    return Ok(None);
                }
                0x9 => {
                    self.send_frame(0xA, &payload)?;
                    continue;
                }
                0xA => continue,
                other => return Err(anyhow!("unsupported ws opcode {other}")),
            }
            if fin {
                break;
            }
        }
        Ok(Some(String::from_utf8(acc)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn sha1_vectors() {
        let hex = |d: [u8; 20]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(hex(sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            hex(sha1(b"The quick brown fox jumps over the lazy dog")),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    #[test]
    fn base64_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn accept_key_matches_rfc6455() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn frames_roundtrip_with_client_mask() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let payload = b"hello";
            let mask = [1u8, 2, 3, 4];
            let mut frame = vec![0x81, 0x80 | payload.len() as u8];
            frame.extend_from_slice(&mask);
            for (i, b) in payload.iter().enumerate() {
                frame.push(b ^ mask[i & 3]);
            }
            stream.write_all(&frame).unwrap();
            let mut buf = [0u8; 6];
            stream.read_exact(&mut buf).unwrap();
            String::from_utf8_lossy(&buf[2..]).to_string()
        });
        let (stream, _) = listener.accept().unwrap();
        let mut conn = WsConn::new(stream);
        let msg = conn.read_message().unwrap();
        assert_eq!(msg.as_deref(), Some("hello"));
        conn.send_text("pong").unwrap();
        assert_eq!(client.join().unwrap(), "pong");
    }

    #[test]
    fn fragments_are_concatenated() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            let mask = [9u8; 4];
            let mut out = Vec::new();
            for (i, part) in [b"he".as_slice(), b"ll", b"o"].iter().enumerate() {
                let mut frame = vec![if i == 2 { 0x80 } else { 0x01 } | 0x00, 0x80 | part.len() as u8];
                if i == 0 {
                    frame[0] = 0x01;
                } else {
                    frame[0] = if i == 2 { 0x80 } else { 0x00 };
                }
                frame.extend_from_slice(&mask);
                for (j, b) in part.iter().enumerate() {
                    frame.push(b ^ mask[j & 3]);
                }
                out.extend_from_slice(&frame);
            }
            stream.write_all(&out).unwrap();
        });
        let (stream, _) = listener.accept().unwrap();
        let mut conn = WsConn::new(stream);
        assert_eq!(conn.read_message().unwrap().as_deref(), Some("hello"));
        client.join().unwrap();
    }
}
