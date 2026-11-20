use crate::serve::hpack_tables::{HUFFMAN, STATIC_TABLE};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::OnceLock;

pub type Header = (String, String);

pub struct HpackCtx {
    dynamic: Vec<Header>,
    size: usize,
    max: usize,
}

impl Default for HpackCtx {
    fn default() -> Self {
        Self::new()
    }
}

pub const HPACK_MAX_TABLE: usize = 4096;
pub const HPACK_MAX_HEADERS: usize = 64;
pub const HPACK_MAX_BLOCK: usize = 32 * 1024;

impl HpackCtx {
    pub fn new() -> Self {
        Self {
            dynamic: Vec::new(),
            size: 0,
            max: 4096,
        }
    }

    #[cfg(test)]
    pub fn dynamic_len(&self) -> usize {
        self.dynamic.len()
    }

    pub fn set_max_size(&mut self, size: usize) -> Result<()> {
        if size > HPACK_MAX_TABLE {
            return Err(anyhow!(
                "hpack: table size {size} exceeds advertised maximum {HPACK_MAX_TABLE}"
            ));
        }
        self.max = size;
        self.evict();
        Ok(())
    }

    pub fn get(&self, idx: u32) -> Option<Header> {
        if idx == 0 {
            return None;
        }
        let i = idx as usize;
        if i <= STATIC_TABLE.len() {
            let (name, value) = STATIC_TABLE[i - 1];
            return Some((name.to_string(), value.to_string()));
        }
        self.dynamic.get(i - STATIC_TABLE.len() - 1).cloned()
    }

    fn insert(&mut self, name: String, value: String) {
        self.size += name.len() + value.len() + 32;
        self.dynamic.insert(0, (name, value));
        self.evict();
    }

    fn evict(&mut self) {
        while self.size > self.max {
            let Some((name, value)) = self.dynamic.pop() else {
                break;
            };
            self.size = self.size.saturating_sub(name.len() + value.len() + 32);
        }
    }
}

pub fn decode(block: &[u8], ctx: &mut HpackCtx) -> Result<Vec<Header>> {
    let mut out: Vec<Header> = Vec::new();
    let mut pos = 0usize;
    let mut total: usize;
    if block.len() > HPACK_MAX_BLOCK {
        return Err(anyhow!("hpack: header block too large"));
    }
    while pos < block.len() {
        if out.len() >= HPACK_MAX_HEADERS {
            return Err(anyhow!("hpack: too many headers"));
        }
        total = out.iter().map(|(n, v)| n.len() + v.len()).sum();
        if total > HPACK_MAX_BLOCK {
            return Err(anyhow!("hpack: headers too large"));
        }
        let byte = block[pos];
        if byte & 0x80 != 0 {
            let (idx, next) = read_int(block, pos, 7)?;
            pos = next;
            out.push(
                ctx.get(idx)
                    .ok_or_else(|| anyhow!("hpack: bad index {idx}"))?,
            );
        } else if byte & 0x40 != 0 {
            let (idx, next) = read_int(block, pos, 6)?;
            let (name, next) = read_name(block, next, idx, ctx)?;
            let (value, next) = read_string(block, next)?;
            pos = next;
            ctx.insert(name.clone(), value.clone());
            out.push((name, value));
        } else if byte & 0x20 != 0 {
            let (max, next) = read_int(block, pos, 5)?;
            pos = next;
            ctx.set_max_size(max as usize)?;
        } else {
            let (idx, next) = read_int(block, pos, 4)?;
            let (name, next) = read_name(block, next, idx, ctx)?;
            let (value, next) = read_string(block, next)?;
            pos = next;
            out.push((name, value));
        }
    }
    Ok(out)
}

fn read_name(block: &[u8], pos: usize, idx: u32, ctx: &HpackCtx) -> Result<(String, usize)> {
    if idx == 0 {
        read_string(block, pos)
    } else {
        let (name, _) = ctx
            .get(idx)
            .ok_or_else(|| anyhow!("hpack: bad name index {idx}"))?;
        Ok((name, pos))
    }
}

fn read_int(buf: &[u8], pos: usize, prefix: u8) -> Result<(u32, usize)> {
    let mask = (1u32 << prefix) - 1;
    let first = *buf
        .get(pos)
        .ok_or_else(|| anyhow!("hpack: truncated int"))? as u32;
    let mut value = first & mask;
    let mut pos = pos + 1;
    if value < mask {
        return Ok((value, pos));
    }
    let mut shift = 0u32;
    loop {
        let byte = *buf
            .get(pos)
            .ok_or_else(|| anyhow!("hpack: truncated int"))?;
        pos += 1;
        let part = ((byte & 0x7f) as u32)
            .checked_shl(shift)
            .ok_or_else(|| anyhow!("hpack: int overflow"))?;
        value = value
            .checked_add(part)
            .ok_or_else(|| anyhow!("hpack: int overflow"))?;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 28 {
            return Err(anyhow!("hpack: int too long"));
        }
    }
    Ok((value, pos))
}

fn read_string(buf: &[u8], pos: usize) -> Result<(String, usize)> {
    let huffman = buf.get(pos).map(|b| b & 0x80 != 0).unwrap_or(false);
    let (len, start) = read_int(buf, pos, 7)?;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| anyhow!("hpack: string length overflow"))?;
    let data = buf
        .get(start..end)
        .ok_or_else(|| anyhow!("hpack: truncated string"))?;
    let text = if huffman {
        huffman_decode(data)?
    } else {
        String::from_utf8(data.to_vec())?
    };
    Ok((text, end))
}

fn huffman_map() -> &'static HashMap<(u8, u32), u16> {
    static MAP: OnceLock<HashMap<(u8, u32), u16>> = OnceLock::new();
    MAP.get_or_init(|| {
        HUFFMAN
            .iter()
            .enumerate()
            .map(|(sym, (code, bits))| ((*bits, *code), sym as u16))
            .collect()
    })
}

pub fn huffman_decode(data: &[u8]) -> Result<String> {
    let map = huffman_map();
    let mut out = Vec::with_capacity(data.len());
    let mut code = 0u32;
    let mut bits = 0u8;
    for byte in data {
        for i in (0..8).rev() {
            code = (code << 1) | ((byte >> i) & 1) as u32;
            bits += 1;
            if let Some(&sym) = map.get(&(bits, code)) {
                if sym == 256 {
                    return Err(anyhow!("hpack: EOS in huffman string"));
                }
                out.push(sym as u8);
                code = 0;
                bits = 0;
            }
            if bits > 30 {
                return Err(anyhow!("hpack: invalid huffman code"));
            }
        }
    }
    if bits > 7 {
        return Err(anyhow!("hpack: huffman padding too long"));
    }
    if bits > 0 && code != (1u32 << bits) - 1 {
        return Err(anyhow!("hpack: bad huffman padding"));
    }
    Ok(String::from_utf8(out)?)
}

pub fn encode(headers: &[(&str, String)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in headers {
        if let Some(idx) = static_index(name, value) {
            write_int(&mut out, 0x80, 7, idx);
        } else {
            out.push(0x00);
            write_string(&mut out, name);
            write_string(&mut out, value);
        }
    }
    out
}

fn static_index(name: &str, value: &str) -> Option<u32> {
    STATIC_TABLE
        .iter()
        .position(|(n, v)| *n == name && *v == value)
        .map(|pos| pos as u32 + 1)
}

fn write_string(out: &mut Vec<u8>, text: &str) {
    write_int(out, 0x00, 7, text.len() as u32);
    out.extend_from_slice(text.as_bytes());
}

fn write_int(out: &mut Vec<u8>, prefix_byte: u8, prefix: u8, value: u32) {
    let mask = (1u32 << prefix) - 1;
    if value < mask {
        out.push(prefix_byte | value as u8);
        return;
    }
    out.push(prefix_byte | mask as u8);
    let mut rest = value - mask;
    while rest >= 128 {
        out.push((rest % 128) as u8 | 0x80);
        rest /= 128;
    }
    out.push(rest as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        text.split_whitespace()
            .map(|b| u8::from_str_radix(b, 16).expect("hex"))
            .collect()
    }

    #[test]
    fn decodes_rfc_c41_request() {
        let block = hex("82 86 84 41 8c f1 e3 c2 e5 f2 3a 6b a0 ab 90 f4 ff");
        let mut ctx = HpackCtx::new();
        let headers = decode(&block, &mut ctx).unwrap();
        let pairs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                (":method", "GET"),
                (":scheme", "http"),
                (":path", "/"),
                (":authority", "www.example.com"),
            ]
        );
        assert_eq!(ctx.dynamic_len(), 1);
    }

    #[test]
    fn decodes_rfc_c61_response() {
        let block = hex(
            "48 82 64 02 58 85 ae c3 77 1a 4b 61 96 d0 7a be 94 10 54 d4 44 a8 20 05 95 04 0b 81 66 e0 82 a6 2d 1b ff 6e 91 9d 29 ad 17 18 63 c7 8f 0b 97 c8 e9 ae 82 ae 43 d3",
        );
        let mut ctx = HpackCtx::new();
        let headers = decode(&block, &mut ctx).unwrap();
        assert_eq!(headers[0], (":status".to_string(), "302".to_string()));
        assert_eq!(
            headers[1],
            ("cache-control".to_string(), "private".to_string())
        );
        assert_eq!(
            headers[2],
            (
                "date".to_string(),
                "Mon, 21 Oct 2013 20:13:21 GMT".to_string()
            )
        );
        assert_eq!(
            headers[3],
            (
                "location".to_string(),
                "https://www.example.com".to_string()
            )
        );
        assert_eq!(ctx.dynamic_len(), 4);
    }

    #[test]
    fn dynamic_table_is_addressable() {
        let mut ctx = HpackCtx::new();
        let first = hex("82 86 84 41 8c f1 e3 c2 e5 f2 3a 6b a0 ab 90 f4 ff");
        decode(&first, &mut ctx).unwrap();
        let entry = ctx.get(62).expect("dynamic entry");
        assert_eq!(entry.0, ":authority");
        assert_eq!(entry.1, "www.example.com");
        let second = hex("82 86 84 be 58 86 a8 eb 10 64 9c bf");
        let headers = decode(&second, &mut ctx).unwrap();
        assert_eq!(headers[3].0, ":authority");
        assert_eq!(headers[3].1, "www.example.com");
    }

    #[test]
    fn huffman_roundtrip_known_strings() {
        for text in ["www.example.com", "Mon, 21 Oct 2013 20:13:21 GMT", "", "a"] {
            let encoded = match text {
                "www.example.com" => hex("f1 e3 c2 e5 f2 3a 6b a0 ab 90 f4 ff"),
                "Mon, 21 Oct 2013 20:13:21 GMT" => {
                    hex("d0 7a be 94 10 54 d4 44 a8 20 05 95 04 0b 81 66 e0 82 a6 2d 1b ff")
                }
                "a" => hex("1f"),
                _ => Vec::new(),
            };
            assert_eq!(huffman_decode(&encoded).unwrap(), text);
        }
    }

    #[test]
    fn encoder_roundtrips_through_decoder() {
        let headers = vec![
            (":status", "200".to_string()),
            ("content-type", "application/json".to_string()),
            ("content-length", "17".to_string()),
        ];
        let block = encode(&headers);
        let mut ctx = HpackCtx::new();
        let decoded = decode(&block, &mut ctx).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], (":status".to_string(), "200".to_string()));
        assert_eq!(decoded[1].0, "content-type");
        assert_eq!(decoded[2].1, "17");
    }

    #[test]
    fn rejects_bad_padding() {
        assert!(huffman_decode(&[0x00]).is_err());
    }
}
