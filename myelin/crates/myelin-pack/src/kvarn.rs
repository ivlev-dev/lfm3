//! `kvarn` page codec: K 4 bit / V 2 or 4 bit with per-channel and per-token
//! scales, Hadamard rotation.
//!
//! This is a port of the canonical scalar codec in
//! `crates/lfm-npu-ref/src/kvarn.rs` (which mirrors
//! `crates/backend-opencl/kernels/kvarn.cl`). The DSP paged-attention kernel
//! does **not** read this format (see [`crate::kvpage`]); `kvarn` is kept here
//! as the reference codec so the two can be compared and so a future OpenCL /
//! DSP port has a byte-identical encoder. `tests/kvpage.rs` checks every field
//! against `lfm-npu-ref` bit for bit.
//!
//! Page layout of one KV head, 32 tokens x 128 dims, little endian:
//!
//! ```text
//! kcode 32*64 B, nibbles (q + 8) | ((q + 8) << 4), low nibble = even dim
//! ka    128 f16   per-channel K scale
//! kc     32 f16   per-token K scale
//! vcode 32*32 B (k4v2, 2 bits per dim, 4 dims per byte) or 32*64 B (k4v4)
//! vb     32 f16   per-token V step
//! vbz    32 f16   per-token V zero point
//! vc    128 f16   per-channel V scale
//! ```

use half::f16;

use crate::crownstone::fnv1a64;
use crate::manifest::SourceInfo;
use crate::{PackError, Result};

/// Tokens per page.
pub const PAGE_TOKENS: usize = 32;
/// Dimensions per head supported by `kvarn`.
pub const HEAD_DIM: usize = 128;
/// Diagonal signs of the rotation (`LFM_TQ_SIGN`).
pub const ROT_SIGN: [u32; 4] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];
/// Hadamard normalisation `1 / sqrt(128)` as used by the kernel.
pub const ROT_SCALE: f32 = 0.088_388_346;
/// Variance floor of the scale iterations.
pub const VAR_FLOOR: f32 = 1e-8;
/// Lower clamp of `ln(scale)`.
pub const LOG_SCALE_MIN: f32 = -0.3;
/// Upper clamp of `ln(scale)`.
pub const LOG_SCALE_MAX: f32 = 10.0;
/// Scale-balancing iterations.
pub const SCALE_ITERS: usize = 3;

/// Variant of the kvarn format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvarnFormat {
    /// K 4 bit, V 2 bit.
    K4V2,
    /// K 4 bit, V 4 bit with a zero point.
    K4V4,
}

impl KvarnFormat {
    /// Bits per V channel.
    pub const fn v_bits(self) -> usize {
        match self {
            Self::K4V2 => 2,
            Self::K4V4 => 4,
        }
    }

    /// Manifest name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::K4V2 => "kvarn-k4v2",
            Self::K4V4 => "kvarn-k4v4",
        }
    }

    /// Parse a manifest name.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "kvarn-k4v2" | "k4v2" => Ok(Self::K4V2),
            "kvarn-k4v4" | "k4v4" => Ok(Self::K4V4),
            other => Err(PackError::Kvarn(format!(
                "unknown kvarn format `{other}` (expected kvarn-k4v2 or kvarn-k4v4)"
            ))),
        }
    }

    /// Bytes of one encoded page of one head.
    pub const fn page_bytes(self) -> usize {
        let vcode = PAGE_TOKENS * HEAD_DIM * self.v_bits() / 8;
        PAGE_TOKENS * HEAD_DIM / 2
            + HEAD_DIM * 2
            + PAGE_TOKENS * 2
            + vcode
            + PAGE_TOKENS * 4
            + HEAD_DIM * 2
    }
}

/// `lfm_tq_sign(d)`: `-1` or `+1` from the sign table.
pub fn sign(d: usize) -> f32 {
    if (ROT_SIGN[d >> 5] >> (d & 31)) & 1 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// In-place Hadamard transform of length 128 (no normalisation).
fn hadamard(v: &mut [f32; HEAD_DIM]) {
    let mut step = 1;
    while step < HEAD_DIM {
        let mut i = 0;
        while i < HEAD_DIM {
            for j in i..i + step {
                let a = v[j];
                let b = v[j + step];
                v[j] = a + b;
                v[j + step] = a - b;
            }
            i += 2 * step;
        }
        step *= 2;
    }
}

/// Forward rotation `H * D / sqrt(128)`.
pub fn rot_fwd(v: &mut [f32; HEAD_DIM]) {
    for (i, x) in v.iter_mut().enumerate() {
        *x *= sign(i);
    }
    hadamard(v);
    for x in v.iter_mut() {
        *x *= ROT_SCALE;
    }
}

/// Inverse rotation `D * H / sqrt(128)`.
pub fn rot_inv(v: &mut [f32; HEAD_DIM]) {
    hadamard(v);
    for (i, x) in v.iter_mut().enumerate() {
        *x *= ROT_SCALE * sign(i);
    }
}

/// Forward rotation of an f16 vector, f32 in the middle.
pub fn rot_fwd_f16(v: &mut [f16; HEAD_DIM]) {
    let mut buf = [0f32; HEAD_DIM];
    for (dst, src) in buf.iter_mut().zip(v.iter()) {
        *dst = src.to_f32();
    }
    rot_fwd(&mut buf);
    for (dst, src) in v.iter_mut().zip(buf.iter()) {
        *dst = f16::from_f32(*src);
    }
}

/// Inverse rotation of an f16 vector.
pub fn rot_inv_f16(v: &mut [f16; HEAD_DIM]) {
    let mut buf = [0f32; HEAD_DIM];
    for (dst, src) in buf.iter_mut().zip(v.iter()) {
        *dst = src.to_f32();
    }
    rot_inv(&mut buf);
    for (dst, src) in v.iter_mut().zip(buf.iter()) {
        *dst = f16::from_f32(*src);
    }
}

/// Forward rotation of a whole page (32 rows of 128 f16 values).
pub fn rot_fwd_page(v: &mut [f16]) -> Result<()> {
    if v.len() != PAGE_TOKENS * HEAD_DIM {
        return Err(PackError::Kvarn(format!(
            "page has {} values, expected {}",
            v.len(),
            PAGE_TOKENS * HEAD_DIM
        )));
    }
    for row in v.chunks_exact_mut(HEAD_DIM) {
        let row: &mut [f16; HEAD_DIM] = row.try_into().expect("chunk is 128 wide");
        rot_fwd_f16(row);
    }
    Ok(())
}

/// Inverse rotation of a whole page (32 rows of 128 values, f32).
pub fn rot_inv_page(v: &mut [f32]) -> Result<()> {
    if v.len() != PAGE_TOKENS * HEAD_DIM {
        return Err(PackError::Kvarn(format!(
            "page has {} values, expected {}",
            v.len(),
            PAGE_TOKENS * HEAD_DIM
        )));
    }
    for row in v.chunks_exact_mut(HEAD_DIM) {
        let row: &mut [f32; HEAD_DIM] = row.try_into().expect("chunk is 128 wide");
        rot_inv(row);
    }
    Ok(())
}

/// Encoded page of one KV head.
#[derive(Debug, Clone, PartialEq)]
pub struct KvarnPage {
    pub format: KvarnFormat,
    pub kcode: Vec<u8>,
    pub ka: Vec<f16>,
    pub kc: Vec<f16>,
    pub vcode: Vec<u8>,
    pub vb: Vec<f16>,
    pub vbz: Vec<f16>,
    pub vc: Vec<f16>,
}

fn push_f16(out: &mut Vec<u8>, values: &[f16]) {
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn take_f16(bytes: &[u8], count: usize, name: &str) -> Result<Vec<f16>> {
    let need = count * 2;
    if bytes.len() != need {
        return Err(PackError::Kvarn(format!(
            "{name} has {} bytes, expected {need}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn take_bytes<'a>(bytes: &'a [u8], count: usize, name: &str) -> Result<&'a [u8]> {
    if bytes.len() != count {
        return Err(PackError::Kvarn(format!(
            "{name} has {} bytes, expected {count}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

impl KvarnPage {
    /// Size of the serialised page.
    pub fn encoded_len(&self) -> usize {
        self.kcode.len()
            + 2 * self.ka.len()
            + 2 * self.kc.len()
            + self.vcode.len()
            + 2 * self.vb.len()
            + 2 * self.vbz.len()
            + 2 * self.vc.len()
    }

    /// Serialise in the canonical field order.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&self.kcode);
        push_f16(&mut out, &self.ka);
        push_f16(&mut out, &self.kc);
        out.extend_from_slice(&self.vcode);
        push_f16(&mut out, &self.vb);
        push_f16(&mut out, &self.vbz);
        push_f16(&mut out, &self.vc);
        out
    }

    /// Parse a page written by [`Self::encode`].
    pub fn decode(format: KvarnFormat, bytes: &[u8]) -> Result<Self> {
        if bytes.len() != format.page_bytes() {
            return Err(PackError::Kvarn(format!(
                "page has {} bytes, {} needs {}",
                bytes.len(),
                format.name(),
                format.page_bytes()
            )));
        }
        let kcode_len = PAGE_TOKENS * HEAD_DIM / 2;
        let vcode_len = PAGE_TOKENS * HEAD_DIM * format.v_bits() / 8;
        let mut offset = 0;
        let kcode = take_bytes(&bytes[offset..offset + kcode_len], kcode_len, "kcode")?.to_vec();
        offset += kcode_len;
        let ka = take_f16(&bytes[offset..offset + HEAD_DIM * 2], HEAD_DIM, "ka")?;
        offset += HEAD_DIM * 2;
        let kc = take_f16(&bytes[offset..offset + PAGE_TOKENS * 2], PAGE_TOKENS, "kc")?;
        offset += PAGE_TOKENS * 2;
        let vcode = take_bytes(&bytes[offset..offset + vcode_len], vcode_len, "vcode")?.to_vec();
        offset += vcode_len;
        let vb = take_f16(&bytes[offset..offset + PAGE_TOKENS * 2], PAGE_TOKENS, "vb")?;
        offset += PAGE_TOKENS * 2;
        let vbz = take_f16(&bytes[offset..offset + PAGE_TOKENS * 2], PAGE_TOKENS, "vbz")?;
        offset += PAGE_TOKENS * 2;
        let vc = take_f16(&bytes[offset..], HEAD_DIM, "vc")?;
        Ok(Self {
            format,
            kcode,
            ka,
            kc,
            vcode,
            vb,
            vbz,
            vc,
        })
    }

    /// Reconstruct the K vector of one token (rotated basis).
    pub fn decode_k(&self, token: usize) -> [f32; HEAD_DIM] {
        debug_assert!(token < PAGE_TOKENS);
        let ct = self.kc[token].to_f32();
        let mut out = [0f32; HEAD_DIM];
        for (ch, dst) in out.iter_mut().enumerate() {
            let byte = self.kcode[token * (HEAD_DIM / 2) + ch / 2];
            let q = if ch % 2 == 0 {
                (byte & 15) as i32
            } else {
                (byte >> 4) as i32
            } - 8;
            *dst = q as f32 * self.ka[ch].to_f32() * ct;
        }
        out
    }

    /// Reconstruct the V vector of one token (rotated basis).
    pub fn decode_v(&self, token: usize) -> [f32; HEAD_DIM] {
        debug_assert!(token < PAGE_TOKENS);
        let bt = self.vb[token].to_f32();
        let zero = self.vbz[token].to_f32();
        let code_stride = HEAD_DIM * self.format.v_bits() / 8;
        let mut out = [0f32; HEAD_DIM];
        for (ch, dst) in out.iter_mut().enumerate() {
            let q = if self.format == KvarnFormat::K4V4 {
                let byte = self.vcode[token * code_stride + ch / 2];
                if ch % 2 == 0 {
                    (byte & 15) as f32
                } else {
                    (byte >> 4) as f32
                }
            } else {
                let byte = self.vcode[token * code_stride + ch / 4];
                let shift = 2 * (ch % 4);
                ((byte >> shift) & 3) as f32
            };
            *dst = (q * bt + zero) * self.vc[ch].to_f32();
        }
        out
    }
}

/// Encode one page (32 x 128) of already rotated f16 vectors.
///
/// Row-major: `input[token * 128 + ch]`, one KV head.
pub fn commit(format: KvarnFormat, input: &[f16]) -> Result<KvarnPage> {
    let want = PAGE_TOKENS * HEAD_DIM;
    if input.len() != want {
        return Err(PackError::Kvarn(format!(
            "page has {} values, expected {want}",
            input.len()
        )));
    }
    let t: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
    let (mut lr, mut lc) = ([0f32; HEAD_DIM], [0f32; PAGE_TOKENS]);

    for _ in 0..SCALE_ITERS {
        let (ir, ic) = scales(&lr, &lc);
        for (token, lc_token) in lc.iter_mut().enumerate() {
            let v = token_var(&t, token, ic[token], &ir).max(VAR_FLOOR);
            *lc_token = (*lc_token + 0.5 * v.ln()).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
        }
        let (ir, ic) = scales(&lr, &lc);
        for (ch, lr_ch) in lr.iter_mut().enumerate() {
            let v = channel_var(&t, ch, &ic, ir[ch]).max(VAR_FLOOR);
            *lr_ch = (*lr_ch + 0.5 * v.ln()).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
        }
    }

    let (ir, ic) = scales(&lr, &lc);
    let (kcode, ka, kc) = commit_k(&t, &ir, &ic, &lr, &lc);
    let (vcode, vb, vbz, vc) = commit_v(format, &t, &ir, &ic, &lr, &lc);
    Ok(KvarnPage {
        format,
        kcode,
        ka,
        kc,
        vcode,
        vb,
        vbz,
        vc,
    })
}

fn commit_k(
    t: &[f32],
    ir: &[f32; HEAD_DIM],
    ic: &[f32; PAGE_TOKENS],
    lr: &[f32; HEAD_DIM],
    lc: &[f32; PAGE_TOKENS],
) -> (Vec<u8>, Vec<f16>, Vec<f16>) {
    let mut amp = [0f32; HEAD_DIM];
    for (ch, a) in amp.iter_mut().enumerate() {
        let mut mx = VAR_FLOOR;
        for token in 0..PAGE_TOKENS {
            mx = mx.max((t[token * HEAD_DIM + ch] * ic[token] * ir[ch]).abs());
        }
        *a = mx / 7.0;
    }
    let ka: Vec<f16> = amp
        .iter()
        .zip(lr.iter())
        .map(|(a, l)| f16::from_f32(a * l.exp()))
        .collect();
    let kc: Vec<f16> = lc.iter().map(|l| f16::from_f32(l.exp())).collect();

    let mut kcode = vec![0u8; PAGE_TOKENS * HEAD_DIM / 2];
    for token in 0..PAGE_TOKENS {
        let ct = ic[token];
        for lane in 0..HEAD_DIM / 2 {
            let c0 = 2 * lane;
            let c1 = c0 + 1;
            let q0 = quant_k(t[token * HEAD_DIM + c0] * ct * ir[c0] / amp[c0]);
            let q1 = quant_k(t[token * HEAD_DIM + c1] * ct * ir[c1] / amp[c1]);
            kcode[token * (HEAD_DIM / 2) + lane] = ((q0 + 8) | ((q1 + 8) << 4)) as u8;
        }
    }
    (kcode, ka, kc)
}

fn commit_v(
    format: KvarnFormat,
    t: &[f32],
    ir: &[f32; HEAD_DIM],
    ic: &[f32; PAGE_TOKENS],
    lr: &[f32; HEAD_DIM],
    lc: &[f32; PAGE_TOKENS],
) -> (Vec<u8>, Vec<f16>, Vec<f16>, Vec<f16>) {
    let v_bits = format.v_bits();
    let max_level = (1i32 << v_bits) - 1;
    let mut vb = vec![f16::ZERO; PAGE_TOKENS];
    let mut vbz = vec![f16::ZERO; PAGE_TOKENS];
    let mut vcode = vec![0u8; PAGE_TOKENS * HEAD_DIM * v_bits / 8];

    for token in 0..PAGE_TOKENS {
        let (mn, mx) = token_range(t, token, ic[token], ir);
        let mut step = (mx - mn) / max_level as f32;
        if step < VAR_FLOOR {
            step = VAR_FLOOR;
        }
        vb[token] = f16::from_f32(step * lc[token].exp());
        vbz[token] = f16::from_f32(mn * lc[token].exp());
        let ct = ic[token];
        if format == KvarnFormat::K4V4 {
            for lane in 0..HEAD_DIM / 2 {
                let c0 = 2 * lane;
                let c1 = c0 + 1;
                let q0 = quant_v(
                    (t[token * HEAD_DIM + c0] * ct * ir[c0] - mn) / step,
                    max_level,
                );
                let q1 = quant_v(
                    (t[token * HEAD_DIM + c1] * ct * ir[c1] - mn) / step,
                    max_level,
                );
                vcode[token * (HEAD_DIM / 2) + lane] = q0 | (q1 << 4);
            }
        } else {
            for byte in 0..HEAD_DIM / 4 {
                let mut value = 0u8;
                for k in 0..4 {
                    let ch = byte * 4 + k;
                    let q = quant_v((t[token * HEAD_DIM + ch] * ct * ir[ch] - mn) / step, 3);
                    value |= q << (2 * k);
                }
                vcode[token * (HEAD_DIM / 4) + byte] = value;
            }
        }
    }
    let vc: Vec<f16> = lr.iter().map(|l| f16::from_f32(l.exp())).collect();
    (vcode, vb, vbz, vc)
}

fn scales(lr: &[f32; HEAD_DIM], lc: &[f32; PAGE_TOKENS]) -> ([f32; HEAD_DIM], [f32; PAGE_TOKENS]) {
    (
        std::array::from_fn(|c| (-lr[c]).exp()),
        std::array::from_fn(|t| (-lc[t]).exp()),
    )
}

fn token_var(t: &[f32], token: usize, ct: f32, ir: &[f32; HEAD_DIM]) -> f32 {
    let row = &t[token * HEAD_DIM..(token + 1) * HEAD_DIM];
    let sum: f32 = row
        .iter()
        .zip(ir.iter())
        .map(|(v, r)| {
            let x = v * ct * r;
            x * x
        })
        .sum();
    sum / HEAD_DIM as f32
}

fn channel_var(t: &[f32], ch: usize, ic: &[f32; PAGE_TOKENS], r: f32) -> f32 {
    let mut sum = 0f32;
    for token in 0..PAGE_TOKENS {
        let x = t[token * HEAD_DIM + ch] * ic[token] * r;
        sum += x * x;
    }
    sum / PAGE_TOKENS as f32
}

fn token_range(t: &[f32], token: usize, ct: f32, ir: &[f32; HEAD_DIM]) -> (f32, f32) {
    let row = &t[token * HEAD_DIM..(token + 1) * HEAD_DIM];
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for (v, r) in row.iter().zip(ir.iter()) {
        let x = v * ct * r;
        mn = mn.min(x);
        mx = mx.max(x);
    }
    (mn, mx)
}

fn quant_k(x: f32) -> i32 {
    (x.round() as i32).clamp(-7, 7)
}

fn quant_v(x: f32, max: i32) -> u8 {
    (x.round() as i32).clamp(0, max) as u8
}

/// JSON description of an encoded `kvarn` page file.
///
/// The page file is `heads * page_bytes` bytes, one page per head in head
/// order. It is the reference codec and is not read by the DSP kernel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KvarnManifest {
    pub format: String,
    pub format_version: u32,
    /// `kvarn-k4v2` or `kvarn-k4v4`.
    pub variant: String,
    pub heads: usize,
    pub page_tokens: usize,
    pub head_dim: usize,
    pub page_bytes: usize,
    /// The input was rotated with `lfm_tq_rot_fwd` before encoding.
    pub rotate: bool,
    pub encoded_bytes: usize,
    pub fnv1a64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub layout: String,
}

/// Stable `format` tag of [`KvarnManifest`].
pub const MANIFEST_FORMAT: &str = "kvarn-page";
/// Manifest version.
pub const MANIFEST_VERSION: u32 = 1;

impl KvarnManifest {
    /// Describe an encoded page file.
    pub fn new(
        format: KvarnFormat,
        heads: usize,
        rotate: bool,
        source: Option<SourceInfo>,
        encoded: &[u8],
    ) -> Self {
        Self {
            format: MANIFEST_FORMAT.to_string(),
            format_version: MANIFEST_VERSION,
            variant: format.name().to_string(),
            heads,
            page_tokens: PAGE_TOKENS,
            head_dim: HEAD_DIM,
            page_bytes: format.page_bytes(),
            rotate,
            encoded_bytes: encoded.len(),
            fnv1a64: format!("{:016x}", fnv1a64(encoded)),
            source,
            layout: "per head: kcode 2048 B, ka 256 B, kc 64 B, vcode 1024/2048 B, vb 64 B, vbz 64 B, vc 256 B".to_string(),
        }
    }

    /// Format recorded in the manifest.
    pub fn format(&self) -> Result<KvarnFormat> {
        KvarnFormat::parse(&self.variant)
    }

    /// Check the manifest against the encoded bytes.
    pub fn verify(&self, encoded: &[u8]) -> Result<()> {
        if self.format != MANIFEST_FORMAT || self.format_version != MANIFEST_VERSION {
            return Err(PackError::ManifestMismatch(format!(
                "kvarn manifest format {}/{} is not {MANIFEST_FORMAT}/{MANIFEST_VERSION}",
                self.format, self.format_version
            )));
        }
        let format = self.format()?;
        if self.page_tokens != PAGE_TOKENS || self.head_dim != HEAD_DIM {
            return Err(PackError::ManifestMismatch(format!(
                "kvarn page is {}x{}, expected {PAGE_TOKENS}x{HEAD_DIM}",
                self.page_tokens, self.head_dim
            )));
        }
        if self.page_bytes != format.page_bytes()
            || self.encoded_bytes != encoded.len()
            || self.encoded_bytes != self.heads * self.page_bytes
        {
            return Err(PackError::ManifestMismatch(format!(
                "{} bytes for {} heads x {} bytes/page",
                encoded.len(),
                self.heads,
                self.page_bytes
            )));
        }
        let checksum = format!("{:016x}", fnv1a64(encoded));
        if checksum != self.fnv1a64 {
            return Err(PackError::ManifestMismatch(format!(
                "kvarn checksum {} != {checksum}",
                self.fnv1a64
            )));
        }
        for (head, page) in encoded.chunks_exact(self.page_bytes).enumerate() {
            KvarnPage::decode(format, page)
                .map_err(|error| PackError::Kvarn(format!("head {head}: {error}")))?;
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn read(path: &std::path::Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}
