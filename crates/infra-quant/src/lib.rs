use thiserror::Error;

pub const QK_K: usize = 256;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid length: expected {expected}, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("invalid n: {n} (must be multiple of {multiple})")]
    InvalidN { n: usize, multiple: usize },
}

pub type QuantError = Error;
pub type Result<T> = std::result::Result<T, Error>;

pub fn half_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

pub fn f16s_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        out.push(half_bits_to_f32(bits));
    }
    out
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub fn dequant_q4_k(block: &[u8; 144]) -> [f32; 256] {
    let d = half_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = half_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut y = [0.0f32; 256];
    let mut q_off = 0usize;
    let mut is = 0usize;
    for j in (0..256).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * (sc1 as f32);
        let m1f = dmin * (m1 as f32);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * (sc2 as f32);
        let m2f = dmin * (m2 as f32);
        for l in 0..32usize {
            y[j + l] = d1 * ((qs[q_off + l] & 0x0F) as f32) - m1f;
        }
        for l in 0..32usize {
            y[j + 32 + l] = d2 * ((qs[q_off + l] >> 4) as f32) - m2f;
        }
        q_off += 32;
        is += 2;
    }
    y
}

pub fn dequant_q6_k(block: &[u8; 210]) -> [f32; 256] {
    let d = half_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql0 = &block[0..128];
    let qh0 = &block[128..192];
    let sc0 = &block[192..208];
    let mut y = [0.0f32; 256];
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for n in (0..256).step_by(128) {
        for l in 0..32usize {
            let is = l / 16;
            let q1 = (((ql0[ql_off + l] & 0x0F) as i32)
                | ((((qh0[qh_off + l] >> 0) & 3) as i32) << 4))
                - 32;
            let q2 = (((ql0[ql_off + l + 32] & 0x0F) as i32)
                | ((((qh0[qh_off + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 = (((ql0[ql_off + l] >> 4) as i32)
                | ((((qh0[qh_off + l] >> 4) & 3) as i32) << 4))
                - 32;
            let q4 = (((ql0[ql_off + l + 32] >> 4) as i32)
                | ((((qh0[qh_off + l] >> 6) & 3) as i32) << 4))
                - 32;
            y[n + l + 0] =
                d * (sc0[sc_off + is + 0] as i8 as f32) * (q1 as f32);
            y[n + l + 32] =
                d * (sc0[sc_off + is + 2] as i8 as f32) * (q2 as f32);
            y[n + l + 64] =
                d * (sc0[sc_off + is + 4] as i8 as f32) * (q3 as f32);
            y[n + l + 96] =
                d * (sc0[sc_off + is + 6] as i8 as f32) * (q4 as f32);
        }
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    y
}

pub fn dequant_q8_0(block: &[u8; 34]) -> [f32; 32] {
    let d = half_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut y = [0.0f32; 32];
    for i in 0..32usize {
        y[i] = d * (block[2 + i] as i8 as f32);
    }
    y
}

pub fn dequant_row_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % 256 != 0 {
        return Err(Error::InvalidN { n, multiple: 256 });
    }
    let nblocks = n / 256;
    let expected = nblocks * 144;
    if bytes.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..nblocks {
        let mut block = [0u8; 144];
        block.copy_from_slice(&bytes[b * 144..(b + 1) * 144]);
        let arr = dequant_q4_k(&block);
        out.extend_from_slice(&arr);
    }
    Ok(out)
}

pub fn dequant_row_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % 256 != 0 {
        return Err(Error::InvalidN { n, multiple: 256 });
    }
    let nblocks = n / 256;
    let expected = nblocks * 210;
    if bytes.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..nblocks {
        let mut block = [0u8; 210];
        block.copy_from_slice(&bytes[b * 210..(b + 1) * 210]);
        let arr = dequant_q6_k(&block);
        out.extend_from_slice(&arr);
    }
    Ok(out)
}

pub fn dot_row_q4_k(bytes: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    if n % 256 != 0 {
        return Err(Error::InvalidN { n, multiple: 256 });
    }
    let nblocks = n / 256;
    let expected = nblocks * 144;
    if bytes.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    let y = dequant_row_q4_k(bytes, n).map_err(|_| Error::InvalidLength {
        expected,
        actual: bytes.len(),
    })?;
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += y[i] * x[i];
    }
    Ok(acc)
}

pub fn dot_row_q6_k(bytes: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    if n % 256 != 0 {
        return Err(Error::InvalidN { n, multiple: 256 });
    }
    let nblocks = n / 256;
    let expected = nblocks * 210;
    if bytes.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    let y = dequant_row_q6_k(bytes, n).map_err(|_| Error::InvalidLength {
        expected,
        actual: bytes.len(),
    })?;
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += y[i] * x[i];
    }
    Ok(acc)
}

pub fn dot_row_q8_0(bytes: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    if n % 32 != 0 {
        return Err(Error::InvalidN { n, multiple: 32 });
    }
    let nblocks = n / 32;
    let expected = nblocks * 34;
    if bytes.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: bytes.len(),
        });
    }
    let mut acc = 0.0f32;
    for b in 0..nblocks {
        let mut block = [0u8; 34];
        block.copy_from_slice(&bytes[b * 34..(b + 1) * 34]);
        let y = dequant_q8_0(&block);
        for i in 0..32usize {
            acc += y[i] * x[b * 32 + i];
        }
    }
    Ok(acc)
}