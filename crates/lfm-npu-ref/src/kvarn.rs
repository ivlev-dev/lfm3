//! Скалярный кодек `kvarn` — эталон раскладки OpenCL-ядра.
//!
//! Реализация повторяет `kvarn_commit` и `lfm_kvarn_k/v` из
//! `crates/backend-opencl/kernels/kvarn.cl`:
//!
//! * вход — страница уже **повёрнутых** f16-векторов (32 токена × 128);
//! * K: 4 бита на канал, масштаб на канал (`ka`) и на токен (`kc`);
//! * V (k4v2): 2 бита на канал, масштаб/ноль на токен, масштаб на канал;
//! * V (k4v4): 4 бита на канал с нулевой точкой (ветка `LFM_KV4`).
//!
//! Раскладка байтов (сверено с ядром):
//!
//! * `kcode[tok * 64 + ch/2] = (q0 + 8) | ((q1 + 8) << 4)` для чётного `ch`;
//! * `vcode` k4v2: `ch = 4 * byte + (sh / 2)`, `sh = 4 * (lane & 1)`,
//!   то есть байт `b` хранит каналы `4b..4b+3` по 2 бита;
//! * `vcode` k4v4: нибблы как у `kcode`, но без смещения +8.
//!
//! Вращение — та же ортогональная матрица, что `lfm_tq_rot_fwd`:
//! диагональ знаков `LFM_TQ_SIGN`, быстрое преобразование Адамара длины
//! 128 и нормировка `1/sqrt(128) = 0.0883883476`.

use half::f16;

use crate::NpuError;
use crate::abi::KvFormat;
use crate::layout::{HEAD_DIM, KV_PAGE, page_bytes};

/// Знаки диагональной матрицы `LFM_TQ_SIGN` из `kernels/kvq4.cl`.
pub const ROT_SIGN: [u32; 4] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];
/// Нормировка Адамара `1/sqrt(128)` (значение из ядра, усечённое до f32).
pub const ROT_SCALE: f32 = 0.088_388_346;
/// Нижняя граница дисперсии в итерациях масштабов.
pub const VAR_FLOOR: f32 = 1e-8;
/// Нижняя граница логарифма масштаба (ядро).
pub const LOG_SCALE_MIN: f32 = -0.3;
/// Верхняя граница логарифма масштаба (ядро).
pub const LOG_SCALE_MAX: f32 = 10.0;
/// Число итераций балансировки масштабов (ядро).
pub const SCALE_ITERS: usize = 3;

/// `lfm_tq_sign(d)` из ядра.
pub const fn sign(d: usize) -> f32 {
    if (ROT_SIGN[d >> 5] >> (d & 31)) & 1 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Быстрое преобразование Адамара длины 128 на месте (без нормировки).
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

/// Прямое вращение `H * D / sqrt(128)` (как `lfm_tq_rot_fwd`).
pub fn rot_fwd(v: &mut [f32; HEAD_DIM]) {
    for (i, x) in v.iter_mut().enumerate() {
        *x *= sign(i);
    }
    hadamard(v);
    for x in v.iter_mut() {
        *x *= ROT_SCALE;
    }
}

/// Обратное вращение `D * H / sqrt(128)` (как `lfm_tq_rot_inv`).
pub fn rot_inv(v: &mut [f32; HEAD_DIM]) {
    hadamard(v);
    for (i, x) in v.iter_mut().enumerate() {
        *x *= ROT_SCALE * sign(i);
    }
}

/// Прямое вращение f16-вектора (с промежуточными вычислениями в f32).
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

/// Обратное вращение f16-вектора.
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

/// Закодированная страница одной KV-головы.
#[derive(Debug, Clone, PartialEq)]
pub struct PageCodes {
    /// Формат кодов.
    pub format: KvFormat,
    /// K-коды: 32 * 64 байт (2 канала на байт).
    pub kcode: Vec<u8>,
    /// Масштаб K на канал (f16).
    pub ka: Vec<f16>,
    /// Масштаб K на токен (f16).
    pub kc: Vec<f16>,
    /// V-коды: 32 * 32 (k4v2) или 32 * 64 (k4v4) байт.
    pub vcode: Vec<u8>,
    /// Шаг V на токен (f16).
    pub vb: Vec<f16>,
    /// Нулевая точка V на токен (f16).
    pub vbz: Vec<f16>,
    /// Масштаб V на канал (f16).
    pub vc: Vec<f16>,
}

impl PageCodes {
    /// Полный размер страницы одной головы в байтах.
    pub fn encoded_len(&self) -> usize {
        self.kcode.len()
            + self.ka.len() * 2
            + self.kc.len() * 2
            + self.vcode.len()
            + self.vb.len() * 2
            + self.vbz.len() * 2
            + self.vc.len() * 2
    }

    /// Восстановить K-вектор токена (в повёрнутом базисе).
    ///
    /// # Panics
    ///
    /// Если `token >= 32`.
    pub fn decode_k(&self, token: usize) -> [f32; HEAD_DIM] {
        assert!(token < KV_PAGE, "token {token} >= {KV_PAGE}");
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

    /// Восстановить V-вектор токена (в повёрнутом базисе).
    ///
    /// # Panics
    ///
    /// Если `token >= 32`.
    pub fn decode_v(&self, token: usize) -> [f32; HEAD_DIM] {
        assert!(token < KV_PAGE, "token {token} >= {KV_PAGE}");
        let bt = self.vb[token].to_f32();
        let zero = self.vbz[token].to_f32();
        let code_stride = HEAD_DIM * self.format.v_bits() as usize / 8;
        let mut out = [0f32; HEAD_DIM];
        for (ch, dst) in out.iter_mut().enumerate() {
            let q = if self.format == KvFormat::KvarnK4V4 {
                let byte = self.vcode[token * code_stride + ch / 2];
                if ch % 2 == 0 {
                    (byte & 15) as f32
                } else {
                    (byte >> 4) as f32
                }
            } else {
                let byte = self.vcode[token * code_stride + ch / 4];
                let sh = 2 * (ch % 4);
                ((byte >> sh) & 3) as f32
            };
            *dst = (q * bt + zero) * self.vc[ch].to_f32();
        }
        out
    }
}

/// Кодек `kvarn` для одной KV-головы.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvarnCodec {
    format: KvFormat,
}

impl KvarnCodec {
    /// Создать кодек для k4v2 или k4v4.
    pub fn new(format: KvFormat) -> Result<Self, NpuError> {
        if format.is_kvarn() {
            Ok(Self { format })
        } else {
            Err(NpuError::NotKvarn(format))
        }
    }

    /// Формат кодов.
    pub const fn format(&self) -> KvFormat {
        self.format
    }

    /// Закодировать страницу повёрнутых f16-векторов `32 × 128`.
    ///
    /// Вход — row-major: `input[tok * 128 + ch]`. Один вызов соответствует
    /// одной KV-голове (`get_group_id(0)` в `kvarn_commit`).
    pub fn commit(&self, input: &[f16]) -> Result<PageCodes, NpuError> {
        let want = KV_PAGE * HEAD_DIM;
        if input.len() != want {
            return Err(NpuError::PageLen {
                want,
                got: input.len(),
            });
        }
        let t: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
        let (mut lr, mut lc) = ([0f32; HEAD_DIM], [0f32; KV_PAGE]);

        for _ in 0..SCALE_ITERS {
            let (ir, ic) = scales(&lr, &lc);
            for (tok, lc_tok) in lc.iter_mut().enumerate() {
                let v = token_var(&t, tok, ic[tok], &ir).max(VAR_FLOOR);
                *lc_tok = (*lc_tok + 0.5 * v.ln()).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
            }
            let (ir, ic) = scales(&lr, &lc);
            for (ch, lr_ch) in lr.iter_mut().enumerate() {
                let v = channel_var(&t, ch, &ic, ir[ch]).max(VAR_FLOOR);
                *lr_ch = (*lr_ch + 0.5 * v.ln()).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
            }
        }

        let (ir, ic) = scales(&lr, &lc);
        let k = self.commit_k(&t, &ir, &ic, &lr, &lc);
        let v = self.commit_v(&t, &ir, &ic, &lr, &lc);

        Ok(PageCodes {
            format: self.format,
            kcode: k.kcode,
            ka: k.ka,
            kc: k.kc,
            vcode: v.vcode,
            vb: v.vb,
            vbz: v.vbz,
            vc: v.vc,
        })
    }

    fn commit_k(
        &self,
        t: &[f32],
        ir: &[f32; HEAD_DIM],
        ic: &[f32; KV_PAGE],
        lr: &[f32; HEAD_DIM],
        lc: &[f32; KV_PAGE],
    ) -> KEncoded {
        let mut amp = [0f32; HEAD_DIM];
        for (ch, a) in amp.iter_mut().enumerate() {
            let mut mx = VAR_FLOOR;
            for tok in 0..KV_PAGE {
                mx = mx.max((t[tok * HEAD_DIM + ch] * ic[tok] * ir[ch]).abs());
            }
            *a = mx / 7.0;
        }
        let ka: Vec<f16> = amp
            .iter()
            .zip(lr.iter())
            .map(|(a, l)| f16::from_f32(a * l.exp()))
            .collect();
        let kc: Vec<f16> = lc.iter().map(|l| f16::from_f32(l.exp())).collect();

        let mut kcode = vec![0u8; KV_PAGE * HEAD_DIM / 2];
        for tok in 0..KV_PAGE {
            let ct = ic[tok];
            for lane in 0..HEAD_DIM / 2 {
                let c0 = 2 * lane;
                let c1 = c0 + 1;
                let q0 = quant_k(t[tok * HEAD_DIM + c0] * ct * ir[c0] / amp[c0]);
                let q1 = quant_k(t[tok * HEAD_DIM + c1] * ct * ir[c1] / amp[c1]);
                kcode[tok * (HEAD_DIM / 2) + lane] = ((q0 + 8) | ((q1 + 8) << 4)) as u8;
            }
        }
        KEncoded { kcode, ka, kc }
    }

    fn commit_v(
        &self,
        t: &[f32],
        ir: &[f32; HEAD_DIM],
        ic: &[f32; KV_PAGE],
        lr: &[f32; HEAD_DIM],
        lc: &[f32; KV_PAGE],
    ) -> VEncoded {
        let v_bits = self.format.v_bits();
        let max_level = (1i32 << v_bits) - 1;
        let mut vb = vec![f16::ZERO; KV_PAGE];
        let mut vbz = vec![f16::ZERO; KV_PAGE];
        let mut vcode = vec![0u8; KV_PAGE * HEAD_DIM * v_bits as usize / 8];

        for tok in 0..KV_PAGE {
            let (mn, mx) = token_range(t, tok, ic[tok], ir);
            let mut step = (mx - mn) / max_level as f32;
            if step < VAR_FLOOR {
                step = VAR_FLOOR;
            }
            vb[tok] = f16::from_f32(step * lc[tok].exp());
            vbz[tok] = f16::from_f32(mn * lc[tok].exp());
            let ct = ic[tok];
            if self.format == KvFormat::KvarnK4V4 {
                for lane in 0..HEAD_DIM / 2 {
                    let c0 = 2 * lane;
                    let c1 = c0 + 1;
                    let q0 = quant_v(
                        (t[tok * HEAD_DIM + c0] * ct * ir[c0] - mn) / step,
                        max_level,
                    );
                    let q1 = quant_v(
                        (t[tok * HEAD_DIM + c1] * ct * ir[c1] - mn) / step,
                        max_level,
                    );
                    vcode[tok * (HEAD_DIM / 2) + lane] = q0 | (q1 << 4);
                }
            } else {
                for byte in 0..HEAD_DIM / 4 {
                    let mut b = 0u8;
                    for k in 0..4 {
                        let ch = byte * 4 + k;
                        let q = quant_v((t[tok * HEAD_DIM + ch] * ct * ir[ch] - mn) / step, 3);
                        b |= q << (2 * k);
                    }
                    vcode[tok * (HEAD_DIM / 4) + byte] = b;
                }
            }
        }
        let vc: Vec<f16> = lr.iter().map(|l| f16::from_f32(l.exp())).collect();
        VEncoded { vcode, vb, vbz, vc }
    }
}

struct KEncoded {
    kcode: Vec<u8>,
    ka: Vec<f16>,
    kc: Vec<f16>,
}

struct VEncoded {
    vcode: Vec<u8>,
    vb: Vec<f16>,
    vbz: Vec<f16>,
    vc: Vec<f16>,
}

fn scales(lr: &[f32; HEAD_DIM], lc: &[f32; KV_PAGE]) -> ([f32; HEAD_DIM], [f32; KV_PAGE]) {
    (
        std::array::from_fn(|c| (-lr[c]).exp()),
        std::array::from_fn(|t| (-lc[t]).exp()),
    )
}

fn token_var(t: &[f32], tok: usize, ct: f32, ir: &[f32; HEAD_DIM]) -> f32 {
    let row = &t[tok * HEAD_DIM..(tok + 1) * HEAD_DIM];
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

fn channel_var(t: &[f32], ch: usize, ic: &[f32; KV_PAGE], r: f32) -> f32 {
    let mut sum = 0f32;
    for tok in 0..KV_PAGE {
        let x = t[tok * HEAD_DIM + ch] * ic[tok] * r;
        sum += x * x;
    }
    sum / KV_PAGE as f32
}

fn token_range(t: &[f32], tok: usize, ct: f32, ir: &[f32; HEAD_DIM]) -> (f32, f32) {
    let row = &t[tok * HEAD_DIM..(tok + 1) * HEAD_DIM];
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

/// Проверка, что размер кодов совпадает с формулой арены (тест/диагностика).
pub fn codes_match_layout(format: KvFormat, kv_heads: usize) -> bool {
    if KvarnCodec::new(format).is_err() {
        return false;
    }
    let per_head = page_bytes(format, 1, HEAD_DIM) as usize;
    let empty = PageCodes {
        format,
        kcode: vec![0; KV_PAGE * HEAD_DIM / 2],
        ka: vec![f16::ZERO; HEAD_DIM],
        kc: vec![f16::ZERO; KV_PAGE],
        vcode: vec![0; KV_PAGE * HEAD_DIM * format.v_bits() as usize / 8],
        vb: vec![f16::ZERO; KV_PAGE],
        vbz: vec![f16::ZERO; KV_PAGE],
        vc: vec![f16::ZERO; HEAD_DIM],
    };
    empty.encoded_len() == per_head
        && page_bytes(format, kv_heads, HEAD_DIM) as usize == per_head * kv_heads
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }

        fn signed(&mut self) -> f32 {
            self.unit() * 2.0 - 1.0
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn page_from(seed: u64, dist: impl Fn(&mut Lcg) -> f32) -> Vec<f16> {
        let mut rng = Lcg::new(seed);
        (0..KV_PAGE * HEAD_DIM)
            .map(|_| f16::from_f32(dist(&mut rng)))
            .collect()
    }

    fn as_f32(page: &[f16]) -> Vec<f32> {
        page.iter().map(|x| x.to_f32()).collect()
    }

    fn cos_per_token(input: &[f32], decoded: impl Fn(usize) -> [f32; HEAD_DIM]) -> Vec<f32> {
        (0..KV_PAGE)
            .map(|tok| {
                let row = &input[tok * HEAD_DIM..(tok + 1) * HEAD_DIM];
                cosine(row, &decoded(tok))
            })
            .collect()
    }

    #[test]
    fn sign_table_matches_kernel() {
        // 0x9E3779B9: бит 0 = 1, бит 1 = 0.
        assert_eq!(sign(0), -1.0);
        assert_eq!(sign(1), 1.0);
        // 0x85EBCA6B: бит 0 = 1.
        assert_eq!(sign(32), -1.0);
        // 0xC2B2AE35: бит 0 = 1.
        assert_eq!(sign(64), -1.0);
        // 0x27D4EB2F: бит 0 = 1.
        assert_eq!(sign(96), -1.0);
        // 0x9E3779B9: бит 2 = 0.
        assert_eq!(sign(2), 1.0);
    }

    #[test]
    fn rotation_is_inverse() {
        let mut rng = Lcg::new(42);
        let orig: [f32; HEAD_DIM] = std::array::from_fn(|_| rng.signed());
        let mut v = orig;
        rot_fwd(&mut v);
        rot_inv(&mut v);
        for (a, b) in orig.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let mut rng = Lcg::new(43);
        let orig: [f32; HEAD_DIM] = std::array::from_fn(|_| rng.signed());
        let norm_in: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut v = orig;
        rot_fwd(&mut v);
        let norm_out: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_in - norm_out).abs() < 1e-4);
        // Эталонная матрица: 1/sqrt(128) * sqrt(128) = 1.
        assert!((ROT_SCALE * (128f32).sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_first_basis_matches_kernel() {
        let mut v = [0f32; HEAD_DIM];
        v[0] = 1.0;
        rot_fwd(&mut v);
        // sign(0) = -1, H * e0 = все единицы, масштаб 1/sqrt(128).
        for x in v {
            assert!((x + ROT_SCALE).abs() < 1e-7, "{x}");
        }
    }

    #[test]
    fn rotation_f16_roundtrip() {
        let mut rng = Lcg::new(44);
        let mut orig = [f16::ZERO; HEAD_DIM];
        for x in orig.iter_mut() {
            *x = f16::from_f32(rng.signed());
        }
        let mut v = orig;
        rot_fwd_f16(&mut v);
        rot_inv_f16(&mut v);
        for (a, b) in orig.iter().zip(v.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 3e-3);
        }
    }

    #[test]
    fn page_sizes_match_layout() {
        assert!(codes_match_layout(KvFormat::KvarnK4V2, 4));
        assert!(codes_match_layout(KvFormat::KvarnK4V4, 4));
        assert_eq!(page_bytes(KvFormat::KvarnK4V2, 1, 128), 3_776);
        assert_eq!(page_bytes(KvFormat::KvarnK4V4, 1, 128), 4_800);
        assert!(!codes_match_layout(KvFormat::F16, 4));
    }

    #[test]
    fn commit_decode_rademacher_cos_above_0999() {
        // ±1-векторы: 15 уровней K и 4 уровня V покрывают их точно,
        // поэтому косинус проверяет именно раскладку/масштабы, а не потери.
        let page = page_from(7, |rng| if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 });
        let input = as_f32(&page);
        for format in [KvFormat::KvarnK4V2, KvFormat::KvarnK4V4] {
            let codec = KvarnCodec::new(format).unwrap();
            let codes = codec.commit(&page).unwrap();
            assert_eq!(
                codes.encoded_len(),
                page_bytes(format, 1, HEAD_DIM) as usize
            );
            let ck = cos_per_token(&input, |t| codes.decode_k(t));
            let cv = cos_per_token(&input, |t| codes.decode_v(t));
            for (tok, c) in ck.iter().enumerate() {
                assert!(*c >= 0.999, "{format:?} K tok {tok}: cos {c}");
            }
            for (tok, c) in cv.iter().enumerate() {
                assert!(*c >= 0.999, "{format:?} V tok {tok}: cos {c}");
            }
        }
    }

    #[test]
    fn continuous_page_quality_bounds() {
        // Непрерывные равномерные векторы: 4 бита K дают cos ≈ 0.9977,
        // 2 бита V — ≈ 0.952 (теоретический предел 4 уровней).
        let page = page_from(11, |rng| rng.signed());
        let input = as_f32(&page);
        let codec = KvarnCodec::new(KvFormat::KvarnK4V2).unwrap();
        let codes = codec.commit(&page).unwrap();
        let ck = cos_per_token(&input, |t| codes.decode_k(t));
        let cv = cos_per_token(&input, |t| codes.decode_v(t));
        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(avg(&ck) >= 0.995, "avg K {}", avg(&ck));
        assert!(avg(&cv) >= 0.93, "avg V {}", avg(&cv));
        assert!(
            ck.iter().all(|c| *c >= 0.99),
            "min K {}",
            ck.iter().cloned().fold(f32::INFINITY, f32::min)
        );
        assert!(
            cv.iter().all(|c| *c >= 0.88),
            "min V {}",
            cv.iter().cloned().fold(f32::INFINITY, f32::min)
        );
    }

    #[test]
    fn commit_is_byte_stable() {
        let page = page_from(13, |rng| rng.signed());
        for format in [KvFormat::KvarnK4V2, KvFormat::KvarnK4V4] {
            let codec = KvarnCodec::new(format).unwrap();
            let a = codec.commit(&page).unwrap();
            let b = codec.commit(&page).unwrap();
            assert_eq!(a, b, "{format:?}");
            assert_eq!(a.kcode, b.kcode);
            assert_eq!(a.vcode, b.vcode);
            let bits = |v: &[f16]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.ka), bits(&b.ka));
            assert_eq!(bits(&a.kc), bits(&b.kc));
            assert_eq!(bits(&a.vb), bits(&b.vb));
            assert_eq!(bits(&a.vbz), bits(&b.vbz));
            assert_eq!(bits(&a.vc), bits(&b.vc));
        }
    }

    #[test]
    fn k_int4_boundaries_reached() {
        // Каждый канал постоянен по токенам: чётные +1, нечётные -1.
        let mut page = vec![f16::ZERO; KV_PAGE * HEAD_DIM];
        for tok in 0..KV_PAGE {
            for ch in 0..HEAD_DIM {
                page[tok * HEAD_DIM + ch] = if ch % 2 == 0 { f16::ONE } else { f16::NEG_ONE };
            }
        }
        let codec = KvarnCodec::new(KvFormat::KvarnK4V2).unwrap();
        let codes = codec.commit(&page).unwrap();
        for tok in 0..KV_PAGE {
            for lane in 0..HEAD_DIM / 2 {
                let b = codes.kcode[tok * (HEAD_DIM / 2) + lane];
                assert_eq!(b & 15, 15, "q=+7 => nibble 15");
                assert_eq!(b >> 4, 1, "q=-7 => nibble 1 (код 0 не используется)");
            }
        }
        assert!(codes.kcode.iter().all(|b| (b & 15) >= 1 && (b >> 4) >= 1));
        let input = as_f32(&page);
        for tok in 0..KV_PAGE {
            let dec = codes.decode_k(tok);
            for ch in 0..HEAD_DIM {
                assert!(
                    (dec[ch] - input[tok * HEAD_DIM + ch]).abs() < 1e-3,
                    "tok {tok} ch {ch}: {} != {}",
                    dec[ch],
                    input[tok * HEAD_DIM + ch]
                );
            }
        }
    }

    #[test]
    fn v2_levels_and_zero_point_reached() {
        // В каждом токене один минимум -3 и один максимум +5.
        let mut page = vec![f16::ZERO; KV_PAGE * HEAD_DIM];
        for tok in 0..KV_PAGE {
            page[tok * HEAD_DIM] = f16::from_f32(-3.0);
            page[tok * HEAD_DIM + 1] = f16::from_f32(5.0);
        }
        let codec = KvarnCodec::new(KvFormat::KvarnK4V2).unwrap();
        let codes = codec.commit(&page).unwrap();
        for tok in 0..KV_PAGE {
            let b = codes.vcode[tok * (HEAD_DIM / 4)];
            assert_eq!(b & 3, 0, "минимум => уровень 0");
            assert_eq!((b >> 2) & 3, 3, "максимум => уровень 3");
            let dec = codes.decode_v(tok);
            assert!((dec[0] + 3.0).abs() < 0.02, "min {}", dec[0]);
            assert!((dec[1] - 5.0).abs() < 0.02, "max {}", dec[1]);
        }
    }

    #[test]
    fn v4_levels_reach_nibble_edges() {
        let mut page = vec![f16::ZERO; KV_PAGE * HEAD_DIM];
        for tok in 0..KV_PAGE {
            page[tok * HEAD_DIM] = f16::from_f32(-2.0);
            page[tok * HEAD_DIM + 1] = f16::from_f32(2.0);
        }
        let codec = KvarnCodec::new(KvFormat::KvarnK4V4).unwrap();
        let codes = codec.commit(&page).unwrap();
        for tok in 0..KV_PAGE {
            let b = codes.vcode[tok * (HEAD_DIM / 2)];
            assert_eq!(b & 15, 0, "минимум => 0");
            assert_eq!(b >> 4, 15, "максимум => 15");
        }
        // Нулевые каналы при 4 битах и широком диапазоне дают заметную
        // ошибку (уровень 8 из 15), поэтому проверяем точность краёв,
        // а не косинус всей страницы.
        for tok in 0..KV_PAGE {
            let dec = codes.decode_v(tok);
            assert!((dec[0] + 2.0).abs() < 0.02, "min {}", dec[0]);
            assert!((dec[1] - 2.0).abs() < 0.02, "max {}", dec[1]);
        }
    }

    #[test]
    fn rejects_bad_input() {
        let codec = KvarnCodec::new(KvFormat::KvarnK4V2).unwrap();
        let short = vec![f16::ZERO; 10];
        assert_eq!(
            codec.commit(&short).unwrap_err(),
            NpuError::PageLen {
                want: KV_PAGE * HEAD_DIM,
                got: 10
            }
        );
        assert_eq!(
            KvarnCodec::new(KvFormat::F16).unwrap_err(),
            NpuError::NotKvarn(KvFormat::F16)
        );
    }

    #[test]
    fn error_message_mentions_parameter() {
        let e = crate::invalid("ctx", "должен быть > 0");
        assert!(e.to_string().contains("ctx"));
        assert!(e.to_abi_oom().is_none());
    }
}
