//! KV page images in the format the DSP paged-attention kernel reads.
//!
//! The kernel is `attn_core()` in `myelin/dsp/hvx/lfmgemv_impl.c`; the layouts
//! below are transcribed from it and from `attn_main.c`, not invented:
//!
//! * **K** — int4 per dimension, offset-8 nibbles (`nibble = q + 8`, low
//!   nibble = even dimension), `q` in `-7..=7`. Per-token scale `kt` is a Q16
//!   `i32` table (`score = mulq16((dot - 8*sum_q) << 8, kt)`), so the decoded
//!   key is `code * kt / 65536`.
//! * **V** — int8 per dimension, one value per token, used by the PV pass as
//!   `out = sum_t p_t * v[t][d] / sum_t p_t` (the kernel applies no scale).
//!   Because tokens are mixed, the scale of V must be constant over tokens:
//!   this packer therefore stores either one scalar per image ([`VScaleMode::Global`])
//!   or one scale per head and dimension ([`VScaleMode::Channel`]), and the
//!   host applies it after attention (`out[d] * v_scale`). A per-token V
//!   scale is mathematically impossible here.
//! * **Two layouts** — `row` (head-grouped, 128 B per token: 64 B codes + 64 B
//!   zero pad, the one `exec` uses: `g_attn_tm = 0`) and `page` (token-major,
//!   the transposed page the token-major kernels use: K `16 x 128` B per page,
//!   V `4096` B per page).
//!
//! This is **not** `kvarn`: the DSP kernel has no Hadamard rotation, no
//! per-channel K scale, no V zero point and reads int8 V, while `kvarn` is K
//! 4 bit + V 2/4 bit with a rotation. [`crate::kvarn`] keeps that reference
//! codec side by side, disconnected from the DSP path.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::crownstone::fnv1a64;
use crate::manifest::SourceInfo;
use crate::{PackError, Result};

/// Tokens per KV page (fixed by the kernel).
pub const PAGE_TOKENS: usize = 32;
/// Dimensions per head the DSP attention kernel is compiled for.
pub const HEAD_DIM: usize = 128;
/// KV heads the current DSP kernel is compiled for (`ATTN_KV`).
pub const DSP_KV_HEADS: usize = 4;
/// Query heads of the current DSP kernel (`ATTN_HQ`).
pub const DSP_QUERY_HEADS: usize = 16;
/// Stable name written into the manifest.
pub const FORMAT_NAME: &str = "myelin-dsp-kv-page";
/// Manifest format version.
pub const FORMAT_VERSION: u32 = 1;
/// Largest `kt` that keeps `mulq16((dot - 8*sum_q) << 8, kt)` inside int32:
/// `|dot| <= 128 * 8 * 128`, so `(dot >> 8) * kt < 2^31` needs `kt <= 4194303`.
pub const MAX_KT_Q16: i32 = 4_194_303;

/// How a page image is addressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// Head-grouped, 128 bytes per token (64 code bytes + 64 zero pad).
    Row,
    /// Token-major, transposed per 32-token page.
    Page,
}

impl KvLayout {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Page => "page",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "row" | "head-grouped" => Ok(Self::Row),
            "page" | "token-major" => Ok(Self::Page),
            other => Err(PackError::KvPage(format!(
                "unknown layout `{other}` (expected row or page)"
            ))),
        }
    }
}

/// How the V scale is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VScaleMode {
    /// One scalar for the whole image; folds into a scalar `out_scale`.
    Global,
    /// One scale per head and dimension; applied after attention.
    Channel,
}

impl VScaleMode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Channel => "channel",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "global" | "head" => Ok(Self::Global),
            "channel" | "per-channel" => Ok(Self::Channel),
            other => Err(PackError::KvPage(format!(
                "unknown V scale mode `{other}` (expected global or channel)"
            ))),
        }
    }
}

/// Shape of the image to pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPackParams {
    pub heads: usize,
    /// Number of 32-token pages.
    pub pages: usize,
    /// Valid tokens in the last page (1..=32); the rest is zero padded.
    pub tokens_in_last_page: usize,
    pub layout: KvLayout,
    pub v_scale_mode: VScaleMode,
}

impl KvPackParams {
    /// Slots (tokens) the image holds, padded to whole pages.
    pub const fn slots(&self) -> usize {
        self.pages * PAGE_TOKENS
    }

    /// Valid tokens across all pages.
    pub const fn valid_tokens(&self) -> usize {
        (self.pages - 1) * PAGE_TOKENS + self.tokens_in_last_page
    }

    /// Bytes of the K section per head.
    pub const fn k_bytes_per_head(&self) -> usize {
        match self.layout {
            KvLayout::Row => self.slots() * HEAD_DIM,
            KvLayout::Page => self.pages * 16 * 128,
        }
    }

    /// Bytes of the V section per head.
    pub const fn v_bytes_per_head(&self) -> usize {
        match self.layout {
            KvLayout::Row => self.slots() * HEAD_DIM,
            KvLayout::Page => self.pages * 4096,
        }
    }

    /// Bytes of the per-token K scale section per head.
    pub const fn kt_bytes_per_head(&self) -> usize {
        self.slots() * 4
    }

    fn validate(&self) -> Result<()> {
        if self.heads == 0 || self.heads > 16 {
            return Err(PackError::KvPage(format!(
                "head count {} is outside 1..=16",
                self.heads
            )));
        }
        if self.pages == 0 || self.pages > 4096 {
            return Err(PackError::KvPage(format!(
                "page count {} is outside 1..=4096",
                self.pages
            )));
        }
        if self.tokens_in_last_page == 0 || self.tokens_in_last_page > PAGE_TOKENS {
            return Err(PackError::KvPage(format!(
                "tokens in the last page {} is outside 1..={PAGE_TOKENS}",
                self.tokens_in_last_page
            )));
        }
        Ok(())
    }
}

/// Packed K/KT/V images plus the scales they were built from.
#[derive(Debug, Clone)]
pub struct KvImage {
    pub params: KvPackParams,
    /// K codes, `heads * k_bytes_per_head`.
    pub k: Vec<u8>,
    /// Per-token K scale, `heads * slots` little-endian Q16 `i32`.
    pub kt: Vec<u8>,
    /// V codes, `heads * v_bytes_per_head`.
    pub v: Vec<u8>,
    /// `kt` as integers, same order as the image.
    pub k_scales_q16: Vec<i32>,
    /// One value ([`VScaleMode::Global`]) or `heads * 128` values
    /// ([`VScaleMode::Channel`]).
    pub v_scales: Vec<f32>,
    /// `max |k|` per head and slot (diagnostics and metrics).
    pub k_max_abs: Vec<f32>,
    /// `max |v|` per head and dimension, or one value for the global mode.
    pub v_max_abs: Vec<f32>,
}

fn quantize_int4(value: f32, scale: f32) -> i8 {
    if scale <= 0.0 {
        return 0;
    }
    ((value / scale).round() as i32).clamp(-7, 7) as i8
}

fn quantize_int8(value: f32, scale: f32) -> i8 {
    if scale <= 0.0 {
        return 0;
    }
    ((value / scale).round() as i32).clamp(-127, 127) as i8
}

/// Pack `[head][page][token][dim]` f32 values into the DSP images.
pub fn build(values: &[f32], params: KvPackParams) -> Result<KvImage> {
    params.validate()?;
    let valid = params.valid_tokens();
    let expected = params
        .heads
        .checked_mul(valid)
        .and_then(|n| n.checked_mul(HEAD_DIM))
        .ok_or_else(|| PackError::KvPage("input size overflows usize".to_string()))?;
    if values.len() != expected {
        return Err(PackError::KvPage(format!(
            "input has {} values, expected heads {heads} * tokens {valid} * {HEAD_DIM} = {expected}",
            values.len(),
            heads = params.heads
        )));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PackError::NonFiniteWeight {
                row: index / HEAD_DIM,
                col: index % HEAD_DIM,
                value: *value,
            });
        }
    }

    // ---- pass 1: K scales, V scales and the max|.| tables ------------------
    let slots = params.slots();
    let mut k_scales_q16 = vec![0i32; params.heads * slots];
    let mut k_max_abs = vec![0f32; params.heads * slots];
    let mut v_max_abs = match params.v_scale_mode {
        VScaleMode::Global => vec![0f32; 1],
        VScaleMode::Channel => vec![0f32; params.heads * HEAD_DIM],
    };

    for head in 0..params.heads {
        for slot in 0..slots {
            if slot >= valid {
                continue;
            }
            let row =
                &values[(head * valid + slot) * HEAD_DIM..(head * valid + slot + 1) * HEAD_DIM];
            let mut max_abs = 0.0f32;
            for &value in row {
                max_abs = max_abs.max(value.abs());
            }
            let scale = max_abs / 7.0;
            let mut q16 = (f64::from(scale) * 65536.0).round();
            if max_abs > 0.0 && q16 < 1.0 {
                q16 = 1.0;
            }
            if !(0.0..=f64::from(MAX_KT_Q16)).contains(&q16) {
                return Err(PackError::KvPage(format!(
                    "head {head} token {slot}: K scale {scale} does not fit the Q16 kernel multiplier (max |k| {} must stay below {})",
                    max_abs,
                    f64::from(MAX_KT_Q16) / 65536.0 * 7.0
                )));
            }
            k_scales_q16[head * slots + slot] = q16 as i32;
            k_max_abs[head * slots + slot] = max_abs;

            match params.v_scale_mode {
                VScaleMode::Global => {
                    for &value in row {
                        v_max_abs[0] = v_max_abs[0].max(value.abs());
                    }
                }
                VScaleMode::Channel => {
                    let base = head * HEAD_DIM;
                    for (dim, &value) in row.iter().enumerate() {
                        v_max_abs[base + dim] = v_max_abs[base + dim].max(value.abs());
                    }
                }
            }
        }
    }

    let v_scales: Vec<f32> = v_max_abs.iter().map(|max| max / 127.0).collect();

    // ---- pass 2: codes ----------------------------------------------------
    let mut k = vec![0u8; params.heads * params.k_bytes_per_head()];
    let mut v = vec![0u8; params.heads * params.v_bytes_per_head()];
    let mut kt = vec![0u8; params.heads * params.kt_bytes_per_head()];

    for head in 0..params.heads {
        let k_stride = params.k_bytes_per_head();
        let v_stride = params.v_bytes_per_head();
        for slot in 0..valid {
            let row =
                &values[(head * valid + slot) * HEAD_DIM..(head * valid + slot + 1) * HEAD_DIM];
            let k_scale = k_scales_q16[head * slots + slot] as f32 / 65536.0;
            let page = slot / PAGE_TOKENS;
            let token = slot % PAGE_TOKENS;
            let mut codes = [0i8; HEAD_DIM];
            for (dim, &value) in row.iter().enumerate() {
                codes[dim] = quantize_int4(value, k_scale);
            }
            match params.layout {
                KvLayout::Row => {
                    let row_base = head * k_stride + slot * HEAD_DIM;
                    for lane in 0..HEAD_DIM / 2 {
                        let lo = (codes[2 * lane] + 8) as u8;
                        let hi = (codes[2 * lane + 1] + 8) as u8;
                        k[row_base + lane] = lo | (hi << 4);
                    }
                }
                KvLayout::Page => {
                    let page_base = head * k_stride + page * 16 * 128;
                    for g in 0..16 {
                        for j in 0..4 {
                            let d0 = 8 * g + 2 * j;
                            let lo = (codes[d0] + 8) as u8;
                            let hi = (codes[d0 + 1] + 8) as u8;
                            k[page_base + g * 128 + 4 * token + j] = lo | (hi << 4);
                        }
                    }
                }
            }
            for (dim, &value) in row.iter().enumerate() {
                let v_scale = match params.v_scale_mode {
                    VScaleMode::Global => v_scales[0],
                    VScaleMode::Channel => v_scales[head * HEAD_DIM + dim],
                };
                let code = quantize_int8(value, v_scale);
                match params.layout {
                    KvLayout::Row => {
                        v[head * v_stride + slot * HEAD_DIM + dim] = code as u8;
                    }
                    KvLayout::Page => {
                        let g = dim >> 5;
                        let d = dim & 31;
                        let c = token / 4;
                        let j = token % 4;
                        v[head * v_stride + page * 4096 + g * 1024 + c * 128 + 4 * d + j] =
                            code as u8;
                    }
                }
            }
        }
    }

    for head in 0..params.heads {
        for slot in 0..slots {
            let offset = (head * slots + slot) * 4;
            kt[offset..offset + 4]
                .copy_from_slice(&k_scales_q16[head * slots + slot].to_le_bytes());
        }
    }

    Ok(KvImage {
        params,
        k,
        kt,
        v,
        k_scales_q16,
        v_scales,
        k_max_abs,
        v_max_abs,
    })
}

impl KvImage {
    /// Effective K scale of one head and slot.
    pub fn k_scale(&self, head: usize, slot: usize) -> f32 {
        self.k_scales_q16[head * self.params.slots() + slot] as f32 / 65536.0
    }

    /// Effective V scale of one head and dimension.
    pub fn v_scale(&self, head: usize, dim: usize) -> f32 {
        match self.params.v_scale_mode {
            VScaleMode::Global => self.v_scales[0],
            VScaleMode::Channel => self.v_scales[head * HEAD_DIM + dim],
        }
    }

    /// Decode the int4 K codes of one head and slot.
    pub fn decode_k_codes(&self, head: usize, slot: usize) -> [i8; HEAD_DIM] {
        let mut codes = [0i8; HEAD_DIM];
        let stride = self.params.k_bytes_per_head();
        let page = slot / PAGE_TOKENS;
        let token = slot % PAGE_TOKENS;
        match self.params.layout {
            KvLayout::Row => {
                let base = head * stride + slot * HEAD_DIM;
                for lane in 0..HEAD_DIM / 2 {
                    let byte = self.k[base + lane];
                    codes[2 * lane] = ((byte & 15) as i8) - 8;
                    codes[2 * lane + 1] = ((byte >> 4) as i8) - 8;
                }
            }
            KvLayout::Page => {
                let base = head * stride + page * 16 * 128;
                for g in 0..16 {
                    for j in 0..4 {
                        let d0 = 8 * g + 2 * j;
                        let byte = self.k[base + g * 128 + 4 * token + j];
                        codes[d0] = ((byte & 15) as i8) - 8;
                        codes[d0 + 1] = ((byte >> 4) as i8) - 8;
                    }
                }
            }
        }
        codes
    }

    /// Decode the int8 V codes of one head and slot.
    pub fn decode_v_codes(&self, head: usize, slot: usize) -> [i8; HEAD_DIM] {
        let mut codes = [0i8; HEAD_DIM];
        let stride = self.params.v_bytes_per_head();
        let page = slot / PAGE_TOKENS;
        let token = slot % PAGE_TOKENS;
        match self.params.layout {
            KvLayout::Row => {
                let base = head * stride + slot * HEAD_DIM;
                for (dim, code) in codes.iter_mut().enumerate() {
                    *code = self.v[base + dim] as i8;
                }
            }
            KvLayout::Page => {
                let base = head * stride + page * 4096;
                for (dim, code) in codes.iter_mut().enumerate() {
                    let g = dim >> 5;
                    let d = dim & 31;
                    let c = token / 4;
                    let j = token % 4;
                    *code = self.v[base + g * 1024 + c * 128 + 4 * d + j] as i8;
                }
            }
        }
        codes
    }

    /// Decode the whole image to f32 in `[head][slot][dim]` order
    /// (`heads * slots * 128` values, padding slots included and zero).
    pub fn decode(&self) -> (Vec<f32>, Vec<f32>) {
        let slots = self.params.slots();
        let mut keys = vec![0f32; self.params.heads * slots * HEAD_DIM];
        let mut values = vec![0f32; self.params.heads * slots * HEAD_DIM];
        for head in 0..self.params.heads {
            for slot in 0..slots {
                let k_scale = self.k_scale(head, slot);
                let k_codes = self.decode_k_codes(head, slot);
                let v_codes = self.decode_v_codes(head, slot);
                for dim in 0..HEAD_DIM {
                    keys[(head * slots + slot) * HEAD_DIM + dim] =
                        f32::from(k_codes[dim]) * k_scale;
                    values[(head * slots + slot) * HEAD_DIM + dim] =
                        f32::from(v_codes[dim]) * self.v_scale(head, dim);
                }
            }
        }
        (keys, values)
    }
}

/// Quality of one KV image, measured after decoding the stored bytes.
#[derive(Debug, Clone, Serialize)]
pub struct KvQuality {
    pub k_cos_total: f64,
    pub k_cos_min_token: f64,
    pub k_cos_mean_token: f64,
    pub k_max_abs_err: f64,
    pub k_max_abs_err_over_scale: f64,
    pub k_saturated: usize,
    pub k_samples: usize,
    pub v_cos_total: f64,
    pub v_cos_min_token: f64,
    pub v_cos_mean_token: f64,
    pub v_max_abs_err: f64,
    pub v_max_abs_err_over_scale: f64,
    pub v_saturated: usize,
    pub v_samples: usize,
}

fn cosine(dot: f64, a_sq: f64, b_sq: f64) -> f64 {
    if a_sq == 0.0 && b_sq == 0.0 {
        return 1.0;
    }
    if a_sq == 0.0 || b_sq == 0.0 {
        return 0.0;
    }
    dot / (a_sq.sqrt() * b_sq.sqrt())
}

/// Compare the source values with the decoded image.
pub fn measure(values: &[f32], image: &KvImage) -> Result<KvQuality> {
    let params = image.params;
    let valid = params.valid_tokens();
    let expected = params.heads * valid * HEAD_DIM;
    if values.len() != expected {
        return Err(PackError::QualityShape(format!(
            "{} source values for {} heads x {valid} tokens x {HEAD_DIM}",
            values.len(),
            params.heads
        )));
    }

    let mut k_dot = 0.0f64;
    let mut k_src_sq = 0.0f64;
    let mut k_dec_sq = 0.0f64;
    let mut k_err = 0.0f64;
    let mut k_err_over_scale = 0.0f64;
    let mut k_saturated = 0usize;
    let mut k_cos: Vec<f64> = Vec::with_capacity(params.heads * valid);

    let mut v_dot = 0.0f64;
    let mut v_src_sq = 0.0f64;
    let mut v_dec_sq = 0.0f64;
    let mut v_err = 0.0f64;
    let mut v_err_over_scale = 0.0f64;
    let mut v_saturated = 0usize;
    let mut v_cos: Vec<f64> = Vec::with_capacity(params.heads * valid);

    for head in 0..params.heads {
        for slot in 0..valid {
            let row =
                &values[(head * valid + slot) * HEAD_DIM..(head * valid + slot + 1) * HEAD_DIM];
            let k_codes = image.decode_k_codes(head, slot);
            let k_scale = image.k_scale(head, slot);
            let v_codes = image.decode_v_codes(head, slot);

            let (mut td, mut ts, mut ds, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for dim in 0..HEAD_DIM {
                let src = f64::from(row[dim]);
                let dec = f64::from(k_codes[dim]) * f64::from(k_scale);
                td += src * dec;
                ts += src * src;
                ds += dec * dec;
                err = err.max((src - dec).abs());
                if k_codes[dim].unsigned_abs() == 7 {
                    k_saturated += 1;
                }
            }
            k_dot += td;
            k_src_sq += ts;
            k_dec_sq += ds;
            k_err = k_err.max(err);
            if k_scale > 0.0 {
                k_err_over_scale = k_err_over_scale.max(err / f64::from(k_scale));
            }
            k_cos.push(cosine(td, ts, ds));

            let (mut td, mut ts, mut ds, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for dim in 0..HEAD_DIM {
                let scale = image.v_scale(head, dim);
                let src = f64::from(row[dim]);
                let dec = f64::from(v_codes[dim]) * f64::from(scale);
                td += src * dec;
                ts += src * src;
                ds += dec * dec;
                err = err.max((src - dec).abs());
                if v_codes[dim].unsigned_abs() == 127 {
                    v_saturated += 1;
                }
                if scale > 0.0 {
                    v_err_over_scale = v_err_over_scale.max((src - dec).abs() / f64::from(scale));
                }
            }
            v_dot += td;
            v_src_sq += ts;
            v_dec_sq += ds;
            v_err = v_err.max(err);
            v_cos.push(cosine(td, ts, ds));
        }
    }

    let k_vectors = k_cos.len().max(1);
    let v_vectors = v_cos.len().max(1);
    Ok(KvQuality {
        k_cos_total: cosine(k_dot, k_src_sq, k_dec_sq),
        k_cos_min_token: k_cos.iter().copied().fold(1.0, f64::min),
        k_cos_mean_token: k_cos.iter().sum::<f64>() / k_vectors as f64,
        k_max_abs_err: k_err,
        k_max_abs_err_over_scale: k_err_over_scale,
        k_saturated,
        k_samples: params.heads * valid * HEAD_DIM,
        v_cos_total: cosine(v_dot, v_src_sq, v_dec_sq),
        v_cos_min_token: v_cos.iter().copied().fold(1.0, f64::min),
        v_cos_mean_token: v_cos.iter().sum::<f64>() / v_vectors as f64,
        v_max_abs_err: v_err,
        v_max_abs_err_over_scale: v_err_over_scale,
        v_saturated,
        v_samples: params.heads * valid * HEAD_DIM,
    })
}

/// One section of the image: sizes and strides for building a view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSection {
    /// Bytes of the image itself.
    pub bytes: usize,
    /// Bytes the runtime must register as the view.
    ///
    /// Equal to `bytes` except for the token-major K image: the kernel derives
    /// its head stride as `wbytes / (ATTN_KV * 2)`, so that view has to be
    /// `heads * slots * 128` bytes (twice the data), exactly as `attn_main.c`
    /// allocates it.
    pub view_bytes: usize,
    pub bytes_per_head: usize,
    /// Stride between heads (equals `bytes_per_head`).
    pub head_stride: usize,
    /// Stride between pages inside one head (0 for the row layout).
    pub page_stride: usize,
    /// Stride between tokens/slots inside one head.
    pub token_stride: usize,
    pub encoding: String,
}

/// Fields the runtime needs to emit the `ATTN` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCommand {
    pub op: String,
    pub keys_arg: String,
    pub key_scales_arg: String,
    pub values_arg: String,
    pub value_scales_arg: String,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_tokens: usize,
    /// Rule for the host to apply the V scale after attention.
    pub out_scale_rule: String,
}

/// JSON description of the K/KT/V images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvManifest {
    pub format: String,
    pub format_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub heads: usize,
    pub pages: usize,
    pub slots: usize,
    pub tokens: usize,
    pub tokens_in_last_page: usize,
    pub head_dim: usize,
    pub page_tokens: usize,
    pub layout: String,
    pub k: KvSection,
    pub kt: KvSection,
    pub v: KvSection,
    pub v_scale_mode: String,
    pub k_scale_encoding: String,
    pub v_scale_encoding: String,
    /// `kt` per head and slot, Q16.
    pub k_scales_q16: Vec<i32>,
    /// One value for the global mode, `heads * head_dim` for the channel mode.
    pub v_scales_f32: Vec<f32>,
    pub k_fnv1a64: String,
    pub kt_fnv1a64: String,
    pub v_fnv1a64: String,
    pub command: KvCommand,
}

impl KvManifest {
    /// Describe `image`.
    pub fn new(image: &KvImage, source: Option<SourceInfo>) -> Self {
        let params = image.params;
        let k_page = match params.layout {
            KvLayout::Row => 0,
            KvLayout::Page => 16 * 128,
        };
        let v_page = match params.layout {
            KvLayout::Row => 0,
            KvLayout::Page => 4096,
        };
        let out_scale_rule = match params.v_scale_mode {
            VScaleMode::Global => {
                "multiply every attention output by out_scale * v_scales_f32[0]".to_string()
            }
            VScaleMode::Channel => {
                "multiply out[head * 128 + d] by out_scale * v_scales_f32[head * 128 + d]"
                    .to_string()
            }
        };
        Self {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            source,
            heads: params.heads,
            pages: params.pages,
            slots: params.slots(),
            tokens: params.valid_tokens(),
            tokens_in_last_page: params.tokens_in_last_page,
            head_dim: HEAD_DIM,
            page_tokens: PAGE_TOKENS,
            layout: params.layout.name().to_string(),
            k: KvSection {
                bytes: image.k.len(),
                view_bytes: match params.layout {
                    KvLayout::Row => image.k.len(),
                    // token-major: head stride = wbytes / (ATTN_KV * 2)
                    KvLayout::Page => params.heads * params.slots() * HEAD_DIM,
                },
                bytes_per_head: params.k_bytes_per_head(),
                head_stride: params.k_bytes_per_head(),
                page_stride: k_page,
                token_stride: HEAD_DIM,
                encoding: match params.layout {
                    KvLayout::Row => "int4 offset-8, 128 B/token: 64 B nibbles (low = even dim) then 64 B zero pad".to_string(),
                    KvLayout::Page => "int4 offset-8, 2048 B/page: [g<16][4*token + j], byte = code(8g+2j) | code(8g+2j+1) << 4; register view_bytes, the kernel halves it for the head stride".to_string(),
                },
            },
            kt: KvSection {
                bytes: image.kt.len(),
                view_bytes: image.kt.len(),
                bytes_per_head: params.kt_bytes_per_head(),
                head_stride: params.kt_bytes_per_head(),
                page_stride: 0,
                token_stride: 4,
                encoding: "int32 little endian Q16 per token; k_real = code * kt / 65536".to_string(),
            },
            v: KvSection {
                bytes: image.v.len(),
                view_bytes: image.v.len(),
                bytes_per_head: params.v_bytes_per_head(),
                head_stride: params.v_bytes_per_head(),
                page_stride: v_page,
                token_stride: HEAD_DIM,
                encoding: match params.layout {
                    KvLayout::Row => "int8 per token, 128 B".to_string(),
                    KvLayout::Page => "int8, 4096 B/page: byte at g*1024 + c*128 + 4*d + j = V[32g + d][4c + j]".to_string(),
                },
            },
            v_scale_mode: params.v_scale_mode.name().to_string(),
            k_scale_encoding: "per token, Q16 i32".to_string(),
            v_scale_encoding: "constant over tokens (required: PV mixes tokens)".to_string(),
            k_scales_q16: image.k_scales_q16.clone(),
            v_scales_f32: image.v_scales.clone(),
            k_fnv1a64: format!("{:016x}", fnv1a64(&image.k)),
            kt_fnv1a64: format!("{:016x}", fnv1a64(&image.kt)),
            v_fnv1a64: format!("{:016x}", fnv1a64(&image.v)),
            command: KvCommand {
                op: "ATTN".to_string(),
                keys_arg: "h0".to_string(),
                key_scales_arg: "h1".to_string(),
                values_arg: "h2".to_string(),
                value_scales_arg: "h3".to_string(),
                query_heads: DSP_QUERY_HEADS,
                kv_heads: params.heads,
                head_dim: HEAD_DIM,
                page_tokens: PAGE_TOKENS,
                out_scale_rule,
            },
        }
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Reconstruct the parameters recorded in the manifest.
    pub fn params(&self) -> Result<KvPackParams> {
        Ok(KvPackParams {
            heads: self.heads,
            pages: self.pages,
            tokens_in_last_page: self.tokens_in_last_page,
            layout: KvLayout::parse(&self.layout)?,
            v_scale_mode: VScaleMode::parse(&self.v_scale_mode)?,
        })
    }

    /// Check the three images against the manifest and catch silent corruption.
    pub fn verify_images(&self, k: &[u8], kt: &[u8], v: &[u8]) -> Result<KvImage> {
        if self.format != FORMAT_NAME {
            return Err(PackError::ManifestMismatch(format!(
                "format `{}` != `{FORMAT_NAME}`",
                self.format
            )));
        }
        if self.format_version != FORMAT_VERSION {
            return Err(PackError::ManifestMismatch(format!(
                "format_version {} != {FORMAT_VERSION}",
                self.format_version
            )));
        }
        if self.head_dim != HEAD_DIM {
            return Err(PackError::ManifestMismatch(format!(
                "head_dim {} != {HEAD_DIM}",
                self.head_dim
            )));
        }
        if self.page_tokens != PAGE_TOKENS {
            return Err(PackError::ManifestMismatch(format!(
                "page_tokens {} != {PAGE_TOKENS}",
                self.page_tokens
            )));
        }
        let params = self.params()?;
        params.validate()?;
        if self.slots != params.slots() || self.tokens != params.valid_tokens() {
            return Err(PackError::ManifestMismatch(
                "slot/token counts do not match pages and the last page".to_string(),
            ));
        }
        let k_page_stride = match params.layout {
            KvLayout::Row => 0,
            KvLayout::Page => 16 * 128,
        };
        let v_page_stride = match params.layout {
            KvLayout::Row => 0,
            KvLayout::Page => 4096,
        };
        if self.k.bytes_per_head != params.k_bytes_per_head()
            || self.v.bytes_per_head != params.v_bytes_per_head()
            || self.kt.bytes_per_head != params.kt_bytes_per_head()
            || self.k.page_stride != k_page_stride
            || self.v.page_stride != v_page_stride
            || self.k.token_stride != HEAD_DIM
            || self.v.token_stride != HEAD_DIM
            || self.kt.token_stride != 4
        {
            return Err(PackError::ManifestMismatch(
                "section strides do not match the recorded layout".to_string(),
            ));
        }
        let expected_start = |section: &KvSection| -> usize {
            match params.layout {
                KvLayout::Row => section.bytes,
                KvLayout::Page => params.heads * params.slots() * HEAD_DIM,
            }
        };
        if self.k.view_bytes != expected_start(&self.k)
            || self.v.view_bytes != self.v.bytes
            || self.kt.view_bytes != self.kt.bytes
            || self.k.view_bytes < self.k.bytes
        {
            return Err(PackError::ManifestMismatch(
                "view sizes do not match the layout".to_string(),
            ));
        }
        for (name, section, bytes) in [("k", &self.k, k), ("kt", &self.kt, kt), ("v", &self.v, v)] {
            if section.bytes != bytes.len()
                || section.bytes != params.heads * section.bytes_per_head
                || section.head_stride != section.bytes_per_head
            {
                return Err(PackError::ManifestMismatch(format!(
                    "{name}: section sizes do not match {} heads x {} bytes",
                    params.heads, section.bytes_per_head
                )));
            }
        }
        for (name, recorded, actual) in [
            ("k", &self.k_fnv1a64, fnv1a64(k)),
            ("kt", &self.kt_fnv1a64, fnv1a64(kt)),
            ("v", &self.v_fnv1a64, fnv1a64(v)),
        ] {
            let actual = format!("{actual:016x}");
            if *recorded != actual {
                return Err(PackError::ManifestMismatch(format!(
                    "{name} checksum {recorded} != {actual}"
                )));
            }
        }
        let expected_scales = match params.v_scale_mode {
            VScaleMode::Global => 1,
            VScaleMode::Channel => params.heads * HEAD_DIM,
        };
        if self.v_scales_f32.len() != expected_scales {
            return Err(PackError::ManifestMismatch(format!(
                "{} V scales for mode {} and {} heads",
                self.v_scales_f32.len(),
                params.v_scale_mode.name(),
                params.heads
            )));
        }
        if self.k_scales_q16.len() != params.heads * params.slots() {
            return Err(PackError::ManifestMismatch(
                "K scale table does not cover every head and slot".to_string(),
            ));
        }
        let mut kt_from_image = Vec::with_capacity(self.k_scales_q16.len());
        for chunk in kt.chunks_exact(4) {
            kt_from_image.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if kt_from_image != self.k_scales_q16 {
            return Err(PackError::ManifestMismatch(
                "K scale table does not match the image".to_string(),
            ));
        }
        for (index, &value) in self.k_scales_q16.iter().enumerate() {
            let slot = index % params.slots();
            let valid = slot < params.valid_tokens();
            if !(0..=MAX_KT_Q16).contains(&value) {
                return Err(PackError::ManifestMismatch(format!(
                    "kt[{index}] = {value} is outside 0..={MAX_KT_Q16}"
                )));
            }
            if !valid && value != 0 {
                return Err(PackError::ManifestMismatch(format!(
                    "padding slot {index} has a non-zero K scale"
                )));
            }
        }
        for &scale in &self.v_scales_f32 {
            if !scale.is_finite() || scale < 0.0 {
                return Err(PackError::ManifestMismatch(format!(
                    "V scale {scale} is not finite and non-negative"
                )));
            }
        }

        let image = KvImage {
            params,
            k: k.to_vec(),
            kt: kt.to_vec(),
            v: v.to_vec(),
            k_scales_q16: self.k_scales_q16.clone(),
            v_scales: self.v_scales_f32.clone(),
            k_max_abs: Vec::new(),
            v_max_abs: Vec::new(),
        };
        image.verify_structure()?;
        Ok(image)
    }
}

impl KvImage {
    /// Structural check of the stored bytes: padding slots are zero, nibbles
    /// are in `1..=15` for real tokens (code 0 would mean q = -8, which this
    /// packer never emits).
    pub fn verify_structure(&self) -> Result<()> {
        let params = self.params;
        let valid = params.valid_tokens();
        let slots = params.slots();
        for head in 0..params.heads {
            for slot in 0..slots {
                let k_codes = self.decode_k_codes(head, slot);
                let v_codes = self.decode_v_codes(head, slot);
                if slot >= valid {
                    // Padding slots are written as zero bytes. The decoder maps
                    // an all-zero int4 byte to -8 and an all-zero int8 byte to 0.
                    if k_codes.iter().any(|&code| code != -8) {
                        return Err(PackError::KvPage(format!(
                            "padding slot {slot} of head {head} has non-zero K bytes"
                        )));
                    }
                    if v_codes.iter().any(|&code| code != 0) {
                        return Err(PackError::KvPage(format!(
                            "padding slot {slot} of head {head} has non-zero V bytes"
                        )));
                    }
                    continue;
                }
                if k_codes.iter().any(|&code| !(-7..=7).contains(&code)) {
                    return Err(PackError::KvPage(format!(
                        "slot {slot} of head {head} has an out-of-range K code"
                    )));
                }
                if self.k_scale(head, slot) == 0.0 && k_codes.iter().any(|&code| code != 0) {
                    return Err(PackError::KvPage(format!(
                        "slot {slot} of head {head} has codes but a zero K scale"
                    )));
                }
                if v_codes.contains(&i8::MIN) {
                    return Err(PackError::KvPage(format!(
                        "slot {slot} of head {head} has an out-of-range V code"
                    )));
                }
            }
        }
        if params.layout == KvLayout::Row {
            let stride = params.k_bytes_per_head();
            for head in 0..params.heads {
                for slot in 0..slots {
                    let base = head * stride + slot * HEAD_DIM;
                    if self.k[base + HEAD_DIM / 2..base + HEAD_DIM]
                        .iter()
                        .any(|byte| *byte != 0)
                    {
                        return Err(PackError::KvPage(format!(
                            "slot {slot} of head {head} has a non-zero K pad"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}
