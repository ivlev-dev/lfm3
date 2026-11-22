//! Quantised shortconv (depthwise, 7 taps) blob for the DSP `CONV` command.
//!
//! The format is not invented here: it is the byte image `conv_blob_build()`
//! in `myelin/dsp/hvx/exec_main.c` fills and `lfmgemv_conv_init()` in
//! `myelin/dsp/conv/lfmconv_impl.c` builds, i.e. exactly what the `CONV`
//! command of the `exec` list reads in `lfmgemv_impl.c`:
//!
//! ```text
//! [0        .. 64*H)    8 phase-rotated copies of the int8 tap weights
//! [64*H     .. 68*H)    kq[H] int32 little endian: out32*c -> y8 multiplier
//! ```
//!
//! Phase block `p` (`p = ctr & 7`, i.e. the ring slot of the newest sample)
//! is `H/16` vectors of 128 B; vector `g` holds channels `16g .. 16g+15`,
//! 8 bytes per channel, one byte per ring position. Byte `s` of channel `c`
//! holds tap `(s - p - 2) & 7` (tap 7 is the unused ring slot and stays 0).
//!
//! `kq[ch] = trunc(K_ch * 2^(ksh-2) + 0.5)` with
//! `K_ch = max|w| * s_bc^3 / (127 * s_y)`; `ksh` is chosen once for the layer
//! so the largest `K` lands close to `2^28`. Both `kq` and `ksh` depend on the
//! layer's activation scales `s_bc` (in_proj output) and `s_y` (conv output),
//! which are calibration results, so they are inputs of this packer and are
//! recorded in the manifest.
//!
//! The integer recipes are copied operation for operation from
//! `myelin/dsp/conv/conv_math.h`; `tests/c/conv_blob_ref.c` is the verbatim
//! C builder and `tests/shortconv.rs` compares the two byte for byte.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::crownstone::fnv1a64;
use crate::manifest::SourceInfo;
use crate::{PackError, Result};

/// Taps of the depthwise filter.
pub const TAPS: usize = 7;
/// Ring positions per channel (7 taps + one unused).
pub const RING: usize = 8;
/// Channels per 128-byte HVX vector.
pub const GROUP: usize = 16;
/// Largest channel count `lfmgemv_conv_init` accepts.
pub const MAX_CHANNELS: usize = 16384;
/// Stable name written into the manifest.
pub const FORMAT_NAME: &str = "myelin-dsp-shortconv";
/// Manifest format version.
pub const FORMAT_VERSION: u32 = 1;
/// Stand defaults used when the caller does not pass calibration scales.
pub const STAND_S_BC: f32 = 0.01;
pub const STAND_S_Y: f32 = 0.05;

/// Bytes of the rotated weight phases for `h` channels.
pub fn wq8_bytes(h: usize) -> usize {
    RING * h * 8
}

/// Offset of `kq[0]` inside the blob.
pub fn kq_base(h: usize) -> usize {
    wq8_bytes(h)
}

/// Total blob size for `h` channels.
pub fn blob_bytes(h: usize) -> usize {
    wq8_bytes(h) + 4 * h
}

/// Tap carried by ring slot `slot` at ring phase `phase` (7 = unused slot).
///
/// Mirrors `conv_slot_tap()`: `(slot - phase - 2) & 7`.
pub fn slot_tap(phase: usize, slot: usize) -> usize {
    ((slot as isize - phase as isize - 2) & 7) as usize
}

/// Verify the geometry the DSP kernel accepts.
pub fn check_channels(h: usize) -> Result<()> {
    if h < GROUP || h % GROUP != 0 {
        return Err(PackError::Shortconv(format!(
            "channel count {h} must be a positive multiple of {GROUP}"
        )));
    }
    if h > MAX_CHANNELS {
        return Err(PackError::Shortconv(format!(
            "channel count {h} exceeds the DSP limit {MAX_CHANNELS}"
        )));
    }
    Ok(())
}

/// Activation scales the layer was calibrated with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationScales {
    /// Scale of the in_proj/int8 `bcx` buffer.
    pub s_bc: f32,
    /// Scale of the int8 conv output `y`.
    pub s_y: f32,
    /// Per-channel `y` scales. When present, `s_y` is only a fallback and the
    /// out projection must fold the same table into its columns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s_y_channels: Option<Vec<f32>>,
    /// Where the values came from: `"cli"` or `"stand_default"`.
    pub source: String,
}

impl ActivationScales {
    /// Scales supplied by the caller.
    pub fn cli(s_bc: f32, s_y: f32) -> Self {
        Self {
            s_bc,
            s_y,
            s_y_channels: None,
            source: "cli".to_string(),
        }
    }

    /// Per-channel `y` scales from a per-channel calibration.
    pub fn channels(s_bc: f32, s_y_channels: Vec<f32>) -> Self {
        Self {
            s_bc,
            s_y: 1.0,
            s_y_channels: Some(s_y_channels),
            source: "cli".to_string(),
        }
    }

    /// The stand values `0.01 / 0.05` (`conv_main.c`, `exec_main.c`).
    pub fn stand_default() -> Self {
        Self {
            s_bc: STAND_S_BC,
            s_y: STAND_S_Y,
            s_y_channels: None,
            source: "stand_default".to_string(),
        }
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [("s_bc", self.s_bc), ("s_y", self.s_y)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PackError::Shortconv(format!(
                    "{name} must be finite and positive, got {value}"
                )));
            }
        }
        Ok(())
    }
}

/// A packed shortconv blob plus the tables it was built from.
#[derive(Debug, Clone)]
pub struct ConvBlob {
    pub h: usize,
    pub scales: ActivationScales,
    /// One shared right shift for `kq`; goes into the `CONV` command (`a1`).
    pub ksh: i32,
    pub blob: Vec<u8>,
    /// Per-channel quantisation scale `max|w|/127` of the 7 taps.
    pub tap_scale: Vec<f32>,
    /// Per-channel `max|w|` of the 7 taps.
    pub tap_max_abs: Vec<f32>,
    /// Unrotated tap codes, `h * 7`.
    pub tap_codes: Vec<i8>,
    /// Per-channel `kq`, little-endian words at [`kq_base`].
    pub kq: Vec<i32>,
}

/// `m` of one channel: the largest `|w|`, or `1.0` for an all-zero channel,
/// exactly as the DSP setup does it (`if (!(m > 0)) m = 1.0f`).
fn channel_max_abs(row: &[f32]) -> f32 {
    let mut max_abs = 0.0f32;
    for &w in row {
        let a = if w < 0.0 { -w } else { w };
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs > 0.0 {
        max_abs
    } else {
        1.0
    }
}

/// Sign-symmetric round-half-away-from-zero int8 code (`conv_wq`).
fn quantize_tap(w: f32, scale: f32) -> i8 {
    let t = w / scale;
    let q = (t + if t >= 0.0 { 0.5 } else { -0.5 }) as i32;
    q.clamp(-127, 127) as i8
}

/// `conv_K`: real value of one unit of `out32 * c8`.
fn conv_k(max_abs: f64, s_bc: f64, s_y: f64) -> f64 {
    max_abs * s_bc * s_bc * s_bc / (127.0 * s_y)
}

/// `conv_pick_ksh`: shift that lifts the largest `K` close to `2^28`.
fn pick_ksh(kmax: f64) -> i32 {
    if kmax.is_nan() || kmax <= 0.0 {
        return 16;
    }
    let mut ksh = 2;
    let mut sc = 4.0f64;
    while ksh < 40 && kmax * sc * 2.0 < 3.0e8 {
        sc *= 2.0;
        ksh += 1;
    }
    ksh
}

/// `conv_make_kq`: Q(ksh) integer multiplier, truncating a positive value.
fn make_kq(k: f64, ksh: i32) -> i32 {
    let mut sc = 4.0f64;
    let mut i = 2;
    while i < ksh {
        sc *= 2.0;
        i += 1;
    }
    let mut value = k * sc + 0.5;
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    if value > 2_147_483_000.0 {
        value = 2_147_483_000.0;
    }
    value as i32
}

/// Build the 8 phase-rotated weight planes from unrotated tap codes.
fn build_wq8(codes: &[i8], h: usize) -> Vec<u8> {
    let ng = h / GROUP;
    let mut out = vec![0u8; wq8_bytes(h)];
    for phase in 0..RING {
        for g in 0..ng {
            for j in 0..GROUP {
                let channel = g * GROUP + j;
                let base = (phase * ng + g) * 128 + j * 8;
                for slot in 0..RING {
                    let tap = slot_tap(phase, slot);
                    out[base + slot] = if tap < TAPS {
                        codes[channel * TAPS + tap] as u8
                    } else {
                        0
                    };
                }
            }
        }
    }
    out
}

/// Pack `h * 7` f32 taps into the DSP shortconv blob.
pub fn build(taps: &[f32], h: usize, scales: ActivationScales) -> Result<ConvBlob> {
    check_channels(h)?;
    scales.validate()?;
    if taps.len() != h * TAPS {
        return Err(PackError::Shortconv(format!(
            "{} taps for {h} channels, expected {}",
            taps.len(),
            h * TAPS
        )));
    }
    for (index, &value) in taps.iter().enumerate() {
        if !value.is_finite() {
            return Err(PackError::NonFiniteWeight {
                row: index / TAPS,
                col: index % TAPS,
                value,
            });
        }
    }

    let mut tap_scale = vec![0.0f32; h];
    let mut tap_max_abs = vec![0.0f32; h];
    let mut tap_codes = vec![0i8; h * TAPS];
    let mut kmax = 0.0f64;

    for channel in 0..h {
        let row = &taps[channel * TAPS..(channel + 1) * TAPS];
        let max_abs = channel_max_abs(row);
        let scale = max_abs / 127.0;
        tap_scale[channel] = scale;
        tap_max_abs[channel] = max_abs;
        for (tap, &w) in row.iter().enumerate() {
            tap_codes[channel * TAPS + tap] = quantize_tap(w, scale);
        }
        let s_y = scales
            .s_y_channels
            .as_ref()
            .map_or(scales.s_y, |v| v[channel]);
        let k = conv_k(f64::from(max_abs), f64::from(scales.s_bc), f64::from(s_y));
        if k > kmax {
            kmax = k;
        }
    }

    let ksh = pick_ksh(kmax);
    let mut kq = Vec::with_capacity(h);
    for (channel, &max_abs) in tap_max_abs.iter().enumerate() {
        let s_y = scales
            .s_y_channels
            .as_ref()
            .map_or(scales.s_y, |v| v[channel]);
        kq.push(make_kq(
            conv_k(f64::from(max_abs), f64::from(scales.s_bc), f64::from(s_y)),
            ksh,
        ));
    }

    let mut blob = build_wq8(&tap_codes, h);
    blob.resize(blob_bytes(h), 0);
    let base = kq_base(h);
    for (channel, &value) in kq.iter().enumerate() {
        blob[base + 4 * channel..base + 4 * channel + 4].copy_from_slice(&value.to_le_bytes());
    }

    Ok(ConvBlob {
        h,
        scales,
        ksh,
        blob,
        tap_scale,
        tap_max_abs,
        tap_codes,
        kq,
    })
}

/// Decode the unrotated tap codes from phase 0 of a blob.
///
/// Phase 0 carries every tap: tap `t` sits in ring slot `(t + 2) & 7`.
pub fn decode_tap_codes(blob: &[u8], h: usize) -> Result<Vec<i8>> {
    check_blob_len(blob, h)?;
    let mut codes = vec![0i8; h * TAPS];
    for channel in 0..h {
        for tap in 0..TAPS {
            let slot = (tap + 2) & 7;
            let offset = (channel / GROUP) * 128 + (channel % GROUP) * 8 + slot;
            codes[channel * TAPS + tap] = blob[offset] as i8;
        }
    }
    Ok(codes)
}

/// Decode the per-channel `kq` table.
pub fn decode_kq(blob: &[u8], h: usize) -> Result<Vec<i32>> {
    check_blob_len(blob, h)?;
    let base = kq_base(h);
    let mut kq = Vec::with_capacity(h);
    for channel in 0..h {
        let word = &blob[base + 4 * channel..base + 4 * channel + 4];
        kq.push(i32::from_le_bytes([word[0], word[1], word[2], word[3]]));
    }
    Ok(kq)
}

fn check_blob_len(blob: &[u8], h: usize) -> Result<()> {
    check_channels(h)?;
    let expected = blob_bytes(h);
    if blob.len() != expected {
        return Err(PackError::BlobTooSmall {
            expected,
            actual: blob.len(),
        });
    }
    Ok(())
}

/// Structural check: every phase is the rotation of the same tap codes and the
/// `kq` words are non-negative. This catches truncated or hand-edited blobs.
pub fn verify_structure(blob: &[u8], h: usize) -> Result<()> {
    let codes = decode_tap_codes(blob, h)?;
    let expected = build_wq8(&codes, h);
    if blob[..wq8_bytes(h)] != expected {
        return Err(PackError::Shortconv(
            "weight phases are not a rotation of one tap table".to_string(),
        ));
    }
    for (channel, &value) in decode_kq(blob, h)?.iter().enumerate() {
        if value < 0 {
            return Err(PackError::Shortconv(format!(
                "kq[{channel}] = {value} is negative"
            )));
        }
    }
    Ok(())
}

/// Per-channel quality of the tap quantisation.
#[derive(Debug, Clone, Serialize)]
pub struct ConvChannelStats {
    pub channel: usize,
    pub scale: f32,
    pub max_abs: f32,
    pub cos: f64,
    pub max_abs_err: f64,
    pub max_abs_err_over_scale: f64,
    pub saturated: usize,
}

/// Quality of the whole shortconv blob.
#[derive(Debug, Clone, Serialize)]
pub struct ConvQuality {
    pub cos_total: f64,
    pub cos_min_channel: f64,
    pub cos_mean_channel: f64,
    pub max_abs_err: f64,
    pub max_abs_err_over_scale: f64,
    pub saturated: usize,
    pub saturated_fraction: f64,
    pub samples: usize,
    pub per_channel: Vec<ConvChannelStats>,
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

/// Compare the f32 taps with the codes stored in `blob`.
pub fn measure(taps: &[f32], h: usize, blob: &[u8]) -> Result<ConvQuality> {
    if taps.len() != h * TAPS {
        return Err(PackError::QualityShape(format!(
            "{} taps for {h} channels",
            taps.len()
        )));
    }
    let codes = decode_tap_codes(blob, h)?;

    let mut per_channel = Vec::with_capacity(h);
    let mut global_dot = 0.0f64;
    let mut global_original_sq = 0.0f64;
    let mut global_decoded_sq = 0.0f64;
    let mut max_abs_err = 0.0f64;
    let mut max_abs_err_over_scale = 0.0f64;
    let mut saturated = 0usize;

    for channel in 0..h {
        let row = &taps[channel * TAPS..(channel + 1) * TAPS];
        let scale = channel_max_abs(row) / 127.0;
        let mut dot = 0.0f64;
        let mut original_sq = 0.0f64;
        let mut decoded_sq = 0.0f64;
        let mut channel_err = 0.0f64;
        let mut channel_saturated = 0usize;
        for (tap, &w) in row.iter().enumerate() {
            let code = codes[channel * TAPS + tap];
            let decoded = f64::from(code) * f64::from(scale);
            let original = f64::from(w);
            dot += original * decoded;
            original_sq += original * original;
            decoded_sq += decoded * decoded;
            let err = (original - decoded).abs();
            if err > channel_err {
                channel_err = err;
            }
            if code.unsigned_abs() == 127 {
                channel_saturated += 1;
            }
        }
        let err_over_scale = if scale > 0.0 {
            channel_err / f64::from(scale)
        } else {
            0.0
        };
        max_abs_err = max_abs_err.max(channel_err);
        max_abs_err_over_scale = max_abs_err_over_scale.max(err_over_scale);
        saturated += channel_saturated;
        global_dot += dot;
        global_original_sq += original_sq;
        global_decoded_sq += decoded_sq;
        let max_abs = row.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
        per_channel.push(ConvChannelStats {
            channel,
            scale,
            max_abs,
            cos: cosine(dot, original_sq, decoded_sq),
            max_abs_err: channel_err,
            max_abs_err_over_scale: err_over_scale,
            saturated: channel_saturated,
        });
    }

    let samples = h * TAPS;
    let cos_min_channel = per_channel
        .iter()
        .map(|channel| channel.cos)
        .fold(1.0f64, f64::min);
    let cos_mean_channel = per_channel.iter().map(|channel| channel.cos).sum::<f64>() / h as f64;

    Ok(ConvQuality {
        cos_total: cosine(global_dot, global_original_sq, global_decoded_sq),
        cos_min_channel,
        cos_mean_channel,
        max_abs_err,
        max_abs_err_over_scale,
        saturated,
        saturated_fraction: saturated as f64 / samples as f64,
        samples,
        per_channel,
    })
}

/// Fields the runtime needs to emit the `CONV` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvCommand {
    /// Command name in the `exec` list.
    pub op: String,
    /// Which command field carries the blob handle.
    pub blob_arg: String,
    pub h: usize,
    pub ksh: i32,
    /// `src_len`: `bcx` is `3 * h` int8.
    pub src_len: usize,
    /// `dst_len`: `y` is `h` int8.
    pub dst_len: usize,
    /// Ring state per slot in DSP scratch (`hi` plane then `lo` plane).
    pub ring_state_bytes_per_slot: usize,
    pub ring_slots: usize,
}

impl ConvCommand {
    fn new(h: usize, ksh: i32) -> Self {
        Self {
            op: "CONV".to_string(),
            blob_arg: "h0".to_string(),
            h,
            ksh,
            src_len: 3 * h,
            dst_len: h,
            ring_state_bytes_per_slot: 2 * h * 8,
            ring_slots: 8,
        }
    }
}

/// JSON description of a shortconv blob: sizes, offsets, scales, checksum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvManifest {
    pub format: String,
    pub format_version: u32,
    pub tensor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub h: usize,
    pub taps: usize,
    pub ring: usize,
    pub group: usize,
    pub wq8_base: usize,
    pub wq8_bytes: usize,
    pub kq_base: usize,
    pub kq_bytes: usize,
    pub blob_bytes: usize,
    pub blob_fnv1a64: String,
    pub ksh: i32,
    pub activation_scales: ActivationScales,
    pub wq8_layout: String,
    pub kq_layout: String,
    pub command: ConvCommand,
    pub tap_scales_f32: Vec<f32>,
    pub tap_max_abs: Vec<f32>,
    pub kq_q16: Vec<i32>,
}

impl ConvManifest {
    /// Describe `blob`.
    pub fn new(blob: &ConvBlob, tensor: String, source: Option<SourceInfo>) -> Self {
        Self {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            tensor,
            source,
            h: blob.h,
            taps: TAPS,
            ring: RING,
            group: GROUP,
            wq8_base: 0,
            wq8_bytes: wq8_bytes(blob.h),
            kq_base: kq_base(blob.h),
            kq_bytes: 4 * blob.h,
            blob_bytes: blob.blob.len(),
            blob_fnv1a64: format!("{:016x}", fnv1a64(&blob.blob)),
            ksh: blob.ksh,
            activation_scales: blob.scales.clone(),
            wq8_layout: "phase-major: [phase][H/16][16 channels][8 ring slots], tap = (slot - phase - 2) & 7, tap 7 unused and zero".to_string(),
            kq_layout: "int32 little endian per channel, Q(ksh): y8 = clamp(((out32>>1)*c8*kq[ch] + 2^(ksh-2)) >> (ksh-1))".to_string(),
            command: ConvCommand::new(blob.h, blob.ksh),
            tap_scales_f32: blob.tap_scale.clone(),
            tap_max_abs: blob.tap_max_abs.clone(),
            kq_q16: blob.kq.clone(),
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

    /// Check that the manifest really describes `blob`.
    pub fn verify_blob(&self, blob: &[u8]) -> Result<()> {
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
        check_channels(self.h)?;
        for (name, manifest_value, expected) in [
            ("taps", self.taps, TAPS),
            ("ring", self.ring, RING),
            ("group", self.group, GROUP),
            ("wq8_base", self.wq8_base, 0),
            ("wq8_bytes", self.wq8_bytes, wq8_bytes(self.h)),
            ("kq_base", self.kq_base, kq_base(self.h)),
            ("kq_bytes", self.kq_bytes, 4 * self.h),
            ("blob_bytes", self.blob_bytes, blob_bytes(self.h)),
            ("blob_bytes", self.blob_bytes, blob.len()),
            ("command.h", self.command.h, self.h),
        ] {
            if manifest_value != expected {
                return Err(PackError::ManifestMismatch(format!(
                    "{name}: manifest {manifest_value} != {expected}"
                )));
            }
        }
        if self.command.ksh != self.ksh || self.command.op != "CONV" {
            return Err(PackError::ManifestMismatch(
                "command fields do not match ksh/op".to_string(),
            ));
        }
        if !(1..=40).contains(&self.ksh) {
            return Err(PackError::ManifestMismatch(format!(
                "ksh {} is outside the DSP range 1..=40",
                self.ksh
            )));
        }
        self.activation_scales.validate()?;
        if self.tap_scales_f32.len() != self.h
            || self.tap_max_abs.len() != self.h
            || self.kq_q16.len() != self.h
        {
            return Err(PackError::ManifestMismatch(
                "scale/kq table length does not match h".to_string(),
            ));
        }
        let checksum = format!("{:016x}", fnv1a64(blob));
        if self.blob_fnv1a64 != checksum {
            return Err(PackError::ManifestMismatch(format!(
                "blob checksum {} != {checksum}",
                self.blob_fnv1a64
            )));
        }
        verify_structure(blob, self.h)?;
        if decode_kq(blob, self.h)? != self.kq_q16 {
            return Err(PackError::ManifestMismatch(
                "kq table does not match the blob".to_string(),
            ));
        }
        Ok(())
    }
}
