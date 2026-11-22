//! On-disk fixture for the device check of one conv and one attention layer.
//!
//! The host example (`examples/make_layer_fixture.rs`) writes the blobs and
//! this manifest; the Android binary (`src/bin/dsp-layer-check.rs`) reads them,
//! stages the weights on the DSP and compares one conv layer and one attention
//! layer against the CPU oracle that produced the expected f32 outputs.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::TransportError;

/// One scalar rope configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureRope {
    /// `"full"` or `"partial"`.
    pub kind: String,
    pub theta: f32,
    pub dim: usize,
    /// Rotated pairs for a partial rope (`0` for full).
    pub rotated_dim: usize,
}

/// Model dimensions needed to rebuild `ModelSpec` on the device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureDims {
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub conv_cache: usize,
    pub sliding_window: usize,
}

/// Scales of the conv chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvFixture {
    pub act_scale: f32,
    pub mix_scale: f32,
    pub y_scale: f32,
    pub out_scale: f32,
    pub ksh: i32,
    pub state_off: i32,
}

/// Scales of the attention chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFixture {
    pub act_scale: f32,
    pub q_scale: f32,
    pub k_scale: f32,
    pub v_scale: f32,
    pub gate_scale: f32,
    pub attn_q_scale: f32,
    pub attn_out_scale: f32,
    pub o_act_scale: f32,
    pub out_scale: f32,
    pub v_channel_scales: Vec<f32>,
    pub has_gate: bool,
    pub cache_pages: usize,
}

/// Root of a layer fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFixture {
    pub model: String,
    pub conv_layer: usize,
    pub attention_layer: usize,
    pub dims: FixtureDims,
    pub rope_full: FixtureRope,
    pub rope_sliding: Option<FixtureRope>,
    /// `"full"` or `"sliding"` for the attention layer under test.
    pub attention_kind: String,
    pub norm_eps: f32,
    /// `"conv"` / `"attention"` per layer index, as in `ModelSpec`.
    pub layer_kinds: Vec<String>,
    pub conv_tokens: usize,
    pub attention_tokens: usize,
    pub conv: ConvFixture,
    pub attention: AttentionFixture,
}

impl LayerFixture {
    pub fn read(dir: &Path) -> Result<Self, TransportError> {
        let text = fs::read_to_string(dir.join("fixture.json")).map_err(|_| bad("fixture.json"))?;
        serde_json::from_str(&text).map_err(|_| bad("fixture.json is not valid JSON"))
    }

    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(
            dir.join("fixture.json"),
            serde_json::to_string_pretty(self)?,
        )
    }

    pub fn path(&self, dir: &Path, name: &str) -> PathBuf {
        dir.join(name)
    }
}

fn bad(what: &'static str) -> TransportError {
    TransportError::InvalidLength { what, len: 0 }
}

/// Read a raw little-endian f32 file and check the expected length.
pub fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>, TransportError> {
    let bytes = fs::read(path).map_err(|_| bad("f32 fixture file cannot be read"))?;
    if bytes.len() != expected * 4 {
        return Err(bad("f32 fixture file has the wrong length"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Write raw little-endian f32 values.
pub fn write_f32(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)
}

/// Cosine similarity and maximum absolute difference.
pub fn compare(expected: &[f32], actual: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut a_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    let mut max = 0.0f64;
    for (a, b) in expected.iter().zip(actual) {
        let a = *a as f64;
        let b = *b as f64;
        dot += a * b;
        a_sq += a * a;
        b_sq += b * b;
        let diff = (a - b).abs();
        if diff > max {
            max = diff;
        }
    }
    let cos = if a_sq == 0.0 || b_sq == 0.0 {
        1.0
    } else {
        dot / (a_sq.sqrt() * b_sq.sqrt())
    };
    (cos, max)
}
