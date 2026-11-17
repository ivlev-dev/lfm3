use anyhow::{anyhow, Context, Result};
use lfm_domain::{LayerKind, ModelConfig};
use lfm_gguf::{GgmlType, GgufFile};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub struct Tensor {
    pub ne0: usize,
    pub ne1: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.ne0..(i + 1) * self.ne0]
    }
}

pub struct Lfm3Ref {
    pub cfg: ModelConfig,
    w4a8: bool,
    pub tensors: HashMap<String, Tensor>,
    rope_full: Vec<f32>,
    rope_sliding: Vec<f32>,
}

pub struct LayerState {
    pub conv: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

pub struct State {
    pub layers: Vec<LayerState>,
    pub len: usize,
}

impl State {
    pub fn new(cfg: &ModelConfig) -> Self {
        let layers = (0..cfg.num_layers)
            .map(|i| match cfg.kind(i) {
                LayerKind::Conv => LayerState {
                    conv: vec![0.0; (cfg.conv_cache - 1) * cfg.hidden_size],
                    k: Vec::new(),
                    v: Vec::new(),
                },
                _ => LayerState {
                    conv: Vec::new(),
                    k: Vec::with_capacity(64 * cfg.kv_dim()),
                    v: Vec::with_capacity(64 * cfg.kv_dim()),
                },
            })
            .collect();
        Self { layers, len: 0 }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for j in 0..8 {
            acc[j] += a[base + j] * b[base + j];
        }
    }
    let mut s = 0.0;
    for v in acc {
        s += v;
    }
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn matvec(t: &Tensor, x: &[f32]) -> Vec<f32> {
    (0..t.ne1).into_par_iter().map(|o| dot(t.row(o), x)).collect()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut sum = 0.0f64;
    for v in x {
        sum += (*v as f64) * (*v as f64);
    }
    let scale = (1.0 / ((sum / x.len() as f64) + eps as f64).sqrt()) as f32;
    x.iter().zip(w).map(|(a, b)| a * scale * b).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rope_inplace(v: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        if inv_freq[i] == 0.0 {
            continue;
        }
        let angle = pos as f32 * inv_freq[i];
        let (s, c) = angle.sin_cos();
        let x0 = v[i];
        let x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x1 * c + x0 * s;
    }
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    best as u32
}

impl Lfm3Ref {
    pub fn load(path: &Path) -> Result<Self> {
        let file = GgufFile::open(path).with_context(|| format!("open {}", path.display()))?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let mut tensors = HashMap::new();
        for info in file.tensors() {
            if info.dims.is_empty() || info.dims.len() > 2 {
                continue;
            }
            let ne0 = info.dims[0] as usize;
            let ne1 = info
                .dims
                .get(1)
                .map(|v| *v as usize)
                .unwrap_or(1)
                .max(1);
            let count = ne0 * ne1;
            let bytes = file.tensor_bytes(&info.name)?;
            let data = match info.ggml_type {
                GgmlType::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>(),
                GgmlType::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect::<Vec<f32>>(),
                GgmlType::Q4_K => lfm_quant::dequant_row_q4_k(bytes, count)?,
                GgmlType::Q6_K => lfm_quant::dequant_row_q6_k(bytes, count)?,
                GgmlType::Q8_0 => {
                    let blocks = count / 32;
                    let mut out = Vec::with_capacity(count);
                    for b in 0..blocks {
                        let mut block = [0u8; 34];
                        block.copy_from_slice(&bytes[b * 34..(b + 1) * 34]);
                        out.extend_from_slice(&lfm_quant::dequant_q8_0(&block));
                    }
                    out
                }
                _ => continue,
            };
            if data.len() != count {
                return Err(anyhow!(
                    "tensor {}: expected {count} values, got {}",
                    info.name,
                    data.len()
                ));
            }
            tensors.insert(info.name.clone(), Tensor { ne0, ne1, data });
        }
        let rope_full = cfg.rope_full.inv_freq();
        let rope_sliding = cfg.rope_sliding.inv_freq();
        Ok(Self {
            cfg,
            w4a8: std::env::var("LFM_CPU_W4A8").is_ok(),
            tensors,
            rope_full,
            rope_sliding,
        })
    }

    fn q(&self, x: &[f32]) -> Vec<f32> {
        if !self.w4a8 {
            return x.to_vec();
        }
        let mut out = x.to_vec();
        for chunk in out.chunks_mut(32) {
            let amax = chunk.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            if amax <= 0.0 {
                continue;
            }
            let sc = amax / 127.0;
            for v in chunk.iter_mut() {
                let qv = (*v / sc).round().clamp(-127.0, 127.0);
                *v = qv * sc;
            }
        }
        out
    }

    fn t(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn forward(&self, state: &mut State, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let emb = self.t("token_embd.weight")?;
        let mut x = emb.row(token as usize).to_vec();
        for layer in 0..cfg.num_layers {
            let prefix = format!("blk.{layer}");
            let residual = x.clone();
            let attn_norm = self.t(&format!("{prefix}.attn_norm.weight"))?;
            let h_norm = rms_norm(&x, &attn_norm.data, cfg.norm_eps);
            let op = match cfg.kind(layer) {
                LayerKind::Conv => self.conv_block(layer, &h_norm, &mut state.layers[layer])?,
                _ => self.attn_block(layer, &h_norm, &mut state.layers[layer], pos)?,
            };
            let post = self.t(&format!("{prefix}.post_attention_norm.weight"))?;
            let mut merged: Vec<f32> = rms_norm(&op, &post.data, cfg.norm_eps)
                .iter()
                .zip(&residual)
                .map(|(a, b)| a + b)
                .collect();
            let ffn_norm = self.t(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_in = rms_norm(&merged, &ffn_norm.data, cfg.norm_eps);
            let up = self.t(&format!("{prefix}.ffn_up.weight"))?;
            let gate = self.t(&format!("{prefix}.ffn_gate.weight"))?;
            let down = self.t(&format!("{prefix}.ffn_down.weight"))?;
            let u = matvec(up, &self.q(&ffn_in));
            let g = matvec(gate, &self.q(&ffn_in));
            let act: Vec<f32> = u.iter().zip(&g).map(|(up, gate)| silu(*gate) * up).collect();
            let ffn_out = matvec(down, &self.q(&act));
            let post_ffn = self.t(&format!("{prefix}.post_ffw_norm.weight"))?;
            let ffn_post = rms_norm(&ffn_out, &post_ffn.data, cfg.norm_eps);
            for i in 0..h {
                merged[i] += ffn_post[i];
            }
            x = merged;
        }
        state.len += 1;
        let norm = self.t("output_norm.weight")?;
        let x = rms_norm(&x, &norm.data, cfg.norm_eps);
        let logits = matvec(emb, &self.q(&x));
        Ok(logits)
    }

    fn conv_block(&self, layer: usize, x: &[f32], st: &mut LayerState) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let prefix = format!("blk.{layer}");
        let in_proj = self.t(&format!("{prefix}.shortconv.in_proj.weight"))?;
        let bcx = matvec(in_proj, &self.q(x));
        let b = &bcx[0..h];
        let c = &bcx[h..2 * h];
        let xv = &bcx[2 * h..3 * h];
        let bx: Vec<f32> = b.iter().zip(xv).map(|(a, b)| a * b).collect();
        let kernel = cfg.conv_cache;
        let w = self.t(&format!("{prefix}.shortconv.conv.weight"))?;
        let mut out = vec![0.0f32; h];
        for ch in 0..h {
            let wrow = w.row(ch);
            let mut acc = wrow[kernel - 1] * bx[ch];
            for t in 0..kernel - 1 {
                acc += wrow[t] * st.conv[t * h + ch];
            }
            out[ch] = acc;
        }
        st.conv.extend_from_slice(&bx);
        if st.conv.len() > (kernel - 1) * h {
            let excess = st.conv.len() - (kernel - 1) * h;
            st.conv.drain(0..excess);
        }
        let y: Vec<f32> = c.iter().zip(&out).map(|(a, b)| a * b).collect();
        let out_proj = self.t(&format!("{prefix}.shortconv.out_proj.weight"))?;
        Ok(matvec(out_proj, &self.q(&y)))
    }

    fn attn_block(
        &self,
        layer: usize,
        x: &[f32],
        st: &mut LayerState,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{layer}");
        let n_heads = cfg.num_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.kv_heads_of(layer);
        let group = n_heads / kv_heads.max(1);
        let inv_freq = match cfg.kind(layer) {
            LayerKind::SlidingAttention => &self.rope_sliding,
            _ => &self.rope_full,
        };
        let q = self.t(&format!("{prefix}.attn_q.weight"))?;
        let k = self.t(&format!("{prefix}.attn_k.weight"))?;
        let v = self.t(&format!("{prefix}.attn_v.weight"))?;
        let gate_w = self.t(&format!("{prefix}.attn_gate.weight"))?;
        let q_norm = self.t(&format!("{prefix}.attn_q_norm.weight"))?;
        let k_norm = self.t(&format!("{prefix}.attn_k_norm.weight"))?;
        let mut qv = matvec(q, &self.q(x));
        let mut kv = matvec(k, &self.q(x));
        let vv = matvec(v, &self.q(x));
        for head in 0..n_heads {
            let base = head * head_dim;
            let normed = rms_norm(&qv[base..base + head_dim], &q_norm.data, cfg.norm_eps);
            qv[base..base + head_dim].copy_from_slice(&normed);
            rope_inplace(&mut qv[base..base + head_dim], pos, inv_freq);
        }
        for head in 0..kv_heads {
            let base = head * head_dim;
            let normed = rms_norm(&kv[base..base + head_dim], &k_norm.data, cfg.norm_eps);
            kv[base..base + head_dim].copy_from_slice(&normed);
            rope_inplace(&mut kv[base..base + head_dim], pos, inv_freq);
        }
        st.k.extend_from_slice(&kv);
        st.v.extend_from_slice(&vv);
        let total = st.k.len() / (kv_heads * head_dim);
        let window = if cfg.kind(layer) == LayerKind::SlidingAttention {
            cfg.sliding_window.min(total)
        } else {
            total
        };
        let start = total - window;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kvh = head / group;
            let qh = &qv[head * head_dim..(head + 1) * head_dim];
            let mut scores = vec![0.0f32; window];
            for (j, slot) in (start..total).enumerate() {
                let base = slot * kv_heads * head_dim + kvh * head_dim;
                let kh = &st.k[base..base + head_dim];
                scores[j] = dot(qh, kh) * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            let out = &mut attn[head * head_dim..(head + 1) * head_dim];
            for (j, slot) in (start..total).enumerate() {
                let base = slot * kv_heads * head_dim + kvh * head_dim;
                let vh = &st.v[base..base + head_dim];
                let w = scores[j];
                for d in 0..head_dim {
                    out[d] += w * vh[d];
                }
            }
        }
        let gates = matvec(gate_w, &self.q(x));
        for head in 0..n_heads {
            let g = sigmoid(gates[head]);
            for d in 0..head_dim {
                attn[head * head_dim + d] *= g;
            }
        }
        let o = self.t(&format!("{prefix}.attn_output.weight"))?;
        Ok(matvec(o, &self.q(&attn)))
    }

    pub fn generate_greedy(&self, prompt: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>> {
        let mut state = State::new(&self.cfg);
        let mut logits = Vec::new();
        for (pos, token) in prompt.iter().enumerate() {
            logits = self.forward(&mut state, *token, pos)?;
        }
        let mut out = Vec::new();
        let mut next = argmax(&logits);
        let mut pos = prompt.len();
        for _ in 0..max_new {
            out.push(next);
            if next == eos {
                break;
            }
            logits = self.forward(&mut state, next, pos)?;
            pos += 1;
            next = argmax(&logits);
        }
        Ok(out)
    }

    pub fn forward_debug(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut state = State::new(&self.cfg);
        let mut norms = Vec::new();
        let mut logits = Vec::new();
        for (pos, token) in tokens.iter().enumerate() {
            logits = self.forward_debug_step(&mut state, *token, pos, &mut norms)?;
        }
        Ok((logits, norms))
    }

    fn forward_debug_step(
        &self,
        state: &mut State,
        token: u32,
        pos: usize,
        norms: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let emb = self.t("token_embd.weight")?;
        let mut x = emb.row(token as usize).to_vec();
        norms.push(norm(&x));
        for layer in 0..cfg.num_layers {
            let prefix = format!("blk.{layer}");
            let residual = x.clone();
            let attn_norm = self.t(&format!("{prefix}.attn_norm.weight"))?;
            let h_norm = rms_norm(&x, &attn_norm.data, cfg.norm_eps);
            let op = match cfg.kind(layer) {
                LayerKind::Conv => self.conv_block(layer, &h_norm, &mut state.layers[layer])?,
                _ => self.attn_block(layer, &h_norm, &mut state.layers[layer], pos)?,
            };
            norms.push(norm(&op));
            let post = self.t(&format!("{prefix}.post_attention_norm.weight"))?;
            let mut merged: Vec<f32> = rms_norm(&op, &post.data, cfg.norm_eps)
                .iter()
                .zip(&residual)
                .map(|(a, b)| a + b)
                .collect();
            let ffn_norm = self.t(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_in = rms_norm(&merged, &ffn_norm.data, cfg.norm_eps);
            let up = self.t(&format!("{prefix}.ffn_up.weight"))?;
            let gate = self.t(&format!("{prefix}.ffn_gate.weight"))?;
            let down = self.t(&format!("{prefix}.ffn_down.weight"))?;
            let u = matvec(up, &self.q(&ffn_in));
            let g = matvec(gate, &self.q(&ffn_in));
            let act: Vec<f32> = u.iter().zip(&g).map(|(up, gate)| silu(*gate) * up).collect();
            let ffn_out = matvec(down, &self.q(&act));
            let post_ffn = self.t(&format!("{prefix}.post_ffw_norm.weight"))?;
            let ffn_post = rms_norm(&ffn_out, &post_ffn.data, cfg.norm_eps);
            norms.push(norm(&ffn_post));
            for i in 0..cfg.hidden_size {
                merged[i] += ffn_post[i];
            }
            x = merged;
            norms.push(norm(&x));
        }
        state.len += 1;
        let norm_t = self.t("output_norm.weight")?;
        let x = rms_norm(&x, &norm_t.data, cfg.norm_eps);
        norms.push(norm(&x));
        let logits = matvec(emb, &self.q(&x));
        Ok(logits)
    }

    pub fn norm_named(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let w = self.t(name)?;
        Ok(rms_norm(x, &w.data, self.cfg.norm_eps))
    }

    pub fn conv_op(&self, layer: usize, h: &[f32]) -> Result<Vec<f32>> {
        let mut st = LayerState {
            conv: vec![0.0; (self.cfg.conv_cache - 1) * self.cfg.hidden_size],
            k: Vec::new(),
            v: Vec::new(),
        };
        self.conv_block(layer, h, &mut st)
    }

    pub fn ffn_debug(&self, layer: usize, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{layer}");
        let ffn_norm = self.t(&format!("{prefix}.ffn_norm.weight"))?;
        let ffn_in = rms_norm(x, &ffn_norm.data, cfg.norm_eps);
        let up = self.t(&format!("{prefix}.ffn_up.weight"))?;
        let gate = self.t(&format!("{prefix}.ffn_gate.weight"))?;
        let down = self.t(&format!("{prefix}.ffn_down.weight"))?;
        let u = matvec(up, &self.q(&ffn_in));
        let g = matvec(gate, &self.q(&ffn_in));
        let act: Vec<f32> = u.iter().zip(&g).map(|(up, gate)| silu(*gate) * up).collect();
        let ffn_out = matvec(down, &self.q(&act));
        let post_ffn = self.t(&format!("{prefix}.post_ffw_norm.weight"))?;
        let post = rms_norm(&ffn_out, &post_ffn.data, cfg.norm_eps);
        let out: Vec<f32> = post.iter().zip(x).map(|(a, b)| a + b).collect();
        Ok((ffn_in, u, g, out))
    }

    pub fn logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut state = State::new(&self.cfg);
        let mut logits = Vec::new();
        for (pos, token) in tokens.iter().enumerate() {
            logits = self.forward(&mut state, *token, pos)?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_small() {
        let t = Tensor {
            ne0: 3,
            ne1: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let y = matvec(&t, &[1.0, 1.0, 1.0]);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn rms_norm_unit() {
        let y = rms_norm(&[1.0, 1.0, 1.0, 1.0], &[1.0; 4], 0.0);
        assert!((y[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_zero_tail_is_identity() {
        let mut v = vec![0.3f32; 128];
        let inv = lfm_domain::RopeParams {
            theta: 1e6,
            dim: 128,
            rotated_dim: 64,
        }
        .inv_freq();
        rope_inplace(&mut v, 7, &inv);
        for i in 32..64 {
            assert_eq!(v[i], 0.3);
        }
        for i in 96..128 {
            assert_eq!(v[i], 0.3);
        }
        assert_ne!(v[0], 0.3);
        assert_ne!(v[64], 0.3);
    }

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[0.1, 0.5, 0.2]), 1);
    }
}
