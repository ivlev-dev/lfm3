use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use lfm_app::{CacheStats, InferenceBackend, SequenceStep};
use lfm_domain::{LayerKind, ModelConfig};
use lfm_gguf::{GgmlType, GgufFile};
use rayon::prelude::*;

const GROUP: usize = 32;
const TILE: usize = 8;

#[repr(C)]
#[derive(Clone)]
struct Q4Block {
    scales: [f32; 8],
    mins: [f32; 8],
    q: [u8; 128],
}

impl Q4Block {
    #[inline]
    fn qgroup(&self, g: usize) -> [u8; 32] {
        let base = (g & 3) * 32;
        let shift: u32 = if g >= 4 { 4 } else { 0 };
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = (self.q[base + i] >> shift) & 0x0F;
        }
        out
    }
}

#[repr(C)]
#[derive(Clone)]
struct Q6Block {
    scales: [f32; 16],
    q: [u8; 256],
}

enum Matrix {
    Q4(Vec<Q4Block>),
    Q6(Vec<Q6Block>),
    Raw(RawKind),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RawKind {
    Q4,
    Q6,
}

struct Quant {
    rows: usize,
    cols: usize,
    data: Matrix,
    raw: Option<(std::sync::Arc<GgufFile>, usize)>,
}

impl Quant {
    fn raw_bytes(&self) -> Option<&[u8]> {
        self.raw.as_ref().map(|(f, off)| &f.raw()[*off..])
    }
}

struct Act {
    q: Vec<i8>,
    s: Vec<f32>,
    sum32: Vec<i32>,
    sum16: Vec<i32>,
}

pub struct CpuBackend {
    cfg: ModelConfig,
    max_ctx: usize,
    quant: HashMap<String, Quant>,
    f32: HashMap<String, Vec<f32>>,
    dims: HashMap<String, (usize, usize)>,
    rope_full: Vec<f32>,
    rope_sliding: Vec<f32>,
    slots: Vec<Option<SeqState>>,
    last_logits: Vec<f32>,
    row_logits: Vec<Vec<f32>>,
}

struct SeqState {
    len: usize,
    layers: Vec<LayerState>,
}

struct LayerState {
    conv: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl CpuBackend {
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        init_threads();
        let mmap_on = std::env::var("LFM_MMAP")
            .map(|v| v != "0")
            .unwrap_or(cfg!(any(target_arch = "arm", target_arch = "aarch64")));
        let file = std::sync::Arc::new(
            GgufFile::open(path).with_context(|| format!("open {}", path.display()))?,
        );
        let cfg = ModelConfig::from_metadata(file.as_ref()).map_err(|e| anyhow!("config: {e}"))?;
        let mut quant = HashMap::new();
        let mut f32s = HashMap::new();
        let mut dims = HashMap::new();
        for info in file.tensors() {
            if info.dims.is_empty() || info.dims.len() > 2 {
                continue;
            }
            let cols = info.dims[0] as usize;
            let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
            let bytes = file.tensor_bytes(&info.name)?;
            match info.ggml_type {
                GgmlType::Q4_K => {
                    let blocks = bytes.len() / 144;
                    if blocks != rows * cols / 256 {
                        return Err(anyhow!("q4 shape mismatch {}", info.name));
                    }
                    if mmap_on {
                        let (start, _) = file.tensor_span(&info.name)?;
                        quant.insert(
                            info.name.clone(),
                            Quant {
                                rows,
                                cols,
                                data: Matrix::Raw(RawKind::Q4),
                                raw: Some((file.clone(), start)),
                            },
                        );
                    } else {
                        let mut out = Vec::with_capacity(blocks);
                        for b in bytes.chunks_exact(144) {
                            out.push(prep_q4(b));
                        }
                        quant.insert(
                            info.name.clone(),
                            Quant {
                                rows,
                                cols,
                                data: Matrix::Q4(out),
                                raw: None,
                            },
                        );
                    }
                }
                GgmlType::Q6_K => {
                    let blocks = bytes.len() / 210;
                    if mmap_on {
                        let (start, _) = file.tensor_span(&info.name)?;
                        quant.insert(
                            info.name.clone(),
                            Quant {
                                rows,
                                cols,
                                data: Matrix::Raw(RawKind::Q6),
                                raw: Some((file.clone(), start)),
                            },
                        );
                    } else {
                        let mut out = Vec::with_capacity(blocks);
                        for b in bytes.chunks_exact(210) {
                            out.push(prep_q6(b));
                        }
                        quant.insert(
                            info.name.clone(),
                            Quant {
                                rows,
                                cols,
                                data: Matrix::Q6(out),
                                raw: None,
                            },
                        );
                    }
                }
                GgmlType::F32 => {
                    let data: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    f32s.insert(info.name.clone(), data);
                    dims.insert(info.name.clone(), (cols, rows));
                }
                GgmlType::F16 => {
                    let data: Vec<f32> = bytes
                        .chunks_exact(2)
                        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                        .collect();
                    f32s.insert(info.name.clone(), data);
                    dims.insert(info.name.clone(), (cols, rows));
                }
                _ => continue,
            }
        }
        let rope_full = cfg.rope_full.inv_freq();
        let rope_sliding = cfg.rope_sliding.inv_freq();
        let slots = (0..cfg.num_layers.max(8)).map(|_| None).collect();
        Ok(Self {
            cfg,
            max_ctx,
            quant,
            f32: f32s,
            dims,
            rope_full,
            rope_sliding,
            slots,
            last_logits: Vec::new(),
            row_logits: Vec::new(),
        })
    }

    fn t(&self, name: &str) -> Result<&Quant> {
        self.quant
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    fn f(&self, name: &str) -> Result<&[f32]> {
        self.f32
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    fn row_f32(&self, name: &str) -> Result<(usize, usize)> {
        self.dims
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("missing dims {name}"))
    }

    fn new_state(&self) -> SeqState {
        let cfg = &self.cfg;
        let layers = (0..cfg.num_layers)
            .map(|i| match cfg.kind(i) {
                LayerKind::Conv => LayerState {
                    conv: vec![0.0; (cfg.conv_cache - 1) * cfg.hidden_size],
                    k: Vec::new(),
                    v: Vec::new(),
                },
                _ => LayerState {
                    conv: Vec::new(),
                    k: Vec::new(),
                    v: Vec::new(),
                },
            })
            .collect();
        SeqState { len: 0, layers }
    }

    fn forward(&self, seq: &mut SeqState, token: u32, pos: usize) -> Result<Vec<f32>> {
        if pos >= self.max_ctx {
            return Err(anyhow!("context overflow {pos} >= {}", self.max_ctx));
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let emb = self.t("token_embd.weight")?;
        let mut x = embed_row(emb, token as usize)?;
        for layer in 0..cfg.num_layers {
            let prefix = format!("blk.{layer}");
            let residual = x.clone();
            let attn_norm = self.f(&format!("{prefix}.attn_norm.weight"))?;
            let h_norm = rms_norm(&x, attn_norm, cfg.norm_eps);
            let act = quantize(&h_norm);
            let op = match cfg.kind(layer) {
                LayerKind::Conv => {
                    self.conv_block(layer, &act, &mut seq.layers[layer], h_norm.len())?
                }
                _ => self.attn_block(layer, &act, &mut seq.layers[layer], pos, &h_norm)?,
            };
            let post = self.f(&format!("{prefix}.post_attention_norm.weight"))?;
            let mut merged: Vec<f32> = rms_norm(&op, post, cfg.norm_eps)
                .iter()
                .zip(&residual)
                .map(|(a, b)| a + b)
                .collect();
            let ffn_norm = self.f(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_in = quantize(&rms_norm(&merged, ffn_norm, cfg.norm_eps));
            let up = self.t(&format!("{prefix}.ffn_up.weight"))?;
            let gate = self.t(&format!("{prefix}.ffn_gate.weight"))?;
            let down = self.t(&format!("{prefix}.ffn_down.weight"))?;
            let u = matvec(up, &ffn_in);
            let g = matvec(gate, &ffn_in);
            let act: Vec<f32> = u.iter().zip(&g).map(|(u, g)| silu(*g) * u).collect();
            let ffn_out = matvec(down, &quantize(&act));
            let post_ffn = self.f(&format!("{prefix}.post_ffw_norm.weight"))?;
            let ffn_post = rms_norm(&ffn_out, post_ffn, cfg.norm_eps);
            for i in 0..h {
                merged[i] += ffn_post[i];
            }
            x = merged;
        }
        seq.len += 1;
        let norm = self.f("output_norm.weight")?;
        let x = rms_norm(&x, norm, cfg.norm_eps);
        Ok(matvec(self.t("token_embd.weight")?, &quantize(&x)))
    }

    fn conv_block_batch(
        &self,
        layer: usize,
        acts: &[Act],
        st: &mut LayerState,
        h: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{layer}");
        let in_proj = self.t(&format!("{prefix}.shortconv.in_proj.weight"))?;
        let mut bcx = vec![0.0f32; TILE * 3 * h];
        matmul_batch(in_proj, acts, &mut bcx);
        let kernel = cfg.conv_cache;
        let (wcols, wrows) = self.row_f32(&format!("{prefix}.shortconv.conv.weight"))?;
        let w = self.f(&format!("{prefix}.shortconv.conv.weight"))?;
        if wcols != kernel || wrows < h {
            return Err(anyhow!("bad conv weight shape"));
        }
        let mut ys: Vec<Act> = Vec::with_capacity(TILE);
        for t in 0..n {
            let src = &bcx[t * 3 * h..(t + 1) * 3 * h];
            let b = &src[0..h];
            let c = &src[h..2 * h];
            let xv = &src[2 * h..3 * h];
            let bx: Vec<f32> = b.iter().zip(xv).map(|(a, b)| a * b).collect();
            let mut out = vec![0.0f32; h];
            out.par_iter_mut().enumerate().for_each(|(ch, o)| {
                let wrow = &w[ch * kernel..(ch + 1) * kernel];
                let mut acc = wrow[kernel - 1] * bx[ch];
                for j in 0..kernel - 1 {
                    acc += wrow[j] * st.conv[j * h + ch];
                }
                *o = acc;
            });
            st.conv.extend_from_slice(&bx);
            if st.conv.len() > (kernel - 1) * h {
                let excess = st.conv.len() - (kernel - 1) * h;
                st.conv.drain(0..excess);
            }
            let y: Vec<f32> = c.iter().zip(&out).map(|(a, b)| a * b).collect();
            ys.push(quantize(&y));
        }
        pad_acts(&mut ys, h);
        let out_proj = self.t(&format!("{prefix}.shortconv.out_proj.weight"))?;
        let mut op = vec![0.0f32; TILE * h];
        matmul_batch(out_proj, &ys, &mut op);
        Ok(op)
    }

    fn attn_block_batch(
        &self,
        layer: usize,
        acts: &[Act],
        st: &mut LayerState,
        pos0: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let prefix = format!("blk.{layer}");
        let n_heads = cfg.num_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.kv_heads_of(layer);
        let group = n_heads / kv_heads.max(1);
        let inv_freq = match cfg.kind(layer) {
            LayerKind::SlidingAttention => &self.rope_sliding,
            _ => &self.rope_full,
        };
        let qd = n_heads * head_dim;
        let kvd = kv_heads * head_dim;
        let mut qv = vec![0.0f32; TILE * qd];
        let mut kvb = vec![0.0f32; TILE * kvd];
        let mut vvb = vec![0.0f32; TILE * kvd];
        let mut gb = vec![0.0f32; TILE * n_heads];
        matmul_batch(self.t(&format!("{prefix}.attn_q.weight"))?, acts, &mut qv);
        matmul_batch(self.t(&format!("{prefix}.attn_k.weight"))?, acts, &mut kvb);
        matmul_batch(self.t(&format!("{prefix}.attn_v.weight"))?, acts, &mut vvb);
        matmul_batch(
            self.t(&format!("{prefix}.attn_gate.weight"))?,
            acts,
            &mut gb,
        );
        let q_norm = self.f(&format!("{prefix}.attn_q_norm.weight"))?;
        let k_norm = self.f(&format!("{prefix}.attn_k_norm.weight"))?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attns: Vec<Act> = Vec::with_capacity(TILE);
        for t in 0..n {
            let pos = pos0 + t;
            {
                let q = &mut qv[t * qd..(t + 1) * qd];
                for head in 0..n_heads {
                    let base = head * head_dim;
                    let normed = rms_norm(&q[base..base + head_dim], q_norm, cfg.norm_eps);
                    q[base..base + head_dim].copy_from_slice(&normed);
                    rope_inplace(&mut q[base..base + head_dim], pos, inv_freq);
                }
            }
            {
                let kv = &mut kvb[t * kvd..(t + 1) * kvd];
                for head in 0..kv_heads {
                    let base = head * head_dim;
                    let normed = rms_norm(&kv[base..base + head_dim], k_norm, cfg.norm_eps);
                    kv[base..base + head_dim].copy_from_slice(&normed);
                    rope_inplace(&mut kv[base..base + head_dim], pos, inv_freq);
                }
            }
            st.k.extend_from_slice(&kvb[t * kvd..(t + 1) * kvd]);
            st.v.extend_from_slice(&vvb[t * kvd..(t + 1) * kvd]);
            let total = st.k.len() / (kv_heads * head_dim);
            let window = if cfg.kind(layer) == LayerKind::SlidingAttention {
                cfg.sliding_window.min(total)
            } else {
                total
            };
            let start = total - window;
            let mut attn = vec![0.0f32; qd];
            let q = &qv[t * qd..(t + 1) * qd];
            for head in 0..n_heads {
                let kvh = head / group;
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                let mut scores = vec![0.0f32; window];
                for (j, slot) in (start..total).enumerate() {
                    let base = slot * kv_heads * head_dim + kvh * head_dim;
                    scores[j] = dot_f32(&st.k[base..base + head_dim], qh) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let out = &mut attn[head * head_dim..(head + 1) * head_dim];
                for (j, slot) in (start..total).enumerate() {
                    let base = slot * kv_heads * head_dim + kvh * head_dim;
                    let vh = &st.v[base..base + head_dim];
                    let w = scores[j] * inv;
                    for d in 0..head_dim {
                        out[d] += w * vh[d];
                    }
                }
            }
            for head in 0..n_heads {
                let g = sigmoid(gb[t * n_heads + head]);
                for d in 0..head_dim {
                    attn[head * head_dim + d] *= g;
                }
            }
            attns.push(quantize(&attn));
        }
        pad_acts(&mut attns, qd);
        let mut op = vec![0.0f32; TILE * h];
        matmul_batch(
            self.t(&format!("{prefix}.attn_output.weight"))?,
            &attns,
            &mut op,
        );
        Ok(op)
    }

    fn forward_batch(
        &self,
        seq: &mut SeqState,
        tokens: &[u32],
        pos0: usize,
        need_logits: bool,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let ffn = cfg.ffn_size;
        let n = tokens.len();
        if n == 0 || n > TILE {
            return Err(anyhow!("bad cpu batch size {n}"));
        }
        if pos0 + n > self.max_ctx {
            return Err(anyhow!("context overflow {} >= {}", pos0 + n, self.max_ctx));
        }
        let emb = self.t("token_embd.weight")?;
        let mut x = Vec::with_capacity(TILE * h);
        for &token in tokens {
            x.extend_from_slice(&embed_row(emb, token as usize)?);
        }
        x.resize(TILE * h, 0.0);
        for layer in 0..cfg.num_layers {
            let prefix = format!("blk.{layer}");
            let attn_norm = self.f(&format!("{prefix}.attn_norm.weight"))?;
            let mut acts: Vec<Act> = (0..n)
                .map(|t| quantize(&rms_norm(&x[t * h..(t + 1) * h], attn_norm, cfg.norm_eps)))
                .collect();
            pad_acts(&mut acts, h);
            let residual = x.clone();
            let op = match cfg.kind(layer) {
                LayerKind::Conv => {
                    self.conv_block_batch(layer, &acts, &mut seq.layers[layer], h, n)?
                }
                _ => self.attn_block_batch(layer, &acts, &mut seq.layers[layer], pos0, n)?,
            };
            let post = self.f(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm = self.f(&format!("{prefix}.ffn_norm.weight"))?;
            let mut merged = vec![0.0f32; TILE * h];
            let mut acts2: Vec<Act> = Vec::with_capacity(TILE);
            for t in 0..n {
                let nr = rms_norm(&op[t * h..(t + 1) * h], post, cfg.norm_eps);
                for i in 0..h {
                    merged[t * h + i] = nr[i] + residual[t * h + i];
                }
                acts2.push(quantize(&rms_norm(
                    &merged[t * h..(t + 1) * h],
                    ffn_norm,
                    cfg.norm_eps,
                )));
            }
            pad_acts(&mut acts2, h);
            let up = self.t(&format!("{prefix}.ffn_up.weight"))?;
            let gate = self.t(&format!("{prefix}.ffn_gate.weight"))?;
            let down = self.t(&format!("{prefix}.ffn_down.weight"))?;
            let mut u = vec![0.0f32; TILE * ffn];
            let mut g = vec![0.0f32; TILE * ffn];
            matmul_batch(up, &acts2, &mut u);
            matmul_batch(gate, &acts2, &mut g);
            let mut acts3: Vec<Act> = Vec::with_capacity(TILE);
            for t in 0..n {
                let a: Vec<f32> = (0..ffn)
                    .map(|i| silu(g[t * ffn + i]) * u[t * ffn + i])
                    .collect();
                acts3.push(quantize(&a));
            }
            pad_acts(&mut acts3, ffn);
            let mut d = vec![0.0f32; TILE * h];
            matmul_batch(down, &acts3, &mut d);
            let post_ffn = self.f(&format!("{prefix}.post_ffw_norm.weight"))?;
            for t in 0..n {
                let nr = rms_norm(&d[t * h..(t + 1) * h], post_ffn, cfg.norm_eps);
                for i in 0..h {
                    x[t * h + i] = merged[t * h + i] + nr[i];
                }
            }
        }
        seq.len += n;
        if !need_logits {
            return Ok(Vec::new());
        }
        let norm = self.f("output_norm.weight")?;
        let last = &x[(n - 1) * h..n * h];
        let xn = rms_norm(last, norm, cfg.norm_eps);
        Ok(matvec(emb, &quantize(&xn)))
    }

    fn conv_block(
        &self,
        layer: usize,
        x: &Act,
        st: &mut LayerState,
        h: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let prefix = format!("blk.{layer}");
        let in_proj = self.t(&format!("{prefix}.shortconv.in_proj.weight"))?;
        let bcx = matvec(in_proj, x);
        let b = &bcx[0..h];
        let c = &bcx[h..2 * h];
        let xv = &bcx[2 * h..3 * h];
        let bx: Vec<f32> = b.iter().zip(xv).map(|(a, b)| a * b).collect();
        let kernel = cfg.conv_cache;
        let (wcols, wrows) = self.row_f32(&format!("{prefix}.shortconv.conv.weight"))?;
        let w = self.f(&format!("{prefix}.shortconv.conv.weight"))?;
        if wcols != kernel || wrows < h {
            return Err(anyhow!("bad conv weight shape"));
        }
        let mut out = vec![0.0f32; h];
        out.par_iter_mut().enumerate().for_each(|(ch, o)| {
            let wrow = &w[ch * kernel..(ch + 1) * kernel];
            let mut acc = wrow[kernel - 1] * bx[ch];
            for t in 0..kernel - 1 {
                acc += wrow[t] * st.conv[t * h + ch];
            }
            *o = acc;
        });
        st.conv.extend_from_slice(&bx);
        if st.conv.len() > (kernel - 1) * h {
            let excess = st.conv.len() - (kernel - 1) * h;
            st.conv.drain(0..excess);
        }
        let y: Vec<f32> = c.iter().zip(&out).map(|(a, b)| a * b).collect();
        Ok(matvec(self.t(&format!("{prefix}.shortconv.out_proj.weight"))?, &quantize(&y)))
    }

    fn attn_block(
        &self,
        layer: usize,
        x: &Act,
        st: &mut LayerState,
        pos: usize,
        xf: &[f32],
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
        let mut qv = matvec(q, x);
        let mut kv = matvec(k, x);
        let vv = matvec(v, x);
        let q_norm = self.f(&format!("{prefix}.attn_q_norm.weight"))?;
        let k_norm = self.f(&format!("{prefix}.attn_k_norm.weight"))?;
        for head in 0..n_heads {
            let base = head * head_dim;
            let normed = rms_norm(&qv[base..base + head_dim], q_norm, cfg.norm_eps);
            qv[base..base + head_dim].copy_from_slice(&normed);
            rope_inplace(&mut qv[base..base + head_dim], pos, inv_freq);
        }
        for head in 0..kv_heads {
            let base = head * head_dim;
            let normed = rms_norm(&kv[base..base + head_dim], k_norm, cfg.norm_eps);
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
                scores[j] = dot_f32(&st.k[base..base + head_dim], qh) * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in scores.iter_mut() {
                *s *= inv;
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
        let gates = matvec(gate_w, &quantize(xf));
        for head in 0..n_heads {
            let g = sigmoid(gates[head]);
            for d in 0..head_dim {
                attn[head * head_dim + d] *= g;
            }
        }
        Ok(matvec(
            self.t(&format!("{prefix}.attn_output.weight"))?,
            &quantize(&attn),
        ))
    }
}

impl InferenceBackend for CpuBackend {
    fn create_sequence(&mut self) -> Result<usize> {
        if let Some(slot) = self.slots.iter().position(|s| s.is_none()) {
            let state = self.new_state();
            self.slots[slot] = Some(state);
            return Ok(slot);
        }
        let state = self.new_state();
        self.slots.push(Some(state));
        Ok(self.slots.len() - 1)
    }

    fn free_sequence(&mut self, slot: usize) -> Result<()> {
        if let Some(entry) = self.slots.get_mut(slot) {
            *entry = None;
        }
        Ok(())
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32], need_logits: bool) -> Result<usize> {
        let reused = self.prefill_begin(slot, tokens)?;
        let mut state = self
            .slots
            .get_mut(slot)
            .and_then(|s| s.take())
            .ok_or_else(|| anyhow!("empty cpu sequence {slot}"))?;
        let mut logits = Vec::new();
        let result = (|| -> Result<()> {
            let mut pos = reused;
            while pos < tokens.len() {
                let end = (pos + TILE).min(tokens.len());
                logits = self.forward_batch(
                    &mut state,
                    &tokens[pos..end],
                    pos,
                    need_logits && end == tokens.len(),
                )?;
                pos = end;
            }
            Ok(())
        })();
        self.slots[slot] = Some(state);
        result?;
        self.prefill_finish(slot)?;
        if need_logits {
            self.last_logits = logits;
        }
        Ok(reused)
    }

    fn prefill_begin(&mut self, _slot: usize, _tokens: &[u32]) -> Result<usize> {
        Ok(0)
    }

    fn prefill_finish(&mut self, _slot: usize) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()> {
        self.row_logits.clear();
        for step in steps {
            let mut state = self
                .slots
                .get_mut(step.slot)
                .and_then(|s| s.take())
                .ok_or_else(|| anyhow!("empty cpu sequence {}", step.slot))?;
            let logits = self.forward(&mut state, step.token, step.pos);
            self.slots[step.slot] = Some(state);
            let logits = logits?;
            if need_logits {
                self.last_logits = logits.clone();
                self.row_logits.push(logits);
            } else {
                self.row_logits.push(Vec::new());
            }
        }
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.last_logits.clone())
    }

    fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        self.row_logits
            .get(row)
            .filter(|v| !v.is_empty())
            .cloned()
            .ok_or_else(|| anyhow!("no cpu logits for row {row}"))
    }

    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn free_pages(&self) -> usize {
        self.max_ctx.div_ceil(32)
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

fn prep_q4(block: &[u8]) -> Q4Block {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut q = [0u8; 128];
    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut g = 0usize;
    for j in (0..256).step_by(64) {
        let (sc1, m1) = scale_min(block, is);
        let (sc2, m2) = scale_min(block, is + 1);
        scales[g] = d * f32::from(sc1);
        mins[g] = dmin * f32::from(m1);
        for l in 0..32 {
            let v = j + l;
            let lo = (block[16 + q_off + l] & 0x0F) as u16;
            q[v & 127] |= (lo << (4 * (v >> 7))) as u8;
        }
        g += 1;
        scales[g] = d * f32::from(sc2);
        mins[g] = dmin * f32::from(m2);
        for l in 0..32 {
            let v = j + 32 + l;
            let hi = (block[16 + q_off + l] >> 4) as u16;
            q[v & 127] |= (hi << (4 * (v >> 7))) as u8;
        }
        g += 1;
        q_off += 32;
        is += 2;
    }
    Q4Block { scales, mins, q }
}

fn scale_min(block: &[u8], j: usize) -> (u8, u8) {
    let q = &block[4..16];
    let jm = j & 3;
    let h = j >> 2;
    let a = q[jm];
    let b = q[jm + 4];
    let c = q[jm + 8];
    let lo_d = a & 63;
    let lo_m = b & 63;
    let hi_d = (c & 15) | ((a >> 6) << 4);
    let hi_m = (c >> 4) | ((b >> 6) << 4);
    if h == 0 {
        (lo_d, lo_m)
    } else {
        (hi_d, hi_m)
    }
}

fn prep_q6(block: &[u8]) -> Q6Block {
    let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut q = [0u8; 256];
    let mut scales = [0.0f32; 16];
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    let mut base = 0usize;
    for _n in 0..2 {
        for l in 0..32usize {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32
                | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 = ((ql[ql_off + l] >> 4) as i32
                | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4))
                - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) as i32
                | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4))
                - 32;
            q[base + l] = (q1 + 32) as u8;
            q[base + l + 32] = (q2 + 32) as u8;
            q[base + l + 64] = (q3 + 32) as u8;
            q[base + l + 96] = (q4 + 32) as u8;
            scales[base / 16 + l / 16] = d * (sc[sc_off + is] as i8 as f32);
            scales[base / 16 + 2 + l / 16] = d * (sc[sc_off + is + 2] as i8 as f32);
            scales[base / 16 + 4 + l / 16] = d * (sc[sc_off + is + 4] as i8 as f32);
            scales[base / 16 + 6 + l / 16] = d * (sc[sc_off + is + 6] as i8 as f32);
        }
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
        base += 128;
    }
    Q6Block { scales, q }
}

fn quantize(x: &[f32]) -> Act {
    let groups = x.len().div_ceil(GROUP);
    let mut q = vec![0i8; x.len()];
    let mut s = vec![0.0f32; groups];
    let q256 = std::env::var("LFM_Q256").map(|v| v != "0").unwrap_or(false);
    if q256 || cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
        return quantize_block(x, q, s);
    }
    let mut sum32 = vec![0i32; groups];
    let mut sum16 = vec![0i32; groups * 2];
    for g in 0..groups {
        let start = g * GROUP;
        let end = (start + GROUP).min(x.len());
        let mut amax = 0.0f32;
        for &v in &x[start..end] {
            amax = amax.max(v.abs());
        }
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        s[g] = scale;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (i, &v) in x[start..end].iter().enumerate() {
            q[start + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        for i in 0..(end - start) {
            sum32[g] += i32::from(q[start + i]);
            sum16[g * 2 + i / 16] += i32::from(q[start + i]);
        }
    }
    Act { q, s, sum32, sum16 }
}

fn quantize_block(x: &[f32], mut q: Vec<i8>, mut s: Vec<f32>) -> Act {
    let groups = s.len();
    let blocks = x.len() / 256;
    let mut sum32 = vec![0i32; groups];
    let mut sum16 = vec![0i32; groups * 2];
    for b in 0..blocks {
        let start = b * 256;
        let end = (start + 256).min(x.len());
        let amax = x[start..end].iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let sc = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for g in b * 8..(b * 8 + 8).min(groups) {
            s[g] = sc;
        }
        for i in start..end {
            let v = (x[i] * inv).round().clamp(-127.0, 127.0) as i8;
            q[i] = v;
            sum32[i / 32] += i32::from(v);
            sum16[i / 16] += i32::from(v);
        }
    }
    Act { q, s, sum32, sum16 }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
unsafe extern "C" {
    fn lfm_q4_rows(
        w: *const u8,
        xq: *const *const i8,
        xs: *const *const f32,
        sums: *const *const i32,
        blocks: i32,
        ntok: i32,
        out: *mut f32,
    );
    fn lfm_q6_rows(
        w: *const u8,
        xq: *const *const i8,
        xs: *const *const f32,
        sums16: *const *const i32,
        blocks: i32,
        ntok: i32,
        out: *mut f32,
    );
    fn lfm_q4_rows_raw(
        w: *const u8,
        xq: *const *const i8,
        xs: *const *const f32,
        sums: *const *const i32,
        blocks: i32,
        ntok: i32,
        out: *mut f32,
    );
    fn lfm_q6_rows_raw(
        w: *const u8,
        xq: *const *const i8,
        xs: *const *const f32,
        sums16: *const *const i32,
        blocks: i32,
        ntok: i32,
        out: *mut f32,
    );
}

fn online_cpus() -> usize {
    let mut best = 0usize;
    for path in ["/sys/devices/system/cpu/present", "/sys/devices/system/cpu/online"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let mut count = 0usize;
            for part in text.trim().split(',') {
                match part.split_once('-') {
                    Some((a, b)) => {
                        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                            count += b.saturating_sub(a) + 1;
                        }
                    }
                    None => count += 1,
                }
            }
            best = best.max(count);
        }
    }
    if best > 0 {
        return best;
    }
    if let Ok(text) = std::fs::read_to_string("/sys/devices/system/cpu/online") {
        let mut count = 0usize;
        for part in text.trim().split(',') {
            match part.split_once('-') {
                Some((a, b)) => {
                    if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        count += b.saturating_sub(a) + 1;
                    }
                }
                None => count += 1,
            }
        }
        if count > 0 {
            return count;
        }
    }
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
}

fn init_threads() {
    let n = std::env::var("LFM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(online_cpus)
        .min(64);
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    });
}

fn q4_raw_scales(blk: &[u8]) -> ([f32; 8], [f32; 8]) {
    let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
    let mut sc = [0.0f32; 8];
    let mut mn = [0.0f32; 8];
    for j in 0..8 {
        let (a, b) = scale_min(blk, j);
        sc[j] = d * f32::from(a);
        mn[j] = dmin * f32::from(b);
    }
    (sc, mn)
}

#[inline]
fn q4_raw_val(qs: &[u8], g: usize, i: usize) -> u8 {
    let base = (g / 2) * 32;
    let shift: u32 = 4 * (g as u32 & 1);
    (qs[base + i] >> shift) & 0x0F
}

#[inline]
fn q6_raw_val(blk: &[u8], v: usize) -> u8 {
    let n = v / 128;
    let j = v % 128;
    let s = j / 32;
    let l = j % 32;
    let ql = blk[64 * n + (s & 1) * 32 + l];
    let qh = blk[128 + 32 * n + l];
    let nib = if s < 2 { ql & 0x0F } else { ql >> 4 };
    nib | (((qh >> (2 * s)) & 3) << 4)
}

#[allow(unreachable_code)]
fn raw_rows(w: &Quant, acts: &[&Act], out: &mut [f32]) {
    let raw = w.raw_bytes().expect("raw weights");
    let neon = std::env::var("LFM_NEON").map(|v| v != "0").unwrap_or(true);
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    if neon {
        let kind = match &w.data {
            Matrix::Raw(k) => *k,
            _ => unreachable!(),
        };
        let blocks = w.cols / 256;
        let rows = w.rows;
        let n = acts.len();
        let qp: Vec<usize> = acts.iter().map(|a| a.q.as_ptr() as usize).collect();
        let sp: Vec<usize> = acts.iter().map(|a| a.s.as_ptr() as usize).collect();
        let up: Vec<usize> = acts.iter().map(|a| a.sum32.as_ptr() as usize).collect();
        let u16p: Vec<usize> = acts.iter().map(|a| a.sum16.as_ptr() as usize).collect();
        let dst = out.as_mut_ptr() as usize;
        let base = raw.as_ptr() as usize;
        let bsz = if kind == RawKind::Q4 { 144 } else { 210 };
        let chunk = 16usize;
        let tasks: Vec<usize> = (0..rows).step_by(chunk).collect();
        tasks.into_par_iter().for_each(|r0| unsafe {
            let r1 = (r0 + chunk).min(rows);
            for row in r0..r1 {
            let mut tmp = [0.0f32; TILE];
            let wptr = (base as *const u8).add(row * blocks * bsz);
            match kind {
                RawKind::Q4 => lfm_q4_rows_raw(
                    wptr,
                    qp.as_ptr() as *const *const i8,
                    sp.as_ptr() as *const *const f32,
                    up.as_ptr() as *const *const i32,
                    blocks as i32,
                    n as i32,
                    tmp.as_mut_ptr(),
                ),
                RawKind::Q6 => lfm_q6_rows_raw(
                    wptr,
                    qp.as_ptr() as *const *const i8,
                    sp.as_ptr() as *const *const f32,
                    u16p.as_ptr() as *const *const i32,
                    blocks as i32,
                    n as i32,
                    tmp.as_mut_ptr(),
                ),
            }
            let p = dst as *mut f32;
            for t in 0..n {
                *p.add(t * rows + row) = tmp[t];
            }
            }
        });
        return;
    }
    let _ = raw;
    let kind = match &w.data {
        Matrix::Raw(k) => *k,
        _ => unreachable!(),
    };
    let blocks = w.cols / 256;
    let rows = w.rows;
    let n = acts.len();
    let ptr = out.as_mut_ptr() as usize;
    (0..rows).into_par_iter().for_each(|row| {
        let mut acc = [0.0f32; TILE];
        match kind {
            RawKind::Q4 => {
                for b in 0..blocks {
                    let blk = &raw[(row * blocks + b) * 144..][..144];
                    let (sc, mn) = q4_raw_scales(blk);
                    let qs = &blk[16..144];
                    for g in 0..8 {
                        let off = b * 256 + g * 32;
                        for (t, a) in acts.iter().enumerate().take(n) {
                            let mut dot = 0i32;
                            for i in 0..32 {
                                dot += i32::from(q4_raw_val(qs, g, i)) * i32::from(a.q[off + i]);
                            }
                            acc[t] += a.s[b * 8 + g] * (sc[g] * dot as f32
                                - mn[g] * a.sum32[b * 8 + g] as f32);
                        }
                    }
                }
            }
            RawKind::Q6 => {
                for b in 0..blocks {
                    let blk = &raw[(row * blocks + b) * 210..][..210];
                    let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
                    for k in 0..16 {
                        let sc = d * f32::from(blk[192 + k] as i8);
                        let off = b * 256 + k * 16;
                        for (t, a) in acts.iter().enumerate().take(n) {
                            let mut dot = 0i32;
                            for i in 0..16 {
                                dot += i32::from(q6_raw_val(blk, k * 16 + i)) * i32::from(a.q[off + i]);
                            }
                            acc[t] += a.s[b * 8 + k / 2] * sc
                                * (dot - 32 * a.sum16[b * 16 + k]) as f32;
                        }
                    }
                }
            }
        }
        let p = ptr as *mut f32;
        for t in 0..n {
            unsafe {
                *p.add(t * rows + row) = acc[t];
            }
        }
    });
}

#[allow(unreachable_code)]
fn matvec(w: &Quant, x: &Act) -> Vec<f32> {
    if w.raw.is_some() {
        let mut out = vec![0.0f32; w.rows];
        raw_rows(w, &[x], &mut out);
        return out;
    }
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        let blocks = w.cols / 256;
        let mut out = vec![0.0f32; w.rows];
        let xq = [x.q.as_ptr() as usize];
        let xs = [x.s.as_ptr() as usize];
        let sums = [x.sum32.as_ptr() as usize];
        let sums16 = [x.sum16.as_ptr() as usize];
        match &w.data {
            Matrix::Raw(_) => unreachable!("raw handled earlier"),
            Matrix::Q4(data) => {
                out.par_iter_mut().enumerate().for_each(|(row, o)| unsafe {
                    let wptr = data.as_ptr().add(row * blocks) as *const u8;
                    lfm_q4_rows(
                        wptr,
                        xq.as_ptr() as *const *const i8,
                        xs.as_ptr() as *const *const f32,
                        sums.as_ptr() as *const *const i32,
                        blocks as i32,
                        1,
                        o as *mut f32,
                    );
                });
            }
            Matrix::Q6(data) => {
                out.par_iter_mut().enumerate().for_each(|(row, o)| unsafe {
                    let wptr = data.as_ptr().add(row * blocks) as *const u8;
                    lfm_q6_rows(
                        wptr,
                        xq.as_ptr() as *const *const i8,
                        xs.as_ptr() as *const *const f32,
                        sums16.as_ptr() as *const *const i32,
                        blocks as i32,
                        1,
                        o as *mut f32,
                    );
                });
            }
        }
        return out;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() && std::env::var("LFM_CPU_SCALAR").is_err() {
            unsafe {
                return matvec_avx2(w, x);
            }
        }
    }
    let blocks = w.cols / 256;
    (0..w.rows)
        .into_par_iter()
        .map(|row| match &w.data {
            Matrix::Raw(_) => unreachable!("raw handled earlier"),
            Matrix::Q4(data) => {
                let mut acc = 0.0f32;
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    let xb = b * 256;
                    for g in 0..8 {
                        let off = xb + g * 32;
                        let qg = blk.qgroup(g);
                        let dot = dot_u8_i8_32(&qg, &x.q[off..off + 32]);
                        acc += x.s[g + b * 8]
                            * (blk.scales[g] * dot as f32 - blk.mins[g] * x.sum32[g + b * 8] as f32);
                    }
                }
                acc
            }
            Matrix::Q6(data) => {
                let mut acc = 0.0f32;
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    let xb = b * 256;
                    for k in 0..16 {
                        let off = xb + k * 16;
                        let dot = dot_u8_i8_16(&blk.q[k * 16..k * 16 + 16], &x.q[off..off + 16]);
                        let sum = x.sum16[b * 16 + k];
                        acc += x.s[(b * 8) + k / 2] * blk.scales[k] * (dot - 32 * sum) as f32;
                    }
                }
                acc
            }
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_batch_avx2(w: &Quant, acts: &[Act], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let rows = w.rows;
    let blocks = w.cols / 256;
    let ptr = out.as_mut_ptr() as usize;
    let ones = _mm256_set1_epi16(1);
    let mut qp = [0usize; TILE];
    let mut sp = [0usize; TILE];
    let mut up = [0usize; TILE];
    for t in 0..TILE {
        qp[t] = acts[t].q.as_ptr() as usize;
        sp[t] = acts[t].s.as_ptr() as usize;
        up[t] = acts[t].sum32.as_ptr() as usize;
    }
    let mut wsum = vec![0.0f32; TILE * blocks * 16];
    for t in 0..TILE {
        let a = &acts[t];
        for b in 0..blocks {
            let base = (t * blocks + b) * 16;
            match &w.data {
                Matrix::Raw(_) => unreachable!("raw handled earlier"),
                Matrix::Q4(_) => {
                    for g in 0..8 {
                        wsum[base + g] = a.s[b * 8 + g] * a.sum32[b * 8 + g] as f32;
                    }
                }
                Matrix::Q6(_) => {
                    for k in 0..16 {
                        wsum[base + k] = 32.0 * a.sum16[b * 16 + k] as f32 * a.s[b * 8 + k / 2];
                    }
                }
            }
        }
    }
    let wsum = &wsum;
    match &w.data {
        Matrix::Raw(_) => unreachable!("raw handled earlier"),
        Matrix::Q4(data) => {
            (0..rows).into_par_iter().for_each(|row| unsafe {
                let mut acc = [_mm256_setzero_ps(); TILE];
                let mut msub = [_mm256_setzero_ps(); TILE];
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    let mnv = _mm256_loadu_ps(blk.mins.as_ptr());
                    for t in 0..TILE {
                        let wv = _mm256_loadu_ps(wsum.as_ptr().add((t * blocks + b) * 16));
                        msub[t] = _mm256_fmadd_ps(mnv, wv, msub[t]);
                    }
                    for g in 0..8 {
                        let p = _mm256_loadu_si256(blk.q[(g & 3) * 32..(g & 3) * 32 + 32].as_ptr() as *const __m256i);
                        let q = if g < 4 {
                            _mm256_and_si256(p, _mm256_set1_epi8(0x0F))
                        } else {
                            _mm256_and_si256(_mm256_srli_epi16(p, 4), _mm256_set1_epi8(0x0F))
                        };
                        let scv = _mm256_set1_ps(blk.scales[g]);
                        let off = b * 256 + g * 32;
                        for t in 0..TILE {
                            let a = &acts[t];
                            let xv = _mm256_loadu_si256(a.q[off..off + 32].as_ptr() as *const __m256i);
                            let pairs = _mm256_maddubs_epi16(q, xv);
                            let sums = _mm256_madd_epi16(pairs, ones);
                            let f = _mm256_cvtepi32_ps(sums);
                            acc[t] = _mm256_fmadd_ps(f, _mm256_mul_ps(scv, _mm256_set1_ps(a.s[b * 8 + g])), acc[t]);
                        }
                    }
                }
                for t in 0..TILE {
                    let v = _mm256_sub_ps(acc[t], msub[t]);
                    let lo = _mm256_castps256_ps128(v);
                    let hi = _mm256_extractf128_ps(v, 1);
                    let s4 = _mm_add_ps(lo, hi);
                    let s2 = _mm_hadd_ps(s4, s4);
                    let s1 = _mm_hadd_ps(s2, s2);
                    *(ptr as *mut f32).add(t * rows + row) = _mm_cvtss_f32(s1);
                }
            });
        }
        Matrix::Q6(data) => {
            (0..rows).into_par_iter().for_each(|row| unsafe {
                let mut acc = [_mm256_setzero_ps(); TILE];
                let mut msub = [_mm256_setzero_ps(); TILE];
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    for t in 0..TILE {
                        let wv0 = _mm256_loadu_ps(wsum.as_ptr().add((t * blocks + b) * 16));
                        let wv1 = _mm256_loadu_ps(wsum.as_ptr().add((t * blocks + b) * 16 + 8));
                        let sc0 = _mm256_loadu_ps(blk.scales.as_ptr());
                        let sc1 = _mm256_loadu_ps(blk.scales.as_ptr().add(8));
                        msub[t] = _mm256_fmadd_ps(sc0, wv0, msub[t]);
                        msub[t] = _mm256_fmadd_ps(sc1, wv1, msub[t]);
                    }
                    for kk in 0..8 {
                        let k0 = kk * 2;
                        let q = _mm256_loadu_si256(blk.q[k0 * 16..k0 * 16 + 32].as_ptr() as *const __m256i);
                        let sa = blk.scales[k0];
                        let sb = blk.scales[k0 + 1];
                        let scv = _mm256_setr_ps(sa, sa, sa, sa, sb, sb, sb, sb);
                        let off = b * 256 + k0 * 16;
                        for t in 0..TILE {
                            let a = &acts[t];
                            let xv = _mm256_loadu_si256(a.q[off..off + 32].as_ptr() as *const __m256i);
                            let pairs = _mm256_maddubs_epi16(q, xv);
                            let sums = _mm256_madd_epi16(pairs, ones);
                            let f = _mm256_cvtepi32_ps(sums);
                            acc[t] = _mm256_fmadd_ps(f, _mm256_mul_ps(scv, _mm256_set1_ps(a.s[b * 8 + k0 / 2])), acc[t]);
                        }
                    }
                }
                for t in 0..TILE {
                    let v = _mm256_sub_ps(acc[t], msub[t]);
                    let lo = _mm256_castps256_ps128(v);
                    let hi = _mm256_extractf128_ps(v, 1);
                    let s4 = _mm_add_ps(lo, hi);
                    let s2 = _mm_hadd_ps(s4, s4);
                    let s1 = _mm_hadd_ps(s2, s2);
                    *(ptr as *mut f32).add(t * rows + row) = _mm_cvtss_f32(s1);
                }
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matvec_avx2(w: &Quant, x: &Act) -> Vec<f32> {
    use std::arch::x86_64::*;
    let blocks = w.cols / 256;
    let ones = _mm256_set1_epi16(1);
    let hsum = |v: __m256| -> f32 {
        let m = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s2 = _mm_hadd_ps(m, m);
        _mm_cvtss_f32(_mm_hadd_ps(s2, s2))
    };
    match &w.data {
        Matrix::Raw(_) => unreachable!("raw handled earlier"),
        Matrix::Q4(data) => (0..w.rows)
            .into_par_iter()
            .map(|row| unsafe {
                let base = row * blocks;
                let mut vacc = _mm256_setzero_ps();
                let mut msum = 0.0f32;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    for g in 0..8 {
                        let p = _mm256_loadu_si256(blk.q[(g & 3) * 32..(g & 3) * 32 + 32].as_ptr() as *const __m256i);
                        let q = if g < 4 {
                            _mm256_and_si256(p, _mm256_set1_epi8(0x0F))
                        } else {
                            _mm256_and_si256(_mm256_srli_epi16(p, 4), _mm256_set1_epi8(0x0F))
                        };
                        let off = b * 256 + g * 32;
                        let xv = _mm256_loadu_si256(x.q[off..off + 32].as_ptr() as *const __m256i);
                        let pairs = _mm256_maddubs_epi16(q, xv);
                        let sums = _mm256_madd_epi16(pairs, ones);
                        let f = _mm256_cvtepi32_ps(sums);
                        let xs = x.s[b * 8 + g];
                        vacc = _mm256_fmadd_ps(f, _mm256_set1_ps(blk.scales[g] * xs), vacc);
                        msum += blk.mins[g] * xs * x.sum32[b * 8 + g] as f32;
                    }
                }
                hsum(vacc) - msum
            })
            .collect(),
        Matrix::Q6(data) => (0..w.rows)
            .into_par_iter()
            .map(|row| unsafe {
                let base = row * blocks;
                let mut vacc = _mm256_setzero_ps();
                let mut msum = 0.0f32;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    for kk in 0..8 {
                        let k0 = kk * 2;
                        let q = _mm256_loadu_si256(blk.q[k0 * 16..k0 * 16 + 32].as_ptr() as *const __m256i);
                        let sa = blk.scales[k0];
                        let sb = blk.scales[k0 + 1];
                        let scv = _mm256_setr_ps(sa, sa, sa, sa, sb, sb, sb, sb);
                        let off = b * 256 + k0 * 16;
                        let xv = _mm256_loadu_si256(x.q[off..off + 32].as_ptr() as *const __m256i);
                        let pairs = _mm256_maddubs_epi16(q, xv);
                        let sums = _mm256_madd_epi16(pairs, ones);
                        let f = _mm256_cvtepi32_ps(sums);
                        let xs = x.s[b * 8 + k0 / 2];
                        vacc = _mm256_fmadd_ps(f, _mm256_mul_ps(scv, _mm256_set1_ps(xs)), vacc);
                        msum += 32.0
                            * (x.sum16[b * 16 + k0] as f32 * sa + x.sum16[b * 16 + k0 + 1] as f32 * sb)
                            * xs;
                    }
                }
                hsum(vacc) - msum
            })
            .collect(),
    }
}

fn pad_acts(acts: &mut Vec<Act>, cols: usize) {
    while acts.len() < TILE {
        acts.push(Act {
            q: vec![0i8; cols],
            s: vec![0.0f32; cols / 32],
            sum32: vec![0i32; cols / 32],
            sum16: vec![0i32; cols / 16],
        });
    }
}

#[allow(unreachable_code)]
fn matmul_batch(w: &Quant, acts: &[Act], out: &mut [f32]) {
    if w.raw.is_some() {
        let refs: Vec<&Act> = acts.iter().collect();
        raw_rows(w, &refs, out);
        return;
    }
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    {
        let n = acts.len();
        let blocks = w.cols / 256;
        let rows = w.rows;
        let qp: Vec<usize> = acts.iter().map(|a| a.q.as_ptr() as usize).collect();
        let sp: Vec<usize> = acts.iter().map(|a| a.s.as_ptr() as usize).collect();
        let up: Vec<usize> = acts.iter().map(|a| a.sum32.as_ptr() as usize).collect();
        let u16p: Vec<usize> = acts.iter().map(|a| a.sum16.as_ptr() as usize).collect();
        let dst = out.as_mut_ptr() as usize;
        match &w.data {
            Matrix::Raw(_) => unreachable!("raw handled earlier"),
            Matrix::Q4(data) => {
                (0..rows).into_par_iter().for_each(|row| unsafe {
                    let mut tmp = [0.0f32; TILE];
                    let wptr = data.as_ptr().add(row * blocks) as *const u8;
                    lfm_q4_rows(
                        wptr,
                        qp.as_ptr() as *const *const i8,
                        sp.as_ptr() as *const *const f32,
                        up.as_ptr() as *const *const i32,
                        blocks as i32,
                        n as i32,
                        tmp.as_mut_ptr(),
                    );
                    let p = dst as *mut f32;
                    for t in 0..n {
                        *p.add(t * rows + row) = tmp[t];
                    }
                });
            }
            Matrix::Q6(data) => {
                (0..rows).into_par_iter().for_each(|row| unsafe {
                    let mut tmp = [0.0f32; TILE];
                    let wptr = data.as_ptr().add(row * blocks) as *const u8;
                    lfm_q6_rows(
                        wptr,
                        qp.as_ptr() as *const *const i8,
                        sp.as_ptr() as *const *const f32,
                        u16p.as_ptr() as *const *const i32,
                        blocks as i32,
                        n as i32,
                        tmp.as_mut_ptr(),
                    );
                    let p = dst as *mut f32;
                    for t in 0..n {
                        *p.add(t * rows + row) = tmp[t];
                    }
                });
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() && std::env::var("LFM_CPU_SCALAR").is_err() {
            unsafe {
                matmul_batch_avx2(w, acts, out);
            }
            return;
        }
    }
    matmul_batch_generic(w, acts, out);
}

fn matmul_batch_generic(w: &Quant, acts: &[Act], out: &mut [f32]) {
    let n = acts.len();
    let rows = w.rows;
    debug_assert!(n == TILE && out.len() >= n * rows);
    let blocks = w.cols / 256;
    let ptr = out.as_mut_ptr() as usize;
    (0..rows).into_par_iter().for_each(|row| {
        let mut acc = [0.0f32; TILE];
        match &w.data {
            Matrix::Raw(_) => unreachable!("raw handled earlier"),
            Matrix::Q4(data) => {
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    for g in 0..8 {
                        let qg = blk.qgroup(g);
                        let q = &qg[..];
                        let sc = blk.scales[g];
                        let mn = blk.mins[g];
                        let off = b * 256 + g * 32;
                        for t in 0..n {
                            let dot = dot_u8_i8_32(q, &acts[t].q[off..off + 32]);
                            acc[t] += acts[t].s[b * 8 + g]
                                * (sc * dot as f32 - mn * acts[t].sum32[b * 8 + g] as f32);
                        }
                    }
                }
            }
            Matrix::Q6(data) => {
                let base = row * blocks;
                for b in 0..blocks {
                    let blk = &data[base + b];
                    for k in 0..16 {
                        let q = &blk.q[k * 16..k * 16 + 16];
                        let sc = blk.scales[k];
                        let off = b * 256 + k * 16;
                        for t in 0..n {
                            let dot = dot_u8_i8_16(q, &acts[t].q[off..off + 16]);
                            let sum = acts[t].sum16[b * 16 + k];
                            acc[t] += acts[t].s[b * 8 + k / 2] * sc * (dot - 32 * sum) as f32;
                        }
                    }
                }
            }
        }
        for t in 0..n {
            unsafe {
                *(ptr as *mut f32).add(t * rows + row) = acc[t];
            }
        }
    });
}

fn embed_row(w: &Quant, row: usize) -> Result<Vec<f32>> {
    let blocks = w.cols / 256;
    if let Matrix::Raw(RawKind::Q6) = w.data {
        let raw = w.raw_bytes().ok_or_else(|| anyhow!("raw q6 bytes missing"))?;
        let mut out = Vec::with_capacity(w.cols);
        for b in 0..blocks {
            let blk = &raw[(row * blocks + b) * 210..][..210];
            let d = half::f16::from_bits(u16::from_le_bytes([blk[208], blk[209]])).to_f32();
            for k in 0..16 {
                let sc = d * f32::from(blk[192 + k] as i8);
                for i in 0..16 {
                    out.push((f32::from(q6_raw_val(blk, k * 16 + i)) - 32.0) * sc);
                }
            }
        }
        return Ok(out);
    }
    match &w.data {
        Matrix::Q6(data) => {
            let base = row * blocks;
            let mut out = Vec::with_capacity(w.cols);
            for b in 0..blocks {
                let blk = &data[base + b];
                for k in 0..16 {
                    let scale = blk.scales[k];
                    for i in 0..16 {
                        out.push((f32::from(blk.q[k * 16 + i]) - 32.0) * scale);
                    }
                }
            }
            Ok(out)
        }
        _ => Err(anyhow!("embedding must be q6")),
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "x86_64")]
fn has_fma() -> bool {
    use std::sync::OnceLock;
    static FMA: OnceLock<bool> = OnceLock::new();
    *FMA.get_or_init(|| std::is_x86_feature_detected!("fma"))
}

fn dot_u8_i8_32(q: &[u8], x: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                return dot_u8_i8_32_avx2(q.as_ptr(), x.as_ptr());
            }
        }
    }
    let mut acc = 0i32;
    for i in 0..q.len() {
        acc += i32::from(q[i]) * i32::from(x[i]);
    }
    acc
}

fn dot_u8_i8_16(q: &[u8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for i in 0..q.len() {
        acc += i32::from(q[i]) * i32::from(x[i]);
    }
    acc
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            unsafe {
                return dot_f32_avx2(a.as_ptr(), b.as_ptr(), a.len());
            }
        }
    }
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_32_avx2(q: *const u8, x: *const i8) -> i32 {
    use std::arch::x86_64::*;
    let qv = _mm256_loadu_si256(q as *const __m256i);
    let xv = _mm256_loadu_si256(x as *const __m256i);
    let pairs = _mm256_maddubs_epi16(qv, xv);
    let ones = _mm256_set1_epi16(1);
    let sums = _mm256_madd_epi16(pairs, ones);
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sums);
    out.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm256_loadu_ps(b.add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * scale * w).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rope_inplace(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let freq = inv_freq.get(i).copied().unwrap_or(0.0) * pos as f32;
        let (s, c) = freq.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qblock(seed: u8) -> Vec<u8> {
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        for (i, slot) in b.iter_mut().enumerate().skip(4) {
            *slot = seed.wrapping_add((i as u8).wrapping_mul(17));
        }
        b
    }

    fn q6block(seed: u8) -> Vec<u8> {
        let mut b = vec![0u8; 210];
        b[208..210].copy_from_slice(&half::f16::from_f32(0.03).to_bits().to_le_bytes());
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = seed.wrapping_add((i as u8).wrapping_mul(29));
        }
        for i in 0..16 {
            b[192 + i] = (i as i8 - 8) as u8;
        }
        b
    }

    #[test]
    fn raw_q4_matches_prepacked() {
        let raw = qblock(5);
        let prep = prep_q4(&raw);
        let (sc, mn) = q4_raw_scales(&raw);
        for g in 0..8 {
            assert!((sc[g] - prep.scales[g]).abs() < 1e-6, "scale g {g}");
            assert!((mn[g] - prep.mins[g]).abs() < 1e-6, "min g {g}");
            for i in 0..32 {
                assert_eq!(
                    q4_raw_val(&raw[16..144], g, i),
                    prep.qgroup(g)[i],
                    "g {g} i {i}"
                );
            }
        }
    }

    #[test]
    fn raw_q6_matches_prepacked() {
        let raw = q6block(7);
        let prep = prep_q6(&raw);
        let d = half::f16::from_bits(u16::from_le_bytes([raw[208], raw[209]])).to_f32();
        for k in 0..16 {
            let sc = d * f32::from(raw[192 + k] as i8);
            assert!((sc - prep.scales[k]).abs() < 1e-6, "scale k {k}");
            for i in 0..16 {
                let v = k * 16 + i;
                let q_raw = f32::from(q6_raw_val(&raw, v)) - 32.0;
                let q_prep = f32::from(prep.q[v]) - 32.0;
                assert_eq!(q_raw, q_prep, "k {k} i {i}");
            }
        }
    }

    #[test]
    fn q4_prepack_matches_scalar_dequant() {
        let raw = qblock(3);
        let mut arr = [0u8; 144];
        arr.copy_from_slice(&raw);
        let expect = lfm_quant::dequant_q4_k(&arr);
        let prep = prep_q4(&raw);
        for g in 0..8 {
            for i in 0..32 {
                let idx = g * 32 + i;
                let qv = prep.qgroup(g)[i];
                let got = prep.scales[g] * f32::from(qv) - prep.mins[g];
                assert!((got - expect[idx]).abs() < 1e-4, "idx {idx}");
            }
        }
    }

    #[test]
    fn quantize_roundtrip_and_sums() {
        let x: Vec<f32> = (0..96).map(|i| (i as f32 * 0.37).sin()).collect();
        let a = quantize(&x);
        assert_eq!(a.q.len(), 96);
        assert_eq!(a.s.len(), 3);
        let sum32: i32 = a.q[0..32].iter().map(|v| i32::from(*v)).sum();
        assert_eq!(a.sum32[0], sum32);
        let sum16: i32 = a.q[16..32].iter().map(|v| i32::from(*v)).sum();
        assert_eq!(a.sum16[1], sum16);
        for i in 0..96 {
            let deq = f32::from(a.q[i]) * a.s[i / 32];
            assert!((deq - x[i]).abs() <= a.s[i / 32] + 1e-5);
        }
    }

    #[test]
    fn dot_avx2_matches_scalar() {
        let q: Vec<u8> = (0..32u8).map(|v| v % 16).collect();
        let x: Vec<i8> = (0..32).map(|i| ((i * 7) % 255) as i8).collect();
        let scalar: i32 = q.iter().zip(&x).map(|(a, b)| i32::from(*a) * i32::from(*b)).sum();
        assert_eq!(dot_u8_i8_32(&q, &x), scalar);
    }
}
