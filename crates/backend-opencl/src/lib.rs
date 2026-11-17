use anyhow::{anyhow, Result};
use lfm_app::Sampler;
use lfm_domain::{LayerKind, ModelConfig, SamplingParams};
use lfm_gguf::{GgmlType, GgufFile};
use lfm_opencl::{Arg, Buffer, ClMem, Gpu, Kernel, Program};
use std::os::raw::c_void;
type ClMemRaw = *mut c_void;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

mod prepack;
mod ports;
mod radix;

pub use radix::{block_hash, chain_hashes, RadixCache, PAGE};

pub fn chunk_ranges(len: usize, batch: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let batch = batch.max(1);
    let mut pos = 0;
    while pos < len {
        let end = (pos + batch).min(len);
        out.push((pos, end));
        pos = end;
    }
    out
}

pub fn attention_splits(span: usize) -> usize {
    (span / 512).clamp(1, MAX_SPLITS)
}

pub fn reuse_cap(prompt_len: usize) -> usize {
    prompt_len.saturating_sub(1) / KV_PAGE
}

const KERNELS: &str = concat!(
    include_str!("../kernels/decode.cl"),
    "\n",
    include_str!("../kernels/batch.cl")
);
const GROUP: usize = 32;
const MAX_SPLITS: usize = 16;
const KV_PAGE: usize = PAGE;

enum Weight {
    Q4(Buffer<u8>),
    Q6(Buffer<u8>),
    F32(Buffer<f32>),
}

enum KvBuf {
    F32(Buffer<f32>),
    F16(Buffer<u16>),
}

impl KvBuf {
    fn mem(&self) -> ClMemRaw {
        match self {
            KvBuf::F32(b) => b.get(),
            KvBuf::F16(b) => b.get(),
        }
    }
}

struct Kernels {
    embed: Kernel,
    rms: Kernel,
    rms_add: Kernel,
    quant: Kernel,
    q4: Kernel,
    q6: Kernel,
    silu: Kernel,
    conv: Kernel,
    attn: Kernel,
    attn_split: Kernel,
    attn_combine: Kernel,
    attn_h: Kernel,
    attn_split_h: Kernel,
    kv_store: Kernel,
    kv_store_h: Kernel,
    fill: Kernel,
    fill_h: Kernel,
    rms_quant: Kernel,
    norm_add_quant: Kernel,
    silu_quant: Kernel,
    rms_rope_qk: Kernel,
    embed_b: Kernel,
    rms_quant_b: Kernel,
    norm_add_quant_b: Kernel,
    silu_quant_b: Kernel,
    quant_b: Kernel,
    q4_b: Kernel,
    q4p_b: Kernel,
    q4x4p_b: Kernel,
    q6_b: Kernel,
    conv_b: Kernel,
    rms_rope_heads_b: Kernel,
    kv_store_b: Kernel,
    kv_store_b_h: Kernel,
    attn_b: Kernel,
    attn_split_b: Kernel,
    attn_combine_b: Kernel,
    attn_b_h: Kernel,
    attn_b_h2: Kernel,
    attn_split_b_h: Kernel,
    attn_split_b_h4: Kernel,
}

struct Bufs {
    x: Buffer<f32>,
    h: Buffer<f32>,
    hq: Buffer<i8>,
    hs: Buffer<f32>,
    op: Buffer<f32>,
    tmp: Buffer<f32>,
    ug: Buffer<f32>,
    act: Buffer<f32>,
    actq: Buffer<i8>,
    acts: Buffer<f32>,
    bcx: Buffer<f32>,
    q: Buffer<f32>,
    kk: Buffer<f32>,
    vv: Buffer<f32>,
    g16: Buffer<f32>,
    attn: Buffer<f32>,
    logits: Buffer<f32>,
    tokens: Buffer<u32>,
    xb: Buffer<f32>,
    opb: Buffer<f32>,
    tmpb: Buffer<f32>,
    hqx: Buffer<i8>,
    hsx: Buffer<f32>,
    ugb: Buffer<f32>,
    actqb: Buffer<i8>,
    actsb: Buffer<f32>,
    bcxb: Buffer<f32>,
    qb: Buffer<f32>,
    kkb: Buffer<f32>,
    vvb: Buffer<f32>,
    gb: Buffer<f32>,
    attnb: Buffer<f32>,
    logitsb: Buffer<f32>,
    xqg: Buffer<i8>,
    xsg: Buffer<f32>,
    logitsg: Buffer<f32>,
    pos_b: Buffer<u32>,
    tbl_off_b: Buffer<u32>,
    conv_off_b: Buffer<u32>,
    tbl_all: Buffer<u32>,
    snap_page: Buffer<u32>,
    snap_dummy: Buffer<f32>,
    pacc: Buffer<f32>,
    pml: Buffer<f32>,
    pacc_b: Buffer<f32>,
    pml_b: Buffer<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct StepReq {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
}

pub struct Sequence {
    pub slot: usize,
    pub len: usize,
    tbl: Buffer<u32>,
    block_table: Vec<u32>,
    tbl_dirty: bool,
    pending: Option<PendingPrefill>,
}

struct PendingPrefill {
    hashes: Vec<u64>,
    reused_pages: usize,
}

struct LayerBufs {
    conv: Option<Buffer<f32>>,
    snap: Option<Buffer<f32>>,
    k: Option<KvBuf>,
    v: Option<KvBuf>,
}

pub struct GpuModel {
    gpu: Gpu,
    pub cfg: ModelConfig,
    pub max_ctx: usize,
    weights: HashMap<String, Weight>,
    prepacked: HashMap<String, Buffer<u8>>,
    prepack_on: bool,
    kernels: Kernels,
    rope_full: Buffer<f32>,
    rope_sliding: Buffer<f32>,
    bufs: Bufs,
    layers: Vec<LayerBufs>,
    ffn: usize,
    pb: usize,
    rgemv: usize,
    rgemv_b: usize,
    wg: usize,
    wg_b: usize,
    kv_blocks: usize,
    cache: RadixCache,
    cache_on: bool,
    kv_f16: bool,
    attn4: bool,
    attn2: bool,
    max_seqs: usize,
    seqs: Vec<Option<Sequence>>,
    logits_off: Option<usize>,
    pub trace: bool,
    pub traced: Vec<Vec<f32>>,
}

fn pick_logits_rows(
    need_logits: bool,
    logits_rows: Option<&[usize]>,
    logits_row: Option<usize>,
    ntok: usize,
) -> Result<Option<Vec<usize>>> {
    if !need_logits {
        return Ok(None);
    }
    let mut sel: Vec<usize> = match (logits_rows, logits_row) {
        (Some(rows), _) => rows.to_vec(),
        (None, Some(r)) => vec![r],
        (None, None) => return Ok(Some(Vec::new())),
    };
    if sel.is_empty() {
        return Err(anyhow!("empty logits row selection"));
    }
    sel.sort_unstable();
    sel.dedup();
    if let Some(&last) = sel.last() {
        if last >= ntok {
            return Err(anyhow!("logits row {last} outside batch of {ntok}"));
        }
    }
    Ok(Some(sel))
}

impl GpuModel {
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        let profiling = std::env::var("LFM_PROFILE").map(|v| v == "1").unwrap_or(false);
        let gpu = Gpu::open_with(0, profiling)?;
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let prepack_on = std::env::var("LFM_PREPACK").map(|v| v != "0").unwrap_or(true);
        let fuse_on = std::env::var("LFM_FUSE").map(|v| v != "0").unwrap_or(true);
        let mut weights = HashMap::new();
        let mut prepacked: HashMap<String, Buffer<u8>> = HashMap::new();
        for info in file.tensors() {
            let bytes = file.tensor_bytes(&info.name)?;
            let weight = match info.ggml_type {
                GgmlType::Q4_K => {
                    if prepack_on {
                        let packed = prepack::prepack_q4_k(bytes);
                        prepacked.insert(info.name.clone(), gpu.upload_bytes(&packed)?);
                    }
                    Weight::Q4(gpu.upload_bytes(bytes)?)
                }
                GgmlType::Q6_K => Weight::Q6(gpu.upload_bytes(bytes)?),
                GgmlType::F32 => {
                    let data: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Weight::F32(gpu.upload(&data)?)
                }
                _ => continue,
            };
            weights.insert(info.name.clone(), weight);
        }
        for layer in 0..cfg.num_layers {
            let up_name = format!("blk.{layer}.ffn_up.weight");
            let gate_name = format!("blk.{layer}.ffn_gate.weight");
            let up_info = file.tensor(&up_name);
            let gate_info = file.tensor(&gate_name);
            if let (Some(up_info), Some(gate_info)) = (up_info, gate_info) {
                if up_info.ggml_type == gate_info.ggml_type
                    && up_info.byte_size == gate_info.byte_size
                {
                    let gate_bytes = file.tensor_bytes(&gate_name)?;
                    let up_bytes = file.tensor_bytes(&up_name)?;
                    if prepack_on && up_info.ggml_type == GgmlType::Q4_K {
                        let mut fused = prepack::prepack_q4_k(&gate_bytes);
                        fused.extend_from_slice(&prepack::prepack_q4_k(&up_bytes));
                        prepacked.insert(
                            format!("blk.{layer}.ffn_ug.weight"),
                            gpu.upload_bytes(&fused)?,
                        );
                    }
                    let mut fused = gpu.alloc::<u8>(gate_bytes.len() + up_bytes.len())?;
                    unsafe {
                        gpu.queue.enqueue_write_buffer(&mut fused, 1, 0, gate_bytes, &[])?;
                        gpu.queue.enqueue_write_buffer(&mut fused, 1, gate_bytes.len(), up_bytes, &[])?;
                    }
                    let weight = match up_info.ggml_type {
                        GgmlType::Q4_K => Weight::Q4(fused),
                        GgmlType::Q6_K => Weight::Q6(fused),
                        _ => continue,
                    };
                    weights.insert(format!("blk.{layer}.ffn_ug.weight"), weight);
                }
            }
            if !prepack_on || !fuse_on {
                continue;
            }
            let q_name = format!("blk.{layer}.attn_q.weight");
            let k_name = format!("blk.{layer}.attn_k.weight");
            let v_name = format!("blk.{layer}.attn_v.weight");
            let g_name = format!("blk.{layer}.attn_gate.weight");
            let q_info = file.tensor(&q_name);
            let k_info = file.tensor(&k_name);
            let g_info = file.tensor(&g_name);
            if let (Some(q_info), Some(k_info), Some(g_info)) = (q_info, k_info, g_info) {
                let q4 = |info: &lfm_gguf::TensorInfo| info.ggml_type == GgmlType::Q4_K;
                if q4(q_info) && q4(k_info) && q4(g_info) && q_info.dims[0] == k_info.dims[0]
                    && q_info.dims[0] == g_info.dims[0]
                    && q_info.dims[1] % 4 == 0
                    && k_info.dims[1] % 4 == 0
                    && g_info.dims[1] % 4 == 0
                {
                    let q_bytes = file.tensor_bytes(&q_name)?;
                    let k_bytes = file.tensor_bytes(&k_name)?;
                    let g_bytes = file.tensor_bytes(&g_name)?;
                    let v_info = file.tensor(&v_name);
                    let fuse_v = v_info.map(q4).unwrap_or(false)
                        && v_info.map(|i| i.dims[0] == q_info.dims[0]).unwrap_or(false)
                        && v_info.map(|i| i.dims[1] % 4 == 0).unwrap_or(false);
                    if fuse_v {
                        let v_bytes = file.tensor_bytes(&v_name)?;
                        let mut fused = prepack::prepack_q4_k(&q_bytes);
                        fused.extend_from_slice(&prepack::prepack_q4_k(&k_bytes));
                        fused.extend_from_slice(&prepack::prepack_q4_k(&v_bytes));
                        fused.extend_from_slice(&prepack::prepack_q4_k(&g_bytes));
                        prepacked.insert(
                            format!("blk.{layer}.attn_qkvg.weight"),
                            gpu.upload_bytes(&fused)?,
                        );
                    }
                    let mut fused = prepack::prepack_q4_k(&q_bytes);
                    fused.extend_from_slice(&prepack::prepack_q4_k(&k_bytes));
                    fused.extend_from_slice(&prepack::prepack_q4_k(&g_bytes));
                    prepacked.insert(
                        format!("blk.{layer}.attn_qkg.weight"),
                        gpu.upload_bytes(&fused)?,
                    );
                }
            }
        }
        let pb: usize = std::env::var("LFM_PB")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (1..=12).contains(v))
            .unwrap_or(6);
        let rgemv: usize = std::env::var("LFM_RGEMV")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (1..=8).contains(v))
            .unwrap_or(4);
        let rgemv_b: usize = std::env::var("LFM_RGEMV_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (1..=8).contains(v))
            .unwrap_or(5);
        let wg: usize = std::env::var("LFM_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (64..=512).contains(v))
            .map(|v| v / 64 * 64)
            .unwrap_or(256);
        let wg_b: usize = std::env::var("LFM_WG_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (64..=512).contains(v))
            .map(|v| v / 64 * 64)
            .unwrap_or(64);
        let options = format!("-cl-std=CL3.0 -DRGEMV={rgemv} -DRGEMV_B={rgemv_b} -DPB={pb}");
        let program: Program = gpu.build(KERNELS, &options)?;
        let kernels = Kernels {
            embed: Kernel::create(&program, "embed_q6k")?,
            rms: Kernel::create(&program, "rms_norm")?,
            rms_add: Kernel::create(&program, "rms_norm_add")?,
            quant: Kernel::create(&program, "quantize_q8")?,
            q4: Kernel::create(&program, "gemv_q4_k_i8")?,
            q6: Kernel::create(&program, "gemv_q6_k_i8")?,
            silu: Kernel::create(&program, "silu_mul")?,
            conv: Kernel::create(&program, "conv_step")?,
            attn: Kernel::create(&program, "attention")?,
            attn_split: Kernel::create(&program, "attention_split")?,
            attn_combine: Kernel::create(&program, "attention_combine")?,
            attn_h: Kernel::create(&program, "attention_h")?,
            attn_split_h: Kernel::create(&program, "attention_split_h")?,
            kv_store: Kernel::create(&program, "kv_store2")?,
            kv_store_h: Kernel::create(&program, "kv_store_h")?,
            fill: Kernel::create(&program, "fill_zero")?,
            fill_h: Kernel::create(&program, "fill_zero_h")?,
            rms_quant: Kernel::create(&program, "rms_quant")?,
            norm_add_quant: Kernel::create(&program, "norm_add_quant")?,
            silu_quant: Kernel::create(&program, "silu_quant")?,
            rms_rope_qk: Kernel::create(&program, "rms_rope_qk")?,
            embed_b: Kernel::create(&program, "embed_q6k_b")?,
            rms_quant_b: Kernel::create(&program, "rms_quant_b")?,
            norm_add_quant_b: Kernel::create(&program, "norm_add_quant_b")?,
            silu_quant_b: Kernel::create(&program, "silu_quant_b")?,
            quant_b: Kernel::create(&program, "quantize_b")?,
            q4_b: Kernel::create(&program, "gemv_q4_k_i8_b")?,
            q4p_b: Kernel::create(&program, "gemv_q4_k_i8_p_b")?,
            q4x4p_b: Kernel::create(&program, "gemv_q4_k_i8_4p_b")?,
            q6_b: Kernel::create(&program, "gemv_q6_k_i8_b")?,
            conv_b: Kernel::create(&program, "conv_step_b")?,
            rms_rope_heads_b: Kernel::create(&program, "rms_rope_heads_b")?,
            kv_store_b: Kernel::create(&program, "kv_store_b")?,
            kv_store_b_h: Kernel::create(&program, "kv_store_b_h")?,
            attn_b: Kernel::create(&program, "attention_b")?,
            attn_split_b: Kernel::create(&program, "attention_split_b")?,
            attn_combine_b: Kernel::create(&program, "attention_combine_b")?,
            attn_b_h: Kernel::create(&program, "attention_b_h")?,
            attn_b_h2: Kernel::create(&program, "attention_b_h2")?,
            attn_split_b_h: Kernel::create(&program, "attention_split_b_h")?,
            attn_split_b_h4: Kernel::create(&program, "attention_split_b_h4")?,
        };
        let h = cfg.hidden_size;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let ffn = cfg.ffn_size;
        let vocab = cfg.vocab_size;
        let kv_blocks = max_ctx.div_ceil(KV_PAGE);
        let kv_slots = kv_blocks * KV_PAGE;
        let max_seqs: usize = std::env::var("LFM_SEQS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (1..=16).contains(v))
            .unwrap_or(8);
        let cache_on = std::env::var("LFM_CACHE").map(|v| v != "0").unwrap_or(true);
        let kv_f16 = gpu.caps.fp16
            && std::env::var("LFM_KV")
                .map(|v| v != "f32")
                .unwrap_or(true);
        let attn4 = std::env::var("LFM_ATTN4").map(|v| v != "0").unwrap_or(true);
        let attn2 = std::env::var("LFM_ATTN2").map(|v| v != "0").unwrap_or(true);
        let bufs = Bufs {
            x: gpu.alloc::<f32>(h)?,
            h: gpu.alloc::<f32>(h)?,
            hq: gpu.alloc::<i8>(ffn.max(h))?,
            hs: gpu.alloc::<f32>(ffn.max(h) / GROUP + 1)?,
            op: gpu.alloc::<f32>(h)?,
            tmp: gpu.alloc::<f32>(h.max(ffn))?,
            ug: gpu.alloc::<f32>(2 * ffn)?,
            act: gpu.alloc::<f32>(ffn)?,
            actq: gpu.alloc::<i8>(ffn)?,
            acts: gpu.alloc::<f32>(ffn / GROUP)?,
            bcx: gpu.alloc::<f32>(3 * h)?,
            q: gpu.alloc::<f32>(qd)?,
            kk: gpu.alloc::<f32>(kvd)?,
            vv: gpu.alloc::<f32>(kvd)?,
            g16: gpu.alloc::<f32>(cfg.num_heads)?,
            attn: gpu.alloc::<f32>(qd)?,
            logits: gpu.alloc::<f32>(vocab)?,
            tokens: gpu.alloc::<u32>(pb)?,
            xb: gpu.alloc::<f32>(pb * h)?,
            opb: gpu.alloc::<f32>(pb * h)?,
            tmpb: gpu.alloc::<f32>(pb * h)?,
            hqx: gpu.alloc::<i8>(pb * h)?,
            hsx: gpu.alloc::<f32>(pb * (h / GROUP))?,
            ugb: gpu.alloc::<f32>(pb * 2 * ffn)?,
            actqb: gpu.alloc::<i8>(pb * ffn)?,
            actsb: gpu.alloc::<f32>(pb * (ffn / GROUP))?,
            bcxb: gpu.alloc::<f32>(pb * 3 * h)?,
            qb: gpu.alloc::<f32>(pb * qd)?,
            kkb: gpu.alloc::<f32>(pb * kvd)?,
            vvb: gpu.alloc::<f32>(pb * kvd)?,
            gb: gpu.alloc::<f32>(pb * cfg.num_heads)?,
            attnb: gpu.alloc::<f32>(pb * qd)?,
            logitsb: gpu.alloc::<f32>(pb * vocab)?,
            xqg: gpu.alloc::<i8>(pb * h)?,
            xsg: gpu.alloc::<f32>(pb * (h / GROUP))?,
            logitsg: gpu.alloc::<f32>(pb * vocab)?,
            pos_b: gpu.alloc::<u32>(pb)?,
            tbl_off_b: gpu.alloc::<u32>(pb)?,
            conv_off_b: gpu.alloc::<u32>(pb)?,
            tbl_all: gpu.alloc::<u32>(max_seqs * kv_blocks.max(1))?,
            snap_page: gpu.alloc::<u32>(pb)?,
            snap_dummy: gpu.alloc::<f32>(1)?,
            pacc: gpu.alloc::<f32>(MAX_SPLITS * cfg.num_heads * cfg.head_dim)?,
            pml: gpu.alloc::<f32>(MAX_SPLITS * cfg.num_heads * 2)?,
            pacc_b: gpu.alloc::<f32>(pb * MAX_SPLITS * cfg.num_heads * cfg.head_dim)?,
            pml_b: gpu.alloc::<f32>(pb * MAX_SPLITS * cfg.num_heads * 2)?,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer = match cfg.kind(i) {
                LayerKind::Conv => LayerBufs {
                    conv: Some(gpu.alloc::<f32>(max_seqs * (cfg.conv_cache - 1) * h)?),
                    snap: if cache_on {
                        Some(gpu.alloc::<f32>(kv_blocks * (cfg.conv_cache - 1) * h)?)
                    } else {
                        None
                    },
                    k: None,
                    v: None,
                },
                _ => LayerBufs {
                    conv: None,
                    snap: None,
                    k: Some(if kv_f16 {
                        KvBuf::F16(gpu.alloc::<u16>(kv_slots * kvd)?)
                    } else {
                        KvBuf::F32(gpu.alloc::<f32>(kv_slots * kvd)?)
                    }),
                    v: Some(if kv_f16 {
                        KvBuf::F16(gpu.alloc::<u16>(kv_slots * kvd)?)
                    } else {
                        KvBuf::F32(gpu.alloc::<f32>(kv_slots * kvd)?)
                    }),
                },
            };
            layers.push(layer);
        }
        let rope_full = gpu.upload(&cfg.rope_full.inv_freq())?;
        let rope_sliding = gpu.upload(&cfg.rope_sliding.inv_freq())?;
        let mut model = Self {
            gpu,
            cfg,
            max_ctx,
            weights,
            prepacked,
            prepack_on,
            kernels,
            rope_full,
            rope_sliding,
            bufs,
            layers,
            ffn,
            pb,
            rgemv,
            rgemv_b,
            wg,
            wg_b,
            kv_blocks,
            cache: RadixCache::new(kv_blocks),
            cache_on,
            kv_f16,
            attn4,
            attn2,
            max_seqs,
            seqs: (0..max_seqs).map(|_| None).collect(),
            logits_off: None,
            trace: false,
            traced: Vec::new(),
        };
        model.reset()?;
        Ok(model)
    }

    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("missing weight {name}"))
    }

    fn gemv_auto(
        &self,
        name: &str,
        xq: &Buffer<i8>,
        xs: &Buffer<f32>,
        y: &Buffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let weight = self.weight(name)?;
        let kernel = match weight {
            Weight::Q4(_) => &self.kernels.q4,
            Weight::Q6(_) => &self.kernels.q6,
            Weight::F32(_) => return Err(anyhow!("gemv on f32 weight {name}")),
        };
        let buffer = match weight {
            Weight::Q4(b) | Weight::Q6(b) => b,
            Weight::F32(_) => unreachable!(),
        };
        let per_wg = (self.wg / 64) * self.rgemv;
        let grid = rows.div_ceil(per_wg) * self.wg;
        unsafe {
            self.gpu.launch(
                kernel,
                grid,
                self.wg,
                &[
                    Arg::Mem(buffer.get()),
                    Arg::Mem(xq.get()),
                    Arg::Mem(xs.get()),
                    Arg::Mem(y.get()),
                    Arg::U32(rows as u32),
                    Arg::U32(cols as u32),
                ],
            )?;
        }
        Ok(())
    }

    fn gemv_auto_b(
        &self,
        name: &str,
        xq: &Buffer<i8>,
        xs: &Buffer<f32>,
        y: &Buffer<f32>,
        rows: usize,
        cols: usize,
        ntok: usize,
        tok0: usize,
    ) -> Result<()> {
        let weight = self.weight(name)?;
        let packed = if self.prepack_on {
            self.prepacked.get(name)
        } else {
            None
        };
        let (kernel, buffer) = match (weight, packed) {
            (Weight::Q4(_), Some(p)) => (&self.kernels.q4p_b, p),
            (Weight::Q4(b), None) => (&self.kernels.q4_b, b),
            (Weight::Q6(b), _) => (&self.kernels.q6_b, b),
            (Weight::F32(_), _) => return Err(anyhow!("gemv on f32 weight {name}")),
        };
        let per_wg = (self.wg_b / 64) * self.rgemv_b;
        let grid = rows.div_ceil(per_wg) * self.wg_b;
        unsafe {
            self.gpu.launch(
                kernel,
                grid,
                self.wg_b,
                &[
                    Arg::Mem(buffer.get()),
                    Arg::Mem(xq.get()),
                    Arg::Mem(xs.get()),
                    Arg::Mem(y.get()),
                    Arg::U32(rows as u32),
                    Arg::U32(cols as u32),
                    Arg::U32(ntok as u32),
                    Arg::U32(tok0 as u32),
                ],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_auto4_b(
        &self,
        name: &str,
        xq: &Buffer<i8>,
        xs: &Buffer<f32>,
        y1: &Buffer<f32>,
        y2: &Buffer<f32>,
        y3: &Buffer<f32>,
        y4: &Buffer<f32>,
        r1: usize,
        r2: usize,
        r3: usize,
        r4: usize,
        cols: usize,
        ntok: usize,
    ) -> Result<()> {
        let packed = self
            .prepacked
            .get(name)
            .ok_or_else(|| anyhow!("missing fused weight {name}"))?;
        let rows = r1 + r2 + r3 + r4;
        let per_wg = (self.wg_b / 64) * 4;
        let grid = rows.div_ceil(per_wg) * self.wg_b;
        unsafe {
            self.gpu.launch(
                &self.kernels.q4x4p_b,
                grid,
                self.wg_b,
                &[
                    Arg::Mem(packed.get()),
                    Arg::Mem(xq.get()),
                    Arg::Mem(xs.get()),
                    Arg::Mem(y1.get()),
                    Arg::Mem(y2.get()),
                    Arg::Mem(y3.get()),
                    Arg::Mem(y4.get()),
                    Arg::U32(r1 as u32),
                    Arg::U32(r2 as u32),
                    Arg::U32(r3 as u32),
                    Arg::U32(r4 as u32),
                    Arg::U32(cols as u32),
                    Arg::U32(ntok as u32),
                ],
            )?;
        }
        Ok(())
    }

    fn quantize(&self, x: &Buffer<f32>, xq: &Buffer<i8>, xs: &Buffer<f32>, n: usize) -> Result<()> {        unsafe {
            let groups = n / GROUP;
            let grid = groups.div_ceil(16) * 256;
            self.gpu.launch(
                &self.kernels.quant,
                grid,
                256,
                &[
                    Arg::Mem(x.get()),
                    Arg::Mem(xq.get()),
                    Arg::Mem(xs.get()),
                    Arg::U32(n as u32),
                    Arg::U32(GROUP as u32),
                ],
            )?;
        }
        Ok(())
    }

    fn rms(&self, x: &Buffer<f32>, w: &Buffer<f32>, out: &Buffer<f32>, n: usize) -> Result<()> {
        unsafe {
            self.gpu.launch(
                &self.kernels.rms,
                64,
                64,
                &[
                    Arg::Mem(x.get()),
                    Arg::Mem(w.get()),
                    Arg::Mem(out.get()),
                    Arg::U32(n as u32),
                ],
            )?;
        }
        Ok(())
    }

    fn rms_add(
        &self,
        x: &Buffer<f32>,
        w: &Buffer<f32>,
        res: &Buffer<f32>,
        out: &Buffer<f32>,
        n: usize,
    ) -> Result<()> {
        unsafe {
            self.gpu.launch(
                &self.kernels.rms_add,
                64,
                64,
                &[
                    Arg::Mem(x.get()),
                    Arg::Mem(w.get()),
                    Arg::Mem(res.get()),
                    Arg::Mem(out.get()),
                    Arg::U32(n as u32),
                ],
            )?;
        }
        Ok(())
    }

    fn norm_add_quant(
        &self,
        op: &Buffer<f32>,
        post: &Buffer<f32>,
        res: &Buffer<f32>,
        next_w: &Buffer<f32>,
        n: usize,
    ) -> Result<()> {
        unsafe {
            self.gpu.launch(
                &self.kernels.norm_add_quant,
                256,
                256,
                &[
                    Arg::Mem(op.get()),
                    Arg::Mem(post.get()),
                    Arg::Mem(res.get()),
                    Arg::Mem(next_w.get()),
                    Arg::Mem(self.bufs.x.get()),
                    Arg::Mem(self.bufs.hq.get()),
                    Arg::Mem(self.bufs.hs.get()),
                    Arg::U32(n as u32),
                ],
            )?;
        }
        Ok(())
    }

    pub fn create_sequence(&mut self) -> Result<usize> {
        let slot = self
            .seqs
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| anyhow!("no free sequence slots (max {})", self.max_seqs))?;
        let tbl = self.gpu.alloc::<u32>(self.kv_blocks.max(1))?;
        self.clear_conv_slot(slot)?;
        self.seqs[slot] = Some(Sequence {
            slot,
            len: 0,
            tbl,
            block_table: Vec::new(),
            tbl_dirty: false,
            pending: None,
        });
        Ok(slot)
    }

    pub fn free_sequence(&mut self, slot: usize) -> Result<()> {
        if let Some(seq) = self.seqs.get_mut(slot).and_then(|s| s.take()) {
            for block in seq.block_table {
                self.cache.release(block as usize);
            }
        }
        Ok(())
    }

    fn clear_conv_slot(&mut self, slot: usize) -> Result<()> {
        let conv_len = (self.cfg.conv_cache - 1) * self.cfg.hidden_size;
        let off = (slot * conv_len) as u32;
        for layer in self.layers.iter() {
            if let Some(conv) = &layer.conv {
                unsafe {
                    self.gpu.launch(
                        &self.kernels.fill,
                        conv_len,
                        256,
                        &[Arg::Mem(conv.get()), Arg::U32(off), Arg::U32(conv_len as u32)],
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn sequence(&self, slot: usize) -> Result<&Sequence> {
        self.seqs
            .get(slot)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| anyhow!("sequence slot {slot} is empty"))
    }

    fn seq_ensure(&mut self, slot: usize, need: usize) -> Result<()> {
        let kv_blocks = self.kv_blocks;
        let want = need.div_ceil(KV_PAGE);
        loop {
            let have = match self.seqs.get(slot).and_then(|s| s.as_ref()) {
                Some(seq) => seq.block_table.len(),
                None => return Err(anyhow!("sequence slot {slot} is empty")),
            };
            if have >= want {
                return Ok(());
            }
            let block = self
                .cache
                .alloc()
                .ok_or_else(|| anyhow!("KV pool exhausted: {} pages", kv_blocks))?;
            if let Some(seq) = self.seqs[slot].as_mut() {
                seq.block_table.push(block as u32);
                seq.tbl_dirty = true;
            }
        }
    }

    fn seq_sync(&mut self, slot: usize) -> Result<()> {
        let seq = self
            .seqs
            .get_mut(slot)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| anyhow!("sequence slot {slot} is empty"))?;
        if !seq.tbl_dirty {
            return Ok(());
        }
        unsafe {
            self.gpu.queue.enqueue_write_buffer(
                &mut seq.tbl,
                1,
                0,
                &seq.block_table[..],
                &[],
            )?;
        }
        seq.tbl_dirty = false;
        Ok(())
    }

    fn seq_launch_params(&mut self, slot: usize, pos_end: usize) -> Result<(ClMemRaw, usize)> {
        self.seq_ensure(slot, pos_end)?;
        self.seq_sync(slot)?;
        let seq = self.sequence(slot)?;
        let tbl_mem = seq.tbl.get();
        let conv_off = slot * (self.cfg.conv_cache - 1) * self.cfg.hidden_size;
        if let Some(seq) = self.seqs[slot].as_mut() {
            seq.len = seq.len.max(pos_end);
        }
        Ok((tbl_mem, conv_off))
    }

    fn f32_weight(&self, name: &str) -> Result<&Buffer<f32>> {
        match self.weight(name)? {
            Weight::F32(b) => Ok(b),
            _ => Err(anyhow!("{name} is not f32")),
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        self.forward_seq(0, token, pos)
    }

    pub fn forward_seq(&mut self, slot: usize, token: u32, pos: usize) -> Result<()> {
        if pos >= self.max_ctx {
            return Err(anyhow!(
                "context overflow: pos {pos} >= max_ctx {}",
                self.max_ctx
            ));
        }
        let (tbl_mem, conv_off) = self.seq_launch_params(slot, pos + 1)?;
        self.logits_off = None;
        let h = self.cfg.hidden_size;
        let qd = self.cfg.q_dim();
        let kvd = self.cfg.kv_dim();
        let heads = self.cfg.num_heads;
        let head_dim = self.cfg.head_dim;
        let emb = match self.weight("token_embd.weight")? {
            Weight::Q6(b) => b,
            _ => return Err(anyhow!("token_embd must be Q6_K for gpu embed")),
        };
        unsafe {
            self.gpu.launch(
                &self.kernels.embed,
                h,
                64,
                &[
                    Arg::Mem(emb.get()),
                    Arg::Mem(self.bufs.x.get()),
                    Arg::U32(token),
                    Arg::U32(h as u32),
                ],
            )?;
        }
        for layer in 0..self.cfg.num_layers {
            let prefix = format!("blk.{layer}");
            if layer == 0 {
                let attn_norm = self.f32_weight(&format!("{prefix}.attn_norm.weight"))?;
                unsafe {
                    self.gpu.launch(
                        &self.kernels.rms_quant,
                        64,
                        64,
                        &[
                            Arg::Mem(self.bufs.x.get()),
                            Arg::Mem(attn_norm.get()),
                            Arg::Mem(self.bufs.hq.get()),
                            Arg::Mem(self.bufs.hs.get()),
                            Arg::U32(h as u32),
                        ],
                    )?;
                }
            }
            match self.cfg.kind(layer) {
                LayerKind::Conv => {
                    self.gemv_auto(
                        &format!("{prefix}.shortconv.in_proj.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.bcx,
                        3 * h,
                        h,
                    )?;
                    let conv_w = self.f32_weight(&format!("{prefix}.shortconv.conv.weight"))?;
                    let state = self.layers[layer]
                        .conv
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing conv state"))?;
                    unsafe {
                        self.gpu.launch(
                            &self.kernels.conv,
                            h,
                            256,
                            &[
                                Arg::Mem(self.bufs.bcx.get()),
                                Arg::Mem(conv_w.get()),
                                Arg::Mem(state.get()),
                                Arg::Mem(self.bufs.op.get()),
                                Arg::U32(conv_off as u32),
                                Arg::U32(h as u32),
                                Arg::U32(self.cfg.conv_cache as u32),
                            ],
                        )?;
                    }
                    self.quantize(&self.bufs.op, &self.bufs.hq, &self.bufs.hs, h)?;
                    self.gemv_auto(
                        &format!("{prefix}.shortconv.out_proj.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.tmp,
                        h,
                        h,
                    )?;
                }
                _ => {
                    let kv_heads = self.cfg.kv_heads_of(layer);
                    self.gemv_auto(
                        &format!("{prefix}.attn_q.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.q,
                        qd,
                        h,
                    )?;
                    self.gemv_auto(
                        &format!("{prefix}.attn_k.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.kk,
                        kv_heads * head_dim,
                        h,
                    )?;
                    self.gemv_auto(
                        &format!("{prefix}.attn_v.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.vv,
                        kv_heads * head_dim,
                        h,
                    )?;
                    self.gemv_auto(
                        &format!("{prefix}.attn_gate.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.g16,
                        heads,
                        h,
                    )?;
                    let q_norm = self.f32_weight(&format!("{prefix}.attn_q_norm.weight"))?;
                    let k_norm = self.f32_weight(&format!("{prefix}.attn_k_norm.weight"))?;
                    let inv = if self.cfg.kind(layer) == LayerKind::SlidingAttention {
                        &self.rope_sliding
                    } else {
                        &self.rope_full
                    };
                    unsafe {
                        self.gpu.launch(
                            &self.kernels.rms_rope_qk,
                            (heads + kv_heads) * 64,
                            64,
                            &[
                                Arg::Mem(self.bufs.q.get()),
                                Arg::Mem(self.bufs.kk.get()),
                                Arg::Mem(q_norm.get()),
                                Arg::Mem(k_norm.get()),
                                Arg::Mem(inv.get()),
                                Arg::U32(pos as u32),
                                Arg::U32(head_dim as u32),
                                Arg::U32(heads as u32),
                                Arg::U32(kv_heads as u32),
                            ],
                        )?;
                    }
                    let (k_buf, v_buf) = {
                        let lb = &self.layers[layer];
                        (
                            lb.k.as_ref().ok_or_else(|| anyhow!("missing kv"))?,
                            lb.v.as_ref().ok_or_else(|| anyhow!("missing kv"))?,
                        )
                    };
                    unsafe {
                        self.gpu.launch(
                            self.kv_store_kernel(),
                            kvd,
                            256,
                            &[
                                Arg::Mem(self.bufs.kk.get()),
                                Arg::Mem(self.bufs.vv.get()),
                                Arg::Mem(k_buf.mem()),
                                Arg::Mem(v_buf.mem()),
                                Arg::Mem(tbl_mem),
                                Arg::U32(pos as u32),
                                Arg::U32(kvd as u32),
                            ],
                        )?;
                    }
                    let start = if self.cfg.kind(layer) == LayerKind::SlidingAttention {
                        (pos + 1).saturating_sub(self.cfg.sliding_window)
                    } else {
                        0
                    };
                    let splits = attention_splits(pos + 1 - start);
                    if splits <= 1 {
                        unsafe {
                            self.gpu.launch(
                                self.attn_kernel(false),
                                heads * 64,
                                64,
                                &[
                                    Arg::Mem(self.bufs.q.get()),
                                    Arg::Mem(k_buf.mem()),
                                    Arg::Mem(v_buf.mem()),
                                    Arg::Mem(tbl_mem),
                                    Arg::Mem(self.bufs.g16.get()),
                                    Arg::Mem(self.bufs.attn.get()),
                                    Arg::U32((pos + 1) as u32),
                                    Arg::U32(start as u32),
                                    Arg::U32(kv_heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(heads as u32),
                                ],
                            )?;
                        }
                    } else {
                        unsafe {
                            self.gpu.launch(
                                self.attn_kernel(true),
                                heads * splits * 64,
                                64,
                                &[
                                    Arg::Mem(self.bufs.q.get()),
                                    Arg::Mem(k_buf.mem()),
                                    Arg::Mem(v_buf.mem()),
                                    Arg::Mem(tbl_mem),
                                    Arg::Mem(self.bufs.pacc.get()),
                                    Arg::Mem(self.bufs.pml.get()),
                                    Arg::U32((pos + 1) as u32),
                                    Arg::U32(start as u32),
                                    Arg::U32(kv_heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(heads as u32),
                                    Arg::U32(splits as u32),
                                ],
                            )?;
                            self.gpu.launch(
                                &self.kernels.attn_combine,
                                heads * 64,
                                64,
                                &[
                                    Arg::Mem(self.bufs.pacc.get()),
                                    Arg::Mem(self.bufs.pml.get()),
                                    Arg::Mem(self.bufs.g16.get()),
                                    Arg::Mem(self.bufs.attn.get()),
                                    Arg::U32(heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(splits as u32),
                                ],
                            )?;
                        }
                    }
                    self.quantize(&self.bufs.attn, &self.bufs.hq, &self.bufs.hs, qd)?;
                    self.gemv_auto(
                        &format!("{prefix}.attn_output.weight"),
                        &self.bufs.hq,
                        &self.bufs.hs,
                        &self.bufs.tmp,
                        h,
                        qd,
                    )?;
                }
            }
            let post = self.f32_weight(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm = self.f32_weight(&format!("{prefix}.ffn_norm.weight"))?;
            self.norm_add_quant(&self.bufs.tmp, post, &self.bufs.x, ffn_norm, h)?;
            self.gemv_auto(
                &format!("{prefix}.ffn_ug.weight"),
                &self.bufs.hq,
                &self.bufs.hs,
                &self.bufs.ug,
                2 * self.ffn,
                h,
            )?;
            unsafe {
                self.gpu.launch(
                    &self.kernels.silu_quant,
                    self.ffn / GROUP,
                    0,
                    &[
                        Arg::Mem(self.bufs.ug.get()),
                        Arg::Mem(self.bufs.actq.get()),
                        Arg::Mem(self.bufs.acts.get()),
                        Arg::U32(self.ffn as u32),
                        Arg::U32(self.ffn as u32),
                    ],
                )?;
            }
            self.gemv_auto(
                &format!("{prefix}.ffn_down.weight"),
                &self.bufs.actq,
                &self.bufs.acts,
                &self.bufs.tmp,
                h,
                self.ffn,
            )?;
            let post_ffn = self.f32_weight(&format!("{prefix}.post_ffw_norm.weight"))?;
            let next_w = if layer + 1 < self.cfg.num_layers {
                self.f32_weight(&format!("blk.{}.attn_norm.weight", layer + 1))?
            } else {
                self.f32_weight("output_norm.weight")?
            };
            self.norm_add_quant(&self.bufs.tmp, post_ffn, &self.bufs.x, next_w, h)?;
            if self.trace {
                self.gpu.queue.finish()?;
                let v = self.gpu.download(&self.bufs.x, h)?;
                self.traced.push(v);
            }
        }
        self.gemv_auto(
            "token_embd.weight",
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.logits,
            self.cfg.vocab_size,
            h,
        )?;
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn batch_size(&self) -> usize {
        self.pb
    }

    fn kv_store_kernel(&self) -> &Kernel {
        if self.kv_f16 {
            &self.kernels.kv_store_h
        } else {
            &self.kernels.kv_store
        }
    }

    fn kv_store_b_kernel(&self) -> &Kernel {
        if self.kv_f16 {
            &self.kernels.kv_store_b_h
        } else {
            &self.kernels.kv_store_b
        }
    }

    fn attn_kernel(&self, split: bool) -> &Kernel {
        match (self.kv_f16, split) {
            (true, false) => &self.kernels.attn_h,
            (true, true) => &self.kernels.attn_split_h,
            (false, false) => &self.kernels.attn,
            (false, true) => &self.kernels.attn_split,
        }
    }

    fn attn_b_kernel(&self, split: bool) -> &Kernel {
        match (self.kv_f16, split) {
            (true, false) => &self.kernels.attn_b_h,
            (true, true) => &self.kernels.attn_split_b_h,
            (false, false) => &self.kernels.attn_b,
            (false, true) => &self.kernels.attn_split_b,
        }
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn finish(&self) -> Result<()> {
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn kernel_stats(&self) -> Vec<(String, u64)> {
        let dev = self.gpu.device.id();
        let list: [(&str, &Kernel); 20] = [
            ("q4", &self.kernels.q4),
            ("q4p_b", &self.kernels.q4p_b),
            ("q4x4p_b", &self.kernels.q4x4p_b),
            ("q6", &self.kernels.q6),
            ("q4_b", &self.kernels.q4_b),
            ("q6_b", &self.kernels.q6_b),
            ("attn", &self.kernels.attn),
            ("attn_split", &self.kernels.attn_split),
            ("attn_h", &self.kernels.attn_h),
            ("attn_split_h", &self.kernels.attn_split_h),
            ("attn_b", &self.kernels.attn_b),
            ("attn_split_b", &self.kernels.attn_split_b),
            ("attn_b_h", &self.kernels.attn_b_h),
            ("attn_b_h2", &self.kernels.attn_b_h2),
            ("attn_split_b_h", &self.kernels.attn_split_b_h),
            ("norm_add_quant", &self.kernels.norm_add_quant),
            ("norm_add_quant_b", &self.kernels.norm_add_quant_b),
            ("silu_quant_b", &self.kernels.silu_quant_b),
            ("conv_b", &self.kernels.conv_b),
            ("kv_store_h", &self.kernels.kv_store_h),
        ];
        list.iter()
            .map(|(n, k)| (n.to_string(), k.get_private_mem_size(dev).unwrap_or(0)))
            .collect()
    }

    pub fn reset_profile(&self) {
        self.gpu.reset_profile();
    }

    pub fn profile_report(&self) -> Vec<(String, u32, f64, f64)> {
        self.gpu.profile_report(0.0)
    }

    pub fn forward_batch(&mut self, tokens: &[u32], pos0: usize, need_logits: bool) -> Result<()> {
        self.forward_batch_seq(0, tokens, pos0, need_logits)
    }

    pub fn forward_batch_seq(
        &mut self,
        slot: usize,
        tokens: &[u32],
        pos0: usize,
        need_logits: bool,
    ) -> Result<()> {
        let reqs: Vec<StepReq> = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| StepReq {
                slot,
                token: *token,
                pos: pos0 + i,
            })
            .collect();
        let row = if need_logits { Some(reqs.len() - 1) } else { None };
        self.forward_mixed(&reqs, need_logits, row)
    }

    pub fn step_mixed(&mut self, reqs: &[StepReq], need_logits: bool) -> Result<()> {
        self.forward_mixed(reqs, need_logits, None)
    }

    pub fn step_mixed_rows(&mut self, reqs: &[StepReq], logits_rows: &[usize]) -> Result<()> {
        self.forward_impl(reqs, !logits_rows.is_empty(), None, Some(logits_rows), None)
    }

    pub fn logits_row(&self, row: usize) -> Result<Vec<f32>> {
        self.gpu
            .download_at(&self.bufs.logitsb, row * self.cfg.vocab_size * 4, self.cfg.vocab_size)
    }

    pub fn forward_mixed(
        &mut self,
        reqs: &[StepReq],
        need_logits: bool,
        logits_row: Option<usize>,
    ) -> Result<()> {
        self.forward_impl(reqs, need_logits, logits_row, None, None)
    }

    fn forward_impl(
        &mut self,
        reqs: &[StepReq],
        need_logits: bool,
        logits_row: Option<usize>,
        logits_rows: Option<&[usize]>,
        snap: Option<&[u32]>,
    ) -> Result<()> {
        let ntok = reqs.len();
        if ntok == 0 {
            return Err(anyhow!("empty batch"));
        }
        if ntok > self.pb {
            return Err(anyhow!("batch of {ntok} exceeds prefetch batch {}", self.pb));
        }
        for req in reqs {
            if req.pos >= self.max_ctx {
                return Err(anyhow!(
                    "context overflow: pos {} >= max_ctx {}",
                    req.pos,
                    self.max_ctx
                ));
            }
        }
        // per-slot page tables
        let mut slots: Vec<usize> = reqs.iter().map(|r| r.slot).collect();
        slots.sort_unstable();
        slots.dedup();
        let kv_blocks = self.kv_blocks;
        let mut tbl_host = vec![0u32; self.max_seqs * kv_blocks.max(1)];
        for &slot in &slots {
            let end = reqs
                .iter()
                .filter(|r| r.slot == slot)
                .map(|r| r.pos + 1)
                .max()
                .unwrap_or(1);
            self.seq_ensure(slot, end)?;
            self.seq_sync(slot)?;
            let seq = self.sequence(slot)?;
            let base = slot * kv_blocks;
            for (i, block) in seq.block_table.iter().enumerate() {
                tbl_host[base + i] = *block;
            }
            if let Some(seq) = self.seqs[slot].as_mut() {
                seq.len = seq.len.max(end);
            }
        }
        let conv_len = (self.cfg.conv_cache - 1) * self.cfg.hidden_size;
        let pos_host: Vec<u32> = reqs.iter().map(|r| r.pos as u32).collect();
        let tbl_off_host: Vec<u32> = reqs.iter().map(|r| (r.slot * kv_blocks) as u32).collect();
        let conv_off_host: Vec<u32> = reqs.iter().map(|r| (r.slot * conv_len) as u32).collect();
        let token_host: Vec<u32> = reqs.iter().map(|r| r.token).collect();
        let snap_host: Vec<u32> = match snap {
            Some(s) if self.layers.iter().any(|l| l.snap.is_some()) => s.to_vec(),
            Some(_) => vec![u32::MAX; ntok],
            None => {
                if self.cache_on && self.layers.iter().any(|l| l.snap.is_some()) {
                    reqs.iter()
                        .map(|r| {
                            if (r.pos + 1) % KV_PAGE != 0 {
                                return u32::MAX;
                            }
                            let pending = self
                                .seqs
                                .get(r.slot)
                                .and_then(|s| s.as_ref())
                                .and_then(|s| s.pending.as_ref())
                                .is_some();
                            if !pending {
                                return u32::MAX;
                            }
                            self.seqs
                                .get(r.slot)
                                .and_then(|s| s.as_ref())
                                .and_then(|s| s.block_table.get(r.pos / KV_PAGE))
                                .copied()
                                .unwrap_or(u32::MAX)
                        })
                        .collect()
                } else {
                    vec![u32::MAX; ntok]
                }
            }
        };
        unsafe {
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.tokens, 1, 0, &token_host, &[])?;
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.snap_page, 1, 0, &snap_host, &[])?;
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.pos_b, 1, 0, &pos_host, &[])?;
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.tbl_off_b, 1, 0, &tbl_off_host, &[])?;
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.conv_off_b, 1, 0, &conv_off_host, &[])?;
            self.gpu.queue.enqueue_write_buffer(&mut self.bufs.tbl_all, 1, 0, &tbl_host, &[])?;
        }
        let h = self.cfg.hidden_size;
        let qd = self.cfg.q_dim();
        let kvd = self.cfg.kv_dim();
        let heads = self.cfg.num_heads;
        let head_dim = self.cfg.head_dim;
        let ffn = self.ffn;
        let emb = match self.weight("token_embd.weight")? {
            Weight::Q6(b) => b,
            _ => return Err(anyhow!("token_embd must be Q6_K for gpu embed")),
        };
        let n32 = ntok as u32;
        unsafe {
            self.gpu.launch(
                &self.kernels.embed_b,
                ntok * h,
                64,
                &[
                    Arg::Mem(emb.get()),
                    Arg::Mem(self.bufs.xb.get()),
                    Arg::Mem(self.bufs.tokens.get()),
                    Arg::U32(n32),
                    Arg::U32(h as u32),
                ],
            )?;
        }
        for layer in 0..self.cfg.num_layers {
            let prefix = format!("blk.{layer}");
            if layer == 0 {
                let attn_norm = self.f32_weight(&format!("{prefix}.attn_norm.weight"))?;
                unsafe {
                    self.gpu.launch(
                        &self.kernels.rms_quant_b,
                        ntok * 64,
                        64,
                        &[
                            Arg::Mem(self.bufs.xb.get()),
                            Arg::Mem(attn_norm.get()),
                            Arg::Mem(self.bufs.hqx.get()),
                            Arg::Mem(self.bufs.hsx.get()),
                            Arg::U32(h as u32),
                            Arg::U32(n32),
                        ],
                    )?;
                }
            }
            match self.cfg.kind(layer) {
                LayerKind::Conv => {
                    self.gemv_auto_b(
                        &format!("{prefix}.shortconv.in_proj.weight"),
                        &self.bufs.hqx,
                        &self.bufs.hsx,
                        &self.bufs.bcxb,
                        3 * h,
                        h,
                        ntok,
                        0,
                    )?;
                    let conv_w = self.f32_weight(&format!("{prefix}.shortconv.conv.weight"))?;
                    let state = self.layers[layer]
                        .conv
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing conv state"))?;
                    let snap = self.layers[layer]
                        .snap
                        .as_ref()
                        .unwrap_or(&self.bufs.snap_dummy);
                    unsafe {
                        self.gpu.launch(
                            &self.kernels.conv_b,
                            h,
                            256,
                            &[
                                Arg::Mem(self.bufs.bcxb.get()),
                                Arg::Mem(conv_w.get()),
                                Arg::Mem(state.get()),
                                Arg::Mem(self.bufs.opb.get()),
                                Arg::Mem(self.bufs.conv_off_b.get()),
                                Arg::Mem(self.bufs.snap_page.get()),
                                Arg::Mem(snap.get()),
                                Arg::U32(h as u32),
                                Arg::U32(self.cfg.conv_cache as u32),
                                Arg::U32(n32),
                                Arg::U32(((self.cfg.conv_cache - 1) * h) as u32),
                            ],
                        )?;
                        self.gpu.launch(
                            &self.kernels.quant_b,
                            ntok * (h / GROUP),
                            0,
                            &[
                                Arg::Mem(self.bufs.opb.get()),
                                Arg::Mem(self.bufs.hqx.get()),
                                Arg::Mem(self.bufs.hsx.get()),
                                Arg::U32(h as u32),
                                Arg::U32(GROUP as u32),
                                Arg::U32(n32),
                            ],
                        )?;
                    }
                    self.gemv_auto_b(
                        &format!("{prefix}.shortconv.out_proj.weight"),
                        &self.bufs.hqx,
                        &self.bufs.hsx,
                        &self.bufs.tmpb,
                        h,
                        h,
                        ntok,
                        0,
                    )?;
                }
                _ => {
                    let kv_heads = self.cfg.kv_heads_of(layer);
                    let kvg = kv_heads * head_dim;
                    let fused = if self
                        .prepacked
                        .contains_key(&format!("{prefix}.attn_qkvg.weight"))
                    {
                        Some((format!("{prefix}.attn_qkvg.weight"), true))
                    } else if self.prepacked.contains_key(&format!("{prefix}.attn_qkg.weight")) {
                        Some((format!("{prefix}.attn_qkg.weight"), false))
                    } else {
                        None
                    };
                    match &fused {
                        Some((name, true)) => {
                            self.gemv_auto4_b(
                                name,
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.qb,
                                &self.bufs.kkb,
                                &self.bufs.vvb,
                                &self.bufs.gb,
                                qd,
                                kvg,
                                kvg,
                                heads,
                                h,
                                ntok,
                            )?;
                        }
                        Some((name, false)) => {
                            self.gemv_auto4_b(
                                name,
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.qb,
                                &self.bufs.kkb,
                                &self.bufs.snap_dummy,
                                &self.bufs.gb,
                                qd,
                                kvg,
                                0,
                                heads,
                                h,
                                ntok,
                            )?;
                            self.gemv_auto_b(
                                &format!("{prefix}.attn_v.weight"),
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.vvb,
                                kvg,
                                h,
                                ntok,
                                0,
                            )?;
                        }
                        None => {
                            self.gemv_auto_b(
                                &format!("{prefix}.attn_q.weight"),
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.qb,
                                qd,
                                h,
                                ntok,
                                0,
                            )?;
                            self.gemv_auto_b(
                                &format!("{prefix}.attn_k.weight"),
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.kkb,
                                kvg,
                                h,
                                ntok,
                                0,
                            )?;
                            self.gemv_auto_b(
                                &format!("{prefix}.attn_v.weight"),
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.vvb,
                                kvg,
                                h,
                                ntok,
                                0,
                            )?;
                            self.gemv_auto_b(
                                &format!("{prefix}.attn_gate.weight"),
                                &self.bufs.hqx,
                                &self.bufs.hsx,
                                &self.bufs.gb,
                                heads,
                                h,
                                ntok,
                                0,
                            )?;
                        }
                    }
                    let q_norm = self.f32_weight(&format!("{prefix}.attn_q_norm.weight"))?;
                    let k_norm = self.f32_weight(&format!("{prefix}.attn_k_norm.weight"))?;
                    let inv = if self.cfg.kind(layer) == LayerKind::SlidingAttention {
                        &self.rope_sliding
                    } else {
                        &self.rope_full
                    };
                    unsafe {
                        self.gpu.launch(
                            &self.kernels.rms_rope_heads_b,
                            ntok * heads * 64,
                            64,
                            &[
                                Arg::Mem(self.bufs.qb.get()),
                                Arg::Mem(q_norm.get()),
                                Arg::Mem(inv.get()),
                                Arg::Mem(self.bufs.pos_b.get()),
                                Arg::U32(head_dim as u32),
                                Arg::U32(heads as u32),
                                Arg::U32(n32),
                            ],
                        )?;
                        self.gpu.launch(
                            &self.kernels.rms_rope_heads_b,
                            ntok * kv_heads * 64,
                            64,
                            &[
                                Arg::Mem(self.bufs.kkb.get()),
                                Arg::Mem(k_norm.get()),
                                Arg::Mem(inv.get()),
                                Arg::Mem(self.bufs.pos_b.get()),
                                Arg::U32(head_dim as u32),
                                Arg::U32(kv_heads as u32),
                                Arg::U32(n32),
                            ],
                        )?;
                    }
                    let (k_buf, v_buf) = {
                        let lb = &self.layers[layer];
                        (
                            lb.k.as_ref().ok_or_else(|| anyhow!("missing kv"))?,
                            lb.v.as_ref().ok_or_else(|| anyhow!("missing kv"))?,
                        )
                    };
                    let window = if self.cfg.kind(layer) == LayerKind::SlidingAttention {
                        self.cfg.sliding_window
                    } else {
                        0
                    };
                    unsafe {
                        self.gpu.launch(
                            self.kv_store_b_kernel(),
                            ntok * kvd,
                            256,
                            &[
                                Arg::Mem(self.bufs.kkb.get()),
                                Arg::Mem(self.bufs.vvb.get()),
                                Arg::Mem(k_buf.mem()),
                                Arg::Mem(v_buf.mem()),
                                Arg::Mem(self.bufs.tbl_all.get()),
                                Arg::Mem(self.bufs.tbl_off_b.get()),
                                Arg::Mem(self.bufs.pos_b.get()),
                                Arg::U32(kvd as u32),
                                Arg::U32(n32),
                            ],
                        )?;
                        let last = reqs.iter().map(|r| r.pos).max().unwrap_or(0) + 1;
                        let span_max = if window > 0 { window.min(last) } else { last };
                        let splits = (span_max / 512).clamp(1, MAX_SPLITS);
                        if splits <= 1 {
                            let use2 = self.attn2
                                && self.kv_f16
                                && head_dim == 128
                                && heads % 4 == 0;
                            let kernel = if use2 {
                                &self.kernels.attn_b_h2
                            } else {
                                self.attn_b_kernel(false)
                            };
                            self.gpu.launch(
                                kernel,
                                ntok * (heads / 4) * 256,
                                256,
                                &[
                                    Arg::Mem(self.bufs.qb.get()),
                                    Arg::Mem(k_buf.mem()),
                                    Arg::Mem(v_buf.mem()),
                                    Arg::Mem(self.bufs.tbl_all.get()),
                                    Arg::Mem(self.bufs.tbl_off_b.get()),
                                    Arg::Mem(self.bufs.pos_b.get()),
                                    Arg::Mem(self.bufs.gb.get()),
                                    Arg::Mem(self.bufs.attnb.get()),
                                    Arg::U32(n32),
                                    Arg::U32(kv_heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(heads as u32),
                                    Arg::U32(window as u32),
                                ],
                            )?;
                        } else {
                            let use4 = self.attn4
                                && self.kv_f16
                                && heads == kv_heads * 4
                                && head_dim == 128;
                            let (kernel, grid) = if use4 {
                                (
                                    &self.kernels.attn_split_b_h4,
                                    ntok * kv_heads * splits * 64,
                                )
                            } else {
                                (self.attn_b_kernel(true), ntok * heads * splits * 64)
                            };
                            self.gpu.launch(
                                kernel,
                                grid,
                                64,
                                &[
                                    Arg::Mem(self.bufs.qb.get()),
                                    Arg::Mem(k_buf.mem()),
                                    Arg::Mem(v_buf.mem()),
                                    Arg::Mem(self.bufs.tbl_all.get()),
                                    Arg::Mem(self.bufs.tbl_off_b.get()),
                                    Arg::Mem(self.bufs.pos_b.get()),
                                    Arg::Mem(self.bufs.pacc_b.get()),
                                    Arg::Mem(self.bufs.pml_b.get()),
                                    Arg::U32(n32),
                                    Arg::U32(kv_heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(heads as u32),
                                    Arg::U32(window as u32),
                                    Arg::U32(splits as u32),
                                ],
                            )?;
                            self.gpu.launch(
                                &self.kernels.attn_combine_b,
                                ntok * heads * 64,
                                64,
                                &[
                                    Arg::Mem(self.bufs.pacc_b.get()),
                                    Arg::Mem(self.bufs.pml_b.get()),
                                    Arg::Mem(self.bufs.gb.get()),
                                    Arg::Mem(self.bufs.attnb.get()),
                                    Arg::U32(n32),
                                    Arg::U32(heads as u32),
                                    Arg::U32(head_dim as u32),
                                    Arg::U32(splits as u32),
                                ],
                            )?;
                        }
                        self.gpu.launch(
                            &self.kernels.quant_b,
                            ntok * (qd / GROUP),
                            0,
                            &[
                                Arg::Mem(self.bufs.attnb.get()),
                                Arg::Mem(self.bufs.hqx.get()),
                                Arg::Mem(self.bufs.hsx.get()),
                                Arg::U32(qd as u32),
                                Arg::U32(GROUP as u32),
                                Arg::U32(n32),
                            ],
                        )?;
                    }
                    self.gemv_auto_b(
                        &format!("{prefix}.attn_output.weight"),
                        &self.bufs.hqx,
                        &self.bufs.hsx,
                        &self.bufs.tmpb,
                        h,
                        qd,
                        ntok,
                        0,
                    )?;
                }
            }
            let post = self.f32_weight(&format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm = self.f32_weight(&format!("{prefix}.ffn_norm.weight"))?;
            unsafe {
                self.gpu.launch(
                    &self.kernels.norm_add_quant_b,
                    ntok * 256,
                    256,
                    &[
                        Arg::Mem(self.bufs.tmpb.get()),
                        Arg::Mem(post.get()),
                        Arg::Mem(self.bufs.xb.get()),
                        Arg::Mem(ffn_norm.get()),
                        Arg::Mem(self.bufs.xb.get()),
                        Arg::Mem(self.bufs.hqx.get()),
                        Arg::Mem(self.bufs.hsx.get()),
                        Arg::U32(h as u32),
                        Arg::U32(n32),
                    ],
                )?;
            }
            self.gemv_auto_b(
                &format!("{prefix}.ffn_ug.weight"),
                &self.bufs.hqx,
                &self.bufs.hsx,
                &self.bufs.ugb,
                2 * ffn,
                h,
                ntok,
                0,
            )?;
            unsafe {
                self.gpu.launch(
                    &self.kernels.silu_quant_b,
                    ntok * (ffn / GROUP),
                    0,
                    &[
                        Arg::Mem(self.bufs.ugb.get()),
                        Arg::Mem(self.bufs.actqb.get()),
                        Arg::Mem(self.bufs.actsb.get()),
                        Arg::U32(ffn as u32),
                        Arg::U32(ffn as u32),
                        Arg::U32(n32),
                    ],
                )?;
            }
            self.gemv_auto_b(
                &format!("{prefix}.ffn_down.weight"),
                &self.bufs.actqb,
                &self.bufs.actsb,
                &self.bufs.tmpb,
                h,
                ffn,
                ntok,
                0,
            )?;
            let post_ffn = self.f32_weight(&format!("{prefix}.post_ffw_norm.weight"))?;
            let next_w = if layer + 1 < self.cfg.num_layers {
                self.f32_weight(&format!("blk.{}.attn_norm.weight", layer + 1))?
            } else {
                self.f32_weight("output_norm.weight")?
            };
            unsafe {
                self.gpu.launch(
                    &self.kernels.norm_add_quant_b,
                    ntok * 256,
                    256,
                    &[
                        Arg::Mem(self.bufs.tmpb.get()),
                        Arg::Mem(post_ffn.get()),
                        Arg::Mem(self.bufs.xb.get()),
                        Arg::Mem(next_w.get()),
                        Arg::Mem(self.bufs.xb.get()),
                        Arg::Mem(self.bufs.hqx.get()),
                        Arg::Mem(self.bufs.hsx.get()),
                        Arg::U32(h as u32),
                        Arg::U32(n32),
                    ],
                )?;
            }
            if self.trace {
                self.gpu.queue.finish()?;
                let v = self.gpu.download_at(&self.bufs.xb, ((ntok - 1) * h) * 4, h)?;
                self.traced.push(v);
            }
        }
        if need_logits {
            let sel = pick_logits_rows(need_logits, logits_rows, logits_row, ntok)?
                .ok_or_else(|| anyhow!("missing logits selection"))?;
            if sel.is_empty() {
                self.gemv_auto_b(
                    "token_embd.weight",
                    &self.bufs.hqx,
                    &self.bufs.hsx,
                    &self.bufs.logitsb,
                    self.cfg.vocab_size,
                    h,
                    ntok,
                    0,
                )?;
                self.logits_off = None;
            } else {
                let k = sel.len();
                let ngs = h / GROUP;
                for (i, &r) in sel.iter().enumerate() {
                    unsafe {
                        self.gpu.queue.enqueue_copy_buffer(
                            &self.bufs.hqx,
                            &mut self.bufs.xqg,
                            r * h,
                            i * h,
                            h,
                            &[],
                        )?;
                        self.gpu.queue.enqueue_copy_buffer(
                            &self.bufs.hsx,
                            &mut self.bufs.xsg,
                            r * ngs * 4,
                            i * ngs * 4,
                            ngs * 4,
                            &[],
                        )?;
                    }
                }
                self.gemv_auto_b(
                    "token_embd.weight",
                    &self.bufs.xqg,
                    &self.bufs.xsg,
                    &self.bufs.logitsg,
                    self.cfg.vocab_size,
                    h,
                    k,
                    0,
                )?;
                let row_bytes = self.cfg.vocab_size * 4;
                for (i, &r) in sel.iter().enumerate() {
                    unsafe {
                        self.gpu.queue.enqueue_copy_buffer(
                            &self.bufs.logitsg,
                            &mut self.bufs.logitsb,
                            i * row_bytes,
                            r * row_bytes,
                            row_bytes,
                            &[],
                        )?;
                    }
                }
                self.logits_off = if k == 1 { Some(sel[0] * self.cfg.vocab_size) } else { None };
            }
        } else {
            self.logits_off = None;
        }
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn debug_embed(&mut self, token: u32) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let emb = match self.weight("token_embd.weight")? {
            Weight::Q6(b) => b,
            _ => return Err(anyhow!("embedding must be Q6_K")),
        };
        unsafe {
            self.gpu.launch(
                &self.kernels.embed,
                h,
                64,
                &[
                    Arg::Mem(emb.get()),
                    Arg::Mem(self.bufs.x.get()),
                    Arg::U32(token),
                    Arg::U32(h as u32),
                ],
            )?
            .wait()?;
        }
        self.gpu.download(&self.bufs.x, h)
    }

    pub fn debug_norm(&mut self, name: &str) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let w = self.f32_weight(name)?;
        unsafe {
            self.gpu.launch(
                &self.kernels.rms,
                64,
                64,
                &[
                    Arg::Mem(self.bufs.x.get()),
                    Arg::Mem(w.get()),
                    Arg::Mem(self.bufs.h.get()),
                    Arg::U32(h as u32),
                ],
            )?
            .wait()?;
        }
        self.gpu.download(&self.bufs.h, h)
    }

    pub fn debug_norm_raw(&mut self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let n = x.len();
        let xb = self.gpu.upload(x)?;
        let w = self.f32_weight(name)?;
        unsafe {
            self.gpu.launch(
                &self.kernels.rms,
                64,
                64,
                &[
                    Arg::Mem(xb.get()),
                    Arg::Mem(w.get()),
                    Arg::Mem(self.bufs.h.get()),
                    Arg::U32(n as u32),
                ],
            )?
            .wait()?;
        }
        self.gpu.download(&self.bufs.h, n)
    }

    pub fn debug_weight_f32(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let w = self.f32_weight(name)?;
        self.gpu.download(w, n)
    }

    pub fn debug_conv(&mut self, layer: usize, h: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let hsize = self.cfg.hidden_size;
        let xh = self.gpu.upload(h)?;
        self.quantize(&xh, &self.bufs.hq, &self.bufs.hs, hsize)?;
        self.gemv_auto(
            &format!("blk.{layer}.shortconv.in_proj.weight"),
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.bcx,
            3 * hsize,
            hsize,
        )?;
        let conv_w = self.f32_weight(&format!("blk.{layer}.shortconv.conv.weight"))?;
        let state = self.layers[layer]
            .conv
            .as_ref()
            .ok_or_else(|| anyhow!("missing conv state"))?;
        unsafe {
            self.gpu.launch(
                &self.kernels.conv,
                hsize,
                256,
                &[
                    Arg::Mem(self.bufs.bcx.get()),
                    Arg::Mem(conv_w.get()),
                    Arg::Mem(state.get()),
                    Arg::Mem(self.bufs.op.get()),
                    Arg::U32(hsize as u32),
                    Arg::U32(self.cfg.conv_cache as u32),
                ],
            )?;
        }
        self.quantize(&self.bufs.op, &self.bufs.hq, &self.bufs.hs, hsize)?;
        self.gemv_auto(
            &format!("blk.{layer}.shortconv.out_proj.weight"),
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.tmp,
            hsize,
            hsize,
        )?;
        self.gpu.queue.finish()?;
        let bcx = self.gpu.download(&self.bufs.bcx, 3 * hsize)?;
        let op = self.gpu.download(&self.bufs.tmp, hsize)?;
        Ok((bcx, op))
    }

    pub fn debug_ffn(&mut self, layer: usize, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hsize = self.cfg.hidden_size;
        let xb = self.gpu.upload(x)?;
        let ffn_norm = self.f32_weight(&format!("blk.{layer}.ffn_norm.weight"))?;
        self.rms(&xb, ffn_norm, &self.bufs.h, hsize)?;
        self.quantize(&self.bufs.h, &self.bufs.hq, &self.bufs.hs, hsize)?;
        self.gemv_auto(&format!("blk.{layer}.ffn_ug.weight"), &self.bufs.hq, &self.bufs.hs, &self.bufs.ug, 2 * self.ffn, hsize)?;
        unsafe {
            self.gpu.launch(
                &self.kernels.silu,
                self.ffn,
                256,
                &[
                    Arg::Mem(self.bufs.ug.get()),
                    Arg::Mem(self.bufs.act.get()),
                    Arg::U32(self.ffn as u32),
                    Arg::U32(self.ffn as u32),
                ],
            )?;
        }
        self.quantize(&self.bufs.act, &self.bufs.actq, &self.bufs.acts, self.ffn)?;
        self.gemv_auto(&format!("blk.{layer}.ffn_down.weight"), &self.bufs.actq, &self.bufs.acts, &self.bufs.tmp, hsize, self.ffn)?;
        let post_ffn = self.f32_weight(&format!("blk.{layer}.post_ffw_norm.weight"))?;
        self.rms_add(&self.bufs.tmp, post_ffn, &xb, &self.bufs.op, hsize)?;
        self.gpu.queue.finish()?;
        let h = self.gpu.download(&self.bufs.h, hsize)?;
        let up = self
            .gpu
            .download_at(&self.bufs.ug, self.ffn * 4, self.ffn)?;
        let act = self.gpu.download(&self.bufs.act, self.ffn)?;
        let out = self.gpu.download(&self.bufs.op, hsize)?;
        Ok((h, up, act, out))
    }

    pub fn debug_lmhead(&mut self, rows: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        self.quantize(&self.bufs.h, &self.bufs.hq, &self.bufs.hs, h)?;
        self.gemv_auto("token_embd.weight", &self.bufs.hq, &self.bufs.hs, &self.bufs.logits, rows, h)?;
        self.gpu.queue.finish()?;
        self.gpu.download(&self.bufs.logits, rows)
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        match self.logits_off {
            Some(off) => self
                .gpu
                .download_at(&self.bufs.logitsb, off * 4, self.cfg.vocab_size),
            None => self.gpu.download(&self.bufs.logits, self.cfg.vocab_size),
        }
    }

    pub fn debug_state(&self, layer: usize, pos: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let h = self.cfg.hidden_size;
        let kvd = self.cfg.kv_dim();
        let conv = match &self.layers[layer].conv {
            Some(c) => self.gpu.download(c, (self.cfg.conv_cache - 1) * h)?,
            None => Vec::new(),
        };
        let slot = self
            .seqs
            .first()
            .and_then(|s| s.as_ref())
            .and_then(|s| s.block_table.get(pos / KV_PAGE))
            .map(|b| (*b as usize) * KV_PAGE + pos % KV_PAGE)
            .unwrap_or(0);
        let k = match &self.layers[layer].k {
            Some(KvBuf::F32(b)) => self.gpu.download_at(b, slot * kvd * 4, kvd)?,
            Some(KvBuf::F16(b)) => self
                .gpu
                .download_at(b, slot * kvd * 2, kvd)?
                .iter()
                .map(|bits| half_to_f32(*bits))
                .collect(),
            None => Vec::new(),
        };
        let v = match &self.layers[layer].v {
            Some(KvBuf::F32(b)) => self.gpu.download_at(b, slot * kvd * 4, kvd)?,
            Some(KvBuf::F16(b)) => self
                .gpu
                .download_at(b, slot * kvd * 2, kvd)?
                .iter()
                .map(|bits| half_to_f32(*bits))
                .collect(),
            None => Vec::new(),
        };
        Ok((conv, k, v))
    }

    pub fn debug_norm_add_quant(
        &self,
        op: &[f32],
        post: &[f32],
        res: &[f32],
        next_w: &[f32],
    ) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>)> {
        let n = op.len();
        let ob = self.gpu.upload(op)?;
        let pb = self.gpu.upload(post)?;
        let rb = self.gpu.upload(res)?;
        let nb = self.gpu.upload(next_w)?;
        self.norm_add_quant(&ob, &pb, &rb, &nb, n)?;
        self.gpu.queue.finish()?;
        let x = self.gpu.download(&self.bufs.x, n)?;
        let q = self.gpu.download(&self.bufs.hq, n)?;
        let s = self.gpu.download(&self.bufs.hs, n / GROUP)?;
        Ok((x, q, s))
    }

    pub fn debug_silu_quant(&self, ug: &[f32], n: usize) -> Result<(Vec<i8>, Vec<f32>)> {
        let ub = self.gpu.upload(ug)?;
        unsafe {
            self.gpu.launch(
                &self.kernels.silu_quant,
                n / GROUP,
                0,
                &[
                    Arg::Mem(ub.get()),
                    Arg::Mem(self.bufs.actq.get()),
                    Arg::Mem(self.bufs.acts.get()),
                    Arg::U32(n as u32),
                    Arg::U32(n as u32),
                ],
            )?;
        }
        self.gpu.queue.finish()?;
        let q = self.gpu.download(&self.bufs.actq, n)?;
        let s = self.gpu.download(&self.bufs.acts, n / GROUP)?;
        Ok((q, s))
    }

    pub fn debug_quant(&self, x: &[f32]) -> Result<(Vec<i8>, Vec<f32>)> {
        let n = x.len();
        let xb = self.gpu.upload(x)?;
        self.quantize(&xb, &self.bufs.hq, &self.bufs.hs, n)?;
        self.gpu.queue.finish()?;
        let q = self.gpu.download(&self.bufs.hq, n)?;
        let s = self.gpu.download(&self.bufs.hs, n / GROUP)?;
        Ok((q, s))
    }

    pub fn debug_x(&self, batch: bool, tok: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden_size;
        if batch {
            self.gpu.download_at(&self.bufs.xb, tok * h * 4, h)
        } else {
            self.gpu.download(&self.bufs.x, h)
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        for slot in 0..self.max_seqs {
            self.free_sequence(slot)?;
        }
        self.cache.clear();
        self.create_sequence()?;
        let h = self.cfg.hidden_size;
        let kvd = self.cfg.kv_dim();
        let conv_len = self.max_seqs * (self.cfg.conv_cache - 1) * h;
        let kv_len = self.kv_blocks * KV_PAGE * kvd;
        for layer in &self.layers {
            if let Some(conv) = &layer.conv {
                unsafe {
                    self.gpu.launch(
                        &self.kernels.fill,
                        conv_len,
                        256,
                        &[Arg::Mem(conv.get()), Arg::U32(0), Arg::U32(conv_len as u32)],
                    )?;
                }
            }
            if let Some(k) = &layer.k {
                let v = layer.v.as_ref().ok_or_else(|| anyhow!("missing v"))?;
                let fill = if self.kv_f16 {
                    &self.kernels.fill_h
                } else {
                    &self.kernels.fill
                };
                unsafe {
                    self.gpu.launch(
                        fill,
                        kv_len,
                        256,
                        &[Arg::Mem(k.mem()), Arg::U32(0), Arg::U32(kv_len as u32)],
                    )?;
                    self.gpu.launch(
                        fill,
                        kv_len,
                        256,
                        &[Arg::Mem(v.mem()), Arg::U32(0), Arg::U32(kv_len as u32)],
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn cache_stats(&self) -> (usize, usize, u64) {
        (self.cache.cached(), self.cache.used_pages(), self.cache.hits())
    }

    pub fn free_pages(&self) -> usize {
        self.cache
            .capacity()
            .saturating_sub(self.cache.used_pages())
    }

    pub fn prefill(&mut self, slot: usize, prompt: &[u32], need_logits: bool) -> Result<usize> {
        let reused = self.prefill_begin(slot, prompt)?;
        let mut pos = reused;
        while pos < prompt.len() {
            let end = (pos + self.pb).min(prompt.len());
            let reqs: Vec<StepReq> = (pos..end)
                .map(|p| StepReq {
                    slot,
                    token: prompt[p],
                    pos: p,
                })
                .collect();
            let last = end == prompt.len();
            let row = last.then_some(reqs.len() - 1);
            self.forward_impl(&reqs, need_logits && last, row, None, None)?;
            pos = end;
        }
        self.prefill_finish(slot)?;
        Ok(reused)
    }

    pub fn prefill_begin(&mut self, slot: usize, prompt: &[u32]) -> Result<usize> {
        if prompt.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        if prompt.len() > self.max_ctx {
            return Err(anyhow!(
                "prompt {} exceeds max_ctx {}",
                prompt.len(),
                self.max_ctx
            ));
        }
        if self.seqs.get(slot).map(|s| s.is_none()).unwrap_or(true) {
            return Err(anyhow!("sequence slot {slot} is empty"));
        }
        if let Some(pending) = self.seqs[slot]
            .as_ref()
            .and_then(|s| s.pending.as_ref())
        {
            return Ok(pending.reused_pages * KV_PAGE);
        }
        let fresh = self
            .seqs
            .get(slot)
            .and_then(|s| s.as_ref())
            .map(|s| s.len == 0 && s.block_table.is_empty())
            .unwrap_or(false);
        if !fresh {
            return Err(anyhow!("prefill expects a fresh sequence in slot {slot}"));
        }
        let hashes = chain_hashes(prompt);
        let cap = reuse_cap(prompt.len());
        let mut reused_pages = 0usize;
        if self.cache_on {
            for &hash in hashes.iter().take(cap) {
                let page = match self.cache.take(hash) {
                    Some(page) => page,
                    None => break,
                };
                if let Some(seq) = self.seqs[slot].as_mut() {
                    seq.block_table.push(page as u32);
                    seq.tbl_dirty = true;
                }
                reused_pages += 1;
            }
        }
        let reused = reused_pages * KV_PAGE;
        if reused_pages > 0 {
            let page = self
                .seqs
                .get(slot)
                .and_then(|s| s.as_ref())
                .and_then(|s| s.block_table.get(reused_pages - 1))
                .copied()
                .ok_or_else(|| anyhow!("missing cached page"))? as usize;
            self.restore_conv(slot, page)?;
        }
        self.seq_ensure(slot, prompt.len())?;
        if let Some(seq) = self.seqs[slot].as_mut() {
            seq.pending = Some(PendingPrefill {
                hashes,
                reused_pages,
            });
        }
        Ok(reused)
    }

    pub fn prefill_finish(&mut self, slot: usize) -> Result<()> {
        let Some(pending) = self
            .seqs
            .get_mut(slot)
            .and_then(|s| s.as_mut())
            .and_then(|s| s.pending.take())
        else {
            return Ok(());
        };
        if !self.cache_on {
            return Ok(());
        }
        let pages = self
            .seqs
            .get(slot)
            .and_then(|s| s.as_ref())
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        for (i, &hash) in pending.hashes.iter().enumerate().skip(pending.reused_pages) {
            if let Some(&page) = pages.get(i) {
                self.cache.publish(page as usize, hash);
            }
        }
        Ok(())
    }

    fn restore_conv(&mut self, slot: usize, page: usize) -> Result<()> {
        let conv_len = (self.cfg.conv_cache - 1) * self.cfg.hidden_size;
        let src_off = page * conv_len * 4;
        let dst_off = slot * conv_len * 4;
        let size = conv_len * 4;
        for layer in self.layers.iter_mut() {
            if let (Some(snap), Some(conv)) = (layer.snap.as_ref(), layer.conv.as_mut()) {
                unsafe {
                    self.gpu
                        .queue
                        .enqueue_copy_buffer(snap, conv, src_off, dst_off, size, &[])?;
                }
            }
        }
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &SamplingParams,
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        self.generate_stream(prompt, params, max_new, eos, |token| out.push(token))?;
        Ok(out)
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt: &[u32],
        params: &SamplingParams,
        max_new: usize,
        eos: u32,
        on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(u32),
    {
        let cancel = AtomicBool::new(false);
        self.generate_stream_cancel(prompt, params, max_new, eos, &cancel, on_token)
    }

    pub fn generate_stream_cancel<F>(
        &mut self,
        prompt: &[u32],
        params: &SamplingParams,
        max_new: usize,
        eos: u32,
        cancel: &AtomicBool,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(u32),
    {
        if prompt.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        if prompt.len() + max_new > self.max_ctx {
            return Err(anyhow!(
                "prompt {} + max_new {} exceeds max_ctx {}",
                prompt.len(),
                max_new,
                self.max_ctx
            ));
        }
        self.free_sequence(0)?;
        self.create_sequence()?;
        self.prefill(0, prompt, true)?;
        let mut sampler = Sampler::new(params.clone());
        let mut pos = prompt.len();
        let mut logits = self.logits()?;
        for token in prompt {
            sampler.push(*token);
        }
        let mut count = 0usize;
        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let mut row = logits.clone();
            let next = sampler.sample(&mut row);
            sampler.push(next);
            count += 1;
            on_token(next);
            if next == eos || count >= max_new || pos >= self.max_ctx {
                break;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            self.forward(next, pos)?;
            pos += 1;
            logits = self.logits()?;
        }
        Ok(count)
    }

    pub fn generate_greedy(&mut self, prompt: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>> {
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        self.generate(prompt, &params, max_new, eos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_covers_all() {
        assert_eq!(chunk_ranges(0, 5), Vec::<(usize, usize)>::new());
        assert_eq!(chunk_ranges(5, 5), vec![(0, 5)]);
        assert_eq!(chunk_ranges(9, 5), vec![(0, 5), (5, 9)]);
        assert_eq!(chunk_ranges(4, 1), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let chunks = chunk_ranges(17, 4);
        assert_eq!(chunks.first(), Some(&(0, 4)));
        assert_eq!(chunks.last(), Some(&(16, 17)));
        assert_eq!(chunks.iter().map(|(a, b)| b - a).sum::<usize>(), 17);
    }

    #[test]
    fn chunk_ranges_handles_zero_batch() {
        assert_eq!(chunk_ranges(3, 0), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn splits_scale_with_context() {
        assert_eq!(attention_splits(0), 1);
        assert_eq!(attention_splits(511), 1);
        assert_eq!(attention_splits(512), 1);
        assert_eq!(attention_splits(1023), 1);
        assert_eq!(attention_splits(1024), 2);
        assert_eq!(attention_splits(4096), 8);
        assert_eq!(attention_splits(8192), MAX_SPLITS);
        assert_eq!(attention_splits(1 << 20), MAX_SPLITS);
    }

    #[test]
    fn half_bits_convert_exactly() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x8000), -0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert!(half_to_f32(0x7c00).is_infinite() && half_to_f32(0x7c00) > 0.0);
        assert!(half_to_f32(0xfc00).is_infinite() && half_to_f32(0xfc00) < 0.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn reuse_keeps_one_token_uncomputed() {
        assert_eq!(reuse_cap(0), 0);
        assert_eq!(reuse_cap(1), 0);
        assert_eq!(reuse_cap(31), 0);
        assert_eq!(reuse_cap(32), 0);
        assert_eq!(reuse_cap(33), 1);
        assert_eq!(reuse_cap(64), 1);
        assert_eq!(reuse_cap(512), 15);
        assert!(reuse_cap(512) * KV_PAGE < 512);
        assert!(reuse_cap(8192) * KV_PAGE < 8192);
    }

    #[test]
    fn logits_rows_all_when_unspecified() {
        assert_eq!(pick_logits_rows(true, None, None, 6).unwrap(), Some(Vec::new()));
        assert_eq!(pick_logits_rows(false, None, Some(2), 6).unwrap(), None);
    }

    #[test]
    fn logits_rows_single_row_from_forward_mixed() {
        assert_eq!(pick_logits_rows(true, None, Some(4), 6).unwrap(), Some(vec![4]));
    }

    #[test]
    fn logits_rows_are_sorted_and_deduplicated() {
        assert_eq!(
            pick_logits_rows(true, Some(&[5, 1, 5, 3]), None, 6).unwrap(),
            Some(vec![1, 3, 5])
        );
    }

    #[test]
    fn logits_rows_reject_out_of_range() {
        assert!(pick_logits_rows(true, Some(&[0, 6]), None, 6).is_err());
        assert!(pick_logits_rows(true, None, Some(9), 6).is_err());
        assert!(pick_logits_rows(true, Some(&[]), None, 6).is_err());
    }
}

pub fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut f = frac;
            let mut e = -1i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let exp32 = (114 + e) as u32;
            (sign << 31) | (exp32 << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

pub fn argmax(x: &[f32]) -> u32 {
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
