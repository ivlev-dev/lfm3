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

mod pipeline;
mod prepack;
mod tp;
mod ports;
mod radix;

pub use pipeline::PipelineModel;
pub use tp::TpModel;

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
    gemm_q4_lds10: Kernel,
    gemm_q4_lds11: Kernel,
    gemm_q6: Kernel,
    gemm_q6_pp: Kernel,
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
    attn_split_b_h4_p4: Kernel,
}

#[derive(Debug, Clone, Copy)]
pub struct TpShard {
    pub rank: usize,
    pub world: usize,
    pub head_start: usize,
    pub head_count: usize,
    pub kv_head_start: usize,
    pub kv_head_count: usize,
    pub out_col_start: usize,
    pub out_col_count: usize,
    pub ffn_row_start: usize,
    pub ffn_row_count: usize,
    pub ffn_col_start: usize,
    pub ffn_col_count: usize,
    pub vocab_start: usize,
    pub vocab_count: usize,
    pub embed: bool,
}

impl TpShard {
    pub fn attention(
        rank: usize,
        world: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let qd = heads * head_dim;
        let per = qd / world;
        debug_assert!(per.is_multiple_of(256));
        Self {
            rank,
            world,
            head_start: rank * heads / world,
            head_count: heads / world,
            kv_head_start: rank * kv_heads / world,
            kv_head_count: kv_heads / world,
            out_col_start: rank * per / head_dim,
            out_col_count: per / head_dim,
            ffn_row_start: 0,
            ffn_row_count: 0,
            ffn_col_start: 0,
            ffn_col_count: 0,
            vocab_start: 0,
            vocab_count: 0,
            embed: rank == 0,
        }
    }

    pub fn full(
        rank: usize,
        world: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        ffn: usize,
        vocab: usize,
    ) -> Self {
        let mut shard = Self::attention(rank, world, heads, kv_heads, head_dim);
        let per = ffn / world;
        shard.ffn_row_start = rank * per;
        shard.ffn_row_count = per;
        shard.ffn_col_start = rank * per;
        shard.ffn_col_count = per;
        let vper = vocab / world;
        shard.vocab_start = rank * vper;
        shard.vocab_count = vper;
        shard
    }
}

enum ShardSlice {
    Rows(usize, usize),
    Blocks(usize, usize, usize, usize),
}

fn shard_slice(
    name: &str,
    rows: usize,
    cols: usize,
    head_dim: usize,
    shard: &TpShard,
) -> Option<ShardSlice> {
    if let Some(rest) = name.strip_prefix("blk.") {
        let (_, field) = rest.split_once('.')?;
        return match field {
            "attn_q.weight" => Some(ShardSlice::Rows(
                shard.head_start * head_dim,
                (shard.head_start + shard.head_count) * head_dim,
            )),
            "attn_k.weight" | "attn_v.weight" => Some(ShardSlice::Rows(
                shard.kv_head_start * head_dim,
                (shard.kv_head_start + shard.kv_head_count) * head_dim,
            )),
            "attn_gate.weight" => Some(ShardSlice::Rows(
                shard.head_start,
                shard.head_start + shard.head_count,
            )),
            "ffn_up.weight" | "ffn_gate.weight" => Some(ShardSlice::Rows(
                shard.ffn_row_start,
                shard.ffn_row_start + shard.ffn_row_count,
            )),
            "ffn_down.weight" => Some(ShardSlice::Blocks(
                0,
                rows,
                shard.ffn_col_start / 256,
                (shard.ffn_col_start + shard.ffn_col_count) / 256,
            )),
            "attn_output.weight" => {
                if cols != shard.world * (cols / shard.world) {
                    return None;
                }
                Some(ShardSlice::Blocks(
                    0,
                    rows,
                    shard.out_col_start * head_dim / 256,
                    (shard.out_col_start + shard.out_col_count) * head_dim / 256,
                ))
            }
            _ => Some(ShardSlice::Rows(0, usize::MAX)),
        };
    }
    if name == "token_embd.weight" {
        return if shard.embed {
            Some(ShardSlice::Rows(0, usize::MAX))
        } else {
            None
        };
    }
    Some(ShardSlice::Rows(0, usize::MAX))
}

fn block_size(kind: GgmlType) -> usize {
    match kind {
        GgmlType::Q4_K => 144,
        GgmlType::Q6_K => 210,
        GgmlType::F32 => 4,
        _ => 1,
    }
}

fn slice_shard(
    bytes: &[u8],
    kind: GgmlType,
    cols: usize,
    spec: &ShardSlice,
) -> Result<Vec<u8>> {
    if matches!(kind, GgmlType::F32) {
        if let ShardSlice::Rows(start, end) = *spec {
            let stride = cols * 4;
            let total = bytes.len() / stride.max(1);
            let start = start.min(total);
            let end = end.min(total);
            return Ok(bytes[start * stride..end * stride].to_vec());
        }
        return Err(anyhow!("f32 shard supports rows only"));
    }
    let bs = block_size(kind);
    let blocks = cols / 256;
    if !cols.is_multiple_of(256) {
        return Err(anyhow!("shard needs cols multiple of 256"));
    }
    match *spec {
        ShardSlice::Rows(start, end) => {
            let stride = blocks * bs;
            let total = bytes.len() / stride;
            let start = start.min(total);
            let end = end.min(total);
            Ok(bytes[start * stride..end * stride].to_vec())
        }
        ShardSlice::Blocks(r0, r1, b0, b1) => {
            let stride = blocks * bs;
            let out_stride = (b1 - b0) * bs;
            let mut out = Vec::with_capacity((r1 - r0) * out_stride);
            for row in r0..r1 {
                let row_start = row * stride;
                out.extend_from_slice(&bytes[row_start + b0 * bs..row_start + b1 * bs]);
            }
            Ok(out)
        }
    }
}

fn tensor_needed(
    name: &str,
    start: usize,
    end: usize,
    has_embed: bool,
    has_head: bool,
) -> bool {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((idx, _)) = rest.split_once('.') {
            if let Ok(layer) = idx.parse::<usize>() {
                return (start..end).contains(&layer);
            }
        }
        return false;
    }
    if name == "token_embd.weight" {
        return has_embed || has_head;
    }
    if name == "output_norm.weight" {
        return has_head;
    }
    true
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
    layer_start: usize,
    layer_end: usize,
    has_embed: bool,
    has_head: bool,
    tp: Option<TpShard>,
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
    prefill_batch: usize,
    batch_cap: usize,
    gemm_force: Option<String>,
    prepacked_i8: HashMap<String, (Buffer<i8>, Buffer<f32>, usize)>,
    gemm_q4: String,
    rgemv: usize,
    rgemv_b: usize,
    wg: usize,
    wg_b: usize,
    subgroup: usize,
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
        Self::load_stage(path, max_ctx, None, None, None)
    }

    pub fn load_stage(
        path: &Path,
        max_ctx: usize,
        device: Option<usize>,
        layers: Option<(usize, usize)>,
        shard: Option<TpShard>,
    ) -> Result<Self> {
        let profiling = std::env::var("LFM_PROFILE").map(|v| v == "1").unwrap_or(false);
        let device = match device {
            Some(device) => device,
            None => std::env::var("LFM_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        };
        let gpu = Gpu::open_with(device, profiling)?;
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_metadata(&file).map_err(|e| anyhow!("config: {e}"))?;
        let (layer_start, layer_end) = layers.unwrap_or((0, cfg.num_layers));
        if layer_start >= layer_end || layer_end > cfg.num_layers {
            return Err(anyhow!(
                "bad layer range {layer_start}..{layer_end} of {}",
                cfg.num_layers
            ));
        }
        let has_embed = layer_start == 0;
        let has_head = layer_end == cfg.num_layers;
        let prepack_on = std::env::var("LFM_PREPACK").map(|v| v != "0").unwrap_or(true);
        let fuse_on = std::env::var("LFM_FUSE").map(|v| v != "0").unwrap_or(true);
        let mut weights = HashMap::new();
        let mut prepacked: HashMap<String, Buffer<u8>> = HashMap::new();
        let mut prepacked_i8: HashMap<String, (Buffer<i8>, Buffer<f32>, usize)> = HashMap::new();
        for info in file.tensors() {
            if !tensor_needed(&info.name, layer_start, layer_end, has_embed, has_head) {
                continue;
            }
            let bytes = file.tensor_bytes(&info.name)?;
            let cols = info.dims[0] as usize;
            let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
            let data: std::borrow::Cow<[u8]> = if let Some(shard) = shard {
                match shard_slice(&info.name, rows, cols, cfg.head_dim, &shard) {
                    None => continue,
                    Some(spec) => {
                        std::borrow::Cow::Owned(slice_shard(&bytes, info.ggml_type, cols, &spec)?)
                    }
                }
            } else {
                std::borrow::Cow::Borrowed(&bytes)
            };
            let weight = match info.ggml_type {
                GgmlType::Q4_K => {
                    if prepack_on {
                        let packed = prepack::prepack_q4_k(data.as_ref());
                        prepacked.insert(info.name.clone(), gpu.upload_bytes(&packed)?);
                    }
                    Weight::Q4(gpu.upload_bytes(data.as_ref())?)
                }
                GgmlType::Q6_K => {
                    if prepack_on && shard.is_none() && info.dims.len() == 2 {
                        let cols = info.dims[0] as usize;
                        let rows = info.dims[1] as usize;
                        if cols % 256 == 0 && rows > 0 {
                            let (q, sc) = prepack::prepack_q6_i8(data.as_ref(), cols);
                            prepacked_i8.insert(
                                info.name.clone(),
                                (gpu.upload(&q)?, gpu.upload(&sc)?, rows),
                            );
                        }
                    }
                    Weight::Q6(gpu.upload_bytes(data.as_ref())?)
                }
                GgmlType::F32 => {
                    let data: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Weight::F32(gpu.upload(&data)?)
                }
                _ => continue,
            };
            weights.insert(info.name.clone(), weight);
        }
        if let Some(shard) = shard {
            let info = file
                .tensor("token_embd.weight")
                .ok_or_else(|| anyhow!("missing token_embd"))?
                .clone();
            let bytes = file.tensor_bytes("token_embd.weight")?;
            let cols = info.dims[0] as usize;
            let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
            let spec = ShardSlice::Rows(
                shard.vocab_start,
                shard.vocab_start + shard.vocab_count,
            );
            let data = slice_shard(&bytes, info.ggml_type, cols, &spec)?;
            let weight = match info.ggml_type {
                GgmlType::Q4_K => Weight::Q4(gpu.upload_bytes(&data)?),
                GgmlType::Q6_K => Weight::Q6(gpu.upload_bytes(&data)?),
                other => return Err(anyhow!("unsupported lm_head type {other:?}")),
            };
            let _ = rows;
            weights.insert("lm_head_shard".to_string(), weight);
        }
        for layer in layer_start..layer_end {
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
                    let slice_ffn = |bytes: &[u8], info: &lfm_gguf::TensorInfo| -> Result<Vec<u8>> {
                        if let Some(shard) = shard {
                            let cols = info.dims[0] as usize;
                            let rows = info.dims.get(1).copied().unwrap_or(1) as usize;
                            let spec = shard_slice(&info.name, rows, cols, cfg.head_dim, &shard)
                                .ok_or_else(|| anyhow!("missing ffn shard spec"))?;
                            slice_shard(bytes, info.ggml_type, cols, &spec)
                        } else {
                            Ok(bytes.to_vec())
                        }
                    };
                    let gate_s = slice_ffn(&gate_bytes, gate_info)?;
                    let up_s = slice_ffn(&up_bytes, up_info)?;
                    if prepack_on && up_info.ggml_type == GgmlType::Q4_K {
                        let mut fused = prepack::prepack_q4_k(&gate_s);
                        fused.extend_from_slice(&prepack::prepack_q4_k(&up_s));
                        prepacked.insert(
                            format!("blk.{layer}.ffn_ug.weight"),
                            gpu.upload_bytes(&fused)?,
                        );
                    }
                    let mut fused = gpu.alloc::<u8>(gate_s.len() + up_s.len())?;
                    unsafe {
                        gpu.queue.enqueue_write_buffer(&mut fused, 1, 0, &gate_s, &[])?;
                        gpu.queue.enqueue_write_buffer(&mut fused, 1, gate_s.len(), &up_s, &[])?;
                    }
                    let weight = match up_info.ggml_type {
                        GgmlType::Q4_K => Weight::Q4(fused),
                        GgmlType::Q6_K => Weight::Q6(fused),
                        _ => continue,
                    };
                    weights.insert(format!("blk.{layer}.ffn_ug.weight"), weight);
                }
            }
            if !prepack_on || !fuse_on || shard.is_some() {
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
        let prefill_batch: usize = std::env::var("LFM_PREFILL_NTOK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &usize| (1..=128).contains(v))
            .unwrap_or(128);
        let batch_cap = pb.max(prefill_batch);
        let gemm_force: Option<String> = std::env::var("LFM_GEMM_FORCE").ok().filter(|v| !v.is_empty());
        let gemm_q4: String = std::env::var("LFM_GEMM_Q4").unwrap_or_else(|_| "lds11".to_string());
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
        let subgroup = gpu.caps.sub_group_size.max(1);
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
            gemm_q4_lds10: Kernel::create(&program, "gemm_q4_k_i8_lds10")?,
            gemm_q4_lds11: Kernel::create(&program, "gemm_q4_k_i8_lds11")?,
            gemm_q6: Kernel::create(&program, "gemm_q6_k_i8")?,
            gemm_q6_pp: Kernel::create(&program, "gemm_q6_i8_pp")?,
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
            attn_split_b_h4_p4: Kernel::create(&program, "attention_split_b_h4_p4")?,
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
            tokens: gpu.alloc::<u32>(batch_cap)?,
            xb: gpu.alloc::<f32>(batch_cap * h)?,
            opb: gpu.alloc::<f32>(batch_cap * h)?,
            tmpb: gpu.alloc::<f32>(batch_cap * h)?,
            hqx: gpu.alloc::<i8>(batch_cap * h)?,
            hsx: gpu.alloc::<f32>(batch_cap * (h / GROUP))?,
            ugb: gpu.alloc::<f32>(batch_cap * 2 * ffn)?,
            actqb: gpu.alloc::<i8>(batch_cap * ffn)?,
            actsb: gpu.alloc::<f32>(batch_cap * (ffn / GROUP))?,
            bcxb: gpu.alloc::<f32>(batch_cap * 3 * h)?,
            qb: gpu.alloc::<f32>(batch_cap * qd)?,
            kkb: gpu.alloc::<f32>(batch_cap * kvd)?,
            vvb: gpu.alloc::<f32>(batch_cap * kvd)?,
            gb: gpu.alloc::<f32>(batch_cap * cfg.num_heads)?,
            attnb: gpu.alloc::<f32>(batch_cap * qd)?,
            logitsb: gpu.alloc::<f32>(batch_cap * vocab)?,
            xqg: gpu.alloc::<i8>(batch_cap * h)?,
            xsg: gpu.alloc::<f32>(batch_cap * (h / GROUP))?,
            logitsg: gpu.alloc::<f32>(batch_cap * vocab)?,
            pos_b: gpu.alloc::<u32>(batch_cap)?,
            tbl_off_b: gpu.alloc::<u32>(batch_cap)?,
            conv_off_b: gpu.alloc::<u32>(batch_cap)?,
            tbl_all: gpu.alloc::<u32>(max_seqs * kv_blocks.max(1))?,
            snap_page: gpu.alloc::<u32>(batch_cap)?,
            snap_dummy: gpu.alloc::<f32>(1)?,
            pacc: gpu.alloc::<f32>(MAX_SPLITS * cfg.num_heads * cfg.head_dim)?,
            pml: gpu.alloc::<f32>(MAX_SPLITS * cfg.num_heads * 2)?,
            pacc_b: gpu.alloc::<f32>(batch_cap * MAX_SPLITS * cfg.num_heads * cfg.head_dim)?,
            pml_b: gpu.alloc::<f32>(batch_cap * MAX_SPLITS * cfg.num_heads * 2)?,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            if i < layer_start || i >= layer_end {
                layers.push(LayerBufs {
                    conv: None,
                    snap: None,
                    k: None,
                    v: None,
                });
                continue;
            }
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
            layer_start,
            layer_end,
            has_embed,
            has_head,
            tp: shard,
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
            prefill_batch,
            batch_cap,
            gemm_force,
            prepacked_i8,
            gemm_q4,
            rgemv,
            rgemv_b,
            wg,
            wg_b,
            subgroup,
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
        let per_wg = (self.wg / self.subgroup).max(1) * self.rgemv;
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
        let force = self
            .gemm_force
            .as_deref()
            .is_some_and(|list| list.split(',').any(|pat| !pat.is_empty() && name.contains(pat)));
        if ntok > self.pb || force {
            if let (Weight::Q4(_), Some(p)) = (weight, packed) {
                let tps = ntok.div_ceil(32);
                let grid = rows.div_ceil(32) * tps * 256;
                unsafe {
                    let kq = if self.gemm_q4 == "lds10" {
                        &self.kernels.gemm_q4_lds10
                    } else {
                        &self.kernels.gemm_q4_lds11
                    };
                    self.gpu.launch(
                        kq,
                        grid,
                        256,
                        &[
                            Arg::Mem(p.get()),
                            Arg::Mem(xq.get()),
                            Arg::Mem(xs.get()),
                            Arg::Mem(y.get()),
                            Arg::U32(rows as u32),
                            Arg::U32(cols as u32),
                            Arg::U32(ntok as u32),
                        ],
                    )?;
                }
                return Ok(());
            }
            if let Some((w8, wsc, _)) = self.prepacked_i8.get(name) {
                if matches!(weight, Weight::Q6(_)) {
                    let grid = rows.div_ceil(32) * ntok.div_ceil(32) * 256;
                    unsafe {
                        self.gpu.launch(
                            &self.kernels.gemm_q6_pp,
                            grid,
                            256,
                            &[
                                Arg::Mem(w8.get()),
                                Arg::Mem(wsc.get()),
                                Arg::Mem(xq.get()),
                                Arg::Mem(xs.get()),
                                Arg::Mem(y.get()),
                                Arg::U32(rows as u32),
                                Arg::U32(cols as u32),
                                Arg::U32(ntok as u32),
                            ],
                        )?;
                    }
                    return Ok(());
                }
            }
            if let Weight::Q6(b) = weight {
                let nsg = (64 / self.gpu.caps.sub_group_size.max(1)).max(1);
                let grid = rows.div_ceil(2 * nsg) * 64;
                unsafe {
                    self.gpu.launch(
                        &self.kernels.gemm_q6,
                        grid,
                        64,
                        &[
                            Arg::Mem(b.get()),
                            Arg::Mem(xq.get()),
                            Arg::Mem(xs.get()),
                            Arg::Mem(y.get()),
                            Arg::U32(rows as u32),
                            Arg::U32(cols as u32),
                            Arg::U32(ntok as u32),
                        ],
                    )?;
                }
                return Ok(());
            }
            return Err(anyhow!("large prefill batch: unsupported weight {name} ntok {ntok}"));
        }
        let (kernel, buffer) = match (weight, packed) {
            (Weight::Q4(_), Some(p)) => (&self.kernels.q4p_b, p),
            (Weight::Q4(b), None) => (&self.kernels.q4_b, b),
            (Weight::Q6(b), _) => (&self.kernels.q6_b, b),
            (Weight::F32(_), _) => return Err(anyhow!("gemv on f32 weight {name}")),
        };
        let per_wg = (self.wg_b / self.subgroup).max(1) * self.rgemv_b;
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
        let per_wg = (self.wg_b / self.subgroup).max(1) * 4;
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
        if !self.has_embed {
            return Err(anyhow!("stage has no embedding"));
        }
        self.forward_step(slot, Some(token), None, pos, true)
    }

    pub fn forward_hidden_seq(
        &mut self,
        slot: usize,
        hidden: &[f32],
        pos: usize,
        need_logits: bool,
    ) -> Result<()> {
        self.forward_step(slot, None, Some(hidden), pos, need_logits)
    }

    pub fn stage_hidden(&self) -> Result<Vec<f32>> {
        self.gpu.download(&self.bufs.x, self.cfg.hidden_size)
    }

    fn forward_step(
        &mut self,
        slot: usize,
        token: Option<u32>,
        hidden: Option<&[f32]>,
        pos: usize,
        need_logits: bool,
    ) -> Result<()> {
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
        if let Some(token) = token {
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
        } else if let Some(hidden) = hidden {
            unsafe {
                self.gpu
                    .queue
                    .enqueue_write_buffer(&mut self.bufs.x, 1, 0, hidden, &[])?;
            }
        } else {
            return Err(anyhow!("no token or hidden input"));
        }
        for layer in self.layer_start..self.layer_end {
            let prefix = format!("blk.{layer}");
            if layer == self.layer_start {
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
            let next_w = if layer + 1 < self.layer_end {
                self.f32_weight(&format!("blk.{}.attn_norm.weight", layer + 1))?
            } else if self.has_head {
                self.f32_weight("output_norm.weight")?
            } else {
                self.f32_weight(&format!("{prefix}.post_ffw_norm.weight"))?
            };
            self.norm_add_quant(&self.bufs.tmp, post_ffn, &self.bufs.x, next_w, h)?;
            if self.trace {
                self.gpu.queue.finish()?;
                let v = self.gpu.download(&self.bufs.x, h)?;
                self.traced.push(v);
            }
        }
        if need_logits && self.has_head {
            self.gemv_auto(
                "token_embd.weight",
                &self.bufs.hq,
                &self.bufs.hs,
                &self.bufs.logits,
                self.cfg.vocab_size,
                h,
            )?;
        }
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn tp_set_x(&mut self, x: &[f32]) -> Result<()> {
        let h = self.cfg.hidden_size;
        if x.len() != h {
            return Err(anyhow!("hidden size mismatch"));
        }
        unsafe {
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.x, 1, 0, x, &[])?;
        }
        Ok(())
    }

    pub fn tp_x(&self) -> Result<Vec<f32>> {
        self.gpu.download(&self.bufs.x, self.cfg.hidden_size)
    }

    pub fn tp_embed(&mut self, token: u32) -> Result<()> {
        if !self.has_embed {
            return Err(anyhow!("tp shard has no embedding"));
        }
        let h = self.cfg.hidden_size;
        let emb = match self.weight("token_embd.weight")? {
            Weight::Q6(b) => b,
            _ => return Err(anyhow!("token_embd must be Q6_K")),
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
        Ok(())
    }

    pub fn tp_norm_attn(&mut self, layer: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let w = self.f32_weight(&format!("blk.{layer}.attn_norm.weight"))?;
        unsafe {
            self.gpu.launch(
                &self.kernels.rms_quant,
                64,
                64,
                &[
                    Arg::Mem(self.bufs.x.get()),
                    Arg::Mem(w.get()),
                    Arg::Mem(self.bufs.hq.get()),
                    Arg::Mem(self.bufs.hs.get()),
                    Arg::U32(h as u32),
                ],
            )?;
        }
        Ok(())
    }

    pub fn tp_norm_inputs(&self) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>)> {
        let h = self.cfg.hidden_size;
        let x = self.gpu.download(&self.bufs.x, h)?;
        let hq = self.gpu.download(&self.bufs.hq, h)?;
        let hs = self.gpu.download(&self.bufs.hs, h / GROUP)?;
        Ok((x, hq, hs))
    }

    pub fn tp_set_norm_inputs(&mut self, x: &[f32], hq: &[i8], hs: &[f32]) -> Result<()> {
        unsafe {
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.x, 1, 0, x, &[])?;
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.hq, 1, 0, hq, &[])?;
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.hs, 1, 0, hs, &[])?;
        }
        Ok(())
    }

    pub fn tp_conv(&mut self, layer: usize, slot: usize, pos: usize) -> Result<Vec<f32>> {
        let (_, conv_off) = self.seq_launch_params(slot, pos + 1)?;
        let prefix = format!("blk.{layer}");
        let h = self.cfg.hidden_size;
        self.gemv_auto(
            &format!("{prefix}.shortconv.in_proj.weight"),
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.bcx,
            3 * h,
            h,
        )?;
        let conv_w = self.f32_weight(&format!("{prefix}.shortconv.conv.weight"))?;
        let state = self
            .layers[layer]
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
        self.gpu.queue.finish()?;
        self.gpu.download(&self.bufs.tmp, h)
    }

    pub fn tp_attn(&mut self, layer: usize, slot: usize, pos: usize) -> Result<Vec<f32>> {
        let shard = self.tp.ok_or_else(|| anyhow!("not a tp shard"))?;
        let (tbl_mem, _) = self.seq_launch_params(slot, pos + 1)?;
        let prefix = format!("blk.{layer}");
        let h = self.cfg.hidden_size;
        let head_dim = self.cfg.head_dim;
        let heads = shard.head_count;
        let kv_heads = shard.kv_head_count;
        let qd = heads * head_dim;
        let kvd = kv_heads * head_dim;
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
            kvd,
            h,
        )?;
        self.gemv_auto(
            &format!("{prefix}.attn_v.weight"),
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.vv,
            kvd,
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
        self.gpu.queue.finish()?;
        self.gpu.download(&self.bufs.tmp, h)
    }

    pub fn tp_attn_norm(&mut self, layer: usize, tmp: &[f32]) -> Result<()> {
        let h = self.cfg.hidden_size;
        unsafe {
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.tmp, 1, 0, tmp, &[])?;
        }
        let prefix = format!("blk.{layer}");
        let post = self.f32_weight(&format!("{prefix}.post_attention_norm.weight"))?;
        let ffn_norm = self.f32_weight(&format!("{prefix}.ffn_norm.weight"))?;
        self.norm_add_quant(&self.bufs.tmp, post, &self.bufs.x, ffn_norm, h)?;
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn tp_ffn_local(&mut self, layer: usize) -> Result<Vec<f32>> {
        let shard = self
            .tp
            .ok_or_else(|| anyhow!("not a tp shard"))?;
        let h = self.cfg.hidden_size;
        let ffn = if shard.ffn_row_count == 0 {
            self.ffn
        } else {
            shard.ffn_row_count
        };
        let prefix = format!("blk.{layer}");
        self.gemv_auto(
            &format!("{prefix}.ffn_ug.weight"),
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.ug,
            2 * ffn,
            h,
        )?;
        unsafe {
            self.gpu.launch(
                &self.kernels.silu_quant,
                ffn / GROUP,
                0,
                &[
                    Arg::Mem(self.bufs.ug.get()),
                    Arg::Mem(self.bufs.actq.get()),
                    Arg::Mem(self.bufs.acts.get()),
                    Arg::U32(ffn as u32),
                    Arg::U32(ffn as u32),
                ],
            )?;
        }
        self.gemv_auto(
            &format!("{prefix}.ffn_down.weight"),
            &self.bufs.actq,
            &self.bufs.acts,
            &self.bufs.tmp,
            h,
            ffn,
        )?;
        self.gpu.queue.finish()?;
        self.gpu.download(&self.bufs.tmp, h)
    }

    pub fn tp_ffn_norm(&mut self, layer: usize, tmp: &[f32]) -> Result<()> {
        let h = self.cfg.hidden_size;
        unsafe {
            self.gpu
                .queue
                .enqueue_write_buffer(&mut self.bufs.tmp, 1, 0, tmp, &[])?;
        }
        let prefix = format!("blk.{layer}");
        let post_ffn = self.f32_weight(&format!("{prefix}.post_ffw_norm.weight"))?;
        let next_w = if layer + 1 < self.cfg.num_layers {
            self.f32_weight(&format!("blk.{}.attn_norm.weight", layer + 1))?
        } else {
            self.f32_weight("output_norm.weight")?
        };
        self.norm_add_quant(&self.bufs.tmp, post_ffn, &self.bufs.x, next_w, h)?;
        self.gpu.queue.finish()?;
        Ok(())
    }

    pub fn tp_lmhead_partial(&mut self) -> Result<Vec<f32>> {
        let shard = self.tp.ok_or_else(|| anyhow!("not a tp shard"))?;
        let h = self.cfg.hidden_size;
        self.gemv_auto(
            "lm_head_shard",
            &self.bufs.hq,
            &self.bufs.hs,
            &self.bufs.logits,
            shard.vocab_count,
            h,
        )?;
        self.gpu.queue.finish()?;
        self.gpu.download(&self.bufs.logits, shard.vocab_count)
    }

    pub fn tp_shard(&self) -> Option<TpShard> {
        self.tp
    }

    pub fn batch_size(&self) -> usize {
        self.pb
    }

    pub fn prefill_batch_size(&self) -> usize {
        self.prefill_batch
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
        let list: [(&str, &Kernel); 23] = [
            ("q4", &self.kernels.q4),
            ("q4p_b", &self.kernels.q4p_b),
            ("q4x4p_b", &self.kernels.q4x4p_b),
            ("gemm_q4_lds10", &self.kernels.gemm_q4_lds10),
            ("gemm_q6", &self.kernels.gemm_q6),
            ("gemm_q6_pp", &self.kernels.gemm_q6_pp),
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

    pub fn step_mixed_input(
        &mut self,
        reqs: &[StepReq],
        hidden: &[f32],
        need_logits: bool,
    ) -> Result<()> {
        self.forward_impl(reqs, need_logits, None, None, None, Some(hidden))
    }

    pub fn forward_batch_hidden(
        &mut self,
        slot: usize,
        hidden: &[f32],
        pos0: usize,
        need_logits: bool,
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        if hidden.is_empty() || !hidden.len().is_multiple_of(h) {
            return Err(anyhow!("hidden batch has wrong size"));
        }
        let ntok = hidden.len() / h;
        let reqs: Vec<StepReq> = (0..ntok)
            .map(|i| StepReq {
                slot,
                token: 0,
                pos: pos0 + i,
            })
            .collect();
        let row = need_logits.then_some(ntok - 1);
        self.forward_impl(&reqs, need_logits, row, None, None, Some(hidden))
    }

    pub fn stage_hidden_batch(&self, ntok: usize) -> Result<Vec<f32>> {
        self.gpu
            .download(&self.bufs.xb, ntok * self.cfg.hidden_size)
    }

    pub fn step_mixed_rows(&mut self, reqs: &[StepReq], logits_rows: &[usize]) -> Result<()> {
        self.forward_impl(reqs, !logits_rows.is_empty(), None, Some(logits_rows), None, None)
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
        self.forward_impl(reqs, need_logits, logits_row, None, None, None)
    }

    fn forward_impl(
        &mut self,
        reqs: &[StepReq],
        need_logits: bool,
        logits_row: Option<usize>,
        logits_rows: Option<&[usize]>,
        snap: Option<&[u32]>,
        input: Option<&[f32]>,
    ) -> Result<()> {
        let ntok = reqs.len();
        if ntok == 0 {
            return Err(anyhow!("empty batch"));
        }
        if ntok > self.batch_cap {
            return Err(anyhow!("batch of {ntok} exceeds prefetch batch {}", self.batch_cap));
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
        let n32 = ntok as u32;
        if let Some(hidden) = input {
            unsafe {
                self.gpu
                    .queue
                    .enqueue_write_buffer(&mut self.bufs.xb, 1, 0, hidden, &[])?;
            }
        } else {
            let emb = match self.weight("token_embd.weight")? {
                Weight::Q6(b) => b,
                _ => return Err(anyhow!("token_embd must be Q6_K for gpu embed")),
            };
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
        }
        for layer in self.layer_start..self.layer_end {
            let prefix = format!("blk.{layer}");
            if layer == self.layer_start {
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
                    let fused = if ntok > self.pb {
                        None
                    } else if self
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
                                    &self.kernels.attn_split_b_h4_p4,
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
            let next_w = if layer + 1 < self.layer_end {
                self.f32_weight(&format!("blk.{}.attn_norm.weight", layer + 1))?
            } else if self.has_head {
                self.f32_weight("output_norm.weight")?
            } else {
                self.f32_weight(&format!("{prefix}.post_ffw_norm.weight"))?
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
        if need_logits && self.has_head {
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
                    Arg::U32(0),
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
            let end = (pos + self.prefill_batch).min(prompt.len());
            let reqs: Vec<StepReq> = (pos..end)
                .map(|p| StepReq {
                    slot,
                    token: prompt[p],
                    pos: p,
                })
                .collect();
            let last = end == prompt.len();
            let row = last.then_some(reqs.len() - 1);
            self.forward_impl(&reqs, need_logits && last, row, None, None, None)?;
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
