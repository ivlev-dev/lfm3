//! Формулы арен paged-KV и валидация конфигурации.
//!
//! Все размеры — в байтах, смещения — байтовые. Формулы совпадают с
//! `crates/backend-opencl/src/lib.rs` (выделение буферов) и
//! `crates/backend-opencl/kernels/kvarn.cl` (раскладка страницы):
//!
//! * `max_pages = ceil(ctx / 32) * kv_pool`;
//! * `pt_stride = max_pages`;
//! * `kv_stage = max(64, next_pow2(prefill_ntok))`;
//! * KV-арена = `kv_layers * max_pages * page_bytes(format)`;
//! * staging = `kv_layers * max_seqs * kv_stage * kv_dim * 4`
//!   (одно кольцо; K и V на DSP-порте выделяются отдельно);
//! * PT = `max_seqs * pt_stride * 4`;
//! * conv = `max_seqs * (conv_cache - 1) * hidden * 4`;
//! * snap = `snap_slots * (conv_cache - 1) * hidden * 4`.
//!
//! Инварианты, которые проверяет [`arena_layout`]:
//! `kv_stage` — степень двойки и >= 64; `head_dim == 128` для `kvarn`;
//! `max_rows <= kv_stage - 32`; `snap_slots <= max_pages`; `kv_pool` в
//! `1..=16`. При превышении бюджета возвращается [`NpuError::Oom`].

use lfm_domain::ModelConfig;

pub use crate::abi::KvFormat;
use crate::{ArenaKind, NpuError, invalid};

/// Токенов в KV-странице.
pub const KV_PAGE: usize = 32;
/// Размер головы, поддерживаемый `kvarn`.
pub const HEAD_DIM: usize = 128;
/// Размер элемента staging/KV в host-эталоне (f32).
pub const ELEM_BYTES: u64 = 4;
/// Запись таблицы страниц.
pub const PT_ENTRY_BYTES: u64 = 4;
/// Выравнивание начала арены (байт).
pub const ARENA_ALIGN: u64 = 256;
/// Максимальный множитель KV-пула (как `backend.kv_pool`).
pub const MAX_KV_POOL: usize = 16;
/// Верхняя граница `prefill_ntok`, защищающая от переполнения `next_pow2`.
pub const MAX_PREFILL_NTOK: usize = 1 << 24;

/// Число страниц KV: `ceil(ctx / 32) * kv_pool`.
pub fn max_pages(ctx: usize, kv_pool: usize) -> usize {
    ctx.div_ceil(KV_PAGE) * kv_pool
}

/// Размер staging-кольца: `max(64, next_pow2(prefill_ntok))`.
///
/// `prefill_ntok = 0` трактуется как 1 (кольцо минимум 64 токена).
pub fn kv_stage(prefill_ntok: usize) -> usize {
    let n = prefill_ntok.max(1).next_power_of_two();
    n.max(64)
}

/// Байт на одну страницу выбранного формата для `kv_heads` голов.
///
/// Для `kvarn` формула общая по `head_dim`, но валидация арен разрешает
/// только 128 (см. [`arena_layout`]).
pub fn page_bytes(format: KvFormat, kv_heads: usize, head_dim: usize) -> u64 {
    match format {
        KvFormat::F16 => {
            // K и V: 32 токена * head_dim * 2 байта, по всем KV-головам.
            KV_PAGE as u64 * kv_heads as u64 * head_dim as u64 * 2 * 2
        }
        KvFormat::KvarnK4V2 => kvarn_page_bytes(kv_heads, head_dim, 2),
        KvFormat::KvarnK4V4 => kvarn_page_bytes(kv_heads, head_dim, 4),
    }
}

/// Байт на страницу `kvarn` при `v_bits` битах на канал V.
///
/// На голову: `kcode` 4 бита, `ka`/`kc` в f16, `vcode` `v_bits`,
/// `vb`/`vbz` по 32 f16, `vc` f16.
fn kvarn_page_bytes(kv_heads: usize, head_dim: usize, v_bits: u64) -> u64 {
    let hd = head_dim as u64;
    let per_head = KV_PAGE as u64 * hd / 2        // kcode
        + hd * 2                                 // ka
        + KV_PAGE as u64 * 2                     // kc
        + KV_PAGE as u64 * hd * v_bits / 8       // vcode
        + KV_PAGE as u64 * 2                     // vb
        + KV_PAGE as u64 * 2                     // vbz
        + hd * 2; // vc
    per_head * kv_heads as u64
}

/// Конфигурация арен.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaConfig {
    /// Максимальный контекст в токенах.
    pub ctx: usize,
    /// Множитель KV-пула (`1..=16`).
    pub kv_pool: usize,
    /// Размер prefill-чанка (влияет на staging и `kv_stage`).
    pub prefill_ntok: usize,
    /// Число слоёв, держащих KV (attention-слои).
    pub kv_layers: usize,
    /// Число KV-голов.
    pub kv_heads: usize,
    /// Размер головы.
    pub head_dim: usize,
    /// Размер скрытого состояния (для conv/snap).
    pub hidden: usize,
    /// Длина кольца shortconv.
    pub conv_cache: usize,
    /// Максимум одновременных последовательностей.
    pub max_seqs: usize,
    /// Максимум строк в одном шаге.
    pub max_rows: usize,
    /// Число слотов снапшотов кэша.
    pub snap_slots: usize,
    /// Формат KV.
    pub kv_format: KvFormat,
    /// Доступная память в байтах; `None` — без проверки.
    pub mem_budget: Option<u64>,
}

impl ArenaConfig {
    /// Собрать конфигурацию из [`ModelConfig`] и параметров запуска.
    ///
    /// `kv_layers` = число не-conv слоёв, `kv_heads` = максимум по слоям,
    /// `hidden`/`conv_cache`/`head_dim` берутся из модели.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model(
        model: &ModelConfig,
        ctx: usize,
        kv_pool: usize,
        prefill_ntok: usize,
        max_seqs: usize,
        max_rows: usize,
        snap_slots: usize,
        kv_format: KvFormat,
        mem_budget: Option<u64>,
    ) -> Self {
        Self {
            ctx,
            kv_pool,
            prefill_ntok,
            kv_layers: model.attention_layers().count(),
            kv_heads: model.kv_heads.iter().copied().max().unwrap_or(0),
            head_dim: model.head_dim,
            hidden: model.hidden_size,
            conv_cache: model.conv_cache,
            max_seqs,
            max_rows,
            snap_slots,
            kv_format,
            mem_budget,
        }
    }
}

/// Арена: смещение и длина в байтах.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Arena {
    /// Смещение от начала общей области.
    pub offset: u64,
    /// Длина в байтах.
    pub len: u64,
}

impl Arena {
    /// Конец арены (не включая).
    pub const fn end(self) -> u64 {
        self.offset + self.len
    }
}

/// Раскладка всех арен.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Arenas {
    /// KV-арена (страницы).
    pub kv: Arena,
    /// Staging-кольца (одно кольцо; K и V выделяются отдельно на DSP).
    pub staging: Arena,
    /// Таблицы страниц.
    pub pt: Arena,
    /// Кольца shortconv.
    pub conv: Arena,
    /// Снапшоты кэша.
    pub snap: Arena,
    /// Итоговый размер с выравниванием.
    pub total: u64,
}

impl Arenas {
    /// Итоговый размер области в байтах.
    pub const fn total_bytes(&self) -> u64 {
        self.total
    }

    /// Арены в порядке раскладки.
    pub const fn entries(&self) -> [(ArenaKind, Arena); 5] {
        [
            (ArenaKind::Kv, self.kv),
            (ArenaKind::Staging, self.staging),
            (ArenaKind::PageTable, self.pt),
            (ArenaKind::Conv, self.conv),
            (ArenaKind::Snap, self.snap),
        ]
    }
}

/// Посчитать раскладку арен и проверить инварианты.
pub fn arena_layout(cfg: &ArenaConfig) -> Result<Arenas, NpuError> {
    let pages = validate(cfg)?;
    let stage = kv_stage(cfg.prefill_ntok);
    let kv_dim = cfg.kv_heads * cfg.head_dim;
    let page = page_bytes(cfg.kv_format, cfg.kv_heads, cfg.head_dim);
    let conv_elems = cfg.conv_cache.saturating_sub(1);

    let kv_bytes = mul("kv", cfg.kv_layers as u64, pages as u64)?;
    let kv_bytes = mul("kv", kv_bytes, page)?;
    let staging_bytes = mul("staging", cfg.kv_layers as u64, cfg.max_seqs as u64)?;
    let staging_bytes = mul("staging", staging_bytes, stage as u64)?;
    let staging_bytes = mul("staging", staging_bytes, kv_dim as u64)?;
    let staging_bytes = mul("staging", staging_bytes, ELEM_BYTES)?;
    let pt_bytes = mul("page_table", cfg.max_seqs as u64, pages as u64)?;
    let pt_bytes = mul("page_table", pt_bytes, PT_ENTRY_BYTES)?;
    let conv_bytes = mul("conv", cfg.max_seqs as u64, conv_elems as u64)?;
    let conv_bytes = mul("conv", conv_bytes, cfg.hidden as u64)?;
    let conv_bytes = mul("conv", conv_bytes, ELEM_BYTES)?;
    let snap_bytes = mul("snap", cfg.snap_slots as u64, conv_elems as u64)?;
    let snap_bytes = mul("snap", snap_bytes, cfg.hidden as u64)?;
    let snap_bytes = mul("snap", snap_bytes, ELEM_BYTES)?;

    let mut cursor = 0u64;
    let kv = place(&mut cursor, kv_bytes);
    let staging = place(&mut cursor, staging_bytes);
    let pt = place(&mut cursor, pt_bytes);
    let conv = place(&mut cursor, conv_bytes);
    let snap = place(&mut cursor, snap_bytes);

    if let Some(have) = cfg.mem_budget
        && cursor > have
    {
        return Err(NpuError::Oom {
            what: ArenaKind::Total,
            need: cursor,
            have,
        });
    }

    Ok(Arenas {
        kv,
        staging,
        pt,
        conv,
        snap,
        total: cursor,
    })
}

fn validate(cfg: &ArenaConfig) -> Result<usize, NpuError> {
    if cfg.ctx == 0 {
        return Err(invalid("ctx", "должен быть > 0"));
    }
    if !(1..=MAX_KV_POOL).contains(&cfg.kv_pool) {
        return Err(invalid(
            "kv_pool",
            format!("должен быть в 1..={MAX_KV_POOL}, получено {}", cfg.kv_pool),
        ));
    }
    if cfg.prefill_ntok > MAX_PREFILL_NTOK {
        return Err(invalid(
            "prefill_ntok",
            format!("должен быть <= {MAX_PREFILL_NTOK}"),
        ));
    }
    if cfg.kv_layers == 0 {
        return Err(invalid("kv_layers", "должен быть > 0"));
    }
    if cfg.kv_heads == 0 {
        return Err(invalid("kv_heads", "должен быть > 0"));
    }
    if cfg.head_dim == 0 {
        return Err(invalid("head_dim", "должен быть > 0"));
    }
    if cfg.hidden == 0 {
        return Err(invalid("hidden", "должен быть > 0"));
    }
    if cfg.conv_cache < 2 {
        return Err(invalid("conv_cache", "должен быть >= 2"));
    }
    if cfg.max_seqs == 0 {
        return Err(invalid("max_seqs", "должен быть > 0"));
    }
    if cfg.kv_format.is_kvarn() && cfg.head_dim != HEAD_DIM {
        return Err(NpuError::HeadDim { got: cfg.head_dim });
    }

    let stage = kv_stage(cfg.prefill_ntok);
    if !stage.is_power_of_two() || stage < 64 {
        return Err(invalid(
            "kv_stage",
            format!("должен быть степенью двойки и >= 64, получено {stage}"),
        ));
    }
    let ring_rows = stage - KV_PAGE;
    if cfg.max_rows > ring_rows {
        return Err(invalid(
            "max_rows",
            format!(
                "{} > kv_stage - 32 = {ring_rows} (kv_stage {stage})",
                cfg.max_rows
            ),
        ));
    }

    let pages = max_pages(cfg.ctx, cfg.kv_pool);
    if cfg.snap_slots > pages {
        return Err(invalid(
            "snap_slots",
            format!("{} > max_pages = {pages}", cfg.snap_slots),
        ));
    }
    if cfg.snap_slots > 0 && cfg.conv_cache < 2 {
        return Err(invalid("conv_cache", "снапшоты требуют conv_cache >= 2"));
    }
    Ok(pages)
}

fn mul(what: &'static str, a: u64, b: u64) -> Result<u64, NpuError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(what, format!("переполнение {a} * {b}")))
}

fn place(cursor: &mut u64, len: u64) -> Arena {
    let offset = cursor.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
    *cursor = offset + len;
    Arena { offset, len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lfm_domain::{LayerKind, RopeParams};

    fn cfg(ctx: usize, kv_pool: usize, format: KvFormat) -> ArenaConfig {
        ArenaConfig {
            ctx,
            kv_pool,
            prefill_ntok: 128,
            kv_layers: 11,
            kv_heads: 4,
            head_dim: 128,
            hidden: 2048,
            conv_cache: 4,
            max_seqs: 16,
            max_rows: 32,
            snap_slots: 0,
            kv_format: format,
            mem_budget: None,
        }
    }

    #[test]
    fn page_sizes_match_kvarn_kernel() {
        assert_eq!(page_bytes(KvFormat::F16, 4, 128), 65_536);
        assert_eq!(page_bytes(KvFormat::KvarnK4V2, 4, 128), 15_104);
        assert_eq!(page_bytes(KvFormat::KvarnK4V4, 4, 128), 19_200);
        // 11 слоёв * 128 страниц (ctx 4096, пул 1).
        assert_eq!(
            11 * 128 * page_bytes(KvFormat::KvarnK4V2, 4, 128),
            21_266_432
        );
        // 11 слоёв * 256 страниц (ctx 8192, пул 1) — сверено с docs/backends.md.
        assert_eq!(11 * 256 * page_bytes(KvFormat::F16, 4, 128), 184_549_376);
    }

    #[test]
    fn kv_stage_is_power_of_two_and_min_64() {
        assert_eq!(kv_stage(0), 64);
        assert_eq!(kv_stage(1), 64);
        assert_eq!(kv_stage(32), 64);
        assert_eq!(kv_stage(64), 64);
        assert_eq!(kv_stage(65), 128);
        assert_eq!(kv_stage(128), 128);
        assert_eq!(kv_stage(1000), 1024);
        for n in [1usize, 7, 63, 64, 65, 127, 128, 129, 4096, 8192] {
            let s = kv_stage(n);
            assert!(s.is_power_of_two(), "kv_stage({n}) = {s}");
            assert!(s >= 64, "kv_stage({n}) = {s}");
        }
    }

    #[test]
    fn max_pages_rounds_up_context() {
        assert_eq!(max_pages(2048, 1), 64);
        assert_eq!(max_pages(4096, 1), 128);
        assert_eq!(max_pages(8192, 1), 256);
        assert_eq!(max_pages(8192, 16), 4096);
        assert_eq!(max_pages(2049, 1), 65);
        assert_eq!(max_pages(2049, 2), 130);
    }

    #[test]
    fn arena_numbers_ctx_pool_matrix() {
        // ctx, kv_pool, ожидаемое число страниц, KV kvarn (11 слоёв).
        let cases = [
            (2048, 1, 10_633_216u64),
            (2048, 2, 21_266_432),
            (2048, 16, 170_131_456),
            (4096, 1, 21_266_432),
            (4096, 2, 42_532_864),
            (4096, 16, 340_262_912),
            (8192, 1, 42_532_864),
            (8192, 2, 85_065_728),
            (8192, 16, 680_525_824),
        ];
        for (ctx, pool, kv) in cases {
            let arenas = arena_layout(&cfg(ctx, pool, KvFormat::KvarnK4V2)).unwrap();
            assert_eq!(arenas.kv.len, kv, "ctx={ctx} pool={pool}");
        }
        // f16: 11 * 256 * 65 536.
        let arenas = arena_layout(&cfg(8192, 1, KvFormat::F16)).unwrap();
        assert_eq!(arenas.kv.len, 184_549_376);
    }

    #[test]
    fn staging_matches_expected_bytes() {
        // 11 слоёв * 16 seq * 128 токенов * 512 (kv_dim) * 4 байта.
        let arenas = arena_layout(&cfg(4096, 1, KvFormat::KvarnK4V2)).unwrap();
        assert_eq!(arenas.staging.len, 46_137_344);
        assert_eq!(11 * 16 * 128 * 512 * 4, 46_137_344);

        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.prefill_ntok = 64;
        assert_eq!(arena_layout(&c).unwrap().staging.len, 23_068_672);
    }

    #[test]
    fn pt_conv_snap_formulas() {
        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.snap_slots = 128;
        let arenas = arena_layout(&c).unwrap();
        assert_eq!(arenas.pt.len, 16 * 128 * 4);
        assert_eq!(arenas.pt.len, 8_192);
        assert_eq!(arenas.conv.len, 16 * 3 * 2048 * 4);
        assert_eq!(arenas.conv.len, 393_216);
        assert_eq!(arenas.snap.len, 128 * 3 * 2048 * 4);
        assert_eq!(arenas.snap.len, 3_145_728);
    }

    #[test]
    fn arenas_are_aligned_and_total_is_last_end() {
        let arenas = arena_layout(&cfg(4096, 1, KvFormat::KvarnK4V2)).unwrap();
        let mut prev_end = 0u64;
        for (_, a) in arenas.entries() {
            assert_eq!(a.offset % ARENA_ALIGN, 0, "offset {}", a.offset);
            assert!(a.offset >= prev_end);
            if a.len == 0 {
                assert_eq!(a.offset, prev_end.div_ceil(ARENA_ALIGN) * ARENA_ALIGN);
            }
            prev_end = a.end();
        }
        assert_eq!(arenas.total_bytes(), prev_end);
        assert_eq!(arenas.total_bytes() % ARENA_ALIGN, 0);
    }

    #[test]
    fn rejects_bad_head_dim_for_kvarn() {
        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.head_dim = 64;
        assert_eq!(arena_layout(&c).unwrap_err(), NpuError::HeadDim { got: 64 });
        // f16 с другим head_dim допустим.
        let mut c = cfg(4096, 1, KvFormat::F16);
        c.head_dim = 64;
        assert!(arena_layout(&c).is_ok());
    }

    #[test]
    fn rejects_too_many_rows_for_stage() {
        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.prefill_ntok = 64; // kv_stage = 64
        c.max_rows = 32;
        assert!(arena_layout(&c).is_ok());
        c.max_rows = 33;
        let err = arena_layout(&c).unwrap_err();
        assert!(matches!(
            err,
            NpuError::Invalid {
                what: "max_rows",
                ..
            }
        ));
    }

    #[test]
    fn rejects_too_many_snap_slots() {
        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.snap_slots = 129;
        let err = arena_layout(&c).unwrap_err();
        assert!(matches!(
            err,
            NpuError::Invalid {
                what: "snap_slots",
                ..
            }
        ));
    }

    #[test]
    fn rejects_kv_pool_out_of_range() {
        for pool in [0, 17, 100] {
            let err = arena_layout(&cfg(4096, pool, KvFormat::KvarnK4V2)).unwrap_err();
            assert!(matches!(
                err,
                NpuError::Invalid {
                    what: "kv_pool",
                    ..
                }
            ));
        }
    }

    #[test]
    fn oom_reports_need_and_have() {
        let mut c = cfg(4096, 1, KvFormat::KvarnK4V2);
        c.snap_slots = 128;
        let total = arena_layout(&c).unwrap().total_bytes();
        c.mem_budget = Some(total - 1);
        let err = arena_layout(&c).unwrap_err();
        match &err {
            NpuError::Oom { what, need, have } => {
                assert_eq!(*what, ArenaKind::Total);
                assert_eq!(*need, total);
                assert_eq!(*have, total - 1);
            }
            other => panic!("ожидался Oom, получено {other:?}"),
        }
        let abi = err.to_abi_oom().unwrap();
        assert_eq!(abi.what, crate::abi::OomWhat::Total);
        assert_eq!(abi.need, total);
        assert_eq!(abi.have, total - 1);

        c.mem_budget = Some(total);
        assert!(arena_layout(&c).is_ok());
    }

    #[test]
    fn zero_snapshots_need_no_memory() {
        let arenas = arena_layout(&cfg(4096, 1, KvFormat::KvarnK4V2)).unwrap();
        assert_eq!(arenas.snap.len, 0);
    }

    #[test]
    fn from_model_collects_kv_layers() {
        let model = ModelConfig {
            arch: "lfm3".into(),
            hidden_size: 2048,
            ffn_size: 8192,
            num_layers: 6,
            num_heads: 16,
            head_dim: 128,
            kv_heads: vec![4, 0, 4, 0, 4, 0],
            layer_kinds: vec![
                LayerKind::FullAttention,
                LayerKind::Conv,
                LayerKind::SlidingAttention,
                LayerKind::Conv,
                LayerKind::FullAttention,
                LayerKind::Conv,
            ],
            vocab_size: 65_536,
            norm_eps: 1e-5,
            conv_cache: 4,
            sliding_window: 512,
            rope_full: RopeParams {
                theta: 1e6,
                dim: 128,
                rotated_dim: 128,
            },
            rope_sliding: RopeParams {
                theta: 1e6,
                dim: 128,
                rotated_dim: 128,
            },
            tied_embeddings: true,
        };
        let c = ArenaConfig::from_model(&model, 4096, 1, 128, 16, 32, 0, KvFormat::KvarnK4V2, None);
        assert_eq!(c.kv_layers, 3);
        assert_eq!(c.kv_heads, 4);
        assert_eq!(c.hidden, 2048);
        assert_eq!(c.conv_cache, 4);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.ctx, 4096);
    }
}
