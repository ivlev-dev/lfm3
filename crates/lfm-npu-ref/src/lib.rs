#![forbid(unsafe_code)]
//! Эталонная (host) реализация семантики paged-KV и кодека `kvarn`.
//!
//! Крейт — независимый оракул для будущего NPU/DSP-бэкенда. Он повторяет
//! ровно те правила, по которым уже работает OpenCL-бэкенд
//! (`crates/backend-opencl/kernels/kvarn.cl`):
//!
//! * [`abi`] — побайтовые зеркала C-ABI (`Row`, `StepDesc`, `ModelDesc`, ...);
//! * [`layout`] — формулы арен, размеры страниц, валидация конфигурации;
//! * [`kvarn`] — скалярный кодек K4/V2 и K4/V4 с вращением Адамара.
//!
//! Forward-модель (шаг, слои, внимание) сознательно не входит в этап P0:
//! сначала фиксируются контракты и кодек, затем поверх них пишется `step`.
//!
//! # Пример
//!
//! ```
//! use lfm_npu_ref::layout::{ArenaConfig, arena_layout};
//! use lfm_npu_ref::KvFormat;
//!
//! let cfg = ArenaConfig {
//!     ctx: 4096,
//!     kv_pool: 1,
//!     prefill_ntok: 128,
//!     kv_layers: 11,
//!     kv_heads: 4,
//!     head_dim: 128,
//!     hidden: 2048,
//!     conv_cache: 4,
//!     max_seqs: 16,
//!     max_rows: 32,
//!     snap_slots: 0,
//!     kv_format: KvFormat::KvarnK4V2,
//!     mem_budget: None,
//! };
//! let arenas = arena_layout(&cfg).unwrap();
//! assert_eq!(arenas.kv.len, 11 * 128 * 15_104);
//! assert_eq!(arenas.staging.len, 11 * 16 * 128 * 512 * 4);
//! ```

pub mod abi;
pub mod kvarn;
pub mod layout;

pub use abi::{Ctl, KvFormat, ModelDesc, Oom, OomWhat, Row, StepDesc, StepOut};
pub use kvarn::{KvarnCodec, PageCodes};
pub use layout::{Arena, ArenaConfig, Arenas, arena_layout, kv_stage, max_pages, page_bytes};

use thiserror::Error;

/// Категория арены/памяти для ошибок нехватки места.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArenaKind {
    /// KV-арена.
    Kv,
    /// Staging-кольца.
    Staging,
    /// Таблицы страниц.
    PageTable,
    /// Кольца shortconv.
    Conv,
    /// Снапшоты кэша.
    Snap,
    /// Сумма всех арен.
    Total,
}

impl ArenaKind {
    /// Человекочитаемое имя для сообщений об ошибке.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Kv => "kv",
            Self::Staging => "staging",
            Self::PageTable => "page_table",
            Self::Conv => "conv",
            Self::Snap => "snap",
            Self::Total => "total",
        }
    }

    /// Код категории для [`Oom`].
    pub const fn oom_code(self) -> OomWhat {
        match self {
            Self::Kv => OomWhat::Kv,
            Self::Staging => OomWhat::Staging,
            Self::PageTable => OomWhat::PageTable,
            Self::Conv => OomWhat::Conv,
            Self::Snap => OomWhat::Snap,
            Self::Total => OomWhat::Total,
        }
    }
}

/// Ошибки эталонной реализации.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NpuError {
    /// Не хватает памяти под арену.
    #[error(
        "нехватка памяти ({}): нужно {} Б, доступно {} Б",
        .what.as_str(),
        .need,
        .have
    )]
    Oom {
        /// Что не помещается.
        what: ArenaKind,
        /// Требуется байт.
        need: u64,
        /// Доступно байт.
        have: u64,
    },

    /// Некорректный параметр конфигурации.
    #[error("некорректный параметр {what}: {detail}")]
    Invalid {
        /// Имя параметра.
        what: &'static str,
        /// Пояснение.
        detail: String,
    },

    /// `kvarn` требует head_dim = 128.
    #[error("kvarn поддерживает только head_dim = 128, получено {got}")]
    HeadDim {
        /// Фактический head_dim.
        got: usize,
    },

    /// Неверная длина входной страницы.
    #[error("буфер страницы: ожидалось {want} элементов f16, получено {got}")]
    PageLen {
        /// Ожидалось элементов.
        want: usize,
        /// Получено элементов.
        got: usize,
    },

    /// Формат не поддерживается кодеком `kvarn`.
    #[error("формат {0:?} не является kvarn-форматом")]
    NotKvarn(KvFormat),
}

impl NpuError {
    /// Преобразовать ошибку в ABI-структуру [`Oom`], если это OOM.
    pub fn to_abi_oom(&self) -> Option<Oom> {
        match *self {
            Self::Oom { what, need, have } => Some(Oom {
                what: what.oom_code(),
                reserved: 0,
                need,
                have,
            }),
            _ => None,
        }
    }
}

/// Удобный конструктор ошибки некорректного параметра.
pub(crate) fn invalid(what: &'static str, detail: impl Into<String>) -> NpuError {
    NpuError::Invalid {
        what,
        detail: detail.into(),
    }
}
