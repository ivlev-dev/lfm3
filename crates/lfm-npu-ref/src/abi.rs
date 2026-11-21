//! Зеркала будущего C-ABI NPU-бэкенда.
//!
//! Модуль фиксирует только раскладку структур и значения перечислений.
//! Никакой логики исполнения здесь нет: DSP-порт получает те же типы
//! побайтово, а host-эталон использует их как общий контракт.
//!
//! Все структуры `#[repr(C)]`. Смещения и размеры проверяются
//! `const`-ассертами и тестами модуля, поэтому случайное изменение поля
//! ломает сборку, а не протокол на телефоне.

/// Токенов в одной KV-странице (`KV_PAGE` в OpenCL-бэкенде).
pub const PAGE_TOKENS: u32 = 32;

/// Строка шага: одна последовательность (или один чанк prefill).
///
/// Смещения — в элементах/записях соответствующих арен, не байтах:
/// DSP-порт сам умножает на размер элемента.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Row {
    /// id входного токена (первый токен строки при prefill).
    pub token: u32,
    /// Слот последовательности в KV-пуле (`0..max_seqs`).
    pub seq: u32,
    /// Абсолютная позиция первого токена строки в последовательности.
    pub pos: u32,
    /// Сколько токенов несёт строка (>= 1).
    pub ntok: u32,
    /// База staging-кольца: `seq * kv_stage` (в токенах KV).
    pub kv_off: u32,
    /// Смещение таблицы страниц: `seq * pt_stride` (в записях `u32`).
    pub tbl_off: u32,
}

/// Описание одного шага (prefill-чанка или decode-пачки).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepDesc {
    /// Число валидных строк.
    pub n_rows: u32,
    /// Суммарное число токенов во всех строках.
    pub ntok: u32,
    /// Окно внимания; `0` — полный контекст.
    pub window: u32,
    /// Битовая маска: [`FLAG_PREFILL`] и зарезервированные биты.
    pub flags: u32,
    /// Ёмкость массива строк.
    pub max_rows: u32,
    /// Выравнивание/резерв, всегда 0.
    pub reserved: u32,
}

/// Шаг является prefill-чанком (несколько токенов на строку).
pub const FLAG_PREFILL: u32 = 1 << 0;

/// Результат шага.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepOut {
    /// Число обработанных строк.
    pub n_rows: u32,
    /// Число обработанных токенов.
    pub ntok: u32,
    /// Смещение логитов последнего токена каждой строки (в f32-элементах).
    pub logits_off: u32,
    /// Статус шага.
    pub status: Ctl,
}

/// Формат KV в арене.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum KvFormat {
    /// K и V в f16, полная точность.
    #[default]
    F16 = 0,
    /// `kvarn`: K 4 бита, V 2 бита.
    KvarnK4V2 = 1,
    /// `kvarn_k4v4`: K 4 бита, V 4 бита.
    KvarnK4V4 = 2,
}

impl KvFormat {
    /// Является ли формат вариантом `kvarn` (страничные коды).
    pub const fn is_kvarn(self) -> bool {
        matches!(self, Self::KvarnK4V2 | Self::KvarnK4V4)
    }

    /// Бит на канал для V (`0` для f16).
    pub const fn v_bits(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::KvarnK4V2 => 2,
            Self::KvarnK4V4 => 4,
        }
    }
}

/// Описание модели для загрузки/инициализации бэкенда.
///
/// Ровно 16 полей по 4 байта — 64 байта, удобно для копирования в DSP.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelDesc {
    /// Размер скрытого состояния.
    pub hidden: u32,
    /// Размер FFN.
    pub ffn: u32,
    /// Число слоёв.
    pub n_layers: u32,
    /// Число голов внимания.
    pub n_heads: u32,
    /// Размер головы.
    pub head_dim: u32,
    /// `kv_heads * head_dim`.
    pub kv_dim: u32,
    /// Размер словаря.
    pub vocab: u32,
    /// Максимальный контекст.
    pub ctx: u32,
    /// Длина кольца shortconv.
    pub conv_cache: u32,
    /// Окно sliding-attention; `0` — нет.
    pub sliding_window: u32,
    /// Число слоёв, держащих KV.
    pub kv_layers: u32,
    /// Число KV-голов.
    pub kv_heads: u32,
    /// Множитель KV-пула (`1..=16`).
    pub kv_pool: u32,
    /// Размер staging-кольца в токенах (степень двойки, >= 64).
    pub kv_stage: u32,
    /// Максимум одновременных последовательностей.
    pub max_seqs: u32,
    /// Формат KV.
    pub kv_format: KvFormat,
}

/// Категория нехватки памяти.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum OomWhat {
    /// Суммарная арена.
    Arena = 0,
    /// KV-арена.
    Kv = 1,
    /// Staging-арена.
    Staging = 2,
    /// Таблицы страниц.
    PageTable = 3,
    /// Кольца shortconv.
    Conv = 4,
    /// Снапшоты кэша.
    Snap = 5,
    /// Итог по всем аренам.
    #[default]
    Total = 6,
}

/// Описание нехватки памяти в терминах ABI.
///
/// `need`/`have` — байты. Структура 24 байта (u32 + выравнивание + два u64).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Oom {
    /// Что именно не помещается.
    pub what: OomWhat,
    /// Резерв, всегда 0.
    pub reserved: u32,
    /// Требуется байт.
    pub need: u64,
    /// Доступно байт.
    pub have: u64,
}

/// Код завершения шага/команды.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Ctl {
    /// Успех, работа продолжается.
    #[default]
    Ok = 0,
    /// Шаг принят, но у вызывающего остались строки.
    More = 1,
    /// Последовательность завершена.
    Done = 2,
    /// Не хватило памяти (см. [`Oom`]).
    Oom = 3,
    /// Некорректный дескриптор/параметр.
    Invalid = 4,
}

impl Ctl {
    /// Является ли код ошибкой.
    pub const fn is_error(self) -> bool {
        matches!(self, Self::Oom | Self::Invalid)
    }
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<Row>() == 24);
    assert!(align_of::<Row>() == 4);
    assert!(offset_of!(Row, token) == 0);
    assert!(offset_of!(Row, seq) == 4);
    assert!(offset_of!(Row, pos) == 8);
    assert!(offset_of!(Row, ntok) == 12);
    assert!(offset_of!(Row, kv_off) == 16);
    assert!(offset_of!(Row, tbl_off) == 20);

    assert!(size_of::<StepDesc>() == 24);
    assert!(align_of::<StepDesc>() == 4);
    assert!(offset_of!(StepDesc, n_rows) == 0);
    assert!(offset_of!(StepDesc, ntok) == 4);
    assert!(offset_of!(StepDesc, window) == 8);
    assert!(offset_of!(StepDesc, flags) == 12);
    assert!(offset_of!(StepDesc, max_rows) == 16);
    assert!(offset_of!(StepDesc, reserved) == 20);

    assert!(size_of::<StepOut>() == 16);
    assert!(align_of::<StepOut>() == 4);
    assert!(offset_of!(StepOut, n_rows) == 0);
    assert!(offset_of!(StepOut, ntok) == 4);
    assert!(offset_of!(StepOut, logits_off) == 8);
    assert!(offset_of!(StepOut, status) == 12);

    assert!(size_of::<ModelDesc>() == 64);
    assert!(align_of::<ModelDesc>() == 4);
    assert!(offset_of!(ModelDesc, hidden) == 0);
    assert!(offset_of!(ModelDesc, kv_dim) == 20);
    assert!(offset_of!(ModelDesc, kv_format) == 60);

    assert!(size_of::<KvFormat>() == 4);
    assert!(align_of::<KvFormat>() == 4);

    assert!(size_of::<OomWhat>() == 4);
    assert!(size_of::<Oom>() == 24);
    assert!(align_of::<Oom>() == 8);
    assert!(offset_of!(Oom, what) == 0);
    assert!(offset_of!(Oom, reserved) == 4);
    assert!(offset_of!(Oom, need) == 8);
    assert!(offset_of!(Oom, have) == 16);

    assert!(size_of::<Ctl>() == 4);
    assert!(align_of::<Ctl>() == 4);
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn row_is_24_bytes() {
        assert_eq!(size_of::<Row>(), 24);
        assert_eq!(align_of::<Row>(), 4);
        assert_eq!(offset_of!(Row, token), 0);
        assert_eq!(offset_of!(Row, seq), 4);
        assert_eq!(offset_of!(Row, pos), 8);
        assert_eq!(offset_of!(Row, ntok), 12);
        assert_eq!(offset_of!(Row, kv_off), 16);
        assert_eq!(offset_of!(Row, tbl_off), 20);
    }

    #[test]
    fn step_structs_are_stable() {
        assert_eq!(size_of::<StepDesc>(), 24);
        assert_eq!(size_of::<StepOut>(), 16);
        assert_eq!(offset_of!(StepDesc, flags), 12);
        assert_eq!(offset_of!(StepDesc, max_rows), 16);
        assert_eq!(offset_of!(StepOut, status), 12);
    }

    #[test]
    fn model_desc_is_64_bytes() {
        assert_eq!(size_of::<ModelDesc>(), 64);
        assert_eq!(size_of::<ModelDesc>(), 16 * 4);
        assert_eq!(offset_of!(ModelDesc, kv_format), 60);
    }

    #[test]
    fn oom_has_aligned_u64() {
        assert_eq!(size_of::<Oom>(), 24);
        assert_eq!(align_of::<Oom>(), 8);
        assert_eq!(offset_of!(Oom, need), 8);
        assert_eq!(offset_of!(Oom, have), 16);
    }

    #[test]
    fn enum_discriminants_match_protocol() {
        assert_eq!(KvFormat::F16 as u32, 0);
        assert_eq!(KvFormat::KvarnK4V2 as u32, 1);
        assert_eq!(KvFormat::KvarnK4V4 as u32, 2);
        assert_eq!(OomWhat::Arena as u32, 0);
        assert_eq!(OomWhat::Total as u32, 6);
        assert_eq!(Ctl::Ok as u32, 0);
        assert_eq!(Ctl::Done as u32, 2);
        assert_eq!(Ctl::Oom as u32, 3);
        assert_eq!(Ctl::Invalid as u32, 4);
        assert!(Ctl::Oom.is_error());
        assert!(!Ctl::Ok.is_error());
    }

    #[test]
    fn kv_format_bits() {
        assert!(!KvFormat::F16.is_kvarn());
        assert!(KvFormat::KvarnK4V2.is_kvarn());
        assert_eq!(KvFormat::KvarnK4V2.v_bits(), 2);
        assert_eq!(KvFormat::KvarnK4V4.v_bits(), 4);
    }

    #[test]
    fn page_tokens_match_kv_page() {
        assert_eq!(PAGE_TOKENS, 32);
    }
}
