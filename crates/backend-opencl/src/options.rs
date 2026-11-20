use crate::MAX_SPLITS;
use std::env;

pub const MAX_PREPACK_THREADS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub device: usize,
    pub profile: bool,
    pub kv: String,
    pub prepack: bool,
    pub fuse: bool,
    pub pb: usize,
    pub prefill_ntok: usize,
    pub max_seqs: usize,
    pub cache: bool,
    pub splits: usize,
    pub attn4: bool,
    pub attn2: bool,
    pub flash: bool,
    pub flash2: bool,
    pub b8p: bool,
    pub i8pf: bool,
    pub q4i8: bool,
    pub q6i8: bool,
    pub wg: usize,
    pub wg_b: usize,
    pub rgemv: usize,
    pub rgemv_b: usize,
    pub gemm_force: Option<String>,
    pub prepack_threads: usize,
    pub program_cache: bool,
    pub kv_pool: usize,
    pub q4i8_ffn_only: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            profile: false,
            kv: "f16".to_string(),
            prepack: true,
            fuse: true,
            pb: 6,
            prefill_ntok: 128,
            max_seqs: 8,
            cache: true,
            splits: MAX_SPLITS,
            attn4: true,
            attn2: true,
            flash: true,
            flash2: true,
            b8p: true,
            i8pf: true,
            q4i8: true,
            q6i8: true,
            wg: 256,
            wg_b: 64,
            rgemv: 4,
            rgemv_b: 5,
            gemm_force: None,
            prepack_threads: 0,
            program_cache: true,
            kv_pool: 1,
            q4i8_ffn_only: false,
        }
    }
}

fn flag(name: &str) -> Option<bool> {
    let raw = env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn num(name: &str, lo: usize, hi: usize) -> Option<usize> {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v: &usize| (lo..=hi).contains(v))
}

impl Options {
    pub fn from_env() -> Self {
        let mut o = Self::default();
        if let Some(v) = num("LFM_DEVICE", 0, 64) {
            o.device = v;
        }
        if let Some(v) = flag("LFM_PROFILE") {
            o.profile = v;
        }
        if let Ok(v) = env::var("LFM_KV")
            && !v.is_empty()
        {
            o.kv = v;
        }
        if let Some(v) = flag("LFM_PREPACK") {
            o.prepack = v;
        }
        if let Some(v) = flag("LFM_FUSE") {
            o.fuse = v;
        }
        if let Some(v) = num("LFM_PB", 1, 12) {
            o.pb = v;
        }
        if let Some(v) = num("LFM_PREFILL_NTOK", 1, 128) {
            o.prefill_ntok = v;
        }
        if let Some(v) = num("LFM_SEQS", 1, 32) {
            o.max_seqs = v;
        }
        if let Some(v) = num("LFM_KV_POOL", 1, 16) {
            o.kv_pool = v;
        }
        if let Some(v) = flag("LFM_Q4I8_FFN_ONLY") {
            o.q4i8_ffn_only = v;
        }
        if let Some(v) = flag("LFM_CACHE") {
            o.cache = v;
        }
        if let Some(v) = num("LFM_SPLITS", 1, MAX_SPLITS) {
            o.splits = v;
        }
        if let Some(v) = flag("LFM_ATTN4") {
            o.attn4 = v;
        }
        if let Some(v) = flag("LFM_ATTN2") {
            o.attn2 = v;
        }
        if let Some(v) = flag("LFM_FLASH") {
            o.flash = v;
        }
        if let Some(v) = flag("LFM_FLASH2") {
            o.flash2 = v;
        }
        if let Some(v) = flag("LFM_B8P") {
            o.b8p = v;
        }
        if let Some(v) = flag("LFM_I8PF") {
            o.i8pf = v;
        }
        if let Some(v) = flag("LFM_Q4I8") {
            o.q4i8 = v;
        }
        if let Some(v) = flag("LFM_Q6I8") {
            o.q6i8 = v;
        }
        if let Some(v) = num("LFM_WG", 64, 512) {
            o.wg = v / 64 * 64;
        }
        if let Some(v) = num("LFM_WG_B", 64, 512) {
            o.wg_b = v / 64 * 64;
        }
        if let Some(v) = num("LFM_RGEMV", 1, 16) {
            o.rgemv = v;
        }
        if let Some(v) = num("LFM_RGEMV_B", 1, 8) {
            o.rgemv_b = v;
        }
        if let Ok(v) = env::var("LFM_GEMM_FORCE")
            && !v.is_empty()
        {
            o.gemm_force = Some(v);
        }
        if let Some(v) = num("LFM_PREPACK_THREADS", 0, MAX_PREPACK_THREADS) {
            o.prepack_threads = v;
        }
        if let Some(v) = flag("LFM_PROGRAM_CACHE") {
            o.program_cache = v;
        }
        o
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.kv.as_str(),
            "f16" | "q4" | "k4v4" | "kvarn" | "kvarn_k4v2" | "kvarn_k4v4"
        ) {
            return Err(format!(
                "unknown kv kind {:?} (f16, q4, k4v4, kvarn, kvarn_k4v2, kvarn_k4v4)",
                self.kv
            ));
        }
        if self.pb == 0 || self.pb > 12 {
            return Err(format!("pb must be in 1..=12, got {}", self.pb));
        }
        if self.prefill_ntok == 0 || self.prefill_ntok > 128 {
            return Err(format!(
                "prefill_ntok must be in 1..=128, got {}",
                self.prefill_ntok
            ));
        }
        if self.splits == 0 || self.splits > MAX_SPLITS {
            return Err(format!(
                "splits must be in 1..={MAX_SPLITS}, got {}",
                self.splits
            ));
        }
        if !self.wg.is_multiple_of(64) || !(64..=512).contains(&self.wg) {
            return Err(format!(
                "wg must be a multiple of 64 in 64..=512, got {}",
                self.wg
            ));
        }
        if !self.wg_b.is_multiple_of(64) || !(64..=512).contains(&self.wg_b) {
            return Err(format!(
                "wg_b must be a multiple of 64 in 64..=512, got {}",
                self.wg_b
            ));
        }
        if !(1..=16).contains(&self.rgemv) || !(1..=8).contains(&self.rgemv_b) {
            return Err("rgemv/rgemv_b must be in 1..=8".to_string());
        }
        if !(1..=32).contains(&self.max_seqs) {
            return Err(format!("max_seqs must be in 1..=32, got {}", self.max_seqs));
        }
        if !(1..=16).contains(&self.kv_pool) {
            return Err(format!("kv_pool must be in 1..=16, got {}", self.kv_pool));
        }
        if self.prepack_threads > MAX_PREPACK_THREADS {
            return Err(format!(
                "prepack_threads must be 0 (auto) or in 1..={MAX_PREPACK_THREADS}, got {}",
                self.prepack_threads
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(Options::default().validate().is_ok());
    }

    #[test]
    fn rejects_unknown_kv() {
        let o = Options {
            kv: "q8".to_string(),
            ..Options::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn rejects_bad_pb_and_wg() {
        let o = Options {
            pb: 13,
            ..Options::default()
        };
        assert!(o.validate().is_err());
        let o = Options {
            wg: 100,
            ..Options::default()
        };
        assert!(o.validate().is_err());
        let o = Options {
            splits: 0,
            ..Options::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn prepack_threads_and_program_cache_defaults() {
        let o = Options::default();
        assert_eq!(o.prepack_threads, 0);
        assert!(o.program_cache);
    }

    #[test]
    fn kv_pool_bounds() {
        for pool in [1usize, 2, 16] {
            let o = Options {
                kv_pool: pool,
                ..Options::default()
            };
            assert!(o.validate().is_ok(), "pool={pool}");
        }
        for pool in [0usize, 17] {
            let o = Options {
                kv_pool: pool,
                ..Options::default()
            };
            assert!(o.validate().is_err(), "pool={pool}");
        }
    }

    #[test]
    fn rejects_excessive_prepack_threads() {
        let too_many = Options {
            prepack_threads: MAX_PREPACK_THREADS + 1,
            ..Options::default()
        };
        assert!(too_many.validate().is_err());
        for threads in [0, 1, MAX_PREPACK_THREADS] {
            let o = Options {
                prepack_threads: threads,
                ..Options::default()
            };
            assert!(o.validate().is_ok(), "threads={threads}");
        }
    }
}
