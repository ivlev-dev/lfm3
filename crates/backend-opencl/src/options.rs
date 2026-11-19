use crate::MAX_SPLITS;
use std::env;

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
        if let Some(v) = num("LFM_SEQS", 1, 16) {
            o.max_seqs = v;
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
        if let Some(v) = num("LFM_RGEMV", 1, 32) {
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
        o
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.kv.as_str(),
            "f16" | "q4" | "k4v4" | "kvarn" | "kvarn_k4v2"
        ) {
            return Err(format!(
                "unknown kv kind {:?} (f16, q4, k4v4, kvarn, kvarn_k4v2)",
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
        if !(1..=8).contains(&self.rgemv) || !(1..=8).contains(&self.rgemv_b) {
            return Err("rgemv/rgemv_b must be in 1..=8".to_string());
        }
        if !(1..=16).contains(&self.max_seqs) {
            return Err(format!("max_seqs must be in 1..=16, got {}", self.max_seqs));
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
        let mut o = Options::default();
        o.kv = "q8".to_string();
        assert!(o.validate().is_err());
    }

    #[test]
    fn rejects_bad_pb_and_wg() {
        let mut o = Options::default();
        o.pb = 13;
        assert!(o.validate().is_err());
        let mut o = Options::default();
        o.wg = 100;
        assert!(o.validate().is_err());
        let mut o = Options::default();
        o.splits = 0;
        assert!(o.validate().is_err());
    }
}
