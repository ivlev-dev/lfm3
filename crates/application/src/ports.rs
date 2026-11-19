use std::path::Path;

use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStep {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub cached_pages: usize,
    pub used_pages: usize,
    pub hits: u64,
}

pub trait TokenizerPort: Send {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn eos_id(&self) -> u32;
}

pub trait InferenceBackend: Send {
    fn create_sequence(&mut self) -> Result<usize>;
    fn free_sequence(&mut self, slot: usize) -> Result<()>;
    fn prefill(&mut self, slot: usize, tokens: &[u32], need_logits: bool) -> Result<usize>;
    fn prefill_begin(&mut self, slot: usize, tokens: &[u32]) -> Result<usize>;
    fn prefill_finish(&mut self, slot: usize) -> Result<()>;
    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()>;
    fn logits(&self) -> Result<Vec<f32>>;
    fn logits_row(&self, row: usize) -> Result<Vec<f32>>;
    fn max_ctx(&self) -> usize;
    fn max_sequences(&self) -> usize {
        16
    }
    fn prefill_batch_size(&self) -> usize {
        128
    }
    fn free_pages(&self) -> usize;
    fn cache_stats(&self) -> CacheStats;
}

pub type BackendLoader = Box<dyn FnOnce(&Path, usize) -> Result<Box<dyn InferenceBackend>> + Send>;
pub type TokenizerLoader = Box<dyn FnOnce(&Path) -> Result<Box<dyn TokenizerPort>> + Send>;
