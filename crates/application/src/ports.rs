use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStep {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
}

pub trait TokenizerPort {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn eos_id(&self) -> u32;
}

pub trait InferenceBackend {
    fn create_sequence(&mut self) -> Result<usize>;
    fn free_sequence(&mut self, slot: usize) -> Result<()>;
    fn prefill(&mut self, slot: usize, tokens: &[u32], need_logits: bool) -> Result<usize>;
    fn step(&mut self, steps: &[SequenceStep], need_logits: bool) -> Result<()>;
    fn logits(&self) -> Result<Vec<f32>>;
    fn logits_row(&self, row: usize) -> Result<Vec<f32>>;
}
