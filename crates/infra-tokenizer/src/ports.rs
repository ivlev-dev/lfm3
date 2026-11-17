use anyhow::Result;

use lfm_app::TokenizerPort;

use crate::LfmTokenizer;

impl TokenizerPort for LfmTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        LfmTokenizer::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        LfmTokenizer::decode(self, ids, skip_special_tokens)
    }

    fn eos_id(&self) -> u32 {
        LfmTokenizer::eos_id(self)
    }
}
