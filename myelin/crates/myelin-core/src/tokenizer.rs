//! Tokenizer boundary.
//!
//! The core only needs text-to-ids, ids-to-text and the EOS id. The host
//! adapter reuses the repository tokenizer (`lfm-tokenizer`, backed by the HF
//! `tokenizers` crate) behind the `hf-tokenizer` feature so the Android/FFI
//! build does not pull the oniguruma C dependency.

use crate::RuntimeError;

/// Text tokenizer used by a session.
pub trait Tokenizer {
    /// Encode text into token ids.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, RuntimeError>;

    /// Decode token ids back into text.
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, RuntimeError>;

    /// End-of-sequence token id of this vocabulary.
    fn eos_id(&self) -> u32;
}

#[cfg(feature = "hf-tokenizer")]
pub use lfm_tokenizer::LfmTokenizer;

#[cfg(feature = "hf-tokenizer")]
impl Tokenizer for lfm_tokenizer::LfmTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, RuntimeError> {
        lfm_tokenizer::LfmTokenizer::encode(self, text, add_special_tokens)
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, RuntimeError> {
        lfm_tokenizer::LfmTokenizer::decode(self, ids, skip_special_tokens)
            .map_err(|error| RuntimeError::Tokenizer(error.to_string()))
    }

    fn eos_id(&self) -> u32 {
        lfm_tokenizer::LfmTokenizer::eos_id(self)
    }
}
