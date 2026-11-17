use std::path::Path;

use serde_json::Value;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("missing special token: {token}")]
    MissingSpecialToken { token: String },
}

pub type TokenizerError = Error;

pub struct LfmTokenizer {
    tokenizer: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
}

const BOS_TOKEN: &str = "<|startoftext|>";
const EOS_TOKEN: &str = "<|im_end|>";
const PAD_TOKEN: &str = "<|pad|>";
const VOCAB_SIZE: usize = 128000;

impl LfmTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(Error::Io)?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| Error::Parse(e.to_string()))?;
        let json = std::str::from_utf8(&bytes).map_err(|e| Error::Parse(e.to_string()))?;
        let (bos_id, eos_id, pad_id) = special_ids(json)?;
        Ok(Self {
            tokenizer,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| Error::Parse(e.to_string()))?;
        let (bos_id, eos_id, pad_id) = special_ids(json)?;
        Ok(Self {
            tokenizer,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Parse(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Parse(e.to_string()).into())
    }

    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

fn special_ids(json: &str) -> anyhow::Result<(u32, u32, u32)> {
    let value: Value = serde_json::from_str(json).map_err(|e| Error::Parse(e.to_string()))?;
    let added = value
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Parse("added_tokens missing or not array".to_string()))?;

    let find = |needle: &str| -> Option<u32> {
        for entry in added {
            if entry.get("content").and_then(|v| v.as_str()) == Some(needle) {
                if let Some(id) = entry.get("id").and_then(|v| v.as_u64()) {
                    return Some(id as u32);
                }
            }
        }
        None
    };

    let bos_id = find(BOS_TOKEN).ok_or_else(|| Error::MissingSpecialToken {
        token: BOS_TOKEN.to_string(),
    })?;
    let eos_id = find(EOS_TOKEN).ok_or_else(|| Error::MissingSpecialToken {
        token: EOS_TOKEN.to_string(),
    })?;
    let pad_id = find(PAD_TOKEN).ok_or_else(|| Error::MissingSpecialToken {
        token: PAD_TOKEN.to_string(),
    })?;

    Ok((bos_id, eos_id, pad_id))
}