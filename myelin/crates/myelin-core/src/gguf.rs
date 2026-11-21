//! Minimal, parser-independent view of GGUF metadata.
//!
//! The real reader lands in a later stage. For now tests build a
//! [`GgufMeta`] by hand, and architecture adapters only see this typed map.

use std::collections::BTreeMap;

use crate::CoreError;

/// One metadata entry. Arrays and tensor dimensions are kept separate from
/// scalars so adapters cannot silently read the wrong shape.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Str(String),
    I32Array(Vec<i32>),
    U32Array(Vec<u32>),
    BoolArray(Vec<bool>),
    /// Tensor dimensions in GGUF order (ne[0] first).
    Dims(Vec<u64>),
}

impl MetaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_i32_array(&self) -> Option<&[i32]> {
        match self {
            Self::I32Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<&[u32]> {
        match self {
            Self::U32Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_bool_array(&self) -> Option<&[bool]> {
        match self {
            Self::BoolArray(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_dims(&self) -> Option<&[u64]> {
        match self {
            Self::Dims(values) => Some(values),
            _ => None,
        }
    }
}

/// Metadata of one GGUF file, keyed by GGUF key names.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMeta {
    architecture: String,
    entries: BTreeMap<String, MetaValue>,
}

impl GgufMeta {
    pub fn new(architecture: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            entries: BTreeMap::new(),
        }
    }

    /// `general.architecture` value.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn set(&mut self, key: impl Into<String>, value: MetaValue) -> &mut Self {
        self.entries.insert(key.into(), value);
        self
    }

    pub fn set_tensor_dims(&mut self, name: impl Into<String>, dims: Vec<u64>) -> &mut Self {
        self.set(name, MetaValue::Dims(dims))
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries.get(key)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        matches!(self.get(name), Some(MetaValue::Dims(_)))
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetaValue::as_str)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(MetaValue::as_u32)
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.get(key).and_then(MetaValue::as_i32)
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(MetaValue::as_f32)
    }

    pub fn get_i32_array(&self, key: &str) -> Option<&[i32]> {
        self.get(key).and_then(MetaValue::as_i32_array)
    }

    pub fn get_u32_array(&self, key: &str) -> Option<&[u32]> {
        self.get(key).and_then(MetaValue::as_u32_array)
    }

    pub fn get_bool_array(&self, key: &str) -> Option<&[bool]> {
        self.get(key).and_then(MetaValue::as_bool_array)
    }

    pub fn tensor_dims(&self, name: &str) -> Option<&[u64]> {
        self.get(name).and_then(MetaValue::as_dims)
    }

    /// Prefix a GGUF key with the architecture name, for example
    /// `block_count` -> `<arch>.block_count`.
    pub fn arch_key(&self, suffix: &str) -> String {
        format!("{}.{}", self.architecture, suffix)
    }

    pub fn get_arch_str(&self, suffix: &str) -> Option<&str> {
        self.get_str(&self.arch_key(suffix))
    }

    pub fn get_arch_u32(&self, suffix: &str) -> Option<u32> {
        self.get_u32(&self.arch_key(suffix))
    }

    pub fn get_arch_f32(&self, suffix: &str) -> Option<f32> {
        self.get_f32(&self.arch_key(suffix))
    }

    pub fn get_arch_i32_array(&self, suffix: &str) -> Option<&[i32]> {
        self.get_i32_array(&self.arch_key(suffix))
    }

    pub fn get_arch_bool_array(&self, suffix: &str) -> Option<&[bool]> {
        self.get_bool_array(&self.arch_key(suffix))
    }

    pub fn require_arch_u32(&self, suffix: &str) -> Result<u32, CoreError> {
        self.get_arch_u32(suffix)
            .ok_or_else(|| CoreError::MissingKey(self.arch_key(suffix)))
    }

    pub fn require_arch_f32(&self, suffix: &str) -> Result<f32, CoreError> {
        self.get_arch_f32(suffix)
            .ok_or_else(|| CoreError::MissingKey(self.arch_key(suffix)))
    }

    pub fn require_arch_str(&self, suffix: &str) -> Result<&str, CoreError> {
        self.get_arch_str(suffix)
            .ok_or_else(|| CoreError::MissingKey(self.arch_key(suffix)))
    }

    pub fn require_arch_i32_array(&self, suffix: &str) -> Result<&[i32], CoreError> {
        self.get_arch_i32_array(suffix)
            .ok_or_else(|| CoreError::MissingKey(self.arch_key(suffix)))
    }

    pub fn require_arch_bool_array(&self, suffix: &str) -> Result<&[bool], CoreError> {
        self.get_arch_bool_array(suffix)
            .ok_or_else(|| CoreError::MissingKey(self.arch_key(suffix)))
    }

    pub fn require_tensor_dims(&self, name: &str) -> Result<&[u64], CoreError> {
        self.tensor_dims(name)
            .ok_or_else(|| CoreError::MissingKey(name.to_owned()))
    }
}
