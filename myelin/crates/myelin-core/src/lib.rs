//! Model-agnostic runtime core for Myelin.
//!
//! The core knows nothing about a concrete model family. All family-specific
//! knowledge lives in an [`arch`] adapter and is reduced to a [`ModelSpec`]
//! (plain data) before the engine starts.

pub mod arch;
pub mod engine;
pub mod error;
pub mod gguf;
pub mod spec;

pub use arch::{build_spec, detect, Arch};
pub use engine::{Engine, StepOut};
pub use error::CoreError;
pub use gguf::{GgufMeta, MetaValue};
pub use spec::{
    AttentionKind, ChatTemplateSpec, DType, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec,
    Quant, RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
};
