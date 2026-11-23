//! Model-agnostic runtime core for Myelin.
//!
//! The core knows nothing about a concrete model family. All family-specific
//! knowledge lives in an [`arch`] adapter and is reduced to a [`ModelSpec`]
//! (plain data) before the engine starts.
//!
//! Two layers sit on top of the spec:
//!
//! * [`backend`] executes layer-level operations, either on the host
//!   ([`CpuBackend`]) or, later, on the Hexagon ([`DspBackend`] seam);
//! * [`session`] owns the sequence state, drives the layer graph and samples.
//!
//! [`engine::Engine`] is the deterministic placeholder kept for the FFI
//! skeleton; it does not run a model.

pub mod arch;
pub mod backend;
pub mod dequant;
pub mod engine;
pub mod error;
pub mod gguf;
pub mod kv;
pub mod ops;
pub mod radix;
pub mod sampling;
pub mod session;
pub mod spec;
pub mod tokenizer;

pub use arch::{build_spec, detect, Arch};
pub use backend::{
    AttentionCall, Backend, BackendError, CpuBackend, DspBackend, EmbeddingRows, HeadMode,
    HostWeights, LayerNorms, NormSlot, NormWeights, StageProvider,
};
pub use engine::{Engine, StepOut};
pub use error::{CoreError, RuntimeError};
pub use gguf::{GgufMeta, MetaValue};
pub use kv::{KvCache, PagePool, PagedKvCache, PAGE_TOKENS};
pub use ops::RopeTable;
pub use radix::{block_hash, chain_hashes, RadixCache, RadixStats};
pub use sampling::{Sampler, SamplingParams};
pub use session::{GenerateConfig, Generation, Model, Session, StopReason};
pub use spec::{
    AttentionKind, ChatTemplateSpec, DType, Dims, KvFormat, LayerSpec, ModelFamily, ModelSpec,
    Quant, RopeConfig, RopeKind, RopeSpec, TokenizerSpec, WeightFormat,
};
pub use tokenizer::Tokenizer;
