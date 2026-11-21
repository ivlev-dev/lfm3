//! Architecture adapters.
//!
//! An adapter is the only place where family-specific knowledge is allowed.
//! It turns raw metadata into a [`ModelSpec`]; the rest of the crate never
//! matches on GGUF key names.

pub mod lfm3;

use crate::{CoreError, GgufMeta, ModelFamily, ModelSpec};

/// Family adapter contract.
///
/// Adapters are stateless and registered by function pointer in [`ARCHES`].
pub trait Arch {
    /// Return the family when this adapter recognises the metadata.
    fn detect(meta: &GgufMeta) -> Option<ModelFamily>
    where
        Self: Sized;

    /// Build a complete, validated specification.
    fn build_spec(meta: &GgufMeta) -> Result<ModelSpec, CoreError>
    where
        Self: Sized;
}

struct ArchImpl {
    family: ModelFamily,
    detect: fn(&GgufMeta) -> Option<ModelFamily>,
    build_spec: fn(&GgufMeta) -> Result<ModelSpec, CoreError>,
}

/// Registration table. A new model is one new file plus one row here.
const ARCHES: &[ArchImpl] = &[ArchImpl {
    family: ModelFamily::Lfm3,
    detect: lfm3::Lfm3::detect,
    build_spec: lfm3::Lfm3::build_spec,
}];

/// Detect the model family without building a spec.
pub fn detect(meta: &GgufMeta) -> Option<ModelFamily> {
    ARCHES.iter().find_map(|entry| (entry.detect)(meta))
}

/// Build a spec with the adapter that recognises `meta`.
pub fn build_spec(meta: &GgufMeta) -> Result<ModelSpec, CoreError> {
    for entry in ARCHES {
        if let Some(family) = (entry.detect)(meta) {
            debug_assert_eq!(family, entry.family);
            let spec = (entry.build_spec)(meta)?;
            debug_assert_eq!(spec.family, entry.family);
            return Ok(spec);
        }
    }
    Err(CoreError::UnsupportedArchitecture(
        meta.architecture().to_owned(),
    ))
}

/// Whether an adapter is compiled in for `architecture`.
pub fn is_supported(architecture: &str) -> bool {
    ARCHES
        .iter()
        .any(|entry| entry.family.as_str().eq_ignore_ascii_case(architecture))
}
