//! Staging of int4 GEMV matrices and raw images on the `lfmgemv` skeleton.
//!
//! `myelin-pack gemv` produces a blob with a 128-byte header, the row-major
//! `[N][K/2]` weights, an f32 scale table and a Q16 scale table. The HVX
//! kernel wants `b->w` to point at the **weights alone**, so this module
//! uploads `blob[weight_base..weight_base + weight_bytes]` and passes the f32
//! table through the stand's `set_scale` payload.
//!
//! Call order is the engineered one from `host_main.c` / `ffn_main.c`:
//! `alloc_buf`, `set_shape(with_scale = 1)`, `load_buf` chunks, `set_scale`.

use crate::error::TransportError;
use crate::hvx::HvxClient;
use crate::method::method_id;
use myelin_pack::gemv::{GemvHeader, GemvManifest};

/// Bytes per `load_buf` RPC. Large chunks amortise the ~150 µs RPC cost.
pub const GEMV_LOAD_CHUNK: usize = 4 * 1024 * 1024;

/// A verified int4 GEMV blob plus its manifest.
#[derive(Debug, Clone)]
pub struct PackedGemv {
    name: String,
    manifest: GemvManifest,
    blob: Vec<u8>,
}

impl PackedGemv {
    /// Adopt a manifest/blob pair after checking that they agree.
    pub fn new(
        name: impl Into<String>,
        manifest: GemvManifest,
        blob: Vec<u8>,
    ) -> Result<Self, TransportError> {
        manifest
            .verify_blob(&blob)
            .map_err(|_| TransportError::InvalidLength {
                what: "gemv manifest does not describe this blob",
                len: blob.len(),
            })?;
        Ok(Self {
            name: name.into(),
            manifest,
            blob,
        })
    }

    /// Read both files from disk.
    pub fn read(
        name: impl Into<String>,
        manifest: impl AsRef<std::path::Path>,
        blob: impl AsRef<std::path::Path>,
    ) -> Result<Self, TransportError> {
        let blob = std::fs::read(blob.as_ref()).map_err(|_| TransportError::InvalidLength {
            what: "gemv blob cannot be read",
            len: 0,
        })?;
        let manifest =
            GemvManifest::read(manifest.as_ref()).map_err(|_| TransportError::InvalidLength {
                what: "gemv manifest cannot be read",
                len: 0,
            })?;
        Self::new(name, manifest, blob)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn manifest(&self) -> &GemvManifest {
        &self.manifest
    }

    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    pub fn header(&self) -> GemvHeader {
        GemvHeader::parse(&self.blob).expect("PackedGemv verified the header")
    }

    /// Upload as a resident GEMV matrix with `set_scale` for these scales.
    pub fn stage(
        &self,
        client: &HvxClient,
        act_scale: f32,
        out_scale: f32,
        chunk: usize,
    ) -> Result<i32, TransportError> {
        let header = self.header();
        let weights = &self.blob[header.weight_base..header.weight_base + header.weight_bytes];
        let handle = client.alloc_buf(weights.len())?;
        let result = (|| {
            client.set_shape(handle, header.n_padded as i32, header.k_padded as i32, 1)?;
            for (index, part) in weights.chunks(chunk.max(1)).enumerate() {
                client.load_buf(handle, (index * chunk.max(1)) as i32, part)?;
            }
            client.set_scale(handle, &self.scale_payload(act_scale, out_scale))
        })();
        if result.is_err() {
            let _ = client.release_buf(handle);
        }
        result.map(|()| handle)
    }

    /// `set_scale` payload: `N`, `K`, activation scale, output scale, `sw[N]`.
    pub fn scale_payload(&self, act_scale: f32, out_scale: f32) -> Vec<u8> {
        let n = self.manifest.n_padded;
        let mut payload = Vec::with_capacity(16 + 4 * n);
        payload.extend_from_slice(&(n as i32).to_le_bytes());
        payload.extend_from_slice(&(self.manifest.k_padded as i32).to_le_bytes());
        payload.extend_from_slice(&act_scale.to_le_bytes());
        payload.extend_from_slice(&out_scale.to_le_bytes());
        for scale in &self.manifest.scales_f32 {
            payload.extend_from_slice(&scale.to_le_bytes());
        }
        payload
    }
}

/// Upload a raw image (conv blob, KV cache, block table) unchanged.
pub fn stage_blob(client: &HvxClient, blob: &[u8], chunk: usize) -> Result<i32, TransportError> {
    let handle = client.alloc_buf(blob.len())?;
    let result = (|| {
        for (index, part) in blob.chunks(chunk.max(1)).enumerate() {
            client.load_buf(handle, (index * chunk.max(1)) as i32, part)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = client.release_buf(handle);
    }
    result.map(|()| handle)
}

/// Allocate a zeroed resident buffer of `bytes`.
pub fn stage_zero(client: &HvxClient, bytes: usize, chunk: usize) -> Result<i32, TransportError> {
    let zeros = vec![0u8; chunk.max(1).min(bytes.max(1))];
    let handle = client.alloc_buf(bytes)?;
    let result = (|| {
        let mut offset = 0usize;
        while offset < bytes {
            let part = (bytes - offset).min(zeros.len());
            client.load_buf(handle, offset as i32, &zeros[..part])?;
            offset += part;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = client.release_buf(handle);
    }
    result.map(|()| handle)
}

/// Upload one row of per-token data into a resident cache at `offset`.
pub fn load_bytes(
    client: &HvxClient,
    handle: i32,
    offset: i32,
    data: &[u8],
) -> Result<(), TransportError> {
    client.load_buf(handle, offset, data)
}

/// `load_buf` method ID, for tests that record the transport calls.
pub const LOAD_BUF_METHOD: u32 = method_id::LOAD_BUF;

#[cfg(test)]
mod tests {
    use super::*;
    use myelin_pack::{gemv, Matrix, PackMeta};

    fn packed() -> PackedGemv {
        let k = 256;
        let n = 32;
        let matrix = Matrix::new(
            k,
            n,
            (0..k * n)
                .map(|index| (index as f32 * 0.13).sin())
                .collect(),
        )
        .expect("matrix");
        let meta = PackMeta {
            tensor: "test".to_string(),
            source: None,
        };
        let blob = gemv::build(&matrix, &meta).expect("build");
        let manifest = GemvManifest::from_blob(&meta, &blob);
        PackedGemv::new("test", manifest, blob.blob).expect("packed")
    }

    #[test]
    fn scale_payload_matches_the_stand_layout() {
        let packed = packed();
        let payload = packed.scale_payload(0.5, 0.25);
        assert_eq!(payload.len(), 16 + 4 * packed.manifest().n_padded);
        assert_eq!(
            &payload[0..4],
            &(packed.manifest().n_padded as i32).to_le_bytes()
        );
        assert_eq!(
            &payload[4..8],
            &(packed.manifest().k_padded as i32).to_le_bytes()
        );
        assert_eq!(&payload[8..12], &0.5f32.to_le_bytes());
        assert_eq!(&payload[12..16], &0.25f32.to_le_bytes());
    }

    #[test]
    fn weights_section_is_not_the_blob_start() {
        let packed = packed();
        let header = packed.header();
        assert_eq!(header.weight_base, 128);
        assert_eq!(
            packed.blob().len(),
            header.scale_q16_base + header.scale_q16_bytes
        );
    }
}
