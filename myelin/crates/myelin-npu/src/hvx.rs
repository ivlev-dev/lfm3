//! Typed wrappers around the `lfmgemv` FastRPC interface.
//!
//! [`HvxClient`] owns one FastRPC handle and turns the IDL methods into Rust
//! requests/responses. It closes the handle on drop.

use std::sync::Arc;

use crate::error::TransportError;
use crate::method::{
    decode_i32s, method_id, AllocBufScalars, AttachOutScalars, AttnRequest, AttnResponse,
    DetachOutScalars, FfnRequest, FfnResponse, GemvRequest, GemvResponse, LoadBufScalars,
    ReleaseBufScalars, SetScaleScalars, SetShapeScalars, TeamInfoScalars,
};
use crate::transport::{Call, Handle, Transport};

/// Client for one open `lfmgemv` handle.
pub struct HvxClient {
    transport: Arc<dyn Transport>,
    module: String,
    handle: Option<Handle>,
}

impl HvxClient {
    /// Open `module` (normally `"lfmgemv"`) on `transport`.
    pub fn open(transport: Arc<dyn Transport>, module: &str) -> Result<Self, TransportError> {
        let handle = transport.open(module)?;
        Ok(Self {
            transport,
            module: module.to_owned(),
            handle: Some(handle),
        })
    }

    /// Open handle, or `None` after [`HvxClient::close`].
    pub const fn handle(&self) -> Option<Handle> {
        self.handle
    }

    /// Module this client talks to.
    pub fn module(&self) -> &str {
        &self.module
    }

    /// Close the handle. Idempotent.
    pub fn close(&mut self) -> Result<(), TransportError> {
        match self.handle.take() {
            Some(handle) => self.transport.close(handle),
            None => Ok(()),
        }
    }

    fn invoke(&self, call: Call<'_, '_>) -> Result<(), TransportError> {
        let handle = self.handle.ok_or_else(|| TransportError::Closed {
            module: self.module.clone(),
        })?;
        self.transport.invoke(handle, call)
    }

    /// Allocate a resident DSP buffer; returns its handle.
    pub fn alloc_buf(&self, nbytes: usize) -> Result<i32, TransportError> {
        let nbytes = i32::try_from(nbytes).map_err(|_| TransportError::BufferTooSmall {
            buffer: "alloc_buf nbytes",
            expected: nbytes,
            got: i32::MAX as usize,
        })?;
        let prim = AllocBufScalars { nbytes }.to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::ALLOC_BUF,
            prim_in: &prim,
            prim_out: &mut out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })?;
        Ok(i32::from_le_bytes(out))
    }

    /// Declare `N x K` shape and whether dequant tables are needed.
    pub fn set_shape(&self, h: i32, n: i32, k: i32, with_scale: i32) -> Result<(), TransportError> {
        let prim = SetShapeScalars {
            h,
            n,
            k,
            with_scale,
        }
        .to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::SET_SHAPE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Copy `data` into the resident buffer at byte offset `off`.
    pub fn load_buf(&self, h: i32, off: i32, data: &[u8]) -> Result<(), TransportError> {
        let prim = LoadBufScalars { h, off }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::LOAD_BUF,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[data],
            out_bufs: &mut no_out,
        })
    }

    /// Upload per-row fp32 scales (the stand's `16 + 4*N` byte prefix layout).
    pub fn set_scale(&self, h: i32, scales: &[u8]) -> Result<(), TransportError> {
        let prim = SetScaleScalars { h }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::SET_SCALE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[scales],
            out_bufs: &mut no_out,
        })
    }

    /// Resident GEMV.
    pub fn gemv(&self, request: GemvRequest<'_>) -> Result<GemvResponse, TransportError> {
        let prim = request.scalars.to_bytes();
        let mut dst = vec![0u8; request.dst_len];
        let mut times = [0i32; 8];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 2] = [dst.as_mut_slice(), i32_bytes_mut(&mut times)];
        self.invoke(Call {
            method: method_id::GEMV,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[request.act],
            out_bufs: &mut out_bufs,
        })?;
        Ok(GemvResponse { dst, times })
    }

    /// One-RPC FFN layer.
    pub fn ffn(&self, request: FfnRequest<'_>) -> Result<FfnResponse, TransportError> {
        let prim = request.scalars.to_bytes();
        let mut dst = vec![0u8; request.dst_len];
        let mut mid = vec![0u8; request.mid_len];
        let mut times = [0i32; 8];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 3] = [
            dst.as_mut_slice(),
            mid.as_mut_slice(),
            i32_bytes_mut(&mut times),
        ];
        self.invoke(Call {
            method: method_id::FFN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[request.act],
            out_bufs: &mut out_bufs,
        })?;
        Ok(FfnResponse { dst, mid, times })
    }

    /// Paged attention, M=1.
    pub fn attn(&self, request: AttnRequest<'_>) -> Result<AttnResponse, TransportError> {
        let prim = request.scalars.to_bytes();
        let table = i32_bytes(request.table);
        let mut outp = vec![0u8; request.out_len];
        let mut times = [0i32; 8];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 2] = [outp.as_mut_slice(), i32_bytes_mut(&mut times)];
        self.invoke(Call {
            method: method_id::ATTN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[request.q, table],
            out_bufs: &mut out_bufs,
        })?;
        Ok(AttnResponse { outp, times })
    }

    /// Map a host rpcmem fd once; subsequent resident calls write straight into
    /// it.
    pub fn attach_out(&self, h: i32, fd: i32, nbytes: i32) -> Result<(), TransportError> {
        let prim = AttachOutScalars { h, fd, nbytes }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::ATTACH_OUT,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Drop the resident output mapping.
    pub fn detach_out(&self, h: i32) -> Result<(), TransportError> {
        let prim = DetachOutScalars { h }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::DETACH_OUT,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Worker-team information (`8` values in the stand).
    pub fn team_info(&self, len: usize) -> Result<Vec<i32>, TransportError> {
        let len_i32 = i32::try_from(len).map_err(|_| TransportError::BufferTooSmall {
            buffer: "team_info",
            expected: len,
            got: i32::MAX as usize,
        })?;
        let prim = TeamInfoScalars { len: len_i32 }.to_bytes();
        let mut out = vec![0u8; len * 4];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 1] = [out.as_mut_slice()];
        self.invoke(Call {
            method: method_id::TEAM_INFO,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut out_bufs,
        })?;
        Ok(decode_i32s(&out))
    }

    /// Release a resident buffer.
    pub fn release_buf(&self, h: i32) -> Result<(), TransportError> {
        let prim = ReleaseBufScalars { h }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::RELEASE_BUF,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }
}

impl Drop for HvxClient {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    if values.is_empty() {
        return &[];
    }
    // SAFETY: `i32` is plain old data and the slice keeps its lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn i32_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    if values.is_empty() {
        return &mut [];
    }
    // SAFETY: `i32` is plain old data and the slice keeps its lifetime.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_buffer_is_little_endian() {
        let mut times = [1i32, 2, 3];
        let bytes = i32_bytes_mut(&mut times);
        assert_eq!(&bytes[0..4], &1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2i32.to_le_bytes());
    }
}
