//! Typed wrappers around the `lfmgemv` FastRPC interface.
//!
//! [`HvxClient`] owns one FastRPC handle and turns the IDL methods into Rust
//! requests/responses. It closes the handle on drop.

use std::sync::Arc;

use crate::error::TransportError;
use crate::method::FfnScalars;
use crate::method::{
    decode_i32s, long_len, method_id, octet_len, AllocBufScalars, AllocInScalars, AllocViewWire,
    AttachInScalars, AttachOutScalars, AttnRequest, AttnResponse, AttnWire, DetachInScalars,
    DetachOutScalars, ExecRequest, ExecResponse, ExecWire, FfnRequest, FfnResidentWire,
    FfnResponse, FfnWire, GemvRequest, GemvResidentWire, GemvResponse, GemvWire, KvWriteWire,
    LoadBufWire, MapArenaWire, ReleaseBufScalars, ReserveScratchScalars, ResetStateScalars,
    SetScaleWire, SetShapeScalars, TeamInfoScalars, UnmapArenaScalars, TIMES_VALUES,
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
        // `load_buf` packs the sequence length as the third scalar
        // (`_primIn[3]` in the stub), not only `{h, off}`.
        let prim = LoadBufWire {
            h,
            off,
            data_len: octets(data.len(), "load_buf data")?,
        }
        .to_bytes();
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
        let prim = SetScaleWire {
            h,
            data_len: octets(scales.len(), "set_scale data")?,
        }
        .to_bytes();
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
        let mut dst = vec![0u8; request.dst_len];
        let mut times = [0i32; 8];
        let prim = GemvWire {
            h: request.scalars.h,
            m: request.scalars.m,
            act_len: octets(request.act.len(), "gemv act")?,
            reps: request.scalars.reps,
            dst_len: octets(dst.len(), "gemv dst")?,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
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
        let mut dst = vec![0u8; request.dst_len];
        let mut mid = vec![0u8; request.mid_len];
        let mut times = [0i32; 8];
        let prim = FfnWire {
            gate: request.scalars.gate,
            up: request.scalars.up,
            down: request.scalars.down,
            act_len: octets(request.act.len(), "ffn act")?,
            reps: request.scalars.reps,
            dst_len: octets(dst.len(), "ffn dst")?,
            mid_len: octets(mid.len(), "ffn mid")?,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
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
        let table = i32_bytes(request.table);
        let mut outp = vec![0u8; request.out_len];
        let mut times = [0i32; 8];
        let prim = AttnWire {
            keys: request.scalars.keys,
            key_scales: request.scalars.key_scales,
            values: request.scalars.values,
            value_scales: request.scalars.value_scales,
            q_len: octets(request.q.len(), "attn q")?,
            tbl_len: longs(table.len(), "attn tbl")?,
            position: request.scalars.position,
            seq: request.scalars.seq,
            window: request.scalars.window,
            reps: request.scalars.reps,
            out_len: octets(outp.len(), "attn outp")?,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
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

    /// GEMV whose result goes into the buffer mapped by [`HvxClient::attach_out`].
    ///
    /// Only the DSP-side `times[]` sequence crosses FastRPC; `dst` is written
    /// in place, so no output copy is marshalled by the host.
    pub fn gemv_resident(
        &self,
        h: i32,
        m: i32,
        act: &[u8],
        reps: i32,
    ) -> Result<[i32; 8], TransportError> {
        let prim = GemvResidentWire {
            h,
            m,
            act_len: octets(act.len(), "gemv_resident act")?,
            reps,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
        let mut times = [0i32; 8];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 1] = [i32_bytes_mut(&mut times)];
        self.invoke(Call {
            method: method_id::GEMV_RESIDENT,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[act],
            out_bufs: &mut out_bufs,
        })?;
        Ok(times)
    }

    /// FFN whose final int8 result goes into the attached buffer.
    pub fn ffn_resident(
        &self,
        scalars: FfnScalars,
        act: &[u8],
        mid: &mut [u8],
    ) -> Result<[i32; 8], TransportError> {
        let prim = FfnResidentWire {
            gate: scalars.gate,
            up: scalars.up,
            down: scalars.down,
            act_len: octets(act.len(), "ffn_resident act")?,
            reps: scalars.reps,
            mid_len: octets(mid.len(), "ffn_resident mid")?,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
        let mut times = [0i32; 8];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 2] = [mid, i32_bytes_mut(&mut times)];
        self.invoke(Call {
            method: method_id::FFN_RESIDENT,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[act],
            out_bufs: &mut out_bufs,
        })?;
        Ok(times)
    }

    /// Map a host rpcmem fd once; subsequent resident calls write straight into
    /// it.
    ///
    /// The host mapping is cached on this platform: after a resident call the
    /// DSP result becomes visible to the CPU mapping when [`HvxClient::detach_out`]
    /// unmaps it. Callers that only hand the buffer to the next DSP stage need
    /// not read it at all; callers that do must detach (or re-attach) first.
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

    /// Map a host rpcmem/DMA-buf fd as the *weights* of `h` (zero-copy input).
    ///
    /// The DSP reads the buffer in place, so the handle no longer holds a heap
    /// copy. The host owns cache maintenance: write and flush the weights
    /// before this call. The mapping is refreshed only by
    /// [`HvxClient::detach_in`] + `attach_in`: rewriting the pages in place
    /// afterwards is not visible to the DSP (it keeps its own cached view).
    pub fn attach_in(&self, h: i32, fd: i32, nbytes: i32) -> Result<(), TransportError> {
        let prim = AttachInScalars { h, fd, nbytes }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::ATTACH_IN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Drop the weight mapping made by [`HvxClient::attach_in`].
    pub fn detach_in(&self, h: i32) -> Result<(), TransportError> {
        let prim = DetachInScalars { h }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::DETACH_IN,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Create a weight handle backed only by the host fd: no DSP heap copy.
    ///
    /// This is the path for models larger than the DSP heap; `set_shape` and
    /// `set_scale` then work exactly as after [`HvxClient::alloc_buf`].
    pub fn alloc_in(&self, fd: i32, nbytes: i32) -> Result<i32, TransportError> {
        let prim = AllocInScalars { fd, nbytes }.to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::ALLOC_IN,
            prim_in: &prim,
            prim_out: &mut out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })?;
        Ok(i32::from_le_bytes(out))
    }

    /// Map one host rpcmem/ION buffer once; model tensors become views of it
    /// ([`HvxClient::alloc_view`]) with no further mappings. Returns the arena
    /// index. The mapping is established here, so write the buffer first.
    pub fn map_arena(&self, fd: i32, nbytes: i32) -> Result<i32, TransportError> {
        let prim = MapArenaWire { fd, nbytes }.to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::MAP_ARENA,
            prim_in: &prim,
            prim_out: &mut out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })?;
        Ok(i32::from_le_bytes(out))
    }

    /// Drop an arena mapping. Every view of it must be released first.
    pub fn unmap_arena(&self, arena: i32) -> Result<(), TransportError> {
        let prim = UnmapArenaScalars { ah: arena }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::UNMAP_ARENA,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Create a weight handle as a `(off, nbytes)` view of `arena`. `off` and
    /// `nbytes` must be 128-byte aligned and lie inside the mapping.
    pub fn alloc_view(&self, arena: i32, off: i32, nbytes: i32) -> Result<i32, TransportError> {
        let prim = AllocViewWire {
            ah: arena,
            off,
            nbytes,
        }
        .to_bytes();
        let mut out = [0u8; 4];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::ALLOC_VIEW,
            prim_in: &prim,
            prim_out: &mut out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })?;
        Ok(i32::from_le_bytes(out))
    }

    /// Allocate (or, with `nbytes <= 0`, free) the DSP-heap scratch arena that
    /// `exec` commands address by byte offset.
    pub fn reserve_scratch(&self, nbytes: usize) -> Result<(), TransportError> {
        let nbytes = i32::try_from(nbytes).map_err(|_| TransportError::BufferTooSmall {
            buffer: "reserve_scratch",
            expected: nbytes,
            got: i32::MAX as usize,
        })?;
        let prim = ReserveScratchScalars { nbytes }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::RESERVE_SCRATCH,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Append one token to all resident K/KT/V heads in one RPC. `k` holds
    /// `heads * 64` bytes (int4), `kt` `heads * 4` (Q16 scale), `v`
    /// `heads * 128` (int8); the row index is `head * slots + slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_write(
        &self,
        hk: i32,
        hkt: i32,
        hv: i32,
        slots: i32,
        slot: i32,
        k: &[u8],
        kt: &[u8],
        v: &[u8],
    ) -> Result<(), TransportError> {
        let prim = KvWriteWire {
            hk,
            hkt,
            hv,
            slots,
            slot,
            k_len: octets(k.len(), "kv_write k")?,
            kt_len: octets(kt.len(), "kv_write kt")?,
            v_len: octets(v.len(), "kv_write v")?,
        }
        .to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::KV_WRITE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[k, kt, v],
            out_bufs: &mut no_out,
        })
    }

    /// Zero DSP session state: bit 0 = scratch (all CONV rings), bit 1 =
    /// counters; `flags = 0` means both.
    pub fn reset_state(&self, flags: i32) -> Result<(), TransportError> {
        let prim = ResetStateScalars { flags }.to_bytes();
        let mut prim_out: [u8; 0] = [];
        let mut no_out: [&mut [u8]; 0] = [];
        self.invoke(Call {
            method: method_id::RESET_STATE,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        })
    }

    /// Run a command list in one FastRPC call. The list itself is in memory
    /// (arena + offset), not in the RPC arguments.
    pub fn exec(&self, request: ExecRequest<'_>) -> Result<ExecResponse, TransportError> {
        let n_cmds =
            i32::try_from(request.commands).map_err(|_| TransportError::BufferTooSmall {
                buffer: "exec commands",
                expected: request.commands,
                got: i32::MAX as usize,
            })?;
        let prim = ExecWire {
            ah: request.arena,
            cmd_off: request.cmd_off,
            n_cmds,
            act_len: octets(request.act.len(), "exec act")?,
            rt_len: longs(request.rt.len() * 4, "exec rt")?,
            outp_len: octets(request.out_len, "exec out")?,
            times_len: TIMES_VALUES as i32,
        }
        .to_bytes();
        let mut out = vec![0u8; request.out_len];
        let mut times = vec![0u8; TIMES_VALUES * 4];
        let mut prim_out: [u8; 0] = [];
        let mut out_bufs: [&mut [u8]; 2] = [out.as_mut_slice(), times.as_mut_slice()];
        self.invoke(Call {
            method: method_id::EXEC,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[request.act, i32_bytes(request.rt)],
            out_bufs: &mut out_bufs,
        })?;
        let mut decoded = [0i32; TIMES_VALUES];
        decoded.copy_from_slice(&decode_i32s(&times));
        Ok(ExecResponse {
            out,
            times: decoded,
        })
    }

    /// Worker-team information (`8` values in the stand).
    pub fn team_info(&self, len: usize) -> Result<Vec<i32>, TransportError> {
        let prim = TeamInfoScalars {
            len: longs(len * 4, "team_info res")?,
        }
        .to_bytes();
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

fn octets(bytes: usize, what: &'static str) -> Result<i32, TransportError> {
    octet_len(bytes).ok_or(TransportError::InvalidLength { what, len: bytes })
}

fn longs(bytes: usize, what: &'static str) -> Result<i32, TransportError> {
    long_len(bytes).ok_or(TransportError::InvalidLength { what, len: bytes })
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
