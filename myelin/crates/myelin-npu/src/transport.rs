//! Transport contract shared by the real FastRPC client and the mock.

use std::sync::Arc;

use crate::error::TransportError;

/// FastRPC handle (`remote_handle`).
///
/// The `qaic`-generated stub and `libcdsprpc.so` use the 32-bit
/// `remote_handle` ABI on the device; `remote_handle64_*` rejects every module
/// with `AEE_EBADPARM`. Keep this signed so the invalid value `-1` is visible.
pub type Handle = i32;

/// One FastRPC invocation.
///
/// Wire order is `prim_in`, input sequences, `prim_out`, output sequences,
/// matching `qaic`'s generated stubs. The transport turns the counts into
/// `REMOTE_SCALARS_MAKEX(0, method, in, out, 0, 0)`.
pub struct Call<'a, 'b> {
    pub method: u32,
    pub prim_in: &'a [u8],
    pub prim_out: &'b mut [u8],
    pub in_bufs: &'a [&'a [u8]],
    pub out_bufs: &'b mut [&'b mut [u8]],
}

impl Call<'_, '_> {
    /// Number of input buffers, including the scalar pack.
    pub fn in_count(&self) -> usize {
        1 + self.in_bufs.len()
    }

    /// Number of output buffers, including the scalar pack when present.
    pub fn out_count(&self) -> usize {
        usize::from(!self.prim_out.is_empty()) + self.out_bufs.len()
    }

    /// `REMOTE_SCALARS_MAKEX(0, method, in, out, 0, 0)`.
    pub fn scalars(&self) -> u32 {
        let method = self.method & 0x1f;
        let inputs = (self.in_count() as u32) & 0xff;
        let outputs = (self.out_count() as u32) & 0xff;
        (method << 24) | (inputs << 16) | (outputs << 8)
    }
}

/// Owner of a shared-memory allocation; runs when [`SharedBuffer`] drops.
pub trait SharedOwner: Send + Sync {
    fn release(&self, ptr: *mut u8, len: usize, fd: i32);
}

/// RAII shared buffer: rpcmem/ION on Android, heap memory in the mock.
pub struct SharedBuffer {
    ptr: *mut u8,
    len: usize,
    fd: i32,
    phys: Option<u64>,
    owner: Arc<dyn SharedOwner>,
}

impl SharedBuffer {
    pub(crate) fn new(
        ptr: *mut u8,
        len: usize,
        fd: i32,
        phys: Option<u64>,
        owner: Arc<dyn SharedOwner>,
    ) -> Self {
        Self {
            ptr,
            len,
            fd,
            phys,
            owner,
        }
    }

    /// Mapped host pointer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Bytes allocated.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the allocation is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// DMA-buf/ION file descriptor used by `attach_out`.
    pub const fn fd(&self) -> i32 {
        self.fd
    }

    /// Physical address, when the allocator can provide one.
    ///
    /// rpcmem on Android exposes only an fd, not a physical address, so this
    /// is `None` there; the DSP maps the DMA-buf by fd.
    pub const fn physical_address(&self) -> Option<u64> {
        self.phys
    }

    /// View the buffer as bytes.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` points to `len` initialised bytes owned by this value.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// View the buffer as mutable bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `ptr` points to `len` initialised bytes owned by this value,
        // and `&mut self` guarantees exclusive access to the mapping.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        self.owner.release(self.ptr, self.len, self.fd);
    }
}

/// `DMA_BUF_IOCTL_SYNC` flags (`linux/dma-buf.h`).
#[cfg(target_os = "android")]
mod dma_buf {
    pub const SYNC_READ: u64 = 1;
    pub const SYNC_WRITE: u64 = 2;
    pub const SYNC_START: u64 = 0;
    pub const SYNC_END: u64 = 4;
    /// `_IOW('b', 2, struct dma_buf_sync)` with an 8-byte payload.
    pub const IOCTL_SYNC: libc::c_ulong = 0x4008_6202;
}

/// Begin or end a host mapping of a shared buffer around CPU access.
///
/// rpcmem allocations are cached (`ION_FLAG_CACHED`), and the DSP maps the
/// same dma-buf. Weights written by the CPU are not visible to the device
/// until the cache is flushed, exactly like the C stands do it
/// (`attach_main.c:dma_sync`, `exec_main.c:dma_sync`). On the host this is a
/// no-op: the mock transport uses plain memory.
pub fn sync_shared(buffer: &SharedBuffer, start: bool, write: bool) -> std::io::Result<()> {
    #[cfg(target_os = "android")]
    {
        #[repr(C)]
        struct DmaBufSync {
            flags: u64,
        }
        let mut flags = if start {
            dma_buf::SYNC_START
        } else {
            dma_buf::SYNC_END
        };
        if write {
            flags |= dma_buf::SYNC_WRITE;
        } else {
            flags |= dma_buf::SYNC_READ;
        }
        let arg = DmaBufSync { flags };
        // SAFETY: the fd is a live dma-buf and `arg` matches the ioctl layout.
        // Android's bionic declares `ioctl` with an `int` request; the constant
        // is a `c_ulong` on the host toolchains, so let inference pick the
        // target's request type.
        let code = unsafe { libc::ioctl(buffer.fd(), dma_buf::IOCTL_SYNC as _, &arg) };
        if code != 0 {
            let error = std::io::Error::last_os_error();
            // `ENOTTY` means the fd is not a dma-buf, which is what we get from
            // `rpcmem_to_fd` on this kernel. Our buffers are allocated uncached
            // (see `CdsprpcConfig::rpcmem_flags`), so there is nothing to flush.
            if error.raw_os_error() == Some(libc::ENOTTY) {
                return Ok(());
            }
            return Err(error);
        }
        Ok(())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (buffer, start, write);
        Ok(())
    }
}

// SAFETY: the allocation is plain memory; the owner's release path is
// thread-safe (`SharedOwner: Send + Sync`).
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

/// FastRPC transport.
pub trait Transport: Send + Sync {
    /// Open one module (for example `"lfmgemv"`).
    fn open(&self, module: &str) -> Result<Handle, TransportError>;
    /// Close a handle.
    fn close(&self, handle: Handle) -> Result<(), TransportError>;
    /// Invoke one method.
    fn invoke(&self, handle: Handle, call: Call<'_, '_>) -> Result<(), TransportError>;
    /// Allocate memory shared with the DSP.
    fn alloc_shared(&self, len: usize) -> Result<SharedBuffer, TransportError>;

    /// Pin a long-lived shared buffer so the driver keeps its mapping.
    ///
    /// FastRPC maps every argument sequence for the duration of a call unless
    /// the buffer is registered; `remote_register_buf_attr` with
    /// `FASTRPC_ATTR_KEEP_MAP` turns that into a one-time cost. Transports
    /// without registration keep the default no-op, so a `Vec` argument still
    /// works.
    fn register_shared(&self, buffer: &SharedBuffer) {
        let _ = buffer;
    }

    /// Drop a mapping created by [`Transport::register_shared`].
    ///
    /// Best effort: the FastRPC build on the device has no
    /// `remote_deregister_buf`, so the C ABI is `remote_register_buf(ptr, len,
    /// -1)`.
    fn unregister_shared(&self, buffer: &SharedBuffer) {
        let _ = buffer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_counts_match_qaic() {
        let prim = [0u8; 16];
        let act = [1u8; 8];
        let mut dst = [0u8; 8];
        let mut mid = [0u8; 8];
        let mut times = [0u8; 32];
        let out: [&mut [u8]; 3] = [&mut dst, &mut mid, &mut times];
        let mut prim_out: [u8; 0] = [];
        let call = Call {
            method: 6,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[&act],
            out_bufs: &mut { out },
        };
        assert_eq!(call.in_count(), 2);
        assert_eq!(call.out_count(), 3);
        // method 6, 2 in, 3 out
        assert_eq!(call.scalars(), 0x0602_0300);
    }

    #[test]
    fn scalar_output_counts_as_one_buffer() {
        let prim = [0u8; 4];
        let mut out = [0u8; 4];
        let mut prim_out = [0u8; 4];
        let call = Call {
            method: 0,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut [out.as_mut_slice()][..0],
        };
        assert_eq!(call.in_count(), 1);
        assert_eq!(call.out_count(), 1);
        assert_eq!(call.scalars(), 0x0001_0100);
    }
}
