//! Transport contract shared by the real FastRPC client and the mock.

use std::sync::Arc;

use crate::error::TransportError;

/// FastRPC handle (`remote_handle64`).
pub type Handle = u64;

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
