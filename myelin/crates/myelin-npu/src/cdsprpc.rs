//! Real FastRPC transport for Android (`libcdsprpc.so`).
//!
//! The library is loaded with `dlopen`, so a host build (or a device without
//! FastRPC) does not fail to link: [`Cdsprpc::load`] returns
//! [`TransportError::LibraryNotFound`] instead.
//!
//! Unsigned PD is enabled exactly as the DSP stands do:
//! `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` with
//! `CDSP_DOMAIN_ID` and `enable = 1`, before the first `remote_handle_open`.
//! The shell name (`fastrpc_shell_unsigned_3`) is diagnostic context kept in
//! [`CdsprpcConfig`]; the driver chooses the shell itself.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::{fastrpc_error_name, rpc_error, TransportError};
use crate::transport::{Call, Handle, SharedBuffer, SharedOwner, Transport};

/// `DSPRPC_CONTROL_UNSIGNED_MODULE` from `remote.h`.
const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;
/// `CDSP_DOMAIN_ID` from `remote.h`.
const CDSP_DOMAIN_ID: i32 = 3;
/// `RPCMEM_HEAP_ID_SYSTEM`.
const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
/// `FASTRPC_ATTR_KEEP_MAP`, the attribute used by the stands for long-lived
/// output buffers.
pub const FASTRPC_ATTR_KEEP_MAP: i32 = 0x8;

type OpenFn = unsafe extern "C" fn(*const c_char, *mut Handle) -> c_int;
type InvokeFn = unsafe extern "C" fn(Handle, u32, *mut RemoteArg) -> c_int;
type CloseFn = unsafe extern "C" fn(Handle) -> c_int;
type SessionControlFn = unsafe extern "C" fn(u32, *mut c_void, u32) -> c_int;
type RpcmemAllocFn = unsafe extern "C" fn(i32, u32, i32) -> *mut c_void;
type RpcmemFreeFn = unsafe extern "C" fn(*mut c_void);
type RpcmemToFdFn = unsafe extern "C" fn(*mut c_void) -> c_int;
/// `remote_register_buf(void *buf, int size, int fd)`; returns `void` in the
/// SDK header (`__QAIC_RETURN void`), and `fd = -1` deregisters.
type RegisterBufFn = unsafe extern "C" fn(*mut c_void, c_int, c_int);
/// `remote_register_buf_attr(void *buf, int size, int fd, int attr)`.
type RegisterBufAttrFn = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int);

#[repr(C)]
#[derive(Clone, Copy)]
struct RemoteBuf {
    pv: *mut c_void,
    n_len: usize,
}

/// `remote_arg` on the 64-bit ABI. `remote_buf` is the largest member, so the
/// union has the same size and alignment as the C type.
#[repr(C)]
#[derive(Clone, Copy)]
union RemoteArg {
    buf: RemoteBuf,
}

#[repr(C)]
struct UnsignedModule {
    domain: i32,
    enable: i32,
}

struct Library {
    handle: *mut c_void,
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle came from dlopen and is closed once.
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }
}

// SAFETY: dlopen/dlclose and the FastRPC entry points are thread-safe; the
// handle itself is only used to resolve symbols.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

struct RpcmemOwner {
    /// Keeps `libcdsprpc.so` loaded until the last shared buffer is freed.
    _library: Arc<Library>,
    free: RpcmemFreeFn,
}

impl SharedOwner for RpcmemOwner {
    fn release(&self, ptr: *mut u8, _len: usize, _fd: i32) {
        // SAFETY: the pointer came from rpcmem_alloc and is freed once.
        unsafe { (self.free)(ptr.cast::<c_void>()) }
    }
}

/// FastRPC transport configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CdsprpcConfig {
    /// Shared library to load.
    pub library: String,
    /// DSP domain (`CDSP_DOMAIN_ID = 3`).
    pub domain: i32,
    /// Enable unsigned PD before opening the first handle.
    pub unsigned: bool,
    /// Unsigned shell used by the device; diagnostic only.
    pub shell: String,
    /// `remote_register_buf_attr` attribute for long-lived buffers.
    pub map_attributes: i32,
    /// rpcmem heap id.
    pub rpcmem_heap: i32,
    /// rpcmem flags.
    pub rpcmem_flags: u32,
}

impl Default for CdsprpcConfig {
    fn default() -> Self {
        Self {
            library: "libcdsprpc.so".to_owned(),
            domain: CDSP_DOMAIN_ID,
            unsigned: true,
            shell: "fastrpc_shell_unsigned_3".to_owned(),
            map_attributes: FASTRPC_ATTR_KEEP_MAP,
            rpcmem_heap: RPCMEM_HEAP_ID_SYSTEM,
            // Uncached on purpose. Cached buffers would need
            // `DMA_BUF_IOCTL_SYNC` around every CPU access, but the fds that
            // `rpcmem_to_fd` hands out on this kernel are not dma-bufs
            // (`ENOTTY`), so there is no way to flush them. Measured CPU fill
            // rate is 10-15 GB/s uncached versus 9-12 cached, i.e. the flag
            // does not cost us anything here.
            rpcmem_flags: 0,
        }
    }
}

/// `dlopen`-based FastRPC client.
pub struct Cdsprpc {
    library: Arc<Library>,
    config: CdsprpcConfig,
    open: OpenFn,
    invoke: InvokeFn,
    close: CloseFn,
    session_control: SessionControlFn,
    rpcmem_alloc: RpcmemAllocFn,
    rpcmem_free: RpcmemFreeFn,
    rpcmem_to_fd: RpcmemToFdFn,
    /// Optional: older FastRPC builds may omit the registration API.
    register_buf: Option<RegisterBufFn>,
    register_buf_attr: Option<RegisterBufAttrFn>,
    unsigned_enabled: AtomicBool,
}

impl Cdsprpc {
    /// Load `config.library` and resolve every entry point.
    pub fn load(config: CdsprpcConfig) -> Result<Self, TransportError> {
        let library_name = config.library.clone();
        let name =
            CString::new(library_name.as_bytes()).map_err(|_| TransportError::LibraryNotFound {
                library: library_name.clone(),
                detail: "library name contains NUL".to_owned(),
            })?;
        // `RTLD_NODELETE`: libcdsprpc starts worker threads and keeps internal
        // state; unloading it while those threads live segfaults the process
        // when the last client drops.
        // SAFETY: `name` is a valid C string.
        let handle = unsafe {
            libc::dlopen(
                name.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            )
        };
        if handle.is_null() {
            let detail = dlerror().unwrap_or_else(|| "dlopen failed".to_owned());
            return Err(TransportError::LibraryNotFound {
                library: library_name,
                detail,
            });
        }
        let library = Arc::new(Library { handle });

        // SAFETY: each symbol is resolved from a live library and stored in a
        // field with the exact FastRPC signature.
        // `remote_handle64_*` exists on newer SDKs but this device rejects it
        // with `AEE_EBADPARM` for every module; the qaic stub and the working
        // C stands use the 32-bit ABI.
        let open = unsafe { resolve::<OpenFn>(&library, "remote_handle_open")? };
        let invoke = unsafe { resolve::<InvokeFn>(&library, "remote_handle_invoke")? };
        let close = unsafe { resolve::<CloseFn>(&library, "remote_handle_close")? };
        let session_control =
            unsafe { resolve::<SessionControlFn>(&library, "remote_session_control")? };
        let rpcmem_alloc = unsafe { resolve::<RpcmemAllocFn>(&library, "rpcmem_alloc")? };
        let rpcmem_free = unsafe { resolve::<RpcmemFreeFn>(&library, "rpcmem_free")? };
        let rpcmem_to_fd = unsafe { resolve::<RpcmemToFdFn>(&library, "rpcmem_to_fd")? };
        let register_buf =
            unsafe { resolve_optional::<RegisterBufFn>(&library, "remote_register_buf") };
        let register_buf_attr =
            unsafe { resolve_optional::<RegisterBufAttrFn>(&library, "remote_register_buf_attr") };

        Ok(Self {
            library,
            config,
            open,
            invoke,
            close,
            session_control,
            rpcmem_alloc,
            rpcmem_free,
            rpcmem_to_fd,
            register_buf,
            register_buf_attr,
            unsigned_enabled: AtomicBool::new(false),
        })
    }

    /// Configuration used by this client.
    pub fn config(&self) -> &CdsprpcConfig {
        &self.config
    }

    /// Unsigned shell requested by the configuration.
    pub fn shell(&self) -> &str {
        &self.config.shell
    }

    fn enable_unsigned_once(&self) -> Result<(), TransportError> {
        if !self.config.unsigned || self.unsigned_enabled.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut module = UnsignedModule {
            domain: self.config.domain,
            enable: 1,
        };
        // SAFETY: the struct matches `remote_rpc_control_unsigned_module`.
        let code = unsafe {
            (self.session_control)(
                DSPRPC_CONTROL_UNSIGNED_MODULE,
                (&mut module as *mut UnsignedModule).cast::<c_void>(),
                std::mem::size_of::<UnsignedModule>() as u32,
            )
        };
        if code != 0 {
            return Err(TransportError::SessionControl {
                code,
                name: fastrpc_error_name(code),
            });
        }
        self.unsigned_enabled.store(true, Ordering::Release);
        Ok(())
    }
}

impl Transport for Cdsprpc {
    fn open(&self, module: &str) -> Result<Handle, TransportError> {
        self.enable_unsigned_once()?;
        let name = CString::new(module).map_err(|_| TransportError::HandleOpen {
            module: module.to_owned(),
            code: 0,
        })?;
        let mut handle: Handle = -1;
        // SAFETY: `name` is a valid C string and `handle` is writable.
        let code = unsafe { (self.open)(name.as_ptr(), &mut handle) };
        if code != 0 || handle < 0 {
            return Err(TransportError::HandleOpen {
                module: module.to_owned(),
                code: if code != 0 { code } else { 0x2c }, // AEE_EINVHANDLE
            });
        }
        Ok(handle)
    }

    fn close(&self, handle: Handle) -> Result<(), TransportError> {
        // SAFETY: `handle` came from `open`.
        let code = unsafe { (self.close)(handle) };
        if code != 0 {
            return Err(rpc_error(u32::MAX, code));
        }
        Ok(())
    }

    fn invoke(&self, handle: Handle, call: Call<'_, '_>) -> Result<(), TransportError> {
        // At most `prim_in + 2 in + prim_out + 3 out = 7` arguments exist in
        // `lfmgemv`; a fixed array keeps the hot path allocation-free.
        const MAX_ARGS: usize = 8;
        let mut args = [empty_arg(); MAX_ARGS];
        let mut count = 0;
        push_arg(
            &mut args,
            &mut count,
            call.prim_in.as_ptr() as *mut u8,
            call.prim_in.len(),
        )?;
        for buffer in call.in_bufs {
            push_arg(
                &mut args,
                &mut count,
                buffer.as_ptr() as *mut u8,
                buffer.len(),
            )?;
        }
        if !call.prim_out.is_empty() {
            push_arg(
                &mut args,
                &mut count,
                call.prim_out.as_mut_ptr(),
                call.prim_out.len(),
            )?;
        }
        for buffer in call.out_bufs.iter_mut() {
            push_arg(&mut args, &mut count, buffer.as_mut_ptr(), buffer.len())?;
        }
        // SAFETY: `args[..count]` is a contiguous `remote_arg[]` for the
        // duration of the call, and every pointer refers to a live Rust slice.
        let code = unsafe { (self.invoke)(handle, call.scalars(), args.as_mut_ptr()) };
        if code != 0 {
            return Err(rpc_error(call.method, code));
        }
        Ok(())
    }

    fn alloc_shared(&self, len: usize) -> Result<SharedBuffer, TransportError> {
        let Ok(size) = i32::try_from(len) else {
            return Err(TransportError::SharedMemory {
                bytes: len,
                code: 0x4d, // AEE_EBADSIZE
            });
        };
        // SAFETY: rpcmem_alloc returns an owned mapping or NULL.
        let ptr =
            unsafe { (self.rpcmem_alloc)(self.config.rpcmem_heap, self.config.rpcmem_flags, size) };
        if ptr.is_null() {
            return Err(TransportError::SharedMemory {
                bytes: len,
                code: 0x68, // AEE_ENORPCMEMORY
            });
        }
        // SAFETY: `ptr` is a live rpcmem allocation.
        let fd = unsafe { (self.rpcmem_to_fd)(ptr) };
        if fd < 0 {
            // SAFETY: `ptr` came from rpcmem_alloc and is not shared yet.
            unsafe { (self.rpcmem_free)(ptr) };
            return Err(TransportError::SharedMemory {
                bytes: len,
                code: fd,
            });
        }
        let owner = RpcmemOwner {
            _library: Arc::clone(&self.library),
            free: self.rpcmem_free,
        };
        Ok(SharedBuffer::new(
            ptr.cast::<u8>(),
            len,
            fd,
            None,
            Arc::new(owner),
        ))
    }

    fn register_shared(&self, buffer: &SharedBuffer) {
        if std::env::var_os("MYELIN_NO_REGISTER").is_some() {
            return;
        }
        let Some(len) = i32::try_from(buffer.len()).ok() else {
            return;
        };
        let ptr = buffer.as_ptr().cast::<c_void>();
        let attr = std::env::var("MYELIN_REG_ATTR")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(self.config.map_attributes);
        // SAFETY: `ptr` is a live rpcmem allocation of `len` bytes; both
        // entry points only record the mapping.
        unsafe {
            if let Some(register) = self.register_buf_attr {
                register(ptr, len, buffer.fd(), attr);
            } else if let Some(register) = self.register_buf {
                register(ptr, len, buffer.fd());
            }
        }
    }

    fn unregister_shared(&self, buffer: &SharedBuffer) {
        let (Some(len), Some(register)) = (i32::try_from(buffer.len()).ok(), self.register_buf)
        else {
            return;
        };
        // SAFETY: `fd = -1` is the documented deregistration form; the buffer
        // is still alive for the duration of the call.
        unsafe { register(buffer.as_ptr().cast::<c_void>(), len, -1) };
    }
}

fn remote_buf(ptr: *mut u8, len: usize) -> RemoteArg {
    RemoteArg {
        buf: RemoteBuf {
            pv: ptr.cast::<c_void>(),
            n_len: len,
        },
    }
}

fn empty_arg() -> RemoteArg {
    remote_buf(std::ptr::null_mut(), 0)
}

fn push_arg(
    args: &mut [RemoteArg; 8],
    count: &mut usize,
    ptr: *mut u8,
    len: usize,
) -> Result<(), TransportError> {
    let slot = args.get_mut(*count).ok_or(TransportError::InvalidLength {
        what: "remote_arg",
        len: *count + 1,
    })?;
    *slot = remote_buf(ptr, len);
    *count += 1;
    Ok(())
}

fn dlerror() -> Option<String> {
    // SAFETY: dlerror returns a NUL-terminated string or NULL.
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        None
    } else {
        // SAFETY: non-NULL dlerror result is a valid C string.
        Some(
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

unsafe fn resolve<T: Copy>(library: &Library, symbol: &'static str) -> Result<T, TransportError> {
    unsafe { resolve_optional(library, symbol) }.ok_or(TransportError::SymbolMissing {
        library: "libcdsprpc.so".to_owned(),
        symbol,
    })
}

/// Resolve a symbol, returning `None` when the loaded library omits it.
unsafe fn resolve_optional<T: Copy>(library: &Library, symbol: &'static str) -> Option<T> {
    let name = CString::new(symbol).ok()?;
    // SAFETY: `library.handle` is a live dlopen handle.
    let pointer = unsafe { libc::dlsym(library.handle, name.as_ptr()) };
    if pointer.is_null() {
        return None;
    }
    // SAFETY: the caller names the exact FastRPC signature for `symbol`.
    Some(unsafe { std::mem::transmute_copy(&pointer) })
}
