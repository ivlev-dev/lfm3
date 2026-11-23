//! Adapters from [`Transport`] to `myelin-core`'s [`DspExecutor`].
//!
//! `myelin-core` cannot depend on this crate (the dependency runs the other
//! way), so the core exposes a narrow object-safe trait and this module
//! implements it for the real and mock transports.

use std::sync::{Arc, Mutex, PoisonError};

use myelin_core::backend::dsp::{DspCall, DspError, DspExecutor, DspHandle, ExecResult, ExecRun};

use crate::error::TransportError;
use crate::method::{
    long_len, method_id, octet_len, ExecWire, MapArenaWire, ReserveScratchScalars,
    UnmapArenaScalars, TIMES_VALUES,
};
use crate::transport::{sync_shared, Call, SharedBuffer, Transport};

impl From<TransportError> for DspError {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::LibraryNotFound { library, detail } => {
                Self::Unavailable(format!("{library}: {detail}"))
            }
            TransportError::SymbolMissing { library, symbol } => {
                Self::Unavailable(format!("{library} does not export {symbol}"))
            }
            TransportError::SessionControl { code, name } => {
                Self::Other(format!("unsigned PD request failed: 0x{code:08x} ({name})"))
            }
            TransportError::HandleOpen { code, .. } => Self::HandleOpen { code },
            TransportError::Rpc { method, code, name } => Self::Rpc { method, code, name },
            TransportError::Timeout { method, millis } => Self::Timeout { method, millis },
            TransportError::BufferTooSmall { expected, got, .. } => {
                Self::BufferTooSmall { expected, got }
            }
            TransportError::InvalidLength { what, len } => Self::Other(format!(
                "{what} length {len} does not fit a 32-bit IDL field"
            )),
            TransportError::SharedMemory { bytes, code } => Self::Other(format!(
                "shared memory allocation of {bytes} bytes failed: 0x{code:08x}"
            )),
            TransportError::Closed { .. } => Self::Closed,
        }
    }
}

fn dispatch<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    call: DspCall<'_, '_>,
) -> Result<(), DspError> {
    let DspCall {
        method,
        prim_in,
        prim_out,
        in_bufs,
        out_bufs,
    } = call;
    let mut outputs: Vec<&mut [u8]> = out_bufs.iter_mut().map(|buffer| &mut **buffer).collect();
    let transport_call = Call {
        method,
        prim_in,
        prim_out,
        in_bufs,
        out_bufs: &mut outputs,
    };
    transport
        .invoke(handle, transport_call)
        .map_err(DspError::from)
}

fn reserve_scratch_on<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    nbytes: i64,
) -> Result<(), DspError> {
    let nbytes = i32::try_from(nbytes).map_err(|_| {
        DspError::Other(format!(
            "reserve_scratch size {nbytes} does not fit a 32-bit IDL field"
        ))
    })?;
    let prim = ReserveScratchScalars { nbytes }.to_bytes();
    let mut prim_out: [u8; 0] = [];
    let mut no_out: [&mut [u8]; 0] = [];
    dispatch(
        transport,
        handle,
        DspCall {
            method: method_id::RESERVE_SCRATCH,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        },
    )
}

fn run_exec_on<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    request: ExecRun<'_>,
) -> Result<ExecResult, DspError> {
    if std::env::var_os("MYELIN_DEBUG").is_some() {
        let head = &request.list[..request.list.len().min(32)];
        eprintln!(
            "exec: list={} n_cmds={} scratch={} head={:02x?}",
            request.list.len(),
            request.commands,
            request.scratch_bytes,
            head
        );
    }
    let (shared, arena) = map_list(transport, handle, request.list, None)?;
    if std::env::var_os("MYELIN_DEBUG").is_some() {
        eprintln!("exec: arena={arena} cmd_off=0");
    }
    let result = exec_mapped(transport, handle, arena, 0, &request, true);
    drop(shared);
    unmap_arena(transport, handle, arena);
    result
}

/// Command-list mappings, keyed by exact bytes.
///
/// The skeleton holds `MAXARENA` mappings per PD (raised to 160 for the full
/// model), and a token touches ~70 distinct lists. Mapping costs ~2.2 ms on
/// this device, so each list is mapped once and reused for the whole
/// generation; the list bytes are immutable while mapped, which is exactly
/// what makes the reuse safe (the DSP caches mapped pages).
struct MappedList {
    /// FNV-1a of `list`: a cheap pre-filter for the byte comparison.
    hash: u64,
    list: Vec<u8>,
    /// Owns the rpcmem buffer: the arena is only valid while this lives.
    _shared: SharedBuffer,
    arena: i32,
    handle: DspHandle,
}

#[derive(Default)]
struct ListCache {
    entries: Vec<MappedList>,
}

/// One entry per `MAXARENA` slot of the DSP skeleton (`lfmgemv_impl.c:43`).
///
/// With the identity block table (O2) the attention lists no longer change
/// with the context, so the steady-state graph is ~71 lists and fits with
/// room for a second sequence; the old 128 started evicting exactly the
/// stable FFN/conv lists once `ATTN` grew one variant per 32-token page.
const LIST_CACHE: usize = 160;

fn list_hash(bytes: &[u8]) -> u64 {
    // FNV-1a: allocation-free, and the result is only a pre-filter.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl ListCache {
    fn lock(cache: &Mutex<Self>) -> std::sync::MutexGuard<'_, Self> {
        cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn find(&self, handle: DspHandle, hash: u64, list: &[u8]) -> Option<i32> {
        self.entries
            .iter()
            .find(|entry| {
                entry.handle == handle
                    && entry.hash == hash
                    && entry.list.len() == list.len()
                    && entry.list == list
            })
            .map(|entry| entry.arena)
    }

    fn arena_for<T: Transport + ?Sized>(
        &mut self,
        transport: &T,
        handle: DspHandle,
        list: &[u8],
    ) -> Result<i32, DspError> {
        let hash = list_hash(list);
        if let Some(arena) = self.find(handle, hash, list) {
            return Ok(arena);
        }
        if self.entries.len() == LIST_CACHE {
            let evicted = self.entries.remove(0);
            unmap_arena(transport, evicted.handle, evicted.arena);
        }
        let (shared, arena) = map_list(transport, handle, list, None)?;
        self.entries.push(MappedList {
            hash,
            list: list.to_vec(),
            _shared: shared,
            arena,
            handle,
        });
        Ok(arena)
    }
}

/// Register `shared` with the skeleton and return the arena handle.
fn map_arena<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    shared: &SharedBuffer,
    nbytes: i32,
) -> Result<i32, DspError> {
    // `map_arena` writes the arena handle through `rout long` (prim_out).
    let mut arena_bytes = [0u8; 4];
    let prim = MapArenaWire {
        fd: shared.fd(),
        nbytes,
    }
    .to_bytes();
    let mut no_out: [&mut [u8]; 0] = [];
    dispatch(
        transport,
        handle,
        DspCall {
            method: method_id::MAP_ARENA,
            prim_in: &prim,
            prim_out: &mut arena_bytes,
            in_bufs: &[],
            out_bufs: &mut no_out,
        },
    )?;
    Ok(i32::from_le_bytes(arena_bytes))
}

/// Round the list to the 128-byte alignment `map_arena` requires, copy it into
/// rpcmem and map it once.
fn map_list<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    list: &[u8],
    reuse: Option<SharedBuffer>,
) -> Result<(SharedBuffer, i32), DspError> {
    let nbytes = list.len().max(1).div_ceil(128) * 128;
    let nbytes = i32::try_from(nbytes).map_err(|_| {
        DspError::Other(format!(
            "exec list of {} bytes does not fit a 32-bit IDL field",
            list.len()
        ))
    })?;
    let mut shared = match reuse {
        Some(buffer) if buffer.len() >= nbytes as usize => buffer,
        _ => transport
            .alloc_shared(nbytes as usize)
            .map_err(DspError::from)?,
    };
    shared.as_mut_slice()[..list.len()].copy_from_slice(list);
    // The command list is read by the DSP from the mapped dma-buf; flush it.
    sync_shared(&shared, true, true).map_err(|error| DspError::Other(error.to_string()))?;
    sync_shared(&shared, false, true).map_err(|error| DspError::Other(error.to_string()))?;
    let arena = map_arena(transport, handle, &shared, nbytes)?;
    Ok((shared, arena))
}

fn unmap_arena<T: Transport + ?Sized>(transport: &T, handle: DspHandle, arena: i32) {
    let prim = UnmapArenaScalars { ah: arena }.to_bytes();
    let mut prim_out: [u8; 0] = [];
    let mut no_out: [&mut [u8]; 0] = [];
    let _ = dispatch(
        transport,
        handle,
        DspCall {
            method: method_id::UNMAP_ARENA,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[],
            out_bufs: &mut no_out,
        },
    );
}

/// Scalar pack of one `exec` call.
fn exec_prim(
    arena: i32,
    cmd_off: i32,
    request: &ExecRun<'_>,
    times_len: i32,
) -> Result<[u8; 28], DspError> {
    Ok(ExecWire {
        ah: arena,
        cmd_off,
        n_cmds: i32::try_from(request.commands)
            .map_err(|_| DspError::Other("exec command count does not fit i32".to_owned()))?,
        act_len: octet_len(request.act.len())
            .ok_or_else(|| DspError::Other("exec act length does not fit i32".to_owned()))?,
        rt_len: long_len(request.rt.len() * 4)
            .ok_or_else(|| DspError::Other("exec rt length does not fit i32".to_owned()))?,
        outp_len: octet_len(request.out_len)
            .ok_or_else(|| DspError::Other("exec out length does not fit i32".to_owned()))?,
        times_len,
    }
    .to_bytes())
}

/// `exec` against an already mapped arena, at `cmd_off`, with heap arguments.
///
/// This is the one-shot path (and the `MYELIN_EXEC_HEAP=1` fallback): every
/// sequence is a Rust slice, so the driver copies it into the RPC buffer on
/// every call. [`Executor`] uses [`exec_shared`] instead.
fn exec_mapped<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    arena: i32,
    cmd_off: i32,
    request: &ExecRun<'_>,
    full_sequences: bool,
) -> Result<ExecResult, DspError> {
    if std::env::var_os("MYELIN_DEBUG").is_some() {
        eprintln!(
            "exec_mapped: arena={arena} cmd_off={cmd_off} list={} commands={} scratch={}",
            request.list.len(),
            request.commands,
            request.scratch_bytes
        );
    }
    let mut out = vec![0u8; request.out_len];
    let mut times = [0i32; TIMES_VALUES];
    times[1] = -1;
    let prim = exec_prim(
        arena,
        cmd_off,
        request,
        if full_sequences {
            TIMES_VALUES as i32
        } else {
            0
        },
    )?;
    let rt_bytes = rt_to_bytes(request.rt);
    let mut prim_out: [u8; 0] = [];
    let exec = {
        let mut in_bufs: [&[u8]; 2] = [request.act, &rt_bytes];
        let in_bufs: &mut [&[u8]] = &mut in_bufs[..];
        let mut out_bufs: [&mut [u8]; 2] = [out.as_mut_slice(), i32_bytes_mut(&mut times)];
        let out_bufs: &mut [&mut [u8]] = &mut out_bufs[..];
        dispatch(
            transport,
            handle,
            DspCall {
                method: method_id::EXEC,
                prim_in: &prim,
                prim_out: &mut prim_out,
                in_bufs,
                out_bufs,
            },
        )
    };
    if let Err(error) = &exec {
        eprintln!("exec failed: {error:?} times={times:?}");
    }
    exec?;
    Ok(ExecResult { out, times })
}

/// A shared buffer pinned with `remote_register_buf_attr`.
///
/// Deregisters on drop. `remote_register_buf_attr` returns `void` in the SDK,
/// so registration is best effort: if the driver rejects it, the call still
/// works through the per-call mapping path.
struct RegisteredBuffer {
    transport: Arc<dyn Transport>,
    buffer: SharedBuffer,
    /// Whether [`Transport::register_shared`] was actually called.
    registered: bool,
}

impl RegisteredBuffer {
    fn new(transport: &Arc<dyn Transport>, len: usize, register: bool) -> Result<Self, DspError> {
        let buffer = transport.alloc_shared(len.max(1)).map_err(DspError::from)?;
        if register {
            transport.register_shared(&buffer);
        }
        Ok(Self {
            transport: Arc::clone(transport),
            buffer,
            registered: register,
        })
    }

    fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    const fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Drop for RegisteredBuffer {
    fn drop(&mut self) {
        if self.registered {
            self.transport.unregister_shared(&self.buffer);
        }
    }
}

/// Long-lived `act`/`rt`/`out` sequences of one FastRPC handle (O1).
///
/// The Rust path used to hand the driver fresh `Vec`s and stack arrays on
/// every call. On this device that copy was not the dominant RPC cost (see
/// `docs/transport-results.md`), but the reusable buffers remove ~200 host
/// allocations per token and are the substrate for the O4/O5 lists, where one
/// call carries a whole layer.
struct ExecBuffers {
    handle: DspHandle,
    act: RegisteredBuffer,
    rt: RegisteredBuffer,
    out: RegisteredBuffer,
}

const ACT_BUFFER_ALIGN: usize = 4096;
const RT_BUFFER_MIN: usize = 64;
const OUT_BUFFER_ALIGN: usize = 4096;

const fn round_up(len: usize, align: usize) -> usize {
    let len = if len == 0 { 1 } else { len };
    len.div_ceil(align) * align
}

impl ExecBuffers {
    fn new(
        transport: &Arc<dyn Transport>,
        handle: DspHandle,
        act_len: usize,
        rt_bytes: usize,
        out_len: usize,
        register: bool,
    ) -> Result<Self, DspError> {
        Ok(Self {
            handle,
            // Registration is opt-in (`MYELIN_REG_ATTR`), because
            // `FASTRPC_ATTR_KEEP_MAP` makes this libcdsprpc skip the cache
            // invalidate after the call: the host reads stale bytes and the
            // first token changes. The output sequence is never registered.
            act: RegisteredBuffer::new(transport, round_up(act_len, ACT_BUFFER_ALIGN), register)?,
            rt: RegisteredBuffer::new(transport, round_up(rt_bytes, RT_BUFFER_MIN), register)?,
            out: RegisteredBuffer::new(transport, round_up(out_len, OUT_BUFFER_ALIGN), false)?,
        })
    }

    fn fits(&self, act_len: usize, rt_bytes: usize, out_len: usize) -> bool {
        self.act.len() >= act_len && self.rt.len() >= rt_bytes && self.out.len() >= out_len
    }
}

/// `exec` with reusable sequences. Only `times` stays on the stack: it is
/// 32 bytes and its value is needed for the failure index; it is sent only
/// under `MYELIN_TRACE` unless `MYELIN_FULL_SEQS` keeps the old wire shape.
fn exec_shared(
    transport: &dyn Transport,
    handle: DspHandle,
    arena: i32,
    cmd_off: i32,
    request: &ExecRun<'_>,
    buffers: &mut ExecBuffers,
    full_sequences: bool,
) -> Result<ExecResult, DspError> {
    let act_len = request.act.len();
    let rt_len = request.rt.len() * 4;
    let out_len = request.out_len;
    buffers.act.as_mut_slice()[..act_len].copy_from_slice(request.act);
    write_rt(request.rt, &mut buffers.rt.as_mut_slice()[..rt_len]);
    let prim = exec_prim(
        arena,
        cmd_off,
        request,
        if full_sequences {
            TIMES_VALUES as i32
        } else {
            0
        },
    )?;
    let mut times = [0i32; TIMES_VALUES];
    times[1] = -1;
    let mut prim_out: [u8; 0] = [];
    let exec = {
        let mut in_bufs: [&[u8]; 2] = [
            &buffers.act.as_slice()[..act_len],
            &buffers.rt.as_slice()[..rt_len],
        ];
        let in_bufs: &mut [&[u8]] = &mut in_bufs[..];
        let mut out_bufs: [&mut [u8]; 2] = [
            &mut buffers.out.as_mut_slice()[..out_len],
            i32_bytes_mut(&mut times),
        ];
        let out_bufs: &mut [&mut [u8]] = &mut out_bufs[..];
        transport.invoke(
            handle,
            Call {
                method: method_id::EXEC,
                prim_in: &prim,
                prim_out: &mut prim_out,
                in_bufs,
                out_bufs,
            },
        )
    };
    if let Err(error) = &exec {
        eprintln!("exec failed: {error:?} times={times:?}");
    }
    exec.map_err(DspError::from)?;
    let out = buffers.out.as_slice()[..out_len].to_vec();
    Ok(ExecResult { out, times })
}

fn write_rt(values: &[i32], out: &mut [u8]) {
    for (slot, value) in out.chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

fn rt_to_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    write_rt(values, &mut out);
    out
}

fn i32_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    // SAFETY: `i32` is plain old data and the slice keeps its lifetime.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

macro_rules! impl_executor {
    ($ty:ty) => {
        impl DspExecutor for $ty {
            fn open(&self, module: &str) -> Result<DspHandle, DspError> {
                Transport::open(self, module).map_err(DspError::from)
            }

            fn close(&self, handle: DspHandle) -> Result<(), DspError> {
                Transport::close(self, handle).map_err(DspError::from)
            }

            fn invoke(&self, handle: DspHandle, call: DspCall<'_, '_>) -> Result<(), DspError> {
                dispatch(self, handle, call)
            }

            fn reserve_scratch(&self, handle: DspHandle, nbytes: i64) -> Result<(), DspError> {
                reserve_scratch_on(self, handle, nbytes)
            }

            fn exec(
                &self,
                handle: DspHandle,
                request: ExecRun<'_>,
            ) -> Result<ExecResult, DspError> {
                run_exec_on(self, handle, request)
            }
        }
    };
}

impl_executor!(crate::mock::MockTransport);

#[cfg(target_os = "android")]
impl_executor!(crate::cdsprpc::Cdsprpc);

/// Object-safe adapter for any [`Transport`] implementation.
///
/// It owns the exec command-arena cache: one mapped arena per distinct command
/// list, so a stable token graph costs no `map_arena`/`unmap_arena` after the
/// first token. `DspBackend` drives this adapter on the device.
pub struct Executor {
    transport: Arc<dyn Transport>,
    exec_arenas: Mutex<ListCache>,
    /// Reusable registered `act`/`rt`/`out` sequences, one set per handle.
    exec_buffers: Mutex<Vec<ExecBuffers>>,
    /// `MYELIN_EXEC_HEAP=1` restores per-call heap sequences (O1 rollback).
    heap_exec: bool,
    /// `MYELIN_FULL_SEQS=1` keeps the empty `rt` sequence and the `times`
    /// output on every call (pre-O1 wire shape; rollback / A-B).
    full_sequences: bool,
    /// `MYELIN_REG_ATTR=N` opts into `remote_register_buf_attr`.
    register_buffers: bool,
    /// `MYELIN_TRACE` diagnostics: host microseconds in arena setup and in the
    /// `exec` round trip itself.
    trace_map_micros: std::sync::atomic::AtomicI64,
    trace_call_micros: std::sync::atomic::AtomicI64,
    trace_calls: std::sync::atomic::AtomicI64,
}

impl Drop for Executor {
    fn drop(&mut self) {
        if std::env::var_os("MYELIN_TRACE").is_none() {
            return;
        }
        use std::sync::atomic::Ordering;
        eprintln!(
            "exec transport: {} calls, arena {:.1} ms, invoke {:.1} ms ({:.2} ms/call)",
            self.trace_calls.load(Ordering::Relaxed),
            self.trace_map_micros.load(Ordering::Relaxed) as f64 / 1000.0,
            self.trace_call_micros.load(Ordering::Relaxed) as f64 / 1000.0,
            self.trace_call_micros.load(Ordering::Relaxed) as f64
                / 1000.0
                / self.trace_calls.load(Ordering::Relaxed).max(1) as f64
        );
    }
}

impl Executor {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            exec_arenas: Mutex::new(ListCache::default()),
            exec_buffers: Mutex::new(Vec::new()),
            heap_exec: std::env::var_os("MYELIN_EXEC_HEAP").is_some(),
            full_sequences: std::env::var_os("MYELIN_FULL_SEQS").is_some()
                || std::env::var_os("MYELIN_TRACE").is_some(),
            register_buffers: std::env::var_os("MYELIN_REG_ATTR").is_some()
                && std::env::var_os("MYELIN_NO_REGISTER").is_none(),
            trace_map_micros: std::sync::atomic::AtomicI64::new(0),
            trace_call_micros: std::sync::atomic::AtomicI64::new(0),
            trace_calls: std::sync::atomic::AtomicI64::new(0),
        }
    }

    fn shared_exec(
        &self,
        handle: DspHandle,
        arena: i32,
        request: &ExecRun<'_>,
    ) -> Result<ExecResult, DspError> {
        let mut cache = self
            .exec_buffers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let rt_bytes = request.rt.len() * 4;
        let index = match cache.iter().position(|buffers| {
            buffers.handle == handle && buffers.fits(request.act.len(), rt_bytes, request.out_len)
        }) {
            Some(index) => index,
            None => {
                if let Some(index) = cache.iter().position(|buffers| buffers.handle == handle) {
                    cache.swap_remove(index);
                }
                cache.push(ExecBuffers::new(
                    &self.transport,
                    handle,
                    request.act.len(),
                    rt_bytes,
                    request.out_len,
                    self.register_buffers,
                )?);
                cache.len() - 1
            }
        };
        exec_shared(
            self.transport.as_ref(),
            handle,
            arena,
            0,
            request,
            &mut cache[index],
            self.full_sequences,
        )
    }

    fn traced_exec(&self, handle: DspHandle, request: ExecRun<'_>) -> Result<ExecResult, DspError> {
        use std::sync::atomic::Ordering;
        let map_started = std::time::Instant::now();
        let mut cache = ListCache::lock(&self.exec_arenas);
        let arena = cache.arena_for(self.transport.as_ref(), handle, request.list)?;
        drop(cache);
        if std::env::var_os("MYELIN_DEBUG").is_some() {
            eprintln!(
                "traced_exec: arena={arena} list={} commands={} scratch={} heap={}",
                request.list.len(),
                request.commands,
                request.scratch_bytes,
                self.heap_exec
            );
        }
        self.trace_map_micros
            .fetch_add(map_started.elapsed().as_micros() as i64, Ordering::Relaxed);
        let call_started = std::time::Instant::now();
        let result = if self.heap_exec {
            exec_mapped(
                self.transport.as_ref(),
                handle,
                arena,
                0,
                &request,
                self.full_sequences,
            )
        } else {
            self.shared_exec(handle, arena, &request)
        };
        self.trace_call_micros
            .fetch_add(call_started.elapsed().as_micros() as i64, Ordering::Relaxed);
        self.trace_calls.fetch_add(1, Ordering::Relaxed);
        result
    }
}

impl DspExecutor for Executor {
    fn open(&self, module: &str) -> Result<DspHandle, DspError> {
        self.transport.open(module).map_err(DspError::from)
    }

    fn close(&self, handle: DspHandle) -> Result<(), DspError> {
        self.transport.close(handle).map_err(DspError::from)
    }

    fn invoke(&self, handle: DspHandle, call: DspCall<'_, '_>) -> Result<(), DspError> {
        dispatch(self.transport.as_ref(), handle, call)
    }

    fn reserve_scratch(&self, handle: DspHandle, nbytes: i64) -> Result<(), DspError> {
        reserve_scratch_on(self.transport.as_ref(), handle, nbytes)
    }

    fn exec(&self, handle: DspHandle, request: ExecRun<'_>) -> Result<ExecResult, DspError> {
        self.traced_exec(handle, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockReply, MockTransport};
    use myelin_core::backend::dsp::exec;

    #[test]
    fn exec_rounds_the_list_arena_to_128_and_returns_output() {
        let transport = Arc::new(MockTransport::new());
        transport.set_handler(method_id::MAP_ARENA, |_| {
            Ok(MockReply {
                prim_out: 7i32.to_le_bytes().to_vec(),
                ..Default::default()
            })
        });
        transport.set_handler(method_id::EXEC, |call| {
            // `rt` is empty for this list and tracing is off in the test, so
            // only `act` is sent as an input sequence.
            assert_eq!(call.in_bufs.len(), 1);
            Ok(MockReply {
                out_bufs: vec![vec![0xaa; 4], vec![0u8; 32]],
                ..Default::default()
            })
        });
        transport.set_handler(method_id::UNMAP_ARENA, |_| Ok(MockReply::default()));

        let executor = Executor::new(transport.clone());
        let mut list = exec::List::new(4096);
        list.push(exec::Command::quant(0, 0, 4, 65536).with_dst(exec::dst::OUT, 0));
        let encoded = list.to_bytes();
        assert_ne!(encoded.len() % 128, 0);
        let result = executor
            .exec(
                3,
                ExecRun {
                    list: &encoded,
                    commands: 1,
                    scratch_bytes: 4096,
                    act: &[1, 2, 3, 4],
                    rt: &[],
                    out_len: 4,
                },
            )
            .expect("exec");
        assert_eq!(result.out, vec![0xaa; 4]);

        let calls = transport.calls();
        let map = calls
            .iter()
            .find(|call| call.method == method_id::MAP_ARENA)
            .expect("map_arena");
        let nbytes = i32::from_le_bytes(map.prim_in[4..8].try_into().unwrap());
        assert_eq!(nbytes as usize, encoded.len().div_ceil(128) * 128);
        let exec_call = calls
            .iter()
            .find(|call| call.method == method_id::EXEC)
            .expect("exec");
        assert_eq!(
            i32::from_le_bytes(exec_call.prim_in[0..4].try_into().unwrap()),
            7
        );
        assert_eq!(
            i32::from_le_bytes(encoded[20..24].try_into().unwrap()),
            4096
        );

        // The command arena stays mapped for the next token: a second exec
        // with the same bytes must not map or unmap anything.
        executor
            .exec(
                3,
                ExecRun {
                    list: &encoded,
                    commands: 1,
                    scratch_bytes: 4096,
                    act: &[1, 2, 3, 4],
                    rt: &[],
                    out_len: 4,
                },
            )
            .expect("second exec");
        let calls = transport.calls();
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.method == method_id::MAP_ARENA)
                .count(),
            1,
            "the list arena is mapped once"
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.method == method_id::UNMAP_ARENA)
                .count(),
            0,
            "a live session keeps its arena mapped"
        );

        // A different list gets its own mapping; the first one is not unmapped
        // until the cache has to evict.
        let mut other = exec::List::new(4096);
        other.push(exec::Command::quant(0, 0, 4, 65536).with_dst(exec::dst::OUT, 0));
        other.push(exec::Command::quant(4, 4, 4, 65536).with_dst(exec::dst::OUT, 4));
        let other = other.to_bytes();
        executor
            .exec(
                3,
                ExecRun {
                    list: &other,
                    commands: 2,
                    scratch_bytes: 4096,
                    act: &[],
                    rt: &[],
                    out_len: 8,
                },
            )
            .expect("other exec");
        assert_eq!(
            transport
                .calls()
                .iter()
                .filter(|call| call.method == method_id::MAP_ARENA)
                .count(),
            2
        );

        drop(executor);
        assert_eq!(transport.live_bytes(), 0);
    }

    #[test]
    fn errors_keep_the_fastrpc_code() {
        let error: DspError = TransportError::Rpc {
            method: 6,
            code: 0x8000_040e_u32 as i32,
            name: "AEE_EBADPARM",
        }
        .into();
        match error {
            DspError::Rpc { method, code, name } => {
                assert_eq!(method, 6);
                assert_eq!(code, 0x8000_040e_u32 as i32);
                assert_eq!(name, "AEE_EBADPARM");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn missing_library_is_unavailable() {
        let error: DspError = TransportError::LibraryNotFound {
            library: "libcdsprpc.so".to_owned(),
            detail: "not found".to_owned(),
        }
        .into();
        assert!(matches!(error, DspError::Unavailable(_)));
    }
}
