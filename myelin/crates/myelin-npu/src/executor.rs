//! Adapters from [`Transport`] to `myelin-core`'s [`DspExecutor`].
//!
//! `myelin-core` cannot depend on this crate (the dependency runs the other
//! way), so the core exposes a narrow object-safe trait and this module
//! implements it for the real and mock transports.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use myelin_core::backend::dsp::{DspCall, DspError, DspExecutor, DspHandle, ExecResult, ExecRun};

use crate::error::TransportError;
use crate::method::{
    long_len, method_id, octet_len, ExecWire, MapArenaWire, ReserveScratchScalars,
    UnmapArenaScalars, TIMES_VALUES,
};
use crate::transport::{Call, SharedBuffer, Transport};

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
    let (shared, arena) = map_list(transport, handle, request.list)?;
    let result = exec_mapped(transport, handle, arena, &request);
    drop(shared);
    unmap_arena(transport, handle, arena);
    result
}

/// One mapped command list. The list bytes are immutable while mapped (the
/// DSP caches every mapped page, so rewriting it in place is invisible to the
/// DSP), which is exactly why a cache can reuse the mapping when the bytes and
/// the handle match.
struct MappedList {
    list: Vec<u8>,
    /// Owns the rpcmem buffer; the arena is only valid while this lives.
    _shared: SharedBuffer,
    arena: i32,
    handle: DspHandle,
}

/// Mapped exec lists, keyed by exact bytes and handle.
///
/// One token touches a handful of stable lists: the conv list, the attention
/// projection list, the ATTN list, the o_proj list and the FFN list. None of
/// them changes between tokens once the CONV counter travels in `rt[]`, so
/// after the first token `map_arena`/`unmap_arena` disappear from the hot
/// path. Lists that do change are remapped and the oldest mapping is evicted;
/// the cap keeps the DSP arena table small.
#[derive(Default)]
struct ExecArenaCache {
    entries: Vec<MappedList>,
}

const EXEC_ARENA_CACHE: usize = 8;

impl ExecArenaCache {
    fn lock(cache: &Mutex<Self>) -> MutexGuard<'_, Self> {
        cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn find(&self, handle: DspHandle, list: &[u8]) -> Option<i32> {
        self.entries
            .iter()
            .find(|entry| entry.handle == handle && entry.list == list)
            .map(|entry| entry.arena)
    }

    fn arena_for<T: Transport + ?Sized>(
        &mut self,
        transport: &T,
        handle: DspHandle,
        list: &[u8],
    ) -> Result<i32, DspError> {
        if let Some(arena) = self.find(handle, list) {
            return Ok(arena);
        }
        if self.entries.len() == EXEC_ARENA_CACHE {
            let evicted = self.entries.remove(0);
            unmap_arena(transport, evicted.handle, evicted.arena);
        }
        let (shared, arena) = map_list(transport, handle, list)?;
        self.entries.push(MappedList {
            list: list.to_vec(),
            _shared: shared,
            arena,
            handle,
        });
        Ok(arena)
    }
}

/// Round the list to the 128-byte alignment `map_arena` requires, copy it into
/// rpcmem and map it once.
fn map_list<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    list: &[u8],
) -> Result<(SharedBuffer, i32), DspError> {
    let nbytes = list.len().max(1).div_ceil(128) * 128;
    let nbytes = i32::try_from(nbytes).map_err(|_| {
        DspError::Other(format!(
            "exec list of {} bytes does not fit a 32-bit IDL field",
            list.len()
        ))
    })?;
    let mut shared = transport
        .alloc_shared(nbytes as usize)
        .map_err(DspError::from)?;
    shared.as_mut_slice()[..list.len()].copy_from_slice(list);

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
    Ok((shared, i32::from_le_bytes(arena_bytes)))
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

/// `exec` against an already mapped arena.
fn exec_mapped<T: Transport + ?Sized>(
    transport: &T,
    handle: DspHandle,
    arena: i32,
    request: &ExecRun<'_>,
) -> Result<ExecResult, DspError> {
    let mut out = vec![0u8; request.out_len];
    let mut times = [0i32; TIMES_VALUES];
    let prim = ExecWire {
        ah: arena,
        cmd_off: 0,
        n_cmds: i32::try_from(request.commands)
            .map_err(|_| DspError::Other("exec command count does not fit i32".to_owned()))?,
        act_len: octet_len(request.act.len())
            .ok_or_else(|| DspError::Other("exec act length does not fit i32".to_owned()))?,
        rt_len: long_len(request.rt.len() * 4)
            .ok_or_else(|| DspError::Other("exec rt length does not fit i32".to_owned()))?,
        outp_len: octet_len(request.out_len)
            .ok_or_else(|| DspError::Other("exec out length does not fit i32".to_owned()))?,
        times_len: TIMES_VALUES as i32,
    }
    .to_bytes();
    let rt_bytes = rt_to_bytes(request.rt);
    let mut prim_out: [u8; 0] = [];
    let mut out_bufs: [&mut [u8]; 2] = [out.as_mut_slice(), i32_bytes_mut(&mut times)];
    let exec = dispatch(
        transport,
        handle,
        DspCall {
            method: method_id::EXEC,
            prim_in: &prim,
            prim_out: &mut prim_out,
            in_bufs: &[request.act, &rt_bytes],
            out_bufs: &mut out_bufs,
        },
    );
    if let Err(error) = &exec {
        eprintln!("exec failed: {error:?} times={times:?}");
    }
    exec?;
    Ok(ExecResult { out, times })
}

/// Cached path: the arena stays mapped for the whole generation.
fn run_exec_cached<T: Transport + ?Sized>(
    transport: &T,
    cache: &Mutex<ExecArenaCache>,
    handle: DspHandle,
    request: ExecRun<'_>,
) -> Result<ExecResult, DspError> {
    // Held across `exec` on purpose: the mapping must outlive the call, and
    // the DSP runs one command list at a time anyway.
    let mut cache = ExecArenaCache::lock(cache);
    let arena = cache.arena_for(transport, handle, request.list)?;
    exec_mapped(transport, handle, arena, &request)
}

fn rt_to_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
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
    exec_arenas: Mutex<ExecArenaCache>,
}

impl Executor {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            exec_arenas: Mutex::new(ExecArenaCache::default()),
        }
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
        run_exec_cached(self.transport.as_ref(), &self.exec_arenas, handle, request)
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
            assert_eq!(call.in_bufs.len(), 2);
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
