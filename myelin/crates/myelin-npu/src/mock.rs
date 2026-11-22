//! Deterministic in-process transport for host tests.
//!
//! The mock records every call, can inject failures, and owns its shared
//! buffers so tests can assert that RAII really frees them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::TransportError;
use crate::transport::{Call, Handle, SharedBuffer, SharedOwner, Transport};

/// One recorded invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedCall {
    pub handle: Handle,
    pub method: u32,
    pub scalars: u32,
    pub prim_in: Vec<u8>,
    pub in_bufs: Vec<Vec<u8>>,
    pub out_lens: Vec<usize>,
    pub prim_out_len: usize,
}

/// Inputs visible to a scripted handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockCall {
    pub method: u32,
    pub prim_in: Vec<u8>,
    pub in_bufs: Vec<Vec<u8>>,
    pub out_lens: Vec<usize>,
    pub prim_out_len: usize,
}

/// Bytes a scripted handler wants written back.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockReply {
    pub prim_out: Vec<u8>,
    pub out_bufs: Vec<Vec<u8>>,
}

type Handler = Box<dyn Fn(&MockCall) -> Result<MockReply, TransportError> + Send + Sync>;

struct MockState {
    next_handle: Handle,
    next_fd: i32,
    opened: Vec<String>,
    live: Vec<Handle>,
    closed: Vec<Handle>,
    calls: Vec<RecordedCall>,
    handlers: HashMap<u32, Handler>,
    open_error: Option<TransportError>,
    invoke_error: Option<TransportError>,
    allocated: usize,
    freed: usize,
    live_bytes: usize,
}

struct MockInner {
    state: Mutex<MockState>,
}

impl MockInner {
    fn lock(&self) -> MutexGuard<'_, MockState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl SharedOwner for MockInner {
    fn release(&self, ptr: *mut u8, len: usize, _fd: i32) {
        if len > 0 && !ptr.is_null() {
            // SAFETY: `alloc_shared` created the Vec with capacity == len and
            // forgot it; this is the only reconstruction.
            unsafe {
                drop(Vec::from_raw_parts(ptr, len, len));
            }
        }
        let mut state = self.lock();
        state.freed += 1;
        state.live_bytes = state.live_bytes.saturating_sub(len);
    }
}

/// Mock FastRPC transport.
#[derive(Clone)]
pub struct MockTransport {
    inner: Arc<MockInner>,
}

impl Default for MockTransport {
    fn default() -> Self {
        Self {
            inner: Arc::new(MockInner {
                state: Mutex::new(MockState {
                    next_handle: 1,
                    next_fd: 1000,
                    opened: Vec::new(),
                    live: Vec::new(),
                    closed: Vec::new(),
                    calls: Vec::new(),
                    handlers: HashMap::new(),
                    open_error: None,
                    invoke_error: None,
                    allocated: 0,
                    freed: 0,
                    live_bytes: 0,
                }),
            }),
        }
    }
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a handler for one method.
    pub fn set_handler(
        &self,
        method: u32,
        handler: impl Fn(&MockCall) -> Result<MockReply, TransportError> + Send + Sync + 'static,
    ) {
        self.inner
            .lock()
            .handlers
            .insert(method, Box::new(handler) as Handler);
    }

    /// Make the next (and every) open fail.
    pub fn fail_open(&self, error: TransportError) {
        self.inner.lock().open_error = Some(error);
    }

    /// Make every invoke fail with `error`.
    pub fn fail_invoke(&self, error: TransportError) {
        self.inner.lock().invoke_error = Some(error);
    }

    /// Modules passed to `open`, in order.
    pub fn opened(&self) -> Vec<String> {
        self.inner.lock().opened.clone()
    }

    /// Handles passed to `close`, in order.
    pub fn closed(&self) -> Vec<Handle> {
        self.inner.lock().closed.clone()
    }

    /// Handles that are still open.
    pub fn live_handles(&self) -> Vec<Handle> {
        self.inner.lock().live.clone()
    }

    /// Every invocation, in order.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.inner.lock().calls.clone()
    }

    /// Last invocation, if any.
    pub fn last_call(&self) -> Option<RecordedCall> {
        self.inner.lock().calls.last().cloned()
    }

    /// Number of `alloc_shared` calls.
    pub fn allocations(&self) -> usize {
        self.inner.lock().allocated
    }

    /// Number of `SharedBuffer` drops.
    pub fn frees(&self) -> usize {
        self.inner.lock().freed
    }

    /// Bytes currently owned by live shared buffers.
    pub fn live_bytes(&self) -> usize {
        self.inner.lock().live_bytes
    }
}

impl Transport for MockTransport {
    fn open(&self, module: &str) -> Result<Handle, TransportError> {
        let mut state = self.inner.lock();
        if let Some(error) = &state.open_error {
            return Err(error.clone());
        }
        let handle = state.next_handle;
        state.next_handle += 1;
        state.opened.push(module.to_owned());
        state.live.push(handle);
        Ok(handle)
    }

    fn close(&self, handle: Handle) -> Result<(), TransportError> {
        let mut state = self.inner.lock();
        state.closed.push(handle);
        if let Some(index) = state.live.iter().position(|live| *live == handle) {
            state.live.swap_remove(index);
            Ok(())
        } else {
            Err(TransportError::Closed {
                module: "mock".to_owned(),
            })
        }
    }

    fn invoke(&self, handle: Handle, call: Call<'_, '_>) -> Result<(), TransportError> {
        let recorded = RecordedCall {
            handle,
            method: call.method,
            scalars: call.scalars(),
            prim_in: call.prim_in.to_vec(),
            in_bufs: call.in_bufs.iter().map(|buf| buf.to_vec()).collect(),
            out_lens: call.out_bufs.iter().map(|buf| buf.len()).collect(),
            prim_out_len: call.prim_out.len(),
        };
        let mock_call = MockCall {
            method: call.method,
            prim_in: recorded.prim_in.clone(),
            in_bufs: recorded.in_bufs.clone(),
            out_lens: recorded.out_lens.clone(),
            prim_out_len: recorded.prim_out_len,
        };
        let mut state = self.inner.lock();
        state.calls.push(recorded);
        if let Some(error) = &state.invoke_error {
            return Err(error.clone());
        }
        let reply = match state.handlers.get(&call.method) {
            Some(handler) => handler(&mock_call)?,
            None => MockReply::default(),
        };
        drop(state);

        if reply.prim_out.len() > call.prim_out.len() {
            return Err(TransportError::BufferTooSmall {
                buffer: "prim_out",
                expected: reply.prim_out.len(),
                got: call.prim_out.len(),
            });
        }
        call.prim_out[..reply.prim_out.len()].copy_from_slice(&reply.prim_out);

        for (index, output) in call.out_bufs.iter_mut().enumerate() {
            let Some(bytes) = reply.out_bufs.get(index) else {
                continue;
            };
            if bytes.len() > output.len() {
                return Err(TransportError::BufferTooSmall {
                    buffer: "out_buf",
                    expected: bytes.len(),
                    got: output.len(),
                });
            }
            output[..bytes.len()].copy_from_slice(bytes);
        }
        Ok(())
    }

    fn alloc_shared(&self, len: usize) -> Result<SharedBuffer, TransportError> {
        let fd = {
            let mut state = self.inner.lock();
            state.allocated += 1;
            state.live_bytes += len;
            let fd = state.next_fd;
            state.next_fd += 1;
            fd
        };
        let mut bytes = vec![0u8; len];
        let raw = bytes.as_mut_ptr();
        std::mem::forget(bytes);
        Ok(SharedBuffer::new(
            raw,
            len,
            fd,
            Some(raw as u64),
            Arc::clone(&self.inner) as Arc<dyn SharedOwner>,
        ))
    }
}
