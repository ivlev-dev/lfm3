//! Stable C ABI.
//!
//! Rules:
//! * handles are opaque pointers owned by the caller;
//! * every input struct starts with `size` so fields can be appended later;
//! * functions return an error code and optionally fill `myelin_error`;
//! * no Rust panic may cross the boundary.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Arc;

use crate::runtime::{
    BackendKind, ChatMessage, EngineHandle, RuntimeError, Sampling, SessionHandle, StreamEvent,
};

/// Version of the C ABI, not of the crate.
pub const MYELIN_ABI_VERSION: u32 = 1;

pub const MYELIN_OK: c_int = 0;
pub const MYELIN_ERR_INVALID_ARGUMENT: c_int = 1;
pub const MYELIN_ERR_INVALID_STATE: c_int = 2;
pub const MYELIN_ERR_IO: c_int = 3;
pub const MYELIN_ERR_BACKEND: c_int = 4;
pub const MYELIN_ERR_CANCELLED: c_int = 5;
pub const MYELIN_ERR_PANIC: c_int = 6;
pub const MYELIN_ERR_INTERNAL: c_int = 7;

pub const MYELIN_BACKEND_AUTO: u32 = 0;
pub const MYELIN_BACKEND_MOCK: u32 = 1;
pub const MYELIN_BACKEND_NPU: u32 = 2;

pub const MYELIN_EVENT_DELTA: u32 = 0;
pub const MYELIN_EVENT_TOOL_CALL: u32 = 1;
pub const MYELIN_EVENT_DONE: u32 = 2;
pub const MYELIN_EVENT_ERROR: u32 = 3;

const ENGINE_MAGIC: u64 = 0x4D59_454C_494E_454E;
const SESSION_MAGIC: u64 = 0x4D59_454C_494E_5345;
const ERROR_MESSAGE_LEN: usize = 512;

/// Out-parameter for the last failure. `message` is NUL-terminated UTF-8.
#[repr(C)]
pub struct myelin_error {
    pub size: usize,
    pub code: c_int,
    pub message: [c_char; ERROR_MESSAGE_LEN],
}

/// Engine construction options.
#[repr(C)]
pub struct myelin_engine_cfg {
    pub size: usize,
    /// One of `MYELIN_BACKEND_*`.
    pub backend: u32,
    /// Optional UTF-8 path to a GGUF file; loaded when non-null.
    pub model_path: *const c_char,
}

/// One chat message. Strings are borrowed for the duration of the call.
#[repr(C)]
pub struct myelin_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

/// Sampling options. The stub validates them but always emits five tokens.
#[repr(C)]
pub struct myelin_sampling {
    pub size: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub seed: u64,
}

/// One streaming event passed to `myelin_token_cb`.
#[repr(C)]
pub struct myelin_token_event {
    pub size: usize,
    /// One of `MYELIN_EVENT_*`.
    pub kind: u32,
    /// Token index for deltas, total for `Done`.
    pub index: u32,
    /// Token id for deltas, zero otherwise.
    pub token: u32,
    /// NUL-terminated UTF-8 payload, may be null.
    pub text: *const c_char,
    /// Set for error events, null otherwise.
    pub error: *const myelin_error,
}

/// Callback. Return `0` to continue or non-zero to cancel the stream.
pub type myelin_token_cb =
    unsafe extern "C" fn(event: *const myelin_token_event, user: *mut c_void) -> c_int;

/// Opaque engine handle.
#[repr(C)]
pub struct myelin_engine {
    _private: [u8; 0],
}

/// Opaque session handle.
#[repr(C)]
pub struct myelin_session {
    _private: [u8; 0],
}

struct EngineBox {
    magic: u64,
    handle: Arc<EngineHandle>,
}

struct SessionBox {
    magic: u64,
    handle: Arc<SessionHandle>,
}

static VERSION_STRING: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// C ABI version.
#[no_mangle]
pub extern "C" fn myelin_version() -> u32 {
    MYELIN_ABI_VERSION
}

/// NUL-terminated crate version string with static lifetime.
#[no_mangle]
pub extern "C" fn myelin_version_string() -> *const c_char {
    VERSION_STRING.as_ptr().cast()
}

/// Create an engine. Returns null on failure.
///
/// # Safety
/// `cfg`, when non-null, must point to a readable `myelin_engine_cfg` whose
/// `model_path` is a valid NUL-terminated string. `err`, when non-null, must
/// point to writable `myelin_error` storage.
#[no_mangle]
pub unsafe extern "C" fn myelin_engine_new(
    cfg: *const myelin_engine_cfg,
    err: *mut myelin_error,
) -> *mut myelin_engine {
    clear_error(err);
    guarded(err, ptr::null_mut(), || {
        let (kind, model_path) = match read_engine_cfg(cfg) {
            Ok(value) => value,
            Err(error) => return report_null(err, &error),
        };
        let handle = match EngineHandle::new(kind) {
            Ok(handle) => handle,
            Err(error) => return report_null(err, &error),
        };
        if let Some(path) = model_path {
            if let Err(error) = handle.load(&path) {
                return report_null(err, &error);
            }
        }
        let boxed = Box::new(EngineBox {
            magic: ENGINE_MAGIC,
            handle,
        });
        Box::into_raw(boxed).cast()
    })
}

/// Validate and load a model path. The GGUF parser is not wired yet.
///
/// # Safety
/// `engine` must be a live handle from `myelin_engine_new`; `path` must be a
/// valid NUL-terminated UTF-8 string; `err` must be writable when non-null.
#[no_mangle]
pub unsafe extern "C" fn myelin_engine_load(
    engine: *mut myelin_engine,
    path: *const c_char,
    err: *mut myelin_error,
) -> c_int {
    clear_error(err);
    guarded(err, MYELIN_ERR_PANIC, || {
        let handle = match engine_handle(engine) {
            Ok(handle) => handle,
            Err(error) => return report(err, &error),
        };
        let path = match read_cstr(path, "path") {
            Ok(path) => path,
            Err(error) => return report(err, &error),
        };
        match handle.load(&path) {
            Ok(()) => MYELIN_OK,
            Err(error) => report(err, &error),
        }
    })
}

/// Create a session for an engine. Returns null on failure.
///
/// # Safety
/// `engine` must be a live handle from `myelin_engine_new`; `err` must be
/// writable when non-null.
#[no_mangle]
pub unsafe extern "C" fn myelin_session_new(
    engine: *mut myelin_engine,
    err: *mut myelin_error,
) -> *mut myelin_session {
    clear_error(err);
    guarded(err, ptr::null_mut(), || {
        let handle = match engine_handle(engine) {
            Ok(handle) => handle,
            Err(error) => return report_null(err, &error),
        };
        let session = handle.session();
        let boxed = Box::new(SessionBox {
            magic: SESSION_MAGIC,
            handle: session,
        });
        Box::into_raw(boxed).cast()
    })
}

/// Stream five fake tokens and a final `Done` event.
///
/// # Safety
/// `session` must be live. `messages` must point to `message_count` valid
/// entries with NUL-terminated role/content strings. `sampling` and `err` must
/// be readable/writable respectively when non-null. `callback`, when set, must
/// be safe to call synchronously with `user`.
#[no_mangle]
pub unsafe extern "C" fn myelin_session_generate(
    session: *mut myelin_session,
    messages: *const myelin_message,
    message_count: usize,
    sampling: *const myelin_sampling,
    callback: Option<myelin_token_cb>,
    user: *mut c_void,
    err: *mut myelin_error,
) -> c_int {
    clear_error(err);
    guarded(err, MYELIN_ERR_PANIC, || {
        let handle = match session_handle(session) {
            Ok(handle) => handle,
            Err(error) => return report(err, &error),
        };
        let messages = match read_messages(messages, message_count) {
            Ok(messages) => messages,
            Err(error) => return report(err, &error),
        };
        let sampling = match read_sampling(sampling) {
            Ok(sampling) => sampling,
            Err(error) => return report(err, &error),
        };

        let mut emit = |event: StreamEvent| -> bool {
            let Some(callback) = callback else {
                return true;
            };
            match event {
                StreamEvent::Delta { index, token, text } => {
                    let text = CString::new(text).unwrap_or_default();
                    let event = myelin_token_event {
                        size: size_of::<myelin_token_event>(),
                        kind: MYELIN_EVENT_DELTA,
                        index,
                        token,
                        text: text.as_ptr(),
                        error: ptr::null(),
                    };
                    unsafe { callback(&event, user) == 0 }
                }
                StreamEvent::ToolCall { payload } => {
                    let payload = CString::new(payload).unwrap_or_default();
                    let event = myelin_token_event {
                        size: size_of::<myelin_token_event>(),
                        kind: MYELIN_EVENT_TOOL_CALL,
                        index: 0,
                        token: 0,
                        text: payload.as_ptr(),
                        error: ptr::null(),
                    };
                    unsafe { callback(&event, user) == 0 }
                }
                StreamEvent::Done { total } => {
                    let event = myelin_token_event {
                        size: size_of::<myelin_token_event>(),
                        kind: MYELIN_EVENT_DONE,
                        index: total,
                        token: 0,
                        text: ptr::null(),
                        error: ptr::null(),
                    };
                    unsafe { callback(&event, user) == 0 }
                }
                StreamEvent::Error { message } => {
                    let text = CString::new(message.clone()).unwrap_or_default();
                    let mut error = myelin_error {
                        size: size_of::<myelin_error>(),
                        code: MYELIN_ERR_INTERNAL,
                        message: [0; ERROR_MESSAGE_LEN],
                    };
                    write_message(&mut error, &message);
                    let event = myelin_token_event {
                        size: size_of::<myelin_token_event>(),
                        kind: MYELIN_EVENT_ERROR,
                        index: 0,
                        token: 0,
                        text: text.as_ptr(),
                        error: &error,
                    };
                    unsafe { callback(&event, user) == 0 }
                }
            }
        };

        match handle.generate(&messages, sampling, &mut emit) {
            Ok(_) => MYELIN_OK,
            Err(error) => report(err, &error),
        }
    })
}

/// Request cancellation. Safe to call while `generate` is running.
///
/// # Safety
/// `session` must be a live handle from `myelin_session_new`.
#[no_mangle]
pub unsafe extern "C" fn myelin_session_cancel(session: *mut myelin_session) -> c_int {
    guarded(ptr::null_mut(), MYELIN_ERR_PANIC, || {
        let handle = match session_handle(session) {
            Ok(handle) => handle,
            Err(error) => return error_code(&error),
        };
        handle.cancel();
        MYELIN_OK
    })
}

/// Free an engine handle. Null is a no-op.
///
/// # Safety
/// `engine` must be null or a live handle that has not been freed before.
/// Sessions created from it may outlive the call.
#[no_mangle]
pub unsafe extern "C" fn myelin_engine_free(engine: *mut myelin_engine) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if engine.is_null() {
            return;
        }
        let boxed = engine.cast::<EngineBox>();
        // SAFETY: the pointer was produced by `myelin_engine_new`.
        unsafe {
            if (*boxed).magic == ENGINE_MAGIC {
                (*boxed).magic = 0;
                drop(Box::from_raw(boxed));
            }
        }
    }));
}

/// Free a session handle. Null is a no-op.
///
/// # Safety
/// `session` must be null or a live handle that has not been freed before and
/// is not currently inside `myelin_session_generate`.
#[no_mangle]
pub unsafe extern "C" fn myelin_session_free(session: *mut myelin_session) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() {
            return;
        }
        let boxed = session.cast::<SessionBox>();
        // SAFETY: the pointer was produced by `myelin_session_new`.
        unsafe {
            if (*boxed).magic == SESSION_MAGIC {
                (*boxed).magic = 0;
                drop(Box::from_raw(boxed));
            }
        }
    }));
}

fn guarded<T>(err: *mut myelin_error, fallback: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_error(err, MYELIN_ERR_PANIC, "panic in myelin FFI");
            fallback
        }
    }
}

fn read_engine_cfg(
    cfg: *const myelin_engine_cfg,
) -> Result<(BackendKind, Option<String>), RuntimeError> {
    if cfg.is_null() {
        return Ok((BackendKind::Auto, None));
    }
    // SAFETY: caller promises a valid pointer to the declared struct.
    let cfg = unsafe { &*cfg };
    if cfg.size < size_of::<myelin_engine_cfg>() {
        return Err(RuntimeError::InvalidArgument(format!(
            "engine cfg size {} < {}",
            cfg.size,
            size_of::<myelin_engine_cfg>()
        )));
    }
    let kind = BackendKind::from_u32(cfg.backend)?;
    let model_path = if cfg.model_path.is_null() {
        None
    } else {
        Some(read_cstr(cfg.model_path, "engine cfg model_path")?)
    };
    Ok((kind, model_path))
}

fn read_sampling(sampling: *const myelin_sampling) -> Result<Sampling, RuntimeError> {
    if sampling.is_null() {
        return Ok(Sampling::default());
    }
    // SAFETY: caller promises a valid pointer to the declared struct.
    let sampling = unsafe { &*sampling };
    if sampling.size < size_of::<myelin_sampling>() {
        return Err(RuntimeError::InvalidArgument(format!(
            "sampling size {} < {}",
            sampling.size,
            size_of::<myelin_sampling>()
        )));
    }
    Ok(Sampling {
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        max_tokens: sampling.max_tokens,
        seed: sampling.seed,
    })
}

fn read_messages(
    messages: *const myelin_message,
    message_count: usize,
) -> Result<Vec<ChatMessage>, RuntimeError> {
    if message_count == 0 {
        return Err(RuntimeError::InvalidArgument(
            "message count is zero".to_owned(),
        ));
    }
    if messages.is_null() {
        return Err(RuntimeError::InvalidArgument(
            "messages pointer is null".to_owned(),
        ));
    }
    // SAFETY: caller promises `message_count` valid entries.
    let raw = unsafe { slice::from_raw_parts(messages, message_count) };
    let mut converted = Vec::with_capacity(message_count);
    for (index, message) in raw.iter().enumerate() {
        let role = read_cstr(message.role, &format!("messages[{index}].role"))?;
        let content = read_cstr(message.content, &format!("messages[{index}].content"))?;
        converted.push(ChatMessage { role, content });
    }
    Ok(converted)
}

fn read_cstr(pointer: *const c_char, field: &str) -> Result<String, RuntimeError> {
    if pointer.is_null() {
        return Err(RuntimeError::InvalidArgument(format!("{field} is null")));
    }
    // SAFETY: caller promises a NUL-terminated string.
    let value = unsafe { CStr::from_ptr(pointer) };
    value
        .to_str()
        .map(str::to_owned)
        .map_err(|_| RuntimeError::InvalidArgument(format!("{field} is not valid UTF-8")))
}

fn engine_handle(engine: *mut myelin_engine) -> Result<Arc<EngineHandle>, RuntimeError> {
    if engine.is_null() {
        return Err(RuntimeError::InvalidArgument(
            "engine handle is null".to_owned(),
        ));
    }
    // SAFETY: pointer must come from `myelin_engine_new`.
    let boxed = unsafe { &*engine.cast::<EngineBox>() };
    if boxed.magic != ENGINE_MAGIC {
        return Err(RuntimeError::InvalidArgument(
            "engine handle is not valid".to_owned(),
        ));
    }
    Ok(Arc::clone(&boxed.handle))
}

fn session_handle(session: *mut myelin_session) -> Result<Arc<SessionHandle>, RuntimeError> {
    if session.is_null() {
        return Err(RuntimeError::InvalidArgument(
            "session handle is null".to_owned(),
        ));
    }
    // SAFETY: pointer must come from `myelin_session_new`.
    let boxed = unsafe { &*session.cast::<SessionBox>() };
    if boxed.magic != SESSION_MAGIC {
        return Err(RuntimeError::InvalidArgument(
            "session handle is not valid".to_owned(),
        ));
    }
    Ok(Arc::clone(&boxed.handle))
}

fn error_code(error: &RuntimeError) -> c_int {
    match error {
        RuntimeError::InvalidArgument(_) => MYELIN_ERR_INVALID_ARGUMENT,
        RuntimeError::InvalidState(_) => MYELIN_ERR_INVALID_STATE,
        RuntimeError::Io(_) => MYELIN_ERR_IO,
        RuntimeError::Backend(_) => MYELIN_ERR_BACKEND,
        RuntimeError::Cancelled => MYELIN_ERR_CANCELLED,
        RuntimeError::Internal(_) => MYELIN_ERR_INTERNAL,
    }
}

fn report(err: *mut myelin_error, error: &RuntimeError) -> c_int {
    let code = error_code(error);
    set_error(err, code, &error.to_string());
    code
}

fn report_null<T>(err: *mut myelin_error, error: &RuntimeError) -> *mut T {
    report(err, error);
    ptr::null_mut()
}

fn clear_error(err: *mut myelin_error) {
    if err.is_null() {
        return;
    }
    // SAFETY: out-parameter contract; caller allocated the struct.
    let err = unsafe { &mut *err };
    err.size = size_of::<myelin_error>();
    err.code = MYELIN_OK;
    err.message = [0; ERROR_MESSAGE_LEN];
}

fn set_error(err: *mut myelin_error, code: c_int, message: &str) {
    if err.is_null() {
        return;
    }
    // SAFETY: out-parameter contract; caller allocated the struct.
    let err = unsafe { &mut *err };
    err.size = size_of::<myelin_error>();
    err.code = code;
    write_message(err, message);
}

fn write_message(err: &mut myelin_error, message: &str) {
    err.message = [0; ERROR_MESSAGE_LEN];
    let bytes = message.as_bytes();
    let length = bytes.len().min(ERROR_MESSAGE_LEN - 1);
    for (slot, byte) in err.message.iter_mut().zip(bytes).take(length) {
        *slot = *byte as c_char;
    }
}
