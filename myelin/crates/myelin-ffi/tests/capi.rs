use std::ffi::{c_char, c_void, CStr, CString};
use std::fs;
use std::mem::size_of;
use std::ptr;

use myelin::capi::{
    myelin_engine_cfg, myelin_engine_free, myelin_engine_load, myelin_engine_new, myelin_error,
    myelin_message, myelin_session, myelin_session_cancel, myelin_session_free,
    myelin_session_generate, myelin_session_new, myelin_token_event, myelin_version_string,
    MYELIN_ABI_VERSION, MYELIN_BACKEND_MOCK, MYELIN_BACKEND_NPU, MYELIN_ERR_BACKEND,
    MYELIN_ERR_INVALID_ARGUMENT, MYELIN_ERR_IO, MYELIN_EVENT_DELTA, MYELIN_EVENT_DONE, MYELIN_OK,
};

struct Sink {
    deltas: Vec<u32>,
    done: usize,
    last_text: Option<String>,
    cancel_at: Option<u32>,
    session: *mut myelin_session,
}

impl Sink {
    fn new(session: *mut myelin_session) -> Self {
        Self {
            deltas: Vec::new(),
            done: 0,
            last_text: None,
            cancel_at: None,
            session,
        }
    }
}

unsafe extern "C" fn on_event(event: *const myelin_token_event, user: *mut c_void) -> i32 {
    // SAFETY: the FFI passes a live event and the sink outlives generate().
    let event = unsafe { &*event };
    let sink = unsafe { &mut *(user.cast::<Sink>()) };
    assert!(event.size >= size_of::<myelin_token_event>());
    match event.kind {
        MYELIN_EVENT_DELTA => {
            sink.deltas.push(event.token);
            if !event.text.is_null() {
                // SAFETY: delta text is NUL-terminated for the callback.
                let text = unsafe { CStr::from_ptr(event.text) };
                sink.last_text = Some(text.to_string_lossy().into_owned());
            }
            if sink.cancel_at == Some(event.index) {
                // SAFETY: the session is alive for the duration of generate().
                let code = unsafe { myelin_session_cancel(sink.session) };
                assert_eq!(code, MYELIN_OK);
            }
        }
        MYELIN_EVENT_DONE => sink.done += 1,
        _ => {}
    }
    0
}

fn new_error() -> myelin_error {
    myelin_error {
        size: size_of::<myelin_error>(),
        code: MYELIN_OK,
        message: [0; 512],
    }
}

fn error_message(err: &myelin_error) -> String {
    // SAFETY: message is a fixed, NUL-terminated buffer owned by the struct.
    let message = unsafe { CStr::from_ptr(err.message.as_ptr().cast::<c_char>()) };
    message.to_string_lossy().into_owned()
}

fn engine_cfg(backend: u32, model_path: *const c_char) -> myelin_engine_cfg {
    myelin_engine_cfg {
        size: size_of::<myelin_engine_cfg>(),
        backend,
        model_path,
    }
}

fn message(role: &CString, content: &CString) -> myelin_message {
    myelin_message {
        role: role.as_ptr(),
        content: content.as_ptr(),
    }
}

#[test]
fn version_is_exposed_through_extern_c() {
    // Declared independently of the Rust wrapper to prove the C symbol exists.
    unsafe extern "C" {
        fn myelin_version() -> u32;
    }
    // SAFETY: plain C function with no preconditions.
    assert_eq!(unsafe { myelin_version() }, MYELIN_ABI_VERSION);
    // SAFETY: static NUL-terminated string.
    let version = unsafe { CStr::from_ptr(myelin_version_string()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn generates_five_tokens_and_done() {
    let mut err = new_error();
    let cfg = engine_cfg(MYELIN_BACKEND_MOCK, ptr::null());
    // SAFETY: all pointers refer to live stack values.
    let engine = unsafe { myelin_engine_new(&cfg, &mut err) };
    assert!(!engine.is_null(), "{}", error_message(&err));
    // SAFETY: engine is a live handle.
    let session = unsafe { myelin_session_new(engine, &mut err) };
    assert!(!session.is_null(), "{}", error_message(&err));

    let role = CString::new("user").unwrap();
    let content = CString::new("hello").unwrap();
    let messages = [message(&role, &content)];
    let mut sink = Sink::new(session);
    // SAFETY: pointers are valid for the call; the callback only touches sink.
    let code = unsafe {
        myelin_session_generate(
            session,
            messages.as_ptr(),
            messages.len(),
            ptr::null(),
            Some(on_event),
            ptr::addr_of_mut!(sink).cast(),
            &mut err,
        )
    };
    assert_eq!(code, MYELIN_OK, "{}", error_message(&err));
    assert_eq!(sink.deltas.len(), 5);
    assert_eq!(sink.done, 1);
    assert!(sink
        .last_text
        .as_deref()
        .is_some_and(|text| text.starts_with('<')));
    assert_eq!(err.code, MYELIN_OK);

    // SAFETY: each handle is freed exactly once.
    unsafe {
        myelin_session_free(session);
        myelin_engine_free(engine);
    }
}

#[test]
fn cancellation_stops_the_stream_and_session_recovers() {
    let mut err = new_error();
    let cfg = engine_cfg(MYELIN_BACKEND_MOCK, ptr::null());
    // SAFETY: live stack pointers.
    let engine = unsafe { myelin_engine_new(&cfg, &mut err) };
    assert!(!engine.is_null());
    // SAFETY: live engine handle.
    let session = unsafe { myelin_session_new(engine, &mut err) };
    assert!(!session.is_null());

    let role = CString::new("user").unwrap();
    let content = CString::new("hello").unwrap();
    let messages = [message(&role, &content)];
    let mut sink = Sink::new(session);
    sink.cancel_at = Some(2);
    // SAFETY: pointers are valid for the call.
    let code = unsafe {
        myelin_session_generate(
            session,
            messages.as_ptr(),
            messages.len(),
            ptr::null(),
            Some(on_event),
            ptr::addr_of_mut!(sink).cast(),
            &mut err,
        )
    };
    assert_eq!(code, myelin::capi::MYELIN_ERR_CANCELLED);
    assert_eq!(sink.deltas.len(), 3);
    assert_eq!(sink.done, 0);

    let mut sink = Sink::new(session);
    // SAFETY: pointers are valid for the call.
    let code = unsafe {
        myelin_session_generate(
            session,
            messages.as_ptr(),
            messages.len(),
            ptr::null(),
            Some(on_event),
            ptr::addr_of_mut!(sink).cast(),
            &mut err,
        )
    };
    assert_eq!(code, MYELIN_OK);
    assert_eq!(sink.deltas.len(), 5);

    // SAFETY: each handle is freed exactly once.
    unsafe {
        myelin_session_free(session);
        myelin_engine_free(engine);
    }
}

#[test]
fn loads_gguf_header_and_rejects_bad_files() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("myelin-capi-{}.gguf", std::process::id()));
    fs::write(&path, b"GGUF\x00\x00\x00\x00").unwrap();
    let path_c = CString::new(path.to_str().unwrap()).unwrap();

    let mut err = new_error();
    let cfg = engine_cfg(MYELIN_BACKEND_MOCK, std::ptr::null());
    // SAFETY: live stack pointers.
    let engine = unsafe { myelin_engine_new(&cfg, &mut err) };
    assert!(!engine.is_null(), "{}", error_message(&err));
    assert_eq!(err.code, MYELIN_OK);
    // The loader now parses the checkpoint, so a magic-only stub is rejected
    // with a backend error instead of being accepted.
    let code = unsafe { myelin_engine_load(engine, path_c.as_ptr(), &mut err) };
    assert_ne!(code, MYELIN_OK);
    assert!(error_message(&err).contains("version"));

    fs::write(&path, b"NOPE").unwrap();
    // SAFETY: live engine handle and C string.
    let code = unsafe { myelin_engine_load(engine, path_c.as_ptr(), &mut err) };
    assert_eq!(code, MYELIN_ERR_INVALID_ARGUMENT);
    assert!(error_message(&err).contains("magic"));

    let missing = CString::new("/definitely/missing/myelin.gguf").unwrap();
    // SAFETY: live engine handle and C string.
    let code = unsafe { myelin_engine_load(engine, missing.as_ptr(), &mut err) };
    assert_eq!(code, MYELIN_ERR_IO);

    // SAFETY: handle freed once.
    unsafe { myelin_engine_free(engine) };
    let _ = fs::remove_file(path);
}

#[test]
fn rejects_invalid_arguments_without_crashing() {
    let mut err = new_error();
    let bad_cfg = myelin_engine_cfg {
        size: 0,
        backend: MYELIN_BACKEND_MOCK,
        model_path: ptr::null(),
    };
    // SAFETY: cfg is a valid pointer with an invalid size field.
    let engine = unsafe { myelin_engine_new(&bad_cfg, &mut err) };
    assert!(engine.is_null());
    assert_eq!(err.code, MYELIN_ERR_INVALID_ARGUMENT);
    assert!(!error_message(&err).is_empty());

    let npu_cfg = engine_cfg(MYELIN_BACKEND_NPU, ptr::null());
    // SAFETY: live stack pointer.
    let engine = unsafe { myelin_engine_new(&npu_cfg, &mut err) };
    assert!(engine.is_null());
    assert_eq!(err.code, MYELIN_ERR_BACKEND);

    // SAFETY: null handles are documented no-ops or argument errors.
    unsafe {
        myelin_engine_free(ptr::null_mut());
        myelin_session_free(ptr::null_mut());
        assert_eq!(
            myelin_session_cancel(ptr::null_mut()),
            MYELIN_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            myelin_engine_load(ptr::null_mut(), ptr::null(), &mut err),
            MYELIN_ERR_INVALID_ARGUMENT
        );
        assert!(myelin_session_new(ptr::null_mut(), &mut err).is_null());
        assert_eq!(
            myelin_session_generate(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null(),
                None,
                ptr::null_mut(),
                &mut err,
            ),
            MYELIN_ERR_INVALID_ARGUMENT
        );
    }
}

#[test]
fn rejects_empty_messages_and_bad_sampling() {
    let mut err = new_error();
    let cfg = engine_cfg(MYELIN_BACKEND_MOCK, ptr::null());
    // SAFETY: live stack pointers.
    let engine = unsafe { myelin_engine_new(&cfg, &mut err) };
    // SAFETY: live engine handle.
    let session = unsafe { myelin_session_new(engine, &mut err) };

    // SAFETY: null messages with zero count is an argument error.
    let code = unsafe {
        myelin_session_generate(
            session,
            ptr::null(),
            0,
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut err,
        )
    };
    assert_eq!(code, MYELIN_ERR_INVALID_ARGUMENT);

    let role = CString::new("user").unwrap();
    let content = CString::new("hello").unwrap();
    let messages = [message(&role, &content)];
    let sampling = myelin::capi::myelin_sampling {
        size: size_of::<myelin::capi::myelin_sampling>(),
        temperature: 1.0,
        top_k: 0,
        top_p: 1.5,
        max_tokens: 5,
        seed: 0,
    };
    // SAFETY: pointers are valid for the call.
    let code = unsafe {
        myelin_session_generate(
            session,
            messages.as_ptr(),
            messages.len(),
            &sampling,
            None,
            ptr::null_mut(),
            &mut err,
        )
    };
    assert_eq!(code, MYELIN_ERR_INVALID_ARGUMENT);

    // SAFETY: each handle is freed exactly once.
    unsafe {
        myelin_session_free(session);
        myelin_engine_free(engine);
    }
}
