//! Typed transport failures.
//!
//! FastRPC returns a single `int` from every entry point. The mapping below is
//! taken from `AEEStdErr.h` (SDK 6.7.0.0): on the DSP the constants carry the
//! `0x80000400` offset, on the host the offset is zero, so the low 16 bits are
//! the stable part and are what [`fastrpc_error_name`] matches.

use std::fmt;

/// Failure of the NPU transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// `libcdsprpc.so` could not be loaded.
    LibraryNotFound { library: String, detail: String },
    /// The library loaded but a required symbol is absent.
    SymbolMissing {
        library: String,
        symbol: &'static str,
    },
    /// `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` failed.
    SessionControl { code: i32, name: &'static str },
    /// `remote_handle_open` failed.
    HandleOpen { module: String, code: i32 },
    /// A method returned a non-zero FastRPC code.
    Rpc {
        method: u32,
        code: i32,
        name: &'static str,
    },
    /// The RPC or the device did not answer in time (`AEE_EEXPIRED`).
    Timeout { method: u32, millis: u64 },
    /// A caller-provided buffer is too small for the method output.
    BufferTooSmall {
        buffer: &'static str,
        expected: usize,
        got: usize,
    },
    /// A sequence length does not fit the 32-bit `long` the IDL uses.
    InvalidLength { what: &'static str, len: usize },
    /// Shared-memory allocation (`rpcmem`) failed.
    SharedMemory { bytes: usize, code: i32 },
    /// The handle has already been closed.
    Closed { module: String },
}

impl TransportError {
    /// Original FastRPC code, when the failure carries one.
    pub const fn code(&self) -> Option<i32> {
        match self {
            Self::SessionControl { code, .. }
            | Self::HandleOpen { code, .. }
            | Self::Rpc { code, .. }
            | Self::SharedMemory { code, .. } => Some(*code),
            Self::LibraryNotFound { .. }
            | Self::SymbolMissing { .. }
            | Self::Timeout { .. }
            | Self::BufferTooSmall { .. }
            | Self::InvalidLength { .. }
            | Self::Closed { .. } => None,
        }
    }

    /// Method index the failure belongs to, when known.
    pub const fn method(&self) -> Option<u32> {
        match self {
            Self::Rpc { method, .. } | Self::Timeout { method, .. } => Some(*method),
            Self::LibraryNotFound { .. }
            | Self::SymbolMissing { .. }
            | Self::SessionControl { .. }
            | Self::HandleOpen { .. }
            | Self::BufferTooSmall { .. }
            | Self::InvalidLength { .. }
            | Self::SharedMemory { .. }
            | Self::Closed { .. } => None,
        }
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound { library, detail } => {
                write!(f, "cannot load {library}: {detail}")
            }
            Self::SymbolMissing { library, symbol } => {
                write!(f, "{library} does not export {symbol}")
            }
            Self::SessionControl { code, name } => write!(
                f,
                "remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) failed: 0x{code:08x} ({name})"
            ),
            Self::HandleOpen { module, code } => write!(
                f,
                "remote_handle_open({module}) failed: 0x{code:08x} ({})",
                fastrpc_error_name(*code)
            ),
            Self::Rpc { method, code, name } => write!(
                f,
                "FastRPC method {method} failed: 0x{code:08x} ({name})"
            ),
            Self::Timeout { method, millis } => {
                write!(f, "FastRPC method {method} timed out after {millis} ms")
            }
            Self::BufferTooSmall {
                buffer,
                expected,
                got,
            } => write!(
                f,
                "{buffer} buffer too small: need {expected} bytes, got {got}"
            ),
            Self::InvalidLength { what, len } => write!(
                f,
                "{what} length {len} does not fit a 32-bit IDL sequence count"
            ),
            Self::SharedMemory { bytes, code } => write!(
                f,
                "rpcmem allocation of {bytes} bytes failed: 0x{code:08x}"
            ),
            Self::Closed { module } => write!(f, "FastRPC handle for {module} is closed"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Human-readable name of a FastRPC/AEE code.
///
/// The DSP offset (`0x80000400`) is stripped, so both the host-side and the
/// device-side spelling resolve to the same name.
pub fn fastrpc_error_name(code: i32) -> &'static str {
    if code == 0 {
        return "AEE_SUCCESS";
    }
    if code == -1 {
        return "AEE_EUNKNOWN";
    }
    let raw = code as u32;
    let base = if raw & 0x8000_0000 != 0 {
        raw.wrapping_sub(0x8000_0400)
    } else {
        raw
    };
    match base & 0xffff {
        0x0001 => "AEE_EFAILED",
        0x0002 => "AEE_ENOMEMORY",
        0x0003 => "AEE_ECLASSNOTSUPPORT",
        0x0004 => "AEE_EVERSIONNOTSUPPORT",
        0x0005 => "AEE_EALREADYLOADED",
        0x0006 => "AEE_EUNABLETOLOAD",
        0x0007 => "AEE_EUNABLETOUNLOAD",
        0x000c => "AEE_EEXPIRED",
        0x000d => "AEE_EBADSTATE",
        0x000e => "AEE_EBADPARM",
        0x0010 => "AEE_EBADITEM",
        0x0014 => "AEE_EUNSUPPORTED",
        0x0016 => "AEE_ERESOURCENOTFOUND",
        0x001a => "AEE_EALREADY",
        0x0027 => "AEE_ENOSUCH",
        0x002c => "AEE_EINVHANDLE",
        0x002d => "AEE_EOUTOFHANDLES",
        0x002e => "AEE_EINTERRUPTED",
        0x003b => "AEE_EINVALIDDEVICE",
        0x003e => "AEE_ENOSUCHMOD",
        0x0042 => "AEE_ENOSUCHSYMBOL",
        0x0046 => "AEE_ENOSUCHHANDLE",
        0x0048 => "AEE_ENOSUCHMAP",
        0x004c => "AEE_EBADOFFSET",
        0x004d => "AEE_EBADSIZE",
        0x0050 => "AEE_EBADPID",
        0x0055 => "AEE_EBADMEMALIGN",
        0x0066 => "AEE_EIONMAP",
        0x0067 => "AEE_EIONALLOC",
        0x0068 => "AEE_ENORPCMEMORY",
        0x006b => "AEE_ENOTINITIALIZED",
        0x006c => "AEE_EUNSUPPORTEDAPI",
        0x0081 => "AEE_EUNSIGNEDMOD",
        0x0200 => "AEE_ERPC",
        0x0204 => "AEE_EWOULDBLOCK",
        _ => "AEE_UNKNOWN",
    }
}

/// Map one FastRPC return code to a typed error.
pub fn rpc_error(method: u32, code: i32) -> TransportError {
    let name = fastrpc_error_name(code);
    if name == "AEE_EEXPIRED" {
        return TransportError::Timeout { method, millis: 0 };
    }
    TransportError::Rpc { method, code, name }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_ignore_the_dsp_offset() {
        assert_eq!(fastrpc_error_name(0), "AEE_SUCCESS");
        assert_eq!(fastrpc_error_name(-1), "AEE_EUNKNOWN");
        assert_eq!(fastrpc_error_name(0x0000_000e), "AEE_EBADPARM");
        assert_eq!(fastrpc_error_name(0x8000_040e_u32 as i32), "AEE_EBADPARM");
        assert_eq!(fastrpc_error_name(0x8000_0600_u32 as i32), "AEE_ERPC");
        assert_eq!(
            fastrpc_error_name(0x8000_0481_u32 as i32),
            "AEE_EUNSIGNEDMOD"
        );
        assert_eq!(fastrpc_error_name(0x1234), "AEE_UNKNOWN");
    }

    #[test]
    fn expired_is_a_timeout() {
        match rpc_error(6, 0x8000_040c_u32 as i32) {
            TransportError::Timeout { method, millis } => {
                assert_eq!(method, 6);
                assert_eq!(millis, 0);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            rpc_error(6, 0x8000_040e_u32 as i32),
            TransportError::Rpc {
                method: 6,
                name: "AEE_EBADPARM",
                ..
            }
        ));
    }
}
