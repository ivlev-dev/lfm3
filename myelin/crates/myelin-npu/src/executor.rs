//! Adapters from [`Transport`] to `myelin-core`'s [`DspExecutor`].
//!
//! `myelin-core` cannot depend on this crate (the dependency runs the other
//! way), so the core exposes a narrow object-safe trait and this module
//! implements it for the real and mock transports.

use std::sync::Arc;

use myelin_core::backend::dsp::{DspCall, DspError, DspExecutor, DspHandle};

use crate::error::TransportError;
use crate::transport::{Call, Transport};

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
        }
    };
}

impl_executor!(crate::mock::MockTransport);

#[cfg(target_os = "android")]
impl_executor!(crate::cdsprpc::Cdsprpc);

/// Object-safe adapter for any [`Transport`] implementation.
pub struct Executor {
    transport: Arc<dyn Transport>,
}

impl Executor {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self { transport }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
