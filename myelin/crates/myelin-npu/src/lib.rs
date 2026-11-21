//! NPU transport boundary.
//!
//! The real FastRPC/DSP backend will implement [`NpuBackend`]. Until then
//! [`MockBackend`] lets the FFI and the Android build exercise the whole
//! call path on the host and on the device.

use std::fmt;

use myelin_core::ModelSpec;

/// Transport-level failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpuError {
    NotPrepared,
    Unsupported(String),
    Transport(String),
}

impl fmt::Display for NpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPrepared => f.write_str("NPU backend is not prepared"),
            Self::Unsupported(reason) => write!(f, "unsupported NPU operation: {reason}"),
            Self::Transport(reason) => write!(f, "NPU transport error: {reason}"),
        }
    }
}

impl std::error::Error for NpuError {}

/// Backend contract. `step` is intentionally model-agnostic: the spec is
/// passed once through `prepare`, and per-token work uses plain buffers.
pub trait NpuBackend: Send {
    fn name(&self) -> &'static str;

    /// Adopt a validated model spec and allocate device-side state.
    fn prepare(&mut self, spec: &ModelSpec) -> Result<(), NpuError>;

    /// Run one step. Returns the number of output elements written.
    fn step(&mut self, input: &[u32], out: &mut [u32]) -> Result<usize, NpuError>;

    fn reset(&mut self) -> Result<(), NpuError>;
}

/// Deterministic host-side backend used by the skeleton.
#[derive(Debug, Default)]
pub struct MockBackend {
    prepared: bool,
    steps: u64,
}

impl MockBackend {
    pub const fn steps(&self) -> u64 {
        self.steps
    }
}

impl NpuBackend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn prepare(&mut self, _spec: &ModelSpec) -> Result<(), NpuError> {
        self.prepared = true;
        self.steps = 0;
        Ok(())
    }

    fn step(&mut self, input: &[u32], out: &mut [u32]) -> Result<usize, NpuError> {
        if !self.prepared {
            return Err(NpuError::NotPrepared);
        }
        let written = out.len().min(input.len());
        let offset = u32::try_from(self.steps).unwrap_or(u32::MAX);
        for (dst, src) in out.iter_mut().zip(input.iter()).take(written) {
            *dst = src.wrapping_add(offset);
        }
        self.steps = self.steps.saturating_add(1);
        Ok(written)
    }

    fn reset(&mut self) -> Result<(), NpuError> {
        self.steps = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_is_deterministic() {
        let spec = ModelSpec::placeholder();
        let mut backend = MockBackend::default();
        backend.prepare(&spec).expect("prepare");
        assert_eq!(backend.name(), "mock");

        let mut a = [0u32; 3];
        let mut b = [0u32; 3];
        assert_eq!(backend.step(&[1, 2, 3], &mut a), Ok(3));
        assert_eq!(backend.step(&[1, 2, 3], &mut b), Ok(3));
        assert_eq!(a, [1, 2, 3]);
        assert_eq!(b, [2, 3, 4]);

        backend.reset().expect("reset");
        let mut c = [0u32; 3];
        backend.step(&[1, 2, 3], &mut c).expect("step");
        assert_eq!(c, a);
    }

    #[test]
    fn unprepared_backend_rejects_step() {
        let mut backend = MockBackend::default();
        let mut out = [0u32; 1];
        assert_eq!(backend.step(&[1], &mut out), Err(NpuError::NotPrepared));
    }
}
