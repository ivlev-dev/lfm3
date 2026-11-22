//! Row-major f32 matrix with `n` output channels (rows) of `k` inputs each.
//!
//! HMX multiplies `activation[32 x K] * weight[K x 32]`, so the GGUF tensor
//! `dims = [K, N]` maps directly: row `n` is the weight vector of output
//! channel `n`.

use crate::{PackError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    k: usize,
    n: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(k: usize, n: usize, data: Vec<f32>) -> Result<Self> {
        if k == 0 || n == 0 {
            return Err(PackError::EmptyMatrix { k, n });
        }
        let expected = k.checked_mul(n).ok_or(PackError::MatrixShape {
            k,
            n,
            expected: usize::MAX,
            actual: data.len(),
        })?;
        if data.len() != expected {
            return Err(PackError::MatrixShape {
                k,
                n,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { k, n, data })
    }

    pub const fn k(&self) -> usize {
        self.k
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn row(&self, n: usize) -> &[f32] {
        &self.data[n * self.k..(n + 1) * self.k]
    }
}
