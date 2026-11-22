//! KV cache contracts and the paged host implementation.
//!
//! The backend only ever sees the [`KvCache`] trait. A DSP backend keeps its
//! own pages resident on the device and implements the same trait as a thin
//! staging proxy, while the CPU backend uses [`PagedKvCache`] with a
//! [`PagePool`] allocator: 32-token pages addressed through a block table, the
//! same granularity the mobile scheduler uses.

use crate::backend::BackendError;

/// Tokens covered by one KV page.
pub const PAGE_TOKENS: usize = 32;

/// Read/write access to one layer's key/value history of one sequence.
pub trait KvCache {
    /// Number of tokens stored so far.
    fn len(&self) -> usize;

    /// Maximum number of tokens that can be appended.
    fn capacity(&self) -> usize;

    /// Width of one key or value vector.
    fn kv_dim(&self) -> usize;

    /// Store the key/value of the next position.
    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), BackendError>;

    /// Key at an absolute token index, `None` when the index is not stored.
    fn key(&self, index: usize) -> Option<&[f32]>;

    /// Value at an absolute token index, `None` when the index is not stored.
    fn value(&self, index: usize) -> Option<&[f32]>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Fixed-size pool of `PAGE_TOKENS`-token pages.
///
/// Keys and values live in two flat arrays so a page is a contiguous range and
/// a device backend can hand the same ranges to ION/rpcmem without repacking.
#[derive(Debug, Clone)]
pub struct PagePool {
    kv_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    free: Vec<usize>,
}

impl PagePool {
    /// Create a pool with `pages` pages of `kv_dim` values each.
    pub fn new(kv_dim: usize, pages: usize) -> Result<Self, BackendError> {
        if kv_dim == 0 {
            return Err(BackendError::Kv("page width is zero".to_owned()));
        }
        if pages == 0 {
            return Err(BackendError::Kv("page pool has no pages".to_owned()));
        }
        let capacity = kv_dim
            .checked_mul(PAGE_TOKENS)
            .and_then(|per_page| per_page.checked_mul(pages))
            .ok_or_else(|| BackendError::Kv("page pool size overflows usize".to_owned()))?;
        Ok(Self {
            kv_dim,
            keys: vec![0.0; capacity],
            values: vec![0.0; capacity],
            free: (0..pages).rev().collect(),
        })
    }

    /// Total number of pages.
    pub fn pages(&self) -> usize {
        self.keys.len() / (self.kv_dim * PAGE_TOKENS)
    }

    /// Number of pages that are not handed out.
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// Reserve a page, `None` when the pool is exhausted.
    pub fn allocate(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// Return a page to the pool.
    pub fn release(&mut self, page: usize) -> Result<(), BackendError> {
        if page >= self.pages() {
            return Err(BackendError::Kv(format!(
                "page {page} is outside the pool of {} pages",
                self.pages()
            )));
        }
        if self.free.contains(&page) {
            return Err(BackendError::Kv(format!("page {page} is already free")));
        }
        self.free.push(page);
        Ok(())
    }

    fn page_range(&self, page: usize) -> Result<std::ops::Range<usize>, BackendError> {
        if page >= self.pages() {
            return Err(BackendError::Kv(format!(
                "page {page} is outside the pool of {} pages",
                self.pages()
            )));
        }
        let start = page * self.kv_dim * PAGE_TOKENS;
        Ok(start..start + self.kv_dim * PAGE_TOKENS)
    }

    /// Key page contents, `kv_dim * PAGE_TOKENS` values.
    pub fn key_page(&self, page: usize) -> Result<&[f32], BackendError> {
        let range = self.page_range(page)?;
        Ok(&self.keys[range])
    }

    /// Value page contents, `kv_dim * PAGE_TOKENS` values.
    pub fn value_page(&self, page: usize) -> Result<&[f32], BackendError> {
        let range = self.page_range(page)?;
        Ok(&self.values[range])
    }
}

/// Paged KV history of one layer of one sequence.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    pool: PagePool,
    table: Vec<usize>,
    len: usize,
    capacity: usize,
    kv_dim: usize,
}

impl PagedKvCache {
    /// Create a cache that can hold `capacity` tokens.
    pub fn new(kv_dim: usize, capacity: usize) -> Result<Self, BackendError> {
        if capacity == 0 {
            return Err(BackendError::Kv("KV capacity is zero".to_owned()));
        }
        let pages = capacity.div_ceil(PAGE_TOKENS);
        let pool = PagePool::new(kv_dim, pages)?;
        Ok(Self {
            pool,
            table: Vec::with_capacity(pages),
            len: 0,
            capacity,
            kv_dim,
        })
    }

    /// Pages currently referenced by this sequence's block table.
    pub fn pages_used(&self) -> usize {
        self.table.len()
    }

    /// Page ids in token order.
    pub fn page_table(&self) -> &[usize] {
        &self.table
    }

    /// Release the token at `index` and everything after it.
    ///
    /// Used when a scheduler rewinds a sequence; pages that become unreferenced
    /// go back to the pool.
    pub fn truncate(&mut self, index: usize) -> Result<(), BackendError> {
        if index >= self.len {
            return Ok(());
        }
        let keep_pages = index.div_ceil(PAGE_TOKENS);
        while self.table.len() > keep_pages {
            if let Some(page) = self.table.pop() {
                self.pool.release(page)?;
            }
        }
        self.len = index;
        Ok(())
    }

    fn slot_of(&self, index: usize) -> Result<std::ops::Range<usize>, BackendError> {
        if index >= self.len {
            return Err(BackendError::Kv(format!(
                "token {index} is not stored (len {})",
                self.len
            )));
        }
        let page = *self
            .table
            .get(index / PAGE_TOKENS)
            .ok_or_else(|| BackendError::Kv(format!("no page for token {index}")))?;
        let offset = index % PAGE_TOKENS;
        let start = page * self.kv_dim * PAGE_TOKENS + offset * self.kv_dim;
        Ok(start..start + self.kv_dim)
    }
}

impl KvCache for PagedKvCache {
    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), BackendError> {
        if key.len() != self.kv_dim || value.len() != self.kv_dim {
            return Err(BackendError::Kv(format!(
                "append expects {0} values, got {1} and {2}",
                self.kv_dim,
                key.len(),
                value.len()
            )));
        }
        if self.len >= self.capacity {
            return Err(BackendError::Kv(format!(
                "cache is full at {} tokens",
                self.capacity
            )));
        }
        if self.len % PAGE_TOKENS == 0 {
            let page = self
                .pool
                .allocate()
                .ok_or_else(|| BackendError::Kv("page pool is exhausted".to_owned()))?;
            self.table.push(page);
        }
        let page = *self
            .table
            .last()
            .ok_or_else(|| BackendError::Kv("block table is empty".to_owned()))?;
        let offset = self.len % PAGE_TOKENS;
        let start = page * self.kv_dim * PAGE_TOKENS + offset * self.kv_dim;
        let slot = start..start + self.kv_dim;
        // Both flat arrays use the same range, so one range is enough.
        self.pool.keys[slot.clone()].copy_from_slice(key);
        self.pool.values[slot].copy_from_slice(value);
        self.len += 1;
        Ok(())
    }

    fn key(&self, index: usize) -> Option<&[f32]> {
        let slot = self.slot_of(index).ok()?;
        Some(&self.pool.keys[slot])
    }

    fn value(&self, index: usize) -> Option<&[f32]> {
        let slot = self.slot_of(index).ok()?;
        Some(&self.pool.values[slot])
    }
}
