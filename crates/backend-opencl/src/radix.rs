use std::collections::HashMap;

pub const PAGE: usize = 32;

pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ parent.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h | 1
}

pub fn chain_hashes(prompt: &[u32]) -> Vec<u64> {
    let mut out = Vec::with_capacity(prompt.len() / PAGE);
    let mut parent = 0u64;
    for block in prompt.as_chunks::<PAGE>().0 {
        parent = block_hash(parent, block);
        out.push(parent);
    }
    out
}

#[derive(Debug, Clone, Copy, Default)]
struct Page {
    hash: u64,
    refs: u32,
    used: u64,
}

const NOT_FREE: usize = usize::MAX;

pub struct RadixCache {
    pages: Vec<Page>,
    index: HashMap<u64, usize>,
    free: Vec<usize>,
    slot: Vec<usize>,
    clock: u64,
    hits: u64,
}

impl RadixCache {
    pub fn new(pages: usize) -> Self {
        let free: Vec<usize> = (0..pages).rev().collect();
        let mut slot = vec![NOT_FREE; pages];
        for (i, &page) in free.iter().enumerate() {
            slot[page] = i;
        }
        Self {
            pages: vec![Page::default(); pages],
            index: HashMap::new(),
            free,
            slot,
            clock: 0,
            hits: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.pages.len()
    }

    pub fn cached(&self) -> usize {
        self.index.len()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn refs(&self, page: usize) -> u32 {
        self.pages[page].refs
    }

    pub fn used_pages(&self) -> usize {
        self.pages.len() - self.free.len()
    }

    pub fn clear(&mut self) {
        self.pages.iter_mut().for_each(|p| *p = Page::default());
        self.index.clear();
        self.free.clear();
        self.free.extend((0..self.pages.len()).rev());
        for (i, &page) in self.free.iter().enumerate() {
            self.slot[page] = i;
        }
        self.clock = 0;
        self.hits = 0;
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn take(&mut self, hash: u64) -> Option<usize> {
        let page = *self.index.get(&hash)?;
        self.remove_free(page);
        let now = self.tick();
        let p = &mut self.pages[page];
        p.refs += 1;
        p.used = now;
        self.hits += 1;
        Some(page)
    }

    pub fn alloc(&mut self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for &page in &self.free {
            let better = match best {
                None => true,
                Some(b) => self.pages[page].used < self.pages[b].used,
            };
            if better {
                best = Some(page);
            }
        }
        let page = best?;
        self.remove_free(page);
        let old = self.pages[page].hash;
        if old != 0 && self.index.get(&old) == Some(&page) {
            self.index.remove(&old);
        }
        let now = self.tick();
        let p = &mut self.pages[page];
        p.hash = 0;
        p.refs = 1;
        p.used = now;
        Some(page)
    }

    pub fn release(&mut self, page: usize) {
        let p = &mut self.pages[page];
        if p.refs == 0 {
            return;
        }
        p.refs -= 1;
        if p.refs == 0 {
            p.used = self.clock + 1;
            self.add_free(page);
        }
    }

    pub fn publish(&mut self, page: usize, hash: u64) {
        if self.index.contains_key(&hash) {
            return;
        }
        let old = self.pages[page].hash;
        if old != 0 && self.index.get(&old) == Some(&page) {
            self.index.remove(&old);
        }
        self.pages[page].hash = hash;
        self.index.insert(hash, page);
    }

    pub fn lookup(&self, hash: u64) -> Option<usize> {
        self.index.get(&hash).copied()
    }

    fn is_free(&self, page: usize) -> bool {
        self.slot[page] != NOT_FREE
    }

    fn add_free(&mut self, page: usize) {
        if self.is_free(page) {
            return;
        }
        self.slot[page] = self.free.len();
        self.free.push(page);
    }

    fn remove_free(&mut self, page: usize) {
        let pos = self.slot[page];
        if pos == NOT_FREE {
            return;
        }
        let last = *self.free.last().expect("free list not empty");
        self.free.swap_remove(pos);
        if pos < self.free.len() {
            self.slot[last] = pos;
        }
        self.slot[page] = NOT_FREE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_chain_depends_on_prefix() {
        let a = chain_hashes(&vec![7u32; 96]);
        let mut other = vec![7u32; 96];
        other[5] = 8;
        let b = chain_hashes(&other);
        assert_eq!(a.len(), 3);
        assert_ne!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        let mut same = vec![7u32; 96];
        same[80] = 9;
        let c = chain_hashes(&same);
        assert_eq!(a[0], c[0]);
        assert_eq!(a[1], c[1]);
        assert_ne!(a[2], c[2]);
    }

    #[test]
    fn chain_hashes_only_full_pages() {
        assert!(chain_hashes(&[1u32; 31]).is_empty());
        assert_eq!(chain_hashes(&[1u32; 32]).len(), 1);
        assert_eq!(chain_hashes(&[1u32; 65]).len(), 2);
    }

    #[test]
    fn hashes_are_never_zero() {
        for i in 0..64u32 {
            let h = block_hash(0, &[i; PAGE]);
            assert_ne!(h, 0);
        }
    }

    #[test]
    fn alloc_release_roundtrip() {
        let mut c = RadixCache::new(4);
        let a = c.alloc().unwrap();
        let b = c.alloc().unwrap();
        assert_ne!(a, b);
        assert_eq!(c.used_pages(), 2);
        c.release(a);
        assert_eq!(c.used_pages(), 1);
        c.release(b);
        assert_eq!(c.used_pages(), 0);
        assert_ne!(c.alloc().unwrap(), c.alloc().unwrap());
    }

    #[test]
    fn refcount_keeps_page_alive() {
        let mut c = RadixCache::new(4);
        let p = c.alloc().unwrap();
        c.publish(p, 42);
        assert_eq!(c.take(42), Some(p));
        c.release(p);
        assert_eq!(c.refs(p), 1);
        assert_eq!(c.cached(), 1);
        assert_eq!(c.take(42), Some(p));
        assert_eq!(c.refs(p), 2);
        c.release(p);
        c.release(p);
        assert_eq!(c.refs(p), 0);
        assert_eq!(c.cached(), 1);
        assert_eq!(c.take(42), Some(p));
        assert_eq!(c.refs(p), 1);
        assert_eq!(c.used_pages(), 1);
    }

    #[test]
    fn alloc_evicts_least_recently_used() {
        let mut c = RadixCache::new(3);
        let a = c.alloc().unwrap();
        let b = c.alloc().unwrap();
        let d = c.alloc().unwrap();
        c.publish(a, 1);
        c.publish(b, 2);
        c.publish(d, 3);
        c.release(a);
        c.release(b);
        c.release(d);
        assert_eq!(c.take(1), Some(a));
        c.release(a);
        let e = c.alloc().unwrap();
        assert_ne!(e, a);
        assert_eq!(c.lookup(1), Some(a));
        let f = c.alloc().unwrap();
        assert!(e == b || e == d);
        assert!(f != e);
    }

    #[test]
    fn alloc_clears_evicted_hash() {
        let mut c = RadixCache::new(1);
        let p = c.alloc().unwrap();
        c.publish(p, 9);
        c.release(p);
        let q = c.alloc().unwrap();
        assert_eq!(q, p);
        assert_eq!(c.lookup(9), None);
        assert_eq!(c.alloc(), None);
    }

    #[test]
    fn publish_keeps_first_owner() {
        let mut c = RadixCache::new(2);
        let a = c.alloc().unwrap();
        let b = c.alloc().unwrap();
        c.publish(a, 5);
        c.publish(b, 5);
        assert_eq!(c.lookup(5), Some(a));
    }

    #[test]
    fn clear_resets_everything() {
        let mut c = RadixCache::new(4);
        let p = c.alloc().unwrap();
        c.publish(p, 7);
        assert_eq!(c.take(7), Some(p));
        c.clear();
        assert_eq!(c.cached(), 0);
        assert_eq!(c.used_pages(), 0);
        assert_eq!(c.hits(), 0);
        assert_eq!(c.lookup(7), None);
    }
}
