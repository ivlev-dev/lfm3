use std::collections::HashMap;

use crate::kv::PAGE_TOKENS;

pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ parent.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &token in tokens {
        hash ^= u64::from(token);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        hash ^= hash >> 29;
    }
    hash | 1
}

pub fn chain_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(tokens.len() / PAGE_TOKENS);
    let mut parent = 0u64;
    for block in tokens.chunks_exact(PAGE_TOKENS) {
        parent = block_hash(parent, block);
        hashes.push(parent);
    }
    hashes
}

#[derive(Debug, Clone, Copy)]
struct Block {
    page: usize,
    refs: u32,
    used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RadixStats {
    pub capacity: usize,
    pub cached: usize,
    pub pinned: usize,
    pub refs: u64,
    pub hits: u64,
    pub evictions: u64,
}

#[derive(Debug)]
pub struct RadixCache {
    blocks: HashMap<u64, Block>,
    capacity: usize,
    clock: u64,
    hits: u64,
    evictions: u64,
}

impl RadixCache {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            blocks: HashMap::with_capacity(max_blocks),
            capacity: max_blocks,
            clock: 0,
            hits: 0,
            evictions: 0,
        }
    }

    pub fn lookup(&self, tokens: &[u32]) -> (usize, usize) {
        let mut parent = 0u64;
        let mut matched = 0usize;
        for block in tokens.chunks_exact(PAGE_TOKENS) {
            let hash = block_hash(parent, block);
            if !self.blocks.contains_key(&hash) {
                break;
            }
            parent = hash;
            matched += 1;
        }
        (matched, matched * PAGE_TOKENS)
    }

    pub fn publish_block(&mut self, parent_hash: u64, tokens: &[u32], page: usize) -> Option<u64> {
        if tokens.len() != PAGE_TOKENS {
            return None;
        }
        let hash = block_hash(parent_hash, tokens);
        if self.blocks.contains_key(&hash) {
            return Some(hash);
        }
        if self.blocks.len() >= self.capacity && !self.evict() {
            return None;
        }
        let used = self.tick();
        self.blocks.insert(
            hash,
            Block {
                page,
                refs: 1,
                used,
            },
        );
        Some(hash)
    }

    pub fn take(&mut self, hash: u64) -> Option<usize> {
        let used = self.tick();
        let block = self.blocks.get_mut(&hash)?;
        block.refs += 1;
        block.used = used;
        self.hits += 1;
        Some(block.page)
    }

    pub fn release(&mut self, hash: u64) {
        if let Some(block) = self.blocks.get_mut(&hash) {
            if block.refs == 0 {
                return;
            }
            block.refs -= 1;
            if block.refs == 0 {
                block.used = self.clock + 1;
            }
        }
    }

    pub fn stats(&self) -> RadixStats {
        RadixStats {
            capacity: self.capacity,
            cached: self.blocks.len(),
            pinned: self.blocks.values().filter(|block| block.refs > 0).count(),
            refs: self
                .blocks
                .values()
                .map(|block| u64::from(block.refs))
                .sum(),
            hits: self.hits,
            evictions: self.evictions,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn evict(&mut self) -> bool {
        let victim = self
            .blocks
            .iter()
            .filter(|(_, block)| block.refs == 0)
            .min_by_key(|(_, block)| block.used)
            .map(|(&hash, _)| hash);
        match victim {
            Some(hash) => {
                self.blocks.remove(&hash);
                self.evictions += 1;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn publish_chain(cache: &mut RadixCache, tokens: &[u32], first_page: usize) -> u64 {
        let mut parent = 0u64;
        for (index, block) in tokens.chunks_exact(PAGE_TOKENS).enumerate() {
            parent = cache
                .publish_block(parent, block, first_page + index)
                .expect("block is published");
        }
        parent
    }

    #[test]
    fn chain_hashes_depend_on_prefix() {
        let base = vec![7u32; 3 * PAGE_TOKENS];
        let mut middle = base.clone();
        middle[5] = 8;
        let mut tail = base.clone();
        tail[2 * PAGE_TOKENS + 3] = 9;

        let a = chain_hashes(&base);
        let b = chain_hashes(&middle);
        let c = chain_hashes(&tail);

        assert_eq!(a.len(), 3);
        assert_ne!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        assert_eq!(a[0], c[0]);
        assert_eq!(a[1], c[1]);
        assert_ne!(a[2], c[2]);
    }

    #[test]
    fn chain_hashes_cover_full_blocks_only() {
        assert!(chain_hashes(&[]).is_empty());
        assert!(chain_hashes(&[1u32; PAGE_TOKENS - 1]).is_empty());
        assert_eq!(chain_hashes(&[1u32; PAGE_TOKENS]).len(), 1);
        assert_eq!(chain_hashes(&[1u32; 2 * PAGE_TOKENS + 1]).len(), 2);
    }

    #[test]
    fn hashes_are_never_zero() {
        for token in 0..64u32 {
            assert_ne!(block_hash(0, &[token; PAGE_TOKENS]), 0);
            assert_ne!(block_hash(u64::MAX, &[token; PAGE_TOKENS]), 0);
        }
    }

    #[test]
    fn lookup_matches_shared_prefix() {
        let mut cache = RadixCache::new(8);
        let shared: Vec<u32> = (0..2 * PAGE_TOKENS as u32).collect();
        publish_chain(&mut cache, &shared, 0);
        assert_eq!(cache.lookup(&shared), (2, 2 * PAGE_TOKENS));
        assert_eq!(cache.stats().refs, 2);

        let mut first = shared.clone();
        first.extend(500u32..532);
        let mut second = shared.clone();
        second.extend(600u32..632);
        assert_eq!(cache.lookup(&first), (2, 2 * PAGE_TOKENS));
        assert_eq!(cache.lookup(&second), (2, 2 * PAGE_TOKENS));
        assert_eq!(cache.stats().refs, 2);

        let mut diverged = shared.clone();
        diverged[3] = 999;
        assert_eq!(cache.lookup(&diverged), (0, 0));
    }

    #[test]
    fn lookup_stops_at_first_missing_block() {
        let mut cache = RadixCache::new(8);
        let tokens: Vec<u32> = (1..=3 * PAGE_TOKENS as u32).collect();
        cache
            .publish_block(0, &tokens[..PAGE_TOKENS], 4)
            .expect("first block is published");
        assert_eq!(cache.lookup(&tokens), (1, PAGE_TOKENS));
    }

    #[test]
    fn identical_middle_tokens_do_not_collide() {
        let mut cache = RadixCache::new(8);
        let mut left = vec![1u32; PAGE_TOKENS];
        left.extend(vec![7u32; PAGE_TOKENS]);
        let mut right = vec![2u32; PAGE_TOKENS];
        right.extend(vec![7u32; PAGE_TOKENS]);

        publish_chain(&mut cache, &left, 0);
        assert_eq!(cache.lookup(&right), (0, 0));

        publish_chain(&mut cache, &right, 2);
        assert_eq!(cache.lookup(&left), (2, 2 * PAGE_TOKENS));
        assert_eq!(cache.lookup(&right), (2, 2 * PAGE_TOKENS));
        assert_eq!(cache.stats().cached, 4);

        let left_hashes = chain_hashes(&left);
        let right_hashes = chain_hashes(&right);
        assert_eq!(
            left_hashes[1],
            block_hash(left_hashes[0], &[7u32; PAGE_TOKENS])
        );
        assert_ne!(left_hashes[1], right_hashes[1]);
    }

    #[test]
    fn refcount_keeps_block_cached() {
        let mut cache = RadixCache::new(2);
        let hash = cache
            .publish_block(0, &[3u32; PAGE_TOKENS], 7)
            .expect("block is published");
        assert_eq!(cache.stats().refs, 1);
        assert_eq!(cache.stats().pinned, 1);

        assert_eq!(cache.take(hash), Some(7));
        assert_eq!(cache.stats().refs, 2);
        assert_eq!(cache.stats().hits, 1);

        cache.release(hash);
        assert_eq!(cache.stats().refs, 1);
        cache.release(hash);
        assert_eq!(cache.stats().refs, 0);
        assert_eq!(cache.stats().cached, 1);
        assert_eq!(cache.stats().pinned, 0);

        cache.release(hash);
        assert_eq!(cache.stats().refs, 0);
        assert_eq!(cache.take(hash), Some(7));
        assert_eq!(cache.stats().refs, 1);
        assert!(cache.take(0).is_none());
        cache.release(0);
    }

    #[test]
    fn block_limit_evicts_released_blocks() {
        let mut cache = RadixCache::new(2);
        let first = cache
            .publish_block(0, &[1u32; PAGE_TOKENS], 0)
            .expect("first block is published");
        let second = cache
            .publish_block(0, &[2u32; PAGE_TOKENS], 1)
            .expect("second block is published");

        assert!(cache.publish_block(0, &[3u32; PAGE_TOKENS], 2).is_none());
        assert_eq!(cache.stats().cached, 2);
        assert_eq!(cache.stats().evictions, 0);

        assert_eq!(cache.take(first), Some(0));
        cache.release(second);
        cache.release(first);

        let third = cache
            .publish_block(0, &[3u32; PAGE_TOKENS], 2)
            .expect("third block is published");
        assert_eq!(cache.stats().cached, 2);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.take(first), Some(0));
        assert_eq!(cache.take(second), None);
        assert_eq!(cache.take(third), Some(2));
    }

    #[test]
    fn publish_keeps_first_page() {
        let mut cache = RadixCache::new(2);
        let hash = cache
            .publish_block(0, &[4u32; PAGE_TOKENS], 0)
            .expect("block is published");
        assert_eq!(cache.publish_block(0, &[4u32; PAGE_TOKENS], 1), Some(hash));
        assert_eq!(cache.stats().cached, 1);
        assert_eq!(cache.stats().refs, 1);
        assert_eq!(cache.take(hash), Some(0));
    }

    #[test]
    fn publish_requires_full_block() {
        let mut cache = RadixCache::new(2);
        assert!(cache
            .publish_block(0, &[1u32; PAGE_TOKENS - 1], 0)
            .is_none());
        assert!(cache
            .publish_block(0, &[1u32; PAGE_TOKENS + 1], 0)
            .is_none());
        assert_eq!(cache.stats().cached, 0);
    }

    #[test]
    fn zero_capacity_cache_stays_empty() {
        let mut cache = RadixCache::new(0);
        assert!(cache.publish_block(0, &[5u32; PAGE_TOKENS], 0).is_none());
        assert_eq!(cache.lookup(&[5u32; PAGE_TOKENS]), (0, 0));
        assert_eq!(cache.stats().capacity, 0);
        assert_eq!(cache.stats().cached, 0);
    }
}
