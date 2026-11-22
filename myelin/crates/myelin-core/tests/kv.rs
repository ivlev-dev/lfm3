//! Paged KV allocation and history behaviour.

use myelin_core::backend::BackendError;
use myelin_core::kv::{KvCache, PagePool, PagedKvCache, PAGE_TOKENS};

#[test]
fn page_pool_reuses_released_pages() {
    let mut pool = PagePool::new(4, 2).expect("pool");
    assert_eq!(pool.pages(), 2);
    assert_eq!(pool.free_pages(), 2);
    let first = pool.allocate().expect("page 0");
    let second = pool.allocate().expect("page 1");
    assert_ne!(first, second);
    assert!(pool.allocate().is_none(), "pool must be exhausted");
    pool.release(first).expect("release");
    assert_eq!(pool.free_pages(), 1);
    assert_eq!(pool.allocate(), Some(first));
    assert!(pool.release(99).is_err(), "page id outside the pool");
    pool.release(second).expect("release second");
    assert!(pool.release(second).is_err(), "double release is rejected");
}

#[test]
fn pool_rejects_empty_shapes() {
    assert!(matches!(PagePool::new(0, 1), Err(BackendError::Kv(_))));
    assert!(matches!(PagePool::new(4, 0), Err(BackendError::Kv(_))));
}

#[test]
fn paged_cache_allocates_a_page_per_window() {
    let mut cache = PagedKvCache::new(2, 3 * PAGE_TOKENS).expect("cache");
    assert_eq!(cache.capacity(), 3 * PAGE_TOKENS);
    for token in 0..(2 * PAGE_TOKENS + 1) {
        cache
            .append(&[token as f32, 0.0], &[0.0, token as f32])
            .expect("append");
        let expected_pages = (token + 1).div_ceil(PAGE_TOKENS);
        assert_eq!(cache.pages_used(), expected_pages, "after token {token}");
    }
    assert_eq!(cache.pages_used(), 3);
}

#[test]
fn paged_cache_reads_across_page_boundaries() {
    let mut cache = PagedKvCache::new(2, 2 * PAGE_TOKENS).expect("cache");
    for token in 0..(PAGE_TOKENS + 4) {
        cache
            .append(&[token as f32, -(token as f32)], &[token as f32 * 2.0, 1.0])
            .expect("append");
    }
    assert_eq!(cache.len(), PAGE_TOKENS + 4);
    for token in 0..(PAGE_TOKENS + 4) {
        let key = cache.key(token).expect("key");
        assert_eq!(key, [token as f32, -(token as f32)]);
        let value = cache.value(token).expect("value");
        assert_eq!(value, [token as f32 * 2.0, 1.0]);
    }
    assert!(cache.key(PAGE_TOKENS + 4).is_none());
}

#[test]
fn paged_cache_rejects_overflow_and_bad_width() {
    let mut cache = PagedKvCache::new(2, 1).expect("cache");
    cache.append(&[1.0, 2.0], &[3.0, 4.0]).expect("append");
    let error = cache
        .append(&[5.0, 6.0], &[7.0, 8.0])
        .expect_err("capacity");
    assert!(matches!(error, BackendError::Kv(_)));
    let mut cache = PagedKvCache::new(2, 4).expect("cache");
    assert!(cache.append(&[1.0], &[2.0]).is_err());
    assert!(PagedKvCache::new(2, 0).is_err());
}

#[test]
fn truncate_releases_unused_pages() {
    let mut cache = PagedKvCache::new(1, 3 * PAGE_TOKENS).expect("cache");
    for token in 0..(2 * PAGE_TOKENS + 5) {
        cache.append(&[token as f32], &[0.0]).expect("append");
    }
    assert_eq!(cache.pages_used(), 3);
    cache.truncate(PAGE_TOKENS).expect("truncate");
    assert_eq!(cache.len(), PAGE_TOKENS);
    assert_eq!(cache.pages_used(), 1);
    assert!(cache.key(PAGE_TOKENS - 1).is_some());
    assert!(cache.key(PAGE_TOKENS).is_none());
    // The freed pages are usable again.
    cache
        .append(&[42.0], &[1.0])
        .expect("append after truncate");
    assert_eq!(cache.pages_used(), 2);
    assert_eq!(cache.key(PAGE_TOKENS), Some(&[42.0][..]));
}

#[test]
fn sliding_window_reads_only_the_tail_of_the_cache() {
    // The cache keeps the whole history; the window is a read range. This is
    // the contract the session relies on for sliding-attention layers.
    let mut cache = PagedKvCache::new(1, 64).expect("cache");
    for token in 0..40 {
        cache.append(&[token as f32], &[1.0]).expect("append");
    }
    let start = myelin_core::ops::window_start(myelin_core::AttentionKind::Sliding, 39, 8);
    assert_eq!(start, 32);
    let window: Vec<f32> = (start..cache.len())
        .map(|token| cache.key(token).expect("key")[0])
        .collect();
    assert_eq!(
        window,
        (32..40).map(|token| token as f32).collect::<Vec<_>>()
    );
}
