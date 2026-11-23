#ifndef LFM_PAGE_H
#define LFM_PAGE_H

#include <stdint.h>

#include "kvarn.h"

#define LFM_PAGE_TOKENS 32
#define LFM_PT_ENTRY_BYTES 4u
#define LFM_ARENA_ALIGN 256u
#define LFM_MAX_KV_POOL 16
#define LFM_MAX_PREFILL_TOKENS (1u << 24)

#define LFM_FLAG_PREFILL (1u << 0)

typedef struct {
    uint32_t token;
    uint32_t seq;
    uint32_t pos;
    uint32_t ntok;
    uint32_t kv_off;
    uint32_t tbl_off;
} lfm_row;

typedef struct {
    uint32_t n_rows;
    uint32_t ntok;
    uint32_t window;
    uint32_t flags;
    uint32_t max_rows;
    uint32_t reserved;
} lfm_step;

typedef struct {
    uint32_t n_rows;
    uint32_t ntok;
    uint32_t logits_off;
    uint32_t status;
} lfm_out;

typedef struct {
    uint32_t hidden;
    uint32_t ffn;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t kv_dim;
    uint32_t vocab;
    uint32_t ctx;
    uint32_t conv_cache;
    uint32_t sliding_window;
    uint32_t kv_layers;
    uint32_t kv_heads;
    uint32_t kv_pool;
    uint32_t kv_stage;
    uint32_t max_seqs;
    uint32_t kv_format;
} lfm_model_desc;

static inline uint32_t lfm_max_pages(uint32_t ctx, uint32_t kv_pool)
{
    return ((ctx + LFM_PAGE_TOKENS - 1u) / LFM_PAGE_TOKENS) * kv_pool;
}

static inline uint32_t lfm_kv_stage(uint32_t prefill_ntok)
{
    uint32_t n = prefill_ntok ? prefill_ntok : 1u;
    uint32_t s = 1u;
    while (s < n && s < (1u << 30))
        s <<= 1;
    return s < 64u ? 64u : s;
}

static inline uint64_t lfm_page_bytes(uint32_t fmt, uint32_t kv_heads, uint32_t head_dim)
{
    if (fmt == LFM_KV_F16)
        return (uint64_t)LFM_PAGE_TOKENS * kv_heads * head_dim * 2u * 2u;
    return (uint64_t)lfm_kvarn_page_bytes((lfm_kv_format)fmt, (int)kv_heads);
}

static inline uint64_t lfm_kv_arena_bytes(uint32_t kv_layers, uint32_t max_pages, uint32_t fmt,
                                          uint32_t kv_heads, uint32_t head_dim)
{
    return (uint64_t)kv_layers * max_pages * lfm_page_bytes(fmt, kv_heads, head_dim);
}

static inline uint32_t lfm_page_of(uint32_t pos)
{
    return pos >> 5;
}

static inline uint32_t lfm_slot_of(uint32_t tbl_entry, uint32_t pos)
{
    return (tbl_entry << 5) | (pos & 31u);
}

static inline uint32_t lfm_kv_logic_slot(const uint32_t *tbl, uint32_t pos)
{
    return lfm_slot_of(tbl[lfm_page_of(pos)], pos);
}

static inline uint32_t lfm_window_start(uint32_t pos, uint32_t window)
{
    if (window == 0)
        return 0;
    return pos + 1u > window ? pos + 1u - window : 0u;
}

#endif
