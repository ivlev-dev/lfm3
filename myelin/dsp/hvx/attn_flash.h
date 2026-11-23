#ifndef LFM_ATTN_FLASH_H
#define LFM_ATTN_FLASH_H

#include <stdint.h>

/*
 * Fused (flash) paged attention on HVX v68, GQA 16/4, head_dim 128.
 * One pass over KV with an online softmax in f32:
 *
 *   s_j = (q . k_j) * qscale * ks[j] / sqrt(128)
 *   acc = sum_j exp(s_j - m) v_j,  l = sum_j exp(s_j - m),  out = acc / l
 *
 * All pointers are 128-byte aligned; the KV layout is the resident cache
 * layout written by LFM_CMD_KVSTORE (slot = page * 32 + token % 32):
 *
 *   q      int8    [16][128]
 *   qscale int32   [16]     Q16 dequant multiplier per q head, NULL = 1.0
 *   k      int4    [4][slots][128] B, 64 B nibbles (code+8) + 64 B pad
 *   ks     int32   [4][slots]      Q16 dequant multiplier per token
 *   v      int8    [4][slots][128]
 *   vscale float   [4][128]        per-channel V dequant, NULL = 1.0
 *   tbl    int     [pages]         page -> physical page
 *   out    float   [16][128]
 *
 * window <= 0 means the whole history; the window includes `pos`.
 * `reps` repeats the computation (bench only), the last result is kept.
 * Not reentrant (static q scratch).
 */
void attn_flash(const int8_t *q, const int32_t *qscale,
                const int8_t *k, const int32_t *ks,
                const int8_t *v, const float *vscale,
                const int *tbl, int slots, int pos, int window, int reps,
                float *out);

#endif /* LFM_ATTN_FLASH_H */
