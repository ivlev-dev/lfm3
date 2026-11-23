#ifndef LFM_HVX_OPS_H
#define LFM_HVX_OPS_H

#include <stdint.h>

#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/q4.h"

void lfm_hvx_gemv_i4(const uint8_t *img, int n, int k, const int8_t *q, const float *ascale,
                     float *y);

void lfm_hvx_gemm_i4(const uint8_t *img, int n, int k, const int8_t *q, int m,
                     const float *ascale, float *y);

void lfm_hvx_ffn_i4(const uint8_t *gate, int ffn, int hidden, const uint8_t *up,
                    const uint8_t *down, const int8_t *xq, const float *xscale, float *out);

void lfm_hvx_attn_flash(lfm_kv_format fmt, const uint32_t *tbl, const uint8_t *kv_pages,
                        int kv_heads, int n_heads, int head_dim, const float *q_rot, uint32_t ntok,
                        uint32_t start, uint32_t window, float *out_rot, float *scratch);

void lfm_hvx_kvcache_commit(lfm_kv_format fmt, uint8_t *page, const float *k_rot,
                            const float *v_rot, int head, int head_dim);

int lfm_hvx_exec(const lfm_model_desc *desc, const lfm_step *step, const lfm_row *rows, void *arena,
                 float *logits);

#endif
