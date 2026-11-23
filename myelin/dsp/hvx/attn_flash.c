#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "hvx_ops.h"

#define LFM_ATTN_DIM 128
#define LFM_ATTN_PAGE LFM_KVARN_PAGE_TOKENS
#define LFM_ATTN_GQA_MAX 32
#define LFM_ATTN_SCRATCH_FLOATS (2 * LFM_ATTN_PAGE * LFM_ATTN_DIM)
#define LFM_ATTN_F16_BLOCK ((size_t)LFM_ATTN_PAGE * LFM_ATTN_DIM * 2u)
#define LFM_ATTN_F16_HEAD (2u * LFM_ATTN_F16_BLOCK)
#define LFM_ATTN_INV_SQRT_D 0.08838834764831845f

static void attn_decode(lfm_kv_format fmt, const uint8_t *page, uint32_t first, uint32_t last,
                        float *kb, float *vb)
{
    uint32_t j;
    int d;

    if (fmt == LFM_KV_F16) {
        for (j = first; j <= last; j++) {
            const uint8_t *ks = page + (size_t)j * LFM_ATTN_DIM * 2u;
            const uint8_t *vs = page + LFM_ATTN_F16_BLOCK + (size_t)j * LFM_ATTN_DIM * 2u;
            float *kd = kb + (size_t)(j - first) * LFM_ATTN_DIM;
            float *vd = vb + (size_t)(j - first) * LFM_ATTN_DIM;
            for (d = 0; d < LFM_ATTN_DIM; d++) {
                kd[d] = lfm_f16_to_f32(lfm_f16_load(ks + (size_t)d * 2u));
                vd[d] = lfm_f16_to_f32(lfm_f16_load(vs + (size_t)d * 2u));
            }
        }
        return;
    }

    for (j = first; j <= last; j++) {
        lfm_kvarn_decode_k(page, (int)j, kb + (size_t)(j - first) * LFM_ATTN_DIM);
        lfm_kvarn_decode_v(fmt, page, (int)j, vb + (size_t)(j - first) * LFM_ATTN_DIM);
    }
}

void lfm_hvx_attn_flash(lfm_kv_format fmt, const uint32_t *tbl, const uint8_t *kv_pages,
                        int kv_heads, int n_heads, int head_dim, const float *q_rot, uint32_t ntok,
                        uint32_t start, uint32_t window, float *out_rot, float *scratch)
{
    size_t head_bytes;
    size_t page_bytes;
    float *kblk;
    float *vblk;
    int group;
    int kh;
    int g;
    int d;
    uint32_t t;

    if (fmt != LFM_KV_F16 && fmt != LFM_KVARN_K4V2 && fmt != LFM_KVARN_K4V4)
        return;
    if (tbl == NULL || kv_pages == NULL || q_rot == NULL || out_rot == NULL || scratch == NULL)
        return;
    if (kv_heads <= 0 || n_heads <= 0 || head_dim != LFM_ATTN_DIM)
        return;
    if (n_heads % kv_heads != 0)
        return;
    group = n_heads / kv_heads;
    if (group > LFM_ATTN_GQA_MAX)
        return;

    head_bytes = fmt == LFM_KV_F16 ? LFM_ATTN_F16_HEAD : lfm_kvarn_head_bytes(fmt);
    page_bytes = head_bytes * (size_t)kv_heads;
    kblk = scratch;
    vblk = scratch + LFM_ATTN_PAGE * LFM_ATTN_DIM;

    for (t = 0; t < ntok; t++) {
        uint32_t pos = start + t;
        uint32_t k0 = lfm_window_start(pos, window);
        uint32_t p0 = k0 >> 5;
        uint32_t p1 = pos >> 5;

        for (kh = 0; kh < kv_heads; kh++) {
            float mrun[LFM_ATTN_GQA_MAX];
            float lrun[LFM_ATTN_GQA_MAX];
            float *acc[LFM_ATTN_GQA_MAX];
            uint32_t lp;

            for (g = 0; g < group; g++) {
                size_t h = (size_t)kh * (size_t)group + (size_t)g;
                acc[g] = out_rot + ((size_t)t * (size_t)n_heads + h) * LFM_ATTN_DIM;
                memset(acc[g], 0, sizeof(float) * LFM_ATTN_DIM);
                mrun[g] = -FLT_MAX;
                lrun[g] = 0.0f;
            }

            for (lp = p0; lp <= p1; lp++) {
                uint32_t first = lp == p0 ? (k0 & 31u) : 0u;
                uint32_t last = lp == p1 ? (pos & 31u) : 31u;
                uint32_t cnt = last - first + 1u;
                uint32_t j;
                const uint8_t *page = kv_pages + (size_t)tbl[lp] * page_bytes +
                                      (size_t)kh * head_bytes;

                attn_decode(fmt, page, first, last, kblk, vblk);

                for (g = 0; g < group; g++) {
                    size_t h = (size_t)kh * (size_t)group + (size_t)g;
                    const float *q = q_rot + ((size_t)t * (size_t)n_heads + h) * LFM_ATTN_DIM;
                    float mg = mrun[g];
                    float lg = lrun[g];
                    float *a = acc[g];

                    for (j = 0; j < cnt; j++) {
                        const float *krow = kblk + (size_t)j * LFM_ATTN_DIM;
                        const float *vrow = vblk + (size_t)j * LFM_ATTN_DIM;
                        float s = 0.0f;

                        for (d = 0; d < LFM_ATTN_DIM; d++)
                            s += q[d] * krow[d];
                        s *= LFM_ATTN_INV_SQRT_D;

                        if (s > mg) {
                            float alpha = expf(mg - s);
                            lg *= alpha;
                            for (d = 0; d < LFM_ATTN_DIM; d++)
                                a[d] *= alpha;
                            mg = s;
                            lg += 1.0f;
                            for (d = 0; d < LFM_ATTN_DIM; d++)
                                a[d] += vrow[d];
                        } else {
                            float p = expf(s - mg);
                            lg += p;
                            for (d = 0; d < LFM_ATTN_DIM; d++)
                                a[d] += p * vrow[d];
                        }
                    }

                    mrun[g] = mg;
                    lrun[g] = lg;
                }
            }

            for (g = 0; g < group; g++) {
                float inv = lrun[g] > 0.0f ? 1.0f / lrun[g] : 0.0f;
                float *a = acc[g];
                for (d = 0; d < LFM_ATTN_DIM; d++)
                    a[d] *= inv;
            }
        }
    }
}
