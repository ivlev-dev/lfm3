#include <stdint.h>
#include <arm_neon.h>

#define Q4_BLOCK 192
#define Q6_BLOCK 320
#define MAX_TOK 8

static inline int32_t hsum_s32(int32x4_t acc) {
    int32x2_t s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
}

static inline int32x4_t dot16(uint8x16_t w, int8x16_t x) {
    int16x8_t wlo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(w)));
    int16x8_t whi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(w)));
    int16x8_t xlo = vmovl_s8(vget_low_s8(x));
    int16x8_t xhi = vmovl_s8(vget_high_s8(x));
    int32x4_t acc = vdupq_n_s32(0);
    acc = vmlal_s16(acc, vget_low_s16(wlo), vget_low_s16(xlo));
    acc = vmlal_s16(acc, vget_high_s16(wlo), vget_high_s16(xlo));
    acc = vmlal_s16(acc, vget_low_s16(whi), vget_low_s16(xhi));
    acc = vmlal_s16(acc, vget_high_s16(whi), vget_high_s16(xhi));
    return acc;
}

static inline int32x4_t mla16(int32x4_t acc, uint8x16_t w, int8x16_t x) {
    return vaddq_s32(acc, dot16(w, x));
}

static inline uint8x16_t nib(uint8x16_t v, int hi) {
    if (hi) {
        return vshrq_n_u8(v, 4);
    }
    return vandq_u8(v, vdupq_n_u8(0x0F));
}

void lfm_q4_rows(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                 const int32_t *const *sums, int blocks, int ntok, float *out) {
    float acc[MAX_TOK];
    for (int t = 0; t < ntok; ++t) {
        acc[t] = 0.0f;
    }
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = w + (size_t)b * Q4_BLOCK;
        __builtin_prefetch(blk + 2 * Q4_BLOCK);
        __builtin_prefetch(blk + 2 * Q4_BLOCK + 64);
        const float *sc = (const float *)blk;
        const float *mn = (const float *)(blk + 32);
        const uint8_t *q = blk + 64;
        for (int g = 0; g < 8; ++g) {
            const uint8_t *wq = q + (g & 3) * 32;
            int hi = (g & 4) != 0;
            uint8x16_t wa = nib(vld1q_u8(wq), hi);
            uint8x16_t wb = nib(vld1q_u8(wq + 16), hi);
            for (int t = 0; t < ntok; ++t) {
                const int8_t *xv = xq[t] + b * 256 + g * 32;
                int32x4_t d = mla16(dot16(wa, vld1q_s8(xv)), wb, vld1q_s8(xv + 16));
                int32_t dot = hsum_s32(d);
                acc[t] += xs[t][b * 8 + g] *
                         (sc[g] * (float)dot - mn[g] * (float)sums[t][b * 8 + g]);
            }
        }
    }
    for (int t = 0; t < ntok; ++t) {
        out[t] = acc[t];
    }
}

void lfm_q6_rows(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                 const int32_t *const *sums16, int blocks, int ntok, float *out) {
    float acc[MAX_TOK];
    for (int t = 0; t < ntok; ++t) {
        acc[t] = 0.0f;
    }
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = w + (size_t)b * Q6_BLOCK;
        __builtin_prefetch(blk + 2 * Q6_BLOCK);
        __builtin_prefetch(blk + 2 * Q6_BLOCK + 64);
        const float *sc = (const float *)blk;
        const uint8_t *q = blk + 64;
        for (int k = 0; k < 16; ++k) {
            uint8x16_t wv = vld1q_u8(q + k * 16);
            for (int t = 0; t < ntok; ++t) {
                const int8_t *xv = xq[t] + b * 256 + k * 16;
                int32_t dot = hsum_s32(dot16(wv, vld1q_s8(xv)));
                int32_t s16 = sums16[t][b * 16 + k];
                acc[t] += xs[t][b * 8 + k / 2] * sc[k] * (float)(dot - 32 * s16);
            }
        }
    }
    for (int t = 0; t < ntok; ++t) {
        out[t] = acc[t];
    }
}
