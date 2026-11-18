#include <stdint.h>
#include <string.h>
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

static inline int32_t dot_i8_32(int8x16_t wa, int8x16_t wb, const int8_t *x) {
    int8x16_t xa = vld1q_s8(x);
    int8x16_t xb = vld1q_s8(x + 16);
    int16x8_t p0 = vmull_s8(vget_low_s8(wa), vget_low_s8(xa));
    int16x8_t p1 = vmull_s8(vget_high_s8(wa), vget_high_s8(xa));
    int16x8_t p2 = vmull_s8(vget_low_s8(wb), vget_low_s8(xb));
    int16x8_t p3 = vmull_s8(vget_high_s8(wb), vget_high_s8(xb));
    int32x4_t z = vdupq_n_s32(0);
    int32x4_t a0 = vpadalq_s16(vpadalq_s16(z, p0), p1);
    int32x4_t a1 = vpadalq_s16(vpadalq_s16(z, p2), p3);
    return hsum_s32(vaddq_s32(a0, a1));
}

static inline uint8x16_t nib(uint8x16_t v, int hi) {
    if (hi) {
        return vshrq_n_u8(v, 4);
    }
    return vandq_u8(v, vdupq_n_u8(0x0F));
}

extern "C" void lfm_q4_rows(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
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

extern "C" void lfm_q6_rows(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
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

static inline float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s;
        } else {
            e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            out = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

static inline float xs_all(const float *const *xs, int t, int b) {
    return xs[t][b * 8];
}

static inline void q4_raw_scales_i(const uint8_t *blk, int *sc, int *mn) {
    const uint8_t *q = blk + 4;
    for (int j = 0; j < 8; ++j) {
        int jm = j & 3;
        int h = j >> 2;
        uint8_t a = q[jm], b = q[jm + 4], c = q[jm + 8];
        uint8_t lo_d = a & 63, lo_m = b & 63;
        uint8_t hi_d = (uint8_t)((c & 15) | ((a >> 6) << 4));
        uint8_t hi_m = (uint8_t)((c >> 4) | ((b >> 6) << 4));
        sc[j] = (int)(h == 0 ? lo_d : hi_d);
        mn[j] = (int)(h == 0 ? lo_m : hi_m);
    }
}

static inline void q4_raw_scales(const uint8_t *blk, float d, float dmin, float *sc, float *mn) {
    const uint8_t *q = blk + 4;
    for (int j = 0; j < 8; ++j) {
        int jm = j & 3;
        int h = j >> 2;
        uint8_t a = q[jm], b = q[jm + 4], c = q[jm + 8];
        uint8_t lo_d = a & 63, lo_m = b & 63;
        uint8_t hi_d = (uint8_t)((c & 15) | ((a >> 6) << 4));
        uint8_t hi_m = (uint8_t)((c >> 4) | ((b >> 6) << 4));
        sc[j] = d * (float)(h == 0 ? lo_d : hi_d);
        mn[j] = dmin * (float)(h == 0 ? lo_m : hi_m);
    }
}

template <int NT>
static void q4_rows_raw_t(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                          const int32_t *const *sums, int blocks, float *out) {
    constexpr int ntok = NT;
    float acc[NT];
    for (int t = 0; t < ntok; ++t) {
        acc[t] = 0.0f;
    }
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = w + (size_t)b * 144;
        __builtin_prefetch(blk + 6 * 144);
        __builtin_prefetch(blk + 6 * 144 + 64);
        __builtin_prefetch(blk + 6 * 144 + 128);
        float dw = half_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
        float dminw = half_to_float((uint16_t)(blk[2] | (blk[3] << 8)));
        int sc6[8], m6[8];
        q4_raw_scales_i(blk, sc6, m6);
        const uint8_t *qs = blk + 16;
        int sumi[NT];
        int sumn[NT];
        for (int t = 0; t < ntok; ++t) {
            sumi[t] = 0;
            sumn[t] = 0;
        }
        const int8x16_t bias = vdupq_n_s8(8);
        for (int g = 0; g < 8; ++g) {
            int8x16_t wa = vsubq_s8(vreinterpretq_s8_u8(nib(vld1q_u8(qs + (g / 2) * 32), g & 1)), bias);
            int8x16_t wb = vsubq_s8(vreinterpretq_s8_u8(nib(vld1q_u8(qs + (g / 2) * 32 + 16), g & 1)), bias);
            const int sc = sc6[g];
            const int mn = m6[g];
            for (int t = 0; t < ntok; ++t) {
                const int32_t *st = sums[t] + b * 8 + g;
                int dot = dot_i8_32(wa, wb, xq[t] + b * 256 + g * 32) + 8 * (*st);
                sumi[t] += dot * sc;
                sumn[t] += (*st) * mn;
            }
        }
        for (int t = 0; t < ntok; ++t) {
            float xsc = xs_all(xs, t, b);
            acc[t] += xsc * (dw * (float)sumi[t] - dminw * (float)sumn[t]);
        }
    }
    for (int t = 0; t < ntok; ++t) {
        out[t] = acc[t];
    }
}

template <int NT>
static void q6_rows_raw_t(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                          const int32_t *const *sums16, int blocks, float *out) {
    constexpr int ntok = NT;
    float acc[NT];
    for (int t = 0; t < ntok; ++t) {
        acc[t] = 0.0f;
    }
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = w + (size_t)b * 210;
        __builtin_prefetch(blk + 6 * 210);
        __builtin_prefetch(blk + 6 * 210 + 64);
        __builtin_prefetch(blk + 6 * 210 + 128);
        float d = half_to_float((uint16_t)(blk[208] | (blk[209] << 8)));
        for (int n = 0; n < 2; ++n) {
            for (int s = 0; s < 4; ++s) {
                const uint8_t *ql = blk + 64 * n + (s & 1) * 32;
                const uint8_t *qh = blk + 128 + 32 * n;
                const uint8x16_t m3 = vdupq_n_u8(3);
                const int8x16_t bias32 = vdupq_n_s8(32);
                for (int half = 0; half < 2; ++half) {
                    int k = n * 8 + s * 2 + half;
                    uint8x16_t qlv = vld1q_u8(ql + half * 16);
                    uint8x16_t qhv = vld1q_u8(qh + half * 16);
                    uint8x16_t nibs = (s < 2) ? vandq_u8(qlv, vdupq_n_u8(0x0F))
                                              : vshrq_n_u8(qlv, 4);
                    uint8x16_t hi = vandq_u8(vshlq_u8(qhv, vdupq_n_s8((int8_t)(-2 * s))), m3);
                    int8x16_t wv = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nibs, vshlq_n_u8(hi, 4))), bias32);
                    float sc = d * (float)(int8_t)blk[192 + k];
                    for (int t = 0; t < ntok; ++t) {
                        const int8_t *xv = xq[t] + b * 256 + k * 16;
                        int16x8_t p = vmull_s8(vget_low_s8(wv), vget_low_s8(vld1q_s8(xv)));
                        p = vmlal_s8(p, vget_high_s8(wv), vget_low_s8(vld1q_s8(xv + 8)));
                        int32x4_t z = vdupq_n_s32(0);
                        int32_t dot = hsum_s32(vpadalq_s16(z, p));
                        acc[t] += xs[t][b * 8 + k / 2] * sc * (float)dot;
                    }
                }
            }
        }
    }
    for (int t = 0; t < ntok; ++t) {
        out[t] = acc[t];
    }
}

extern "C" void lfm_q4_rows_raw(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                     const int32_t *const *sums, int blocks, int ntok, float *out) {
    if (ntok == 8) {
        q4_rows_raw_t<8>(w, xq, xs, sums, blocks, out);
    } else if (ntok == 1) {
        q4_rows_raw_t<1>(w, xq, xs, sums, blocks, out);
    } else {
        q4_rows_raw_t<2>(w, xq, xs, sums, blocks, out);
    }
}

extern "C" void lfm_q6_rows_raw(const uint8_t *w, const int8_t *const *xq, const float *const *xs,
                     const int32_t *const *sums16, int blocks, int ntok, float *out) {
    if (ntok == 8) {
        q6_rows_raw_t<8>(w, xq, xs, sums16, blocks, out);
    } else if (ntok == 1) {
        q6_rows_raw_t<1>(w, xq, xs, sums16, blocks, out);
    } else {
        q6_rows_raw_t<2>(w, xq, xs, sums16, blocks, out);
    }
}

