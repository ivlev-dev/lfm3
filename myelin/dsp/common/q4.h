#ifndef LFM_Q4_H
#define LFM_Q4_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "fp16.h"
#include "quant.h"
#include "rot.h"

#define LFM_Q4_GROUP 32
#define LFM_Q4_ALIGN 128

static inline size_t lfm_q4_scales_bytes(int n, int k)
{
    return (size_t)n * (size_t)(k / LFM_Q4_GROUP) * 2u;
}

static inline size_t lfm_q4_planes_off(int n, int k)
{
    size_t s = lfm_q4_scales_bytes(n, k);
    return (s + (LFM_Q4_ALIGN - 1)) & ~(size_t)(LFM_Q4_ALIGN - 1);
}

static inline size_t lfm_q4_image_bytes(int n, int k)
{
    return lfm_q4_planes_off(n, k) + (size_t)n * (size_t)(k / 2);
}

static inline const uint8_t *lfm_q4_planes(const uint8_t *img, int n, int k)
{
    return img + lfm_q4_planes_off(n, k);
}

static inline const uint16_t *lfm_q4_row_scales(const uint8_t *img, int n, int k, int row)
{
    (void)n;
    return (const uint16_t *)img + (size_t)row * (size_t)(k / LFM_Q4_GROUP);
}

static inline const uint8_t *lfm_q4_row_qs(const uint8_t *img, int n, int k, int row)
{
    return lfm_q4_planes(img, n, k) + (size_t)row * (size_t)(k / 2);
}

static inline void lfm_q4_pack_row(const float *w, int k, uint8_t *qs, uint16_t *sc)
{
    int g;
    for (g = 0; g < k / LFM_Q4_GROUP; g++) {
        const float *wg = w + (size_t)g * LFM_Q4_GROUP;
        float amax = 0.0f;
        float step;
        int i;
        for (i = 0; i < LFM_Q4_GROUP; i++) {
            float a = fabsf(wg[i]);
            if (a > amax)
                amax = a;
        }
        step = amax > 0.0f ? amax / 7.0f : 0.0f;
        sc[g] = lfm_f32_to_f16(step);
        for (i = 0; i < LFM_Q4_GROUP; i += 2) {
            int q0 = 0;
            int q1 = 0;
            if (step > 0.0f) {
                float z0 = wg[i] / step;
                float z1 = wg[i + 1] / step;
                q0 = (int)(z0 >= 0.0f ? floorf(z0 + 0.5f) : ceilf(z0 - 0.5f));
                q1 = (int)(z1 >= 0.0f ? floorf(z1 + 0.5f) : ceilf(z1 - 0.5f));
                if (q0 > 7)
                    q0 = 7;
                if (q0 < -7)
                    q0 = -7;
                if (q1 > 7)
                    q1 = 7;
                if (q1 < -7)
                    q1 = -7;
            }
            qs[(size_t)g * (LFM_Q4_GROUP / 2) + i / 2] =
                (uint8_t)(((q0 + 8) & 0xF) | (((q1 + 8) & 0xF) << 4));
        }
    }
}

static inline void lfm_q4_pack_matrix(const float *w, int n, int k, uint8_t *img, int rot)
{
    uint8_t *planes = (uint8_t *)lfm_q4_planes(img, n, k);
    float *tmp = (float *)malloc((size_t)k * sizeof(float));
    int r;
    for (r = 0; r < n; r++) {
        const float *src = w + (size_t)r * k;
        uint16_t *sc = (uint16_t *)img + (size_t)r * (size_t)(k / LFM_Q4_GROUP);
        uint8_t *qs = planes + (size_t)r * (size_t)(k / 2);
        int i;
        for (i = 0; i < k; i++)
            tmp[i] = src[i];
        lfm_rot_fwd_blocks_f32(tmp, k, rot);
        lfm_q4_pack_row(tmp, k, qs, sc);
    }
    free(tmp);
}

static inline void lfm_q4_dequant_row(const uint8_t *qs, const uint16_t *sc, int k, float *out)
{
    int g;
    for (g = 0; g < k / LFM_Q4_GROUP; g++) {
        float step = lfm_f16_to_f32(sc[g]);
        int i;
        for (i = 0; i < LFM_Q4_GROUP; i += 2) {
            uint8_t b = qs[(size_t)g * (LFM_Q4_GROUP / 2) + i / 2];
            int q0 = (int)(b & 0xF) - 8;
            int q1 = (int)(b >> 4) - 8;
            out[g * LFM_Q4_GROUP + i] = (float)q0 * step;
            out[g * LFM_Q4_GROUP + i + 1] = (float)q1 * step;
        }
    }
}

static inline void lfm_q4_dot32(const uint8_t *qs, const int8_t *q, int32_t *dot, int32_t *sumq)
{
    int32_t d = 0;
    int32_t s = 0;
    int i;
    for (i = 0; i < LFM_Q4_GROUP; i += 2) {
        uint8_t b = qs[i / 2];
        int q0 = (int)(b & 0xF) - 8;
        int q1 = (int)(b >> 4) - 8;
        d += q0 * (int)q[i] + q1 * (int)q[i + 1];
        s += (int)q[i] + (int)q[i + 1];
    }
    *dot = d;
    *sumq = s;
}

static inline float lfm_q4_row_dot_i8(const uint8_t *qs, const uint16_t *sc, const int8_t *q,
                                      const float *ascale, int k)
{
    float acc = 0.0f;
    int g;
    for (g = 0; g < k / LFM_Q4_GROUP; g++) {
        int32_t dot;
        int32_t sumq;
        lfm_q4_dot32(qs + (size_t)g * (LFM_Q4_GROUP / 2), q + (size_t)g * LFM_Q4_GROUP, &dot,
                     &sumq);
        (void)sumq;
        acc += (float)dot * lfm_f16_to_f32(sc[g]) * ascale[g];
    }
    return acc;
}

static inline void lfm_q4_gemv_i8(const uint8_t *img, int n, int k, const int8_t *q,
                                  const float *ascale, float *y)
{
    int r;
    for (r = 0; r < n; r++)
        y[r] = lfm_q4_row_dot_i8(lfm_q4_row_qs(img, n, k, r), lfm_q4_row_scales(img, n, k, r), q,
                                 ascale, k);
}

static inline void lfm_q4_gemm_i8(const uint8_t *img, int n, int k, const int8_t *q, int m,
                                  const float *ascale, float *y)
{
    int r;
    int t;
    for (r = 0; r < n; r++) {
        const uint8_t *qs = lfm_q4_row_qs(img, n, k, r);
        const uint16_t *sc = lfm_q4_row_scales(img, n, k, r);
        for (t = 0; t < m; t++)
            y[(size_t)t * n + r] =
                lfm_q4_row_dot_i8(qs, sc, q + (size_t)t * k, ascale + (size_t)t * (k / 32), k);
    }
}

#endif
