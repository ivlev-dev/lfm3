#ifndef LFM_QUANT_H
#define LFM_QUANT_H

#include <math.h>
#include <stdint.h>

#include "rot.h"

#define LFM_QGROUP 32
#define LFM_QZP 128

static inline int lfm_rne_f32(float v)
{
    float f = floorf(v);
    float d = v - f;
    int fi = (int)f;
    if (d > 0.5f)
        return fi + 1;
    if (d < 0.5f)
        return fi;
    return (fi & 1) ? fi + 1 : fi;
}

static inline int lfm_quant_i8_group32(const float *x, int n, int8_t *q, float *scale)
{
    int g;
    int clamped = 0;
    if (n % LFM_QGROUP)
        return -1;
    for (g = 0; g < n / LFM_QGROUP; g++) {
        const float *xg = x + (size_t)g * LFM_QGROUP;
        int8_t *qg = q + (size_t)g * LFM_QGROUP;
        float amax = 0.0f;
        float inv;
        int i;
        for (i = 0; i < LFM_QGROUP; i++) {
            float a = fabsf(xg[i]);
            if (a > amax)
                amax = a;
        }
        if (amax > 0.0f) {
            scale[g] = amax / 127.0f;
            inv = 127.0f / amax;
        } else {
            scale[g] = 0.0f;
            inv = 0.0f;
        }
        for (i = 0; i < LFM_QGROUP; i++) {
            int v = lfm_rne_f32(xg[i] * inv);
            if (v > 127) {
                v = 127;
                clamped++;
            }
            if (v < -127) {
                v = -127;
                clamped++;
            }
            qg[i] = (int8_t)v;
        }
    }
    return clamped;
}

static inline int lfm_act_i8_rot(const float *x, int k, int rot, int8_t *q, float *scale)
{
    int b;
    if (k % LFM_QGROUP)
        return -1;
    if (rot <= 0) {
        return lfm_quant_i8_group32(x, k, q, scale);
    }
    for (b = 0; b < k; b += rot) {
        float blk[128];
        int n = k - b < rot ? k - b : rot;
        int i;
        if (n > 128)
            return -1;
        for (i = 0; i < n; i++)
            blk[i] = x[b + i];
        if (n == rot && (n == 32 || n == 128))
            lfm_rot_fwd_f32(blk, n);
        if (lfm_quant_i8_group32(blk, n, q + b, scale + b / LFM_QGROUP) < 0)
            return -1;
    }
    return 0;
}

static inline void lfm_q_to_u8(const int8_t *q, uint8_t *u, int n)
{
    int i;
    for (i = 0; i < n; i++)
        u[i] = (uint8_t)((int)q[i] + LFM_QZP);
}

static inline int32_t lfm_dot_i8(const int8_t *a, const int8_t *w, int n)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < n; i++)
        s += (int32_t)a[i] * (int32_t)w[i];
    return s;
}

static inline int32_t lfm_rowsum_i8(const int8_t *w, int n)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < n; i++)
        s += (int32_t)w[i];
    return s;
}

static inline int32_t lfm_dot_u8_zp(const uint8_t *u, const int8_t *w, int n, int zp)
{
    int32_t s = 0;
    int32_t sw = 0;
    int i;
    for (i = 0; i < n; i++) {
        s += (int32_t)u[i] * (int32_t)w[i];
        sw += (int32_t)w[i];
    }
    return s - (int32_t)zp * sw;
}

static inline void lfm_deq_i8_group32(const int8_t *q, const float *scale, float *x, int n)
{
    int g;
    for (g = 0; g < n / LFM_QGROUP; g++) {
        int i;
        for (i = 0; i < LFM_QGROUP; i++)
            x[g * LFM_QGROUP + i] = (float)q[g * LFM_QGROUP + i] * scale[g];
    }
}

#endif
