#ifndef LFM_ROT_H
#define LFM_ROT_H

#include <math.h>
#include <stdint.h>

#include "fp16.h"

#define LFM_ROT_SIGN0 0x9E3779B9u
#define LFM_ROT_SIGN1 0x85EBCA6Bu
#define LFM_ROT_SIGN2 0xC2B2AE35u
#define LFM_ROT_SIGN3 0x27D4EB2Fu
#define LFM_ROT_S32 0.17677669529663687f
#define LFM_ROT_S128 0.08838834764831845f

static inline uint32_t lfm_rot_sign_word(int i)
{
    switch (i) {
    case 0: return LFM_ROT_SIGN0;
    case 1: return LFM_ROT_SIGN1;
    case 2: return LFM_ROT_SIGN2;
    default: return LFM_ROT_SIGN3;
    }
}

static inline float lfm_rot_sign(int d)
{
    return ((lfm_rot_sign_word(d >> 5) >> (d & 31u)) & 1u) ? -1.0f : 1.0f;
}

static inline void lfm_hadamard_f32(float *v, int n)
{
    int step;
    for (step = 1; step < n; step <<= 1) {
        int base;
        for (base = 0; base < n; base += 2 * step) {
            int i;
            for (i = 0; i < step; i++) {
                float a = v[base + i];
                float b = v[base + i + step];
                v[base + i] = a + b;
                v[base + i + step] = a - b;
            }
        }
    }
}

static inline float lfm_rot_scale(int n)
{
    return n == 32 ? LFM_ROT_S32 : LFM_ROT_S128;
}

static inline void lfm_rot_fwd_f32(float *v, int n)
{
    float s = lfm_rot_scale(n);
    int i;
    for (i = 0; i < n; i++)
        v[i] *= lfm_rot_sign(i);
    lfm_hadamard_f32(v, n);
    for (i = 0; i < n; i++)
        v[i] *= s;
}

static inline void lfm_rot_inv_f32(float *v, int n)
{
    float s = lfm_rot_scale(n);
    int i;
    lfm_hadamard_f32(v, n);
    for (i = 0; i < n; i++)
        v[i] *= s * lfm_rot_sign(i);
}

static inline void lfm_rot_fwd_blocks_f32(float *x, int k, int n)
{
    int b;
    if (n <= 0)
        return;
    for (b = 0; b + n <= k; b += n)
        lfm_rot_fwd_f32(x + b, n);
}

static inline void lfm_rot_inv_blocks_f32(float *x, int k, int n)
{
    int b;
    if (n <= 0)
        return;
    for (b = 0; b + n <= k; b += n)
        lfm_rot_inv_f32(x + b, n);
}

static inline void lfm_rot_fwd_f16(uint16_t *v, int n)
{
    float buf[128];
    int i;
    if (n > 128)
        return;
    for (i = 0; i < n; i++)
        buf[i] = lfm_f16_to_f32(v[i]);
    lfm_rot_fwd_f32(buf, n);
    for (i = 0; i < n; i++)
        v[i] = lfm_f32_to_f16(buf[i]);
}

static inline void lfm_rot_inv_f16(uint16_t *v, int n)
{
    float buf[128];
    int i;
    if (n > 128)
        return;
    for (i = 0; i < n; i++)
        buf[i] = lfm_f16_to_f32(v[i]);
    lfm_rot_inv_f32(buf, n);
    for (i = 0; i < n; i++)
        v[i] = lfm_f32_to_f16(buf[i]);
}

static inline void lfm_rot_fwd_page_f16(uint16_t *v, int tokens, int head_dim)
{
    int t;
    if (head_dim != 128)
        return;
    for (t = 0; t < tokens; t++)
        lfm_rot_fwd_f16(v + t * head_dim, head_dim);
}

#endif
