#ifndef LFM_KVARN_H
#define LFM_KVARN_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "fp16.h"
#include "rot.h"

#define LFM_KVARN_PAGE_TOKENS 32
#define LFM_KVARN_HEAD_DIM 128
#define LFM_KVARN_KCODE_BYTES (LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM / 2)
#define LFM_KVARN_KA_BYTES (LFM_KVARN_HEAD_DIM * 2)
#define LFM_KVARN_KC_BYTES (LFM_KVARN_PAGE_TOKENS * 2)
#define LFM_KVARN_VB_BYTES (LFM_KVARN_PAGE_TOKENS * 2)
#define LFM_KVARN_VBZ_BYTES (LFM_KVARN_PAGE_TOKENS * 2)
#define LFM_KVARN_VC_BYTES (LFM_KVARN_HEAD_DIM * 2)
#define LFM_KVARN_VAR_FLOOR 1e-8f
#define LFM_KVARN_LOG_MIN (-0.3f)
#define LFM_KVARN_LOG_MAX 10.0f
#define LFM_KVARN_ITERS 3

typedef enum {
    LFM_KV_F16 = 0,
    LFM_KVARN_K4V2 = 1,
    LFM_KVARN_K4V4 = 2
} lfm_kv_format;

static inline int lfm_kvarn_v_bits(lfm_kv_format fmt)
{
    return fmt == LFM_KVARN_K4V2 ? 2 : 4;
}

static inline size_t lfm_kvarn_head_bytes(lfm_kv_format fmt)
{
    size_t vcode = (size_t)LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM * (size_t)lfm_kvarn_v_bits(fmt) / 8u;
    return (size_t)LFM_KVARN_KCODE_BYTES + LFM_KVARN_KA_BYTES + LFM_KVARN_KC_BYTES + vcode +
           LFM_KVARN_VB_BYTES + LFM_KVARN_VBZ_BYTES + LFM_KVARN_VC_BYTES;
}

static inline size_t lfm_kvarn_page_bytes(lfm_kv_format fmt, int heads)
{
    return lfm_kvarn_head_bytes(fmt) * (size_t)heads;
}

static inline size_t lfm_kvarn_vcode_off(lfm_kv_format fmt)
{
    (void)fmt;
    return (size_t)LFM_KVARN_KCODE_BYTES + LFM_KVARN_KA_BYTES + LFM_KVARN_KC_BYTES;
}

static inline size_t lfm_kvarn_vcode_bytes(lfm_kv_format fmt)
{
    return (size_t)LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM * (size_t)lfm_kvarn_v_bits(fmt) / 8u;
}

static inline int lfm_kvarn_round_i(float x)
{
    return (int)roundf(x);
}

static inline float lfm_kvarn_token_var(const float *t, int token, float ct, const float *ir)
{
    float sum = 0.0f;
    int ch;
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
        float x = t[token * LFM_KVARN_HEAD_DIM + ch] * ct * ir[ch];
        sum += x * x;
    }
    return sum / (float)LFM_KVARN_HEAD_DIM;
}

static inline float lfm_kvarn_channel_var(const float *t, int ch, const float *ic, float r)
{
    float sum = 0.0f;
    int token;
    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
        float x = t[token * LFM_KVARN_HEAD_DIM + ch] * ic[token] * r;
        sum += x * x;
    }
    return sum / (float)LFM_KVARN_PAGE_TOKENS;
}

static inline void lfm_kvarn_scales(const float *lr, const float *lc, float *ir, float *ic)
{
    int i;
    for (i = 0; i < LFM_KVARN_HEAD_DIM; i++)
        ir[i] = expf(-lr[i]);
    for (i = 0; i < LFM_KVARN_PAGE_TOKENS; i++)
        ic[i] = expf(-lc[i]);
}

static inline void lfm_kvarn_commit_work(lfm_kv_format fmt, const uint16_t *src_f16, uint8_t *out,
                                         float *t)
{
    float lr[LFM_KVARN_HEAD_DIM];
    float lc[LFM_KVARN_PAGE_TOKENS];
    float ir[LFM_KVARN_HEAD_DIM];
    float ic[LFM_KVARN_PAGE_TOKENS];
    int token;
    int ch;
    int it;
    int vmax = (1 << lfm_kvarn_v_bits(fmt)) - 1;
    size_t voff = lfm_kvarn_vcode_off(fmt);
    size_t vbytes = lfm_kvarn_vcode_bytes(fmt);
    uint8_t *ka = out + LFM_KVARN_KCODE_BYTES;
    uint8_t *kc = ka + LFM_KVARN_KA_BYTES;
    uint8_t *vcode = out + voff;
    uint8_t *vb = out + voff + vbytes;
    uint8_t *vbz = vb + LFM_KVARN_VB_BYTES;
    uint8_t *vc = vbz + LFM_KVARN_VBZ_BYTES;

    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
        for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++)
            t[token * LFM_KVARN_HEAD_DIM + ch] =
                lfm_f16_to_f32(src_f16[token * LFM_KVARN_HEAD_DIM + ch]);
    }
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++)
        lr[ch] = 0.0f;
    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++)
        lc[token] = 0.0f;

    for (it = 0; it < LFM_KVARN_ITERS; it++) {
        lfm_kvarn_scales(lr, lc, ir, ic);
        for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
            float v = lfm_kvarn_token_var(t, token, ic[token], ir);
            float l;
            if (v < LFM_KVARN_VAR_FLOOR)
                v = LFM_KVARN_VAR_FLOOR;
            l = lc[token] + 0.5f * logf(v);
            if (l < LFM_KVARN_LOG_MIN)
                l = LFM_KVARN_LOG_MIN;
            if (l > LFM_KVARN_LOG_MAX)
                l = LFM_KVARN_LOG_MAX;
            lc[token] = l;
        }
        lfm_kvarn_scales(lr, lc, ir, ic);
        for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
            float v = lfm_kvarn_channel_var(t, ch, ic, ir[ch]);
            float l;
            if (v < LFM_KVARN_VAR_FLOOR)
                v = LFM_KVARN_VAR_FLOOR;
            l = lr[ch] + 0.5f * logf(v);
            if (l < LFM_KVARN_LOG_MIN)
                l = LFM_KVARN_LOG_MIN;
            if (l > LFM_KVARN_LOG_MAX)
                l = LFM_KVARN_LOG_MAX;
            lr[ch] = l;
        }
    }
    lfm_kvarn_scales(lr, lc, ir, ic);

    {
        float amp[LFM_KVARN_HEAD_DIM];
        for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
            float mx = LFM_KVARN_VAR_FLOOR;
            for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
                float a = fabsf(t[token * LFM_KVARN_HEAD_DIM + ch] * ic[token] * ir[ch]);
                if (a > mx)
                    mx = a;
            }
            amp[ch] = mx / 7.0f;
            lfm_f16_store(ka + ch * 2, lfm_f32_to_f16(amp[ch] * expf(lr[ch])));
        }
        for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++)
            lfm_f16_store(kc + token * 2, lfm_f32_to_f16(expf(lc[token])));
        for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
            int lane;
            for (lane = 0; lane < LFM_KVARN_HEAD_DIM / 2; lane++) {
                int c0 = 2 * lane;
                int q0 = lfm_kvarn_round_i(t[token * LFM_KVARN_HEAD_DIM + c0] * ic[token] * ir[c0] /
                                           amp[c0]);
                int q1 = lfm_kvarn_round_i(t[token * LFM_KVARN_HEAD_DIM + c0 + 1] * ic[token] *
                                           ir[c0 + 1] / amp[c0 + 1]);
                if (q0 < -7)
                    q0 = -7;
                if (q0 > 7)
                    q0 = 7;
                if (q1 < -7)
                    q1 = -7;
                if (q1 > 7)
                    q1 = 7;
                out[token * (LFM_KVARN_HEAD_DIM / 2) + lane] =
                    (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }

    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
        float mn = INFINITY;
        float mx = -INFINITY;
        float step;
        int lane;
        for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
            float x = t[token * LFM_KVARN_HEAD_DIM + ch] * ic[token] * ir[ch];
            if (x < mn)
                mn = x;
            if (x > mx)
                mx = x;
        }
        step = (mx - mn) / (float)vmax;
        if (step < LFM_KVARN_VAR_FLOOR)
            step = LFM_KVARN_VAR_FLOOR;
        lfm_f16_store(vb + token * 2, lfm_f32_to_f16(step * expf(lc[token])));
        lfm_f16_store(vbz + token * 2, lfm_f32_to_f16(mn * expf(lc[token])));
        if (fmt == LFM_KVARN_K4V4) {
            for (lane = 0; lane < LFM_KVARN_HEAD_DIM / 2; lane++) {
                int c0 = 2 * lane;
                int q0 = lfm_kvarn_round_i((t[token * LFM_KVARN_HEAD_DIM + c0] * ic[token] *
                                            ir[c0] - mn) /
                                           step);
                int q1 = lfm_kvarn_round_i((t[token * LFM_KVARN_HEAD_DIM + c0 + 1] * ic[token] *
                                            ir[c0 + 1] - mn) /
                                           step);
                if (q0 < 0)
                    q0 = 0;
                if (q0 > vmax)
                    q0 = vmax;
                if (q1 < 0)
                    q1 = 0;
                if (q1 > vmax)
                    q1 = vmax;
                vcode[token * (LFM_KVARN_HEAD_DIM / 2) + lane] = (uint8_t)(q0 | (q1 << 4));
            }
        } else {
            int byte;
            for (byte = 0; byte < LFM_KVARN_HEAD_DIM / 4; byte++) {
                uint8_t value = 0;
                int k;
                for (k = 0; k < 4; k++) {
                    int c = byte * 4 + k;
                    int q = lfm_kvarn_round_i((t[token * LFM_KVARN_HEAD_DIM + c] * ic[token] *
                                               ir[c] - mn) /
                                              step);
                    if (q < 0)
                        q = 0;
                    if (q > 3)
                        q = 3;
                    value = (uint8_t)(value | (q << (2 * k)));
                }
                vcode[token * (LFM_KVARN_HEAD_DIM / 4) + byte] = value;
            }
        }
    }
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++)
        lfm_f16_store(vc + ch * 2, lfm_f32_to_f16(expf(lr[ch])));
}

static inline void lfm_kvarn_commit(lfm_kv_format fmt, const uint16_t *src_f16, uint8_t *out)
{
    float *t = (float *)malloc(sizeof(float) * LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM);
    if (t == NULL)
        return;
    lfm_kvarn_commit_work(fmt, src_f16, out, t);
    free(t);
}

static inline void lfm_kvarn_decode_k(const uint8_t *page, int token, float *out)
{
    const uint8_t *kcode = page;
    const uint8_t *ka = page + LFM_KVARN_KCODE_BYTES;
    const uint8_t *kc = ka + LFM_KVARN_KA_BYTES;
    float ct = lfm_f16_to_f32(lfm_f16_load(kc + token * 2));
    int ch;
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
        uint8_t b = kcode[token * (LFM_KVARN_HEAD_DIM / 2) + ch / 2];
        int q = (ch % 2 == 0) ? (int)(b & 0xF) : (int)(b >> 4);
        q -= 8;
        out[ch] = (float)q * lfm_f16_to_f32(lfm_f16_load(ka + ch * 2)) * ct;
    }
}

static inline void lfm_kvarn_decode_v(lfm_kv_format fmt, const uint8_t *page, int token,
                                      float *out)
{
    size_t voff = lfm_kvarn_vcode_off(fmt);
    size_t vbytes = lfm_kvarn_vcode_bytes(fmt);
    const uint8_t *vcode = page + voff;
    const uint8_t *vb = page + voff + vbytes;
    const uint8_t *vbz = vb + LFM_KVARN_VB_BYTES;
    const uint8_t *vc = vbz + LFM_KVARN_VBZ_BYTES;
    float bt = lfm_f16_to_f32(lfm_f16_load(vb + token * 2));
    float zero = lfm_f16_to_f32(lfm_f16_load(vbz + token * 2));
    int ch;
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
        int q;
        if (fmt == LFM_KVARN_K4V4) {
            uint8_t b = vcode[token * (LFM_KVARN_HEAD_DIM / 2) + ch / 2];
            q = (ch % 2 == 0) ? (int)(b & 0xF) : (int)(b >> 4);
        } else {
            uint8_t b = vcode[token * (LFM_KVARN_HEAD_DIM / 4) + ch / 4];
            q = (int)((b >> (2 * (ch % 4))) & 3);
        }
        out[ch] = ((float)q * bt + zero) * lfm_f16_to_f32(lfm_f16_load(vc + ch * 2));
    }
}

static inline void lfm_kvarn_decode_page(lfm_kv_format fmt, const uint8_t *page, float *k,
                                         float *v)
{
    int token;
    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
        lfm_kvarn_decode_k(page, token, k + token * LFM_KVARN_HEAD_DIM);
        lfm_kvarn_decode_v(fmt, page, token, v + token * LFM_KVARN_HEAD_DIM);
    }
}

#endif
