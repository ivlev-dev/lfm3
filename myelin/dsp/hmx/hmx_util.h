#ifndef LFM_HMX_UTIL_H
#define LFM_HMX_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../common/fp16.h"

#if defined(__HEXAGON__)
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include "qurt_hmx.h"
#endif

#define LFM_HMX_KC 32
#define LFM_HMX_N 32
#define LFM_HMX_SPATIAL 64
#define LFM_HMX_ACT_BYTES 2048u
#define LFM_HMX_WN_BYTES 512u
#define LFM_HMX_WB_BYTES 1024u
#define LFM_HMX_OUT_BYTES 2048u
#define LFM_HMX_BIAS_BYTES 256u
#define LFM_HMX_ALIGN 2048u
#define LFM_HMX_QZP 128
#define LFM_HMX_BIAS_SCALE_IDENTITY 0x6000u
#define LFM_HMX_BIAS_SCALE_DIV 512.0f
#define LFM_HMX_ACT_CTRL 0x3FFu
#define LFM_HMX_WN_CTRL 511u
#define LFM_HMX_WB_CTRL 1023u
#define LFM_HMX_STORE_CTRL 0u

typedef struct {
    uint32_t word0[LFM_HMX_N];
    uint32_t word1[LFM_HMX_N];
} lfm_hmx_bias_block;

typedef struct {
    int32_t v[LFM_HMX_SPATIAL][LFM_HMX_N];
} lfm_hmx_acc;

typedef struct {
    uint8_t *base;
    size_t bytes;
    unsigned ctx;
} lfm_hmx_vtcm;

_Static_assert(sizeof(lfm_hmx_bias_block) == LFM_HMX_BIAS_BYTES, "bias block size");
_Static_assert(sizeof(lfm_hmx_acc) == LFM_HMX_SPATIAL * LFM_HMX_N * 4u, "acc size");

static inline size_t lfm_hmx_align_up(size_t v, size_t a)
{
    return (v + a - 1u) & ~(a - 1u);
}

static inline int lfm_hmx_is_aligned(const void *p, size_t a)
{
    return ((uintptr_t)p & (uintptr_t)(a - 1u)) == 0u;
}

static inline size_t lfm_hmx_act_off(int spatial, int k)
{
    return (size_t)LFM_HMX_ACT_BYTES * (size_t)(k >> 5) + 128u * (size_t)(spatial >> 2) +
           4u * (size_t)(k & 31) + (size_t)(spatial & 3);
}

static inline size_t lfm_hmx_act_off_local(int spatial, int kk)
{
    return 128u * (size_t)(spatial >> 2) + 4u * (size_t)kk + (size_t)(spatial & 3);
}

static inline size_t lfm_hmx_wn_off(int k, int n)
{
    return (size_t)LFM_HMX_WN_BYTES * (size_t)(k >> 5) + 128u * (size_t)((k & 31) >> 3) +
           4u * (size_t)n + (size_t)(k & 3);
}

static inline size_t lfm_hmx_wn_off_local(int kk, int n)
{
    return 128u * (size_t)(kk >> 3) + 4u * (size_t)n + (size_t)(kk & 3);
}

static inline size_t lfm_hmx_wb_off(int k, int n)
{
    return (size_t)LFM_HMX_WB_BYTES * (size_t)(k >> 5) + 128u * (size_t)((k & 31) >> 2) +
           4u * (size_t)n + (size_t)(k & 3);
}

static inline size_t lfm_hmx_wb_off_local(int kk, int n)
{
    return 128u * (size_t)(kk >> 2) + 4u * (size_t)n + (size_t)(kk & 3);
}

static inline size_t lfm_hmx_out_off(int spatial, int n)
{
    return 128u * (size_t)(spatial >> 2) + 4u * (size_t)n + (size_t)(spatial & 3);
}

static inline uint8_t lfm_hmx_wn_nibble(const uint8_t *tile, int kk, int n)
{
    uint8_t b = tile[lfm_hmx_wn_off_local(kk, n)];
    return (uint8_t)(((kk >> 2) & 1) ? (b >> 4) : (b & 0x0Fu));
}

static inline void lfm_hmx_wn_put(uint8_t *tile, int kk, int n, uint8_t v)
{
    size_t off = lfm_hmx_wn_off_local(kk, n);
    if ((kk >> 2) & 1)
        tile[off] = (uint8_t)((tile[off] & 0x0Fu) | (uint8_t)((v & 0x0Fu) << 4));
    else
        tile[off] = (uint8_t)((tile[off] & 0xF0u) | (uint8_t)(v & 0x0Fu));
}

static inline int lfm_hmx_nibble_signed(uint8_t v)
{
    return (v & 8u) ? (int)v - 16 : (int)v;
}

static inline void lfm_hmx_bias_identity(lfm_hmx_bias_block *b, int cols)
{
    int n;
    for (n = 0; n < cols; n++) {
        b->word0[n] = LFM_HMX_BIAS_SCALE_IDENTITY;
        b->word1[n] = 0u;
    }
}

static inline void lfm_hmx_bias_zero_point(lfm_hmx_bias_block *b, const int32_t *sum_w, int cols)
{
    int n;
    for (n = 0; n < cols; n++) {
        b->word0[n] = LFM_HMX_BIAS_SCALE_IDENTITY;
        b->word1[n] = (uint32_t)(int32_t)(-LFM_HMX_QZP * sum_w[n]);
    }
}

static inline float lfm_hmx_bias_scale(const lfm_hmx_bias_block *b, int n)
{
    return lfm_f16_to_f32((uint16_t)(b->word0[n] & 0xFFFFu)) / LFM_HMX_BIAS_SCALE_DIV;
}

static inline void lfm_hmx_acc_clear(lfm_hmx_acc *a, int rows)
{
    int s;
    for (s = 0; s < rows; s++)
        memset(a->v[s], 0, sizeof(a->v[s]));
}

static inline void lfm_hmx_acc_bias(lfm_hmx_acc *a, const lfm_hmx_bias_block *b, int rows)
{
    int s;
    int n;
    for (s = 0; s < rows; s++) {
        for (n = 0; n < LFM_HMX_N; n++) {
            int64_t v = (int64_t)a->v[s][n] + (int64_t)(int32_t)b->word1[n];
            if (v > INT32_MAX)
                v = INT32_MAX;
            if (v < INT32_MIN)
                v = INT32_MIN;
            a->v[s][n] = (int32_t)v;
        }
    }
}

static inline void lfm_hmx_store_u8(const lfm_hmx_acc *a, const lfm_hmx_bias_block *b, int rows,
                                    uint8_t *out)
{
    int n;
    for (n = 0; n < LFM_HMX_N; n++) {
        float sc = lfm_hmx_bias_scale(b, n);
        int s;
        for (s = 0; s < rows; s++) {
            int64_t v = (int64_t)a->v[s][n] + (int64_t)(int32_t)b->word1[n];
            int64_t q = (int64_t)((double)v * (double)sc);
            if (q < 0)
                q = 0;
            if (q > 255)
                q = 255;
            out[lfm_hmx_out_off(s, n)] = (uint8_t)q;
        }
    }
}

static inline void lfm_hmx_pack_act_ub(uint8_t *dst, const int8_t *q, int ld, int rows, int kc_count)
{
    int c;
    if (rows > LFM_HMX_SPATIAL)
        rows = LFM_HMX_SPATIAL;
    memset(dst, 0, (size_t)kc_count * LFM_HMX_ACT_BYTES);
    for (c = 0; c < kc_count; c++) {
        int s;
        for (s = 0; s < rows; s++) {
            int kk;
            for (kk = 0; kk < 32; kk++) {
                dst[(size_t)c * LFM_HMX_ACT_BYTES + lfm_hmx_act_off_local(s, kk)] =
                    (uint8_t)((int)q[(size_t)s * (size_t)ld + (size_t)c * 32u + (size_t)kk] +
                              LFM_HMX_QZP);
            }
        }
    }
}

static inline void lfm_hmx_pack_act_crouton(uint8_t *dst, const int8_t *q, int ld, int rows, int kc)
{
    int s;
    if (rows > LFM_HMX_SPATIAL)
        rows = LFM_HMX_SPATIAL;
    memset(dst, 0, LFM_HMX_ACT_BYTES);
    for (s = 0; s < rows; s++) {
        int kk;
        for (kk = 0; kk < 32; kk++) {
            dst[lfm_hmx_act_off_local(s, kk)] =
                (uint8_t)((int)q[(size_t)s * (size_t)ld + (size_t)kc * 32u + (size_t)kk] +
                          LFM_HMX_QZP);
        }
    }
}

static inline void lfm_hmx_pack_wn_tile(uint8_t *dst, const uint8_t *planes, int plane_ld,
                                        int row0, int kc, int cols)
{
    int c;
    memset(dst, 0, LFM_HMX_WN_BYTES);
    if (cols > 32)
        cols = 32;
    for (c = 0; c < cols; c++) {
        const uint8_t *src = planes + (size_t)(row0 + c) * (size_t)plane_ld + (size_t)kc * 16u;
        int kk;
        for (kk = 0; kk < 32; kk++) {
            uint8_t b = src[kk >> 1];
            uint8_t v = (uint8_t)((kk & 1) ? (b >> 4) : (b & 0x0Fu));
            lfm_hmx_wn_put(dst, kk, c, (uint8_t)(v ^ 0x08u));
        }
    }
}

static inline void lfm_hmx_pack_wb_tile(uint8_t *dst, const int8_t *w, int ld, int row0, int kc,
                                        int cols)
{
    int c;
    memset(dst, 0, LFM_HMX_WB_BYTES);
    if (cols > 32)
        cols = 32;
    for (c = 0; c < cols; c++) {
        const int8_t *src = w + (size_t)(row0 + c) * (size_t)ld + (size_t)kc * 32u;
        int kk;
        for (kk = 0; kk < 32; kk++)
            dst[lfm_hmx_wb_off_local(kk, c)] = (uint8_t)src[kk];
    }
}

static inline int32_t lfm_hmx_sum_wn_tile(const uint8_t *tile, int cols)
{
    int32_t sum = 0;
    int c;
    for (c = 0; c < cols; c++) {
        int kk;
        for (kk = 0; kk < 32; kk++)
            sum += (int32_t)lfm_hmx_nibble_signed(lfm_hmx_wn_nibble(tile, kk, c));
    }
    return sum;
}

static inline int32_t lfm_hmx_sum_wn_tile_n(const uint8_t *tile, int n)
{
    int32_t sum = 0;
    int kk;
    for (kk = 0; kk < 32; kk++)
        sum += (int32_t)lfm_hmx_nibble_signed(lfm_hmx_wn_nibble(tile, kk, n));
    return sum;
}

static inline int32_t lfm_hmx_sum_wb_tile_n(const uint8_t *tile, int n)
{
    int32_t sum = 0;
    int kk;
    for (kk = 0; kk < 32; kk++)
        sum += (int32_t)(int8_t)tile[lfm_hmx_wb_off_local(kk, n)];
    return sum;
}

static inline void lfm_hmx_mac_i8(lfm_hmx_acc *a, const uint8_t *wb, const uint8_t *ub, int rows)
{
    int kk;
    int n;
    for (kk = 0; kk < 32; kk++) {
        for (n = 0; n < LFM_HMX_N; n++) {
            int wv = (int)(int8_t)wb[lfm_hmx_wb_off_local(kk, n)];
            int s;
            for (s = 0; s < rows; s++)
                a->v[s][n] += wv * (int)ub[lfm_hmx_act_off_local(s, kk)];
        }
    }
}

static inline void lfm_hmx_mac_i4(lfm_hmx_acc *a, const uint8_t *wn, const uint8_t *ub, int rows)
{
    int kk;
    int n;
    for (kk = 0; kk < 32; kk++) {
        for (n = 0; n < LFM_HMX_N; n++) {
            int wv = lfm_hmx_nibble_signed(lfm_hmx_wn_nibble(wn, kk, n));
            int s;
            for (s = 0; s < rows; s++)
                a->v[s][n] += wv * (int)ub[lfm_hmx_act_off_local(s, kk)];
        }
    }
}

static inline int lfm_hmx_vtcm_alloc(lfm_hmx_vtcm *v, size_t bytes)
{
    v->base = NULL;
    v->bytes = 0;
    v->ctx = 0;
#if defined(__HEXAGON__)
    {
        compute_res_attr_t attr;
        unsigned ctx;
        void *p;
        if (!compute_resource_attr_init || compute_resource_attr_init(&attr) != 0)
            return -1;
        if (compute_resource_attr_set_hmx_param &&
            compute_resource_attr_set_hmx_param(&attr, 1) != 0)
            return -2;
        if (compute_resource_attr_set_vtcm_param &&
            compute_resource_attr_set_vtcm_param(&attr, (unsigned)bytes, 0) != 0)
            return -3;
        ctx = HAP_compute_res_acquire(&attr, 1000000u);
        if (ctx == 0u)
            return -4;
        p = HAP_compute_res_attr_get_vtcm_ptr(&attr);
        if (p == NULL) {
            HAP_compute_res_release(ctx);
            return -5;
        }
        v->base = (uint8_t *)p;
        v->bytes = bytes;
        v->ctx = ctx;
    }
#else
    {
        void *p = NULL;
        size_t sz = lfm_hmx_align_up(bytes, (size_t)LFM_HMX_ALIGN);
        if (sz == 0u)
            return -1;
        p = aligned_alloc((size_t)LFM_HMX_ALIGN, sz);
        if (p == NULL)
            return -2;
        memset(p, 0, sz);
        v->base = (uint8_t *)p;
        v->bytes = sz;
    }
#endif
    return 0;
}

static inline void lfm_hmx_vtcm_free(lfm_hmx_vtcm *v)
{
#if defined(__HEXAGON__)
    if (v->ctx != 0u)
        HAP_compute_res_release(v->ctx);
#else
    free(v->base);
#endif
    v->base = NULL;
    v->bytes = 0;
    v->ctx = 0;
}

static inline int lfm_hmx_power(int up)
{
#if defined(__HEXAGON__)
    HAP_power_request_t req;
    memset(&req, 0, sizeof req);
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = up ? TRUE : FALSE;
    return HAP_power_set(0, &req);
#else
    (void)up;
    return 0;
#endif
}

static inline int lfm_hmx_lock(void)
{
#if defined(__HEXAGON__)
    return qurt_hmx_lock2(QURT_HMX_SHARED_LOCK);
#else
    return 0;
#endif
}

static inline int lfm_hmx_unlock(void)
{
#if defined(__HEXAGON__)
    return qurt_hmx_unlock2(QURT_HMX_SHARED_UNLOCK);
#else
    return 0;
#endif
}

static inline void lfm_hmx_sync(void)
{
#if defined(__HEXAGON__)
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
#endif
}

#if defined(__HEXAGON__)
static inline void lfm_hmx_hw_clear(void)
{
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
}

static inline void lfm_hmx_hw_bias_load(const lfm_hmx_bias_block *b)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
                         :
                         : "r"((uint32_t)(uintptr_t)b)
                         : "memory");
}

static inline void lfm_hmx_hw_pair_i8(const uint8_t *act, const uint8_t *wgt)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3)\n\t}"
                         :
                         : "r"((uint32_t)(uintptr_t)act), "r"(LFM_HMX_ACT_CTRL),
                           "r"((uint32_t)(uintptr_t)wgt), "r"(LFM_HMX_WB_CTRL)
                         : "memory");
}

static inline void lfm_hmx_hw_pair_i4(const uint8_t *act, const uint8_t *wgt)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
                         :
                         : "r"((uint32_t)(uintptr_t)act), "r"(LFM_HMX_ACT_CTRL),
                           "r"((uint32_t)(uintptr_t)wgt), "r"(LFM_HMX_WN_CTRL)
                         : "memory");
}

static inline void lfm_hmx_hw_store_ub(uint8_t *out)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :
                         : "r"((uint32_t)(uintptr_t)out), "r"(LFM_HMX_STORE_CTRL)
                         : "memory");
}
#endif

static inline int lfm_hmx_tile_u8(const uint8_t *act_ub, const uint8_t *wgt, int is_i4, int kc,
                                  const lfm_hmx_bias_block *bias, uint8_t *out, int rows)
{
#if defined(__HEXAGON__)
    int c;
    (void)rows;
    lfm_hmx_hw_clear();
    for (c = 0; c < kc; c++) {
        if (is_i4)
            lfm_hmx_hw_pair_i4(act_ub + (size_t)c * LFM_HMX_ACT_BYTES,
                               wgt + (size_t)c * LFM_HMX_WN_BYTES);
        else
            lfm_hmx_hw_pair_i8(act_ub + (size_t)c * LFM_HMX_ACT_BYTES,
                               wgt + (size_t)c * LFM_HMX_WB_BYTES);
    }
    lfm_hmx_hw_bias_load(bias);
    lfm_hmx_hw_store_ub(out);
    lfm_hmx_sync();
    return 0;
#else
    lfm_hmx_acc acc;
    int c;
    if (rows > LFM_HMX_SPATIAL)
        rows = LFM_HMX_SPATIAL;
    lfm_hmx_acc_clear(&acc, rows);
    for (c = 0; c < kc; c++) {
        if (is_i4)
            lfm_hmx_mac_i4(&acc, wgt + (size_t)c * LFM_HMX_WN_BYTES,
                           act_ub + (size_t)c * LFM_HMX_ACT_BYTES, rows);
        else
            lfm_hmx_mac_i8(&acc, wgt + (size_t)c * LFM_HMX_WB_BYTES,
                           act_ub + (size_t)c * LFM_HMX_ACT_BYTES, rows);
    }
    lfm_hmx_store_u8(&acc, bias, rows, out);
    return 0;
#endif
}

#endif
