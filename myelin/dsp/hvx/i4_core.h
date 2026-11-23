#ifndef LFM_I4_CORE_H
#define LFM_I4_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../common/fp16.h"
#include "../common/q4.h"

#define LFM_I4_MAX_M 8

#if defined(__HVX__)
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#endif

static inline void lfm_i4_group_dot(const uint8_t *qs, const int8_t *q, int32_t *out)
{
    int32_t d = 0;
    int i;
    for (i = 0; i < LFM_Q4_GROUP; i += 2) {
        uint8_t b = qs[(size_t)(i >> 1)];
        int q0 = (int)(b & 0x0Fu) - 8;
        int q1 = (int)(b >> 4) - 8;
        d += q0 * (int)q[i];
        d += q1 * (int)q[i + 1];
    }
    *out = d;
}

static inline void lfm_i4_rows_scalar(const uint8_t *img, int n, int k, const int8_t *q, int m,
                                      const float *ascale, float *y)
{
    int groups = k / LFM_Q4_GROUP;
    int r;
    int t;
    int g;
    for (r = 0; r < n; r++) {
        const uint8_t *qs = lfm_q4_row_qs(img, n, k, r);
        const uint16_t *sc = lfm_q4_row_scales(img, n, k, r);
        for (t = 0; t < m; t++) {
            float acc = 0.0f;
            for (g = 0; g < groups; g++) {
                int32_t dot;
                lfm_i4_group_dot(qs + (size_t)g * 16u,
                                 q + (size_t)t * (size_t)k + (size_t)g * 32u, &dot);
                acc += (float)dot * lfm_f16_to_f32(sc[g]) *
                       ascale[(size_t)t * (size_t)groups + (size_t)g];
            }
            y[(size_t)t * (size_t)n + (size_t)r] = acc;
        }
    }
}

#if defined(__HVX__)

static inline HVX_Vector lfm_i4_loadu(const uint8_t *p)
{
    uintptr_t addr = (uintptr_t)p;
    unsigned int off = (unsigned int)(addr & (uintptr_t)127u);
    const HVX_Vector *base = (const HVX_Vector *)(addr & ~(uintptr_t)127u);
    if (off == 0u)
        return *base;
    return Q6_V_valign_VVR(*(base + 1), *base, (int)off);
}

static inline float lfm_i4_scale_f16(uint16_t h)
{
    uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    if (e - 1u < 30u) {
        uint32_t x = ((uint32_t)(h & 0x8000u) << 16) | ((e + 112u) << 23) |
                     ((uint32_t)(h & 0x3FFu) << 13);
        float f;
        memcpy(&f, &x, sizeof f);
        return f;
    }
    return lfm_f16_to_f32(h);
}

static inline void lfm_i4_prefetch(const uint8_t *p, int bytes)
{
    unsigned int lines = (unsigned int)(bytes / 128);
    if (lines == 0u)
        return;
    if (lines > 16u)
        lines = 16u;
    Q6_l2fetch_AR((void *)(uintptr_t)p, (int)((lines << 16) | 128u));
}

static inline void lfm_i4_rows_hvx(const uint8_t *img, int n, int k, const int8_t *q, int m,
                                   const float *ascale, float *y)
{
    int groups = k / LFM_Q4_GROUP;
    int pairs = k / 256;
    HVX_Vector ones = Q6_V_vsplat_R(-1);
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector mask_lo = Q6_V_valign_VVR(zero, ones, 64);
    HVX_Vector mask_hi = Q6_V_valign_VVR(ones, zero, 64);
    HVX_Vector mask_nib = Q6_V_vsplat_R(0x0F0F0F0Fu);
    HVX_Vector mask_eight = Q6_V_vsplat_R(0x08080808u);
    int r;
    for (r = 0; r < n; r++) {
        const uint8_t *qs = lfm_q4_row_qs(img, n, k, r);
        const uint16_t *sc = lfm_q4_row_scales(img, n, k, r);
        float acc[LFM_I4_MAX_M];
        int ci;
        int t;
        for (t = 0; t < m; t++)
            acc[t] = 0.0f;
        if (r + 1 < n)
            lfm_i4_prefetch(lfm_q4_row_qs(img, n, k, r + 1), k / 2);
        for (ci = 0; ci < pairs; ci++) {
            HVX_Vector w = lfm_i4_loadu(qs + (size_t)ci * 128u);
            HVX_Vector lo = Q6_V_vand_VV(w, mask_nib);
            HVX_Vector hi = Q6_Vub_vlsr_VubR(w, 4);
            HVX_Vector los = Q6_Vb_vsub_VbVb(lo, mask_eight);
            HVX_Vector his = Q6_Vb_vsub_VbVb(hi, mask_eight);
            for (t = 0; t < m; t++) {
                const int8_t *qt = q + (size_t)t * (size_t)k + (size_t)ci * 256u;
                HVX_Vector va = lfm_i4_loadu((const uint8_t *)qt);
                HVX_Vector vb = lfm_i4_loadu((const uint8_t *)(qt + 128));
                HVX_Vector da = Q6_Vb_vdeal_Vb(va);
                HVX_Vector db = Q6_Vb_vdeal_Vb(vb);
                HVX_Vector rot = Q6_V_vror_VR(db, 64);
                HVX_Vector even = Q6_V_vor_VV(Q6_V_vand_VV(da, mask_lo),
                                              Q6_V_vand_VV(rot, mask_hi));
                HVX_Vector odd = Q6_V_vror_VR(Q6_V_vor_VV(Q6_V_vand_VV(da, mask_hi),
                                                          Q6_V_vand_VV(rot, mask_lo)),
                                              64);
                HVX_Vector s = Q6_Vw_vrmpyacc_VwVbVb(zero, los, even);
                HVX_Vector p = Q6_Vw_vrmpyacc_VwVbVb(s, his, odd);
                const uint16_t *scb = sc + (size_t)ci * 8u;
                const float *asb = ascale + (size_t)t * (size_t)groups + (size_t)ci * 8u;
                float a0 = acc[t];
                int j;
                p = Q6_Vw_vadd_VwVw(p, Q6_V_vror_VR(p, 4));
                p = Q6_Vw_vadd_VwVw(p, Q6_V_vror_VR(p, 8));
                for (j = 0; j < 8; j++) {
                    int32_t dot = Q6_R_vextract_VR(p, 16 * j);
                    a0 += (float)dot * lfm_i4_scale_f16(scb[j]) * asb[j];
                }
                acc[t] = a0;
            }
        }
        for (t = 0; t < m; t++) {
            float a0 = acc[t];
            int g;
            for (g = pairs * 8; g < groups; g++) {
                int32_t dot;
                lfm_i4_group_dot(qs + (size_t)g * 16u,
                                 q + (size_t)t * (size_t)k + (size_t)g * 32u, &dot);
                a0 += (float)dot * lfm_i4_scale_f16(sc[g]) *
                      ascale[(size_t)t * (size_t)groups + (size_t)g];
            }
            y[(size_t)t * (size_t)n + (size_t)r] = a0;
        }
    }
}

#endif

static inline void lfm_i4_rows(const uint8_t *img, int n, int k, const int8_t *q, int m,
                               const float *ascale, float *y)
{
    if (n <= 0 || k <= 0 || m <= 0 || (k % LFM_Q4_GROUP) != 0)
        return;
#if defined(__HVX__)
    if (m <= LFM_I4_MAX_M) {
        lfm_i4_rows_hvx(img, n, k, q, m, ascale, y);
        return;
    }
#endif
    lfm_i4_rows_scalar(img, n, k, q, m, ascale, y);
}

#endif
