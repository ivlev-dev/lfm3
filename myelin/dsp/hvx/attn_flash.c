/*
 * attn_flash: fused paged attention, online softmax, one pass over KV.
 * Format and scales follow attn_core (lfmgemv_impl.c): int4 K with per-token
 * Q16 scales, int8 V, 16 q heads / 4 kv heads, 32-token pages.
 *
 * The softmax (running max, exp, l, rescale factors, normalisation) is f32.
 * v68 has no IEEE f32 vector SIMD: sf vadd/vmpy and every vconv are v73+ and
 * the qf32 accumulator does not hold a whole window, so acc = sum_j p_j v_j
 * runs in int32.  p is f32 and enters the MAC as Q12, exact enough for the
 * int8 V of this model; l stays f32.
 */
#include "attn_flash.h"

#include <hexagon_types.h>
#include "hvx_hexagon_protos.h"
#include "hexagon_protos.h"

#include <stdint.h>
#include <string.h>

#define AF_HQ 16
#define AF_KV 4
#define AF_D  128

#define AF_LOG2E      1.4426950408889634f
#define AF_LN2        0.6931471805599453f
#define AF_INV_SQRT_D 0.08838834764831845f
#define AF_MIN_EXP    (-87.0f)

#define AF_PS 4096              /* Q12 weight scale, halved for long windows */

typedef union { HVX_Vector v; int32_t w[32]; float f[32]; } af_vec;
typedef union { HVX_VectorPair p; HVX_Vector v[2]; } af_pair;

static int8_t  g_qe[AF_HQ][AF_D] __attribute__((aligned(128)));
static int8_t  g_qo[AF_HQ][AF_D] __attribute__((aligned(128)));
static int32_t g_sumq[AF_HQ];
static float   g_qf[AF_HQ];

/* lane gather after AF_VSUM5: keep word 0/1/2/3 of the rotated vectors */
static const int32_t g_l1[32] __attribute__((aligned(128))) = { -1 };
static const int32_t g_l2[32] __attribute__((aligned(128))) = { 0, -1 };
static const int32_t g_l3[32] __attribute__((aligned(128))) = { 0, 0, -1 };
static const int32_t g_l4[32] __attribute__((aligned(128))) = { 0, 0, 0, -1 };

#define AF_VSUM5(v) do {                                      \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 64));     \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 32));     \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 16));     \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 8));      \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 4));      \
    } while (0)

/* scalar -> both halfwords of every word (operand of vmpy(Vu.w, Vv.h)) */
static inline HVX_Vector af_splat_h(int32_t v)
{
    return Q6_V_vsplat_R((uint32_t)v * 0x00010001u);
}

/* exp(x) for x <= 0: 2^round(y) * poly(2^frac), y = x * log2(e). */
static inline float af_exp(float x)
{
    union { float f; uint32_t u; } sc;
    float y, f, p;
    int i;

    if (x <= AF_MIN_EXP) return 0.0f;
    y = x * AF_LOG2E;
    i = (int)(y + (y >= 0.0f ? 0.5f : -0.5f));
    f = (y - (float)i) * AF_LN2;
    p = 1.0f / 720.0f;
    p = p * f + 1.0f / 120.0f;
    p = p * f + 1.0f / 24.0f;
    p = p * f + 1.0f / 6.0f;
    p = p * f + 0.5f;
    p = p * f + 1.0f;
    p = p * f + 1.0f;
    sc.u = (uint32_t)(i + 127) << 23;
    return p * sc.f;
}

static void af_group(int kh, const int8_t *k, const int32_t *ks,
                     const int8_t *v, const float *vscale,
                     const int *tbl, int slots, int tok0, int n, float *out)
{
    const int8_t *kb = k + (size_t)kh * (size_t)slots * AF_D;
    const int32_t *ktp = ks + (size_t)kh * (size_t)slots;
    const int8_t *vb = v + (size_t)kh * (size_t)slots * AF_D;
    const float *vs = vscale ? vscale + kh * AF_D : 0;
    HVX_Vector qa[4][2], acc[4][4];
    float m[4], l[4];
    int32_t pq[4];
    float pscale = (float)AF_PS;
    int h, q, t, d, psh;

    /* |acc| = pscale * 127 * n must stay inside int32 */
    for (psh = 0; psh < 8 && (n >> psh) > 2048; psh++) pscale *= 0.5f;

    for (h = 0; h < 4; h++) {
        qa[h][0] = *(const HVX_Vector *)((const int8_t *)g_qe +
                                        (size_t)(kh * 4 + h) * AF_D);
        qa[h][1] = *(const HVX_Vector *)((const int8_t *)g_qo +
                                        (size_t)(kh * 4 + h) * AF_D);
        m[h] = -1.0e30f;
        l[h] = 0.0f;
        pq[h] = 0;
        for (q = 0; q < 4; q++) acc[h][q] = Q6_V_vzero();
    }

    for (t = 0; t < n; t++) {
        int tok = tok0 + t;
        int slot = (tbl[tok >> 5] << 5) | (tok & 31);
        HVX_Vector krow, lo, hi, a[4], g;
        af_vec u;
        float kc, sc[4], pp[4];
        int32_t d4[4];

        if (slot < 0 || slot >= slots) continue;

        krow = *(const HVX_Vector *)(kb + (size_t)slot * AF_D);
        lo = Q6_V_vand_VV(krow, Q6_V_vsplat_R(0x0f0f0f0fu));
        hi = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(krow, 4), Q6_V_vsplat_R(0x0f0f0f0fu));
        kc = (float)ktp[slot] * (AF_INV_SQRT_D / 65536.0f);

#pragma unroll
        for (h = 0; h < 4; h++) {
            a[h] = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[h][0]);
            a[h] = Q6_Vw_vrmpyacc_VwVubVb(a[h], hi, qa[h][1]);
            AF_VSUM5(a[h]);
        }
        g = Q6_V_vor_VV(
              Q6_V_vor_VV(Q6_V_vand_VV(a[0], *(const HVX_Vector *)g_l1),
                          Q6_V_vand_VV(Q6_V_vror_VR(a[1], 124),
                                       *(const HVX_Vector *)g_l2)),
              Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(a[2], 120),
                                       *(const HVX_Vector *)g_l3),
                          Q6_V_vand_VV(Q6_V_vror_VR(a[3], 116),
                                       *(const HVX_Vector *)g_l4)));
        u.v = g;
        for (h = 0; h < 4; h++) d4[h] = u.w[h];

#pragma unroll
        for (h = 0; h < 4; h++)
            sc[h] = (float)(d4[h] - 8 * g_sumq[kh * 4 + h]) *
                    g_qf[kh * 4 + h] * kc;

#pragma unroll
        for (h = 0; h < 4; h++) {
            if (sc[h] > m[h]) {
                float c = af_exp(m[h] - sc[h]);
                int f15 = (int)(c * 32768.0f + 0.5f);
                if (f15 > 32767) f15 = 32767;
                l[h] *= c;
                m[h] = sc[h];
                if (f15 < 32767) {
                    HVX_Vector cv = af_splat_h(f15);
#pragma unroll
                    for (q = 0; q < 4; q++)
                        acc[h][q] = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(acc[h][q], cv);
                }
            }
        }

        pp[0] = af_exp(sc[0] - m[0]);
        pp[1] = af_exp(sc[1] - m[1]);
        pp[2] = af_exp(sc[2] - m[2]);
        pp[3] = af_exp(sc[3] - m[3]);
#pragma unroll
        for (h = 0; h < 4; h++) {
            l[h] += pp[h];
            pq[h] = (int32_t)(pp[h] * pscale + 0.5f);
        }

        if (!(pq[0] | pq[1] | pq[2] | pq[3]))
            continue;
        {
            af_pair b, w0, w1;
            HVX_Vector vw[4];
            HVX_Vector vv = *(const HVX_Vector *)(vb + (size_t)slot * AF_D);

            b.p = Q6_Wh_vunpack_Vb(vv);
            w0.p = Q6_Ww_vunpack_Vh(b.v[0]);
            w1.p = Q6_Ww_vunpack_Vh(b.v[1]);
            vw[0] = w0.v[0];
            vw[1] = w0.v[1];
            vw[2] = w1.v[0];
            vw[3] = w1.v[1];
#pragma unroll
            for (h = 0; h < 4; h++) {
                HVX_Vector ps;
                if (!pq[h]) continue;
                ps = af_splat_h(pq[h]);
#pragma unroll
                for (q = 0; q < 4; q++)
                    acc[h][q] = Q6_Vw_vmpyieacc_VwVwVh(acc[h][q], vw[q], ps);
            }
        }
    }

    /* int32 -> f32 is scalar: v68 has no vector conversion */
    for (h = 0; h < 4; h++) {
        float inv = l[h] > 0.0f ? (1.0f / l[h]) / pscale : 0.0f;
        float *dst = out + (size_t)(kh * 4 + h) * AF_D;
        for (q = 0; q < 4; q++) {
            af_vec u;
            u.v = acc[h][q];
            if (vs) {
                for (d = 0; d < 32; d++)
                    dst[q * 32 + d] = (float)u.w[d] * inv * vs[q * 32 + d];
            } else {
                for (d = 0; d < 32; d++)
                    dst[q * 32 + d] = (float)u.w[d] * inv;
            }
        }
    }
}

void attn_flash(const int8_t *q, const int32_t *qscale,
                const int8_t *k, const int32_t *ks,
                const int8_t *v, const float *vscale,
                const int *tbl, int slots, int pos, int window, int reps,
                float *out)
{
    int h, d, kh, r, tok0, n;

    if (!q || !k || !ks || !v || !tbl || !out) return;
    if (slots < 1 || pos < 0) return;
    if (reps < 1) reps = 1;

    for (h = 0; h < AF_HQ; h++) {
        const int8_t *qp = q + (size_t)h * AF_D;
        int32_t sq = 0;
        memset(g_qe[h], 0, AF_D);
        memset(g_qo[h], 0, AF_D);
        for (d = 0; d < AF_D; d++) sq += qp[d];
        for (d = 0; d < AF_D / 2; d++) {
            g_qe[h][d] = qp[2 * d];
            g_qo[h][d] = qp[2 * d + 1];
        }
        g_sumq[h] = sq;
        g_qf[h] = qscale ? (float)qscale[h] * (1.0f / 65536.0f) : 1.0f;
    }

    tok0 = (window > 0 && pos + 1 - window > 0) ? pos + 1 - window : 0;
    n = pos + 1 - tok0;

    for (r = 0; r < reps; r++)
        for (kh = 0; kh < AF_KV; kh++)
            af_group(kh, k, ks, v, vscale, tbl, slots, tok0, n, out);
}
