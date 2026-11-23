/*
 * hmx_blkscale_sim — REPORT24: end-to-end functional model of the block-scale
 * int4 prefill path, run in hexagon-sim (Hexagon v68, SDK 6.7).
 *
 * This is the same arithmetic that `mx4_blkpref` (mx4_impl.c) runs on the DSP,
 * written out with explicit VTCM addresses and checked against a CPU reference
 * computed from the same packed operands.
 *
 * Recipe per (32-N tile, 64-spatial tile):
 *   for g in 0..ng-1:                       one block-scale group of G croutons
 *       acc = 0
 *       mul(offset_act, offset_wgt)          adds the constant C0 to every lane
 *       for c in g*G .. g*G+G-1:
 *           mul(act[c], wgt[c])              one 32 K x 32 N instruction
 *       bias = 1 group of 32 words           word = fp16(s_g[n] * Kc * 512)
 *       store .ub  -> partial[g]             8-bit, layout out(s,n)
 *   combine: sum the ng partials byte-wise (identical layout, so no transpose
 *            in the hot loop), transpose once, subtract C0*sum_g(s_g[n])*Kc,
 *            divide by Kc.
 *
 * Established semantics (hmx_ctrl_probe / hmx_part_probe*):
 *   - one int4 instruction pair = 32 K x 32 N x 64 spatial;
 *   - `.ub` store writes 2 KB, layout out(s,n) = 128*(s>>2) + 4*n + (s&3);
 *   - the store multiplies the accumulator by fp16(bias_word)/512 and
 *     saturates to unsigned 8 bit, so a negative partial needs the offset C0;
 *   - `.uh` (16-bit) only covers 32 of the 64 spatial rows, so 8-bit it is.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define VTCM_BASE 0xd8400000ul
#define OFF_ACT    0x00000u
#define OFF_WGT    0x20000u
#define OFF_PART   0x30000u
#define OFF_BIAS   0x40000u
#define OFF_SUM    0x41000u
#define OFF_OUT    0x42000u

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}
static inline void hmx_clr(void) { __asm__ __volatile__("mxclracc" ::: "memory"); }
static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}
static inline void hmx_bias2(const void *p)
{
    __asm__ __volatile__("{ bias = mxmem2(%0) }"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}
static inline void hmx_mul(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}
static inline void hmx_st_ub(void *o)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(0u) : "memory");
}

/* ---- packing helpers: identical math to mx4_impl.c / mkblob.c ---- */
static void pack_act(uint8_t *dst, const uint8_t *A, int s0, int rows, int K, int ncr, int mt)
{
    int s, k;
    (void)mt;
    for (s = 0; s < rows; s++)
        for (k = 0; k < K; k++)
            dst[(unsigned)(s0 >> 6) * (unsigned)ncr * 2048u
                + 2048u * (unsigned)(k >> 5)
                + 128u * (unsigned)((s & 63) >> 2)
                + 4u * (unsigned)(k & 31) + (unsigned)(s & 3)] =
                A[(size_t)s * K + k];
}
static void pack_w4(uint8_t *dst, const int8_t *q, int n0, int K)
{
    int k, n;
    memset(dst, 0, (size_t)(K / 32) * 512u);
    for (n = 0; n < 32; n++)
        for (k = 0; k < K; k++) {
            unsigned v = (unsigned)(q[(size_t)(n0 + n) * K + k] & 0xF);
            unsigned off = 512u * (unsigned)(k >> 5)
                         + 128u * (unsigned)((k & 31) >> 3)
                         + 4u * (unsigned)n + (unsigned)(k & 3);
            if ((k >> 2) & 1) dst[off] |= (uint8_t)(v << 4);
            else              dst[off] |= (uint8_t)v;
        }
}

static uint16_t f16(float f)
{
    union { float f; uint32_t u; } v;
    uint32_t s, e, m;
    v.f = f;
    s = (v.u >> 31) & 1u;
    if (v.u == 0) return 0;
    e = ((v.u >> 23) & 0xFFu) - 127 + 15;
    m = v.u & 0x7FFFFFu;
    return (uint16_t)((s << 15) | ((e & 0x1Fu) << 10) | (m >> 13));
}

int main(void)
{
    /* shape: one 32-N tile, 64 spatial rows (32 real), K = 128, G = 2 */
    enum { M = 32, N = 32, K = 128, G = 2 };
    int ncr = K / 32, ng = ncr / G;
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *vact = p + OFF_ACT;
    uint8_t *vwgt = p + OFF_WGT;
    uint8_t *vpart = p + OFF_PART;
    uint8_t *vbias = p + OFF_BIAS;
    uint8_t *vsum = p + OFF_SUM;
    uint8_t *voact = p + OFF_ACT + 0x10000;         /* offset croutons */
    uint8_t *vowgt = p + OFF_WGT + 0x8000;
    /* host-side model data */
    static uint8_t A[M * K];
    static int8_t q[N * K];
    static float sc[64];
    static double yref[M * N];
    static float ygot[M * N];
    uint32_t rng = 1234567u;
    double C0, Kc, smax = 0, maxP = 0, Aoff;
    int Aq;
    int s, n, k, g, c, i, j;

    set_xe2();
    memset(p, 0, 0x50000);

    /* ---- synthetic operands ---- */
    for (s = 0; s < M; s++)
        for (k = 0; k < K; k++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            A[s * K + k] = (uint8_t)(rng & 0xFF);
        }
    /* weights: independent per (n, 32G) block, quantised to [-8,7] */
    for (n = 0; n < N; n++)
        for (g = 0; g < ng; g++) {
            float amax = 0, *w = (float *)malloc(sizeof(float) * 32 * G);
            for (k = 0; k < 32 * G; k++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                w[k] = ((float)(rng % 20001) - 10000.0f) / 10000.0f * 0.03f;
                if (fabsf(w[k]) > amax) amax = fabsf(w[k]);
            }
            sc[n * ng + g] = amax > 0 ? amax / 7.0f : 0.0f;
            for (k = 0; k < 32 * G; k++) {
                int v = sc[n * ng + g] > 0 ?
                        (int)lrintf(w[k] / sc[n * ng + g]) : 0;
                if (v > 7) v = 7;
                if (v < -8) v = -8;
                q[n * K + g * 32 * G + k] = (int8_t)v;
            }
            free(w);
        }
    for (i = 0; i < N * ng; i++) if (sc[i] > smax) smax = sc[i];

    /* ---- reference (exact, int64 partials) ---- */
    for (s = 0; s < M; s++)
        for (n = 0; n < N; n++) {
            double y = 0;
            for (g = 0; g < ng; g++) {
                long pacc = 0;
                for (k = 0; k < 32 * G; k++)
                    pacc += (long)A[s * K + g * 32 * G + k]
                          * (long)q[n * K + g * 32 * G + k];
                if (fabs((double)pacc) > maxP) maxP = fabs((double)pacc);
                y += (double)sc[n * ng + g] * (double)pacc;
            }
            yref[s * N + n] = y;
        }
    /* offset C0 must clear max|partial|; build it as 224*A (a_k = A, q_k = 7) */
    Aq = (int)ceil(maxP * 1.05 / 224.0);
    if (Aq > 255) { printf("FAIL: offset does not fit one crouton (A=%d)\n", Aq); return 1; }
    Aoff = Aq;
    C0 = 224.0 * Aoff;
    printf("max|partial| over the tile = %.0f  ->  C0 = %.0f (a_k=%.0f, q_k=7)\n",
           maxP, C0, Aoff);

    /* store normalisation: every partial uses the full 0..255 range; the
     * per-(n,group) scale is applied later in the combine (REPORT24) */
    Kc = 255.0 / (2.0 * C0);
    printf("smax = %.6g   store m = %.5f (constant, block scale in the combine)\n",
           smax, Kc);


    /* ---- VTCM operands ---- */
    pack_act(vact, A, 0, 64, K, ncr, 1);
    pack_w4(vwgt, q, 0, K);
    /* offset crouton: sum_k a_k * q_k = C0 = 224 * 74 */
    memset(voact, 0, 2048);
    for (s = 0; s < 64; s++)
        for (k = 0; k < 32; k++)
            voact[128u * (unsigned)(s >> 2) + 4u * (unsigned)k + (unsigned)(s & 3)] = (uint8_t)Aq;
    memset(vowgt, 0, 512);
    for (k = 0; k < 32; k++)
        for (n = 0; n < 32; n++) {
            unsigned off = 128u * (unsigned)(k >> 3) + 4u * (unsigned)n
                         + (unsigned)(k & 3);
            if ((k >> 2) & 1) vowgt[off] |= 0x70u;
            else              vowgt[off] |= 0x07u;
        }

    /* ---- DSP side ---- */
    memset(vsum, 0, 2048);
    for (g = 0; g < ng; g++) {
        /* bias block for this group: word = fp16(s*Kc*512), both halves */
        memset(vbias, 0, 512);
        for (n = 0; n < 32; n++) {
            uint32_t w = (uint32_t)f16((float)(Kc * 512.0));
            ((uint32_t *)vbias)[n] = w;
            ((uint32_t *)vbias)[64 + n] = w;
        }
        hmx_clr();
        hmx_mul(voact, 0x3FFu, vowgt, 511u);        /* +C0 on every lane */
        for (c = 0; c < G; c++)
            hmx_mul(vact + (unsigned)(g * G + c) * 2048u, 0x3FFu,
                    vwgt + (unsigned)(g * G + c) * 512u, 511u);
        hmx_bias2(vbias);
        hmx_st_ub(vpart + (unsigned)g * 2048u);
    }
    hmx_sync();

    /* combine: transpose once, apply the per-(n,group) block scale */
    for (s = 0; s < 64; s++)
        for (n = 0; n < N; n++) {
            unsigned off = 128u * (unsigned)(s >> 2) + 4u * (unsigned)n
                         + (unsigned)(s & 3);
            double acc = 0, scs = 0;
            for (g = 0; g < ng; g++) {
                double sc1 = (double)sc[n * ng + g];
                acc += sc1 * ((double)vpart[(size_t)g * 2048u + off] + 0.5);
                scs += sc1;
            }
            ygot[s * N + n] = (float)(acc / Kc - C0 * scs);
        }

    /* ---- compare ---- */
    {
        double dot = 0, na = 0, nb = 0, mx = 0;
        int bad = 0;
        for (s = 0; s < M; s++)
            for (n = 0; n < N; n++) {
                double a = yref[s * N + n], b = ygot[s * N + n];
                double e = fabs(a - b);
                if (e > mx) mx = e;
                if (e > 1e-3 * (fabs(a) + 1e-9)) bad++;
                dot += a * b; na += a * a; nb += b * b;
            }
        printf("\ncheck M=%d N=%d K=%d G=%d: cos=%.7f  maxabs=%.4f  "
               "cells>0.1%%rel=%d / %d\n",
               M, N, K, G, dot / sqrt(na * nb), mx, bad, M * N);
        /* print a few cells for the record */
        for (i = 0; i < 3; i++) {
            s = i; n = i * 7;
            printf("   y[%d][%d]: ref=%.3f got=%.3f (delta %.3f)\n",
                   s, n, yref[s * N + n], (double)ygot[s * N + n],
                   (double)ygot[s * N + n] - yref[s * N + n]);
        }
        printf("\nRESULT: %s\n", (dot / sqrt(na * nb) > 0.9995) ? "OK" : "CHECK");
    }
    return 0;
}
