/*
 * attn_flash_main: DSP stand for the fused attention kernel under hexagon-sim.
 *
 * Deterministic LCG data in the resident KV layout (int4 K with per-token Q16
 * scales, int8 V, per-channel V dequant), a shuffled block table, and an f64
 * two-pass softmax on the DSP as the oracle.  Checks cos and max|d| for
 * ctx 64/256/512, a narrow window and a single token, then reports pcycles.
 *
 *   build/run: ./attn_flash_build.sh
 */
#include "attn_flash.h"

#include <qurt_alloc.h>
#include <HAP_perf.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

QURT_DECLARE_STATIC_HEAP(16u * 1024u * 1024u);

#define HQ     16
#define KVH    4
#define D      128
#define PAGE   32
#define CTXMAX 512
#define PAGES  (CTXMAX / PAGE)

#define COS_MIN 0.9995
#define ABS_MAX 5.0e-3

static int8_t  g_k[KVH][CTXMAX][D] __attribute__((aligned(128)));
static int8_t  g_v[KVH][CTXMAX][D] __attribute__((aligned(128)));
static int32_t g_ks[KVH][CTXMAX] __attribute__((aligned(128)));
static int32_t g_qs[HQ] __attribute__((aligned(128)));
static int8_t  g_q[HQ][D] __attribute__((aligned(128)));
static float   g_vs[KVH][D] __attribute__((aligned(128)));
static float   g_out[HQ][D] __attribute__((aligned(128)));
static double  g_ref[HQ][D];
static int     g_tbl[PAGES] __attribute__((aligned(128)));
static double  g_sc[CTXMAX];

static int g_fail;
static uint32_t g_rng = 0x9e3779b9u;

static uint32_t rnd(void)
{
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return x;
}

static int rnd_int(int lo, int hi)
{
    return lo + (int)(rnd() % (uint32_t)(hi - lo + 1));
}

static void gen_data(void)
{
    int kh, s, d, h, p;

    for (kh = 0; kh < KVH; kh++) {
        for (s = 0; s < CTXMAX; s++) {
            int8_t *row = g_k[kh][s];
            for (d = 0; d < D / 2; d++)
                row[d] = (int8_t)(rnd_int(0, 15) | (rnd_int(0, 15) << 4));
            memset(row + D / 2, 0, D / 2);
            for (d = 0; d < D; d++)
                g_v[kh][s][d] = (int8_t)rnd_int(-127, 127);
            g_ks[kh][s] = 6000 + rnd_int(0, 6000);
        }
        for (d = 0; d < D; d++)
            g_vs[kh][d] = 0.002f + (float)rnd_int(0, 1000) * 1.0e-5f;
    }
    for (h = 0; h < HQ; h++) {
        for (d = 0; d < D; d++) g_q[h][d] = (int8_t)rnd_int(-127, 127);
        g_qs[h] = 1600 + rnd_int(0, 800);
    }
    for (p = 0; p < PAGES; p++) g_tbl[p] = p;
    for (p = PAGES - 1; p > 0; p--) {
        int j = rnd_int(0, p);
        int t = g_tbl[p];
        g_tbl[p] = g_tbl[j];
        g_tbl[j] = t;
    }
}

static int32_t dot_i4(int h, int kh, int slot)
{
    const int8_t *row = g_k[kh][slot];
    const int8_t *qp = g_q[h];
    int32_t acc = 0;
    int d;

    for (d = 0; d < D / 2; d++) {
        int u = (uint8_t)row[d];
        acc += qp[2 * d] * ((u & 0xf) - 8);
        acc += qp[2 * d + 1] * ((u >> 4) - 8);
    }
    return acc;
}

static void ref_attn(int pos, int window)
{
    int tok0 = (window > 0 && pos + 1 - window > 0) ? pos + 1 - window : 0;
    int n = pos + 1 - tok0;
    int h, kh, t, d;

    for (h = 0; h < HQ; h++) {
        double mx = -1.0e300, sum = 0.0;
        kh = h / (HQ / KVH);
        for (t = 0; t < n; t++) {
            int tok = tok0 + t;
            int slot = (g_tbl[tok >> 5] << 5) | (tok & 31);
            double sc = (double)dot_i4(h, kh, slot) *
                        ((double)g_qs[h] / 65536.0) *
                        ((double)g_ks[kh][slot] / 65536.0) *
                        (1.0 / sqrt((double)D));
            g_sc[t] = sc;
            if (sc > mx) mx = sc;
        }
        for (d = 0; d < D; d++) g_ref[h][d] = 0.0;
        for (t = 0; t < n; t++) {
            int tok = tok0 + t;
            int slot = (g_tbl[tok >> 5] << 5) | (tok & 31);
            double p = exp(g_sc[t] - mx);
            sum += p;
            for (d = 0; d < D; d++)
                g_ref[h][d] += p * (double)g_v[kh][slot][d];
        }
        for (d = 0; d < D; d++)
            g_ref[h][d] = sum > 0.0 ?
                g_ref[h][d] / sum * (double)g_vs[kh][d] : 0.0;
    }
}

static void check(int pos, int window, const char *tag)
{
    double num = 0.0, da = 0.0, db = 0.0, mx = 0.0, cos;
    int h, d;
    int n = pos + 1 - ((window > 0 && pos + 1 - window > 0) ?
                       pos + 1 - window : 0);

    ref_attn(pos, window);
    attn_flash((const int8_t *)g_q, g_qs, (const int8_t *)g_k,
               (const int32_t *)g_ks, (const int8_t *)g_v,
               (const float *)g_vs, g_tbl, CTXMAX, pos, window, 1,
               (float *)g_out);
    for (h = 0; h < HQ; h++) {
        for (d = 0; d < D; d++) {
            double a = (double)g_out[h][d];
            double b = g_ref[h][d];
            double e = fabs(a - b);
            num += a * b;
            da += a * a;
            db += b * b;
            if (e > mx) mx = e;
        }
    }
    cos = (da > 0.0 && db > 0.0) ? num / sqrt(da * db) : 0.0;
    printf("AF %-12s n=%-4d cos=%.9f maxabs=%.3e  %s\n", tag, n, cos, mx,
           (cos >= COS_MIN && mx <= ABS_MAX) ? "OK" : "FAIL");
    if (!(cos >= COS_MIN && mx <= ABS_MAX)) g_fail = 1;
}

/* qscale/vscale = NULL paths: reference with qs = 1.0 and vs = 1.0 */
static void check_null_scales(void)
{
    double num = 0.0, da = 0.0, db = 0.0, mx = 0.0, cos;
    int32_t sq[HQ];
    float sv[KVH][D];
    int h, d;
    const int pos = 63, window = 64;
    int n = 64;

    for (h = 0; h < HQ; h++) {
        sq[h] = g_qs[h];
        g_qs[h] = 65536;
    }
    for (h = 0; h < KVH; h++) {
        for (d = 0; d < D; d++) {
            sv[h][d] = g_vs[h][d];
            g_vs[h][d] = 1.0f;
        }
    }
    ref_attn(pos, window);
    attn_flash((const int8_t *)g_q, 0, (const int8_t *)g_k,
               (const int32_t *)g_ks, (const int8_t *)g_v, 0, g_tbl,
               CTXMAX, pos, window, 1, (float *)g_out);
    for (h = 0; h < HQ; h++) {
        for (d = 0; d < D; d++) {
            double a = (double)g_out[h][d];
            double b = g_ref[h][d];
            double e = fabs(a - b);
            num += a * b;
            da += a * a;
            db += b * b;
            if (e > mx) mx = e;
        }
    }
    cos = (da > 0.0 && db > 0.0) ? num / sqrt(da * db) : 0.0;
    printf("AF %-12s n=%-4d cos=%.9f maxabs=%.3e  %s\n", "null scales", n, cos,
           mx, cos >= COS_MIN ? "OK" : "FAIL");
    if (cos < COS_MIN) g_fail = 1;
    for (h = 0; h < HQ; h++) g_qs[h] = sq[h];
    for (h = 0; h < KVH; h++)
        for (d = 0; d < D; d++) g_vs[h][d] = sv[h][d];
}

static void bench(int pos, int window, int reps)
{
    uint64_t t0, t1;
    double pc;

    attn_flash((const int8_t *)g_q, g_qs, (const int8_t *)g_k,
               (const int32_t *)g_ks, (const int8_t *)g_v,
               (const float *)g_vs, g_tbl, CTXMAX, pos, window, 1,
               (float *)g_out);
    t0 = HAP_perf_get_pcycles();
    attn_flash((const int8_t *)g_q, g_qs, (const int8_t *)g_k,
               (const int32_t *)g_ks, (const int8_t *)g_v,
               (const float *)g_vs, g_tbl, CTXMAX, pos, window, reps,
               (float *)g_out);
    t1 = HAP_perf_get_pcycles();
    pc = (double)(t1 - t0) / (double)reps;
    printf("AF ctx=%-4d reps=%-2d pcycles=%.0f per_token=%.1f us@1.42GHz=%.1f\n",
           pos + 1, reps, pc, pc / (double)(pos + 1), pc / 1420.0);
}

int main(void)
{
    printf("ATTN FLASH stand: HQ=%d KV=%d D=%d page=%d slots=%d "
           "cos>=%.4f maxabs<=%.0e\n", HQ, KVH, D, PAGE, CTXMAX,
           COS_MIN, ABS_MAX);

    gen_data();
    check(0, 0, "one token");
    check(63, 64, "ctx 64");
    check(255, 256, "ctx 256");
    check(511, 512, "ctx 512");
    check(511, 100, "window 100");
    check(511, 0, "window none");
    check_null_scales();

    bench(63, 64, 8);
    bench(255, 256, 4);
    bench(511, 512, 3);

    printf("ATTN FLASH RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
