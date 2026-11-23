/* gemv2_main: stand for the v2 int4 GEMV (HVX) against the f64 reference. */
#include "gemv2.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GEMV2_HOST
#include <qurt_alloc.h>
#include "HAP_perf.h"
QURT_DECLARE_STATIC_HEAP(16u * 1024u * 1024u);
#else
#include <time.h>
static uint64_t HAP_perf_get_pcycles(void) { return (uint64_t)clock(); }
#endif

#define N 256
#define K 2048

static uint8_t codes[N * (K / 2)] __attribute__((aligned(128)));
static float row_scale[N] __attribute__((aligned(128)));
static uint8_t ratio[N * (K / 256)] __attribute__((aligned(128)));
static int8_t xq[K] __attribute__((aligned(128)));
static float sa[K / 32] __attribute__((aligned(128)));
static float out_ref[N] __attribute__((aligned(128)));
static float out_hvx[N] __attribute__((aligned(128)));

static uint32_t g_rng = 0x1234567u;
static uint32_t rnd(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

int main(void)
{
    int i;
    gemv2_matrix m;
    double dot = 0, na = 0, nb = 0, mx = 0;

    for (i = 0; i < N * (K / 2); i++)
        codes[i] = (uint8_t)(rnd() & 0xff);
    for (i = 0; i < N; i++)
        row_scale[i] = 0.002f + (float)(rnd() % 1000) * 1e-6f;
    for (i = 0; i < N * (K / 256); i++)
        ratio[i] = (uint8_t)(1 + rnd() % 128);
    for (i = 0; i < K; i++)
        xq[i] = (int8_t)((int)(rnd() >> 24));
    for (i = 0; i < K / 32; i++)
        sa[i] = 0.001f + (float)(rnd() % 1000) * 1e-6f;

    m.n = N;
    m.k = K;
    m.codes = codes;
    m.row_scale = row_scale;
    m.seg_ratio = ratio;
    m.seg_max = K / 256;

    gemv2_ref(&m, xq, sa, out_ref);
    gemv2_hvx(&m, xq, sa, out_hvx);

    for (i = 0; i < N; i++) {
        double d = fabs((double)out_hvx[i] - (double)out_ref[i]);
        if (d > mx) mx = d;
        dot += (double)out_hvx[i] * out_ref[i];
        na += (double)out_hvx[i] * out_hvx[i];
        nb += (double)out_ref[i] * out_ref[i];
    }
    printf("GEMV2 N=%d K=%d cos=%.9f max|d|=%.3e  %s\n", N, K,
           dot / sqrt(na * nb), mx,
           mx < 1e-4 ? "OK" : "FAIL");
    printf("GEMV2 RESULT: %s\n", mx < 1e-4 ? "PASS" : "FAIL");
    return mx < 1e-4 ? 0 : 1;
}
