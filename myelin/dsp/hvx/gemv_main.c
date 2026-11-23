#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../common/fp16.h"
#include "../common/q4.h"
#include "../common/quant.h"
#include "hvx_ops.h"

#define LFM_TOL 1e-5f
#define LFM_COS 0.9999f
#define LFM_COS_INT4 0.99f

static uint32_t g_seed = 12345u;
static int g_fail;

static uint32_t rnd_u32(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}

static float rnd_uni(void)
{
    return (float)(rnd_u32() >> 8) * (1.0f / 16777216.0f);
}

static void *xalloc(size_t bytes)
{
    void *raw = malloc(bytes + 256u);
    uintptr_t a;
    if (!raw)
        return NULL;
    a = ((uintptr_t)raw + 128u) & ~(uintptr_t)127u;
    ((void **)a)[-1] = raw;
    return (void *)a;
}

static void xfree(void *p)
{
    if (p)
        free(((void **)p)[-1]);
}

static double now_sec(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static float dot_f32(const float *a, const float *b, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++)
        s += (double)a[i] * (double)b[i];
    return (float)s;
}

static float cos_f32(const float *a, const float *b, int n)
{
    double d = 0.0;
    double na = 0.0;
    double nb = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        d += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0f;
    return (float)(d / sqrt(na * nb));
}

static int run_case(int n, int k)
{
    size_t img_bytes = lfm_q4_image_bytes(n, k);
    float *w = xalloc((size_t)n * (size_t)k * sizeof(float));
    float *x = xalloc((size_t)k * sizeof(float));
    float *xr = xalloc((size_t)k * sizeof(float));
    int8_t *q = xalloc((size_t)k + 256u);
    float *ascale = xalloc(((size_t)k / 32u) * sizeof(float));
    float *y_ref = xalloc((size_t)n * sizeof(float));
    float *y = xalloc((size_t)n * sizeof(float));
    float *y_f32 = xalloc((size_t)n * sizeof(float));
    float *y_dq = xalloc((size_t)n * sizeof(float));
    float *xd = xalloc((size_t)k * sizeof(float));
    float *wr = xalloc((size_t)k * sizeof(float));
    uint8_t *img = xalloc(img_bytes);
    double t0;
    double t1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float cos_dq;
    float cos_real;
    int i;
    int r;

    if (!w || !x || !xr || !q || !ascale || !y_ref || !y || !y_f32 || !y_dq || !xd || !wr || !img) {
        printf("FAIL gemv N=%d K=%d: alloc\n", n, k);
        g_fail++;
        goto out;
    }
    for (i = 0; i < n * k; i++)
        w[i] = (rnd_uni() - 0.5f) * 0.7f;
    for (i = 0; i < k; i++)
        x[i] = (rnd_uni() - 0.5f) * 1.5f;
    memset(img, 0, img_bytes);
    lfm_q4_pack_matrix(w, n, k, img, 128);
    memcpy(xr, x, (size_t)k * sizeof(float));
    if (lfm_act_i8_rot(xr, k, 128, q, ascale) != 0) {
        printf("FAIL gemv N=%d K=%d: quant\n", n, k);
        g_fail++;
        goto out;
    }
    lfm_q4_gemv_i8(img, n, k, q, ascale, y_ref);
    t0 = now_sec();
    lfm_hvx_gemv_i4(img, n, k, q, ascale, y);
    t1 = now_sec();
    for (r = 0; r < n; r++) {
        float d = fabsf(y[r] - y_ref[r]);
        float s = fabsf(y_ref[r]) > 1.0f ? fabsf(y_ref[r]) : 1.0f;
        if (d > max_abs)
            max_abs = d;
        if (d / s > max_rel)
            max_rel = d / s;
    }
    for (r = 0; r < n; r++)
        y_f32[r] = dot_f32(w + (size_t)r * (size_t)k, x, k);
    for (i = 0; i < k; i++)
        xd[i] = (float)q[i] * ascale[i / 32];
    for (r = 0; r < n; r++) {
        lfm_q4_dequant_row(lfm_q4_row_qs(img, n, k, r), lfm_q4_row_scales(img, n, k, r), k, wr);
        y_dq[r] = dot_f32(wr, xd, k);
    }
    cos_dq = cos_f32(y, y_dq, n);
    cos_real = cos_f32(y, y_f32, n);
    printf("gemv N=%-4d K=%-4d max_abs=%.3e max_rel=%.3e cos_dq=%.7f cos_int4=%.6f ", n, k,
           (double)max_abs, (double)max_rel, (double)cos_dq, (double)cos_real);
    if (t1 > t0)
        printf("time=%.3f ms\n", (t1 - t0) * 1e3);
    else
        printf("time=n/a\n");
    if (max_abs > LFM_TOL || max_rel > LFM_TOL) {
        printf("FAIL gemv N=%d K=%d: ref mismatch\n", n, k);
        g_fail++;
    }
    if (cos_dq < LFM_COS) {
        printf("FAIL gemv N=%d K=%d: cos_dq %.7f < %.7f\n", n, k, (double)cos_dq, (double)LFM_COS);
        g_fail++;
    }
    if (cos_real < LFM_COS_INT4) {
        printf("FAIL gemv N=%d K=%d: cos_int4 %.7f < %.7f\n", n, k, (double)cos_real,
               (double)LFM_COS_INT4);
        g_fail++;
    }
out:
    xfree(w);
    xfree(x);
    xfree(xr);
    xfree(q);
    xfree(ascale);
    xfree(y_ref);
    xfree(y);
    xfree(y_f32);
    xfree(y_dq);
    xfree(xd);
    xfree(wr);
    xfree(img);
    return g_fail == 0;
}

static void bench(int n, int k, int iters)
{
    size_t img_bytes = lfm_q4_image_bytes(n, k);
    uint8_t *img = xalloc(img_bytes);
    int8_t *q = xalloc((size_t)k + 256u);
    float *ascale = xalloc(((size_t)k / 32u) * sizeof(float));
    float *y = xalloc((size_t)n * sizeof(float));
    size_t bytes;
    double best = 0.0;
    uint16_t *sc;
    uint8_t *planes;
    int it;
    int i;
    int r;

    if (!img || !q || !ascale || !y) {
        printf("FAIL bench: alloc\n");
        g_fail++;
        goto out;
    }
    planes = (uint8_t *)lfm_q4_planes(img, n, k);
    sc = (uint16_t *)img;
    for (r = 0; r < n; r++) {
        memset(planes + (size_t)r * (size_t)(k / 2), (0x11 + r * 37) & 0xFF, (size_t)(k / 2));
        for (i = 0; i < k / 32; i++)
            sc[(size_t)r * (size_t)(k / 32) + (size_t)i] = lfm_f32_to_f16(1e-4f);
    }
    for (i = 0; i < k; i++)
        q[i] = (int8_t)((int)(rnd_u32() & 0xFFu) - 128);
    for (i = 0; i < k / 32; i++)
        ascale[i] = 1e-6f;
    lfm_hvx_gemv_i4(img, n, k, q, ascale, y);
    for (it = 0; it < iters; it++) {
        double t0 = now_sec();
        lfm_hvx_gemv_i4(img, n, k, q, ascale, y);
        {
            double dt = now_sec() - t0;
            if (best == 0.0 || dt < best)
                best = dt;
        }
    }
    bytes = (size_t)n * (size_t)(k / 2) + (size_t)n * (size_t)(k / 32) * 2u + (size_t)k +
            (size_t)n * sizeof(float);
#if defined(__HVX__)
    printf("bench N=%d K=%d M=1: %d iters, %.1f MB (время: отчёт hexagon-sim)\n", n, k, iters,
           (double)bytes / 1048576.0);
    (void)best;
#else
    if (best > 0.0) {
        printf("bench N=%d K=%d M=1: %.3f ms, %.2f GB/s (%.1f MB, %d iters)\n", n, k, best * 1e3,
               (double)bytes / best / 1e9, (double)bytes / 1048576.0, iters);
    } else {
        printf("bench N=%d K=%d M=1: n/a (clock() не идёт)\n", n, k);
    }
#endif
    {
        float mx = 0.0f;
        for (i = 0; i < n; i++) {
            float a = fabsf(y[i]);
            if (a > mx)
                mx = a;
        }
        printf("bench max|y|=%.3e\n", (double)mx);
    }
out:
    xfree(img);
    xfree(q);
    xfree(ascale);
    xfree(y);
}

int main(int argc, char **argv)
{
#if defined(__HVX__)
    static const int cases[][2] = {
        {64, 512}, {64, 768}, {96, 640}, {128, 1024},
    };
    static const int ncases = (int)(sizeof cases / sizeof cases[0]);
#else
    static const int cases[][2] = {
        {64, 512}, {96, 640}, {128, 768}, {128, 1024}, {256, 1280}, {64, 2048}, {256, 2048},
    };
    static const int ncases = (int)(sizeof cases / sizeof cases[0]);
#endif
    int i;

    if (argc > 1 && argv[1][0] == 'b') {
        bench(7680, 2048, argc > 2 ? atoi(argv[2]) : 3);
        return g_fail ? 1 : 0;
    }
    printf("gemv_i4 stand: %s\n",
#if defined(__HVX__)
           "hexagon HVX"
#else
           "portable C11"
#endif
    );
    for (i = 0; i < ncases; i++)
        run_case(cases[i][0], cases[i][1]);
    bench(7680, 2048, 3);
    if (g_fail) {
        printf("gemv FAIL (%d)\n", g_fail);
        return 1;
    }
    printf("gemv PASS\n");
    return 0;
}
