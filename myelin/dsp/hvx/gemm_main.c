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

static uint32_t g_seed = 987654321u;
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

static int run_case(int n, int k, int m)
{
    size_t img_bytes = lfm_q4_image_bytes(n, k);
    size_t groups = (size_t)(k / 32);
    float *w = xalloc((size_t)n * (size_t)k * sizeof(float));
    float *x = xalloc((size_t)m * (size_t)k * sizeof(float));
    float *xr = xalloc((size_t)m * (size_t)k * sizeof(float));
    int8_t *q = xalloc((size_t)m * (size_t)k + 256u);
    float *ascale = xalloc((size_t)m * groups * sizeof(float));
    float *y_ref = xalloc((size_t)m * (size_t)n * sizeof(float));
    float *y = xalloc((size_t)m * (size_t)n * sizeof(float));
    float *y_gemv = xalloc((size_t)n * sizeof(float));
    float *y_f32 = xalloc((size_t)m * (size_t)n * sizeof(float));
    float *y_dq = xalloc((size_t)m * (size_t)n * sizeof(float));
    float *xd = xalloc((size_t)m * (size_t)k * sizeof(float));
    float *wr = xalloc((size_t)k * sizeof(float));
    uint8_t *img = xalloc(img_bytes);
    double t0;
    double t1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float min_cos_dq = 1.0f;
    float min_cos_real = 1.0f;
    int i;
    int r;
    int t;

    if (!w || !x || !xr || !q || !ascale || !y_ref || !y || !y_gemv || !y_f32 || !y_dq || !xd ||
        !wr || !img) {
        printf("FAIL gemm N=%d K=%d M=%d: alloc\n", n, k, m);
        g_fail++;
        goto out;
    }
    for (i = 0; i < n * k; i++)
        w[i] = (rnd_uni() - 0.5f) * 0.7f;
    for (i = 0; i < m * k; i++)
        x[i] = (rnd_uni() - 0.5f) * 1.5f;
    memset(img, 0, img_bytes);
    lfm_q4_pack_matrix(w, n, k, img, 128);
    memcpy(xr, x, (size_t)m * (size_t)k * sizeof(float));
    for (t = 0; t < m; t++) {
        if (lfm_act_i8_rot(xr + (size_t)t * (size_t)k, k, 128, q + (size_t)t * (size_t)k,
                           ascale + (size_t)t * groups) != 0) {
            printf("FAIL gemm N=%d K=%d M=%d: quant\n", n, k, m);
            g_fail++;
            goto out;
        }
    }
    lfm_q4_gemm_i8(img, n, k, q, m, ascale, y_ref);
    t0 = now_sec();
    lfm_hvx_gemm_i4(img, n, k, q, m, ascale, y);
    t1 = now_sec();
    for (t = 0; t < m; t++) {
        for (r = 0; r < n; r++) {
            size_t idx = (size_t)t * (size_t)n + (size_t)r;
            float d = fabsf(y[idx] - y_ref[idx]);
            float s = fabsf(y_ref[idx]) > 1.0f ? fabsf(y_ref[idx]) : 1.0f;
            if (d > max_abs)
                max_abs = d;
            if (d / s > max_rel)
                max_rel = d / s;
        }
        for (i = 0; i < k; i++)
            xd[(size_t)t * (size_t)k + (size_t)i] = (float)q[(size_t)t * (size_t)k + (size_t)i] *
                                                   ascale[(size_t)t * groups + (size_t)(i / 32)];
        for (r = 0; r < n; r++) {
            size_t idx = (size_t)t * (size_t)n + (size_t)r;
            y_f32[idx] = dot_f32(w + (size_t)r * (size_t)k, x + (size_t)t * (size_t)k, k);
            lfm_q4_dequant_row(lfm_q4_row_qs(img, n, k, r), lfm_q4_row_scales(img, n, k, r), k,
                               wr);
            y_dq[idx] = dot_f32(wr, xd + (size_t)t * (size_t)k, k);
        }
        {
            float c = cos_f32(y + (size_t)t * (size_t)n, y_dq + (size_t)t * (size_t)n, n);
            if (c < min_cos_dq)
                min_cos_dq = c;
        }
        {
            float c = cos_f32(y + (size_t)t * (size_t)n, y_f32 + (size_t)t * (size_t)n, n);
            if (c < min_cos_real)
                min_cos_real = c;
        }
    }
    lfm_hvx_gemv_i4(img, n, k, q, ascale, y_gemv);
    {
        float d1 = 0.0f;
        for (r = 0; r < n; r++)
            if (fabsf(y_gemv[r] - y[r]) > d1)
                d1 = fabsf(y_gemv[r] - y[r]);
        if (d1 > LFM_TOL) {
            printf("FAIL gemm N=%d K=%d M=%d: gemv/gemm row0 diff %.3e\n", n, k, m, (double)d1);
            g_fail++;
        }
    }
    printf("gemm N=%-4d K=%-4d M=%d max_abs=%.3e max_rel=%.3e cos_dq=%.7f cos_int4=%.6f ", n, k,
           m, (double)max_abs, (double)max_rel, (double)min_cos_dq, (double)min_cos_real);
    if (t1 > t0)
        printf("time=%.3f ms\n", (t1 - t0) * 1e3);
    else
        printf("time=n/a\n");
    if (max_abs > LFM_TOL || max_rel > LFM_TOL) {
        printf("FAIL gemm N=%d K=%d M=%d: ref mismatch\n", n, k, m);
        g_fail++;
    }
    if (min_cos_dq < LFM_COS) {
        printf("FAIL gemm N=%d K=%d M=%d: cos_dq %.7f < %.7f\n", n, k, m, (double)min_cos_dq,
               (double)LFM_COS);
        g_fail++;
    }
    if (min_cos_real < LFM_COS_INT4) {
        printf("FAIL gemm N=%d K=%d M=%d: cos_int4 %.7f < %.7f\n", n, k, m, (double)min_cos_real,
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
    xfree(y_gemv);
    xfree(y_f32);
    xfree(y_dq);
    xfree(xd);
    xfree(wr);
    xfree(img);
    return g_fail == 0;
}

static void bench(int n, int k, int m, int iters)
{
    size_t img_bytes = lfm_q4_image_bytes(n, k);
    size_t groups = (size_t)(k / 32);
    uint8_t *img = xalloc(img_bytes);
    int8_t *q = xalloc((size_t)m * (size_t)k + 256u);
    float *ascale = xalloc((size_t)m * groups * sizeof(float));
    float *y = xalloc((size_t)m * (size_t)n * sizeof(float));
    double best = 0.0;
    int it;
    int i;

    if (!img || !q || !ascale || !y) {
        printf("FAIL gemm bench: alloc\n");
        g_fail++;
        goto out;
    }
    memset(img, 0x5A, img_bytes);
    for (i = 0; i < m * k; i++)
        q[i] = (int8_t)((int)(rnd_u32() & 0xFFu) - 128);
    for (i = 0; i < m * (int)groups; i++)
        ascale[i] = 1e-6f;
    lfm_hvx_gemm_i4(img, n, k, q, m, ascale, y);
    for (it = 0; it < iters; it++) {
        double t0 = now_sec();
        lfm_hvx_gemm_i4(img, n, k, q, m, ascale, y);
        {
            double dt = now_sec() - t0;
            if (best == 0.0 || dt < best)
                best = dt;
        }
    }
    printf("bench gemm N=%d K=%d M=%d: ", n, k, m);
#if defined(__HVX__)
    printf("%d iters, %.1f MMAC (время: отчёт hexagon-sim)\n", iters,
           (double)n * (double)k * (double)m / 1e6);
    (void)best;
#else
    if (best > 0.0)
        printf("%.3f ms, %.2f GMAC/s\n", best * 1e3,
               (double)n * (double)k * (double)m / best / 1e9);
    else
        printf("n/a (clock() не идёт)\n");
#endif
out:
    xfree(img);
    xfree(q);
    xfree(ascale);
    xfree(y);
}

int main(int argc, char **argv)
{
    static const int cases[][3] = {
        {64, 512, 1}, {64, 512, 2}, {96, 640, 3}, {128, 768, 4},
        {128, 1024, 5}, {64, 2048, 8},
    };
    static const int ncases = (int)(sizeof cases / sizeof cases[0]);
    int i;

    if (argc > 1 && argv[1][0] == 'b') {
#if defined(__HVX__)
        bench(128, 1024, 8, argc > 2 ? atoi(argv[2]) : 2);
#else
        bench(1024, 2048, 8, argc > 2 ? atoi(argv[2]) : 2);
#endif
        return g_fail ? 1 : 0;
    }
    printf("gemm_i4 stand: %s\n",
#if defined(__HVX__)
           "hexagon HVX"
#else
           "portable C11"
#endif
    );
    for (i = 0; i < ncases; i++)
        run_case(cases[i][0], cases[i][1], cases[i][2]);
#if defined(__HVX__)
    bench(128, 1024, 8, 2);
#else
    bench(1024, 2048, 8, 2);
    bench(7680, 2048, 1, 2);
#endif
    if (g_fail) {
        printf("gemm FAIL (%d)\n", g_fail);
        return 1;
    }
    printf("gemm PASS\n");
    return 0;
}
