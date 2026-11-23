#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/q4.h"
#include "../common/quant.h"
#include "hmx_ops.h"

#if defined(__HEXAGON__)
#define TEST_M 32
#define TEST_N 64
#define TEST_K 256
#else
#define TEST_M 32
#define TEST_N 256
#define TEST_K 2048
#endif
#define TEST_ROT 128

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint64_t rng_next(void)
{
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}

static float rng_gauss(void)
{
    double u1 = ((double)(rng_next() >> 11) + 1.0) / 9007199254740992.0;
    double u2 = (double)(rng_next() >> 11) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static void fill_gauss(float *x, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        x[i] = rng_gauss();
}

static void quant_i8_rows(const float *x, int rows, int k, int rot, int8_t *q, float *sc)
{
    float *tmp = (float *)malloc((size_t)k * sizeof(float));
    int r;
    if (tmp == NULL)
        abort();
    for (r = 0; r < rows; r++) {
        memcpy(tmp, x + (size_t)r * (size_t)k, (size_t)k * sizeof(float));
        lfm_rot_fwd_blocks_f32(tmp, k, rot);
        if (lfm_quant_i8_group32(tmp, k, q + (size_t)r * (size_t)k,
                                 sc + (size_t)r * (size_t)(k / LFM_QGROUP)) < 0)
            abort();
    }
    free(tmp);
}

static void quant_act_rows(const float *x, int rows, int k, int rot, int8_t *q, float *sc)
{
    int r;
    for (r = 0; r < rows; r++) {
        if (lfm_act_i8_rot(x + (size_t)r * (size_t)k, k, rot, q + (size_t)r * (size_t)k,
                           sc + (size_t)r * (size_t)(k / LFM_QGROUP)) != 0)
            abort();
    }
}

static void ref_i8(const int8_t *w, const float *wscale, const int8_t *act, const float *ascale,
                   int m, int n, int k, float *y)
{
    int t;
    int j;
    int ng = k / LFM_HMX_KC;
    for (t = 0; t < m; t++) {
        for (j = 0; j < n; j++) {
            float acc = 0.0f;
            int g;
            for (g = 0; g < ng; g++) {
                int32_t d = 0;
                int i;
                for (i = 0; i < LFM_HMX_KC; i++)
                    d += (int32_t)w[(size_t)j * (size_t)k + (size_t)g * 32u + (size_t)i] *
                         (int32_t)act[(size_t)t * (size_t)k + (size_t)g * 32u + (size_t)i];
                acc += ((float)d * wscale[(size_t)j * (size_t)ng + (size_t)g]) *
                       ascale[(size_t)t * (size_t)ng + (size_t)g];
            }
            y[(size_t)t * (size_t)n + (size_t)j] = acc;
        }
    }
}

static void ref_i8_acc(const int8_t *w, const int8_t *act, int m, int n, int k, int32_t *acc)
{
    int t;
    int j;
    for (t = 0; t < m; t++) {
        for (j = 0; j < n; j++) {
            int64_t d = 0;
            int i;
            for (i = 0; i < k; i++)
                d += (int64_t)w[(size_t)j * (size_t)k + (size_t)i] *
                     (int64_t)act[(size_t)t * (size_t)k + (size_t)i];
            acc[(size_t)t * (size_t)n + (size_t)j] = (int32_t)d;
        }
    }
}

static void ref_f64_dot(const float *w, const float *x, int m, int n, int k, float *y)
{
    int t;
    int j;
    for (t = 0; t < m; t++) {
        for (j = 0; j < n; j++) {
            double d = 0.0;
            int i;
            for (i = 0; i < k; i++)
                d += (double)w[(size_t)j * (size_t)k + (size_t)i] *
                     (double)x[(size_t)t * (size_t)k + (size_t)i];
            y[(size_t)t * (size_t)n + (size_t)j] = (float)d;
        }
    }
}

static float max_abs_diff(const float *a, const float *b, size_t count)
{
    float worst = 0.0f;
    size_t i;
    for (i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst)
            worst = d;
    }
    return worst;
}

static int32_t max_i32_diff(const int32_t *a, const int32_t *b, size_t count)
{
    int32_t worst = 0;
    size_t i;
    for (i = 0; i < count; i++) {
        int64_t d = (int64_t)a[i] - (int64_t)b[i];
        if (d < 0)
            d = -d;
        if (d > (int64_t)worst)
            worst = (int32_t)d;
    }
    return worst;
}

static double cos_sim(const float *a, const float *b, size_t count)
{
    double ab = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    size_t i;
    for (i = 0; i < count; i++) {
        ab += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    if (aa == 0.0 || bb == 0.0)
        return 0.0;
    return ab / sqrt(aa * bb);
}

static int test_tile_u8(void)
{
    int8_t q[32 * 32];
    int8_t w[32 * 32];
    uint8_t actbuf[LFM_HMX_ACT_BYTES];
    uint8_t wbuf[LFM_HMX_WB_BYTES];
    uint8_t out[LFM_HMX_OUT_BYTES];
    lfm_hmx_bias_block bias;
    int32_t sumw[32];
    int worst = 0;
    int s;
    int n;
    int i;

    for (i = 0; i < 32 * 32; i++) {
        q[i] = (int8_t)((int)(rng_next() % 11u) - 5);
        w[i] = (int8_t)((int)(rng_next() % 7u) - 3);
    }
    lfm_hmx_pack_act_ub(actbuf, q, 32, 32, 1);
    lfm_hmx_pack_wb_tile(wbuf, w, 32, 0, 0, 32);
    for (n = 0; n < 32; n++) {
        sumw[n] = lfm_hmx_sum_wb_tile_n(wbuf, n);
    }
    lfm_hmx_bias_zero_point(&bias, sumw, 32);
    if (lfm_hmx_tile_u8(actbuf, wbuf, 0, 1, &bias, out, 32) != 0)
        return 1;
    for (s = 0; s < 32; s++) {
        for (n = 0; n < 32; n++) {
            int32_t acc = 0;
            int exp;
            int got;
            for (i = 0; i < 32; i++)
                acc += (int32_t)w[n * 32 + i] * (int32_t)((int)q[s * 32 + i] + LFM_HMX_QZP);
            acc += (int32_t)bias.word1[n];
            exp = acc < 0 ? 0 : (acc > 255 ? 255 : (int)acc);
            got = (int)out[lfm_hmx_out_off(s, n)];
            if (abs(exp - got) > worst)
                worst = abs(exp - got);
        }
    }
    printf("tile  int8 u8-store M=32 K=32  max_abs_err=%d\n", worst);
    return worst != 0;
}

static int test_i8(int m, int n, int k, const float *w, const float *x)
{
    int8_t *wq = (int8_t *)malloc((size_t)n * (size_t)k);
    int8_t *aq = (int8_t *)malloc((size_t)m * (size_t)k);
    float *ws = (float *)malloc((size_t)n * (size_t)(k / LFM_QGROUP) * sizeof(float));
    float *as = (float *)malloc((size_t)m * (size_t)(k / LFM_QGROUP) * sizeof(float));
    float *y = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    float *yr = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    int32_t *acc = (int32_t *)malloc((size_t)m * (size_t)n * sizeof(int32_t));
    int32_t *accr = (int32_t *)malloc((size_t)m * (size_t)n * sizeof(int32_t));
    float err;
    int32_t aerr;
    int rc = 0;

    if (wq == NULL || aq == NULL || ws == NULL || as == NULL || y == NULL || yr == NULL ||
        acc == NULL || accr == NULL)
        abort();
    quant_i8_rows(w, n, k, TEST_ROT, wq, ws);
    quant_act_rows(x, m, k, TEST_ROT, aq, as);
    lfm_hmx_gemm_i8(wq, ws, n, k, aq, as, m, y);
    ref_i8(wq, ws, aq, as, m, n, k, yr);
    err = max_abs_diff(y, yr, (size_t)m * (size_t)n);
    lfm_hmx_gemm_i8_acc(wq, n, k, aq, m, acc);
    ref_i8_acc(wq, aq, m, n, k, accr);
    aerr = max_i32_diff(acc, accr, (size_t)m * (size_t)n);
    printf("int8  M=%d N=%d K=%d  max_abs_err=%.9g  int32_err=%d\n", m, n, k, (double)err,
           (int)aerr);
    if (err != 0.0f || aerr != 0)
        rc = 1;
    free(wq);
    free(aq);
    free(ws);
    free(as);
    free(y);
    free(yr);
    free(acc);
    free(accr);
    return rc;
}

static int test_i4_gate(int m, int n, int k, const float *w, const float *x, const uint8_t *img,
                        const int8_t *aq, const float *as)
{
    float *y = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    float *yr = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    double cs;
    float err;
    int rc = 0;

    if (y == NULL || yr == NULL)
        abort();
    lfm_hmx_gemm_i4(img, n, k, aq, as, m, y);
    ref_f64_dot(w, x, m, n, k, yr);
    cs = cos_sim(y, yr, (size_t)m * (size_t)n);
    err = max_abs_diff(y, yr, (size_t)m * (size_t)n);
    printf("int4  M=%d N=%d K=%d  cos=%.9f  max_abs_err=%.9g  gate=0.99\n", m, n, k, cs,
           (double)err);
    if (!(cs >= 0.99))
        rc = 1;
    free(y);
    free(yr);
    return rc;
}

static int test_prefill(int m, int n, int k, const uint8_t *img, const float *x)
{
    int8_t *aq = (int8_t *)malloc((size_t)m * (size_t)k);
    float *as = (float *)malloc((size_t)m * (size_t)(k / LFM_QGROUP) * sizeof(float));
    float *y = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    float *yp = (float *)malloc((size_t)m * (size_t)n * sizeof(float));
    lfm_hmx_layer_i4 layer;
    lfm_hmx_vtcm vtcm;
    float err;
    int rc = 0;

    if (aq == NULL || as == NULL || y == NULL || yp == NULL)
        abort();
    quant_act_rows(x, m, k, TEST_ROT, aq, as);
    layer.weights = img;
    layer.ascale = as;
    layer.n = n;
    layer.k = k;
    lfm_hmx_gemm_i4(img, n, k, aq, as, m, y);
    if (lfm_hmx_vtcm_alloc(&vtcm, lfm_hmx_prefill_vtcm_bytes()) != 0)
        abort();
    if (lfm_hmx_prefill_i4(&layer, aq, m, yp, &vtcm) != 0)
        rc = 1;
    err = max_abs_diff(y, yp, (size_t)m * (size_t)n);
    printf("prefill M=%d N=%d K=%d  vtcm=%zu B  vs gemm_i4 max_abs_err=%.9g\n", m, n, k,
           lfm_hmx_prefill_vtcm_bytes(), (double)err);
    if (err != 0.0f)
        rc = 1;
    lfm_hmx_vtcm_free(&vtcm);
    free(aq);
    free(as);
    free(y);
    free(yp);
    return rc;
}

int main(void)
{
    const int m = TEST_M;
    const int n = TEST_N;
    const int k = TEST_K;
    const int ng = k / LFM_HMX_KC;
    size_t img_bytes = lfm_q4_image_bytes(n, k);
    float *w = (float *)malloc((size_t)n * (size_t)k * sizeof(float));
    float *x = (float *)malloc((size_t)128 * (size_t)k * sizeof(float));
    int8_t *aq = (int8_t *)malloc((size_t)m * (size_t)k);
    float *as = (float *)malloc((size_t)m * (size_t)ng * sizeof(float));
    uint8_t *img = (uint8_t *)malloc(img_bytes);
    int rc = 0;

    if (w == NULL || x == NULL || aq == NULL || as == NULL || img == NULL)
        abort();
    memset(img, 0, img_bytes);
    fill_gauss(w, (size_t)n * (size_t)k);
    fill_gauss(x, (size_t)128 * (size_t)k);
    lfm_q4_pack_matrix(w, n, k, img, TEST_ROT);
    quant_act_rows(x, m, k, TEST_ROT, aq, as);

    if (test_i4_gate(m, n, k, w, x, img, aq, as) != 0)
        rc = 1;
    if (test_tile_u8() != 0)
        rc = 1;
    if (test_i8(m, n, k, w, x) != 0)
        rc = 1;
    if (test_i8(64, n, k, w, x) != 0)
        rc = 1;
    if (test_i8(128, n, k, w, x) != 0)
        rc = 1;
    if (test_prefill(m, n, k, img, x) != 0)
        rc = 1;
    if (test_prefill(64, n, k, img, x) != 0)
        rc = 1;
    if (test_prefill(128, n, k, img, x) != 0)
        rc = 1;

    printf("RESULT %s\n", rc == 0 ? "PASS" : "FAIL");
    free(w);
    free(x);
    free(aq);
    free(as);
    free(img);
    return rc;
}
