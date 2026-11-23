/*
 * q4k_test: Q4_K GEMV through the FastRPC runtime (set_q4k / gemv_q4k).
 *
 * The stand builds a synthetic but valid Q4_K matrix (GGUF 144-B blocks with
 * f16 d/dmin, 6-bit (sc,m) pairs and 128 random qs bytes), packs it with
 * q4k_pack_image and uploads it as a resident image.  Three oracles:
 *
 *   - host scalar twin of the DSP formula with the same int8/32 activation
 *     (q4k_gemv_scalar): must be bit-exact against the HVX kernel;
 *   - host f32 oracle (dequant every weight, f32 activation): cos target
 *     0.9999, the same gate as dsp/hvx/q4k_main.c;
 *   - timing from the DSP (best/worst/avg microseconds).
 *
 * Run on device: ./q4k_test [N] [K] [reps] [rot]
 *   rot (default 0) = 128: weights are requantised after the H128 Hadamard
 *   rotation and the activation is rotated with the matching forward pair
 *   before int8/32 quantisation (the product path does this with ROTQ a0=128).
 * Exit code: 0 pass, 2 numerics, 10.. RPC failure.
 */
#include "lfmgemv.h"
#include "q4k_impl.h"
#include "../quant/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rpcmem.h"

static uint32_t g_rng = 0x1234abcd;

static uint32_t rnd(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

static float frnd(void)
{
    return (float)(rnd() >> 8) / 16777216.0f - 0.5f;
}

/* one GGUF Q4_K block: d, dmin, 12 B of packed (sc,m), 128 B of qs */
static void make_block(uint8_t *b)
{
    uint16_t db = (uint16_t)(0x0b00u | (rnd() & 0x3ffu));
    uint16_t mb = (uint16_t)(0x0900u | (rnd() & 0x3ffu));
    int i;
    b[0] = (uint8_t)(db & 0xff);
    b[1] = (uint8_t)(db >> 8);
    b[2] = (uint8_t)(mb & 0xff);
    b[3] = (uint8_t)(mb >> 8);
    for (i = 4; i < Q4K_GGUF_BLOCK; i++)
        b[i] = (uint8_t)rnd();
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                  (uint32_t)sizeof(um));
}

static int upload(const uint8_t *img, int bytes, int N, int K, int *h)
{
    size_t off = 0, chunk = 256 * 1024;
    int rc;

    if ((rc = lfmgemv_alloc_buf(bytes, h)) != 0)
        return rc;
    if ((rc = lfmgemv_set_shape(*h, N, K, 0)) != 0)
        return rc;
    while (off < (size_t)bytes) {
        size_t n = (size_t)bytes - off < chunk ? (size_t)bytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, img + off, (int)n)) != 0)
            return rc;
        off += n;
    }
    return lfmgemv_set_q4k(*h);
}

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int K = argc > 2 ? atoi(argv[2]) : 2048;
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    int rot = argc > 4 ? atoi(argv[4]) : 0;
    int blocks, img_bytes, h = -1, rc, n, b, i, times[8];
    uint8_t *gguf, *img;
    int8_t *xq;
    float *x, *xs, *yref, *ysc, *ydsp;
    q4k_mat m;
    q4k_plan pl;
    int bad = 0;
    double cos_num = 0.0, cos_a = 0.0, cos_b = 0.0, maxerr = 0.0;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (N < 1 || K < 256 || (K & 255) != 0) {
        printf("bad shape N=%d K=%d\n", N, K);
        return 1;
    }
    blocks = K / 256;

    if ((rc = enable_unsigned()) != 0)
        printf("unsigned PD rc=%d (continuing)\n", rc);

    gguf = (uint8_t *)malloc((size_t)N * blocks * Q4K_GGUF_BLOCK);
    img = (uint8_t *)calloc(1, q4k_image_bytes(N, K) + 128);
    x = (float *)malloc((size_t)K * sizeof(float));
    xq = (int8_t *)malloc((size_t)K);
    xs = (float *)malloc((size_t)(K / 32) * sizeof(float));
    yref = (float *)malloc((size_t)N * sizeof(float));
    ysc = (float *)malloc((size_t)N * sizeof(float));
    ydsp = (float *)malloc((size_t)N * sizeof(float));
    if (!gguf || !img || !x || !xq || !xs || !yref || !ysc || !ydsp) {
        printf("host alloc failed\n");
        return 1;
    }

    for (n = 0; n < N * blocks; n++)
        make_block(gguf + (size_t)n * Q4K_GGUF_BLOCK);
    if (q4k_pack_image_rotated(gguf, N, K, rot, img) != 0) {
        printf("pack failed\n");
        return 1;
    }
    q4k_mat_init(&m, img, N, K);

    for (i = 0; i < K; i++)
        x[i] = frnd() * 4.0f;
    if (rot)
        lfm_rot_blocks_f32(x, K, rot);
    q4k_act_q8(x, K, xq, xs);
    if (q4k_plan_init(&pl, &m, xq, xs, 0) != 0) {
        printf("plan failed\n");
        return 1;
    }
    q4k_gemv_scalar(&m, &pl, xq, ysc, 0, N);
    q4k_gemv_f32act(&m, x, yref, 0, N);
    q4k_plan_free(&pl);

    rc = upload(img, q4k_image_bytes(N, K), N, K, &h);
    if (rc != 0) {
        printf("upload rc=%d\n", rc);
        return 10 + (rc & 7);
    }
    {
        unsigned char *act = (unsigned char *)malloc((size_t)K + (size_t)(K / 32) * 4);
        if (!act) return 1;
        memcpy(act, xq, (size_t)K);
        memcpy(act + K, xs, (size_t)(K / 32) * 4);
        rc = lfmgemv_gemv_q4k(h, act, K + (K / 32) * 4, reps,
                              (unsigned char *)ydsp, N * 4, times, 8);
        free(act);
    }
    if (rc != 0) {
        printf("gemv_q4k rc=%d\n", rc);
        return 10 + (rc & 7);
    }
    lfmgemv_release_buf(h);

    for (i = 0; i < N; i++) {
        double d = fabs((double)ydsp[i] - (double)ysc[i]);
        double r = fabs((double)ydsp[i] - (double)yref[i]);
        if (ydsp[i] != ysc[i])
            bad++;
        if (d > maxerr)
            maxerr = d;
        cos_num += (double)ydsp[i] * yref[i];
        cos_a += (double)ydsp[i] * ydsp[i];
        cos_b += (double)yref[i] * yref[i];
        (void)r;
    }
    {
        double cosv = cos_num / sqrt(cos_a * cos_b);
        printf("Q4K N=%d K=%d reps=%d rot=%d\n", N, K, reps, rot);
        printf("Q4K hvx vs scalar twin : mismatches=%d/%d  %s\n", bad, N,
               bad == 0 ? "OK" : "FAIL");
        printf("Q4K hvx vs f32 oracle  : cos=%.9f  %s\n", cosv,
               cosv >= 0.9999 ? "OK" : "FAIL");
        printf("Q4K dsp us: best=%d worst=%d avg=%d\n", times[0], times[1],
               times[2]);
        if (bad != 0 || cosv < 0.9999)
            return 2;
    }
    printf("Q4K RESULT: PASS\n");
    return 0;
}
