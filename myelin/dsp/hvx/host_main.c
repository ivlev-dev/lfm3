/*
 * Host driver for int4 GEMV HVX benchmark.
 */
#include "lfmgemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    int rc;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
    printf("enable unsigned PD: rc=%d\n", rc);
    return rc;
}

/* deterministic pseudo random, explicit state (no globals: clang -O2 hoists them) */
static inline uint32_t lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static int N = 7680, K = 2048;

/* packed weights: offset-8 nibbles, lo = k even, hi = k odd */
static void build_weights(uint8_t *w, int8_t *wsigned)
{
    int r, k;
    uint32_t st = 0x12345678u;
    for (r = 0; r < N; r++) {
        uint8_t *row = w + (size_t)r * (K / 2);
        int8_t *sr = wsigned + (size_t)r * K;
        for (k = 0; k < K; k += 2) {
            int lo = (int)(lcg(&st) % 16) - 8;   /* -8..7 */
            int hi = (int)(lcg(&st) % 16) - 8;
            sr[k] = (int8_t)lo;
            sr[k + 1] = (int8_t)hi;
            row[k / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
        }
    }
}

static int32_t dot_ref(const int8_t *w, const int8_t *a, int k)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < k; i++) s += (int32_t)w[i] * (int32_t)a[i];
    return s;
}

int main(int argc, char **argv)
{
    int h = -1, rc;
    uint8_t *w, *wup;
    int8_t *wsigned, *act;
    uint8_t *dst;
    int32_t *acc;
    size_t wbytes = (size_t)N * (K / 2);
    size_t chunk = 256 * 1024;
    int reps = (argc > 1) ? atoi(argv[1]) : 20;
    int i, m, r;
    int times[8];
    float *srow;
    unsigned char *sbuf;

    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    wbytes = (size_t)N * (K / 2);
    printf("shape N=%d K=%d weights=%zu bytes (%.2f MB)\n", N, K, wbytes, wbytes / 1e6);

    if (enable_unsigned() != 0) return 10;

    w       = malloc(wbytes);
    wsigned = malloc((size_t)N * K);
    wup     = malloc(wbytes);
    act     = malloc((size_t)8 * K + 128);
    dst     = malloc((size_t)8 * N + 128);
    acc     = malloc((size_t)8 * N * sizeof(int32_t));
    srow    = malloc((size_t)N * sizeof(float));
    sbuf    = malloc(16 + (size_t)N * sizeof(float));

    build_weights(w, wsigned);
    {
        static const int pr[] = {0, 1, 33, 100};
        int j, q;
        for (j = 0; j < 4; j++) {
            const int8_t *sr = wsigned + (size_t)pr[j] * K;
            const uint8_t *wr = w + (size_t)pr[j] * (K / 2);
            long sw = 0;
            for (q = 0; q < K; q++) sw += sr[q];
            printf("row %3d: w[0..5]=%d,%d,%d  packed=%02x %02x %02x  rowsum=%ld\n",
                   pr[j], sr[0], sr[1], sr[2], wr[0], wr[1], wr[2], sw);
        }
        printf("act[0..5]=%d,%d,%d,%d  dot(row0)=%d dot(row33)=%d\n",
               act[0], act[1], act[2], act[3],
               dot_ref(wsigned, act, K), dot_ref(wsigned + 33 * K, act, K));
    }
    for (r = 0; r < N; r++) srow[r] = 0.001f + (float)(r % 7) * 0.0001f;
    {
        uint32_t st = 0xdeadbeefu;
        for (i = 0; i < 8 * K; i++) act[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }

    rc = lfmgemv_alloc_buf((int)wbytes, &h);
    printf("alloc_buf rc=%d h=%d\n", rc, h);
    if (rc != 0) return 11;

    rc = lfmgemv_set_shape(h, N, K, 0);
    if (rc != 0) { printf("set_shape(int32) rc=%d\n", rc); return 12; }

    /* upload weights */
    {
        double t0 = now_us();
        size_t off = 0;
        while (off < wbytes) {
            size_t n = wbytes - off < chunk ? wbytes - off : chunk;
            rc = lfmgemv_load_buf(h, (int)off, w + off, (int)n);
            if (rc != 0) { printf("load_buf rc=%d off=%zu\n", rc, off); return 13; }
            off += n;
        }
        printf("weights upload: %.1f us total (%.1f MB), rpc only\n",
               now_us() - t0, wbytes / 1e6);
    }

    /* --------- correctness: int32 mode, M=1 and M=4 --------- */
    for (m = 1; m <= 4; m *= 2) {
        int bad = 0, maxerr = 0;
        rc = lfmgemv_gemv(h, m, (const unsigned char *)act, m * K, 1,
                          (unsigned char *)acc, m * N * (int)sizeof(int32_t), times, 8);
        if (rc != 0) { printf("gemv int32 M=%d rc=%d\n", m, rc); return 14; }
        for (r = 0; r < N; r++) {
            for (i = 0; i < m; i++) {
                int32_t ref = dot_ref(wsigned + (size_t)r * K, act + (size_t)i * K, K);
                int32_t got = acc[(size_t)i * N + r];
                int32_t e = ref - got;
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = (int)e;
                if (e != 0) bad++;
            }
        }
        printf("correctness M=%d: rows=%d elems=%d mismatches=%d max_abs_err=%d\n",
               m, N, N * m, bad, maxerr);
        if (bad) {
            int rr;
            static const int probe[] = {0, 1, 32, 33, 34, 35, 100};
            int32_t g1[7];
            lfmgemv_gemv(h, 1, (const unsigned char *)act, K, 1,
                         (unsigned char *)acc, N * (int)sizeof(int32_t), times, 8);
            for (rr = 0; rr < 7; rr++) g1[rr] = acc[probe[rr]];
            lfmgemv_gemv(h, m, (const unsigned char *)act, m * K, 1,
                         (unsigned char *)acc, m * N * (int)sizeof(int32_t), times, 8);
            for (rr = 0; rr < 7; rr++) {
                int32_t ref = dot_ref(wsigned + (size_t)probe[rr] * K, act, K);
                printf("   row %4d ref=%8d  got_M1=%8d  got_M%d=%8d\n",
                       probe[rr], ref, g1[rr], m, acc[probe[rr]]);
            }
            {
                int nbad = 0, first = -1, last = -1;
                for (r = 0; r < N; r++) {
                    int32_t ref = dot_ref(wsigned + (size_t)r * K, act, K);
                    if (ref != acc[r]) {
                        nbad++;
                        if (first < 0) first = r;
                        last = r;
                    }
                }
                printf("   lane0: nbad=%d first=%d last=%d\n", nbad, first, last);
            }
            int shown = 0;
            for (r = 0; r < N && shown < 3; r++) {
                int32_t ref = dot_ref(wsigned + (size_t)r * K, act, K);
                int32_t got = acc[(size_t)0 * N + r];
                if (ref != got) {
                    printf("   row %d: ref=%d got=%d\n", r, ref, got);
                    shown++;
                }
            }
        }
    }

    /* --------- perf: int8 output mode --------- */
    rc = lfmgemv_set_shape(h, N, K, 1);
    if (rc != 0) { printf("set_shape(i8) rc=%d\n", rc); return 15; }
    memcpy(sbuf, &N, 4);
    memcpy(sbuf + 4, &K, 4);
    {
        float sa = 0.05f, so = 0.5f;
        memcpy(sbuf + 8, &sa, 4);
        memcpy(sbuf + 12, &so, 4);
    }
    memcpy(sbuf + 16, srow, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(h, sbuf, 16 + (int)N * 4);
    printf("set_scale rc=%d\n", rc);

    printf("\n%-6s %-8s %-10s %-10s %-10s %-9s %-9s\n",
           "M", "reps", "kernel_us", "wall_us", "GB/s", "GMAC/s", "RPC_us");
    for (m = 1; m <= 4; m *= 2) {
        double t0, tw;
        double wall;
        int best;

        rc = lfmgemv_gemv(h, m, (const unsigned char *)act, m * K, reps,
                          dst, m * N, times, 8);
        if (rc != 0) { printf("gemv i8 M=%d rc=%d\n", m, rc); return 16; }

        t0 = now_us();
        rc = lfmgemv_gemv(h, m, (const unsigned char *)act, m * K, 1,
                          dst, m * N, times, 8);
        tw = now_us() - t0;
        best = times[0];
        wall = tw;

        printf("%-6d %-8d %-10d %-10.1f %-10.1f %-9.1f %-9.1f\n",
               m, reps, best, wall,
               (double)wbytes / (best * 1e3),
               (double)m * N * K / (best * 1e3),
               wall - best);
        (void)t0;
    }

    /* M=1 detail */
    rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, K, reps, dst, N, times, 8);
    printf("\nM=1 detail: best=%d worst=%d avg=%d deint=%d pack=%d us (nvec=%d)\n",
           times[0], times[1], times[2], times[3], times[4], times[5]);

    /* timing without deinterleave bias: single row buffer check */
    {
        long sum = 0;
        for (i = 0; i < N; i++) sum += (int8_t)dst[i];
        printf("dst checksum=%ld (nonzero sanity)\n", sum);
    }

    lfmgemv_release_buf(h);
    printf("done\n");
    return 0;
}
