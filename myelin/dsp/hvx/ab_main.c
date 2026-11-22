/*
 * A/B driver: int4 GEMV HVX variants, interleaved inside one series.
 */
#include "lfmgemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static inline uint32_t lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

static int N = 7680, K = 2048;

static void build_weights(uint8_t *w, int8_t *wsigned)
{
    int r, k;
    uint32_t st = 0x12345678u;
    for (r = 0; r < N; r++) {
        uint8_t *row = w + (size_t)r * (K / 2);
        int8_t *sr = wsigned + (size_t)r * K;
        for (k = 0; k < K; k += 2) {
            int lo = (int)(lcg(&st) % 16) - 8;
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

/* variant table: {pf_kb, rows, usepf, label} */
static const int VAR[][4] = {
    { 8,  2, 1, 1 },   /* V0 pf8 rows2 thr1 */
    { 8,  2, 1, 2 },   /* V1 thr2 */
    { 8,  2, 1, 3 },   /* V2 thr3 */
    { 8,  2, 1, 4 },   /* V3 thr4 */
    { 8,  2, 1, 8 },   /* V4 thr8 */
};
#define NVAR ((int)(sizeof(VAR) / sizeof(VAR[0])))

int main(int argc, char **argv)
{
    int h = -1, rc, i, m, r, v, round;
    int times[8];
    size_t wbytes = (size_t)N * (K / 2);
    uint8_t *w, *dst;
    int8_t *wsigned, *act;
    int32_t *acc;
    float *srow;
    unsigned char *sbuf;
    int rounds = (argc > 1) ? atoi(argv[1]) : 3;
    int reps = (argc > 2) ? atoi(argv[2]) : 2;
    double best_us[NVAR];
    int pf_used = 0;

    printf("shape N=%d K=%d weights=%.2f MB\n", N, K, wbytes / 1e6);
    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }

    w       = malloc(wbytes);
    wsigned = malloc((size_t)N * K);
    act     = malloc((size_t)8 * K + 128);
    dst     = malloc((size_t)8 * N + 128);
    acc     = malloc((size_t)8 * N * sizeof(int32_t));
    srow    = malloc((size_t)N * sizeof(float));
    sbuf    = malloc(16 + (size_t)N * sizeof(float));

    build_weights(w, wsigned);
    {
        uint32_t st = 0xdeadbeefu;
        for (i = 0; i < 8 * K; i++) act[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }
    for (r = 0; r < N; r++) srow[r] = 0.001f + (float)(r % 7) * 0.0001f;

    rc = lfmgemv_alloc_buf((int)wbytes, &h);
    printf("alloc_buf rc=%d h=%d\n", rc, h);
    if (rc != 0) return 11;

    rc = lfmgemv_set_shape(h, N, K, 0);
    if (rc != 0) { printf("set_shape rc=%d\n", rc); return 12; }

    {
        double t0 = now_us();
        size_t off = 0, chunk = 256 * 1024;
        while (off < wbytes) {
            size_t n = wbytes - off < chunk ? wbytes - off : chunk;
            rc = lfmgemv_load_buf(h, (int)off, w + off, (int)n);
            if (rc != 0) { printf("load_buf rc=%d\n", rc); return 13; }
            off += n;
        }
        printf("upload: %.1f ms for %.2f MB\n", (now_us() - t0) / 1000.0, wbytes / 1e6);
    }
    pf_used = 0;

    /* ---- correctness, int32 raw accumulators ---- */
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
        printf("correctness M=%d: elems=%d mismatches=%d max_abs_err=%d\n",
               m, N * m, bad, maxerr);
    }

    /* ---- raw bandwidth probes ---- */
    {
        int res[8];
        struct { int kind; int mib; const char *label; } bp[] = {
            { 2,  0,    "loads only (whole 7.9MB buffer, dependent)" },
            { 3,  0,    "loads + unpack + 2 vrmpy (kb pattern)"     },
            { 6,  128,  "linear 128 MiB, 1 chain (latency bound)"   },
            { 9,  128,  "linear 128 MiB, 8 chains, no l2fetch"      },
        };
        int bi;
        for (bi = 0; bi < 4; bi++) {
            uint64 bytes;
            rc = lfmgemv_probe(h, bp[bi].kind, bp[bi].mib, res, 8);
            if (rc != 0) { printf("probe %d rc=%d\n", bp[bi].kind, rc); continue; }
            bytes = (bp[bi].kind == 6 || bp[bi].kind == 7 || bp[bi].kind == 9)
                    ? (uint64)res[6] : wbytes;
            printf("bw[%d] %-42s %8.2f ms  %6.2f GB/s  (pc=%d)\n",
                   bp[bi].kind, bp[bi].label, res[1] / 1000.0,
                   (double)bytes / (res[1] * 1e3), res[0]);
        }
    }

    {
        int res[8];
        lfmgemv_probe(h, 8, 8 | (1 << 8) | (1 << 16), res, 8);
        lfmgemv_probe(h, 9, 128, res, 8);
        printf("bw[9] linear 128 MiB, 8 chains + l2fetch 8 KB     %8.2f ms  %6.2f GB/s  (pc=%d)\n",
               res[1] / 1000.0, (double)res[6] / (res[1] * 1e3), res[0]);
    }

    /* ---- switch to int8 output and A/B the kernel variants ---- */
    rc = lfmgemv_set_shape(h, N, K, 1);
    if (rc != 0) { printf("set_shape(i8) rc=%d\n", rc); return 15; }
    memcpy(sbuf, &N, 4);
    memcpy(sbuf + 4, &K, 4);
    { float sa = 0.05f, so = 0.5f;
      memcpy(sbuf + 8, &sa, 4); memcpy(sbuf + 12, &so, 4); }
    memcpy(sbuf + 16, srow, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(h, sbuf, 16 + (int)N * 4);
    if (rc != 0) { printf("set_scale rc=%d\n", rc); return 16; }
    pf_used = 0;

    for (v = 0; v < NVAR; v++) best_us[v] = 1e18;

    for (round = 0; round < rounds; round++) {
        for (i = 0; i < NVAR; i++) {
            v = (round & 1) ? (NVAR - 1 - i) : i;     /* alternate direction */
            lfmgemv_probe(h, 8, VAR[v][0] | (VAR[v][1] << 8) | (VAR[v][2] << 16) | (VAR[v][3] << 24), times, 8);
            rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, K, reps,
                              dst, N, times, 8);
            if (rc != 0) { printf("gemv v%d rc=%d\n", v, rc); return 17; }
            if ((double)times[0] < best_us[v]) best_us[v] = times[0];
        }
        usleep(200000);
    }

    printf("\n%-4s %-22s %10s %10s %9s %9s\n",
           "var", "config", "kernel_us", "GB/s", "GMAC/s", "cyc/row");
    for (v = 0; v < NVAR; v++) {
        char cfg[32];
        snprintf(cfg, sizeof(cfg), "pf=%d rows=%d thr=%d", VAR[v][0], VAR[v][1], VAR[v][3]);
        printf("V%-3d %-22s %10.0f %10.2f %9.1f %9.0f\n", v, cfg, best_us[v],
               (double)wbytes / (best_us[v] * 1e3),
               (double)N * K / (best_us[v] * 1e3),
               best_us[v] * 1.42e3 / N);
    }

    /* ---- int8 weights, identical harness: my int8 kernel vs my int4 kernel ---- */
    {
        int h8 = -1, hh, best8 = 0, worst8 = 0;
        size_t w8bytes = (size_t)N * K;
        uint8_t *w8 = malloc(w8bytes);
        int8_t *w8signed = malloc(w8bytes);
        int times8[8];
        uint32_t st = 0x11223344u;

        for (r = 0; r < N; r++) {
            uint8_t *row = w8 + (size_t)r * K;
            int8_t *sr = w8signed + (size_t)r * K;
            for (i = 0; i < K; i++) {
                int v = (int)(lcg(&st) % 255) - 127;
                sr[i] = (int8_t)v;
                row[i] = (uint8_t)(v + 128);
            }
        }
        rc = lfmgemv_alloc_buf((int)w8bytes, &h8);
        if (rc != 0) { printf("int8 alloc rc=%d\n", rc); }
        else {
            rc = lfmgemv_set_shape(h8, N, K, 1 | 2);
            if (rc != 0) { printf("int8 set_shape rc=%d\n", rc); return 20; }
            {
                size_t off = 0, chunk = 256 * 1024;
                while (off < w8bytes) {
                    size_t n = w8bytes - off < chunk ? w8bytes - off : chunk;
                    rc = lfmgemv_load_buf(h8, (int)off, w8 + off, (int)n);
                    if (rc != 0) { printf("int8 load rc=%d\n", rc); return 21; }
                    off += n;
                }
            }
            rc = lfmgemv_set_scale(h8, sbuf, 16 + (int)N * 4);
            if (rc != 0) { printf("int8 set_scale rc=%d\n", rc); return 22; }

            /* correctness of the int8 path (raw int32) */
            lfmgemv_set_shape(h8, N, K, 2);
            rc = lfmgemv_gemv(h8, 1, (const unsigned char *)act, K, 1,
                              (unsigned char *)acc, N * (int)sizeof(int32_t), times8, 8);
            if (rc != 0) { printf("int8 gemv rc=%d\n", rc); return 23; }
            {
                int bad = 0, maxerr = 0;
                for (r = 0; r < N; r++) {
                    int32_t ref = dot_ref(w8signed + (size_t)r * K, act, K);
                    int32_t got = acc[r];
                    int32_t e = ref - got;
                    if (e < 0) e = -e;
                    if (e > maxerr) maxerr = (int)e;
                    if (e) bad++;
                }
                printf("correctness int8 M=1: elems=%d mismatches=%d max_abs_err=%d\n",
                       N, bad, maxerr);
            }
            lfmgemv_set_shape(h8, N, K, 1 | 2);
            lfmgemv_set_scale(h8, sbuf, 16 + (int)N * 4);

            /* interleaved A/B: int4 (h) vs int8 (h8), same conditions */
            {
                int bi4 = 0, bi8 = 0;
                for (round = 0; round < rounds; round++) {
                    lfmgemv_probe(h, 8, 8 | (2 << 8) | (1 << 16), times8, 8);
                    if (lfmgemv_gemv(h, 1, (const unsigned char *)act, K, reps, dst, N, times8, 8) == 0)
                        if (bi4 == 0 || times8[0] < bi4) bi4 = times8[0];
                    usleep(100000);
                    lfmgemv_probe(h8, 8, 8 | (1 << 8) | (1 << 16), times8, 8);
                    if (lfmgemv_gemv(h8, 1, (const unsigned char *)act, K, reps, dst, N, times8, 8) == 0)
                        if (bi8 == 0 || times8[0] < bi8) bi8 = times8[0];
                    usleep(100000);
                }
                printf("AB interleaved: int4=%.0f us (%.2f GB/s)  int8=%.0f us (%.2f GB/s)  ratio=%.2fx\n",
                       (double)bi4, (double)wbytes / (bi4 * 1e3),
                       (double)bi8, (double)w8bytes / (bi8 * 1e3),
                       (double)bi8 / (double)bi4);
                best8 = bi8;
            }
            printf("int8 kernel (same harness): %.0f us  %.2f GB/s  %.1f GMAC/s\n",
                   (double)best8, (double)w8bytes / (best8 * 1e3),
                   (double)N * K / (best8 * 1e3));
            (void)hh;
            lfmgemv_release_buf(h8);
        }
    }

    /* ---- final: best config, spread, M scaling, RPC cost ---- */
    {
        int best_cfg = 8 | (4 << 8) | (1 << 16);   /* pf 8 KB, rows 4 */
        int t[8];
        double wall;
        lfmgemv_probe(h, 8, best_cfg, t, 8);
        printf("\nfinal config: pf=8KB rows=4\n");
        lfmgemv_gemv(h, 1, (const unsigned char *)act, K, 8, dst, N, t, 8);
        printf("M=1: kernel min=%d us  avg=%d us  max=%d us | deint=%d us pack=%d us\n",
               t[0], t[2], t[1], t[3], t[4]);
        {
            double wmin = 1e18, wsum = 0;
            int n = 20, kk;
            for (kk = 0; kk < n; kk++) {
                double t0 = now_us();
                lfmgemv_gemv(h, 1, (const unsigned char *)act, K, 1, dst, N, t, 8);
                wall = now_us() - t0;
                if (wall < wmin) wmin = wall;
                wsum += wall;
                usleep(2000);
            }
            printf("M=1: call end-to-end: min=%.0f us avg=%.0f us | kernel=625 us -> RPC overhead min=%.0f us\n",
                   wmin, wsum / n, wmin - 625);
        }
        for (m = 1; m <= 4; m *= 2) {
            lfmgemv_gemv(h, m, (const unsigned char *)act, m * K, 8, dst, m * N, t, 8);
            printf("M=%d: kernel min=%d us (%.2f GB/s weights, %.2f GB/s weights+act, %.1f GMAC/s)\n",
                   m, t[0], (double)wbytes / (t[0] * 1e3),
                   ((double)wbytes + (double)m * K + (double)m * N) / (t[0] * 1e3),
                   (double)m * N * K / (t[0] * 1e3));
        }
        usleep(200000);
    }

    /* ---- pure RPC cost: trivial kernel (N=64) and tight back-to-back calls ---- */
    {
        int hs = -1, t2[8], kk;
        size_t wb = (size_t)64 * (K / 2);
        uint8_t *ws = malloc(wb);
        double wmin = 1e18, wsum = 0;
        memcpy(ws, w, wb);
        if (lfmgemv_alloc_buf((int)wb, &hs) == 0 &&
            lfmgemv_set_shape(hs, 64, K, 1) == 0 &&
            lfmgemv_load_buf(hs, 0, ws, (int)wb) == 0) {
            unsigned char *ss = malloc(16 + 64 * 4);
            float s2[64];
            for (i = 0; i < 64; i++) s2[i] = 0.01f;
            memcpy(ss, &N, 4); memcpy(ss + 4, &K, 4);
            { float a = 0.05f, o2 = 0.5f; memcpy(ss + 8, &a, 4); memcpy(ss + 12, &o2, 4); }
            memcpy(ss + 16, s2, 64 * 4);
            lfmgemv_set_scale(hs, ss, 16 + 64 * 4);
            for (kk = 0; kk < 40; kk++) {
                double t0 = now_us();
                lfmgemv_gemv(hs, 1, (const unsigned char *)act, K, 1, dst, 64, t2, 8);
                {
                    double w = now_us() - t0;
                    if (w < wmin) wmin = w;
                    wsum += w;
                }
            }
            printf("RPC only (N=64 kernel ~0): min=%.0f us avg=%.0f us, kernel=%d us\n",
                   wmin, wsum / 40, t2[0]);
            lfmgemv_release_buf(hs);
            free(ss);
        } else printf("RPC probe: alloc/setup failed\n");
        free(ws);
    }

    /* QNN int8 reference for the same tensor */
    printf("\nQNN int8 ref: 1.37 ms -> %.1f GB/s (int8 weights 15.7 MB)\n",
           15.73 / 1.37);

    lfmgemv_release_buf(h);
    printf("done\n");
    return 0;
}
