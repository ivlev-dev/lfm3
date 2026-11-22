/*
 * Prototype: one FastRPC call per FFN layer (gate/up/down int4 + SiLU),
 * multi-context HVX sweep, and RPC cost with persistent buffers.
 */
#include "lfmgemv.h"
#include "silu_lut.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "rpcmem.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static inline uint32_t lcg(uint32_t *st)
{
    uint32_t x = *st; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *st = x; return x;
}
static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

#define NH 7680     /* hidden */
#define K  2048     /* model width */

/* ---- same integer pipeline as the DSP ---- */
static inline int32_t mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((uint32_t)lo * (uint32_t)b) >> 8);
}
static int32_t silu_q16(int32_t zq)
{
    int32_t off, idx, frac, a, b;
    if (zq > 8 * 65536) zq = 8 * 65536;
    if (zq < -8 * 65536) zq = -8 * 65536;
    off = zq + 8 * 65536;
    idx = off >> 10;
    frac = off & 1023;
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> 10);
}

static int32_t mark_q16(float m)
{
    unsigned q;
    if (m < 0.0f) m = 0.0f;
    if (m > 0.99997f) m = 0.99997f;
    q = (unsigned)(m * 65536.0f + 0.5f);
    if (q > 65535u) q = 65535u;
    return (int32_t)q;
}
static int32_t mark_q30(float m)
{
    int64_t q;
    if (m < 0.0f) m = 0.0f;
    if (m > 0.99997f) m = 0.99997f;
    q = (int64_t)((double)m * 1073741824.0 + 0.5);
    if (q > 0x7fffffffLL) q = 0x7fffffffLL;
    return (int32_t)q;
}

/* pack int4 rows: row-major [N][K], offset-8 nibbles */
static void pack_int4(uint8_t *w, const int8_t *ws, int n, int k, uint32_t seed)
{
    uint32_t st = seed;
    int r, i;
    for (r = 0; r < n; r++) {
        uint8_t *row = w + (size_t)r * (k / 2);
        int8_t *sr = (int8_t *)ws + (size_t)r * k;
        for (i = 0; i < k; i += 2) {
            int lo = (int)(lcg(&st) % 16) - 8;
            int hi = (int)(lcg(&st) % 16) - 8;
            sr[i] = (int8_t)lo; sr[i + 1] = (int8_t)hi;
            row[i / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
        }
    }
}

static int32_t dot_i8(const int8_t *w, const int8_t *a, int k)
{
    int32_t s = 0; int i;
    for (i = 0; i < k; i++) s += (int32_t)w[i] * (int32_t)a[i];
    return s;
}

int main(int argc, char **argv)
{
    int hg = -1, hu = -1, hd = -1, rc, i, r, round;
    int times[8], info[8];
    size_t wb = (size_t)NH * (K / 2);
    uint8_t *wg, *wu, *wd;
    int8_t *sg, *su, *sd, *x;
    int8_t *mid_out, *dst_out;
    int32_t *acc;
    float *sgw, *suw, *sdw;
    unsigned char *sbuf;
    int rounds = (argc > 1) ? atoi(argv[1]) : 3;
    const float s_mid = 0.0625f, s_out = 0.5f;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }

    wg = malloc(wb); wu = malloc(wb); wd = malloc(wb);
    sg = malloc((size_t)NH * K); su = malloc((size_t)NH * K); sd = malloc((size_t)K * NH);
    x = malloc(K + 128);
    mid_out = malloc(NH + 128);
    dst_out = malloc(K + 128);
    acc = malloc((size_t)4 * NH * sizeof(int32_t));
    sgw = malloc((size_t)NH * sizeof(float));
    suw = malloc((size_t)NH * sizeof(float));
    sdw = malloc((size_t)K * sizeof(float));
    sbuf = malloc(16 + (size_t)NH * sizeof(float));

    pack_int4(wg, sg, NH, K, 0x1111u);
    pack_int4(wu, su, NH, K, 0x2222u);
    pack_int4(wd, sd, K, NH, 0x3333u);
    {
        uint32_t st = 0xabcdef01u;
        for (i = 0; i < K; i++) x[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }
    /* per-row weight scales: Q16 multiplier for gate/up is s_w (sa=so=1) */
    for (r = 0; r < NH; r++) {
        sgw[r] = 0.00010f + (float)(r % 13) * 0.000002f;
        suw[r] = 0.00010f + (float)(r % 11) * 0.000002f;
    }
    for (r = 0; r < K; r++) sdw[r] = 0.00010f + (float)(r % 9) * 0.000002f;

    /* ---- allocate + upload ---- */
    if ((rc = lfmgemv_alloc_buf((int)wb, &hg)) != 0) { printf("alloc g rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_alloc_buf((int)wb, &hu)) != 0) { printf("alloc u rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_alloc_buf((int)wb, &hd)) != 0) { printf("alloc d rc=%d\n", rc); return 11; }
    printf("handles g=%d u=%d d=%d (%.1f MB total int4)\n", hg, hu, hd, 3.0 * wb / 1e6);

    {
        struct { int h; uint8_t *src; } up[] = { {hg, wg}, {hu, wu}, {hd, wd} };
        for (i = 0; i < 3; i++) {
            double t0 = now_us();
            size_t off = 0, chunk = 256 * 1024;
            if ((rc = lfmgemv_set_shape(up[i].h, i == 2 ? K : NH, i == 2 ? NH : K, 1)) != 0) {
                printf("set_shape %d rc=%d\n", i, rc); return 12;
            }
            while (off < wb) {
                size_t n = wb - off < chunk ? wb - off : chunk;
                if ((rc = lfmgemv_load_buf(up[i].h, (int)off, up[i].src + off, (int)n)) != 0) {
                    printf("load %d rc=%d\n", i, rc); return 13;
                }
                off += n;
            }
            printf("upload %d: %.1f ms\n", i, (now_us() - t0) / 1000.0);
        }
    }

    /* gate/up: sa=1, so=1 -> qp = s_w  (dequantised units) */
    {
        float sa = 1.0f, so = 1.0f;
        int vNH = NH, vK = K;
        memcpy(sbuf, &vNH, 4); memcpy(sbuf + 4, &vK, 4);
        memcpy(sbuf + 8, &sa, 4); memcpy(sbuf + 12, &so, 4);
        memcpy(sbuf + 16, sgw, (size_t)NH * 4);
        if ((rc = lfmgemv_set_scale(hg, sbuf, 16 + NH * 4)) != 0) { printf("scale g rc=%d\n", rc); return 14; }
        memcpy(sbuf + 16, suw, (size_t)NH * 4);
        if ((rc = lfmgemv_set_scale(hu, sbuf, 16 + NH * 4)) != 0) { printf("scale u rc=%d\n", rc); return 14; }
        /* down: sa = s_mid, so = s_out */
        int vNH2 = NH, vK2 = K;
        memcpy(sbuf, &vK2, 4); memcpy(sbuf + 4, &vNH2, 4);
        memcpy(sbuf + 8, &s_mid, 4); memcpy(sbuf + 12, &s_out, 4);
        memcpy(sbuf + 16, sdw, (size_t)K * 4);
        if ((rc = lfmgemv_set_scale(hd, sbuf, 16 + K * 4)) != 0) { printf("scale d rc=%d\n", rc); return 14; }
    }

    /* ---- team info ---- */

    if (lfmgemv_team_info(info, 8) == 0)
        printf("team before start: state=%d hvx_rc=%d mode=%d created=%d threads=%d\n",
               info[0], info[1], info[2], info[3], info[4]);

    if (argc > 3 && strcmp(argv[3], "bw") == 0) {
        /* minimal bandwidth probe: N iterations of one GEMV, print min */
        double wmin = 1e18;
        int kk;
        lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16), times, 8);
        for (kk = 0; kk < 40; kk++) {
            double t0 = now_us();
            lfmgemv_gemv(hg, 1, (const unsigned char *)x, K, 1, (unsigned char *)dst_out, NH, times, 8);
            { double w = now_us() - t0; if (w < wmin) wmin = w; }
        }
        printf("BW pid=%d kernel_min=%d us  wall_min=%.0f us  %.2f GB/s\n",
               (int)getpid(), times[0], wmin, wb / (times[0] * 1e3));
        lfmgemv_release_buf(hg); lfmgemv_release_buf(hu); lfmgemv_release_buf(hd);
        return 0;
    }

    /* ---- 1. multi-context sweep on the gate matrix ---- */
    printf("\n%-6s %-12s %10s %9s %9s\n", "thr", "config", "us", "GB/s", "GMAC/s");
    {
        static const int TH[] = { 1, 2, 4, 8 };
        double best[4];
        int t;
        for (t = 0; t < 4; t++) best[t] = 1e18;
        for (round = 0; round < rounds; round++) {
            for (i = 0; i < 4; i++) {
                t = (round & 1) ? (3 - i) : i;
                lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16) | (TH[t] << 24), times, 8);
                rc = lfmgemv_gemv(hg, 1, (const unsigned char *)x, K, 3,
                                  (unsigned char *)dst_out, NH, times, 8);
                if (rc != 0) { printf("gemv thr=%d rc=%d\n", TH[t], rc); return 15; }
                if ((double)times[0] < best[t]) best[t] = times[0];
            }
            usleep(150000);
        }
        for (t = 0; t < 4; t++)
            printf("%-6d %-12s %10.0f %9.2f %9.1f\n", TH[t], "pf8KB r2",
                   best[t], wb / (best[t] * 1e3), (double)NH * K / (best[t] * 1e3));
        lfmgemv_team_info(info, 8);
        printf("team after start: state=%d hvx_rc=%d mode=%d created=%d thr_rc=%d hvx_units=%d\n",
               info[0], info[1], info[2], info[3], info[5], info[6]);
    }

    /* ---- 2. correctness of the three GEMMs (int32, bit exact) ---- */
    lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16) | (2 << 24), times, 8);
    {
        struct { int h; int8_t *ws; int n; int k; const char *nm; } m3[] = {
            { hg, sg, NH, K,  "gate" },
            { hu, su, NH, K,  "up  " },
            { hd, sd, K,  NH, "down" },
        };
        int mi;
        for (mi = 0; mi < 3; mi++) {
            int bad = 0, maxerr = 0;
            lfmgemv_set_shape(m3[mi].h, m3[mi].n, m3[mi].k, 0);
            rc = lfmgemv_gemv(m3[mi].h, 1, (const unsigned char *)x, m3[mi].k, 1,
                              (unsigned char *)acc, m3[mi].n * 4, times, 8);
            if (rc != 0) { printf("gemv %s rc=%d\n", m3[mi].nm, rc); return 16; }
            for (r = 0; r < m3[mi].n; r++) {
                int32_t ref = dot_i8(m3[mi].ws + (size_t)r * m3[mi].k, x, m3[mi].k);
                int32_t e = ref - acc[r];
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = e;
                if (e) bad++;
            }
            printf("correctness %s: rows=%d mismatches=%d max_abs_err=%d\n",
                   m3[mi].nm, m3[mi].n, bad, maxerr);
            lfmgemv_set_shape(m3[mi].h, m3[mi].n, m3[mi].k, 1);
        }
    }
    /* restore scales after set_shape */
    {
        float sa = 1.0f, so = 1.0f;
        int vNH = NH, vK = K;
        memcpy(sbuf, &vNH, 4); memcpy(sbuf + 4, &vK, 4);
        memcpy(sbuf + 8, &sa, 4); memcpy(sbuf + 12, &so, 4);
        memcpy(sbuf + 16, sgw, (size_t)NH * 4); lfmgemv_set_scale(hg, sbuf, 16 + NH * 4);
        memcpy(sbuf + 16, suw, (size_t)NH * 4); lfmgemv_set_scale(hu, sbuf, 16 + NH * 4);
        int vNH2 = NH, vK2 = K;
        memcpy(sbuf, &vK2, 4); memcpy(sbuf + 4, &vNH2, 4);
        memcpy(sbuf + 8, &s_mid, 4); memcpy(sbuf + 12, &s_out, 4);
        memcpy(sbuf + 16, sdw, (size_t)K * 4); lfmgemv_set_scale(hd, sbuf, 16 + K * 4);
    }

    /* ---- 3. one-RPC FFN layer ---- */
    printf("\n");
    {
        double bestffn = 1e18, wallmin = 1e18, wallsum = 0;
        int n = 8, kk;
        int32_t mq_mid = (int32_t)(4096.0f / s_mid);

        /* CPU reference of the composed layer */
        int32_t *accg = malloc((size_t)NH * 4), *accu = malloc((size_t)NH * 4);
        int8_t *h = malloc(NH);
        for (r = 0; r < NH; r++) {
            accg[r] = dot_i8(sg + (size_t)r * K, x, K);
            accu[r] = dot_i8(su + (size_t)r * K, x, K);
        }
        for (i = 0; i < NH; i++) {
            int32_t zg = mulq16(accg[i], mark_q30(sgw[i]) >> 14);
            int32_t zu = mulq16(accu[i], mark_q30(suw[i]) >> 14);
            int32_t sv = silu_q16(zg);
            int32_t pr = mulq16(sv, zu);
            int32_t ps = pr >> 12;
            int32_t hv = mulq16(ps, mq_mid);
            if (hv > 127) hv = 127;
            if (hv < -127) hv = -127;
            h[i] = (int8_t)hv;
        }

        for (kk = 0; kk < n; kk++) {
            double t0 = now_us();
            rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)x, K, 1,
                             (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH, times, 8);
            {
                double w = now_us() - t0;
                if (w < wallmin) wallmin = w;
                wallsum += w;
            }
            if (rc != 0) { printf("ffn rc=%d\n", rc); return 17; }
            if ((double)times[0] < bestffn) bestffn = times[0];
            usleep(20000);
        }

        /* verify mid (int8 after SiLU) and the final output */
        {
            int badmid = 0, maxd = 0, badout = 0, maxo = 0;
            for (r = 0; r < NH; r++) {
                int d = (int)mid_out[r] - (int)h[r];
                if (d < 0) d = -d;
                if (d > maxd) maxd = d;
                if (d) badmid++;
            }
            for (r = 0; r < K; r++) {
                int32_t ref = dot_i8(sd + (size_t)r * NH, h, NH);
                int32_t o = mulq16(ref, mark_q16(sdw[r] * s_mid / s_out));
                int got = dst_out[r];
                int d;
                if (o > 127) o = 127;
                if (o < -127) o = -127;
                d = (int)o - got;
                if (d < 0) d = -d;
                if (d > maxo) maxo = d;
                if (d) badout++;
            }
            printf("FFN correctness: mid int8 mismatches=%d max|d|=%d\n", badmid, maxd);
            printf("FFN correctness: out int8 mismatches=%d max|d|=%d\n", badout, maxo);
        }
        free(accg); free(accu); free(h);
        (void)bestffn;
        printf("FFN layer (1 RPC): %.2f ms  [gate=%.2f up=%.2f silu=%.2f down=%.2f pack=%.2f ms]\n",
               wallmin / 1000.0, times[1] / 1e3, times[2] / 1e3, times[3] / 1e3,
               times[4] / 1e3, times[5] / 1e3);
        printf("FFN host wall: min=%.0f us avg=%.0f us over %d calls\n", wallmin, wallsum / n, n);
        printf("QNN reference for the same layer: 3 x 1.37 = 4.11 ms\n");
        /* amortisation: several layers' worth of work inside one RPC */
        {
            int rp[] = { 1, 2, 4, 8 };
            int z;
            for (z = 0; z < 4; z++) {
                double t0 = now_us();
                int kk2;
                double bb = 0;
                for (kk2 = 0; kk2 < 4; kk2++) {
                    t0 = now_us();
                    rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)x, K, rp[z],
                                     (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH, times, 8);
                    { double w = now_us() - t0; if (kk2 == 0 || w < bb) bb = w; }
                    usleep(15000);
                }
                if (rc != 0) break;
                printf("FFN reps=%d: wall=%7.0f us  per-layer amortised=%6.0f us  (dsp/rep=%d us)\n",
                       rp[z], bb, bb / rp[z], times[0]);
            }
        }
    }

    /* ---- 3b. resident output buffer (apps_mem_share_map, mapped once) ---- */
    {
        void *pdst = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        int rc2;
        if (!pdst) { printf("resident: rpcmem_alloc failed\n"); }
        else {
            int fd = rpcmem_to_fd(pdst);
            int8_t ref_out[K];
            double wmin = 1e18, wsum = 0;
            int kk, bad = 0, maxd = 0;

            rc2 = lfmgemv_attach_out(hg, fd, NH);
            rc2 = lfmgemv_attach_out(hd, fd, NH);   /* hd: K=2048 out fits */
            printf("resident: attach fd=%d rc=%d\n", fd, rc2);

            /* reference result via the ordinary rout path */
            lfmgemv_ffn(hg, hu, hd, (const unsigned char *)x, K, 1,
                        (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH, times, 8);
            memcpy(ref_out, dst_out, K);

            for (kk = 0; kk < 12; kk++) {
                double t0 = now_us();
                rc2 = lfmgemv_ffn_resident(hg, hu, hd, (const unsigned char *)x, K, 1,
                                           0, 0, times, 8);
                { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w; }
                if (rc2 != 0) { printf("ffn_resident rc=%d\n", rc2); break; }
                usleep(20000);
            }
            for (r = 0; r < K; r++) {
                int dd = (int)((int8_t *)pdst)[r] - (int)ref_out[r];
                if (dd < 0) dd = -dd;
                if (dd > maxd) maxd = dd;
                if (dd) bad++;
            }
            printf("FFN resident out: wall min=%.0f us avg=%.0f us | mismatches=%d max|d|=%d\n",
                   wmin, wsum / 12, bad, maxd);
            lfmgemv_detach_out(hg); lfmgemv_detach_out(hd);
            rpcmem_free(pdst);
        }
    }

    /* ---- 4. RPC cost: plain heap buffers vs persistent rpcmem ---- */
    {
        double wmin = 1e18, wsum = 0, kmin = 1e18;
        int kk;
        for (kk = 0; kk < 30; kk++) {
            double t0 = now_us();
            lfmgemv_gemv(hg, 1, (const unsigned char *)x, K, 1, (unsigned char *)dst_out, NH, times, 8);
            { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w;
              if (times[0] < kmin) kmin = times[0]; }
        }
        printf("\nRPC plain heap:   min=%.0f us avg=%.0f us  (kernel min=%.0f us)\n",
               wmin, wsum / 30, kmin);
    }
    {
        void *pd2 = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        if (pd2) {
            double wmin = 1e18, kmin = 1e18;
            int kk;
            lfmgemv_attach_out(hg, rpcmem_to_fd(pd2), NH);
            for (kk = 0; kk < 30; kk++) {
                double t0 = now_us();
                if (lfmgemv_gemv_resident(hg, 1, (const unsigned char *)x, K, 1, times, 8) == 0) {
                    double w = now_us() - t0;
                    if (w < wmin) wmin = w;
                    if (times[0] < kmin) kmin = times[0];
                }
            }
            printf("gemv_resident (no output transfer): min=%.0f us (kernel min=%.0f us)\n",
                   wmin, kmin);
            lfmgemv_detach_out(hg);
            rpcmem_free(pd2);
        }
    }
    {
        void *pa = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, K + 4096);
        void *pd = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        if (pa && pd) {
            double wmin = 1e18, wsum = 0, kmin = 1e18;
            int kk;
            memcpy(pa, x, K);
            for (kk = 0; kk < 30; kk++) {
                double t0 = now_us();
                lfmgemv_gemv(hg, 1, (const unsigned char *)pa, K, 1, (unsigned char *)pd, NH, times, 8);
                { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w;
                  if (times[0] < kmin) kmin = times[0]; }
            }
            printf("RPC rpcmem ION:   min=%.0f us avg=%.0f us  (kernel min=%.0f us), fd=%d\n",
                   wmin, wsum / 30, kmin, rpcmem_to_fd(pa));
            rpcmem_free(pa); rpcmem_free(pd);
        } else printf("rpcmem alloc failed\n");
    }

    lfmgemv_release_buf(hg); lfmgemv_release_buf(hu); lfmgemv_release_buf(hd);
    printf("done\n");
    return 0;
}
