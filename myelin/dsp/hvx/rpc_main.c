/*
 * FastRPC latency stand: what does an empty call cost, and what can be
 * removed with the DSPRPC_CONTROL_LATENCY QOS modes.
 *
 * Answers three questions on the real phone:
 *   1. baseline (QOS off) vs PM_QOS / ADAPTIVE_QOS on an empty call
 *      (detach_out: 4 B of scalar in, no sequences at all).  POLL_QOS is
 *      reported separately: this libcdsprpc rejects it (rc=0xE);
 *   2. does the latency depend on the number of in/out sequences and on the
 *      payload size (0/1/3 sequences, 16 B .. 256 KiB);
 *   3. what that means per token: 48 calls x median.
 *
 * The QOS mode is sticky for the FastRPC session, so every configuration is
 * selected explicitly and the run ends with RPC_DISABLE_QOS.  Rounds are
 * interleaved (base / PM / adaptive) because the device drifts between
 * series: the median of medians is used for the verdict.
 *
 * Build: see build.sh.  Run on device: ./run.sh rpc_test [calls] [rounds]
 */
#include "lfmgemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include "rpcmem.h"

#define CALLS_DEFAULT  300
#define ROUNDS_DEFAULT 3

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y);
}

typedef struct {
    double min, median, p95, max;
} stats_t;

static stats_t stats(const double *v, int n)
{
    double *s = (double *)malloc((size_t)n * sizeof(double));
    stats_t r;
    int i;
    memcpy(s, v, (size_t)n * sizeof(double));
    qsort(s, (size_t)n, sizeof(double), cmp_double);
    r.min = s[0];
    r.median = n & 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    i = (int)(0.95 * n);                    /* nearest-rank p95 */
    if (i >= n) i = n - 1;
    r.p95 = s[i];
    r.max = s[n - 1];
    free(s);
    return r;
}

typedef int (*call_fn)(void *ctx);

static stats_t series(call_fn fn, void *ctx, int n, int *rc_out)
{
    double *v = (double *)malloc((size_t)n * sizeof(double));
    int i, rc = 0;
    for (i = 0; i < n; i++) {
        double t0 = now_us();
        int r = fn(ctx);
        v[i] = now_us() - t0;
        if (r != 0 && !rc) rc = r;
    }
    if (rc_out) *rc_out = rc;
    {
        stats_t s = stats(v, n);
        free(v);
        return s;
    }
}

static int set_qos(const char *name, uint32_t mode, uint32_t latency_us, int verbose)
{
    struct remote_rpc_control_latency lat;
    int rc;
    lat.enable = mode;
    lat.latency = latency_us;
    rc = remote_handle_control(DSPRPC_CONTROL_LATENCY, &lat, (uint32_t)sizeof(lat));
    if (verbose || rc)
        printf("  control(LATENCY mode=%u %u us) rc=0x%x%s\n", mode, latency_us,
               (unsigned)rc, rc ? "  <- not supported" : "");
    (void)name;
    return rc;
}

/* ---------------- call shapes ---------------- */

typedef struct {
    int  h, hres;                 /* h: plain handle, hres: resident-output */
    int  hg, hu, hd;              /* tiny FFN (1 in / 3 out) */
    int  N, K;
    unsigned char *act;
    unsigned char *dst;
    unsigned char *mid;
    int32_t *times;
    uint8_t *big;
    int   big_len;
} ctx_t;

static int c_empty(void *p)            /* 0 in seq, 0 out seq, 4 B scalar */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_detach_out(c->h);
}

static int c_team_info(void *p)        /* 0 in seq, 1 out seq (32 B) */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_team_info(c->times, 8);
}

static int c_probe(void *p)            /* 0 in seq, 1 out seq, trivial kernel */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_probe(c->h, 45, 0, c->times, 8);
}

static int c_load16(void *p)           /* 1 in seq (16 B) */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_load_buf(c->hres, 0, c->big, 16);
}

static int c_load4k(void *p)
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_load_buf(c->hres, 0, c->big, 4096);
}

static int c_load256k(void *p)
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_load_buf(c->hres, 0, c->big, 256 * 1024);
}

static int c_gemv(void *p)             /* 1 in seq + 2 out seq, tiny kernel */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_gemv(c->hres, 1, c->act, c->K, 1, c->dst, c->N, c->times, 8);
}

static int c_gemv_res(void *p)         /* 1 in seq + 1 out seq */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_gemv_resident(c->hres, 1, c->act, c->K, 1, c->times, 8);
}

static int c_ffn(void *p)              /* 1 in seq + 3 out seq (dst, mid, times) */
{
    ctx_t *c = (ctx_t *)p;
    return lfmgemv_ffn(c->hg, c->hu, c->hd, c->act, c->K, 1,
                       c->dst, c->N, c->mid, c->N, c->times, 8);
}

static void row(const char *label, stats_t s, int rc)
{
    printf("  %-34s %8.1f %8.1f %8.1f %8.1f%s\n",
           label, s.median, s.p95, s.min, s.max,
           rc ? "  rc!=0" : "");
}

int main(int argc, char **argv)
{
    int calls = argc > 1 ? atoi(argv[1]) : CALLS_DEFAULT;
    int rounds = argc > 2 ? atoi(argv[2]) : ROUNDS_DEFAULT;
    ctx_t c;
    int h = -1, hres = -1, rc, i, k;
    const int N = 256, Kd = 256;
    const int nb = N * Kd / 2;             /* 32 KiB of int4 weights */
    const int nbig = 1 << 20;              /* resident weight buffer */
    void *shared;
    stats_t base_all[8], pm_all[8], ad_all[8];
    double med_base, med_pm, med_ad;
    int qrc_poll;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }
    printf("FastRPC latency: %d calls/point, %d interleaved rounds\n", calls, rounds);

    memset(&c, 0, sizeof(c));
    c.N = N; c.K = Kd;
    c.act = (unsigned char *)malloc(Kd);
    c.dst = (unsigned char *)malloc(N);
    c.mid = (unsigned char *)malloc(256 + 128);
    c.times = (int32_t *)malloc(16 * sizeof(int32_t));
    c.big = (uint8_t *)malloc(nbig);
    if (!c.act || !c.dst || !c.mid || !c.times || !c.big) return 1;
    for (i = 0; i < Kd; i++) c.act[i] = (unsigned char)((i * 11 % 31) - 15);
    for (i = 0; i < nbig; i++) c.big[i] = (uint8_t)i;

    /* h: plain 32 KiB weight handle (the empty call is detach_out on it)
     * hres: 1 MiB weight handle with a resident output buffer */
    rc = lfmgemv_alloc_buf(nb, &h);
    if (rc != 0) { printf("alloc_buf rc=0x%x\n", (unsigned)rc); return 11; }
    if ((rc = lfmgemv_set_shape(h, N, Kd, 1)) != 0) {
        printf("set_shape rc=0x%x\n", (unsigned)rc); return 11;
    }
    lfmgemv_load_buf(h, 0, c.big, nb);

    if ((rc = lfmgemv_alloc_buf(nbig, &hres)) != 0) {
        printf("alloc_buf(hres) rc=0x%x\n", (unsigned)rc); return 11;
    }
    if ((rc = lfmgemv_set_shape(hres, N, Kd, 1)) != 0) return 11;
    lfmgemv_load_buf(hres, 0, c.big, nb);
    shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4096);
    if (!shared) { printf("rpcmem_alloc failed\n"); return 11; }
    rc = lfmgemv_attach_out(hres, rpcmem_to_fd(shared), N);
    if (rc != 0) { printf("attach_out rc=0x%x\n", (unsigned)rc); return 11; }
    c.h = h; c.hres = hres;

    /* tiny FFN state: gate/up/down [256,256], one in + three out sequences */
    {
        float *sw = (float *)malloc((size_t)N * sizeof(float));
        unsigned char *sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
        int hh[3];
        if (!sw || !sbuf) return 1;
        for (i = 0; i < N; i++) sw[i] = 1.0f;
        memcpy(sbuf + 0, &N, 4); memcpy(sbuf + 4, &Kd, 4);
        {
            float one = 1.0f;
            memcpy(sbuf + 8, &one, 4); memcpy(sbuf + 12, &one, 4);
        }
        memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
        for (k = 0; k < 3; k++) {
            if ((rc = lfmgemv_alloc_buf(nb, &hh[k])) != 0) return 11;
            if ((rc = lfmgemv_set_shape(hh[k], N, Kd, 1)) != 0) return 11;
            lfmgemv_load_buf(hh[k], 0, c.big, nb);
            if ((rc = lfmgemv_set_scale(hh[k], sbuf, 16 + N * (int)sizeof(float))) != 0) return 11;
        }
        c.hg = hh[0]; c.hu = hh[1]; c.hd = hh[2];
        free(sw); free(sbuf);
    }

    /* is POLL_QOS there at all? (one attempt, mode 3) */
    qrc_poll = set_qos("poll", RPC_POLL_QOS, 1000, 1);
    set_qos("off", RPC_DISABLE_QOS, 0, 0);
    usleep(100000);

    /* ---------------- 1. QOS modes on an empty call ---------------- */
    printf("\n== 1. empty call (detach_out: 4 B scalar, 0 sequences), us ==\n");
    printf("  %-34s %8s %8s %8s %8s\n", "config", "median", "p95", "min", "max");
    for (i = 0; i < rounds && i < 8; i++) {
        base_all[i] = series(c_empty, &c, calls, &rc);
        set_qos("pm", RPC_PM_QOS, 100, 0);
        usleep(50000);
        pm_all[i] = series(c_empty, &c, calls, &rc);
        set_qos("ad", RPC_ADAPTIVE_QOS, 100, 0);
        usleep(50000);
        ad_all[i] = series(c_empty, &c, calls, &rc);
        set_qos("off", RPC_DISABLE_QOS, 0, 0);
        usleep(50000);
        printf("  round %d: off %6.1f | PM_QOS %6.1f | ADAPTIVE %6.1f   (us, median)\n",
               i, base_all[i].median, pm_all[i].median, ad_all[i].median);
    }
    /* median of the round medians */
    {
        double m[8];
        int n = rounds < 8 ? rounds : 8;
        for (i = 0; i < n; i++) m[i] = base_all[i].median;
        med_base = stats(m, n).median;
        for (i = 0; i < n; i++) m[i] = pm_all[i].median;
        med_pm = stats(m, n).median;
        for (i = 0; i < n; i++) m[i] = ad_all[i].median;
        med_ad = stats(m, n).median;
    }
    printf("  %-34s %8.1f\n", "median of rounds: QOS off", med_base);
    printf("  %-34s %8.1f\n", "median of rounds: PM_QOS 100 us", med_pm);
    printf("  %-34s %8.1f\n", "median of rounds: ADAPTIVE_QOS 100 us", med_ad);
    printf("  POLL_QOS (mode %d): control rc=0x%x%s\n", RPC_POLL_QOS, (unsigned)qrc_poll,
           qrc_poll ? "  -> not supported by this libcdsprpc" : "  -> accepted");

    /* ---------------- 2. call shape at QOS off ---------------- */
    set_qos("off", RPC_DISABLE_QOS, 0, 0);
    usleep(100000);
    printf("\n== 2. call shape, QOS off, us (%d calls each) ==\n", calls);
    printf("  %-34s %8s %8s %8s %8s\n", "shape", "median", "p95", "min", "max");
    {
        static const struct { const char *name; call_fn fn; } shapes[] = {
            { "detach_out        0 in / 0 out",   c_empty },
            { "team_info         0 in / 1 out",   c_team_info },
            { "probe 45          0 in / 1 out",   c_probe },
            { "load_buf 16 B     1 in / 0 out",   c_load16 },
            { "load_buf 4 KiB    1 in / 0 out",   c_load4k },
            { "load_buf 256 KiB  1 in / 0 out",   c_load256k },
            { "gemv 256x256      1 in / 2 out",   c_gemv },
            { "gemv_resident     1 in / 1 out",   c_gemv_res },
            { "ffn 256x256       1 in / 3 out",   c_ffn },
        };
        for (k = 0; k < (int)(sizeof(shapes) / sizeof(shapes[0])); k++) {
            stats_t best = series(shapes[k].fn, &c, calls, &rc);
            int r2 = 0;
            stats_t s2 = series(shapes[k].fn, &c, calls, &r2);
            if (s2.median < best.median) best = s2;
            rc |= r2;
            row(shapes[k].name, best, rc);
            if (rc) printf("      rc=0x%x\n", (unsigned)rc);
        }
    }

    printf("\nbaseline empty call: median=%.1f us -> 48 calls = %.2f ms/token\n",
           med_base, 48.0 * med_base / 1000.0);
    printf("best QOS mode      : median=%.1f us -> 48 calls = %.2f ms/token\n",
           med_pm < med_ad ? med_pm : med_ad,
           48.0 * (med_pm < med_ad ? med_pm : med_ad) / 1000.0);

    set_qos("off", RPC_DISABLE_QOS, 0, 0);
    lfmgemv_detach_out(hres);
    lfmgemv_release_buf(h);
    lfmgemv_release_buf(hres);
    lfmgemv_release_buf(c.hg);
    lfmgemv_release_buf(c.hu);
    lfmgemv_release_buf(c.hd);
    rpcmem_free(shared);
    free(c.act); free(c.dst); free(c.mid); free(c.times); free(c.big);
    printf("done\n");
    return 0;
}
