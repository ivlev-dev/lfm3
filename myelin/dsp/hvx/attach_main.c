/*
 * Host stand for zero-copy weights: attach_in / alloc_in / detach_in.
 *
 * The weights stay in an rpcmem/ION buffer (DDR) owned by the host; the DSP
 * only maps the same pages (apps_mem_share_map) and reads them in place, so a
 * layer costs no DSP heap and no per-layer upload.
 *
 * Checks:
 *   1. GEMV bit-exactness: heap (alloc_buf+load_buf) vs alloc_in vs
 *      alloc_buf+attach_in, identical int4 bytes -> identical int8 output.
 *   2. Cache coherence of the mapped buffer: a cached ION buffer written by
 *      the CPU is read by the DSP (a) without a flush, (b) after an explicit
 *      DMA_BUF_IOCTL_SYNC, (c) after rewriting the weights and flushing, twice.
 *      Also the same with an uncached allocation.  This decides whether the
 *      weights path needs a flush or can use uncached memory.
 *   3. Whole FFN layer (gate/up/down, SiLU fused, one RPC) bit-exact:
 *      heap weights vs attached weights, mid[NH] and out[K], 2 contexts;
 *      timing over 5 runs per configuration.
 *   4. `attach_test big`: how much weight memory fits in ION (beyond the
 *      512 MiB DSP heap), one GEMV across the whole buffer.
 *
 * Exit code: 0 pass, 1 alloc/calibration failure, 2 bit-exactness failure,
 * 10.. RPC failure (matches the other stands).
 *
 * Build: see build.sh.  Run on device: ./run.sh attach_test [ntok]
 *                                              ./run.sh attach_test big
 */
#include "lfmgemv.h"
#include "ffn_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include "rpcmem.h"

#define NH 7680     /* hidden */
#define K  2048     /* model width */

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

/* ---------------- host ION buffer ---------------- */

typedef struct {
    void *p;
    int   fd;
    int   bytes;      /* rounded to 4 KiB */
} ionbuf_t;

static int ion_alloc(ionbuf_t *b, int bytes, uint32_t flags)
{
    int n = (bytes + 4095) & ~4095;
    b->p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, flags, n);
    if (!b->p) { b->fd = -1; b->bytes = 0; return -1; }
    b->fd = rpcmem_to_fd(b->p);
    b->bytes = n;
    return 0;
}

static void ion_free(ionbuf_t *b)
{
    if (b->p) rpcmem_free(b->p);
    b->p = 0; b->fd = -1; b->bytes = 0;
}

static int dma_sync(int fd, uint64_t flags)
{
    struct dma_buf_sync arg;
    arg.flags = flags;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &arg);
}

/* make the CPU writes visible to the DSP's view of the same pages */
static int ion_flush(ionbuf_t *b)
{
    if (dma_sync(b->fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE) != 0) return -errno;
    if (dma_sync(b->fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE) != 0) return -errno;
    return 0;
}

/* ---------------- weight upload helpers ---------------- */

static int set_scale(int h, int N, int Kdim, float sa, float so, const float *sw)
{
    unsigned char *sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
    int rc;
    if (!sbuf) return AEE_ENOMEMORY;
    memcpy(sbuf + 0, &N, 4);
    memcpy(sbuf + 4, &Kdim, 4);
    memcpy(sbuf + 8, &sa, 4);
    memcpy(sbuf + 12, &so, 4);
    memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(h, sbuf, 16 + N * (int)sizeof(float));
    free(sbuf);
    return rc;
}

/* heap resident weights: alloc_buf + chunked load_buf (the old path) */
static int upload_heap(int *h, const uint8_t *w, int nbytes,
                       int N, int Kdim, float sa, float so, const float *sw)
{
    size_t off = 0, chunk = 256 * 1024;
    int rc;
    if ((rc = lfmgemv_alloc_buf(nbytes, h)) != 0) return rc;
    if ((rc = lfmgemv_set_shape(*h, N, Kdim, 1)) != 0) return rc;
    while (off < (size_t)nbytes) {
        size_t n = (size_t)nbytes - off < chunk ? (size_t)nbytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, w + off, (int)n)) != 0) return rc;
        off += n;
    }
    return set_scale(*h, N, Kdim, sa, so, sw);
}

/* weights in ION; handle either alloc_in (no heap at all) or
 * alloc_buf + attach_in (the heap copy is dropped at attach) */
static int upload_attached(int *h, ionbuf_t *b, const uint8_t *w, int nbytes,
                           int N, int Kdim, float sa, float so, const float *sw,
                           int use_alloc_in, uint32_t flags)
{
    int rc;
    if (ion_alloc(b, nbytes, flags) != 0) return AEE_ENOMEMORY;
    memcpy(b->p, w, (size_t)nbytes);
    if (flags == RPCMEM_DEFAULT_FLAGS) {
        rc = ion_flush(b);
        if (rc != 0) { printf("ion flush rc=%d (%s)\n", rc, strerror(-rc)); return rc; }
    }
    if (use_alloc_in) {
        rc = lfmgemv_alloc_in(b->fd, b->bytes, h);
        if (rc != 0) return rc;
        rc = lfmgemv_set_shape(*h, N, Kdim, 1);
    } else {
        rc = lfmgemv_alloc_buf(nbytes, h);
        if (rc != 0) return rc;
        rc = lfmgemv_attach_in(*h, b->fd, b->bytes);
        if (rc != 0) return rc;
        rc = lfmgemv_set_shape(*h, N, Kdim, 1);
    }
    if (rc != 0) return rc;
    return set_scale(*h, N, Kdim, sa, so, sw);
}

static int cmp_i8(const char *label, const int8_t *a, const int8_t *b, int n)
{
    int i, bad = 0, first = -1, maxd = 0;
    for (i = 0; i < n; i++) {
        int d = (int)a[i] - (int)b[i];
        if (d) {
            if (first < 0) first = i;
            bad++;
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
        }
    }
    printf("%-52s %6d mismatches", label, bad);
    if (bad) printf("  first=%d max|d|=%d", first, maxd);
    printf("\n");
    return bad;
}

static int cmp_i32(const char *label, const int32_t *a, const int32_t *b, int n)
{
    int i, bad = 0, first = -1, maxd = 0;
    for (i = 0; i < n; i++) {
        int64_t d = (int64_t)a[i] - (int64_t)b[i];
        if (d) {
            if (first < 0) first = i;
            bad++;
            if (d < 0) d = -d;
            if (d > maxd) maxd = (int)d;
        }
    }
    printf("%-52s %6d mismatches", label, bad);
    if (bad) printf("  first=%d max|d|=%d", first, maxd);
    printf("\n");
    return bad;
}

static double median(double *v, int n)
{
    int i, j;
    for (i = 1; i < n; i++) {
        double x = v[i];
        for (j = i - 1; j >= 0 && v[j] > x; j--) v[j + 1] = v[j];
        v[j + 1] = x;
    }
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ---------------- 1+2: small matrix, three sources, cache coherence ---------------- */

static int small_and_cache_tests(void)
{
    const int N = 512, Kd = 256;
    const int nbytes = N * Kd / 2;            /* 64 KiB of int4 */
    uint8_t *w0 = (uint8_t *)malloc(nbytes);
    uint8_t *w1 = (uint8_t *)malloc(nbytes);
    float *sw = (float *)malloc((size_t)N * sizeof(float));
    int8_t act[Kd], *ref, *got;
    int32_t *ref32, *got32;
    int h_heap = -1, h_in = -1, h_ab = -1, h_cc = -1, h_uc = -1;
    ionbuf_t b_in, b_ab, b_cc, b_uc;
    int32_t times[8];
    int i, rc, bad = 0, fails = 0;

    memset(&b_in, 0, sizeof(b_in)); memset(&b_ab, 0, sizeof(b_ab));
    memset(&b_cc, 0, sizeof(b_cc)); memset(&b_uc, 0, sizeof(b_uc));
    if (!w0 || !w1 || !sw) return 1;

    ffn_pack_i4_random(w0, N, Kd, 0x51eeu);
    ffn_pack_i4_random(w1, N, Kd, 0x77a7u);
    for (i = 0; i < N; i++) sw[i] = (0.35f + 0.05f * (float)(i % 40)) / (float)Kd;
    for (i = 0; i < Kd; i++) act[i] = (int8_t)((i * 13 % 37) - 18);
    ref = (int8_t *)malloc((size_t)N);
    got = (int8_t *)malloc((size_t)N);
    ref32 = (int32_t *)malloc((size_t)N * 4);
    got32 = (int32_t *)malloc((size_t)N * 4);
    if (!ref || !got || !ref32 || !got32) return 1;

    printf("\n== 1. GEMV bit-exactness, N=%d K=%d (%.0f KiB int4) ==\n", N, Kd, nbytes / 1024.0);

    /* int32 reference on the heap handle, checked against the host model */
    rc = upload_heap(&h_heap, w0, nbytes, N, Kd, 1.0f, 1.0f, sw);
    if (rc != 0) { printf("upload_heap rc=%d\n", rc); return 10; }
    rc = lfmgemv_set_shape(h_heap, N, Kd, 0);          /* int32 output */
    if (rc != 0) { printf("set_shape(int32) rc=%d\n", rc); return 10; }
    rc = lfmgemv_gemv(h_heap, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)ref32, N * 4, times, 8);
    if (rc != 0) { printf("gemv int32 rc=%d\n", rc); return 10; }
    {
        int32_t bias = -8 * ffn_sum_i8(act, Kd);
        for (i = 0; i < N; i++)
            got32[i] = ffn_dot_i4(w0 + (size_t)i * (Kd / 2), act, Kd, bias);
        fails += cmp_i32("heap int32 vs host ffn_dot_i4", ref32, got32, N) != 0;
    }
    /* back to the Q16 path for the byte comparison */
    if ((rc = lfmgemv_set_shape(h_heap, N, Kd, 1)) != 0) return 10;
    if ((rc = set_scale(h_heap, N, Kd, 1.0f, 1.0f, sw)) != 0) return 10;
    rc = lfmgemv_gemv(h_heap, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)ref, N, times, 8);
    if (rc != 0) { printf("gemv heap rc=%d\n", rc); return 10; }

    /* (a) alloc_in: no DSP heap copy at all, uncached ION */
    rc = upload_attached(&h_in, &b_in, w0, nbytes, N, Kd, 1.0f, 1.0f, sw, 1, RPCMEM_FLAG_UNCACHED);
    if (rc != 0) { printf("alloc_in rc=%d\n", rc); return 11; }
    rc = lfmgemv_gemv(h_in, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) { printf("gemv alloc_in rc=%d\n", rc); return 11; }
    bad = cmp_i8("alloc_in (uncached) vs heap", got, ref, N);
    fails += bad != 0;

    /* (b) alloc_buf + attach_in: heap copy dropped, cached ION + flush */
    rc = upload_attached(&h_ab, &b_ab, w0, nbytes, N, Kd, 1.0f, 1.0f, sw, 0, RPCMEM_DEFAULT_FLAGS);
    if (rc != 0) { printf("alloc_buf+attach_in rc=%d\n", rc); return 11; }
    rc = lfmgemv_gemv(h_ab, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) { printf("gemv attach_in rc=%d\n", rc); return 11; }
    bad = cmp_i8("alloc_buf+attach_in (cached, flushed) vs heap", got, ref, N);
    fails += bad != 0;

    /* ---- 2. cache coherence ---- */
    printf("\n== 2. cache coherence of the mapped weights ==\n");

    /* host write bandwidth: is RPCMEM_DEFAULT_FLAGS a cached mapping here? */
    {
        static const struct { const char *name; uint32_t flags; } fl[] = {
            { "RPCMEM_DEFAULT_FLAGS", RPCMEM_DEFAULT_FLAGS },
            { "RPCMEM_FLAG_UNCACHED", RPCMEM_FLAG_UNCACHED },
        };
        uint8_t *src = (uint8_t *)malloc(8 << 20);
        int k;
        if (src) {
            memset(src, 0x5a, 8 << 20);
            printf("host memcpy into ION, 8 MiB x 4:\n");
            for (k = 0; k < 2; k++) {
                ionbuf_t bt;
                double t0, us;
                memset(&bt, 0, sizeof(bt));
                if (ion_alloc(&bt, 8 << 20, fl[k].flags) != 0) {
                    printf("  %-24s alloc failed\n", fl[k].name);
                    continue;
                }
                t0 = now_us();
                for (i = 0; i < 4; i++) memcpy(bt.p, src, 8 << 20);
                us = now_us() - t0;
                printf("  %-24s %.0f us for 32 MiB -> %.2f GB/s\n",
                       fl[k].name, us, 33.554432e6 / (us * 1e3));
                ion_free(&bt);
            }
            free(src);
        }
    }

    /* (i) cached buffer, CPU writes, NO flush: what does the DSP see? */
    if (ion_alloc(&b_cc, nbytes, RPCMEM_DEFAULT_FLAGS) != 0) {
        printf("cached ion alloc failed\n"); return 11;
    }
    memcpy(b_cc.p, w0, (size_t)nbytes);
    rc = lfmgemv_alloc_in(b_cc.fd, b_cc.bytes, &h_cc);
    if (rc != 0) { printf("alloc_in(cached) rc=%d\n", rc); return 11; }
    lfmgemv_set_shape(h_cc, N, Kd, 1);
    set_scale(h_cc, N, Kd, 1.0f, 1.0f, sw);
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) { printf("gemv cached rc=%d\n", rc); return 11; }
    bad = cmp_i8("cached, CPU writes, no flush (stale?)", got, ref, N);
    printf("   -> %s\n", bad ? "flush/invalidate is REQUIRED" :
                                 "mapping already gives a coherent view");
    /* the number itself is the measurement; it is not a pass/fail gate */

    /* (ii) explicit flush, then the same call twice */
    rc = ion_flush(&b_cc);
    if (rc != 0) { printf("flush before attach rc=%d (%s)\n", rc, strerror(-rc)); return 11; }
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("cached, flushed, call 1 vs heap", got, ref, N) != 0;
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("cached, flushed, call 2 vs heap (no stuck data)", got, ref, N) != 0;

    /* (iii) rewrite the weights in the same buffer, flush, call twice.
     * The host flush is not enough: the DSP keeps its own cached view of the
     * pages it has already read, so this is expected to be stale. */
    memcpy(b_cc.p, w1, (size_t)nbytes);
    if (ion_flush(&b_cc) != 0) { printf("flush after rewrite failed\n"); return 11; }
    if ((rc = lfmgemv_load_buf(h_heap, 0, w1, nbytes)) != 0) {
        printf("load_buf w1 rc=%d\n", rc); return 11;
    }
    rc = lfmgemv_gemv(h_heap, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)ref, N, times, 8);
    if (rc != 0) return 11;
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    bad = cmp_i8("rewritten in place + host flush (expected stale)", got, ref, N);
    printf("   -> %s\n", bad ? "DSP keeps its own cached view: re-map is REQUIRED"
                             : "in-place rewrite visible without re-map");
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    bad = cmp_i8("rewritten in place, call 2 (expected stale too)", got, ref, N);

    /* (iii-b) unmap + map again refreshes the DSP view: the documented way to
     * swap weights inside one ION arena */
    if (lfmgemv_detach_in(h_cc) != 0) { printf("detach_in rc!=0\n"); fails++; }
    rc = lfmgemv_attach_in(h_cc, b_cc.fd, b_cc.bytes);
    if (rc != 0) { printf("re-attach rc=%d\n", rc); return 11; }
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("rewritten weights, detach+attach, call 1 vs heap(w1)", got, ref, N) != 0;
    rc = lfmgemv_gemv(h_cc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("rewritten weights, detach+attach, call 2 vs heap(w1)", got, ref, N) != 0;

    /* (iv) uncached buffer: no flush, rewrite, two calls */
    rc = upload_attached(&h_uc, &b_uc, w0, nbytes, N, Kd, 1.0f, 1.0f, sw, 1, RPCMEM_FLAG_UNCACHED);
    if (rc != 0) { printf("alloc_in(uncached) rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_load_buf(h_heap, 0, w0, nbytes)) != 0) return 11;
    lfmgemv_gemv(h_heap, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)ref, N, times, 8);
    rc = lfmgemv_gemv(h_uc, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("uncached, no flush, call 1 vs heap", got, ref, N) != 0;
    memcpy(b_uc.p, w1, (size_t)nbytes);
    if ((rc = lfmgemv_load_buf(h_heap, 0, w1, nbytes)) != 0) return 11;
    lfmgemv_gemv(h_heap, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)ref, N, times, 8);
    rc = lfmgemv_gemv(h_uc, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    bad = cmp_i8("uncached, rewritten in place (expected stale)", got, ref, N);
    printf("   -> uncached host allocation behaves the same: %s\n",
           bad ? "the stale copy is on the DSP side" : "no DSP-side copy");
    rc = lfmgemv_gemv(h_uc, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    bad = cmp_i8("uncached, rewritten, call 2 (expected stale too)", got, ref, N);
    if (lfmgemv_detach_in(h_uc) != 0) { printf("detach_in rc!=0\n"); fails++; }
    rc = lfmgemv_attach_in(h_uc, b_uc.fd, b_uc.bytes);
    if (rc != 0) { printf("re-attach(uncached) rc=%d\n", rc); return 11; }
    rc = lfmgemv_gemv(h_uc, 1, (const unsigned char *)act, Kd, 1, (unsigned char *)got, N, times, 8);
    if (rc != 0) return 11;
    fails += cmp_i8("uncached, rewritten, detach+attach vs heap(w1)", got, ref, N) != 0;

    /* detach_in: the handle loses its weights and the call is refused */
    if (lfmgemv_detach_in(h_uc) != 0) { printf("detach_in rc!=0\n"); fails++; }
    rc = lfmgemv_gemv(h_uc, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, N, times, 8);
    printf("%-52s rc=0x%08x%s\n", "gemv after detach_in", (unsigned)rc,
           ((unsigned)rc & 0xffu) == 0x0eu ? "  (AEE_EBADPARM, as intended)" : "  unexpected");

    lfmgemv_release_buf(h_heap);
    lfmgemv_release_buf(h_in);
    lfmgemv_release_buf(h_ab);
    lfmgemv_release_buf(h_cc);
    lfmgemv_release_buf(h_uc);
    ion_free(&b_in); ion_free(&b_ab); ion_free(&b_cc); ion_free(&b_uc);
    free(w0); free(w1); free(sw); free(ref); free(got); free(ref32); free(got32);
    return fails ? 2 : 0;
}

/* ---------------- 3: whole FFN layer, heap vs attached ---------------- */

static void time_ffn(const char *label, int hg, int hu, int hd, const int8_t *act,
                     int runs, int8_t *dst, int8_t *mid, int32_t *times, double *wall)
{
    int i;
    for (i = 0; i < runs; i++) {
        double t0 = now_us();
        int rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)act, K, 1,
                             (unsigned char *)dst, K, (unsigned char *)mid, NH, times, 12);
        wall[i] = now_us() - t0;
        if (rc != 0) { printf("%s ffn rc=%d\n", label, rc); wall[i] = -1; }
        usleep(15000);
    }
}

static int ffn_tests(int ntok)
{
    ffn_model_t m;
    ffn_calib_t ci;
    const int nb = NH * (K / 2);            /* one int4 matrix */
    int8_t *mid[2], *dst[2], *mid_a[2], *dst_a[2];
    int hg = -1, hu = -1, hd = -1, hg2 = -1, hu2 = -1, hd2 = -1;
    ionbuf_t bg, bu, bd;
    double wall_h[10], wall_a[10], med_h, med_a;
    int32_t times[12];
    int t, i, rc, fails = 0;
    float sag;

    memset(&bg, 0, sizeof(bg)); memset(&bu, 0, sizeof(bu)); memset(&bd, 0, sizeof(bd));

    if (ffn_model_synth(&m, K, NH, ntok < 2 ? 2 : ntok, 0x5eedu) != 0) {
        printf("model alloc failed\n"); return 1;
    }
    if (ffn_calibrate(&m, &ci) != 0) { printf("calibration failed\n"); return 1; }
    sag = m.s_in * FFN_Q16_ONE;
    for (t = 0; t < 2; t++) {
        mid[t] = (int8_t *)malloc(NH + 128); dst[t] = (int8_t *)malloc(K + 128);
        mid_a[t] = (int8_t *)malloc(NH + 128); dst_a[t] = (int8_t *)malloc(K + 128);
        if (!mid[t] || !dst[t] || !mid_a[t] || !dst_a[t]) return 1;
    }

    printf("\n== 3. FFN layer (K=%d NH=%d, one RPC) ==\n", K, NH);

    /* heap weights */
    rc = upload_heap(&hg, m.wg, nb, NH, K, sag, 1.0f, m.swg);
    if (!rc) rc = upload_heap(&hu, m.wu, nb, NH, K, sag, 1.0f, m.swu);
    if (!rc) rc = upload_heap(&hd, m.wd, nb, K, NH, m.s_mid, m.s_out, m.swd);
    if (rc != 0) { printf("heap upload rc=%d\n", rc); return 11; }
    for (t = 0; t < 2; t++) {
        rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)(m.x8 + (size_t)t * K), K, 1,
                         (unsigned char *)dst[t], K, (unsigned char *)mid[t], NH, times, 12);
        if (rc != 0) { printf("ffn heap ctx=%d rc=%d\n", t, rc); return 12; }
    }
    /* the layer itself must still match the integer model */
    for (t = 0; t < 2; t++) {
        int8_t *mid_ref = (int8_t *)malloc(NH), *out_ref = (int8_t *)malloc(K);
        char label[80];
        if (!mid_ref || !out_ref) return 1;
        ffn_int_mid(&m, t, mid_ref, NULL);
        ffn_int_out(&m, mid_ref, out_ref, NULL);
        snprintf(label, sizeof(label), "heap ctx=%d vs host integer model (mid)", t);
        fails += cmp_i8(label, mid[t], mid_ref, NH) != 0;
        snprintf(label, sizeof(label), "heap ctx=%d vs host integer model (out)", t);
        fails += cmp_i8(label, dst[t], out_ref, K) != 0;
        free(mid_ref); free(out_ref);
    }
    /* warm up the worker team before timing */
    time_ffn("warm", hg, hu, hd, m.x8, 3, dst[0], mid[0], times, wall_h);
    /* 2 contexts x 5 runs */
    for (t = 0; t < 2; t++)
        time_ffn("heap", hg, hu, hd, m.x8 + (size_t)t * K, 5,
                 dst[t], mid[t], times, wall_h + 5 * t);
    med_h = median(wall_h, 10);
    printf("heap weights     : wall median=%6.0f us  dsp=%4d us  [g=%d u=%d s=%d d=%d p=%d]\n",
           med_h, times[0], times[1], times[2], times[3], times[4], times[5]);

    /* attached weights: alloc_in, cached + flushed once */
    rc = upload_attached(&hg2, &bg, m.wg, nb, NH, K, sag, 1.0f, m.swg, 1, RPCMEM_DEFAULT_FLAGS);
    if (!rc) rc = upload_attached(&hu2, &bu, m.wu, nb, NH, K, sag, 1.0f, m.swu, 1, RPCMEM_DEFAULT_FLAGS);
    if (!rc) rc = upload_attached(&hd2, &bd, m.wd, nb, K, NH, m.s_mid, m.s_out, m.swd, 1, RPCMEM_DEFAULT_FLAGS);
    if (rc != 0) { printf("attach upload rc=%d\n", rc); return 11; }
    for (t = 0; t < 2; t++) {
        rc = lfmgemv_ffn(hg2, hu2, hd2, (const unsigned char *)(m.x8 + (size_t)t * K), K, 1,
                         (unsigned char *)dst_a[t], K, (unsigned char *)mid_a[t], NH, times, 12);
        if (rc != 0) { printf("ffn attached ctx=%d rc=%d\n", t, rc); return 12; }
    }
    for (t = 0; t < 2; t++) {
        char label[80];
        snprintf(label, sizeof(label), "attached ctx=%d vs heap (mid[%d])", t, NH);
        fails += cmp_i8(label, mid_a[t], mid[t], NH) != 0;
        snprintf(label, sizeof(label), "attached ctx=%d vs heap (out[%d])", t, K);
        fails += cmp_i8(label, dst_a[t], dst[t], K) != 0;
    }
    for (t = 0; t < 2; t++)
        time_ffn("attach", hg2, hu2, hd2, m.x8 + (size_t)t * K, 5,
                 dst_a[t], mid_a[t], times, wall_a + 5 * t);
    med_a = median(wall_a, 10);
    printf("attached weights : wall median=%6.0f us  dsp=%4d us  [g=%d u=%d s=%d d=%d p=%d]\n",
           med_a, times[0], times[1], times[2], times[3], times[4], times[5]);
    printf("wall heap median=%.0f us, attached median=%.0f us, delta=%+.0f us (%.1f%%)\n",
           med_h, med_a, med_a - med_h, 100.0 * (med_a - med_h) / med_h);

    /* order check: the two phases above always run heap-first, so interleave
     * them in both orders; no mode may win because it ran second */
    {
        double wq[4];
        printf("interleaved A/B (both contexts, 4 runs per point):\n");
        for (i = 0; i < 4; i++) {
            int first_attached = i & 1;
            double m_first, m_second;
            if (!first_attached) {
                for (t = 0; t < 4; t++)
                    time_ffn("x", hg, hu, hd, m.x8 + (size_t)(t & 1) * K, 1,
                             dst[t & 1], mid[t & 1], times, wq + t);
                m_first = median(wq, 4);
                for (t = 0; t < 4; t++)
                    time_ffn("x", hg2, hu2, hd2, m.x8 + (size_t)(t & 1) * K, 1,
                             dst_a[t & 1], mid_a[t & 1], times, wq + t);
                m_second = median(wq, 4);
                printf("  round %d: heap %5.0f us | attached %5.0f us\n", i, m_first, m_second);
            } else {
                for (t = 0; t < 4; t++)
                    time_ffn("x", hg2, hu2, hd2, m.x8 + (size_t)(t & 1) * K, 1,
                             dst_a[t & 1], mid_a[t & 1], times, wq + t);
                m_first = median(wq, 4);
                for (t = 0; t < 4; t++)
                    time_ffn("x", hg, hu, hd, m.x8 + (size_t)(t & 1) * K, 1,
                             dst[t & 1], mid[t & 1], times, wq + t);
                m_second = median(wq, 4);
                printf("  round %d: attached %5.0f us | heap %5.0f us\n", i, m_first, m_second);
            }
        }
    }

    lfmgemv_release_buf(hg); lfmgemv_release_buf(hu); lfmgemv_release_buf(hd);
    lfmgemv_release_buf(hg2); lfmgemv_release_buf(hu2); lfmgemv_release_buf(hd2);
    ion_free(&bg); ion_free(&bu); ion_free(&bd);
    for (i = 0; i < 2; i++) { free(mid[i]); free(dst[i]); free(mid_a[i]); free(dst_a[i]); }
    ffn_model_free(&m);
    return fails ? 2 : 0;
}

/* ---------------- 4: how much weight memory fits in ION ---------------- */

static int big_test(void)
{
    static const int mib[] = { 256, 512, 640, 900 };
    const int Kd = 2048;
    int i, h = -1, rc, ok = 0;
    ionbuf_t b;
    double t0, ms;
    int32_t times[8];
    int32_t *ref, *got;
    int8_t act[2048];
    int rows = 0;

    memset(&b, 0, sizeof(b));
    printf("\n== 4. weight memory in ION beyond the DSP heap (512 MiB) ==\n");
    for (i = 0; i < (int)(sizeof(mib) / sizeof(mib[0])); i++) {
        int bytes = mib[i] << 20;
        int r = ion_alloc(&b, bytes, RPCMEM_DEFAULT_FLAGS);
        printf("ION %4d MiB: %s\n", mib[i], r == 0 ? "ok" : "FAIL");
        if (r == 0) { ok = mib[i]; ion_free(&b); }
    }
    if (!ok) return 0;

    /* fill the largest successful buffer and run one GEMV across all of it */
    if (ion_alloc(&b, ok << 20, RPCMEM_DEFAULT_FLAGS) != 0) return 11;
    rows = (b.bytes / (Kd / 2)) & ~0x3ff;      /* whole 1024-row blocks */
    printf("GEMV over N=%d rows, K=%d (%.0f MiB of int4 weights)\n",
           rows, Kd, (double)rows * (Kd / 2) / 1048576.0);
    {
        /* realistic upload: one 1 MiB pattern, memcpy'd over the arena twice
         * (pass 1 pays the first-touch of the freshly allocated pages, pass 2
         * is the steady-state host->ION bandwidth) */
        const size_t chunk = 1u << 20;
        uint8_t *pat = (uint8_t *)malloc(chunk);
        double ms1, ms2;
        size_t off;
        int pass;
        if (!pat) return 1;
        ffn_pack_i4_random(pat, (int)(chunk / (Kd / 2)), Kd, 0xbeefu);
        ms1 = ms2 = 0;
        for (pass = 0; pass < 2; pass++) {
            t0 = now_us();
            for (off = 0; off + chunk <= (size_t)rows * (Kd / 2); off += chunk)
                memcpy((uint8_t *)b.p + off, pat, chunk);
            ms = (now_us() - t0) / 1e3;
            if (pass == 0) ms1 = ms; else ms2 = ms;
        }
        free(pat);
        printf("host upload: pass1 (cold pages) %.1f ms (%.2f GB/s), "
               "pass2 %.1f ms (%.2f GB/s)\n",
               ms1, (double)rows * (Kd / 2) / (ms1 * 1e6),
               ms2, (double)rows * (Kd / 2) / (ms2 * 1e6));
        ms = ms2;
    }
    if (ion_flush(&b) != 0) { printf("flush failed\n"); return 11; }

    rc = lfmgemv_alloc_in(b.fd, b.bytes, &h);
    if (rc != 0) { printf("alloc_in(big) rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_set_shape(h, rows, Kd, 0)) != 0) { printf("set_shape rc=%d\n", rc); return 11; }
    ref = (int32_t *)malloc((size_t)rows * 4);
    got = (int32_t *)malloc((size_t)rows * 4);
    if (!ref || !got) return 1;
    for (i = 0; i < Kd; i++) act[i] = (int8_t)((i * 7 % 29) - 14);
    rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, Kd, 1,
                      (unsigned char *)got, rows * 4, times, 8);
    if (rc != 0) { printf("gemv(big) rc=%d\n", rc); return 12; }
    {
        int32_t bias = -8 * ffn_sum_i8(act, Kd);
        int r, bad = 0;
        /* check the first, a middle and the last rows: the point is that rows
         * spread over the whole 256..900 MiB are read correctly */
        for (r = 0; r < rows; r += (rows > 3 ? (rows - 1) / 3 : 1)) {
            ref[r] = ffn_dot_i4((const uint8_t *)b.p + (size_t)r * (Kd / 2), act, Kd, bias);
            if (ref[r] != got[r]) { bad++; printf("row %d ref=%d got=%d\n", r, ref[r], got[r]); }
        }
        printf("spot check over the whole buffer: %d bad rows\n", bad);
        if (bad) return 2;
    }
    printf("GEMV kernel min=%d us for %.0f MiB -> %.2f GB/s from ION\n",
           times[0], (double)rows * (Kd / 2) / 1048576.0,
           (double)rows * (Kd / 2) / (times[0] * 1e3));
    lfmgemv_release_buf(h);
    ion_free(&b);
    free(ref); free(got);
    return 0;
}

int main(int argc, char **argv)
{
    int rc;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }
    if (argc > 1 && strcmp(argv[1], "big") == 0) return big_test();

    rc = small_and_cache_tests();
    if (rc == 10 || rc == 11) return rc;
    {
        int ffn_rc = ffn_tests(argc > 1 ? atoi(argv[1]) : 2);
        if (ffn_rc == 10 || ffn_rc == 11 || ffn_rc == 12) return ffn_rc;
        if (ffn_rc) rc = ffn_rc;
    }
    printf("\n%s\n", rc == 0 ? "ATTACH PASS" : "ATTACH FAIL");
    return rc;
}
