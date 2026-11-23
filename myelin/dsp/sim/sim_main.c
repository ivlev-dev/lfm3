/*
 * Local QEMU/QuRT stand for the DSP exec framework (lfmgemv_impl.c).
 *
 * Runs the same integer kernels as the on-device stands, but without a phone
 * and without FastRPC: lfmgemv_impl.c is linked directly into a QuRT qexe and
 * every test calls the lfmgemv_* entry points in-process.  Under QEMU the DSP
 * image is flat memory, so an address is a pointer and VTCM/ION are not
 * needed; the fd-based arena API is emulated by shim_hap.c (see there).
 *
 * Groups (same data and same reference formulas as the host stands):
 *   gemv    7680x2048 int4, int32 out M=1/2/4 and int8 out with scales,
 *           compared against the scalar dot product (host_main.c)
 *   ffn     7680x2048 gate/up/down, compared bit-for-bit against the shared
 *           integer model and by cos against the f32 truth (ffn_ref.h),
 *           single-thread result == 4-thread result
 *   exec    2048x2048 int4 GEMV, FFN, ATTN, CONV and QUANT in one command
 *           list: every op vs a direct call, CONV vs a host ring model,
 *           token counter via rt[], reset_state() identity
 *   team    team_start/team_kick/team_join: 1 thread vs 4 threads bit-exact,
 *           worker creation and fallback counters reported
 *
 * Exit code: 0 all checks passed, 1 failure.  The runner keys on the
 * "SIM RESULT: PASS" line because sdksim's runelf misbehaves on non-zero
 * main() return codes (it raises an exception during teardown).
 */
#include "lfmgemv.h"
#include "ffn_ref.h"
#include "sim_shim.h"
#include "../conv/conv_math.h"
#include "../quant/quant.h"

#include <qurt_alloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* QEMU/QuRT gives user code no heap by default (sbrk fails immediately), so
 * the qexe declares a static one.  64 MiB is the largest size that still
 * starts on the V68N_1024 machine here; 96 MiB fails in the QuRT user-client
 * mapping ("failed to allocate virtual memory with MEM_MAPPING_NONE"). */
QURT_DECLARE_STATIC_HEAP(64u * 1024u * 1024u);

#define GEMV_N 7680
#define GEMV_K 2048
#define FFN_K  2048
#define FFN_NH 7680

static int g_groups;
static int g_failed;

static void group_result(const char *name, int failed)
{
    g_groups++;
    if (failed) g_failed++;
    printf("SIM %-8s %s\n", name, failed ? "FAIL" : "PASS");
}

/* deterministic pseudo random, explicit state (no globals: clang -O2 hoists) */
static inline uint32_t lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

/* packed weights: offset-8 nibbles, lo = k even, hi = k odd */
static void build_weights(uint8_t *w, int8_t *wsigned, int N, int K)
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

/* set tuning in the kernel: probe kind 8, arg = pf_kb | rows<<8 | usepf<<16 |
 * threads<<24 */
static int set_threads(int h, int threads)
{
    int res[8];
    int arg = 32 | (2 << 8) | (1 << 16) | ((threads & 0xff) << 24);
    int rc = lfmgemv_probe(h, 8, arg, res, 8);

    if (rc != 0) return rc;
    if (res[6] != threads) return -1;
    return 0;
}

static void print_team(const char *when)
{
    int info[8];
    int rc = lfmgemv_team_info(info, 8);

    if (rc != 0) {
        printf("SIM [team] %s: team_info rc=%d\n", when, rc);
        return;
    }
    printf("SIM [team] %s: state=%d hvx_rc=%d mode=%d created=%d threads=%d "
           "thr_rc=%d hvx_units=%d rows=%d\n",
           when, info[0], info[1], info[2], info[3], info[4], info[5],
           info[6], info[7]);
}

/* ======================= GEMV (host_main.c mirror) ======================= */

static int test_gemv(void)
{
    int bad = 0, h = -1, rc, m, r, i;
    size_t wbytes = (size_t)GEMV_N * (GEMV_K / 2);
    size_t chunk = 256 * 1024;
    uint8_t *w = NULL;
    int8_t *wsigned = NULL, *act = NULL;
    uint8_t *dst = NULL;
    int32_t *acc = NULL;
    float *srow = NULL;
    unsigned char *sbuf = NULL;
    int times[12];

    printf("SIM [gemv] shape N=%d K=%d weights=%.2f MB\n",
           GEMV_N, GEMV_K, (double)wbytes / 1e6);

    w       = (uint8_t *)malloc(wbytes);
    wsigned = (int8_t *)malloc((size_t)GEMV_N * GEMV_K);
    act     = (int8_t *)malloc((size_t)8 * GEMV_K + 128);
    dst     = (uint8_t *)malloc((size_t)8 * GEMV_N + 128);
    acc     = (int32_t *)malloc((size_t)8 * GEMV_N * sizeof(int32_t));
    srow    = (float *)malloc((size_t)GEMV_N * sizeof(float));
    sbuf    = (unsigned char *)malloc(16 + (size_t)GEMV_N * 4);
    if (!w || !wsigned || !act || !dst || !acc || !srow || !sbuf) {
        printf("SIM [gemv] host allocation failed (static heap exhausted)\n");
        bad = 1;
        goto out;
    }

    build_weights(w, wsigned, GEMV_N, GEMV_K);

    {
        static const int pr[] = {0, 1, 33, 100};
        int j, q;
        for (j = 0; j < 4; j++) {
            const int8_t *sr = wsigned + (size_t)pr[j] * GEMV_K;
            const uint8_t *wr = w + (size_t)pr[j] * (GEMV_K / 2);
            long sw = 0;
            for (q = 0; q < GEMV_K; q++) sw += sr[q];
            printf("SIM [gemv] row %3d: w[0..2]=%d,%d,%d packed=%02x %02x %02x rowsum=%ld\n",
                   pr[j], sr[0], sr[1], sr[2], wr[0], wr[1], wr[2], sw);
        }
    }
    for (r = 0; r < GEMV_N; r++) srow[r] = 0.001f + (float)(r % 7) * 0.0001f;
    {
        uint32_t st = 0xdeadbeefu;
        for (i = 0; i < 8 * GEMV_K; i++) act[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }
    printf("SIM [gemv] act[0..3]=%d,%d,%d,%d dot(row0)=%d dot(row33)=%d\n",
           (int)act[0], (int)act[1], (int)act[2], (int)act[3],
           (int)dot_ref(wsigned, act, GEMV_K),
           (int)dot_ref(wsigned + 33 * GEMV_K, act, GEMV_K));

    rc = lfmgemv_alloc_buf((int)wbytes, &h);
    printf("SIM [gemv] alloc_buf rc=%d h=%d\n", rc, h);
    if (rc != 0) { bad = 1; goto out; }

    if ((rc = lfmgemv_set_shape(h, GEMV_N, GEMV_K, 0)) != 0) {
        printf("SIM [gemv] set_shape(int32) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    if ((rc = set_threads(h, 1)) != 0) {
        printf("SIM [gemv] probe(set threads=1) rc=%d\n", rc);
        bad = 1;
        goto out;
    }

    {
        size_t off = 0;
        while (off < wbytes) {
            size_t n = wbytes - off < chunk ? wbytes - off : chunk;
            rc = lfmgemv_load_buf(h, (int)off, w + off, (int)n);
            if (rc != 0) { printf("SIM [gemv] load_buf rc=%d off=%zu\n", rc, off); bad = 1; goto out; }
            off += n;
        }
    }

    for (m = 1; m <= 4; m *= 2) {
        int nbad = 0, maxerr = 0;
        rc = lfmgemv_gemv(h, m, (const unsigned char *)act, m * GEMV_K, 1,
                          (unsigned char *)acc, m * GEMV_N * (int)sizeof(int32_t),
                          times, 12);
        if (rc != 0) { printf("SIM [gemv] gemv int32 M=%d rc=%d\n", m, rc); bad = 1; goto out; }
        for (r = 0; r < GEMV_N; r++) {
            for (i = 0; i < m; i++) {
                int32_t ref = dot_ref(wsigned + (size_t)r * GEMV_K,
                                      act + (size_t)i * GEMV_K, GEMV_K);
                int32_t got = acc[(size_t)i * GEMV_N + r];
                int32_t e = ref - got;
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = (int)e;
                if (e != 0) nbad++;
            }
        }
        printf("SIM [gemv] int32 M=%d: elems=%d mismatches=%d max_abs_err=%d "
               "kernel_min=%d cycles\n",
               m, GEMV_N * m, nbad, maxerr, times[0]);
        if (nbad) {
            int rr;
            static const int probe[] = {0, 1, 32, 33, 34, 35, 100};
            for (rr = 0; rr < 7; rr++) {
                int32_t ref = dot_ref(wsigned + (size_t)probe[rr] * GEMV_K, act, GEMV_K);
                printf("SIM [gemv]   row %4d ref=%8d got_M%d=%8d\n",
                       probe[rr], (int)ref, m,
                       (int)acc[(size_t)(0) * GEMV_N + probe[rr]]);
            }
            bad = 1;
        }
    }

    if ((rc = lfmgemv_set_shape(h, GEMV_N, GEMV_K, 1)) != 0) {
        printf("SIM [gemv] set_shape(i8) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    {
        int vN = GEMV_N, vK = GEMV_K;
        memcpy(sbuf, &vN, 4);
        memcpy(sbuf + 4, &vK, 4);
    }
    {
        float sa = 0.05f, so = 0.5f;
        memcpy(sbuf + 8, &sa, 4);
        memcpy(sbuf + 12, &so, 4);
    }
    memcpy(sbuf + 16, srow, (size_t)GEMV_N * sizeof(float));
    rc = lfmgemv_set_scale(h, sbuf, 16 + GEMV_N * 4);
    printf("SIM [gemv] set_scale rc=%d\n", rc);
    if (rc != 0) { bad = 1; goto out; }

    {
        const float sa_check = 0.05f, so_check = 0.5f;
        int nbad = 0, maxerr = 0;
        rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, GEMV_K, 1, dst, GEMV_N,
                          times, 12);
        if (rc != 0) { printf("SIM [gemv] gemv i8 rc=%d\n", rc); bad = 1; goto out; }
        for (r = 0; r < GEMV_N; r++) {
            int32_t acc_ref = dot_ref(wsigned + (size_t)r * GEMV_K, act, GEMV_K);
            float mm = srow[r] * sa_check / so_check;
            int32_t q;
            int64_t v;
            int32_t got = (int8_t)dst[r];
            if (mm < 0.0f) mm = 0.0f;
            if (mm > 0.99997f) mm = 0.99997f;
            q = (int32_t)(mm * 65536.0f + 0.5f);
            v = ((int64_t)acc_ref * (int64_t)q + 32768) >> 16;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            if (v != got) {
                int32_t e = v > got ? (int32_t)(v - got) : (int32_t)(got - v);
                nbad++;
                if (e > maxerr) maxerr = (int)e;
                if (nbad <= 5)
                    printf("SIM [gemv]   i8 row %d ref=%lld got=%d acc=%d m=%.6f\n",
                           r, (long long)v, (int)got, (int)acc_ref, mm);
            }
        }
        printf("SIM [gemv] int8+scales: mismatches=%d/%d max_err=%d\n",
               nbad, GEMV_N, maxerr);
        if (nbad) bad = 1;
    }

    {
        long sum = 0;
        for (i = 0; i < GEMV_N; i++) sum += (int8_t)dst[i];
        printf("SIM [gemv] dst checksum=%ld (nonzero sanity)\n", sum);
    }

out:
    if (h >= 0) lfmgemv_release_buf(h);
    free(w); free(wsigned); free(act); free(dst); free(acc); free(srow); free(sbuf);
    group_result("gemv", bad);
    return bad;
}

/* ============================ FFN layer ================================= */

static int upload_matrix(int *h, const uint8_t *w, int nbytes, int N, int Kdim,
                         float sa, float so, const float *sw)
{
    unsigned char *sbuf;
    size_t off = 0, chunk = 256 * 1024;
    int rc;

    if ((rc = lfmgemv_alloc_buf(nbytes, h)) != 0) return rc;
    if ((rc = lfmgemv_set_shape(*h, N, Kdim, 1)) != 0) return rc;
    while (off < (size_t)nbytes) {
        size_t n = (size_t)nbytes - off < chunk ? (size_t)nbytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, w + off, (int)n)) != 0) return rc;
        off += n;
    }
    sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
    if (!sbuf) return AEE_ENOMEMORY;
    memcpy(sbuf + 0, &N, 4);
    memcpy(sbuf + 4, &Kdim, 4);
    memcpy(sbuf + 8, &sa, 4);
    memcpy(sbuf + 12, &so, 4);
    memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(*h, sbuf, 16 + N * (int)sizeof(float));
    free(sbuf);
    return rc;
}

static int test_ffn(void)
{
    int hg = -1, hu = -1, hd = -1, rc, r, bad = 0;
    int times[12];
    size_t wb = (size_t)FFN_NH * (FFN_K / 2);
    ffn_model_t m;
    ffn_calib_t ci;
    int8_t *mid_out = NULL, *dst_out = NULL, *mid_ref = NULL, *out_ref = NULL;
    int8_t *mid_single = NULL, *out_single = NULL;
    float *yref = NULL, *ppipe = NULL;
    int bad_mid = 0, bad_out = 0;

    printf("SIM [ffn] K=%d NH=%d ntok=2\n", FFN_K, FFN_NH);
    memset(&m, 0, sizeof(m));
    if (ffn_model_synth(&m, FFN_K, FFN_NH, 2, 0x5eedu) != 0) {
        printf("SIM [ffn] model allocation failed\n");
        bad = 1;
        goto out;
    }
    if (ffn_calibrate(&m, &ci) != 0) {
        printf("SIM [ffn] calibration failed\n");
        bad = 1;
        goto out;
    }
    printf("SIM [ffn] s_in=%.6g s_mid=%.6g mq_mid=%d s_out=%.6g k_gate=%d k_up=%d\n",
           ci.s_in, ci.s_mid, (int)ci.mq_mid, ci.s_out, ci.kg, ci.ku);

    mid_out    = (int8_t *)malloc((size_t)FFN_NH + 128);
    dst_out    = (int8_t *)malloc((size_t)FFN_K + 128);
    mid_ref    = (int8_t *)malloc((size_t)FFN_NH);
    out_ref    = (int8_t *)malloc((size_t)FFN_K);
    mid_single = (int8_t *)malloc((size_t)FFN_NH);
    out_single = (int8_t *)malloc((size_t)FFN_K);
    yref       = (float *)malloc((size_t)FFN_K * sizeof(float));
    ppipe      = (float *)malloc((size_t)FFN_NH * sizeof(float));
    if (!mid_out || !dst_out || !mid_ref || !out_ref || !mid_single ||
        !out_single || !yref || !ppipe) {
        printf("SIM [ffn] host allocation failed\n");
        bad = 1;
        goto out;
    }

    rc = upload_matrix(&hg, m.wg, (int)wb, FFN_NH, FFN_K,
                       m.s_in * FFN_Q16_ONE, 1.0f, m.swg);
    if (rc == 0)
        rc = upload_matrix(&hu, m.wu, (int)wb, FFN_NH, FFN_K,
                           m.s_in * FFN_Q16_ONE, 1.0f, m.swu);
    if (rc == 0)
        rc = upload_matrix(&hd, m.wd, (int)wb, FFN_K, FFN_NH,
                           m.s_mid, m.s_out, m.swd);
    if (rc != 0) {
        printf("SIM [ffn] upload rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    printf("SIM [ffn] handles g=%d u=%d d=%d (%.1f MB int4)\n",
           hg, hu, hd, 3.0 * wb / 1e6);

    if ((rc = set_threads(hg, 1)) != 0) {
        printf("SIM [ffn] probe(set threads=1) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    print_team("before first FFN");

    rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)m.x8, FFN_K, 1,
                     (unsigned char *)dst_out, FFN_K,
                     (unsigned char *)mid_out, FFN_NH, times, 12);
    if (rc != 0) {
        printf("SIM [ffn] lfmgemv_ffn rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    memcpy(mid_single, mid_out, (size_t)FFN_NH);
    memcpy(out_single, dst_out, (size_t)FFN_K);

    /* bit-exact against the shared integer model (the on-device check) */
    ffn_int_mid(&m, 0, mid_ref, NULL);
    ffn_int_out(&m, mid_ref, out_ref, NULL);
    for (r = 0; r < FFN_NH; r++) {
        int d = (int)mid_out[r] - (int)mid_ref[r];
        if (d < 0) d = -d;
        if (d) bad_mid++;
    }
    for (r = 0; r < FFN_K; r++) if (dst_out[r] != out_ref[r]) bad_out++;
    printf("SIM [ffn] tok0 single-thread vs integer model: mid mismatches=%d/%d "
           "out mismatches=%d/%d\n", bad_mid, FFN_NH, bad_out, FFN_K);
    if (bad_mid || bad_out) bad = 1;

    {
        double c;
        ffn_ref_token(&m, 0, yref, NULL, 0);
        c = ffn_cos_i8(dst_out, yref, FFN_K, m.s_out);
        printf("SIM [ffn] tok0 cos(out vs f32 truth)=%.6f\n", c);
        if (c < 0.99) {
            printf("SIM [ffn] cos below the 0.99 gate\n");
            bad = 1;
        }
    }

    /* ---- worker team: same layer, 4 threads, must be bit-identical ---- */
    if ((rc = set_threads(hg, 4)) != 0) {
        printf("SIM [ffn] probe(set threads=4) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)m.x8, FFN_K, 1,
                     (unsigned char *)dst_out, FFN_K,
                     (unsigned char *)mid_out, FFN_NH, times, 12);
    if (rc != 0) {
        printf("SIM [ffn] lfmgemv_ffn (4 threads) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    print_team("after 4-thread FFN");
    {
        int dmid = memcmp(mid_out, mid_single, (size_t)FFN_NH) != 0;
        int dout = memcmp(dst_out, out_single, (size_t)FFN_K) != 0;
        int info[8];
        int created = 0, fallbacks = (times[8] >= 0) ? times[8] : -1;
        if (lfmgemv_team_info(info, 8) == 0) created = info[3];
        printf("SIM [ffn] tok0 4-thread vs 1-thread: mid_diff=%d out_diff=%d "
               "workers=%d fallbacks=%d\n", dmid, dout, created, fallbacks);
        if (dmid || dout) bad = 1;
        if (created < 1) {
            printf("SIM [ffn] worker team was not created\n");
            bad = 1;
        }
    }

    /* ---- second token, 4 threads, still bit-exact vs the integer model --- */
    bad_mid = 0;
    bad_out = 0;
    ffn_int_mid(&m, 1, mid_ref, NULL);
    ffn_int_out(&m, mid_ref, out_ref, NULL);
    rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)(m.x8 + (size_t)FFN_K),
                     FFN_K, 1, (unsigned char *)dst_out, FFN_K,
                     (unsigned char *)mid_out, FFN_NH, times, 12);
    if (rc != 0) {
        printf("SIM [ffn] lfmgemv_ffn tok1 rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    for (r = 0; r < FFN_NH; r++) if (mid_out[r] != mid_ref[r]) bad_mid++;
    for (r = 0; r < FFN_K; r++) if (dst_out[r] != out_ref[r]) bad_out++;
    printf("SIM [ffn] tok1 4-thread vs integer model: mid mismatches=%d/%d "
           "out mismatches=%d/%d\n", bad_mid, FFN_NH, bad_out, FFN_K);
    if (bad_mid || bad_out) bad = 1;

out:
    if (hg >= 0) lfmgemv_release_buf(hg);
    if (hu >= 0) lfmgemv_release_buf(hu);
    if (hd >= 0) lfmgemv_release_buf(hd);
    ffn_model_free(&m);
    free(mid_out); free(dst_out); free(mid_ref); free(out_ref);
    free(mid_single); free(out_single); free(yref); free(ppipe);
    group_result("ffn", bad);
    return bad;
}

/* ====================== exec framework (one list) ======================= */

/* exec ABI mirror (must match lfmgemv_impl.c) */
#define LFM_EXEC_MAGIC 0x454D464Cu
#define LFM_EXEC_VERSION 1u
#define LFM_CMD_MAX 256
enum { LFM_CMD_QUANT = 0, LFM_CMD_GEMV = 1, LFM_CMD_FFN = 2,
       LFM_CMD_ATTN = 3, LFM_CMD_CONV = 4, LFM_CMD_DUMP = 5 };
enum { LFM_SRC_SCRATCH = 0, LFM_SRC_OUT = 1 };
enum { LFM_DST_SCRATCH = 0, LFM_DST_OUT = 1, LFM_DST_ATTACHED = 2 };

typedef struct {
    uint32_t magic, version, n_cmds, cmd_size, list_bytes, scratch_bytes;
    uint32_t flags, hdr_size;
    uint64_t reserved[4];
} lfm_exec_hdr;

typedef struct {
    uint32_t type, src, dst, pad;
    int32_t h0, h1, h2, h3, dst_h;
    int32_t src_off, src_len, dst_off, dst_len;
    int32_t tbl_h, tbl_len, tbl_off;
    int32_t a0, a1, a2, a3, rt_base;
    int32_t reserved[3];
} lfm_cmd;

typedef char lfm_exec_hdr_size_check[(sizeof(lfm_exec_hdr) == 64) ? 1 : -1];
typedef char lfm_cmd_size_check[(sizeof(lfm_cmd) == 96) ? 1 : -1];

#define EX_N     2048
#define EX_K     2048
#define EX_HID   2048
#define EX_FFNW  7680
#define EX_CTX   256
#define EX_HQ    16
#define EX_HD    128
#define EX_HKV   4
#define EX_TAPS  7
#define EX_RING  8
#define EX_PAGES (EX_CTX / 32)

/* scratch layout (byte offsets, 128 B aligned) */
#define S_INT32  0                       /* 2048 int32 = 8192 */
#define S_Q8     8192                    /* quantised activation */
#define S_ACT8   10240                   /* FFN input */
#define S_Q      12288                   /* attention q, 16*128 */
#define S_BCX    14336                   /* 3*HID conv input */
#define S_STATE  20480                   /* conv ring: 2*H*8*2 */
#define SCRATCH  131072

/* result layout in the OUT sequence */
#define O_GEMV   0
#define O_FFN    2048
#define O_ATTN   4096
#define O_CONV   12288
#define O_QUANT  14336
#define OUT_BYTES 16384

#define ACT_BYTES (S_BCX + 3 * EX_HID)

typedef struct {
    int8_t *blob;
    int8_t *wq1;
    int     H, ksh;
} conv_blob_t;

static void fill_i4(uint8_t *dst, size_t bytes, uint32_t seed)
{
    size_t i;
    for (i = 0; i < bytes; i++) {
        seed = seed * 1103515245u + 12345u;
        dst[i] = (uint8_t)(seed >> 24);
    }
}

static void fill_i8(int8_t *dst, size_t n, uint32_t seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        dst[i] = (int8_t)((seed >> 24) % 255 - 127);
    }
}

/* shortconv blob builder: exact mirror of the one in exec_main.c */
static int conv_blob_build(conv_blob_t *b, int H, float s_bc, float s_y,
                           const float *wf)
{
    double kmax = 0.0;
    int ch, j, s, p, g;

    b->H = H;
    b->blob = (int8_t *)malloc((size_t)8 * H * 8 + (size_t)4 * H);
    b->wq1 = (int8_t *)malloc((size_t)H * EX_TAPS);
    if (!b->blob || !b->wq1) return -1;
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * EX_TAPS;
        float mx = 0.0f, sw;
        for (j = 0; j < EX_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > mx) mx = a;
        }
        if (!(mx > 0.0f)) mx = 1.0f;
        sw = mx / 127.0f;
        for (j = 0; j < EX_TAPS; j++)
            b->wq1[(size_t)ch * EX_TAPS + j] = (int8_t)conv_wq(w[j], sw);
        {
            double K = conv_K((double)mx, (double)s_bc, (double)s_y);
            if (K > kmax) kmax = K;
        }
    }
    b->ksh = conv_pick_ksh(kmax);
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * EX_TAPS;
        float mx = 0.0f;
        for (j = 0; j < EX_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > mx) mx = a;
        }
        if (!(mx > 0.0f)) mx = 1.0f;
        ((int32_t *)(b->blob + (size_t)8 * H * 8))[ch] =
            conv_make_kq(conv_K((double)mx, (double)s_bc, (double)s_y), b->ksh);
    }
    for (p = 0; p < EX_RING; p++) {
        for (g = 0; g < H / 16; g++) {
            for (j = 0; j < 16; j++) {
                int c = g * 16 + j;
                int8_t *dst = b->blob + ((size_t)p * (H / 16) + g) * 128 + (size_t)j * 8;
                for (s = 0; s < EX_RING; s++) {
                    int tap = conv_slot_tap(p, s);
                    dst[s] = (tap < EX_TAPS) ? b->wq1[(size_t)c * EX_TAPS + tap] : 0;
                }
            }
        }
    }
    return 0;
}

/* host ring model: same operations as xconv_bx_write/xconv_dot/pack_y */
typedef struct {
    int H, ksh;
    int8_t *wq1;
    const int32_t *kq;
    int8_t *hi, *lo;
} conv_ring_t;

static void conv_ring_init(conv_ring_t *r, const conv_blob_t *b)
{
    r->H = b->H;
    r->ksh = b->ksh;
    r->wq1 = b->wq1;
    r->kq = (const int32_t *)(b->blob + (size_t)8 * b->H * 8);
    r->hi = (int8_t *)calloc((size_t)b->H * EX_RING, 1);
    r->lo = (int8_t *)calloc((size_t)b->H * EX_RING, 1);
}

static void conv_ring_free(conv_ring_t *r)
{
    free(r->hi);
    free(r->lo);
    memset(r, 0, sizeof(*r));
}

static void conv_ring_step(conv_ring_t *r, const int8_t *bcx, int ctr, int8_t *y)
{
    int H = r->H, ch, s;
    int p = ctr & (EX_RING - 1);

    for (ch = 0; ch < H; ch++) {
        int32_t t = (int32_t)bcx[ch] * (int32_t)bcx[2 * H + ch];
        r->hi[(size_t)ch * EX_RING + p] = (int8_t)(t >> 8);
        r->lo[(size_t)ch * EX_RING + p] = (int8_t)(t & 0xff);
    }
    for (ch = 0; ch < H; ch++) {
        int32_t out32 = 0;
        for (s = 0; s < EX_RING; s++) {
            int tap = conv_slot_tap(p, s);
            int w;
            if (tap >= EX_TAPS) continue;          /* unused ring slot */
            w = r->wq1[(size_t)ch * EX_TAPS + tap];
            out32 += (int32_t)r->hi[(size_t)ch * EX_RING + s] * w * 256
                   + (int32_t)(uint8_t)r->lo[(size_t)ch * EX_RING + s] * w;
        }
        y[ch] = conv_glue(out32, bcx[H + ch], r->kq[ch], r->ksh);
    }
}

/* arena offset allocator, 128 B aligned */
typedef struct {
    uint8_t *base;
    size_t   bytes;
    size_t   off;
} arena_build_t;

static size_t ab_take(arena_build_t *a, size_t bytes)
{
    size_t at = (a->off + 127) & ~(size_t)127;
    if (at + bytes > a->bytes) {
        printf("SIM [exec] arena overflow\n");
        return 0;
    }
    a->off = at + bytes;
    return at;
}

typedef struct {
    lfm_exec_hdr *hdr;
    lfm_cmd      *cmd;
    int           n;
} list_t;

static void list_init(uint8_t *buf, size_t off, int n, list_t *l)
{
    size_t bytes = sizeof(lfm_exec_hdr) + (size_t)n * sizeof(lfm_cmd);
    l->hdr = (lfm_exec_hdr *)(buf + off);
    l->cmd = (lfm_cmd *)(buf + off + sizeof(lfm_exec_hdr));
    l->n = n;
    memset(l->hdr, 0, bytes);
    l->hdr->magic = LFM_EXEC_MAGIC;
    l->hdr->version = LFM_EXEC_VERSION;
    l->hdr->n_cmds = (uint32_t)n;
    l->hdr->cmd_size = sizeof(lfm_cmd);
    l->hdr->list_bytes = (uint32_t)bytes;
    l->hdr->scratch_bytes = SCRATCH;
    l->hdr->hdr_size = sizeof(lfm_exec_hdr);
}

static lfm_cmd *cmd_at(list_t *l, int index)
{
    lfm_cmd *c = &l->cmd[index];
    memset(c, 0, sizeof(*c));
    c->h0 = c->h1 = c->h2 = c->h3 = -1;
    c->dst_h = -1;
    c->tbl_h = -1;
    c->rt_base = -1;
    return c;
}

/* ================= activation rotation (Hadamard-32/128) ================ */

static int test_rot(void)
{
    const int K = 2048;
    const int G = K / 32;
    int bad = 0, rc, ah = -1, i, rot, r;
    uint8_t *arena = NULL;
    float *x = NULL, *wr = NULL, *xr = NULL, *ref = NULL;
    uint8_t *act = NULL, *out = NULL;
    int res[8];
    static const int ROTS[3] = { 0, 32, 128 };

    arena = (uint8_t *)memalign(128, 262144);
    x   = (float *)malloc((size_t)K * 4);
    wr  = (float *)malloc((size_t)K * 4);
    xr  = (float *)malloc((size_t)K * 4);
    ref = (float *)malloc((size_t)K * 4);
    act = (uint8_t *)malloc((size_t)K * 4);
    out = (uint8_t *)malloc((size_t)K + (size_t)G * 4 + 128);
    if (!arena || !x || !wr || !xr || !ref || !act || !out) {
        printf("SIM [rot] allocation failed\n");
        bad = 1;
        goto out;
    }
    {
        uint32_t st = 0xc0ffee11u;
        for (i = 0; i < K; i++) {
            st = st * 1664525u + 1013904223u;
            x[i] = (float)((int)((st >> 8) % 2000u) - 1000) / 300.0f;
        }
        x[7] = 12.5f; x[100] = -9.0f;   /* explicit outliers */
        memcpy(act, x, (size_t)K * 4);
    }

    printf("SIM [rot] K=%d, HVX rotation + per-group-32 int8 (canonical quantiser)\n", K);
    for (r = 0; r < 3; r++) {
        rot = ROTS[r];
        rc = lfmgemv_probe(-1, 50, rot, res, 8);
        if (rc != 0) { printf("SIM [rot] probe(50, rot=%d) rc=%d\n", rot, rc); bad = 1; continue; }
        printf("SIM [rot] rot=%-3d: %d pcycles/token, HVX-vs-scalar mismatches=%d, "
               "clamps=%d, nonzero=%d\n", rot, res[2], res[3], res[4], res[5]);
        if (res[3] != 0) bad = 1;
    }

    if (lfmgemv_probe(-1, 51, 200, res, 8) == 0)
        printf("SIM [rot] canonical quantiser only: %d pcycles/token\n", res[2]);
    if (lfmgemv_probe(-1, 50, 0, res, 8) == 0)
        printf("SIM [rot] copy baseline (8 KB): %d pcycles/token\n", res[2]);

    /* one-command exec list: f32 -> ROTQ -> int8 + f32 scales */
    memset(arena, 0, 262144);
    {
        int fd = sim_ion_register(arena, 262144);
        rc = lfmgemv_map_arena(fd, 262144, &ah);
        if (rc != 0) {
            printf("SIM [rot] map_arena rc=%d fd=%d base=%p\n", rc, fd, (void *)arena);
            bad = 1; goto out;
        }
    }
    for (r = 0; r < 3; r++) {
        list_t list;
        lfm_cmd *c;
        int nq = 0, nsc = 0, maxq = 0;
        rot = ROTS[r];
        list_init(arena, 0, 1, &list);
        c = cmd_at(&list, 0);
        c->type = 9;                    /* LFM_CMD_ROTQ */
        c->src = LFM_SRC_SCRATCH;
        c->dst = LFM_DST_OUT;
        c->src_off = 0;
        c->src_len = K * 4;
        c->dst_off = 0;
        c->dst_len = K + G * 4;
        c->a0 = rot;
        memset(out, 0xCD, (size_t)K + (size_t)G * 4);
        rc = lfmgemv_exec(ah, 0, 1, act, K * 4, NULL, 0, out,
                          K + G * 4, NULL, 0);
        if (rc != 0) { printf("SIM [rot] exec ROTQ rot=%d rc=%d\n", rot, rc); bad = 1; continue; }
        /* independent scalar reference: normalised rotation + canonical quant */
        for (i = 0; i < K; i += (rot ? rot : K)) {
            int blk = rot ? rot : K, g, j;
            memcpy(ref, x + i, (size_t)blk * 4);
            lfm_rot_blocks_f32(ref, blk, rot);
            {
                int8_t qref[2048];
                float sref[64];
                (void)lfm_quant_i8_group32(ref, blk, qref, sref);
                for (j = 0; j < blk; j++) {
                    int d = (int)(int8_t)out[i + j] - (int)qref[j];
                    if (d < 0) d = -d;
                    if (d > maxq) maxq = d;
                    if (d) nq++;
                }
                for (g = 0; g < blk / 32; g++) {
                    float a = ((float *)(out + K))[i / 32 + g];
                    float e = a - sref[g];
                    if (e < 0) e = -e;
                    if (e > 1e-5f * (sref[g] < 0 ? -sref[g] : sref[g]) + 1e-7f) nsc++;
                }
            }
        }
        printf("SIM [rot] exec rot=%-3d: q mismatches=%d (max |d|=%d), "
               "scale mismatches=%d\n", rot, nq, maxq, nsc);
        if (nq || nsc) bad = 1;
        (void)rc;
    }

    /* orthogonality: <R w, R x> == <w, x> for the same R (f32) */
    {
        uint32_t st = 0x5eed1234u;
        double d0 = 0.0, d1 = 0.0, mag = 0.0;
        for (i = 0; i < K; i++) {
            st = st * 1664525u + 1013904223u;
            wr[i] = (float)((int)((st >> 8) % 2000u) - 1000) / 700.0f;
        }
        memcpy(xr, x, (size_t)K * 4);
        for (i = 0; i < K; i++) {
            d0 += (double)wr[i] * x[i];
            mag += (double)x[i] * x[i];
        }
        lfm_rot_blocks_f32(wr, K, 32);
        lfm_rot_blocks_f32(xr, K, 32);
        for (i = 0; i < K; i++) d1 += (double)wr[i] * xr[i];
        {
            double err = d0 != 0.0 ? (d1 - d0) / (d0 < 0 ? -d0 : d0) : 0.0;
            if (err < 0) err = -err;
            printf("SIM [rot] orthogonality H32: dot=%.4f rotated=%.4f rel=%.2e\n",
                   d0, d1, err);
            if (err > 1e-5) bad = 1;
        }
    }

out:
    if (ah >= 0) lfmgemv_unmap_arena(ah);
    free(arena); free(x); free(wr); free(xr); free(ref); free(act); free(out);
    group_result("rot", bad);
    return bad;
}


static int set_scale_of(int h, int N, int Kdim, float sa, float so, const float *sw)
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

static int test_exec(void)
{
    int bad = 0, rc, i;
    uint8_t *arena = NULL;
    conv_blob_t blob;
    conv_ring_t ring;
    list_t list;
    int hg = -1, hgate = -1, hup = -1, hdown = -1;
    int hk = -1, hkt = -1, hv = -1, htbl = -1, hblob = -1;
    int ah = -1;
    int times[12];
    int8_t *act = NULL;
    int8_t *q = NULL, *bcx = NULL, *act8 = NULL;
    int32_t *src32 = NULL;
    int8_t *q8_ref = NULL;
    int8_t *out = NULL;
    uint8_t *out0 = NULL;
    int8_t *gemv_ref = NULL, *ffn_ref = NULL;
    int32_t *attn_ref = NULL, *attn_dev = NULL;
    int8_t *conv_ref = NULL, *conv_dev = NULL;
    size_t arena_bytes;
    arena_build_t ab;
    float *swbuf = NULL;
    float *cwf = NULL;
    uint32_t seed = 0x1234567u;

    printf("SIM [exec] 2048x2048 int4 GEMV + FFN %dx2048 + ATTN(16x128, ctx=%d) "
           "+ CONV H=%d, one list\n", EX_FFNW, EX_CTX, EX_HID);

    memset(&blob, 0, sizeof(blob));
    memset(&ring, 0, sizeof(ring));

    /* ---- arena layout (one mapping; written before map) ---- */
    arena_bytes = 6096;                                  /* command list + slack */
    {
        size_t ffn_g = (size_t)EX_FFNW * (EX_HID / 2);
        size_t ffn_d = (size_t)EX_HID * (EX_FFNW / 2);
        size_t conv_blob_sz = (size_t)8 * EX_HID * 8 + (size_t)4 * EX_HID;
        arena_bytes += (size_t)EX_N * (EX_K / 2);
        arena_bytes += 2 * ffn_g + ffn_d;
        arena_bytes += conv_blob_sz;
        arena_bytes += (size_t)EX_HKV * EX_CTX * 128 * 2;   /* K + V */
        arena_bytes += (size_t)EX_HKV * EX_CTX * 4;         /* KT */
        arena_bytes += EX_PAGES * 4;                        /* table */
        arena_bytes += 4096 + 64 * 128;                     /* alignment slack */
    }
    arena_bytes = (arena_bytes + 4095u) & ~(size_t)4095u;   /* ION page size */
    arena = (uint8_t *)memalign(4096, arena_bytes);
    if (!arena) {
        printf("SIM [exec] arena allocation failed (%zu bytes)\n", arena_bytes);
        bad = 1;
        goto out;
    }
    memset(arena, 0x88, arena_bytes);   /* valid int4 default */
    ab.base = arena;
    ab.bytes = arena_bytes;
    ab.off = 6096;                      /* command lists live in [0, 6096) */

    {
        size_t off_gemv  = ab_take(&ab, (size_t)EX_N * (EX_K / 2));
        size_t off_gate  = ab_take(&ab, (size_t)EX_FFNW * (EX_HID / 2));
        size_t off_up    = ab_take(&ab, (size_t)EX_FFNW * (EX_HID / 2));
        size_t off_down  = ab_take(&ab, (size_t)EX_HID * (EX_FFNW / 2));
        size_t off_blob  = ab_take(&ab, (size_t)8 * EX_HID * 8 + (size_t)4 * EX_HID);
        size_t off_k     = ab_take(&ab, (size_t)EX_HKV * EX_CTX * 128);
        size_t off_v     = ab_take(&ab, (size_t)EX_HKV * EX_CTX * 128);
        size_t off_kt    = ab_take(&ab, (size_t)EX_HKV * EX_CTX * 4);
        size_t off_tbl   = ab_take(&ab, EX_PAGES * 4);

        if (ab.off == 0 || off_gemv == 0) { bad = 1; goto out; }

        fill_i4(arena + off_gemv, (size_t)EX_N * (EX_K / 2), 0x91u);
        fill_i4(arena + off_gate, (size_t)EX_FFNW * (EX_HID / 2), 0xa1u);
        fill_i4(arena + off_up,   (size_t)EX_FFNW * (EX_HID / 2), 0xa2u);
        fill_i4(arena + off_down, (size_t)EX_HID * (EX_FFNW / 2), 0xa3u);
        fill_i4(arena + off_k, (size_t)EX_HKV * EX_CTX * 128, 0x77u);
        fill_i8((int8_t *)(arena + off_v), (size_t)EX_HKV * EX_CTX * 128, 0x78u);
        for (i = 0; i < EX_HKV * EX_CTX; i++)
            ((int32_t *)(arena + off_kt))[i] = 30000 + (i % 4096);
        for (i = 0; i < EX_PAGES; i++) ((int32_t *)(arena + off_tbl))[i] = i;

        /* conv weights: same generator as exec_main.c */
        cwf = (float *)malloc((size_t)EX_HID * EX_TAPS * sizeof(float));
        if (!cwf) { printf("SIM [exec] cwf allocation failed\n"); bad = 1; goto out; }
        for (i = 0; i < EX_HID * EX_TAPS; i++) {
            seed = seed * 1103515245u + 12345u;
            cwf[i] = ((float)((int)(seed >> 20) % 2001) - 1000.0f) * 1.0e-3f;
        }
        if (conv_blob_build(&blob, EX_HID, 0.01f, 0.05f, cwf) != 0) {
            printf("SIM [exec] conv blob build failed\n");
            bad = 1;
            goto out;
        }
        memcpy(arena + off_blob, blob.blob,
               (size_t)8 * EX_HID * 8 + (size_t)4 * EX_HID);

        /* register the arena as a fake ION buffer and map it */
        {
            int fd = sim_ion_register(arena, arena_bytes);
            if (fd < 0) { printf("SIM [exec] fake fd registration failed\n"); bad = 1; goto out; }
            rc = lfmgemv_map_arena(fd, (int)arena_bytes, &ah);
            if (rc != 0) { printf("SIM [exec] map_arena rc=%d\n", rc); bad = 1; goto out; }
        }
        printf("SIM [exec] arena %d: %.1f MiB mapped once\n",
               ah, (double)arena_bytes / 1048576.0);

        /* views + shapes + scales */
        if ((rc = lfmgemv_alloc_view(ah, (int)off_gemv, (int)((size_t)EX_N * (EX_K / 2)), &hg)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_gate, (int)((size_t)EX_FFNW * (EX_HID / 2)), &hgate)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_up, (int)((size_t)EX_FFNW * (EX_HID / 2)), &hup)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_down, (int)((size_t)EX_HID * (EX_FFNW / 2)), &hdown)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_blob,
                                     (int)((size_t)8 * EX_HID * 8 + (size_t)4 * EX_HID), &hblob)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_k, (int)((size_t)EX_HKV * EX_CTX * 128), &hk)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_v, (int)((size_t)EX_HKV * EX_CTX * 128), &hv)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_kt, (int)((size_t)EX_HKV * EX_CTX * 4), &hkt)) != 0 ||
            (rc = lfmgemv_alloc_view(ah, (int)off_tbl, EX_PAGES * 4, &htbl)) != 0) {
            printf("SIM [exec] alloc_view rc=%d\n", rc);
            bad = 1;
            goto out;
        }

        swbuf = (float *)malloc((size_t)EX_FFNW * sizeof(float));
        if (!swbuf) { printf("SIM [exec] swbuf allocation failed\n"); bad = 1; goto out; }

        if ((rc = lfmgemv_set_shape(hg, EX_N, EX_K, 1)) != 0) {
            printf("SIM [exec] shape gemv rc=%d\n", rc); bad = 1; goto out;
        }
        for (i = 0; i < EX_N; i++) swbuf[i] = 1.2e-3f;
        if ((rc = set_scale_of(hg, EX_N, EX_K, 0.01f, 0.05f, swbuf)) != 0) {
            printf("SIM [exec] scale gemv rc=%d\n", rc); bad = 1; goto out;
        }
        if ((rc = lfmgemv_set_shape(hgate, EX_FFNW, EX_HID, 1)) != 0 ||
            (rc = lfmgemv_set_shape(hup, EX_FFNW, EX_HID, 1)) != 0) {
            printf("SIM [exec] shape gate/up rc=%d\n", rc); bad = 1; goto out;
        }
        for (i = 0; i < EX_FFNW; i++) swbuf[i] = 1.0e-3f;
        if ((rc = set_scale_of(hgate, EX_FFNW, EX_HID, 0.005f, 1.0f, swbuf)) != 0 ||
            (rc = set_scale_of(hup, EX_FFNW, EX_HID, 0.005f, 1.0f, swbuf)) != 0) {
            printf("SIM [exec] scale gate/up rc=%d\n", rc); bad = 1; goto out;
        }
        if ((rc = lfmgemv_set_shape(hdown, EX_HID, EX_FFNW, 1)) != 0) {
            printf("SIM [exec] shape down rc=%d\n", rc); bad = 1; goto out;
        }
        for (i = 0; i < EX_HID; i++) swbuf[i] = 1.1e-3f;
        if ((rc = set_scale_of(hdown, EX_HID, EX_FFNW, 0.02f, 0.03f, swbuf)) != 0) {
            printf("SIM [exec] scale down rc=%d\n", rc); bad = 1; goto out;
        }

        printf("SIM [exec] handles gemv=%d gate=%d up=%d down=%d conv=%d "
               "k=%d v=%d kt=%d tbl=%d\n", hg, hgate, hup, hdown, hblob,
               hk, hv, hkt, htbl);
    }

    if ((rc = lfmgemv_reserve_scratch(SCRATCH)) != 0) {
        printf("SIM [exec] reserve_scratch rc=%d\n", rc);
        bad = 1;
        goto out;
    }

    /* ---- host inputs ---- */
    act = (int8_t *)calloc(1, ACT_BYTES);
    q = (int8_t *)malloc(EX_HQ * EX_HD);
    bcx = (int8_t *)malloc(3 * EX_HID);
    act8 = (int8_t *)malloc(EX_HID);
    src32 = (int32_t *)malloc(EX_HID * sizeof(int32_t));
    q8_ref = (int8_t *)malloc(EX_HID);
    out = (int8_t *)malloc(OUT_BYTES);
    out0 = (uint8_t *)malloc(OUT_BYTES);
    gemv_ref = (int8_t *)malloc(EX_N);
    ffn_ref = (int8_t *)malloc(EX_HID);
    attn_ref = (int32_t *)malloc(EX_HQ * EX_HD * sizeof(int32_t));
    attn_dev = (int32_t *)malloc(EX_HQ * EX_HD * sizeof(int32_t));
    conv_ref = (int8_t *)malloc(EX_HID);
    conv_dev = (int8_t *)malloc(EX_HID);
    if (!act || !q || !bcx || !act8 || !src32 || !q8_ref || !out || !out0 ||
        !gemv_ref || !ffn_ref || !attn_ref || !attn_dev || !conv_ref || !conv_dev) {
        printf("SIM [exec] host allocation failed\n");
        bad = 1;
        goto out;
    }

    for (i = 0; i < EX_HID; i++) src32[i] = (int32_t)((i * 2654435761u) & 0x3ffff) - 0x1ffff;
    for (i = 0; i < EX_HID; i++) {
        int64_t v = ((int64_t)src32[i] * 4096 + 32768) >> 16;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q8_ref[i] = (int8_t)v;
    }
    fill_i8(act8, EX_HID, 0x55u);
    fill_i8(q, EX_HQ * EX_HD, 0x66u);
    fill_i8(bcx, 3 * EX_HID, 0x77u);

    /* act blob: everything one exec copies into scratch[0..ACT_BYTES) */
    memcpy(act + S_INT32, src32, (size_t)EX_HID * 4);
    memset(act + S_Q8, 0xaa, EX_HID);          /* must be overwritten by QUANT */
    memcpy(act + S_ACT8, act8, EX_HID);
    memcpy(act + S_Q, q, EX_HQ * EX_HD);
    memcpy(act + S_BCX, bcx, 3 * EX_HID);

    /* ---- command list: 6 commands, all sources live in the act blob ---- */
    list_init(arena, 0, 6, &list);
    {
        lfm_cmd *c = cmd_at(&list, 0);
        c->type = LFM_CMD_QUANT;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_SCRATCH;
        c->src_off = S_INT32; c->src_len = EX_HID * 4;
        c->dst_off = S_Q8;    c->dst_len = EX_HID;
        c->a0 = 4096;

        c = cmd_at(&list, 1);
        c->type = LFM_CMD_GEMV;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_OUT;
        c->h0 = hg; c->a0 = 1; c->a1 = 1;
        c->src_off = S_Q8; c->src_len = EX_K;
        c->dst_off = O_GEMV; c->dst_len = EX_N;

        c = cmd_at(&list, 2);
        c->type = LFM_CMD_FFN;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_OUT;
        c->h0 = hgate; c->h1 = hup; c->h2 = hdown;
        c->src_off = S_ACT8; c->src_len = EX_HID;
        c->dst_off = O_FFN;  c->dst_len = EX_HID;
        c->a0 = 1;

        c = cmd_at(&list, 3);
        c->type = LFM_CMD_ATTN;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_OUT;
        c->h0 = hk; c->h1 = hkt; c->h2 = hv; c->h3 = 0;
        c->src_off = S_Q; c->src_len = EX_HQ * EX_HD;
        c->dst_off = O_ATTN; c->dst_len = EX_HQ * EX_HD * 4;
        c->tbl_h = htbl; c->tbl_len = EX_PAGES;
        c->a0 = EX_CTX - 1; c->a1 = EX_CTX; c->a2 = 0; c->a3 = 1;

        c = cmd_at(&list, 4);
        c->type = LFM_CMD_CONV;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_OUT;
        c->h0 = hblob;
        c->src_off = S_BCX; c->src_len = 3 * EX_HID;
        c->dst_off = O_CONV; c->dst_len = EX_HID;
        c->tbl_off = S_STATE;
        c->a0 = EX_HID; c->a1 = blob.ksh; c->a2 = 0; c->a3 = 0;
        c->rt_base = 0;

        c = cmd_at(&list, 5);
        c->type = LFM_CMD_DUMP;
        c->src = LFM_SRC_SCRATCH; c->dst = LFM_DST_OUT;
        c->src_off = S_Q8; c->src_len = EX_HID;
        c->dst_off = O_QUANT; c->dst_len = EX_HID;
    }

    if ((rc = set_threads(hg, 1)) != 0) {
        printf("SIM [exec] probe(set threads=1) rc=%d\n", rc);
        bad = 1;
        goto out;
    }

    /* ---- run the list for token 0 ---- */
    {
        int rt0[1] = { 0 };
        rc = lfmgemv_exec(ah, 0, list.n, (const unsigned char *)act, ACT_BYTES,
                          rt0, 1, (unsigned char *)out, OUT_BYTES, times, 12);
        if (rc != 0) {
            printf("SIM [exec] exec(token0) rc=0x%x at cmd=%d\n", rc, times[1]);
            bad = 1;
            goto out;
        }
        printf("SIM [exec] exec(token0): dsp=%d us cmds=%d fallbacks=%d\n",
               times[0], times[2], times[3]);
        memcpy(out0, out, OUT_BYTES);
        memcpy(conv_dev, out + O_CONV, EX_HID);
    }

    /* QUANT + GEMV chain: the GEMV must see the quantised activation */
    {
        int nbad = 0;
        for (i = 0; i < EX_HID; i++)
            if ((int8_t)out[O_QUANT + i] != q8_ref[i]) nbad++;
        printf("SIM [exec] QUANT vs host formula: mismatches=%d/%d\n", nbad, EX_HID);
        if (nbad) bad = 1;
    }
    {
        int nbad = 0;
        rc = lfmgemv_gemv(hg, 1, (const unsigned char *)q8_ref, EX_K, 1,
                          (unsigned char *)gemv_ref, EX_N, times, 12);
        if (rc != 0) { printf("SIM [exec] direct gemv rc=%d\n", rc); bad = 1; goto out; }
        for (i = 0; i < EX_N; i++) if (out[O_GEMV + i] != gemv_ref[i]) nbad++;
        printf("SIM [exec] GEMV(quant-chain) vs direct call: mismatches=%d/%d\n",
               nbad, EX_N);
        if (nbad) bad = 1;
    }

    /* FFN exec vs direct call (same resident weights) */
    {
        int nbad = 0;
        int8_t *mid = (int8_t *)malloc(EX_FFNW);
        if (!mid) { printf("SIM [exec] mid allocation failed\n"); bad = 1; goto out; }
        rc = lfmgemv_ffn(hgate, hup, hdown, (const unsigned char *)act8, EX_HID, 1,
                         (unsigned char *)ffn_ref, EX_HID, (unsigned char *)mid,
                         EX_FFNW, times, 12);
        free(mid);
        if (rc != 0) { printf("SIM [exec] direct ffn rc=%d\n", rc); bad = 1; goto out; }
        for (i = 0; i < EX_HID; i++) if (out[O_FFN + i] != ffn_ref[i]) nbad++;
        printf("SIM [exec] FFN vs direct call: mismatches=%d/%d\n", nbad, EX_HID);
        if (nbad) bad = 1;
    }

    /* ATTN exec vs direct call */
    {
        int nbad = 0;
        static const int tbl[EX_PAGES] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        rc = lfmgemv_attn(hk, hkt, hv, 0, (const unsigned char *)q, EX_HQ * EX_HD,
                          tbl, EX_PAGES, EX_CTX - 1, EX_CTX, 0, 1,
                          (unsigned char *)attn_ref, EX_HQ * EX_HD * 4, times, 12);
        if (rc != 0) { printf("SIM [exec] direct attn rc=%d\n", rc); bad = 1; goto out; }
        memcpy(attn_dev, out + O_ATTN, (size_t)EX_HQ * EX_HD * 4);
        for (i = 0; i < EX_HQ * EX_HD; i++) if (attn_dev[i] != attn_ref[i]) nbad++;
        printf("SIM [exec] ATTN vs direct call: mismatches=%d/%d\n",
               nbad, EX_HQ * EX_HD);
        if (nbad) bad = 1;
    }

    /* CONV vs host ring model, token 0 */
    conv_ring_init(&ring, &blob);
    if (!ring.hi || !ring.lo) {
        printf("SIM [exec] ring allocation failed\n");
        bad = 1;
        goto out;
    }
    {
        int nbad = 0;
        conv_ring_step(&ring, bcx, 0, conv_ref);
        for (i = 0; i < EX_HID; i++) if (conv_dev[i] != conv_ref[i]) nbad++;
        printf("SIM [exec] CONV token0 vs host ring model: mismatches=%d/%d\n",
               nbad, EX_HID);
        if (nbad) bad = 1;
    }

    /* ---- token 1: same list, counter from rt[]; ring state must persist --- */
    {
        int rt1[1] = { 1 };
        int nbad = 0, nchanged = 0;
        rc = lfmgemv_exec(ah, 0, list.n, (const unsigned char *)act, ACT_BYTES,
                          rt1, 1, (unsigned char *)out, OUT_BYTES, times, 12);
        if (rc != 0) {
            printf("SIM [exec] exec(token1) rc=0x%x at cmd=%d\n", rc, times[1]);
            bad = 1;
            goto out;
        }
        conv_ring_step(&ring, bcx, 1, conv_ref);
        for (i = 0; i < EX_HID; i++) if (out[O_CONV + i] != conv_ref[i]) nbad++;
        /* the other commands do not depend on rt[] and must be unchanged */
        for (i = 0; i < OUT_BYTES; i++) {
            if (i >= O_CONV && i < O_CONV + EX_HID) continue;
            if ((uint8_t)out[i] != out0[i]) nchanged++;
        }
        printf("SIM [exec] CONV token1 via rt[] + persistent ring: mismatches=%d/%d; "
               "other outputs changed=%d\n", nbad, EX_HID, nchanged);
        if (nbad || nchanged) bad = 1;
    }

    /* ---- reset_state: token 0 must reproduce exactly ---- */
    if ((rc = lfmgemv_reset_state(0)) != 0) {
        printf("SIM [exec] reset_state rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    {
        int rt0[1] = { 0 };
        int nbad = 0;
        rc = lfmgemv_exec(ah, 0, list.n, (const unsigned char *)act, ACT_BYTES,
                          rt0, 1, (unsigned char *)out, OUT_BYTES, times, 12);
        if (rc != 0) {
            printf("SIM [exec] exec(after reset) rc=0x%x at cmd=%d\n", rc, times[1]);
            bad = 1;
            goto out;
        }
        for (i = 0; i < OUT_BYTES; i++) if ((uint8_t)out[i] != out0[i]) nbad++;
        printf("SIM [exec] reset_state identity: changed bytes=%d/%d\n", nbad, OUT_BYTES);
        if (nbad) bad = 1;
    }

out:
    if (hg >= 0) lfmgemv_release_buf(hg);
    if (hgate >= 0) lfmgemv_release_buf(hgate);
    if (hup >= 0) lfmgemv_release_buf(hup);
    if (hdown >= 0) lfmgemv_release_buf(hdown);
    if (hblob >= 0) lfmgemv_release_buf(hblob);
    if (hk >= 0) lfmgemv_release_buf(hk);
    if (hv >= 0) lfmgemv_release_buf(hv);
    if (hkt >= 0) lfmgemv_release_buf(hkt);
    if (htbl >= 0) lfmgemv_release_buf(htbl);
    if (ah >= 0) lfmgemv_unmap_arena(ah);
    conv_ring_free(&ring);
    free(blob.blob);
    free(blob.wq1);
    free(arena);
    free(act); free(q); free(bcx); free(act8); free(src32); free(q8_ref);
    free(out); free(out0); free(gemv_ref); free(ffn_ref);
    free(attn_ref); free(attn_dev); free(conv_ref); free(conv_dev);
    free(swbuf); free(cwf);
    group_result("exec", bad);
    return bad;
}

/* ===================== worker team, 1 vs 4 threads ====================== */

static int test_team(void)
{
    int bad = 0, h = -1, rc, r, i;
    size_t wbytes = (size_t)GEMV_N * (GEMV_K / 2);
    uint8_t *w = NULL;
    int8_t *act = NULL;
    int32_t *acc1 = NULL, *acc2 = NULL;
    int times[12];

    printf("SIM [team] GEMV %dx%d int32 M=1: threads=1 vs threads=4\n",
           GEMV_N, GEMV_K);

    w    = (uint8_t *)malloc(wbytes);
    act  = (int8_t *)malloc((size_t)GEMV_K + 128);
    acc1 = (int32_t *)malloc((size_t)GEMV_N * sizeof(int32_t));
    acc2 = (int32_t *)malloc((size_t)GEMV_N * sizeof(int32_t));
    if (!w || !act || !acc1 || !acc2) {
        printf("SIM [team] host allocation failed\n");
        bad = 1;
        goto out;
    }
    {
        uint32_t st = 0x6b6bu;
        for (i = 0; i < (int)wbytes; i++) w[i] = (uint8_t)(lcg(&st) >> 24);
        for (i = 0; i < GEMV_K; i++) act[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }

    rc = lfmgemv_alloc_buf((int)wbytes, &h);
    if (rc == 0) rc = lfmgemv_set_shape(h, GEMV_N, GEMV_K, 0);
    if (rc == 0) rc = lfmgemv_load_buf(h, 0, w, (int)wbytes);
    if (rc != 0) {
        printf("SIM [team] setup rc=%d\n", rc);
        bad = 1;
        goto out;
    }

    if ((rc = set_threads(h, 1)) != 0) {
        printf("SIM [team] probe(threads=1) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    print_team("threads=1");
    rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, GEMV_K, 1,
                      (unsigned char *)acc1, GEMV_N * (int)sizeof(int32_t),
                      times, 12);
    if (rc != 0) { printf("SIM [team] gemv single rc=%d\n", rc); bad = 1; goto out; }
    printf("SIM [team] threads=1: kernel_min=%d cycles\n", times[0]);

    if ((rc = set_threads(h, 4)) != 0) {
        printf("SIM [team] probe(threads=4) rc=%d\n", rc);
        bad = 1;
        goto out;
    }
    rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, GEMV_K, 1,
                      (unsigned char *)acc2, GEMV_N * (int)sizeof(int32_t),
                      times, 12);
    if (rc != 0) { printf("SIM [team] gemv multi rc=%d\n", rc); bad = 1; goto out; }
    print_team("threads=4");

    {
        int nbad = 0;
        int info[8];
        int created = 0, fallbacks = (times[8] >= 0) ? times[8] : -1;
        if (lfmgemv_team_info(info, 8) == 0) created = info[3];
        for (r = 0; r < GEMV_N; r++) if (acc1[r] != acc2[r]) nbad++;
        printf("SIM [team] 4-thread vs 1-thread rows: mismatches=%d/%d "
               "workers=%d fallbacks=%d kernel_min=%d cycles\n",
               nbad, GEMV_N, created, fallbacks, times[0]);
        if (nbad) bad = 1;
        if (created < 1) {
            printf("SIM [team] worker team was not created (qurt_thread_create failed)\n");
            bad = 1;
        }
    }

out:
    if (h >= 0) lfmgemv_release_buf(h);
    free(w); free(act); free(acc1); free(acc2);
    group_result("team", bad);
    return bad;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("SIM lfm3 DSP stand: QuRT v68 under qemu-system-hexagon V68N_1024\n");
    printf("SIM exec framework: lfmgemv_impl.c linked directly, no FastRPC\n\n");

    test_gemv();
    test_ffn();
    test_exec();
    test_rot();
    test_team();

    printf("\nSIM RESULT: %s (%d/%d groups failed)\n",
           g_failed ? "FAIL" : "PASS", g_failed, g_groups);
    return g_failed ? 1 : 0;
}
