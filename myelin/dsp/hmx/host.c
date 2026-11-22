#include "mx4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <unistd.h>

#define SLOT 2048u
#define TILE 32

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    int rc;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                (uint32_t)sizeof(um));
    printf("unsigned PD: rc=%d\n", rc);
    return rc;
}

static uint32_t rng = 987654321u;
static uint32_t rnd(void)
{
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

/* ---- tile layouts (must match mx4_impl.c) -------------------------- */

static int g_act_layout;
static int g_act_delta;
static int g_act_line;
static int g_act_cshift;
static int g_slot0, g_slotstep;

static void pack_act_slot(uint8_t *dst, const uint8_t *A, int M, int K,
                          int mt, int kc, int amax)
{
    int s, c;
    memset(dst, 0, SLOT);
    for (s = 0; s < TILE; s++) {
        int row = mt * TILE + s;
        for (c = 0; c < TILE; c++) {
            int ch = kc * TILE + c;
            unsigned off;
            if (g_act_layout == 100)
                off = (unsigned)(c >> 2) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)(c & 3) * 32u
                    + (unsigned)(g_slot0 + g_slotstep * (s & 3)) * 2u;
            else if (g_act_layout == 91)
                off = (unsigned)(c >> 2) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)(c & 3) * 32u + 8u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 92)
                off = (unsigned)(c >> 2) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)(c & 3) * 32u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 93)
                off = (unsigned)(c >> 2) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)(c & 3) * 32u + 16u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 90) {
                int cc = c + g_act_cshift;
                if (cc < 0 || cc > 31) { continue; }
                off = (unsigned)(cc >> 4) * 1024u + (unsigned)(s >> 1) * 128u
                    + (unsigned)(cc & 15) * 8u + (unsigned)g_act_delta
                    + (unsigned)(s & 1) * 2u + (unsigned)g_act_line;
            }
            else if (g_act_layout == 88)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 1) * 128u
                    + (unsigned)(c & 15) * 8u + 2u + (unsigned)(s & 1) * 2u;
            else if (g_act_layout == 86)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)(c & 15) * 8u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 82)
                off = (unsigned)(c >> 4) * 2048u + (unsigned)s * 128u
                    + (unsigned)(c & 15) * 8u + 4u;
            else if (g_act_layout == 80)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 2u + (unsigned)(s & 3) * 8u;
            else if (g_act_layout == 81)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 4u + (unsigned)(s & 3) * 8u;
            else if (g_act_layout == 77)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 8u + 2u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 76)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 4u + (unsigned)(s & 3);
            else if (g_act_layout == 75)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 8u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 74)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 8u + (unsigned)(s & 3) * 2u + 4u;
            else if (g_act_layout == 72)
                off = (unsigned)(c >> 4) * 1024u + (unsigned)(s >> 2) * 128u
                    + (unsigned)c * 8u + 4u + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 70)
                off = (unsigned)(s >> 2) * 128u + (unsigned)c * 8u
                    + (unsigned)(s & 3) * 2u;
            else if (g_act_layout == 110)
                off = 4u + (unsigned)(s >> 2) * 128u + (unsigned)c * 4u + (unsigned)(s & 3);
            else if (g_act_layout == 111)
                off = 4u + (unsigned)(s >> 2) * 128u + (unsigned)c * 8u
                    + (unsigned)(s & 3) * 2u + 1u;
            else if (g_act_layout == 112)
                off = 4u + (unsigned)(s >> 2) * 128u + (unsigned)c * 8u
                    + (unsigned)(s & 3) * 2u;
            else if (g_act_layout >= 100) {
                int v = g_act_layout - 100;
                if (v == 0 || v == 1)
                    off = (unsigned)(s >> 2) * 128u + (unsigned)c * 8u
                        + (unsigned)(s & 3) * 2u + (unsigned)(v & 1);
                else if (v == 2 || v == 3)
                    off = (unsigned)(s >> 3) * 128u + (unsigned)c * 16u
                        + (unsigned)(s & 7) * 2u + (unsigned)(v & 1);
                else if (v == 4 || v == 5)
                    off = (unsigned)(s >> 2) * 128u + (unsigned)(s & 3) * 8u
                        + (unsigned)c * 2u + (unsigned)(v & 1);
                else
                    off = (unsigned)(s >> 2) * 128u + (unsigned)c * 4u
                        + (unsigned)(s & 3);
            } else if (g_act_layout == 4) {
                /* 128 B line = 4 spatials x 16 channels x 2 B; 1 KB = 32 sp x 16 ch */
                unsigned sub = (unsigned)(c >> 4), cc = (unsigned)(c & 15);
                off = sub * 1024u + (unsigned)(s >> 2) * 128u + cc * 8u
                    + (unsigned)(s & 3) * 2u;
            } else if (g_act_layout == 3)
                off = (unsigned)(s >> 1) * 128u + (unsigned)c * 4u
                    + (unsigned)(s & 1) * 2u;
            else if (g_act_layout == 1)
                off = (unsigned)(s >> 1) * 128u + (unsigned)c * 4u
                    + (unsigned)(s & 1) * 2u;
            else if (g_act_layout == 2)
                off = (unsigned)(c >> 2) * 128u + (unsigned)s * 4u
                    + (unsigned)(c & 3);
            else
                off = (unsigned)(s >> 2) * 128u + (unsigned)c * 4u
                    + (unsigned)(s & 3);
            if (row < M) {
                unsigned char v = (unsigned char)((uint32_t)A[(size_t)row * K + ch] % (unsigned)(amax + 1));
                dst[off] = v;
                if (g_act_layout >= 70 && g_act_layout != 76 && g_act_layout != 80 &&
                    g_act_layout != 81 && g_act_layout != 82 && g_act_layout != 110 &&
                    g_act_layout != 111 && off + 1 < SLOT)
                    dst[off + 1] = v;
            }
        }
    }
}

static int g_wgt_layout;

static void pack_wgt_slot(uint8_t *dst, const int8_t *B, int N, int K,
                          int nt, int kc)
{
    int n, k;
    memset(dst, 0, SLOT);
    for (n = 0; n < TILE; n++) {
        int col = nt * TILE + n;
        for (k = 0; k < TILE; k++) {
            int ch = kc * TILE + k;
            unsigned off;
            if (g_wgt_layout == 84)
                off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 2) * 128u
                    + (unsigned)(n & 15) * 8u + (unsigned)(k & 3) * 2u;
            else if (g_wgt_layout == 85)
                off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 2) * 128u
                    + (unsigned)(n & 15) * 8u + (unsigned)(k & 3) * 2u + 1u;
            else if (g_wgt_layout == 80)
                off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 1) * 128u
                    + (unsigned)(n & 15) * 8u + 6u + (unsigned)(k & 1) * 2u;
            else if (g_wgt_layout == 81)
                off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 1) * 128u
                    + (unsigned)(n & 15) * 8u + 4u + (unsigned)(k & 1) * 4u;
            else if (g_wgt_layout == 73)
                off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 1) * 128u
                    + (unsigned)(n & 15) * 8u + 4u + (unsigned)(k & 1) * 2u;
            else if (g_wgt_layout == 71)
                off = (unsigned)(k >> 1) * 128u + (unsigned)(n & 15) * 8u + 4u
                    + (unsigned)(k & 1) * 2u;
            else if (g_wgt_layout == 36)
                off = 4u + (unsigned)(k >> 1) * 128u + (unsigned)n * 8u
                    + (unsigned)(k & 1) * 2u + 1u;
            else if (g_wgt_layout == 37)
                off = 4u + (unsigned)(k >> 1) * 128u + (unsigned)n * 8u
                    + (unsigned)(k & 1) * 2u;
            else if (g_wgt_layout == 38)
                off = 4u + (unsigned)(k >> 1) * 128u + (unsigned)n * 4u
                    + (unsigned)(k & 1) * 2u + 1u;
            else if (g_wgt_layout == 30)
                off = 4u + (unsigned)(k >> 2) * 128u + (unsigned)n * 4u + (unsigned)(k & 3);
            else if (g_wgt_layout == 31)
                off = 4u + (unsigned)(k >> 2) * 128u + (unsigned)n * 8u
                    + (unsigned)(k & 3) * 2u + 1u;
            else if (g_wgt_layout == 32)
                off = 4u + (unsigned)(k >> 2) * 128u + (unsigned)n * 8u
                    + (unsigned)(k & 3) * 2u;
            else if (g_wgt_layout >= 10) {
                int v = g_wgt_layout - 10;
                if (v == 0)
                    off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 2) * 128u
                        + (unsigned)(n & 15) * 8u + (unsigned)(k & 3) * 2u + 1u;
                else if (v == 1)
                    off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 2) * 128u
                        + (unsigned)(n & 15) * 8u + (unsigned)(k & 3) * 2u;
                else if (v == 2)
                    off = (unsigned)(k >> 1) * 128u + (unsigned)n * 8u
                        + (unsigned)(k & 1) * 2u + 1u;
                else if (v == 3)
                    off = (unsigned)(n >> 4) * 1024u + (unsigned)(k >> 1) * 128u
                        + (unsigned)(n & 15) * 8u + (unsigned)(k & 1) * 2u + 1u;
                else
                    off = (unsigned)(k >> 3) * 128u + (unsigned)n * 8u
                        + (unsigned)(k & 7);
            } else if (g_wgt_layout == 2)
                off = (unsigned)(k >> 1) * 128u + (unsigned)n * 4u
                    + (unsigned)(k & 1) * 2u + 1u;
            else if (g_wgt_layout == 1)
                off = (unsigned)(k >> 1) * 128u + (unsigned)n * 4u
                    + (unsigned)(k & 1) * 2u;
            else
                off = (unsigned)(k >> 2) * 128u + (unsigned)n * 4u
                    + (unsigned)(k & 3);
            if (col < N) {
                unsigned char v = (uint8_t)B[(size_t)col * K + ch];
                dst[off] = v;
                if (g_wgt_layout >= 71 && g_wgt_layout != 85 && off + 1 < SLOT) dst[off + 1] = v;
            }
        }
    }
}

static void prepack_act(uint8_t *dst, const uint8_t *A, int M, int K, int amax)
{
    int m32 = (M + TILE - 1) / TILE, ncr = K / TILE, mt, kc;
    for (mt = 0; mt < m32; mt++)
        for (kc = 0; kc < ncr; kc++)
            pack_act_slot(dst + (size_t)(mt * ncr + kc) * SLOT, A, M, K, mt, kc, amax);
}

static void prepack_wgt(uint8_t *dst, const int8_t *B, int N, int K)
{
    int n32 = (N + TILE - 1) / TILE, ncr = K / TILE, nt, kc;
    for (nt = 0; nt < n32; nt++)
        for (kc = 0; kc < ncr; kc++)
            pack_wgt_slot(dst + (size_t)(nt * ncr + kc) * SLOT, B, N, K, nt, kc);
}

/* unpack one 2 KB output slot into C[rows][cols] starting at (m0,n0) */
static void unpack_slot(const uint8_t *src, int rows, int cols,
                        uint8_t *C, int ldc, int m0, int n0)
{
    int s, n;
    for (s = 0; s < rows; s++) {
        const uint8_t *line = src + (unsigned)(s >> 2) * 128u + (unsigned)(s & 3);
        uint8_t *dst = C + (size_t)(m0 + s) * ldc + n0;
        for (n = 0; n < cols; n++)
            dst[n] = line[(unsigned)n * 4u];
    }
}

/* ---- reference ----------------------------------------------------- */

static void ref_gemm(const uint8_t *A, const int8_t *B, int M, int N, int K,
                     uint8_t *C)
{
    int m, n, k;
    for (m = 0; m < M; m++) {
        const uint8_t *ar = A + (size_t)m * K;
        for (n = 0; n < N; n++) {
            const int8_t *br = B + (size_t)n * K;
            int32_t acc = 0;
            for (k = 0; k < K; k++)
                acc += (int32_t)ar[k] * (int32_t)br[k];
            C[(size_t)m * N + n] = (uint8_t)(acc < 0 ? 0 : (acc > 255 ? 255 : acc));
        }
    }
}

static void cmp(const uint8_t *got, const uint8_t *ref, int M, int N, int show)
{
    long long maxerr = 0, bad = 0;
    int m, n, shown = 0;
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            long long e = (long long)got[(size_t)m * N + n] - ref[(size_t)m * N + n];
            if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
            if (e) {
                bad++;
                if (shown < show) {
                    printf("    C[%d][%d] hw=%u ref=%u\n", m, n,
                           got[(size_t)m * N + n], ref[(size_t)m * N + n]);
                    shown++;
                }
            }
        }
    printf("  max_abs_err=%lld  mismatches=%lld / %d (%.2f%%)\n",
           maxerr, bad, M * N, 100.0 * (double)bad / (double)(M * N));
}

/* ---- steps --------------------------------------------------------- */

static int do_info(void)
{
    int r[8];
    int rc = mx4_info(r, 8);
    printf("info rc=%d: total_rc=%d page=%u pages=%u avail_rc=%d avail=%u "
           "maxpage=%u npages=%u hmxbits=%d\n",
           rc, r[0], (unsigned)r[1], (unsigned)r[2], r[3],
           (unsigned)r[4], (unsigned)r[5], (unsigned)r[6], r[7]);
    return 0;
}

static int do_recipe(void)
{
    int r[8];
    int rc = mx4_recipe(r, 8);
    printf("recipe rc=%d: status=%d lock=%d unlock=%d size=%u ctx=%u magic=0x%x\n",
           rc, r[0], r[1], r[2], (unsigned)r[3], (unsigned)r[4], (unsigned)r[7]);
    return (rc == 0 && r[0] == 0 && r[1] == 0 && r[2] == 0) ? 0 : 1;
}

/* single tile stream: 32 x 32 x 32*ncr */
static int do_tile(int ncr, int ac, int wc, int srt, int variant, int ops, int verbose)
{
    uint8_t *A, *B, *Ap, *Bp, *raw, *got, *ref;
    int K = ncr * TILE;
    int info[6];
    int rc, i;
    uint64_t t0, t1;

    A = malloc((size_t)TILE * K);
    B = malloc((size_t)TILE * K);
    Ap = malloc((size_t)ncr * SLOT);
    Bp = malloc((size_t)ncr * SLOT);
    raw = malloc(SLOT);
    got = malloc((size_t)TILE * TILE);
    ref = malloc((size_t)TILE * TILE);
    if (!A || !B || !Ap || !Bp || !raw || !got || !ref) return 1;

    for (i = 0; i < TILE * K; i++) A[i] = (uint8_t)(rnd() % 4u);
    for (i = 0; i < TILE * K; i++) B[i] = (int8_t)((int)(rnd() % 3u) - 1);

    prepack_act(Ap, A, TILE, K, 3);
    prepack_wgt(Bp, (int8_t *)B, TILE, K);

    memset(raw, 0xAA, SLOT);
    t0 = 0;
    {
        struct timespec ts, te;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        rc = mx4_tile(Ap, (int)(ncr * SLOT), Bp, (int)(ncr * SLOT),
                      ac, wc, srt, variant, ops, raw, (int)SLOT, info, 6);
        clock_gettime(CLOCK_MONOTONIC, &te);
        t0 = (uint64_t)(te.tv_sec - ts.tv_sec) * 1000000u
           + (uint64_t)(te.tv_nsec - ts.tv_nsec) / 1000u;
    }
    t1 = (uint64_t)(TILE * TILE * K);

    printf("tile ncr=%d (32x32x%d) ac=0x%x wc=0x%x srt=0x%x variant=%d ops=%d: "
           "rc=%d status=%d vtcm=%d\n", ncr, K, ac, wc, srt, variant, ops, rc,
           info[0], info[5]);
    if (rc != 0 || info[0] != 0) { printf("  FAIL status\n"); return 1; }
    if (ops != 3) { printf("  (no full multiply/store requested: stage survived)\n"); return 0; }

    unpack_slot(raw, TILE, TILE, got, TILE, 0, 0);
    ref_gemm(A, (int8_t *)B, TILE, TILE, K, ref);
    cmp(got, ref, TILE, TILE, verbose ? 8 : 0);
    printf("  rpc call: %llu us, %.1f GMAC/s (incl. rpc overhead)\n",
           (unsigned long long)t0, (double)t1 / ((double)(t0 ? t0 : 1) * 1e3));

    if (verbose) {
        int s;
        printf("  raw[0..63]:");
        for (s = 0; s < 64; s++) printf(" %u", raw[s]);
        printf("\n");
    }
    return 0;
}

static int do_sweep(void)
{
    static const int acs[] = { 0x7C, 0x87C };
    static const int wcs[] = { 1023, 2047 };
    int a, w, ok = -1;
    printf("== S1 sweep (32x32x32) ==\n");
    for (a = 0; a < 2; a++)
        for (w = 0; w < 2; w++) {
            int r = do_tile(1, acs[a], wcs[w], 0, 3, 3, 0);
            if (r == 0) ok = a * 2 + w;
        }
    printf("sweep: %s\n", ok >= 0 ? "at least one combo matched" : "NONE matched");
    return ok >= 0 ? 0 : 1;
}

static int do_gemm(int M, int N, int K, int reps, int mode, int amax,
                   int do_ref, int verbose)
{
    uint8_t *A, *B, *C, *R, *Ap, *Bp;
    int res[12];
    int rc, m32, n32, ncr, i;
    size_t ab, bb, cb;
    unsigned long long macs;
    double gmac_hmx, gmac_full;
    uint64_t wall0, wall1;
    struct timespec ts, te;

    m32 = (M + TILE - 1) / TILE;
    n32 = (N + TILE - 1) / TILE;
    ncr = K / TILE;
    ab = (size_t)m32 * ncr * SLOT;
    bb = (size_t)n32 * ncr * SLOT;
    cb = (size_t)M * N;

    A = malloc((size_t)M * K);
    B = malloc((size_t)N * K);
    C = calloc(cb + 64, 1);
    R = do_ref ? malloc(cb) : NULL;
    Ap = malloc(ab);
    Bp = malloc(bb);
    if (!A || !B || !C || !Ap || !Bp || (do_ref && !R)) {
        printf("host alloc failed\n");
        return 1;
    }

    for (i = 0; i < M * K; i++) A[i] = (uint8_t)(rnd() % (unsigned)(amax + 1));
    for (i = 0; i < N * K; i++) B[i] = (int8_t)((int)(rnd() % 3u) - 1);

    printf("prepack A %.1f MB, B %.1f MB\n", (double)ab / 1e6, (double)bb / 1e6);
    prepack_act(Ap, A, M, K, amax);
    prepack_wgt(Bp, (int8_t *)B, N, K);

    clock_gettime(CLOCK_MONOTONIC, &ts);
    rc = mx4_gemm(Ap, (int)ab, Bp, (int)bb, M, N, K, reps, mode,
                  C, (int)cb, res, 12);
    clock_gettime(CLOCK_MONOTONIC, &te);
    wall0 = (uint64_t)(te.tv_sec - ts.tv_sec) * 1000000u
          + (uint64_t)(te.tv_nsec - ts.tv_nsec) / 1000u;

    printf("gemm %dx%dx%d reps=%d mode=%d: rc=%d status=%d tiles=%d "
           "m32=%d n32=%d ncr=%d vtcm=%d stage_mode=%d\n",
           M, N, K, reps, mode, rc, res[0], res[4], res[5], res[6], res[7],
           res[8], res[11]);
    printf("  act_bytes=%d wgt_bytes=%d\n", res[9], res[10]);
    if (rc != 0 || res[0] != 0) { printf("  FAIL status\n"); return 1; }
    wall1 = wall0;

    macs = (unsigned long long)m32 * TILE * n32 * TILE * (unsigned)K
         * (unsigned)reps;
    gmac_hmx  = (double)macs / ((double)(res[1] ? res[1] : 1) * 1e3);
    gmac_full = (double)macs / ((double)(res[2] ? res[2] : 1) * 1e3);

    printf("  pure HMX  : %d us -> %.1f GMAC/s (%.3f TMAC/s)\n",
           res[1], gmac_hmx, gmac_hmx / 1000.0);
    printf("  full pass : %d us -> %.1f GMAC/s (stream wgt DDR->VTCM + unpack)\n",
           res[2], gmac_full);
    printf("  stage act : %d us, wall (rpc+tests) %llu us\n", res[3],
           (unsigned long long)wall1);

    if (do_ref) {
        printf("  computing cpu ref (%llu MACs)...\n", macs / (unsigned)reps);
        fflush(stdout);
        ref_gemm(A, (int8_t *)B, M, N, K, R);
        cmp(C, R, M, N, verbose ? 8 : 0);
        if (verbose) {
            int s, n;
            printf("  C[0][0..15]:");
            for (n = 0; n < 16; n++) printf(" %u/%u", C[n], R[n]);
            printf("\n");
            (void)s;
        }
    }
    (void)wall1;
    return 0;
}

static double h2d16(int v) { return (double)v; }

static double h2d(unsigned short h)
{
    unsigned sign = h >> 15, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    double v;
    if (exp == 0) v = (double)man * (1.0 / (1u << 24));
    else if (exp == 31) v = man ? 0.0 : 1e30;
    else v = (double)(man | 0x400) * ldexp(1.0, (int)exp - 25);
    return sign ? -v : v;
}

/* deterministic pattern diagnostics: reveal the real output layout */
static int do_diag2(int pattern, int variant, int srt, int ops)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, i, rc, line, cnt, hist[40];

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (pattern == 2) {
                /* A has data only in spatials 16..31 */
                A[s * 32 + n] = (n == 0 && s >= 16) ? (uint8_t)(s + 1) : 0;
                B[s * 32 + n] = (s == n) ? 1 : 0;
            } else if (pattern == 1) {
                /* A all ones, B = single 1 at (n=0,k=0) -> out[s][0] = 1 for all s */
                A[s * 32 + n] = 1;
                B[s * 32 + n] = (s == 0 && n == 0) ? 1 : 0;
            } else {
                /* A[s][k] = s+1 on k=0, B[n][k] = 1 iff k == n -> out[s][n] = (n==0)?s+1:0 */
                A[s * 32 + n] = (n == 0) ? (uint8_t)(s + 1) : 0;
                B[s * 32 + n] = (s == n) ? 1 : 0;
            }
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0xEE, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, srt, variant, ops, raw, 2048, info, 6);
    printf("diag pattern=%d variant=%d srt=0x%x ops=%d: rc=%d status=%d\n",
           pattern, variant, srt, ops, rc, info[0]);
    if (rc || info[0]) return 1;
    memset(hist, 0, sizeof(hist));
    cnt = 0;
    for (i = 0; i < 2048; i++) { if (raw[i] && raw[i] != 0xEE) cnt++; }
    for (i = 0; i < 2048; i++) if (raw[i] < 40) hist[raw[i]]++;
    printf("  nonzero=%d  hist:", cnt);
    for (i = 0; i < 40; i++) if (hist[i]) printf(" %d:%d", i, hist[i]);
    printf("\n");
    for (n = 1; n <= 8; n++) {
        int f = 0;
        printf("  v=%d at:", n);
        for (i = 0; i < 2048 && f < 10; i++) if (raw[i] == n) { printf(" %d", i); f++; }
        printf("\n");
    }
    for (line = 0; line < 2; line++) {
        printf("  [%4d]", line * 128);
        for (i = 0; i < 32; i++) printf(" %02x", raw[line * 128 + i]);
        printf("\n");
    }
    return 0;
}

static int do_diag(int pattern)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, i, rc, found;

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (pattern == 0) { A[s * 32 + n] = 1; B[s * 32 + n] = (s == n); }
            else if (pattern == 1) { A[s * 32 + n] = (n == 0) ? (uint8_t)(s + 1) : 0;
                                     B[s * 32 + n] = (n == 0); }
            else { A[s * 32 + n] = (n == 0);
                   B[s * 32 + n] = (n == 0) ? (uint8_t)(s + 1) : 0; }
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0xEE, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 3, 3, raw, 2048, info, 6);
    printf("diag pattern=%d: rc=%d status=%d\n", pattern, rc, info[0]);
    if (rc || info[0]) return 1;
    printf("  raw[0..95]:");
    for (i = 0; i < 96; i++) printf(" %02x", raw[i]);
    printf("\n");
    for (i = 0; i < 96; i += 2)
        if (raw[i] || raw[i + 1]) { }
    for (n = 1; n <= 6; n++) {
        printf("  value %d at:", n);
        found = 0;
        for (i = 0; i < 2048 && found < 24; i++)
            if (raw[i] == n) { printf(" %d", i); found++; }
        printf("\n");
    }
    return 0;
}

/* int8 GEMM check with the verified subset: 16 spatial rows, uh (16-bit) output,
 * layouts: act (s>>2)*128+c*4+(s&3), wgt (k>>2)*128+n*4+(k&3),
 *          out (s>>1)*128+n*4+(s&1)*2 (16-bit LE) */
/* int8 channel-coverage probes.
 * 10: A[s][k]=k+1, B=all ones        -> out = sum(1..32) = 528 if all act channels read
 * 11: A=all ones, B[n][k]=k+1        -> out = 528 if all weight channels read
 * 12: A[s][k]=k+1, B[n][k]=(k==n)    -> out[s][n] = n+1 (identity mapping check)
 */
static int do_probe_ch(int pat, int ac, int wc)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc, i;
    int hist[300];

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (pat == 30) { A[s * 32 + n] = 1; B[s * 32 + n] = (int8_t)(s + 1); }
            else if (pat == 31) { A[s * 32 + n] = (uint8_t)(s + 1); B[s * 32 + n] = 1; }
            else if (pat == 13) { A[s * 32 + n] = 1; B[s * 32 + n] = 1; }
            else if (pat == 15) { A[s * 32 + n] = (n < 16) ? 1 : 0; B[s * 32 + n] = 1; }
            else if (pat == 16) { A[s * 32 + n] = (n >= 16) ? 1 : 0; B[s * 32 + n] = 1; }
            else if (pat == 17) { A[s * 32 + n] = 1; B[s * 32 + n] = (n < 16) ? 1 : 0; }
            else if (pat == 18) { A[s * 32 + n] = 1; B[s * 32 + n] = (n >= 16) ? 1 : 0; }
            else if (pat == 19) { A[s * 32 + n] = 1; B[s * 32 + n] = (s == n) ? 1 : 0; }
            else if (pat == 10) { A[s * 32 + n] = (uint8_t)(n + 1); B[s * 32 + n] = 1; }
            else if (pat == 11) { A[s * 32 + n] = 1; B[s * 32 + n] = (uint8_t)(n + 1); }
            else { A[s * 32 + n] = (uint8_t)(n + 1); B[s * 32 + n] = (n == s) ? 1 : 0; }
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, ac, wc, 0, 6, 3, raw, 2048, info, 6);
    printf("probe_ch pat=%d ac=0x%x wc=%d: rc=%d status=%d\n", pat, ac, wc, rc, info[0]);
    if (rc || info[0]) return 1;

    memset(hist, 0, sizeof(hist));
    for (i = 0; i < 2048; i += 2) {
        int v = raw[i] | (raw[i + 1] << 8);
        if (v < 300) hist[v]++;
    }
    printf("  word hist:");
    for (i = 0; i < 300; i++) if (hist[i]) printf(" %d:%d", i, hist[i]);
    printf("\n");
    if (pat == 30 || pat == 31) {
        int s2;
        for (s2 = 0; s2 < 3; s2++) {
            printf("  out row s=%d:", s2);
            for (i = 0; i < 32; i++) {
                unsigned off = (unsigned)(s2 >> 1) * 128u + (unsigned)i * 4u
                             + (unsigned)(s2 & 1) * 2u;
                printf(" %d", raw[off] | (raw[off + 1] << 8));
            }
            printf("\n");
        }
        return 0;
    }
    if (pat == 12) {
        /* where does each value live? use first 8 values */
        for (n = 1; n <= 4; n++) {
            int f = 0;
            printf("  v=%d at:", n);
            for (i = 0; i < 2048 && f < 12; i += 2)
                if ((raw[i] | (raw[i + 1] << 8)) == n) { printf(" %d", i); f++; }
            printf("\n");
        }
    }
    return 0;
}

/* S1: full 32x32x32 int8 GEMM check with the 2-sub-crouton (16 ch/1 KB) model.
 * act   : (s>>1)*128 + c*4 + (s&1)*2      (2 B cells, 1 KB per 16 channels)
 * weight: (k>>1)*128 + n*4 + (k&1)*2
 * out   : (s>>1)*128 + n*4 + (s&1)*2      16-bit
 */
static int do_s1(int mode, int ac, int wc, int st)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0;
    int32_t refv, acc;

    g_act_layout = 4;
    g_wgt_layout = 1;

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (mode == 0) {
                A[s * 32 + n] = (uint8_t)(n + 1);             /* k+1 */
                B[s * 32 + n] = (s == n) ? 1 : 0;             /* identity */
            } else if (mode == 2) {
                A[s * 32 + n] = (uint8_t)((s + n) & 0x3);     /* A[s][k]=s+k */
                B[s * 32 + n] = (int8_t)((s - n) & 0x3);      /* B[n][k]=n-k */
                if (B[s * 32 + n] > 1) B[s * 32 + n] = (int8_t)(B[s * 32 + n] - 4);
            } else {
                A[s * 32 + n] = (uint8_t)(1 + rnd() % 4);
                B[s * 32 + n] = (int8_t)((int)(rnd() % 3) - 1);
            }
        }

    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, ac, wc, 0, 6, 35, raw, 2048, info, 6);
    printf("S1 mode=%d ac=0x%x wc=%d: rc=%d status=%d\n", mode, ac, wc, rc, info[0]);
    if (rc || info[0]) return 1;

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            unsigned off = (unsigned)(s >> 1) * 128u + (unsigned)n * 4u
                         + (unsigned)(s & 1) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            acc = 0;
            for (k = 0; k < 32; k++)
                acc += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            refv = acc < 0 ? 0 : (acc > 65535 ? 65535 : acc);
            {
                long long e = (long long)hw - refv;
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = e;
                if (e) {
                    if (bad < 6)
                        printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                    bad++;
                }
            }
        }
    printf("  S1 max_abs_err=%lld  mismatches=%lld / 1024\n", maxerr, bad);
    return maxerr ? 1 : 0;
}

/* 16-channel S1 check: A[s][k]=(k<16)?k+1:0, B=I -> out[s][n]=(n<16)?n+1:0 */
/* signature of the set of input channels the HW reads:
 *  who=0: weight side (A=1, B[n][k]=(k>>j)&1)
 *  who=1: activation side (A[s][k]=(k>>j)&1, B=1)
 * out = number of read channels with bit j set; expect 8,4,2,1,0 for k=0..15 */
static int do_bits(int who, int alay, int wlay)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int j, s, n, rc;

    g_act_layout = alay; g_wgt_layout = wlay;
    printf("bits who=%d alay=%d wlay=%d -> ", who, alay, wlay);
    for (j = 0; j < 5; j++) {
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                if (who == 0) { A[s * 32 + n] = 1;  B[s * 32 + n] = (int8_t)((n >> j) & 1); }
                else          { A[s * 32 + n] = (uint8_t)((n >> j) & 1); B[s * 32 + n] = 1; }
            }
        Ap = malloc(SLOT); Bp = malloc(SLOT);
        prepack_act(Ap, A, 32, 32, 255);
        prepack_wgt(Bp, (int8_t *)B, 32, 32);
        memset(raw, 0, sizeof(raw));
        rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
        {
            unsigned off = 0;   /* s=0, n=0 */
            printf("b%d=%d ", j, raw[off] | (raw[off + 1] << 8));
        }
        if (rc || info[0]) printf("[rc=%d st=%d] ", rc, info[0]);
        free(Ap); free(Bp);
        usleep(300000);
    }
    printf("\n");
    return 0;
}

/* scan: which packed input channels does the HW actually read?
 * who=0 -> weight side: B[n][k] = (k==c), A=1  => out[0][0] = 1 if c is read
 * who=1 -> act side   : A[s][k] = (k==c), B=1  => out[0][0] = 1 if c is read
 * prints a 32-bit mask over c and the count.                                   */
/* fingerprint: A=1, B[n][k]=k+1  =>  out[s][0] = sum of the values the HW reads.
 * candidates: 136 (k=0..15), 200 (k=0..7 + 16..23), 392 (k=16..31),
 *             528 (all 32), 265 (layout-0 measured earlier)                     */
static int do_sumtest(int wlay)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, rc;

    g_act_layout = 0; g_wgt_layout = wlay;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = 1;
            B[s * 32 + n] = (int8_t)(n + 1);
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("sumtest wlay=%3d rc=%d st=%d  out[0][0]=%d  out[0][1]=%d  out[1][0]=%d\n",
           wlay, rc, info[0],
           raw[4] | (raw[5] << 8),
           raw[12] | (raw[13] << 8),
           raw[4 + 128] | (raw[5 + 128] << 8));
    free(Ap); free(Bp);
    return 0;
}

static int do_scan(int who, int alay, int wlay)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int c, s, n, rc, cnt = 0;
    unsigned mask = 0;

    g_act_layout = alay; g_wgt_layout = wlay;
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    printf("scan who=%d alay=%d wlay=%d mask: ", who, alay, wlay);
    for (c = 0; c < 32; c++) {
        int hit;
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                if (who == 0) { A[s * 32 + n] = 1; B[s * 32 + n] = (n == c) ? 1 : 0; }
                else          { A[s * 32 + n] = (n == c) ? 1 : 0; B[s * 32 + n] = 1; }
            }
        prepack_act(Ap, A, 32, 32, 255);
        prepack_wgt(Bp, (int8_t *)B, 32, 32);
        memset(raw, 0, sizeof(raw));
        rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
        /* cell (s=0,n=0) is at byte 4, not 0 (see measured output layout) */
        hit = (raw[4] | (raw[5] << 8)) != 0;
        if (rc || info[0]) { printf("[rc=%d st=%d] ", rc, info[0]); }
        printf("%c", hit ? '1' : '0');
        if (hit) { mask |= 1u << c; cnt++; }
        usleep(150000);
    }
    printf("  count=%d mask=0x%08x\n", cnt, mask);
    free(Ap); free(Bp);
    return 0;
}

/* 16-channel S1 with the measured layouts:
 *  act: (c>>4)*1024 + (s>>2)*128 + (c&15)*8 + (s&3)*2 + 1
 *  wgt: (n>>4)*1024 + (k>>2)*128 + (n&15)*8 + (k&3)*2 + 1
 *  out: 4 + (n>>4)*1024 + (s>>2)*128 + (n&15)*8 + (s&3)*2     (16-bit)
 */
static int do_s1h2(int alay, int wlay, int nch, int quiet)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0, nz = 0;

    g_act_layout = alay; g_wgt_layout = wlay;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (n < nch) ? (uint8_t)(n + 1) : 0;
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("s1h2 alay=%d wlay=%d nch=%d: rc=%d status=%d\n", alay, wlay, nch, rc, info[0]);
    if (rc || info[0]) return 1;

    for (s = 0; s < 16; s++)
        for (n = 0; n < nch; n++) {
            unsigned off = 2u + (unsigned)(n >> 4) * 1024u
                         + (unsigned)(s >> 1) * 128u + (unsigned)(n & 15) * 8u
                         + (unsigned)(s & 1) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            int32_t refv = 0;
            for (k = 0; k < nch; k++)
                refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            if (hw) nz++;
            if (hw != refv) {
                if (!quiet && bad < 6)
                    printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                { long long e = hw - refv; if (e < 0) e = -e; if (e > maxerr) maxerr = e; }
                bad++;
            }
        }
    printf("  max_abs_err=%lld mismatches=%lld/%d nonzero=%lld (rows 0..15)\n",
           maxerr, bad, 16 * nch, nz);
    return maxerr ? 1 : 0;
}

/* random S1 on the reachable subset: K = nch input channels, N = nch output
 * channels, all 32 spatial rows.  out byte = 4 + (s>>2)*128 + n*8 + (s&3)*2 */
static int do_s1r2(int nch, int wlay)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0;

    g_act_layout = 0; g_wgt_layout = wlay;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (n < nch) ? (uint8_t)(1 + rnd() % 4) : 0;
            B[s * 32 + n] = (s < nch) ? (int8_t)((int)(rnd() % 3) - 1) : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("s1r random K=%d N=%d wlay=%d: rc=%d status=%d\n", nch, nch, wlay, rc, info[0]);
    if (rc || info[0]) return 1;
    for (s = 0; s < 32; s++)
        for (n = 0; n < nch; n++) {
            unsigned off = 4u + (unsigned)(s >> 2) * 128u + (unsigned)n * 8u
                         + (unsigned)(s & 3) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            int32_t refv = 0;
            for (k = 0; k < nch; k++)
                refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            if (refv < 0) refv = 0;
            if (refv > 65535) refv = 65535;
            if (hw != refv) {
                if (bad < 6)
                    printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                { long long e = hw - refv; if (e < 0) e = -e; if (e > maxerr) maxerr = e; }
                bad++;
            }
        }
    printf("  S1(%d x %d x 32) wlay=%d max_abs_err=%lld mismatches=%lld/%d\n",
           nch, nch, wlay, maxerr, bad, 32 * nch);
    return maxerr ? 1 : 0;
}

/* full 32x32x32 int8 S1 with layouts measured by the byte-position scan:
 *   act (70): 128*(s>>2) + 8*c + 2*(s&3)          (both bytes = value)
 *   wgt (71): 128*(k>>1) + 8*(n&15) + 4 + 2*(k&1) (both bytes = value)
 *   out     : 1024*(n>>4) + 128*(s>>2) + 8*(n&15) + 4 + 2*(s&3)
 * 16 MAC per multiply -> two multiply pairs (+1024) cover k / n 0..31.        */
/* output-channel map with the measured weight layout: B[n][k]=n+1, A=1 */
static int do_wmap2(void)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, q, rc;

    g_act_layout = 0; g_wgt_layout = 71;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) { A[s * 32 + n] = 1; B[s * 32 + n] = (int8_t)(s + 1); }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("wmap2 rc=%d st=%d (expect 16*(n+1) at the cell of channel n)\n", rc, info[0]);
    for (q = 0; q < 2; q++) {
        printf("  words[%3d..]:", q * 32);
        for (n = 0; n < 32; n++) {
            int o = (q * 32 + n) * 2;
            printf(" %d", raw[o] | (raw[o + 1] << 8));
        }
        printf("\n");
    }
    return 0;
}

/* Full maps in two runs:
 *  run A: act[s][k] = s+1 (all k), wgt = 1   -> each output cell = 16*(row read)
 *  run B: act = 1, wgt[n][k] = n+1 (all k)   -> each output cell = 16*(channel read)
 * Printing (value/16 - 1) per 16-bit cell reveals both maps without assumptions. */
/* k map: B[n][k] = 1 only at k=k0 (all n), A=1 -> every written cell must be 1.
 * A misplaced k cell shows up as "this k contributes nothing" (0 cells). */
/* identity test: A[s][k]=k+1, B=I  ->  out[s][n] = n+1 (all 1024 cells).
 * Sweeps the activation offset (delta) to find the exact address. */
/* activation channel map: A[s][k]=k+1, B[n][k]=(k==n) -> out[s][n] = pi(n)+1 */
static int do_actch(int lay, int cshift, int delta)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, rc;

    g_act_layout = lay; g_wgt_layout = 84; g_act_cshift = cshift; g_act_delta = delta;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (uint8_t)(n + 1);
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("actch lay=%d cshift=%d delta=%d rc=%d st=%d\n  s=0 pi(n)+1:", lay, cshift, delta, rc, info[0]);
    for (n = 0; n < 32; n++) {
        unsigned off = 2u + (unsigned)(n >> 4) * 1024u + (unsigned)(n & 15) * 8u;
        printf(" %d", raw[off] | (raw[off + 1] << 8));
    }
    printf("\n  s=1 pi(n)+1:");
    for (n = 0; n < 32; n++) {
        unsigned off = 2u + (unsigned)(n >> 4) * 1024u + 128u + (unsigned)(n & 15) * 8u + 2u;
        printf(" %d", raw[off] | (raw[off + 1] << 8));
    }
    printf("\n");
    free(Ap); free(Bp);
    return 0;
}

/* ONE-SHOT activation map: A[s][c] = s*16+c (unique), B = identity.
 * out(s,n) = the activation value the HW read for (s,n); since values are unique
 * we recover the exact VTCM byte the HW read for every (s,c). */
static int do_amap(void)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, c, rc, v, sr, cr, off;

    g_act_layout = 90; g_act_delta = 0; g_act_cshift = 0; g_wgt_layout = 84;
    for (s = 0; s < 32; s++)
        for (c = 0; c < 32; c++) {
            A[s * 32 + c] = (s < 16 && c < 16) ? (uint8_t)(s * 16 + c) : 0;
            B[s * 32 + c] = (s == c) ? 1 : 0;      /* B[out=s][k=c] identity */
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("amap rc=%d st=%d; entries = VTCM byte the HW read for (s,c)\n", rc, info[0]);
    for (s = 0; s < 32; s++) {
        printf("  s=%2d:", s);
        for (n = 0; n < 16; n++) {
            unsigned o = 2u + (unsigned)(n >> 4) * 1024u
                       + (unsigned)(s >> 1) * 128u + (unsigned)(n & 15) * 8u
                       + (unsigned)(s & 1) * 2u;
            v = raw[o] | (raw[o + 1] << 8);
            sr = v >> 4; cr = v & 15;
            if (v == 0) printf("  .");
            else printf(" %d:%d", sr, cr);
        }
        printf("\n");
    }
    free(Ap); free(Bp);
    return 0;
}

static int do_asweep(void)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, k, rc, d, bad;
    static const int deltas[] = { 0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 24, 32 };
    static const int shifts[] = { -2, -1, 0, 1, 2, 3, 4 };

    Ap = malloc(SLOT); Bp = malloc(SLOT);
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (uint8_t)(n + 1);
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    g_act_layout = 90; g_wgt_layout = 84;
    (void)deltas; (void)shifts;
    for (d = 0; d < 12; d++) {
        static const int cs[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        static const int dl[] = { 56, 58, 60, 62, 64, 66, 68, 72, 60+128, 64+128, 60+256, 64+256 };
        g_act_layout = 90;
        g_act_cshift = cs[d]; g_act_delta = dl[d];
        g_slot0 = g_slotstep = 0;
        prepack_act(Ap, A, 32, 32, 255);
        prepack_wgt(Bp, (int8_t *)B, 32, 32);
        memset(raw, 0, sizeof(raw));
        rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
        bad = 0;
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                unsigned off = 2u + (unsigned)(n >> 4) * 1024u
                             + (unsigned)(s >> 1) * 128u + (unsigned)(n & 15) * 8u
                             + (unsigned)(s & 1) * 2u;
                int hw = raw[off] | (raw[off + 1] << 8);
                if (hw != n + 1) bad++;
            }
        printf("  cshift=%3d delta=%4d: rc=%d bad=%4d/1024  w0=%2d,%2d,%2d,%2d\n",
               g_act_cshift, g_act_delta, rc, bad,
               raw[2] | (raw[3] << 8), raw[10] | (raw[11] << 8),
               raw[18] | (raw[19] << 8), raw[26] | (raw[27] << 8));
        usleep(150000);
    }
    free(Ap); free(Bp);
    return 0;
}

static int do_kmap(int alay, int wlay)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, k, k0, rc, ones, firstword;

    g_act_layout = alay; g_wgt_layout = wlay;
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    printf("kmap alay=%d wlay=%d\n", alay, wlay);
    for (k0 = 0; k0 < 32; k0++) {
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                A[s * 32 + n] = 1;
                B[s * 32 + n] = (n == k0) ? 1 : 0;   /* B[out=s][k=n] */
            }
        prepack_act(Ap, A, 32, 32, 255);
        prepack_wgt(Bp, (int8_t *)B, 32, 32);
        memset(raw, 0, sizeof(raw));
        rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
        ones = 0; firstword = -1;
        for (s = 0; s < 1024; s++) {
            int v = raw[s * 2] | (raw[s * 2 + 1] << 8);
            if (v) { ones++; if (firstword < 0) firstword = s; }
        }
        printf("  k=%2d: rc=%d cells=%3d firstword=%d\n", k0, rc, ones, firstword);
        usleep(100000);
    }
    free(Ap); free(Bp);
    return 0;
}

static int do_maps(int which)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, i, rc;

    if (!g_act_layout) g_act_layout = 77;
    if (!g_wgt_layout) g_wgt_layout = 73;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (which == 0) { A[s * 32 + n] = (uint8_t)(s + 1); B[s * 32 + n] = 1; }
            else            { A[s * 32 + n] = 1; B[s * 32 + n] = (int8_t)(s + 1); }
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("maps which=%d (%s) rc=%d st=%d\n", which,
           which ? "channel map" : "spatial map", rc, info[0]);
    if (rc || info[0]) return 1;
    for (i = 0; i < 4; i++) {
        printf("  w%4d raw:", i * 32);
        for (n = 0; n < 32; n++) {
            int v = raw[(i * 32 + n) * 2] | (raw[(i * 32 + n) * 2 + 1] << 8);
            printf(" %3d", v);
        }
        printf("\n");
    }
    return 0;
}

static int do_srtest(int srt)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6], s, n, k, rc, rows = 0;

    g_act_layout = 77; g_wgt_layout = 73;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (uint8_t)(1 + (s % 4));   /* distinct per spatial */
            B[s * 32 + n] = (int8_t)((s == 0) ? 1 : 0); /* only n=0 weights */
        }
    for (s = 0; s < 32; s++) for (n = 1; n < 32; n++) B[s * 32 + n] = 0;
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, srt, 6, 3 | 32, raw, 2048, info, 6);
    printf("srtest srt=0x%03x rc=%d st=%d rows(nonzero out[s][0]):", srt, rc, info[0]);
    for (s = 0; s < 32; s++) {
        unsigned off = 4u + (unsigned)(s >> 2) * 128u + (unsigned)(s & 3) * 2u;
        int v = raw[off] | (raw[off + 1] << 8);
        if (v) { printf(" %d=%d", s, v); rows++; }
    }
    printf("  (count=%d)\n", rows);
    free(Ap); free(Bp);
    return 0;
}

static int do_s1full(int mode)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0;

    if (!g_act_layout) g_act_layout = 76;
    if (!g_wgt_layout) g_wgt_layout = 73;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            if (mode == 0) {                      /* deterministic: A=i+k, B=k-j */
                A[s * 32 + n] = (uint8_t)((s + n) & 0x3);
                B[s * 32 + n] = (int8_t)(((n - s) & 0x3) - 2);
            } else if (mode == 1) {               /* identity check */
                A[s * 32 + n] = (uint8_t)(n + 1);
                B[s * 32 + n] = (s == n) ? 1 : 0;
            } else {                              /* random */
                A[s * 32 + n] = (uint8_t)(1 + rnd() % 4);
                B[s * 32 + n] = (int8_t)((int)(rnd() % 3) - 1);
            }
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3 | 32, raw, 2048, info, 6);
    printf("S1full mode=%d: rc=%d status=%d\n", mode, rc, info[0]);
    if (rc || info[0]) return 1;
    if (mode == 2) {
        int q;
        for (q = 0; q < 8; q++) {
            unsigned off = 4u + (unsigned)(q >> 2) * 128u + (unsigned)(q & 3) * 2u;
            int32_t rv = 0; int kk;
            for (kk = 0; kk < 32; kk++) rv += (int32_t)A[q * 32 + kk] * (int32_t)(int8_t)B[kk];
            printf("  s=%d n=0: hw=%d ref=%d\n", q, raw[off] | (raw[off + 1] << 8), rv);
        }
    }
    if (mode == 1) {
        int q;
        printf("  raw words[0..15]:");
        for (q = 0; q < 16; q++) printf(" %d", raw[q * 2] | (raw[q * 2 + 1] << 8));
        printf("\n  ref rows: 1,1,1,1 / 2,2,2,2 / 3,3,3,3 ...\n");
    }
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            unsigned off = 2u + (unsigned)(n >> 4) * 1024u
                         + (unsigned)(s >> 1) * 128u + (unsigned)(n & 15) * 8u
                         + (unsigned)(s & 1) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            int32_t refv = 0;
            for (k = 0; k < 32; k++)
                refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            if (refv < 0) refv = 0;
            if (refv > 65535) refv = 65535;
            if (hw != refv) {
                if (bad < 6) printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                { long long e = hw - refv; if (e < 0) e = -e; if (e > maxerr) maxerr = e; }
                bad++;
            }
        }
    {
        long long e0 = 0, b0 = 0, e1 = 0, b1 = 0;
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                unsigned off = 2u + (unsigned)(n >> 4) * 1024u
                             + (unsigned)(s >> 1) * 128u + (unsigned)(n & 15) * 8u
                             + (unsigned)(s & 1) * 2u;
                int hw = raw[off] | (raw[off + 1] << 8);
                int32_t refv = 0;
                long long e;
                for (k = 0; k < 32; k++)
                    refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
                if (refv < 0) refv = 0;
                e = (long long)hw - refv; if (e < 0) e = -e;
                if (n < 16) { if (e) { b0++; if (e > e0) e0 = e; } }
                else        { if (e) { b1++; if (e > e1) e1 = e; } }
            }
        {
            long long e2 = 0, b2 = 0;
            for (s = 0; s < 16; s++)
                for (n = 0; n < 16; n++) {
                    unsigned off = 2u + (unsigned)(s >> 1) * 128u
                                 + (unsigned)(n & 15) * 8u + (unsigned)(s & 1) * 2u;
                    int hw = raw[off] | (raw[off + 1] << 8);
                    int32_t refv = 0; long long e;
                    for (k = 0; k < 32; k++)
                        refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
                    if (refv < 0) refv = 0;
                    e = (long long)hw - refv; if (e < 0) e = -e;
                    if (e) { if (b2 < 12) printf("    fail s=%d n=%d hw=%d ref=%d\n", s, n, hw, refv); b2++; if (e > e2) e2 = e; }
                    (void)refv;
                }
            printf("  S1 verified 16x16x32 (s<16,n<16): max_abs_err=%lld mismatches=%lld/256\n",
                   e2, b2);
        }
        printf("  half n=0..15 : max_abs_err=%lld mismatches=%lld/512\n", e0, b0);
        printf("  half n=16..31: max_abs_err=%lld mismatches=%lld/512\n", e1, b1);
    }
    printf("  S1 max_abs_err=%lld mismatches=%lld/1024\n", maxerr, bad);
    return maxerr ? 1 : 0;
}

static int do_dump(int alay, int wlay, int nch)
{
    uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, rc, i;

    g_act_layout = alay; g_wgt_layout = wlay;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (n < nch) ? (uint8_t)(n + 1) : 0;
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("dump alay=%d wlay=%d nch=%d rc=%d st=%d; expect words: 1,1,1,1,2,2,2,2,...,%d\n",
           alay, wlay, nch, rc, info[0], nch);
    for (i = 0; i < 1; i++) {
        printf("  [%3d]:", i * 128);
        for (n = 0; n < 64; n++) {
            unsigned o = (unsigned)i * 128u + (unsigned)n * 2u;
            printf(" %3d", raw[o] | (raw[o + 1] << 8));
        }
        printf("\n");
    }
    /* stats: which word offsets hold value 1 */
    {
        int v, f;
        for (v = 1; v <= 16; v++) {
            f = 0;
            printf("  v=%2d at words:", v);
            for (i = 0; i < 1024 && f < 8; i++)
                if ((raw[i * 2] | (raw[i * 2 + 1] << 8)) == v) { printf(" %d", i); f++; }
            printf("\n");
        }
    }
    return 0;
}

static int do_grid(int alay, int wlay)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;

    g_act_layout = alay; g_wgt_layout = wlay;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (n < 16) ? (uint8_t)(n + 1) : 0;
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("grid alay=%d wlay=%d: rc=%d status=%d (ref: n<16 -> n+1, all s)\n",
           alay, wlay, rc, info[0]);
    if (rc || info[0]) return 1;
    for (s = 0; s < 6; s++) {
        printf("  s=%2d:", s);
        for (n = 0; n < 16; n++) {
            unsigned off = (unsigned)(s >> 1) * 128u + (unsigned)n * 4u
                         + (unsigned)(s & 1) * 2u;
            printf(" %2d", raw[off] | (raw[off + 1] << 8));
        }
        printf("\n");
    }
    return 0;
}

static int do_s1h(int alay, int wlay, int ac, int wc, int quiet)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0, nz = 0;

    g_act_layout = alay;
    g_wgt_layout = wlay;

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            A[s * 32 + n] = (n < 16) ? (uint8_t)(n + 1) : 0;
            B[s * 32 + n] = (s == n) ? 1 : 0;
        }
    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, ac, wc, 0, 6, 3, raw, 2048, info, 6);
    if (rc || info[0]) { printf("  alay=%d rc=%d status=%d\n", alay, rc, info[0]); return 1; }

    for (s = 0; s < 32; s++)
        for (n = 0; n < 16; n++) {
            unsigned off = (unsigned)(s >> 1) * 128u + (unsigned)n * 4u
                         + (unsigned)(s & 1) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            int32_t refv = 0;
            for (k = 0; k < 16; k++)
                refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            if (hw) nz++;
            if (hw != refv) {
                if (!quiet && bad < 3)
                    printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                { long long e = hw - refv; if (e < 0) e = -e; if (e > maxerr) maxerr = e; }
                bad++;
            }
        }
    printf("  alay=%3d wlay=%d ac=0x%x: max_abs_err=%3lld mismatches=%3lld/512 nonzero=%lld\n",
           alay, wlay, ac, maxerr, bad, nz);
    (void)s; (void)n; (void)nz;
    return maxerr ? 1 : 0;
}

static int do_check16(void)
{
    uint8_t A[32 * 32], B[32 * 32], *Ap, *Bp, raw[2048];
    int info[6];
    int s, n, k, rc;
    long long maxerr = 0, bad = 0;
    int32_t refv;

    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            /* B[n][k]: n = output channel row, k = input channel */
        }
    for (s = 0; s < 32; s++)
        for (k = 0; k < 32; k++)
            A[s * 32 + k] = (s < 16) ? (uint8_t)(1 + (rnd() % 4)) : 0;
    for (n = 0; n < 32; n++)
        for (k = 0; k < 32; k++)
            B[n * 32 + k] = (int8_t)((int)(rnd() % 3) - 1);

    Ap = malloc(SLOT); Bp = malloc(SLOT);
    prepack_act(Ap, A, 32, 32, 255);
    prepack_wgt(Bp, (int8_t *)B, 32, 32);
    memset(raw, 0, sizeof(raw));
    rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
    printf("check16 (32x32x32, rows 0..15): rc=%d status=%d\n", rc, info[0]);
    if (rc || info[0]) return 1;

    for (s = 0; s < 16; s++)
        for (n = 0; n < 32; n++) {
            unsigned off = (unsigned)(s >> 1) * 128u + (unsigned)n * 4u
                         + (unsigned)(s & 1) * 2u;
            int hw = raw[off] | (raw[off + 1] << 8);
            refv = 0;
            for (k = 0; k < 32; k++)
                refv += (int32_t)A[s * 32 + k] * (int32_t)(int8_t)B[n * 32 + k];
            if (refv < 0) refv = 0;
            if (refv > 65535) refv = 65535;
            {
                long long e = (long long)hw - refv;
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = e;
                if (e) {
                    if (bad < 6)
                        printf("    out[%d][%d] hw=%d ref=%d\n", s, n, hw, refv);
                    bad++;
                }
            }
        }
    printf("  max_abs_err=%lld mismatches=%lld / %d\n", maxerr, bad, 16 * 32);
    return maxerr ? 1 : 0;
}

/* validate the PRM 3.2 example: out[s][n] = (s + 0.01n) * n, 1024 cells */
static int do_cov(int off, int step, int nst, int rt, int form)
{
    unsigned char raw[2048];
    int r[6], rc;
    rc = mx4_cov(off, step, nst, rt, form, raw, 2048, r, 6);
    printf("cov off=%d step=%d nst=%d rt=0x%x form=%d: rc=%d st=%d filled=%d/1024\n",
           off, step, nst, rt, form, rc, r[0], r[1]);
    return 0;
}

static int do_fp16ex(int mask)
{
    uint8_t raw[2048];
    int info[6];
    int s, n, rc, i, j;
    static double want[1024], got[1024];
    double maxerr = 0;
    int bad = 0;

    rc = mx4_fp16ex(mask, raw, 2048, info, 6);
    printf("fp16ex mask=0b%05b: rc=%d status=%d act_rt=0x%x wgt_rt=%d wr_rt=0x%x\n",
           mask, rc, info[0], info[1], info[2], info[3]);
    if (rc || info[0]) return 1;

    i = 0;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            double a = (double)s + 0.01 * (double)n;
            want[i++] = a * (double)n;
        }
    for (i = 0; i < 1024; i++) got[i] = h2d((unsigned short)(raw[i * 2] | (raw[i * 2 + 1] << 8)));
    /* multiset comparison: sort copies */
    {
        static double w2[1024], g2[1024];
        memcpy(w2, want, sizeof want); memcpy(g2, got, sizeof got);
        for (i = 0; i < 1024; i++)
            for (j = i + 1; j < 1024; j++) {
                if (w2[j] < w2[i]) { double t = w2[i]; w2[i] = w2[j]; w2[j] = t; }
                if (g2[j] < g2[i]) { double t = g2[i]; g2[i] = g2[j]; g2[j] = t; }
            }
        for (i = 0; i < 1024; i++) {
            double e = g2[i] - w2[i]; if (e < 0) e = -e;
            if (e > maxerr) maxerr = e;
            if (e > 1e-2) bad++;
        }
    }
    printf("  multiset: max_abs_err=%.5f  cells_out_of_tol=%d/1024\n", maxerr, bad);
    {
        /* by-position check with the doc's layout: 2 spatials per 128 B line */
        double mx = 0; int bd = 0;
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                unsigned widx = (unsigned)(s >> 1) * 64u + (unsigned)n * 2u + (unsigned)(s & 1);
                double a = (double)s + 0.01 * (double)n;
                double wv = a * (double)n;
                double gv = got[widx];
                double e = gv - wv; if (e < 0) e = -e;
                if (e > mx) mx = e;
                if (e > 1e-2) bd++;
            }
        printf("  by-position (2 sp/line): max_abs_err=%.5f bad=%d/1024\n", mx, bd);
    }
    printf("  want[0..7] = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
           want[0], want[1], want[2], want[3], want[4], want[5], want[6], want[7]);
    printf("  got [0..7] = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
           got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7]);
    /* where does want[s=1][n=0] (==0) ... use unique-ish values: report positions of
     * out[16][16]=272.16, out[8][8]=68.64, out[1][31]=32.11 */
    {
        static const double probe[3] = { 272.16, 68.64, 32.11 };
        for (i = 0; i < 3; i++) {
            int f = 0;
            printf("  value %.2f at words:", probe[i]);
            for (j = 0; j < 1024 && f < 4; j++)
                if (fabs(got[j] - probe[i]) < 0.02) { printf(" %d", j); f++; }
            printf("\n");
        }
    }
    return (maxerr <= 1e-2) ? 0 : 1;
}

static int do_fp16(void)
{
    uint8_t raw[2048], unused[4] = {0};
    int info[6];
    int rc, s, n, bad = 0;
    unsigned short h;

    rc = mx4_fp16(unused, 0, raw, 2048, info, 6);
    printf("fp16 32x32x32: rc=%d status=%d vtcm=%d base=%p\n", rc, info[0],
           info[5], (void *)(uintptr_t)(unsigned)info[3]);
    if (rc || info[0]) return 1;
    for (s = 0; s < 32; s++)
        for (n = 0; n < 32; n++) {
            unsigned off = (unsigned)(s >> 1) * 128u + (unsigned)n * 4u
                         + (unsigned)(s & 1) * 2u;
            h = (unsigned short)(raw[off] | (raw[off + 1] << 8));
            {
                double v = h2d(h);
                double e = (n == 0) ? (double)(s + 1) : 0.0;
                if (fabs(v - e) > 1e-3) {
                    if (bad < 6)
                        printf("    out[%d][%d]=%f (h=0x%04x) expected %f\n",
                               s, n, v, h, e);
                    bad++;
                }
            }
        }
    printf("  fp16 mismatches=%d / 1024\n", bad);
    {
        static const float want[] = { 1.0f, 2.0f, 3.0f, 5.0f, 9.0f, 17.0f, 33.0f };
        int w, i2;
        for (w = 0; w < 7; w++) {
            unsigned short enc = 0;
            float f = want[w];
            /* encode via ldexpf-free path: use host conversion */
            { unsigned int bits; float v = f; unsigned exp = 0; double m;
              (void)bits;
              if (v < 1.0f) { exp = 0; m = v * 16777216.0; enc = (unsigned short)m; }
              else { exp = 15; m = v; while (m >= 2.0) { m /= 2.0; exp++; }
                     enc = (unsigned short)((exp << 10) | (((unsigned)((m - 1.0) * 1024.0 + 0.5)) & 0x3ff)); } }
            printf("    %.1f (0x%04x) at:", (double)f, enc);
            for (i2 = 0; i2 < 2048; i2 += 2) {
                unsigned short hh = (unsigned short)(raw[i2] | (raw[i2 + 1] << 8));
                if (hh == enc) printf(" %d", i2);
            }
            printf("\n");
        }
    }
    return bad ? 1 : 0;
}

/* ---- qhl_hmx SDK reference path ----------------------------------- */

static double g_href_err, g_href_cos;
static int    g_href_rc, g_href_st;

static int do_href_one(int N, int K, int mode, int reps, int verbose)
{
    static __fp16 *A, *B, *C;
    int info[16];
    int rc, s, k, n;
    double mx = 0, dot = 0, na = 0, nb = 0, a, b;
    int bad = 0;

    if (!A) {
        A = malloc(32 * 256 * 2);
        B = malloc(256 * 64 * 2);
        C = malloc(32 * 64 * 2);
        if (!A || !B || !C) { printf("alloc failed\n"); return 1; }
        memset(A, 0, 32 * 256 * 2);
        memset(B, 0, 256 * 64 * 2);
        memset(C, 0, 32 * 64 * 2);
    }

    for (s = 0; s < 32; s++)
        for (k = 0; k < K; k++)
            A[s * K + k] = (__fp16)(((int)(rnd() % 2001) - 1000) / 1000.0f);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++)
            B[k * N + n] = (__fp16)(((int)(rnd() % 2001) - 1000) / 1000.0f);

    rc = mx4_hmxref((unsigned char *)A, 32 * K * 2,
                    (unsigned char *)B, K * N * 2,
                    N, K, mode, reps,
                    (unsigned char *)C, 32 * N * 2, info, 16);

    g_href_rc = rc;
    g_href_st = info[0];
    if (rc || info[0]) {
        printf("  K=%3d N=%2d rc=%d st=%d (power=%d lock=%d hvx=%d pack=%d/%d)\n",
               K, N, rc, info[0], info[6], info[2], info[8], info[9], info[12]);
        g_href_err = 9e9; g_href_cos = 0;
        return 1;
    }
    for (s = 0; s < 32; s++)
        for (n = 0; n < N; n++) {
            double ref = 0, got, e;
            for (k = 0; k < K; k++)
                ref += (double)A[s * K + k] * (double)B[k * N + n];
            got = (double)C[s * N + n];
            e = got - ref; if (e < 0) e = -e;
            if (e > mx) mx = e;
            if (e > 1e-2 + 1e-2 * ((ref < 0) ? -ref : ref)) bad++;
            dot += got * ref; na += got * got; nb += ref * ref;
        }
    a = (na > 0) ? na : 1; b = (nb > 0) ? nb : 1;
    g_href_err = mx;
    g_href_cos = (na > 0 && nb > 0) ? dot / sqrt(a * b) : 0;
    printf("  K=%3d N=%2d err=%.5f cos=%.6f bad=%d/%d us=%.1f "
           "act=0x%05x wgt=0x%05x ncr=%d npad=%2d pwr=%d lock=%d hvx=%d\n",
           K, N, mx, g_href_cos, bad, 32 * N, (double)info[1],
           (unsigned)info[3], (unsigned)info[4], info[5], info[11],
           info[6], info[2], info[8]);
    if (verbose) {
        printf("    info: pack=%d/%d vtcm=%d mpy=%d unlock=%d pcyc=%d clk=%d\n",
               info[9], info[12], info[10], info[7], info[13], info[14], info[15]);
        printf("    C[0][0..3]=%.4f %.4f %.4f %.4f  C[31][0..3]=%.4f %.4f %.4f %.4f\n",
               (double)C[0], (double)C[1], (double)C[2], (double)C[3],
               (double)C[31 * N], (double)C[31 * N + 1],
               (double)C[31 * N + 2], (double)C[31 * N + 3]);
    }
    return bad ? 1 : 0;
}

static int do_hrefall(int mode, int reps, int verbose)
{
    static const int Ks[] = { 32, 64, 128, 256 };
    static const int Ns[] = { 32, 64 };
    int i, j, fail = 0;
    printf("qhl_hmx ref path, mode=%#x reps=%d\n", mode, reps);
    for (i = 0; i < 4; i++)
        for (j = 0; j < 2; j++)
            fail |= do_href_one(Ns[j], Ks[i], mode, reps, verbose);
    printf("RESULT %s\n", fail ? "FAIL" : "OK");
    return fail;
}

static int do_refdump(int K, int side)
{
    static unsigned char buf[32 * 256 * 2];
    unsigned short *v = (unsigned short *)buf;
    int info[4];
    int rc, i, bad1 = 0, bad2 = 0, bad3 = 0, bad4 = 0, n = 0;
    int maxv;

    rc = mx4_refdump(K, side, buf, (int)sizeof(buf), info, 4);
    printf("refdump K=%d side=%d rc=%d st=%d hvx=%d pack=%d bytes=%d\n",
           K, side, rc, info[0], info[1], info[2], info[3]);
    if (rc || info[0]) return 1;

    maxv = (side == 0) ? 32 * K : K * 64;
    for (i = 0; i < info[3] / 2; i++)
        if (v[i] >= maxv) { printf("  ! elem %d = %u out of range\n", i, v[i]); return 1; }

    /* candidate formulas (element index inside the 2 KB crouton pair) */
    for (i = 0; i < info[3] / 2; i++) {
        int src = v[i], r, c, f1, f2, f3, f4;
        if (side == 0) { r = src / K; c = src % K; }
        else           { r = src / 64; c = src % 64; }
        f1 = 64 * (r >> 1) + 2 * c + (r & 1);                    /* SDK comment "Elements" */
        f2 = 64 * (r >> 1) + 32 * (r & 1) + 16 * (c >> 4) + (c & 15); /* git-blame variant */
        f3 = 128 * (r >> 2) + 4 * c + (r & 3);                   /* old bench layout */
        /* crouton = 32 of the "other axis" per 2 KB, f1 inside */
        if (side == 0)
            f4 = 1024 * (c >> 5) + 64 * (r >> 1) + 2 * (c & 31) + (r & 1);
        else
            f4 = 1024 * (r >> 5) + 64 * ((r & 31) >> 1) + 2 * c + ((r & 31) & 1);
        if (f1 != i) bad1++;
        if (f2 != i) bad2++;
        if (f3 != i) bad3++;
        if (f4 != i) bad4++;
        if (n < 8 && (r < 2 || r == (side == 0 ? 31 : K - 1))) {
            printf("  dest[%4d] <- (%2d,%2d)   f1=%4d f2=%4d f3=%4d f4=%4d\n",
                   i, r, c, f1, f2, f3, f4);
            n++;
        }
    }
    printf("  mismatches: f1=%d f2=%d f3=%d f4=%d of %d elems\n",
           bad1, bad2, bad3, bad4, info[3] / 2);
    return 0;
}

/* ---- int8 probe --------------------------------------------------- */

static int i8_off(int cand, int s, int n)
{
    switch (cand) {
    case 0:  /* 64 ch/crouton, spatial pairs interleaved 1:1 */
        return 2048 * (n >> 6) + 128 * (s >> 1) + 2 * (n & 63) + (s & 1);
    case 1:  /* 64 ch/crouton, 64-byte row blocks */
        return 2048 * (n >> 6) + 128 * (s >> 1) + 64 * (s & 1) + (n & 63);
    case 2:  /* 32 ch/crouton, 1 KB payload */
        return 2048 * (n >> 5) + 64 * (s >> 1) + 2 * (n & 31) + (s & 1);
    case 3:  /* old bench layout (s>>2)*128 + n*4 + (s&3) */
        return (s >> 2) * 128 + 4 * n + (s & 3);
    case 4:  /* plain row-major 32x64 */
        return 64 * s + n;
    case 6:  /* int8 hw layout, 2 KB group per 32 output channels */
        return 2048 * (n >> 5) + (s >> 2) * 128 + 4 * (n & 31) + (s & 3);
    default: /* 32 ch/crouton packed back to back (1 KB per crouton) */
        return 1024 * (n >> 5) + 64 * (s >> 1) + 2 * (n & 31) + (s & 1);
    }
}

/* A=B=1 (amax<0): every output cell must read (#MACs accumulated), which
 * decouples "does it multiply" from "is the layout right". */
/* A=B=1 (amax<0): every output cell reads the number of MACs accumulated,
 * which decouples "does it multiply" from "is the layout right". */
static int do_i8sweep(int N, int K, int lay, int wl, int mode)
{
    static const struct { int art, wc, srt, mul, np, bs; } sw[] = {
        { 0x3FF, 1023, 0, 0, 0, 0 },          /* no bias at all */
        { 0x3FF, 1023, 0, 0, 0, 0x6000 },     /* verified int8 recipe */
        { 0x3FF, 1023, 0, 0, 0, 0x3C00 },     /* 1.0 as plain fp16 */
        { 0x3FF, 1023, 0, 1, 0, 0x6000 },     /* adeep instead of pairs */
        { 0x3FF, 1023, 3, 0, 0, 0x6000 },     /* uh bank form */
        { 0x07C, 1023, 0, 0, 0, 0x6000 },     /* act_rt control point */
        { 0x77C, 1023, 0, 0, 0, 0x6000 },     /* act_rt control point */
        { 0x3FF, 1023, 0, 0, 1, 0x6000 },     /* only the first pair */
    };
    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16];
    unsigned cfg;

    A = malloc(32 * 256); B = malloc(64 * 256);
    if (!A || !B) { printf("alloc failed\n"); return 1; }
    memset(A, 1, 32 * 256);
    memset(B, 1, 64 * 256);
    for (cfg = 0; cfg < sizeof(sw) / sizeof(sw[0]); cfg++) {
        int mx = 0, nz = 0, i, vals[12], nv = 0;
        int rc = mx4_i8((unsigned char *)A, 32 * K, (unsigned char *)B, K * N,
                        N, K, lay, wl, sw[cfg].art, sw[cfg].wc, sw[cfg].srt,
                        sw[cfg].mul, sw[cfg].np, sw[cfg].bs, mode, 1,
                        raw, 4096, info, 16);
        if (rc || info[0]) {
            printf("  cfg%u art=0x%05x wc=%d srt=%d mul=%d bs=%#x -> rc=%d st=%d\n",
                   cfg, (unsigned)sw[cfg].art, sw[cfg].wc, sw[cfg].srt,
                   sw[cfg].mul, (unsigned)sw[cfg].bs, rc, info[0]);
            continue;
        }
        for (i = 0; i < 4096; i++) {
            int v = raw[i];
            if (v) nz++;
            if (v > mx) mx = v;
            if (v && nv < 12) {
                int j, seen = 0;
                for (j = 0; j < nv; j++) if (vals[j] == v) seen = 1;
                if (!seen) vals[nv++] = v;
            }
        }
        printf("  cfg%u art=0x%05x wc=%d srt=%d mul=%d np=%d bs=%#x -> "
               "max=%3d nz=%4d us=%d vals:", cfg, (unsigned)sw[cfg].art,
               sw[cfg].wc, sw[cfg].srt, sw[cfg].mul, sw[cfg].np,
               (unsigned)sw[cfg].bs, mx, nz, info[1]);
        for (i = 0; i < nv; i++) printf(" %d", vals[i]);
        printf("%s\n", nz ? "" : "   (empty)");
    }
    free(A); free(B);
    return 0;
}

static int do_i8(int N, int K, int lay, int wl, int art, int wc,
                 int srt, int mul, int npair, int bs, int amax, int mode, int reps)
{    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16];
    int rc, s, n, k, c, bad, best = -1;
    double bcos = -1;
    int n255 = 0, n0 = 0, sat = 0;

    if (!A) {
        A = malloc(32 * 2048);
        B = malloc(64 * 2048);
        if (!A || !B) { printf("alloc failed\n"); return 1; }
    }
    for (s = 0; s < 32; s++)
        for (k = 0; k < K; k++)
            A[s * K + k] = (int8_t)(amax == -2 ? ((rnd() % 8) == 0)
                              : amax == -1 ? 1
                              : amax ? (int)(rnd() % (unsigned)(amax + 1)) : 0);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++)
            B[n * K + k] = (int8_t)(amax == -2 ? ((rnd() % 2) ? 0 : ((rnd() & 1) ? 1 : -1))
                              : amax == -1 ? 1
                              : amax ? (int)(rnd() % (unsigned)(amax + 1)) : 0);

    rc = mx4_i8((unsigned char *)A, 32 * K, (unsigned char *)B, K * N,
                N, K, lay, wl, art, wc, srt, mul, npair, bs, mode, reps,
                raw, 4096, info, 16);
    printf("i8 K=%3d N=%2d lay=%d wl=%d art=0x%05x wc=0x%04x srt=%d mul=%d "
           "np=%d bs=%#x reps=%d rc=%d st=%d us=%d pcyc=%d act=%d wgt=%d\n",
           K, N, lay, wl, (unsigned)art, (unsigned)wc, srt, mul,
           npair, (unsigned)bs, reps, rc, info[0], info[1], info[2], info[12], info[13]);
    if (rc || info[0]) return 1;

    for (c = 0; c < 2048; c++) {
        if (raw[c] == 255) n255++;
        if (raw[c] == 0) n0++;
    }
    printf("  tile: 255=%d 0=%d of 2048   head:", n255, n0);
    for (c = 0; c < 16; c++) printf(" %3d", raw[c]);
    printf("\n");

    for (c = 0; c < 7; c++) {
        double mx = 0, dot = 0, na = 0, nb = 0, cs;
        bad = 0; sat = 0;
        for (s = 0; s < 32; s++)
            for (n = 0; n < N; n++) {
                int ref = 0, got, e;
                for (k = 0; k < K; k++)
                    ref += (int)A[s * K + k] * (int)B[n * K + k];
                got = raw[i8_off(c, s, n)];
                if (ref > 255) { ref = 255; sat++; }
                if (ref < 0) ref = 0;
                e = got - ref; if (e < 0) e = -e;
                if (e > mx) mx = e;
                if (e > 1) bad++;
                dot += (double)got * ref; na += (double)got * got;
                nb += (double)ref * ref;
            }
        cs = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0;
        printf("  cand%d err=%.1f cos=%.6f bad=%d/%d sat=%d%s\n", c, mx, cs, bad,
               32 * N, sat, (cs > 0.999 && bad == 0) ? "   <== MATCH" : "");
        if (cs > bcos) { bcos = cs; best = c; }
    }
    printf("  best=cand%d cos=%.6f\n", best, bcos);
    return 0;
}

/* ---- int8 with M spatial rows (32/64) --------------------------------- */
static int do_i8m(int M, int N, int K, int bs, int amax, int mode, int reps,
                  int mul, int npair)
{
    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16];
    int rc, s, n, k, c, bad;
    double mxerr, cosv, dot, na, nb;
    int sat = 0, nz = 0;
    long long macs;
    double usrep;

    if (!A) {
        A = malloc(64 * 2048);
        B = malloc(64 * 2048);
        if (!A || !B) { printf("alloc failed\n"); return 1; }
    }
    for (s = 0; s < M; s++)
        for (k = 0; k < K; k++)
            A[s * K + k] = (int8_t)(amax == -2 ? ((rnd() % 8) == 0)
                              : amax == -1 ? 1
                              : amax ? (int)(rnd() % (unsigned)(amax + 1)) : 0);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++)
            B[n * K + k] = (int8_t)(amax == -2 ? ((rnd() % 2) ? 0 : ((rnd() & 1) ? 1 : -1))
                              : amax == -1 ? 1
                              : amax ? (int)(rnd() % (unsigned)(amax + 1)) : 0);

    rc = mx4_i8m((unsigned char *)A, M * K, (unsigned char *)B, K * N,
                 M, N, K, 3, (N > 32 ? 4 : 3), 0x3FF, (N > 32 ? 2047 : 1023), 0,
                 mul, npair, bs, mode, reps,
                 raw, 4096, info, 16);
    if (rc || info[0]) {
        printf("i8m M=%d N=%d K=%d rc=%d st=%d\n", M, N, K, rc, info[0]);
        return 1;
    }
    usrep = (double)info[1] / (double)reps;
    macs = (long long)M * N * K;
    printf("i8m M=%d N=%d K=%d mul=%d bs=%#x amax=%d reps=%d us/rep=%.3f "
           "pcyc/rep=%d TMAC/s=%.3f\n", M, N, K, mul, (unsigned)bs, amax, reps,
           usrep, info[2], usrep > 0 ? macs / usrep * 1e-6 : 0.0);
    for (c = 0; c < 2048; c++) if (raw[c]) nz++;
    printf("  tile: nz=%d/%d head:", nz, 2048);
    for (c = 0; c < 12; c++) printf(" %3d", raw[c]);
    printf(" | s32 half:");
    for (c = 1024; c < 1030; c++) printf(" %3d", raw[c]);
    printf("\n");

    for (c = 3; c <= 6; c += 3) {
        mxerr = 0; dot = 0; na = 0; nb = 0; bad = 0; sat = 0;
        for (s = 0; s < M; s++)
            for (n = 0; n < N; n++) {
                int ref = 0, got, e;
                for (k = 0; k < K; k++)
                    ref += (int)A[s * K + k] * (int)B[n * K + k];
                got = raw[i8_off(c, s, n)];
                if (ref > 255) { ref = 255; sat++; }
                if (ref < 0) ref = 0;
                e = got - ref; if (e < 0) e = -e;
                if (e > mxerr) mxerr = (double)e;
                if (e > 0) bad++;
                dot += (double)got * ref; na += (double)got * got;
                nb += (double)ref * ref;
            }
        cosv = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0;
        printf("  layout%d maxerr=%.1f cos=%.7f bad=%d/%d sat=%d%s\n", c,
               mxerr, cosv, bad, M * N, sat,
               (bad == 0 && cosv > 0.9999) ? "   <== MATCH" : "");
    }
    return 0;
}

/* one-hot activation probe: A[s0][k0]=1, B=1 everywhere -> which output rows
 * consumed that activation byte? (a nonzero output row == the row that read it) */
static int do_i8one(int s0, int k0, int K, int N, int bs, int mode, int reps)
{
    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16], rc, s, n, k, i, hits = 0;

    A = calloc(64 * 2048, 1);
    B = malloc(64 * 2048);
    if (!A || !B) { printf("alloc failed\n"); return 1; }
    memset(B, 1, 64 * 2048);
    A[s0 * K + k0] = 1;
    rc = mx4_i8m((unsigned char *)A, 64 * K, (unsigned char *)B, K * N,
                 64, N, K, 3, 3, 0x3FF, 1023, 0, 0, 0, bs, mode, reps,
                 raw, 4096, info, 16);
    if (rc || info[0]) { printf("i8one rc=%d st=%d\n", rc, info[0]); return 1; }
    printf("i8one act(s=%d,k=%d) K=%d N=%d ->", s0, k0, K, N);
    for (s = 0; s < 64; s++)
        for (n = 0; n < N; n++)
            if (raw[i8_off(3, s, n)]) {
                if (hits < 8) printf(" (s=%d,n=%d)=%d", s, n, raw[i8_off(3, s, n)]);
                hits++;
            }
    printf("  hits=%d\n", hits);
    free(A); free(B);
    return 0;
}

/* spatial identity ramp: A[s][k]=s (k independent), B[n][0]=1, B[n][k>0]=0.
 * out[s][n] then equals the spatial row s' whose activation byte was read.
 * A duplicate/aliased bank shows up as out[32+i] == i. */
static int do_i8ramp(int K, int N, int bs, int mul, int npair, int reps)
{
    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16], rc, s, n, bad = 0;

    A = malloc(64 * 2048); B = calloc(64 * 2048, 1);
    if (!A || !B) { printf("alloc failed\n"); return 1; }
    for (s = 0; s < 64; s++)
        for (int k = 0; k < K; k++) A[s * K + k] = (int8_t)(s + 1);
    for (n = 0; n < N; n++) B[n * K] = 1;
    rc = mx4_i8m((unsigned char *)A, 64 * K, (unsigned char *)B, K * N,
                 64, N, K, 3, 3, 0x3FF, 1023, 0, mul, npair, bs, 0, reps,
                 raw, 4096, info, 16);
    if (rc || info[0]) { printf("i8ramp rc=%d st=%d\n", rc, info[0]); return 1; }
    printf("i8ramp K=%d N=%d mul=%d np=%d -> out rows (should be s+1):\n   ", K, mul, npair);
    for (s = 0; s < 64; s++) {
        int v = raw[i8_off(3, s, 0)];
        if (v != s + 1) bad++;
        printf("%d%s", v, (s % 16 == 15) ? "\n   " : " ");
    }
    printf("\n   bad=%d/64\n", bad);
    free(A); free(B);
    return 0;
}

/* ---- batched prefill GEMM -------------------------------------------- */
static int do_bw2(int mb, int kind, int chunkKb, int reps, int useHost)
{
    int r[16];
    int n = (mb << 20) + (1 << 18);
    unsigned char *src = (unsigned char *)malloc(n);
    int rc;

    if (!src) return 1;
    memset(src, 0x5a, n);
    rc = mx4_bw2(src, n, kind, chunkKb, reps,
                 (useHost ? 1 : 0) | (getenv("BW_HMX") ? 0x40 : 0) |
                 (getenv("BW_VOTE") ? 6 : 0), r, 16);
    printf("bw2 %dMB kind=%d chunk=%dKB reps=%d rc=%d st=%d src=%s hmx=%d\n"
           "  A : %.2f GB/s (us=%u pcyc=%u)\n",
           mb, kind, chunkKb, reps, rc, r[0], useHost ? "host" : "heap",
           getenv("BW_HMX") ? 1 : 0,
           (double)r[2] / (r[1] ? r[1] * 1e3 : 1.0), (unsigned)r[1], (unsigned)r[3]);
    if (getenv("BW_VOTE"))
        printf("  B vote : %.2f GB/s (us=%d pcyc=%d) vote_rc=%#x\n"
               "  C undo : %.2f GB/s (us=%d pcyc=%d)\n",
               (double)r[10] / 100.0, r[8], r[9], r[6],
               (double)r[14] / 100.0, r[12], r[13]);
    free(src);
    return 0;
}

static const char *bw2_kind_name(int k)
{
    static const char *n[24] = {
        "read-sum 4acc", "read-sum 16-inflight", "copy 2KB croutons",
        "copy chunk plain", "copy chunk l2fetch", "read-sum l2fetch",
        "copy chunk memcpy", "copy 2KB 8+8",
        "read-sum 2 streams", "read-sum 4 streams",
        "read-sum 8 streams", "read-sum 16 streams",
        "copy 2 streams", "copy 4 streams", "copy 8 streams",
        "copy l2f 8 streams", "copy l2f 4 streams", "copy L2-resident 512K",
        "copy 4x4 batched", "copy 8x2 batched", "2-pass L2 staging"
    };
    return (k >= 0 && k < 21) ? n[k] : "?";
}

static int do_bwt(int mb, int chunkKb, int reps, int rounds)
{
    int r[64];
    int n = (mb << 20) + (1 << 18);
    unsigned char *src = (unsigned char *)malloc(n);
    int rc, k;

    if (!src) return 1;
    memset(src, 0x5a, n);
    rc = mx4_bwt(src, n, chunkKb, reps, rounds, r, 64);
    printf("bwt %dMB chunk=%dKB reps=%d rounds=%d rc=%d st=%d\n",
           mb, chunkKb, reps, rounds, rc, r[0]);
    if (rc == 0 && r[0] == 0) {
        for (k = 0; k < 21; k++)
            printf("  %-20s last %6.2f  best %6.2f GB/s\n", bw2_kind_name(k),
                   r[4 + k] / 100.0, r[28 + k] / 100.0);
    }
    free(src);
    return 0;
}

static int do_pref(int M, int N, int K, int mode, int reps)
{
    static int8_t *A, *B;
    static unsigned char *C;
    int info[16];
    int rc, s, n, k, bad = 0, sat = 0;
    double macs, us;
    long long lmacs;

    A = malloc((size_t)M * K);
    B = malloc((size_t)N * K);
    C = malloc((size_t)M * N);
    if (!A || !B || !C) { printf("alloc failed\n"); return 1; }
    for (s = 0; s < M; s++)
        for (k = 0; k < K; k++) A[(size_t)s * K + k] = (int8_t)((rnd() % 8) == 0 ? 1 : 0);
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++) B[(size_t)n * K + k] = (int8_t)((int)(rnd() % 3) - 1);

    rc = mx4_pref((unsigned char *)A, M * K, (unsigned char *)B, N * K,
                  M, N, K, mode, reps, C, M * N, info, 16);
    if (rc || info[0]) { printf("pref M=%d N=%d K=%d rc=%d st=%d\n", M, N, K, rc, info[0]); return 1; }
    lmacs = (long long)M * N * K;
    us = (double)info[1] / (double)reps;
    printf("pref M=%d N=%d K=%d mode=%d reps=%d -> us/rep=%.1f pcyc/rep=%d "
           "TMAC/s=%.2f  (ntiles=%d mtiles=%d ncr=%d vtcm=%d wgt=%d)\n",
           M, N, K, mode, reps, us, info[2] / reps,
           us > 0 ? lmacs / us * 1e-6 : 0.0, info[3], info[4], info[5],
           info[10], info[11]);
    printf("  copycheck(bad bytes in first staged tile)=%d  copy_us/rep=%.1f\n",
           info[12], (double)info[15] / reps);
    if (us > 0) printf("  per token: %.3f us -> %.1f tok/s (batch %d)\n",
                       us / M, M * 1e6 / us, M);

    if (mode & 8) {
        long long cmacs = 0;
        double dot = 0, na = 0, nb = 0, mx = 0;
        for (s = 0; s < M; s++)
            for (n = 0; n < N; n++) {
                int ref = 0, got = C[(size_t)s * N + n], e;
                for (k = 0; k < K; k++) ref += (int)A[(size_t)s * K + k] * (int)B[(size_t)n * K + k];
                cmacs++;
                if (ref > 255) { ref = 255; sat++; }
                if (ref < 0) ref = 0;
                e = got - ref; if (e < 0) e = -e;
                if (e > mx) mx = (double)e;
                if (e) { if (bad < 6) printf("  [bad s=%d n=%d got=%d ref=%d]%s", s, n, got, ref, (bad==5)?"\n":"  "); bad++; }
                dot += (double)got * ref; na += (double)got * got; nb += (double)ref * ref;
            }
        printf("  check: cells=%lld maxerr=%.0f cos=%.7f bad=%d sat=%d%s\n",
               cmacs, mx, (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0, bad, sat,
               bad ? "" : "   <== MATCH");
    }
    free(A); free(B); free(C);
    return 0;
}

/* ---- int4 weights (weight.n) ------------------------------------------ */
static int do_i4(int M, int N, int K, int wl, int bs, int amax, int mode, int reps)
{
    static int8_t *A, *B;
    static unsigned char raw[4096];
    int info[16], rc, s, n, k, c, bad = 0, sat = 0, nz = 0;
    double mx = 0, cos = 0, dot = 0, na = 0, nb = 0, us;

    A = malloc(64 * 2048); B = malloc(32 * 2048);
    if (!A || !B) { printf("alloc failed\n"); return 1; }
    for (s = 0; s < 64; s++)
        for (k = 0; k < K; k++)
            A[s * K + k] = (int8_t)(amax == -1 ? 1
                              : (amax <= -8) ? 1
                              : (amax == -7) ? (k >= 32 ? 1 : 0)
                              : (amax <= -2) ? (int)(rnd() % 4u)
                              : amax ? (int)(rnd() % (unsigned)(amax + 1)) : 0);
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++) {
            int v;
            if (amax == -1 || amax == -4) v = 1;
            else if (amax == -2) v = (int)(rnd() % 16) - 8;         /* -8..7 */
            else if (amax == -5) v = (k >= 32) ? 1 : 0;             /* 2nd half */
            else if (amax == -6) v = (k < 32) ? 1 : 0;              /* 1st half */
            else if (amax == -7) v = 1;                              /* act 2nd */
            else if (amax <= -8) { int cc = -8 - amax;
                                   v = (((int)k >> 5) == cc) ? 1 : 0; }
            else if (amax == -3) v = ((int)(rnd() % 2) ? 7 : -7);   /* +-7 */
            else v = (int)(rnd() % (unsigned)(amax + 1));
            B[n * K + k] = (int8_t)v;
        }
    rc = mx4_i4((unsigned char *)A, 64 * K, (unsigned char *)B, N * K,
                M, N, K, wl, 0x3FF,
                getenv("I4_WC") ? atoi(getenv("I4_WC")) : 0, bs, mode, reps,
                raw, 4096, info, 16);
    if (rc || info[0]) { printf("i4 rc=%d st=%d\n", rc, info[0]); return 1; }
    us = (double)info[1] / (double)reps;
    printf("i4 M=%d N=%d K=%d wl=%d bs=%#x amax=%d mode=%d reps=%d us/rep=%.3f "
           "pcyc=%d\n", M, N, K, wl, (unsigned)bs, amax, mode, reps, us, info[2]);
    for (c = 0; c < 2048; c++) if (raw[c]) nz++;
    { int nz2 = 0;
      for (c = 1024; c < 2048; c++) if (raw[c]) nz2++;
      printf("  tile nz=%d nz2=%d head:", nz, nz2);
      for (c = 0; c < 12; c++) printf(" %3d", raw[c]);
      printf("  b1:"); for (c = 1024; c < 1036; c++) printf(" %3d", raw[c]);
      printf("\\n"); }
    bad = 0; sat = 0;
    for (s = 0; s < 32; s++)
        for (n = 0; n < N; n++) {
            int ref = 0, got, e;
            for (k = 0; k < K; k++) ref += (int)A[s * K + k] * (int)B[n * K + k];
            got = raw[(s >> 2) * 128 + 4 * n + (s & 3)];
            if (ref > 255) { ref = 255; sat++; }
            if (ref < 0) ref = 0;
            e = got - ref; if (e < 0) e = -e;
            if ((double)e > mx) mx = (double)e;
            if (e) bad++;
            dot += (double)got * ref; na += (double)got * got; nb += (double)ref * ref;
        }
    cos = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0;
    printf("  check maxerr=%.0f cos=%.7f bad=%d/%d sat=%d%s\n", mx, cos, bad,
           32 * N, sat, (bad == 0 && cos > 0.9999) ? "   <== MATCH" : "");
    free(A); free(B);
    return 0;
}

int main(int argc, char **argv)
{
    const char *cmd = (argc > 1) ? argv[1] : "info";
    setvbuf(stdout, NULL, _IONBF, 0);
    if (enable_unsigned() != 0) return 10;

    if (!strcmp(cmd, "info")) return do_info();
    if (!strcmp(cmd, "recipe")) return do_recipe();
    if (!strcmp(cmd, "s1fp16")) {
        static __fp16 A[1024], B[1024], C[1024];
        int info[2], rc, s, n, k, i;
        double mx = 0; int bad = 0;
        for (i = 0; i < 1024; i++) {
            A[i] = (__fp16)((int)(rnd() % 17) - 8);
            B[i] = (__fp16)((int)(rnd() % 3) - 1);
        }
        { int pk = argc > 2 ? atoi(argv[2]) : 0, wr = argc > 3 ? atoi(argv[3]) : 2047;
          mx4_cfg(pk, wr);
          printf("  cfg pack=%d wrt=%d\n", pk, wr); }
        rc = mx4_s1f16((unsigned char *)A, 2048, (unsigned char *)B, 2048,
                       (unsigned char *)C, 2048, info, 2);
        printf("s1fp16 32x32x32 rc=%d st=%d\n", rc, info[0]);
        if (rc || info[0]) return 1;
        for (s = 0; s < 32; s++)
            for (n = 0; n < 32; n++) {
                double ref = 0;
                for (k = 0; k < 32; k++) ref += (double)A[s * 32 + k] * (double)B[n * 32 + k];
                {
                    double e = (double)C[s * 32 + n] - ref;
                    if (e < 0) e = -e;
                    if (e > mx) mx = e;
                    if (e > 1e-2) bad++;
                }
            }
        printf("  fp16 S1: max_abs_err=%.5f  bad=%d/1024\n", mx, bad);
        return 0;
    }
    if (!strcmp(cmd, "s1q")) {
        static unsigned char A[1024], B[1024], raw[2048];
        int info[3], rc, q, s2, n, k;
        static const int cs[3][2] = { { 2048, 2047 }, { 2048, 2047 }, { 2048, 2047 } };
        for (q = 0; q < 2; q++) {
            long long mx = 0, bad = 0;
            for (s2 = 0; s2 < 1024; s2++) {
                if (q == 0) {   /* A[i][k]=i+k, B[j][k]=k-j+64 (по заданию) */
                    int i2 = s2 / 32, k2 = s2 % 32;
                    A[s2] = (unsigned char)(i2 + k2);
                    B[s2] = (unsigned char)(k2 - i2 + 64);
                } else {        /* малые значения: сильная проверка без насыщения */
                    int i2 = s2 / 32, k2 = s2 % 32;
                    A[s2] = (unsigned char)((i2 + k2) & 7);
                    B[s2] = (unsigned char)(int8_t)(((k2 - i2) & 7) - 3);
                }
            }
            rc = mx4_s1q(A, 1024, B, 1024, cs[q][0], cs[q][1], raw, 2048, info, 3);
            for (s2 = 0; s2 < 32; s2++)
                for (n = 0; n < 32; n++) {
                    unsigned o = (unsigned)(s2 >> 1) * 128u + (unsigned)n * 4u
                               + (unsigned)(s2 & 1) * 2u;
                    int hw = raw[o] | (raw[o + 1] << 8);
                    int32_t ref = 0;
                    long long e;
                    for (k = 0; k < 32; k++)
                        ref += (int32_t)A[s2 * 32 + k] * (int32_t)(int8_t)B[k * 32 + n];
                    if (ref < 0) ref = 0;
                    e = (long long)hw - ref; if (e < 0) e = -e;
                    if (e) { bad++; if (e > mx) mx = e; }
                }
            printf("  s1q wstride=%4d wgt_rt=%4d: rc=%d st=%d max_abs_err=%lld bad=%lld/1024\n",
                   cs[q][0], cs[q][1], rc, info[0], mx, bad);
            usleep(200000);
        }
        return 0;
    }
    if (!strcmp(cmd, "dlprobe")) {
        int r[5], rc = mx4_dlprobe(r, 5);
        printf("dlprobe: rc=%d loaded=%d handle=0x%x deep_sym=0x%x found=%d bbb_sym=0x%x\n",
               rc, r[0], (unsigned)r[1], (unsigned)r[2], r[3], (unsigned)r[4]);
        return 0;
    }
    if (!strcmp(cmd, "rs")) {
        unsigned char raw[2048];
        int info[3], q, rc;
        static const int ar[] = { 0x03C, 0x07C, 0x07FF, 0x0FFF, 0x03C };
        printf("rs (A=1,B=1): expect 32\n");
        for (q = 0; q < 5; q++) {
            mx4_cfg2(ar[q], 2047);
            mx4_cfg2(ar[q], (q == 2) ? 2047 : 1023);
            rc = mx4_qnn8(2, 512, (q == 2) ? 2047 : 1023, 512, raw, 2048, info, 3);
            printf("  act_rt=0x%03x wgt_rt=%4d: rc=%d st=%d out[0]=%d out[512]=%d\n",
                   ar[q], (q == 2) ? 2047 : 1023, rc, info[0], raw[0], raw[512]);
            usleep(150000);
        }
        return 0;
    }
    if (!strcmp(cmd, "qnn8")) {
        unsigned char raw[2048];
        int info[3], rc, q;
        static const int cfg[4][4] = {
            { 2, 512, 1023, 512 }, { 1, 0, 1023, 0 }, { 2, 1024, 1023, 1024 }, { 2, 512, 2047, 512 },
        };
        printf("qnn8 A=1,B=1: expect 32 when both croutons accumulate\n");
        for (q = 0; q < 4; q++) {
            rc = mx4_qnn8(cfg[q][0], cfg[q][1], cfg[q][2], cfg[q][3], raw, 2048, info, 3);
            printf("  ncr=%d wstride=%4d wgt_rt=%4d astep=%4d: rc=%d st=%d out=%.1f\n",
                   cfg[q][0], cfg[q][1], cfg[q][2], cfg[q][3], rc, info[0],
                   (double)h2d16(raw[0] | (raw[1] << 8)));
            usleep(150000);
        }
        return 0;
    }
    if (!strcmp(cmd, "swap")) {
        unsigned char raw[2048];
        int info[2], rc;
        printf("swap test (A=1,B=1,cnt=32): expected 32 when acc sums all\n");
        mx4_cfg(0, 2047); mx4_cfg2(0x7C, 2047);
        rc = mx4_psum(32, raw, 2048, info, 2);
        printf("  no swap : rc=%d out[0][0]=%.4f\n", rc,
               (double)h2d((unsigned short)(raw[0] | (raw[1] << 8))));
        usleep(200000);
        mx4_cfg(1, 2047);
        rc = mx4_psum(32, raw, 2048, info, 2);
        printf("  swapped : rc=%d out[0][0]=%.4f\n", rc,
               (double)h2d((unsigned short)(raw[0] | (raw[1] << 8))));
        return 0;
    }
    if (!strcmp(cmd, "wsweep")) {
        unsigned char raw[2048];
        int info[2], rc, q;
        static const int wr[] = { 127, 255, 511, 1023, 2047, 4095 };
        static const int ar[] = { 0x7C, 0x3C, 0x1C };
        printf("wsweep (A=1, B=1, 32 channels) -> out should be 32:\n");
        for (q = 0; q < 6; q++) {
            mx4_cfg2(0x7C, wr[q]);
            rc = mx4_psum(32, raw, 2048, info, 2);
            printf("  wgt_rt=%4d: rc=%d out=%.4f\n", wr[q], rc,
                   (double)h2d((unsigned short)(raw[0] | (raw[1] << 8))));
            usleep(150000);
        }
        for (q = 0; q < 3; q++) {
            mx4_cfg2(ar[q], 2047);
            rc = mx4_psum(32, raw, 2048, info, 2);
            printf("  act_rt=0x%03x: rc=%d out=%.4f\n", ar[q], rc,
                   (double)h2d((unsigned short)(raw[0] | (raw[1] << 8))));
            usleep(150000);
        }
        return 0;
    }
    if (!strcmp(cmd, "psum")) {
        unsigned char raw[2048];
        int info[2], rc, cnt;
        static const int cs[] = { 1, 2, 3, 4, 8, 16, 24, 32 };
        int q;
        printf("psum (A=1, B=1 for k<cnt) -> out should equal cnt:\n");
        for (q = 0; q < 8; q++) {
            cnt = cs[q];
            rc = mx4_psum(cnt, raw, 2048, info, 2);
            printf("  cnt=%2d: rc=%d out[0][0]=%.4f  out[1][1]=%.4f\n", cnt, rc,
                   (double)h2d((unsigned short)(raw[0] | (raw[1] << 8))),
                   (double)h2d((unsigned short)(raw[2 * 64 + 2 * 1 + 1] |
                                              (raw[(64 + 3) * 2 + 1] << 8))));
            usleep(150000);
        }
        return 0;
    }
    if (!strcmp(cmd, "kscan")) {
        unsigned char raw[2048];
        int info[2], k0, rc, hits = 0;
        printf("kscan (fp16, A=delta(k), B=1): ");
        for (k0 = 0; k0 < 32; k0++) {
            rc = mx4_kscan(k0, raw, 2048, info, 2);
            {
                double v = (double)h2d((unsigned short)(raw[0] | (raw[1] << 8)));
                printf("%c", (v > 0.5) ? '1' : '.');
                if (v > 0.5) hits++;
            }
            usleep(120000);
        }
        printf("  hits=%d/32\n", hits);
        return 0;
    }
    if (!strcmp(cmd, "amap2")) {
        int mode = argc > 2 ? atoi(argv[2]) : 0;
        unsigned char raw[2048];
        int info[2], rc, s, n, i, sr, cr;
        double v;
        rc = mx4_amap2(mode, raw, 2048, info, 2);
        printf("amap2 mode=%d rc=%d st=%d\n", mode, rc, info[0]);
        printf("  words[0..7]:");
        for (i = 0; i < 8; i++)
            printf(" %.4f", (double)h2d((unsigned short)(raw[i*2] | (raw[i*2+1] << 8))));
        printf("\n");
        for (s = 0; s < 4; s++) {
            printf("  s=%d:", s);
            for (n = 0; n < 8; n++) {
                unsigned widx = (unsigned)(s >> 1) * 64u + (unsigned)n * 2u + (unsigned)(s & 1);
                v = (double)h2d((unsigned short)(raw[widx*2] | (raw[widx*2+1] << 8)));
                sr = (int)(v + 1e-4);
                cr = (int)((v - (double)sr) * 32.0 + 0.5);
                printf(" %d:%d", sr, cr);
            }
            printf("\n");
        }
        return 0;
    }
    if (!strcmp(cmd, "idmap")) {
        int mode = argc > 2 ? atoi(argv[2]) : 0;
        unsigned char raw[2048];
        int info[4], rc, i;
        rc = mx4_idmap(mode, raw, 2048, info, 4);
        printf("idmap mode=%d rc=%d st=%d\n  ", mode, rc, info[0]);
        for (i = 0; i < 32; i++) {
            unsigned widx = (unsigned)(i >> 1) * 64u + (unsigned)(i & 1);  /* (s=i,n=0) */
            printf(" %d", (int)h2d((unsigned short)(raw[widx*2] | (raw[widx*2+1] << 8))));
        }
        printf("\n  ch row s=0:");
        for (i = 0; i < 32; i++) {
            unsigned widx = (unsigned)i * 2u;   /* (s=0,n=i) */
            printf(" %d", (int)h2d((unsigned short)(raw[widx*2] | (raw[widx*2+1] << 8))));
        }
        printf("\n");
        return 0;
    }
    if (!strcmp(cmd, "covsweep")) {
        /* one store per run, address swept 0..1920 step 128 */
        int off, r[6], rc, filled_total = 0;
        unsigned char raw[2048];
        printf("covsweep (A=1,B=I -> expect 1024 words == 0x3C00 when full)\n");
        { int r8[6]; unsigned char raw8[2048];
          int cfg[6][3] = {{0,0,1},{0,2,2},{0,4,2},{0,8,2},{0,1024,2},{0,2,16}};
          int q;
          for (q = 0; q < 6; q++) {
              mx4_cov(cfg[q][0], cfg[q][1], cfg[q][2], 0, 3, raw8, 2048, r8, 6);
              printf("  int8 off=%d step=%d nst=%d -> filled(16s)=%3d/1024\n",
                     cfg[q][0], cfg[q][1], cfg[q][2], r8[1]);
              usleep(150000);
          }
        }
        for (off = 0; off <= 1920; off += 128) {
            rc = mx4_cov(off, 0, 1, 0, 0, raw, 2048, r, 6);
            printf("  off=%4d: rc=%d st=%d filled=%3d\n", off, rc, r[0], r[1]);
            if (r[1] > filled_total) filled_total = r[1];
        }
        return 0;
    }
    if (!strcmp(cmd, "cov"))
        return do_cov(argc > 2 ? atoi(argv[2]) : 0, argc > 3 ? atoi(argv[3]) : 0,
                      argc > 4 ? atoi(argv[4]) : 1, argc > 5 ? (int)strtol(argv[5],0,0) : 0,
                      argc > 6 ? atoi(argv[6]) : 0);
    if (!strcmp(cmd, "fp16ex"))
        return do_fp16ex(argc > 2 ? (int)strtol(argv[2], 0, 0) : 28);
    if (!strcmp(cmd, "sweep")) return do_sweep();
    if (!strcmp(cmd, "fp16")) return do_fp16();
    if (!strcmp(cmd, "s1r"))
        return do_s1r2(argc > 2 ? atoi(argv[2]) : 8, argc > 3 ? atoi(argv[3]) : 0);
    if (!strcmp(cmd, "dump"))
        return do_dump(argc > 2 ? atoi(argv[2]) : 0, argc > 3 ? atoi(argv[3]) : 0,
                       argc > 4 ? atoi(argv[4]) : 16);
    if (!strcmp(cmd, "s1h2"))
        return do_s1h2(argc > 2 ? atoi(argv[2]) : 101, argc > 3 ? atoi(argv[3]) : 10,
                       argc > 4 ? atoi(argv[4]) : 16, argc > 5 ? atoi(argv[5]) : 0);
    if (!strcmp(cmd, "amap")) return do_amap();
    if (!strcmp(cmd, "actch"))
        return do_actch(argc > 2 ? atoi(argv[2]) : 90, argc > 3 ? atoi(argv[3]) : 0,
                        argc > 4 ? atoi(argv[4]) : 0);
    if (!strcmp(cmd, "asweep")) return do_asweep();
    if (!strcmp(cmd, "kmap"))
        return do_kmap(argc > 2 ? atoi(argv[2]) : 77, argc > 3 ? atoi(argv[3]) : 73);
    if (!strcmp(cmd, "maps2")) {
        int wh = argc > 2 ? atoi(argv[2]) : 1;
        g_act_layout = argc > 3 ? atoi(argv[3]) : 77;
        g_wgt_layout = argc > 4 ? atoi(argv[4]) : 73;
        return do_maps(wh);
    }
    if (!strcmp(cmd, "maps")) return do_maps(argc > 2 ? atoi(argv[2]) : 0);
    if (!strcmp(cmd, "srtest")) {
        do_srtest(0);
        return 0;
    }
    if (!strcmp(cmd, "wmap2")) return do_wmap2();
    if (!strcmp(cmd, "s1full")) {
        if (argc > 3) g_act_layout = atoi(argv[3]);
        if (argc > 4) g_wgt_layout = atoi(argv[4]);
        return do_s1full(argc > 2 ? atoi(argv[2]) : 1);
    }
    if (!strcmp(cmd, "rngscan")) {
        int gran = argc > 2 ? atoi(argv[2]) : 64;
        int sd = argc > 3 ? atoi(argv[3]) : 3;
        int start, r[5], rc;
        printf("rngscan gran=%d side=%d:", gran, sd);
        for (start = 0; start + gran <= 2048; start += gran) {
            rc = mx4_pos(sd, (start << 11) | (gran & 0x7ff), 0x7C, 1023, r, 5);
            if (rc || r[0]) { printf("[rc=%d st=%d@%d]", rc, r[0], start); continue; }
            printf(" %d:[%d,%d,%d,%d]", start, r[1], r[2], r[3], r[4]);
        }
        printf("\n");
        return 0;
    }
    if (!strcmp(cmd, "rng")) {
        int start = argc > 2 ? atoi(argv[2]) : 0;
        int len = argc > 3 ? atoi(argv[3]) : 1024;
        int r[5];
        int rc = mx4_pos(3, (start << 11) | (len & 0x7ff), 0x7C, 1023, r, 5);
        printf("rng [%d,%d): rc=%d st=%d out=%d\n", start, start + len, rc, r[0], r[1]);
        return 0;
    }
    if (!strcmp(cmd, "postest")) {
        int r[5], rc = mx4_pos(argc > 2 ? atoi(argv[2]) : 2,
                               argc > 3 ? atoi(argv[3]) : 0, 0x7C, 1023, r, 5);
        printf("postest side=%d q=%d: rc=%d st=%d words=%d,%d,%d,%d\n",
               argc > 2 ? atoi(argv[2]) : 2, argc > 3 ? atoi(argv[3]) : 0,
               rc, r[0], r[1], r[2], r[3], r[4]);
        return 0;
    }
    if (!strcmp(cmd, "posscan")) {
        int side = argc > 2 ? atoi(argv[2]) : 1;
        int ac = argc > 3 ? (int)strtol(argv[3], 0, 0) : 0x7C;
        int q, r[5], rc, hits = 0;
        printf("posscan side=%d ac=0x%x: ", side, ac);
        for (q = 0; q < 2048; q++) {
            rc = mx4_pos(side, q, ac, 1023, r, 5);
            if (rc || r[0]) { printf("[rc=%d st=%d]", rc, r[0]); break; }
            if (r[1]) { printf("%d:%d ", q, r[1]); hits++; }
        }
        printf(" hits=%d\n", hits);
        return 0;
    }
    if (!strcmp(cmd, "sumtest")) {
        static const int ws[] = { 0, 31, 36, 30, 2 };
        int i;
        for (i = 0; i < 5; i++) { do_sumtest(ws[i]); usleep(300000); }
        return 0;
    }
    if (!strcmp(cmd, "scan")) {
        return do_scan(argc > 2 ? atoi(argv[2]) : 0,
                       argc > 3 ? atoi(argv[3]) : 0,
                       argc > 4 ? atoi(argv[4]) : 0);
    }
    if (!strcmp(cmd, "bits")) {
        int who = argc > 2 ? atoi(argv[2]) : 0;
        int al = argc > 3 ? atoi(argv[3]) : 0;
        int wl = argc > 4 ? atoi(argv[4]) : 0;
        return do_bits(who, al, wl);
    }
    if (!strcmp(cmd, "sweep2")) {
        static const int acts[] = { 101, 101, 101, 106, 101 };
        static const int wgts[] = { 10, 11, 12, 10, 13 };
        int i, j;
        for (i = 0; i < 5; i++) {
            g_act_layout = acts[i]; g_wgt_layout = wgts[i];
            {
                uint8_t A[1024], B[1024], *Ap, *Bp, raw[2048];
                int info[6], s2, n2, rc;
                for (s2 = 0; s2 < 32; s2++)
                    for (n2 = 0; n2 < 32; n2++) {
                        A[s2 * 32 + n2] = (n2 < 16) ? (uint8_t)(n2 + 1) : 0;
                        B[s2 * 32 + n2] = (s2 == n2) ? 1 : 0;
                    }
                Ap = malloc(SLOT); Bp = malloc(SLOT);
                prepack_act(Ap, A, 32, 32, 255);
                prepack_wgt(Bp, (int8_t *)B, 32, 32);
                memset(raw, 0, sizeof(raw));
                rc = mx4_tile(Ap, SLOT, Bp, SLOT, 0x7C, 1023, 0, 6, 3, raw, 2048, info, 6);
                printf("act=%d wgt=%d rc=%d st=%d | s=0 row:");
                for (n2 = 0; n2 < 16; n2++) {
                    unsigned off = (unsigned)n2 * 4u;
                    printf(" %2d", raw[off] | (raw[off + 1] << 8));
                }
                printf("   (ref 1..16)\n");
                free(Ap); free(Bp);
            }
            usleep(400000);
        }
        return 0;
    }
    if (!strcmp(cmd, "grid"))
        return do_grid(argc > 2 ? atoi(argv[2]) : 101, argc > 3 ? atoi(argv[3]) : 2);
    if (!strcmp(cmd, "s1h")) {
        static const int lays[] = { 101, 103, 105 };
        int i;
        for (i = 0; i < 3; i++) {
            do_s1h(lays[i], 2, 0x7C, 1023, 1);
            usleep(400000);
        }
        return 0;
    }
    if (!strcmp(cmd, "s1"))
        return do_s1(argc > 2 ? atoi(argv[2]) : 0,
                     argc > 3 ? (int)strtol(argv[3], 0, 0) : 0x3C,
                     argc > 4 ? (int)strtol(argv[4], 0, 0) : 1023,
                     argc > 5 ? atoi(argv[5]) : 0);
    if (!strcmp(cmd, "pc"))
        return do_probe_ch(argc > 2 ? atoi(argv[2]) : 10,
                           argc > 3 ? (int)strtol(argv[3], 0, 0) : 0x7C,
                           argc > 4 ? (int)strtol(argv[4], 0, 0) : 1023);
    if (!strcmp(cmd, "check16")) return do_check16();
    if (!strcmp(cmd, "diag")) return do_diag(argc > 2 ? atoi(argv[2]) : 0);
    if (!strcmp(cmd, "diag5"))
        return do_diag2(2, argc > 2 ? atoi(argv[2]) : 6, 0, 3);
    if (!strcmp(cmd, "diag4")) {
        g_act_layout = argc > 2 ? atoi(argv[2]) : 0;
        return do_diag2(0, argc > 3 ? atoi(argv[3]) : 6, 0, 3);
    }
    if (!strcmp(cmd, "diag3"))
        return do_diag2(1, argc > 2 ? atoi(argv[2]) : 6, argc > 3 ? atoi(argv[3]) : 0,
                        argc > 4 ? atoi(argv[4]) : 3);
    if (!strcmp(cmd, "diag2"))
        return do_diag2(argc > 5 ? atoi(argv[5]) : 0, argc > 2 ? atoi(argv[2]) : 3,
                        argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? atoi(argv[4]) : 3);
    if (!strcmp(cmd, "probe2")) {
        int r[8];
        int rc;
        rc = mx4_probe(8, r, 8);
        printf("step1 write: rc=%d status=%d base=0x%08x magic=0x%x\n",
               rc, r[0], (unsigned)r[1], (unsigned)r[5]);
        if (rc || r[0]) return 1;
        sleep(1);
        rc = mx4_probe(16, r, 8);
        printf("step2 read : rc=%d status=%d sum=%d edge=0x%x magic=0x%x\n",
               rc, r[0], r[4], (unsigned)r[3], (unsigned)r[5]);
        return 0;
    }
    if (!strcmp(cmd, "probe")) {
        int r[8];
        int rc = mx4_probe(argc > 2 ? atoi(argv[2]) : 0, r, 8);
        printf("probe mode=%d: rc=%d status=%d base=0x%08x size=%u edges=0x%x "
               "sum=%d expect=%d pwr_rc=%d hmx_clk=%d\n",
               argc > 2 ? atoi(argv[2]) : 0, rc, r[0], (unsigned)r[1],
               (unsigned)r[2], (unsigned)r[3], r[4], r[5], r[6], r[7]);
        return 0;
    }
    if (!strcmp(cmd, "tile"))
        return do_tile(argc > 2 ? atoi(argv[2]) : 1,
                       argc > 3 ? (int)strtol(argv[3], 0, 0) : 0x7C,
                       argc > 4 ? (int)strtol(argv[4], 0, 0) : 1023,
                       argc > 5 ? (int)strtol(argv[5], 0, 0) : 0,
                       argc > 6 ? atoi(argv[6]) : 1,
                       argc > 7 ? atoi(argv[7]) : 3,
                       argc > 8 ? atoi(argv[8]) : 1);
    if (!strcmp(cmd, "gemm")) {
        int M = argc > 2 ? atoi(argv[2]) : 128;
        int N = argc > 3 ? atoi(argv[3]) : 64;
        int K = argc > 4 ? atoi(argv[4]) : 256;
        int reps = argc > 5 ? atoi(argv[5]) : 1;
        int mode = argc > 6 ? atoi(argv[6]) : 0;
        int amax = argc > 7 ? atoi(argv[7]) : 3;
        int do_ref = argc > 8 ? atoi(argv[8]) : 1;
        return do_gemm(M, N, K, reps, mode, amax, do_ref, 0);
    }
    if (!strcmp(cmd, "href")) {
        int N = argc > 2 ? atoi(argv[2]) : 32;
        int K = argc > 3 ? atoi(argv[3]) : 32;
        int mode = argc > 4 ? (int)strtol(argv[4], 0, 0) : 0;
        int reps = argc > 5 ? atoi(argv[5]) : 1;
        return do_href_one(N, K, mode, reps, 1) ? 1 : 0;
    }
    if (!strcmp(cmd, "hrefall")) {
        int mode = argc > 2 ? (int)strtol(argv[2], 0, 0) : 0;
        int reps = argc > 3 ? atoi(argv[3]) : 3;
        return do_hrefall(mode, reps, argc > 4 ? atoi(argv[4]) : 1);
    }
    if (!strcmp(cmd, "i8")) {
        int N = argc > 2 ? atoi(argv[2]) : 32;
        int K = argc > 3 ? atoi(argv[3]) : 32;
        int lay = argc > 4 ? atoi(argv[4]) : 3;
        int wl = argc > 5 ? atoi(argv[5]) : (N > 32 ? 4 : 3);
        int art = argc > 6 ? (int)strtol(argv[6], 0, 0) : 0x3FF;
        int wc = argc > 7 ? (int)strtol(argv[7], 0, 0) : 1023;
        int srt = argc > 8 ? atoi(argv[8]) : 0;
        int mul = argc > 9 ? atoi(argv[9]) : 0;
        int npair = argc > 10 ? atoi(argv[10]) : 0;
        int bs = argc > 11 ? (int)strtol(argv[11], 0, 0) : 0x6000;
        int amax = argc > 12 ? atoi(argv[12]) : 1;
        int mode = argc > 13 ? (int)strtol(argv[13], 0, 0) : 0;
        int reps = argc > 14 ? atoi(argv[14]) : 1;
        return do_i8(N, K, lay, wl, art, wc, srt, mul, npair, bs, amax, mode, reps);
    }
    if (!strcmp(cmd, "i8m")) {
        int M = argc > 2 ? atoi(argv[2]) : 64;
        int N = argc > 3 ? atoi(argv[3]) : 32;
        int K = argc > 4 ? atoi(argv[4]) : 2048;
        int bs = argc > 5 ? (int)strtol(argv[5], 0, 0) : 0x6000;
        int amax = argc > 6 ? atoi(argv[6]) : 0;
        int mode = argc > 7 ? (int)strtol(argv[7], 0, 0) : 0;
        int reps = argc > 8 ? atoi(argv[8]) : 100;
        int mul = argc > 9 ? atoi(argv[9]) : 4;
        int npair = argc > 10 ? atoi(argv[10]) : 0;
        return do_i8m(M, N, K, bs, amax, mode, reps, mul, npair);
    }
    if (!strcmp(cmd, "i8one"))
        return do_i8one(argc > 2 ? atoi(argv[2]) : 0,
                        argc > 3 ? atoi(argv[3]) : 0,
                        argc > 4 ? atoi(argv[4]) : 32,
                        argc > 5 ? atoi(argv[5]) : 32,
                        argc > 6 ? (int)strtol(argv[6], 0, 0) : 0x6000,
                        argc > 7 ? (int)strtol(argv[7], 0, 0) : 0,
                        argc > 8 ? atoi(argv[8]) : 20);
    if (!strcmp(cmd, "i8ramp"))
        return do_i8ramp(argc > 2 ? atoi(argv[2]) : 32,
                         argc > 3 ? atoi(argv[3]) : 32,
                         argc > 4 ? (int)strtol(argv[4], 0, 0) : 0x6000,
                         argc > 5 ? atoi(argv[5]) : 0,
                         argc > 6 ? atoi(argv[6]) : 0,
                         argc > 7 ? atoi(argv[7]) : 20);
    if (!strcmp(cmd, "pref"))
        return do_pref(argc > 2 ? atoi(argv[2]) : 128,
                       argc > 3 ? atoi(argv[3]) : 128,
                       argc > 4 ? atoi(argv[4]) : 64,
                       argc > 5 ? atoi(argv[5]) : 0,
                       argc > 6 ? atoi(argv[6]) : 1);
    if (!strcmp(cmd, "bw")) {
        int r8[8];
        int mb = argc > 2 ? atoi(argv[2]) : 16;
        int md = argc > 3 ? atoi(argv[3]) : 0;
        int rp = argc > 4 ? atoi(argv[4]) : 3;
        int rc = mx4_bw(mb, md, rp, r8, 8);
        printf("bw %d MB mode=%d reps=%d rc=%d st=%d us=%u bytes=%u pcyc=%u "
               "-> %.2f GB/s (align=%d)\n", mb, md, rp, rc, r8[0],
               (unsigned)r8[1], (unsigned)r8[2], (unsigned)r8[3],
               (double)r8[2] / (r8[1] ? r8[1] * 1e3 : 1.0), r8[5]);
        return 0;
    }
    if (!strcmp(cmd, "bw2"))
        return do_bw2(argc > 2 ? atoi(argv[2]) : 16,
                      argc > 3 ? atoi(argv[3]) : 0,
                      argc > 4 ? atoi(argv[4]) : 32,
                      argc > 5 ? atoi(argv[5]) : 3,
                      argc > 6 ? atoi(argv[6]) : 0);
    if (!strcmp(cmd, "bwt"))
        return do_bwt(argc > 2 ? atoi(argv[2]) : 8,
                      argc > 3 ? atoi(argv[3]) : 128,
                      argc > 4 ? atoi(argv[4]) : 1,
                      argc > 5 ? atoi(argv[5]) : 4);
    if (!strcmp(cmd, "i4map")) {
        unsigned char map[512];
        int info[8];
        int rc = mx4_i4map(argc > 2 ? atoi(argv[2]) : 0,
                           argc > 3 ? (int)strtol(argv[3], 0, 0) : 0x3FF,
                           argc > 4 ? (int)strtol(argv[4], 0, 0) : 511,
                           argc > 5 ? (int)strtol(argv[5], 0, 0) : 0x6000,
                           argc > 6 ? atoi(argv[6]) : 0,
                           argc > 7 ? atoi(argv[7]) : 0,
                           map, 512, info, 8);
        int b;
        printf("i4map rc=%d st=%d maxv=%d hits=%d nibhi=%d maxcols=%d k0=%d\n",
               rc, info[0], info[1], info[2], info[3], info[4], info[5]);
        for (b = 0; b < (info[3] == 0 && info[5] >= 0 && info[5] < 32 ? 64 : 512); b++)
            printf("%d %d\n", b, map[b] == 0xFF ? -1 : map[b]);
        return 0;
    }
    if (!strcmp(cmd, "l2t")) {
        int r[8];
        int kb = argc > 2 ? atoi(argv[2]) : 512;
        int rp = argc > 3 ? atoi(argv[3]) : 5;
        int n = (kb << 10) + (1 << 18);
        unsigned char *src = (unsigned char *)malloc(n);
        int rc;
        if (!src) return 1;
        memset(src, 0x33, n);
        rc = mx4_l2t(src, n, kb, rp, r, 8);
        printf("l2t %dKB reps=%d rc=%d st=%d bytes=%d | cold %.2f GB/s (%d us) "
               "warm %.2f GB/s (%d us) pref1 %.2f GB/s (%d us) pref8 %.2f GB/s "
               "(%d us)\n", kb, rp, rc, r[0], r[1],
               r[1] / (r[2] ? r[2] * 1e3 : 1.0), r[2],
               r[1] / (r[3] ? r[3] * 1e3 : 1.0), r[3],
               r[1] / (r[4] ? r[4] * 1e3 : 1.0), r[4],
               r[1] / (r[5] ? r[5] * 1e3 : 1.0), r[5]);
        free(src);
        return 0;
    }
    if (!strcmp(cmd, "i4"))
        return do_i4(argc > 2 ? atoi(argv[2]) : 32,
                     argc > 3 ? atoi(argv[3]) : 32,
                     argc > 4 ? atoi(argv[4]) : 32,
                     argc > 5 ? atoi(argv[5]) : 0,
                     argc > 6 ? (int)strtol(argv[6], 0, 0) : 0x6000,
                     argc > 7 ? atoi(argv[7]) : 1,
                     argc > 8 ? atoi(argv[8]) : 0,
                     argc > 9 ? atoi(argv[9]) : 100);
    if (!strcmp(cmd, "rpc")) {
        int it = argc > 2 ? atoi(argv[2]) : 31;
        int r[4], i, rc, sz;
        double s[256];
        int sizes[3] = {0, 1024, 1 << 20}, si;
        for (si = 0; si < 3; si++) {
            sz = sizes[si];
            unsigned char *buf = sz ? (unsigned char *)malloc(sz) : 0;
            if (buf) memset(buf, 7, sz);
            for (i = 0; i < 3; i++) {
                if (sz) rc = mx4_pingbuf(buf, sz, r, 4);
                else rc = mx4_ping0(1, r, 4);
            }
            for (i = 0; i < it && i < 256; i++) {
                struct timespec a, b;
                clock_gettime(CLOCK_MONOTONIC, &a);
                if (sz) rc = mx4_pingbuf(buf, sz, r, 4);
                else rc = mx4_ping0(1, r, 4);
                clock_gettime(CLOCK_MONOTONIC, &b);
                s[i] = (double)(b.tv_sec - a.tv_sec) * 1e6
                     + (double)(b.tv_nsec - a.tv_nsec) / 1e3;
            }
            {   /* insertion sort */
                int j;
                for (i = 1; i < it; i++) {
                    double v = s[i];
                    for (j = i - 1; j >= 0 && s[j] > v; j--) s[j + 1] = s[j];
                    s[j + 1] = v;
                }
            }
            printf("rpc %7s rc=%d  min %8.1f us  median %8.1f us  p95 %8.1f us\n",
                   sz >= (1 << 20) ? "1 MiB" : sz ? "1 KiB" : "empty", rc,
                   s[0], s[it / 2], s[(it * 95 / 100) < it ? it * 95 / 100 : it - 1]);
            free(buf);
        }
        return 0;
    }
    if (!strcmp(cmd, "clk")) {
        int r[4];
        int rc = mx4_clk(r, 4);
        printf("clk rc=%d pcycles=%u us=%u -> %d pcycles/us  (%.2f GHz)\n",
               rc, (unsigned)r[0], (unsigned)r[1], r[2], r[2] / 1000.0);
        return 0;
    }
    if (!strcmp(cmd, "i8sweep")) {
        int N = argc > 2 ? atoi(argv[2]) : 32;
        int K = argc > 3 ? atoi(argv[3]) : 32;
        int lay = argc > 4 ? atoi(argv[4]) : 0;
        int wl = argc > 5 ? atoi(argv[5]) : 0;
        int mode = argc > 6 ? (int)strtol(argv[6], 0, 0) : 0;
        return do_i8sweep(N, K, lay, wl, mode);
    }
    if (!strcmp(cmd, "refdump"))
        return do_refdump(argc > 2 ? atoi(argv[2]) : 32,
                          argc > 3 ? atoi(argv[3]) : 0);
    printf("usage: info | recipe | sweep | tile ncr ac wc srt variant ops | "
           "gemm M N K reps mode amax do_ref | href N K mode reps | "
           "hrefall mode reps | refdump K side\n");
    return 2;
}
