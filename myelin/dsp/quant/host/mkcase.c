/*
 * mkcase.c - host tool for the REPORT25 HMX zero-point + rotation study.
 *
 *   cc -O2 -I<repo>/myelin/dsp/quant -o mkcase mkcase.c -lm
 *
 *   ./mkcase gen   <gguf> <tensor> <M> <N> <K> <rot> <kind> <case.bin>
 *   ./mkcase check <gguf> <tensor> <case.bin> <hmx_fixed.bin> <hmx_raw.bin>
 *
 * Weights: Q4_K from the GGUF -> f32 -> optional Hadamard rotation along K
 * -> int4/32 (scale = amax/7, half-away-from-zero, q in [-7,7]) -> HMX
 * `weight.n` tiles (4-bit two's complement, REPORT21 layout).  The weight
 * scales are f32 per (row, group).
 *
 * Activations: f32 -> same rotation -> per-group-32 absmax int8 with the
 * canonical OpenCL rule (quant.h), f32 scale.
 *
 * Case file:
 *   "LFMQ25\0" | u32 ver,M,N,K,rot | i8 qa[M*K] | f32 sa[M*K/32]
 *              | f32 sw[N*K/32] | f32 x[M*K]
 *              | i32 tbias[N*K/32] | u8 tshift[N*K/32] | u8 blob[]
 *
 * tbias/tshift are the per-(row,group) u8 readout calibration:
 *   out_u8 = (dot + tbias) >> tshift,  chosen to use the full 0..255 range
 *   for the observed M tokens (power-of-two scale, so the HMX conversion is
 *   exact truncation).  Production calibrates the same way.
 *
 * blob tile t (= 32 out rows), group g (= 32 K):
 *   byte = t*(K/32)*512 + g*512 + 128*((kk&31)>>3) + 4*n + (kk&3)
 *   nibble = q & 0xF, high iff ((kk>>2)&1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../../hvx/gguf_f32.h"
#include "quant.h"

#define MAGIC "LFMQ25"

static uint32_t g_rng = 987654321u;
static uint32_t rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

static void gen_act(float *x, int M, int K, int kind)
{
    int s, k;
    for (s = 0; s < M; s++) {
        if (kind == 1) {
            for (k = 0; k < K; k++)
                x[(size_t)s * K + k] = (float)((int)(rnd() % 2000u) - 1000) / 1000.0f;
            continue;
        }
        for (k = 0; k < K; k++) {
            double u1 = ((double)(rnd() % 100000u) + 1.0) / 100001.0;
            double u2 = ((double)(rnd() % 100000u) + 1.0) / 100001.0;
            double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
            double y = z / (1.0 + exp(-z));
            if (kind == 2) y += 0.35;
            x[(size_t)s * K + k] = (float)y;
        }
    }
}

static void put_nibble(uint8_t *tile, int kk, int n, int q)
{
    unsigned off = 128u * (unsigned)((kk & 31) >> 3)
                 + 4u * (unsigned)n + (unsigned)(kk & 3);
    unsigned sh = ((kk >> 2) & 1) ? 4u : 0u;
    tile[off] = (uint8_t)((tile[off] & ~(0xFu << sh)) | ((unsigned)(q & 0xF) << sh));
}

static void deq_row(int64_t off, int K, float *out)
{
    int nb = K / 256, i;
    uint8_t *buf = malloc((size_t)nb * 144);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    fseek(gf, (long)off, SEEK_SET);
    gguf_grd(buf, (size_t)nb * 144);
    for (i = 0; i < nb; i++) gguf_deq_q4k(buf + (size_t)i * 144, out + i * 256);
    free(buf);
}

static int cmd_gen(int argc, char **argv)
{
    const char *gguf, *name, *out;
    int M, N, K, rot, kind, m, n, g, l, clamped, G;
    const tinfo_t *t;
    float *x, *xr, *sa, *sw, *w;
    int8_t *qa, *q4;
    int32_t *tbias;
    uint8_t *tshift;
    uint8_t *blob;
    int64_t rowbytes;
    FILE *f;
    uint32_t hdr[5];

    if (argc < 10) {
        fprintf(stderr, "gen: <gguf> <tensor> <M> <N> <K> <rot> <kind> <out>\n");
        return 2;
    }
    gguf = argv[2]; name = argv[3];
    M = atoi(argv[4]); N = atoi(argv[5]); K = atoi(argv[6]);
    rot = atoi(argv[7]); kind = atoi(argv[8]); out = argv[9];
    if (K % 128 || N % 32 || M < 1 || M > 32 || (rot != 0 && rot != 32 && rot != 128)) {
        fprintf(stderr, "need K%%128==0, N%%32==0, 1<=M<=32, rot in {0,32,128}\n");
        return 2;
    }
    gguf_open(gguf);
    t = gguf_find(name);
    if ((int)t->dim[0] != K || (int)t->dim[1] < N) {
        fprintf(stderr, "tensor %s is %llu x %llu, wanted K=%d N=%d\n", name,
                (unsigned long long)t->dim[0], (unsigned long long)t->dim[1], K, N);
        return 2;
    }
    rowbytes = gguf_bytes(t) / (int64_t)t->dim[1];
    G = K / 32;
    printf("tensor %s: %llu x %llu type=%u row=%lld B, rot=%d\n", name,
           (unsigned long long)t->dim[0], (unsigned long long)t->dim[1],
           t->type, (long long)rowbytes, rot);

    x   = malloc((size_t)M * K * 4);
    xr  = malloc((size_t)M * K * 4);
    qa  = malloc((size_t)M * K);
    q4  = malloc((size_t)K);
    sa  = malloc((size_t)(M * K / 32) * 4);
    sw  = malloc((size_t)(N * K / 32) * 4);
    w   = malloc((size_t)K * 4);
    tbias = malloc((size_t)(N * K / 32) * 4);
    tshift = malloc((size_t)(N * K / 32));
    blob = calloc(1, (size_t)(N / 32) * (K / 32) * 512u);
    if (!x || !xr || !qa || !q4 || !sa || !sw || !w || !tbias || !tshift || !blob) {
        fprintf(stderr, "oom\n"); return 1;
    }

    gen_act(x, M, K, kind);
    clamped = 0;
    for (m = 0; m < M; m++) {
        memcpy(xr + (size_t)m * K, x + (size_t)m * K, (size_t)K * 4);
        lfm_rot_blocks_f32(xr + (size_t)m * K, K, rot);
        clamped += lfm_quant_i8_group32(xr + (size_t)m * K, K,
                                        qa + (size_t)m * K, sa + (size_t)m * (K / 32));
    }
    {
        int nz = 0, neg = 0, mn = 0, mx = 0;
        long long i;
        for (i = 0; i < (long long)M * K; i++) {
            if (qa[i]) nz++;
            if (qa[i] < 0) neg++;
            if (qa[i] < mn) mn = qa[i];
            if (qa[i] > mx) mx = qa[i];
        }
        printf("act: rot=%d kind=%d M=%d K=%d nonzero=%d/%d neg=%d min=%d max=%d clamps=%d\n",
               rot, kind, M, K, nz, M * K, neg, mn, mx, clamped);
    }

    for (n = 0; n < N; n++) {
        deq_row(g_data + (int64_t)t->off + (int64_t)n * rowbytes, K, w);
        lfm_rot_blocks_f32(w, K, rot);
        lfm_pack_i4_row(w, K, NULL, sw + (size_t)n * (K / 32), q4);
        for (g = 0; g < K / 32; g++) {
            uint8_t *tile = blob + ((size_t)(n / 32) * (K / 32) + g) * 512u;
            int mn = 0x7fffffff, mx = -0x7fffffff, k, m2;
            for (l = 0; l < 32; l++)
                put_nibble(tile, l, n & 31, (int)q4[g * 32 + l]);
            for (m2 = 0; m2 < M; m2++) {
                int32_t d = 0;
                for (l = 0; l < 32; l++)
                    d += (int32_t)qa[(size_t)m2 * K + g * 32 + l] * q4[g * 32 + l];
                if (d < mn) mn = d;
                if (d > mx) mx = d;
            }
            k = 0;
            while (((mx - mn) >> k) > 255) k++;
            tbias[(size_t)n * G + g] = -mn;
            tshift[(size_t)n * G + g] = (uint8_t)k;
        }
    }

    {
        char mf[1024];
        FILE *mfp;
        snprintf(mf, sizeof(mf), "%s.meta", out);
        mfp = fopen(mf, "w");
        if (mfp) {
            fprintf(mfp, "format=lfm-int4-32\nrotation=%d\nsign0=0x%08x\n"
                         "scale=%.9g\ngroup=32\nk=%d\nn=%d\nm=%d\n",
                    rot, rot ? lfm_rot_sign_tab[0] : 0u,
                    rot ? (rot == 32 ? LFM_ROT_S32 : LFM_ROT_S128) : 1.0f,
                    K, N, M);
            fclose(mfp);
            printf("wrote %s (rotation metadata)\n", mf);
        }
    }
    f = fopen(out, "wb");
    if (!f) { perror(out); return 1; }
    hdr[0] = 1; hdr[1] = (uint32_t)M; hdr[2] = (uint32_t)N;
    hdr[3] = (uint32_t)K; hdr[4] = (uint32_t)rot;
    fwrite(MAGIC, 1, 8, f);
    fwrite(hdr, 4, 5, f);
    fwrite(qa, 1, (size_t)M * K, f);
    fwrite(sa, 4, (size_t)M * K / 32, f);
    fwrite(sw, 4, (size_t)N * K / 32, f);
    fwrite(x, 4, (size_t)M * K, f);
    fwrite(tbias, 4, (size_t)N * G, f);
    fwrite(tshift, 1, (size_t)N * G, f);
    fwrite(blob, 1, (size_t)(N / 32) * (K / 32) * 512u, f);
    fclose(f);
    printf("wrote %s: qa %d + sa %d + sw %d + x %d + blob %d B\n", out,
           M * K, (int)(M * K / 32 * 4), (int)(N * K / 32 * 4), M * K * 4,
           (N / 32) * (K / 32) * 512);
    return 0;
}

/* ------------------------------------------------------------------ */
#define HMX_B 28448          /* |per-group int4 dot| <= 32*127*7 */
#define HMX_SH  8

static int cmd_check(int argc, char **argv)
{
    const char *gguf, *name, *casef, *fixedf, *rawf;
    FILE *f, *ff, *fr;
    char magic[8];
    uint32_t hdr[5];
    int M, N, K, rot, G, tiles, m, n, g, l, c, i;
    int8_t *qa, *qw;
    float *sa, *sw, *x, *w;
    uint8_t *ubuf, *tshift;
    int32_t *tbias;
    float *wraw;
    double *out_truth, *out_ref, *out_fix, *out_raw;
    double dot_fix = 0, na_fix = 0, dot_ref = 0, na_ref = 0, dot_raw = 0, na_raw = 0;
    double nb = 0, maxerr_truth = 0, maxerr_raw_truth = 0, maxerr_readout = 0;
    long long cells = 0;
    int maxerr_u8 = 0, bad_u8 = 0, sat_u8 = 0;
    int maxerr_raw_u8 = 0, bad_raw_u8 = 0;
    const tinfo_t *t;
    int64_t rowbytes;

    if (argc < 7) { fprintf(stderr, "check: <gguf> <tensor> <case> <fixed> <raw>\n"); return 2; }
    gguf = argv[2]; name = argv[3]; casef = argv[4];
    fixedf = argv[5]; rawf = argv[6];

    f = fopen(casef, "rb");
    if (!f) { perror(casef); return 1; }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8)) {
        fprintf(stderr, "bad case magic\n"); return 1;
    }
    if (fread(hdr, 4, 5, f) != 5) return 1;
    M = (int)hdr[1]; N = (int)hdr[2]; K = (int)hdr[3]; rot = (int)hdr[4];
    G = K / 32; tiles = N / 32;
    qa = malloc((size_t)M * K);
    sa = malloc((size_t)M * G * 4);
    sw = malloc((size_t)N * G * 4);
    x  = malloc((size_t)M * K * 4);
    w  = malloc((size_t)K * 4);
    wraw = malloc((size_t)K * 4);
    qw = malloc((size_t)K);
    ubuf = malloc((size_t)G * 32 * 32);
    tbias = malloc((size_t)N * G * 4);
    tshift = malloc((size_t)N * G);
    if (!qa || !sa || !sw || !x || !w || !wraw || !qw || !ubuf || !tbias || !tshift) { fprintf(stderr, "oom\n"); return 1; }
    if (fread(qa, 1, (size_t)M * K, f) != (size_t)M * K) return 1;
    if (fread(sa, 4, (size_t)M * G, f) != (size_t)M * G) return 1;
    if (fread(sw, 4, (size_t)N * G, f) != (size_t)N * G) return 1;
    if (fread(x, 4, (size_t)M * K, f) != (size_t)M * K) return 1;
    if (fread(tbias, 4, (size_t)N * G, f) != (size_t)N * G) return 1;
    if (fread(tshift, 1, (size_t)N * G, f) != (size_t)N * G) return 1;
    fclose(f);
    printf("case: M=%d N=%d K=%d rot=%d groups=%d tiles=%d  hmx readout: "
           "u8, per-(row,group) power-of-two scale+offset\n",
           M, N, K, rot, G, tiles);

    gguf_open(gguf);
    t = gguf_find(name);
    rowbytes = gguf_bytes(t) / (int64_t)t->dim[1];

    out_truth = calloc((size_t)M * N, 8);
    out_ref   = calloc((size_t)M * N, 8);
    out_fix   = calloc((size_t)M * N, 8);
    out_raw   = calloc((size_t)M * N, 8);

    ff = fopen(fixedf, "rb");
    fr = fopen(rawf, "rb");
    if (!ff || !fr) { perror("dump"); return 1; }

    for (i = 0; i < tiles; i++) {
        if (fread(ubuf, 1, (size_t)G * 32 * 32, ff) != (size_t)G * 32 * 32) {
            fprintf(stderr, "short fixed dump\n"); return 1;
        }
        for (c = 0; c < 32; c++) {
            int row = i * 32 + c;
            deq_row(g_data + (int64_t)t->off + (int64_t)row * rowbytes, K, wraw);
            memcpy(w, wraw, (size_t)K * 4);
            lfm_rot_blocks_f32(w, K, rot);
            lfm_pack_i4_row(w, K, NULL, sw + (size_t)row * G, qw);
            for (m = 0; m < M; m++) {
                double s = 0.0;
                for (l = 0; l < K; l++)
                    s += (double)wraw[l] * (double)x[(size_t)m * K + l];
                out_truth[(size_t)m * N + row] = s;
            }
            for (g = 0; g < G; g++)
                for (m = 0; m < M; m++) {
                    int32_t refq = 0;
                    int got = ubuf[((size_t)g * 32 + m) * 32 + c];
                    int ref_u8, bsel = 0, shsel = 0;
                    double dot_est = 0.0;
                    for (l = 0; l < 32; l++)
                        refq += (int32_t)qa[(size_t)m * K + g * 32 + l] * qw[g * 32 + l];
                    {
                        int B = tbias[(size_t)row * G + g];
                        int sh = tshift[(size_t)row * G + g];
                        ref_u8 = (refq + B) >> sh;
                        if (ref_u8 < 0) ref_u8 = 0;
                        if (ref_u8 > 255) { ref_u8 = 255; sat_u8++; }
                        dot_est = (sh > 0)
                            ? (double)(((long long)got << sh) + (1LL << (sh - 1)) - B)
                            : (double)((long long)got - B);
                        bsel = B; shsel = sh;
                    }
                    {
                        int e = got - ref_u8;
                        if (e < 0) e = -e;
                        if (e > maxerr_u8) maxerr_u8 = e;
                        if (e) bad_u8++;
                    }
                    {
                        double e = fabs(dot_est - (double)refq);
                        if (e > maxerr_readout) maxerr_readout = e;
                        out_fix[(size_t)m * N + row] +=
                            (double)sa[(size_t)m * G + g] * (double)sw[(size_t)row * G + g]
                            * dot_est;
                    }
                    (void)bsel; (void)shsel;
                    out_ref[(size_t)m * N + row] +=
                        (double)sa[(size_t)m * G + g] * (double)sw[(size_t)row * G + g]
                        * (double)refq;
                    cells++;
                }
        }
    }
    fclose(ff);

    /* broken mode: exact CPU value of int8-as-u8, and the HMX u8 dump */
    for (i = 0; i < tiles; i++) {
        if (fread(ubuf, 1, (size_t)G * 32 * 32, fr) != (size_t)G * 32 * 32) {
            fprintf(stderr, "short raw dump\n"); return 1;
        }
        for (c = 0; c < 32; c++) {
            int row = i * 32 + c;
            deq_row(g_data + (int64_t)t->off + (int64_t)row * rowbytes, K, wraw);
            memcpy(w, wraw, (size_t)K * 4);
            lfm_rot_blocks_f32(w, K, rot);
            lfm_pack_i4_row(w, K, NULL, sw + (size_t)row * G, qw);
            for (g = 0; g < G; g++)
                for (m = 0; m < M; m++) {
                    int32_t rawq = 0;
                    int got = ubuf[((size_t)g * 32 + m) * 32 + c];
                    int raw_u8;
                    for (l = 0; l < 32; l++)
                        rawq += (int32_t)(uint8_t)(int8_t)qa[(size_t)m * K + g * 32 + l]
                                * qw[g * 32 + l];
                    {
                        int B = tbias[(size_t)row * G + g];
                        int sh = tshift[(size_t)row * G + g];
                        raw_u8 = (rawq + B) >> sh;
                        if (raw_u8 < 0) raw_u8 = 0;
                        if (raw_u8 > 255) raw_u8 = 255;
                    }
                    {
                        int e = got - raw_u8;
                        if (e < 0) e = -e;
                        if (e > maxerr_raw_u8) maxerr_raw_u8 = e;
                        if (e) bad_raw_u8++;
                    }
                    out_raw[(size_t)m * N + row] +=
                        (double)sa[(size_t)m * G + g] * (double)sw[(size_t)row * G + g]
                        * (double)rawq;
                }
        }
    }
    fclose(fr);

    for (i = 0; i < M * N; i++) {
        double a = out_fix[i], b = out_ref[i], r = out_raw[i], tval = out_truth[i];
        double e = fabs(a - tval), er = fabs(r - tval);
        if (e > maxerr_truth) maxerr_truth = e;
        if (er > maxerr_raw_truth) maxerr_raw_truth = er;
        dot_fix += a * tval; na_fix += a * a;
        dot_ref += b * tval; na_ref += b * b;
        dot_raw += r * tval; na_raw += r * r;
        nb += tval * tval;
    }
    printf("u8 plane (u8+zp vs signed CPU): cells=%lld maxerr=%d bad=%d\n",
           cells, maxerr_u8, bad_u8);
    printf("u8 plane (raw int8-as-u8):      maxerr=%d bad=%d\n",
           maxerr_raw_u8, bad_raw_u8);
    printf("exact-dot reference cos(truth)=%.7f\n",
           (na_ref > 0 && nb > 0) ? dot_ref / sqrt(na_ref * nb) : 0.0);
    printf("u8-readout cos(truth)=%.7f  maxerr=%.4g  (max ideal-dot deviation "
           "from the u8 readout=%.0f)\n",
           (na_fix > 0 && nb > 0) ? dot_fix / sqrt(na_fix * nb) : 0.0,
           maxerr_truth, maxerr_readout);
    printf("raw-int8-as-u8 cos(truth)=%.7f  maxerr=%.4g\n",
           (na_raw > 0 && nb > 0) ? dot_raw / sqrt(na_raw * nb) : 0.0,
           maxerr_raw_truth);
    printf("MACs checked: %lld\n", (long long)M * N * K);
    if (maxerr_u8 || bad_u8) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: OK\n");
    return 0;
}


int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s gen|check ...\n", argv[0]); return 2; }
    if (!strcmp(argv[1], "gen")) return cmd_gen(argc, argv);
    if (!strcmp(argv[1], "check")) return cmd_check(argc, argv);
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}
