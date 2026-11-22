/*
 * Paged attention test: CPU f32 reference, correctness (cos / max_abs_err),
 * timings for ctx 512/2048 and 1/2 HVX contexts, plus an offline accuracy
 * check of the two integer exp variants.
 */
#include "lfmgemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

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
    um.domain = CDSP_DOMAIN_ID; um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

#define HQ   16
#define KVH  4
#define D    128
#define PAGE 32

/* ---- the two integer exp variants, mirroring the DSP code ---- */
static int exp_bit(int32_t xq)
{
    int32_t y, n, f, p;
    if (xq < -16 * 65536) return 0;
    if (xq > 0) xq = 0;
    y = (xq * 94548) >> 16;
    n = y >> 16;
    f = (y - (n << 16)) >> 2;
    p = 3637;
    p = ((f * p) >> 14) + 15742;
    p = ((f * p) >> 14) + 45426;
    p = ((f * p) >> 14) + 65536;
    n = -n;
    if (n >= 15) return 0;
    return (int32_t)((p >> n) >> 8);
}
static const int16_t exptab[33] = {
    255, 155, 94, 57, 35, 21, 13, 8, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static int exp_tab(int32_t xq)
{
    int32_t y, n, f, r;
    if (xq >= 0) return 255;
    if (xq < -16 * 65536) return 0;
    y = ((xq >> 16) * 94548) + ((((uint32_t)xq >> 8) & 0xFFu) * (uint32_t)94548 >> 8);
    n = y >> 16;
    f = (y >> 8) & 0xff;
    r = (int32_t)(256.0 * pow(2.0, f / 256.0) + 0.5);
    n = -n;
    if (n >= 9) return 0;
    r >>= n;
    return r > 255 ? 255 : r;
}

int main(int argc, char **argv)
{
    int pages = (argc > 1) ? atoi(argv[1]) : 64;      /* 64 pages = 2048 tokens */
    int win = (argc > 2) ? atoi(argv[2]) : 0;          /* 0 = full */
    int threads = (argc > 3) ? atoi(argv[3]) : 2;
    int slots = pages * PAGE;
    int pos = slots - 1, seq = slots;
    int hk = -1, hkt = -1, hv = -1, rc, i, qh, t, k, r;
    int times[8], info[8];
    size_t kbytes = (size_t)KVH * slots * 128;         /* int4 packed + 64B pad */
    size_t ktbytes = (size_t)KVH * slots * 4;          /* Q16 per-token scales */
    size_t vbytes = (size_t)KVH * slots * D;           /* int8 */
    uint8_t *K, *V, *Kup, *Vup;

    int tm = getenv("ATTN_HG") ? 0 : 1;
    int32_t *KT;
    int8_t *q8;
    int *tbl;
    int32_t *out32;
    uint32_t st = 0x1234567u;
    int tok0, n;

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("attn test: pages=%d slots=%d pos=%d window=%s threads=%d\n",
           pages, slots, pos, win ? "sliding" : "full", threads);

    /* ---------- exp accuracy (host, offline) ---------- */
    {
        double eb = 0, ep = 0;
        double worstb = 0, worstp = 0;
        for (i = 0; i <= 100000; i++) {
            double x = -16.0 + 16.0 * i / 100000.0;
            int32_t xq = (int32_t)llround(x * 65536.0);
            double ref = exp(x);
            double vb = exp_bit(xq) / 255.0;
            double vp = exp_tab(xq) / 255.0;
            double reb = fabs(vb - ref) / (ref + 1e-12);
            double rep = fabs(vp - ref) / (ref + 1e-12);
            if (ref > 1e-4) {           /* only where it matters */
                if (reb > eb) { eb = reb; worstb = x; }
                if (rep > ep) { ep = rep; worstp = x; }
            }
        }
        printf("exp bit-trick : max rel err %.3e at x=%.3f\n", eb, worstb);
        printf("exp table     : max rel err %.3e at x=%.3f\n", ep, worstp);
    }

    /* ---------- data ---------- */
    K  = malloc(kbytes);       /* row-major (reference) */
    Kup = malloc(kbytes);      /* layout for the selected kernel */
    KT = malloc(ktbytes);
    V  = malloc(vbytes);
    Vup = malloc(vbytes);
    q8 = malloc(HQ * D);
    tbl = malloc(pages * sizeof(int));
    out32 = malloc(HQ * D * sizeof(int32_t));

    for (i = 0; i < pages; i++) tbl[i] = i;            /* page id == page index */
    /* K: int4 offset-8 nibbles, random; V: int8 in [-2..1] */
    for (k = 0; k < KVH; k++) {
        for (t = 0; t < slots; t++) {
            uint8_t *row = K + ((size_t)k * slots + t) * 128;
            for (i = 64; i < 128; i++) row[i] = 0;
            for (i = 0; i < 64; i++) {
                int lo = (int)(lcg(&st) % 16);
                int hi = (int)(lcg(&st) % 16);
                row[i] = (uint8_t)(lo | (hi << 4));
            }
            KT[k * slots + t] = (int32_t)(65536.0 * (0.0009 + 0.0000005 * (t % 17)));
            {
                int8_t *vr = (int8_t *)(V + ((size_t)k * slots + t) * D);
                for (i = 0; i < D; i++) vr[i] = (int8_t)((int)(lcg(&st) % 4) - 2);
            }
        }
    }
    for (i = 0; i < HQ * D; i++) q8[i] = (int8_t)((int)(lcg(&st) % 255) - 127);

    /* token-major layout: [kv][page][dim-group g<16][128B], byte 4t+j holds
       dims 8g+2j and 8g+2j+1 of token t (offset-8 nibbles) */
    memset(Kup, 0, kbytes);
    for (k = 0; k < KVH; k++)
        for (i = 0; i < pages; i++)
            for (t = 0; t < 32; t++) {
                const uint8_t *rrow = K + ((size_t)k * slots + i * 32 + t) * 128;
                int g, j;
                for (g = 0; g < 16; g++)
                    for (j = 0; j < 4; j++) {
                        int d0 = 8 * g + 2 * j, d1 = d0 + 1;
                        uint8_t lo = (d0 & 1) ? (uint8_t)(rrow[d0 >> 1] >> 4) : (uint8_t)(rrow[d0 >> 1] & 0xf);
                        uint8_t hi = (uint8_t)(rrow[d1 >> 1] >> 4);
                        Kup[((size_t)k * pages + i) * 16 * 128 + (size_t)g * 128 + 4 * t + j] =
                            (uint8_t)((lo & 0xf) | ((hi & 0xf) << 4));
                    }
            }

    /* token-major V: per 32-token page, vector (g,c) = 32 dims x 4 tokens,
       byte 4*d'+j = V[32g+d'][32c+j] */
    for (k = 0; k < KVH; k++)
        for (i = 0; i < pages; i++)
            for (t = 0; t < 32; t++) {
                const int8_t *vrow = (const int8_t *)(V + ((size_t)k * slots + i * 32 + t) * D);
                int g2, j2, d2;
                for (d2 = 0; d2 < D; d2++) {
                    g2 = d2 >> 5;
                    j2 = t & 3;
                    Vup[((size_t)k * pages + i) * 4096 + (size_t)g2 * 1024
                        + (size_t)(t >> 2) * 128 + 4 * (d2 & 31) + j2] = (uint8_t)vrow[d2];
                }
            }

    /* verify the transposed V layout round-trips */
    if (tm) {
        int mism = 0, g;
        uint8_t *chkv = malloc(vbytes);
        memset(chkv, 0, vbytes);
        for (k = 0; k < KVH; k++)
            for (i = 0; i < pages; i++)
                for (t = 0; t < 32; t++)
                    for (g = 0; g < D; g++) {
                        int8_t val = (int8_t)Vup[((size_t)k * pages + i) * 4096
                                     + (size_t)(g >> 5) * 1024 + (size_t)(t >> 2) * 128
                                     + 4 * (g & 31) + (t & 3)];
                        chkv[((size_t)k * slots + i * 32 + t) * D + g] = (uint8_t)val;
                    }
        for (i = 0; i < (int)vbytes; i++) if (chkv[i] != V[i]) mism++;
        printf("V transpose round-trip: %d mismatching bytes of %d\n", mism, (int)vbytes);
        free(chkv);
    }

    /* verify the transposed layout round-trips */
    {
        int mism = 0;
        int g, j;
        uint8_t *chk = calloc(kbytes, 1);
        for (k = 0; k < KVH; k++)
            for (i = 0; i < pages; i++)
                for (t = 0; t < 32; t++)
                    for (g = 0; g < 16; g++)
                        for (j = 0; j < 4; j++) {
                            int d0 = 8 * g + 2 * j, d1 = d0 + 1;
                            uint8_t b = Kup[((size_t)k * pages + i) * 16 * 128 + (size_t)g * 128 + 4 * t + j];
                            uint8_t *dst = chk + ((size_t)k * slots + i * 32 + t) * 128;
                            dst[d0 >> 1] = (uint8_t)((dst[d0 >> 1] & (d0 & 1 ? 0x0f : 0xf0)) |
                                                     ((d0 & 1) ? (uint8_t)((b & 0xf) << 4) : (uint8_t)(b & 0xf)));
                            dst[d1 >> 1] = (uint8_t)((dst[d1 >> 1] & 0x0f) | (uint8_t)((b >> 4) << 4));
                        }
        for (i = 0; i < (int)kbytes; i++) if (chk[i] != K[i]) mism++;
        printf("transpose round-trip: %d mismatching bytes of %d\n", mism, (int)kbytes);
        free(chk);
    }

    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }
    if ((rc = lfmgemv_alloc_buf((int)kbytes, &hk)) != 0) { printf("alloc K rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_alloc_buf((int)ktbytes, &hkt)) != 0) { printf("alloc KT rc=%d\n", rc); return 11; }
    if ((rc = lfmgemv_alloc_buf((int)vbytes, &hv)) != 0) { printf("alloc V rc=%d\n", rc); return 11; }
    {
        size_t off = 0, chunk = 256 * 1024;
        while (off < kbytes) {
            size_t m = kbytes - off < chunk ? kbytes - off : chunk;
            if ((rc = lfmgemv_load_buf(hk, (int)off, (tm ? Kup : K) + off, (int)m)) != 0) { printf("load K rc=%d\n", rc); return 12; }
            off += m;
        }
        off = 0;
        while (off < vbytes) {
            size_t m = vbytes - off < chunk ? vbytes - off : chunk;
            if ((rc = lfmgemv_load_buf(hv, (int)off, (tm ? Vup : V) + off, (int)m)) != 0) { printf("load V rc=%d\n", rc); return 12; }
            off += m;
        }
        if ((rc = lfmgemv_load_buf(hkt, 0, (uint8_t *)KT, (int)ktbytes)) != 0) { printf("load KT rc=%d\n", rc); return 12; }
    }
    /* correctness pass always single-threaded; thread count applies to timing */
    lfmgemv_probe(-1, 8, 8 | (2 << 8) | (1 << 16) | (1 << 24), times, 8);
    lfmgemv_probe(-1, 27, tm, times, 8);
    printf("kernel: %s, K layout: %s\n", tm ? "token-major" : "head-grouped",
           tm ? "transposed" : "row-major");

    tok0 = (win > 0 && seq - win > 0) ? seq - win : 0;
    n = seq - tok0;

    /* ---------- CPU reference (f32) ---------- */
    {
        float *out_ref = malloc(HQ * D * sizeof(float));
        int tc;
        for (qh = 0; qh < HQ; qh++) {
            int kh = qh >> 2;
            float *sc = malloc((size_t)n * sizeof(float));
            float mx = -1e30f, lsum = 0;
            double *accf = calloc(D, sizeof(double));
            for (tc = 0; tc < n; tc++) {
                int tok = tok0 + tc;
                int slot = (tbl[tok >> 5] << 5) | (tok & 31);
                const uint8_t *krow = K + ((size_t)kh * slots + slot) * 128;
                int32_t dot = 0;
                for (k = 0; k < 64; k++) {
                    int lo = (int)(krow[k] & 0xf) - 8;
                    int hi = (int)(krow[k] >> 4) - 8;
                    dot += (int32_t)q8[qh * D + 2 * k] * lo;
                    dot += (int32_t)q8[qh * D + 2 * k + 1] * hi;
                }
                sc[tc] = (float)dot * (float)KT[kh * slots + slot] / 65536.0f;
                if (sc[tc] > mx) mx = sc[tc];
            }
            for (tc = 0; tc < n; tc++) {
                int tok = tok0 + tc;
                int slot = (tbl[tok >> 5] << 5) | (tok & 31);
                const int8_t *vr = (const int8_t *)(V + ((size_t)kh * slots + slot) * D);
                float p = expf(sc[tc] - mx);
                lsum += p;
                for (i = 0; i < D; i++) accf[i] += (double)p * (double)vr[i];
            }
            for (i = 0; i < D; i++) out_ref[qh * D + i] = (float)(accf[i] / (lsum > 0 ? lsum : 1));
            free(sc); free(accf);
        }
        /* run on DSP and compare */
        rc = lfmgemv_attn(hk, hkt, hv, 0, (const unsigned char *)q8, HQ * D,
                          tbl, pages, pos, seq, win, 1, (unsigned char *)out32, HQ * D * 4, times, 8);
        if (rc != 0) { printf("attn rc=%d\n", rc); return 13; }
        {
            int dbg[8], z, hd2;
            {
                int hh2;
                for (hh2 = 0; hh2 < 2; hh2++) {
                    int d30[8], y;
                    int hd3 = hh2 * 4;    /* head 0 and head 4 */
                    if (lfmgemv_probe(hk, 30, hd3, d30, 8) != 0) continue;
                    printf("uq head%d dsp:", hd3);
                    for (y = 0; y < 8; y++) printf(" %08x", d30[y]);
                    printf("\n");
                    {
                        uint32_t exp8[8];
                        int gg, j2;
                        for (gg = 0; gg < 4; gg++) {
                            uint32_t lo2 = 0, hi2 = 0;
                            for (j2 = 0; j2 < 4; j2++) {
                                lo2 |= (uint32_t)(uint8_t)q8[hd3 * D + 8 * gg + 2 * j2] << (8 * j2);
                                hi2 |= (uint32_t)(uint8_t)q8[hd3 * D + 8 * gg + 2 * j2 + 1] << (8 * j2);
                            }
                            exp8[2 * gg] = lo2; exp8[2 * gg + 1] = hi2;
                        }
                        printf("uq head%d cpu:", hd3);
                        for (y = 0; y < 8; y++) printf(" %08x", exp8[y]);
                        printf("\n");
                    }
                }
            }
            for (hd2 = 0; hd2 < 4; hd2++) {
                printf("dbg head%d dsp [0..7]: ", hd2);
                if (lfmgemv_probe(hk, 29, hd2 * 4096 + 0, dbg, 8) != 0) { printf("probe fail\n"); continue; }
                for (z = 0; z < 8; z++) printf("%6d ", dbg[z]);
                printf("\n");
                printf("dbg head%d cpu [0..7]: ", hd2);
                for (z = 0; z < 8; z++) {
                    int tok = tok0 + z;
                    int slot = (tbl[tok >> 5] << 5) | (tok & 31);
                    const uint8_t *krow = K + ((size_t)(hd2 >> 2) * slots + slot) * 128;
                    int32_t dt = 0;
                    for (k = 0; k < 64; k++) {
                        int lo2 = (int)(krow[k] & 0xf) - 8;
                        int hi2 = (int)(krow[k] >> 4) - 8;
                        dt += (int32_t)q8[hd2 * D + 2 * k] * lo2;
                        dt += (int32_t)q8[hd2 * D + 2 * k + 1] * hi2;
                    }
                    printf("%6d ", (int)((float)dt * (float)KT[(hd2 >> 2) * slots + slot] / 65536.0f * 256.0f));
                }
                printf("\n");
            }
        }

        {
            double dot = 0, na = 0, nb = 0, mx = 0;
            for (qh = 0; qh < HQ; qh++) {
                for (i = 0; i < D; i++) {
                    double a = out32[qh * D + i] / 4096.0;
                    double b = out_ref[qh * D + i];
                    dot += a * b; na += a * a; nb += b * b;
                    if (fabs(a - b) > mx) mx = fabs(a - b);
                }
            }
            printf("correctness: cos=%.6f  max_abs_err=%.4f  (tokens=%d)\n",
                   dot / sqrt(na * nb + 1e-30), mx, n);
            {
                double c = dot / sqrt(na * nb + 1e-30);
                double need = (n <= 256) ? 0.999 : (n <= 512 ? 0.999 : 0.998);
                printf("VERDICT: cos %.6f vs threshold %.3f -> %s\n",
                       c, need, (c >= need) ? "PASS" : "FAIL");
            }
            {
                int hh;
                printf("per-head cos:");
                for (hh = 0; hh < HQ; hh++) {
                    double d2 = 0, a2 = 0, b2 = 0;
                    for (i = 0; i < D; i++) {
                        double a = out32[hh * D + i] / 4096.0, b = out_ref[hh * D + i];
                        d2 += a * b; a2 += a * a; b2 += b * b;
                    }
                    printf(" %d:%.3f", hh, d2 / sqrt(a2 * b2 + 1e-30));
                }
                printf("\n");
            }
        }
        free(out_ref);
    }

    /* ---------- timing ---------- */
    lfmgemv_probe(-1, 8, 8 | (2 << 8) | (1 << 16) | (threads << 24), times, 8);
    {
        int reps = 3;
        double wmin = 1e18, kmin = 1e18;
        for (r = 0; r < 8; r++) {
            double t0 = now_us();
            rc = lfmgemv_attn(hk, hkt, hv, 0, (const unsigned char *)q8, HQ * D,
                              tbl, pages, pos, seq, win, reps, (unsigned char *)out32,
                              HQ * D * 4, times, 8);
            { double w = now_us() - t0; if (w < wmin) wmin = w; }
            if (rc != 0) { printf("attn timing rc=%d\n", rc); break; }
            if (times[0] < kmin) kmin = times[0];
            usleep(20000);
        }
        printf("timing: kernel=%.0f us  wall=%.0f us | prep=%d heads=%d norm=%d us\n",
               kmin, wmin, times[1], times[2], times[3]);
        printf("        phase: QK^T=%d us  exp+PV=%d us  (tokens=%d)\n",
               times[5], times[6], times[4]);
        printf("        %.0f us / 1k tokens\n", 1000.0 * kmin / n);
    }
    lfmgemv_team_info(info, 8);
    printf("team: created=%d hvx_units=%d\n", info[3], info[6]);
    if (argc > 4 && strcmp(argv[4], "pwl") == 0) {
        /* second run with the piecewise-linear exp */
        lfmgemv_probe(-1, 26, 1, times, 8);
        rc = lfmgemv_attn(hk, hkt, hv, 0, (const unsigned char *)q8, HQ * D,
                          tbl, pages, pos, seq, win, 3, (unsigned char *)out32,
                          HQ * D * 4, times, 8);
        printf("pwl exp: kernel=%d us (tokens=%d)\n", times[0], times[4]);
    }
    lfmgemv_release_buf(hk); lfmgemv_release_buf(hkt); lfmgemv_release_buf(hv);
    printf("done\n");
    return 0;
}
