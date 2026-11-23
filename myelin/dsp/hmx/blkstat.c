/* blkstat — REPORT24: accuracy/cost study of per-block int4 scales.
 *
 * Dev-machine tool (system cc, no SDK, no device).  Reads a real Q4_K/Q6_K
 * GGUF tensor, builds our int4 weight format with a block scale over G*32 K
 * values (G = 1 is REPORT23's format: symmetric int4, scale = max|w|/7 per
 * 32 K), and compares on realistic uint8 activations:
 *
 *   f32     y = sum_k a[k] * w_f32[k]                     (quality reference)
 *   per-row int4 with one scale per row                    (production today)
 *   xactG   exact arithmetic with G-block scales
 *   modelG  bit-model of the HMX implementation of REPORT24:
 *             one accumulate per group, add offset C, store as unsigned
 *             16-bit with multiplier `mult` (saturating), combine on HVX.
 *           Its error is the implementation's, not the format's.
 *
 * Per G it prints
 *   cos(modelG, xactG)  - the acceptance criterion (>= 0.9995)
 *   cos(xactG, f32)     - format quality (REPORT23: 0.995341 at G=1)
 *   max|partial|        - sizes the offset C and the multiplier
 *   stores per matrix, partial bytes per matrix, combine MACs per matrix
 *
 *   ./blkstat <model.gguf> <tensor> [rows] [actrows]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../hvx/gguf_f32.h"

static uint32_t rng = 2463534242u;
static double urand(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((double)(rng & 0xFFFFFF) + 1.0) / 16777217.0;
}
static double gauss(void)
{
    return sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand());
}

typedef struct { double dot, na, nb, mx; long cells, over; } stat_t;

static void st_add(stat_t *s, double a, double b)
{
    double e = a - b;
    if (e < 0) e = -e;
    if (e > s->mx) s->mx = e;
    s->dot += a * b; s->na += a * a; s->nb += b * b; s->cells++;
}
static double st_cos(const stat_t *s)
{
    return (s->na > 0 && s->nb > 0) ? s->dot / sqrt(s->na * s->nb) : 0.0;
}

#define MAXG 6
#define MAXR 8

static int gs[MAXG] = { 1, 2, 4, 8, 16, 64 };

/* int4 quantisation of one K-row with one scale per G*32 values */
static void quant_row(const float *w, int K, int G, int8_t *q, float *sc)
{
    int nb = K / (32 * G), b, k;
    for (b = 0; b < nb; b++) {
        float amax = 0, s;
        for (k = 0; k < 32 * G; k++) {
            float a = fabsf(w[b * 32 * G + k]);
            if (a > amax) amax = a;
        }
        s = amax > 0 ? amax / 7.0f : 0.0f;
        sc[b] = s;
        for (k = 0; k < 32 * G; k++) {
            int v = s > 0 ? (int)lrintf(w[b * 32 * G + k] / s) : 0;
            if (v > 7) v = 7;
            if (v < -8) v = -8;
            q[b * 32 * G + k] = (int8_t)v;
        }
    }
}

int main(int argc, char **argv)
{
    const tinfo_t *t;
    int N, K, rows, arows = MAXR, i, k, n, r, g;
    float *W;
    unsigned char *A;
    stat_t s_f32_row = {0}, s_f32[MAXG], s_impl[MAXG], s_impl8[MAXG], s_impl8g[MAXG], s_v32[MAXG];
    double maxpart[MAXG], maxneg[MAXG], smaxg[MAXG];
    int8_t *qg[MAXG];
    float *scg[MAXG];
    int nbG[MAXG];
    int rowbytes;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor> [rows] [actrows]\n", argv[0]);
        return 2;
    }
    gguf_open(argv[1]);
    t = gguf_find(argv[2]);
    K = (int)t->dim[0];
    N = (int)t->dim[1];
    rows = argc > 3 ? atoi(argv[3]) : N;
    arows = argc > 4 ? atoi(argv[4]) : MAXR;
    if (rows > N) rows = N;
    if (arows > MAXR) arows = MAXR;
    if (K > 8192) { fprintf(stderr, "K=%d too big\n", K); return 1; }

    printf("tensor %s: N=%d K=%d  rows=%d activations=%d\n", argv[2], N, K, rows, arows);
    rowbytes = (int)(gguf_bytes(t) / N);

    W = (float *)malloc((size_t)K * sizeof(float));
    A = (unsigned char *)malloc((size_t)arows * K);
    if (!W || !A) return 1;

    for (r = 0; r < arows; r++) {
        float mx = 0, *row = (float *)malloc((size_t)K * sizeof(float));
        for (k = 0; k < K; k++) {
            double z = gauss();
            double y = z / (1.0 + exp(-z)) + 0.35;          /* SiLU >= -0.28 */
            row[k] = (float)y;
            if (fabs(y) > mx) mx = (float)fabs(y);
        }
        for (k = 0; k < K; k++) {
            int v = (int)lrint((double)row[k] / mx * 255.0);
            if (v > 255) v = 255;
            if (v < 0) v = 0;
            A[(size_t)r * K + k] = (unsigned char)v;
        }
        free(row);
    }

    memset(s_f32, 0, sizeof(s_f32));
    memset(s_impl, 0, sizeof(s_impl));
    memset(s_impl8, 0, sizeof(s_impl8));
    memset(s_impl8g, 0, sizeof(s_impl8g));
    memset(s_v32, 0, sizeof(s_v32));
    for (i = 0; i < MAXG; i++) {
        g = gs[i];
        nbG[i] = K / (32 * g);
        qg[i] = (int8_t *)malloc((size_t)K);
        scg[i] = (float *)malloc((size_t)nbG[i] * sizeof(float));
        maxpart[i] = 0; maxneg[i] = 0; smaxg[i] = 0;
    }

    for (int pass = 0; pass < 2; pass++) {
    for (n = 0; n < rows; n++) {
        int64_t off = g_data + (int64_t)t->off + (int64_t)n * rowbytes;
        float promax = 0, prosc;
        int8_t *proq = (int8_t *)malloc((size_t)K);

        gguf_row_f32(off, K, (int)t->type, W);

        for (i = 0; i < MAXG; i++) quant_row(W, K, gs[i], qg[i], scg[i]);
        if (pass == 0)
            for (i = 0; i < MAXG; i++) {
                int b3;
                for (b3 = 0; b3 < nbG[i]; b3++)
                    if (scg[i][b3] > smaxg[i]) smaxg[i] = scg[i][b3];
            }
        /* per-row (production) */
        for (k = 0; k < K; k++) { float a = fabsf(W[k]); if (a > promax) promax = a; }
        prosc = promax > 0 ? promax / 7.0f : 0.0f;
        for (k = 0; k < K; k++) {
            int v = prosc > 0 ? (int)lrintf(W[k] / prosc) : 0;
            if (v > 7) v = 7;
            if (v < -8) v = -8;
            proq[k] = (int8_t)v;
        }

        for (r = 0; r < arows; r++) {
            const unsigned char *a = A + (size_t)r * K;
            double yf = 0, yrow = 0, y32;
            double p32;
            /* f32 reference and per-row format */
            for (k = 0; k < K; k++) {
                yf += (double)a[k] * (double)W[k];
                yrow += (double)a[k] * (double)prosc * (double)proq[k];
            }
            st_add(&s_f32_row, yrow, yf);
            /* per-32 exact (also yields y32 used as the G=1 reference) */
            {
                double acc = 0;
                int b;
                for (b = 0; b < K / 32; b++) {
                    long p = 0;
                    for (k = 0; k < 32; k++) p += (long)a[b * 32 + k] * (long)qg[0][b * 32 + k];
                    acc += (double)scg[0][b] * (double)p;
                }
                y32 = acc;
                p32 = acc;
            }
            (void)p32;

            for (i = 0; i < MAXG; i++) {
                int b2, G = gs[i], W32 = 32 * G;
                double yex = 0;
                long *pp = (long *)malloc((size_t)nbG[i] * sizeof(long));
                for (b2 = 0; b2 < nbG[i]; b2++) {
                    long p = 0;
                    for (k = 0; k < W32; k++)
                        p += (long)a[b2 * W32 + k] * (long)qg[i][b2 * W32 + k];
                    pp[b2] = p;
                    if ((double)p > maxpart[i]) maxpart[i] = (double)p;
                    if ((double)p < maxneg[i]) maxneg[i] = (double)p;
                    yex += (double)scg[i][b2] * (double)p;
                }
                st_add(&s_f32[i], yex, yf);          /* format quality */
                st_add(&s_v32[i], yex, y32);         /* format vs per-32  */
                if (pass == 1) {
                    /* implementation model A: 16-bit store, offset C, mult 65535/(2C+1) */
                    double mp = (maxpart[i] > -maxneg[i]) ? maxpart[i] : -maxneg[i];
                    double C = mp + 1.0;
                    double mult = 65535.0 / (2.0 * mp + 1.0);
                    double ym = 0, ym8 = 0, ym8g = 0;
                    /* implementation model B: the exact mx4_blkpref recipe --
                     * C0 = 224*Aq (1.25x the sampled max), store multiplier
                     * Kc = 250/(2*C0*smax) applied per (n,group), truncating
                     * 8-bit store, +0.5 per group for the truncation bias */
                    double Cd = 224.0 * ceil(1.25 * mp / 224.0);
                    double Kc = smaxg[i] > 0 ? 250.0 / (2.0 * Cd * smaxg[i]) : 1.0;
                    if (Cd < mp + 1) Cd = 224.0 * ceil((mp + 1) / 224.0);
                    for (b2 = 0; b2 < nbG[i]; b2++) {
                        double st = floor(((double)pp[b2] + C) * mult + 0.5);
                        double st8 = floor(((double)pp[b2] + Cd) * (double)scg[i][b2] * Kc);
                        if (st < 0) st = 0;
                        if (st > 65535.0) st = 65535.0;
                        if (st8 < 0) st8 = 0;
                        if (st8 > 255.0) st8 = 255.0;
                        ym += (double)scg[i][b2] * (st / mult - C);
                        ym8 += (st8 + 0.5) / Kc - Cd * (double)scg[i][b2];
                        {   /* model C: store normalised per (n,group) --
                             * 8 bits of (P+C0) for every block, scale later */
                            double m8 = 255.0 / (2.0 * Cd);
                            double u8 = floor(((double)pp[b2] + Cd) * m8);
                            if (u8 < 0) u8 = 0;
                            if (u8 > 255.0) u8 = 255.0;
                            ym8g += (double)scg[i][b2]
                                  * ((u8 + 0.5) / m8 - Cd);
                        }
                    }
                    st_add(&s_impl[i], ym, yex);
                    st_add(&s_impl8[i], ym8, yex);
                    st_add(&s_impl8g[i], ym8g, yex);
                }
                free(pp);
            }
        }
        free(proq);
    }
    }

    printf("\n%-3s %-5s %-11s %-11s %-10s %-11s %-11s %-9s\n",
           "G", "B(K)", "cos16(exact)", "cos8(unifKc)", "cos8(perGrp)",
           "maxabs8", "cos xact/f32", "max|part|");
    for (i = 0; i < MAXG; i++)
        printf("%-3d %-5d %-11.7f %-13.7f %-11.7f %-10.2f %-11.7f %-9.0f\n",
               gs[i], gs[i] * 32, st_cos(&s_impl[i]), st_cos(&s_impl8[i]),
               st_cos(&s_impl8g[i]), s_impl8g[i].mx, st_cos(&s_f32[i]),
               maxpart[i]);
    printf("%-3s %-6s %-11.7f\n", "row", "-", st_cos(&s_f32_row));

    /* cost model for this matrix shape, M=128 */
    printf("\ncost model per matrix (M=128, N=%d, K=%d; %d tiles of 32 N):\n",
           N, K, 2 * (N / 32));
    printf("%-3s %-6s %-9s %-13s %-13s %-12s %s\n",
           "G", "B(K)", "stores", "store bytes", "combine reads", "combine MACs",
           "extra acc. instr");
    for (i = 0; i < MAXG; i++) {
        double tiles = 2.0 * (N / 32);               /* mt=2, nt=N/32 */
        double stores = tiles * nbG[i];
        double bytes = stores * 2048.0;
        double macs = 128.0 * (double)N * nbG[i];
        printf("%-3d %-6d %-9.0f %-13.1f %-13.1f %-12.0f +%.0f%%\n",
               gs[i], gs[i] * 32, stores, bytes / 1e6, bytes / 1e6, macs / 1e6,
               100.0 / gs[i]);
    }

    for (i = 0; i < MAXG; i++) { free(qg[i]); free(scg[i]); }
    free(W); free(A);
    return 0;
}
