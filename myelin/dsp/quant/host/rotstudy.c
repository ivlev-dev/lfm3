/*
 * rotstudy.c - quality of int4/32 + int8/32 with a Hadamard rotation.
 *
 *   cc -O2 -o rotstudy rotstudy.c -lm
 *   ./rotstudy <gguf> <tensor> <M> <N> <K> <actkind>
 *
 * Reads the real Q4_K tensor from the GGUF, dequantises to f32, rotates
 * weights and activations by the same orthogonal block transform *before*
 * the int4/32 / int8/32 quantisation, and compares:
 *
 *   truth  y = sum_k w_f32[k] * x_f32[k]
 *   w-only y = sum_k deq4(w_rot)[k] * x_rot[k]        (weight quant only)
 *   full   y = sum_g sw[g]*sa[g]*dot(qw,qa)           (weight + act int8)
 *
 * for rot = 0 / 32 / 128, plus the per-32 amax/rms outlier ratio.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../../hvx/gguf_f32.h"
#include "quant.h"

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

typedef struct { double dot, na, nb, maxerr, maxabs; } stat_t;

static void acc_stat(stat_t *s, double y, double ref)
{
    double e = fabs(y - ref);
    s->dot += y * ref;
    s->na += y * y;
    s->nb += ref * ref;
    if (e > s->maxerr) s->maxerr = e;
    if (fabs(ref) > s->maxabs) s->maxabs = fabs(ref);
}

static double outlier_ratio32(const float *v)
{
    int i;
    double sum = 0.0, mx = 0.0, rms;
    for (i = 0; i < 32; i++) {
        double a = fabs(v[i]);
        sum += (double)v[i] * v[i];
        if (a > mx) mx = a;
    }
    rms = sqrt(sum / 32.0);
    return rms > 1e-30 ? mx / rms : 0.0;
}

int main(int argc, char **argv)
{
    const char *gguf, *name;
    int M, N, K, kind;
    const tinfo_t *t;
    int64_t rowbytes;
    float *w, *wrot, *x, *xrot;
    int8_t *qx, *qw;
    float *sx, *sw;
    double *truth, *yw, *yf;
    int n, m, k, g, cfg;
    static const int ROTS[3] = { 0, 32, 128 };
    double oor_w[3] = {0}, oor_x[3] = {0};
    long long oor_wn[3] = {0}, oor_xn[3] = {0};

    if (argc < 7) {
        fprintf(stderr, "usage: %s <gguf> <tensor> <M> <N> <K> <actkind>\n", argv[0]);
        return 2;
    }
    gguf = argv[1]; name = argv[2];
    M = atoi(argv[3]); N = atoi(argv[4]); K = atoi(argv[5]); kind = atoi(argv[6]);
    if (K % 128 || M < 1) { fprintf(stderr, "K %% 128, M>=1\n"); return 2; }

    gguf_open(gguf);
    t = gguf_find(name);
    if ((int)t->dim[0] != K || (int)t->dim[1] < N) {
        fprintf(stderr, "tensor is %llu x %llu\n",
                (unsigned long long)t->dim[0], (unsigned long long)t->dim[1]);
        return 2;
    }
    rowbytes = gguf_bytes(t) / (int64_t)t->dim[1];

    w = malloc((size_t)K * 4);
    wrot = malloc((size_t)K * 4);
    x = malloc((size_t)M * K * 4);
    xrot = malloc((size_t)M * K * 4);
    qx = malloc((size_t)M * K);
    qw = malloc((size_t)K);
    sx = malloc((size_t)(M * K / 32) * 4);
    sw = malloc((size_t)(K / 32) * 4);
    truth = malloc((size_t)M * 8);
    yw = malloc((size_t)M * 8);
    yf = malloc((size_t)M * 8);
    if (!w || !wrot || !x || !xrot || !qx || !qw || !sx || !sw || !truth || !yw || !yf) {
        fprintf(stderr, "oom\n"); return 1;
    }
    gen_act(x, M, K, kind);

    printf("tensor %s: %d rows x K=%d, M=%d tokens, act kind=%d (0=SiLU, 1=uniform, 2=SiLU+0.35)\n",
           name, N, K, M, kind);
    printf("%-5s %-13s %-12s %-13s %-12s %-11s\n",
           "rot", "cos(w-only)", "maxerr(w)", "cos(full)", "maxerr(full)", "rel(full)");

    for (cfg = 0; cfg < 3; cfg++) {
        int rot = ROTS[cfg];
        stat_t sw_only = {0}, sf = {0};

        for (m = 0; m < M; m++) {
            memcpy(xrot + (size_t)m * K, x + (size_t)m * K, (size_t)K * 4);
            lfm_rot_blocks_f32(xrot + (size_t)m * K, K, rot);
            lfm_quant_i8_group32(xrot + (size_t)m * K, K, qx + (size_t)m * K,
                                 sx + (size_t)m * (K / 32));
            for (g = 0; g < K / 32; g++) {
                oor_x[cfg] += outlier_ratio32(xrot + (size_t)m * K + g * 32);
                oor_xn[cfg]++;
            }
        }

        for (n = 0; n < N; n++) {
            int64_t off = g_data + (int64_t)t->off + (int64_t)n * rowbytes;
            float *xr;
            gguf_row_f32(off, K, (int)t->type, w);
            memcpy(wrot, w, (size_t)K * 4);
            lfm_rot_blocks_f32(wrot, K, rot);
            lfm_pack_i4_row(wrot, K, NULL, sw, qw);
            for (g = 0; g < K / 32; g++) {
                oor_w[cfg] += outlier_ratio32(wrot + g * 32);
                oor_wn[cfg]++;
            }
            for (m = 0; m < M; m++) {
                double s = 0.0, swq = 0.0, sfq = 0.0;
                xr = xrot + (size_t)m * K;
                for (k = 0; k < K; k++) {
                    int32_t qwk = qw[k];
                    int gk = k / 32;
                    s += (double)w[k] * (double)x[(size_t)m * K + k];
                    swq += (double)qwk * (double)sw[gk] * (double)xr[k];
                    sfq += (double)(qwk * (int32_t)qx[(size_t)m * K + k])
                         * (double)sw[gk] * (double)sx[(size_t)m * (K / 32) + gk];
                }
                truth[m] = s;
                yw[m] = swq;
                yf[m] = sfq;
                acc_stat(&sw_only, yw[m], truth[m]);
                acc_stat(&sf, yf[m], truth[m]);
            }
        }
        {
            double cw = (sw_only.na > 0 && sw_only.nb > 0)
                        ? sw_only.dot / sqrt(sw_only.na * sw_only.nb) : 0.0;
            double cf = (sf.na > 0 && sf.nb > 0) ? sf.dot / sqrt(sf.na * sf.nb) : 0.0;
            printf("%-5d %-13.7f %-12.4g %-13.7f %-12.4g %-11.3g\n",
                   rot, cw, sw_only.maxerr, cf, sf.maxerr,
                   sf.maxabs > 0 ? sf.maxerr / sf.maxabs : 0.0);
        }
    }
    printf("\nmean per-32 amax/rms (outlier ratio):\n");
    for (cfg = 0; cfg < 3; cfg++)
        printf("  rot=%-3d weights %.3f   activations %.3f\n", ROTS[cfg],
               oor_wn[cfg] ? oor_w[cfg] / oor_wn[cfg] : 0.0,
               oor_xn[cfg] ? oor_x[cfg] / oor_xn[cfg] : 0.0);
    return 0;
}
