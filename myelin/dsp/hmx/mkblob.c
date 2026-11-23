/* mkblob — вырезать FFN-тензор из GGUF, переквантовать и упаковать под стенд HMX.
 *
 *   ./mkblob <model.gguf> <tensor> <out-prefix>
 *
 * Пример:
 *   ./mkblob ../../../models/LFM3-1.7B-Q4_K_M.gguf blk.0.ffn_gate.weight /tmp/opencode/gate
 *
 * На выходе:
 *   <out>.i4   — int4, блочный масштаб на 32 K, раскладка стенда:
 *                byte(t,k,n) = t*(K/32)*512 + 512*(k>>5) + 128*((k&31)>>3)
 *                              + 4*n + (k&3),  ниббл старший при ((k>>2)&1)
 *   <out>.i4s  — f32[N * K/32], масштабы блоков (n-major, блок K младший)
 *   <out>.i8   — int8, per-row масштаб, row-major N x K (для пути `pref`)
 *   <out>.i8s  — f32[N], per-row масштабы int8
 *   <out>.meta — текстом: форма, параметры квантования, точность (cos)
 *
 * Считается на dev-машине: модель на телефон не копируется, на устройство
 * уезжают только блобы весов.
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

static int8_t *q4;                 /* [N][K] */
static float *sc4;                 /* [N][ng] : one scale per G*32 K */
static int8_t *q8;                 /* [N][K] */
static float *sc8;                 /* [N] */
static int N, K, ncr, nt;
static int G = 1;                  /* croutons per block scale */

static void write_file(const char *prefix, const char *suffix,
                       const void *data, size_t bytes)
{
    char path[1024];
    FILE *f;
    snprintf(path, sizeof(path), "%s%s", prefix, suffix);
    f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (fwrite(data, 1, bytes, f) != bytes) { perror(path); exit(1); }
    fclose(f);
    printf("  %-28s %9zu bytes\n", path, bytes);
}

int main(int argc, char **argv)
{
    const tinfo_t *t;
    float *w;
    uint8_t *blob;
    char path[1024];
    FILE *fm;
    int n, k;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor> <out-prefix> [G]\n"
                        "  G = 32-K croutons per block scale (default 1)\n",
                argv[0]);
        return 2;
    }
    if (argc > 4) G = atoi(argv[4]);
    if (G < 1 || G > 64 || (K % (32 * G))) {
        fprintf(stderr, "bad G=%d (1..64, K must be a multiple of 32*G)\n", G);
        return 2;
    }
    gguf_open(argv[1]);
    t = gguf_find(argv[2]);
    K = (int)t->dim[0];
    N = (int)t->dim[1];
    if (t->nd != 2 || K % 32 || N % 32) {
        fprintf(stderr, "tensor %s: nd=%u K=%d N=%d, need 2-D and 32-aligned\n",
                argv[2], t->nd, K, N);
        return 1;
    }
    ncr = K / 32;
    nt = N / 32;
    printf("tensor %s: N=%d K=%d type=%u  (%.2f MB f32)\n", argv[2], N, K,
           t->type, (double)N * K * 4.0 / 1e6);

    w = gguf_matrix(t, N, K);

    /* ---- int4: симметрично, один масштаб на блок 32*G значений K ---- */
    if (K % (32 * G)) { fprintf(stderr, "K=%d not a multiple of 32*G\n", K); return 1; }
    q4 = (int8_t *)malloc((size_t)N * K);
    sc4 = (float *)malloc((size_t)N * (K / (32 * G)) * sizeof(float));
    if (!q4 || !sc4) { fprintf(stderr, "oom\n"); return 1; }
    for (n = 0; n < N; n++) {
        for (k = 0; k < K / (32 * G); k++) {
            const float *wv = w + (size_t)n * K + (size_t)k * 32 * G;
            float amax = 0, sc;
            int l;
            for (l = 0; l < 32 * G; l++) { float a = fabsf(wv[l]); if (a > amax) amax = a; }
            sc = amax > 0 ? amax / 7.0f : 0.0f;
            sc4[(size_t)n * (K / (32 * G)) + k] = sc;
            for (l = 0; l < 32 * G; l++) {
                int v = sc > 0 ? (int)lrintf(wv[l] / sc) : 0;
                if (v > 7) v = 7;
                if (v < -8) v = -8;
                q4[(size_t)n * K + k * 32 * G + l] = (int8_t)v;
            }
        }
    }

    /* ---- упаковка int4 (раскладка стенда) ---- */
    blob = (uint8_t *)calloc((size_t)nt * ncr * 512u, 1);
    if (!blob) { fprintf(stderr, "oom\n"); return 1; }
        for (n = 0; n < N; n++) {
        unsigned char *base = blob + (size_t)(n / 32) * (size_t)ncr * 512u;
        int nn = n & 31;
        for (k = 0; k < K; k++) {
            unsigned v = (unsigned)(q4[(size_t)n * K + k] & 0x0F);
            unsigned off = 512u * (unsigned)(k >> 5)
                         + 128u * (unsigned)((k & 31) >> 3)
                         + 4u * (unsigned)nn + (unsigned)(k & 3);
            if ((k >> 2) & 1)
                base[off] = (uint8_t)((base[off] & 0x0Fu) | (v << 4));
            else
                base[off] = (uint8_t)((base[off] & 0xF0u) | v);
        }
    }

    /* ---- int8: per-row масштаб ---- */
    q8 = (int8_t *)malloc((size_t)N * K);
    sc8 = (float *)malloc((size_t)N * sizeof(float));
    if (!q8 || !sc8) { fprintf(stderr, "oom\n"); return 1; }
    for (n = 0; n < N; n++) {
        const float *row = w + (size_t)n * K;
        float amax = 0, sc;
        for (k = 0; k < K; k++) { float a = fabsf(row[k]); if (a > amax) amax = a; }
        sc = amax > 0 ? amax / 127.0f : 0.0f;
        sc8[n] = sc;
        for (k = 0; k < K; k++) {
            int v = sc > 0 ? (int)lrintf(row[k] / sc) : 0;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q8[(size_t)n * K + k] = (int8_t)v;
        }
    }

    /* ---- точность: cos на 8 строках x 32 активациях ---- */
    {
        static const int rows = 8;
        double c4 = 0, c4row = 0, c8 = 0;
        int r, j;
        for (r = 0; r < rows; r++) {
            int n0 = (int)((double)r * N / rows);
            const float *row = w + (size_t)n0 * K;
            float rowmax = 0, srow;
            float *a = (float *)malloc((size_t)K * sizeof(float));
            for (k = 0; k < K; k++) { float v = (float)gauss(); a[k] = (float)(v / (1.0 + exp(-v))); }
            for (k = 0; k < K; k++) { float av = fabsf(row[k]); if (av > rowmax) rowmax = av; }
            srow = rowmax > 0 ? rowmax / 7.0f : 0.0f;
            for (j = 0; j < 32; j++) {
                double d4 = 0, d4r = 0, d8 = 0;
                double n4 = 0, n4r = 0, n8 = 0, nref = 0;
                for (k = 0; k < K; k++) {
                    float av = a[k];
                    float ref = row[k] * av;
                    float v4 = sc4[(size_t)n0 * (K / (32 * G)) + k / (32 * G)] * (float)q4[(size_t)n0 * K + k] * av;
                    float v4r = srow * (float)(srow > 0 ? (int)lrintf(row[k] / srow) : 0) * av;
                    float v8 = sc8[n0] * (float)q8[(size_t)n0 * K + k] * av;
                    nref += (double)ref * ref;
                    d4 += (double)ref * v4;    n4 += (double)v4 * v4;
                    d4r += (double)ref * v4r;  n4r += (double)v4r * v4r;
                    d8 += (double)ref * v8;    n8 += (double)v8 * v8;
                }
                c4    += (n4 > 0 && nref > 0)   ? d4 / sqrt(n4 * nref)     : 0;
                c4row += (n4r > 0 && nref > 0)  ? d4r / sqrt(n4r * nref)   : 0;
                c8    += (n8 > 0 && nref > 0)   ? d8 / sqrt(n8 * nref)     : 0;
            }
            free(a);
        }
        c4 /= (rows * 32); c8 /= (rows * 32); c4row /= (rows * 32);
        printf("fidelity (f32 activations, %d rows x 32 cols):\n", rows);
        printf("  int4 block-%d scale: cos=%.6f\n", 32 * G, c4);
        printf("  int4 per-row scale : cos=%.6f\n", c4row);
        printf("  int8 per-row scale : cos=%.6f\n", c8);
    }

    write_file(argv[3], ".i4", blob, (size_t)nt * ncr * 512u);
    write_file(argv[3], ".i4s", sc4, (size_t)N * (K / (32 * G)) * sizeof(float));
    write_file(argv[3], ".i8", q8, (size_t)N * K);
    write_file(argv[3], ".i8s", sc8, (size_t)N * sizeof(float));

    snprintf(path, sizeof(path), "%s.meta", argv[3]);
    fm = fopen(path, "w");
    if (fm) {
        double mn = 1e30, mxs = 0, sum = 0;
        long cnt = 0, sat7 = 0;
        for (n = 0; n < N * (K / (32 * G)); n++) {
            double v = sc4[n];
            if (v <= 0) continue;
            if (v < mn) mn = v;
            if (v > mxs) mxs = v;
            sum += v; cnt++;
        }
        for (n = 0; n < N * K; n++) if (q4[n] == 7 || q4[n] == -8) sat7++;
        fprintf(fm, "tensor=%s\nN=%d\nK=%d\nint4_block=%d\nint4_range=-8..7\n"
                    "int4_ng=%d\n", argv[2], N, K, 32 * G, K / (32 * G));
        fprintf(fm, "int8_per_row=1\n");
        fprintf(fm, "block_scale_min=%.8g\nblock_scale_max=%.8g\nblock_scale_mean=%.8g\n",
                mn, mxs, cnt ? sum / cnt : 0);
        fprintf(fm, "int4_values_at_rail=%.4f%%\n", 100.0 * (double)sat7 / ((double)N * K));
        fclose(fm);
        printf("  %-28s meta\n", path);
    }
    free(w); free(q4); free(sc4); free(q8); free(sc8); free(blob);
    return 0;
}
