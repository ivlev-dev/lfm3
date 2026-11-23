/*
 * q4k_host: host stand for the Q4_K path.
 *
 * Runs LFM3-1.7B greedy generation with the matvecs routed through selectable
 * weight formats, so the *same* pipeline can be compared format by format:
 *
 *   f32   exact dequant of every weight (the CPU reference, lfm-bench ref-gen)
 *   q4k   Q4_K packed + int8/32 activations (int32 per-32 dot, f32 combine);
 *         Q6_K tensors (ffn_down, token_embd, 4x attn_v) stay exact
 *   q4kf  Q4_K packed + f32 activations (weight format only, no act quant)
 *   i8    int8/32 requant of every packed weight (the OpenCL "int4/32" prepack)
 *   i4    int4/32 requant of every packed weight (accepted path, amax/7)
 *
 * The q4k maths is q4k_impl.c -- the same source the HVX kernel is built from,
 * so what this stand measures is what the DSP kernel computes (the DSP stand
 * q4k_main.c additionally checks HVX == the scalar twin bit for bit).
 *
 * Usage:
 *   --mode <f32|q4k|q4kf|i8|i4>   weight format              (default q4k)
 *   --ids "597 5205 302 3980 355" prompt token ids
 *   --tokens N                    greedy continuation length (default 64)
 *   --selftest                    pack/dequant exactness + cos vs the f32 oracle
 *   --emit-case <file>            write the DSP stand case header
 *   --case-rows N                 rows of blk.0.ffn_gate.weight (default 512)
 *
 * Build:
 *   cc -O2 -march=native -fopenmp -o build/q4k_host q4k_host.c q4k_impl.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gguf_f32.h"       /* tensor directory; dequant helpers; gguf_open */
#include "q4k_impl.h"
#include "../quant/quant.h"

#define MAXT 400
#define MAXL 32
#define KV_SLOTS 512

enum { M_F32 = 0, M_Q4K, M_Q4KF, M_I8, M_I4, M_I4M, M_I4R };

static int g_mode = M_Q4K;

/* ------------------------------------------------------------------ */
/* model                                                               */
/* ------------------------------------------------------------------ */
typedef struct {
    int N, K;               /* rows (out), cols (in)                        */
    int type;               /* 0 F32, 1 F16, 12 Q4_K, 14 Q6_K               */
    int packable;           /* type 12 and K % 256 == 0                     */
    int blocks_ok;          /* type 12/14 and K % 256 == 0                  */
    int64_t off;            /* absolute offset in the mapped file           */
    size_t row_bytes;
    uint8_t *img;           /* q4k packed image                             */
    q4k_mat qm;
    int8_t *q8; float *s8;  /* mode i8                                      */
    int8_t *q4; float *s4;  /* mode i4 (offset-8 nibbles)                   */
    float *m4;              /* mode i4m: per-group min                       */
    float *sr;              /* mode i4r: one scale per row                   */
} wmat;

static wmat g_w[MAXT];
static int g_nw;
static tinfo_t *g_ti[MAXT];
static uint8_t *g_map;      /* whole GGUF, read-only                        */
static size_t g_map_len;

static int g_NL = 24, g_H = 2048, g_FFN = 7680, g_HQ = 16, g_HKV = 4, g_D = 128;
static int g_WIN = 256, g_V = 128000;
static double g_eps = 1e-5;
static float *g_rope_full, *g_rope_slide;

static int lkind(int L)
{
    if (L < 2 || (L & 1)) return 0;
    return (L % 4 == 2) ? 1 : 2;
}

static void *xalloc(size_t n, size_t align)
{
    void *p = NULL;
    if (posix_memalign(&p, align, n) != 0) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

/* one f32 row out of the mapped GGUF; thread safe */
static void row_f32(const wmat *w, int r, float *out)
{
    const uint8_t *p = g_map + w->off + (int64_t)r * (int64_t)w->row_bytes;
    int nb, i;
    if (w->type == 0) { memcpy(out, p, (size_t)w->K * 4); return; }
    if (w->type == 1) {
        for (i = 0; i < w->K; i++) {
            uint16_t h = (uint16_t)(p[2 * i] | (p[2 * i + 1] << 8));
            out[i] = gguf_f16(h);
        }
        return;
    }
    nb = w->K / 256;
    for (i = 0; i < nb; i++) {
        if (w->type == 12) gguf_deq_q4k(p + (size_t)i * 144, out + i * 256);
        else               gguf_deq_q6k(p + (size_t)i * 210, out + i * 256);
    }
}

/* int8/32 requant of a f32 row, exactly prepack.rs:48-76 (round half away) */
static void requant_i8(const float *v, int K, int8_t *q, float *sc)
{
    int g, i;
    for (g = 0; g < K / 32; g++) {
        float amax = 0.0f, inv;
        for (i = 0; i < 32; i++) { float a = fabsf(v[32 * g + i]); if (a > amax) amax = a; }
        inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        sc[g] = (amax > 0.0f) ? amax / 127.0f : 0.0f;
        for (i = 0; i < 32; i++) {
            float t = roundf(v[32 * g + i] * inv);
            if (t > 127.0f) t = 127.0f;
            if (t < -127.0f) t = -127.0f;
            q[32 * g + i] = (int8_t)t;
        }
    }
}

/* int4/32 requant, myelin/dsp/quant/host/mkcase.c: scale = amax/7, q in [-7,7] */
static void requant_i4(const float *v, int K, int8_t *q, float *sc)
{
    int g, i;
    for (g = 0; g < K / 32; g++) {
        float amax = 0.0f, inv;
        for (i = 0; i < 32; i++) { float a = fabsf(v[32 * g + i]); if (a > amax) amax = a; }
        inv = (amax > 0.0f) ? 7.0f / amax : 0.0f;
        sc[g] = (amax > 0.0f) ? amax / 7.0f : 0.0f;
        for (i = 0; i < 32; i++) {
            float t = roundf(v[32 * g + i] * inv);
            if (t > 7.0f) t = 7.0f;
            if (t < -7.0f) t = -7.0f;
            q[32 * g + i] = (int8_t)t;
        }
    }
}

/* int4/32 with a per-group min (the Q4_K idea, 6-bit sc/m replaced by f32) */
static void requant_i4m(const float *v, int K, int8_t *q, float *sc, float *mn)
{
    int g, i;
    for (g = 0; g < K / 32; g++) {
        float lo = v[32 * g], hi = v[32 * g], range;
        for (i = 1; i < 32; i++) {
            if (v[32 * g + i] < lo) lo = v[32 * g + i];
            if (v[32 * g + i] > hi) hi = v[32 * g + i];
        }
        range = hi - lo;
        sc[g] = (range > 0.0f) ? range / 15.0f : 0.0f;
        mn[g] = lo;
        for (i = 0; i < 32; i++) {
            float t = sc[g] > 0.0f ? roundf((v[32 * g + i] - lo) / sc[g]) : 0.0f;
            if (t > 15.0f) t = 15.0f;
            if (t < 0.0f) t = 0.0f;
            q[32 * g + i] = (int8_t)t;
        }
    }
}

static void wmat_prepare(wmat *w)
{
    int r;
    if (g_mode == M_Q4K || g_mode == M_Q4KF) {
        if (w->packable) {
            w->img = xalloc(q4k_image_bytes(w->N, w->K), 128);
            if (q4k_pack_image(g_map + w->off, w->N, w->K, w->img) != 0) {
                fprintf(stderr, "pack failed %d x %d\n", w->N, w->K);
                exit(1);
            }
            q4k_mat_init(&w->qm, w->img, w->N, w->K);
        }
        return;                       /* Q6_K / F32 read through row_f32 */
    }
    if (!w->blocks_ok) return;
    {
        int groups = w->K / 32;
        float *row = malloc((size_t)w->K * 4);
        int8_t *q = malloc((size_t)w->K);
        if (g_mode == M_I8) {
            w->q8 = xalloc((size_t)w->N * w->K, 64);
            w->s8 = xalloc((size_t)w->N * groups * 4, 64);
        } else if (g_mode == M_I4R) {
            w->q4 = xalloc((size_t)w->N * (w->K / 2), 64);
            w->sr = xalloc((size_t)w->N * 4, 64);
        } else if (g_mode == M_I4M) {
            w->q4 = xalloc((size_t)w->N * (w->K / 2), 64);
            w->s4 = xalloc((size_t)w->N * groups * 4, 64);
            w->m4 = xalloc((size_t)w->N * groups * 4, 64);
        } else {
            w->q4 = xalloc((size_t)w->N * (w->K / 2), 64);
            w->s4 = xalloc((size_t)w->N * groups * 4, 64);
        }
        for (r = 0; r < w->N; r++) {
            row_f32(w, r, row);
            if (g_mode == M_I8) {
                requant_i8(row, w->K, w->q8 + (size_t)r * w->K, w->s8 + (size_t)r * groups);
            } else if (g_mode == M_I4R) {
            w->q4 = xalloc((size_t)w->N * (w->K / 2), 64);
            w->sr = xalloc((size_t)w->N * 4, 64);
        } else if (g_mode == M_I4M) {
                int i;
                requant_i4m(row, w->K, q, w->s4 + (size_t)r * groups,
                            w->m4 + (size_t)r * groups);
                for (i = 0; i < w->K / 2; i++)
                    w->q4[(size_t)r * (w->K / 2) + i] =
                        (uint8_t)((q[2 * i] & 15) | ((q[2 * i + 1] & 15) << 4));
            } else {
                int i;
                requant_i4(row, w->K, q, w->s4 + (size_t)r * groups);
                for (i = 0; i < w->K / 2; i++)
                    w->q4[(size_t)r * (w->K / 2) + i] =
                        (uint8_t)(((q[2 * i] + 8) & 15) | (((q[2 * i + 1] + 8) & 15) << 4));
            }
        }
        free(row); free(q);
    }
}

/* ------------------------------------------------------------------ */
/* matvec                                                              */
/* ------------------------------------------------------------------ */
static inline float dot8_f32(const float *a, const float *b, int n)
{
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int c = 0, i;
    for (; c + 8 <= n; c += 8)
        for (i = 0; i < 8; i++) acc[i] += a[c + i] * b[c + i];
    {
        float s = 0.0f;
        for (i = 0; i < 8; i++) s += acc[i];
        for (; c < n; c++) s += a[c] * b[c];
        return s;
    }
}

static void deq_q8(const int8_t *xq, const float *xs, float *out, int K)
{
    int i;
    for (i = 0; i < K; i++) out[i] = (float)xq[i] * xs[i / 32];
}

static __thread float *t_row;      /* scratch of exactly-K floats    */
static __thread int t_row_cap;
static float *row_scratch(int K)
{
    if (t_row_cap < K) {
        free(t_row);
        t_row = malloc((size_t)K * 4);
        t_row_cap = K;
    }
    return t_row;
}

/* exact f32 weights through row_f32 */
static void mv_exact(const wmat *w, const float *xr, float *y)
{
    int r, K = w->K;
#pragma omp parallel for schedule(static)
    for (r = 0; r < w->N; r++) {
        float *row = row_scratch(K);
        row_f32(w, r, row);
        y[r] = dot8_f32(row, xr, K);
    }
}

static void matvec(const wmat *w, const float *x, float *y);

static void mv(const wmat *w, const float *x, float *y)
{
    int r, K = w->K;

    if (g_mode == M_F32) { mv_exact(w, x, y); return; }

    if (g_mode == M_Q4K || g_mode == M_Q4KF) {
        if (!w->packable) {
            /* Q6_K / F32 tensor: exact weights; q8 activations when the mode
             * quantises activations (q4k), plain f32 otherwise (q4kf) */
            if (g_mode == M_Q4K && w->K % 32 == 0) {
                int8_t *xq = xalloc(K, 64);
                float *xs = xalloc((size_t)(K / 32) * 4, 64);
                float *xr = malloc((size_t)K * 4);
                q4k_act_q8(x, K, xq, xs);
                deq_q8(xq, xs, xr, K);
                mv_exact(w, xr, y);
                free(xq); free(xs); free(xr);
            } else {
                mv_exact(w, x, y);
            }
            return;
        }
        if (g_mode == M_Q4KF) {
#pragma omp parallel for schedule(static)
            for (r = 0; r < w->N; r++) q4k_gemv_f32act(&w->qm, x, y, r, r + 1);
            return;
        }
        {
            int8_t *xq = xalloc(K, 128);
            float *xs = xalloc((size_t)(K / 32) * 4, 64);
            q4k_plan plan;
            q4k_act_q8(x, K, xq, xs);
            if (q4k_plan_init(&plan, &w->qm, xq, xs, NULL) != 0) { fprintf(stderr, "plan\n"); exit(1); }
#pragma omp parallel for schedule(static)
            for (r = 0; r < w->N; r++) q4k_gemv_scalar(&w->qm, &plan, xq, y, r, r + 1);
            q4k_plan_free(&plan);
            free(xq); free(xs);
        }
        return;
    }

    if (!w->packable) {
        /* Q6_K / F32 tensor under i8/i4 mode: exact weights, q8 activations */
        int8_t *xq = xalloc(K, 64);
        float *xs = xalloc((size_t)(K / 32) * 4, 64);
        float *xr = malloc((size_t)K * 4);
        q4k_act_q8(x, K, xq, xs);
        deq_q8(xq, xs, xr, K);
        mv_exact(w, xr, y);
        free(xq); free(xs); free(xr);
        return;
    }

    if (g_mode == M_I8) {
        int groups = K / 32, g;
        int8_t *xq = xalloc(K, 64);
        float *xs = xalloc((size_t)groups * 4, 64);
        q4k_act_q8(x, K, xq, xs);
#pragma omp parallel for schedule(static)
        for (r = 0; r < w->N; r++) {
            const int8_t *q = w->q8 + (size_t)r * K;
            const float *s = w->s8 + (size_t)r * groups;
            float acc = 0.0f;
            for (g = 0; g < groups; g++) {
                int32_t d = 0;
                int i;
                for (i = 0; i < 32; i++) d += (int32_t)q[32 * g + i] * (int32_t)xq[32 * g + i];
                acc += (float)d * s[g] * xs[g];
            }
            y[r] = acc;
        }
        free(xq); free(xs);
        return;
    }

    if (g_mode == M_I4R) {
        int groups = K / 32, g;
        int8_t *xq = xalloc(K, 64);
        float *xs = xalloc((size_t)groups * 4, 64);
        int32_t *sums = xalloc((size_t)groups * 4, 64);
        q4k_act_q8(x, K, xq, xs);
        q4k_act_sums(xq, K, sums);
#pragma omp parallel for schedule(static)
        for (r = 0; r < w->N; r++) {
            const uint8_t *q = (const uint8_t *)w->q4 + (size_t)r * (K / 2);
            float acc = 0.0f;
            for (g = 0; g < groups; g++) {
                const uint8_t *qb = q + 16 * g;
                int32_t d = 0;
                int i;
                for (i = 0; i < 16; i++) {
                    d += (int32_t)(qb[i] & 15) * (int32_t)xq[32 * g + 2 * i];
                    d += (int32_t)(qb[i] >> 4) * (int32_t)xq[32 * g + 2 * i + 1];
                }
                d -= 8 * sums[g];
                acc += (float)d * xs[g];
            }
            y[r] = acc * w->sr[r];
        }
        free(xq); free(xs); free(sums);
        return;
    }

    if (g_mode == M_I4M) {
        int groups = K / 32, g;
        int8_t *xq = xalloc(K, 64);
        float *xs = xalloc((size_t)groups * 4, 64);
        int32_t *sums = xalloc((size_t)groups * 4, 64);
        q4k_act_q8(x, K, xq, xs);
        q4k_act_sums(xq, K, sums);
#pragma omp parallel for schedule(static)
        for (r = 0; r < w->N; r++) {
            const uint8_t *q = (const uint8_t *)w->q4 + (size_t)r * (K / 2);
            const float *s = w->s4 + (size_t)r * groups;
            const float *mn = w->m4 + (size_t)r * groups;
            float acc = 0.0f;
            for (g = 0; g < groups; g++) {
                const uint8_t *qb = q + 16 * g;
                int32_t d = 0;
                int i;
                for (i = 0; i < 16; i++) {
                    d += (int32_t)(qb[i] & 15) * (int32_t)xq[32 * g + 2 * i];
                    d += (int32_t)(qb[i] >> 4) * (int32_t)xq[32 * g + 2 * i + 1];
                }
                acc += xs[g] * ((float)d * s[g] + mn[g] * (float)sums[g]);
            }
            y[r] = acc;
        }
        free(xq); free(xs); free(sums);
        return;
    }

    /* i4/32, offset-8 nibbles: dot(u,x) - 8*sum(x) per group, exact in int32 */
    {
        int groups = K / 32, g;
        int8_t *xq = xalloc(K, 64);
        float *xs = xalloc((size_t)groups * 4, 64);
        int32_t *sums = xalloc((size_t)groups * 4, 64);
        q4k_act_q8(x, K, xq, xs);
        q4k_act_sums(xq, K, sums);
#pragma omp parallel for schedule(static)
        for (r = 0; r < w->N; r++) {
            const uint8_t *q = (const uint8_t *)w->q4 + (size_t)r * (K / 2);
            const float *s = w->s4 + (size_t)r * groups;
            float acc = 0.0f;
            for (g = 0; g < groups; g++) {
                const uint8_t *qb = q + 16 * g;
                int32_t d = 0;
                int i;
                for (i = 0; i < 16; i++) {
                    d += (int32_t)(qb[i] & 15) * (int32_t)xq[32 * g + 2 * i];
                    d += (int32_t)(qb[i] >> 4) * (int32_t)xq[32 * g + 2 * i + 1];
                }
                d -= 8 * sums[g];
                acc += (float)d * s[g] * xs[g];
            }
            y[r] = acc;
        }
        free(xq); free(xs); free(sums);
    }
}

static void matvec(const wmat *w, const float *x, float *y) { mv(w, x, y); }

/* ------------------------------------------------------------------ */
/* tensor lookup                                                       */
/* ------------------------------------------------------------------ */
static wmat *W(const char *name)
{
    int i;
    for (i = 0; i < g_nw; i++)
        if (!strcmp(g_ti[i]->name, name)) return &g_w[i];
    fprintf(stderr, "no tensor %s\n", name);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* numerics                                                            */
/* ------------------------------------------------------------------ */
static void rmsnorm(const float *v, const float *w, int n, float *out)
{
    double s = 0.0;
    int i;
    float inv;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    inv = (float)(1.0 / sqrt(s / n + g_eps));
    for (i = 0; i < n; i++) out[i] = v[i] * inv * w[i];
}

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf_(float x) { return x / (1.0f + expf(-x)); }

static void rope_inplace(float *v, int pos, const float *inv)
{
    int half = g_D / 2, i;
    for (i = 0; i < half; i++) {
        float a, s, c, x0, x1;
        if (inv[i] == 0.0f) continue;
        a = (float)pos * inv[i];
        s = sinf(a); c = cosf(a);
        x0 = v[i]; x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x1 * c + x0 * s;
    }
}

/* ------------------------------------------------------------------ */
/* state / forward                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    float *conv[MAXL];
    float *k[MAXL], *v[MAXL];
    int len[MAXL];
    int pos;
} state_t;

static float *g_cap;        /* capture buffer for the layer-0 ffn input */

static state_t *st_new(void)
{
    state_t *s = calloc(1, sizeof(state_t));
    int L;
    for (L = 0; L < g_NL; L++) {
        s->conv[L] = calloc((size_t)6 * g_H, 4);
        s->k[L] = calloc((size_t)KV_SLOTS * g_HKV * g_D, 4);
        s->v[L] = calloc((size_t)KV_SLOTS * g_HKV * g_D, 4);
    }
    return s;
}

/* one token; top1 != NULL -> greedy argmax of the logits */
static void forward(state_t *S, int tok, int *top1)
{
    const int H = g_H, F = g_FFN, D = g_D, QD = g_HQ * D, KD = g_HKV * D;
    const int pos = S->pos;
    float *x = calloc((size_t)H, 4), *res = calloc((size_t)H, 4);
    float *nrm = calloc((size_t)H, 4), *blk = calloc((size_t)H, 4);
    float *mrg = calloc((size_t)H, 4), *tmp = calloc((size_t)H, 4);
    float *q = calloc((size_t)QD, 4), *kk = calloc((size_t)KD, 4), *vv = calloc((size_t)KD, 4);
    float *gat = calloc((size_t)g_HQ, 4), *att = calloc((size_t)QD, 4);
    float *mix = calloc((size_t)3 * H, 4), *bx = calloc((size_t)H, 4);
    float *cv = calloc((size_t)H, 4), *cvg = calloc((size_t)H, 4);
    float *fa = calloc((size_t)F, 4), *fu = calloc((size_t)F, 4), *fg = calloc((size_t)F, 4);
    double *sc = calloc((size_t)4096, 8);
    int L, i, h, j, z;

    row_f32(W("token_embd.weight"), tok, x);

    for (L = 0; L < g_NL; L++) {
        char nm[64];
        const int kind = lkind(L);
        float v7[8];

        memcpy(res, x, (size_t)H * 4);
        snprintf(nm, 64, "blk.%d.attn_norm.weight", L);
        {
            float *v = gguf_vec_alloc(nm, H);
            rmsnorm(x, v, H, nrm);
            free(v);
        }

        if (kind == 0) {
            wmat *cin, *ccw, *cout;
            snprintf(nm, 64, "blk.%d.shortconv.in_proj.weight", L); cin = W(nm);
            snprintf(nm, 64, "blk.%d.shortconv.conv.weight", L);   ccw = W(nm);
            snprintf(nm, 64, "blk.%d.shortconv.out_proj.weight", L); cout = W(nm);
            matvec(cin, nrm, mix);
            for (i = 0; i < H; i++) bx[i] = mix[i] * mix[2 * H + i];
            for (i = 0; i < H; i++) {
                float acc;
                row_f32(ccw, i, v7);
                acc = v7[6] * bx[i];
                for (j = 0; j < 6; j++) acc += v7[j] * S->conv[L][j * H + i];
                cv[i] = acc;
            }
            memmove(S->conv[L], S->conv[L] + H, (size_t)5 * H * 4);
            memcpy(S->conv[L] + 5 * H, bx, (size_t)H * 4);
            for (i = 0; i < H; i++) cvg[i] = mix[H + i] * cv[i];
            matvec(cout, cvg, blk);
        } else {
            wmat *wq, *wk, *wv, *wg, *wo;
            int start, n, grp = g_HQ / g_HKV;
            float *qn, *kn;
            snprintf(nm, 64, "blk.%d.attn_q.weight", L); wq = W(nm);
            snprintf(nm, 64, "blk.%d.attn_k.weight", L); wk = W(nm);
            snprintf(nm, 64, "blk.%d.attn_v.weight", L); wv = W(nm);
            snprintf(nm, 64, "blk.%d.attn_gate.weight", L); wg = W(nm);
            snprintf(nm, 64, "blk.%d.attn_output.weight", L); wo = W(nm);
            matvec(wq, nrm, q);
            matvec(wk, nrm, kk);
            matvec(wv, nrm, vv);
            matvec(wg, nrm, gat);
            snprintf(nm, 64, "blk.%d.attn_q_norm.weight", L); qn = gguf_vec_alloc(nm, D);
            snprintf(nm, 64, "blk.%d.attn_k_norm.weight", L); kn = gguf_vec_alloc(nm, D);
            for (h = 0; h < g_HQ; h++) {
                rmsnorm(q + h * D, qn, D, tmp);
                memcpy(q + h * D, tmp, (size_t)D * 4);
                rope_inplace(q + h * D, pos, kind == 1 ? g_rope_full : g_rope_slide);
            }
            for (h = 0; h < g_HKV; h++) {
                rmsnorm(kk + h * D, kn, D, tmp);
                memcpy(kk + h * D, tmp, (size_t)D * 4);
                rope_inplace(kk + h * D, pos, kind == 1 ? g_rope_full : g_rope_slide);
            }
            free(qn); free(kn);
            memcpy(S->k[L] + (size_t)S->len[L] * KD, kk, (size_t)KD * 4);
            memcpy(S->v[L] + (size_t)S->len[L] * KD, vv, (size_t)KD * 4);
            S->len[L]++;
            start = (kind == 2) ? (S->len[L] > g_WIN ? S->len[L] - g_WIN : 0) : 0;
            n = S->len[L] - start;
            memset(att, 0, (size_t)QD * 4);
            for (h = 0; h < g_HQ; h++) {
                int kh = h / grp;
                double scale = 1.0 / sqrt((double)D);
                double mx, sum;
                for (j = 0; j < n; j++) {
                    const float *kb = S->k[L] + (size_t)(start + j) * KD + kh * D;
                    const float *qb = q + h * D;
                    double d = 0.0;
                    for (z = 0; z < D; z++) d += (double)qb[z] * kb[z];
                    sc[j] = d * scale;
                }
                mx = sc[0];
                for (j = 1; j < n; j++) if (sc[j] > mx) mx = sc[j];
                sum = 0.0;
                for (j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
                for (j = 0; j < n; j++) sc[j] /= sum;
                for (j = 0; j < n; j++) {
                    const float *vb = S->v[L] + (size_t)(start + j) * KD + kh * D;
                    float *o = att + h * D;
                    for (z = 0; z < D; z++) o[z] += (float)(sc[j] * vb[z]);
                }
            }
            for (h = 0; h < g_HQ; h++) {
                float g = sigmoidf_(gat[h]);
                for (i = 0; i < D; i++) att[h * D + i] *= g;
            }
            matvec(wo, att, blk);
        }

        snprintf(nm, 64, "blk.%d.post_attention_norm.weight", L);
        {
            float *v = gguf_vec_alloc(nm, H);
            rmsnorm(blk, v, H, tmp);
            free(v);
        }
        for (i = 0; i < H; i++) mrg[i] = res[i] + tmp[i];
        snprintf(nm, 64, "blk.%d.ffn_norm.weight", L);
        {
            float *v = gguf_vec_alloc(nm, H);
            rmsnorm(mrg, v, H, nrm);
            free(v);
        }
        if (L == 0 && g_cap) memcpy(g_cap, nrm, (size_t)H * 4);

        {
            wmat *wu, *wga, *wd;
            snprintf(nm, 64, "blk.%d.ffn_up.weight", L); wu = W(nm);
            snprintf(nm, 64, "blk.%d.ffn_gate.weight", L); wga = W(nm);
            snprintf(nm, 64, "blk.%d.ffn_down.weight", L); wd = W(nm);
            matvec(wu, nrm, fu);
            matvec(wga, nrm, fg);
            for (i = 0; i < F; i++) fa[i] = siluf_(fg[i]) * fu[i];
            matvec(wd, fa, blk);
        }
        snprintf(nm, 64, "blk.%d.post_ffw_norm.weight", L);
        {
            float *v = gguf_vec_alloc(nm, H);
            rmsnorm(blk, v, H, tmp);
            free(v);
        }
        for (i = 0; i < H; i++) x[i] = mrg[i] + tmp[i];
    }
    S->pos++;

    if (top1) {
        float *logits = malloc((size_t)g_V * 4);
        float *onorm = gguf_vec_alloc("output_norm.weight", H);
        int r, best = 0;
        float bv = -1e38f;
        rmsnorm(x, onorm, H, nrm);
        free(onorm);
        matvec(W("token_embd.weight"), nrm, logits);
        for (r = 0; r < g_V; r++) if (logits[r] > bv) { bv = logits[r]; best = r; }
        *top1 = best;
        free(logits);
    }
    free(x); free(res); free(nrm); free(blk); free(mrg); free(tmp);
    free(q); free(kk); free(vv); free(gat); free(att);
    free(mix); free(bx); free(cv); free(cvg);
    free(fa); free(fu); free(fg); free(sc);
}

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */
static void model_init(const char *path)
{
    int i, fd;
    struct stat st;
    size_t rb;

    gguf_open(path);          /* tensor directory + g_data (file offset) */
    fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
    g_map_len = (size_t)st.st_size;
    g_map = mmap(NULL, g_map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g_map == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);

    {
        int half = g_D / 2, rot = g_D / 4, i2;   /* full: rotated_dim = D/2 */
        g_rope_full = calloc((size_t)half, 4);
        g_rope_slide = calloc((size_t)half, 4);
        for (i2 = 0; i2 < rot; i2++)
            g_rope_full[i2] = (float)(1.0 / pow(1e6, (2.0 * i2) / (double)g_D));
        for (i2 = 0; i2 < half; i2++)
            g_rope_slide[i2] = (float)(1.0 / pow(1e4, (2.0 * i2) / (double)g_D));
    }

    g_nw = 0;
    for (i = 0; i < g_nt; i++) {
        tinfo_t *t = &g_t[i];
        if (t->nd != 2) continue;
        if (t->type != 0 && t->type != 1 && t->type != 12 && t->type != 14) continue;
        g_ti[g_nw] = t;
        memset(&g_w[g_nw], 0, sizeof(wmat));
        g_w[g_nw].N = (int)t->dim[1];
        g_w[g_nw].K = (int)t->dim[0];
        g_w[g_nw].type = (int)t->type;
        g_w[g_nw].off = g_data + (int64_t)t->off;
        if (t->type == 12) rb = (size_t)(g_w[g_nw].K / 256) * Q4K_GGUF_BLOCK;
        else if (t->type == 14) rb = (size_t)(g_w[g_nw].K / 256) * 210;
        else if (t->type == 1) rb = (size_t)g_w[g_nw].K * 2;
        else rb = (size_t)g_w[g_nw].K * 4;
        g_w[g_nw].row_bytes = rb;
        g_w[g_nw].packable = (t->type == 12 && (g_w[g_nw].K % 256) == 0);
        g_w[g_nw].blocks_ok = ((t->type == 12 || t->type == 14) &&
                               (g_w[g_nw].K % 256) == 0);
        g_nw++;
    }
    fprintf(stderr, "[q4k] %d 2-D tensors, mode %d, model %.2f GB\n",
            g_nw, g_mode, (double)g_map_len / 1e9);
    for (i = 0; i < g_nw; i++) {
        if (g_w[i].blocks_ok) wmat_prepare(&g_w[i]);
        if ((i % 24) == 0) fprintf(stderr, "[q4k] prepared %d/%d %s\n", i, g_nw, g_ti[i]->name);
    }
}

/* ------------------------------------------------------------------ */
/* selftest / case emission                                            */
/* ------------------------------------------------------------------ */
static void metrics(const char *tag, const float *a, const float *b, int n)
{
    double ab = 0, aa = 0, bb = 0, mx = 0, mxa = 0;
    int i;
    for (i = 0; i < n; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        ab += x * y; aa += x * x; bb += y * y;
        if (d > mx) mx = d;
        if (fabs(x) > mxa) mxa = fabs(x);
    }
    printf("%-30s cos=%.9f  maxerr=%.3e  rel=%.2e\n",
           tag, ab / (sqrt(aa) * sqrt(bb) + 1e-30), mx, mx / (mxa + 1e-30));
}

static void selftest(void)
{
    wmat *w = W("blk.0.ffn_gate.weight");
    int N = w->N, K = w->K, r, i;
    float *xf = malloc((size_t)K * 4);
    float *yref = malloc((size_t)N * 4);
    float *yq = malloc((size_t)N * 4);
    float *yq4 = malloc((size_t)N * 4);
    int8_t *xq = xalloc(K, 128);
    float *xs = xalloc((size_t)(K / 32) * 4, 64);
    q4k_plan plan;

    /* 1. pack/dequant exactness against the GGUF reference dequant */
    {
        double mx = 0.0;
        float a[256], b[256];
        for (i = 0; i < 64; i++) {
            gguf_deq_q4k(g_map + w->off + (int64_t)i * 144, a);
            q4k_dequant_block(w->qm.hdr + (size_t)i * Q4K_HDR_BLOCK,
                              w->qm.p + (size_t)i * Q4K_PLANE, b);
            for (r = 0; r < 256; r++) {
                double d = fabs((double)a[r] - (double)b[r]);
                if (d > mx) mx = d;
            }
        }
        printf("pack/dequant vs gguf_deq_q4k: max|diff|=%.3e (64 blocks)\n", mx);
    }

    /* 1b. weight fidelity of each 4/8-bit representation, same tensor */
    {
        int groups = K / 32, g, i2;
        double num[6] = {0}, den = 0.0;
        int8_t q8b[8192]; float s8b[256];
        int8_t q4b[8192]; float s4b[256];
        float sr[8192];
        int nr = N < 512 ? N : 512;
        for (r = 0; r < nr; r++) {
            float *row = row_scratch(K);
            float wq[256];
            int nb;
            row_f32(w, r, row);
            /* q4k */
            for (nb = 0; nb < K / 256; nb++) {
                q4k_dequant_block(w->qm.hdr + ((size_t)r * w->qm.blocks + nb) * Q4K_HDR_BLOCK,
                                  w->qm.p + ((size_t)r * w->qm.blocks + nb) * Q4K_PLANE, wq);
                for (i2 = 0; i2 < 256; i2++) {
                    double d = (double)row[256 * nb + i2] - (double)wq[i2];
                    num[0] += d * d;
                }
            }
            /* i8/32 */
            requant_i8(row, K, q8b, s8b);
            for (i2 = 0; i2 < K; i2++) {
                double d = (double)row[i2] - (double)q8b[i2] * s8b[i2 / 32];
                num[1] += d * d;
            }
            /* i4/32, no rotation */
            requant_i4(row, K, q4b, s4b);
            for (i2 = 0; i2 < K; i2++) {
                double d = (double)row[i2] - (double)q4b[i2] * s4b[i2 / 32];
                num[2] += d * d;
            }
            /* i4/32 after the H32 rotation (HMX path) */
            memcpy(sr, row, (size_t)K * 4);
            lfm_rot_blocks_f32(sr, K, 32);
            requant_i4(sr, K, q4b, s4b);
            lfm_rot_blocks_f32(sr, K, 32);        /* back: not orthogonal-inverse
                                                     per group here, so rotate
                                                     the reconstruction instead */
            for (g = 0; g < groups; g++) {
                float back[32];
                int t2;
                for (t2 = 0; t2 < 32; t2++) back[t2] = (float)q4b[32 * g + t2] * s4b[g];
                lfm_rot_inv_f32(back, 32);
                for (t2 = 0; t2 < 32; t2++) {
                    double d = (double)row[32 * g + t2] - (double)back[t2];
                    num[3] += d * d;
                }
            }
            /* i4/32 with per-group min (Q4_K scheme) */
            {
                int8_t qm4[8192]; float sm4[256], mm4[256];
                requant_i4m(row, K, qm4, sm4, mm4);
                for (i2 = 0; i2 < K; i2++) {
                    double d = (double)row[i2] - (double)(qm4[i2] * sm4[i2 / 32] + mm4[i2 / 32]);
                    num[5] += d * d;
                }
            }
            /* i4 per row, no rotation (lfmgemv_impl.c format) */
            {
                float amax = 0.0f, sc;
                for (i2 = 0; i2 < K; i2++) { float a = fabsf(row[i2]); if (a > amax) amax = a; }
                sc = amax / 7.0f;
                for (i2 = 0; i2 < K; i2++) {
                    float t = roundf(row[i2] / sc);
                    double v;
                    if (t > 7.0f) t = 7.0f;
                    if (t < -7.0f) t = -7.0f;
                    v = (double)t * sc;
                    num[4] += ((double)row[i2] - v) * ((double)row[i2] - v);
                }
            }
            for (i2 = 0; i2 < K; i2++) den += (double)row[i2] * row[i2];
        }
        printf("\nweight rel-RMS error over %d rows:\n", nr);
        printf("  q4k (gguf)            %.5f\n", sqrt(num[0] / den));
        printf("  i8/32                 %.5f\n", sqrt(num[1] / den));
        printf("  i4/32                 %.5f\n", sqrt(num[2] / den));
        printf("  i4/32 + H32 rotation  %.5f\n", sqrt(num[3] / den));
        printf("  i4/row                %.5f\n", sqrt(num[4] / den));
        printf("  i4/32 + group min     %.5f\n", sqrt(num[5] / den));
    }

    /* 2. real layer-0 ffn activation, from the f32 forward of token 597 */
    {
        state_t *S = st_new();
        g_cap = xf;
        forward(S, 597, NULL);
        g_cap = NULL;
    }

    q4k_act_q8(xf, K, xq, xs);
    if (q4k_plan_init(&plan, &w->qm, xq, xs, NULL) != 0) { fprintf(stderr, "plan\n"); exit(1); }

#pragma omp parallel for schedule(static)
    for (r = 0; r < N; r++) q4k_gemv_f32act(&w->qm, xf, yref, r, r + 1);
#pragma omp parallel for schedule(static)
    for (r = 0; r < N; r++) q4k_gemv_scalar(&w->qm, &plan, xq, yq, r, r + 1);
    {
        float *xr = malloc((size_t)K * 4);
        deq_q8(xq, xs, xr, K);
#pragma omp parallel for schedule(static)
        for (r = 0; r < N; r++) {
            float *row = row_scratch(K);
            row_f32(w, r, row);
            yq4[r] = dot8_f32(row, xr, K);
        }
        free(xr);
    }
    printf("\n=== blk.0.ffn_gate.weight [%d x %d], real layer-0 ffn activation "
           "(token 597) ===\n", N, K);
    metrics("q4k(pack+q8act) vs f32", yq, yref, N);
    metrics("f32w+q8act vs f32", yq4, yref, N);
    {
        double mx = 0.0;
        for (r = 0; r < N; r++) {
            double d = fabs((double)yq[r] - (double)yq4[r]);
            if (d > mx) mx = d;
        }
        printf("%-30s maxerr=%.3e (weight-format error only)\n", "q4k vs f32w+q8act", mx);
    }

    /* the two requantised formats, same activation, same matrix */
    {
        float *yi8 = malloc((size_t)N * 4), *yi4 = malloc((size_t)N * 4);
        int groups = K / 32, g, i2, G;
        int32_t sums[64];
        int8_t qi8[8192]; float si8[256];
        int8_t qi4[8192]; float si4[256];
        r = 0;
        q4k_act_sums(xq, K, sums);
        for (r = 0; r < N; r++) {
            float *row = row_scratch(K);
            float a8 = 0.0f, a4 = 0.0f;
            row_f32(w, r, row);
            requant_i8(row, K, qi8, si8);
            requant_i4(row, K, qi4, si4);
            for (g = 0; g < groups; g++) {
                int32_t d8 = 0, d4 = 0;
                for (i2 = 0; i2 < 32; i2++) {
                    d8 += (int32_t)qi8[32 * g + i2] * (int32_t)xq[32 * g + i2];
                    d4 += (int32_t)qi4[32 * g + i2] * (int32_t)xq[32 * g + i2];
                }
                a8 += (float)d8 * si8[g] * xs[g];
                a4 += (float)(d4 - 8 * sums[g]) * si4[g] * xs[g];
            }
            yi8[r] = a8;
            yi4[r] = a4;
        }
        (void)G;
        metrics("i8/32 requant + q8act vs f32", yi8, yref, N);
        metrics("i4/32 requant + q8act vs f32", yi4, yref, N);
        {
            float *yim = malloc((size_t)N * 4);
            for (r = 0; r < N; r++) {
                float *row = row_scratch(K);
                int8_t qm4[8192]; float sm4[256], mm4[256];
                float acc = 0.0f;
                row_f32(w, r, row);
                requant_i4m(row, K, qm4, sm4, mm4);
                for (g = 0; g < groups; g++) {
                    int32_t d = 0;
                    for (i2 = 0; i2 < 32; i2++)
                        d += (int32_t)qm4[32 * g + i2] * (int32_t)xq[32 * g + i2];
                    acc += xs[g] * ((float)d * sm4[g] + mm4[g] * (float)sums[g]);
                }
                yim[r] = acc;
            }
            metrics("i4/32+min requant + q8act vs f32", yim, yref, N);
            free(yim);
        }
        {
            double mx = 0.0, mx4 = 0.0;
            for (r = 0; r < N; r++) {
                double d = fabs((double)yi8[r] - (double)yq4[r]);
                double e = fabs((double)yi4[r] - (double)yq4[r]);
                if (d > mx) mx = d;
                if (e > mx4) mx4 = e;
            }
            printf("%-30s maxerr=%.3e\n", "i8 vs f32w+q8act", mx);
            printf("%-30s maxerr=%.3e\n", "i4 vs f32w+q8act", mx4);
        }
        free(yi8); free(yi4);
    }
    q4k_plan_free(&plan);
    free(xq); free(xs); free(xf); free(yref); free(yq); free(yq4);
}

static void emit_case(const char *path, int rows)
{
    wmat *w = W("blk.0.ffn_gate.weight");
    int K = w->K, N = w->N, r, i;
    FILE *f;
    float *xf, *yref, *yq4, *xr;
    int8_t *xq;
    float *xs;

    if (rows > N) rows = N;
    xf = malloc((size_t)K * 4);
    {
        state_t *S = st_new();
        g_cap = xf;
        forward(S, 597, NULL);
        g_cap = NULL;
    }
    xq = xalloc(K, 128);
    xs = xalloc((size_t)(K / 32) * 4, 64);
    q4k_act_q8(xf, K, xq, xs);
    yref = malloc((size_t)rows * 4);
    yq4 = malloc((size_t)rows * 4);
    xr = malloc((size_t)K * 4);
    deq_q8(xq, xs, xr, K);
#pragma omp parallel for schedule(static)
    for (r = 0; r < rows; r++) {
        float *row = row_scratch(K);
        q4k_gemv_f32act(&w->qm, xf, yref, r, r + 1);
        row_f32(w, r, row);
        yq4[r] = dot8_f32(row, xr, K);
    }

    f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "/* generated by q4k_host --emit-case\n");
    fprintf(f, " * real blk.0.ffn_gate.weight rows 0..%d,\n"
               " * real layer-0 ffn activation (token 597 = \"The\").\n */\n",
            rows - 1);
    fprintf(f, "#ifndef Q4K_CASE_H\n#define Q4K_CASE_H\n");
    fprintf(f, "#define CASE_N %d\n#define CASE_K %d\n#define CASE_BLOCKS %d\n",
            rows, K, K / 256);
    fprintf(f, "#define CASE_NAMEPTR \"blk.0.ffn_gate.weight\"\n");
    fprintf(f, "static const unsigned char case_w[] = {");
    for (i = 0; i < rows * (int)w->row_bytes; i++) {
        if ((i % 24) == 0) fprintf(f, "\n");
        fprintf(f, "%u,", g_map[w->off + i]);
    }
    fprintf(f, "\n};\n");
    fprintf(f, "static const signed char case_xq[] = {");
    for (i = 0; i < K; i++) { if ((i % 24) == 0) fprintf(f, "\n"); fprintf(f, "%d,", xq[i]); }
    fprintf(f, "\n};\n");
    fprintf(f, "static const float case_xf[] = {");
    for (i = 0; i < K; i++) { if ((i % 5) == 0) fprintf(f, "\n"); fprintf(f, "%.9ef,", xf[i]); }
    fprintf(f, "\n};\n");
    fprintf(f, "static const float case_xs[] = {");
    for (i = 0; i < K / 32; i++) { if ((i % 5) == 0) fprintf(f, "\n"); fprintf(f, "%.9ef,", xs[i]); }
    fprintf(f, "\n};\n");
    fprintf(f, "/* y = f32 dequant of the weights x f32 activation */\n");
    fprintf(f, "static const float case_ref[] = {");
    for (i = 0; i < rows; i++) { if ((i % 5) == 0) fprintf(f, "\n"); fprintf(f, "%.9ef,", yref[i]); }
    fprintf(f, "\n};\n");
    fprintf(f, "/* y = f32 dequant of the weights x dequantised q8 activation */\n");
    fprintf(f, "static const float case_refq[] = {");
    for (i = 0; i < rows; i++) { if ((i % 5) == 0) fprintf(f, "\n"); fprintf(f, "%.9ef,", yq4[i]); }
    fprintf(f, "\n};\n");
    fprintf(f, "#endif\n");
    fclose(f);
    fprintf(stderr, "[q4k] wrote %s (%d rows, %d B raw weights)\n",
            path, rows, rows * (int)w->row_bytes);
    free(xf); free(xq); free(xs); free(yref); free(yq4); free(xr);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *model = "models/LFM3-1.7B-Q4_K_M.gguf";
    char prompt_ids[4096] = "597 5205 302 3980 355";   /* The capital of France is */
    int ntokens = 64, i, selft = 0, case_rows = 512;
    const char *case_out = NULL;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) model = argv[++i];
        else if (!strcmp(argv[i], "--mode")) {
            const char *m = argv[++i];
            if (!strcmp(m, "f32")) g_mode = M_F32;
            else if (!strcmp(m, "q4k")) g_mode = M_Q4K;
            else if (!strcmp(m, "q4kf")) g_mode = M_Q4KF;
            else if (!strcmp(m, "i8")) g_mode = M_I8;
            else if (!strcmp(m, "i4")) g_mode = M_I4;
            else if (!strcmp(m, "i4m")) g_mode = M_I4M;
            else if (!strcmp(m, "i4r")) g_mode = M_I4R;
            else { fprintf(stderr, "bad mode %s\n", m); return 2; }
        }
        else if (!strcmp(argv[i], "--ids")) { i++; snprintf(prompt_ids, sizeof(prompt_ids), "%s", argv[i]); }
        else if (!strcmp(argv[i], "--tokens")) ntokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) selft = 1;
        else if (!strcmp(argv[i], "--emit-case")) case_out = argv[++i];
        else if (!strcmp(argv[i], "--case-rows")) case_rows = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    model_init(model);

    if (selft) { selftest(); return 0; }
    if (case_out) { emit_case(case_out, case_rows); return 0; }

    {
        int ids[4096], nids = 0, t, next;
        char *p = prompt_ids;
        state_t *S;
        clock_t t0 = clock();

        while (*p && nids < 4096) {
            while (*p == ' ') p++;
            if (!*p) break;
            ids[nids++] = (int)strtol(p, &p, 10);
        }
        S = st_new();
        next = ids[0];
        for (t = 0; t < nids; t++) {
            int top = -1;
            forward(S, ids[t], &top);
            next = top;
        }
        printf("ids:");
        for (t = 0; t < ntokens; t++) {
            int top = -1;
            printf(" %d", next);
            fflush(stdout);
            forward(S, next, &top);
            next = top;
        }
        printf("\n");
        fprintf(stderr, "[q4k] mode %d, %.2f s (%.2f s/token)\n", g_mode,
                (double)(clock() - t0) / CLOCKS_PER_SEC,
                (double)(clock() - t0) / CLOCKS_PER_SEC / (nids + ntokens));
    }
    return 0;
}
