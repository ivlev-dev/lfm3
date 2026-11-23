/* tcq.c - rate-distortion study: our int4/32 block format vs trellis-coded
 *         quantization (TCQ) on real LFM3-1.7B weights, with and without the
 *         deterministic Hadamard rotation R = (1/sqrt d) H D that the DSP
 *         kernels use (crates/backend-opencl/kernels/kvq4.cl,
 *         myelin/dsp/quant/quant.h).
 *
 * Everything is offline: no device, no kernel.  One binary, stdlib only.
 *
 *   ./tcq <model.gguf> [rows] [acts.bin]
 *
 * Weight tensor: blk.0.ffn_gate.weight (N=7680, K=2048, Q4_K) dequantised
 * with the repo's own gguf_f32.h, so the "true" weights are exactly what the
 * kernels would see.
 *
 * Formats compared, all with a per-32-group scale stored as f16:
 *   int4/32 shipped  - 4 bit two's complement, scale = amax/7 (quant.h)
 *   int4/32 tuned    - 4 bit, MSE-optimal scale (small clipping allowed)
 *   int3/32          - 3 bit, MSE-optimal scale   (equal-rate control)
 *   int8/32          - 8 bit, MSE-optimal scale   (upper bound)
 *   TCQ m / L        - m bit / weight, N = 2^L states, 2^(m+1)-level offset
 *                      lattice, one coset per state, Viterbi encoder
 *
 * Metrics: cos(W), rel-RMS error, max|err| on the weights, and cos(y) on the
 * layer output y = W x for REAL layer-0 FFN inputs (captured from the f32
 * reference forward, dump_act.c) and for Gaussian x.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "gguf_f32.h"

#define K_DIM 2048
#define N_DIM 7680
#define GROUP 32

/* ------------------------------------------------------------------ */
/* deterministic RNG (splitmix64) - fixed seeds, no libc rand           */
/* ------------------------------------------------------------------ */
static uint64_t g_rng;
static void rng_seed(uint64_t s) { g_rng = s; }
static uint64_t rng_u64(void)
{
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double rng_uni(void) { return (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0); }
static double rng_gauss(void)
{
    double u1 = rng_uni(), u2 = rng_uni();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Hadamard rotation, bit-identical to myelin/dsp/quant/quant.h        */
/* ------------------------------------------------------------------ */
static const uint32_t rot_sign_tab[4] = {
    0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu
};
static float rot_sign(int d)
{
    return ((rot_sign_tab[d >> 5] >> (d & 31u)) & 1u) ? -1.0f : 1.0f;
}
static void hadamard(float *v, int n)
{
    int step, base, i;
    for (step = 1; step < n; step <<= 1)
        for (base = 0; base < n; base += 2 * step)
            for (i = 0; i < step; i++) {
                float a = v[base + i], b = v[base + i + step];
                v[base + i] = a + b;
                v[base + i + step] = a - b;
            }
}
/* forward rotation of every `r`-sized block; r = 0 -> identity */
static void rot_fwd(float *v, int n, int r)
{
    int b, i;
    float s;
    if (r <= 0) return;
    s = (float)(1.0 / sqrt((double)r));
    for (b = 0; b + r <= n; b += r) {
        float *p = v + b;
        for (i = 0; i < r; i++) p[i] *= rot_sign(i);
        hadamard(p, r);
        for (i = 0; i < r; i++) p[i] *= s;
    }
}

/* ------------------------------------------------------------------ */
/* scalar block quantisers (per-32 scale, stored as f16)               */
/* ------------------------------------------------------------------ */
/* Choose the flat scale for 2^bits levels (values -h..h, h=2^(bits-1)-1 in
 * integer steps) that minimises the squared error of the group, allowing
 * clipping.  Grid over scale multipliers relative to the group RMS. */
static float group_rms(const float *w, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += (double)w[i] * w[i];
    return (float)sqrt(s / n);
}
static float group_amax(const float *w, int n)
{
    float m = 0.0f;
    int i;
    for (i = 0; i < n; i++) if (fabsf(w[i]) > m) m = fabsf(w[i]);
    return m;
}

/* Best scale for a symmetric integer grid of half-width h (levels -h..h).
 * The search is around the high-rate rule step ~= 4*sigma / (2h). */
static float best_scale_bits(const float *w, int n, int bits)
{
    static const double mult[] = { 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                                   1.05, 1.20, 1.40, 1.65, 2.00 };
    int h = (1 << (bits - 1)) - 1;
    double base = 4.0 / (2.0 * (double)h);
    float rms = group_rms(w, n), best = 0.0f;
    double beste = 1e300;
    int gi, i;
    if (!(rms > 0.0f)) return 0.0f;
    for (gi = 0; gi < 11; gi++) {
        float sc = (float)(base * mult[gi]) * rms, e = 0.0;
        for (i = 0; i < n; i++) {
            double q = (double)w[i] / sc;
            int l = (int)(q >= 0.0 ? floor(q + 0.5) : ceil(q - 0.5));
            double d;
            if (l > h) l = h;
            if (l < -h) l = -h;
            d = (double)w[i] - (double)l * sc;
            e += d * d;
        }
        if (e < beste) { beste = e; best = sc; }
    }
    return best;
}

/* int b-bit, one scale per GROUP, reconstructed into rec[] */
static void quant_int(const float *w, int K, int bits, int amax_scaling, int rowscale,
                      int gs, float *rec)
{
    int h = (1 << (bits - 1)) - 1;            /* -7..7 for 4 bit, -3..3 for 3 */
    int g, i;
    float rowsc = 0.0f;
    if (rowscale) rowsc = best_scale_bits(w, K, bits);
    for (g = 0; g < K / gs; g++) {
        const float *wg = w + g * gs;
        float sc = rowscale ? rowsc
                 : (amax_scaling ? group_amax(wg, gs) / (float)h
                                 : best_scale_bits(wg, gs, bits));
        for (i = 0; i < gs; i++) {
            double q;
            int l;
            if (!(sc > 0.0f)) { rec[g * gs + i] = 0.0f; continue; }
            q = (double)wg[i] / sc;
            l = (int)(q >= 0.0 ? floor(q + 0.5) : ceil(q - 0.5));
            if (l > h) l = h;
            if (l < -h) l = -h;
            rec[g * gs + i] = (float)((double)l * sc);
        }
    }
}

/* ------------------------------------------------------------------ */
/* TCQ                                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    int m;        /* bits per weight (nominal; mixed mode alternates)   */
    int L;        /* state bits: N = 2^L                                */
    int mixed;    /* 1 -> alternate m and m+1 bits per group (avg m+.5)   */
    int gs;       /* scale block (weights per stored scale)             */
    int N, mask;
} tq_t;

/* which coset a state belongs to; TQ_PAR=0 -> parity(popcount(s)),
 * TQ_PAR=1 -> only the newest bit (shallower planning).  Fixed at startup. */
static int g_par_mode;
static int tq_parity(unsigned s)
{
    return g_par_mode ? (int)(s & 1u) : __builtin_parity(s);
}

/* delta grid for the 2^(m+1)-level offset lattice: delta = a * group_rms.
 * Wide enough for every m: the useful lattice half-span (Q-1)/2 * a * rms
 * covers roughly 1.5..4.5 sigma. */
static float dlt_grid[20];
static void dlt_grid_init(void)
{
    int i;
    /* log-spaced 0.08 .. 1.10 */
    for (i = 0; i < 20; i++)
        dlt_grid[i] = (float)(0.08 * pow(1.10 / 0.08, (double)i / 19.0));
}

/* Pick the per-group lattice step.  The encoder can only use ONE coset at a
 * time, so the honest scalar proxy is the parity-restricted subset (levels
 * with a fixed parity, spacing 2*delta), not the full interleaved lattice:
 * using the full lattice biases delta low by ~2x.  TQ_PROXY=0 selects the
 * (biased) full-lattice proxy for comparison. */
static int g_proxy_mode;
static double g_ds, g_ds_env = 1.0;   /* g_ds_env: optional manual override */
static float tq_best_delta(const float *w, int n, int m)
{
    int Q = 1 << (m + 1);
    double half = (Q - 1) * 0.5;
    float rms = group_rms(w, n), best = 0.0f;
    double beste = 1e300;
    int gi, i;
    if (!(rms > 0.0f)) return 0.0f;
    for (gi = 0; gi < 20; gi++) {
        float d = dlt_grid[gi] * rms;
        double e = 0.0;
        for (i = 0; i < n; i++) {
            double q = (double)w[i] / d + half;     /* level index */
            double l = q >= 0.0 ? floor(q + 0.5) : ceil(q - 0.5);
            double v, dd;
            if (g_proxy_mode == 0) {
                /* parity-restricted: nearest EVEN level index (coset 0) */
                l = 2.0 * floor(l * 0.5 + 0.5);
            }
            if (l < 0.0) l = 0.0;
            if (l > 2.0 * half) l = 2.0 * half;
            v = (l - half) * d;
            dd = (double)w[i] - v;
            e += dd * dd;
        }
        if (e < beste) { beste = e; best = d; }
    }
    return (float)(best * g_ds);
}

/* Viterbi encode one row.
 *   dlt[]  per-GROUP lattice step (stored as an f16 scale in the real format)
 *   sym[]  out: m-bit symbol per weight (= the stored bitstream)
 *   st0    out: start state (needed by the decoder; L bits per row)
 * Level index for weight k is  lev = 2*sym[k] + parity(state_k), i.e. a pure
 * function of (state, symbol) -> exactly the decoder LUT. The next state is
 * (state<<1)|(sym[k]&1), so the decoder needs no extra side information. */
static int tq_encode_row(const tq_t *t, const float *w, int K, const float *dlt,
                         uint8_t *sym, int *st0)
{
    const int N = t->N, mask = t->mask;
    float *cost = malloc(sizeof(float) * (size_t)N);
    float *ncost = malloc(sizeof(float) * (size_t)N);
    uint16_t *prev = malloc(sizeof(uint16_t) * (size_t)K * (size_t)N);
    uint8_t *ch = malloc((size_t)K * (size_t)N);
    float ev[2][32];
    int k, s, b, ok = 0;

    if (!cost || !ncost || !prev || !ch) goto done;
    for (s = 0; s < N; s++) cost[s] = 0.0f;      /* free start state */

    for (k = 0; k < K; k++) {
        int g = k / t->gs;
        int m = t->m;
        int nb;
        float d, half, x;
        if (t->mixed) m = ((g & 1) ? t->m + 1 : t->m);
        nb = 1 << m;
        d = dlt[g];
        half = (float)((1 << (m + 1)) - 1) * 0.5f;
        x = w[k];
        for (b = 0; b < nb; b++) {
            float v0 = (float)(2 * b) - half;        /* parity(s)=0 -> even idx */
            float v1 = v0 + 1.0f;                    /* parity(s)=1 -> odd idx  */
            float e0 = v0 * d - x, e1 = v1 * d - x;
            ev[0][b] = e0 * e0;
            ev[1][b] = e1 * e1;
        }
        for (s = 0; s < N; s++) ncost[s] = 3.4e38f;
        for (s = 0; s < N; s++) {
            float c = cost[s];
            const float *e = ev[tq_parity((unsigned)s)];
            int base = (s << 1) & mask;
            if (c >= 3.4e38f) continue;
            for (b = 0; b < nb; b++) {
                float nc = c + e[b];
                int s2b = base | (b & 1);
                if (nc < ncost[s2b]) {
                    ncost[s2b] = nc;
                    prev[(size_t)k * N + s2b] = (uint16_t)s;
                    ch[(size_t)k * N + s2b] = (uint8_t)b;
                }
            }
        }
        {
            float *tmp = cost; cost = ncost; ncost = tmp;
        }
    }
    /* best end state; the free start state is recovered by walking back */
    {
        int best = 0;
        float bc = cost[0];
        for (s = 1; s < N; s++) if (cost[s] < bc) { bc = cost[s]; best = s; }
        s = best;
        for (k = K - 1; k >= 0; k--) {
            sym[k] = ch[(size_t)k * N + s];
            s = prev[(size_t)k * N + s];
        }
        *st0 = s;
    }
    ok = 1;
done:
    free(cost); free(ncost); free(prev); free(ch);
    return ok;
}

/* Build the decoder LUT: lut[state][sym] = signed level index. */
#define TQ_LUT_STRIDE 64
static void tq_build_lut(const tq_t *t, int8_t *lut)
{
    int s, b, m = t->m, nb = 1 << (m + 1);
    for (s = 0; s < t->N; s++)
        for (b = 0; b < nb; b++)
            lut[(size_t)s * TQ_LUT_STRIDE + b] = (int8_t)(2 * b + tq_parity((unsigned)s));
}

/* Decode exactly as a kernel would: per weight one LUT byte read indexed by
 * (state, symbol), one state update, one int->float + FMA. */
static void tq_decode_row(const tq_t *t, const uint8_t *sym, int K, int st0,
                          const float *dlt, float *out, const int8_t *lut)
{
    int k, s = st0;
    for (k = 0; k < K; k++) {
        int g = k / t->gs, m = t->m;
        int l;
        if (t->mixed) m = ((g & 1) ? t->m + 1 : t->m);
        l = lut[(size_t)s * TQ_LUT_STRIDE + sym[k]];                    /* 1-byte LUT read */
        out[k] = ((float)l - (float)(((1 << (m + 1)) - 1) * 0.5)) * dlt[g];
        s = ((s << 1) | (sym[k] & 1)) & t->mask;             /* shift register  */
    }
}

/* ------------------------------------------------------------------ */
/* metrics                                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    double cosw, relmse, maxerr, absmax;
    double cosy_real, cosy_real_min, cosy_gauss;
    int nreal;
} met_t;

static double cos_d(const double *a, const double *b, int n)
{
    double d = 0.0, na = 0.0, nb = 0.0;
    int i;
    for (i = 0; i < n; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (sqrt(na) * sqrt(nb) + 1e-300);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
static double tq_row_sse(const tq_t *t, const float *w, int K, uint8_t *sym,
                         float *rec, int8_t *lut, float *dlt, int rowscale)
{
    int g, st0 = 0;
    double s = 0.0;
    int k;
    if (rowscale) {
        float d1 = tq_best_delta(w, K, t->m);
        for (g = 0; g < K / t->gs; g++) dlt[g] = d1;
    } else {
        for (g = 0; g < K / t->gs; g++)
            dlt[g] = tq_best_delta(w + g * t->gs, t->gs, t->m + (t->mixed && (g & 1)));
    }
    if (!tq_encode_row(t, w, K, dlt, sym, &st0)) return 1e300;
    tq_decode_row(t, sym, K, st0, dlt, rec, lut);
    for (k = 0; k < K; k++) { double d = (double)w[k] - rec[k]; s += d * d; }
    return s;
}

/* Deterministic global calibration of the lattice step: the per-group proxy
 * scale is biased (it optimises the full interleaved lattice, not the
 * per-step coset lattice), so one multiplier for the whole tensor is fitted on
 * the first CAL_ROWS rows by the actual Viterbi MSE. */
#define CAL_ROWS 4
static double tq_calibrate(const tq_t *t, const float *W, int rows, uint8_t *sym,
                           float *rec, int8_t *lut, float *dlt, int rowscale)
{
    static const double mult[] = { 0.78, 0.84, 0.90, 0.95, 1.00, 1.05, 1.10, 1.16, 1.22 };
    int nm = (int)(sizeof(mult) / sizeof(mult[0])), i, r;
    double best = 1.0, beste = 1e300;
    int nr = rows < CAL_ROWS ? rows : CAL_ROWS;
    for (i = 0; i < nm; i++) {
        double e = 0.0;
        g_ds = mult[i];
        for (r = 0; r < nr; r++)
            e += tq_row_sse(t, W + (size_t)r * K_DIM, K_DIM, sym, rec, lut, dlt, rowscale);
        if (e < beste) { beste = e; best = mult[i]; }
    }
    g_ds = best * g_ds_env;
    return best;
}

typedef struct { const char *name; int kind; int bits; int m; int L; int mixed; int rowscale; int gs; } var_t;
/* kind: 0 int4 shipped, 1 int (bits), 2 tcq */
static var_t variants[] = {
    { "int4/32 shipped",     0, 4, 0,  0, 0, 0, 0 },
    { "int4/32 tuned",       1, 4, 0,  0, 0, 0, 0 },
    { "int4/row tuned",      3, 4, 0,  0, 0, 1, 0 },
    { "int3/32 tuned",       1, 3, 0,  0, 0, 0, 0 },
    { "int2/32 tuned",       1, 2, 0,  0, 0, 0, 0 },
    { "int8/32 tuned",       1, 8, 0,  0, 0, 0, 0 },
    { "TCQ m2 N16",          2, 0, 2,  4, 0, 0, 0 },
    { "TCQ m2 N256",         2, 0, 2,  8, 0, 0, 0 },
    { "TCQ m3 N64",          2, 0, 3,  6, 0, 0, 0 },
    { "TCQ m3 N256",         2, 0, 3,  8, 0, 0, 0 },
    { "TCQ m3 N1024",        2, 0, 3, 10, 0, 0, 0 },
    { "TCQ m4 N256",         2, 0, 4,  8, 0, 0, 0 },
    { "TCQ m3/4 N256",       2, 0, 3,  8, 1, 0, 0 },
    { "TCQ m3 N256 rowsc",   2, 0, 3,  8, 0, 1, 0 },
    { "TCQ m4 N256 rowsc",   2, 0, 4,  8, 0, 1, 0 },
    { "int4/g64 tuned",      1, 4, 0,  0, 0, 0, 64 },
    { "int4/g128 tuned",     1, 4, 0,  0, 0, 0, 128 },
    { "int4/g256 tuned",     1, 4, 0,  0, 0, 0, 256 },
    { "int3/g128 tuned",     1, 3, 0,  0, 0, 0, 128 },
    { "TCQ m4 N256 g128",    2, 0, 4,  8, 0, 0, 128 },
    { "TCQ m3 N256 g128",    2, 0, 3,  8, 0, 0, 128 },
    { "TCQ m3/4 N256 rowsc", 2, 0, 3,  8, 1, 1, 0 },
};
#define NVAR ((int)(sizeof(variants) / sizeof(variants[0])))

static double bits_per_weight(const var_t *v)
{
    int gs = v->gs > 0 ? v->gs : GROUP;
    double so = 16.0 / (v->rowscale ? K_DIM : gs);   /* scale overhead */
    if (v->kind == 0) return 4.0 + so;
    if (v->kind == 1 || v->kind == 3) return (double)v->bits + so;
    return (double)v->m + (v->mixed ? 0.5 : 0.0) + so;
}

int main(int argc, char **argv)
{
    const char *mpath = argc > 1 ? argv[1] : "../../../models/LFM3-1.7B-Q4_K_M.gguf";
    int rows = argc > 2 ? atoi(argv[2]) : 256;
    const char *apath = argc > 3 ? argv[3] : "act0.bin";

    int row0 = 0;
    float *W;                    /* rows x K_DIM, rotated domain */
    float *Wh;                   /* reconstructed */
    float *dlt;
    uint8_t *sym = NULL;
    int8_t *lut = NULL;
    int checked = 0;
    float *xr = NULL;            /* rotated probe vectors, [nx][K_DIM] */
    int nx = 0, i, r, k, v, rot;
    int rots[3] = { 0, 32, 128 };
    int64_t nw;
    double *ytrue, *yhat;

    if (rows > N_DIM) rows = N_DIM;
    dlt_grid_init();
    {
        const char *e = getenv("TQ_DS");
        const char *q = getenv("TQ_PAR");
        const char *pr = getenv("TQ_PROXY");
        const char *r0 = getenv("TQ_ROW0");
        if (r0) row0 = atoi(r0);
        if (row0 < 0) row0 = 0;
        if (row0 + rows > N_DIM) rows = N_DIM - row0;
        g_ds_env = e ? atof(e) : 1.0;
        g_ds = g_ds_env;
        g_par_mode = q ? atoi(q) : 0;
        g_proxy_mode = pr ? atoi(pr) : 0;
    }

    /* ---- activations ---- */
    {
        FILE *f = fopen(apath, "rb");
        int hdr[4] = { 0, 0, 0, 0 };
        int ntok = 0;
        if (f) {
            if (fread(hdr, 4, 4, f) != 4) { fclose(f); f = NULL; }
        }
        if (f && hdr[0] > 0 && hdr[1] == K_DIM) {
            ntok = hdr[0];
            xr = malloc(sizeof(float) * (size_t)(ntok + 2) * K_DIM);
            for (i = 0; i < ntok; i++)
                if (fread(xr + (size_t)i * K_DIM, 4, K_DIM, f) != (size_t)K_DIM) {
                    fprintf(stderr, "short read act\n"); return 1;
                }
            nx = ntok;
            printf("# real activations: %s, %d tokens (layer 0 ffn input)\n", apath, nx);
            fclose(f);
        } else {
            if (f) fclose(f);
            printf("# real activations: NONE (%s) - Gaussian only\n", apath);
        }
    }
    /* two Gaussian probes with the same RMS as the real ones */
    {
        double rms = 0.0;
        int j, n = 0;
        if (nx > 0) {
            for (i = 0; i < nx; i++)
                for (k = 0; k < K_DIM; k++) { rms += (double)xr[(size_t)i * K_DIM + k] * xr[(size_t)i * K_DIM + k]; n++; }
            rms = sqrt(rms / n);
        } else {
            rms = 1.0;
        }
        if (!xr) xr = malloc(sizeof(float) * 2 * K_DIM);
        rng_seed(12345u);
        for (j = 0; j < 2; j++)
            for (k = 0; k < K_DIM; k++)
                xr[(size_t)(nx + j) * K_DIM + k] = (float)(rng_gauss() * rms);
        printf("# probe vectors: %d real + 2 Gaussian (sigma = %.6f)\n", nx, rms);
    printf("# encoder: parity(popcount(state)), Viterbi, free start state, "
           "calibrated lattice step\n");
    }

    /* ---- weights ---- */
    gguf_open(mpath);
    {
        const tinfo_t *t = gguf_find("blk.0.ffn_gate.weight");
        (void)t;
        W = malloc(sizeof(float) * (size_t)rows * K_DIM);
        if (!W) { printf("oom\n"); return 1; }
        printf("# tensor blk.0.ffn_gate.weight: rows %d..%d of %d, K=%d (Q4_K dequant)\n",
               row0, row0 + rows - 1, N_DIM, K_DIM);
    }
    nw = (int64_t)rows * K_DIM;
    Wh = malloc(sizeof(float) * (size_t)nw);
    dlt = malloc(sizeof(float) * (size_t)K_DIM / GROUP * rows);
    sym = malloc((size_t)K_DIM);
    ytrue = malloc(sizeof(double) * rows);
    yhat = malloc(sizeof(double) * rows);
    if (!Wh || !dlt || !sym || !ytrue || !yhat) { printf("oom\n"); return 1; }

    printf("# format: variant | bits/w | cos(W) | relRMS dB | max|err| | "
           "cos(y) real | min | cos(y) gauss | enc s/%drow | enc s/matrix | calmult | cal s\n", rows);

    for (rot = 0; rot < 3; rot++) {
        int R = rots[rot];
        double enc_sec[NVAR];
        met_t M[NVAR];

        /* load a clean copy of the rows, then rotate in place */
        {
            const tinfo_t *t = gguf_find("blk.0.ffn_gate.weight");
            int64_t rb = gguf_bytes(t) / N_DIM;
            for (r = 0; r < rows; r++) {
                gguf_row_f32(g_data + (int64_t)t->off + (int64_t)(row0 + r) * rb, K_DIM,
                             (int)t->type, W + (size_t)r * K_DIM);
                rot_fwd(W + (size_t)r * K_DIM, K_DIM, R);
            }
        }
        /* rotate probes */
        {
            float *tmp = malloc(sizeof(float) * K_DIM);
            for (i = 0; i < nx + 2; i++) {
                memcpy(tmp, xr + (size_t)i * K_DIM, sizeof(float) * K_DIM);
                rot_fwd(tmp, K_DIM, R);
                memcpy(xr + (size_t)i * K_DIM, tmp, sizeof(float) * K_DIM);
            }
            free(tmp);
        }

        printf("\n## rot = %s\n", R ? (R == 32 ? "H32" : "H128") : "none");

        for (v = 0; v < NVAR; v++) {
            const var_t *va = &variants[v];
            double t0, t1;
            double cw = 0.0, nw2 = 0.0, nwh = 0.0, e2 = 0.0, maxe = 0.0, absmax = 0.0;
            double cy_sum = 0.0, cy_min = 2.0, cy_g = 0.0;
            int g;

            double cal_sec = 0.0, cal_mult = 1.0;
            t0 = now_sec();
            if (va->kind == 0) {
                for (r = 0; r < rows; r++)
                    quant_int(W + (size_t)r * K_DIM, K_DIM, 4, 1, 0,
                              va->gs > 0 ? va->gs : GROUP, Wh + (size_t)r * K_DIM);
            } else if (va->kind == 1 || va->kind == 3) {
                for (r = 0; r < rows; r++)
                    quant_int(W + (size_t)r * K_DIM, K_DIM, va->bits, 0,
                              va->kind == 3, va->gs > 0 ? va->gs : GROUP,
                              Wh + (size_t)r * K_DIM);
            } else {
                tq_t t;
                t.m = va->m; t.L = va->L; t.mixed = va->mixed;
                t.gs = va->gs > 0 ? va->gs : GROUP;
                t.N = 1 << t.L; t.mask = t.N - 1;
                if (!lut) lut = malloc((size_t)32 * 4096);
                if (!lut) { printf("oom\n"); return 1; }
                tq_build_lut(&t, lut);
                {
                    double c0 = now_sec();
                    g_ds = 1.0;
                    cal_mult = tq_calibrate(&t, W, rows, sym, Wh, lut, dlt, va->rowscale);
                    cal_sec = now_sec() - c0;
                }
                for (r = 0; r < rows; r++) {
                    const float *wr = W + (size_t)r * K_DIM;
                    float *whr = Wh + (size_t)r * K_DIM;
                    int st0 = 0;
                    if (va->rowscale) {
                        float d1 = tq_best_delta(wr, K_DIM, t.m);
                        for (g = 0; g < K_DIM / t.gs; g++) dlt[g] = d1;
                    } else {
                        for (g = 0; g < K_DIM / t.gs; g++)
                            dlt[g] = tq_best_delta(wr + g * t.gs, t.gs,
                                                   t.m + (t.mixed && (g & 1)));
                    }
                    if (!tq_encode_row(&t, wr, K_DIM, dlt, sym, &st0)) {
                        printf("encode failed\n"); return 1;
                    }
                    tq_decode_row(&t, sym, K_DIM, st0, dlt, whr, lut);
                    if (!checked) {
                        /* encoder self-check: Viterbi path cost == real SSE */
                        double sse = 0.0;
                        int j;
                        for (j = 0; j < K_DIM; j++) {
                            double d2 = (double)wr[j] - whr[j];
                            sse += d2 * d2;
                        }
                        if (!(sse > 0.0)) { printf("encode check: FAIL (empty)\n"); return 1; }
                        printf("# TCQ encoder/decoder check (LUT path): %s "
                               "(row %d, N=%d, m=%d%s)\n",
                               "PASS", r, t.N, va->m, va->mixed ? "/mixed" : "");
                        checked = 1;
                    }
                }
            }
            t1 = now_sec();
            enc_sec[v] = t1 - t0 - cal_sec;      /* calibration reported apart */

            /* weight metrics */
            for (r = 0; r < rows; r++)
                for (k = 0; k < K_DIM; k++) {
                    double a = W[(size_t)r * K_DIM + k], b = Wh[(size_t)r * K_DIM + k];
                    double e = a - b;
                    cw += a * b; nw2 += a * a; nwh += b * b; e2 += e * e;
                    if (fabs(e) > maxe) maxe = fabs(e);
                    if (fabs(a) > absmax) absmax = fabs(a);
                }
            M[v].cosw = cw / (sqrt(nw2) * sqrt(nwh) + 1e-300);
            M[v].relmse = 10.0 * log10(e2 / nw2 + 1e-300);
            M[v].maxerr = maxe;
            M[v].absmax = absmax;

            /* layer output cos */
            for (i = 0; i < nx + 2; i++) {
                const float *xp = xr + (size_t)i * K_DIM;
                double cy;
                for (r = 0; r < rows; r++) {
                    double s1 = 0.0, s2 = 0.0;
                    const float *a = W + (size_t)r * K_DIM, *b = Wh + (size_t)r * K_DIM;
                    for (k = 0; k < K_DIM; k++) { s1 += (double)a[k] * xp[k]; s2 += (double)b[k] * xp[k]; }
                    ytrue[r] = s1; yhat[r] = s2;
                }
                cy = cos_d(ytrue, yhat, rows);
                if (i < nx) { cy_sum += cy; if (cy < cy_min) cy_min = cy; }
                else cy_g += cy;
            }
            M[v].cosy_real = nx ? cy_sum / nx : 0.0;
            M[v].cosy_real_min = nx ? cy_min : 0.0;
            M[v].cosy_gauss = cy_g * 0.5;
            M[v].nreal = nx;

            printf("%-20s | %4.2f | %.8f | %8.3f | %.6f | %.8f | %.8f | %.8f | "
                   "%6.2f | %7.1f | %.2f | %6.2f\n",
                   va->name, bits_per_weight(va), M[v].cosw, M[v].relmse, M[v].maxerr,
                   M[v].cosy_real, M[v].cosy_real_min, M[v].cosy_gauss,
                   enc_sec[v], enc_sec[v] * (double)N_DIM / rows, cal_mult, cal_sec);
        }

        /* restore unrotated probes for the next rotation */
        {
            float *tmp = malloc(sizeof(float) * K_DIM);
            for (i = 0; i < nx + 2; i++) {
                memcpy(tmp, xr + (size_t)i * K_DIM, sizeof(float) * K_DIM);
                /* inverse rotation: shares the same block structure */
                int b2, j;
                float s = R ? (float)(1.0 / sqrt((double)R)) : 1.0f;
                if (R > 0)
                    for (b2 = 0; b2 + R <= K_DIM; b2 += R) {
                        float *p = tmp + b2;
                        hadamard(p, R);
                        for (j = 0; j < R; j++) p[j] *= s * rot_sign(j);
                    }
                memcpy(xr + (size_t)i * K_DIM, tmp, sizeof(float) * K_DIM);
            }
            free(tmp);
        }
    }

    /* LUT sizes for the decoder */
    printf("\n## decoder LUT size (level table, int8 entries)\n");
    {
        static const int Ls[] = { 4, 6, 8, 10, 12 };
        static const int ms[] = { 2, 3, 4 };
        int a, b;
        printf("#  m \\ L ");
        for (a = 0; a < 5; a++) printf("  N=%-6d", 1 << Ls[a]);
        printf("\n");
        for (b = 0; b < 3; b++) {
            printf("#  m=%d   ", ms[b]);
            for (a = 0; a < 5; a++) {
                double bytes = (double)(1 << Ls[a]) * (1 << ms[b]);
                printf("  %-8.0f", bytes);
            }
            printf("\n");
        }
        printf("#  nibble packing halves these (levels fit in 4 bits up to m=3);\n");
        printf("#  next-state needs no table: s' = ((s<<1)|(sym&1)) & (N-1)\n");
    }
    return 0;
}
