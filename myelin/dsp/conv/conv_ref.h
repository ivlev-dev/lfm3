#ifndef CONV_REF_H
#define CONV_REF_H
/*
 * Host-side references for the shortconv layer, shared by conv_sim (offline,
 * runs on the build machine) and conv_main (on-device test driver).
 *
 * Three models of the same layer:
 *   ref_f32_run()    f32 arithmetic on the *dequantised* int4 weights -> truth
 *   ref_i8_run(),    the integer pipeline with a plain 6-tap shift register
 *   ref_i8_run_y()   (y8 for every token; out_proj is a separate pass, its
 *                     scale is calibrated from the accumulator range)
 *   ref_ring_token() the integer pipeline exactly as the DSP does it:
 *                    8 ring positions, phase-rotated int8 weights, byte planes
 *
 * The integer models must agree with each other bit for bit; that is what
 * validates the ring/phase trick without a device.
 *
 * Scale calibration: s_bc / s_y from the f32 absmax (110/127 of full scale),
 * s_out from the out_proj accumulator range (ref_calib_s_out below).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "conv_math.h"

#define CONV_KSIZE 7

/* ------------------------------------------------------------------ */
/* model + weights                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    int      H, ntok;
    uint8_t *w_in;                 /* [3H][H] int4 packed */
    uint8_t *w_out;                /* [H][H]  int4 packed */
    float   *sw_in, *sw_out;       /* per-row scales */
    float   *cw;                   /* [H][7] depthwise */
    int8_t  *tok;                  /* [ntok][H] int8 activations */
    float    s_in, s_bc, s_y, s_out;
} ref_model_t;

/* quantised depthwise weights + dequant tables, as lfmgemv_conv_init builds */
typedef struct {
    int      H;
    int8_t  *wq1;                  /* [H][7] */
    int8_t  *wq8;                  /* [8][H][8] phase-rotated */
    int32_t *kq;                   /* [H] */
    int      ksh;
} ref_convw_t;

static inline uint32_t ref_lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static inline void ref_pack_int4(uint8_t *w, int8_t *ws, int n, int k, uint32_t seed)
{
    uint32_t st = seed;
    int r, i;
    for (r = 0; r < n; r++) {
        uint8_t *row = w + (size_t)r * (k / 2);
        for (i = 0; i < k; i += 2) {
            int lo = (int)(ref_lcg(&st) % 16) - 8;
            int hi = (int)(ref_lcg(&st) % 16) - 8;
            if (ws) {
                ws[(size_t)r * k + i]     = (int8_t)lo;
                ws[(size_t)r * k + i + 1] = (int8_t)hi;
            }
            row[i / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
        }
    }
}

/* int4 row dot, offset-8 form, exact integers (same value as the HVX kernel).
 * The kernel multiplies the *unsigned* nibble (0..15) by the activation and
 * applies the offset once, row-independently, through `bias`:
 *     sum(w_s*a) = sum(u*a) - 8*sum(a),   u = w_s + 8
 * Do NOT subtract 8 per element here: the callers already pass
 * bias = -8*sum(act), exactly as lfm_bridge_gemv() does on the device
 * (see lfmgemv_impl.c, gemv_rows_m1_generic).  Subtracting twice throws the
 * row output off by 8*sum(act)*sw, which is the same order as the signal. */
static inline int32_t ref_dot_int4(const uint8_t *row, const int8_t *act, int K, int32_t bias)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < K; i += 2) {
        uint8_t b = row[i >> 1];
        s += (int)(b & 0xf) * (int)act[i];
        s += (int)(b >> 4) * (int)act[i + 1];
    }
    return s + bias;
}

static inline int32_t ref_sum_i8(const int8_t *a, int K)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < K; i++) s += a[i];
    return s;
}

static inline void ref_model_free(ref_model_t *m)
{
    free(m->w_in); free(m->w_out); free(m->sw_in); free(m->sw_out);
    free(m->cw); free(m->tok);
    memset(m, 0, sizeof(*m));
}

static inline int ref_model_build(ref_model_t *m, int H, int ntok, uint32_t seed)
{
    uint32_t st = seed;
    int i, ch, j;
    memset(m, 0, sizeof(*m));
    m->H = H; m->ntok = ntok;
    m->w_in  = (uint8_t *)malloc((size_t)3 * H * H / 2);
    m->w_out = (uint8_t *)malloc((size_t)H * H / 2);
    m->sw_in = (float *)malloc((size_t)3 * H * sizeof(float));
    m->sw_out = (float *)malloc((size_t)H * sizeof(float));
    m->cw    = (float *)malloc((size_t)H * CONV_KSIZE * sizeof(float));
    m->tok   = (int8_t *)malloc((size_t)ntok * H);
    if (!m->w_in || !m->w_out || !m->sw_in || !m->sw_out || !m->cw || !m->tok) {
        ref_model_free(m);
        return -1;
    }
    ref_pack_int4(m->w_in, NULL, 3 * H, H, 0x1111u ^ seed);
    ref_pack_int4(m->w_out, NULL, H, H, 0x2222u ^ seed);
    for (i = 0; i < 3 * H; i++) m->sw_in[i] = 0.00010f + (float)(i % 13) * 0.000002f;
    for (i = 0; i < H; i++)     m->sw_out[i] = 0.00010f + (float)(i % 11) * 0.000002f;
    for (ch = 0; ch < H; ch++)
        for (j = 0; j < CONV_KSIZE; j++)
            m->cw[(size_t)ch * CONV_KSIZE + j] =
                (float)((int)(ref_lcg(&st) % 2001) - 1000) * 1.0e-4f;
    for (i = 0; i < ntok * H; i++)
        m->tok[i] = (int8_t)((int)(ref_lcg(&st) % 255) - 127);
    m->s_in = 1.0f; m->s_bc = 1.0f; m->s_y = 1.0f; m->s_out = 1.0f;
    return 0;
}

/* mirror of lfmgemv_conv_init: quantise taps, build phases, build kq/ksh */
static inline int ref_convw_init(ref_convw_t *W, const float *cw, int H, float s_bc, float s_y)
{
    double kmax = 0.0;
    int ch, j, s, p;
    memset(W, 0, sizeof(*W));
    W->H = H;
    W->wq1 = (int8_t *)malloc((size_t)H * CONV_KSIZE);
    W->wq8 = (int8_t *)malloc((size_t)8 * H * 8);
    W->kq  = (int32_t *)malloc((size_t)H * sizeof(int32_t));
    if (!W->wq1 || !W->wq8 || !W->kq) return -1;

    for (ch = 0; ch < H; ch++) {
        const float *w = cw + (size_t)ch * CONV_KSIZE;
        float m = 0.0f, sw;
        for (j = 0; j < CONV_KSIZE; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        sw = m / 127.0f;
        for (j = 0; j < CONV_KSIZE; j++)
            W->wq1[(size_t)ch * CONV_KSIZE + j] = (int8_t)conv_wq(w[j], sw);
        {
            double K = conv_K((double)m, (double)s_bc, (double)s_y);
            if (K > kmax) kmax = K;
        }
    }
    W->ksh = conv_pick_ksh(kmax);
    for (ch = 0; ch < H; ch++) {
        const float *w = cw + (size_t)ch * CONV_KSIZE;
        float m = 0.0f;
        for (j = 0; j < CONV_KSIZE; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        W->kq[ch] = conv_make_kq(conv_K((double)m, (double)s_bc, (double)s_y), W->ksh);
    }
    for (p = 0; p < 8; p++)
        for (ch = 0; ch < H; ch++)
            for (s = 0; s < 8; s++) {
                int tap = conv_slot_tap(p, s);
                W->wq8[((size_t)p * H + ch) * 8 + s] =
                    (tap < CONV_KSIZE) ? W->wq1[(size_t)ch * CONV_KSIZE + tap] : 0;
            }
    return 0;
}

static inline void ref_convw_free(ref_convw_t *W)
{
    free(W->wq1); free(W->wq8); free(W->kq);
    memset(W, 0, sizeof(*W));
}

/* ------------------------------------------------------------------ */
/* f32 reference                                                      */
/* ------------------------------------------------------------------ */

static inline void ref_f32_matvec(const uint8_t *w, const float *sw,
                                  const float *x, float *y, int N, int K)
{
    int r, i;
    for (r = 0; r < N; r++) {
        const uint8_t *row = w + (size_t)r * (K / 2);
        double s = 0.0;
        for (i = 0; i < K; i += 2) {
            uint8_t b = row[i >> 1];
            s += (double)((int)(b & 0xf) - 8) * (double)x[i];
            s += (double)((int)(b >> 4) - 8) * (double)x[i + 1];
        }
        y[r] = (float)(s * (double)sw[r]);
    }
}

typedef struct {
    float *hist;                   /* [6*H] oldest..newest */
    float *act, *bcx, *y;          /* [H], [3H], [H] */
    float *bcx_all, *y_all;        /* [ntok*3H], [ntok*H] (for calibration) */
} ref_f32_t;

/* runs the whole token sequence; out_real[ntok*H] receives the layer output */
static inline void ref_f32_run(const ref_model_t *m, const ref_f32_t *st, float *out_real)
{
    int H = m->H, t, ch, j;
    for (t = 0; t < m->ntok; t++) {
        const int8_t *a8 = m->tok + (size_t)t * H;
        for (j = 0; j < H; j++) st->act[j] = (float)a8[j] * m->s_in;
        ref_f32_matvec(m->w_in, m->sw_in, st->act, st->bcx, 3 * H, H);
        for (ch = 0; ch < H; ch++) {
            float bx = st->bcx[ch] * st->bcx[2 * H + ch];
            float acc = m->cw[(size_t)ch * CONV_KSIZE + 6] * bx;
            for (j = 0; j < 6; j++) acc += m->cw[(size_t)ch * CONV_KSIZE + j] * st->hist[j * H + ch];
            for (j = 0; j < 5; j++) st->hist[j * H + ch] = st->hist[(j + 1) * H + ch];
            st->hist[5 * H + ch] = bx;
            st->y[ch] = st->bcx[H + ch] * acc;
        }
        if (st->bcx_all) memcpy(st->bcx_all + (size_t)t * 3 * H, st->bcx, (size_t)3 * H * sizeof(float));
        if (st->y_all)   memcpy(st->y_all + (size_t)t * H, st->y, (size_t)H * sizeof(float));
        ref_f32_matvec(m->w_out, m->sw_out, st->y, out_real + (size_t)t * H, H, H);
    }
}

/* ------------------------------------------------------------------ */
/* integer pipeline, plain 6-tap shift register                       */
/* ------------------------------------------------------------------ */

/* bcx8 for one token (identical arithmetic to the int4 GEMV + pack) */
static inline void ref_bcx8(const ref_model_t *m, const int8_t *act, int8_t *bcx8)
{
    int H = m->H, r;
    int32_t bias = -8 * ref_sum_i8(act, H);
    for (r = 0; r < 3 * H; r++)
        bcx8[r] = conv_pack_i8(ref_dot_int4(m->w_in + (size_t)r * (H / 2), act, H, bias),
                               conv_row_scale(m->sw_in[r], m->s_in, m->s_bc));
}

/* out8 = out_proj(y8) with the host's output scale */
static inline void ref_out8(const ref_model_t *m, const int8_t *y8, int8_t *out8)
{
    int H = m->H, r;
    int32_t bias = -8 * ref_sum_i8(y8, H);
    for (r = 0; r < H; r++)
        out8[r] = conv_pack_i8(ref_dot_int4(m->w_out + (size_t)r * (H / 2), y8, H, bias),
                               conv_row_scale(m->sw_out[r], m->s_y, m->s_out));
}

/* one token of the shift-register pipeline: y8 only, no out_proj */
static inline void ref_i8_token(const ref_model_t *m, const ref_convw_t *W,
                                int16_t *hist6, int8_t *bcx8, int8_t *y8)
{
    int H = m->H, ch, j;
    for (ch = 0; ch < H; ch++) {
        int32_t bx = (int32_t)bcx8[ch] * (int32_t)bcx8[2 * H + ch];
        int32_t a = (int32_t)W->wq1[(size_t)ch * CONV_KSIZE + 6] * bx;
        for (j = 0; j < 6; j++)
            a += (int32_t)W->wq1[(size_t)ch * CONV_KSIZE + j] *
                 (int32_t)hist6[(size_t)j * H + ch];
        y8[ch] = conv_glue(a, bcx8[H + ch], W->kq[ch], W->ksh);
    }
    for (ch = 0; ch < H; ch++) {
        int32_t bx = (int32_t)bcx8[ch] * (int32_t)bcx8[2 * H + ch];
        for (j = 0; j < 5; j++)
            hist6[(size_t)j * H + ch] = hist6[(size_t)(j + 1) * H + ch];
        hist6[(size_t)5 * H + ch] = (int16_t)bx;
    }
}

/* y8 of every token, state carried across tokens.
 * out_proj is a separate pass: its scale s_out must be calibrated from the
 * out_proj accumulator range first (see ref_calib_s_out below).            */
static inline void ref_i8_run_y(const ref_model_t *m, const ref_convw_t *W,
                                int16_t *hist6, int8_t *bcx8, int8_t *y_all)
{
    int t;
    for (t = 0; t < m->ntok; t++) {
        ref_bcx8(m, m->tok + (size_t)t * m->H, bcx8);
        ref_i8_token(m, W, hist6, bcx8, y_all + (size_t)t * m->H);
    }
}

static inline void ref_i8_run(const ref_model_t *m, const ref_convw_t *W,
                              int16_t *hist6, int8_t *bcx8,
                              int8_t *y8, int8_t *out8)
{
    int t;
    for (t = 0; t < m->ntok; t++) {
        ref_bcx8(m, m->tok + (size_t)t * m->H, bcx8);
        ref_i8_token(m, W, hist6, bcx8, y8);
        ref_out8(m, y8, out8 + (size_t)t * m->H);
    }
}

/* ------------------------------------------------------------------ */
/* integer pipeline exactly as the DSP: 8-slot ring + phase weights    */
/* ------------------------------------------------------------------ */

typedef struct {
    int8_t *hi, *lo;               /* [H*8] byte planes of one slot */
    unsigned ctr;
} ref_ring_t;

static inline int ref_ring_init(ref_ring_t *g, int H)
{
    memset(g, 0, sizeof(*g));
    g->hi = (int8_t *)calloc((size_t)H * 8, 1);
    g->lo = (int8_t *)calloc((size_t)H * 8, 1);
    return (g->hi && g->lo) ? 0 : -1;
}
static inline void ref_ring_reset(ref_ring_t *g, int H)
{
    memset(g->hi, 0, (size_t)H * 8);
    memset(g->lo, 0, (size_t)H * 8);
    g->ctr = 0;
}
static inline void ref_ring_free(ref_ring_t *g)
{
    free(g->hi); free(g->lo);
    memset(g, 0, sizeof(*g));
}

/* out8 == NULL: single token; otherwise out8 receives out_proj(y8) */
static inline void ref_ring_token(const ref_model_t *m, const ref_convw_t *W,
                                  ref_ring_t *g, const int8_t *act,
                                  int8_t *bcx8, int8_t *y8, int8_t *out8)
{
    int H = m->H, ch, s;
    int p = conv_ring_slot(g->ctr);

    ref_bcx8(m, act, bcx8);
    for (ch = 0; ch < H; ch++) {
        int32_t bx = (int32_t)bcx8[ch] * (int32_t)bcx8[2 * H + ch];
        g->hi[(size_t)ch * 8 + p] = (int8_t)(bx >> 8);
        g->lo[(size_t)ch * 8 + p] = (int8_t)(bx & 0xff);
    }
    for (ch = 0; ch < H; ch++) {
        const int8_t *w = W->wq8 + ((size_t)p * H + ch) * 8;
        int32_t ah = 0, al = 0;
        for (s = 0; s < 8; s++) {
            ah += (int32_t)w[s] * (int32_t)g->hi[(size_t)ch * 8 + s];
            al += (int32_t)w[s] * (int32_t)(uint8_t)g->lo[(size_t)ch * 8 + s];
        }
        y8[ch] = conv_glue((ah << 8) + al, bcx8[H + ch], W->kq[ch], W->ksh);
    }
    g->ctr++;
    if (out8) ref_out8(m, y8, out8);
}

/* ------------------------------------------------------------------ */
/* metrics + scale calibration                                        */
/* ------------------------------------------------------------------ */

/* cos(a,b) over n elements after scaling a by s */
static inline double ref_cos_i8(const int8_t *a, const float *b, int n, float s)
{
    double ab = 0.0, aa = 0.0, bb = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double x = (double)a[i] * (double)s, y = (double)b[i];
        ab += x * y; aa += x * x; bb += y * y;
    }
    if (aa <= 0.0 || bb <= 0.0) return 1.0;
    return ab / (sqrt(aa) * sqrt(bb));
}

static inline double ref_maxabs_i8(const int8_t *a, const float *b, int n, float s)
{
    double m = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double d = (double)a[i] * (double)s - (double)b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

static inline double ref_rms(float *a, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += (double)a[i] * (double)a[i];
    return sqrt(s / (double)n);
}

/* ------------------------------------------------------------------ */
/* out_proj scale calibration                                         */
/* ------------------------------------------------------------------ */
/*
 * The out_proj output for row r is
 *     out8[r] = clamp(acc[r] * m_r),   m_r = sw_out[r]*s_y/s_out,
 *     acc[r]  = sum_j wq[r][j]*y8[j]   (int4 dot, exact int32)
 * so row r clips iff m_r * acc_max[r] > 127.  Deriving s_out from
 * absmax(out_f32) bounds the real output on the calibration set but not the
 * accumulator that produces it; a scale that is a few times too small then
 * clips whole rows (the original 64 % saturation).  Bound the accumulator
 * instead:
 *     s_out = max_r( sw_out[r]*s_y*acc_max[r] ) / target,  target <= 127
 * which makes max_r m_r*acc_max[r] == target for the binding row.
 * target = 110 keeps the same 13 % headroom as the s_bc / s_y calibration.
 */

/* acc_max[row] = max_t |acc| over the calibration tokens; returns the global
 * max over rows (the number that has to fit the int8 output range).
 * Per row/token: ref_dot_int4(row, y8, H, -8*sum(y8)) -- the GEMV algebra. */
static inline int32_t ref_out_acc_profile(const ref_model_t *m, const int8_t *y8_all,
                                          int ntok, int32_t *acc_max)
{
    int H = m->H, t, r;
    int32_t gmax = 0;
    for (r = 0; r < H; r++) acc_max[r] = 0;
    for (t = 0; t < ntok; t++) {
        const int8_t *y8 = y8_all + (size_t)t * H;
        int32_t bias = -8 * ref_sum_i8(y8, H);
        for (r = 0; r < H; r++) {
            int32_t a = ref_dot_int4(m->w_out + (size_t)r * (H / 2), y8, H, bias);
            if (a < 0) a = -a;
            if (a > acc_max[r]) acc_max[r] = a;
        }
    }
    for (r = 0; r < H; r++) if (acc_max[r] > gmax) gmax = acc_max[r];
    return gmax;
}

/* s_out from the accumulator range (see the block comment above). */
static inline float ref_calib_s_out(const ref_model_t *m, const int32_t *acc_max, float target)
{
    double need = 0.0;
    int r;
    for (r = 0; r < m->H; r++) {
        double s = (double)m->sw_out[r] * (double)m->s_y *
                   (double)acc_max[r] / (double)target;
        if (s > need) need = s;
    }
    return (float)need;
}

/* Invariant the calibration must satisfy: for every row m_r <= 127/acc_max[r].
 * Returns max_r m_r*acc_max[r] (int8 units, must stay <= 127) and, through
 * mrmax, max_r m_r (compare with 127/acc_max over the whole tensor).       */
static inline double ref_s_out_worst(const ref_model_t *m, const int32_t *acc_max,
                                     float s_out, double *mrmax)
{
    double worst = 0.0, mx = 0.0;
    int r;
    for (r = 0; r < m->H; r++) {
        double mr = (double)m->sw_out[r] * (double)m->s_y / (double)s_out;
        if (mr > mx) mx = mr;
        if (mr * (double)acc_max[r] > worst) worst = mr * (double)acc_max[r];
    }
    if (mrmax) *mrmax = mx;
    return worst;
}

#endif /* CONV_REF_H */
