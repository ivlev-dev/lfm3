#ifndef FFN_REF_H
#define FFN_REF_H
/*
 * Host-side references for the one-RPC FFN layer, shared by ffn_sim (offline,
 * runs on the build machine) and ffn_main (on-device test driver).
 *
 * The layer is gate/up int4 GEMV -> SiLU(gate)*up -> int8 mid -> down int4
 * GEMV -> int8 out.  Three models of the same computation:
 *
 *   ffn_int_mid()/ffn_int_out()
 *       the DSP pipeline, expression for expression (deq_wide, silu_qk,
 *       prod_p, requant_mid, pack_out_i8), so a device run must match bit for
 *       bit; ffn_int_mid_mode() adds the two legacy dequantisers used only by
 *       the regression gates.
 *   ffn_ref_token(..., quantised=0)
 *       f32 arithmetic on the *dequantised* int4 weights with the original
 *       activation, i.e. the truth for the numerics of that layer;
 *   ffn_ref_token(..., quantised=1)
 *       the same f32 truth on the quantised activation (x8*s_in); comparing
 *       against it isolates the mid-buffer requant from the activation int8
 *       step.
 *
 * Units.  silu_q16() takes Q16 of the real gate, and the gate/up dequantiser
 * has to keep
 *     g = acc * sw * s_in = acc * m / 2^16,
 * with m = sw*sa/so (sa = 65536*s_in, so = 1) -- the same m as the kernel.
 * On real LFM3 weights m is 0.3..12 for int8 and 6..215 for int4 per-row
 * scales, so a Q16 multiplier clamps and a Q30 int32 field overflows.  The
 * kernel therefore stores m/2^k as Q16.16 and takes acc*m in 64 bits:
 *     z = acc*m/2^k = g*2^(16-k)   (deq_wide(), k <= 8).
 * k is chosen from the worst-case |acc| <= 127*|w|max*K and the largest m so
 * that z stays below 2^30; on the LFM3 tensors it is 0.  silu_qk() consumes
 * Q(16-k) directly (k = 0 is bit-identical to silu_q16).
 *
 * The down projection reuses the mid buffer, so its dequantiser sees an int8
 * activation with scale s_mid: sa = s_mid, so = s_out,
 *     s_out = max_r(swd[r]*s_mid*acc_max[r]) / 110
 * (the same accumulator calibration as the shortconv stand, conv_ref.h).
 *
 * Calibration (ffn_calibrate):
 *   s_in  = absmax(xr) / 110          (int8 full scale with 13 % headroom)
 *   s_mid = absmax(P)  / 127          (measured P range; no headroom, the
 *                                      requant itself is the object of study)
 *   s_out from the down accumulator, 110/127 of full scale.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "silu_lut.h"

#define FFN_CAL_TARGET 110.0
#define FFN_Q16_CLAMP  0.99997f
#define FFN_Q16_ONE    65536.0f

typedef struct {
    int      K, NH, ntok;
    uint8_t *wg, *wu, *wd;         /* int4 packed, row-major [N][K/2] */
    float   *swg, *swu, *swd;      /* per-row weight scales */
    int32_t *qmg, *qmu;            /* Q16.16 m/2^k for gate/up */
    int      kg, ku;               /* widening shift of the gate/up dequant */
    float   *x;                    /* [ntok][K] real input */
    int8_t  *x8;                   /* [ntok][K] quantised input */
    int32_t *accg, *accu, *accd;   /* [NH]/[NH]/[K] scratch */
    float   *p;                    /* [NH] scratch, P = silu(gate)*up */
    float    s_in, s_mid, s_out;
} ffn_model_t;

/* ------------------------------------------------------------------ */
/* integer primitives, copied from lfmgemv_impl.c                     */
/* ------------------------------------------------------------------ */

/* (a*b)>>16 with the low 8 bits of a dropped, exactly as the DSP does it.
 * The low partial product is signed: silu_range() passes a negative b (= up),
 * and the unsigned form the DSP used to have returned ~2^24 for it. */
static inline int32_t ffn_mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((int64_t)lo * (int64_t)b) >> 8);
}

/* SiLU in Q16: 1025-point LUT over [-8,8], step 1/64, linear interpolation */
static inline int32_t ffn_silu_q16(int32_t zq)
{
    int32_t off, idx, frac, a, b;
    if (zq > 8 * 65536) zq = 8 * 65536;
    if (zq < -8 * 65536) zq = -8 * 65536;
    off = zq + 8 * 65536;
    idx = off >> 10;
    frac = off & 1023;
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> 10);
}

/* SiLU in Q(16-k): the arg grid is 1/64 = 1024 in Q16 and 1024>>k in
 * Q(16-k).  k = 0 is bit-identical to ffn_silu_q16(). */
static inline int32_t ffn_silu_qk(int32_t zq, int k)
{
    int32_t lim, step, off, idx, frac, a, b;
    if (k <= 0) return ffn_silu_q16(zq);
    lim  = 8 << (16 - k);
    step = 1024 >> k;
    if (zq > lim) zq = lim;
    if (zq < -lim) zq = -lim;
    off = zq + lim;
    idx = off >> (10 - k);
    frac = off & (step - 1);
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> (10 - k));
}

/* mirror of deq_wide()/deq_shift_for() in lfmgemv_impl.c */
static inline int32_t ffn_deq_wide(int32_t acc, int32_t qm)
{
    return (int32_t)(((int64_t)acc * (int64_t)qm + 32768) >> 16);
}

static inline int ffn_deq_shift(double mmax, int K, int wbits)
{
    double bound = (double)(1 << wbits) * 127.0 * (double)K;
    int k = 0;
    while (k < 8 &&
           (bound * mmax / (double)(1u << k) > 1073741824.0 ||
            mmax / (double)(1u << k) > 32767.0))
        k++;
    return k;
}

static inline int32_t ffn_qm_of(double m, int k)
{
    int64_t q = (int64_t)(m / (double)(1u << k) * 65536.0 + 0.5);
    if (q > 0x7fffffffLL) q = 0x7fffffffLL;
    if (q < 0) q = 0;
    return (int32_t)q;
}

static inline int32_t ffn_prod_q16(int32_t sv, int32_t zu, int k)
{
    int64_t v = (int64_t)sv * (int64_t)zu + ((int64_t)1 << (15 - k));
    v >>= (16 - k);
    if (v > 0x7fffffffLL) v = 0x7fffffffLL;
    if (v < -0x7fffffffLL) v = -0x7fffffffLL;
    return (int32_t)v;
}

/* mid-buffer requant, same expression as requant_mid() in lfmgemv_impl.c:
 * pr is Q16 of P, mq_mid = 4096/s_mid, P/s_mid = pr*mq_mid/2^28.  The
 * product is taken in 64 bits -- the regression this stand guards. */
static inline int32_t ffn_requant_mid(int32_t pr, int32_t mq_mid)
{
    int64_t v = ((int64_t)pr * (int64_t)mq_mid + ((int64_t)1 << 27)) >> 28;
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int32_t)v;
}

/* mirror of lfmgemv_set_scale(): Q30 multiplier for the scalar SiLU path
 * (legacy Q16 format, kept only for the regression gates) */
static inline int32_t ffn_q30_of(float m)
{
    int64_t q;
    if (m < 0.0f) m = 0.0f;
    if (m > FFN_Q16_CLAMP) m = FFN_Q16_CLAMP;
    q = (int64_t)((double)m * 1073741824.0 + 0.5);
    if (q > 0x7fffffffLL) q = 0x7fffffffLL;
    return (int32_t)q;
}

/* legacy gate/up dequant: Q16 multiplier clamped at 0.99997, then mulq16 */
static inline int32_t ffn_deq_legacy(int32_t acc, double m)
{
    return ffn_mulq16(acc, ffn_q30_of((float)m) >> 14);
}

/* the pre-9892795 low partial product: lo is uint32_t, so a negative b is
 * converted to unsigned and the lane blows up to ~2^24 instead of going
 * negative.  Regression gate only. */
static inline int32_t ffn_mulq16_unsigned(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((uint32_t)lo * (uint32_t)b) >> 8);
}

/* mirror of lfmgemv_set_scale(): Q16 multiplier used by pack_out_i8() */
static inline int32_t ffn_qp16_of(float m)
{
    unsigned q;
    if (m < 0.0f) m = 0.0f;
    if (m > FFN_Q16_CLAMP) m = FFN_Q16_CLAMP;
    q = (unsigned)(m * 65536.0f + 0.5f);
    if (q > 65535u) q = 65535u;
    return (int32_t)q;
}

/* exact int32 int4 dot in the kernel's offset-8 algebra:
 * sum(w_s*a) = sum(u*a) - 8*sum(a); callers pass bias = -8*sum(act). */
static inline int32_t ffn_dot_i4(const uint8_t *row, const int8_t *act, int K, int32_t bias)
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

static inline int32_t ffn_sum_i8(const int8_t *a, int K)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < K; i++) s += a[i];
    return s;
}

/* f32 dot with the dequantised int4 row: sum((u-8)*sw*act) */
static inline double ffn_dot_i4_f32(const uint8_t *row, const float *act, int K, double sw)
{
    double s = 0.0;
    int i;
    for (i = 0; i < K; i += 2) {
        uint8_t b = row[i >> 1];
        s += (double)((int)(b & 0xf) - 8) * (double)act[i];
        s += (double)((int)(b >> 4) - 8) * (double)act[i + 1];
    }
    return s * sw;
}

static inline double ffn_silu_f32(double g)
{
    return g / (1.0 + exp(-g));
}

static inline double ffn_cos_f32(const float *a, const float *b, int n)
{
    double ab = 0.0, aa = 0.0, bb = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double x = (double)a[i], y = (double)b[i];
        ab += x * y; aa += x * x; bb += y * y;
    }
    if (aa <= 0.0 || bb <= 0.0) return 1.0;
    return ab / (sqrt(aa) * sqrt(bb));
}

static inline double ffn_maxabs_f32(const float *a, const float *b, int n)
{
    double m = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

/* cos / maxabs of an int8 buffer against f32 truth, scale s */
static inline double ffn_cos_i8(const int8_t *a, const float *b, int n, float s)
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

static inline double ffn_maxabs_i8(const int8_t *a, const float *b, int n, float s)
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

static inline int ffn_count_clip(const int8_t *a, int n)
{
    int i, c = 0;
    for (i = 0; i < n; i++) if (a[i] >= 127 || a[i] <= -127) c++;
    return c;
}

/* ------------------------------------------------------------------ */
/* synthetic model: random int4 weights, real-valued input            */
/* ------------------------------------------------------------------ */

static inline uint32_t ffn_lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static inline void ffn_pack_i4_random(uint8_t *w, int n, int k, uint32_t seed)
{
    uint32_t st = seed;
    int r, i;
    for (r = 0; r < n; r++) {
        uint8_t *row = w + (size_t)r * (k / 2);
        for (i = 0; i < k; i += 2) {
            int lo = (int)(ffn_lcg(&st) % 16) - 8;
            int hi = (int)(ffn_lcg(&st) % 16) - 8;
            row[i / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
        }
    }
}

/* real f32 row -> int4, per-row absmax/7, rounded, codes stored offset-8 */
static inline float ffn_pack_i4_row(uint8_t *dst, const float *row, int k)
{
    float amax = 0.0f, sw;
    int i;
    for (i = 0; i < k; i++) {
        float a = row[i] < 0.0f ? -row[i] : row[i];
        if (a > amax) amax = a;
    }
    sw = amax > 0.0f ? amax / 7.0f : 1.0f;
    for (i = 0; i < k; i += 2) {
        int lo = (int)lrintf(row[i] / sw);
        int hi = (int)lrintf(row[i + 1] / sw);
        if (lo > 7) lo = 7;
        if (lo < -8) lo = -8;
        if (hi > 7) hi = 7;
        if (hi < -8) hi = -8;
        dst[i / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
    }
    return sw;
}

static inline int ffn_model_alloc(ffn_model_t *m, int K, int NH, int ntok)
{
    memset(m, 0, sizeof(*m));
    m->K = K; m->NH = NH; m->ntok = ntok;
    m->wg  = (uint8_t *)malloc((size_t)NH * (K / 2));
    m->wu  = (uint8_t *)malloc((size_t)NH * (K / 2));
    m->wd  = (uint8_t *)malloc((size_t)K * (NH / 2));
    m->swg = (float *)malloc((size_t)NH * sizeof(float));
    m->swu = (float *)malloc((size_t)NH * sizeof(float));
    m->swd = (float *)malloc((size_t)K * sizeof(float));
    m->qmg = (int32_t *)calloc((size_t)NH, sizeof(int32_t));
    m->qmu = (int32_t *)calloc((size_t)NH, sizeof(int32_t));
    m->x   = (float *)malloc((size_t)ntok * K * sizeof(float));
    m->x8  = (int8_t *)malloc((size_t)ntok * K);
    m->accg = (int32_t *)malloc((size_t)NH * sizeof(int32_t));
    m->accu = (int32_t *)malloc((size_t)NH * sizeof(int32_t));
    m->accd = (int32_t *)malloc((size_t)K * sizeof(int32_t));
    m->p    = (float *)malloc((size_t)NH * sizeof(float));
    if (!m->wg || !m->wu || !m->wd || !m->swg || !m->swu || !m->swd ||
        !m->qmg || !m->qmu ||
        !m->x || !m->x8 || !m->accg || !m->accu || !m->accd || !m->p) {
        return -1;
    }
    m->s_in = m->s_mid = m->s_out = 1.0f;
    return 0;
}

static inline void ffn_model_free(ffn_model_t *m)
{
    free(m->wg); free(m->wu); free(m->wd);
    free(m->swg); free(m->swu); free(m->swd);
    free(m->qmg); free(m->qmu);
    free(m->x); free(m->x8);
    free(m->accg); free(m->accu); free(m->accd); free(m->p);
    memset(m, 0, sizeof(*m));
}

/* Synthetic layer with representative numbers:
 *   - input: uniform in [-amp_t, amp_t], amp_t varies per token;
 *   - gate/up: the physical multiplier m = sw*s_in*2^16 is chosen in
 *     [m_lo, m_hi].  The default spans 0.35..2.6, i.e. both sides of the old
 *     Q16 clamp, so the same stand exercises the old bug and the wide path.
 *   - int4 codes, so the accumulator bound is 8*127*K. */
static inline int ffn_model_synth_scaled(ffn_model_t *m, int K, int NH, int ntok,
                                         uint32_t seed, float m_lo, float m_hi)
{
    uint32_t st = seed ^ 0xabcdef01u;
    int t, i, r;
    float amax = 0.0f;

    if (ffn_model_alloc(m, K, NH, ntok) != 0) return -1;

    ffn_pack_i4_random(m->wg, NH, K, seed ^ 0x1111u);
    ffn_pack_i4_random(m->wu, NH, K, seed ^ 0x2222u);
    ffn_pack_i4_random(m->wd, K, NH, seed ^ 0x3333u);

    for (t = 0; t < ntok; t++) {
        float amp = 0.90f + 0.02f * (float)t;
        for (i = 0; i < K; i++) {
            float v = ((float)((int)(ffn_lcg(&st) % 2001) - 1000) * 1.0e-3f) * amp;
            m->x[(size_t)t * K + i] = v;
            if (v < 0.0f) v = -v;
            if (v > amax) amax = v;
        }
    }
    m->s_in = amax / (float)FFN_CAL_TARGET;

    for (r = 0; r < NH; r++) {
        float mg = m_lo + (m_hi - m_lo) * (float)(r % 17) / 16.0f;
        float mu = m_lo + (m_hi - m_lo) * (float)((r * 7) % 13) / 12.0f;
        m->swg[r] = mg / (m->s_in * FFN_Q16_ONE);
        m->swu[r] = mu / (m->s_in * FFN_Q16_ONE);
    }
    /* down rows: independent scale, s_out comes out of the accumulator */
    for (r = 0; r < K; r++)
        m->swd[r] = 1.10e-3f + (float)(r % 9) * 2.0e-5f;
    return 0;
}

static inline int ffn_model_synth(ffn_model_t *m, int K, int NH, int ntok, uint32_t seed)
{
    return ffn_model_synth_scaled(m, K, NH, ntok, seed, 0.35f, 2.60f);
}

/* ------------------------------------------------------------------ */
/* the DSP pipeline                                                   */
/* ------------------------------------------------------------------ */

static inline void ffn_quantize_x(ffn_model_t *m)
{
    int t, i;
    for (t = 0; t < m->ntok; t++) {
        const float *x = m->x + (size_t)t * m->K;
        int8_t *q = m->x8 + (size_t)t * m->K;
        for (i = 0; i < m->K; i++) {
            long v = lrintf(x[i] / m->s_in);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[i] = (int8_t)v;
        }
    }
}

/* exact int32 gate/up accumulators for token t (same value as the HVX dot) */
static inline void ffn_gate_up_acc(const ffn_model_t *m, int t)
{
    const int8_t *x8 = m->x8 + (size_t)t * m->K;
    int32_t bias = -8 * ffn_sum_i8(x8, m->K);
    int r;
    for (r = 0; r < m->NH; r++) {
        m->accg[r] = ffn_dot_i4(m->wg + (size_t)r * (m->K / 2), x8, m->K, bias);
        m->accu[r] = ffn_dot_i4(m->wu + (size_t)r * (m->K / 2), x8, m->K, bias);
    }
}

/* the SiLU stage exactly as silu_range(); p_out = P as the kernel produced it.
 * mode 0 = wide dequant (the kernel), 1 = legacy Q16+clamp (signed low
 * product), 2 = legacy Q16+clamp with the pre-fix unsigned low product.
 * The two legacy modes exist only for the regression gates. */
static inline void ffn_int_mid_mode(const ffn_model_t *m, int t, int8_t *mid,
                                    float *p_out, int mode)
{
    int32_t mq_mid = (int32_t)(4096.0f / m->s_mid);
    int r;
    ffn_gate_up_acc(m, t);
    for (r = 0; r < m->NH; r++) {
        double mg = (double)m->swg[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        double mu = (double)m->swu[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        int32_t zg, zu, pr;
        if (mode == 0) {
            zg = ffn_deq_wide(m->accg[r], m->qmg[r]);
            zu = ffn_deq_wide(m->accu[r], m->qmu[r]);
            pr = ffn_prod_q16(ffn_silu_qk(zg, m->kg), zu, m->ku);
        } else {
            zg = ffn_deq_legacy(m->accg[r], mg);
            zu = ffn_deq_legacy(m->accu[r], mu);
            pr = (mode == 2)
                ? ffn_mulq16_unsigned(ffn_silu_q16(zg), zu)
                : ffn_mulq16(ffn_silu_q16(zg), zu);
        }
        mid[r] = (int8_t)ffn_requant_mid(pr, mq_mid);
        if (p_out) p_out[r] = (float)((double)pr / 65536.0);
    }
}

static inline void ffn_int_mid(const ffn_model_t *m, int t, int8_t *mid, float *p_out)
{
    ffn_int_mid_mode(m, t, mid, p_out, 0);
}

/* down GEMV + pack_out_i8, exactly as ffn_impl() does after SiLU */
static inline void ffn_int_out(const ffn_model_t *m, const int8_t *mid, int8_t *out,
                               int32_t *accd)
{
    int32_t biasd = -8 * ffn_sum_i8(mid, m->NH);
    int r;
    for (r = 0; r < m->K; r++) {
        int32_t acc = ffn_dot_i4(m->wd + (size_t)r * (m->NH / 2), mid, m->NH, biasd);
        float md = m->swd[r] * m->s_mid / m->s_out;
        int64_t v = ((int64_t)acc * (int64_t)ffn_qp16_of(md) + 32768) >> 16;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[r] = (int8_t)v;
        if (accd) accd[r] = acc;
    }
}

/* f32 truth of the layer: on the original activation (quantised=0) or on the
 * quantised one (quantised=1).  out[K] real, p_out[NH] real (optional). */
static inline void ffn_ref_token(const ffn_model_t *m, int t, float *out, float *p_out,
                                 int quantised)
{
    const float *x = m->x + (size_t)t * m->K;
    const int8_t *x8 = m->x8 + (size_t)t * m->K;
    float *p = m->p;
    int r, i;
    for (r = 0; r < m->NH; r++) {
        double g, u, pv;
        const uint8_t *rg = m->wg + (size_t)r * (m->K / 2);
        const uint8_t *ru = m->wu + (size_t)r * (m->K / 2);
        if (quantised) {
            double sg = 0.0, su = 0.0;
            for (i = 0; i < m->K; i += 2) {
                uint8_t b = rg[i >> 1];
                sg += (double)((int)(b & 0xf) - 8) * (double)x8[i];
                sg += (double)((int)(b >> 4) - 8) * (double)x8[i + 1];
                b = ru[i >> 1];
                su += (double)((int)(b & 0xf) - 8) * (double)x8[i];
                su += (double)((int)(b >> 4) - 8) * (double)x8[i + 1];
            }
            g = sg * (double)m->swg[r] * (double)m->s_in;
            u = su * (double)m->swu[r] * (double)m->s_in;
        } else {
            g = ffn_dot_i4_f32(rg, x, m->K, m->swg[r]);
            u = ffn_dot_i4_f32(ru, x, m->K, m->swu[r]);
        }
        pv = ffn_silu_f32(g) * u;
        p[r] = (float)pv;
    }
    if (p_out) memcpy(p_out, p, (size_t)m->NH * sizeof(float));
    for (r = 0; r < m->K; r++)
        out[r] = (float)ffn_dot_i4_f32(m->wd + (size_t)r * (m->NH / 2), p, m->NH, m->swd[r]);
}

/* ------------------------------------------------------------------ */
/* scale calibration, in the order the device needs it                */
/* ------------------------------------------------------------------ */

typedef struct {
    float s_in, s_mid, s_out;
    float gate_absmax, up_absmax, p_absmax;   /* real units */
    float zg_absmax;                          /* |g| in Q16 units (z << k) */
    float z_g_max, z_u_max;                   /* max |acc*m|, int32 units */
    int   kg, ku;                             /* widening shifts */
    int   g_clamped, u_clamped, d_clamped;    /* rows the old Q16 format clamps */
    float m_g_min, m_g_max, m_d_min, m_d_max;
    int32_t mq_mid, acc_dmax;
    double worst_row;                         /* max_r m_d*acc_max[r] */
    int   silu_clamp;                         /* |g| > 8 lanes */
    int   mid_zero, mid_n;                    /* lanes with mid == 0 (degeneracy) */
} ffn_calib_t;

static inline int ffn_calibrate(ffn_model_t *m, ffn_calib_t *ci)
{
    int8_t *mid_all = NULL;
    int32_t *acc_max = NULL;
    float pmax = 0.0f, gmax = 0.0f, umax = 0.0f, zgmax = 0.0f;
    float zgm = 0.0f, zum = 0.0f;
    float mg_min = 1e30f, mg_max = 0.0f, md_min = 1e30f, md_max = 0.0f;
    int32_t gmaxacc = 0;
    double need = 0.0, mmaxg = 0.0, mmaxu = 0.0;
    int t, r, i, rc = -1;

    memset(ci, 0, sizeof(*ci));

    /* ---- s_in from the input absmax, 13 % headroom ---- */
    m->s_in = 0.0f;
    for (i = 0; i < m->ntok * m->K; i++) {
        float a = m->x[i] < 0.0f ? -m->x[i] : m->x[i];
        if (a > m->s_in) m->s_in = a;
    }
    if (!(m->s_in > 0.0f)) return -1;
    m->s_in /= (float)FFN_CAL_TARGET;
    ffn_quantize_x(m);
    ci->s_in = m->s_in;

    /* ---- multipliers and their widening shifts (per buffer, as the kernel) */
    for (r = 0; r < m->NH; r++) {
        double mg = (double)m->swg[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        double mu = (double)m->swu[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        if (mg > mmaxg) mmaxg = mg;
        if (mu > mmaxu) mmaxu = mu;
        if (mg < mg_min) mg_min = (float)mg;
        if (mg > mg_max) mg_max = (float)mg;
        if (mg > FFN_Q16_CLAMP) ci->g_clamped++;
        if (mu > FFN_Q16_CLAMP) ci->u_clamped++;
    }
    m->kg = ffn_deq_shift(mmaxg, m->K, 3);       /* int4 codes, |w| <= 8 */
    m->ku = ffn_deq_shift(mmaxu, m->K, 3);
    for (r = 0; r < m->NH; r++) {
        double mg = (double)m->swg[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        double mu = (double)m->swu[r] * (double)m->s_in * (double)FFN_Q16_ONE;
        m->qmg[r] = ffn_qm_of(mg, m->kg);
        m->qmu[r] = ffn_qm_of(mu, m->ku);
    }
    ci->kg = m->kg;
    ci->ku = m->ku;
    ci->m_g_min = mg_min;
    ci->m_g_max = mg_max;

    /* ---- s_mid from the measured P = silu(gate)*up range ---- */
    for (t = 0; t < m->ntok; t++) {
        ffn_gate_up_acc(m, t);
        for (r = 0; r < m->NH; r++) {
            double g = (double)m->accg[r] * (double)m->swg[r] * (double)m->s_in;
            double u = (double)m->accu[r] * (double)m->swu[r] * (double)m->s_in;
            double pv = fabs(ffn_silu_f32(g) * u);
            int32_t zg = ffn_deq_wide(m->accg[r], m->qmg[r]);
            int32_t zu = ffn_deq_wide(m->accu[r], m->qmu[r]);
            if (pv > pmax) pmax = pv;
            if (fabs(g) > gmax) gmax = (float)fabs(g);
            if (fabs(u) > umax) umax = (float)fabs(u);
            if (fabs((double)zg) > zgm) zgm = (float)fabs((double)zg);
            if (fabs((double)zu) > zum) zum = (float)fabs((double)zu);
            if (fabs(g) * 65536.0 > zgmax) zgmax = (float)(fabs(g) * 65536.0);
            if (g > 8.0 || g < -8.0) ci->silu_clamp++;
        }
    }
    if (!(pmax > 0.0f)) return -1;
    m->s_mid = pmax / 127.0f;
    ci->s_mid = m->s_mid;
    ci->mq_mid = (int32_t)(4096.0f / m->s_mid);
    ci->gate_absmax = gmax;
    ci->up_absmax = umax;
    ci->p_absmax = pmax;
    ci->zg_absmax = zgmax;
    ci->z_g_max = zgm;
    ci->z_u_max = zum;

    /* ---- mid buffer for every token, then s_out from the accumulator ---- */
    mid_all = (int8_t *)malloc((size_t)m->ntok * m->NH);
    acc_max = (int32_t *)calloc((size_t)m->K, sizeof(int32_t));
    if (!mid_all || !acc_max) goto out;
    for (t = 0; t < m->ntok; t++) {
        int8_t *mid = mid_all + (size_t)t * m->NH;
        int32_t biasd;
        ffn_int_mid(m, t, mid, NULL);
        biasd = -8 * ffn_sum_i8(mid, m->NH);            /* same as ffn_int_out() */
        for (i = 0; i < m->NH; i++) if (mid[i] == 0) ci->mid_zero++;
        ci->mid_n += m->NH;
        for (r = 0; r < m->K; r++) {
            int32_t a = ffn_dot_i4(m->wd + (size_t)r * (m->NH / 2), mid, m->NH, biasd);
            if (a < 0) a = -a;
            if (a > acc_max[r]) acc_max[r] = a;
        }
    }
    for (r = 0; r < m->K; r++) {
        double s = (double)m->swd[r] * (double)m->s_mid *
                   (double)acc_max[r] / FFN_CAL_TARGET;
        if (acc_max[r] > gmaxacc) gmaxacc = acc_max[r];
        if (s > need) need = s;
    }
    ci->acc_dmax = gmaxacc;
    if (!(need > 0.0)) goto out;
    m->s_out = (float)need;
    ci->s_out = m->s_out;
    for (r = 0; r < m->K; r++) {
        float md = m->swd[r] * m->s_mid / m->s_out;
        double wr = (double)md * (double)acc_max[r];
        if (md > FFN_Q16_CLAMP) ci->d_clamped++;
        if (md < md_min) md_min = md;
        if (md > md_max) md_max = md;
        if (wr > ci->worst_row) ci->worst_row = wr;
    }
    ci->m_d_min = md_min;
    ci->m_d_max = md_max;
    rc = 0;

out:
    free(mid_all); free(acc_max);
    return rc;
}

#endif /* FFN_REF_H */
