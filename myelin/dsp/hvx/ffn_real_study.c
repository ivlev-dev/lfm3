/*
 * Real-weights study of the FFN layer: does the quantised gate/up -> SiLU ->
 * int8 mid -> down path stay numerically meaningful on the LFM3 tensors?
 *
 * Input:  the real layer-0 FFN input, built exactly as in silu_err_study.c
 *         (token_embd -> attn_norm -> shortconv -> post_attention_norm ->
 *         ffn_norm) for real token ids.  The same real vector is used for
 *         every probed layer; only the layer's ffn_gate/ffn_up/ffn_down
 *         tensors change.
 * Truth:  f32 on the *dequantised* weights of the same format, so the number
 *         isolates the integer pipeline from the weight quantisation.  The
 *         cost of the weight quantisation itself is printed separately
 *         (f32 on the original GGUF weights vs f32 on the dequantised ones,
 *         same activation).
 *
 * Two weight formats are measured per layer:
 *   int8  per-row absmax/127 -- the format for which the ~0.995 ceiling
 *         quoted for int8 mid + int8 weights holds.  The gate/up Q16
 *         multiplier is m = sw*s_in*2^16 (see ffn_ref.h); for int8 it stays
 *         below the 0.99997 clamp.
 *   int4  per-row absmax/7  -- what lfmgemv_ffn() consumes today.  The same
 *         multiplier comes out ~6x over the clamp on every row, i.e. the
 *         dequantiser cannot represent the gate in Q16 at all; the study
 *         simulates the clamped kernel (that is what the DSP would do) so the
 *         size of the resulting error is visible.
 *
 * Scales: s_in = absmax(ffn_in)/110, s_mid = absmax(P)/127,
 *         s_out = max_r(swd*s_mid*acc_max[r])/110   (same rules as ffn_sim).
 *
 * Build:  cc -O2 -o ffn_real_study ffn_real_study.c -lm
 * Run:    ./ffn_real_study ../../../models/LFM3-1.7B-Q4_K_M.gguf [ntok]
 */
#include "ffn_ref.h"
#include "gguf_f32.h"
#include "lfm3_fwd.h"

#define HID 2048
#define INT 7680
#define MAX_TOK 64

/* one probed layer: original f32 rows + per-row int8 and int4 codes */
typedef struct {
    int      K, NH;
    float   *wg, *wu, *wd;                  /* original GGUF dequantised f32 */
    uint8_t *qg8, *qu8, *qd8;               /* offset-128 codes */
    float   *sg8, *su8, *sd8;
    uint8_t *qg4, *qu4, *qd4;               /* offset-8 packed nibbles */
    float   *sg4, *su4, *sd4;
} layer_t;

static double dot_f32_row(const float *w, const float *x, int k)
{
    double s = 0.0;
    int i;
    for (i = 0; i < k; i++) s += (double)w[i] * (double)x[i];
    return s;
}

static int layer_alloc(layer_t *L, int K, int NH)
{
    memset(L, 0, sizeof(*L));
    L->K = K; L->NH = NH;
    L->wg = malloc((size_t)NH * K * 4); L->wu = malloc((size_t)NH * K * 4);
    L->wd = malloc((size_t)K * NH * 4);
    L->qg8 = malloc((size_t)NH * K);    L->qu8 = malloc((size_t)NH * K);
    L->qd8 = malloc((size_t)K * NH);
    L->sg8 = malloc((size_t)NH * 4);    L->su8 = malloc((size_t)NH * 4);
    L->sd8 = malloc((size_t)K * 4);
    L->qg4 = malloc((size_t)NH * K / 2); L->qu4 = malloc((size_t)NH * K / 2);
    L->qd4 = malloc((size_t)K * NH / 2);
    L->sg4 = malloc((size_t)NH * 4);    L->su4 = malloc((size_t)NH * 4);
    L->sd4 = malloc((size_t)K * 4);
    return (L->wg && L->wu && L->wd && L->qg8 && L->qu8 && L->qd8 &&
            L->sg8 && L->su8 && L->sd8 && L->qg4 && L->qu4 && L->qd4 &&
            L->sg4 && L->su4 && L->sd4) ? 0 : -1;
}

static void layer_free(layer_t *L)
{
    free(L->wg); free(L->wu); free(L->wd);
    free(L->qg8); free(L->qu8); free(L->qd8);
    free(L->sg8); free(L->su8); free(L->sd8);
    free(L->qg4); free(L->qu4); free(L->qd4);
    free(L->sg4); free(L->su4); free(L->sd4);
    memset(L, 0, sizeof(*L));
}

/* symmetric per-row int8: codes stored offset-128 (the DSP int8 kernel) */
static float pack_i8_row(uint8_t *dst, const float *row, int k)
{
    float amax = 0.0f, sw;
    int i;
    for (i = 0; i < k; i++) { float a = fabsf(row[i]); if (a > amax) amax = a; }
    sw = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (i = 0; i < k; i++) {
        long q = lrintf(row[i] / sw);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[i] = (uint8_t)(q + 128);
    }
    return sw;
}

/* signed int8 dot with the offset-128 codes; exact int32 as the DSP kernel */
static int32_t dot_i8(const uint8_t *row, const int8_t *act, int k)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < k; i++) s += (int32_t)((int)row[i] - 128) * (int32_t)act[i];
    return s;
}

/* f32 dot with the dequantised int8 row */
static double dot_i8_f32(const uint8_t *row, const float *act, int k, double sw)
{
    double s = 0.0;
    int i;
    for (i = 0; i < k; i++) s += (double)((int)row[i] - 128) * (double)act[i];
    return s * sw;
}

/* f32 dot with the dequantised int4 row */
static double dot_i4_f32(const uint8_t *row, const float *act, int k, double sw)
{
    double s = 0.0;
    int i;
    for (i = 0; i < k; i += 2) {
        uint8_t b = row[i >> 1];
        s += (double)((int)(b & 0xf) - 8) * (double)act[i];
        s += (double)((int)(b >> 4) - 8) * (double)act[i + 1];
    }
    return s * sw;
}

/* result of one integer pipeline run for one format */
typedef struct {
    double  cos_out, cos_out_min, cos_mid, cos_ideal, max_out;
    double  m_g_min, m_g_max, m_d_min, m_d_max;
    int     g_clamped, d_clamped, zero_out, gshift, deq64;
    int32_t acc_dmax, mq_mid;
    double  worst_row;
    double  sat_mid, sat_out;
    double  lsb_p50, lsb_p90, lsb_p99, lsb_max;
    double  err_p50, err_p90, err_p99, err_max;
    int     ntok;
    float   s_mid, s_out;
    double  p_absmax;
} run_t;

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void pct4(double *v, int n, double *p50, double *p90, double *p99, double *mx)
{
    int i;
    double m = 0.0;
    for (i = 0; i < n; i++) if (v[i] > m) m = v[i];
    qsort(v, (size_t)n, sizeof(double), cmp_d);
    *p50 = v[(int)(0.50 * (n - 1))];
    *p90 = v[(int)(0.90 * (n - 1))];
    *p99 = v[(int)(0.99 * (n - 1))];
    *mx = m;
}

/* gate/up dequant as the kernel does it (Q16 multiplier, mulq16), or the
 * 64-bit variant (no clamp, no 8-bit truncation) used as a what-if */
static inline int32_t deq_acc(int32_t acc, double m, int shift, int deq64)
{
    if (deq64) {
        int64_t q30 = (int64_t)llround(m * 1073741824.0);
        return (int32_t)(((int64_t)acc * q30 + ((int64_t)1 << 29)) >> 30);
    }
    return ffn_mulq16(acc, ffn_q30_of((float)(m / (double)(1u << shift))) >> 14);
}

static inline int32_t silu_of(int32_t zg, int shift, int deq64)
{
    return ffn_silu_q16(deq64 ? zg : (zg << shift));
}

static inline int32_t prod_p(int32_t sv, int32_t zu, int shift, int deq64)
{
    if (deq64) return (int32_t)(((int64_t)sv * (int64_t)zu + 32768) >> 16);
    return ffn_mulq16(sv, zu) << shift;
}

/*
 * Run one format of one layer.  x8[t][K] is the quantised input, s_in its
 * scale.  The pipeline is the DSP one expression for expression; a clamped
 * Q16 multiplier is used as clamped, which is what the DSP would do.
 */
static int run_layer(const layer_t *L, int mode, const int8_t *x8, int ntok,
                     double s_in, int gshift, int deq64, int cal, run_t *R)
{
    int K = L->K, NH = L->NH;
    int32_t *accg = malloc((size_t)NH * 4), *accu = malloc((size_t)NH * 4);
    int8_t *mid = malloc((size_t)NH), *midi = malloc((size_t)NH);
    int32_t *acc_max = calloc((size_t)K, 4);
    float *P = malloc((size_t)ntok * NH * 4);
    float *yq = malloc((size_t)K * 4), *yref = malloc((size_t)K * 4);
    int8_t *out8 = malloc((size_t)K);
    double *lsbv = malloc((size_t)ntok * NH * sizeof(double));
    double *errv = malloc((size_t)ntok * K * sizeof(double));
    double pmax = 0.0;
    int t, r, i, nl = 0, ne = 0;

    if (!accg || !accu || !mid || !midi || !acc_max || !P || !yq || !yref ||
        !out8 || !lsbv || !errv) return -1;
    memset(R, 0, sizeof(*R));
    R->ntok = ntok;
    R->gshift = gshift;
    R->deq64 = deq64;

    /* ---- P on the dequantised weights and the quantised input ---- */
    for (t = 0; t < ntok; t++) {
        for (r = 0; r < NH; r++) {
            double g, u;
            if (mode == 8) {
                double sg = 0.0, su = 0.0;
                const uint8_t *rg = L->qg8 + (size_t)r * K;
                const uint8_t *ru = L->qu8 + (size_t)r * K;
                for (i = 0; i < K; i++) {
                    sg += (double)((int)rg[i] - 128) * (double)x8[(size_t)t * K + i];
                    su += (double)((int)ru[i] - 128) * (double)x8[(size_t)t * K + i];
                }
                g = sg * (double)L->sg8[r] * s_in;
                u = su * (double)L->su8[r] * s_in;
            } else {
                double sg = 0.0, su = 0.0;
                const uint8_t *rg = L->qg4 + (size_t)r * (K / 2);
                const uint8_t *ru = L->qu4 + (size_t)r * (K / 2);
                for (i = 0; i < K; i += 2) {
                    uint8_t b = rg[i >> 1];
                    sg += (double)((int)(b & 0xf) - 8) * (double)x8[(size_t)t * K + i];
                    sg += (double)((int)(b >> 4) - 8) * (double)x8[(size_t)t * K + i + 1];
                    b = ru[i >> 1];
                    su += (double)((int)(b & 0xf) - 8) * (double)x8[(size_t)t * K + i];
                    su += (double)((int)(b >> 4) - 8) * (double)x8[(size_t)t * K + i + 1];
                }
                g = sg * (double)L->sg4[r] * s_in;
                u = su * (double)L->su4[r] * s_in;
            }
            P[(size_t)t * NH + r] = (float)(ffn_silu_f32(g) * u);
            if (fabs((double)P[(size_t)t * NH + r]) > pmax) pmax = fabs((double)P[(size_t)t * NH + r]);
        }
    }
    if (!(pmax > 0.0)) return -1;
    if (cal == 0) {
        /* absmax/127: what silu_err_study used */
        R->s_mid = (float)(pmax / 127.0);
    } else {
        /* p99.9 / 110: 13 % headroom, clips the 0.1 % tail instead of spending
         * the whole int8 range on one outlier lane */
        double *ap = malloc((size_t)ntok * NH * sizeof(double));
        int n = 0;
        for (t = 0; t < ntok; t++)
            for (r = 0; r < NH; r++) ap[n++] = fabs((double)P[(size_t)t * NH + r]);
        qsort(ap, (size_t)n, sizeof(double), cmp_d);
        R->s_mid = (float)(ap[(int)(0.999 * (double)(n - 1))] / FFN_CAL_TARGET);
        free(ap);
    }
    R->p_absmax = pmax;
    R->mq_mid = (int32_t)(4096.0f / R->s_mid);

    /* ---- integer mid buffer, then s_out from the down accumulator ---- */
    R->m_g_min = R->m_d_min = 1e30;
    for (t = 0; t < ntok; t++) {
        int32_t bias = 0;
        if (mode == 4) {
            int32_t s = 0;
            for (i = 0; i < K; i++) s += x8[(size_t)t * K + i];
            bias = -8 * s;
        }
        for (r = 0; r < NH; r++) {
            double mg, mu;
            int32_t zg, zu, pr;
            if (mode == 8) {
                accg[r] = dot_i8(L->qg8 + (size_t)r * K, x8 + (size_t)t * K, K);
                accu[r] = dot_i8(L->qu8 + (size_t)r * K, x8 + (size_t)t * K, K);
                mg = (double)L->sg8[r] * s_in * 65536.0;
                mu = (double)L->su8[r] * s_in * 65536.0;
            } else {
                accg[r] = ffn_dot_i4(L->qg4 + (size_t)r * (K / 2), x8 + (size_t)t * K, K, bias);
                accu[r] = ffn_dot_i4(L->qu4 + (size_t)r * (K / 2), x8 + (size_t)t * K, K, bias);
                mg = (double)L->sg4[r] * s_in * 65536.0;
                mu = (double)L->su4[r] * s_in * 65536.0;
            }
            if (mg > FFN_Q16_CLAMP) R->g_clamped++;
            if (mg < R->m_g_min) R->m_g_min = mg;
            if (mg > R->m_g_max) R->m_g_max = mg;
            zg = deq_acc(accg[r], mg, gshift, deq64);
            zu = deq_acc(accu[r], mu, gshift, deq64);
            pr = prod_p(silu_of(zg, gshift, deq64), zu, gshift, deq64);
            mid[r] = (int8_t)ffn_requant_mid(pr, R->mq_mid);
        }
        for (r = 0; r < K; r++) {
            int32_t a;
            if (mode == 8) {
                a = dot_i8(L->qd8 + (size_t)r * NH, mid, NH);
            } else {
                int32_t s = 0;
                for (i = 0; i < NH; i++) s += mid[i];
                a = ffn_dot_i4(L->qd4 + (size_t)r * (NH / 2), mid, NH, -8 * s);
            }
            if (a < 0) a = -a;
            if (a > acc_max[r]) acc_max[r] = a;
        }
    }
    {
        double need = 0.0;
        for (r = 0; r < K; r++) {
            double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
            double s = swd * (double)R->s_mid * (double)acc_max[r] / FFN_CAL_TARGET;
            if (s > need) need = s;
            if (acc_max[r] > R->acc_dmax) R->acc_dmax = acc_max[r];
        }
        if (!(need > 0.0)) {          /* the clamped path produced an all-zero mid */
            R->zero_out = 1;
            need = 1.0;
        }
        R->s_out = (float)need;
    }

    /* ---- evaluate: cos/maxerr/LSB/saturation vs the f32 truth ---- */
    for (t = 0; t < ntok; t++) {
        double dout = 0.0, da = 0.0, dbb = 0.0, dmid = 0.0;
        int32_t bias = 0;
        if (mode == 4) {
            int32_t s = 0;
            for (i = 0; i < K; i++) s += x8[(size_t)t * K + i];
            bias = -8 * s;
        }
        for (r = 0; r < NH; r++) {
            int32_t zg, zu, pr;
            double mg, mu;
            if (mode == 8) {
                accg[r] = dot_i8(L->qg8 + (size_t)r * K, x8 + (size_t)t * K, K);
                accu[r] = dot_i8(L->qu8 + (size_t)r * K, x8 + (size_t)t * K, K);
                mg = (double)L->sg8[r] * s_in * 65536.0;
                mu = (double)L->su8[r] * s_in * 65536.0;
            } else {
                accg[r] = ffn_dot_i4(L->qg4 + (size_t)r * (K / 2), x8 + (size_t)t * K, K, bias);
                accu[r] = ffn_dot_i4(L->qu4 + (size_t)r * (K / 2), x8 + (size_t)t * K, K, bias);
                mg = (double)L->sg4[r] * s_in * 65536.0;
                mu = (double)L->su4[r] * s_in * 65536.0;
            }
            zg = deq_acc(accg[r], mg, gshift, deq64);
            zu = deq_acc(accu[r], mu, gshift, deq64);
            pr = prod_p(silu_of(zg, gshift, deq64), zu, gshift, deq64);
            mid[r] = (int8_t)ffn_requant_mid(pr, R->mq_mid);
            {
                long ideal = lrintf(P[(size_t)t * NH + r] / R->s_mid);
                double d;
                if (ideal > 127) ideal = 127;
                if (ideal < -127) ideal = -127;
                d = fabs((double)mid[r] - (double)ideal);
                lsbv[nl++] = d;
                midi[r] = (int8_t)ideal;
            }
            dmid += (double)mid[r] * (double)R->s_mid * (double)P[(size_t)t * NH + r];
            da += (double)mid[r] * (double)R->s_mid * (double)mid[r] * (double)R->s_mid;
            dbb += (double)P[(size_t)t * NH + r] * (double)P[(size_t)t * NH + r];
        }
        R->cos_mid += (da > 0.0 && dbb > 0.0) ? dmid / (sqrt(da) * sqrt(dbb)) : 0.0;
        /* format ceiling: the same down path with ideally rounded mid */
        {
            double di = 0.0, ai = 0.0, bi = 0.0;
            int32_t bias_i = 0;
            if (mode == 4) for (i = 0; i < NH; i++) bias_i += midi[i];
            bias_i *= -8;
            for (r = 0; r < K; r++) {
                double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
                int32_t acc = (mode == 8) ? dot_i8(L->qd8 + (size_t)r * NH, midi, NH)
                    : ffn_dot_i4(L->qd4 + (size_t)r * (NH / 2), midi, NH, bias_i);
                double y = (double)acc * swd * (double)R->s_mid;
                double t2 = (mode == 8)
                    ? dot_i8_f32(L->qd8 + (size_t)r * NH, P + (size_t)t * NH, NH, swd)
                    : dot_i4_f32(L->qd4 + (size_t)r * (NH / 2), P + (size_t)t * NH, NH, swd);
                di += y * t2; ai += y * y; bi += t2 * t2;
            }
            R->cos_ideal += (ai > 0.0 && bi > 0.0) ? di / (sqrt(ai) * sqrt(bi)) : 0.0;
        }

        /* down projection: kernel dot + pack_out_i8 */
        {
            int32_t s = 0;
            if (mode == 4) for (i = 0; i < NH; i++) s += mid[i];
            for (r = 0; r < K; r++) {
                int32_t acc = (mode == 8)
                    ? dot_i8(L->qd8 + (size_t)r * NH, mid, NH)
                    : ffn_dot_i4(L->qd4 + (size_t)r * (NH / 2), mid, NH, -8 * s);
                double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
                double md = swd * (double)R->s_mid / (double)R->s_out;
                int64_t v = ((int64_t)acc * (int64_t)ffn_qp16_of((float)md) + 32768) >> 16;
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                out8[r] = (int8_t)v;
            }
        }
        dout = 0.0; da = 0.0; dbb = 0.0;
        for (r = 0; r < K; r++) {
            double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
            double y = (mode == 8)
                ? dot_i8_f32(L->qd8 + (size_t)r * NH, P + (size_t)t * NH, NH, swd)
                : dot_i4_f32(L->qd4 + (size_t)r * (NH / 2), P + (size_t)t * NH, NH, swd);
            double ym = (double)out8[r] * (double)R->s_out;
            yref[r] = (float)y;
            yq[r] = (float)ym;
            dout += ym * y;
            da += ym * ym;
            dbb += y * y;
            errv[ne++] = fabs(ym - y);
        }
        {
            double c = (da > 0.0 && dbb > 0.0) ? dout / (sqrt(da) * sqrt(dbb)) : 0.0;
            R->cos_out += c;
            if (t == 0 || c < R->cos_out_min) R->cos_out_min = c;
        }
        {
            int sm = 0, so = 0;
            for (r = 0; r < NH; r++) if (mid[r] >= 127 || mid[r] <= -127) sm++;
            for (r = 0; r < K; r++) if (out8[r] >= 127 || out8[r] <= -127) so++;
            R->sat_mid += 100.0 * sm / NH;
            R->sat_out += 100.0 * so / K;
        }
    }
    R->cos_out /= ntok;
    R->cos_mid /= ntok;
    R->cos_ideal /= ntok;
    R->sat_mid /= ntok;
    R->sat_out /= ntok;
    pct4(lsbv, nl, &R->lsb_p50, &R->lsb_p90, &R->lsb_p99, &R->lsb_max);
    pct4(errv, ne, &R->err_p50, &R->err_p90, &R->err_p99, &R->err_max);
    R->max_out = R->err_max;

    /* down multiplier range and the m_r*acc_max invariant */
    for (r = 0; r < K; r++) {
        double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
        double md = swd * (double)R->s_mid / (double)R->s_out;
        double wr = md * (double)acc_max[r];
        if (md > FFN_Q16_CLAMP) R->d_clamped++;
        if (md < R->m_d_min) R->m_d_min = md;
        if (md > R->m_d_max) R->m_d_max = md;
        if (wr > R->worst_row) R->worst_row = wr;
    }

    free(accg); free(accu); free(mid); free(midi); free(acc_max);
    free(P); free(yq); free(yref); free(out8); free(lsbv); free(errv);
    return 0;
}

/* smallest gate-output widening k such that every row's dequant multiplier
 * fits the Q16 clamp; the DSP kernel has no such field today -- this is the
 * host simulation of what a per-buffer `mbits` would buy. */
static int gate_shift_for(const layer_t *L, int mode, double s_in, double target)
{
    double m = 0.0;
    int i, k = 0;
    for (i = 0; i < L->NH; i++) {
        double sw = (mode == 8) ? (double)L->sg8[i] : (double)L->sg4[i];
        double v = sw * s_in * 65536.0;
        if (v > m) m = v;
    }
    while (m / (double)(1u << k) > target) k++;
    return k;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "../../../models/LFM3-1.7B-Q4_K_M.gguf";
    int ntok = argc > 2 ? atoi(argv[2]) : 8;
    const int lay[3] = { 0, 12, 23 };
    lfm_t spec;
    uint32_t *tok;
    float *cap;
    int top1 = -1;
    int i, t, slot;
    uint32_t st = 20240517u;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (ntok < 1) ntok = 1;
    if (ntok > MAX_TOK) ntok = MAX_TOK;
    tok = malloc((size_t)ntok * 4);
    if (!tok) { printf("oom\n"); return 1; }

    gguf_open(path);
    /* optional explicit token ids: ./ffn_real_study model.gguf N id0 id1 ... */
    if (argc > 4) {
        int n = argc - 3;
        if (n > ntok) n = ntok;
        for (i = 0; i < n; i++) tok[i] = (uint32_t)atoi(argv[3 + i]);
        ntok = n;
    } else {
        for (t = 0; t < ntok; t++) {
            st = st * 1664525u + 1013904223u;
            tok[t] = st % 127999u;
        }
    }

    lfm_init(&spec, 24, HID, INT, 16, 4, 128, 256, 128000);
    cap = malloc((size_t)3 * ntok * HID * 4);
    if (!cap) { printf("oom\n"); return 1; }
    printf("real FFN study: %s, %d tokens\n", path, ntok);
    printf("input: full 24-layer f32 forward (mirror of backend/cpu.rs); "
           "FFN inputs captured at layers %d/%d/%d\n", lay[0], lay[1], lay[2]);
    if (lfm_forward(&spec, tok, ntok, lay[0], lay[1], lay[2], cap, &top1) != 0) {
        printf("forward failed\n");
        return 1;
    }
    printf("lm_head argmax of the last token = %d; reproduce the full-stack check "
           "with 5 explicit ids:\n  ./ffn_real_study <model> 5 597 5205 302 3980 355"
           "  -> expect 4741 (\" Paris\")\n", top1);

    for (slot = 0; slot < 3; slot++) {
        char nm[64];
        int li = lay[slot], li_c = lay[slot];
        const tinfo_t *tg, *tu, *td;
        layer_t Ly;
        run_t R8, R4, R8w, R8i, R8p;
        int k8;
        double s_in = 0.0, cos_w8 = 0.0, cos_w4 = 0.0;
        int8_t *x8;
        float *P = malloc((size_t)INT * 4);
        float *yorig = malloc((size_t)HID * 4);
        float *ydeq = malloc((size_t)HID * 4);
        int rc;

        /* ---- real input of this layer: s_in from its own absmax ---- */
        x8 = malloc((size_t)ntok * HID);
        if (!x8) { printf("oom\n"); return 1; }
        {
            double amax = 0.0, rms = 0.0;
            for (i = 0; i < ntok * HID; i++) {
                double v = fabs((double)cap[((size_t)slot * ntok * HID) + i]);
                if (v > amax) amax = v;
                rms += v * v;
            }
            s_in = amax / FFN_CAL_TARGET;
            printf("\nlayer %d: ffn_in rms=%.4f absmax=%.4f -> s_in=%.6g\n",
                   li, sqrt(rms / (ntok * HID)), amax, s_in);
        }
        for (t = 0; t < ntok; t++)
            for (i = 0; i < HID; i++) {
                long q = lrintf(cap[((size_t)slot * ntok + t) * HID + i] / (float)s_in);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                x8[(size_t)t * HID + i] = (int8_t)q;
            }

        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", li_c); tg = gguf_find(nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", li_c);   tu = gguf_find(nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", li_c); td = gguf_find(nm);

        if (layer_alloc(&Ly, HID, INT) != 0 || !P || !yorig || !ydeq) {
            printf("oom\n");
            return 1;
        }
        Ly.wg = gguf_matrix(tg, INT, HID);
        Ly.wu = gguf_matrix(tu, INT, HID);
        Ly.wd = gguf_matrix(td, HID, INT);
        for (i = 0; i < INT; i++) {
            Ly.sg8[i] = pack_i8_row(Ly.qg8 + (size_t)i * HID, Ly.wg + (size_t)i * HID, HID);
            Ly.su8[i] = pack_i8_row(Ly.qu8 + (size_t)i * HID, Ly.wu + (size_t)i * HID, HID);
            Ly.sg4[i] = ffn_pack_i4_row(Ly.qg4 + (size_t)i * (HID / 2),
                                        Ly.wg + (size_t)i * HID, HID);
            Ly.su4[i] = ffn_pack_i4_row(Ly.qu4 + (size_t)i * (HID / 2),
                                        Ly.wu + (size_t)i * HID, HID);
        }
        for (i = 0; i < HID; i++) {
            Ly.sd8[i] = pack_i8_row(Ly.qd8 + (size_t)i * INT, Ly.wd + (size_t)i * INT, INT);
            Ly.sd4[i] = ffn_pack_i4_row(Ly.qd4 + (size_t)i * (INT / 2),
                                        Ly.wd + (size_t)i * INT, INT);
        }
        {
            double wr = 0.0, wa = 0.0;
            for (i = 0; i < INT * HID; i++) {
                double v = fabs((double)Ly.wg[i]);
                wr += v * v;
                if (v > wa) wa = v;
            }
            printf("  gate weights: rms=%.5f absmax=%.5f -> row int4 sw ~ %.5f, "
                   "int8 sw ~ %.6f\n", sqrt(wr / ((double)INT * HID)), wa,
                   wa / 7.0, wa / 127.0);
        }

        /* weight-quantisation cost: f32 on dequantised vs f32 on the original */
        for (t = 0; t < ntok; t++) {
            const float *xr = cap + ((size_t)slot * ntok + t) * HID;
            for (i = 0; i < INT; i++) {
                double g = dot_f32_row(Ly.wg + (size_t)i * HID, xr, HID);
                double u = dot_f32_row(Ly.wu + (size_t)i * HID, xr, HID);
                P[i] = (float)(ffn_silu_f32(g) * u);
            }
            for (i = 0; i < HID; i++)
                yorig[i] = (float)dot_f32_row(Ly.wd + (size_t)i * INT, P, INT);
            for (i = 0; i < HID; i++)
                ydeq[i] = (float)dot_i8_f32(Ly.qd8 + (size_t)i * INT, P, INT, Ly.sd8[i]);
            cos_w8 += f32_cosine(yorig, ydeq, HID);
            for (i = 0; i < HID; i++)
                ydeq[i] = (float)dot_i4_f32(Ly.qd4 + (size_t)i * (INT / 2), P, INT, Ly.sd4[i]);
            cos_w4 += f32_cosine(yorig, ydeq, HID);
        }
        cos_w8 /= ntok;
        cos_w4 /= ntok;
        free(P); free(yorig); free(ydeq);

        k8 = gate_shift_for(&Ly, 8, s_in, 0.9);
        rc = run_layer(&Ly, 8, x8, ntok, s_in, 0, 0, 0, &R8);
        if (rc != 0) { printf("run int8 failed\n"); return 1; }
        rc = run_layer(&Ly, 4, x8, ntok, s_in, 0, 0, 0, &R4);
        if (rc != 0) { printf("run int4 failed\n"); return 1; }
        rc = run_layer(&Ly, 8, x8, ntok, s_in, k8, 0, 0, &R8w);
        if (rc != 0) { printf("run int8 wide failed\n"); return 1; }
        rc = run_layer(&Ly, 8, x8, ntok, s_in, 0, 1, 0, &R8i);
        if (rc != 0) { printf("run int8 int64 failed\n"); return 1; }
        rc = run_layer(&Ly, 8, x8, ntok, s_in, 0, 1, 1, &R8p);
        if (rc != 0) { printf("run int8 int64 p999 failed\n"); return 1; }

        printf("  weight quantisation alone (f32 weights vs dequantised, same "
               "activation): int8 cos=%.6f  int4 cos=%.6f\n", cos_w8, cos_w4);

        printf("\n  int8 weights (per-row absmax/127):\n");
        printf("    m_gate=[%.4f..%.4f] clamps=%d  m_down=[%.5f..%.5f] clamps=%d\n",
               R8.m_g_min, R8.m_g_max, R8.g_clamped, R8.m_d_min, R8.m_d_max, R8.d_clamped);
        printf("    s_mid=%.6g mq_mid=%d s_out=%.6g (acc_max=%d, worst m*acc=%.1f)\n",
               R8.s_mid, R8.mq_mid, R8.s_out, R8.acc_dmax, R8.worst_row);
        printf("    cos(out)=%.6f (min tok %.6f) cos(mid)=%.6f  ideal-mid ceiling=%.6f  sat mid/out=%.2f%%/%.2f%%\n",
               R8.cos_out, R8.cos_out_min, R8.cos_mid, R8.cos_ideal, R8.sat_mid, R8.sat_out);
        printf("    mid LSB p50/p90/p99/max = %.2f/%.2f/%.2f/%.0f\n",
               R8.lsb_p50, R8.lsb_p90, R8.lsb_p99, R8.lsb_max);
        printf("    out err p50/p90/p99/max = %.5g/%.5g/%.5g/%.5g\n",
               R8.err_p50, R8.err_p90, R8.err_p99, R8.err_max);

        printf("\n  int4 weights (per-row absmax/7, what lfmgemv_ffn consumes):\n");
        printf("    m_gate=[%.3f..%.3f] clamps=%d/%d rows -> Q16 multiplier sits at the clamp\n",
               R4.m_g_min, R4.m_g_max, R4.g_clamped, R4.ntok * INT);
        printf("    cos(out)=%.6f cos(mid)=%.6f  sat mid/out=%.2f%%/%.2f%%\n",
               R4.cos_out, R4.cos_mid, R4.sat_mid, R4.sat_out);
        printf("    mid LSB p50/p90/p99/max = %.2f/%.2f/%.2f/%.0f\n",
               R4.lsb_p50, R4.lsb_p90, R4.lsb_p99, R4.lsb_max);
        printf("    m_down=[%.5f..%.5f] clamps=%d  s_out worst m*acc=%.1f%s\n",
               R4.m_d_min, R4.m_d_max, R4.d_clamped, R4.worst_row,
               R4.zero_out ? "  [mid all zero: no signal reaches down]" : "");

        printf("\n  simulated wide gate output (host only, kernel would need a "
               "per-buffer shift; int8 at Q%d):\n", 16 - k8);
        printf("    int8: m_gate max=%.3f / 2^%d  cos(out)=%.6f  mid LSB p99=%.0f\n",
               R8w.m_g_max, k8, R8w.cos_out, R8w.lsb_p99);
        printf("  int8 + 64-bit gate/up dequant, no Q16 clamp (host only):\n");
        printf("    absmax/127 mid cal: cos(out)=%.6f (min tok %.6f) cos(mid)=%.6f  ideal-mid ceiling=%.6f  LSB p50/p99/max=%.2f/%.0f/%.0f\n",
               R8i.cos_out, R8i.cos_out_min, R8i.cos_mid, R8i.cos_ideal,
               R8i.lsb_p50, R8i.lsb_p99, R8i.lsb_max);
        printf("    p99.9/110 mid cal : cos(out)=%.6f cos(mid)=%.6f  LSB p50/p99/max=%.2f/%.0f/%.0f"
               "  s_mid=%.6g\n", R8p.cos_out, R8p.cos_mid, R8p.lsb_p50, R8p.lsb_p99,
               R8p.lsb_max, R8p.s_mid);

        free(x8);
        layer_free(&Ly);
    }

    free(tok); free(cap);
    return 0;
}
