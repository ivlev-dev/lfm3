/*
 * Real-weights study of the FFN layer: does the wide gate/up dequant keep the
 * quantised path numerically meaningful on every LFM3-1.7B layer?
 *
 * Input:  the real FFN input of every layer, from one full f32 forward pass
 *         (token_embd -> ... -> ffn_norm, mirror of backend/cpu.rs).
 * Truth:  f32 on the *dequantised* weights of the same format, so the number
 *         isolates the integer pipeline from the weight quantisation.
 *
 * Two weight formats per layer:
 *   int8 per-row absmax/127 -- m = sw*s_in*2^16 is 0.3..12, the old Q16
 *         multiplier clamped on the deep layers;
 *   int4 per-row absmax/7   -- m is 6..215, the old kernel clamped every row.
 *
 * Dequant (kernel format, see ffn_ref.h / lfmgemv_impl.c):
 *   qm = Q16.16(m/2^k),  z = (acc*qm + 2^15) >> 16 = g*2^(16-k),
 *   k chosen from 127*|w|max*K*mmax so z fits int32 with 2x headroom
 *   (ffn_deq_shift); silu_qk(z,k) consumes Q(16-k).  On these tensors k = 0.
 *
 * Scales: s_in = absmax(ffn_in)/110, s_mid = absmax(P)/127,
 *         s_out = max_r(swd*s_mid*acc_max[r])/110   (same rules as ffn_sim).
 *
 * Regression gates: for layers 0/12/23 the same int4 path is run with the old
 * Q16+clamp dequant and with the pre-fix unsigned mulq16; both must fail
 * (cos < 0.99, and the unsigned bug must saturate ~half of the mid buffer).
 *
 * Build:  cc -O2 -o ffn_real_study ffn_real_study.c -lm
 * Run:    ./ffn_real_study ../../../models/LFM3-1.7B-Q4_K_M.gguf [ntok]
 *
 * Exit code: 0 pass, 1 allocation/forward failure, 2 threshold failure.
 */
#include "ffn_ref.h"
#include "gguf_f32.h"
#include "lfm3_fwd.h"

#define HID 2048
#define INT 7680
#define NL  24
#define MAX_TOK 32

#define PASS_COS     0.99
#define PASS_SAT_PCT 5.0

/* one probed layer: per-row int8 and int4 codes (the f32 rows are freed after
 * packing, so the study holds ~3x int8 weights instead of ~10x f32) */
typedef struct {
    int      K, NH;
    uint8_t *qg8, *qu8, *qd8;               /* offset-128 codes */
    float   *sg8, *su8, *sd8;
    uint8_t *qg4, *qu4, *qd4;               /* offset-8 packed nibbles */
    float   *sg4, *su4, *sd4;
} layer_t;

static int layer_alloc(layer_t *L, int K, int NH)
{
    memset(L, 0, sizeof(*L));
    L->K = K; L->NH = NH;
    L->qg8 = malloc((size_t)NH * K);    L->qu8 = malloc((size_t)NH * K);
    L->qd8 = malloc((size_t)K * NH);
    L->sg8 = malloc((size_t)NH * 4);    L->su8 = malloc((size_t)NH * 4);
    L->sd8 = malloc((size_t)K * 4);
    L->qg4 = malloc((size_t)NH * K / 2); L->qu4 = malloc((size_t)NH * K / 2);
    L->qd4 = malloc((size_t)K * NH / 2);
    L->sg4 = malloc((size_t)NH * 4);    L->su4 = malloc((size_t)NH * 4);
    L->sd4 = malloc((size_t)K * 4);
    return (L->qg8 && L->qu8 && L->qd8 && L->sg8 && L->su8 && L->sd8 &&
            L->qg4 && L->qu4 && L->qd4 && L->sg4 && L->su4 && L->sd4) ? 0 : -1;
}

static void layer_free(layer_t *L)
{
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

/* f32 dot with the dequantised down row */
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

typedef struct {
    double  cos_out, cos_out_min, cos_mid, cos_p, cos_down, cos_ideal, max_out;
    double  m_g_min, m_g_max, m_d_min, m_d_max, z_g_max, z_u_max;
    int     k_g, k_u, old_clamp_g, old_clamp_u, d_clamped, zero_out;
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

/*
 * Run one format of one layer.  deq_mode 0 is the kernel's wide dequantiser;
 * 1 is the old Q16+clamp with the signed low product, 2 the old code with the
 * unsigned low product (regression gates).
 */
static int run_layer(const layer_t *L, int mode, int deq_mode, const int8_t *x8,
                     int ntok, double s_in, run_t *R)
{
    const int K = L->K, NH = L->NH;
    int32_t *accg = malloc((size_t)ntok * NH * 4);
    int32_t *accu = malloc((size_t)ntok * NH * 4);
    int32_t *accd = malloc((size_t)ntok * K * 4);
    int8_t *mid = malloc((size_t)ntok * NH);
    int8_t *midi = malloc((size_t)ntok * NH);
    float *P = malloc((size_t)ntok * NH * 4);
    int32_t *qmg = malloc((size_t)NH * 4), *qmu = malloc((size_t)NH * 4);
    int32_t *acc_max = calloc((size_t)K, 4);
    double *lsbv = malloc((size_t)ntok * NH * sizeof(double));
    double *errv = malloc((size_t)ntok * K * sizeof(double));
    int8_t *out8 = malloc((size_t)K);
    double pmax = 0.0, gmax = 0.0, umax = 0.0, mmaxg = 0.0, mmaxu = 0.0;
    double mg_min = 1e30, mg_max = 0.0, md_min = 1e30, md_max = 0.0;
    double zgm = 0.0, zum = 0.0, need = 0.0;
    int t, r, nl = 0, ne = 0, kg, ku;

    if (!accg || !accu || !accd || !mid || !midi || !P || !qmg || !qmu ||
        !acc_max || !lsbv || !errv || !out8) {
        free(accg); free(accu); free(accd); free(mid); free(midi); free(P);
        free(qmg); free(qmu); free(acc_max); free(lsbv); free(errv); free(out8);
        return -1;
    }
    memset(R, 0, sizeof(*R));
    R->ntok = ntok;

    /* ---- multipliers, shift and per-row Q16.16 fields ---- */
    for (r = 0; r < NH; r++) {
        double mg = (mode == 8) ? (double)L->sg8[r] * s_in * 65536.0
                                : (double)L->sg4[r] * s_in * 65536.0;
        double mu = (mode == 8) ? (double)L->su8[r] * s_in * 65536.0
                                : (double)L->su4[r] * s_in * 65536.0;
        if (mg < mg_min) mg_min = mg;
        if (mg > mg_max) mg_max = mg;
        if (mg > mmaxg) mmaxg = mg;
        if (mu > mmaxu) mmaxu = mu;
        if (mg > FFN_Q16_CLAMP) R->old_clamp_g++;
        if (mu > FFN_Q16_CLAMP) R->old_clamp_u++;
    }
    kg = ffn_deq_shift(mmaxg, K, mode == 8 ? 7 : 3);
    ku = ffn_deq_shift(mmaxu, K, mode == 8 ? 7 : 3);
    R->k_g = kg; R->k_u = ku;
    for (r = 0; r < NH; r++) {
        double mg = (mode == 8) ? (double)L->sg8[r] * s_in * 65536.0
                                : (double)L->sg4[r] * s_in * 65536.0;
        double mu = (mode == 8) ? (double)L->su8[r] * s_in * 65536.0
                                : (double)L->su4[r] * s_in * 65536.0;
        qmg[r] = ffn_qm_of(mg, kg);
        qmu[r] = ffn_qm_of(mu, ku);
    }
    R->m_g_min = mg_min;
    R->m_g_max = mg_max;

    /* ---- gate/up accumulators and the f32 truth P on the same weights ---- */
    for (t = 0; t < ntok; t++) {
        const int8_t *at = x8 + (size_t)t * K;
        int32_t bias = (mode == 8) ? 0 : -8 * ffn_sum_i8(at, K);
        for (r = 0; r < NH; r++) {
            double g, u;
            int32_t ag, au;
            if (mode == 8) {
                ag = dot_i8(L->qg8 + (size_t)r * K, at, K);
                au = dot_i8(L->qu8 + (size_t)r * K, at, K);
                g = (double)ag * (double)L->sg8[r] * s_in;
                u = (double)au * (double)L->su8[r] * s_in;
            } else {
                ag = ffn_dot_i4(L->qg4 + (size_t)r * (K / 2), at, K, bias);
                au = ffn_dot_i4(L->qu4 + (size_t)r * (K / 2), at, K, bias);
                g = (double)ag * (double)L->sg4[r] * s_in;
                u = (double)au * (double)L->su4[r] * s_in;
            }
            accg[(size_t)t * NH + r] = ag;
            accu[(size_t)t * NH + r] = au;
            P[(size_t)t * NH + r] = (float)(ffn_silu_f32(g) * u);
            if (fabs((double)P[(size_t)t * NH + r]) > pmax)
                pmax = fabs((double)P[(size_t)t * NH + r]);
            if (fabs(g) > gmax) gmax = fabs(g);
            if (fabs(u) > umax) umax = fabs(u);
        }
    }
    if (!(pmax > 0.0)) { free(accg); /* keep the early exit simple */
        free(accu); free(accd); free(mid); free(midi); free(P); free(qmg);
        free(qmu); free(acc_max); free(lsbv); free(errv); free(out8);
        return -1;
    }
    R->p_absmax = pmax;
    R->s_mid = (float)(pmax / 127.0);
    R->mq_mid = (int32_t)(4096.0f / R->s_mid);

    /* ---- integer mid, LSB error, down accumulators ---- */
    for (t = 0; t < ntok; t++) {
        int8_t *mt = mid + (size_t)t * NH;
        double dmid = 0.0, da = 0.0, dbb = 0.0;
        double dp = 0.0, pa = 0.0, pb = 0.0;
        int32_t biasd;
        for (r = 0; r < NH; r++) {
            double mg = (mode == 8) ? (double)L->sg8[r] * s_in * 65536.0
                                    : (double)L->sg4[r] * s_in * 65536.0;
            double mu = (mode == 8) ? (double)L->su8[r] * s_in * 65536.0
                                    : (double)L->su4[r] * s_in * 65536.0;
            int32_t ag = accg[(size_t)t * NH + r];
            int32_t au = accu[(size_t)t * NH + r];
            int32_t zg, zu, pr;
            long ideal;
            double d;
            if (deq_mode == 0) {
                zg = ffn_deq_wide(ag, qmg[r]);
                zu = ffn_deq_wide(au, qmu[r]);
                pr = ffn_prod_q16(ffn_silu_qk(zg, kg), zu, ku);
                if (fabs((double)zg) > zgm) zgm = fabs((double)zg);
                if (fabs((double)zu) > zum) zum = fabs((double)zu);
            } else {
                zg = ffn_deq_legacy(ag, mg);
                zu = ffn_deq_legacy(au, mu);
                pr = (deq_mode == 2)
                    ? ffn_mulq16_unsigned(ffn_silu_q16(zg), zu)
                    : ffn_mulq16(ffn_silu_q16(zg), zu);
            }
            mt[r] = (int8_t)ffn_requant_mid(pr, R->mq_mid);
            {
                double pv = (double)pr / 65536.0;
                double pt = (double)P[(size_t)t * NH + r];
                dp += pv * pt; pa += pv * pv; pb += pt * pt;
            }

            ideal = lrint(P[(size_t)t * NH + r] / R->s_mid);
            if (ideal > 127) ideal = 127;
            if (ideal < -127) ideal = -127;
            midi[(size_t)t * NH + r] = (int8_t)ideal;
            d = fabs((double)mt[r] - (double)ideal);
            lsbv[nl++] = d;
            dmid += (double)mt[r] * (double)R->s_mid * (double)P[(size_t)t * NH + r];
            da += (double)mt[r] * (double)R->s_mid * (double)mt[r] * (double)R->s_mid;
            dbb += (double)P[(size_t)t * NH + r] * (double)P[(size_t)t * NH + r];
        }
        R->cos_mid += (da > 0.0 && dbb > 0.0) ? dmid / (sqrt(da) * sqrt(dbb)) : 0.0;
        R->cos_p += (pa > 0.0 && pb > 0.0) ? dp / (sqrt(pa) * sqrt(pb)) : 0.0;

        biasd = -8 * ffn_sum_i8(mt, NH);
        for (r = 0; r < K; r++) {
            int32_t a;
            if (mode == 8) {
                a = dot_i8(L->qd8 + (size_t)r * NH, mt, NH);
            } else {
                a = ffn_dot_i4(L->qd4 + (size_t)r * (NH / 2), mt, NH, biasd);
            }
            accd[(size_t)t * K + r] = a;
            if (a < 0) a = -a;
            if (a > acc_max[r]) acc_max[r] = a;
        }
    }

    /* ---- s_out from the down accumulator, as the device calibrates it ---- */
    {
        double need0 = 0.0;
        for (r = 0; r < K; r++) {
            double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
            double s = swd * (double)R->s_mid * (double)acc_max[r] / FFN_CAL_TARGET;
            if (s > need0) need0 = s;
            if (acc_max[r] > R->acc_dmax) R->acc_dmax = acc_max[r];
        }
        if (!(need0 > 0.0)) { R->zero_out = 1; need0 = 1.0; }
        need = need0;
        R->s_out = (float)need;
    }

    /* ---- isolate the stages: mid -> down -> int8 out ---- */
    for (t = 0; t < ntok; t++) {
        const int8_t *it = midi + (size_t)t * NH;
        double cd = 0.0, ci = 0.0, ad = 0.0, ai = 0.0, bb = 0.0;
        int32_t bi = -8 * ffn_sum_i8(it, NH);
        for (r = 0; r < K; r++) {
            double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
            double y = (mode == 8)
                ? dot_i8_f32(L->qd8 + (size_t)r * NH, P + (size_t)t * NH, NH, swd)
                : dot_i4_f32(L->qd4 + (size_t)r * (NH / 2), P + (size_t)t * NH, NH, swd);
            double yd = (double)accd[(size_t)t * K + r] * swd * (double)R->s_mid;
            int32_t acci = (mode == 8)
                ? dot_i8(L->qd8 + (size_t)r * NH, it, NH)
                : ffn_dot_i4(L->qd4 + (size_t)r * (NH / 2), it, NH, bi);
            double yi = (double)acci * swd * (double)R->s_mid;
            cd += yd * y; ad += yd * yd; bb += y * y;
            ci += yi * y; ai += yi * yi;
        }
        R->cos_down += (ad > 0.0 && bb > 0.0) ? cd / (sqrt(ad) * sqrt(bb)) : 0.0;
        R->cos_ideal += (ai > 0.0 && bb > 0.0) ? ci / (sqrt(ai) * sqrt(bb)) : 0.0;
    }
    R->cos_down /= ntok;
    R->cos_ideal /= ntok;

    /* ---- evaluate: out int8 vs f32 truth, cos, saturation ---- */
    for (t = 0; t < ntok; t++) {
        const int8_t *mt = mid + (size_t)t * NH;
        double dout = 0.0, da = 0.0, dbb = 0.0;
        int sm = 0, so = 0;
        for (r = 0; r < K; r++) {
            int32_t acc = accd[(size_t)t * K + r];
            double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
            double md = swd * (double)R->s_mid / (double)R->s_out;
            double y, ym;
            int64_t v = ((int64_t)acc * (int64_t)ffn_qp16_of((float)md) + 32768) >> 16;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            out8[r] = (int8_t)v;
            y = (mode == 8)
                ? dot_i8_f32(L->qd8 + (size_t)r * NH, P + (size_t)t * NH, NH, swd)
                : dot_i4_f32(L->qd4 + (size_t)r * (NH / 2), P + (size_t)t * NH, NH, swd);
            ym = (double)out8[r] * (double)R->s_out;
            dout += ym * y;
            da += ym * ym;
            dbb += y * y;
            errv[ne++] = fabs(ym - y);
            if (out8[r] >= 127 || out8[r] <= -127) so++;
        }
        {
            double c = (da > 0.0 && dbb > 0.0) ? dout / (sqrt(da) * sqrt(dbb)) : 0.0;
            R->cos_out += c;
            if (t == 0 || c < R->cos_out_min) R->cos_out_min = c;
        }
        for (r = 0; r < NH; r++) if (mt[r] >= 127 || mt[r] <= -127) sm++;
        R->sat_mid += 100.0 * sm / NH;
        R->sat_out += 100.0 * so / K;
    }
    R->cos_out /= ntok;
    R->cos_mid /= ntok;
    R->cos_p /= ntok;
    R->sat_mid /= ntok;
    R->sat_out /= ntok;
    R->z_g_max = zgm;
    R->z_u_max = zum;
    pct4(lsbv, nl, &R->lsb_p50, &R->lsb_p90, &R->lsb_p99, &R->lsb_max);
    pct4(errv, ne, &R->err_p50, &R->err_p90, &R->err_p99, &R->err_max);
    R->max_out = R->err_max;

    /* down multiplier range and the m_r*acc_max invariant */
    for (r = 0; r < K; r++) {
        double swd = (mode == 8) ? (double)L->sd8[r] : (double)L->sd4[r];
        double md = swd * (double)R->s_mid / (double)R->s_out;
        double wr = md * (double)acc_max[r];
        if (md > FFN_Q16_CLAMP) R->d_clamped++;
        if (md < md_min) md_min = md;
        if (md > md_max) md_max = md;
        if (wr > R->worst_row) R->worst_row = wr;
    }
    R->m_d_min = md_min;
    R->m_d_max = md_max;

    free(accg); free(accu); free(accd); free(mid); free(midi); free(P);
    free(qmg); free(qmu); free(acc_max); free(lsbv); free(errv); free(out8);
    return 0;
}

static void print_table(const char *title, const run_t *R)
{
    int li;
    printf("\n%s\n", title);
    printf("%-3s %-9s %-9s %-9s %-9s %-9s %-9s %-7s %-7s %-7s %-7s %-7s %-8s %-3s %-9s\n",
           "lay", "cos_avg", "cos_min", "cos_mid", "cos_P", "cos_dn", "cos_id",
           "LSB50", "LSB99", "LSBmax", "satM%", "satO%", "m_max", "k", "clamp0/old");
    for (li = 0; li < NL; li++) {
        printf("%-3d %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-9.6f %-7.2f %-7.2f %-7.0f"
               " %-7.2f %-7.2f %-8.3f %-3d %d/%d\n",
               li, R[li].cos_out, R[li].cos_out_min, R[li].cos_mid, R[li].cos_p,
               R[li].cos_down, R[li].cos_ideal,
               R[li].lsb_p50, R[li].lsb_p99, R[li].lsb_max,
               R[li].sat_mid, R[li].sat_out, R[li].m_g_max, R[li].k_g,
               0, R[li].old_clamp_g);
    }
}

/* pool a per-token run into a per-layer summary */
static void merge_run(run_t *a, const run_t *b)
{
    int n = a->ntok;
    double den = (double)(n + 1);
    a->cos_out = (a->cos_out * n + b->cos_out) / den;
    a->cos_mid = (a->cos_mid * n + b->cos_mid) / den;
    a->cos_p = (a->cos_p * n + b->cos_p) / den;
    a->cos_down = (a->cos_down * n + b->cos_down) / den;
    a->cos_ideal = (a->cos_ideal * n + b->cos_ideal) / den;
    a->sat_mid = (a->sat_mid * n + b->sat_mid) / den;
    a->sat_out = (a->sat_out * n + b->sat_out) / den;
    if (n == 0 || b->cos_out_min < a->cos_out_min) a->cos_out_min = b->cos_out_min;
    if (b->max_out > a->max_out) a->max_out = b->max_out;
    if (b->err_p99 > a->err_p99) a->err_p99 = b->err_p99;
    if (b->lsb_p50 > a->lsb_p50) a->lsb_p50 = b->lsb_p50;
    if (b->lsb_p90 > a->lsb_p90) a->lsb_p90 = b->lsb_p90;
    if (b->lsb_p99 > a->lsb_p99) a->lsb_p99 = b->lsb_p99;
    if (b->lsb_max > a->lsb_max) a->lsb_max = b->lsb_max;
    if (b->err_max > a->err_max) a->err_max = b->err_max;
    if (b->z_g_max > a->z_g_max) a->z_g_max = b->z_g_max;
    if (b->z_u_max > a->z_u_max) a->z_u_max = b->z_u_max;
    if (b->m_g_max > a->m_g_max) a->m_g_max = b->m_g_max;
    if (a->m_g_min <= 0.0 || b->m_g_min < a->m_g_min) a->m_g_min = b->m_g_min;
    if (b->m_d_max > a->m_d_max) a->m_d_max = b->m_d_max;
    if (a->m_d_min <= 0.0 || b->m_d_min < a->m_d_min) a->m_d_min = b->m_d_min;
    if (b->worst_row > a->worst_row) a->worst_row = b->worst_row;
    if (b->acc_dmax > a->acc_dmax) a->acc_dmax = b->acc_dmax;
    if (b->p_absmax > a->p_absmax) a->p_absmax = b->p_absmax;
    if (b->old_clamp_g > a->old_clamp_g) a->old_clamp_g = b->old_clamp_g;
    if (b->old_clamp_u > a->old_clamp_u) a->old_clamp_u = b->old_clamp_u;
    if (b->d_clamped > a->d_clamped) a->d_clamped = b->d_clamped;
    if (b->k_g > a->k_g) a->k_g = b->k_g;
    if (b->k_u > a->k_u) a->k_u = b->k_u;
    a->mq_mid = b->mq_mid;
    a->s_mid = b->s_mid;
    a->s_out = b->s_out;
    a->ntok = n + 1;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "../../../models/LFM3-1.7B-Q4_K_M.gguf";
    int ntok = argc > 2 ? atoi(argv[2]) : 4;
    static const int regress_layers[3] = { 0, 12, 23 };
    run_t t8[NL], t4[NL], b8[NL], b4[NL], lg1, lg2;
    lfm_t spec;
    uint32_t *tok;
    float *cap;
    int top1 = -1;
    int li, i, t, rc = 0, have_regress = 0, regress_ok = 1;
    int pass_cos = 1, worst8_li = -1, worst8m_li = -1, worst4_li = -1, worst4m_li = -1;
    double worst8 = 2.0, worst8min = 2.0, worst4 = 2.0, worst4min = 2.0;
    double worstb8 = 2.0, worstb8min = 2.0, worstb4 = 2.0, worstb4min = 2.0;
    uint32_t st = 20240517u;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (ntok < 1) ntok = 1;
    if (ntok > MAX_TOK) ntok = MAX_TOK;
    tok = malloc((size_t)ntok * 4);
    if (!tok) { printf("oom\n"); return 1; }

    gguf_open(path);
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

    lfm_init(&spec, NL, HID, INT, 16, 4, 128, 256, 128000);
    cap = malloc((size_t)NL * ntok * HID * 4);
    if (!cap) { printf("oom\n"); return 1; }
    printf("real FFN study: %s, %d tokens, all %d layers\n", path, ntok, NL);
    printf("input: full f32 forward (mirror of backend/cpu.rs); FFN input "
           "captured before every ffn_gate/up\n");
    if (lfm_forward(&spec, tok, ntok, cap, &top1) != 0) {
        printf("forward failed\n");
        return 1;
    }
    printf("lm_head argmax of the last token = %d\n", top1);

    for (li = 0; li < NL; li++) {
        char nm[64];
        const tinfo_t *tg, *tu, *td;
        layer_t Ly;
        float *wg, *wu, *wd;
        int8_t *x8;
        double rms = 0.0, amax = 0.0;

        /* ---- this layer's real inputs ---- */
        for (t = 0; t < ntok; t++) {
            const float *src = cap + ((size_t)li * ntok + t) * HID;
            for (i = 0; i < HID; i++) {
                double v = fabs((double)src[i]);
                if (v > amax) amax = v;
                rms += v * v;
            }
        }
        if (!(amax > 0.0)) { printf("layer %d: zero input\n", li); rc = 1; break; }
        x8 = malloc((size_t)ntok * HID);
        if (!x8) { printf("oom\n"); rc = 1; break; }

        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", li); tg = gguf_find(nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", li);   tu = gguf_find(nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", li); td = gguf_find(nm);
        wg = gguf_matrix(tg, INT, HID);
        wu = gguf_matrix(tu, INT, HID);
        wd = gguf_matrix(td, HID, INT);
        if (!wg || !wu || !wd || layer_alloc(&Ly, HID, INT) != 0) {
            printf("layer %d: load/alloc failed\n", li);
            free(wg); free(wu); free(wd); free(x8);
            rc = 1; break;
        }
        for (i = 0; i < INT; i++) {
            Ly.sg8[i] = pack_i8_row(Ly.qg8 + (size_t)i * HID, wg + (size_t)i * HID, HID);
            Ly.su8[i] = pack_i8_row(Ly.qu8 + (size_t)i * HID, wu + (size_t)i * HID, HID);
            Ly.sg4[i] = ffn_pack_i4_row(Ly.qg4 + (size_t)i * (HID / 2),
                                        wg + (size_t)i * HID, HID);
            Ly.su4[i] = ffn_pack_i4_row(Ly.qu4 + (size_t)i * (HID / 2),
                                        wu + (size_t)i * HID, HID);
        }
        for (i = 0; i < HID; i++) {
            Ly.sd8[i] = pack_i8_row(Ly.qd8 + (size_t)i * INT, wd + (size_t)i * INT, INT);
            Ly.sd4[i] = ffn_pack_i4_row(Ly.qd4 + (size_t)i * (INT / 2),
                                        wd + (size_t)i * INT, INT);
        }
        free(wg); free(wu); free(wd);

        printf("layer %2d: ffn_in rms=%.4f absmax=%.4f  tok_absmax:",
               li, sqrt(rms / ((double)ntok * HID)), amax);
        for (t = 0; t < ntok; t++) {
            const float *src = cap + ((size_t)li * ntok + t) * HID;
            float ta = 0.0f;
            for (i = 0; i < HID; i++) { float v = fabsf(src[i]); if (v > ta) ta = v; }
            printf(" %.3f", ta);
        }
        printf("\n");

        /* ---- per-token calibration: each token gets its own s_in/s_mid, so
         * the number isolates the dequant from a batch-wide scale outlier ---- */
        memset(&t8[li], 0, sizeof(t8[li]));
        memset(&t4[li], 0, sizeof(t4[li]));
        memset(&lg1, 0, sizeof(lg1));
        memset(&lg2, 0, sizeof(lg2));
        for (t = 0; t < ntok; t++) {
            const float *src = cap + ((size_t)li * ntok + t) * HID;
            double s_in = 0.0;
            run_t tmp;
            for (i = 0; i < HID; i++) {
                double v = fabs((double)src[i]);
                if (v > s_in) s_in = v;
            }
            s_in /= FFN_CAL_TARGET;
            for (i = 0; i < HID; i++) {
                long q = lrintf(src[i] / (float)s_in);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                x8[i] = (int8_t)q;
            }
            if (run_layer(&Ly, 8, 0, x8, 1, s_in, &tmp) != 0) {
                printf("layer %d tok %d: int8 run failed\n", li, t);
                rc = 1; break;
            }
            merge_run(&t8[li], &tmp);
            if (run_layer(&Ly, 4, 0, x8, 1, s_in, &tmp) != 0) {
                printf("layer %d tok %d: int4 run failed\n", li, t);
                rc = 1; break;
            }
            merge_run(&t4[li], &tmp);
            if (li == regress_layers[0] || li == regress_layers[1] ||
                li == regress_layers[2]) {
                if (run_layer(&Ly, 4, 1, x8, 1, s_in, &tmp) != 0) {
                    printf("layer %d tok %d: regression run failed\n", li, t);
                    rc = 1; break;
                }
                merge_run(&lg1, &tmp);
                if (run_layer(&Ly, 4, 2, x8, 1, s_in, &tmp) != 0) {
                    printf("layer %d tok %d: regression run failed\n", li, t);
                    rc = 1; break;
                }
                merge_run(&lg2, &tmp);
            }
        }
        if (rc != 0) { layer_free(&Ly); free(x8); break; }

        /* ---- device-like check: one s_in/s_mid shared by the whole batch ---- */
        if (ntok > 0) {
            double s_in_b = amax / FFN_CAL_TARGET;
            for (t = 0; t < ntok; t++) {
                const float *src = cap + ((size_t)li * ntok + t) * HID;
                for (i = 0; i < HID; i++) {
                    long q = lrintf(src[i] / (float)s_in_b);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    x8[(size_t)t * HID + i] = (int8_t)q;
                }
            }
            if (run_layer(&Ly, 8, 0, x8, ntok, s_in_b, &b8[li]) != 0 ||
                run_layer(&Ly, 4, 0, x8, ntok, s_in_b, &b4[li]) != 0) {
                printf("layer %d: batch run failed\n", li);
                rc = 1; layer_free(&Ly); free(x8); break;
            }
        }

        if (li == regress_layers[0] || li == regress_layers[1] ||
            li == regress_layers[2]) {
            have_regress = 1;
            printf("  regress lay %2d int4: old Q16+clamp cos=%.6f min=%.6f satM=%.2f%% | "
                   "old unsigned cos=%.6f min=%.6f satM=%.2f%%\n",
                   li, lg1.cos_out, lg1.cos_out_min, lg1.sat_mid,
                   lg2.cos_out, lg2.cos_out_min, lg2.sat_mid);
            if (!(lg1.cos_out < PASS_COS || lg1.sat_mid > PASS_SAT_PCT)) regress_ok = 0;
            if (!(lg2.cos_out < PASS_COS && lg2.sat_mid > 20.0)) regress_ok = 0;
        }

        layer_free(&Ly);
        free(x8);
    }

    if (rc == 0) {
        print_table("int8 weights (per-row absmax/127), wide dequant",
                    t8);
        print_table("int4 weights (per-row absmax/7), wide dequant",
                    t4);

        for (li = 0; li < NL; li++) {
            if (t8[li].cos_out < worst8) { worst8 = t8[li].cos_out; worst8_li = li; }
            if (t8[li].cos_out_min < worst8min) { worst8min = t8[li].cos_out_min; worst8m_li = li; }
            if (t4[li].cos_out < worst4) { worst4 = t4[li].cos_out; worst4_li = li; }
            if (t4[li].cos_out_min < worst4min) { worst4min = t4[li].cos_out_min; worst4m_li = li; }
            if (t8[li].cos_out < PASS_COS || t8[li].cos_out_min < PASS_COS ||
                t4[li].cos_out < PASS_COS || t4[li].cos_out_min < PASS_COS ||
                t8[li].sat_mid > PASS_SAT_PCT || t8[li].sat_out > PASS_SAT_PCT ||
                t4[li].sat_mid > PASS_SAT_PCT || t4[li].sat_out > PASS_SAT_PCT)
                pass_cos = 0;
        }
        printf("\nper-token scales: worst int8 cos(avg)=%.6f (lay %d) cos(min-token)=%.6f (lay %d)\n",
               worst8, worst8_li, worst8min, worst8m_li);
        printf("per-token scales: worst int4 cos(avg)=%.6f (lay %d) cos(min-token)=%.6f (lay %d)\n",
               worst4, worst4_li, worst4min, worst4m_li);
        for (li = 0; li < NL; li++) {
            if (b8[li].cos_out < worstb8) worstb8 = b8[li].cos_out;
            if (b8[li].cos_out_min < worstb8min) worstb8min = b8[li].cos_out_min;
            if (b4[li].cos_out < worstb4) worstb4 = b4[li].cos_out;
            if (b4[li].cos_out_min < worstb4min) worstb4min = b4[li].cos_out_min;
        }
        printf("one scale per batch: worst int8 cos(avg)=%.6f cos(min-token)=%.6f\n",
               worstb8, worstb8min);
        printf("one scale per batch: worst int4 cos(avg)=%.6f cos(min-token)=%.6f\n",
               worstb4, worstb4min);
        printf("layers below %.2f with one batch scale: int8", PASS_COS);
        for (li = 0; li < NL; li++)
            if (b8[li].cos_out_min < PASS_COS) printf(" %d(%.4f)", li, b8[li].cos_out_min);
        printf(" | int4");
        for (li = 0; li < NL; li++)
            if (b4[li].cos_out_min < PASS_COS) printf(" %d(%.4f)", li, b4[li].cos_out_min);
        printf("\n");
        printf("regression gates: %s\n",
               (have_regress && regress_ok) ? "both old formats rejected"
                                            : "FAILED (old format was not rejected)");
        {
            int n8 = 0, n4 = 0;
            for (li = 0; li < NL; li++) {
                if (t8[li].cos_out_min >= PASS_COS) n8++;
                if (t4[li].cos_out_min >= PASS_COS) n4++;
            }
            printf("\nper-token min-token >= %.2f: %d/%d layers int8, %d/%d layers int4\n",
                   PASS_COS, n8, NL, n4, NL);
        }
        printf("%s (need: cos(out) >= %.3f on every layer and token, int8 and "
               "int4; saturation <= %.1f%%; both regression gates rejected)\n",
               (pass_cos && have_regress && regress_ok) ? "REAL SIM OK" : "REAL SIM FAIL",
               PASS_COS, PASS_SAT_PCT);
        if (!(pass_cos && have_regress && regress_ok)) rc = 2;
    }

    free(tok); free(cap);
    return rc;
}
