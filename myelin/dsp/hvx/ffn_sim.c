/*
 * Offline validation of the FFN layer (gate/up int4 -> SiLU -> mid int8 ->
 * down int4) -- no device needed.
 *
 * The stand exists because the first version of it was numerically degenerate:
 * gate/up were fed with sa = 1, so silu_q16() received the dequantised gate in
 * Q0 instead of Q16 and the mid buffer came out identically zero.  "0
 * mismatches" then only proved that the host mirrored the same mistake.
 *
 * What is checked here
 *   1. the scales are the ones the kernels need: the gate/up Q16 multiplier
 *      m = sw*s_in*2^16 stays below the 0.99997 clamp, the SiLU argument is a
 *      real Q16 value (|zg| >= 1024), and the mid buffer is not degenerate;
 *   2. s_mid from the measured P = silu(gate)*up range, s_out from the down
 *      accumulator range (same rule as conv_sim);
 *   3. cos / max_abs_err of the integer pipeline against an f32 reference on
 *      the *same dequantised weights*, per token, plus the mid-buffer LSB
 *      error and the int8 saturation of mid and out.
 *
 * Build:  cc -O2 -o ffn_sim ffn_sim.c -lm
 * Run:    ./ffn_sim [ntok]
 *
 * Exit code: 0 pass, 1 allocation/calibration failure, 2 threshold failure.
 */
#include "ffn_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The gate/up dequantiser is wide (see ffn_ref.h): m = sw*s_in*2^16 spans
 * 0.3..12 (int8) and 6..215 (int4) on the real tensors, so the Q16 clamp is
 * gone.  The synthetic model crosses m = 1 (0.35..2.60) and the stand also
 * has explicit regression gates that must reject the two old bugs.
 * cos 0.99: with int4 weights and an int8 mid the per-lane error is ~7 % on
 * the weights and ~2 % on the mid, so after the 7680-lane down projection the
 * ceiling is ~0.995-0.997; asking for 0.999 would ask for more precision than
 * the format holds. */
#define PASS_COS      0.99
#define PASS_SAT_PCT  5.0
#define ZG_MIN_Q16    1024.0f       /* |zg| below 1/64 means "Q0, not Q16" */

/* run one token set through the integer pipeline in a given dequant mode
 * (0 wide, 1 legacy clamp, 2 legacy clamp + unsigned low product) */
static int eval_mode(ffn_model_t *m, int mode, double *worst_cos,
                     double *worst_cos_mid, double *sat_mid, double *sat_out)
{
    float *yref = (float *)malloc((size_t)m->K * sizeof(float));
    float *ppipe = (float *)malloc((size_t)m->NH * sizeof(float));
    int8_t *mid = (int8_t *)malloc((size_t)m->NH);
    int8_t *out8 = (int8_t *)malloc((size_t)m->K);
    long clip_mid = 0, clip_out = 0, n = 0;
    int t;

    if (!yref || !ppipe || !mid || !out8) {
        free(yref); free(ppipe); free(mid); free(out8);
        return -1;
    }
    *worst_cos = 2.0;
    *worst_cos_mid = 2.0;
    for (t = 0; t < m->ntok; t++) {
        double c, cm;
        ffn_ref_token(m, t, yref, ppipe, 1);
        ffn_int_mid_mode(m, t, mid, NULL, mode);
        ffn_int_out(m, mid, out8, NULL);
        c = ffn_cos_i8(out8, yref, m->K, m->s_out);
        cm = ffn_cos_i8(mid, ppipe, m->NH, m->s_mid);
        if (c < *worst_cos) *worst_cos = c;
        if (cm < *worst_cos_mid) *worst_cos_mid = cm;
        clip_mid += ffn_count_clip(mid, m->NH);
        clip_out += ffn_count_clip(out8, m->K);
        n += m->NH;
    }
    *sat_mid = 100.0 * (double)clip_mid / (double)n;
    *sat_out = 100.0 * (double)clip_out / (double)n;
    free(yref); free(ppipe); free(mid); free(out8);
    return 0;
}

int main(int argc, char **argv)
{
    int ntok = (argc > 1) ? atoi(argv[1]) : 8;
    ffn_model_t m;
    ffn_calib_t ci;
    float *yref = NULL, *ypipe = NULL, *ppipe = NULL;
    int8_t *mid = NULL, *out8 = NULL;
    double worst_cos = 2.0, worst_cos_pipe = 2.0, worst_err = 0.0;
    double lsb_rms = 0.0, lsb_max = 0.0;
    long lsb_n = 0, lsb_big = 0;
    int worst_tok = 0, t, r, pass;
    double sat_mid, sat_out;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (ntok < 1) ntok = 1;

    if (ffn_model_synth(&m, 2048, 7680, ntok, 0x5eedu) != 0) {
        printf("model alloc failed\n");
        return 1;
    }
    if (ffn_calibrate(&m, &ci) != 0) {
        printf("calibration failed (no signal in x or P)\n");
        return 1;
    }

    printf("FFN sim: K=%d NH=%d ntok=%d   int4 weights, int8 mid/out\n",
           m.K, m.NH, m.ntok);
    printf("scales: s_in=%.6g  s_mid=%.6g (P absmax/127)  mq_mid=%d  s_out=%.6g\n",
           ci.s_in, ci.s_mid, ci.mq_mid, ci.s_out);
    printf("real ranges: |gate|=%.4f |up|=%.4f |P|=%.4f   |zg| max=%.0f (Q16 units)\n",
           ci.gate_absmax, ci.up_absmax, ci.p_absmax, ci.zg_absmax);
    printf("dequant: m_gate=[%.4f..%.4f] m_down=[%.4f..%.4f] k_gate=%d k_up=%d"
           "  |acc*m| max g=%.0f u=%.0f (int32)\n",
           ci.m_g_min, ci.m_g_max, ci.m_d_min, ci.m_d_max,
           ci.kg, ci.ku, ci.z_g_max, ci.z_u_max);
    printf("old Q16 format would clamp: gate %d / %d rows, up %d / %d rows, down %d\n",
           ci.g_clamped, m.NH, ci.u_clamped, m.NH, ci.d_clamped);
    printf("down: acc_max=%d worst m_d*acc_max=%.1f (<= 127)  silu clamp lanes=%d\n",
           ci.acc_dmax, ci.worst_row, ci.silu_clamp);
    printf("mid: %.1f%% lanes == 0   occupied range %.1f/127 LSB\n",
           100.0 * (double)ci.mid_zero / (double)ci.mid_n,
           (double)ci.p_absmax / (double)ci.s_mid);

    yref  = (float *)malloc((size_t)m.K * sizeof(float));
    ypipe = (float *)malloc((size_t)m.K * sizeof(float));
    ppipe = (float *)malloc((size_t)m.NH * sizeof(float));
    mid   = (int8_t *)malloc((size_t)m.NH);
    out8  = (int8_t *)malloc((size_t)m.K);
    if (!yref || !ypipe || !ppipe || !mid || !out8) {
        printf("alloc failed\n");
        return 1;
    }

    printf("\n%-4s %-10s %-11s %-9s %-9s %-8s %-8s\n",
           "tok", "cos(out)", "maxerr(out)", "cos(mid)", "LSB rms", "LSB max", "mid0%");
    for (t = 0; t < m.ntok; t++) {
        double dsum = 0.0, dmax = 0.0, c, e, cm;
        long n = 0, nz = 0;

        ffn_ref_token(&m, t, yref, NULL, 0);
        ffn_ref_token(&m, t, ypipe, ppipe, 1);
        ffn_int_mid(&m, t, mid, NULL);

        for (r = 0; r < m.NH; r++) {
            long ideal = lrintf(ppipe[r] / m.s_mid);
            double d;
            if (ideal > 127) ideal = 127;
            if (ideal < -127) ideal = -127;
            d = (double)mid[r] - (double)ideal;
            if (d < 0) d = -d;
            dsum += d * d;
            if (d > dmax) dmax = d;
            if (d >= 3.0) lsb_big++;
            n++;
            if (mid[r] == 0) nz++;
        }
        lsb_rms += dsum;
        lsb_n += n;
        if (dmax > lsb_max) lsb_max = dmax;

        cm = ffn_cos_i8(mid, ppipe, m.NH, m.s_mid);
        ffn_int_out(&m, mid, out8, NULL);
        c = ffn_cos_i8(out8, yref, m.K, m.s_out);
        e = ffn_maxabs_i8(out8, yref, m.K, m.s_out);

        printf("%-4d %-10.6f %-11.5g %-9.6f %-9.3f %-8.0f %-8.1f\n",
               t, c, e, cm, sqrt(dsum / (double)n), dmax, 100.0 * (double)nz / (double)n);
        if (c < worst_cos) { worst_cos = c; worst_tok = t; }
        if (cm < worst_cos_pipe) worst_cos_pipe = cm;
        if (e > worst_err) worst_err = e;
    }
    lsb_rms = sqrt(lsb_rms / (double)lsb_n);

    /* saturation of the final buffers, all tokens */
    {
        long clip_mid = 0, clip_out = 0, n = 0;
        for (t = 0; t < m.ntok; t++) {
            ffn_ref_token(&m, t, ypipe, ppipe, 1);
            ffn_int_mid(&m, t, mid, NULL);
            clip_mid += ffn_count_clip(mid, m.NH);
            ffn_int_out(&m, mid, out8, NULL);
            clip_out += ffn_count_clip(out8, m.K);
            n += m.NH;
        }
        sat_mid = 100.0 * (double)clip_mid / (double)n;
        sat_out = 100.0 * (double)clip_out / (double)n;
    }

    printf("\nworst: cos(out)=%.6f (tok %d)  cos(mid)=%.6f  maxerr(out)=%.4g\n",
           worst_cos, worst_tok, worst_cos_pipe, worst_err);
    printf("mid LSB error: rms=%.3f max=%.0f  (%ld lanes with |d|>=3)\n",
           lsb_rms, lsb_max, lsb_big);
    printf("int8 saturation: mid=%.2f%% out=%.2f%% (pass: <= %.1f%%)\n",
           sat_mid, sat_out, PASS_SAT_PCT);

    pass = (worst_cos >= PASS_COS)
        && sat_mid <= PASS_SAT_PCT && sat_out <= PASS_SAT_PCT
        && ci.zg_absmax >= ZG_MIN_Q16
        && ci.d_clamped == 0
        && 100.0 * (double)ci.mid_zero / (double)ci.mid_n < 50.0;

    if (ci.zg_absmax < ZG_MIN_Q16)
        printf("\nDEGENERATE: |zg| max = %.0f Q16 units -> silu_qk() sees Q0, not Q16\n",
               ci.zg_absmax);

    /* ---- regression gates: the old formats must fail this stand ---- */
    {
        ffn_model_t gm;
        ffn_calib_t gc;
        double gc_new, gc_clamp, gc_uns, m_mid;
        double sat_new_mid, sat_new_out, sat_clamp_mid, sat_clamp_out;
        double sat_uns_mid, sat_uns_out;
        int r1 = -1, r2 = -1, r3 = -1;
        int gate_ok = 1;

        if (ffn_model_synth_scaled(&gm, 2048, 7680, m.ntok, 0x6a7e5u, 1.30f, 2.60f) != 0
            || ffn_calibrate(&gm, &gc) != 0) {
            printf("regression model failed\n");
            ffn_model_free(&gm);
            return 1;
        }
        r1 = eval_mode(&gm, 0, &gc_new, &m_mid, &sat_new_mid, &sat_new_out);
        r2 = eval_mode(&gm, 1, &gc_clamp, &m_mid, &sat_clamp_mid, &sat_clamp_out);
        r3 = eval_mode(&gm, 2, &gc_uns, &m_mid, &sat_uns_mid, &sat_uns_out);
        if (r1 || r2 || r3) {
            printf("regression eval failed\n");
            ffn_model_free(&gm);
            return 1;
        }
        printf("\nregression gates (model with m in [1.3,2.6], so every gate/up row "
               "exceeds the old Q16 clamp):\n");
        printf("  wide dequant        : cos(out)=%.6f  sat(mid)=%.2f%%  -> %s\n",
               gc_new, sat_new_mid, (gc_new >= PASS_COS && sat_new_mid <= PASS_SAT_PCT) ? "PASS" : "FAIL");
        printf("  old Q16+clamp       : cos(out)=%.6f  sat(mid)=%.2f%%  -> %s\n",
               gc_clamp, sat_clamp_mid, (gc_clamp < PASS_COS || sat_clamp_mid > PASS_SAT_PCT) ? "rejected (good)" : "NOT REJECTED");
        printf("  old unsigned mulq16 : cos(out)=%.6f  sat(mid)=%.2f%%  -> %s\n",
               gc_uns, sat_uns_mid, (gc_uns < PASS_COS && sat_uns_mid > 20.0) ? "rejected (good)" : "NOT REJECTED");
        if (!(gc_new >= PASS_COS && sat_new_mid <= PASS_SAT_PCT)) gate_ok = 0;
        if (!(gc_clamp < PASS_COS || sat_clamp_mid > PASS_SAT_PCT)) gate_ok = 0;
        if (!(gc_uns < PASS_COS && sat_uns_mid > 20.0)) gate_ok = 0;
        if (!gate_ok) pass = 0;
        ffn_model_free(&gm);
    }

    printf("\n%s (need: worst cos(out) >= %.3f, saturation <= %.1f%%, |zg| >= %.0f,"
           " mid buffer not degenerate, both regression gates rejected)\n",
           pass ? "SIM OK" : "SIM FAIL", PASS_COS, PASS_SAT_PCT, ZG_MIN_Q16);

    ffn_model_free(&m);
    free(yref); free(ypipe); free(ppipe); free(mid); free(out8);
    return pass ? 0 : 2;
}
