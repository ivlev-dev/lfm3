/*
 * Offline validation of the shortconv algorithm -- no device needed.
 *
 *   1. f32 reference on the dequantised int4 weights      (truth)
 *   2. integer pipeline with a plain 6-tap shift register
 *   3. integer pipeline exactly as the DSP: 8-slot ring + phase-rotated
 *      weights + hi/lo byte planes (the scalar model of the HVX vrmpy code)
 *
 * Checks
 *   - (2) and (3) agree bit for bit    -> validates the ring/phase trick
 *   - cos / max_abs_err of (2),(3) vs (1) -> quantisation loss of the layer
 *   - out_proj scale invariant: max_r m_r*acc_max[r] <= 127 (see conv_ref.h)
 *
 * Scale calibration, in the order the device needs it
 *   s_bc, s_y  per-tensor from the f32 absmax, 110/127 of full scale
 *              (13 % headroom; int8 saturation stays ~0)
 *   s_out      from the out_proj *accumulator* range:
 *              s_out = max_r(sw_out[r]*s_y*acc_max[r]) / 110.
 *              Deriving it from absmax(out_f32) instead bounds the real
 *              output but not the accumulators, and lets whole rows clip.
 *
 * Build:  cc -O2 -o conv_sim conv_sim.c -lm
 * Run:    ./conv_sim [H] [ntok]
 */
#include "conv_ref.h"

/* int8 full scale used by the calibrations (127 minus 13 % headroom). */
#define CAL_TARGET 110.0

/* Pass gate.  cos 0.99: the layer's residual error is set by the int8
 * quantisation of the 7 depthwise taps and of y, which costs ~0.4-0.6 %
 * (measured ceiling ~0.996); 0.999 would ask for more precision than the
 * format holds.  Saturation: a few percent is tolerated, 60 % is a bug.   */
#define PASS_COS      0.99
#define PASS_SAT_PCT  5.0

static float absmax32(const float *a, int n)
{
    float m = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float v = a[i] < 0.0f ? -a[i] : a[i];
        if (v > m) m = v;
    }
    return m;
}

static int count_clip(const int8_t *a, int n)
{
    int i, c = 0;
    for (i = 0; i < n; i++) if (a[i] >= 127 || a[i] <= -127) c++;
    return c;
}

int main(int argc, char **argv)
{
    int H    = (argc > 1) ? atoi(argv[1]) : 2048;
    int ntok = (argc > 2) ? atoi(argv[2]) : 12;
    int i, t, ch, bad_y = 0, bad_o = 0, worst_tok = 0, pass;
    double worst_cos = 2.0, worst_err = 0.0, worst_cos_y = 2.0;
    double sat_bcx, sat_y, sat_out, mrmax = 0.0, worst_row = 0.0;
    int32_t acc_gmax;
    ref_model_t m;
    ref_f32_t st;
    ref_convw_t W;
    ref_ring_t ring;
    float *out32 = NULL;
    float ab, ay, ao;
    int8_t *bcx8 = NULL, *y8a = NULL, *y8b = NULL;
    int8_t *out8a = NULL, *out8b = NULL;
    int16_t *hist6 = NULL;
    int32_t *acc_max = NULL;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (ref_model_build(&m, H, ntok, 0x5eed) != 0) { printf("alloc failed\n"); return 1; }

    memset(&st, 0, sizeof(st));
    st.hist    = (float *)calloc((size_t)6 * H, sizeof(float));
    st.act     = (float *)calloc((size_t)H, sizeof(float));
    st.bcx     = (float *)calloc((size_t)3 * H, sizeof(float));
    st.y       = (float *)calloc((size_t)H, sizeof(float));
    st.bcx_all = (float *)calloc((size_t)ntok * 3 * H, sizeof(float));
    st.y_all   = (float *)calloc((size_t)ntok * H, sizeof(float));
    out32      = (float *)calloc((size_t)ntok * H, sizeof(float));
    if (!st.hist || !st.act || !st.bcx || !st.y || !st.bcx_all || !st.y_all || !out32) return 1;

    /* ---- pass 1: f32 truth, then pick per-tensor scales with headroom ---- */
    ref_f32_run(&m, &st, out32);
    ab = absmax32(st.bcx_all, ntok * 3 * H);
    ay = absmax32(st.y_all,   ntok * H);
    ao = absmax32(out32,      ntok * H);
    m.s_in  = 1.0f;
    m.s_bc  = ab / CAL_TARGET;
    m.s_y   = ay / CAL_TARGET;
    printf("H=%d ntok=%d  calibration target=%.0f/127 (13%% headroom)\n", H, ntok, CAL_TARGET);
    printf("absmax real: bcx=%.4g y=%.4g out=%.4g\n", ab, ay, ao);
    printf("scales: s_bc=%.6g s_y=%.6g   (int8 = real/scale)\n", m.s_bc, m.s_y);

    if (ref_convw_init(&W, m.cw, H, m.s_bc, m.s_y) != 0) { printf("convw alloc failed\n"); return 1; }
    {
        int nz = 0;
        double mn = 1e30, mx = 0.0;
        for (ch = 0; ch < H; ch++) {
            double v = (double)m.sw_in[ch] * (double)m.s_in / (double)m.s_bc;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (W.kq[ch] == 0) nz++;
        }
        printf("conv kq: ksh=%d  zero-multiplier channels=%d\n", W.ksh, nz);
        printf("in_proj multiplier m = sw_in*s_in/s_bc: min=%.5f max=%.5f (Q16 clamp 0.99997)\n", mn, mx);
    }

    bcx8   = (int8_t *)calloc((size_t)3 * H, 1);
    y8a    = (int8_t *)calloc((size_t)ntok * H, 1);
    y8b    = (int8_t *)calloc((size_t)ntok * H, 1);
    out8a  = (int8_t *)calloc((size_t)ntok * H, 1);
    out8b  = (int8_t *)calloc((size_t)ntok * H, 1);
    hist6  = (int16_t *)calloc((size_t)6 * H, sizeof(int16_t));
    acc_max = (int32_t *)calloc((size_t)H, sizeof(int32_t));
    if (!bcx8 || !y8a || !y8b || !out8a || !out8b || !hist6 || !acc_max) return 1;

    /* ---- pass 2a: shift-register pipeline, y8 per token ---- */
    ref_i8_run_y(&m, &W, hist6, bcx8, y8a);

    /* ---- pass 2b: DSP ring/phase pipeline, y8 per token ---- */
    if (ref_ring_init(&ring, H) != 0) return 1;
    for (t = 0; t < ntok; t++)
        ref_ring_token(&m, &W, &ring, m.tok + (size_t)t * H, bcx8,
                       y8b + (size_t)t * H, NULL);

    /* ---- compare the two integer models (must be identical) ---- */
    for (i = 0; i < ntok * H; i++) if (y8a[i] != y8b[i]) bad_y++;
    printf("\nring/phase vs shift register: %d mismatches out of %d (y8)\n", bad_y, ntok * H);

    /* ---- pass 3: s_out from the out_proj accumulator range ---- */
    acc_gmax = ref_out_acc_profile(&m, y8b, ntok, acc_max);
    m.s_out  = ref_calib_s_out(&m, acc_max, (float)CAL_TARGET);
    worst_row = ref_s_out_worst(&m, acc_max, m.s_out, &mrmax);
    printf("\ns_out from the accumulator range:\n");
    printf("  acc_max (max |sum wq*y8| over rows/tokens) = %d\n", acc_gmax);
    printf("  s_out = max_r(sw_out*s_y*acc_max[r]) / %.0f = %.6g\n", CAL_TARGET, m.s_out);
    printf("  m_r max = %.5f   127/acc_max = %.5f   worst row m_r*acc_max[r] = %.1f (<= 127)\n",
           mrmax, 127.0 / (double)acc_gmax, worst_row);
    printf("  (m_r max and 127/acc_max come from different rows; the invariant is the row-wise\n"
           "   product in the third column, which must stay <= 127)\n");
    {
        double mn = 1e30, mx = 0.0;
        for (ch = 0; ch < H; ch++) {
            double v = (double)m.sw_out[ch] * (double)m.s_y / (double)m.s_out;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        printf("  out_proj multiplier m = sw_out*s_y/s_out: min=%.5f max=%.5f (Q16 clamp 0.99997)\n",
               mn, mx);
    }

    /* ---- pass 4: out_proj requantisation for both models ---- */
    for (t = 0; t < ntok; t++) {
        ref_out8(&m, y8a + (size_t)t * H, out8a + (size_t)t * H);
        ref_out8(&m, y8b + (size_t)t * H, out8b + (size_t)t * H);
    }
    for (i = 0; i < ntok * H; i++) if (out8a[i] != out8b[i]) bad_o++;

    /* ---- accuracy vs f32, per token ---- */
    printf("\n%-5s %-10s %-12s %-10s %-12s\n", "tok", "cos(out)", "maxerr(out)",
           "cos(y)", "maxerr(y)");
    for (t = 0; t < ntok; t++) {
        double c = ref_cos_i8(out8b + (size_t)t * H, out32 + (size_t)t * H, H, m.s_out);
        double e = ref_maxabs_i8(out8b + (size_t)t * H, out32 + (size_t)t * H, H, m.s_out);
        double cy = ref_cos_i8(y8b + (size_t)t * H, st.y_all + (size_t)t * H, H, m.s_y);
        double ey = ref_maxabs_i8(y8b + (size_t)t * H, st.y_all + (size_t)t * H, H, m.s_y);
        printf("%-5d %-10.6f %-12.5f %-10.6f %-12.5f\n", t, c, e, cy, ey);
        if (c < worst_cos) { worst_cos = c; worst_tok = t; }
        if (e > worst_err) worst_err = e;
        if (cy < worst_cos_y) worst_cos_y = cy;
    }
    printf("\nworst: cos(out)=%.6f (tok %d) maxerr(out)=%.5f  cos(y)=%.6f\n",
           worst_cos, worst_tok, worst_err, worst_cos_y);

    /* bcx8 is per token (the buffer is reused), so count its clips over all
     * tokens explicitly instead of looking at the last one only */
    {
        int clip_bcx = 0;
        for (t = 0; t < ntok; t++) {
            ref_bcx8(&m, m.tok + (size_t)t * H, bcx8);
            clip_bcx += count_clip(bcx8, 3 * H);
        }
        sat_bcx = 100.0 * clip_bcx / (3.0 * H * ntok);
    }
    sat_y   = 100.0 * count_clip(y8b, ntok * H) / (double)(ntok * H);
    sat_out = 100.0 * count_clip(out8b, ntok * H) / (double)(ntok * H);
    printf("int8 saturation: bcx=%.2f%% y=%.2f%% out=%.2f%%  (pass: <= %.1f%%)\n",
           sat_bcx, sat_y, sat_out, PASS_SAT_PCT);

    pass = (bad_y == 0 && bad_o == 0)
        && worst_cos >= PASS_COS
        && sat_out <= PASS_SAT_PCT && sat_y <= PASS_SAT_PCT && sat_bcx <= PASS_SAT_PCT
        && worst_row <= 127.0;
    printf("\n%s (need: 0 mismatches, worst cos(out) >= %.3f, saturation <= %.1f%%, m_r*acc_max <= 127)\n",
           pass ? "SIM OK" : "SIM FAIL", PASS_COS, PASS_SAT_PCT);

    ref_model_free(&m);
    ref_convw_free(&W);
    ref_ring_free(&ring);
    free(st.hist); free(st.act); free(st.bcx); free(st.y);
    free(st.bcx_all); free(st.y_all); free(out32);
    free(bcx8); free(y8a); free(y8b); free(out8a); free(out8b);
    free(hist6); free(acc_max);
    return pass ? 0 : 2;
}
