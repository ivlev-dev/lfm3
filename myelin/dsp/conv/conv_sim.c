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
 *
 * Build:  cc -O2 -o conv_sim conv_sim.c -lm
 * Run:    ./conv_sim [H] [ntok]
 */
#include "conv_ref.h"

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
    int i, t, ch, bad = 0, worst_tok = 0;
    double worst_cos = 2.0, worst_err = 0.0, worst_cos_y = 2.0;
    ref_model_t m;
    ref_f32_t st;
    ref_convw_t W;
    ref_ring_t ring;
    float *out32 = NULL;
    float ab, ay, ao;
    int8_t *bcx8 = NULL, *y8 = NULL, *out8a = NULL, *out8b = NULL, *y8_dsp = NULL;
    int16_t *hist6 = NULL;

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
    m.s_bc  = ab / 110.0f;
    m.s_y   = ay / 110.0f;
    m.s_out = ao / 110.0f;
    printf("H=%d ntok=%d\n", H, ntok);
    printf("absmax real: bcx=%.4g y=%.4g out=%.4g\n", ab, ay, ao);
    printf("scales: s_bc=%.6g s_y=%.6g s_out=%.6g  (int8 = real/scale)\n",
           m.s_bc, m.s_y, m.s_out);
    {
        /* the dequant multiplier per row must fit the fixed-point field:
           m = s_w[row]*s_a/s_out <= 1 (Q15 allows 2) */
        int r;
        double mx = 0, mn = 1e30;
        for (r = 0; r < H; r++) {
            double v = (double)m.sw_out[r] * (double)m.s_y / (double)m.s_out;
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        printf("out_proj multiplier m: min=%.3f max=%.3f  (must be <=1 for Q16, <=2 for Q15)\n", mn, mx);
        mx = 0; mn = 1e30;
        for (r = 0; r < H; r++) {
            double v = (double)m.sw_in[r] * (double)m.s_in / (double)m.s_bc;
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        printf("bcx multiplier       : min=%.3f max=%.3f\n", mn, mx);
    }

    if (ref_convw_init(&W, m.cw, H, m.s_bc, m.s_y) != 0) { printf("convw alloc failed\n"); return 1; }
    {
        int nz = 0;
        for (ch = 0; ch < H; ch++) if (W.kq[ch] == 0) nz++;
        printf("conv kq: ksh=%d  zero-multiplier channels=%d\n", W.ksh, nz);
    }

    bcx8  = (int8_t *)calloc((size_t)3 * H, 1);
    y8    = (int8_t *)calloc((size_t)H, 1);
    y8_dsp = (int8_t *)calloc((size_t)ntok * H, 1);
    out8a = (int8_t *)calloc((size_t)ntok * H, 1);
    out8b = (int8_t *)calloc((size_t)ntok * H, 1);
    hist6 = (int16_t *)calloc((size_t)6 * H, sizeof(int16_t));
    if (!bcx8 || !y8 || !y8_dsp || !out8a || !out8b || !hist6) return 1;

    /* ---- pass 2a: shift-register pipeline ---- */
    ref_i8_run(&m, &W, hist6, bcx8, y8, out8a);

    /* ---- pass 2b: DSP ring/phase pipeline ---- */
    if (ref_ring_init(&ring, H) != 0) return 1;
    for (t = 0; t < ntok; t++)
        ref_ring_token(&m, &W, &ring, m.tok + (size_t)t * H, bcx8,
                       y8_dsp + (size_t)t * H, out8b + (size_t)t * H);

    /* ---- compare the two integer models (must be identical) ---- */
    for (i = 0; i < ntok * H; i++) if (out8a[i] != out8b[i]) bad++;
    printf("\nring/phase vs shift register: %d mismatches out of %d\n", bad, ntok * H);

    /* ---- accuracy vs f32, per token ---- */
    printf("\n%-5s %-10s %-12s %-10s %-12s\n", "tok", "cos(out)", "maxerr(out)",
           "cos(y)", "maxerr(y)");
    for (t = 0; t < ntok; t++) {
        double c = ref_cos_i8(out8b + (size_t)t * H, out32 + (size_t)t * H, H, m.s_out);
        double e = ref_maxabs_i8(out8b + (size_t)t * H, out32 + (size_t)t * H, H, m.s_out);
        double cy = ref_cos_i8(y8_dsp + (size_t)t * H, st.y_all + (size_t)t * H, H, m.s_y);
        double ey = ref_maxabs_i8(y8_dsp + (size_t)t * H, st.y_all + (size_t)t * H, H, m.s_y);
        printf("%-5d %-10.6f %-12.5f %-10.6f %-12.5f\n", t, c, e, cy, ey);
        if (c < worst_cos) { worst_cos = c; worst_tok = t; }
        if (e > worst_err) worst_err = e;
        if (cy < worst_cos_y) worst_cos_y = cy;
    }
    printf("\nworst: cos(out)=%.6f (tok %d) maxerr(out)=%.5f  cos(y)=%.6f\n",
           worst_cos, worst_tok, worst_err, worst_cos_y);
    printf("int8 saturation: bcx=%.2f%% y=%.2f%% out=%.2f%%\n",
           100.0 * count_clip(bcx8, 3 * H) / (3.0 * H),
           100.0 * count_clip(y8_dsp, ntok * H) / (double)(ntok * H),
           100.0 * count_clip(out8b, ntok * H) / (double)(ntok * H));

    printf("\n%s\n", (bad == 0 && worst_cos > 0.999) ? "SIM OK" : "SIM FAIL");

    ref_model_free(&m);
    ref_convw_free(&W);
    ref_ring_free(&ring);
    free(st.hist); free(st.act); free(st.bcx); free(st.y);
    free(st.bcx_all); free(st.y_all); free(out32);
    free(bcx8); free(y8); free(y8_dsp); free(out8a); free(out8b); free(hist6);
    return (bad == 0 && worst_cos > 0.999) ? 0 : 2;
}
