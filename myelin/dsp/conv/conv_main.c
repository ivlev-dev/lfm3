/*
 * Host driver for the shortconv stand (FastRPC, unsigned PD).
 *
 * One RPC per layer: in_proj int4 GEMV -> depthwise k=7 int8 conv with the
 * 8-slot ring state -> out_proj int4 GEMV.  The CPU reference is conv_ref.h
 * (the same file conv_sim.c uses offline): the f32 model is the truth, the
 * ring model is the bit-exact integer model of the DSP.
 *
 * Checks per token and over all tokens
 *   - device output vs the integer ring model : must be bit-exact
 *   - cos / max_abs_err of the device output vs the f32 truth (scale s_out)
 *   - int8 saturation of the output
 *   - per-phase DSP time (in_proj / conv / out_proj / bx / dot / glue)
 *
 * Exit code: 0 pass, 1 allocation/calibration failure, 2 threshold failure,
 * 10.. RPC failure (matches the other stands).
 *
 * Build: see build.sh (needs HEXAGON_SDK_ROOT and ANDROID_NDK_ROOT).
 * Run on device: ./conv_test [H] [ntok] [reps] [rows] [threads] [pf_kb] [usepf]
 *   H       = channels (default 2048, multiple of 16)
 *   ntok    = tokens per pass (default 12)
 *   reps    = full passes over the token sequence (default 3; pass 1 is checked)
 *   rows    = GEMV rows per iteration, 1|2 (default 2)
 *   threads = HVX contexts used by the GEMV worker team (default 2)
 *   pf_kb   = l2fetch window in KB (default 32)
 *   usepf   = 0 disables l2fetch (default 1)
 */
#include "lfmgemv.h"
#include "conv_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define CAL_TARGET    110.0
#define PASS_COS      0.99
#define PASS_SAT_PCT  5.0

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

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

/* upload a resident int4 matrix and its per-row Q16 scales */
static int upload_matrix(int *h, const uint8_t *w, int nbytes,
                         int N, int K, float sa, float so, const float *sw)
{
    unsigned char *sbuf;
    size_t off = 0, chunk = 256 * 1024;
    int rc;

    if ((rc = lfmgemv_alloc_buf(nbytes, h)) != 0) return rc;
    if ((rc = lfmgemv_set_shape(*h, N, K, 1)) != 0) return rc;
    while (off < (size_t)nbytes) {
        size_t n = (size_t)nbytes - off < chunk ? (size_t)nbytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, w + off, (int)n)) != 0) return rc;
        off += n;
    }
    sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
    if (!sbuf) return AEE_ENOMEMORY;
    memcpy(sbuf + 0, &N, 4);
    memcpy(sbuf + 4, &K, 4);
    memcpy(sbuf + 8, &sa, 4);
    memcpy(sbuf + 12, &so, 4);
    memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(*h, sbuf, 16 + N * (int)sizeof(float));
    free(sbuf);
    return rc;
}

int main(int argc, char **argv)
{
    int H    = (argc > 1) ? atoi(argv[1]) : 2048;
    int ntok = (argc > 2) ? atoi(argv[2]) : 12;
    int reps = (argc > 3) ? atoi(argv[3]) : 3;
    int g_rows = (argc > 4) ? atoi(argv[4]) : 2;
    int g_thr  = (argc > 5) ? atoi(argv[5]) : 2;
    int g_pf   = (argc > 6) ? atoi(argv[6]) : 32;
    int g_usepf= (argc > 7) ? atoi(argv[7]) : 1;
    const int layer = 0, slot = 0;
    int hi = -1, ho = -1, rc, t, rep, ch;
    int times[16];
    int64_t t_in = 0, t_conv = 0, t_out = 0, t_bx = 0, t_dot = 0, t_glue = 0;
    int64_t t_in_deint = 0, t_in_kern = 0, t_in_pack = 0, t_out_deint = 0;
    int64_t t_out_kern = 0, t_out_pack = 0, t_fb = 0;
    double wall_sum = 0.0, wall_min = 1e18;
    int calls = 0;

    ref_model_t m;
    ref_convw_t W;
    ref_ring_t ring;
    ref_f32_t st;
    float *out32 = NULL, *abuf = NULL;
    int8_t *bcx8 = NULL, *y8 = NULL, *out8 = NULL;
    int32_t *acc_max = NULL;
    int8_t *dev_out = NULL, *ref_out = NULL;
    unsigned char *cfg = NULL;
    int32_t acc_gmax;
    double worst_cos = 2.0, worst_err = 0.0, mrmax = 0.0, worst_row = 0.0;
    int worst_tok = 0, mismatch = 0, clip_dev = 0, clip_y = 0;
    int pass;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (H < 16 || (H & 15)) { printf("H must be a multiple of 16\n"); return 1; }
    if (ntok < 1) ntok = 1;
    if (reps < 1) reps = 1;
    if (g_rows < 1) g_rows = 1;
    if (g_thr < 1) g_thr = 1;
    printf("conv stand: H=%d ntok=%d reps=%d tune: rows=%d threads=%d pf=%dKB usepf=%d\n",
           H, ntok, reps, g_rows, g_thr, g_pf, g_usepf ? 1 : 0);

    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }

    /* ---------------- model, f32 truth, scales ---------------- */
    if (ref_model_build(&m, H, ntok, 0x5eed) != 0) { printf("model alloc failed\n"); return 1; }
    memset(&st, 0, sizeof(st));
    st.hist    = (float *)calloc((size_t)6 * H, sizeof(float));
    st.act     = (float *)calloc((size_t)H, sizeof(float));
    st.bcx     = (float *)calloc((size_t)3 * H, sizeof(float));
    st.y       = (float *)calloc((size_t)H, sizeof(float));
    st.bcx_all = (float *)calloc((size_t)ntok * 3 * H, sizeof(float));
    st.y_all   = (float *)calloc((size_t)ntok * H, sizeof(float));
    out32      = (float *)calloc((size_t)ntok * H, sizeof(float));
    abuf       = (float *)calloc((size_t)8 * H, sizeof(float));
    if (!st.hist || !st.act || !st.bcx || !st.y || !st.bcx_all || !st.y_all ||
        !out32 || !abuf) { printf("host alloc failed\n"); return 1; }

    ref_f32_run(&m, &st, out32);
    m.s_in = 1.0f;
    m.s_bc = absmax32(st.bcx_all, ntok * 3 * H) / (float)CAL_TARGET;
    m.s_y  = absmax32(st.y_all,   ntok * H)     / (float)CAL_TARGET;
    printf("scales: s_bc=%.6g s_y=%.6g (target %.0f/127)\n",
           m.s_bc, m.s_y, CAL_TARGET);

    if (ref_convw_init(&W, m.cw, H, m.s_bc, m.s_y) != 0) { printf("convw alloc failed\n"); return 1; }

    bcx8    = (int8_t *)calloc((size_t)3 * H, 1);
    y8      = (int8_t *)calloc((size_t)ntok * H, 1);
    out8    = (int8_t *)calloc((size_t)ntok * H, 1);
    acc_max = (int32_t *)calloc((size_t)H, sizeof(int32_t));
    dev_out = (int8_t *)calloc((size_t)H, 1);
    ref_out = (int8_t *)calloc((size_t)H, 1);
    if (!bcx8 || !y8 || !out8 || !acc_max || !dev_out || !ref_out) {
        printf("host alloc failed\n"); return 1;
    }

    /* integer ring model: y8 per token, then s_out from the accumulator */
    if (ref_ring_init(&ring, H) != 0) { printf("ring alloc failed\n"); return 1; }
    for (t = 0; t < ntok; t++)
        ref_ring_token(&m, &W, &ring, m.tok + (size_t)t * H, bcx8,
                       y8 + (size_t)t * H, NULL);
    acc_gmax  = ref_out_acc_profile(&m, y8, ntok, acc_max);
    m.s_out   = ref_calib_s_out(&m, acc_max, (float)CAL_TARGET);
    worst_row = ref_s_out_worst(&m, acc_max, m.s_out, &mrmax);
    printf("acc_max=%d s_out=%.6g worst m_r*acc_max=%.1f (<=127)\n",
           acc_gmax, m.s_out, worst_row);
    if (worst_row > 127.0) { printf("out_proj scale invariant broken\n"); return 1; }

    /* ---------------- upload weights, init the layer ---------------- */
    rc = upload_matrix(&hi, m.w_in, 3 * H * (H / 2), 3 * H, H,
                       m.s_in, m.s_bc, m.sw_in);
    if (rc != 0) { printf("upload in_proj rc=%d\n", rc); return 11; }
    rc = upload_matrix(&ho, m.w_out, H * (H / 2), H, H,
                       m.s_y, m.s_out, m.sw_out);
    if (rc != 0) { printf("upload out_proj rc=%d\n", rc); return 11; }

    cfg = (unsigned char *)malloc(32 + (size_t)H * CONV_KSIZE * sizeof(float));
    if (!cfg) { printf("cfg alloc failed\n"); return 1; }
    {
        int nslots = 1;
        memcpy(cfg + 0, &H, 4);
        memcpy(cfg + 4, &nslots, 4);
        memcpy(cfg + 8, &m.s_bc, 4);
        memcpy(cfg + 12, &m.s_y, 4);
        memcpy(cfg + 16, m.cw, (size_t)H * CONV_KSIZE * sizeof(float));
        {
            int tune[4] = { g_pf, g_rows, g_usepf, g_thr };
            memcpy(cfg + 16 + (size_t)H * CONV_KSIZE * sizeof(float), tune, 16);
        }
        rc = lfmgemv_conv_init(layer, cfg,
                               32 + H * CONV_KSIZE * (int)sizeof(float));
        if (rc != 0) { printf("conv_init rc=%d\n", rc); return 12; }
    }
    printf("weights uploaded, conv layer %d initialised (ksh=%d)\n", layer, W.ksh);

    /* reference outputs for the ring model (s_out is final now) */
    for (t = 0; t < ntok; t++)
        ref_out8(&m, y8 + (size_t)t * H, out8 + (size_t)t * H);

    /* ---------------- run: correctness on pass 0, timing on all ---------------- */
    printf("\n%-5s %-10s %-12s %-10s\n", "tok", "cos(out)", "maxerr(out)", "mismatch");
    for (rep = 0; rep < reps; rep++) {
        rc = lfmgemv_conv_reset(layer, slot);
        if (rc != 0) { printf("conv_reset rc=%d\n", rc); return 13; }
        for (t = 0; t < ntok; t++) {
            double t0 = now_us(), wall;
            rc = lfmgemv_conv(layer, hi, ho, (const unsigned char *)(m.tok + (size_t)t * H),
                              H, 1, slot, 1, (unsigned char *)dev_out, H, times, 16);
            wall = now_us() - t0;
            if (rc != 0) { printf("conv tok=%d rc=%d\n", t, rc); return 14; }
            wall_sum += wall;
            if (wall < wall_min) wall_min = wall;
            t_in += times[1]; t_conv += times[2]; t_out += times[3];
            t_bx += times[4]; t_dot += times[5]; t_glue += times[6];
            t_in_deint += times[8];  t_in_kern += times[9];  t_in_pack += times[10];
            t_out_deint += times[11]; t_out_kern += times[12]; t_out_pack += times[13];
            t_fb = times[15];            /* jobs the team had to redo */
            calls++;

            if (rep == 0) {
                const int8_t *r = out8 + (size_t)t * H;
                double c = ref_cos_i8(dev_out, out32 + (size_t)t * H, H, m.s_out);
                double e = ref_maxabs_i8(dev_out, out32 + (size_t)t * H, H, m.s_out);
                int bad = 0;
                for (ch = 0; ch < H; ch++) if (dev_out[ch] != r[ch]) bad++;
                mismatch += bad;
                clip_dev += count_clip(dev_out, H);
                printf("%-5d %-10.6f %-12.5f %d\n", t, c, e, bad);
                if (c < worst_cos) { worst_cos = c; worst_tok = t; }
                if (e > worst_err) worst_err = e;
            }
        }
    }

    clip_y = count_clip(y8, ntok * H);
    printf("\nworst: cos(out)=%.6f (tok %d) maxerr(out)=%.5f\n",
           worst_cos, worst_tok, worst_err);
    printf("device vs ring model: %d mismatches out of %d\n", mismatch, ntok * H);
    printf("int8 saturation: y=%.2f%% out=%.2f%% (pass: <= %.1f%%)\n",
           100.0 * clip_y / (double)(ntok * H),
           100.0 * clip_dev / (double)(ntok * H), PASS_SAT_PCT);
    printf("time per layer (%d calls): %.0f us  [in_proj=%.0f conv=%.0f out_proj=%.0f"
           "  bx=%.0f dot=%.0f glue=%.0f us]\n",
           calls, (double)(t_in + t_conv + t_out) / calls,
           (double)t_in / calls, (double)t_conv / calls, (double)t_out / calls,
           (double)t_bx / calls, (double)t_dot / calls, (double)t_glue / calls);
    {
        double in_b = 3.0 * H * (H / 2.0), out_b = 1.0 * H * (H / 2.0);
        double in_mac = 3.0 * H * H, out_mac = 1.0 * H * H;
        printf("in_proj  split: deint=%.0f kern=%.0f pack=%.0f us  (%.2f GB/s, %.1f GMAC/s)\n",
               (double)t_in_deint / calls, (double)t_in_kern / calls,
               (double)t_in_pack / calls,
               in_b / ((double)t_in_kern / calls * 1e3),
               in_mac / ((double)t_in_kern / calls * 1e3));
        printf("out_proj split: deint=%.0f kern=%.0f pack=%.0f us  (%.2f GB/s, %.1f GMAC/s)"
               "  team_fallbacks=%d\n",
               (double)t_out_deint / calls, (double)t_out_kern / calls,
               (double)t_out_pack / calls,
               out_b / ((double)t_out_kern / calls * 1e3),
               out_mac / ((double)t_out_kern / calls * 1e3),
               (int)t_fb);
    }
    printf("host wall per layer: min=%.1f avg=%.1f us\n",
           wall_min, wall_sum / (calls ? calls : 1));

    pass = (mismatch == 0)
        && worst_cos >= PASS_COS
        && 100.0 * clip_dev / (double)(ntok * H) <= PASS_SAT_PCT;
    printf("\n%s (need: 0 mismatches vs ring model, worst cos(out) >= %.3f,"
           " out saturation <= %.1f%%)\n",
           pass ? "CONV OK" : "CONV FAIL", PASS_COS, PASS_SAT_PCT);

    lfmgemv_conv_free(layer);
    lfmgemv_release_buf(hi);
    lfmgemv_release_buf(ho);
    ref_model_free(&m);
    ref_convw_free(&W);
    ref_ring_free(&ring);
    free(st.hist); free(st.act); free(st.bcx); free(st.y);
    free(st.bcx_all); free(st.y_all); free(out32); free(abuf);
    free(bcx8); free(y8); free(out8); free(acc_max);
    free(dev_out); free(ref_out); free(cfg);
    return pass ? 0 : 2;
}
