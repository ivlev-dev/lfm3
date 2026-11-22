/*
 * Host driver for the FFN stand (FastRPC, unsigned PD): gate/up int4 GEMV ->
 * SiLU(gate)*up -> int8 mid -> down int4 GEMV, one RPC per layer.
 *
 * The numerics are shared with ffn_sim.c through ffn_ref.h: the host model is
 * the DSP pipeline expression for expression (the wide Q16.16 dequantiser,
 * silu_qk, requant_mid, pack_out_i8), and the truth is the same dequantised
 * int4 weights in f32.  Checks per token:
 *   - device mid/out vs the integer model       : must be bit-exact
 *   - cos / max_abs_err of mid and out vs f32   : numerics, not bit-exactness
 *   - int8 saturation of mid and out
 *   - dequant: m = sw*s_in*2^16 is reported with its widening shift k; the
 *     old Q16 clamp is gone, so m > 1 is expected on int4 tensors
 *
 * Scale protocol (see ffn_ref.h): gate/up get sa = s_in*65536, so = 1;
 * down gets sa = s_mid, so = s_out.  s_mid and s_out are calibrated on the
 * host from the measured P = silu(gate)*up range and the down accumulator
 * range.
 *
 * Exit code: 0 pass, 1 allocation/calibration failure, 2 threshold failure,
 * 10.. RPC failure (matches the other stands).
 *
 * Build: see build.sh (needs HEXAGON_SDK_ROOT and ANDROID_NDK_ROOT).
 * Run on device: ./ffn_test [rounds] [ntok]
 */
#include "lfmgemv.h"
#include "ffn_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "rpcmem.h"

#define NH 7680     /* hidden */
#define K  2048     /* model width */

/* same gate as ffn_sim: cos 0.99 is the int4 weights + int8 mid ceiling */
#define PASS_COS      0.99
#define PASS_SAT_PCT  5.0
#define ZG_MIN_Q16    1024.0f

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

/* upload a resident int4 matrix and its per-row Q16 scales */
static int upload_matrix(int *h, const uint8_t *w, int nbytes,
                         int N, int Kdim, float sa, float so, const float *sw)
{
    unsigned char *sbuf;
    size_t off = 0, chunk = 256 * 1024;
    int rc;

    if ((rc = lfmgemv_alloc_buf(nbytes, h)) != 0) return rc;
    if ((rc = lfmgemv_set_shape(*h, N, Kdim, 1)) != 0) return rc;
    while (off < (size_t)nbytes) {
        size_t n = (size_t)nbytes - off < chunk ? (size_t)nbytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, w + off, (int)n)) != 0) return rc;
        off += n;
    }
    sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
    if (!sbuf) return AEE_ENOMEMORY;
    memcpy(sbuf + 0, &N, 4);
    memcpy(sbuf + 4, &Kdim, 4);
    memcpy(sbuf + 8, &sa, 4);
    memcpy(sbuf + 12, &so, 4);
    memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(*h, sbuf, 16 + N * (int)sizeof(float));
    free(sbuf);
    return rc;
}

int main(int argc, char **argv)
{
    int hg = -1, hu = -1, hd = -1, rc, i, r, round, t;
    int times[8], info[8];
    size_t wb = (size_t)NH * (K / 2);
    int rounds = (argc > 1) ? atoi(argv[1]) : 3;
    int ntok = (argc > 2) ? atoi(argv[2]) : 4;
    ffn_model_t m;
    ffn_calib_t ci;
    float *yref = NULL, *ppipe = NULL;
    int8_t *mid_ref = NULL, *out_ref = NULL;
    int8_t *mid_out, *dst_out;
    int32_t *acc;
    int pass = 0, bad_mid = 0, bad_out = 0;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (ntok < 1) ntok = 1;

    if (ffn_model_synth(&m, K, NH, ntok, 0x5eedu) != 0) {
        printf("model alloc failed\n");
        return 1;
    }
    if (ffn_calibrate(&m, &ci) != 0) {
        printf("calibration failed (no signal in x or P)\n");
        return 1;
    }
    printf("FFN stand: K=%d NH=%d ntok=%d\n", K, NH, ntok);
    printf("scales: s_in=%.6g  s_mid=%.6g (P absmax/127)  mq_mid=%d  s_out=%.6g\n",
           ci.s_in, ci.s_mid, ci.mq_mid, ci.s_out);
    printf("real ranges: |gate|=%.4f |up|=%.4f |P|=%.4f   |zg| max=%.0f (Q16 units)\n",
           ci.gate_absmax, ci.up_absmax, ci.p_absmax, ci.zg_absmax);
    printf("dequant: m_gate=[%.4f..%.4f] m_down=[%.4f..%.4f] k_gate=%d k_up=%d"
           " (old Q16 would clamp g=%d u=%d, down=%d)\n",
           ci.m_g_min, ci.m_g_max, ci.m_d_min, ci.m_d_max,
           ci.kg, ci.ku, ci.g_clamped, ci.u_clamped, ci.d_clamped);
    printf("down: acc_max=%d worst m_d*acc_max=%.1f\n",
           ci.acc_dmax, ci.worst_row);
    if (ci.zg_absmax < ZG_MIN_Q16 || ci.d_clamped) {
        printf("stand is not usable: SiLU input is not Q16 or the down multiplier clamps\n");
        return 1;
    }

    yref   = (float *)malloc((size_t)K * sizeof(float));
    ppipe  = (float *)malloc((size_t)NH * sizeof(float));
    mid_ref = (int8_t *)malloc((size_t)NH);
    out_ref = (int8_t *)malloc((size_t)K);
    mid_out = (int8_t *)malloc((size_t)NH + 128);
    dst_out = (int8_t *)malloc((size_t)K + 128);
    acc     = (int32_t *)malloc((size_t)4 * NH * sizeof(int32_t));
    if (!yref || !ppipe || !mid_ref || !out_ref || !mid_out || !dst_out || !acc) {
        printf("host alloc failed\n");
        return 1;
    }

    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }

    /* ---- allocate + upload ---- */
    rc = upload_matrix(&hg, m.wg, (int)wb, NH, K, m.s_in * FFN_Q16_ONE, 1.0f, m.swg);
    if (rc != 0) { printf("upload gate rc=%d\n", rc); return 11; }
    rc = upload_matrix(&hu, m.wu, (int)wb, NH, K, m.s_in * FFN_Q16_ONE, 1.0f, m.swu);
    if (rc != 0) { printf("upload up rc=%d\n", rc); return 11; }
    rc = upload_matrix(&hd, m.wd, (int)wb, K, NH, m.s_mid, m.s_out, m.swd);
    if (rc != 0) { printf("upload down rc=%d\n", rc); return 11; }
    printf("handles g=%d u=%d d=%d (%.1f MB total int4)\n", hg, hu, hd, 3.0 * wb / 1e6);

    /* ---- team info ---- */
    if (lfmgemv_team_info(info, 8) == 0)
        printf("team before start: state=%d hvx_rc=%d mode=%d created=%d threads=%d\n",
               info[0], info[1], info[2], info[3], info[4]);

    if (argc > 3 && strcmp(argv[3], "bw") == 0) {
        /* minimal bandwidth probe: N iterations of one GEMV, print min */
        double wmin = 1e18;
        int kk;
        lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16), times, 8);
        for (kk = 0; kk < 40; kk++) {
            double t0 = now_us();
            lfmgemv_gemv(hg, 1, (const unsigned char *)m.x8, K, 1,
                         (unsigned char *)dst_out, NH, times, 8);
            { double w = now_us() - t0; if (w < wmin) wmin = w; }
        }
        printf("BW pid=%d kernel_min=%d us  wall_min=%.0f us  %.2f GB/s\n",
               (int)getpid(), times[0], wmin, wb / (times[0] * 1e3));
        lfmgemv_release_buf(hg); lfmgemv_release_buf(hu); lfmgemv_release_buf(hd);
        return 0;
    }

    /* ---- 1. multi-context sweep on the gate matrix ---- */
    printf("\n%-6s %-12s %10s %9s %9s\n", "thr", "config", "us", "GB/s", "GMAC/s");
    {
        static const int TH[] = { 1, 2, 4, 8 };
        double best[4];
        for (t = 0; t < 4; t++) best[t] = 1e18;
        for (round = 0; round < rounds; round++) {
            for (i = 0; i < 4; i++) {
                t = (round & 1) ? (3 - i) : i;
                lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16) | (TH[t] << 24), times, 8);
                rc = lfmgemv_gemv(hg, 1, (const unsigned char *)m.x8, K, 3,
                                  (unsigned char *)dst_out, NH, times, 8);
                if (rc != 0) { printf("gemv thr=%d rc=%d\n", TH[t], rc); return 15; }
                if ((double)times[0] < best[t]) best[t] = times[0];
            }
            usleep(150000);
        }
        for (t = 0; t < 4; t++)
            printf("%-6d %-12s %10.0f %9.2f %9.1f\n", TH[t], "pf8KB r2",
                   best[t], wb / (best[t] * 1e3), (double)NH * K / (best[t] * 1e3));
        lfmgemv_team_info(info, 8);
        printf("team after start: state=%d hvx_rc=%d mode=%d created=%d thr_rc=%d hvx_units=%d\n",
               info[0], info[1], info[2], info[3], info[5], info[6]);
    }

    /* ---- 2. correctness of the three GEMMs (int32, bit exact) ---- */
    lfmgemv_probe(hg, 8, 8 | (2 << 8) | (1 << 16) | (2 << 24), times, 8);
    ffn_int_mid(&m, 0, mid_ref, NULL);   /* activation for the down GEMM check */
    {
        struct { int h; const uint8_t *w; const int8_t *act; int n; int k; const char *nm; } m3[] = {
            { hg, m.wg, m.x8,    NH, K,  "gate" },
            { hu, m.wu, m.x8,    NH, K,  "up  " },
            { hd, m.wd, mid_ref, K,  NH, "down" },
        };
        int mi;
        for (mi = 0; mi < 3; mi++) {
            int bad = 0, maxerr = 0;
            int32_t bias = -8 * ffn_sum_i8(m3[mi].act, m3[mi].k);
            lfmgemv_set_shape(m3[mi].h, m3[mi].n, m3[mi].k, 0);
            rc = lfmgemv_gemv(m3[mi].h, 1, (const unsigned char *)m3[mi].act,
                              m3[mi].k, 1, (unsigned char *)acc,
                              m3[mi].n * 4, times, 8);
            if (rc != 0) { printf("gemv %s rc=%d\n", m3[mi].nm, rc); return 16; }
            for (r = 0; r < m3[mi].n; r++) {
                int32_t ref = ffn_dot_i4(m3[mi].w + (size_t)r * (m3[mi].k / 2),
                                         m3[mi].act, m3[mi].k, bias);
                int32_t e = ref - acc[r];
                if (e < 0) e = -e;
                if (e > maxerr) maxerr = (int)e;
                if (e) bad++;
            }
            printf("correctness %s: rows=%d mismatches=%d max_abs_err=%d\n",
                   m3[mi].nm, m3[mi].n, bad, maxerr);
            lfmgemv_set_shape(m3[mi].h, m3[mi].n, m3[mi].k, 1);
        }
    }
    /* restore scales after set_shape */
    {
        unsigned char *sbuf = (unsigned char *)malloc(16 + (size_t)NH * 4);
        float sa = m.s_in * FFN_Q16_ONE, so = 1.0f;
        int vNH = NH, vK = K;
        if (!sbuf) { printf("alloc failed\n"); return 1; }
        memcpy(sbuf, &vNH, 4); memcpy(sbuf + 4, &vK, 4);
        memcpy(sbuf + 8, &sa, 4); memcpy(sbuf + 12, &so, 4);
        memcpy(sbuf + 16, m.swg, (size_t)NH * 4); lfmgemv_set_scale(hg, sbuf, 16 + NH * 4);
        memcpy(sbuf + 16, m.swu, (size_t)NH * 4); lfmgemv_set_scale(hu, sbuf, 16 + NH * 4);
        {
            int vNH2 = NH, vK2 = K;
            float sa2 = m.s_mid, so2 = m.s_out;
            memcpy(sbuf, &vK2, 4); memcpy(sbuf + 4, &vNH2, 4);
            memcpy(sbuf + 8, &sa2, 4); memcpy(sbuf + 12, &so2, 4);
            memcpy(sbuf + 16, m.swd, (size_t)K * 4);
            lfmgemv_set_scale(hd, sbuf, 16 + K * 4);
        }
        free(sbuf);
    }

    /* ---- 3. one-RPC FFN layer: numerics + timing ---- */
    printf("\n%-4s %-10s %-11s %-9s %-9s %-8s %-8s\n", "tok", "cos(out)", "maxerr(out)",
           "mism mid", "mism out", "cos(mid)", "LSB mid");
    {
        double bestffn = 1e18, wallmin = 1e18, wallsum = 0;
        double worst_cos = 2.0, worst_cos_mid = 2.0, worst_err = 0.0;
        int n = 8, kk, worst_tok = 0;

        /* timing on token 0 */
        for (kk = 0; kk < n; kk++) {
            double t0 = now_us();
            rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)m.x8, K, 1,
                             (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH,
                             times, 8);
            {
                double w = now_us() - t0;
                if (w < wallmin) wallmin = w;
                wallsum += w;
            }
            if (rc != 0) { printf("ffn rc=%d\n", rc); return 17; }
            if ((double)times[0] < bestffn) bestffn = times[0];
            usleep(20000);
        }

        for (t = 0; t < ntok; t++) {
            int badm = 0, bado = 0, mo = 0;
            double c, e, cm, lsb = 0.0;
            ffn_ref_token(&m, t, yref, NULL, 0);
            ffn_ref_token(&m, t, yref, ppipe, 1);
            ffn_int_mid(&m, t, mid_ref, NULL);
            ffn_int_out(&m, mid_ref, out_ref, NULL);

            rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)(m.x8 + (size_t)t * K), K, 1,
                             (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH,
                             times, 8);
            if (rc != 0) { printf("ffn tok=%d rc=%d\n", t, rc); return 17; }
            for (r = 0; r < NH; r++) {
                int d = (int)mid_out[r] - (int)mid_ref[r];
                if (d < 0) d = -d;
                if (d) badm++;
                if (d > mo) mo = d;
            }
            for (r = 0; r < K; r++) if (dst_out[r] != out_ref[r]) bado++;
            for (r = 0; r < NH; r++) {
                long ideal = lrintf(ppipe[r] / m.s_mid);
                double d;
                if (ideal > 127) ideal = 127;
                if (ideal < -127) ideal = -127;
                d = (double)mid_out[r] - (double)ideal;
                lsb += d * d;
            }
            bad_mid += badm;
            bad_out += bado;

            c  = ffn_cos_i8(dst_out, yref, K, m.s_out);
            e  = ffn_maxabs_i8(dst_out, yref, K, m.s_out);
            cm = ffn_cos_i8(mid_out, ppipe, NH, m.s_mid);
            printf("%-4d %-10.6f %-11.5g %-9d %-9d %-8.6f %-8.3f\n", t, c, e, badm,
                   bado, cm, sqrt(lsb / NH));
            if (c < worst_cos) { worst_cos = c; worst_tok = t; }
            if (cm < worst_cos_mid) worst_cos_mid = cm;
            if (e > worst_err) worst_err = e;
        }

        {
            long clip_mid = 0, clip_out = 0, cnt = 0;
            for (t = 0; t < ntok; t++) {
                ffn_int_mid(&m, t, mid_ref, NULL);
                clip_mid += ffn_count_clip(mid_ref, NH);
                ffn_int_out(&m, mid_ref, out_ref, NULL);
                clip_out += ffn_count_clip(out_ref, K);
                cnt += NH;
            }
            printf("\nworst: cos(out)=%.6f (tok %d) cos(mid)=%.6f maxerr(out)=%.5g\n",
                   worst_cos, worst_tok, worst_cos_mid, worst_err);
            printf("device vs integer model: mid %d, out %d mismatches out of %d\n",
                   bad_mid, bad_out, ntok * K);
            printf("int8 saturation (model): mid=%.2f%% out=%.2f%% (pass: <= %.1f%%)\n",
                   100.0 * clip_mid / (double)cnt, 100.0 * clip_out / (double)cnt, PASS_SAT_PCT);
            pass = (bad_mid == 0 && bad_out == 0)
                && worst_cos >= PASS_COS
                && 100.0 * clip_mid / (double)cnt <= PASS_SAT_PCT
                && 100.0 * clip_out / (double)cnt <= PASS_SAT_PCT;
        }
        printf("FFN layer (1 RPC): %.2f ms  [gate=%.2f up=%.2f silu=%.2f down=%.2f pack=%.2f ms]\n",
               wallmin / 1000.0, times[1] / 1e3, times[2] / 1e3, times[3] / 1e3,
               times[4] / 1e3, times[5] / 1e3);
        printf("FFN host wall: min=%.0f us avg=%.0f us over %d calls\n", wallmin, wallsum / n, n);
        printf("QNN reference for the same layer: 3 x 1.37 = 4.11 ms\n");
        (void)bestffn;
        /* amortisation: several layers' worth of work inside one RPC */
        {
            int rp[] = { 1, 2, 4, 8 };
            int z;
            for (z = 0; z < 4; z++) {
                double bb = 0;
                for (kk = 0; kk < 4; kk++) {
                    double t0 = now_us();
                    rc = lfmgemv_ffn(hg, hu, hd, (const unsigned char *)m.x8, K, rp[z],
                                     (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH,
                                     times, 8);
                    { double w = now_us() - t0; if (kk == 0 || w < bb) bb = w; }
                    usleep(15000);
                }
                if (rc != 0) break;
                printf("FFN reps=%d: wall=%7.0f us  per-layer amortised=%6.0f us  (dsp/rep=%d us)\n",
                       rp[z], bb, bb / rp[z], times[0]);
            }
        }
    }

    /* ---- 3b. resident output buffer (apps_mem_share_map, mapped once) ---- */
    {
        void *pdst = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        int rc2;
        if (!pdst) { printf("resident: rpcmem_alloc failed\n"); }
        else {
            int fd = rpcmem_to_fd(pdst);
            int8_t ref_out[K];
            double wmin = 1e18, wsum = 0;
            int kk, bad = 0, maxd = 0;

            rc2 = lfmgemv_attach_out(hg, fd, NH);
            rc2 = lfmgemv_attach_out(hd, fd, NH);   /* hd: K=2048 out fits */
            printf("resident: attach fd=%d rc=%d\n", fd, rc2);

            /* reference result via the ordinary rout path */
            lfmgemv_ffn(hg, hu, hd, (const unsigned char *)m.x8, K, 1,
                        (unsigned char *)dst_out, K, (unsigned char *)mid_out, NH, times, 8);
            memcpy(ref_out, dst_out, K);

            for (kk = 0; kk < 12; kk++) {
                double t0 = now_us();
                rc2 = lfmgemv_ffn_resident(hg, hu, hd, (const unsigned char *)m.x8, K, 1,
                                           0, 0, times, 8);
                { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w; }
                if (rc2 != 0) { printf("ffn_resident rc=%d\n", rc2); break; }
                usleep(20000);
            }
            for (r = 0; r < K; r++) {
                int dd = (int)((int8_t *)pdst)[r] - (int)ref_out[r];
                if (dd < 0) dd = -dd;
                if (dd > maxd) maxd = dd;
                if (dd) bad++;
            }
            printf("FFN resident out: wall min=%.0f us avg=%.0f us | mismatches=%d max|d|=%d\n",
                   wmin, wsum / 12, bad, maxd);
            lfmgemv_detach_out(hg); lfmgemv_detach_out(hd);
            rpcmem_free(pdst);
        }
    }

    /* ---- 4. RPC cost: plain heap buffers vs persistent rpcmem ---- */
    {
        double wmin = 1e18, wsum = 0, kmin = 1e18;
        int kk;
        for (kk = 0; kk < 30; kk++) {
            double t0 = now_us();
            lfmgemv_gemv(hg, 1, (const unsigned char *)m.x8, K, 1,
                         (unsigned char *)dst_out, NH, times, 8);
            { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w;
              if (times[0] < kmin) kmin = times[0]; }
        }
        printf("\nRPC plain heap:   min=%.0f us avg=%.0f us  (kernel min=%.0f us)\n",
               wmin, wsum / 30, kmin);
    }
    {
        void *pd2 = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        if (pd2) {
            double wmin = 1e18, kmin = 1e18;
            int kk;
            lfmgemv_attach_out(hg, rpcmem_to_fd(pd2), NH);
            for (kk = 0; kk < 30; kk++) {
                double t0 = now_us();
                if (lfmgemv_gemv_resident(hg, 1, (const unsigned char *)m.x8, K, 1, times, 8) == 0) {
                    double w = now_us() - t0;
                    if (w < wmin) wmin = w;
                    if (times[0] < kmin) kmin = times[0];
                }
            }
            printf("gemv_resident (no output transfer): min=%.0f us (kernel min=%.0f us)\n",
                   wmin, kmin);
            lfmgemv_detach_out(hg);
            rpcmem_free(pd2);
        }
    }
    {
        void *pa = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, K + 4096);
        void *pd = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, NH + 4096);
        if (pa && pd) {
            double wmin = 1e18, wsum = 0, kmin = 1e18;
            int kk;
            memcpy(pa, m.x8, K);
            for (kk = 0; kk < 30; kk++) {
                double t0 = now_us();
                lfmgemv_gemv(hg, 1, (const unsigned char *)pa, K, 1,
                             (unsigned char *)pd, NH, times, 8);
                { double w = now_us() - t0; if (w < wmin) wmin = w; wsum += w;
                  if (times[0] < kmin) kmin = times[0]; }
            }
            printf("RPC rpcmem ION:   min=%.0f us avg=%.0f us  (kernel min=%.0f us), fd=%d\n",
                   wmin, wsum / 30, kmin, rpcmem_to_fd(pa));
            rpcmem_free(pa); rpcmem_free(pd);
        } else printf("rpcmem alloc failed\n");
    }

    printf("\n%s (need: 0 mismatches vs the integer model, worst cos(out) >= %.3f,"
           " saturation <= %.1f%%)\n", pass ? "FFN OK" : "FFN FAIL", PASS_COS, PASS_SAT_PCT);

    lfmgemv_release_buf(hg); lfmgemv_release_buf(hu); lfmgemv_release_buf(hd);
    ffn_model_free(&m);
    free(yref); free(ppipe); free(mid_ref); free(out_ref);
    free(mid_out); free(dst_out); free(acc);
    return pass ? 0 : 2;
}
