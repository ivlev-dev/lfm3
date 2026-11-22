/*
 * LFM3 shortconv layer on Hexagon HVX (v68) -- one FastRPC call per layer.
 *
 *   bcx  = in_proj(h)                     int4 [3H][H], M = 1|2|4 tokens
 *   b,c,x = bcx[0:H], bcx[H:2H], bcx[2H:3H]        (int8, one scale s_bc)
 *   bx   = b .* x                         exact int16, 7 taps of history
 *   conv = sum_{j=0..6} wq[ch][j] * hist[j][ch]     depthwise, ksize = 7
 *   y    = c .* conv                      int8, one scale s_y
 *   out  = out_proj(y)                    int4 [H][H]
 *
 * Shape note: the depthwise conv runs over H channels (2048 for LFM3); the
 * 3H = 6144 figure is only the in_proj output width.
 *
 * History / ring
 *   Per slot the state is 8 ring positions per channel (7 taps + 1 pad).
 *   State layout is channel-major: byte[ch*8 + s], 16 channels = one 128 B
 *   HVX vector, 8 bytes per channel.  bx16 is exact, stored as two byte
 *   planes so that a single vrmpy (4 bytes/lane) covers 4 ring positions:
 *     hi plane: signed   (int8)(bx16>>8)   -> vrmpyacc Vb x Vb
 *     lo plane: unsigned (uint8)(bx16&ff)  -> vrmpyacc Vub x Vb
 *     out32[ch] = 256*(hi_dot) + lo_dot
 *   No requantisation of the state, no clipping: the product b*x is never
 *   rounded (|b*x| <= 16129, so hi fits int8).
 *
 *   The ring position of the newest sample is ctr&7, so the tap->slot map
 *   rotates with the token counter.  Instead of shuffling 14 KB of state per
 *   token, eight phase-rotated copies of the int8 conv weights are built once
 *   in conv_init (8 * H * 8 B = 128 KB for H=2048); the phase is just a
 *   pointer offset.
 */

#include "lfmgemv.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include <hexagon_types.h>
#include "hvx_hexagon_protos.h"
#include "hexagon_protos.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "lfmconv_bridge.h"
#include "conv_math.h"

#define CONV_TAPS   7
#define CONV_RING   8            /* ring positions: 7 taps + 1 unused */
#define CONV_SLOTS  8            /* history slots (sequences) */
#define CONV_LAYERS 16
#define CONV_GRP    16           /* channels per 128 B vector */
#define CONV_M_MAX  4

typedef struct {
    int      ready;
    int      H, ng;              /* ng = H/16 vrmpy groups */
    int      nslots;
    int8_t  *wq8;                /* [CONV_RING][ng][128] rotated int8 weights */
    int8_t  *wq1;                /* [H][7] unrotated, kept for reference */
    int8_t  *sthi;               /* [slots][H*8] signed   high-byte plane */
    int8_t  *stlo;               /* [slots][H*8] unsigned low-byte plane  */
    int32_t *kq;                 /* [H] out32*c -> y8 multiplier, Q(ksh)   */
    int      ksh;
    unsigned ctr[CONV_SLOTS];
} conv_layer_t;

static conv_layer_t g_cl[CONV_LAYERS];

/* per-call scratch, shared by all layers (one RPC at a time) */
static int8_t  *g_bcx;           /* [CONV_M_MAX * 3H] */
static int8_t  *g_y;             /* [CONV_M_MAX * H]  */
static int32_t *g_res;           /* [2H]              */
static int      g_scratch_h;

/* ------------------------------------------------------------------ */
/* kernels                                                            */
/* ------------------------------------------------------------------ */

/* bx16 = b8*x8 written straight into the ring position `p` of both planes.
 * Scalar on purpose: 2048 iterations, ~7 us, ~1% of the layer.            */
static void conv_bx_write(int8_t *hi, int8_t *lo, int p,
                          const int8_t *b, const int8_t *x, int H)
{
    int ch;
    for (ch = 0; ch < H; ch++) {
        int32_t t = (int32_t)b[ch] * (int32_t)x[ch];
        hi[(size_t)ch * 8 + p] = (int8_t)(t >> 8);
        lo[(size_t)ch * 8 + p] = (int8_t)(t & 0xff);
    }
}

/* out32 for 16 channels per iteration.
 * Lane 2c of the vrmpy result sums ring positions 0..3 of channel c, lane
 * 2c+1 sums positions 4..7; conv_glue() adds the two lanes.               */
static void conv_dot(const int8_t *hi, const int8_t *lo, const int8_t *wq,
                     int32_t *res, int ng)
{
    const HVX_Vector *W  = (const HVX_Vector *)wq;
    const HVX_Vector *PH = (const HVX_Vector *)hi;
    const HVX_Vector *PL = (const HVX_Vector *)lo;
    HVX_Vector z = Q6_V_vzero();
    int g;

    for (g = 0; g < ng; g++) {
        HVX_Vector w  = W[g];
        HVX_Vector ah = Q6_Vw_vrmpyacc_VwVbVb(z, PH[g], w);
        HVX_Vector al = Q6_Vw_vrmpyacc_VwVubVb(z, PL[g], w);
        HVX_Vector r  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(ah, 8), al);
        *(HVX_Vector *)(res + (size_t)g * 32) = r;
    }
}

/* out32 -> y8, scalar; shares conv_glue() with the host reference. */
static void conv_pack_y(const int32_t *res, const int8_t *c, const int32_t *kq,
                        int ksh, int8_t *y, int H)
{
    int ch;
    for (ch = 0; ch < H; ch++)
        y[ch] = conv_glue(res[2 * ch] + res[2 * ch + 1], c[ch], kq[ch], ksh);
}

/* ------------------------------------------------------------------ */
/* setup                                                              */
/* ------------------------------------------------------------------ */

static int conv_scratch(int H)
{
    if (g_scratch_h >= H && g_bcx && g_y && g_res) return 0;
    free(g_bcx); free(g_y); free(g_res);
    g_bcx = (int8_t *)memalign(128, (size_t)CONV_M_MAX * 3 * H);
    g_y   = (int8_t *)memalign(128, (size_t)CONV_M_MAX * H);
    g_res = (int32_t *)memalign(128, (size_t)2 * H * sizeof(int32_t));
    if (!g_bcx || !g_y || !g_res) { g_scratch_h = 0; return AEE_ENOMEMORY; }
    g_scratch_h = H;
    return 0;
}

int lfmgemv_conv_free(int layer)
{
    conv_layer_t *L;
    if (layer < 0 || layer >= CONV_LAYERS) return AEE_EBADPARM;
    L = &g_cl[layer];
    free(L->wq8); free(L->wq1); free(L->sthi); free(L->stlo); free(L->kq);
    memset(L, 0, sizeof(*L));
    return 0;
}

int lfmgemv_conv_init(int layer, const unsigned char *cfg, int cfgLen)
{
    conv_layer_t *L;
    const float *wf;
    int H, nslots, ch, j, s, p, g, rc;
    float s_bc, s_y;
    double kmax = 0.0;

    if (layer < 0 || layer >= CONV_LAYERS) return AEE_EBADPARM;
    if (!cfg || cfgLen < 16) return AEE_EBADPARM;
    memcpy(&H, cfg + 0, 4);
    memcpy(&nslots, cfg + 4, 4);
    memcpy(&s_bc, cfg + 8, 4);
    memcpy(&s_y, cfg + 12, 4);
    if (H < CONV_GRP || (H & (CONV_GRP - 1)) || H > 16384) return AEE_EBADPARM;
    if (nslots < 1 || nslots > CONV_SLOTS) return AEE_EBADPARM;
    if (!(s_bc > 0.0f) || !(s_y > 0.0f)) return AEE_EBADPARM;
    if (cfgLen < 16 + H * CONV_TAPS * (int)sizeof(float)) return AEE_EBADPARM;

    if ((rc = conv_scratch(H)) != 0) return rc;
    lfm_bridge_tune(8, 1, 1, -1);        /* l2fetch 8 KB, prefetch on */

    L = &g_cl[layer];
    if (L->ready && L->H != H) return AEE_EBADPARM;   /* conv_free() first */

    if (!L->ready) {
        L->H  = H;
        L->ng = H / CONV_GRP;
        L->nslots = nslots;
        L->wq8  = (int8_t *)memalign(128, (size_t)CONV_RING * H * 8);
        L->wq1  = (int8_t *)memalign(128, (size_t)H * CONV_TAPS);
        L->sthi = (int8_t *)memalign(128, (size_t)nslots * H * 8);
        L->stlo = (int8_t *)memalign(128, (size_t)nslots * H * 8);
        L->kq   = (int32_t *)memalign(128, (size_t)H * sizeof(int32_t));
        if (!L->wq8 || !L->wq1 || !L->sthi || !L->stlo || !L->kq) {
            lfmgemv_conv_free(layer);
            return AEE_ENOMEMORY;
        }
        memset(L->sthi, 0, (size_t)nslots * H * 8);
        memset(L->stlo, 0, (size_t)nslots * H * 8);
        memset(L->ctr, 0, sizeof(L->ctr));
    }

    /* 1. quantise the 7 depthwise taps, per-channel max-abs scale */
    wf = (const float *)(cfg + 16);
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * CONV_TAPS;
        float m = 0.0f, sw;
        for (j = 0; j < CONV_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        sw = m / 127.0f;
        for (j = 0; j < CONV_TAPS; j++)
            L->wq1[(size_t)ch * CONV_TAPS + j] = (int8_t)conv_wq(w[j], sw);
        {
            double K = conv_K((double)m, (double)s_bc, (double)s_y);
            if (K > kmax) kmax = K;
        }
    }

    /* 2. per-channel dequant multiplier, one shared shift */
    L->ksh = conv_pick_ksh(kmax);
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * CONV_TAPS;
        float m = 0.0f;
        for (j = 0; j < CONV_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        L->kq[ch] = conv_make_kq(conv_K((double)m, (double)s_bc, (double)s_y), L->ksh);
    }

    /* 3. eight phase rotations of the tap weights:
     *    wq8[p][ch][s] = wq[ch][ (s-p-2)&7 ]   (tap 7 -> 0)               */
    for (p = 0; p < CONV_RING; p++) {
        for (g = 0; g < L->ng; g++) {
            for (j = 0; j < CONV_GRP; j++) {
                int ch = g * CONV_GRP + j;
                int8_t *dst = L->wq8 + ((size_t)p * L->ng + g) * 128 + (size_t)j * 8;
                for (s = 0; s < CONV_RING; s++) {
                    int tap = conv_slot_tap(p, s);
                    dst[s] = (tap < CONV_TAPS) ? L->wq1[(size_t)ch * CONV_TAPS + tap] : 0;
                }
            }
        }
    }

    L->ready = 1;
    return 0;
}

int lfmgemv_conv_reset(int layer, int slot)
{
    conv_layer_t *L;
    if (layer < 0 || layer >= CONV_LAYERS) return AEE_EBADPARM;
    L = &g_cl[layer];
    if (!L->ready) return AEE_EBADPARM;
    if (slot < 0 || slot >= L->nslots) return AEE_EBADPARM;
    memset(L->sthi + (size_t)slot * L->H * 8, 0, (size_t)L->H * 8);
    memset(L->stlo + (size_t)slot * L->H * 8, 0, (size_t)L->H * 8);
    L->ctr[slot] = 0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* the layer                                                          */
/* ------------------------------------------------------------------ */

int lfmgemv_conv(int layer, int hi, int ho, const unsigned char *act, int actLen,
                 int M, int slot, int reps, unsigned char *outp, int outLen,
                 int *times, int timesLen)
{
    conv_layer_t *L;
    int n, k, i8, hs, rc;
    int t_in = 0, t_conv = 0, t_out = 0, t_bx = 0, t_dot = 0, t_glue = 0;
    int r, m;

    if (layer < 0 || layer >= CONV_LAYERS) return AEE_EBADPARM;
    L = &g_cl[layer];
    if (!L->ready) return AEE_EBADPARM;
    if (M != 1 && M != 2 && M != 4) return AEE_EBADPARM;
    if (slot < 0 || slot >= L->nslots) return AEE_EBADPARM;
    if (!act || actLen != M * L->H) return AEE_EBADPARM;
    if (!outp || outLen != M * L->H) return AEE_EBADPARM;
    if (reps < 1) reps = 1;

    /* in_proj: [3H][H], out_proj: [H][H], both int4 with a Q16 scale table */
    if (lfm_bridge_shape(hi, &n, &k, &i8, &hs) != 0) return AEE_EBADPARM;
    if (n != 3 * L->H || k != L->H || !hs) return AEE_EBADPARM;
    if (lfm_bridge_shape(ho, &n, &k, &i8, &hs) != 0) return AEE_EBADPARM;
    if (n != L->H || k != L->H || !hs) return AEE_EBADPARM;

    for (r = 0; r < reps; r++) {
        int64_t t0 = HAP_perf_get_time_us(), t1;
        int bt[3] = { 0, 0, 0 };

        /* --- in_proj for all M tokens in one weight pass --- */
        rc = lfm_bridge_gemv(hi, M, (const int8_t *)act, g_bcx, bt);
        if (rc != 0) return rc;
        t1 = HAP_perf_get_time_us(); t_in += (int)(t1 - t0); t0 = t1;

        /* --- conv stage, token by token (the state is sequential) --- */
        for (m = 0; m < M; m++) {
            const int8_t *bc = g_bcx + (size_t)m * 3 * L->H;
            size_t off = (size_t)slot * L->H * 8;
            int8_t *sthi = L->sthi + off;
            int8_t *stlo = L->stlo + off;
            unsigned ctr = L->ctr[slot];
            int p = conv_ring_slot(ctr);
            int64_t u0 = HAP_perf_get_time_us(), u1;

            conv_bx_write(sthi, stlo, p, bc, bc + 2 * L->H, L->H);
            u1 = HAP_perf_get_time_us(); t_bx += (int)(u1 - u0); u0 = u1;

            conv_dot(sthi, stlo, L->wq8 + (size_t)p * L->ng * 128, g_res, L->ng);
            u1 = HAP_perf_get_time_us(); t_dot += (int)(u1 - u0); u0 = u1;

            conv_pack_y(g_res, bc + L->H, L->kq, L->ksh,
                        g_y + (size_t)m * L->H, L->H);
            u1 = HAP_perf_get_time_us(); t_glue += (int)(u1 - u0);

            L->ctr[slot] = ctr + 1;
        }
        t1 = HAP_perf_get_time_us(); t_conv += (int)(t1 - t0); t0 = t1;

        /* --- out_proj for all M tokens --- */
        rc = lfm_bridge_gemv(ho, M, g_y, (int8_t *)outp, bt);
        if (rc != 0) return rc;
        t1 = HAP_perf_get_time_us(); t_out += (int)(t1 - t0);
    }

    if (times && timesLen >= 8) {
        times[0] = (t_in + t_conv + t_out) / reps;
        times[1] = t_in / reps;
        times[2] = t_conv / reps;
        times[3] = t_out / reps;
        times[4] = t_bx / reps;
        times[5] = t_dot / reps;
        times[6] = t_glue / reps;
        times[7] = M;
    }
    return 0;
}
