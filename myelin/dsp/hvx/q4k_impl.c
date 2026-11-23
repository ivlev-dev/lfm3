/*
 * Q4_K GEMV: pack, scalar reference and HVX kernel.
 * See q4k_impl.h for the block format and the formula provenance.
 */
#include "q4k_impl.h"

#include <math.h>
#include "../quant/quant.h"

/* ------------------------------------------------------------------ */
/* packing: GGUF 144-B Q4_K block -> 24-B header + 128-B plane         */
/* ------------------------------------------------------------------ */

/* qs chunks (32 B each) that feed one plane chunk, and the nibble taken.
 * plane chunk 0 = lo(qs[0..32])  | lo(qs[64..96])<<4   -> K[  0.. 31] / K[128..159]
 * plane chunk 1 = hi(qs[0..32])  | hi(qs[64..96])<<4   -> K[ 32.. 63] / K[160..191]
 * plane chunk 2 = lo(qs[32..64]) | lo(qs[96..128])<<4  -> K[ 64.. 95] / K[192..223]
 * plane chunk 3 = hi(qs[32..64]) | hi(qs[96..128])<<4  -> K[ 96..127] / K[224..255]
 */
static const int q4k_chunk_a[4] = { 0, 0, 32, 32 };
static const int q4k_chunk_b[4] = { 64, 64, 96, 96 };
static const int q4k_chunk_hi[4] = { 0, 1, 0, 1 };

static void q4k_pack_block(const uint8_t *src, uint8_t *hdr, uint8_t *plane)
{
    const uint8_t *qs = src + 16;
    int j, i, k;

    memcpy(hdr + 0, src + 0, 2);
    memcpy(hdr + 4, src + 2, 2);
    {
        float d = q4k_f16_to_f32((uint16_t)(src[0] | (src[1] << 8)));
        float dmin = q4k_f16_to_f32((uint16_t)(src[2] | (src[3] << 8)));
        memcpy(hdr + 0, &d, 4);
        memcpy(hdr + 4, &dmin, 4);
    }
    for (j = 0; j < 8; j++) {
        uint8_t sc, mn;
        q4k_scale_min(src + 4, j, &sc, &mn);
        hdr[8 + j] = sc;
        hdr[16 + j] = mn;
    }
    for (k = 0; k < 4; k++) {
        const uint8_t *a = qs + q4k_chunk_a[k];
        const uint8_t *b = qs + q4k_chunk_b[k];
        uint8_t *d = plane + 32 * k;
        if (q4k_chunk_hi[k]) {
            for (i = 0; i < 32; i++) d[i] = (uint8_t)((a[i] >> 4) | (b[i] & 0xf0u));
        } else {
            for (i = 0; i < 32; i++) d[i] = (uint8_t)((a[i] & 15u) | ((b[i] & 15u) << 4));
        }
    }
}

size_t q4k_image_bytes(int rows, int K)
{
    size_t blocks = (size_t)K / 256;
    size_t hdr = (size_t)rows * blocks * Q4K_HDR_BLOCK;
    size_t p_off = (hdr + 127u) & ~(size_t)127u;
    return p_off + (size_t)rows * blocks * Q4K_PLANE;
}

int q4k_pack_image(const uint8_t *src, int rows, int K, uint8_t *img)
{
    size_t blocks, hdr_bytes, p_off, n, r, blk;
    uint8_t *plane;

    if (K <= 0 || (K & 255) || rows <= 0) return -1;
    if (((uintptr_t)img) & 127u) return -2;
    blocks = (size_t)K / 256;
    hdr_bytes = (size_t)rows * blocks * Q4K_HDR_BLOCK;
    p_off = (hdr_bytes + 127u) & ~(size_t)127u;
    plane = img + p_off;
    n = 0;
    for (r = 0; r < (size_t)rows; r++) {
        for (blk = 0; blk < blocks; blk++) {
            q4k_pack_block(src + n * Q4K_GGUF_BLOCK,
                           img + ((size_t)r * blocks + blk) * Q4K_HDR_BLOCK,
                           plane + ((size_t)r * blocks + blk) * Q4K_PLANE);
            n++;
        }
    }
    return 0;
}

void q4k_mat_init(q4k_mat *m, const uint8_t *img, int N, int K)
{
    size_t blocks = (size_t)K / 256;
    size_t hdr_bytes = (size_t)N * blocks * Q4K_HDR_BLOCK;
    size_t p_off = (hdr_bytes + 127u) & ~(size_t)127u;
    m->N = N;
    m->K = K;
    m->blocks = (int)blocks;
    m->img = img;
    m->hdr = img;
    m->p = img + p_off;
}

void q4k_dequant_block(const uint8_t *hdr, const uint8_t *plane, float *out)
{
    float d, dmin;
    int g, i;
    memcpy(&d, hdr + 0, 4);
    memcpy(&dmin, hdr + 4, 4);
    for (g = 0; g < 8; g++) {
        float s = d * (float)hdr[8 + g];
        float mn = dmin * (float)hdr[16 + g];
        for (i = 0; i < 32; i++) {
            int k = 32 * g + i;
            out[k] = s * (float)Q4K_NIB(plane, k) - mn;
        }
    }
}

/* ------------------------------------------------------------------ */
/* activations                                                         */
/* ------------------------------------------------------------------ */

void q4k_act_q8(const float *x, int K, int8_t *xq, float *xs)
{
    (void)lfm_quant_i8_group32(x, K, xq, xs);
}

void q4k_act_sums(const int8_t *xq, int K, int32_t *sums)
{
    int g, i;
    for (g = 0; g < K / 32; g++) {
        int32_t s = 0;
        for (i = 0; i < 32; i++) s += (int32_t)xq[32 * g + i];
        sums[g] = s;
    }
}

/* ------------------------------------------------------------------ */
/* plan: the row-independent part (activation scales and group sums)   */
/* ------------------------------------------------------------------ */

int q4k_plan_init(q4k_plan *pl, const q4k_mat *m, const int8_t *xq,
                  const float *xs, const int32_t *sums)
{
    int g, n = m->blocks * 8;
    int32_t *own = NULL;
    const int32_t *sg = sums;

    pl->xs = NULL;
    pl->xsum = NULL;
    pl->blocks = m->blocks;
    if (!sg) {
        own = (int32_t *)malloc((size_t)n * sizeof(int32_t));
        if (!own) return -1;
        q4k_act_sums(xq, m->K, own);
        sg = own;
    }
    pl->xs = (float *)malloc((size_t)n * sizeof(float));
    pl->xsum = (float *)malloc((size_t)n * sizeof(float));
    if (!pl->xs || !pl->xsum) {
        free(own);
        q4k_plan_free(pl);
        return -1;
    }
    for (g = 0; g < n; g++) {
        pl->xs[g] = xs[g];
        pl->xsum[g] = (float)sg[g];   /* reference: xs * sum * dmin*m */
    }
    free(own);
    return 0;
}

void q4k_plan_free(q4k_plan *p)
{
    free(p->xs);
    free(p->xsum);
    p->xs = NULL;
    p->xsum = NULL;
}

/* ------------------------------------------------------------------ */
/* scalar reference of the exact kernel formula                        */
/* ------------------------------------------------------------------ */

void q4k_gemv_scalar(const q4k_mat *m, const q4k_plan *pl, const int8_t *xq,
                     float *y, int r0, int r1)
{
    int n, b, g, i;
    const size_t hrow = (size_t)m->blocks * Q4K_HDR_BLOCK;
    const size_t prow = (size_t)m->blocks * Q4K_PLANE;

    for (n = r0; n < r1; n++) {
        const uint8_t *hdr = m->hdr + (size_t)n * hrow;
        const uint8_t *plane = m->p + (size_t)n * prow;
        float acc = 0.0f;
        for (b = 0; b < m->blocks; b++) {
            const uint8_t *h = hdr + (size_t)b * Q4K_HDR_BLOCK;
            const uint8_t *pl_ = plane + (size_t)b * Q4K_PLANE;
            const int8_t *xg = xq + 256 * b;
            const float *xs = pl->xs + b * 8;
            float d, dmin;
            memcpy(&d, h + 0, 4);
            memcpy(&dmin, h + 4, 4);
            for (g = 0; g < 8; g++) {
                int32_t dot = 0, s = 0;
                float ds = d * (float)h[8 + g];
                float dm = dmin * (float)h[16 + g];
                for (i = 0; i < 32; i++) {
                    int q = Q4K_NIB(pl_, 32 * g + i);
                    dot += (int32_t)q * (int32_t)xg[32 * g + i];
                    s += (int32_t)xg[32 * g + i];
                }
                acc += xs[g] * (ds * (float)dot - dm * (float)s);
            }
        }
        y[n] = acc;
    }
}

/* Q4_K weights, f32 activations, double accumulation (oracle) */
void q4k_gemv_f32act(const q4k_mat *m, const float *x, float *y, int r0, int r1)
{
    int n, b, i;
    const size_t hrow = (size_t)m->blocks * Q4K_HDR_BLOCK;
    const size_t prow = (size_t)m->blocks * Q4K_PLANE;
    float wblk[256];

    for (n = r0; n < r1; n++) {
        const uint8_t *hdr = m->hdr + (size_t)n * hrow;
        const uint8_t *plane = m->p + (size_t)n * prow;
        double acc = 0.0;
        for (b = 0; b < m->blocks; b++) {
            q4k_dequant_block(hdr + (size_t)b * Q4K_HDR_BLOCK,
                              plane + (size_t)b * Q4K_PLANE, wblk);
            for (i = 0; i < 256; i++) acc += (double)wblk[i] * (double)x[256 * b + i];
        }
        y[n] = (float)acc;
    }
}

/* ------------------------------------------------------------------ */
/* HVX kernel                                                          */
/* ------------------------------------------------------------------ */
#ifdef __HVX__

#include <hexagon_types.h>
#include "hvx_hexagon_protos.h"

static inline HVX_Vector q4k_mask_lo(HVX_Vector v)
{
    return Q6_V_vand_VV(v, Q6_V_vsplat_R(0x0f0f0f0fu));
}
static inline HVX_Vector q4k_shr_hi(HVX_Vector v)
{
    return Q6_Vub_vlsr_VubR(v, 4);
}

/* lane l <- sum of lanes l..l+7 (mod 32); groups land in lanes 0, 8, 16, 24.
 * 3 x (vror, vadd) after the 128-byte product vector. */
static inline HVX_Vector q4k_reduce8(HVX_Vector v)
{
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    return v;
}

void q4k_gemv_hvx(const q4k_mat *m, const q4k_plan *pl, const int8_t *xq,
                  float *y, int r0, int r1)
{
    const size_t hrow = (size_t)m->blocks * Q4K_HDR_BLOCK;
    const size_t prow = (size_t)m->blocks * Q4K_PLANE;
    int n, b;

    for (n = r0; n < r1; n++) {
        const uint8_t *hdr = m->hdr + (size_t)n * hrow;
        const uint8_t *plane = m->p + (size_t)n * prow;
        float acc = 0.0f;

        for (b = 0; b < m->blocks; b++) {
            const uint8_t *h = hdr + (size_t)b * Q4K_HDR_BLOCK;
            const HVX_Vector pv = *(const HVX_Vector *)(plane + (size_t)b * Q4K_PLANE);
            const HVX_Vector xl = *(const HVX_Vector *)(xq + 256 * b);
            const HVX_Vector xh = *(const HVX_Vector *)(xq + 256 * b + 128);
            const float *xs = pl->xs + b * 8;
            const float *xsum = pl->xsum + b * 8;
            float d, dmin;
            HVX_Vector vlo, vhi;
            union { HVX_Vector v; int32_t s[32]; } ulo, uhi;
            int g;

            memcpy(&d, h + 0, 4);
            memcpy(&dmin, h + 4, 4);

            /* groups 0..3: low nibbles of the plane == K[0..127] */
            vlo = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), q4k_mask_lo(pv), xl);
            /* groups 4..7: high nibbles == K[128..255] */
            vhi = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), q4k_shr_hi(pv), xh);
            ulo.v = q4k_reduce8(vlo);
            uhi.v = q4k_reduce8(vhi);

            for (g = 0; g < 4; g++) {
                float ds = d * (float)h[8 + g];
                float dm = dmin * (float)h[16 + g];
                acc += xs[g] * (ds * (float)ulo.s[8 * g] - dm * xsum[g]);
            }
            for (g = 0; g < 4; g++) {
                float ds = d * (float)h[12 + g];
                float dm = dmin * (float)h[20 + g];
                acc += xs[4 + g] * (ds * (float)uhi.s[8 * g] - dm * xsum[4 + g]);
            }
        }
        y[n] = acc;
    }
}

#endif /* __HVX__ */
