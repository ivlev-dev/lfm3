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

/* ------------------------------------------------------------------ */
/* GGUF dequant and Q4_K requantisation (Hadamard weight rotation)     */
/* ------------------------------------------------------------------ */

void q4k_dequant_gguf(const uint8_t *blk, float *out)
{
    float d = q4k_f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = q4k_f16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t *sc = blk + 4, *qs = blk + 16;
    int j, half, l, q_off = 0, is = 0;

    for (j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        q4k_scale_min(sc, is, &s1, &m1);
        q4k_scale_min(sc, is + 1, &s2, &m2);
        for (l = 0; l < 32; l++)
            out[j + l] = d * (float)s1 * (float)(qs[q_off + l] & 15) -
                         dmin * (float)m1;
        for (l = 0; l < 32; l++)
            out[j + 32 + l] = d * (float)s2 * (float)(qs[q_off + l] >> 4) -
                              dmin * (float)m2;
        q_off += 32;
        is += 2;
    }
}

static void q4k_put_scale_min(uint8_t *q12, int j, int s, int m)
{
    int jm = j & 3;
    if (s < 0) s = 0;
    if (s > 63) s = 63;
    if (m < 0) m = 0;
    if (m > 63) m = 63;
    if (j < 4) {
        q12[jm] = (uint8_t)((q12[jm] & 0xc0u) | (s & 63));
        q12[jm + 4] = (uint8_t)((q12[jm + 4] & 0xc0u) | (m & 63));
    } else {
        q12[jm] = (uint8_t)((q12[jm] & 0x3fu) | ((s >> 4) << 6));
        q12[jm + 4] = (uint8_t)((q12[jm + 4] & 0x3fu) | ((m >> 4) << 6));
        q12[jm + 8] = (uint8_t)((q12[jm + 8] & 0xf0u) | (s & 15));
        q12[jm + 8] = (uint8_t)((q12[jm + 8] & 0x0fu) | ((m & 15) << 4));
    }
}

static int q4k_iround(float x)
{
    return (int)roundf(x);
}

void q4k_quant_block(const float *x, uint8_t *blk)
{
    float scale[8], minv[8], max_scale = 0.0f, max_min = 0.0f;
    uint8_t q12[12];
    float d, dmin;
    int j, l, q[256];

    for (j = 0; j < 8; j++) {
        float mn = x[32 * j], mx = x[32 * j];
        for (l = 1; l < 32; l++) {
            float v = x[32 * j + l];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (mn > 0.0f) mn = 0.0f;
        if (mx < 0.0f) mx = 0.0f;
        scale[j] = (mx - mn) / 15.0f;
        minv[j] = -mn;
        if (scale[j] > max_scale) max_scale = scale[j];
        if (minv[j] > max_min) max_min = minv[j];
    }
    if (!(max_scale > 0.0f)) {
        memset(blk, 0, Q4K_GGUF_BLOCK);
        return;
    }
    {
        uint16_t db = q4k_f32_to_f16(max_scale / 63.0f);
        uint16_t mb = q4k_f32_to_f16(max_min / 63.0f);
        blk[0] = (uint8_t)(db & 0xff);
        blk[1] = (uint8_t)(db >> 8);
        blk[2] = (uint8_t)(mb & 0xff);
        blk[3] = (uint8_t)(mb >> 8);
        d = q4k_f16_to_f32(db);
        dmin = q4k_f16_to_f32(mb);
    }
    memset(q12, 0, sizeof(q12));
    for (j = 0; j < 8; j++) {
        int s = q4k_iround(scale[j] / d);
        int m = q4k_iround(minv[j] / dmin);
        q4k_put_scale_min(q12, j, s, m);
        for (l = 0; l < 32; l++) {
            float mn = -minv[j];
            float inv = 1.0f / scale[j];
            int v = q4k_iround((x[32 * j + l] - mn) * inv);
            if (v < 0) v = 0;
            if (v > 15) v = 15;
            q[32 * j + l] = v;
        }
    }
    memcpy(blk + 4, q12, 12);
    for (j = 0; j < 4; j++)
        for (l = 0; l < 32; l++)
            blk[16 + 32 * j + l] = (uint8_t)(q[64 * j + l] |
                                             (q[64 * j + 32 + l] << 4));
}

int q4k_pack_image_rotated(const uint8_t *src, int rows, int K, int rot,
                           uint8_t *img)
{
    size_t blocks = (size_t)K / 256, n = (size_t)rows * blocks;
    uint8_t *tmp;
    float w[256];
    int r, b, rc;

    if (!src || !img || rows < 1 || K < 256 || (K & 255)) return -1;
    if (rot != 0 && rot != 128) return -1;
    tmp = (uint8_t *)malloc(n * Q4K_GGUF_BLOCK);
    if (!tmp) return -1;
    for (r = 0; r < rows; r++) {
        for (b = 0; b < (int)blocks; b++) {
            size_t idx = (size_t)r * blocks + (size_t)b;
            const uint8_t *sb = src + idx * Q4K_GGUF_BLOCK;
            uint8_t *db = tmp + idx * Q4K_GGUF_BLOCK;
            if (rot == 0) {
                memcpy(db, sb, Q4K_GGUF_BLOCK);
                continue;
            }
            q4k_dequant_gguf(sb, w);
            lfm_rot_fwd_f32(w, 128);
            lfm_rot_fwd_f32(w + 128, 128);
            q4k_quant_block(w, db);
        }
    }
    rc = q4k_pack_image(tmp, rows, K, img);
    free(tmp);
    return rc;
}
