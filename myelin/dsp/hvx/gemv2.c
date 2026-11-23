/*
 * int4 GEMV v2.  Reference: crates/backend-opencl (prepack + int4/32 GEMV).
 *
 * w = row_scale * ratio/128 (one ratio per 256 weights); activations are int8
 * with one f32 scale per 32, so every per-32 dot is an exact int32 and the
 * combine is f32 in the same order as the reference.
 */
#include "gemv2.h"

#include <math.h>
#include <string.h>

#ifdef __HVX__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static inline int q_of(const uint8_t *codes, int k)
{
    int byte = codes[k >> 1];
    return ((k & 1) ? (byte >> 4) : (byte & 15)) - 8;
}

void gemv2_ref(const gemv2_matrix *m, const int8_t *xq, const float *sa,
               float *out)
{
    int n, b, g, i;
    for (n = 0; n < m->n; n++) {
        double acc = 0.0;
        const uint8_t *row = m->codes + (size_t)n * (m->k / 2);
        for (b = 0; b < m->k / GEMV2_BLOCK; b++) {
            double wscale = (double)m->row_scale[n] *
                            (double)m->seg_ratio[(size_t)n * m->seg_max + b] / 128.0;
            for (g = 0; g < 8; g++) {
                int32_t dot = 0;
                int base = b * GEMV2_BLOCK + g * GEMV2_GROUP;
                for (i = 0; i < GEMV2_GROUP; i++)
                    dot += q_of(row, base + i) * (int)xq[base + i];
                acc += wscale * (double)sa[b * 8 + g] * (double)dot;
            }
        }
        out[n] = (float)acc;
    }
}

#ifdef __HVX__
static inline HVX_Vector codes32(const uint8_t *p)
{
    HVX_Vector v = *(const HVX_Vector *)p;
    HVX_Vector lo = Q6_V_vand_VV(v, Q6_V_vsplat_R(0x0f0f0f0f));
    HVX_Vector hi = Q6_Vub_vlsr_VubR(v, 4);
#if GEMV2_SHUFFLE == 0
    HVX_Vector packed = Q6_Vb_vshuffe_VbVb(lo, hi);
#elif GEMV2_SHUFFLE == 1
    HVX_Vector packed = Q6_Vb_vshuffe_VbVb(hi, lo);
#elif GEMV2_SHUFFLE == 2
    HVX_Vector packed = Q6_Vb_vshuffo_VbVb(lo, hi);
#else
    HVX_Vector packed = Q6_Vb_vshuffo_VbVb(hi, lo);
#endif
    /* nibble + 8 form is offset-8 unsigned; the dot needs signed codes */
    return Q6_Vb_vsub_VbVb(packed, Q6_V_vsplat_R(0x08080808));
}

void gemv2_hvx(const gemv2_matrix *m, const int8_t *xq, const float *sa,
               float *out)
{
    int n, b, g;
    for (n = 0; n < m->n; n++) {
        const uint8_t *row = m->codes + (size_t)n * (m->k / 2);
        double acc = 0.0;
        for (b = 0; b < m->k / GEMV2_BLOCK; b++) {
            double wscale = (double)m->row_scale[n] *
                            (double)m->seg_ratio[(size_t)n * m->seg_max + b] / 128.0;
            for (g = 0; g < 8; g++) {
                int base = b * GEMV2_BLOCK + g * GEMV2_GROUP;
                HVX_Vector codes;
                HVX_Vector x = *(const HVX_Vector *)(xq + base);
                int8_t qbuf[128] __attribute__((aligned(128)));
                for (i = 0; i < 32; i++)
                    qbuf[i] = (int8_t)q_of(row, base + i);
                codes = *(const HVX_Vector *)qbuf;
                HVX_Vector prod = Q6_Vw_vrmpy_VbVb(codes, x);
                int32_t lanes[32];
                int i;
                int32_t dot = 0;
                memcpy(lanes, &prod, sizeof(lanes));
                for (i = 0; i < 32; i++)
                    dot += lanes[i];
                acc += wscale * (double)sa[b * 8 + g] * (double)dot;
            }
        }
        out[n] = (float)acc;
    }
}
#else
void gemv2_hvx(const gemv2_matrix *m, const int8_t *xq, const float *sa,
               float *out)
{
    gemv2_ref(m, xq, sa, out);
}
#endif
