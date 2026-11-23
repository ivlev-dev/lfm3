#ifndef LFM_GEMV2_H
#define LFM_GEMV2_H

#include <stdint.h>

/*
 * int4 GEMV, v2: per-256 weight scales (7-bit ratio to the row scale) and
 * per-32 activation scales, the same numerics as the OpenCL reference
 * (`crates/backend-opencl`): out[n] = sum_b sum_g sa[8b+g] * wref[b] *
 * ratio[b] * dot32(q, x).
 *
 * Weight layout (per row, K weights, K % 256 == 0):
 *   [0, K/2)                  int4 codes, 16 bytes per 32 weights,
 *                             stored nibble = q + 8, low nibble = even k
 *   row_scale[n]              f32  (absmax/7 of the row)
 *   seg_ratio[n][K/256]       u8   (1..128, scale = row_scale * ratio / 128)
 *
 * Activations: int8 xq[K] and f32 sa[K/32].
 */
#define GEMV2_BLOCK 256
#define GEMV2_GROUP 32

typedef struct {
    int n;
    int k;
    const uint8_t *codes;
    const float *row_scale;
    const uint8_t *seg_ratio;
    int seg_max;
} gemv2_matrix;

void gemv2_ref(const gemv2_matrix *m, const int8_t *xq, const float *sa,
               float *out);
void gemv2_hvx(const gemv2_matrix *m, const int8_t *xq, const float *sa,
               float *out);

#endif
