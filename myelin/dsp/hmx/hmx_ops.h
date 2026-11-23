#ifndef LFM_HMX_OPS_H
#define LFM_HMX_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "../common/q4.h"
#include "hmx_util.h"

/*
 * Input contract for lfm_hmx_gemm_i4.
 *
 * w        q4 weight image produced by common/q4.h:
 *            [ f16 scales: n * (k/32) ][ pad to 128 ][ planes: n * (k/2) ]
 *          Row r starts at scales + r*(k/32); plane row r at planes + r*(k/2),
 *          nibble k/2, low nibble for even k, high for odd k, value q+8.
 *          The rows must already be rotated by R = (1/sqrt(128)) H128 D
 *          (lfm_q4_pack_matrix(w, n, k, img, 128)).
 * n        output columns, any positive value; tiles of 32 are used.
 * k        inner dimension, multiple of 32, any value.
 * act      activations, row-major m x k, int8 in [-127,127], already rotated
 *          by the same R in blocks of 128 (lfm_act_i8_rot(x, k, 128, q, s)).
 * ascale   activation scales, m * (k/32), scale of group g = ascale[t*(k/32)+g].
 * m        rows, 32, 64 or 128 are the tuned sizes; any positive value works.
 * y        output, row-major m x n, f32.
 *
 * Fixed-point contract: every 32-K group is accumulated in an int32
 * accumulator, the activation zero point is folded into the HMX input-bias
 * word (-128 * sum(w) over the group), and the f32 result is
 *
 *   y[t][j] = sum_g ((float)acc[t][j][g] * wscale[j][g]) * ascale[t][g]
 *
 * with g ascending and each product rounded to f32 before the next multiply.
 */
void lfm_hmx_gemm_i4(const uint8_t *w, int n, int k, const int8_t *act,
                     const float *ascale, int m, float *y);

/*
 * Input contract for lfm_hmx_gemm_i8.
 *
 * w        row-major n x k int8 weights, already rotated in blocks of 128.
 * wscale   weight scales, n * (k/32); group g scale = wscale[j*(k/32)+g].
 * act      row-major m x k int8 activations, already rotated.
 * ascale   activation scales, m * (k/32).
 * m, n, k  same limits as lfm_hmx_gemm_i4.
 * y        row-major m x n f32.
 *
 * The integer path is bit exact against the CPU reference: each 32-K group is
 * an exact int32 dot with the q+128 / -128*sum(w) zero-point compensation,
 * then combined in the same f32 order as lfm_hmx_gemm_i4.
 */
void lfm_hmx_gemm_i8(const int8_t *w, const float *wscale, int n, int k,
                     const int8_t *act, const float *ascale, int m, float *y);

/*
 * Exact int32 accumulator view used by the verification gate. acc is
 * row-major m x n and receives sum_k w[j][k] * act[t][k] with no scaling.
 */
void lfm_hmx_gemm_i8_acc(const int8_t *w, int n, int k, const int8_t *act,
                         int m, int32_t *acc);

typedef struct {
    const uint8_t *weights;
    const float *ascale;
    int n;
    int k;
} lfm_hmx_layer_i4;

size_t lfm_hmx_prefill_vtcm_bytes(void);

/*
 * Chunked prefill of one int4 layer. Rows are processed in 128/64/32 chunks,
 * weights are staged from the DDR image into VTCM in 128 KiB blocks through
 * two alternating buffers. vtcm must come from lfm_hmx_vtcm_alloc().
 */
int lfm_hmx_prefill_i4(const lfm_hmx_layer_i4 *layer, const int8_t *act, int m,
                       float *y, lfm_hmx_vtcm *vtcm);

int lfm_hmx_prefill_layers(const lfm_hmx_layer_i4 *layers, int n_layers,
                           const int8_t *act, float *const *y, int m,
                           lfm_hmx_vtcm *vtcm);

#endif
