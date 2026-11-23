#include "hmx_ops.h"
#include "hmx_impl.h"

void lfm_hmx_i4_range(const uint8_t *img, int n, int k, int n0, int cols,
                      const uint8_t *wn, int kc0, int kc, const int8_t *act,
                      int row0, int rows, const float *ascale, float *y, int ldy)
{
    const uint8_t *planes = lfm_q4_planes(img, n, k);
    int plane_ld = k / 2;
    int ng = k / LFM_HMX_KC;
    uint8_t wnbuf[LFM_HMX_WN_BYTES];
    uint8_t actbuf[LFM_HMX_ACT_BYTES];
    lfm_hmx_acc acc;
    lfm_hmx_bias_block bias;
    int c;
    int s;
    int j;

    if (cols > LFM_HMX_N)
        cols = LFM_HMX_N;
    if (rows > LFM_HMX_SPATIAL)
        rows = LFM_HMX_SPATIAL;

    for (c = 0; c < kc; c++) {
        const uint8_t *tile;
        const uint16_t *scrow;
        if (wn != NULL)
            tile = wn + (size_t)c * (size_t)LFM_HMX_N * (size_t)LFM_HMX_WN_BYTES;
        else {
            lfm_hmx_pack_wn_tile(wnbuf, planes, plane_ld, n0, kc0 + c, cols);
            tile = wnbuf;
        }
        lfm_hmx_pack_act_crouton(actbuf, act + (size_t)row0 * (size_t)k, k, rows, kc0 + c);
        lfm_hmx_acc_clear(&acc, rows);
        lfm_hmx_mac_i4(&acc, tile, actbuf, rows);
        for (j = 0; j < cols; j++) {
            int32_t sw = lfm_hmx_sum_wn_tile_n(tile, j);
            bias.word0[j] = LFM_HMX_BIAS_SCALE_IDENTITY;
            bias.word1[j] = (uint32_t)(int32_t)(-LFM_HMX_QZP * sw);
        }
        lfm_hmx_acc_bias(&acc, &bias, rows);
        scrow = lfm_q4_row_scales(img, n, k, n0);
        for (s = 0; s < rows; s++) {
            const float *asrow = ascale + (size_t)(row0 + s) * (size_t)ng;
            float *yrow = y + (size_t)(row0 + s) * (size_t)ldy + (size_t)n0;
            for (j = 0; j < cols; j++) {
                float wsc = lfm_f16_to_f32(scrow[(size_t)j * (size_t)ng + (size_t)(kc0 + c)]);
                float term = (float)acc.v[s][j] * wsc * asrow[kc0 + c];
                yrow[j] += term;
            }
        }
    }
}

void lfm_hmx_gemm_i4(const uint8_t *w, int n, int k, const int8_t *act,
                     const float *ascale, int m, float *y)
{
    int n0;
    int t0;

    if (w == NULL || act == NULL || ascale == NULL || y == NULL)
        return;
    if (n <= 0 || k <= 0 || m <= 0 || (k % LFM_HMX_KC) != 0)
        return;
    memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
    for (n0 = 0; n0 < n; n0 += LFM_HMX_N) {
        int cols = n - n0 < LFM_HMX_N ? n - n0 : LFM_HMX_N;
        for (t0 = 0; t0 < m; t0 += LFM_HMX_SPATIAL) {
            int rows = m - t0 < LFM_HMX_SPATIAL ? m - t0 : LFM_HMX_SPATIAL;
            lfm_hmx_i4_range(w, n, k, n0, cols, NULL, 0, k / LFM_HMX_KC, act, t0,
                             rows, ascale, y, n);
        }
    }
}
