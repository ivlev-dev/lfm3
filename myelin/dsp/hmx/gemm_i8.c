#include "hmx_ops.h"

static void i8_combine(const int8_t *w, const float *wscale, int k, int n0, int cols,
                       const int8_t *act, int row0, int rows, const float *ascale, float *y,
                       int ldy)
{
    int ng = k / LFM_HMX_KC;
    uint8_t wbuf[LFM_HMX_WB_BYTES];
    uint8_t actbuf[LFM_HMX_ACT_BYTES];
    lfm_hmx_acc acc;
    lfm_hmx_bias_block bias;
    int c;
    int s;
    int j;

    for (c = 0; c < ng; c++) {
        lfm_hmx_pack_wb_tile(wbuf, w, k, n0, c, cols);
        lfm_hmx_pack_act_crouton(actbuf, act + (size_t)row0 * (size_t)k, k, rows, c);
        lfm_hmx_acc_clear(&acc, rows);
        lfm_hmx_mac_i8(&acc, wbuf, actbuf, rows);
        for (j = 0; j < cols; j++) {
            int32_t sw = lfm_hmx_sum_wb_tile_n(wbuf, j);
            bias.word0[j] = LFM_HMX_BIAS_SCALE_IDENTITY;
            bias.word1[j] = (uint32_t)(int32_t)(-LFM_HMX_QZP * sw);
        }
        lfm_hmx_acc_bias(&acc, &bias, rows);
        for (s = 0; s < rows; s++) {
            const float *asrow = ascale + (size_t)(row0 + s) * (size_t)ng;
            float *yrow = y + (size_t)(row0 + s) * (size_t)ldy + (size_t)n0;
            for (j = 0; j < cols; j++) {
                float wsc = wscale != NULL ? wscale[(size_t)(n0 + j) * (size_t)ng + (size_t)c]
                                           : 1.0f;
                float term = (float)acc.v[s][j] * wsc * asrow[c];
                yrow[j] += term;
            }
        }
    }
}

static void i8_accumulate(const int8_t *w, int k, int n0, int cols, const int8_t *act,
                          int row0, int rows, int32_t *out, int ld)
{
    int ng = k / LFM_HMX_KC;
    uint8_t wbuf[LFM_HMX_WB_BYTES];
    uint8_t actbuf[LFM_HMX_ACT_BYTES];
    lfm_hmx_acc acc;
    lfm_hmx_bias_block bias;
    int c;
    int s;
    int j;

    for (c = 0; c < ng; c++) {
        lfm_hmx_pack_wb_tile(wbuf, w, k, n0, c, cols);
        lfm_hmx_pack_act_crouton(actbuf, act + (size_t)row0 * (size_t)k, k, rows, c);
        lfm_hmx_acc_clear(&acc, rows);
        lfm_hmx_mac_i8(&acc, wbuf, actbuf, rows);
        for (j = 0; j < cols; j++) {
            int32_t sw = lfm_hmx_sum_wb_tile_n(wbuf, j);
            bias.word0[j] = LFM_HMX_BIAS_SCALE_IDENTITY;
            bias.word1[j] = (uint32_t)(int32_t)(-LFM_HMX_QZP * sw);
        }
        lfm_hmx_acc_bias(&acc, &bias, rows);
        for (s = 0; s < rows; s++) {
            int32_t *orow = out + (size_t)(row0 + s) * (size_t)ld + (size_t)n0;
            for (j = 0; j < cols; j++) {
                int64_t v = (int64_t)orow[j] + (int64_t)acc.v[s][j];
                if (v > INT32_MAX)
                    v = INT32_MAX;
                if (v < INT32_MIN)
                    v = INT32_MIN;
                orow[j] = (int32_t)v;
            }
        }
    }
}

void lfm_hmx_gemm_i8(const int8_t *w, const float *wscale, int n, int k, const int8_t *act,
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
            i8_combine(w, wscale, k, n0, cols, act, t0, rows, ascale, y, n);
        }
    }
}

void lfm_hmx_gemm_i8_acc(const int8_t *w, int n, int k, const int8_t *act, int m, int32_t *acc)
{
    int n0;
    int t0;

    if (w == NULL || act == NULL || acc == NULL)
        return;
    if (n <= 0 || k <= 0 || m <= 0 || (k % LFM_HMX_KC) != 0)
        return;
    memset(acc, 0, (size_t)m * (size_t)n * sizeof(int32_t));
    for (n0 = 0; n0 < n; n0 += LFM_HMX_N) {
        int cols = n - n0 < LFM_HMX_N ? n - n0 : LFM_HMX_N;
        for (t0 = 0; t0 < m; t0 += LFM_HMX_SPATIAL) {
            int rows = m - t0 < LFM_HMX_SPATIAL ? m - t0 : LFM_HMX_SPATIAL;
            i8_accumulate(w, k, n0, cols, act, t0, rows, acc, n);
        }
    }
}
