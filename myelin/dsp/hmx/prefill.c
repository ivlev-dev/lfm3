#include "hmx_ops.h"
#include "hmx_impl.h"

#define LFM_HMX_STAGE_KC 8
#define LFM_HMX_STAGE_BYTES (LFM_HMX_STAGE_KC * LFM_HMX_N * LFM_HMX_WN_BYTES)
#define LFM_HMX_STAGE_ACT_BYTES (2u * LFM_HMX_ACT_BYTES)

size_t lfm_hmx_prefill_vtcm_bytes(void)
{
    size_t need = 2u * LFM_HMX_STAGE_BYTES + LFM_HMX_STAGE_ACT_BYTES;
    return lfm_hmx_align_up(need, (size_t)LFM_HMX_ALIGN);
}

static void i4_stage_block(uint8_t *dst, const uint8_t *img, int n, int k, int n0, int cols,
                           int kc0, int kc)
{
    const uint8_t *planes = lfm_q4_planes(img, n, k);
    int plane_ld = k / 2;
    int c;
    for (c = 0; c < kc; c++)
        lfm_hmx_pack_wn_tile(dst + (size_t)c * (size_t)LFM_HMX_N * (size_t)LFM_HMX_WN_BYTES,
                             planes, plane_ld, n0, kc0 + c, cols);
}

int lfm_hmx_prefill_i4(const lfm_hmx_layer_i4 *layer, const int8_t *act, int m, float *y,
                       lfm_hmx_vtcm *vtcm)
{
    size_t need = lfm_hmx_prefill_vtcm_bytes();
    int ng;
    int n0;
    int t0;

    if (layer == NULL || layer->weights == NULL || layer->ascale == NULL)
        return -1;
    if (act == NULL || y == NULL || vtcm == NULL || vtcm->base == NULL)
        return -1;
    if (layer->n <= 0 || layer->k <= 0 || m <= 0 || (layer->k % LFM_HMX_KC) != 0)
        return -1;
    if (vtcm->bytes < need)
        return -2;
    if (!lfm_hmx_is_aligned(vtcm->base, (size_t)LFM_HMX_ALIGN))
        return -3;

    ng = layer->k / LFM_HMX_KC;
    memset(y, 0, (size_t)m * (size_t)layer->n * sizeof(float));

    for (n0 = 0; n0 < layer->n; n0 += LFM_HMX_N) {
        int cols = layer->n - n0 < LFM_HMX_N ? layer->n - n0 : LFM_HMX_N;
        uint8_t *buf0 = vtcm->base;
        uint8_t *buf1 = vtcm->base + LFM_HMX_STAGE_BYTES;
        int nblocks = (ng + LFM_HMX_STAGE_KC - 1) / LFM_HMX_STAGE_KC;
        int b;

        i4_stage_block(buf0, layer->weights, layer->n, layer->k, n0, cols, 0,
                       ng < LFM_HMX_STAGE_KC ? ng : LFM_HMX_STAGE_KC);
        for (b = 0; b < nblocks; b++) {
            int kc0 = b * LFM_HMX_STAGE_KC;
            int kc = ng - kc0 < LFM_HMX_STAGE_KC ? ng - kc0 : LFM_HMX_STAGE_KC;
            uint8_t *cur = (b & 1) ? buf1 : buf0;
            uint8_t *next = (b & 1) ? buf0 : buf1;
            if (b + 1 < nblocks) {
                int nc0 = (b + 1) * LFM_HMX_STAGE_KC;
                int nc = ng - nc0 < LFM_HMX_STAGE_KC ? ng - nc0 : LFM_HMX_STAGE_KC;
                i4_stage_block(next, layer->weights, layer->n, layer->k, n0, cols, nc0, nc);
            }
            for (t0 = 0; t0 < m;) {
                int rem = m - t0;
                int chunk = rem >= 128 ? 128 : (rem >= 64 ? 64 : 32);
                int rows = rem < chunk ? rem : chunk;
                int u;
                for (u = 0; u < rows; u += LFM_HMX_SPATIAL) {
                    int sub = rows - u < LFM_HMX_SPATIAL ? rows - u : LFM_HMX_SPATIAL;
                    lfm_hmx_i4_range(layer->weights, layer->n, layer->k, n0, cols, cur, kc0,
                                     kc, act, t0 + u, sub, layer->ascale, y, layer->n);
                }
                t0 += rows;
            }
        }
    }
    return 0;
}

int lfm_hmx_prefill_layers(const lfm_hmx_layer_i4 *layers, int n_layers, const int8_t *act,
                           float *const *y, int m, lfm_hmx_vtcm *vtcm)
{
    int l;
    if (layers == NULL || y == NULL || n_layers <= 0)
        return -1;
    for (l = 0; l < n_layers; l++) {
        int rc = lfm_hmx_prefill_i4(&layers[l], act, m, y[l], vtcm);
        if (rc != 0)
            return rc;
    }
    return 0;
}
