#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "../common/q4.h"
#include "../common/quant.h"

#include "hvx_ops.h"

static float ffn_silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

void lfm_hvx_ffn_i4(const uint8_t *gate, int ffn, int hidden, const uint8_t *up,
                    const uint8_t *down, const int8_t *xq, const float *xscale, float *out)
{
    float *mid;
    float *g;
    float *u;
    int8_t *midq;
    float *midscale;
    int groups;
    int r;

    if (gate == NULL || up == NULL || down == NULL || xq == NULL || xscale == NULL ||
        out == NULL)
        return;
    if (ffn <= 0 || hidden <= 0)
        return;
    if ((ffn % LFM_QGROUP) != 0 || (hidden % LFM_QGROUP) != 0)
        return;

    groups = ffn / LFM_QGROUP;
    mid = (float *)malloc((size_t)ffn * sizeof(float));
    g = (float *)malloc((size_t)ffn * sizeof(float));
    u = (float *)malloc((size_t)ffn * sizeof(float));
    midq = (int8_t *)malloc((size_t)ffn);
    midscale = (float *)malloc((size_t)groups * sizeof(float));
    if (mid == NULL || g == NULL || u == NULL || midq == NULL || midscale == NULL) {
        free(mid);
        free(g);
        free(u);
        free(midq);
        free(midscale);
        return;
    }

    lfm_hvx_gemv_i4(gate, ffn, hidden, xq, xscale, g);
    lfm_hvx_gemv_i4(up, ffn, hidden, xq, xscale, u);
    for (r = 0; r < ffn; r++)
        mid[r] = ffn_silu_f32(g[r]) * u[r];

    if (lfm_act_i8_rot(mid, ffn, LFM_Q4_ALIGN, midq, midscale) == 0)
        lfm_hvx_gemv_i4(down, hidden, ffn, midq, midscale, out);

    free(mid);
    free(g);
    free(u);
    free(midq);
    free(midscale);
}
