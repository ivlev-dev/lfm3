#include "hvx_ops.h"

#include "i4_core.h"

void lfm_hvx_gemv_i4(const uint8_t *img, int n, int k, const int8_t *q, const float *ascale,
                     float *y)
{
    lfm_i4_rows(img, n, k, q, 1, ascale, y);
}
