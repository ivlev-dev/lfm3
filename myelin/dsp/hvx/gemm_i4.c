#include "hvx_ops.h"

#include "i4_core.h"

void lfm_hvx_gemm_i4(const uint8_t *img, int n, int k, const int8_t *q, int m,
                     const float *ascale, float *y)
{
    lfm_i4_rows(img, n, k, q, m, ascale, y);
}
