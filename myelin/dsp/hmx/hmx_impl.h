#ifndef LFM_HMX_IMPL_H
#define LFM_HMX_IMPL_H

#include <stdint.h>

void lfm_hmx_i4_range(const uint8_t *img, int n, int k, int n0, int cols,
                      const uint8_t *wn, int kc0, int kc, const int8_t *act,
                      int row0, int rows, const float *ascale, float *y, int ldy);

#endif
