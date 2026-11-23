#ifndef LFM_PREFILL_IMPL_H
#define LFM_PREFILL_IMPL_H

#include <stdint.h>

/*
 * HMX int4 prefill with per-group scales, ported from `dsp/hmx` (mx4_blkpref,
 * REPORT24).  The research stand stays the reference; this copy is compiled
 * into the product `lfmgemv` skeleton so the same .so can run prefill.
 *
 * Operands:
 *   A  M x K int8, row major (activations, M padded to 64 spatial)
 *   B  packed int4 weights, N/32 tiles of (K/32)*512 bytes:
 *      byte(t,k,n) = t*(K/32)*512 + 512*(k>>5) + 128*((k&31)>>3) + 4*n + (k&3),
 *      high nibble iff ((k>>2)&1), two's complement
 *   S  float[N * K/(32*G)] per-group scales
 *   C  float[M*N], written when mode bit8 is set
 *
 * mode bits: 1 stage weights per N tile (double buffered), 2 staging only,
 * 4 keep HMX power/lock, 8 write C.
 */
int lfm_prefill(const int8_t *A, const uint8_t *B, const float *S,
                const float *Q, int M, int N, int K, int G, int C0, int mode,
                int reps, float *C, int *info, int info_len);

#endif /* LFM_PREFILL_IMPL_H */
