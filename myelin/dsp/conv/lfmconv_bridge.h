#ifndef LFMCONV_BRIDGE_H
#define LFMCONV_BRIDGE_H
/*
 * Internal API that lfmgemv_impl.c exposes to the shortconv layer
 * (lfmconv_impl.c).  Not part of the FastRPC interface.
 */
#include <stdint.h>

/* int4 GEMV tuning: l2fetch lookahead in KB, rows per iteration,
 * prefetch on/off, GEMV threads.  Values < 0 are ignored, rows/threads < 1
 * are ignored. */
int lfm_bridge_tune(int pf_kb, int rows, int usepf, int threads);

/* Geometry of a resident weight buffer created by alloc_buf/set_shape. */
int lfm_bridge_shape(int h, int *N, int *K, int *is_int8, int *has_scale);

/* One full GEMV for M = 1|2|4: deinterleave -> int4 dot -> int8 pack.
 * act = M*K int8, out = M*N int8 (buffer must have a Q16 scale table).
 * t[3] (optional) receives microseconds for deinterleave / matmul / pack. */
int lfm_bridge_gemv(int h, int M, const int8_t *act, int8_t *out, int *t);

#endif /* LFMCONV_BRIDGE_H */
