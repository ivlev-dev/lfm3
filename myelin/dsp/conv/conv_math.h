#ifndef CONV_MATH_H
#define CONV_MATH_H
/*
 * Fixed-point recipes shared by the DSP conv layer (lfmconv_impl.c) and the
 * host reference (conv_main.c).  Plain integer / float code, no HVX, no
 * intrinsics: both builds execute literally the same operations, so the two
 * sides are supposed to agree bit for bit.
 *
 * Number formats
 *   bcx[]      int8, one scale s_bc            (in_proj / out_proj activation)
 *   bx16       exact int16 product b8*x8, NOT stored as such: it is split into
 *              hi = bx16>>8 (signed) and lo = bx16&0xff (unsigned) planes
 *   out32[ch]  sum_j wq[ch][j]*bx16[ch][j]     (exact int32)
 *   y[]        int8, one scale s_y
 */
#include <stdint.h>

/* Q16 per-row dequant multiplier, exactly as lfm_bridge set_scale() builds it
 * (lfmgemv_impl.c):  m = sw*sa/so, field clamped to 0.99997.
 *
 * History: an earlier revision switched this to Q15 (clamp 1.9999) to "fix"
 * int8 saturation of out_proj.  That was a red herring: the saturation came
 * from a double offset-8 subtraction in conv_ref.h (ref_dot_int4() already
 * removes the offset and the callers removed it again).  Q16 is restored
 * because it is the device's format; m is ~0.0015 for in_proj and ~0.015 for
 * out_proj on this layer, so the clamp is never reached.                   */
static inline int32_t conv_row_scale(float sw, float sa, float so)
{
    float m = sw * sa / so;
    unsigned q;
    if (m < 0.0f) m = 0.0f;
    if (m > 0.99997f) m = 0.99997f;
    q = (unsigned)(m * 65536.0f + 0.5f);
    if (q > 65535u) q = 65535u;
    return (int32_t)(q | (q << 16));
}

/* int8 dequant of one GEMV row: out = clamp(round(acc*qp/65536), +-127).
 *
 * The product is taken in 64 bits on purpose.  The GEMV harness idiom
 * mulq16(a,b) = (a>>16)*b + (((a>>8)&0xff)*b)>>8 resolves `a` with a step of
 * 256, so the int8 output error is (acc & 0xff)*m: harmless when m ~ 1 (the
 * comment in lfmgemv_impl.c quotes 0.03), but ~4 LSB of the 127-wide output
 * when m ~ 0.015, which is exactly this layer's out_proj.  Measured on the
 * stand (global cos(out) over all tokens): 0.986 with the truncation,
 * 0.996 without.
 * pack_out_i8() in lfmgemv_impl.c uses the same expression (same file set,
 * same arithmetic -- the host reference must stay bit-exact with the DSP). */
static inline int8_t conv_pack_i8(int32_t acc, int32_t qp)
{
    int32_t q = (int32_t)((uint32_t)qp & 0xffffu);
    int64_t v = ((int64_t)acc * (int64_t)q + 32768) >> 16;
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t)v;
}

/* depthwise weight quantisation; sign-symmetric round-half-away-from-zero */
static inline int conv_wq(float w, float sw)
{
    float t = w / sw;
    int q = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return q;
}

/* Real value of one unit of out32*c, i.e.
 *   y_real = out32 * c8 * K,   K = maxw*s_bc^3/(127*s_y)                  */
static inline double conv_K(double maxw, double s_bc, double s_y)
{
    return maxw * s_bc * s_bc * s_bc / (127.0 * s_y);
}

/* Shift that lifts the largest K close to 2^28 (no log2/ldexp: pure loop,
 * so the DSP and the host evaluate it identically). */
static inline int conv_pick_ksh(double kmax)
{
    int ksh = 2;
    double sc = 4.0;
    if (!(kmax > 0.0)) return 16;
    while (ksh < 40 && kmax * sc * 2.0 < 3.0e8) { sc *= 2.0; ksh++; }
    return ksh;
}

static inline int32_t conv_make_kq(double K, int ksh)
{
    double sc = 4.0;
    int i;
    for (i = 2; i < ksh; i++) sc *= 2.0;
    K = K * sc + 0.5;
    if (!(K > 0.0)) return 0;
    if (K > 2147483000.0) K = 2147483000.0;
    return (int32_t)K;               /* truncation of a positive value */
}

/* out32 -> int8 gate output.
 *   y8 = clamp( ((out32>>1)*c8*kq) >> (ksh-1) )
 * out32>>1 keeps (out32>>1)*c8 inside int32 (|out32| <= 7*127*16129).      */
static inline int8_t conv_glue(int32_t out32, int c8, int32_t kq, int ksh)
{
    int64_t v = (int64_t)((out32 >> 1) * (int32_t)c8) * (int64_t)kq;
    v += (int64_t)1 << (ksh - 2);
    v >>= (ksh - 1);
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t)v;
}

/* Ring bookkeeping shared by both sides.
 * ctr = tokens already seen in this slot; the new bx goes to slot ctr&7.
 * tap j (0 = oldest = t-6 .. 6 = current) lives in ring slot (p+j+2)&7.     */
static inline int conv_ring_slot(unsigned ctr)
{
    return (int)(ctr & 7u);
}
static inline int conv_tap_slot(int phase, int tap)
{
    return (phase + tap + 2) & 7;
}
/* inverse: which tap does ring slot s carry, for this phase (7 = unused) */
static inline int conv_slot_tap(int phase, int s)
{
    return (s - phase - 2) & 7;
}

#endif /* CONV_MATH_H */
