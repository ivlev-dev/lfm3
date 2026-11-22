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

/* (a*b)>>16 without 64-bit helpers -- identical to the GEMV harness idiom. */
static inline int32_t conv_mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((uint32_t)lo * (uint32_t)b) >> 8);
}

/* Q15 per-row dequant multiplier.  A conv output projection easily needs
 * s_w*sa/s_out > 1, so a Q16 field with a 1.0 clamp saturated the output
 * (64 % of values).  Q15 keeps 1 bit less precision but covers m up to 2.0. */
static inline int32_t conv_row_scale(float sw, float sa, float so)
{
    float m = sw * sa / so;
    unsigned q;
    if (m < 0.0f) m = 0.0f;
    if (m > 1.9999f) m = 1.9999f;
    q = (unsigned)(m * 32768.0f + 0.5f);
    if (q > 65535u) q = 65535u;
    return (int32_t)(q | (q << 16));
}

/* int8 dequant of one GEMV row: out = clamp((acc*qp)>>15, +-127) */
static inline int8_t conv_pack_i8(int32_t acc, int32_t qp)
{
    int32_t q = (int32_t)((uint32_t)qp & 0xffffu);
    int32_t v = conv_mulq16(acc, q) << 1;   /* >>16 then <<1 == >>15 */
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
