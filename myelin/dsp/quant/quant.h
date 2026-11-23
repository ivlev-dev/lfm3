/*
 * quant.h - canonical activation quantisation for the LFM3 DSP kernels.
 *
 * This is the *reference* behaviour of the OpenCL backend
 * (crates/backend-opencl/kernels/decode.cl:78-121, `quantize_q8`):
 *
 *   group            G = 32 consecutive input channels
 *   amax             = max |x| over the group
 *   scale            = amax / 127.0f            (f32, stored as is)
 *   inv              = 127.0f / amax            (f32; the kernel multiplies)
 *   q[i]             = convert_int_rte(x[i]*inv) in [-127, 127]
 *   zero group       -> scale = 0, q = 0
 *   no clipping: |x*inv| <= 127 by construction, -128 is never produced.
 *
 * It also provides the zero-point view used by the HMX path.  HMX
 * `activation.ub = mxmem(...)` reads UNsigned bytes, therefore a signed int8
 * activation q is stored as
 *
 *   u = (uint8_t)(q + 128)                       zp = 128
 *
 * and, because dot(w, q) = dot(w, u) - zp*sum_k w[n][k], the row term
 * zp*sum(w) must be removed once per output row n.  On v68 that term fits
 * exactly in the signed int32 "input bias" word of the HMX bias block
 * (myelin/dsp/quant/hmxsim/probe_bias.c), so the correction is free.
 *
 * Header-only: it is compiled into the DSP kernels (hexagon-clang) and into
 * the host tools (cc) alike, with no libm rounding-mode setup.
 */
#ifndef LFM_QUANT_H
#define LFM_QUANT_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#define LFM_QGROUP 32
#define LFM_QZP    128

/* round to nearest, ties to even, matching OpenCL convert_int_rte().
 * Implemented on floor() so it does not depend on the FP rounding mode. */
static inline int lfm_rne_f32(float v)
{
    float f = floorf(v);
    float d = v - f;
    int fi = (int)f;
    if (d > 0.5f) return fi + 1;
    if (d < 0.5f) return fi;
    return (fi & 1) ? fi + 1 : fi;      /* exact .5 -> even */
}

/* Quantise `n` floats (n % 32 == 0) into int8 with one f32 scale per group.
 * Returns the number of values that had to be clamped (must be 0 for inputs
 * produced by the reference pipeline; kept as a diagnostic). */
static inline int lfm_quant_i8_group32(const float *x, int n,
                                       int8_t *q, float *scale)
{
    int g, i, clamped = 0;
    if (n % LFM_QGROUP) return -1;
    for (g = 0; g < n / LFM_QGROUP; g++) {
        const float *xg = x + (size_t)g * LFM_QGROUP;
        int8_t *qg = q + (size_t)g * LFM_QGROUP;
        float amax = 0.0f, inv;
        for (i = 0; i < LFM_QGROUP; i++) {
            float a = fabsf(xg[i]);
            if (a > amax) amax = a;
        }
        if (amax > 0.0f) {
            scale[g] = amax / 127.0f;
            inv = 127.0f / amax;
        } else {
            scale[g] = 0.0f;
            inv = 0.0f;
        }
        for (i = 0; i < LFM_QGROUP; i++) {
            int v = lfm_rne_f32(xg[i] * inv);
            if (v > 127) { v = 127; clamped++; }
            if (v < -127) { v = -127; clamped++; }
            qg[i] = (int8_t)v;
        }
    }
    return clamped;
}

/* signed int8 -> unsigned byte view (HMX activation.ub) */
static inline void lfm_q_to_u8(const int8_t *q, uint8_t *u, int n)
{
    int i;
    for (i = 0; i < n; i++)
        u[i] = (uint8_t)((int)q[i] + LFM_QZP);
}

/* exact integer dot products; int32 is enough for the shapes we use
 * (|K| <= 8192, |a| <= 127, |w| <= 127 -> 132M). */
static inline int32_t lfm_dot_i8(const int8_t *a, const int8_t *w, int n)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)w[i];
    return s;
}

static inline int32_t lfm_rowsum_i8(const int8_t *w, int n)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < n; i++) s += (int32_t)w[i];
    return s;
}

/* dot(w, q) computed from the unsigned byte view u = q + zp:
 *     dotu - zp * rowsum(w) == dot(q, w), exactly, in int32.
 * This is the identity the HMX bias word implements in hardware. */
static inline int32_t lfm_dot_u8_zp(const uint8_t *u, const int8_t *w, int n, int zp)
{
    int32_t s = 0, sw = 0;
    int i;
    for (i = 0; i < n; i++) {
        s  += (int32_t)u[i] * (int32_t)w[i];
        sw += (int32_t)w[i];
    }
    return s - (int32_t)zp * sw;
}

/* dequantise one group: x = q * scale (reference reconstruction) */
static inline void lfm_deq_i8_group32(const int8_t *q, const float *scale,
                                      float *x, int n)
{
    int g, i;
    for (g = 0; g < n / LFM_QGROUP; g++)
        for (i = 0; i < LFM_QGROUP; i++)
            x[g * LFM_QGROUP + i] = (float)q[g * LFM_QGROUP + i] * scale[g];
}

/* ================================================================== */
/* Hadamard rotation (orthogonal, deterministic)                        */
/* ================================================================== */
/*
 * Same construction as crates/backend-opencl/kernels/kvq4.cl:
 *
 *   R = (1/sqrt(d)) * H_d * D,     D = diag(+/-1 from LFM_TQ_SIGN)
 *   lfm_tq_rot_fwd(v) = (1/sqrt(d)) H_d (D v)
 *   lfm_tq_rot_inv(v) = (1/sqrt(d)) D (H_d v)
 *
 * H_d is the unnormalised +/-1 Hadamard butterfly, so
 * R^T R = I and <R w, R x> = <w, x>: rotating both the weight row and the
 * activation vector leaves every dot product unchanged, while per-group-32
 * int4/int8 quantisation sees a Gaussianised block instead of outliers.
 *
 * The 32-bit sign pattern is the first word of the reference table (so the
 * H32 rotation is a literal sub-block of the reference's H128 sign mask);
 * H128 uses all four words exactly as the reference does.
 */
static const uint32_t lfm_rot_sign_tab[4] = {
    0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu
};
#define LFM_ROT_SIGN(d) \
    (((lfm_rot_sign_tab[(d) >> 5] >> ((d) & 31u)) & 1u) ? -1.0f : 1.0f)

#define LFM_ROT_S32 0.17677669529663687f   /* 1/sqrt(32)  */
#define LFM_ROT_S128 0.08838834764831845f  /* 1/sqrt(128) */

/* in-place unnormalised Hadamard butterfly, n a power of two */
static inline void lfm_hadamard_f32(float *v, int n)
{
    int step, base, i;
    for (step = 1; step < n; step <<= 1)
        for (base = 0; base < n; base += 2 * step)
            for (i = 0; i < step; i++) {
                float a = v[base + i], b = v[base + i + step];
                v[base + i] = a + b;
                v[base + i + step] = a - b;
            }
}

static inline void lfm_rot_fwd_f32(float *v, int n)
{
    float s = (n == 32) ? LFM_ROT_S32 : LFM_ROT_S128;
    int i;
    for (i = 0; i < n; i++) v[i] *= LFM_ROT_SIGN(i);
    lfm_hadamard_f32(v, n);
    for (i = 0; i < n; i++) v[i] *= s;
}

static inline void lfm_rot_inv_f32(float *v, int n)
{
    float s = (n == 32) ? LFM_ROT_S32 : LFM_ROT_S128;
    int i;
    lfm_hadamard_f32(v, n);
    for (i = 0; i < n; i++) v[i] *= s * LFM_ROT_SIGN(i);
}

/* apply the forward rotation to every n-sized block of x (n = 0: no-op) */
static inline void lfm_rot_blocks_f32(float *x, int K, int n)
{
    int b;
    if (n <= 0) return;
    for (b = 0; b + n <= K; b += n) lfm_rot_fwd_f32(x + b, n);
}

/* metadata: which rotation was baked into a weight pack */
typedef struct {
    int   size;        /* 0, 32 or 128 */
    int   group;       /* quantisation group (32) */
    float scale;       /* 1/sqrt(size) */
    uint32_t sign0;    /* first sign word, for a cheap sanity check */
} lfm_rot_meta;

static inline lfm_rot_meta lfm_rot_meta_of(int size)
{
    lfm_rot_meta m;
    m.size = size;
    m.group = LFM_QGROUP;
    m.scale = (size == 32) ? LFM_ROT_S32 : (size == 128 ? LFM_ROT_S128 : 0.0f);
    m.sign0 = size ? lfm_rot_sign_tab[0] : 0u;
    return m;
}

/* ================================================================== */
/* int4 / 32 block packing (the DSP weight format, offset-8 nibbles)    */
/* ================================================================== */
/* Quantise f32 -> signed int4 in [-7,7] with one scale per 32 block,
 * half-away-from-zero (reference `f32::round`), then pack nibbles as
 * (q + 8) in 0..15, low nibble = even k.  `packed` holds n/2 bytes. */
static inline void lfm_pack_i4_row(const float *w, int n, uint8_t *packed,
                                   float *scale, int8_t *qout)
{
    int g, i;
    for (g = 0; g < n / LFM_QGROUP; g++) {
        const float *wg = w + g * LFM_QGROUP;
        float amax = 0.0f;
        for (i = 0; i < LFM_QGROUP; i++) {
            float a = fabsf(wg[i]);
            if (a > amax) amax = a;
        }
        scale[g] = (amax > 0.0f) ? amax / 7.0f : 0.0f;
        for (i = 0; i < LFM_QGROUP; i++) {
            int v = 0;
            if (scale[g] > 0.0f) {
                float z = wg[i] / scale[g];
                v = (int)(z >= 0.0f ? floorf(z + 0.5f) : ceilf(z - 0.5f));
                if (v > 7) v = 7;
                if (v < -7) v = -7;
            }
            if (qout) qout[g * LFM_QGROUP + i] = (int8_t)v;
            if (packed && (i & 1) == 0)
                packed[(g * LFM_QGROUP + i) / 2] =
                    (uint8_t)((v + 8) & 0xF);
            else if (packed)
                packed[(g * LFM_QGROUP + i) / 2] |= (uint8_t)(((v + 8) & 0xF) << 4);
        }
    }
}

#endif /* LFM_QUANT_H */
