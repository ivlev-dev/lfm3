#ifndef Q4K_IMPL_H
#define Q4K_IMPL_H
/*
 * Q4_K GEMV, 1:1 with the OpenCL reference (crates/backend-opencl).
 *
 * The reference pipeline is
 *   prepack_q4_k   : 144-B GGUF block -> 148-B block (d, dmin, 8 unpacked
 *                    (sc,m) pairs, 128 B of qs), crates/backend-opencl/src/prepack.rs
 *   gemv_q4_k_i8*  : int8/32 activations, per-32-group int32 dot, f32 combine,
 *                    kernels/decode.cl:137-270
 * This file reproduces that maths on HVX.  Two layout changes, both lossless:
 *
 *  1. the 128 qs bytes are permuted with a 32-byte-granular nibble move so
 *     that the low nibbles hold K[0..127] and the high nibbles K[128..255] of
 *     the block.  GGUF interleaves the two halves of a 64-weight quarter;
 *     HVX needs 128 consecutive activations per vector product.  Q4K.md s.2.
 *  2. the 128-B plane is split off the 24-B header so every plane load is
 *     128-B aligned (HVX has no unaligned vector load).
 *
 * Packed image (one malloc, 128-B aligned base):
 *   [0, hdr_bytes)              N*blocks * 24 B:  f32 d, f32 dmin, sc[8], m[8]
 *   [p_off, p_off + N*blocks*128)  planes, block (n,b) at (n*blocks+b)*128
 * Block (n,b) covers weights [256b, 256b+256) of row n; plane byte i holds
 * weight 256b+i in the low nibble and 256b+128+i in the high nibble.
 *
 * Activations: int8 with one f32 scale per 32 channels (lfm_quant_i8_group32,
 * myelin/dsp/quant/quant.h == decode.cl:78-121 `quantize_q8`).
 *
 *   y[n] = sum_b sum_g xs[b*8+g] * ( d_b*sc[b*8+g]*dot(q, xq)
 *                                    - dmin_b*m[b*8+g]*sum(xq) )
 * dot() and sum() are exact int32 per 32-activation group; the second term
 * does not depend on the row and is folded into one scalar per (token) set
 * in q4k_plan.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Q4K_GGUF_BLOCK 144
#define Q4K_HDR_BLOCK   24
#define Q4K_PLANE      128

typedef struct {
    int N, K, blocks;
    const uint8_t *img;   /* 128-B aligned packed image                        */
    const uint8_t *hdr;   /* = img                                            */
    const uint8_t *p;     /* = img + p_off, 128-B aligned planes               */
} q4k_mat;

/* bytes of a packed image for rows x K (K % 256 == 0) */
size_t q4k_image_bytes(int rows, int K);

/* GGUF Q4_K rows -> packed image.  src: rows * (K/256) * 144 bytes.
 * img must be 128-B aligned and at least q4k_image_bytes() long. */
int q4k_pack_image(const uint8_t *src, int rows, int K, uint8_t *img);

/* GGUF 144-B block -> 256 f32, exactly infra-quant dequant_q4_k. */
void q4k_dequant_gguf(const uint8_t *blk, float *out);

/* f32 256 weights -> GGUF 144-B Q4_K block (llama.cpp reference recipe:
 * 8 sub-blocks of 32, 6-bit sc/m, d = max_scale/63, dmin = max_min/63). */
void q4k_quant_block(const float *x, uint8_t *blk);

/* GGUF rows -> Hadamard-rotated and requantised packed image.
 * rot = 0: identical to q4k_pack_image; rot = 128: every 128-weight chunk is
 * multiplied by R (lfm_rot_fwd_f32) before the Q4_K quantiser, so the host
 * must rotate the activation with the same R (lfm_rot_blocks_f32 / ROTQ
 * a0=128): <R w, R x> = <w, x> because R^T R = I. */
int q4k_pack_image_rotated(const uint8_t *src, int rows, int K, int rot,
                           uint8_t *img);

/* fill the view (img must be the one q4k_pack_image filled) */
void q4k_mat_init(q4k_mat *m, const uint8_t *img, int N, int K);

/* per-token, row-independent part of the kernel (the OpenCL kernel keeps the
 * same quantities in registers): xs[b*8+g] and xs*sum(xq) per group. */
typedef struct {
    float *xs;
    float *xsum;
    int blocks;
} q4k_plan;

/* reference dequant of one packed block, 256 f32, exactly infra-quant's
 * formula w = d*sc*q - dmin*m in f32. */
void q4k_dequant_block(const uint8_t *hdr, const uint8_t *plane, float *out);

/* ---- activations ------------------------------------------------------- */
/* int8/32 symmetric quantisation, ties-to-even, no clipping (reference). */
void q4k_act_q8(const float *x, int K, int8_t *xq, float *xs);
/* sum of each 32-group of an int8 activation row (int32, exact). */
void q4k_act_sums(const int8_t *xq, int K, int32_t *sums);

/* ---- plan -------------------------------------------------------------- */
/* c[]/min_total from a pre-quantised activation row; c is malloc'ed.
 * sums may be NULL (computed internally). */
int q4k_plan_init(q4k_plan *p, const q4k_mat *m, const int8_t *xq,
                  const float *xs, const int32_t *sums);
void q4k_plan_free(q4k_plan *p);

/* ---- GEMV -------------------------------------------------------------- */
/* portable reference of the exact formula (the DSP scalar twin), rows [r0,r1) */
void q4k_gemv_scalar(const q4k_mat *m, const q4k_plan *p, const int8_t *xq,
                     float *y, int r0, int r1);

/* Q4_K weights, f32 activations (no activation quantisation): the "dequant
 * every weight to f32" oracle.  Slow; double accumulation. */
void q4k_gemv_f32act(const q4k_mat *m, const float *x, float *y, int r0, int r1);

#ifdef __HVX__
/* HVX kernel, same arguments as q4k_gemv_scalar. */
void q4k_gemv_hvx(const q4k_mat *m, const q4k_plan *p, const int8_t *xq,
                  float *y, int r0, int r1);
#endif

/* ---- helpers ----------------------------------------------------------- */
static inline float q4k_f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15) & 1u, e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu, f;
    if (e == 0u) {
        if (m == 0u) {
            f = s << 31;
        } else {
            uint32_t ex = 127u - 15u + 1u;
            while (!(m & 0x400u)) { m <<= 1; ex--; }
            m &= 0x3ffu;
            f = (s << 31) | (ex << 23) | (m << 13);
        }
    } else if (e == 31u) {
        f = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        f = (s << 31) | ((e - 15u + 127u) << 23) | (m << 13);
    }
    {
        float out;
        memcpy(&out, &f, 4);
        return out;
    }
}

/* f32 -> f16 (round to nearest even, as half::f16::from_f32). */
static inline uint16_t q4k_f32_to_f16(float x)
{
    uint32_t u, sign, a, exp, mant, he, m, rem;
    memcpy(&u, &x, 4);
    sign = (u >> 16) & 0x8000u;
    a = u & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return (uint16_t)(sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(sign | 0x7c00u);
    if (a < 0x33000000u) return (uint16_t)sign;
    if (a < 0x38800000u) {
        uint32_t shift;
        exp = a >> 23;
        mant = (a & 0x7fffffu) | 0x800000u;
        shift = 126u - exp;
        m = mant >> shift;
        rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1u)) ||
            (rem == (1u << (shift - 1u)) && (m & 1u)))
            m++;
        return (uint16_t)(sign | m);
    }
    exp = a >> 23;
    mant = a & 0x7fffffu;
    he = exp - 112u;
    m = mant >> 13;
    rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) { m = 0; he++; }
    }
    if (he >= 0x1fu) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (he << 10) | m);
}

/* 6-bit (sc,m) unpack, identical to prepack.rs:165-177 / infra-quant:30-38 */
static inline void q4k_scale_min(const uint8_t *q12, int j, uint8_t *sc, uint8_t *mm)
{
    int jm = j & 3, h = j >> 2;
    uint8_t a = q12[jm], b = q12[jm + 4], c = q12[jm + 8];
    uint8_t lo_d = a & 63u, lo_m = b & 63u;
    uint8_t hi_d = (uint8_t)((c & 15u) | ((a >> 6) << 4));
    uint8_t hi_m = (uint8_t)((c >> 4) | ((b >> 6) << 4));
    *sc = h ? hi_d : lo_d;
    *mm = h ? hi_m : lo_m;
}

/* weight K index within a 256 block -> nibble from the plane byte */
#define Q4K_NIB(plane, i) ((i) < 128 ? ((plane)[i] & 15) : ((plane)[(i)-128] >> 4))

#endif /* Q4K_IMPL_H */
