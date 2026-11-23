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
