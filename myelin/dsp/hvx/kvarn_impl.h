#ifndef LFM_KVARN_IMPL_H
#define LFM_KVARN_IMPL_H

#include <stddef.h>
#include <stdint.h>

/*
 * kvarn page codec for the HVX path: K 4 bit, V 2 or 4 bit, per-channel and
 * per-token scales, Hadamard rotation.  1:1 port of
 * `myelin/crates/myelin-pack/src/kvarn.rs` (which is checked bit for bit
 * against `crates/lfm-npu-ref/src/kvarn.rs`), so the DSP encoder can be
 * compared with the reference byte for byte.
 *
 * Page of one KV head, 32 tokens x 128 dims, little endian:
 *
 *   kcode 2048 B   nibbles (q + 8) | ((q + 8) << 4), low nibble = even dim
 *   ka     256 B   128 f16 per-channel K scale
 *   kc      64 B    32 f16 per-token   K scale
 *   vcode 1024 B (v2) / 2048 B (v4)
 *   vb      64 B    32 f16 per-token V step
 *   vbz     64 B    32 f16 per-token V zero point
 *   vc     256 B   128 f16 per-channel V scale
 *
 * k4v2 = 3776 B, k4v4 = 4800 B per head.
 */

#define KVARN_TOKENS 32
#define KVARN_DIM 128

#define KVARN_KCODE_BYTES (KVARN_TOKENS * KVARN_DIM / 2)
#define KVARN_KA_BYTES (KVARN_DIM * 2)
#define KVARN_KC_BYTES (KVARN_TOKENS * 2)
#define KVARN_VB_BYTES (KVARN_TOKENS * 2)
#define KVARN_VBZ_BYTES (KVARN_TOKENS * 2)
#define KVARN_VC_BYTES (KVARN_DIM * 2)

#define KVARN_K4V2_BYTES (KVARN_KCODE_BYTES + KVARN_KA_BYTES + KVARN_KC_BYTES + \
                          KVARN_TOKENS * KVARN_DIM * 2 / 8 + KVARN_VB_BYTES +    \
                          KVARN_VBZ_BYTES + KVARN_VC_BYTES)
#define KVARN_K4V4_BYTES (KVARN_KCODE_BYTES + KVARN_KA_BYTES + KVARN_KC_BYTES + \
                          KVARN_TOKENS * KVARN_DIM * 4 / 8 + KVARN_VB_BYTES +    \
                          KVARN_VBZ_BYTES + KVARN_VC_BYTES)

/* Byte sizes of one encoded page for `v_bits` (2 or 4). */
size_t kvarn_page_bytes(int v_bits);

/* f16 helpers (round to nearest even, as `half::f16::from_f32`). */
uint16_t kvarn_f32_to_f16(float x);
float kvarn_f16_to_f32(uint16_t h);

/* Hadamard rotation, 128 values, in place. */
void kvarn_rot_fwd(float *v);
void kvarn_rot_inv(float *v);
void kvarn_rot_fwd_f16(uint16_t *v);
void kvarn_rot_inv_f16(uint16_t *v);

/*
 * Encode one page of already rotated f16 values (`rot[token*128 + dim]`,
 * f16 bits) into `out`, v_bits = 2 or 4.  Returns 0 on success.
 * Not reentrant (uses static scratch).
 */
int kvarn_commit(const uint16_t *rot, int v_bits, uint8_t *out);

/* Reconstruct one token of K or V in the rotated basis. */
void kvarn_decode_k(const uint8_t *page, int token, float *out);
void kvarn_decode_v(const uint8_t *page, int v_bits, int token, float *out);

#endif /* LFM_KVARN_IMPL_H */
