/*
 * kvarn page codec on the DSP: scalar reference implementation.
 *
 * Port of `myelin/crates/myelin-pack/src/kvarn.rs`.  Field order, quantiser
 * (`roundf` = half away from zero), clamps and the three scale iterations are
 * the same; `kvarn_main.c` checks the result against a page produced by the
 * Rust codec.
 */
#include "kvarn_impl.h"

#include <math.h>
#include <string.h>

#define KV_ROT_SCALE 0.088388346f
#define KV_VAR_FLOOR 1e-8f
#define KV_LOG_MIN (-0.3f)
#define KV_LOG_MAX (10.0f)
#define KV_ITERS 3

static const uint32_t kv_rot_sign[4] = {
    0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu
};

static float kv_sign(int d)
{
    return ((kv_rot_sign[d >> 5] >> (d & 31)) & 1u) ? -1.0f : 1.0f;
}

static uint16_t kv_get_u16(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static void kv_put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8);
}

size_t kvarn_page_bytes(int v_bits)
{
    return (size_t)KVARN_KCODE_BYTES + KVARN_KA_BYTES + KVARN_KC_BYTES +
           (size_t)KVARN_TOKENS * KVARN_DIM * (size_t)v_bits / 8 +
           KVARN_VB_BYTES + KVARN_VBZ_BYTES + KVARN_VC_BYTES;
}

/* ------------------------------------------------------------------ */
/* f16                                                                 */
/* ------------------------------------------------------------------ */

uint16_t kvarn_f32_to_f16(float x)
{
    uint32_t u, sign, abs, exp, mant, half_exp, m, rem;

    memcpy(&u, &x, 4);
    sign = (u >> 16) & 0x8000u;
    abs = u & 0x7fffffffu;

    if (abs >= 0x7f800000u)
        return (uint16_t)(sign | 0x7c00u | (abs > 0x7f800000u ? 0x0200u : 0u));
    if (abs >= 0x47800000u)
        return (uint16_t)(sign | 0x7c00u);
    if (abs < 0x33000000u)
        return (uint16_t)sign;
    if (abs < 0x38800000u) {
        uint32_t shift;
        exp = abs >> 23;
        mant = (abs & 0x7fffffu) | 0x800000u;
        shift = 126u - exp;
        m = mant >> shift;
        rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1u)) ||
            (rem == (1u << (shift - 1u)) && (m & 1u)))
            m++;
        return (uint16_t)(sign | m);
    }

    exp = abs >> 23;
    mant = abs & 0x7fffffu;
    half_exp = exp - 112u;
    m = mant >> 13;
    rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) {
            m = 0;
            half_exp++;
        }
    }
    if (half_exp >= 0x1fu)
        return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (half_exp << 10) | m);
}

float kvarn_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t u;
    float f;

    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int shift = 0;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3ffu;
            u = sign | ((uint32_t)(113 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&f, &u, 4);
    return f;
}

/* ------------------------------------------------------------------ */
/* rotation                                                            */
/* ------------------------------------------------------------------ */

static void kv_hadamard(float *v)
{
    int step, i, j;
    for (step = 1; step < KVARN_DIM; step *= 2) {
        for (i = 0; i < KVARN_DIM; i += 2 * step) {
            for (j = i; j < i + step; j++) {
                float a = v[j];
                float b = v[j + step];
                v[j] = a + b;
                v[j + step] = a - b;
            }
        }
    }
}

void kvarn_rot_fwd(float *v)
{
    int i;
    for (i = 0; i < KVARN_DIM; i++)
        v[i] *= kv_sign(i);
    kv_hadamard(v);
    for (i = 0; i < KVARN_DIM; i++)
        v[i] *= KV_ROT_SCALE;
}

void kvarn_rot_inv(float *v)
{
    int i;
    kv_hadamard(v);
    for (i = 0; i < KVARN_DIM; i++)
        v[i] *= KV_ROT_SCALE * kv_sign(i);
}

void kvarn_rot_fwd_f16(uint16_t *v)
{
    float buf[KVARN_DIM];
    int i;
    for (i = 0; i < KVARN_DIM; i++)
        buf[i] = kvarn_f16_to_f32(v[i]);
    kvarn_rot_fwd(buf);
    for (i = 0; i < KVARN_DIM; i++)
        v[i] = kvarn_f32_to_f16(buf[i]);
}

void kvarn_rot_inv_f16(uint16_t *v)
{
    float buf[KVARN_DIM];
    int i;
    for (i = 0; i < KVARN_DIM; i++)
        buf[i] = kvarn_f16_to_f32(v[i]);
    kvarn_rot_inv(buf);
    for (i = 0; i < KVARN_DIM; i++)
        v[i] = kvarn_f32_to_f16(buf[i]);
}

/* ------------------------------------------------------------------ */
/* encode                                                              */
/* ------------------------------------------------------------------ */

static float kv_t[KVARN_TOKENS * KVARN_DIM];
static float kv_lr[KVARN_DIM];
static float kv_lc[KVARN_TOKENS];
static float kv_ir[KVARN_DIM];
static float kv_ic[KVARN_TOKENS];
static float kv_amp[KVARN_DIM];

static void kv_scales(void)
{
    int i;
    for (i = 0; i < KVARN_DIM; i++)
        kv_ir[i] = expf(-kv_lr[i]);
    for (i = 0; i < KVARN_TOKENS; i++)
        kv_ic[i] = expf(-kv_lc[i]);
}

static float kv_token_var(int token, float ct)
{
    const float *row = kv_t + (size_t)token * KVARN_DIM;
    float sum = 0.0f;
    int ch;
    for (ch = 0; ch < KVARN_DIM; ch++) {
        float x = row[ch] * ct * kv_ir[ch];
        sum += x * x;
    }
    return sum / (float)KVARN_DIM;
}

static float kv_channel_var(int ch, float r)
{
    float sum = 0.0f;
    int token;
    for (token = 0; token < KVARN_TOKENS; token++) {
        float x = kv_t[(size_t)token * KVARN_DIM + ch] * kv_ic[token] * r;
        sum += x * x;
    }
    return sum / (float)KVARN_TOKENS;
}

static float kv_clampf(float x, float lo, float hi)
{
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}

static int kv_quant_k(float x)
{
    int q = (int)roundf(x);
    if (q < -7)
        q = -7;
    if (q > 7)
        q = 7;
    return q;
}

static int kv_quant_v(float x, int max_level)
{
    int q = (int)roundf(x);
    if (q < 0)
        q = 0;
    if (q > max_level)
        q = max_level;
    return q;
}

int kvarn_commit(const uint16_t *rot, int v_bits, uint8_t *out)
{
    uint8_t *kcode = out;
    uint8_t *ka = out + KVARN_KCODE_BYTES;
    uint8_t *kc = ka + KVARN_KA_BYTES;
    uint8_t *vcode = kc + KVARN_KC_BYTES;
    size_t vcode_len = (size_t)KVARN_TOKENS * KVARN_DIM * (size_t)v_bits / 8;
    uint8_t *vb = vcode + vcode_len;
    uint8_t *vbz = vb + KVARN_VB_BYTES;
    uint8_t *vc = vbz + KVARN_VBZ_BYTES;
    int max_level = (1 << v_bits) - 1;
    int token, ch, iter;

    if ((v_bits != 2 && v_bits != 4) || !rot || !out)
        return -1;

    for (token = 0; token < KVARN_TOKENS; token++) {
        for (ch = 0; ch < KVARN_DIM; ch++)
            kv_t[(size_t)token * KVARN_DIM + ch] =
                kvarn_f16_to_f32(rot[(size_t)token * KVARN_DIM + ch]);
    }
    for (ch = 0; ch < KVARN_DIM; ch++)
        kv_lr[ch] = 0.0f;
    for (token = 0; token < KVARN_TOKENS; token++)
        kv_lc[token] = 0.0f;

    for (iter = 0; iter < KV_ITERS; iter++) {
        float v;
        kv_scales();
        for (token = 0; token < KVARN_TOKENS; token++) {
            v = kv_token_var(token, kv_ic[token]);
            if (v < KV_VAR_FLOOR)
                v = KV_VAR_FLOOR;
            kv_lc[token] = kv_clampf(kv_lc[token] + 0.5f * logf(v),
                                     KV_LOG_MIN, KV_LOG_MAX);
        }
        kv_scales();
        for (ch = 0; ch < KVARN_DIM; ch++) {
            v = kv_channel_var(ch, kv_ir[ch]);
            if (v < KV_VAR_FLOOR)
                v = KV_VAR_FLOOR;
            kv_lr[ch] = kv_clampf(kv_lr[ch] + 0.5f * logf(v),
                                  KV_LOG_MIN, KV_LOG_MAX);
        }
    }
    kv_scales();

    /* K: symmetric int4, one amplitude per channel */
    for (ch = 0; ch < KVARN_DIM; ch++) {
        float mx = KV_VAR_FLOOR;
        for (token = 0; token < KVARN_TOKENS; token++) {
            float x = kv_t[(size_t)token * KVARN_DIM + ch] *
                      kv_ic[token] * kv_ir[ch];
            float a = fabsf(x);
            if (a > mx)
                mx = a;
        }
        kv_amp[ch] = mx / 7.0f;
    }
    for (ch = 0; ch < KVARN_DIM; ch++)
        kv_put_u16(ka + 2 * ch,
                   kvarn_f32_to_f16(kv_amp[ch] * expf(kv_lr[ch])));
    for (token = 0; token < KVARN_TOKENS; token++)
        kv_put_u16(kc + 2 * token,
                   kvarn_f32_to_f16(expf(kv_lc[token])));

    for (token = 0; token < KVARN_TOKENS; token++) {
        float ct = kv_ic[token];
        for (ch = 0; ch < KVARN_DIM / 2; ch++) {
            int c0 = 2 * ch;
            int c1 = c0 + 1;
            int q0 = kv_quant_k(kv_t[(size_t)token * KVARN_DIM + c0] * ct *
                                kv_ir[c0] / kv_amp[c0]);
            int q1 = kv_quant_k(kv_t[(size_t)token * KVARN_DIM + c1] * ct *
                                kv_ir[c1] / kv_amp[c1]);
            kcode[token * (KVARN_DIM / 2) + ch] =
                (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
        }
    }

    /* V: unsigned, per-token range, k4v4 also gets a zero point */
    for (token = 0; token < KVARN_TOKENS; token++) {
        float ct = kv_ic[token];
        float mn = 1e30f, mx = -1e30f, step;
        for (ch = 0; ch < KVARN_DIM; ch++) {
            float x = kv_t[(size_t)token * KVARN_DIM + ch] * ct * kv_ir[ch];
            if (x < mn)
                mn = x;
            if (x > mx)
                mx = x;
        }
        step = (mx - mn) / (float)max_level;
        if (!(step >= KV_VAR_FLOOR))
            step = KV_VAR_FLOOR;
        kv_put_u16(vb + 2 * token,
                   kvarn_f32_to_f16(step * expf(kv_lc[token])));
        kv_put_u16(vbz + 2 * token,
                   kvarn_f32_to_f16(mn * expf(kv_lc[token])));

        if (v_bits == 4) {
            for (ch = 0; ch < KVARN_DIM / 2; ch++) {
                int c0 = 2 * ch;
                int c1 = c0 + 1;
                int q0 = kv_quant_v(
                    (kv_t[(size_t)token * KVARN_DIM + c0] * ct * kv_ir[c0] - mn) /
                        step, max_level);
                int q1 = kv_quant_v(
                    (kv_t[(size_t)token * KVARN_DIM + c1] * ct * kv_ir[c1] - mn) /
                        step, max_level);
                vcode[token * (KVARN_DIM / 2) + ch] =
                    (uint8_t)(q0 | (q1 << 4));
            }
        } else {
            for (ch = 0; ch < KVARN_DIM / 4; ch++) {
                int value = 0;
                int k;
                for (k = 0; k < 4; k++) {
                    int c = ch * 4 + k;
                    int q = kv_quant_v(
                        (kv_t[(size_t)token * KVARN_DIM + c] * ct * kv_ir[c] - mn) /
                            step, max_level);
                    value |= q << (2 * k);
                }
                vcode[token * (KVARN_DIM / 4) + ch] = (uint8_t)value;
            }
        }
    }

    for (ch = 0; ch < KVARN_DIM; ch++)
        kv_put_u16(vc + 2 * ch, kvarn_f32_to_f16(expf(kv_lr[ch])));

    return 0;
}

/* ------------------------------------------------------------------ */
/* decode                                                              */
/* ------------------------------------------------------------------ */

void kvarn_decode_k(const uint8_t *page, int token, float *out)
{
    const uint8_t *kcode = page;
    const uint8_t *ka = page + KVARN_KCODE_BYTES;
    const uint8_t *kc = ka + KVARN_KA_BYTES;
    float ct = kvarn_f16_to_f32(kv_get_u16(kc + 2 * token));
    int ch;

    for (ch = 0; ch < KVARN_DIM; ch++) {
        uint8_t byte = kcode[token * (KVARN_DIM / 2) + ch / 2];
        int q = (ch & 1) ? (byte >> 4) : (byte & 15);
        out[ch] = (float)(q - 8) * kvarn_f16_to_f32(kv_get_u16(ka + 2 * ch)) * ct;
    }
}

void kvarn_decode_v(const uint8_t *page, int v_bits, int token, float *out)
{
    const uint8_t *vcode = page + KVARN_KCODE_BYTES + KVARN_KA_BYTES +
                           KVARN_KC_BYTES;
    size_t vcode_len = (size_t)KVARN_TOKENS * KVARN_DIM * (size_t)v_bits / 8;
    const uint8_t *vb = vcode + vcode_len;
    const uint8_t *vbz = vb + KVARN_VB_BYTES;
    const uint8_t *vc = vbz + KVARN_VBZ_BYTES;
    float bt = kvarn_f16_to_f32(kv_get_u16(vb + 2 * token));
    float zero = kvarn_f16_to_f32(kv_get_u16(vbz + 2 * token));
    size_t stride = (size_t)KVARN_DIM * (size_t)v_bits / 8;
    int ch;

    for (ch = 0; ch < KVARN_DIM; ch++) {
        float q;
        if (v_bits == 4) {
            uint8_t byte = vcode[(size_t)token * stride + ch / 2];
            q = (float)((ch & 1) ? (byte >> 4) : (byte & 15));
        } else {
            uint8_t byte = vcode[(size_t)token * stride + ch / 4];
            q = (float)((byte >> (2 * (ch & 3))) & 3);
        }
        out[ch] = (q * bt + zero) * kvarn_f16_to_f32(kv_get_u16(vc + 2 * ch));
    }
}
