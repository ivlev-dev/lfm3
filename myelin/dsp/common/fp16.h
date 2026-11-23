#ifndef LFM_FP16_H
#define LFM_FP16_H

#include <stdint.h>
#include <string.h>

#define LFM_F16_ZERO 0x0000u
#define LFM_F16_ONE  0x3C00u

static inline float lfm_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFu;
            x = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    {
        float f;
        memcpy(&f, &x, sizeof f);
        return f;
    }
}

static inline uint16_t lfm_f32_to_f16(float f)
{
    uint32_t x;
    uint32_t sign;
    uint32_t absx;
    uint32_t exp;
    uint32_t mant;
    uint32_t value;
    uint32_t rem;
    uint32_t halfway;
    int shift;
    memcpy(&x, &f, sizeof x);
    sign = (x >> 16) & 0x8000u;
    absx = x & 0x7FFFFFFFu;
    if (absx >= 0x47800000u) {
        if (absx > 0x7F800000u)
            return (uint16_t)(sign | 0x7E00u);
        return (uint16_t)(sign | 0x7C00u);
    }
    if (absx < 0x38800000u) {
        if (absx < 0x33000000u)
            return (uint16_t)sign;
        exp = absx >> 23;
        mant = (absx & 0x7FFFFFu) | 0x800000u;
        shift = 126 - (int)exp;
        value = mant >> shift;
        rem = mant & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (value & 1u)))
            value++;
        return (uint16_t)(sign | value);
    }
    exp = (absx >> 23) - 112u;
    mant = absx & 0x7FFFFFu;
    value = (exp << 10) | (mant >> 13);
    rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (value & 1u)))
        value++;
    return (uint16_t)(sign | value);
}

static inline void lfm_f16_store(uint8_t *dst, uint16_t h)
{
    dst[0] = (uint8_t)(h & 0xFFu);
    dst[1] = (uint8_t)(h >> 8);
}

static inline uint16_t lfm_f16_load(const uint8_t *src)
{
    return (uint16_t)(src[0] | ((uint16_t)src[1] << 8));
}

#endif
