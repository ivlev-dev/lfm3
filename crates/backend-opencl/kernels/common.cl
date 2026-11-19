#ifndef LFM_F32_KV
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#ifndef LFM_AMD
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#if defined(__AMDGCN__)
static inline int lfm_dot4(uint a, uint b, int c) {
    return (int)__builtin_amdgcn_sdot4((int)a, (int)b, c, 0);
}
static inline int lfm_sum4(uint a, int c) {
    return (int)__builtin_amdgcn_sdot4((int)a, 0x01010101, c, 0);
}
#else
static inline int lfm_dot4(uint a, uint b, int c) {
    return c + (int)dot(as_char4(a), as_char4(b));
}
static inline int lfm_sum4(uint a, int c) {
    return c + (int)dot(as_char4(a), (char4)(1));
}
#endif

#define LFM_DOT(a, b) lfm_dot4(as_uint((char4)(a)), as_uint((char4)(b)), 0)

// На устройстве без cl_khr_fp16 (Intel Gen9 через Rusticl) половины нет как типа.
// Тогда заголовки блоков (d/dmin) читаются из байтов и конвертируются вручную,
// а KV-кэш хранится во float.
static inline float lfm_h2f(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) {
            return as_float(sign);
        }
        float v = (float)m * 5.9604644775390625e-8f;
        return sign != 0u ? -v : v;
    }
    if (e == 31u) {
        return as_float(sign | 0x7F800000u | (m << 13));
    }
    return as_float(sign | ((e + 112u) << 23) | (m << 13));
}

#ifdef LFM_F32_KV
static inline float lfm_scale(__global const ushort *p) {
    uint bits = (uint)p[0];
    return lfm_h2f(bits);
}
#define KV_T float
#define KV_V16 float16
#define kv_load2(l, p) vload2((l), (__global const float *)(p))
#define kv_load16(l, p) vload16((l), (__global const float *)(p))
#define kv_load8(l, p) vload8((l), (__global const float *)(p))
#define kv_load4(l, p) vload4((l), (__global const float *)(p))
static inline float lfm_scale_local(__local const uchar *p) {
    uint bits = (uint)p[0] | ((uint)p[1] << 8);
    return lfm_h2f(bits);
}
#else
static inline float lfm_scale(__global const ushort *p) {
    return (float)vload_half(0, (__global const half *)p);
}
#define KV_T half
#define KV_V16 half16
#define kv_load2(l, p) vload_half2((l), (__global const half *)(p))
#define kv_load16(l, p) vload16((l), (__global const half *)(p))
#define kv_load8(l, p) vload8((l), (__global const half *)(p))
#define kv_load4(l, p) vload4((l), (__global const half *)(p))
static inline float lfm_scale_local(__local const uchar *p) {
    return (float)vload_half(0, (__local const half *)p);
}
#endif
