#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
