#ifndef LFM_F32_KV
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void bw_read(__global const uint4 *src, __global float *sink, uint n4) {
    size_t gid = get_global_id(0);
    size_t stride = get_global_size(0);
    uint acc = 0u;
    for (size_t i = gid; i < n4; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) {
        sink[0] = 1.0f;
    }
}

__kernel void bw_copy(__global const uint4 *src, __global uint4 *dst, uint n4) {
    size_t i = get_global_id(0);
    if (i < n4) {
        dst[i] = src[i];
    }
}

__kernel void peak_fp32(__global float *sink, uint iters) {
    float b = 0.999999f;
    float c = 0.5f;
    float a0 = (float)get_global_id(0) * 1e-7f + 1.0f;
    float a1 = a0 + 0.1f;
    float a2 = a0 + 0.2f;
    float a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f;
    float a5 = a0 + 0.5f;
    float a6 = a0 + 0.6f;
    float a7 = a0 + 0.7f;
    for (uint i = 0u; i < iters; ++i) {
        a0 = a0 * b + c;
        a1 = a1 * b + c;
        a2 = a2 * b + c;
        a3 = a3 * b + c;
        a4 = a4 * b + c;
        a5 = a5 * b + c;
        a6 = a6 * b + c;
        a7 = a7 * b + c;
    }
    sink[get_global_id(0)] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

#ifndef LFM_F32_KV
__kernel void peak_fp16(__global float *sink, uint iters) {
    half2 b = (half2)((half)0.9995f, (half)0.9995f);
    half2 c = (half2)((half)0.5f, (half)0.5f);
    half2 a0 = (half2)((half)1.0f, (half)1.1f);
    half2 a1 = (half2)((half)1.2f, (half)1.3f);
    half2 a2 = (half2)((half)1.4f, (half)1.5f);
    half2 a3 = (half2)((half)1.6f, (half)1.7f);
    half2 a4 = (half2)((half)1.8f, (half)1.9f);
    half2 a5 = (half2)((half)2.0f, (half)2.1f);
    half2 a6 = (half2)((half)2.2f, (half)2.3f);
    half2 a7 = (half2)((half)2.4f, (half)2.5f);
    for (uint i = 0u; i < iters; ++i) {
        a0 = a0 * b + c;
        a1 = a1 * b + c;
        a2 = a2 * b + c;
        a3 = a3 * b + c;
        a4 = a4 * b + c;
        a5 = a5 * b + c;
        a6 = a6 * b + c;
        a7 = a7 * b + c;
    }
    sink[get_global_id(0)] = (float)(a0.s0 + a0.s1 + a1.s0 + a1.s1 + a2.s0 + a2.s1 + a3.s0 + a3.s1
        + a4.s0 + a4.s1 + a5.s0 + a5.s1 + a6.s0 + a6.s1 + a7.s0 + a7.s1);
}

#endif

__kernel void peak_i8dot(__global uint *sink, uint iters) {
    uchar4 a = (uchar4)(1u, 2u, 3u, 4u);
    uchar4 b0 = (uchar4)(2u, 1u, 2u, 1u);
    uchar4 b1 = (uchar4)(3u, 1u, 4u, 1u);
    uchar4 b2 = (uchar4)(5u, 9u, 2u, 6u);
    uchar4 b3 = (uchar4)(7u, 1u, 8u, 1u);
    uchar4 b4 = (uchar4)(1u, 3u, 1u, 5u);
    uchar4 b5 = (uchar4)(2u, 7u, 1u, 9u);
    uchar4 b6 = (uchar4)(4u, 1u, 6u, 1u);
    uchar4 b7 = (uchar4)(8u, 5u, 4u, 3u);
    uint acc0 = 0u;
    uint acc1 = 0u;
    uint acc2 = 0u;
    uint acc3 = 0u;
    uint acc4 = 0u;
    uint acc5 = 0u;
    uint acc6 = 0u;
    uint acc7 = 0u;
    for (uint i = 0u; i < iters; ++i) {
        a.s0 = (uchar)(i & 7u);
        acc0 += dot(a, b0);
        acc1 += dot(a, b1);
        acc2 += dot(a, b2);
        acc3 += dot(a, b3);
        acc4 += dot(a, b4);
        acc5 += dot(a, b5);
        acc6 += dot(a, b6);
        acc7 += dot(a, b7);
    }
    sink[get_global_id(0)] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
}

__kernel void nop(__global uint *sink) {
    sink[get_global_id(0)] = 1u;
}

__kernel void peak_mixed(__global float *sinkf, __global uint *sinku, uint iters, uint ratio) {
    uint g = get_group_id(0);
    uint ng = get_num_groups(0);
    uint use_i8 = (g * 100u) / ng < ratio;
    if (use_i8 != 0u) {
        uchar4 a = (uchar4)(1u, 2u, 3u, 4u);
        uchar4 b0 = (uchar4)(2u, 1u, 2u, 1u);
        uchar4 b1 = (uchar4)(3u, 1u, 4u, 1u);
        uchar4 b2 = (uchar4)(5u, 9u, 2u, 6u);
        uchar4 b3 = (uchar4)(7u, 1u, 8u, 1u);
        uchar4 b4 = (uchar4)(1u, 3u, 1u, 5u);
        uchar4 b5 = (uchar4)(2u, 7u, 1u, 9u);
        uchar4 b6 = (uchar4)(4u, 1u, 6u, 1u);
        uchar4 b7 = (uchar4)(8u, 5u, 4u, 3u);
        uint acc0 = 0u;
        uint acc1 = 0u;
        uint acc2 = 0u;
        uint acc3 = 0u;
        uint acc4 = 0u;
        uint acc5 = 0u;
        uint acc6 = 0u;
        uint acc7 = 0u;
        for (uint i = 0u; i < iters; ++i) {
            a.s0 = (uchar)(i & 7u);
            acc0 += dot(a, b0);
            acc1 += dot(a, b1);
            acc2 += dot(a, b2);
            acc3 += dot(a, b3);
            acc4 += dot(a, b4);
            acc5 += dot(a, b5);
            acc6 += dot(a, b6);
            acc7 += dot(a, b7);
        }
        sinku[get_global_id(0)] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
    }
#ifndef LFM_F32_KV
    else {
        half2 b = (half2)((half)0.9995f, (half)0.9995f);
        half2 c = (half2)((half)0.5f, (half)0.5f);
        half2 a0 = (half2)((half)1.0f, (half)1.1f);
        half2 a1 = (half2)((half)1.2f, (half)1.3f);
        half2 a2 = (half2)((half)1.4f, (half)1.5f);
        half2 a3 = (half2)((half)1.6f, (half)1.7f);
        half2 a4 = (half2)((half)1.8f, (half)1.9f);
        half2 a5 = (half2)((half)2.0f, (half)2.1f);
        half2 a6 = (half2)((half)2.2f, (half)2.3f);
        half2 a7 = (half2)((half)2.4f, (half)2.5f);
        for (uint i = 0u; i < iters; ++i) {
            a0 = a0 * b + c;
            a1 = a1 * b + c;
            a2 = a2 * b + c;
            a3 = a3 * b + c;
            a4 = a4 * b + c;
            a5 = a5 * b + c;
            a6 = a6 * b + c;
            a7 = a7 * b + c;
        }
        sinkf[get_global_id(0)] = (float)(a0.s0 + a0.s1 + a1.s0 + a1.s1 + a2.s0 + a2.s1 + a3.s0 + a3.s1
            + a4.s0 + a4.s1 + a5.s0 + a5.s1 + a6.s0 + a6.s1 + a7.s0 + a7.s1);
    }
#else
    else {
        float b = 0.9995f;
        float c = 0.5f;
        float a0 = 1.0f + (float)get_global_id(0) * 1e-7f;
        float a1 = a0 + 0.1f;
        float a2 = a0 + 0.2f;
        float a3 = a0 + 0.3f;
        float a4 = a0 + 0.4f;
        float a5 = a0 + 0.5f;
        float a6 = a0 + 0.6f;
        float a7 = a0 + 0.7f;
        for (uint i = 0u; i < iters; ++i) {
            a0 = a0 * b + c;
            a1 = a1 * b + c;
            a2 = a2 * b + c;
            a3 = a3 * b + c;
            a4 = a4 * b + c;
            a5 = a5 * b + c;
            a6 = a6 * b + c;
            a7 = a7 * b + c;
        }
        sinkf[get_global_id(0)] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
#endif
}
