#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant float LFM_TQ_CB[16] = {
    -0.237629839f, -0.180795565f, -0.141763277f, -0.110247883f,
    -0.082792693f, -0.057744624f, -0.034133885f, -0.011296587f,
     0.011296085f,  0.034133404f,  0.057743630f,  0.082791700f,
     0.110246892f,  0.141762679f,  0.180795257f,  0.237628942f
};

__constant uint LFM_TQ_SIGN[4] = { 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu };

static float lfm_tq_sign(uint d) {
    return ((LFM_TQ_SIGN[d >> 5] >> (d & 31u)) & 1u) ? -1.0f : 1.0f;
}

static void lfm_tq_hadamard(__local float *v, uint lane) {
    #pragma unroll
    for (uint step = 1u; step < 128u; step <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint p = lane;
        uint i = (p / step) * (2u * step) + (p % step);
        float a = v[i];
        float b = v[i + step];
        v[i] = a + b;
        v[i + step] = a - b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static void lfm_tq_rot_fwd(__local float *v, uint lane) {
    v[2u * lane] *= lfm_tq_sign(2u * lane);
    v[2u * lane + 1u] *= lfm_tq_sign(2u * lane + 1u);
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_hadamard(v, lane);
    float s = 0.0883883476f;
    v[2u * lane] *= s;
    v[2u * lane + 1u] *= s;
}

static void lfm_tq_rot_inv(__local float *v, uint lane) {
    lfm_tq_hadamard(v, lane);
    float s = 0.0883883476f;
    v[2u * lane] *= s * lfm_tq_sign(2u * lane);
    v[2u * lane + 1u] *= s * lfm_tq_sign(2u * lane + 1u);
    barrier(CLK_LOCAL_MEM_FENCE);
}

static uint lfm_tq_nearest(float x, __local const float *cb) {
    uint best = 0u;
    float bd = fabs(x - cb[0]);
    #pragma unroll
    for (uint i = 1u; i < 16u; ++i) {
        float d = fabs(x - cb[i]);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

static void lfm_tq_store_k(__global const float *src, __global uchar *code, __global half *scale,
                           __local float *buf, __local float *red, __local const float *cb, uint lane) {
    float a = src[2u * lane];
    float b = src[2u * lane + 1u];
    buf[2u * lane] = a;
    buf[2u * lane + 1u] = b;
    red[lane] = a * a + b * b;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint st = 32u; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(red[0]);
    float inv = norm > 1e-8f ? 1.0f / norm : 0.0f;
    buf[2u * lane] = a * inv;
    buf[2u * lane + 1u] = b * inv;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    uint i0 = lfm_tq_nearest(buf[2u * lane], cb);
    uint i1 = lfm_tq_nearest(buf[2u * lane + 1u], cb);
    code[lane] = (uchar)(i0 | (i1 << 4));
    if (lane == 0u) {
        scale[0] = (half)norm;
    }
}

static void lfm_tq_store_v(__global const float *src, __global uchar *code, __global half *scale,
                           __local float *buf, __local float *red, uint lane) {
    float a = src[2u * lane];
    float b = src[2u * lane + 1u];
    buf[2u * lane] = a;
    buf[2u * lane + 1u] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    float x = buf[2u * lane];
    float y = buf[2u * lane + 1u];
    red[lane] = fmax(fabs(x), fabs(y));
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint st = 32u; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] = fmax(red[lane], red[lane + st]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sc = red[0] / 7.0f;
    if (sc < 1e-8f) {
        sc = 1e-8f;
    }
    int q0 = (int)round(x / sc);
    int q1 = (int)round(y / sc);
    q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
    q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);
    code[lane] = (uchar)((q0 + 7) | ((q1 + 7) << 4));
    if (lane == 0u) {
        scale[0] = (half)sc;
    }
}

__kernel void kv_store_q(__global const float *ksrc, __global const float *vsrc,
                         __global uchar *kc, __global half *kn,
                         __global uchar *vc, __global half *vs,
                         __global const uint *tbl, uint pos, uint kv_heads, uint head_dim) {
    uint lane = get_local_id(0);
    uint h = get_group_id(0);
    if (h >= kv_heads) {
        return;
    }
    __local float buf[128];
    __local float red[64];
    __local float cb[16];
    if (lane < 16u) {
        cb[lane] = LFM_TQ_CB[lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ulong slot = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
    uint base = (uint)slot * kv_heads + h;
    lfm_tq_store_k(ksrc + h * head_dim, kc + base * 64u, kn + base, buf, red, cb, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_store_v(vsrc + h * head_dim, vc + base * 64u, vs + base, buf, red, lane);
}

__kernel void kv_store_b_q(__global const float *ksrc, __global const float *vsrc,
                           __global uchar *kc, __global half *kn,
                           __global uchar *vc, __global half *vs,
                           __global const uint *tbl, __global const uint *tbl_off,
                           __global const uint *pos_b, uint kv_heads, uint head_dim, uint ntok) {
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint t = gid / kv_heads;
    uint h = gid % kv_heads;
    if (t >= ntok) {
        return;
    }
    __local float buf[128];
    __local float red[64];
    __local float cb[16];
    if (lane < 16u) {
        cb[lane] = LFM_TQ_CB[lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint pos = pos_b[t];
    ulong slot = (ulong)((tbl[tbl_off[t] + (pos >> 5)] << 5) | (pos & 31u));
    uint base = (uint)slot * kv_heads + h;
    ulong off = ((ulong)t * kv_heads + h) * head_dim;
    lfm_tq_store_k(ksrc + off, kc + base * 64u, kn + base, buf, red, cb, lane);
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_store_v(vsrc + off, vc + base * 64u, vs + base, buf, red, lane);
}

static float2 lfm_tq_load_k(__global const uchar *kc, __global const half *kn,
                            __local const float *cb, ulong slot, uint kvh, uint kv_heads, uint lane) {
    uint base = (uint)slot * kv_heads + kvh;
    uchar b = kc[base * 64u + lane];
    float n = (float)vload_half(0, kn + base);
    float2 k;
    k.x = cb[b & 15u] * n;
    k.y = cb[b >> 4] * n;
    return k;
}

static float2 lfm_tq_load_v(__global const uchar *vc, __global const half *vs,
                            ulong slot, uint kvh, uint kv_heads, uint lane) {
    uint base = (uint)slot * kv_heads + kvh;
    uchar b = vc[base * 64u + lane];
    float s = (float)vload_half(0, vs + base);
    float2 v;
    v.x = (float)((int)(b & 15u) - 7) * s;
    v.y = (float)((int)(b >> 4) - 7) * s;
    return v;
}

static void lfm_tq_load_q(__global const float *q, __local float *qs, uint lane, uint head) {
    qs[2u * lane] = q[head * 128u + 2u * lane];
    qs[2u * lane + 1u] = q[head * 128u + 2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(qs, lane);
}

__kernel void attention_h_q(__global const float *q, __global const uchar *kc, __global const half *kn,
                            __global const uchar *vc, __global const half *vs,
                            __global const uint *tbl, __global const float *gate,
                            __global float *out, uint len, uint start, uint kv_heads, uint n_heads) {
    uint lane = get_local_id(0);
    uint head = get_group_id(0);
    __local float sred[128];
    __local float qs[128];
    __local float cb[16];
    if (lane < 16u) {
        cb[lane] = LFM_TQ_CB[lane];
    }
    uint kvh = head / (n_heads / kv_heads);
    lfm_tq_load_q(q, qs, lane, head);
    float q0 = qs[2u * lane];
    float q1 = qs[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 0.0883883476f;
    uint pos = start;
    for (; pos + 1u < len; pos += 2u) {
        ulong s0 = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        ulong s1 = (ulong)((tbl[(pos + 1u) >> 5] << 5) | ((pos + 1u) & 31u));
        float2 k0 = lfm_tq_load_k(kc, kn, cb, s0, kvh, kv_heads, lane);
        float2 k1 = lfm_tq_load_k(kc, kn, cb, s1, kvh, kv_heads, lane);
        sred[lane] = q0 * k0.x + q1 * k0.y;
        sred[64u + lane] = q0 * k1.x + q1 * k1.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
            sred[64u + lane] = sred[64u + lane] + sred[64u + lane + 8u] + sred[64u + lane + 16u]
                             + sred[64u + lane + 24u] + sred[64u + lane + 32u] + sred[64u + lane + 40u]
                             + sred[64u + lane + 48u] + sred[64u + lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
            sred[64u] = sred[64u] + sred[65u] + sred[66u] + sred[67u] + sred[68u] + sred[69u] + sred[70u] + sred[71u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        float2 v0 = lfm_tq_load_v(vc, vs, s0, kvh, kv_heads, lane);
        float2 v1 = lfm_tq_load_v(vc, vs, s1, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < len) {
        ulong sp = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        float2 k = lfm_tq_load_k(kc, kn, cb, sp, kvh, kv_heads, lane);
        sred[lane] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sv = sred[0] * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        float2 v = lfm_tq_load_v(vc, vs, sp, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = mn;
    }
    float inv = 1.0f / l;
    qs[2u * lane] = acc0 * inv;
    qs[2u * lane + 1u] = acc1 * inv;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_inv(qs, lane);
    float gs = 1.0f / (1.0f + exp(-gate[head]));
    out[head * 128u + 2u * lane] = qs[2u * lane] * gs;
    out[head * 128u + 2u * lane + 1u] = qs[2u * lane + 1u] * gs;
}

__kernel void attention_split_h_q(__global const float *q, __global const uchar *kc, __global const half *kn,
                                  __global const uchar *vc, __global const half *vs,
                                  __global const uint *tbl, __global float *pacc, __global float *pml,
                                  uint len, uint start, uint kv_heads, uint n_heads, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint head = gid / splits;
    uint sp = gid % splits;
    if (head >= n_heads) {
        return;
    }
    __local float sred[128];
    __local float qs[128];
    __local float cb[16];
    if (lane < 16u) {
        cb[lane] = LFM_TQ_CB[lane];
    }
    uint kvh = head / (n_heads / kv_heads);
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    lfm_tq_load_q(q, qs, lane, head);
    float q0 = qs[2u * lane];
    float q1 = qs[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 0.0883883476f;
    uint pos = p0;
    for (; pos + 1u < p1; pos += 2u) {
        ulong s0 = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        ulong s1 = (ulong)((tbl[(pos + 1u) >> 5] << 5) | ((pos + 1u) & 31u));
        float2 k0 = lfm_tq_load_k(kc, kn, cb, s0, kvh, kv_heads, lane);
        float2 k1 = lfm_tq_load_k(kc, kn, cb, s1, kvh, kv_heads, lane);
        sred[lane] = q0 * k0.x + q1 * k0.y;
        sred[64u + lane] = q0 * k1.x + q1 * k1.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
            sred[64u + lane] = sred[64u + lane] + sred[64u + lane + 8u] + sred[64u + lane + 16u]
                             + sred[64u + lane + 24u] + sred[64u + lane + 32u] + sred[64u + lane + 40u]
                             + sred[64u + lane + 48u] + sred[64u + lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
            sred[64u] = sred[64u] + sred[65u] + sred[66u] + sred[67u] + sred[68u] + sred[69u] + sred[70u] + sred[71u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        float2 v0 = lfm_tq_load_v(vc, vs, s0, kvh, kv_heads, lane);
        float2 v1 = lfm_tq_load_v(vc, vs, s1, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < p1) {
        ulong sp2 = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        float2 k = lfm_tq_load_k(kc, kn, cb, sp2, kvh, kv_heads, lane);
        sred[lane] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sv = sred[0] * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        float2 v = lfm_tq_load_v(vc, vs, sp2, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = mn;
    }
    float inv = 1.0f / l;
    qs[2u * lane] = acc0 * inv;
    qs[2u * lane + 1u] = acc1 * inv;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_inv(qs, lane);
    ulong pbase = ((ulong)head * splits + sp) * 128u;
    pacc[pbase + 2u * lane] = qs[2u * lane];
    pacc[pbase + 2u * lane + 1u] = qs[2u * lane + 1u];
    if (lane == 0u) {
        pml[((ulong)head * splits + sp) * 2u] = m;
        pml[((ulong)head * splits + sp) * 2u + 1u] = l;
    }
}

__kernel void attention_b_h_q(__global const float *q, __global const uchar *kc, __global const half *kn,
                              __global const uchar *vc, __global const half *vs,
                              __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b,
                              __global const float *gate, __global float *out,
                              uint ntok, uint kv_heads, uint n_heads, uint window) {
    uint lid = get_local_id(0);
    uint lane = lid & 63u;
    uint grp = lid >> 6;
    uint hg = n_heads >> 2;
    uint gid = get_group_id(0);
    uint t = gid / hg;
    uint head = (gid % hg) * 4u + grp;
    if (t >= ntok) {
        return;
    }
    __local float sred[256];
    __local float qs[4 * 128];
    __local float cb[16];
    if (lid < 16u) {
        cb[lid] = LFM_TQ_CB[lid];
    }
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    __global const float *qh = q + (ulong)t * n_heads * 128u + head * 128u;
    qs[grp * 128u + 2u * lane] = qh[2u * lane];
    qs[grp * 128u + 2u * lane + 1u] = qh[2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(qs + grp * 128u, lane);
    float q0 = qs[grp * 128u + 2u * lane];
    float q1 = qs[grp * 128u + 2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 0.0883883476f;
    for (uint pos = start; pos < len; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        float2 k = lfm_tq_load_k(kc, kn, cb, slot, kvh, kv_heads, lane);
        sred[lid] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 32u; st > 0u; st >>= 1) {
            if (lane < st) {
                sred[lid] += sred[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float s = sred[grp * 64u] * scale;
        float mn = fmax(m, s);
        float corr = exp(m - mn);
        float wgt = exp(s - mn);
        l = l * corr + wgt;
        float2 v = lfm_tq_load_v(vc, vs, slot, kvh, kv_heads, lane);
        acc0 = acc0 * corr + wgt * v.x;
        acc1 = acc1 * corr + wgt * v.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / l;
    qs[grp * 128u + 2u * lane] = acc0 * inv;
    qs[grp * 128u + 2u * lane + 1u] = acc1 * inv;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_inv(qs + grp * 128u, lane);
    float gs = 1.0f / (1.0f + exp(-gate[t * n_heads + head]));
    __global float *o = out + (ulong)t * n_heads * 128u + head * 128u;
    o[2u * lane] = qs[grp * 128u + 2u * lane] * gs;
    o[2u * lane + 1u] = qs[grp * 128u + 2u * lane + 1u] * gs;
}

__kernel void attention_split_b_h_q(__global const float *q, __global const uchar *kc, __global const half *kn,
                                    __global const uchar *vc, __global const half *vs,
                                    __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b,
                                    __global float *pacc, __global float *pml,
                                    uint ntok, uint kv_heads, uint n_heads, uint window, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint per_tok = n_heads * splits;
    uint t = gid / per_tok;
    uint rem = gid % per_tok;
    uint head = rem / splits;
    uint sp = rem % splits;
    if (t >= ntok) {
        return;
    }
    __local float sred[128];
    __local float qs[128];
    __local float cb[16];
    if (lane < 16u) {
        cb[lane] = LFM_TQ_CB[lane];
    }
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    __global const float *qh = q + (ulong)t * n_heads * 128u + head * 128u;
    qs[2u * lane] = qh[2u * lane];
    qs[2u * lane + 1u] = qh[2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(qs, lane);
    float q0 = qs[2u * lane];
    float q1 = qs[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 0.0883883476f;
    uint pos = p0;
    for (; pos + 1u < p1; pos += 2u) {
        ulong s0 = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        ulong s1 = (ulong)((tbl[toff + ((pos + 1u) >> 5)] << 5) | ((pos + 1u) & 31u));
        float2 k0 = lfm_tq_load_k(kc, kn, cb, s0, kvh, kv_heads, lane);
        float2 k1 = lfm_tq_load_k(kc, kn, cb, s1, kvh, kv_heads, lane);
        sred[lane] = q0 * k0.x + q1 * k0.y;
        sred[64u + lane] = q0 * k1.x + q1 * k1.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
            sred[64u + lane] = sred[64u + lane] + sred[64u + lane + 8u] + sred[64u + lane + 16u]
                             + sred[64u + lane + 24u] + sred[64u + lane + 32u] + sred[64u + lane + 40u]
                             + sred[64u + lane + 48u] + sred[64u + lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
            sred[64u] = sred[64u] + sred[65u] + sred[66u] + sred[67u] + sred[68u] + sred[69u] + sred[70u] + sred[71u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        float2 v0 = lfm_tq_load_v(vc, vs, s0, kvh, kv_heads, lane);
        float2 v1 = lfm_tq_load_v(vc, vs, s1, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < p1) {
        ulong sp2 = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        float2 k = lfm_tq_load_k(kc, kn, cb, sp2, kvh, kv_heads, lane);
        sred[lane] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            sred[lane] = sred[lane] + sred[lane + 8u] + sred[lane + 16u] + sred[lane + 24u]
                       + sred[lane + 32u] + sred[lane + 40u] + sred[lane + 48u] + sred[lane + 56u];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            sred[0] = sred[0] + sred[1] + sred[2] + sred[3] + sred[4] + sred[5] + sred[6] + sred[7];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sv = sred[0] * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        float2 v = lfm_tq_load_v(vc, vs, sp2, kvh, kv_heads, lane);
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = mn;
    }
    float inv = 1.0f / l;
    qs[2u * lane] = acc0 * inv;
    qs[2u * lane + 1u] = acc1 * inv;
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_inv(qs, lane);
    ulong pbase = ((ulong)t * n_heads + head) * splits + sp;
    __global float *pa = pacc + pbase * 128u;
    pa[2u * lane] = qs[2u * lane];
    pa[2u * lane + 1u] = qs[2u * lane + 1u];
    if (lane == 0u) {
        pml[pbase * 2u] = m;
        pml[pbase * 2u + 1u] = l;
    }
}
