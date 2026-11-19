
#ifndef RGEMV
#define RGEMV 5
#endif

__kernel void embed_q6k(__global const uchar *w, __global float *out, uint token, uint cols) {
    uint i = get_global_id(0);
    if (i >= cols) {
        return;
    }
    uint blocks = cols >> 8;
    __global const uchar *base = w + (ulong)token * ((ulong)blocks * 210ul);
    uint b = i >> 8;
    uint r = i & 255u;
    __global const uchar *blk = base + (ulong)b * 210ul;
    float d = lfm_scale((__global const ushort *)(blk + 208));
    __global const uchar *ql = blk;
    __global const uchar *qh = blk + 128;
    __global const char *sc = (__global const char *)(blk + 192);
    uint n = r >> 7;
    uint rr = r & 127u;
    uint l = rr & 31u;
    uint q = rr >> 5;
    uint qlb = (n << 6) + l + ((q & 1u) ? 32u : 0u);
    uint nib = (q >= 2u) ? (ql[qlb] >> 4) : (ql[qlb] & 15u);
    uint qhb = (n << 5) + l;
    int qv = (int)(nib | (((qh[qhb] >> (q << 1)) & 3u) << 4)) - 32;
    float sv = (float)sc[(n << 3) + (l >> 4) + (q << 1)];
    out[i] = d * sv * (float)qv;
}

__kernel void rms_norm(__global const float *x, __global const float *w, __global float *out, uint n) {
    uint lane = get_local_id(0);
    uint nl = get_local_size(0);
    __local float red[64];
    float acc = 0.0f;
    for (uint i = lane; i < n; i += nl) {
        acc += x[i] * x[i];
    }
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = nl >> 1; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = red[0];
    float scale = rsqrt(total / (float)n + 1e-5f);
    for (uint i = lane; i < n; i += nl) {
        out[i] = x[i] * scale * w[i];
    }
}

__kernel void rms_norm_add(__global const float *x, __global const float *w, __global const float *res, __global float *out, uint n) {
    uint lane = get_local_id(0);
    uint nl = get_local_size(0);
    __local float red[64];
    float acc = 0.0f;
    for (uint i = lane; i < n; i += nl) {
        acc += x[i] * x[i];
    }
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = nl >> 1; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = red[0];
    float scale = rsqrt(total / (float)n + 1e-5f);
    for (uint i = lane; i < n; i += nl) {
        out[i] = x[i] * scale * w[i] + res[i];
    }
}

__kernel void quantize_q8(__global const float *x, __global char *xq, __global float *xs, uint n, uint group) {
    uint lid = get_local_id(0);
    uint ng = n / group;
    uint iters = (ng + 15u) >> 4;
    uint sub = lid & 3u;
    uint gbase = (lid >> 2) << 2;
    __local float red[256];
    uint per = group >> 2;
    uint quads = per >> 2;
    for (uint it = 0u; it < iters; ++it) {
        uint g = it * 16u + (lid >> 2);
        bool act = g < ng;
        uint base = g * group + sub * per;
        float amax = 0.0f;
        float4 vals[8];
        if (act) {
            for (uint i = 0u; i < quads; ++i) {
                float4 v = vload4(0, x + base + i * 4u);
                vals[i] = v;
                amax = fmax(amax, fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
            }
        }
        red[lid] = amax;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sub == 0u) {
            red[lid] = fmax(fmax(red[lid], red[lid + 1u]), fmax(red[lid + 2u], red[lid + 3u]));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float m = red[gbase];
        float sc = m > 0.0f ? m / 127.0f : 0.0f;
        float inv = m > 0.0f ? 127.0f / m : 0.0f;
        if (act) {
            if (sub == 0u) {
                xs[g] = sc;
            }
            for (uint i = 0u; i < quads; ++i) {
                float4 v = vals[i] * inv;
                char4 q = (char4)(convert_int_rte(v.x), convert_int_rte(v.y), convert_int_rte(v.z), convert_int_rte(v.w));
                vstore4(q, 0, xq + base + i * 4u);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

static void gsm_k4(uint j, __global const uchar *q, uchar *d, uchar *m) {
    uint jm = j & 3u;
    uint h = j >> 2;
    uchar a = q[jm];
    uchar b = q[jm + 4u];
    uchar c = q[jm + 8u];
    uchar lo_d = a & 63u;
    uchar lo_m = b & 63u;
    uchar hi_d = (c & 15u) | ((a >> 6) << 4);
    uchar hi_m = (c >> 4) | ((b >> 6) << 4);
    *d = h ? hi_d : lo_d;
    *m = h ? hi_m : lo_m;
}

__kernel void gemv_q4_k_i8(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    if (row0 >= rows) {
        return;
    }
    uint rlim = min((uint)RGEMV, rows - row0);
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 144ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
    const char4 ones = (char4)(1);
    float acc[RGEMV];
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        acc[r] = 0.0f;
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bpi) {
        uint b = b0 + (lane >> 3);
        if (b < blocks) {
            uint w_lo = b * 256u + wbase;
            char16 xl = vload16(0, xq + w_lo);
            char16 xh = vload16(0, xq + w_lo + 32u);
            float xs_lo = xs[w_lo >> 5];
            float xs_hi = xs[(w_lo + 32u) >> 5];
            int sum_lo = (int)LFM_DOT(xl.s0123, ones) + (int)LFM_DOT(xl.s4567, ones)
                       + (int)LFM_DOT(xl.s89ab, ones) + (int)LFM_DOT(xl.scdef, ones);
            int sum_hi = (int)LFM_DOT(xh.s0123, ones) + (int)LFM_DOT(xh.s4567, ones)
                       + (int)LFM_DOT(xh.s89ab, ones) + (int)LFM_DOT(xh.scdef, ones);
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)min(r, rlim - 1u) * row_stride + (ulong)b * 144ul;
                __global const uchar *scp = blk + 4;
                float d = lfm_scale((__global const ushort *)blk);
                float dmin = lfm_scale((__global const ushort *)(blk + 2));
                uchar sl = 0u;
                uchar ml = 0u;
                uchar sh = 0u;
                uchar mh = 0u;
                gsm_k4(sb_lo, scp, &sl, &ml);
                gsm_k4(sb_lo + 1u, scp, &sh, &mh);
                uint4 pk = vload4(0, (__global const uint *)(blk + 16u + u * 16u));
                uint4 lo = pk & (uint4)(0x0F0F0F0Fu);
                uint4 hi = (pk >> 4) & (uint4)(0x0F0F0F0Fu);
                float dl = (float)(LFM_DOT(as_char4(lo.x), xl.s0123) + LFM_DOT(as_char4(lo.y), xl.s4567)
                                 + LFM_DOT(as_char4(lo.z), xl.s89ab) + LFM_DOT(as_char4(lo.w), xl.scdef));
                float dh = (float)(LFM_DOT(as_char4(hi.x), xh.s0123) + LFM_DOT(as_char4(hi.y), xh.s4567)
                                 + LFM_DOT(as_char4(hi.z), xh.s89ab) + LFM_DOT(as_char4(hi.w), xh.scdef));
                acc[r] += xs_lo * (dl * (d * (float)sl) - (float)sum_lo * (dmin * (float)ml));
                acc[r] += xs_hi * (dh * (d * (float)sh) - (float)sum_hi * (dmin * (float)mh));
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add(acc[r]);
        if (lane == 0u && r < rlim) {
            y[row0 + r] = total;
        }
    }
}


__kernel void gemv_q4_k_i8_pk(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    if (row0 >= rows) {
        return;
    }
    uint rlim = min((uint)RGEMV, rows - row0);
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 148ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
    const char4 ones = (char4)(1);
    float acc[RGEMV];
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        acc[r] = 0.0f;
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bpi) {
        uint b = b0 + (lane >> 3);
        if (b < blocks) {
            uint w_lo = b * 256u + wbase;
            char16 xl = vload16(0, xq + w_lo);
            char16 xh = vload16(0, xq + w_lo + 32u);
            float xs_lo = xs[w_lo >> 5];
            float xs_hi = xs[(w_lo + 32u) >> 5];
            int sum_lo = (int)LFM_DOT(xl.s0123, ones) + (int)LFM_DOT(xl.s4567, ones)
                       + (int)LFM_DOT(xl.s89ab, ones) + (int)LFM_DOT(xl.scdef, ones);
            int sum_hi = (int)LFM_DOT(xh.s0123, ones) + (int)LFM_DOT(xh.s4567, ones)
                       + (int)LFM_DOT(xh.s89ab, ones) + (int)LFM_DOT(xh.scdef, ones);
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)min(r, rlim - 1u) * row_stride + (ulong)b * 148ul;
                __global const uchar *scp = blk + 4;
                float d = lfm_scale((__global const ushort *)blk);
                float dmin = lfm_scale((__global const ushort *)(blk + 2));
                uchar sl = scp[2u * sb_lo];
                uchar ml = scp[2u * sb_lo + 1u];
                uchar sh = scp[2u * (sb_lo + 1u)];
                uchar mh = scp[2u * (sb_lo + 1u) + 1u];
                uint4 pkl = vload4(0, (__global const uint *)(blk + 20u + u * 16u));
                uint4 lo = pkl & (uint4)(0x0F0F0F0Fu);
                uint4 hi = (pkl >> 4) & (uint4)(0x0F0F0F0Fu);
                float dl = (float)(LFM_DOT(as_char4(lo.x), xl.s0123) + LFM_DOT(as_char4(lo.y), xl.s4567)
                                 + LFM_DOT(as_char4(lo.z), xl.s89ab) + LFM_DOT(as_char4(lo.w), xl.scdef));
                float dh = (float)(LFM_DOT(as_char4(hi.x), xh.s0123) + LFM_DOT(as_char4(hi.y), xh.s4567)
                                 + LFM_DOT(as_char4(hi.z), xh.s89ab) + LFM_DOT(as_char4(hi.w), xh.scdef));
                acc[r] += xs_lo * (dl * (d * (float)sl) - (float)sum_lo * (dmin * (float)ml));
                acc[r] += xs_hi * (dh * (d * (float)sh) - (float)sum_hi * (dmin * (float)mh));
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add(acc[r]);
        if (lane == 0u && r < rlim) {
            y[row0 + r] = total;
        }
    }
}

__kernel void gemv_q6_k_i8(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols) {
    uint lane = get_sub_group_local_id();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    if (row0 >= rows) {
        return;
    }
    uint rlim = min((uint)RGEMV, rows - row0);
    uint sub = lane & 15u;
    uint n = sub >> 3;
    uint l = (sub & 7u) * 4u;
    uint lq = l >> 4;
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 210ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    float acc[RGEMV];
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        acc[r] = 0.0f;
    }
    uint bps = get_sub_group_size() >> 4;
    if (bps == 0u) {
        bps = 1u;
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bps) {
        uint b = b0 + (lane >> 4);
        if (b < blocks) {
            uint w0 = b * 256u + n * 128u + l;
            uint xg = w0 >> 5;
            char4 x0 = vload4(0, xq + w0);
            char4 x1 = vload4(0, xq + w0 + 32u);
            char4 x2 = vload4(0, xq + w0 + 64u);
            char4 x3 = vload4(0, xq + w0 + 96u);
            float xs0 = xs[xg + 0u];
            float xs1 = xs[xg + 1u];
            float xs2 = xs[xg + 2u];
            float xs3 = xs[xg + 3u];
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)min(r, rlim - 1u) * row_stride + (ulong)b * 210ul;
                __global const char *sc = (__global const char *)(blk + 192u);
                float d = lfm_scale((__global const ushort *)(blk + 208u));
                uint so = n * 8u + lq;
                float s0 = d * (float)sc[so + 0u];
                float s1 = d * (float)sc[so + 2u];
                float s2 = d * (float)sc[so + 4u];
                float s3 = d * (float)sc[so + 6u];
                __global const uchar *ql = blk + n * 64u + l;
                __global const uchar *qh = blk + 128u + n * 32u + l;
                uint lo = as_uint(vload4(0, ql));
                uint hi = as_uint(vload4(0, ql + 32u));
                uint hb = as_uint(vload4(0, qh));
                char4 q0 = as_char4((lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4)) - (char4)(32);
                char4 q1 = as_char4((hi & 0x0F0F0F0Fu) | (((hb >> 2) & 0x03030303u) << 4)) - (char4)(32);
                char4 q2 = as_char4(((lo >> 4) & 0x0F0F0F0Fu) | (((hb >> 4) & 0x03030303u) << 4)) - (char4)(32);
                char4 q3 = as_char4(((hi >> 4) & 0x0F0F0F0Fu) | (((hb >> 6) & 0x03030303u) << 4)) - (char4)(32);
                acc[r] += xs0 * s0 * (float)LFM_DOT(q0, x0);
                acc[r] += xs1 * s1 * (float)LFM_DOT(q1, x1);
                acc[r] += xs2 * s2 * (float)LFM_DOT(q2, x2);
                acc[r] += xs3 * s3 * (float)LFM_DOT(q3, x3);
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add(acc[r]);
        if (lane == 0u && r < rlim) {
            y[row0 + r] = total;
        }
    }
}

__kernel void silu_mul(__global const float *ug, __global float *out, uint n, uint up_off) {
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float g = ug[i];
    float u = ug[up_off + i];
    out[i] = (g / (1.0f + exp(-g))) * u;
}

__kernel void conv_step(__global const float *bcx, __global const float *w, __global float *state, __global float *y, uint state_off, uint h, uint ksize) {
    uint ch = get_global_id(0);
    if (ch >= h) {
        return;
    }
    float b = bcx[ch];
    float c = bcx[h + ch];
    float xv = bcx[2u * h + ch];
    float bx = b * xv;
    float acc = w[ch * ksize + ksize - 1] * bx;
    float prev[8];
    __global float *st = state + state_off;
    for (uint t = 0; t < ksize - 1; ++t) {
        prev[t] = st[t * h + ch];
        acc += w[ch * ksize + t] * prev[t];
    }
    for (uint t = 0; t + 1 < ksize - 1; ++t) {
        st[t * h + ch] = prev[t + 1];
    }
    st[(ksize - 2) * h + ch] = bx;
    y[ch] = c * acc;
}

__kernel void rms_quant(__global const float *x, __global const float *w, __global char *xq, __global float *xs, uint n) {
    uint lane = get_local_id(0);
    __local float red[64];
    float acc = 0.0f;
    for (uint i = lane; i < n; i += 64u) {
        acc += x[i] * x[i];
    }
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = 32u; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(red[0] / (float)n + 1e-5f);
    for (uint g = lane; g < (n >> 5); g += 64u) {
        uint base = g << 5;
        float vals[32];
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; ++i) {
            float v = x[base + i] * scale * w[base + i];
            vals[i] = v;
            amax = fmax(amax, fabs(v));
        }
        float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
        float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        xs[g] = sc;
        for (uint i = 0u; i < 32u; ++i) {
            xq[base + i] = (char)convert_int_rte(vals[i] * inv);
        }
    }
}

__kernel void silu_quant(__global const float *ug, __global char *xq, __global float *xs, uint n, uint up_off) {
    uint g = get_global_id(0);
    if (g >= (n >> 5)) {
        return;
    }
    uint base = g << 5;
    float4 vals[8];
    float4 amax = (float4)(0.0f);
    for (uint i = 0u; i < 8u; ++i) {
        float4 gv = vload4(0, ug + base + i * 4u);
        float4 uv = vload4(0, ug + up_off + base + i * 4u);
        float4 sig = (float4)(1.0f) / ((float4)(1.0f) + exp(-gv));
        float4 v = gv * sig * uv;
        vals[i] = v;
        amax = fmax(amax, fabs(v));
    }
    float m = fmax(fmax(amax.x, amax.y), fmax(amax.z, amax.w));
    float sc = m > 0.0f ? m / 127.0f : 0.0f;
    float inv = m > 0.0f ? 127.0f / m : 0.0f;
    xs[g] = sc;
    for (uint i = 0u; i < 8u; ++i) {
        float4 v = vals[i] * inv;
        char4 q = (char4)(convert_int_rte(v.x), convert_int_rte(v.y), convert_int_rte(v.z), convert_int_rte(v.w));
        vstore4(q, 0, xq + base + i * 4u);
    }
}

__kernel void norm_add_quant(__global const float *op, __global const float *post, __global const float *res, __global const float *next_w, __global float *x_out, __global char *xq, __global float *xs, uint n) {
    uint lid = get_local_id(0);
    uint sub = lid & 3u;
    uint gbase = (lid >> 2) << 2;
    __local float red[256];
    uint ng = n >> 5;
    uint iters = (ng + 63u) >> 6;
    for (uint it = 0u; it < iters; ++it) {
        uint g = it * 64u + (lid >> 2);
        bool act = g < ng;
        uint base = (g << 5) + sub * 8u;
        float s1 = 0.0f;
        float4 o0 = (float4)(0.0f);
        float4 o1 = (float4)(0.0f);
        if (act) {
            o0 = vload4(0, op + base);
            o1 = vload4(0, op + base + 4u);
            s1 = o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w
               + o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w;
        }
        red[lid] = s1;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 128u; st > 0u; st >>= 1) {
            if (lid < st) {
                red[lid] += red[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float scale = rsqrt(red[0] / (float)n + 1e-5f);
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 v0 = (float4)(0.0f);
        float4 v1 = (float4)(0.0f);
        float s2 = 0.0f;
        if (act) {
            float4 p0 = vload4(0, post + base);
            float4 p1 = vload4(0, post + base + 4u);
            float4 r0 = vload4(0, res + base);
            float4 r1 = vload4(0, res + base + 4u);
            v0 = o0 * scale * p0 + r0;
            v1 = o1 * scale * p1 + r1;
            vstore4(v0, 0, x_out + base);
            vstore4(v1, 0, x_out + base + 4u);
            s2 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w
               + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
        }
        red[lid] = s2;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 128u; st > 0u; st >>= 1) {
            if (lid < st) {
                red[lid] += red[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float scale2 = rsqrt(red[0] / (float)n + 1e-5f);
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 w0 = (float4)(0.0f);
        float4 w1 = (float4)(0.0f);
        float amax = 0.0f;
        if (act) {
            float4 n0 = vload4(0, next_w + base);
            float4 n1 = vload4(0, next_w + base + 4u);
            w0 = v0 * scale2 * n0;
            w1 = v1 * scale2 * n1;
            amax = fmax(fmax(fmax(fabs(w0.x), fabs(w0.y)), fmax(fabs(w0.z), fabs(w0.w))),
                        fmax(fmax(fabs(w1.x), fabs(w1.y)), fmax(fabs(w1.z), fabs(w1.w))));
        }
        red[lid] = amax;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sub == 0u) {
            red[lid] = fmax(fmax(red[lid], red[lid + 1u]), fmax(red[lid + 2u], red[lid + 3u]));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float m = red[gbase];
        float sc = m > 0.0f ? m / 127.0f : 0.0f;
        float inv = m > 0.0f ? 127.0f / m : 0.0f;
        if (act) {
            if (sub == 0u) {
                xs[g] = sc;
            }
            float4 q0 = w0 * inv;
            float4 q1 = w1 * inv;
            char4 c0 = (char4)(convert_int_rte(q0.x), convert_int_rte(q0.y), convert_int_rte(q0.z), convert_int_rte(q0.w));
            char4 c1 = (char4)(convert_int_rte(q1.x), convert_int_rte(q1.y), convert_int_rte(q1.z), convert_int_rte(q1.w));
            vstore4(c0, 0, xq + base);
            vstore4(c1, 0, xq + base + 4u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void rms_rope_qk(__global float *q, __global float *k, __global const float *wq, __global const float *wk, __global const float *inv_freq, uint pos, uint head_dim, uint nq, uint nk) {
    uint gid = get_group_id(0);
    __global float *v;
    __global const float *w;
    uint head;
    if (gid < nq) {
        v = q + gid * head_dim;
        w = wq;
        head = gid;
    } else {
        uint h = gid - nq;
        if (h >= nk) {
            return;
        }
        v = k + h * head_dim;
        w = wk;
        head = h;
    }
    uint lane = get_local_id(0);
    __local float red[64];
    float acc = 0.0f;
    for (uint i = lane; i < head_dim; i += 64u) {
        acc += v[i] * v[i];
    }
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = 32u; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(red[0] / (float)head_dim + 1e-5f);
    for (uint i = lane; i < head_dim; i += 64u) {
        v[i] = v[i] * scale * w[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint hd2 = head_dim / 2u;
    if (lane >= hd2) {
        return;
    }
    float f = inv_freq[lane];
    if (f == 0.0f) {
        return;
    }
    float angle = (float)pos * f;
    float sn = sin(angle);
    float cs = cos(angle);
    float a = v[lane];
    float b = v[lane + hd2];
    v[lane] = a * cs - b * sn;
    v[lane + hd2] = b * cs + a * sn;
}

__kernel void attention_combine(__global const float *pacc, __global const float *pml, __global const float *gate, __global float *out, uint n_heads, uint head_dim, uint splits) {
    uint head = get_group_id(0);
    uint lane = get_local_id(0);
    if (head >= n_heads) {
        return;
    }
    __global const float *pa = pacc + (ulong)head * splits * head_dim;
    __global const float *pm = pml + (ulong)head * splits * 2u;
    float m = -INFINITY;
    for (uint s = 0u; s < splits; ++s) {
        m = fmax(m, pm[s * 2u]);
    }
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (uint s = 0u; s < splits; ++s) {
        float w = exp(pm[s * 2u] - m);
        l += pm[s * 2u + 1u] * w;
        a0 += pa[s * head_dim + lane] * w;
        a1 += pa[s * head_dim + lane + 64u] * w;
    }
    float gs = 1.0f / (1.0f + exp(-gate[head]));
    out[head * head_dim + lane] = (a0 / l) * gs;
    out[head * head_dim + lane + 64u] = (a1 / l) * gs;
}

__kernel void kv_store_h(__global const float *ksrc, __global const float *vsrc, __global KV_T *kdst, __global KV_T *vdst, __global const uint *tbl, uint pos, uint n) {
    uint i = get_global_id(0);
    if (i < n) {
        ulong slot = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        kdst[slot * n + i] = (KV_T)ksrc[i];
        vdst[slot * n + i] = (KV_T)vsrc[i];
    }
}

__kernel void fill_zero_b(__global uchar *dst, ulong off, ulong n) {
    ulong i = (ulong)get_global_id(0) * 16;
    if (i + 16 <= n) {
        __global uchar *p = dst + off + i;
        vstore16((uchar16)(0), 0, p);
    } else if (i < n) {
        for (ulong j = i; j < n; j++) {
            dst[off + j] = 0;
        }
    }
}

__kernel void attention_h(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const float *gate, __global float *out, uint len, uint start, uint kv_heads, uint head_dim, uint n_heads) {
    uint lane = get_local_id(0);
    uint head = get_group_id(0);
    __local float sred[128];
    uint kvh = head / (n_heads / kv_heads);
    __global const float *qh = q + head * head_dim;
    float q0 = qh[2u * lane];
    float q1 = qh[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint pos = start;
    for (; pos + 1u < len; pos += 2u) {
        ulong slot0 = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        ulong slot1 = (ulong)((tbl[(pos + 1u) >> 5] << 5) | ((pos + 1u) & 31u));
        __global const KV_T *kp0 = kbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *kp1 = kbuf + slot1 * kv_stride + kvh * head_dim;
        float2 k0 = kv_load2(lane, kp0);
        float2 k1 = kv_load2(lane, kp1);
#ifdef LFM_SG64
        float s0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y) * scale;
        float s1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y) * scale;
#else
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
        float s0 = sred[0] * scale;
        float s1 = sred[64u] * scale;
#endif
        float m_new = fmax(m, fmax(s0, s1));
        float corr = exp(m - m_new);
        float w0 = exp(s0 - m_new);
        float w1 = exp(s1 - m_new);
        l = l * corr + w0 + w1;
        __global const KV_T *vp0 = vbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *vp1 = vbuf + slot1 * kv_stride + kvh * head_dim;
        float2 v0 = kv_load2(lane, vp0);
        float2 v1 = kv_load2(lane, vp1);
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < len) {
        ulong slots = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        __global const KV_T *kp = kbuf + slots * kv_stride + kvh * head_dim;
        float2 k = kv_load2(lane, kp);
#ifdef LFM_SG64
        float s = sub_group_reduce_add(q0 * k.x + q1 * k.y) * scale;
#else
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
        float s = sred[0] * scale;
#endif
        float m_new = fmax(m, s);
        float corr = exp(m - m_new);
        float w = exp(s - m_new);
        l = l * corr + w;
        __global const KV_T *vp = vbuf + slots * kv_stride + kvh * head_dim;
        float2 v = kv_load2(lane, vp);
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = m_new;
    }
    float gs = 1.0f / (1.0f + exp(-gate[head]));
    out[head * head_dim + 2u * lane] = (acc0 / l) * gs;
    out[head * head_dim + 2u * lane + 1u] = (acc1 / l) * gs;
}

__kernel void attention_split_h(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global float *pacc, __global float *pml, uint len, uint start, uint kv_heads, uint head_dim, uint n_heads, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint head = gid / splits;
    uint sp = gid % splits;
    if (head >= n_heads) {
        return;
    }
    __local float sred[128];
    uint kvh = head / (n_heads / kv_heads);
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    __global const float *qh = q + head * head_dim;
    float q0 = qh[2u * lane];
    float q1 = qh[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint pos = p0;
    for (; pos + 1u < p1; pos += 2u) {
        ulong slot0 = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        ulong slot1 = (ulong)((tbl[(pos + 1u) >> 5] << 5) | ((pos + 1u) & 31u));
        __global const KV_T *kp0 = kbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *kp1 = kbuf + slot1 * kv_stride + kvh * head_dim;
        float2 k0 = kv_load2(lane, kp0);
        float2 k1 = kv_load2(lane, kp1);
#ifdef LFM_SG64
        float s0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y) * scale;
        float s1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y) * scale;
#else
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
        float s0 = sred[0] * scale;
        float s1 = sred[64u] * scale;
#endif
        float m_new = fmax(m, fmax(s0, s1));
        float corr = exp(m - m_new);
        float w0 = exp(s0 - m_new);
        float w1 = exp(s1 - m_new);
        l = l * corr + w0 + w1;
        __global const KV_T *vp0 = vbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *vp1 = vbuf + slot1 * kv_stride + kvh * head_dim;
        float2 v0 = kv_load2(lane, vp0);
        float2 v1 = kv_load2(lane, vp1);
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < p1) {
        ulong slots = (ulong)((tbl[pos >> 5] << 5) | (pos & 31u));
        __global const KV_T *kp = kbuf + slots * kv_stride + kvh * head_dim;
        float2 k = kv_load2(lane, kp);
#ifdef LFM_SG64
        float sv = sub_group_reduce_add(q0 * k.x + q1 * k.y) * scale;
#else
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
#endif
        float m_new = fmax(m, sv);
        float corr = exp(m - m_new);
        float w = exp(sv - m_new);
        l = l * corr + w;
        __global const KV_T *vp = vbuf + slots * kv_stride + kvh * head_dim;
        float2 v = kv_load2(lane, vp);
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = m_new;
    }
    ulong pbase = ((ulong)head * splits + sp) * head_dim;
    pacc[pbase + 2u * lane] = acc0;
    pacc[pbase + 2u * lane + 1u] = acc1;
    if (lane == 0u) {
        pml[((ulong)head * splits + sp) * 2u] = m;
        pml[((ulong)head * splits + sp) * 2u + 1u] = l;
    }
}
