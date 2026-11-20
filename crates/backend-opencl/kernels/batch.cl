
#ifndef RGEMV_B
#define RGEMV_B 5
#endif

#ifndef RGEMV4P
#define RGEMV4P 4
#endif

#ifndef PB
#define PB 4
#endif
__kernel void embed_q6k_b(__global const uchar *w, __global float *out, __global const uint *tokens, uint ntok, uint cols) {
    uint i = get_global_id(0);
    uint total = ntok * cols;
    if (i >= total) {
        return;
    }
    uint t = i / cols;
    uint c = i % cols;
    uint token = tokens[t];
    uint blocks = cols >> 8;
    __global const uchar *base = w + (ulong)token * ((ulong)blocks * 210ul);
    uint b = c >> 8;
    uint r = c & 255u;
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
    out[t * cols + c] = d * sv * (float)qv;
}

__kernel void rms_quant_b(__global const float *x, __global const float *w, __global char *xq, __global float *xs, uint n, uint ntok) {
    uint t = get_group_id(0);
    if (t >= ntok) {
        return;
    }
    uint lane = get_local_id(0);
    __local float red[64];
    __global const float *xt = x + t * n;
    float acc = 0.0f;
    for (uint i = lane; i < n; i += 64u) {
        acc += xt[i] * xt[i];
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
    __global char *xqt = xq + t * n;
    __global float *xst = xs + t * (n >> 5);
    for (uint g = lane; g < (n >> 5); g += 64u) {
        uint base = g << 5;
        float vals[32];
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; ++i) {
            float v = xt[base + i] * scale * w[base + i];
            vals[i] = v;
            amax = fmax(amax, fabs(v));
        }
        float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
        float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        xst[g] = sc;
        for (uint i = 0u; i < 32u; ++i) {
            xqt[base + i] = (char)convert_int_rte(vals[i] * inv);
        }
    }
}

__kernel void norm_add_quant_b(__global const float *op, __global const float *post, __global const float *res, __global const float *next_w, __global float *x_out, __global char *xq, __global float *xs, uint n, uint ntok) {
    uint t = get_group_id(0);
    if (t >= ntok) {
        return;
    }
    uint lid = get_local_id(0);
    uint sub = lid & 3u;
    uint gbase = (lid >> 2) << 2;
    __local float red[256];
    __local float sred[256];
    uint ng = n >> 5;
    uint iters = (ng + 63u) >> 6;
    uint sgl = get_sub_group_local_id();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    __global const float *opt = op + (ulong)t * n;
    __global const float *rest = res + (ulong)t * n;
    __global float *xot = x_out + (ulong)t * n;
    __global char *xqt = xq + (ulong)t * n;
    __global float *xst = xs + (ulong)t * ng;
    for (uint it = 0u; it < iters; ++it) {
        uint g = it * 64u + (lid >> 2);
        bool act = g < ng;
        uint base = (g << 5) + sub * 8u;
        float s1 = 0.0f;
        float4 o0 = (float4)(0.0f);
        float4 o1 = (float4)(0.0f);
        if (act) {
            o0 = vload4(0, opt + base);
            o1 = vload4(0, opt + base + 4u);
            s1 = o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w
               + o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w;
        }
#ifdef LFM_SG64
        float part = sub_group_reduce_add(s1);
        if (sgl == 0u) {
            sred[sgid] = part;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float ssum = 0.0f;
        for (uint i = 0u; i < nsg; ++i) {
            ssum += sred[i];
        }
        float scale = rsqrt(ssum / (float)n + 1e-5f);
        barrier(CLK_LOCAL_MEM_FENCE);
#else
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
#endif
        float4 v0 = (float4)(0.0f);
        float4 v1 = (float4)(0.0f);
        float s2 = 0.0f;
        if (act) {
            float4 p0 = vload4(0, post + base);
            float4 p1 = vload4(0, post + base + 4u);
            float4 r0 = vload4(0, rest + base);
            float4 r1 = vload4(0, rest + base + 4u);
            v0 = o0 * scale * p0 + r0;
            v1 = o1 * scale * p1 + r1;
            vstore4(v0, 0, xot + base);
            vstore4(v1, 0, xot + base + 4u);
            s2 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w
               + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
        }
#ifdef LFM_SG64
        part = sub_group_reduce_add(s2);
        if (sgl == 0u) {
            sred[sgid] = part;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        ssum = 0.0f;
        for (uint i = 0u; i < nsg; ++i) {
            ssum += sred[i];
        }
        float scale2 = rsqrt(ssum / (float)n + 1e-5f);
        barrier(CLK_LOCAL_MEM_FENCE);
#else
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
#endif
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
                xst[g] = sc;
            }
            float4 q0 = w0 * inv;
            float4 q1 = w1 * inv;
            char4 c0 = (char4)(convert_int_rte(q0.x), convert_int_rte(q0.y), convert_int_rte(q0.z), convert_int_rte(q0.w));
            char4 c1 = (char4)(convert_int_rte(q1.x), convert_int_rte(q1.y), convert_int_rte(q1.z), convert_int_rte(q1.w));
            vstore4(c0, 0, xqt + base);
            vstore4(c1, 0, xqt + base + 4u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void silu_quant_b(__global const float *ug, __global char *xq, __global float *xs, uint n, uint up_off, uint ntok) {
    uint g = get_global_id(0);
    uint ng = n >> 5;
    if (g >= ntok * ng) {
        return;
    }
    uint t = g / ng;
    uint gg = g % ng;
    uint base = (t * 2u * n) + (gg << 5);
    uint uoff = base + up_off;
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0u; i < 32u; ++i) {
        float gv = ug[base + i];
        float v = (gv / (1.0f + exp(-gv))) * ug[uoff + i];
        vals[i] = v;
        amax = fmax(amax, fabs(v));
    }
    float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    xs[t * ng + gg] = sc;
    uint ob = (t * n) + (gg << 5);
    for (uint i = 0u; i < 32u; ++i) {
        xq[ob + i] = (char)convert_int_rte(vals[i] * inv);
    }
}

__kernel void quantize_b(__global const float *x, __global char *xq, __global float *xs, uint n, uint group, uint ntok) {
    uint g = get_global_id(0);
    uint ng = n / group;
    if (g >= ntok * ng) {
        return;
    }
    uint t = g / ng;
    uint gg = g % ng;
    uint base = t * n + gg * group;
    float amax = 0.0f;
    for (uint i = 0u; i < group; i += 4u) {
        float4 v = vload4(0, x + base + i);
        amax = fmax(amax, fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
    }
    float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    xs[t * ng + gg] = sc;
    for (uint i = 0u; i < group; i += 4u) {
        float4 v = vload4(0, x + base + i) * inv;
        char4 q = (char4)(convert_int_rte(v.x), convert_int_rte(v.y), convert_int_rte(v.z), convert_int_rte(v.w));
        vstore4(q, 0, xq + base + i);
    }
}

__kernel void gemv_q4_k_i8_b(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok, uint tok0) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV_B;
    if (rows < RGEMV_B) {
        return;
    }
    if (row0 + RGEMV_B > rows) {
        row0 = rows - RGEMV_B;
    }
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 144ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
    const char4 ones = (char4)(1);
    float acc[PB][RGEMV_B];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV_B; ++r) {
            acc[t][r] = 0.0f;
        }
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bpi) {
        uint b = b0 + (lane >> 3);
        if (b < blocks) {
            uint w_lo = b * 256u + wbase;
            uint ski = w_lo >> 5;
            uint ski2 = (w_lo + 32u) >> 5;
            char16 xl[PB];
            char16 xh[PB];
            float xsl[PB];
            float xsh[PB];
            int suml[PB];
            int sumh[PB];
            #pragma unroll
            for (uint t = 0u; t < PB; ++t) {
                xl[t] = (char16)(0);
                xh[t] = (char16)(0);
                xsl[t] = 0.0f;
                xsh[t] = 0.0f;
                suml[t] = 0;
                sumh[t] = 0;
                if (t >= tok0 && t < ntok) {
                    xl[t] = vload16(0, xq + (ulong)t * cols + w_lo);
                    xh[t] = vload16(0, xq + (ulong)t * cols + w_lo + 32u);
                    xsl[t] = xs[t * ngs + ski];
                    xsh[t] = xs[t * ngs + ski2];
                    suml[t] = (int)LFM_DOT(xl[t].s0123, ones) + (int)LFM_DOT(xl[t].s4567, ones)
                            + (int)LFM_DOT(xl[t].s89ab, ones) + (int)LFM_DOT(xl[t].scdef, ones);
                    sumh[t] = (int)LFM_DOT(xh[t].s0123, ones) + (int)LFM_DOT(xh[t].s4567, ones)
                            + (int)LFM_DOT(xh[t].s89ab, ones) + (int)LFM_DOT(xh[t].scdef, ones);
                }
            }
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 144ul;
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
                char4 l0 = as_char4(pk.x & 0x0F0F0F0Fu);
                char4 l1 = as_char4(pk.y & 0x0F0F0F0Fu);
                char4 l2 = as_char4(pk.z & 0x0F0F0F0Fu);
                char4 l3 = as_char4(pk.w & 0x0F0F0F0Fu);
                char4 h0 = as_char4((pk.x >> 4) & 0x0F0F0F0Fu);
                char4 h1 = as_char4((pk.y >> 4) & 0x0F0F0F0Fu);
                char4 h2 = as_char4((pk.z >> 4) & 0x0F0F0F0Fu);
                char4 h3 = as_char4((pk.w >> 4) & 0x0F0F0F0Fu);
                float dl_coef = d * (float)sl;
                float dh_coef = d * (float)sh;
                float ml_coef = dmin * (float)ml;
                float mh_coef = dmin * (float)mh;
                #pragma unroll
                for (uint t = 0u; t < PB; ++t) {
                    if (t >= tok0 && t < ntok) {
                        float dl = (float)(LFM_DOT(l0, xl[t].s0123) + LFM_DOT(l1, xl[t].s4567)
                                         + LFM_DOT(l2, xl[t].s89ab) + LFM_DOT(l3, xl[t].scdef));
                        float dh = (float)(LFM_DOT(h0, xh[t].s0123) + LFM_DOT(h1, xh[t].s4567)
                                         + LFM_DOT(h2, xh[t].s89ab) + LFM_DOT(h3, xh[t].scdef));
                        acc[t][r] += xsl[t] * (dl * dl_coef - (float)suml[t] * ml_coef);
                        acc[t][r] += xsh[t] * (dh * dh_coef - (float)sumh[t] * mh_coef);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        if (t >= tok0 && t < ntok) {
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u) {
                    y[(ulong)t * rows + row0 + r] = total;
                }
            }
        }
    }
}

__kernel void gemv_q4_k_i8_p_b(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok, uint tok0) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV_B;
    if (rows < RGEMV_B) {
        return;
    }
    if (row0 + RGEMV_B > rows) {
        row0 = rows - RGEMV_B;
    }
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 148ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
    const char4 ones = (char4)(1);
    float acc[PB][RGEMV_B];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV_B; ++r) {
            acc[t][r] = 0.0f;
        }
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bpi) {
        uint b = b0 + (lane >> 3);
        if (b < blocks) {
            uint w_lo = b * 256u + wbase;
            uint ski = w_lo >> 5;
            uint ski2 = (w_lo + 32u) >> 5;
            char16 xl[PB];
            char16 xh[PB];
            float xsl[PB];
            float xsh[PB];
            int suml[PB];
            int sumh[PB];
            #pragma unroll
            for (uint t = 0u; t < PB; ++t) {
                xl[t] = (char16)(0);
                xh[t] = (char16)(0);
                xsl[t] = 0.0f;
                xsh[t] = 0.0f;
                suml[t] = 0;
                sumh[t] = 0;
                if (t >= tok0 && t < ntok) {
                    xl[t] = vload16(0, xq + (ulong)t * cols + w_lo);
                    xh[t] = vload16(0, xq + (ulong)t * cols + w_lo + 32u);
                    xsl[t] = xs[t * ngs + ski];
                    xsh[t] = xs[t * ngs + ski2];
                    suml[t] = (int)LFM_DOT(xl[t].s0123, ones) + (int)LFM_DOT(xl[t].s4567, ones)
                            + (int)LFM_DOT(xl[t].s89ab, ones) + (int)LFM_DOT(xl[t].scdef, ones);
                    sumh[t] = (int)LFM_DOT(xh[t].s0123, ones) + (int)LFM_DOT(xh[t].s4567, ones)
                            + (int)LFM_DOT(xh[t].s89ab, ones) + (int)LFM_DOT(xh[t].scdef, ones);
                }
            }
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 148ul;
                float d = lfm_scale((__global const ushort *)blk);
                float dmin = lfm_scale((__global const ushort *)(blk + 2));
                uchar4 sm = vload4(0, blk + 4u + 4u * (u >> 1));
                uint4 pk = vload4(0, (__global const uint *)(blk + 20u + u * 16u));
                char4 l0 = as_char4(pk.x & 0x0F0F0F0Fu);
                char4 l1 = as_char4(pk.y & 0x0F0F0F0Fu);
                char4 l2 = as_char4(pk.z & 0x0F0F0F0Fu);
                char4 l3 = as_char4(pk.w & 0x0F0F0F0Fu);
                char4 h0 = as_char4((pk.x >> 4) & 0x0F0F0F0Fu);
                char4 h1 = as_char4((pk.y >> 4) & 0x0F0F0F0Fu);
                char4 h2 = as_char4((pk.z >> 4) & 0x0F0F0F0Fu);
                char4 h3 = as_char4((pk.w >> 4) & 0x0F0F0F0Fu);
                float dl_coef = d * (float)sm.x;
                float dh_coef = d * (float)sm.z;
                float ml_coef = dmin * (float)sm.y;
                float mh_coef = dmin * (float)sm.w;
                #pragma unroll
                for (uint t = 0u; t < PB; ++t) {
                    if (t >= tok0 && t < ntok) {
                        float dl = (float)(LFM_DOT(l0, xl[t].s0123) + LFM_DOT(l1, xl[t].s4567)
                                         + LFM_DOT(l2, xl[t].s89ab) + LFM_DOT(l3, xl[t].scdef));
                        float dh = (float)(LFM_DOT(h0, xh[t].s0123) + LFM_DOT(h1, xh[t].s4567)
                                         + LFM_DOT(h2, xh[t].s89ab) + LFM_DOT(h3, xh[t].scdef));
                        acc[t][r] += xsl[t] * (dl * dl_coef - (float)suml[t] * ml_coef);
                        acc[t][r] += xsh[t] * (dh * dh_coef - (float)sumh[t] * mh_coef);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        if (t >= tok0 && t < ntok) {
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u) {
                    y[(ulong)t * rows + row0 + r] = total;
                }
            }
        }
    }
}


__kernel void gemv_q6_k_i8_b(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok, uint tok0) {
    uint lane = get_sub_group_local_id();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV_B;
    if (rows < RGEMV_B) {
        return;
    }
    if (row0 + RGEMV_B > rows) {
        row0 = rows - RGEMV_B;
    }
    uint sub = lane & 15u;
    uint n = sub >> 3;
    uint l = (sub & 7u) * 4u;
    uint lq = l >> 4;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 210ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    float acc[PB][RGEMV_B];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV_B; ++r) {
            acc[t][r] = 0.0f;
        }
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
            char4 x0[PB];
            char4 x1[PB];
            char4 x2[PB];
            char4 x3[PB];
            float xs0[PB];
            float xs1[PB];
            float xs2[PB];
            float xs3[PB];
            #pragma unroll
            for (uint t = 0u; t < PB; ++t) {
                x0[t] = (char4)(0);
                x1[t] = (char4)(0);
                x2[t] = (char4)(0);
                x3[t] = (char4)(0);
                xs0[t] = 0.0f;
                xs1[t] = 0.0f;
                xs2[t] = 0.0f;
                xs3[t] = 0.0f;
                if (t >= tok0 && t < ntok) {
                    ulong xb = (ulong)t * cols + w0;
                    x0[t] = vload4(0, xq + xb);
                    x1[t] = vload4(0, xq + xb + 32u);
                    x2[t] = vload4(0, xq + xb + 64u);
                    x3[t] = vload4(0, xq + xb + 96u);
                    uint sbase = t * ngs + xg;
                    xs0[t] = xs[sbase + 0u];
                    xs1[t] = xs[sbase + 1u];
                    xs2[t] = xs[sbase + 2u];
                    xs3[t] = xs[sbase + 3u];
                }
            }
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 210ul;
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
                #pragma unroll
                for (uint t = 0u; t < PB; ++t) {
                    if (t >= tok0 && t < ntok) {
                        acc[t][r] += xs0[t] * s0 * (float)LFM_DOT(q0, x0[t]);
                        acc[t][r] += xs1[t] * s1 * (float)LFM_DOT(q1, x1[t]);
                        acc[t][r] += xs2[t] * s2 * (float)LFM_DOT(q2, x2[t]);
                        acc[t][r] += xs3[t] * s3 * (float)LFM_DOT(q3, x3[t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        if (t >= tok0 && t < ntok) {
            #pragma unroll
            for (uint r = 0u; r < RGEMV_B; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u) {
                    y[(ulong)t * rows + row0 + r] = total;
                }
            }
        }
    }
}

__kernel void conv_step_b(__global const float *bcx, __global const float *w, __global float *state, __global float *y, __global const uint *offs, __global const uint *snap_page, __global float *snap, uint h, uint ksize, uint ntok, uint snap_stride) {
    uint ch = get_global_id(0);
    if (ch >= h) {
        return;
    }
    float prev[8];
    uint off0 = offs[0];
    __global float *st = state + off0;
    for (uint t = 0u; t < ksize - 1u; ++t) {
        prev[t] = st[t * h + ch];
    }
    for (uint t = 0u; t < ntok; ++t) {
        if (offs[t] != off0) {
            for (uint j = 0u; j < ksize - 1u; ++j) {
                st[j * h + ch] = prev[j];
            }
            off0 = offs[t];
            st = state + off0;
            for (uint j = 0u; j < ksize - 1u; ++j) {
                prev[j] = st[j * h + ch];
            }
        }
        __global const float *src = bcx + (ulong)t * 3u * h;
        float bx = src[ch] * src[2u * h + ch];
        float acc = w[ch * ksize + ksize - 1u] * bx;
        for (uint j = 0u; j + 1u < ksize; ++j) {
            acc += w[ch * ksize + j] * prev[j];
        }
        for (uint j = 0u; j + 1u < ksize - 1u; ++j) {
            prev[j] = prev[j + 1u];
        }
        prev[ksize - 2u] = bx;
        y[(ulong)t * h + ch] = src[h + ch] * acc;
        uint sp = snap_page[t];
        if (sp != 0xffffffffu) {
            ulong base = (ulong)sp * (ulong)snap_stride;
            for (uint j = 0u; j + 1u < ksize; ++j) {
                snap[base + (ulong)j * h + ch] = prev[j];
            }
        }
    }
    for (uint t = 0u; t < ksize - 1u; ++t) {
        st[t * h + ch] = prev[t];
    }
}

__kernel void conv_step_b7(__global const float *bcx, __global const float *w, __global float *state, __global float *y, __global const uint *offs, __global const uint *snap_page, __global float *snap, __global char *xq, __global float *xs, uint h, uint ntok, uint snap_stride) {
    uint ch = get_global_id(0);
    uint lid = get_local_id(0);
    bool act = ch < h;
    __local float qred[512];
    uint off0 = offs[0];
    __global float *st = state + off0;
    float p0 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
    float p3 = 0.0f;
    float p4 = 0.0f;
    float p5 = 0.0f;
    if (act) {
        p0 = st[ch];
        p1 = st[h + ch];
        p2 = st[2u * h + ch];
        p3 = st[3u * h + ch];
        p4 = st[4u * h + ch];
        p5 = st[5u * h + ch];
    }
    uint ng = h >> 5;
    for (uint t = 0u; t < ntok; ++t) {
        float yv = 0.0f;
        if (offs[t] != off0) {
            if (act) {
                st[ch] = p0;
                st[h + ch] = p1;
                st[2u * h + ch] = p2;
                st[3u * h + ch] = p3;
                st[4u * h + ch] = p4;
                st[5u * h + ch] = p5;
            }
            off0 = offs[t];
            st = state + off0;
            if (act) {
                p0 = st[ch];
                p1 = st[h + ch];
                p2 = st[2u * h + ch];
                p3 = st[3u * h + ch];
                p4 = st[4u * h + ch];
                p5 = st[5u * h + ch];
            }
        }
        __global const float *src = bcx + (ulong)t * 3u * h;
        if (act) {
            float bx = src[ch] * src[2u * h + ch];
            uint wb = ch * 7u;
            float acc = w[wb + 6u] * bx;
            acc += w[wb] * p0;
            acc += w[wb + 1u] * p1;
            acc += w[wb + 2u] * p2;
            acc += w[wb + 3u] * p3;
            acc += w[wb + 4u] * p4;
            acc += w[wb + 5u] * p5;
            p0 = p1;
            p1 = p2;
            p2 = p3;
            p3 = p4;
            p4 = p5;
            p5 = bx;
            yv = src[h + ch] * acc;
            y[(ulong)t * h + ch] = yv;
        }
        uint qb = (t & 1u) << 8;
        qred[qb + lid] = act ? fabs(yv) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (act && (lid & 31u) == 0u) {
            float m = qred[qb + lid];
            for (uint i = 1u; i < 32u; ++i) {
                m = fmax(m, qred[qb + lid + i]);
            }
            qred[qb + lid] = m;
            xs[(ulong)t * ng + (ch >> 5)] = m > 0.0f ? m / 127.0f : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (act) {
            float m = qred[qb + ((lid >> 5) << 5)];
            float inv = m > 0.0f ? 127.0f / m : 0.0f;
            xq[(ulong)t * h + ch] = (char)convert_int_rte(yv * inv);
            uint sp = snap_page[t];
            if (sp != 0xffffffffu) {
                ulong base = (ulong)sp * (ulong)snap_stride;
                snap[base + ch] = p0;
                snap[base + h + ch] = p1;
                snap[base + 2u * h + ch] = p2;
                snap[base + 3u * h + ch] = p3;
                snap[base + 4u * h + ch] = p4;
                snap[base + 5u * h + ch] = p5;
            }
        }
    }
    if (act) {
        st[ch] = p0;
        st[h + ch] = p1;
        st[2u * h + ch] = p2;
        st[3u * h + ch] = p3;
        st[4u * h + ch] = p4;
        st[5u * h + ch] = p5;
    }
}

__kernel void rms_rope_heads_b(__global float *x, __global const float *w, __global const float *inv_freq, __global const uint *pos_b, uint head_dim, uint nheads, uint ntok) {
    uint gid = get_group_id(0);
    uint nt = nheads * ntok;
    if (gid >= nt) {
        return;
    }
    uint t = gid / nheads;
    uint head = gid % nheads;
    uint lane = get_local_id(0);
    __global float *v = x + (ulong)t * nheads * head_dim + head * head_dim;
    float acc = 0.0f;
    for (uint i = lane; i < head_dim; i += 64u) {
        acc += v[i] * v[i];
    }
#ifdef LFM_SG64
    float total = sub_group_reduce_add(acc);
    float scale = rsqrt(total / (float)head_dim + 1e-5f);
#else
    __local float red[64];
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = 32u; st > 0u; st >>= 1) {
        if (lane < st) {
            red[lane] += red[lane + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(red[0] / (float)head_dim + 1e-5f);
#endif
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
    float angle = (float)pos_b[t] * f;
    float s = sin(angle);
    float c = cos(angle);
    float a = v[lane];
    float b = v[lane + hd2];
    v[lane] = a * c - b * s;
    v[lane + hd2] = b * c + a * s;
}

__kernel void attention_combine_b(__global const float *pacc, __global const float *pml, __global const float *gate, __global float *out, uint ntok, uint n_heads, uint head_dim, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint t = gid / n_heads;
    uint head = gid % n_heads;
    if (t >= ntok) {
        return;
    }
    uint base = t * n_heads + head;
    __global const float *pa = pacc + (ulong)base * splits * head_dim;
    __global const float *pm = pml + (ulong)base * splits * 2u;
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
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float gs = 1.0f / (1.0f + exp(-gate[base]));
    __global float *o = out + (ulong)(t * n_heads + head) * head_dim;
    o[lane] = a0 * inv * gs;
    o[lane + 64u] = a1 * inv * gs;
}

__kernel void kv_store_b_h(__global const float *ksrc, __global const float *vsrc, __global KV_T *kdst, __global KV_T *vdst, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, uint n, uint ntok) {
    uint i = get_global_id(0);
    if (i >= ntok * n) {
        return;
    }
    uint t = i / n;
    uint j = i % n;
    uint pos = pos_b[t];
    ulong slot = (ulong)((tbl[tbl_off[t] + (pos >> 5)] << 5) | (pos & 31u));
    kdst[slot * n + j] = (KV_T)ksrc[i];
    vdst[slot * n + j] = (KV_T)vsrc[i];
}

__kernel void attention_b_h(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global const float *gate, __global float *out, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window) {
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
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    __global const float *qh = q + (ulong)t * n_heads * head_dim + head * head_dim;
    float q0 = qh[2u * lane];
    float q1 = qh[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 1.0f / sqrt((float)head_dim);
    for (uint pos = start; pos < len; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        __global const KV_T *kp = kbuf + slot * kv_heads * head_dim + kvh * head_dim;
        float2 k = kv_load2(lane, kp);
        sred[lid] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 32u; st > 0u; st >>= 1) {
            if (lane < st) {
                sred[lid] += sred[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float s = sred[grp * 64u] * scale;
        float m_new = fmax(m, s);
        float corr = exp(m - m_new);
        float wgt = exp(s - m_new);
        l = l * corr + wgt;
        __global const KV_T *vp = vbuf + slot * kv_heads * head_dim + kvh * head_dim;
        float2 v = kv_load2(lane, vp);
        acc0 = acc0 * corr + wgt * v.x;
        acc1 = acc1 * corr + wgt * v.y;
        m = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float gs = 1.0f / (1.0f + exp(-gate[t * n_heads + head]));
    __global float *o = out + (ulong)t * n_heads * head_dim + head * head_dim;
    o[2u * lane] = (acc0 / l) * gs;
    o[2u * lane + 1u] = (acc1 / l) * gs;
}

__kernel void attention_b_fa(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global const float *gate, __global float *out, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window) {
    uint lid = get_local_id(0);
    uint lane = lid & 63u;
    uint sg = lid >> 6;
    uint t = get_group_id(0);
    if (t >= ntok) {
        return;
    }
    uint head = sg * 4u + (lane >> 4);
    uint j = lane & 15u;
    uint kvh = sg;
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    float8 qv = vload8(0, q + ((ulong)t * n_heads + head) * head_dim + j * 8u);
    float m = -INFINITY;
    float l = 0.0f;
    float8 acc = (float8)(0.0f);
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint koff = kvh * head_dim + j * 8u;
    for (uint pos = start; pos < len; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        float8 kv = convert_float8(vload8(0, kbuf + slot * kv_stride + koff));
        float8 vv = convert_float8(vload8(0, vbuf + slot * kv_stride + koff));
        float d = qv.s0 * kv.s0 + qv.s1 * kv.s1 + qv.s2 * kv.s2 + qv.s3 * kv.s3
                + qv.s4 * kv.s4 + qv.s5 * kv.s5 + qv.s6 * kv.s6 + qv.s7 * kv.s7;
        d += sub_group_shuffle_xor(d, 8u);
        d += sub_group_shuffle_xor(d, 4u);
        d += sub_group_shuffle_xor(d, 2u);
        d += sub_group_shuffle_xor(d, 1u);
        float sv = d * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        acc = acc * corr + w * vv;
        m = mn;
    }
    float gs = 1.0f / (1.0f + exp(-gate[t * n_heads + head]));
    float8 res = (acc / l) * gs;
    vstore8(res, 0, out + ((ulong)t * n_heads + head) * head_dim + j * 8u);
}

__kernel void attention_b_h2(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global const float *gate, __global float *out, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window) {
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
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    __global const float *qh = q + (ulong)t * n_heads * head_dim + head * head_dim;
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
        ulong slot0 = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        ulong slot1 = (ulong)((tbl[toff + ((pos + 1u) >> 5)] << 5) | ((pos + 1u) & 31u));
        __global const KV_T *kp0 = kbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *kp1 = kbuf + slot1 * kv_stride + kvh * head_dim;
        float2 k0 = kv_load2(lane, kp0);
        float2 k1 = kv_load2(lane, kp1);
        float r0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y);
        float r1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y);
        __global const KV_T *vp0 = vbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *vp1 = vbuf + slot1 * kv_stride + kvh * head_dim;
        float s0 = r0 * scale;
        float m_new = fmax(m, s0);
        float corr = exp(m - m_new);
        float wgt = exp(s0 - m_new);
        l = l * corr + wgt;
        float2 v0 = kv_load2(lane, vp0);
        acc0 = acc0 * corr + wgt * v0.x;
        acc1 = acc1 * corr + wgt * v0.y;
        m = m_new;
        float s1 = r1 * scale;
        m_new = fmax(m, s1);
        corr = exp(m - m_new);
        wgt = exp(s1 - m_new);
        l = l * corr + wgt;
        float2 v1 = kv_load2(lane, vp1);
        acc0 = acc0 * corr + wgt * v1.x;
        acc1 = acc1 * corr + wgt * v1.y;
        m = m_new;
    }
    if (pos < len) {
        ulong slots = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        __global const KV_T *kp = kbuf + slots * kv_stride + kvh * head_dim;
        float2 k = kv_load2(lane, kp);
        float rv = sub_group_reduce_add(q0 * k.x + q1 * k.y);
        float sv = rv * scale;
        float m_new = fmax(m, sv);
        float corr = exp(m - m_new);
        float wgt = exp(sv - m_new);
        l = l * corr + wgt;
        __global const KV_T *vp = vbuf + slots * kv_stride + kvh * head_dim;
        float2 v = kv_load2(lane, vp);
        acc0 = acc0 * corr + wgt * v.x;
        acc1 = acc1 * corr + wgt * v.y;
        m = m_new;
    }
    float gs = 1.0f / (1.0f + exp(-gate[t * n_heads + head]));
    __global float *o = out + (ulong)t * n_heads * head_dim + head * head_dim;
    o[2u * lane] = (acc0 / l) * gs;
    o[2u * lane + 1u] = (acc1 / l) * gs;
}

__kernel void attention_split_b_h(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global float *pacc, __global float *pml, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits) {
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
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    __global const float *qh = q + (ulong)t * n_heads * head_dim + head * head_dim;
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
        ulong slot0 = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        ulong slot1 = (ulong)((tbl[toff + ((pos + 1u) >> 5)] << 5) | ((pos + 1u) & 31u));
        __global const KV_T *kp0 = kbuf + slot0 * kv_stride + kvh * head_dim;
        __global const KV_T *kp1 = kbuf + slot1 * kv_stride + kvh * head_dim;
        float2 k0 = kv_load2(lane, kp0);
        float2 k1 = kv_load2(lane, kp1);
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
            sred[64] = sred[64] + sred[65] + sred[66] + sred[67] + sred[68] + sred[69] + sred[70] + sred[71];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float s0 = sred[0] * scale;
        float s1 = sred[64u] * scale;
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
        ulong slots = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        __global const KV_T *kp = kbuf + slots * kv_stride + kvh * head_dim;
        float2 k = kv_load2(lane, kp);
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
    ulong pbase = ((ulong)t * n_heads + head) * splits + sp;
    __global float *pa = pacc + pbase * head_dim;
    pa[2u * lane] = acc0;
    pa[2u * lane + 1u] = acc1;
    if (lane == 0u) {
        pml[pbase * 2u] = m;
        pml[pbase * 2u + 1u] = l;
    }
}

__kernel void gemv_q4_k_i8_4p_b(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y1, __global float *y2, __global float *y3, __global float *y4, uint r1, uint r2, uint r3, uint r4, uint cols, uint ntok) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint rows = r1 + r2 + r3 + r4;
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV4P;
    if (rows < RGEMV4P) {
        return;
    }
    if (row0 + RGEMV4P > rows) {
        row0 = rows - RGEMV4P;
    }
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 148ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
    const char4 ones = (char4)(1);
    float acc[PB][RGEMV4P];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV4P; ++r) {
            acc[t][r] = 0.0f;
        }
    }
    for (uint b0 = 0u; b0 < blocks; b0 += bpi) {
        uint b = b0 + (lane >> 3);
        if (b < blocks) {
            uint w_lo = b * 256u + wbase;
            uint ski = w_lo >> 5;
            uint ski2 = (w_lo + 32u) >> 5;
            char16 xl[PB];
            char16 xh[PB];
            float xsl[PB];
            float xsh[PB];
            int suml[PB];
            int sumh[PB];
            #pragma unroll
            for (uint t = 0u; t < PB; ++t) {
                xl[t] = (char16)(0);
                xh[t] = (char16)(0);
                xsl[t] = 0.0f;
                xsh[t] = 0.0f;
                suml[t] = 0;
                sumh[t] = 0;
                if (t < ntok) {
                    xl[t] = vload16(0, xq + (ulong)t * cols + w_lo);
                    xh[t] = vload16(0, xq + (ulong)t * cols + w_lo + 32u);
                    xsl[t] = xs[t * ngs + ski];
                    xsh[t] = xs[t * ngs + ski2];
                    suml[t] = (int)LFM_DOT(xl[t].s0123, ones) + (int)LFM_DOT(xl[t].s4567, ones)
                            + (int)LFM_DOT(xl[t].s89ab, ones) + (int)LFM_DOT(xl[t].scdef, ones);
                    sumh[t] = (int)LFM_DOT(xh[t].s0123, ones) + (int)LFM_DOT(xh[t].s4567, ones)
                            + (int)LFM_DOT(xh[t].s89ab, ones) + (int)LFM_DOT(xh[t].scdef, ones);
                }
            }
            #pragma unroll
            for (uint r = 0u; r < RGEMV4P; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 148ul;
                float d = lfm_scale((__global const ushort *)blk);
                float dmin = lfm_scale((__global const ushort *)(blk + 2));
                uchar4 sm = vload4(0, blk + 4u + 4u * (u >> 1));
                uint4 pk = vload4(0, (__global const uint *)(blk + 20u + u * 16u));
                char4 l0 = as_char4(pk.x & 0x0F0F0F0Fu);
                char4 l1 = as_char4(pk.y & 0x0F0F0F0Fu);
                char4 l2 = as_char4(pk.z & 0x0F0F0F0Fu);
                char4 l3 = as_char4(pk.w & 0x0F0F0F0Fu);
                char4 h0 = as_char4((pk.x >> 4) & 0x0F0F0F0Fu);
                char4 h1 = as_char4((pk.y >> 4) & 0x0F0F0F0Fu);
                char4 h2 = as_char4((pk.z >> 4) & 0x0F0F0F0Fu);
                char4 h3 = as_char4((pk.w >> 4) & 0x0F0F0F0Fu);
                float dl_coef = d * (float)sm.x;
                float dh_coef = d * (float)sm.z;
                float ml_coef = dmin * (float)sm.y;
                float mh_coef = dmin * (float)sm.w;
                #pragma unroll
                for (uint t = 0u; t < PB; ++t) {
                    if (t < ntok) {
                        float dl = (float)(LFM_DOT(l0, xl[t].s0123) + LFM_DOT(l1, xl[t].s4567)
                                         + LFM_DOT(l2, xl[t].s89ab) + LFM_DOT(l3, xl[t].scdef));
                        float dh = (float)(LFM_DOT(h0, xh[t].s0123) + LFM_DOT(h1, xh[t].s4567)
                                         + LFM_DOT(h2, xh[t].s89ab) + LFM_DOT(h3, xh[t].scdef));
                        acc[t][r] += xsl[t] * (dl * dl_coef - (float)suml[t] * ml_coef);
                        acc[t][r] += xsh[t] * (dh * dh_coef - (float)sumh[t] * mh_coef);
                    }
                }
            }
        }
    }
    __global float *yd = y4;
    uint stride = r4;
    uint rofs = row0 - r1 - r2 - r3;
    if (row0 < r1) {
        yd = y1;
        stride = r1;
        rofs = row0;
    } else if (row0 < r1 + r2) {
        yd = y2;
        stride = r2;
        rofs = row0 - r1;
    } else if (row0 < r1 + r2 + r3) {
        yd = y3;
        stride = r3;
        rofs = row0 - r1 - r2;
    }
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        if (t < ntok) {
            #pragma unroll
            for (uint r = 0u; r < RGEMV4P; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u) {
                    yd[(ulong)t * stride + rofs + r] = total;
                }
            }
        }
    }
}


__kernel void gemm_q6_k_i8(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * 2u;
    if (rows < 2u) {
        return;
    }
    if (row0 + 2u > rows) {
        row0 = rows - 2u;
    }
    uint sub = lane & 15u;
    uint n = sub >> 3;
    uint l = (sub & 7u) * 4u;
    uint lq = l >> 4;
    uint bps = sgs >> 4;
    if (bps == 0u) {
        bps = 1u;
    }
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 210ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    for (uint t0 = 0u; t0 < ntok; t0 += 16u) {
        float acc[2][16];
        #pragma unroll
        for (uint r = 0u; r < 2u; ++r) {
            #pragma unroll
            for (uint tt = 0u; tt < 16u; ++tt) {
                acc[r][tt] = 0.0f;
            }
        }
        for (uint b0 = 0u; b0 < blocks; b0 += bps) {
            uint b = b0 + (lane >> 4);
            if (b < blocks) {
                char4 q0[2];
                char4 q1[2];
                char4 q2[2];
                char4 q3[2];
                float s0[2];
                float s1[2];
                float s2[2];
                float s3[2];
                #pragma unroll
                for (uint r = 0u; r < 2u; ++r) {
                    __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 210ul;
                    __global const char *sc = (__global const char *)(blk + 192u);
                    float d = lfm_scale((__global const ushort *)(blk + 208u));
                    uint so = n * 8u + lq;
                    s0[r] = d * (float)sc[so + 0u];
                    s1[r] = d * (float)sc[so + 2u];
                    s2[r] = d * (float)sc[so + 4u];
                    s3[r] = d * (float)sc[so + 6u];
                    __global const uchar *ql = blk + n * 64u + l;
                    __global const uchar *qh = blk + 128u + n * 32u + l;
                    uint lo = as_uint(vload4(0, ql));
                    uint hi = as_uint(vload4(0, ql + 32u));
                    uint hb = as_uint(vload4(0, qh));
                    q0[r] = as_char4((lo & 0x0F0F0F0Fu) | ((hb & 0x03030303u) << 4)) - (char4)(32);
                    q1[r] = as_char4((hi & 0x0F0F0F0Fu) | (((hb >> 2) & 0x03030303u) << 4)) - (char4)(32);
                    q2[r] = as_char4(((lo >> 4) & 0x0F0F0F0Fu) | (((hb >> 4) & 0x03030303u) << 4)) - (char4)(32);
                    q3[r] = as_char4(((hi >> 4) & 0x0F0F0F0Fu) | (((hb >> 6) & 0x03030303u) << 4)) - (char4)(32);
                }
                uint w0 = b * 256u + n * 128u + l;
                uint xg = w0 >> 5;
                #pragma unroll
                for (uint tt = 0u; tt < 16u; ++tt) {
                    uint t = t0 + tt;
                    if (t < ntok) {
                        char4 x0 = vload4(0, xq + (ulong)t * cols + w0);
                        char4 x1 = vload4(0, xq + (ulong)t * cols + w0 + 32u);
                        char4 x2 = vload4(0, xq + (ulong)t * cols + w0 + 64u);
                        char4 x3 = vload4(0, xq + (ulong)t * cols + w0 + 96u);
                        float xs0 = xs[t * ngs + xg + 0u];
                        float xs1 = xs[t * ngs + xg + 1u];
                        float xs2 = xs[t * ngs + xg + 2u];
                        float xs3 = xs[t * ngs + xg + 3u];
                        #pragma unroll
                        for (uint r = 0u; r < 2u; ++r) {
                            acc[r][tt] += xs0 * s0[r] * (float)LFM_DOT(q0[r], x0);
                            acc[r][tt] += xs1 * s1[r] * (float)LFM_DOT(q1[r], x1);
                            acc[r][tt] += xs2 * s2[r] * (float)LFM_DOT(q2[r], x2);
                            acc[r][tt] += xs3 * s3[r] * (float)LFM_DOT(q3[r], x3);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (uint r = 0u; r < 2u; ++r) {
            #pragma unroll
            for (uint tt = 0u; tt < 16u; ++tt) {
                uint t = t0 + tt;
                if (t < ntok) {
                    float total = sub_group_reduce_add(acc[r][tt]);
                    if (lane == 0u) {
                        y[(ulong)t * rows + row0 + r] = total;
                    }
                }
            }
        }
    }
}

#ifndef RT
#define RT 32
#endif
#ifndef TT
#define TT 32
#endif

__kernel void gemm_q6_i8_pp(__global const char *w, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lid = get_local_id(0);
    uint tps = (ntok + 31u) / 32u;
    uint gid = get_group_id(0);
    uint row0 = (gid / tps) * 32u;
    uint t0 = (gid % tps) * 32u;
    if (row0 >= rows) {
        return;
    }
    uint valid_rows = min(32u, rows - row0);
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char wq[8 * 32 * 32];
    __local float wqsc[8 * 32];
    __local char xt[32 * 256];
    uint lane = lid & 31u;
    uint warp = lid >> 5;
    uint tok_base = warp * 4u;
    uint lrow = lid & 31u;
    uint lg = lid >> 5;
    uint xt_t = lid >> 3;
    uint xt_off = (lid & 7u) * 32u;
    float acc[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        acc[j] = 0.0f;
    }
    for (uint b = 0u; b < blocks; ++b) {
        {
            uchar16 a = (uchar16)(0);
            uchar16 c = (uchar16)(0);
            if (lrow < valid_rows) {
                __global const char *src = w + ((ulong)(b * 8u + lg) * rows + row0 + lrow) * 32u;
                a = as_uchar16(vload16(0, src));
                c = as_uchar16(vload16(0, src + 16u));
            }
            vstore16(a, 0, (__local uchar *)(wq + (lg * 32u + lrow) * 32u));
            vstore16(c, 0, (__local uchar *)(wq + (lg * 32u + lrow) * 32u + 16u));
        }
        if (lid < 256u) {
            float v = 0.0f;
            if (lrow < valid_rows) {
                v = wsc[(ulong)(b * 8u + lg) * rows + row0 + lrow];
            }
            wqsc[lg * 32u + lrow] = v;
        }
        {
            uint token = t0 + xt_t;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            if (token < ntok) {
                __global const char *src = xq + (ulong)token * cols + (ulong)b * 256u + xt_off;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
            }
            vstore16(a, 0, xt + xt_t * 256u + xt_off);
            vstore16(c, 0, xt + xt_t * 256u + xt_off + 16u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < valid_rows && t0 + tok_base < ntok) {
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                __local const char *wp = wq + (g * 32u + lane) * 32u;
                char16 wa = vload16(0, wp);
                char16 wb = vload16(0, wp + 16u);
                float ws = wqsc[g * 32u + lane];
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    uint t = tok_base + j;
                    uint token = t0 + t;
                    if (token < ntok) {
                        __local const char *xp = xt + t * 256u + g * 32u;
                        char16 xa = vload16(0, xp);
                        char16 xb = vload16(0, xp + 16u);
                        float d = (float)(LFM_DOT(wa.s0123, xa.s0123) + LFM_DOT(wa.s4567, xa.s4567)
                                        + LFM_DOT(wa.s89ab, xa.s89ab) + LFM_DOT(wa.scdef, xa.scdef)
                                        + LFM_DOT(wb.s0123, xb.s0123) + LFM_DOT(wb.s4567, xb.s4567)
                                        + LFM_DOT(wb.s89ab, xb.s89ab) + LFM_DOT(wb.scdef, xb.scdef));
                        acc[j] += xs[(ulong)token * ngs + (ulong)b * 8u + g] * ws * d;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane < valid_rows) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint t = tok_base + j;
            if (t0 + t < ntok) {
                y[(ulong)(t0 + t) * rows + row0 + lane] = acc[j];
            }
        }
    }
}

__kernel void attention_split_b_fa(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global float *pacc, __global float *pml, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint per_tok = kv_heads * splits;
    uint t = gid / per_tok;
    uint rem = gid % per_tok;
    uint kvh = rem / splits;
    uint sp = rem % splits;
    if (t >= ntok) {
        return;
    }
    uint h = lane >> 4;
    uint j = lane & 15u;
    uint head = kvh * (n_heads / kv_heads) + h;
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    float8 qv = vload8(0, q + ((ulong)t * n_heads + head) * head_dim + j * 8u);
    float m = -INFINITY;
    float l = 0.0f;
    float8 acc = (float8)(0.0f);
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint koff = kvh * head_dim + j * 8u;
    uint pos = p0;
    for (; pos + 3u < p1; pos += 4u) {
        float2 s[4];
        float8 v[4];
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            ulong slot = (ulong)((tbl[toff + ((pos + u) >> 5)] << 5) | ((pos + u) & 31u));
            float8 kv = convert_float8(vload8(0, kbuf + slot * kv_stride + koff));
            v[u] = convert_float8(vload8(0, vbuf + slot * kv_stride + koff));
            float d = qv.s0 * kv.s0 + qv.s1 * kv.s1 + qv.s2 * kv.s2 + qv.s3 * kv.s3
                    + qv.s4 * kv.s4 + qv.s5 * kv.s5 + qv.s6 * kv.s6 + qv.s7 * kv.s7;
            d += sub_group_shuffle_xor(d, 8u);
            d += sub_group_shuffle_xor(d, 4u);
            d += sub_group_shuffle_xor(d, 2u);
            d += sub_group_shuffle_xor(d, 1u);
            s[u] = (float2)(d * scale, 0.0f);
        }
        float m_new = fmax(fmax(m, s[0].x), fmax(fmax(s[1].x, s[2].x), s[3].x));
        float corr = exp(m - m_new);
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            float w = exp(s[u].x - m_new);
            l = l * (u == 0u ? corr : 1.0f) + w;
            acc = acc * (u == 0u ? corr : 1.0f) + w * v[u];
        }
        m = m_new;
    }
    for (; pos < p1; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        float8 kv = convert_float8(vload8(0, kbuf + slot * kv_stride + koff));
        float8 vv = convert_float8(vload8(0, vbuf + slot * kv_stride + koff));
        float d = qv.s0 * kv.s0 + qv.s1 * kv.s1 + qv.s2 * kv.s2 + qv.s3 * kv.s3
                + qv.s4 * kv.s4 + qv.s5 * kv.s5 + qv.s6 * kv.s6 + qv.s7 * kv.s7;
        d += sub_group_shuffle_xor(d, 8u);
        d += sub_group_shuffle_xor(d, 4u);
        d += sub_group_shuffle_xor(d, 2u);
        d += sub_group_shuffle_xor(d, 1u);
        float sv = d * scale;
        float m_new = fmax(m, sv);
        float corr = exp(m - m_new);
        float w = exp(sv - m_new);
        l = l * corr + w;
        acc = acc * corr + w * vv;
        m = m_new;
    }
    ulong pbase = ((ulong)t * n_heads + head) * splits + sp;
    __global float *pa = pacc + pbase * head_dim;
    vstore8(acc, 0, pa + j * 8u);
    if (j == 0u) {
        pml[pbase * 2u] = m;
        pml[pbase * 2u + 1u] = l;
    }
}

__kernel void attention_split_b_fa2(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global float *pacc, __global float *pml, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint per_tok = kv_heads * splits;
    uint tgroup = gid / per_tok;
    uint rem = gid % per_tok;
    uint kvh = rem / splits;
    uint sp = rem % splits;
    uint s = lane >> 5;
    uint hh = (lane >> 3) & 3u;
    uint j = lane & 7u;
    uint t = tgroup * 2u + s;
    bool active = t < ntok;
    uint t0 = tgroup * 2u;
    uint gstart = 0xFFFFFFFFu;
    uint gend = 0u;
    #pragma unroll
    for (uint v = 0u; v < 2u; ++v) {
        uint tt = t0 + v;
        if (tt < ntok) {
            uint ln = pos_b[tt] + 1u;
            uint st = (window > 0u && ln > window) ? ln - window : 0u;
            gstart = min(gstart, st);
            gend = max(gend, ln);
        }
    }
    if (gend <= gstart) {
        gend = gstart + 1u;
    }
    uint len = 0u;
    uint start = 0u;
    if (active) {
        len = pos_b[t] + 1u;
        start = (window > 0u && len > window) ? len - window : 0u;
    }
    uint span = gend - gstart;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = gstart + sp * chunk;
    uint p1 = min(p0 + chunk, gend);
    uint toff = tbl_off[active ? t : t0];
    uint head = kvh * 4u + hh;
    __global const float *qp = q + ((ulong)(t0 + s) * n_heads + head) * head_dim + j * 16u;
    float16 qv = (float16)(0.0f);
    if (active) {
        qv = vload16(0, qp);
    }
    float m = -INFINITY;
    float l = 0.0f;
    float16 acc = (float16)(0.0f);
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint koff = kvh * head_dim + j * 16u;
    uint pos = p0;
    for (; pos + 3u < p1; pos += 4u) {
        float sv[4];
        float16 vv[4];
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            ulong slot = (ulong)((tbl[toff + ((pos + u) >> 5)] << 5) | ((pos + u) & 31u));
            KV_V16 kh = kv_load16(0, kbuf + slot * kv_stride + koff);
            vv[u] = convert_float16(kv_load16(0, vbuf + slot * kv_stride + koff));
            float d = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 16u; ++i) {
                d = fma(qv[i], (float)kh[i], d);
            }
            d += sub_group_shuffle_xor(d, 4u);
            d += sub_group_shuffle_xor(d, 2u);
            d += sub_group_shuffle_xor(d, 1u);
            sv[u] = d * scale;
        }
        float mn = m;
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            uint pp = pos + u;
            if (active && pp >= start && pp < len) {
                mn = fmax(mn, sv[u]);
            }
        }
        float corr = exp(m - mn);
        l *= corr;
        acc *= corr;
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            uint pp = pos + u;
            if (active && pp >= start && pp < len) {
                float w = exp(sv[u] - mn);
                l += w;
                acc = fma(w, vv[u], acc);
            }
        }
        m = mn;
    }
    for (; pos < p1; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        KV_V16 kh = kv_load16(0, kbuf + slot * kv_stride + koff);
        float16 vv = convert_float16(kv_load16(0, vbuf + slot * kv_stride + koff));
        float d = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) {
            d = fma(qv[i], (float)kh[i], d);
        }
        d += sub_group_shuffle_xor(d, 4u);
        d += sub_group_shuffle_xor(d, 2u);
        d += sub_group_shuffle_xor(d, 1u);
        if (active && pos >= start && pos < len) {
            float sv = d * scale;
            float mn = fmax(m, sv);
            float corr = exp(m - mn);
            float w = exp(sv - mn);
            l = l * corr + w;
            acc = fma(w, vv, acc * corr);
            m = mn;
        }
    }
    if (active) {
        ulong pbase = ((ulong)t * n_heads + head) * splits + sp;
        __global float *pa = pacc + pbase * head_dim;
        vstore16(acc, 0, pa + j * 16u);
        if (j == 0u) {
            pml[pbase * 2u] = m;
            pml[pbase * 2u + 1u] = l;
        }
    }
}

__kernel void attention_split_b_h4_p4(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf, __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b, __global float *pacc, __global float *pml, uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint per_tok = kv_heads * splits;
    uint t = gid / per_tok;
    uint rem = gid % per_tok;
    uint kvh = rem / splits;
    uint sp = rem % splits;
    if (t >= ntok) {
        return;
    }
    __local float sred[16 * 64];
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    uint head0 = kvh * (n_heads / kv_heads);
    float q0[4];
    float q1[4];
    float m[4];
    float l[4];
    float acc0[4];
    float acc1[4];
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        __global const float *qh = q + ((ulong)t * n_heads + head0 + g) * head_dim;
        q0[g] = qh[2u * lane];
        q1[g] = qh[2u * lane + 1u];
        m[g] = -INFINITY;
        l[g] = 0.0f;
        acc0[g] = 0.0f;
        acc1[g] = 0.0f;
    }
    float scale = 1.0f / sqrt((float)head_dim);
    uint kv_stride = kv_heads * head_dim;
    uint pos = p0;
    for (; pos + 3u < p1; pos += 4u) {
        float2 k[4];
        float2 v[4];
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            ulong slot = (ulong)((tbl[toff + ((pos + u) >> 5)] << 5) | ((pos + u) & 31u));
            k[u] = kv_load2(lane, kbuf + slot * kv_stride + kvh * head_dim);
            v[u] = kv_load2(lane, vbuf + slot * kv_stride + kvh * head_dim);
        }
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            #pragma unroll
            for (uint u = 0u; u < 4u; ++u) {
                sred[(u * 4u + g) * 64u + lane] = q0[g] * k[u].x + q1[g] * k[u].y;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                sred[j * 64u + lane] = sred[j * 64u + lane] + sred[j * 64u + lane + 8u]
                    + sred[j * 64u + lane + 16u] + sred[j * 64u + lane + 24u]
                    + sred[j * 64u + lane + 32u] + sred[j * 64u + lane + 40u]
                    + sred[j * 64u + lane + 48u] + sred[j * 64u + lane + 56u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                sred[j * 64u] = sred[j * 64u] + sred[j * 64u + 1u] + sred[j * 64u + 2u]
                    + sred[j * 64u + 3u] + sred[j * 64u + 4u] + sred[j * 64u + 5u]
                    + sred[j * 64u + 6u] + sred[j * 64u + 7u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            float s0 = sred[(0u * 4u + g) * 64u] * scale;
            float s1 = sred[(1u * 4u + g) * 64u] * scale;
            float s2 = sred[(2u * 4u + g) * 64u] * scale;
            float s3 = sred[(3u * 4u + g) * 64u] * scale;
            float m_new = fmax(fmax(m[g], s0), fmax(s2, s3));
            float corr = exp(m[g] - m_new);
            float w0 = exp(s0 - m_new);
            float w1 = exp(s1 - m_new);
            float w2 = exp(s2 - m_new);
            float w3 = exp(s3 - m_new);
            l[g] = l[g] * corr + w0 + w1 + w2 + w3;
            acc0[g] = acc0[g] * corr + w0 * v[0].x + w1 * v[1].x + w2 * v[2].x + w3 * v[3].x;
            acc1[g] = acc1[g] * corr + w0 * v[0].y + w1 * v[1].y + w2 * v[2].y + w3 * v[3].y;
            m[g] = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; pos < p1; ++pos) {
        ulong slot = (ulong)((tbl[toff + (pos >> 5)] << 5) | (pos & 31u));
        float2 k = kv_load2(lane, kbuf + slot * kv_stride + kvh * head_dim);
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            sred[g * 64u + lane] = q0[g] * k.x + q1[g] * k.y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8u) {
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                sred[j * 64u + lane] = sred[j * 64u + lane] + sred[j * 64u + lane + 8u]
                    + sred[j * 64u + lane + 16u] + sred[j * 64u + lane + 24u]
                    + sred[j * 64u + lane + 32u] + sred[j * 64u + lane + 40u]
                    + sred[j * 64u + lane + 48u] + sred[j * 64u + lane + 56u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == 0u) {
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                sred[j * 64u] = sred[j * 64u] + sred[j * 64u + 1u] + sred[j * 64u + 2u]
                    + sred[j * 64u + 3u] + sred[j * 64u + 4u] + sred[j * 64u + 5u]
                    + sred[j * 64u + 6u] + sred[j * 64u + 7u];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float2 v = kv_load2(lane, vbuf + slot * kv_stride + kvh * head_dim);
        #pragma unroll
        for (uint g = 0u; g < 4u; ++g) {
            float sv = sred[g * 64u] * scale;
            float m_new = fmax(m[g], sv);
            float corr = exp(m[g] - m_new);
            float w = exp(sv - m_new);
            l[g] = l[g] * corr + w;
            acc0[g] = acc0[g] * corr + w * v.x;
            acc1[g] = acc1[g] * corr + w * v.y;
            m[g] = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        ulong pbase = ((ulong)t * n_heads + head0 + g) * splits + sp;
        __global float *pa = pacc + pbase * head_dim;
        pa[2u * lane] = acc0[g];
        pa[2u * lane + 1u] = acc1[g];
        if (lane == 0u) {
            pml[pbase * 2u] = m[g];
            pml[pbase * 2u + 1u] = l[g];
        }
    }
}
__kernel void gemm_q4_k_i8_b8(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint row = gid * 64u + lane;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 148ul;
    __local char xt[8 * 256];
    __local int xsum[8 * 8];
    __local uchar ws[64 * 16];
    __local uchar wh[64 * 4];
    const char4 ones = (char4)(1);
    const uint M = 0x0F0F0F0Fu;
    uint st = lane >> 3;
    uint sc = lane & 7u;
    float acc[8];
    #pragma unroll
    for (uint t = 0u; t < 8u; ++t) {
        acc[t] = 0.0f;
    }
    __global const uchar *base = w + (ulong)row * row_stride;
    for (uint b = 0u; b < blocks; ++b) {
        {
            uint token = st;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            if (token < ntok) {
                __global const char *src = xq + (ulong)token * cols + (ulong)b * 256u + sc * 32u;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
            }
            vstore16(a, 0, xt + token * 256u + sc * 32u);
            vstore16(c, 0, xt + token * 256u + sc * 32u + 16u);
            int s0 = (int)LFM_DOT(a.s0123, ones) + (int)LFM_DOT(a.s4567, ones)
                   + (int)LFM_DOT(a.s89ab, ones) + (int)LFM_DOT(a.scdef, ones)
                   + (int)LFM_DOT(c.s0123, ones) + (int)LFM_DOT(c.s4567, ones)
                   + (int)LFM_DOT(c.s89ab, ones) + (int)LFM_DOT(c.scdef, ones);
            xsum[token * 8u + sc] = s0;
        }
        if (row < rows) {
            uchar16 sv = vload16(0, base + (ulong)b * 148ul + 4ul);
            vstore16(sv, 0, ws + lane * 16u);
            uchar4 hv = vload4(0, base + (ulong)b * 148ul);
            vstore4(hv, 0, wh + lane * 4u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < rows) {
            __local const uchar *srow = ws + lane * 16u;
            float d = lfm_scale_local(wh + lane * 4u);
            float dmin = lfm_scale_local(wh + lane * 4u + 2u);
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                uint4 pa = vload4(0, (__global const uint *)(base + (ulong)b * 148ul + 20ul + q * 32u));
                uint4 pb = vload4(0, (__global const uint *)(base + (ulong)b * 148ul + 20ul + q * 32u + 16u));
                uint lo0 = pa.x & M, lo1 = pa.y & M, lo2 = pa.z & M, lo3 = pa.w & M;
                uint lo4 = pb.x & M, lo5 = pb.y & M, lo6 = pb.z & M, lo7 = pb.w & M;
                uint hi0 = (pa.x >> 4) & M, hi1 = (pa.y >> 4) & M, hi2 = (pa.z >> 4) & M, hi3 = (pa.w >> 4) & M;
                uint hi4 = (pb.x >> 4) & M, hi5 = (pb.y >> 4) & M, hi6 = (pb.z >> 4) & M, hi7 = (pb.w >> 4) & M;
                float scl = d * (float)srow[4u * q], mnl = dmin * (float)srow[4u * q + 1u];
                float sch = d * (float)srow[4u * q + 2u], mnh = dmin * (float)srow[4u * q + 3u];
                #pragma unroll
                for (uint t = 0u; t < 8u; ++t) {
                    if (t < ntok) {
                        __local const uint *xp = (__local const uint *)(xt + t * 256u + q * 64u);
                        uint x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                        uint x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
                        uint y0 = xp[8], y1 = xp[9], y2 = xp[10], y3 = xp[11];
                        uint y4 = xp[12], y5 = xp[13], y6 = xp[14], y7 = xp[15];
                        int dl = 0, dh = 0;
                        dl = lfm_dot4(lo0, x0, dl); dl = lfm_dot4(lo1, x1, dl); dl = lfm_dot4(lo2, x2, dl); dl = lfm_dot4(lo3, x3, dl);
                        dl = lfm_dot4(lo4, x4, dl); dl = lfm_dot4(lo5, x5, dl); dl = lfm_dot4(lo6, x6, dl); dl = lfm_dot4(lo7, x7, dl);
                        dh = lfm_dot4(hi0, y0, dh); dh = lfm_dot4(hi1, y1, dh); dh = lfm_dot4(hi2, y2, dh); dh = lfm_dot4(hi3, y3, dh);
                        dh = lfm_dot4(hi4, y4, dh); dh = lfm_dot4(hi5, y5, dh); dh = lfm_dot4(hi6, y6, dh); dh = lfm_dot4(hi7, y7, dh);
                        int suml = xsum[t * 8u + 2u * q];
                        int sumh = xsum[t * 8u + 2u * q + 1u];
                        ulong g = (ulong)t * ngs + (ulong)b * 8u + 2u * q;
                        acc[t] += xs[g] * ((float)dl * scl - (float)suml * mnl);
                        acc[t] += xs[g + 1u] * ((float)dh * sch - (float)sumh * mnh);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < ntok) {
                y[(ulong)t * rows + row] = acc[t];
            }
        }
    }
}

__kernel void gemm_q4_k_i8_b8p(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint row = gid * 64u + lane;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 148ul;
    __local char xt[2 * 8 * 256];
    __local int xsum[2 * 8 * 8];
    __local uchar ws[2 * 64 * 16];
    __local uchar wh[2 * 64 * 4];
    const char4 ones = (char4)(1);
    const uint M = 0x0F0F0F0Fu;
    uint st = lane >> 3;
    uint sc = lane & 7u;
    bool stok = st < ntok;
    float acc[8];
    #pragma unroll
    for (uint t = 0u; t < 8u; ++t) {
        acc[t] = 0.0f;
    }
    __global const uchar *base = w + (ulong)row * row_stride;
    bool active = row < rows;
    char16 pa = (char16)(0);
    char16 pc = (char16)(0);
    int psn = 0;
    if (stok) {
        __global const char *src = xq + (ulong)st * cols + sc * 32u;
        pa = vload16(0, src);
        pc = vload16(0, src + 16u);
        psn = (int)LFM_DOT(pa.s0123, ones) + (int)LFM_DOT(pa.s4567, ones)
            + (int)LFM_DOT(pa.s89ab, ones) + (int)LFM_DOT(pa.scdef, ones)
            + (int)LFM_DOT(pc.s0123, ones) + (int)LFM_DOT(pc.s4567, ones)
            + (int)LFM_DOT(pc.s89ab, ones) + (int)LFM_DOT(pc.scdef, ones);
    }
    uchar16 pws = (uchar16)(0);
    uchar4 pwh = (uchar4)(0);
    if (active) {
        pws = vload16(0, base + 4ul);
        pwh = vload4(0, base);
    }
    vstore16(pa, 0, xt + st * 256u + sc * 32u);
    vstore16(pc, 0, xt + st * 256u + sc * 32u + 16u);
    xsum[st * 8u + sc] = psn;
    if (active) {
        vstore16(pws, 0, ws + lane * 16u);
        vstore4(pwh, 0, wh + lane * 4u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = 0u; b < blocks; ++b) {
        uint cur = b & 1u;
        if (b + 1u < blocks) {
            uint nb = 1u - cur;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            int sn = 0;
            if (stok) {
                __global const char *src = xq + (ulong)st * cols + (ulong)(b + 1u) * 256u + sc * 32u;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
                sn = (int)LFM_DOT(a.s0123, ones) + (int)LFM_DOT(a.s4567, ones)
                   + (int)LFM_DOT(a.s89ab, ones) + (int)LFM_DOT(a.scdef, ones)
                   + (int)LFM_DOT(c.s0123, ones) + (int)LFM_DOT(c.s4567, ones)
                   + (int)LFM_DOT(c.s89ab, ones) + (int)LFM_DOT(c.scdef, ones);
            }
            uchar16 sws = (uchar16)(0);
            uchar4 swh = (uchar4)(0);
            if (active) {
                sws = vload16(0, base + (ulong)(b + 1u) * 148ul + 4ul);
                swh = vload4(0, base + (ulong)(b + 1u) * 148ul);
            }
            vstore16(a, 0, xt + nb * 2048u + st * 256u + sc * 32u);
            vstore16(c, 0, xt + nb * 2048u + st * 256u + sc * 32u + 16u);
            xsum[nb * 64u + st * 8u + sc] = sn;
            if (active) {
                vstore16(sws, 0, ws + nb * 1024u + lane * 16u);
                vstore4(swh, 0, wh + nb * 256u + lane * 4u);
            }
        }
        if (active) {
            __local const uchar *srow = ws + cur * 1024u + lane * 16u;
            __local const char *xb = xt + cur * 2048u;
            __local const int *sm = xsum + cur * 64u;
            float d = lfm_scale_local(wh + cur * 256u + lane * 4u);
            float dmin = lfm_scale_local(wh + cur * 256u + lane * 4u + 2u);
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                uint4 wa = vload4(0, (__global const uint *)(base + (ulong)b * 148ul + 20ul + q * 32u));
                uint4 wb = vload4(0, (__global const uint *)(base + (ulong)b * 148ul + 20ul + q * 32u + 16u));
                uint lo0 = wa.x & M, lo1 = wa.y & M, lo2 = wa.z & M, lo3 = wa.w & M;
                uint lo4 = wb.x & M, lo5 = wb.y & M, lo6 = wb.z & M, lo7 = wb.w & M;
                uint hi0 = (wa.x >> 4) & M, hi1 = (wa.y >> 4) & M, hi2 = (wa.z >> 4) & M, hi3 = (wa.w >> 4) & M;
                uint hi4 = (wb.x >> 4) & M, hi5 = (wb.y >> 4) & M, hi6 = (wb.z >> 4) & M, hi7 = (wb.w >> 4) & M;
                float scl = d * (float)srow[4u * q], mnl = dmin * (float)srow[4u * q + 1u];
                float sch = d * (float)srow[4u * q + 2u], mnh = dmin * (float)srow[4u * q + 3u];
                #pragma unroll
                for (uint t = 0u; t < 8u; ++t) {
                    if (t < ntok) {
                        __local const uint *xp = (__local const uint *)(xb + t * 256u + q * 64u);
                        uint x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                        uint x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
                        uint y0 = xp[8], y1 = xp[9], y2 = xp[10], y3 = xp[11];
                        uint y4 = xp[12], y5 = xp[13], y6 = xp[14], y7 = xp[15];
                        int dl = 0, dh = 0;
                        dl = lfm_dot4(lo0, x0, dl); dl = lfm_dot4(lo1, x1, dl); dl = lfm_dot4(lo2, x2, dl); dl = lfm_dot4(lo3, x3, dl);
                        dl = lfm_dot4(lo4, x4, dl); dl = lfm_dot4(lo5, x5, dl); dl = lfm_dot4(lo6, x6, dl); dl = lfm_dot4(lo7, x7, dl);
                        dh = lfm_dot4(hi0, y0, dh); dh = lfm_dot4(hi1, y1, dh); dh = lfm_dot4(hi2, y2, dh); dh = lfm_dot4(hi3, y3, dh);
                        dh = lfm_dot4(hi4, y4, dh); dh = lfm_dot4(hi5, y5, dh); dh = lfm_dot4(hi6, y6, dh); dh = lfm_dot4(hi7, y7, dh);
                        int suml = sm[t * 8u + 2u * q];
                        int sumh = sm[t * 8u + 2u * q + 1u];
                        ulong g = (ulong)t * ngs + (ulong)b * 8u + 2u * q;
                        acc[t] += xs[g] * ((float)dl * scl - (float)suml * mnl);
                        acc[t] += xs[g + 1u] * ((float)dh * sch - (float)sumh * mnh);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < ntok) {
                y[(ulong)t * rows + row] = acc[t];
            }
        }
    }
}

__kernel void gemm_q6_i8_b8(__global const char *w, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint row = gid * 64u + lane;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char xt[8 * 256];
    uint st = lane >> 3;
    uint sc = lane & 7u;
    float acc[8];
    #pragma unroll
    for (uint t = 0u; t < 8u; ++t) {
        acc[t] = 0.0f;
    }
    for (uint b = 0u; b < blocks; ++b) {
        {
            uint token = st;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            if (token < ntok) {
                __global const char *src = xq + (ulong)token * cols + (ulong)b * 256u + sc * 32u;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
            }
            vstore16(a, 0, xt + token * 256u + sc * 32u);
            vstore16(c, 0, xt + token * 256u + sc * 32u + 16u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < rows) {
            ulong gbase = (ulong)b * 8u;
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                __global const char *wp = w + ((gbase + g) * rows + row) * 32u;
                uint4 wa = vload4(0, (__global const uint *)wp);
                uint4 wb = vload4(0, (__global const uint *)(wp + 16u));
                float ws = wsc[(gbase + g) * rows + row];
                __local const uint *xp = (__local const uint *)(xt + g * 32u);
                #pragma unroll
                for (uint t = 0u; t < 8u; ++t) {
                    if (t < ntok) {
                        __local const uint *xq2 = (__local const uint *)(xt + t * 256u + g * 32u);
                        uint4 xa = vload4(0, xq2);
                        uint4 xb = vload4(0, xq2 + 4u);
                        int dl = 0;
                        dl = lfm_dot4(wa.x, xa.x, dl);
                        dl = lfm_dot4(wa.y, xa.y, dl);
                        dl = lfm_dot4(wa.z, xa.z, dl);
                        dl = lfm_dot4(wa.w, xa.w, dl);
                        dl = lfm_dot4(wb.x, xb.x, dl);
                        dl = lfm_dot4(wb.y, xb.y, dl);
                        dl = lfm_dot4(wb.z, xb.z, dl);
                        dl = lfm_dot4(wb.w, xb.w, dl);
                        acc[t] += ws * xs[(ulong)t * ngs + gbase + g] * (float)dl;
                    }
                }
                (void)xp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < ntok) {
                y[(ulong)t * rows + row] = acc[t];
            }
        }
    }
}

__kernel void gemm_i8_gz(__global const char *wq8, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lid = get_local_id(0);
    uint tps = (ntok + 31u) / 32u;
    uint gid = get_group_id(0);
    uint row0 = (gid / tps) * 64u;
    uint t0 = (gid % tps) * 32u;
    if (row0 >= rows) {
        return;
    }
    uint valid_rows = min(64u, rows - row0);
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char xt[2 * 32 * 256];
    __local float xsl[2 * 32 * 8];
    uint lane = lid & 31u;
    uint warp = lid >> 5;
    uint whalf = warp >> 3;
    uint tok_base = (warp & 7u) * 4u;
    uint rloc = whalf * 32u + lane;
    float acc[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        acc[j] = 0.0f;
    }
    uint row = row0 + rloc;
    bool active = rloc < valid_rows;
    uint st_tt = lid >> 3;
    uint st_off = (lid & 7u) * 32u;
    uint st_token = t0 + st_tt;
    bool stager = lid < 256u;
    bool stok = stager && st_token < ntok;
    char16 pa = (char16)(0);
    char16 pc = (char16)(0);
    float psc = 0.0f;
    if (stok) {
        __global const char *src = xq + (ulong)st_token * cols + st_off;
        pa = vload16(0, src);
        pc = vload16(0, src + 16u);
        psc = xs[(ulong)st_token * ngs + (st_off >> 5)];
    }
    if (stager) {
        vstore16(pa, 0, xt + st_tt * 256u + st_off);
        vstore16(pc, 0, xt + st_tt * 256u + st_off + 16u);
        xsl[st_tt * 8u + (st_off >> 5)] = psc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = 0u; b < blocks; ++b) {
        uint cur = b & 1u;
        if (b + 1u < blocks && stager) {
            uint nb = 1u - cur;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            float sc = 0.0f;
            if (stok) {
                __global const char *src = xq + (ulong)st_token * cols + (ulong)(b + 1u) * 256u + st_off;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
                sc = xs[(ulong)st_token * ngs + (ulong)(b + 1u) * 8u + (st_off >> 5)];
            }
            vstore16(a, 0, xt + nb * 8192u + st_tt * 256u + st_off);
            vstore16(c, 0, xt + nb * 8192u + st_tt * 256u + st_off + 16u);
            xsl[nb * 256u + st_tt * 8u + (st_off >> 5)] = sc;
        }
        if (active) {
            __local const char *xb = xt + cur * 8192u;
            __local const float *sb = xsl + cur * 256u;
            ulong gbase = (ulong)b * 8u * rows + row;
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                ulong gi = gbase + (ulong)g * rows;
                __global const char *wp = wq8 + gi * 32ul;
                uint4 wa = vload4(0, (__global const uint *)wp);
                uint4 wb = vload4(0, (__global const uint *)(wp + 16u));
                float sc = wsc[gi];
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    uint token = t0 + tok_base + j;
                    if (token < ntok) {
                        __local const char *xp = xb + (tok_base + j) * 256u + g * 32u;
                        uint4 xa = vload4(0, (__local const uint *)xp);
                        uint4 xc = vload4(0, (__local const uint *)(xp + 16u));
                        int dl = 0;
                        dl = lfm_dot4(wa.x, xa.x, dl); dl = lfm_dot4(wa.y, xa.y, dl);
                        dl = lfm_dot4(wa.z, xa.z, dl); dl = lfm_dot4(wa.w, xa.w, dl);
                        dl = lfm_dot4(wb.x, xc.x, dl); dl = lfm_dot4(wb.y, xc.y, dl);
                        dl = lfm_dot4(wb.z, xc.z, dl); dl = lfm_dot4(wb.w, xc.w, dl);
                        acc[j] += (float)dl * sc * sb[(tok_base + j) * 8u + g];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint t = tok_base + j;
            if (t0 + t < ntok) {
                y[(ulong)(t0 + t) * rows + row] = acc[j];
            }
        }
    }
}


__kernel void gemm_i8_pf(__global const char *wq8, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lid = get_local_id(0);
    uint tps = (ntok + 31u) / 32u;
    uint gid = get_group_id(0);
    uint row0 = (gid / tps) * 64u;
    uint t0 = (gid % tps) * 32u;
    if (row0 >= rows) {
        return;
    }
    uint valid_rows = min(64u, rows - row0);
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char xt[2 * 32 * 256];
    __local float xsl[2 * 32 * 8];
    uint lane = lid & 31u;
    uint warp = lid >> 5;
    uint whalf = warp >> 3;
    uint tok_base = (warp & 7u) * 4u;
    uint rloc = whalf * 32u + lane;
    float acc[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        acc[j] = 0.0f;
    }
    uint row = row0 + rloc;
    bool active = rloc < valid_rows;
    uint st_tt = lid >> 3;
    uint st_off = (lid & 7u) * 32u;
    uint st_token = t0 + st_tt;
    bool stager = lid < 256u;
    bool stok = stager && st_token < ntok;
    char16 pa = (char16)(0);
    char16 pc = (char16)(0);
    float psc = 0.0f;
    if (stok) {
        __global const char *src = xq + (ulong)st_token * cols + st_off;
        pa = vload16(0, src);
        pc = vload16(0, src + 16u);
        psc = xs[(ulong)st_token * ngs + (st_off >> 5)];
    }
    if (stager) {
        vstore16(pa, 0, xt + st_tt * 256u + st_off);
        vstore16(pc, 0, xt + st_tt * 256u + st_off + 16u);
        xsl[st_tt * 8u + (st_off >> 5)] = psc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = 0u; b < blocks; ++b) {
        uint cur = b & 1u;
        if (b + 1u < blocks && stager) {
            uint nb = 1u - cur;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            float sc = 0.0f;
            if (stok) {
                __global const char *src = xq + (ulong)st_token * cols + (ulong)(b + 1u) * 256u + st_off;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
                sc = xs[(ulong)st_token * ngs + (ulong)(b + 1u) * 8u + (st_off >> 5)];
            }
            vstore16(a, 0, xt + nb * 8192u + st_tt * 256u + st_off);
            vstore16(c, 0, xt + nb * 8192u + st_tt * 256u + st_off + 16u);
            xsl[nb * 256u + st_tt * 8u + (st_off >> 5)] = sc;
        }
        if (active) {
            __local const char *xb = xt + cur * 8192u;
            __local const float *sb = xsl + cur * 256u;
            ulong gbase = (ulong)b * 8u * rows + row;
            uint4 wa = vload4(0, (__global const uint *)(wq8 + gbase * 32ul));
            uint4 wb = vload4(0, (__global const uint *)(wq8 + gbase * 32ul + 16u));
            float sc = wsc[gbase];
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                uint4 pa = wa;
                uint4 pb = wb;
                float psc = sc;
                if (g + 1u < 8u) {
                    ulong gn = gbase + (ulong)(g + 1u) * rows;
                    wa = vload4(0, (__global const uint *)(wq8 + gn * 32ul));
                    wb = vload4(0, (__global const uint *)(wq8 + gn * 32ul + 16u));
                    sc = wsc[gn];
                }
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    uint token = t0 + tok_base + j;
                    if (token < ntok) {
                        __local const char *xp = xb + (tok_base + j) * 256u + g * 32u;
                        uint4 xa = vload4(0, (__local const uint *)xp);
                        uint4 xc = vload4(0, (__local const uint *)(xp + 16u));
                        int dl = 0;
                        dl = lfm_dot4(pa.x, xa.x, dl); dl = lfm_dot4(pa.y, xa.y, dl);
                        dl = lfm_dot4(pa.z, xa.z, dl); dl = lfm_dot4(pa.w, xa.w, dl);
                        dl = lfm_dot4(pb.x, xc.x, dl); dl = lfm_dot4(pb.y, xc.y, dl);
                        dl = lfm_dot4(pb.z, xc.z, dl); dl = lfm_dot4(pb.w, xc.w, dl);
                        acc[j] += (float)dl * psc * sb[(tok_base + j) * 8u + g];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint t = tok_base + j;
            if (t0 + t < ntok) {
                y[(ulong)(t0 + t) * rows + row] = acc[j];
            }
        }
    }
}

__kernel void gemm_i8_pf16(__global const char *wq8, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lid = get_local_id(0);
    uint tps = (ntok + 31u) / 32u;
    uint gid = get_group_id(0);
    uint row0 = (gid / tps) * 64u;
    uint t0 = (gid % tps) * 32u;
    if (row0 >= rows) {
        return;
    }
    uint valid_rows = min(64u, rows - row0);
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char xt[2 * 16 * 256];
    __local float xsl[2 * 16 * 8];
    uint lane = lid & 31u;
    uint warp = lid >> 5;
    uint whalf = warp >> 2;
    uint tok_base = (warp & 3u) * 4u;
    uint rloc = whalf * 32u + lane;
    float acc[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        acc[j] = 0.0f;
    }
    uint row = row0 + rloc;
    bool active = rloc < valid_rows;
    uint st_tt = lid >> 3;
    uint st_off = (lid & 7u) * 32u;
    uint st_token = t0 + st_tt;
    bool stager = lid < 128u;
    bool stok = stager && st_token < ntok;
    char16 pa = (char16)(0);
    char16 pc = (char16)(0);
    float psc = 0.0f;
    if (stok) {
        __global const char *src = xq + (ulong)st_token * cols + st_off;
        pa = vload16(0, src);
        pc = vload16(0, src + 16u);
        psc = xs[(ulong)st_token * ngs + (st_off >> 5)];
    }
    if (stager) {
        vstore16(pa, 0, xt + st_tt * 256u + st_off);
        vstore16(pc, 0, xt + st_tt * 256u + st_off + 16u);
        xsl[st_tt * 8u + (st_off >> 5)] = psc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = 0u; b < blocks; ++b) {
        uint cur = b & 1u;
        if (b + 1u < blocks && stager) {
            uint nb = 1u - cur;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            float sc = 0.0f;
            if (stok) {
                __global const char *src = xq + (ulong)st_token * cols + (ulong)(b + 1u) * 256u + st_off;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
                sc = xs[(ulong)st_token * ngs + (ulong)(b + 1u) * 8u + (st_off >> 5)];
            }
            vstore16(a, 0, xt + nb * 4096u + st_tt * 256u + st_off);
            vstore16(c, 0, xt + nb * 4096u + st_tt * 256u + st_off + 16u);
            xsl[nb * 128u + st_tt * 8u + (st_off >> 5)] = sc;
        }
        if (active) {
            __local const char *xb = xt + cur * 4096u;
            __local const float *sb = xsl + cur * 128u;
            ulong gbase = (ulong)b * 8u * rows + row;
            uint4 wa = vload4(0, (__global const uint *)(wq8 + gbase * 32ul));
            uint4 wb = vload4(0, (__global const uint *)(wq8 + gbase * 32ul + 16u));
            float sc = wsc[gbase];
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                uint4 pa = wa;
                uint4 pb = wb;
                float psc = sc;
                if (g + 1u < 8u) {
                    ulong gn = gbase + (ulong)(g + 1u) * rows;
                    wa = vload4(0, (__global const uint *)(wq8 + gn * 32ul));
                    wb = vload4(0, (__global const uint *)(wq8 + gn * 32ul + 16u));
                    sc = wsc[gn];
                }
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    uint token = t0 + tok_base + j;
                    if (token < ntok) {
                        __local const char *xp = xb + (tok_base + j) * 256u + g * 32u;
                        uint4 xa = vload4(0, (__local const uint *)xp);
                        uint4 xc = vload4(0, (__local const uint *)(xp + 16u));
                        int dl = 0;
                        dl = lfm_dot4(pa.x, xa.x, dl); dl = lfm_dot4(pa.y, xa.y, dl);
                        dl = lfm_dot4(pa.z, xa.z, dl); dl = lfm_dot4(pa.w, xa.w, dl);
                        dl = lfm_dot4(pb.x, xc.x, dl); dl = lfm_dot4(pb.y, xc.y, dl);
                        dl = lfm_dot4(pb.z, xc.z, dl); dl = lfm_dot4(pb.w, xc.w, dl);
                        acc[j] += (float)dl * psc * sb[(tok_base + j) * 8u + g];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint t = tok_base + j;
            if (t0 + t < ntok) {
                y[(ulong)(t0 + t) * rows + row] = acc[j];
            }
        }
    }
}

__kernel void gemm_q6_i8_b8p(__global const char *w, __global const float *wsc, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint row = gid * 64u + lane;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    __local char xt[2 * 8 * 256];
    uint st = lane >> 3;
    uint sc = lane & 7u;
    bool stok = st < ntok;
    float acc[8];
    #pragma unroll
    for (uint t = 0u; t < 8u; ++t) {
        acc[t] = 0.0f;
    }
    bool active = row < rows;
    char16 pa = (char16)(0);
    char16 pc = (char16)(0);
    if (stok) {
        __global const char *src = xq + (ulong)st * cols + sc * 32u;
        pa = vload16(0, src);
        pc = vload16(0, src + 16u);
    }
    vstore16(pa, 0, xt + st * 256u + sc * 32u);
    vstore16(pc, 0, xt + st * 256u + sc * 32u + 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = 0u; b < blocks; ++b) {
        uint cur = b & 1u;
        if (b + 1u < blocks) {
            uint nb = 1u - cur;
            char16 a = (char16)(0);
            char16 c = (char16)(0);
            if (stok) {
                __global const char *src = xq + (ulong)st * cols + (ulong)(b + 1u) * 256u + sc * 32u;
                a = vload16(0, src);
                c = vload16(0, src + 16u);
            }
            vstore16(a, 0, xt + nb * 2048u + st * 256u + sc * 32u);
            vstore16(c, 0, xt + nb * 2048u + st * 256u + sc * 32u + 16u);
        }
        if (active) {
            __local const char *xb = xt + cur * 2048u;
            ulong gbase = (ulong)b * 8u;
            #pragma unroll
            for (uint g = 0u; g < 8u; ++g) {
                __global const char *wp = w + ((gbase + g) * rows + row) * 32u;
                uint4 wa = vload4(0, (__global const uint *)wp);
                uint4 wb = vload4(0, (__global const uint *)(wp + 16u));
                float ws = wsc[(gbase + g) * rows + row];
                #pragma unroll
                for (uint t = 0u; t < 8u; ++t) {
                    if (t < ntok) {
                        __local const uint *xq2 = (__local const uint *)(xb + t * 256u + g * 32u);
                        uint4 xa = vload4(0, xq2);
                        uint4 xc = vload4(0, xq2 + 4u);
                        int dl = 0;
                        dl = lfm_dot4(wa.x, xa.x, dl);
                        dl = lfm_dot4(wa.y, xa.y, dl);
                        dl = lfm_dot4(wa.z, xa.z, dl);
                        dl = lfm_dot4(wa.w, xa.w, dl);
                        dl = lfm_dot4(wb.x, xc.x, dl);
                        dl = lfm_dot4(wb.y, xc.y, dl);
                        dl = lfm_dot4(wb.z, xc.z, dl);
                        dl = lfm_dot4(wb.w, xc.w, dl);
                        acc[t] += ws * xs[(ulong)t * ngs + gbase + g] * (float)dl;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        #pragma unroll
        for (uint t = 0u; t < 8u; ++t) {
            if (t < ntok) {
                y[(ulong)t * rows + row] = acc[t];
            }
        }
    }
}

__kernel void gemm_q4_k_i8_lds17(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok) {
    uint lid = get_local_id(0);
    uint tps = (ntok + 31u) / 32u;
    uint gid = get_group_id(0);
    uint row0 = (gid / tps) * 64u;
    uint t0 = (gid % tps) * 32u;
    if (row0 >= rows) {
        return;
    }
    uint valid_rows = min(64u, rows - row0);
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 148ul;
    __local uint wq[8 * 64 * 4];
    __local uchar ws[64 * 16];
    __local uchar wh[64 * 4];
    __local char xt[32 * 256];
    __local int xsum[32 * 8];
    const char4 ones = (char4)(1);
    const uint M = 0x0F0F0F0Fu;
    uint lane = lid & 31u;
    uint warp = lid >> 5;
    uint whalf = warp >> 3;
    uint tok_base = (warp & 7u) * 4u;
    uint rloc = whalf * 32u + lane;
    float acc[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        acc[j] = 0.0f;
    }
    for (uint b = 0u; b < blocks; ++b) {
        {
            uint c = lid >> 6;
            uint rr = lid & 63u;
            uint qoff = (c >> 1) * 32u + (c & 1u) * 16u;
            uint4 v = (uint4)(0);
            if (rr < valid_rows) {
                __global const uchar *src = w + (ulong)(row0 + rr) * row_stride + (ulong)b * 148ul + 20ul + qoff;
                v = vload4(0, (__global const uint *)src);
            }
            vstore4(v, 0, wq + (c * 64u + rr) * 4u);
        }
        {
            uint si = lid * 2u;
            uint r = si >> 4;
            uint off = si & 15u;
            uchar2 v = (uchar2)(0);
            if (r < valid_rows) {
                v = vload2(0, w + (ulong)(row0 + r) * row_stride + (ulong)b * 148ul + 4ul + off);
            }
            vstore2(v, 0, ws + r * 16u + off);
        }
        if (lid < 256u) {
            uint r = lid >> 2;
            uint off = lid & 3u;
            uchar v = 0u;
            if (r < valid_rows) {
                v = w[(ulong)(row0 + r) * row_stride + (ulong)b * 148ul + off];
            }
            wh[lid] = v;
        }
        {
            uint4 v = (uint4)(0);
            if (lid < 256u) {
                uint token = t0 + (lid >> 3);
                uint off = (lid & 7u) * 32u;
                char16 a = vload16(0, xq + (ulong)token * cols + (ulong)b * 256u + off);
                char16 c = vload16(0, xq + (ulong)token * cols + (ulong)b * 256u + off + 16u);
                vstore16(a, 0, xt + (lid >> 3) * 256u + off);
                vstore16(c, 0, xt + (lid >> 3) * 256u + off + 16u);
                int sx = (int)LFM_DOT(a.s0123, ones) + (int)LFM_DOT(a.s4567, ones)
                       + (int)LFM_DOT(a.s89ab, ones) + (int)LFM_DOT(a.scdef, ones)
                       + (int)LFM_DOT(c.s0123, ones) + (int)LFM_DOT(c.s4567, ones)
                       + (int)LFM_DOT(c.s89ab, ones) + (int)LFM_DOT(c.scdef, ones);
                xsum[(lid >> 3) * 8u + (off >> 5)] = sx;
            }
            (void)v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (rloc < valid_rows) {
            __local const uchar *srow = ws + rloc * 16u;
            float d = lfm_scale_local(wh + rloc * 4u);
            float dmin = lfm_scale_local(wh + rloc * 4u + 2u);
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                uint4 pa = vload4(0, wq + ((2u * q) * 64u + rloc) * 4u);
                uint4 pb = vload4(0, wq + ((2u * q + 1u) * 64u + rloc) * 4u);
                char4 l0 = as_char4(pa.x & M), l1 = as_char4(pa.y & M), l2 = as_char4(pa.z & M), l3 = as_char4(pa.w & M);
                char4 l4 = as_char4(pb.x & M), l5 = as_char4(pb.y & M), l6 = as_char4(pb.z & M), l7 = as_char4(pb.w & M);
                char4 h0 = as_char4((pa.x >> 4) & M), h1 = as_char4((pa.y >> 4) & M), h2 = as_char4((pa.z >> 4) & M), h3 = as_char4((pa.w >> 4) & M);
                char4 h4 = as_char4((pb.x >> 4) & M), h5 = as_char4((pb.y >> 4) & M), h6 = as_char4((pb.z >> 4) & M), h7 = as_char4((pb.w >> 4) & M);
                float scl = d * (float)srow[4u * q], mnl = dmin * (float)srow[4u * q + 1u];
                float sch = d * (float)srow[4u * q + 2u], mnh = dmin * (float)srow[4u * q + 3u];
                uint xbase = q * 64u;
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    uint t = tok_base + j;
                    uint token = t0 + t;
                    if (token < ntok) {
                        __local const char *xp = xt + t * 256u + xbase;
                        char16 xl0 = vload16(0, xp);
                        char16 xl1 = vload16(0, xp + 16u);
                        char16 xh0 = vload16(0, xp + 32u);
                        char16 xh1 = vload16(0, xp + 48u);
                        int suml = xsum[t * 8u + 2u * q];
                        int sumh = xsum[t * 8u + 2u * q + 1u];
                        float dl = (float)(LFM_DOT(l0, xl0.s0123) + LFM_DOT(l1, xl0.s4567) + LFM_DOT(l2, xl0.s89ab) + LFM_DOT(l3, xl0.scdef)
                                         + LFM_DOT(l4, xl1.s0123) + LFM_DOT(l5, xl1.s4567) + LFM_DOT(l6, xl1.s89ab) + LFM_DOT(l7, xl1.scdef));
                        float dh = (float)(LFM_DOT(h0, xh0.s0123) + LFM_DOT(h1, xh0.s4567) + LFM_DOT(h2, xh0.s89ab) + LFM_DOT(h3, xh0.scdef)
                                         + LFM_DOT(h4, xh1.s0123) + LFM_DOT(h5, xh1.s4567) + LFM_DOT(h6, xh1.s89ab) + LFM_DOT(h7, xh1.scdef));
                        ulong g = (ulong)token * ngs + (ulong)b * 8u + 2u * q;
                        acc[j] += xs[g] * (dl * scl - (float)suml * mnl);
                        acc[j] += xs[g + 1u] * (dh * sch - (float)sumh * mnh);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (rloc < valid_rows) {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint t = tok_base + j;
            if (t0 + t < ntok) {
                y[(ulong)(t0 + t) * rows + row0 + rloc] = acc[j];
            }
        }
    }
}

__kernel void logits_argmax_b(__global const float *src, __global const uint *rowids, __global uint *ids, __global float *vals, uint ntok, uint vocab) {
    uint r = get_group_id(0);
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    if (r >= ntok) {
        return;
    }
    uint row = rowids[r];
    __local float bv[1024];
    __local uint bi[1024];
    __global const float *p = src + (ulong)row * vocab;
    uint nv = vocab >> 2;
    float best = -INFINITY;
    uint idx = 0u;
    for (uint i = lid; i < nv; i += lsz) {
        float4 v = vload4(0, p + i * 4u);
        if (v.x > best) { best = v.x; idx = i * 4u; }
        if (v.y > best) { best = v.y; idx = i * 4u + 1u; }
        if (v.z > best) { best = v.z; idx = i * 4u + 2u; }
        if (v.w > best) { best = v.w; idx = i * 4u + 3u; }
    }
    for (uint i = nv * 4u + lid; i < vocab; i += lsz) {
        float v = p[i];
        if (v > best) { best = v; idx = i; }
    }
    bv[lid] = best;
    bi[lid] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = lsz >> 1; st > 0u; st >>= 1) {
        if (lid < st) {
            float v = bv[lid + st];
            if (v > bv[lid]) {
                bv[lid] = v;
                bi[lid] = bi[lid + st];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0u) {
        ids[r] = bi[0];
        vals[r] = bv[0];
    }
}
