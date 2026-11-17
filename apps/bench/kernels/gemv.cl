#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef RGEMV
#define RGEMV 4
#endif

__kernel void quantize_q8(__global const float *x, __global char *xq, __global float *xs, uint n, uint group) {
    uint g = get_global_id(0);
    uint ng = n / group;
    if (g >= ng) {
        return;
    }
    uint base = g * group;
    float amax = 0.0f;
    for (uint i = 0u; i < group; i += 4u) {
        float4 v = vload4(0, x + base + i);
        amax = fmax(amax, fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
    }
    float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    xs[g] = sc;
    for (uint i = 0u; i < group; i += 4u) {
        float4 v = vload4(0, x + base + i) * inv;
        char4 q = (char4)(convert_int_rte(v.x), convert_int_rte(v.y), convert_int_rte(v.z), convert_int_rte(v.w));
        vstore4(q, 0, xq + base + i);
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
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
    }
    uint u = lane & 7u;
    uint bpi = sgs >> 3;
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 144ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    uint sb_lo = (u >> 1) << 1;
    uint wbase = (u >> 1) * 64u + (u & 1u) * 16u;
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
            int sum_lo = (int)xl.s0 + (int)xl.s1 + (int)xl.s2 + (int)xl.s3 + (int)xl.s4 + (int)xl.s5 + (int)xl.s6 + (int)xl.s7
                       + (int)xl.s8 + (int)xl.s9 + (int)xl.sa + (int)xl.sb + (int)xl.sc + (int)xl.sd + (int)xl.se + (int)xl.sf;
            int sum_hi = (int)xh.s0 + (int)xh.s1 + (int)xh.s2 + (int)xh.s3 + (int)xh.s4 + (int)xh.s5 + (int)xh.s6 + (int)xh.s7
                       + (int)xh.s8 + (int)xh.s9 + (int)xh.sa + (int)xh.sb + (int)xh.sc + (int)xh.sd + (int)xh.se + (int)xh.sf;
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 144ul;
                __global const uchar *scp = blk + 4;
                float d = (float)vload_half(0, (__global const half *)blk);
                float dmin = (float)vload_half(0, (__global const half *)(blk + 2));
                uchar sl = 0u;
                uchar ml = 0u;
                uchar sh = 0u;
                uchar mh = 0u;
                gsm_k4(sb_lo, scp, &sl, &ml);
                gsm_k4(sb_lo + 1u, scp, &sh, &mh);
                uint4 pk = vload4(0, (__global const uint *)(blk + 16u + u * 16u));
                uint4 lo = pk & (uint4)(0x0F0F0F0Fu);
                uint4 hi = (pk >> 4) & (uint4)(0x0F0F0F0Fu);
                float dl = (float)(dot(as_char4(lo.x), xl.s0123) + dot(as_char4(lo.y), xl.s4567)
                                 + dot(as_char4(lo.z), xl.s89ab) + dot(as_char4(lo.w), xl.scdef));
                float dh = (float)(dot(as_char4(hi.x), xh.s0123) + dot(as_char4(hi.y), xh.s4567)
                                 + dot(as_char4(hi.z), xh.s89ab) + dot(as_char4(hi.w), xh.scdef));
                acc[r] += xs_lo * (dl * (d * (float)sl) - (float)sum_lo * (dmin * (float)ml));
                acc[r] += xs_hi * (dh * (d * (float)sh) - (float)sum_hi * (dmin * (float)mh));
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add(acc[r]);
        if (lane == 0u && valid) {
            y[row0 + r] = total;
        }
    }
}

__kernel void gemv_q6_k_i8(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols) {
    uint lane = get_sub_group_local_id();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
    }
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
    for (uint b0 = 0u; b0 < blocks; b0 += 4u) {
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
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 210ul;
                __global const char *sc = (__global const char *)(blk + 192u);
                float d = (float)vload_half(0, (__global const half *)(blk + 208u));
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
                acc[r] += xs0 * s0 * (float)dot(q0, x0);
                acc[r] += xs1 * s1 * (float)dot(q1, x1);
                acc[r] += xs2 * s2 * (float)dot(q2, x2);
                acc[r] += xs3 * s3 * (float)dot(q3, x3);
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add(acc[r]);
        if (lane == 0u && valid) {
            y[row0 + r] = total;
        }
    }
}

__kernel void gemv_q4_k_sum(__global const uchar *w, __global float *y, uint rows, uint cols) {
    uint lid = get_local_id(0);
    uint lane = lid & (get_sub_group_size() - 1u);
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
    }
    uint sub = lane & 31u;
    int acc[RGEMV];
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        acc[r] = 0;
    }
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 144ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    for (uint b0 = 0u; b0 < blocks; b0 += 2u) {
        uint b = b0 + (lane >> 5);
        if (b < blocks) {
            __global const uchar *qrow = base_w + (ulong)b * 144ul + 16u + sub * 4u;
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                uchar4 pk = vload4(0, qrow + (ulong)r * row_stride);
                acc[r] += (int)pk.s0 + (int)pk.s1 + (int)pk.s2 + (int)pk.s3;
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add((float)acc[r]);
        if (lane == 0u) {
            if (valid) {
                y[row0 + r] = total;
            }
        }
    }
}

__kernel void gemv_q6_k_sum(__global const uchar *w, __global float *y, uint rows, uint cols) {
    uint lid = get_local_id(0);
    uint lane = lid & (get_sub_group_size() - 1u);
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
    }
    uint sub = lane & 31u;
    int acc[RGEMV];
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        acc[r] = 0;
    }
    uint blocks = cols >> 8;
    ulong row_stride = (ulong)blocks * 210ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    for (uint b0 = 0u; b0 < blocks; b0 += 2u) {
        uint b = b0 + (lane >> 5);
        if (b < blocks) {
            __global const uchar *qrow = base_w + (ulong)b * 210ul + sub * 4u;
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                uchar4 pk = vload4(0, qrow + (ulong)r * row_stride);
                acc[r] += (int)pk.s0 + (int)pk.s1 + (int)pk.s2 + (int)pk.s3;
            }
        }
    }
    #pragma unroll
    for (uint r = 0u; r < RGEMV; ++r) {
        float total = sub_group_reduce_add((float)acc[r]);
        if (lane == 0u) {
            if (valid) {
                y[row0 + r] = total;
            }
        }
    }
}

__kernel void gemv_q4_k_i8_b(__global const uchar *w, __global const char *xq, __global const float *xs, __global float *y, uint rows, uint cols, uint ntok, uint tok0) {
    uint lane = get_sub_group_local_id();
    uint sgs = get_sub_group_size();
    uint sgid = get_sub_group_id();
    uint nsg = get_num_sub_groups();
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
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
    float acc[PB][RGEMV];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV; ++r) {
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
                    suml[t] = (int)dot(xl[t].s0123, ones) + (int)dot(xl[t].s4567, ones)
                            + (int)dot(xl[t].s89ab, ones) + (int)dot(xl[t].scdef, ones);
                    sumh[t] = (int)dot(xh[t].s0123, ones) + (int)dot(xh[t].s4567, ones)
                            + (int)dot(xh[t].s89ab, ones) + (int)dot(xh[t].scdef, ones);
                }
            }
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 144ul;
                __global const uchar *scp = blk + 4;
                float d = (float)vload_half(0, (__global const half *)blk);
                float dmin = (float)vload_half(0, (__global const half *)(blk + 2));
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
                        float dl = (float)(dot(l0, xl[t].s0123) + dot(l1, xl[t].s4567)
                                         + dot(l2, xl[t].s89ab) + dot(l3, xl[t].scdef));
                        float dh = (float)(dot(h0, xh[t].s0123) + dot(h1, xh[t].s4567)
                                         + dot(h2, xh[t].s89ab) + dot(h3, xh[t].scdef));
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
            for (uint r = 0u; r < RGEMV; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u && valid) {
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
    uint row0 = (get_group_id(0) * nsg + sgid) * RGEMV;
    bool valid = row0 + RGEMV <= rows;
    if (!valid) {
        row0 = rows - RGEMV;
    }
    uint sub = lane & 15u;
    uint n = sub >> 3;
    uint l = (sub & 7u) * 4u;
    uint lq = l >> 4;
    uint blocks = cols >> 8;
    uint ngs = cols >> 5;
    ulong row_stride = (ulong)blocks * 210ul;
    __global const uchar *base_w = w + (ulong)row0 * row_stride;
    float acc[PB][RGEMV];
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        #pragma unroll
        for (uint r = 0u; r < RGEMV; ++r) {
            acc[t][r] = 0.0f;
        }
    }
    for (uint b0 = 0u; b0 < blocks; b0 += 4u) {
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
            for (uint r = 0u; r < RGEMV; ++r) {
                __global const uchar *blk = base_w + (ulong)r * row_stride + (ulong)b * 210ul;
                __global const char *sc = (__global const char *)(blk + 192u);
                float d = (float)vload_half(0, (__global const half *)(blk + 208u));
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
                        acc[t][r] += xs0[t] * s0 * (float)dot(q0, x0[t]);
                        acc[t][r] += xs1[t] * s1 * (float)dot(q1, x1[t]);
                        acc[t][r] += xs2[t] * s2 * (float)dot(q2, x2[t]);
                        acc[t][r] += xs3[t] * s3 * (float)dot(q3, x3[t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint t = 0u; t < PB; ++t) {
        if (t >= tok0 && t < ntok) {
            #pragma unroll
            for (uint r = 0u; r < RGEMV; ++r) {
                float total = sub_group_reduce_add(acc[t][r]);
                if (lane == 0u && valid) {
                    y[(ulong)t * rows + row0 + r] = total;
                }
            }
        }
    }
}

