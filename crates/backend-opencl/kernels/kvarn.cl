#ifndef LFM_F32_KV
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


__kernel void kv_store_h_rot(__global const float *ksrc, __global const float *vsrc,
                             __global KV_T *kdst, __global KV_T *vdst,
                             __global const uint *tbl, uint pos, uint kv_heads, uint head_dim, uint seq, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint lane = get_local_id(0);
    uint h = get_group_id(0);
    __local float buf[128];
    ulong slot = ((ulong)seq << rbits) + (pos & rmask);
    ulong base = slot * kv_heads * 128u + (ulong)h * 128u;
    ulong src = (ulong)h * 128u;
    buf[2u * lane] = ksrc[src + 2u * lane];
    buf[2u * lane + 1u] = ksrc[src + 2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    kdst[base + 2u * lane] = (KV_T)buf[2u * lane];
    kdst[base + 2u * lane + 1u] = (KV_T)buf[2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    buf[2u * lane] = vsrc[src + 2u * lane];
    buf[2u * lane + 1u] = vsrc[src + 2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    vdst[base + 2u * lane] = (KV_T)buf[2u * lane];
    vdst[base + 2u * lane + 1u] = (KV_T)buf[2u * lane + 1u];
}

__kernel void kv_store_b_h_rot(__global const float *ksrc, __global const float *vsrc,
                               __global KV_T *kdst, __global KV_T *vdst,
                               __global const uint *tbl, __global const uint *tbl_off,
                               __global const uint *pos_b, __global const uint *seq_b,
                               uint kv_heads, uint head_dim, uint ntok, uint seq_cap, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint lane = get_local_id(0);
    uint gid = get_group_id(0);
    uint t = gid / kv_heads;
    uint h = gid % kv_heads;
    if (t >= ntok) {
        return;
    }
    __local float buf[128];
    uint seq = seq_b[t];
    if (seq >= seq_cap) {
        return;
    }
    uint pos = pos_b[t];
    ulong slot = ((ulong)seq << rbits) + (pos & rmask);
    ulong base = slot * kv_heads * 128u + (ulong)h * 128u;
    ulong src = ((ulong)t * kv_heads + h) * 128u;
    buf[2u * lane] = ksrc[src + 2u * lane];
    buf[2u * lane + 1u] = ksrc[src + 2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    kdst[base + 2u * lane] = (KV_T)buf[2u * lane];
    kdst[base + 2u * lane + 1u] = (KV_T)buf[2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    buf[2u * lane] = vsrc[src + 2u * lane];
    buf[2u * lane + 1u] = vsrc[src + 2u * lane + 1u];
    barrier(CLK_LOCAL_MEM_FENCE);
    lfm_tq_rot_fwd(buf, lane);
    vdst[base + 2u * lane] = (KV_T)buf[2u * lane];
    vdst[base + 2u * lane + 1u] = (KV_T)buf[2u * lane + 1u];
}

static void lfm_kvarn_scale(__local float *lr, __local float *lc, __local float *ir, __local float *ic,
                            uint lane, uint chans, uint toks) {
    if (lane < toks) {
        ic[lane] = exp(-lc[lane]);
    }
    for (uint c = lane; c < chans; c += 64u) {
        ir[c] = exp(-lr[c]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static float lfm_kvarn_tok_var(__local const float *t, __local const float *ic, __local const float *ir,
                               uint tok, uint chans) {
    float s = 0.0f;
    float c = ic[tok];
    for (uint ch = 0u; ch < chans; ++ch) {
        float v = t[tok * chans + ch] * c * ir[ch];
        s += v * v;
    }
    return s / (float)chans;
}

static float lfm_kvarn_ch_var(__local const float *t, __local const float *ic, __local const float *ir,
                              uint ch, uint toks) {
    float s = 0.0f;
    float r = ir[ch];
    for (uint tok = 0u; tok < toks; ++tok) {
        float v = t[tok * 128u + ch] * ic[tok] * r;
        s += v * v;
    }
    return s / (float)toks;
}

__kernel void kvarn_commit(__global const KV_T *kst, __global const KV_T *vst,
                           __global uchar *kcode, __global KV_T *ka, __global KV_T *kc,
                           __global uchar *vcode, __global KV_T *vb, __global KV_T *vbz, __global KV_T *vc,
                           __global const uint *tbl, uint tbl_off, uint page, uint kv_heads, uint head_dim, uint seq, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint lane = get_local_id(0);
    uint h = get_group_id(0);
    if (h >= kv_heads || head_dim != 128u) {
        return;
    }
    __local float t[32 * 128];
    __local float lr[128];
    __local float lc[32];
    __local float ir[128];
    __local float ic[32];
    __local float a[128];
    __local float b[32];
    __local float bz[32];
    ulong phys = (ulong)tbl[tbl_off + page];
    uint kstride = kv_heads * head_dim;
    ulong stbase = (((ulong)seq << rbits) + ((ulong)page * 32u & rmask)) * kstride + (ulong)h * head_dim;
    for (uint tok = 0u; tok < 32u; ++tok) {
        ulong off = stbase + (ulong)tok * kstride;
        t[tok * 128u + 2u * lane] = (float)kst[off + 2u * lane];
        t[tok * 128u + 2u * lane + 1u] = (float)kst[off + 2u * lane + 1u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < 32u) {
        lc[lane] = 0.0f;
    }
    for (uint c = lane; c < 128u; c += 64u) {
        lr[c] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint it = 0u; it < 3u; ++it) {
        lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
        if (lane < 32u) {
            float v = fmax(lfm_kvarn_tok_var(t, ic, ir, lane, 128u), 1e-8f);
            lc[lane] = clamp(lc[lane] + 0.5f * log(v), -0.3f, 10.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
        for (uint c = lane; c < 128u; c += 64u) {
            float v = fmax(lfm_kvarn_ch_var(t, ic, ir, c, 32u), 1e-8f);
            lr[c] = clamp(lr[c] + 0.5f * log(v), -0.3f, 10.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
    {
        uint base = (uint)phys * kv_heads + h;
        for (uint c = lane; c < 128u; c += 64u) {
            float mx = 1e-8f;
            float r = ir[c];
            for (uint tok = 0u; tok < 32u; ++tok) {
                mx = fmax(mx, fabs(t[tok * 128u + c] * ic[tok] * r));
            }
            a[c] = mx / 7.0f;
            ka[((uint)phys * kv_heads + h) * 128u + c] = (KV_T)(a[c] * exp(lr[c]));
        }
        if (lane < 32u) {
            kc[((uint)phys * kv_heads + h) * 32u + lane] = (KV_T)exp(lc[lane]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint tok = 0u; tok < 32u; ++tok) {
            ulong slot = (phys << 5) | tok;
            float ct = exp(lc[tok]);
            int q0 = (int)round((t[tok * 128u + 2u * lane] * ic[tok] * ir[2u * lane]) / a[2u * lane]);
            int q1 = (int)round((t[tok * 128u + 2u * lane + 1u] * ic[tok] * ir[2u * lane + 1u]) / a[2u * lane + 1u]);
            q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
            q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);
            kcode[(slot * kv_heads + h) * 64u + lane] = (uchar)((q0 + 8) | ((q1 + 8) << 4));
        }
    }
    for (uint tok = 0u; tok < 32u; ++tok) {
        ulong off = stbase + (ulong)tok * kstride;
        t[tok * 128u + 2u * lane] = (float)vst[off + 2u * lane];
        t[tok * 128u + 2u * lane + 1u] = (float)vst[off + 2u * lane + 1u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < 32u) {
        lc[lane] = 0.0f;
    }
    for (uint c = lane; c < 128u; c += 64u) {
        lr[c] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint it = 0u; it < 3u; ++it) {
        lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
        if (lane < 32u) {
            float v = fmax(lfm_kvarn_tok_var(t, ic, ir, lane, 128u), 1e-8f);
            lc[lane] = clamp(lc[lane] + 0.5f * log(v), -0.3f, 10.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
        for (uint c = lane; c < 128u; c += 64u) {
            float v = fmax(lfm_kvarn_ch_var(t, ic, ir, c, 32u), 1e-8f);
            lr[c] = clamp(lr[c] + 0.5f * log(v), -0.3f, 10.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    lfm_kvarn_scale(lr, lc, ir, ic, lane, 128u, 32u);
    {
        for (uint t2 = lane; t2 < 32u; t2 += 64u) {
            float mn = 1e9f;
            float mx = -1e9f;
            float c = ic[t2];
            for (uint ch = 0u; ch < 128u; ++ch) {
                float v = t[t2 * 128u + ch] * c * ir[ch];
                mn = fmin(mn, v);
                mx = fmax(mx, v);
            }
#ifdef LFM_KV4
            float step = (mx - mn) / 15.0f;
#else
            float step = (mx - mn) / 3.0f;
#endif
            if (step < 1e-8f) {
                step = 1e-8f;
            }
            b[t2] = step;
            bz[t2] = mn;
            vb[((uint)phys * kv_heads + h) * 32u + t2] = (KV_T)(step * exp(lc[t2]));
            vbz[((uint)phys * kv_heads + h) * 32u + t2] = (KV_T)(mn * exp(lc[t2]));
        }
        for (uint c = lane; c < 128u; c += 64u) {
            vc[((uint)phys * kv_heads + h) * 128u + c] = (KV_T)exp(lr[c]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef LFM_KV4
        for (uint tok = 0u; tok < 32u; ++tok) {
            ulong slot = (phys << 5) | tok;
            float ct = ic[tok];
            int q0 = (int)round((t[tok * 128u + 2u * lane] * ct * ir[2u * lane] - bz[tok]) / b[tok]);
            int q1 = (int)round((t[tok * 128u + 2u * lane + 1u] * ct * ir[2u * lane + 1u] - bz[tok]) / b[tok]);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            vcode[(slot * kv_heads + h) * 64u + lane] = (uchar)(q0 | (q1 << 4));
        }
#else
        if (lane < 32u) {
            for (uint tok = 0u; tok < 32u; ++tok) {
                ulong slot = (phys << 5) | tok;
                uchar byte = 0u;
                float ct = ic[tok];
                for (uint k = 0u; k < 4u; ++k) {
                    uint ch = lane * 4u + k;
                    int q = (int)round((t[tok * 128u + ch] * ct * ir[ch] - bz[tok]) / b[tok]);
                    q = q < 0 ? 0 : (q > 3 ? 3 : q);
                    byte |= (uchar)(q << (2u * k));
                }
                vcode[(slot * kv_heads + h) * 32u + lane] = byte;
            }
        }
#endif
    }
}

static float2 lfm_kvarn_k(__global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                          uint phys, uint tok, uint kvh, uint kv_heads, uint lane) {
    uint base = phys * kv_heads + kvh;
    uchar b = kcode[((phys << 5 | tok) * kv_heads + kvh) * 64u + lane];
    float ct = (float)kc[base * 32u + tok];
    float2 k;
    k.x = (float)((int)(b & 15u) - 8) * (float)ka[base * 128u + 2u * lane] * ct;
    k.y = (float)((int)(b >> 4) - 8) * (float)ka[base * 128u + 2u * lane + 1u] * ct;
    return k;
}

static float2 lfm_kvarn_v(__global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz,
                          __global const KV_T *vc, uint phys, uint tok, uint kvh, uint kv_heads, uint lane) {
    uint base = phys * kv_heads + kvh;
    float bt = (float)vb[base * 32u + tok];
    float zero = (float)vbz[base * 32u + tok];
    float2 v;
#ifdef LFM_KV4
    uchar b = vcode[((phys << 5 | tok) * kv_heads + kvh) * 64u + lane];
    v.x = (float)(b & 15u) * bt + zero;
    v.y = (float)((b >> 4) & 15u) * bt + zero;
#else
    uchar b = vcode[((phys << 5 | tok) * kv_heads + kvh) * 32u + (lane >> 1)];
    uint sh = 4u * (lane & 1u);
    v.x = (float)((b >> sh) & 3u) * bt + zero;
    v.y = (float)((b >> (sh + 2u)) & 3u) * bt + zero;
#endif
    v.x *= (float)vc[base * 128u + 2u * lane];
    v.y *= (float)vc[base * 128u + 2u * lane + 1u];
    return v;
}

__kernel void attention_h_qk(__global const float *q, __global const KV_T *kst, __global const KV_T *vst,
                             __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                             __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                             __global const uint *tbl, __global const float *gate, __global float *out,
                             uint len, uint start, uint full_end, uint kv_heads, uint n_heads, uint seq, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint lane = get_local_id(0);
    uint head = get_group_id(0);
    __local float sred[128];
    __local float qs[128];
    uint kvh = head / (n_heads / kv_heads);
    lfm_tq_load_q(q, qs, lane, head);
    float q0 = qs[2u * lane];
    float q1 = qs[2u * lane + 1u];
    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = 0.0883883476f;
    uint kstride = kv_heads * 128u;
    uint pos = start;
    for (; pos + 1u < len; pos += 2u) {
        float2 k0;
        float2 k1;
        if (pos < full_end) {
            k0 = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            k0 = kv_load2(lane, kst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
        if (pos + 1u < full_end) {
            k1 = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[(pos + 1u) >> 5], (pos + 1u) & 31u, kvh, kv_heads, lane);
        } else {
            k1 = kv_load2(lane, kst + (((ulong)seq << rbits) + ((pos + 1u) & rmask)) * kstride + kvh * 128u);
        }
#ifdef LFM_SG64
        float sc0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y) * scale;
        float sc1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y) * scale;
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
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
#endif
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        float2 v0;
        float2 v1;
        if (pos < full_end) {
            v0 = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            v0 = kv_load2(lane, vst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
        if (pos + 1u < full_end) {
            v1 = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[(pos + 1u) >> 5], (pos + 1u) & 31u, kvh, kv_heads, lane);
        } else {
            v1 = kv_load2(lane, vst + (((ulong)seq << rbits) + ((pos + 1u) & rmask)) * kstride + kvh * 128u);
        }
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < len) {
        float2 k;
        if (pos < full_end) {
            k = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            k = kv_load2(lane, kst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
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
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        float2 v;
        if (pos < full_end) {
            v = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            v = kv_load2(lane, vst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
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

__kernel void attention_split_h_qk(__global const float *q, __global const KV_T *kst, __global const KV_T *vst,
                                   __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                                   __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                                   __global const uint *tbl, __global float *pacc, __global float *pml,
                                   uint len, uint start, uint full_end, uint kv_heads, uint n_heads, uint splits, uint seq, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint head = gid / splits;
    uint sp = gid % splits;
    if (head >= n_heads) {
        return;
    }
    __local float sred[128];
    __local float qs[128];
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
    uint kstride = kv_heads * 128u;
    uint pos = p0;
    for (; pos + 1u < p1; pos += 2u) {
        float2 k0;
        float2 k1;
        if (pos < full_end) {
            k0 = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            k0 = kv_load2(lane, kst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
        if (pos + 1u < full_end) {
            k1 = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[(pos + 1u) >> 5], (pos + 1u) & 31u, kvh, kv_heads, lane);
        } else {
            k1 = kv_load2(lane, kst + (((ulong)seq << rbits) + ((pos + 1u) & rmask)) * kstride + kvh * 128u);
        }
#ifdef LFM_SG64
        float sc0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y) * scale;
        float sc1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y) * scale;
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
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
#endif
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        float2 v0;
        float2 v1;
        if (pos < full_end) {
            v0 = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            v0 = kv_load2(lane, vst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
        if (pos + 1u < full_end) {
            v1 = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[(pos + 1u) >> 5], (pos + 1u) & 31u, kvh, kv_heads, lane);
        } else {
            v1 = kv_load2(lane, vst + (((ulong)seq << rbits) + ((pos + 1u) & rmask)) * kstride + kvh * 128u);
        }
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < p1) {
        float2 k;
        if (pos < full_end) {
            k = lfm_kvarn_k(kcode, ka, kc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            k = kv_load2(lane, kst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
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
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        float2 v;
        if (pos < full_end) {
            v = lfm_kvarn_v(vcode, vb, vbz, vc, (uint)tbl[pos >> 5], pos & 31u, kvh, kv_heads, lane);
        } else {
            v = kv_load2(lane, vst + (((ulong)seq << rbits) + (pos & rmask)) * kstride + kvh * 128u);
        }
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = mn;
    }
    qs[2u * lane] = acc0;
    qs[2u * lane + 1u] = acc1;
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

__kernel void attention_b_h_rot(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf,
                                __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                                __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                                __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b,
                                __global const uint *seq_b, __global const float *gate, __global float *out,
                                uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint seq_cap, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
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
    uint seq = seq_b[t];
    if (seq >= seq_cap) {
        return;
    }
    __local float sred[256];
    __local float qs[4 * 128];
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint full_end = (len / 32u) * 32u;
    uint kv_shift = (seq << rbits);
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
        float2 k;
        float2 v;
        if (pos < full_end) {
            k = lfm_kvarn_k(kcode, ka, kc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
            v = lfm_kvarn_v(vcode, vb, vbz, vc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
        } else {
            ulong slot = (ulong)(kv_shift + (pos & rmask));
            k = kv_load2(lane, kbuf + slot * kv_heads * 128u + kvh * 128u);
            v = kv_load2(lane, vbuf + slot * kv_heads * 128u + kvh * 128u);
        }
#ifdef LFM_SG64
        float sv = sub_group_reduce_add(q0 * k.x + q1 * k.y) * scale;
#else
        sred[lid] = q0 * k.x + q1 * k.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint st = 32u; st > 0u; st >>= 1) {
            if (lane < st) {
                sred[lid] += sred[lid + st];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float sv = sred[grp * 64u] * scale;
#endif
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float wgt = exp(sv - mn);
        l = l * corr + wgt;
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

__kernel void attention_split_b_h_rot(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf,
                                      __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                                      __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                                      __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b,
                                      __global const uint *seq_b, __global float *pacc, __global float *pml,
                                      uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits, uint seq_cap, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
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
    uint seq = seq_b[t];
    if (seq >= seq_cap) {
        return;
    }
    __local float sred[128];
    __local float qs[128];
    uint kvh = head / (n_heads / kv_heads);
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint full_end = (len / 32u) * 32u;
    uint kv_shift = (seq << rbits);
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
        float2 k0;
        float2 k1;
        float2 v0;
        float2 v1;
        if (pos < full_end) {
            k0 = lfm_kvarn_k(kcode, ka, kc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
            v0 = lfm_kvarn_v(vcode, vb, vbz, vc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
        } else {
            ulong sl0 = (ulong)(kv_shift + (pos & rmask));
            k0 = kv_load2(lane, kbuf + sl0 * kv_heads * 128u + kvh * 128u);
            v0 = kv_load2(lane, vbuf + sl0 * kv_heads * 128u + kvh * 128u);
        }
        if (pos + 1u < full_end) {
            k1 = lfm_kvarn_k(kcode, ka, kc, tbl[toff + ((pos + 1u) >> 5)], (pos + 1u) & 31u, kvh, kv_heads, lane);
            v1 = lfm_kvarn_v(vcode, vb, vbz, vc, tbl[toff + ((pos + 1u) >> 5)], (pos + 1u) & 31u, kvh, kv_heads, lane);
        } else {
            ulong sl1 = (ulong)(kv_shift + ((pos + 1u) & rmask));
            k1 = kv_load2(lane, kbuf + sl1 * kv_heads * 128u + kvh * 128u);
            v1 = kv_load2(lane, vbuf + sl1 * kv_heads * 128u + kvh * 128u);
        }
#ifdef LFM_SG64
        float sc0 = sub_group_reduce_add(q0 * k0.x + q1 * k0.y) * scale;
        float sc1 = sub_group_reduce_add(q0 * k1.x + q1 * k1.y) * scale;
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
        float sc0 = sred[0] * scale;
        float sc1 = sred[64u] * scale;
#endif
        float mn = fmax(m, fmax(sc0, sc1));
        float corr = exp(m - mn);
        float w0 = exp(sc0 - mn);
        float w1 = exp(sc1 - mn);
        l = l * corr + w0 + w1;
        acc0 = acc0 * corr + w0 * v0.x + w1 * v1.x;
        acc1 = acc1 * corr + w0 * v0.y + w1 * v1.y;
        m = mn;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (pos < p1) {
        float2 k;
        float2 v;
        if (pos < full_end) {
            k = lfm_kvarn_k(kcode, ka, kc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
            v = lfm_kvarn_v(vcode, vb, vbz, vc, tbl[toff + (pos >> 5)], pos & 31u, kvh, kv_heads, lane);
        } else {
            ulong sl = (ulong)(kv_shift + (pos & rmask));
            k = kv_load2(lane, kbuf + sl * kv_heads * 128u + kvh * 128u);
            v = kv_load2(lane, vbuf + sl * kv_heads * 128u + kvh * 128u);
        }
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
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        acc0 = acc0 * corr + w * v.x;
        acc1 = acc1 * corr + w * v.y;
        m = mn;
    }
    qs[2u * lane] = acc0;
    qs[2u * lane + 1u] = acc1;
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

static void lfm_tq_had8(float v[8], uint jlane) {
    #pragma unroll
    for (uint st = 1u; st <= 4u; st <<= 1) {
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            if ((i & st) == 0u) {
                float a = v[i];
                float b = v[i + st];
                v[i] = a + b;
                v[i + st] = a - b;
            }
        }
    }
    #pragma unroll
    for (uint mask = 1u; mask <= 8u; mask <<= 1) {
        float sgn = ((jlane & mask) == 0u) ? 1.0f : -1.0f;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            float b = sub_group_shuffle_xor(v[i], mask);
            v[i] = sgn * v[i] + b;
        }
    }
}

static void lfm_tq_rot_fwd8(float v[8], uint jlane) {
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        v[i] *= lfm_tq_sign(jlane * 8u + i);
    }
    lfm_tq_had8(v, jlane);
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        v[i] *= 0.0883883476f;
    }
}

static void lfm_tq_rot_inv8(float v[8], uint jlane) {
    lfm_tq_had8(v, jlane);
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        v[i] *= 0.0883883476f * lfm_tq_sign(jlane * 8u + i);
    }
}


static void lfm_kvarn_k8_pq(__global const uchar *kcode, float ct, const float a[8],
                            uint phys, uint tok, uint kvh, uint kv_heads, uint j, float k[8]) {
    __global const uchar *cp = kcode + ((phys << 5 | tok) * kv_heads + kvh) * 64u + j * 4u;
    uchar4 c = vload4(0, cp);
    uchar8 q = (uchar8)(
        c.s0 & 15u, (c.s0 >> 4) & 15u, c.s1 & 15u, (c.s1 >> 4) & 15u,
        c.s2 & 15u, (c.s2 >> 4) & 15u, c.s3 & 15u, (c.s3 >> 4) & 15u);
    float8 f = (convert_float8(q) - 8.0f) * vload8(0, a) * ct;
    vstore8(f, 0, k);
}

static void lfm_kvarn_v8_v(__global const uchar *vcode, float bt, float zero, const float sc[8],
                           uint phys, uint tok, uint kvh, uint kv_heads, uint j, float v[8]) {
    __global const uchar *cp = vcode + ((phys << 5 | tok) * kv_heads + kvh) * 32u + j * 2u;
    uchar b0 = cp[0];
    uchar b1 = cp[1];
    uchar8 q = (uchar8)(
        b0 & 3u, (b0 >> 2) & 3u, (b0 >> 4) & 3u, (b0 >> 6) & 3u,
        b1 & 3u, (b1 >> 2) & 3u, (b1 >> 4) & 3u, (b1 >> 6) & 3u);
    float8 f = convert_float8(q) * bt + zero;
    vstore8(f * vload8(0, sc), 0, v);
}

static void lfm_kvarn_v8_v4(__global const uchar *vcode, float bt, float zero, const float sc[8],
                            uint phys, uint tok, uint kvh, uint kv_heads, uint j, float v[8]) {
    __global const uchar *cp = vcode + ((phys << 5 | tok) * kv_heads + kvh) * 64u + j * 4u;
    uchar4 c = vload4(0, cp);
    uchar8 q = (uchar8)(
        c.s0 & 15u, (c.s0 >> 4) & 15u, c.s1 & 15u, (c.s1 >> 4) & 15u,
        c.s2 & 15u, (c.s2 >> 4) & 15u, c.s3 & 15u, (c.s3 >> 4) & 15u);
    float8 f = convert_float8(q) * bt + zero;
    vstore8(f * vload8(0, sc), 0, v);
}

static void lfm_kvarn_v8_best(__global const uchar *vcode, float bt, float zero, const float sc[8],
                              uint phys, uint tok, uint kvh, uint kv_heads, uint j, float v[8]) {
#ifdef LFM_KV4
    lfm_kvarn_v8_v4(vcode, bt, zero, sc, phys, tok, kvh, kv_heads, j, v);
#else
    lfm_kvarn_v8_v(vcode, bt, zero, sc, phys, tok, kvh, kv_heads, j, v);
#endif
}

__kernel void attention_split_b_fa_qk(__global const float *q, __global const KV_T *kbuf, __global const KV_T *vbuf,
                                      __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                                      __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                                      __global const uint *tbl, __global const uint *tbl_off, __global const uint *pos_b,
                                      __global const uint *seq_b, __global float *pacc, __global float *pml,
                                      uint ntok, uint kv_heads, uint head_dim, uint n_heads, uint window, uint splits, uint seq_cap, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
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
    uint seq = seq_b[t];
    if (seq >= seq_cap) {
        return;
    }
    uint h = lane >> 4;
    uint j = lane & 15u;
    uint head = kvh * (n_heads / kv_heads) + h;
    uint len = pos_b[t] + 1u;
    uint start = (window > 0u && len > window) ? len - window : 0u;
    uint toff = tbl_off[t];
    uint full_end = (len / 32u) * 32u;
    uint kv_shift = (seq << rbits);
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    float qv[8];
    vstore8(vload8(0, q + ((ulong)t * n_heads + head) * head_dim + j * 8u), 0, qv);
    lfm_tq_rot_fwd8(qv, j);
    float m = -INFINITY;
    float l = 0.0f;
    float acc[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        acc[i] = 0.0f;
    }
    float scale = 1.0f / sqrt((float)head_dim);
    uint kvd = kv_heads * head_dim;
    uint last_phys = 0xFFFFFFFFu;
    float ka8[8];
    float vc8[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        ka8[i] = 1.0f;
        vc8[i] = 1.0f;
    }
    uint pos = p0;
    for (; pos + 3u < p1; pos += 4u) {
        float s[4];
        float vv[4][8];
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            uint pp = pos + u;
            float k8[8];
            if (pp < full_end) {
                uint phys = tbl[toff + (pp >> 5)];
                if (phys != last_phys) {
                    uint base = phys * kv_heads + kvh;
                    vstore8(convert_float8(kv_load8(0, ka + base * 128u + j * 8u)), 0, ka8);
                    vstore8(convert_float8(kv_load8(0, vc + base * 128u + j * 8u)), 0, vc8);
                    last_phys = phys;
                }
                uint base = phys * kv_heads + kvh;
                float ct = (float)kc[base * 32u + (pp & 31u)];
                float bt = (float)vb[base * 32u + (pp & 31u)];
                float zero = (float)vbz[base * 32u + (pp & 31u)];
                lfm_kvarn_k8_pq(kcode, ct, ka8, phys, pp & 31u, kvh, kv_heads, j, k8);
                lfm_kvarn_v8_best(vcode, bt, zero, vc8, phys, pp & 31u, kvh, kv_heads, j, vv[u]);
            } else {
                ulong slot = (ulong)(kv_shift + (pp & rmask));
                vstore8(convert_float8(kv_load8(0, kbuf + slot * kvd + kvh * head_dim + j * 8u)), 0, k8);
                    vstore8(convert_float8(kv_load8(0, vbuf + slot * kvd + kvh * head_dim + j * 8u)), 0, vv[u]);
            }
            float d = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                d += qv[i] * k8[i];
            }
            d += sub_group_shuffle_xor(d, 8u);
            d += sub_group_shuffle_xor(d, 4u);
            d += sub_group_shuffle_xor(d, 2u);
            d += sub_group_shuffle_xor(d, 1u);
            s[u] = d * scale;
        }
        float mn = fmax(fmax(m, s[0]), fmax(fmax(s[1], s[2]), s[3]));
        float corr = exp(m - mn);
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            float w = exp(s[u] - mn);
            l = l * (u == 0u ? corr : 1.0f) + w;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                acc[i] = acc[i] * (u == 0u ? corr : 1.0f) + w * vv[u][i];
            }
        }
        m = mn;
    }
    for (; pos < p1; ++pos) {
        float k8[8];
        float v8[8];
        if (pos < full_end) {
            uint phys = tbl[toff + (pos >> 5)];
            if (phys != last_phys) {
                uint base = phys * kv_heads + kvh;
                vstore8(convert_float8(kv_load8(0, ka + base * 128u + j * 8u)), 0, ka8);
                vstore8(convert_float8(kv_load8(0, vc + base * 128u + j * 8u)), 0, vc8);
                last_phys = phys;
            }
            uint base = phys * kv_heads + kvh;
            float ct = (float)kc[base * 32u + (pos & 31u)];
            float bt = (float)vb[base * 32u + (pos & 31u)];
            float zero = (float)vbz[base * 32u + (pos & 31u)];
            lfm_kvarn_k8_pq(kcode, ct, ka8, phys, pos & 31u, kvh, kv_heads, j, k8);
            lfm_kvarn_v8_best(vcode, bt, zero, vc8, phys, pos & 31u, kvh, kv_heads, j, v8);
        } else {
            ulong slot = (ulong)(kv_shift + (pos & rmask));
            vstore8(convert_float8(kv_load8(0, kbuf + slot * kvd + kvh * head_dim + j * 8u)), 0, k8);
            vstore8(convert_float8(kv_load8(0, vbuf + slot * kvd + kvh * head_dim + j * 8u)), 0, v8);
        }
        float d = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            d += qv[i] * k8[i];
        }
        d += sub_group_shuffle_xor(d, 8u);
        d += sub_group_shuffle_xor(d, 4u);
        d += sub_group_shuffle_xor(d, 2u);
        d += sub_group_shuffle_xor(d, 1u);
        float sv = d * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            acc[i] = acc[i] * corr + w * v8[i];
        }
        m = mn;
    }
    lfm_tq_rot_inv8(acc, j);
    ulong pbase = ((ulong)t * n_heads + head) * splits + sp;
    __global float *pa = pacc + pbase * head_dim;
    vstore8(vload8(0, acc), 0, pa + j * 8u);
    if (j == 0u) {
        pml[pbase * 2u] = m;
        pml[pbase * 2u + 1u] = l;
    }
}


__kernel void attention_split_h_fa_qk(__global const float *q, __global const KV_T *kst, __global const KV_T *vst,
                                      __global const uchar *kcode, __global const KV_T *ka, __global const KV_T *kc,
                                      __global const uchar *vcode, __global const KV_T *vb, __global const KV_T *vbz, __global const KV_T *vc,
                                      __global const uint *tbl, __global float *pacc, __global float *pml,
                                      uint len, uint start, uint full_end, uint kv_heads, uint n_heads, uint splits, uint seq, uint rbits) {
    uint rmask = (1u << rbits) - 1u;
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    uint per_head = splits;
    uint kvh = gid / per_head;
    uint sp = gid % per_head;
    uint h = lane >> 4;
    uint j = lane & 15u;
    uint head = kvh * (n_heads / kv_heads) + h;
    uint span = len - start;
    uint chunk = (span + splits - 1u) / splits;
    uint p0 = start + sp * chunk;
    uint p1 = min(p0 + chunk, len);
    float qv[8];
    vstore8(vload8(0, q + head * 128u + j * 8u), 0, qv);
    lfm_tq_rot_fwd8(qv, j);
    float m = -INFINITY;
    float l = 0.0f;
    float acc[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        acc[i] = 0.0f;
    }
    float scale = 0.0883883476f;
    uint kst_stride = kv_heads * 128u;
    uint stage_shift = rbits;
    uint last_phys = 0xFFFFFFFFu;
    float ka8[8];
    float vc8[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
        ka8[i] = 1.0f;
        vc8[i] = 1.0f;
    }
    uint pos = p0;
    for (; pos + 3u < p1; pos += 4u) {
        float s[4];
        float vv[4][8];
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            uint pp = pos + u;
            float k8[8];
            if (pp < full_end) {
                uint phys = tbl[pp >> 5];
                if (phys != last_phys) {
                    uint base = phys * kv_heads + kvh;
                    vstore8(convert_float8(kv_load8(0, ka + base * 128u + j * 8u)), 0, ka8);
                    vstore8(convert_float8(kv_load8(0, vc + base * 128u + j * 8u)), 0, vc8);
                    last_phys = phys;
                }
                uint base = phys * kv_heads + kvh;
                float ct = (float)kc[base * 32u + (pp & 31u)];
                float bt = (float)vb[base * 32u + (pp & 31u)];
                float zero = (float)vbz[base * 32u + (pp & 31u)];
                lfm_kvarn_k8_pq(kcode, ct, ka8, phys, pp & 31u, kvh, kv_heads, j, k8);
                lfm_kvarn_v8_best(vcode, bt, zero, vc8, phys, pp & 31u, kvh, kv_heads, j, vv[u]);
            } else {
                ulong slot = ((ulong)seq << stage_shift) + (pp & rmask);
                vstore8(convert_float8(kv_load8(0, kst + slot * kst_stride + kvh * 128u + j * 8u)), 0, k8);
                    vstore8(convert_float8(kv_load8(0, vst + slot * kst_stride + kvh * 128u + j * 8u)), 0, vv[u]);
            }
            float d = 0.0f;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                d += qv[i] * k8[i];
            }
            d += sub_group_shuffle_xor(d, 8u);
            d += sub_group_shuffle_xor(d, 4u);
            d += sub_group_shuffle_xor(d, 2u);
            d += sub_group_shuffle_xor(d, 1u);
            s[u] = d * scale;
        }
        float mn = fmax(fmax(m, s[0]), fmax(fmax(s[1], s[2]), s[3]));
        float corr = exp(m - mn);
        #pragma unroll
        for (uint u = 0u; u < 4u; ++u) {
            float w = exp(s[u] - mn);
            l = l * (u == 0u ? corr : 1.0f) + w;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                acc[i] = acc[i] * (u == 0u ? corr : 1.0f) + w * vv[u][i];
            }
        }
        m = mn;
    }
    for (; pos < p1; ++pos) {
        float k8[8];
        float v8[8];
        if (pos < full_end) {
            uint phys = tbl[pos >> 5];
            if (phys != last_phys) {
                uint base = phys * kv_heads + kvh;
                vstore8(convert_float8(kv_load8(0, ka + base * 128u + j * 8u)), 0, ka8);
                vstore8(convert_float8(kv_load8(0, vc + base * 128u + j * 8u)), 0, vc8);
                last_phys = phys;
            }
            uint base = phys * kv_heads + kvh;
            float ct = (float)kc[base * 32u + (pos & 31u)];
            float bt = (float)vb[base * 32u + (pos & 31u)];
            float zero = (float)vbz[base * 32u + (pos & 31u)];
            lfm_kvarn_k8_pq(kcode, ct, ka8, phys, pos & 31u, kvh, kv_heads, j, k8);
            lfm_kvarn_v8_best(vcode, bt, zero, vc8, phys, pos & 31u, kvh, kv_heads, j, v8);
        } else {
            ulong slot = ((ulong)seq << stage_shift) + (pos & rmask);
            vstore8(convert_float8(kv_load8(0, kst + slot * kst_stride + kvh * 128u + j * 8u)), 0, k8);
            vstore8(convert_float8(kv_load8(0, vst + slot * kst_stride + kvh * 128u + j * 8u)), 0, v8);
        }
        float d = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            d += qv[i] * k8[i];
        }
        d += sub_group_shuffle_xor(d, 8u);
        d += sub_group_shuffle_xor(d, 4u);
        d += sub_group_shuffle_xor(d, 2u);
        d += sub_group_shuffle_xor(d, 1u);
        float sv = d * scale;
        float mn = fmax(m, sv);
        float corr = exp(m - mn);
        float w = exp(sv - mn);
        l = l * corr + w;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
            acc[i] = acc[i] * corr + w * v8[i];
        }
        m = mn;
    }
    lfm_tq_rot_inv8(acc, j);
    ulong pbase = ((ulong)head * splits + sp) * 128u;
    vstore8(vload8(0, acc), 0, pacc + pbase + j * 8u);
    if (j == 0u) {
        pml[((ulong)head * splits + sp) * 2u] = m;
        pml[((ulong)head * splits + sp) * 2u + 1u] = l;
    }
}
