#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../common/fp16.h"
#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/q4.h"

#include "hvx_ops.h"

#define LFM_STUB_WEAK __attribute__((weak))

LFM_STUB_WEAK void lfm_hvx_gemv_i4(const uint8_t *img, int n, int k, const int8_t *q,
                                   const float *ascale, float *y)
{
    lfm_q4_gemv_i8(img, n, k, q, ascale, y);
}

LFM_STUB_WEAK void lfm_hvx_gemm_i4(const uint8_t *img, int n, int k, const int8_t *q, int m,
                                   const float *ascale, float *y)
{
    lfm_q4_gemm_i8(img, n, k, q, m, ascale, y);
}

static void stub_f16_store_token(uint8_t *dst, int token, int head_dim, const float *src)
{
    int d;
    for (d = 0; d < head_dim; d++)
        lfm_f16_store(dst + (size_t)(token * head_dim + d) * 2u, lfm_f32_to_f16(src[d]));
}

static void stub_f16_load_token(const uint8_t *src, int token, int head_dim, float *dst)
{
    int d;
    for (d = 0; d < head_dim; d++)
        dst[d] = lfm_f16_to_f32(lfm_f16_load(src + (size_t)(token * head_dim + d) * 2u));
}

static void stub_kvarn_v_encode(lfm_kv_format fmt, uint8_t *head_page, const float *v)
{
    uint8_t *vcode = head_page + lfm_kvarn_vcode_off(fmt);
    size_t vbytes = lfm_kvarn_vcode_bytes(fmt);
    uint8_t *vb = head_page + lfm_kvarn_vcode_off(fmt) + vbytes;
    uint8_t *vbz = vb + LFM_KVARN_VB_BYTES;
    uint8_t *vc = vbz + LFM_KVARN_VBZ_BYTES;
    int token;
    int ch;
    int vmax = (1 << lfm_kvarn_v_bits(fmt)) - 1;
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++)
        lfm_f16_store(vc + (size_t)ch * 2u, lfm_f32_to_f16(1.0f));
    for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
        const float *src = v + (size_t)token * LFM_KVARN_HEAD_DIM;
        float mn = src[0];
        float mx = src[0];
        float step;
        for (ch = 1; ch < LFM_KVARN_HEAD_DIM; ch++) {
            if (src[ch] < mn)
                mn = src[ch];
            if (src[ch] > mx)
                mx = src[ch];
        }
        step = (mx - mn) / (float)vmax;
        if (step < LFM_KVARN_VAR_FLOOR)
            step = LFM_KVARN_VAR_FLOOR;
        lfm_f16_store(vb + (size_t)token * 2u, lfm_f32_to_f16(step));
        lfm_f16_store(vbz + (size_t)token * 2u, lfm_f32_to_f16(mn));
        if (fmt == LFM_KVARN_K4V4) {
            for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch += 2) {
                int q0 = lfm_kvarn_round_i((src[ch] - mn) / step);
                int q1 = lfm_kvarn_round_i((src[ch + 1] - mn) / step);
                if (q0 < 0)
                    q0 = 0;
                if (q0 > vmax)
                    q0 = vmax;
                if (q1 < 0)
                    q1 = 0;
                if (q1 > vmax)
                    q1 = vmax;
                vcode[(size_t)token * (LFM_KVARN_HEAD_DIM / 2) + (size_t)(ch / 2)] =
                    (uint8_t)(q0 | (q1 << 4));
            }
        } else {
            for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch += 4) {
                uint8_t value = 0;
                int k;
                for (k = 0; k < 4; k++) {
                    int q = lfm_kvarn_round_i((src[ch + k] - mn) / step);
                    if (q < 0)
                        q = 0;
                    if (q > 3)
                        q = 3;
                    value = (uint8_t)(value | (q << (2 * k)));
                }
                vcode[(size_t)token * (LFM_KVARN_HEAD_DIM / 4) + (size_t)(ch / 4)] = value;
            }
        }
    }
}

static void stub_kvarn_v_decode(lfm_kv_format fmt, const uint8_t *head_page, int token, float *out)
{
    size_t vbytes = lfm_kvarn_vcode_bytes(fmt);
    const uint8_t *vcode = head_page + lfm_kvarn_vcode_off(fmt);
    const uint8_t *vb = head_page + lfm_kvarn_vcode_off(fmt) + vbytes;
    const uint8_t *vbz = vb + LFM_KVARN_VB_BYTES;
    const uint8_t *vc = vbz + LFM_KVARN_VBZ_BYTES;
    float step = lfm_f16_to_f32(lfm_f16_load(vb + (size_t)token * 2u));
    float zero = lfm_f16_to_f32(lfm_f16_load(vbz + (size_t)token * 2u));
    int ch;
    for (ch = 0; ch < LFM_KVARN_HEAD_DIM; ch++) {
        int q;
        if (fmt == LFM_KVARN_K4V4) {
            uint8_t b = vcode[(size_t)token * (LFM_KVARN_HEAD_DIM / 2) + (size_t)(ch / 2)];
            q = (ch % 2 == 0) ? (int)(b & 0xF) : (int)(b >> 4);
        } else {
            uint8_t b = vcode[(size_t)token * (LFM_KVARN_HEAD_DIM / 4) + (size_t)(ch / 4)];
            q = (int)((b >> (2 * (ch % 4))) & 3);
        }
        out[ch] = ((float)q * step + zero) * lfm_f16_to_f32(lfm_f16_load(vc + (size_t)ch * 2u));
    }
}

LFM_STUB_WEAK void lfm_hvx_kvcache_commit(lfm_kv_format fmt, uint8_t *page, const float *k_rot,
                                          const float *v_rot, int head, int head_dim)
{
    int token;
    if (page == NULL || k_rot == NULL || v_rot == NULL || head < 0)
        return;
    if (fmt == LFM_KV_F16) {
        size_t block = (size_t)LFM_PAGE_TOKENS * (size_t)head_dim * 2u * 2u;
        uint8_t *hp = page + (size_t)head * block;
        uint8_t *kf = hp;
        uint8_t *vf = hp + (size_t)LFM_PAGE_TOKENS * (size_t)head_dim * 2u;
        for (token = 0; token < LFM_PAGE_TOKENS; token++) {
            stub_f16_store_token(kf, token, head_dim, k_rot + (size_t)token * head_dim);
            stub_f16_store_token(vf, token, head_dim, v_rot + (size_t)token * head_dim);
        }
        return;
    }
    {
        uint8_t *hp = page + (size_t)head * lfm_kvarn_head_bytes(fmt);
        uint16_t src[LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM];
        for (token = 0; token < LFM_KVARN_PAGE_TOKENS; token++) {
            int d;
            for (d = 0; d < LFM_KVARN_HEAD_DIM; d++)
                src[(size_t)token * LFM_KVARN_HEAD_DIM + d] =
                    lfm_f32_to_f16(k_rot[(size_t)token * head_dim + d]);
        }
        lfm_kvarn_commit(fmt, src, hp);
        stub_kvarn_v_encode(fmt, hp, v_rot);
    }
}

static float stub_dot_f32(const float *a, const float *b, int n)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

LFM_STUB_WEAK void lfm_hvx_attn_flash(lfm_kv_format fmt, const uint32_t *tbl,
                                      const uint8_t *kv_pages, int kv_heads, int n_heads,
                                      int head_dim, const float *q_rot, uint32_t ntok,
                                      uint32_t start, uint32_t window, float *out_rot,
                                      float *scratch)
{
    size_t page_bytes = lfm_page_bytes((uint32_t)fmt, (uint32_t)kv_heads, (uint32_t)head_dim);
    size_t head_bytes = fmt == LFM_KV_F16
                            ? (size_t)LFM_PAGE_TOKENS * (size_t)head_dim * 2u * 2u
                            : lfm_kvarn_head_bytes(fmt);
    int group = kv_heads > 0 ? n_heads / kv_heads : 0;
    uint32_t qi;
    (void)scratch;

    if (tbl == NULL || kv_pages == NULL || q_rot == NULL || out_rot == NULL || group <= 0)
        return;
    for (qi = 0; qi < ntok; qi++) {
        uint32_t pos = start + qi;
        uint32_t kstart = window == 0u ? 0u : (pos + 1u > window ? pos + 1u - window : 0u);
        uint32_t n = pos - kstart + 1u;
        int h;
        for (h = 0; h < n_heads; h++) {
            int kh = h / group;
            const float *qv = q_rot + ((size_t)qi * (size_t)n_heads + (size_t)h) * head_dim;
            float *ov = out_rot + ((size_t)qi * (size_t)n_heads + (size_t)h) * head_dim;
            float *scores = scratch;
            float mx = -INFINITY;
            float sum = 0.0f;
            uint32_t p;
            int d;
            if (scores == NULL)
                return;
            for (d = 0; d < head_dim; d++)
                ov[d] = 0.0f;
            for (p = kstart; p <= pos; p++) {
                uint32_t pg = tbl[p >> 5];
                const uint8_t *hp = kv_pages + (size_t)pg * page_bytes + (size_t)kh * head_bytes;
                float kv[LFM_KVARN_HEAD_DIM];
                int slot = (int)(p & 31u);
                float s;
                if (fmt == LFM_KV_F16)
                    stub_f16_load_token(hp, slot, head_dim, kv);
                else
                    lfm_kvarn_decode_k(hp, slot, kv);
                s = stub_dot_f32(qv, kv, head_dim) / sqrtf((float)head_dim);
                scores[p - kstart] = s;
                if (s > mx)
                    mx = s;
            }
            for (p = 0; p < n; p++) {
                scores[p] = expf(scores[p] - mx);
                sum += scores[p];
            }
            if (sum > 0.0f) {
                for (p = 0; p < n; p++)
                    scores[p] /= sum;
            }
            for (p = kstart; p <= pos; p++) {
                uint32_t pg = tbl[p >> 5];
                const uint8_t *hp = kv_pages + (size_t)pg * page_bytes + (size_t)kh * head_bytes;
                float vv[LFM_KVARN_HEAD_DIM];
                int slot = (int)(p & 31u);
                float prob = scores[p - kstart];
                if (fmt == LFM_KV_F16)
                    stub_f16_load_token(hp + (size_t)LFM_PAGE_TOKENS * (size_t)head_dim * 2u, slot,
                                        head_dim, vv);
                else
                    stub_kvarn_v_decode(fmt, hp, slot, vv);
                for (d = 0; d < head_dim; d++)
                    ov[d] += prob * vv[d];
            }
        }
    }
}
