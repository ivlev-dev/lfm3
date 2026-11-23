#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/fp16.h"
#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/q4.h"
#include "../common/quant.h"
#include "../common/rot.h"

#include "exec.h"
#include "hvx_ops.h"

static int failures;

#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                            \
            printf(__VA_ARGS__);                                                                   \
            printf("\n");                                                                          \
        }                                                                                          \
    } while (0)

static unsigned g_seed = 12345u;

static float rnd(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)(g_seed >> 8) / 16777216.0f - 0.5f;
}

static float rndn(void)
{
    return (rnd() + rnd() + rnd()) * 1.2f;
}

static float dot_f32(const float *a, const float *b, int n)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static float cos_f32(const float *a, const float *b, int n)
{
    float d = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f)
        return 0.0f;
    {
        float c = d / sqrtf(na * nb);
        return c > 1.0f ? 1.0f : c;
    }
}

static uint8_t *make_image(const float *w, int n, int k)
{
    uint8_t *img = (uint8_t *)malloc(lfm_q4_image_bytes(n, k));
    if (img == NULL)
        return NULL;
    memset(img, 0, lfm_q4_image_bytes(n, k));
    lfm_q4_pack_matrix(w, n, k, img, 128);
    return img;
}

static void test_ffn(int N, int K)
{
    size_t kn = (size_t)K;
    size_t nn = (size_t)N;
    float *wg = (float *)malloc(nn * kn * sizeof(float));
    float *wu = (float *)malloc(nn * kn * sizeof(float));
    float *wd = (float *)malloc(nn * kn * sizeof(float));
    float *x = (float *)malloc(kn * sizeof(float));
    float *xhat = (float *)malloc(kn * sizeof(float));
    float *row = (float *)malloc(kn * sizeof(float));
    float *midh = (float *)malloc(nn * sizeof(float));
    float *mid = (float *)malloc(nn * sizeof(float));
    float *ref = (float *)malloc(nn * sizeof(float));
    float *refq = (float *)malloc(nn * sizeof(float));
    float *out = (float *)malloc(nn * sizeof(float));
    int8_t *xq = (int8_t *)malloc(kn);
    float *xs = (float *)malloc((size_t)(K / 32) * sizeof(float));
    int8_t *mq = (int8_t *)malloc(nn);
    float *ms = (float *)malloc((size_t)(N / 32) * sizeof(float));
    uint8_t *ig;
    uint8_t *iu;
    uint8_t *id;
    float c_true;
    float c_q;
    int i;
    int r;

    CHECK(wg && wu && wd && x && xhat && row && midh && mid && ref && refq && out && xq && xs &&
              mq && ms,
          "ffn alloc");
    if (!(wg && wu && wd && x && xhat && row && midh && mid && ref && refq && out && xq && xs &&
          mq && ms))
        return;
    g_seed = 4242u;
    for (i = 0; i < N * K; i++) {
        wg[i] = rndn();
        wu[i] = rndn();
        wd[i] = rndn();
    }
    for (i = 0; i < K; i++)
        x[i] = rndn();
    ig = make_image(wg, N, K);
    iu = make_image(wu, N, K);
    id = make_image(wd, N, K);
    CHECK(ig && iu && id, "ffn images");
    if (!(ig && iu && id))
        return;

    CHECK(lfm_act_i8_rot(x, K, 128, xq, xs) == 0, "ffn act quant");
    lfm_hvx_ffn_i4(ig, N, K, iu, id, xq, xs, out);

    for (i = 0; i < N; i++) {
        float g = dot_f32(wg + (size_t)i * K, x, K);
        float u = dot_f32(wu + (size_t)i * K, x, K);
        mid[i] = (g / (1.0f + expf(-g))) * u;
    }
    for (r = 0; r < N; r++)
        ref[r] = dot_f32(wd + (size_t)r * K, mid, K);
    c_true = cos_f32(out, ref, N);

    lfm_deq_i8_group32(xq, xs, xhat, K);
    for (i = 0; i < N; i++) {
        float g;
        float u;
        lfm_q4_dequant_row(lfm_q4_row_qs(ig, N, K, i), lfm_q4_row_scales(ig, N, K, i), K, row);
        g = dot_f32(row, xhat, K);
        lfm_q4_dequant_row(lfm_q4_row_qs(iu, N, K, i), lfm_q4_row_scales(iu, N, K, i), K, row);
        u = dot_f32(row, xhat, K);
        midh[i] = (g / (1.0f + expf(-g))) * u;
    }
    CHECK(lfm_act_i8_rot(midh, N, 128, mq, ms) == 0, "ffn mid quant");
    {
        int g;
        for (g = 0; g < N / 32; g++) {
            for (i = 0; i < 32; i++)
                mid[g * 32 + i] = (float)mq[g * 32 + i] * ms[g];
        }
    }
    for (r = 0; r < N; r++) {
        lfm_q4_dequant_row(lfm_q4_row_qs(id, N, N, r), lfm_q4_row_scales(id, N, N, r), N, row);
        refq[r] = dot_f32(row, mid, N);
    }
    c_q = cos_f32(out, refq, N);

    printf("ffn[%dx%d]: cos(int4 kernel vs f32 pipeline on dequantised weights/act)=%.7f\n", N, K,
           c_q);
    printf("ffn[%dx%d]: cos(kernel vs original f32 weights/input)=%.7f (4-bit format limit)\n", N, K,
           c_true);
    CHECK(c_q >= 0.998f, "ffn quant-emulation cos %.7f < 0.998", c_q);
    CHECK(c_true >= 0.98f, "ffn true-f32 cos %.7f < 0.98", c_true);

    free(wg);
    free(wu);
    free(wd);
    free(x);
    free(xhat);
    free(row);
    free(midh);
    free(mid);
    free(ref);
    free(refq);
    free(out);
    free(xq);
    free(xs);
    free(mq);
    free(ms);
    free(ig);
    free(iu);
    free(id);
}

enum {
    T_H = 64,
    T_F = 128,
    T_V = 128,
    T_NL = 2,
    T_D = 128,
    T_HQ = 1,
    T_HKV = 1,
    T_CTX = 64,
    T_WIN = 8,
    T_KSTAGE = 64,
    T_MAXSEQ = 2,
    T_POOL = 2
};

static float *g_embed;
static float *g_out_norm;
static float *g_lm_head;
static float *g_an[T_NL];
static float *g_pan[T_NL];
static float *g_fnn[T_NL];
static float *g_pfn[T_NL];
static float *g_cw[T_NL];
static float *g_qn[T_NL];
static float *g_kn[T_NL];
static uint8_t *g_fg[T_NL];
static uint8_t *g_fu[T_NL];
static uint8_t *g_fd[T_NL];
static uint8_t *g_wq[T_NL];
static uint8_t *g_wk[T_NL];
static uint8_t *g_wv[T_NL];
static uint8_t *g_wg[T_NL];
static uint8_t *g_wo[T_NL];
static uint8_t *g_cin[T_NL];
static uint8_t *g_cout[T_NL];

static float *make_vec(int n, float base, float spread)
{
    float *v = (float *)malloc((size_t)n * sizeof(float));
    int i;
    if (v == NULL)
        return NULL;
    for (i = 0; i < n; i++)
        v[i] = base + spread * rnd();
    return v;
}

static float *make_mat(int n, int k, float spread)
{
    float *v = (float *)malloc((size_t)n * (size_t)k * sizeof(float));
    int i;
    if (v == NULL)
        return NULL;
    for (i = 0; i < n * k; i++)
        v[i] = spread * rnd();
    return v;
}

static uint8_t *pack_mat(float *w, int n, int k)
{
    uint8_t *img = make_image(w, n, k);
    free(w);
    return img;
}

static int build_model(lfm_exec_model *m)
{
    int l;
    g_seed = 2024u;
    memset(m, 0, sizeof *m);
    g_embed = make_mat(T_V, T_H, 0.5f);
    g_out_norm = make_vec(T_H, 1.0f, 0.1f);
    g_lm_head = make_mat(T_V, T_H, 0.5f);
    m->embed = g_embed;
    m->out_norm = g_out_norm;
    m->lm_head = g_lm_head;
    m->n_layers = T_NL;
    m->layer_kind[0] = LFM_LAYER_CONV;
    m->layer_kind[1] = LFM_LAYER_SLIDING;
    for (l = 0; l < T_NL; l++) {
        lfm_exec_layer *w = &m->layer[l];
        g_an[l] = make_vec(T_H, 1.0f, 0.1f);
        g_pan[l] = make_vec(T_H, 1.0f, 0.1f);
        g_fnn[l] = make_vec(T_H, 1.0f, 0.1f);
        g_pfn[l] = make_vec(T_H, 1.0f, 0.1f);
        w->attn_norm = g_an[l];
        w->post_attn_norm = g_pan[l];
        w->ffn_norm = g_fnn[l];
        w->post_ffw_norm = g_pfn[l];
        g_fg[l] = pack_mat(make_mat(T_F, T_H, 0.5f), T_F, T_H);
        g_fu[l] = pack_mat(make_mat(T_F, T_H, 0.5f), T_F, T_H);
        g_fd[l] = pack_mat(make_mat(T_H, T_F, 0.5f), T_H, T_F);
        w->ffn_gate = g_fg[l];
        w->ffn_up = g_fu[l];
        w->ffn_down = g_fd[l];
        if (l == 0) {
            g_cin[l] = pack_mat(make_mat(3 * T_H, T_H, 0.5f), 3 * T_H, T_H);
            g_cout[l] = pack_mat(make_mat(T_H, T_H, 0.5f), T_H, T_H);
            g_cw[l] = make_mat(T_H, LFM_EXEC_CONV_TAPS, 0.4f);
            w->conv_in = g_cin[l];
            w->conv_out = g_cout[l];
            w->conv_w = g_cw[l];
        } else {
            g_wq[l] = pack_mat(make_mat(T_HQ * T_D, T_H, 0.5f), T_HQ * T_D, T_H);
            g_wk[l] = pack_mat(make_mat(T_HKV * T_D, T_H, 0.5f), T_HKV * T_D, T_H);
            g_wv[l] = pack_mat(make_mat(T_HKV * T_D, T_H, 0.5f), T_HKV * T_D, T_H);
            g_wg[l] = pack_mat(make_mat(T_HQ, T_H, 0.5f), T_HQ, T_H);
            g_wo[l] = pack_mat(make_mat(T_H, T_HQ * T_D, 0.5f), T_H, T_HQ * T_D);
            g_qn[l] = make_vec(T_D, 1.0f, 0.1f);
            g_kn[l] = make_vec(T_D, 1.0f, 0.1f);
            w->attn_q = g_wq[l];
            w->attn_k = g_wk[l];
            w->attn_v = g_wv[l];
            w->attn_gate = g_wg[l];
            w->attn_out = g_wo[l];
            w->q_norm = g_qn[l];
            w->k_norm = g_kn[l];
        }
    }
    if (!m->embed || !m->out_norm || !m->lm_head)
        return -1;
    for (l = 0; l < T_NL; l++) {
        const lfm_exec_layer *w = &m->layer[l];
        if (!w->attn_norm || !w->post_attn_norm || !w->ffn_norm || !w->post_ffw_norm ||
            !w->ffn_gate || !w->ffn_up || !w->ffn_down)
            return -1;
    }
    return 0;
}

static void free_model(void)
{
    int l;
    free(g_embed);
    free(g_out_norm);
    free(g_lm_head);
    g_embed = NULL;
    g_out_norm = NULL;
    g_lm_head = NULL;
    for (l = 0; l < T_NL; l++) {
        free(g_an[l]);
        free(g_pan[l]);
        free(g_fnn[l]);
        free(g_pfn[l]);
        free(g_cw[l]);
        free(g_qn[l]);
        free(g_kn[l]);
        free(g_fg[l]);
        free(g_fu[l]);
        free(g_fd[l]);
        free(g_wq[l]);
        free(g_wk[l]);
        free(g_wv[l]);
        free(g_wg[l]);
        free(g_wo[l]);
        free(g_cin[l]);
        free(g_cout[l]);
    }
}

static void fill_desc(lfm_model_desc *d)
{
    memset(d, 0, sizeof *d);
    d->hidden = T_H;
    d->ffn = T_F;
    d->n_layers = T_NL;
    d->n_heads = T_HQ;
    d->head_dim = T_D;
    d->kv_dim = T_HKV * T_D;
    d->vocab = T_V;
    d->ctx = T_CTX;
    d->conv_cache = LFM_EXEC_CONV_HISTORY;
    d->sliding_window = T_WIN;
    d->kv_layers = T_NL;
    d->kv_heads = T_HKV;
    d->kv_pool = T_POOL;
    d->kv_stage = T_KSTAGE;
    d->max_seqs = T_MAXSEQ;
    d->kv_format = LFM_KV_F16;
}

static int run_rows(const lfm_model_desc *d, void *arena, const lfm_row *rows, uint32_t n,
                    uint32_t window, float *logits)
{
    lfm_step step;
    uint32_t i;
    memset(&step, 0, sizeof step);
    step.n_rows = n;
    step.window = window;
    step.max_rows = T_MAXSEQ;
    for (i = 0; i < n; i++) {
        step.ntok += rows[i].ntok;
        if (rows[i].ntok > 1u)
            step.flags |= LFM_FLAG_PREFILL;
    }
    return lfm_hvx_exec(d, &step, rows, arena, logits);
}

static void ref_rms(const float *x, const float *w, int n, float *y)
{
    float acc = 0.0f;
    float inv;
    int i;
    for (i = 0; i < n; i++)
        acc += x[i] * x[i];
    inv = 1.0f / sqrtf(acc / (float)n + 1e-5f);
    for (i = 0; i < n; i++)
        y[i] = x[i] * inv * w[i];
}

static void ref_rope(float *v, uint32_t pos, int dim, int pairs, double theta)
{
    int half = dim / 2;
    int i;
    for (i = 0; i < pairs; i++) {
        double inv = 1.0 / pow(theta, (2.0 * (double)i) / (double)dim);
        double a = (double)pos * inv;
        float sn = (float)sin(a);
        float cs = (float)cos(a);
        float x0 = v[i];
        float x1 = v[i + half];
        v[i] = x0 * cs - x1 * sn;
        v[i + half] = x1 * cs + x0 * sn;
    }
}

static void ref_head_norm(float *v, const float *w, int dim)
{
    float acc = 0.0f;
    float inv;
    int i;
    for (i = 0; i < dim; i++)
        acc += v[i] * v[i];
    inv = 1.0f / sqrtf(acc / (float)dim + 1e-5f);
    for (i = 0; i < dim; i++)
        v[i] = v[i] * inv * w[i];
}

static void ref_proj(const uint8_t *img, int n, int k, const float *x, float *y)
{
    int8_t *q = (int8_t *)malloc((size_t)k);
    float *s = (float *)malloc((size_t)(k / 32) * sizeof(float));
    if (q == NULL || s == NULL) {
        free(q);
        free(s);
        return;
    }
    if (lfm_act_i8_rot(x, k, 128, q, s) == 0)
        lfm_hvx_gemv_i4(img, n, k, q, s, y);
    free(q);
    free(s);
}

static void ref_forward(const lfm_model_desc *d, const lfm_exec_model *m, void *arena,
                        const lfm_row *row, float *logits)
{
    const int h = T_H;
    const int f = T_F;
    const int dim = T_D;
    const int hq = T_HQ;
    const int hkv = T_HKV;
    const int qd = hq * dim;
    const int kd = hkv * dim;
    float *x = (float *)calloc((size_t)h, sizeof(float));
    float *res = (float *)calloc((size_t)h, sizeof(float));
    float *nrm = (float *)calloc((size_t)h, sizeof(float));
    float *blk = (float *)calloc((size_t)h, sizeof(float));
    float *mrg = (float *)calloc((size_t)h, sizeof(float));
    float *tmp = (float *)calloc((size_t)h, sizeof(float));
    float *q = (float *)calloc((size_t)qd, sizeof(float));
    float *k = (float *)calloc((size_t)kd, sizeof(float));
    float *v = (float *)calloc((size_t)kd, sizeof(float));
    float *gat = (float *)calloc((size_t)hq, sizeof(float));
    float *att = (float *)calloc((size_t)qd, sizeof(float));
    float *mix = (float *)calloc((size_t)3 * h, sizeof(float));
    float *cvg = (float *)calloc((size_t)h, sizeof(float));
    float *cin = (float *)calloc((size_t)h, sizeof(float));
    float *cvout = (float *)calloc((size_t)h, sizeof(float));
    float *fa = (float *)calloc((size_t)f, sizeof(float));
    float *fu = (float *)calloc((size_t)f, sizeof(float));
    float *mid = (float *)calloc((size_t)f, sizeof(float));
    float *scratch =
        (float *)calloc((size_t)(4 * (T_CTX + LFM_EXEC_CHUNK) * dim + 4096), sizeof(float));
    uint32_t *tbl = lfm_exec_table(d, arena, row->seq);
    uint32_t pos = row->pos;
    float *hist = lfm_exec_conv_state(d, arena, row->seq);
    int i;
    int l;
    int hh;

    for (i = 0; i < h; i++)
        x[i] = m->embed[(size_t)row->token * h + i];
    for (l = 0; l < (int)d->n_layers; l++) {
        const lfm_exec_layer *w = &m->layer[l];
        uint32_t kind = m->layer_kind[l];
        memcpy(res, x, (size_t)h * sizeof(float));
        ref_rms(x, w->attn_norm, h, nrm);
        if (kind == LFM_LAYER_CONV) {
            int j;
            ref_proj(w->conv_in, 3 * h, h, nrm, mix);
            for (i = 0; i < h; i++)
                cin[i] = mix[i] * mix[2 * h + i];
            for (i = 0; i < h; i++) {
                float acc = w->conv_w[(size_t)i * LFM_EXEC_CONV_TAPS + 6] * cin[i];
                for (j = 0; j < LFM_EXEC_CONV_HISTORY; j++)
                    acc += w->conv_w[(size_t)i * LFM_EXEC_CONV_TAPS + j] * hist[(size_t)j * h + i];
                cvout[i] = acc;
            }
            memmove(hist, hist + h, (size_t)(LFM_EXEC_CONV_HISTORY - 1) * h * sizeof(float));
            memcpy(hist + (size_t)(LFM_EXEC_CONV_HISTORY - 1) * h, cin, (size_t)h * sizeof(float));
            for (i = 0; i < h; i++)
                cvg[i] = mix[h + i] * cvout[i];
            ref_proj(w->conv_out, h, h, cvg, blk);
        } else {
            int pairs = kind == LFM_LAYER_FULL ? dim / 4 : dim / 2;
            double theta = kind == LFM_LAYER_FULL ? 1e6 : 1e4;
            uint32_t win = kind == LFM_LAYER_SLIDING ? d->sliding_window : 0u;
            ref_proj(w->attn_q, qd, h, nrm, q);
            ref_proj(w->attn_k, kd, h, nrm, k);
            ref_proj(w->attn_v, kd, h, nrm, v);
            ref_proj(w->attn_gate, hq, h, nrm, gat);
            for (hh = 0; hh < hq; hh++) {
                float *qh = q + (size_t)hh * dim;
                ref_head_norm(qh, w->q_norm, dim);
                ref_rope(qh, pos, dim, pairs, theta);
                if (dim == 128)
                    lfm_rot_fwd_f32(qh, dim);
            }
            for (hh = 0; hh < hkv; hh++) {
                float *kh = k + (size_t)hh * dim;
                float *vh = v + (size_t)hh * dim;
                ref_head_norm(kh, w->k_norm, dim);
                ref_rope(kh, pos, dim, pairs, theta);
                if (dim == 128) {
                    lfm_rot_fwd_f32(kh, dim);
                    lfm_rot_fwd_f32(vh, dim);
                }
            }
            for (hh = 0; hh < hkv; hh++) {
                uint32_t slot = pos & 31u;
                uint8_t *stage = lfm_exec_kv_stage(d, arena, row->seq, (uint32_t)l, (uint32_t)hh);
                float *sbuf = (float *)stage;
                if (slot == 0u)
                    memset(sbuf, 0, 2u * LFM_PAGE_TOKENS * (size_t)dim * sizeof(float));
                memcpy(sbuf + (size_t)slot * dim, k + (size_t)hh * dim,
                       (size_t)dim * sizeof(float));
                memcpy(sbuf + (size_t)LFM_PAGE_TOKENS * dim + (size_t)slot * dim,
                       v + (size_t)hh * dim, (size_t)dim * sizeof(float));
                lfm_hvx_kvcache_commit((lfm_kv_format)d->kv_format,
                                       lfm_exec_kv_pages(d, arena, (uint32_t)l) +
                                           (size_t)tbl[pos >> 5] * lfm_page_bytes(d->kv_format,
                                                                                  d->kv_heads,
                                                                                  d->head_dim),
                                       sbuf, sbuf + (size_t)LFM_PAGE_TOKENS * dim, hh, dim);
            }
            lfm_hvx_attn_flash((lfm_kv_format)d->kv_format, tbl,
                               lfm_exec_kv_pages(d, arena, (uint32_t)l), hkv, hq, dim, q, 1, pos,
                               win, att, scratch);
            for (hh = 0; hh < hq; hh++) {
                float *oh = att + (size_t)hh * dim;
                float g = 1.0f / (1.0f + expf(-gat[hh]));
                for (i = 0; i < dim; i++)
                    oh[i] *= g;
                if (dim == 128)
                    lfm_rot_inv_f32(oh, dim);
            }
            ref_proj(w->attn_out, h, qd, att, blk);
        }
        ref_rms(blk, w->post_attn_norm, h, tmp);
        for (i = 0; i < h; i++)
            mrg[i] = res[i] + tmp[i];
        ref_rms(mrg, w->ffn_norm, h, nrm);
        ref_proj(w->ffn_gate, f, h, nrm, fa);
        ref_proj(w->ffn_up, f, h, nrm, fu);
        for (i = 0; i < f; i++)
            mid[i] = (fa[i] / (1.0f + expf(-fa[i]))) * fu[i];
        ref_proj(w->ffn_down, h, f, mid, blk);
        ref_rms(blk, w->post_ffw_norm, h, tmp);
        for (i = 0; i < h; i++)
            x[i] = mrg[i] + tmp[i];
    }
    ref_rms(x, m->out_norm, h, tmp);
    for (i = 0; i < T_V; i++)
        logits[i] = dot_f32(m->lm_head + (size_t)i * h, tmp, h);

    free(x);
    free(res);
    free(nrm);
    free(blk);
    free(mrg);
    free(tmp);
    free(q);
    free(k);
    free(v);
    free(gat);
    free(att);
    free(mix);
    free(cvg);
    free(cin);
    free(cvout);
    free(fa);
    free(fu);
    free(mid);
    free(scratch);
}

static const char *fmt_name(lfm_kv_format fmt)
{
    if (fmt == LFM_KVARN_K4V2)
        return "k4v2";
    if (fmt == LFM_KVARN_K4V4)
        return "k4v4";
    return "f16";
}

static void test_exec(lfm_kv_format fmt)
{
    lfm_model_desc desc;
    lfm_exec_model model;
    size_t arena_bytes;
    uint8_t *arena0;
    uint8_t *arena_j;
    uint8_t *play;
    float logits_a[2 * T_V];
    float logits_b[T_V];
    float logits_c[T_V];
    float logits_d[2 * T_V];
    float logits_e[2 * T_V];
    float ref_logits[T_V];
    lfm_row warm[1];
    lfm_row rows[2];
    uint32_t *tok;
    int rc;

    fill_desc(&desc);
    desc.kv_format = fmt;
    if (build_model(&model) != 0) {
        CHECK(0, "exec model build");
        return;
    }
    rc = lfm_hvx_exec_bind(&model);
    CHECK(rc == LFM_EXEC_OK, "bind rc=%d", rc);
    arena_bytes = lfm_exec_arena_bytes(&desc);
    CHECK(arena_bytes > 0u, "arena bytes");
    printf("exec[%s]: arena=%zu B hidden=%u ffn=%u layers=%u heads=%u/%u dim=%u ctx=%u window=%u\n",
           fmt_name(fmt), arena_bytes, desc.hidden, desc.ffn, desc.n_layers, desc.n_heads,
           desc.kv_heads, desc.head_dim, desc.ctx, desc.sliding_window);
    arena0 = (uint8_t *)malloc(arena_bytes);
    arena_j = (uint8_t *)malloc(arena_bytes);
    play = (uint8_t *)malloc(arena_bytes);
    CHECK(arena0 != NULL && arena_j != NULL && play != NULL, "arena alloc");
    if (arena0 == NULL || arena_j == NULL || play == NULL)
        return;
    memset(arena0, 0, arena_bytes);
    lfm_exec_init_tables(&desc, arena0);

    memset(&warm[0], 0, sizeof warm[0]);
    warm[0].seq = 0;
    warm[0].pos = 0;
    warm[0].ntok = 34;
    warm[0].kv_off = 0;
    warm[0].tbl_off = 0;
    tok = lfm_exec_tokens(&desc, arena0, 0);
    for (rc = 0; rc < 34; rc++)
        tok[rc] = (uint32_t)((rc * 7 + 3) % T_V);
    rc = run_rows(&desc, arena0, warm, 1, 0, NULL);
    CHECK(rc == LFM_EXEC_OK, "warm seq0 rc=%d", rc);

    memset(&warm[0], 0, sizeof warm[0]);
    warm[0].seq = 1;
    warm[0].pos = 0;
    warm[0].ntok = 2;
    warm[0].kv_off = T_KSTAGE;
    warm[0].tbl_off = lfm_max_pages(T_CTX, T_POOL);
    tok = lfm_exec_tokens(&desc, arena0, 0);
    tok[0] = 7;
    tok[1] = 9;
    rc = run_rows(&desc, arena0, warm, 1, 0, NULL);
    CHECK(rc == LFM_EXEC_OK, "warm seq1 rc=%d", rc);

    memset(&rows[0], 0, sizeof rows[0]);
    rows[0].token = 91;
    rows[0].seq = 0;
    rows[0].pos = 34;
    rows[0].ntok = 1;
    rows[0].kv_off = 0;
    rows[0].tbl_off = 0;
    memset(&rows[1], 0, sizeof rows[1]);
    rows[1].seq = 1;
    rows[1].pos = 2;
    rows[1].ntok = 3;
    rows[1].kv_off = T_KSTAGE;
    rows[1].tbl_off = lfm_max_pages(T_CTX, T_POOL);

    memcpy(play, arena0, arena_bytes);
    tok = lfm_exec_tokens(&desc, play, 1);
    tok[0] = 11;
    tok[1] = 23;
    tok[2] = 37;
    rc = run_rows(&desc, play, rows, 2, 0, logits_a);
    CHECK(rc == LFM_EXEC_OK, "joint step rc=%d", rc);
    memcpy(arena_j, play, arena_bytes);

    memcpy(play, arena0, arena_bytes);
    rc = run_rows(&desc, play, &rows[0], 1, 0, logits_b);
    CHECK(rc == LFM_EXEC_OK, "seq0 alone rc=%d", rc);

    memcpy(play, arena0, arena_bytes);
    tok = lfm_exec_tokens(&desc, play, 0);
    tok[0] = 11;
    tok[1] = 23;
    tok[2] = 37;
    rc = run_rows(&desc, play, &rows[1], 1, 0, logits_c);
    CHECK(rc == LFM_EXEC_OK, "seq1 alone rc=%d", rc);

    memcpy(play, arena0, arena_bytes);
    {
        lfm_row perm[2];
        perm[0] = rows[1];
        perm[1] = rows[0];
        tok = lfm_exec_tokens(&desc, play, 0);
        tok[0] = 11;
        tok[1] = 23;
        tok[2] = 37;
        rc = run_rows(&desc, play, perm, 2, 0, logits_d);
        CHECK(rc == LFM_EXEC_OK, "permuted step rc=%d", rc);
    }

    CHECK(memcmp(logits_a, logits_b, sizeof logits_b) == 0,
          "joint seq0 differs from solo (max %.3g)",
          memcmp(logits_a, logits_b, sizeof logits_b) == 0
              ? 0.0
              : (double)cos_f32(logits_a, logits_b, T_V));
    CHECK(memcmp(logits_a + T_V, logits_c, sizeof logits_c) == 0, "joint seq1 differs from solo");
    CHECK(memcmp(logits_d, logits_c, sizeof logits_c) == 0, "permuted seq1 differs");
    CHECK(memcmp(logits_d + T_V, logits_b, sizeof logits_b) == 0, "permuted seq0 differs");
    {
        size_t state_bytes = (size_t)((uint8_t *)lfm_exec_tokens(&desc, arena0, 0) - arena0);
        CHECK(memcmp(play, arena_j, state_bytes) == 0, "permuted arena state differs");
    }

    memcpy(play, arena0, arena_bytes);
    tok = lfm_exec_tokens(&desc, play, 1);
    tok[0] = 11;
    tok[1] = 23;
    tok[2] = 37;
    rc = run_rows(&desc, play, rows, 2, 0, logits_e);
    CHECK(rc == LFM_EXEC_OK, "repeat step rc=%d", rc);
    CHECK(memcmp(logits_e, logits_a, sizeof logits_a) == 0, "repeat logits differ");
    CHECK(memcmp(play, arena_j, arena_bytes) == 0, "repeat arena state differs");

    memcpy(play, arena0, arena_bytes);
    {
        lfm_row dec;
        float lg[T_V];
        float c;
        float md = 0.0f;
        int exact = 1;
        int i;
        memset(&dec, 0, sizeof dec);
        dec.seq = 1;
        dec.ntok = 1;
        dec.kv_off = T_KSTAGE;
        dec.tbl_off = lfm_max_pages(T_CTX, T_POOL);
        dec.token = 11;
        dec.pos = 2;
        rc = run_rows(&desc, play, &dec, 1, 0, lg);
        CHECK(rc == LFM_EXEC_OK, "split decode rc=%d", rc);
        dec.token = 23;
        dec.pos = 3;
        rc = run_rows(&desc, play, &dec, 1, 0, lg);
        CHECK(rc == LFM_EXEC_OK, "split decode rc=%d", rc);
        dec.token = 37;
        dec.pos = 4;
        rc = run_rows(&desc, play, &dec, 1, 0, lg);
        CHECK(rc == LFM_EXEC_OK, "split decode rc=%d", rc);
        c = cos_f32(lg, logits_c, T_V);
        for (i = 0; i < T_V; i++) {
            float diff = fabsf(lg[i] - logits_c[i]);
            if (diff > md)
                md = diff;
            if (diff != 0.0f)
                exact = 0;
        }
        printf("exec[%s]: prefill-chunk(3) vs tokenwise-decode cos=%.7f max_abs=%.3g exact=%d\n",
               fmt_name(fmt), c, md, exact);
        CHECK(c >= 0.9999f, "prefill vs tokenwise cos %.7f", c);
    }

    memcpy(play, arena0, arena_bytes);
    ref_forward(&desc, &model, play, &rows[0], ref_logits);
    {
        float c = cos_f32(ref_logits, logits_b, T_V);
        float md = 0.0f;
        int i;
        int exact = 1;
        for (i = 0; i < T_V; i++) {
            float diff = fabsf(ref_logits[i] - logits_b[i]);
            if (diff > md)
                md = diff;
            if (diff != 0.0f)
                exact = 0;
        }
        printf("exec[%s]: decode-vs-gemv cos=%.7f max_abs=%.3g exact=%d\n", fmt_name(fmt), c, md,
               exact);
        CHECK(c >= 0.9999f, "decode vs gemv cos %.7f", c);
        CHECK(md <= 1e-5f, "decode vs gemv max abs %.3g", md);
    }

    printf("exec[%s]: joint-vs-solo bitwise, permute-invariant, repeat-deterministic "
           "(%ux%u tokens)\n",
           fmt_name(fmt), rows[0].ntok, rows[1].ntok);

    free(arena0);
    free(arena_j);
    free(play);
    lfm_hvx_exec_unbind();
    lfm_hvx_exec_release();
    free_model();
}

int main(void)
{
    test_ffn(256, 256);
    test_ffn(160, 160);
    test_exec(LFM_KV_F16);
    test_exec(LFM_KVARN_K4V2);
    test_exec(LFM_KVARN_K4V4);
    if (failures != 0) {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
    printf("hvx exec/ffn: all checks passed\n");
    return 0;
}
