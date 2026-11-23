#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/q4.h"
#include "../common/quant.h"
#include "../common/rot.h"

#include "exec.h"
#include "hvx_ops.h"

#define LFM_EXEC_EPS 1e-5f
#define LFM_EXEC_ROT_BLOCK 128

typedef struct {
    float *buf;
    size_t cap;
    int8_t *act;
    size_t act_cap;
    float *scale;
    size_t scale_cap;
    float *prob;
    size_t prob_cap;
    float *attn;
    size_t attn_cap;
} exec_ws;

static exec_ws g_ws;
static lfm_exec_model g_model;
static int g_bound;

static size_t exec_align(size_t v)
{
    size_t a = (size_t)LFM_ARENA_ALIGN;
    return (v + a - 1u) & ~(a - 1u);
}

static size_t exec_pages(const lfm_model_desc *d)
{
    return (size_t)lfm_max_pages(d->ctx, d->kv_pool);
}

static size_t exec_page_bytes(const lfm_model_desc *d)
{
    return (size_t)lfm_page_bytes(d->kv_format, d->kv_heads, d->head_dim);
}

static size_t exec_kv_bytes(const lfm_model_desc *d)
{
    return (size_t)d->kv_layers * exec_pages(d) * exec_page_bytes(d);
}

static size_t exec_tbl_off(const lfm_model_desc *d)
{
    return exec_align(exec_kv_bytes(d));
}

static size_t exec_tbl_bytes(const lfm_model_desc *d)
{
    return (size_t)d->max_seqs * exec_pages(d) * (size_t)LFM_PT_ENTRY_BYTES;
}

static size_t exec_conv_slots(const lfm_model_desc *d)
{
    return d->conv_cache < LFM_EXEC_CONV_HISTORY ? (size_t)LFM_EXEC_CONV_HISTORY
                                                 : (size_t)d->conv_cache;
}

static size_t exec_conv_off(const lfm_model_desc *d)
{
    return exec_align(exec_tbl_off(d) + exec_tbl_bytes(d));
}

static size_t exec_conv_bytes(const lfm_model_desc *d)
{
    return (size_t)d->max_seqs * exec_conv_slots(d) * (size_t)d->hidden * sizeof(float);
}

static size_t exec_stage_off(const lfm_model_desc *d)
{
    return exec_align(exec_conv_off(d) + exec_conv_bytes(d));
}

static size_t exec_stage_bytes(const lfm_model_desc *d)
{
    return (size_t)d->max_seqs * (size_t)d->kv_layers * (size_t)d->kv_heads * 2u *
           (size_t)LFM_PAGE_TOKENS * (size_t)d->head_dim * sizeof(float);
}

static size_t exec_tok_off(const lfm_model_desc *d)
{
    return exec_align(exec_stage_off(d) + exec_stage_bytes(d));
}

static size_t exec_tok_bytes(const lfm_model_desc *d)
{
    return (size_t)d->max_seqs * (size_t)d->ctx * sizeof(uint32_t);
}

size_t lfm_exec_arena_bytes(const lfm_model_desc *desc)
{
    if (desc == NULL)
        return 0;
    return exec_tok_off(desc) + exec_align(exec_tok_bytes(desc));
}

void lfm_exec_init_tables(const lfm_model_desc *desc, void *arena)
{
    size_t i;
    size_t n;
    uint32_t *tbl;
    if (desc == NULL || arena == NULL)
        return;
    tbl = (uint32_t *)((uint8_t *)arena + exec_tbl_off(desc));
    n = (size_t)desc->max_seqs * exec_pages(desc);
    for (i = 0; i < n; i++)
        tbl[i] = LFM_EXEC_PAGE_NONE;
}

uint8_t *lfm_exec_kv_pages(const lfm_model_desc *desc, void *arena, uint32_t layer)
{
    if (desc == NULL || arena == NULL || layer >= desc->kv_layers)
        return NULL;
    return (uint8_t *)arena + (size_t)layer * exec_pages(desc) * exec_page_bytes(desc);
}

uint32_t *lfm_exec_table(const lfm_model_desc *desc, void *arena, uint32_t seq)
{
    uint32_t *base;
    if (desc == NULL || arena == NULL || seq >= desc->max_seqs)
        return NULL;
    base = (uint32_t *)((uint8_t *)arena + exec_tbl_off(desc));
    return base + (size_t)seq * exec_pages(desc);
}

uint32_t *lfm_exec_tokens(const lfm_model_desc *desc, void *arena, uint32_t row)
{
    uint32_t *base;
    if (desc == NULL || arena == NULL || row >= desc->max_seqs)
        return NULL;
    base = (uint32_t *)((uint8_t *)arena + exec_tok_off(desc));
    return base + (size_t)row * desc->ctx;
}

float *lfm_exec_conv_state(const lfm_model_desc *desc, void *arena, uint32_t seq)
{
    float *base;
    if (desc == NULL || arena == NULL || seq >= desc->max_seqs)
        return NULL;
    base = (float *)((uint8_t *)arena + exec_conv_off(desc));
    return base + (size_t)seq * exec_conv_slots(desc) * desc->hidden;
}

uint8_t *lfm_exec_kv_stage(const lfm_model_desc *desc, void *arena, uint32_t seq, uint32_t layer,
                           uint32_t head)
{
    uint8_t *base;
    size_t stride;
    if (desc == NULL || arena == NULL || seq >= desc->max_seqs || layer >= desc->kv_layers ||
        head >= desc->kv_heads)
        return NULL;
    stride = 2u * (size_t)LFM_PAGE_TOKENS * desc->head_dim * sizeof(float);
    base = (uint8_t *)arena + exec_stage_off(desc);
    return base + ((size_t)seq * desc->kv_layers + layer) * desc->kv_heads * stride +
           (size_t)head * stride;
}

int lfm_hvx_exec_bind(const lfm_exec_model *model)
{
    if (model == NULL || model->n_layers > LFM_EXEC_MAX_LAYERS)
        return LFM_EXEC_ERR_MODEL;
    g_model = *model;
    g_bound = 1;
    return LFM_EXEC_OK;
}

void lfm_hvx_exec_unbind(void)
{
    g_bound = 0;
    memset(&g_model, 0, sizeof g_model);
}

void lfm_hvx_exec_release(void)
{
    free(g_ws.buf);
    free(g_ws.act);
    free(g_ws.scale);
    free(g_ws.prob);
    free(g_ws.attn);
    memset(&g_ws, 0, sizeof g_ws);
}

static int desc_valid(const lfm_model_desc *d, const lfm_exec_model *m)
{
    if (d == NULL || m == NULL)
        return LFM_EXEC_ERR_ARGS;
    if (d->hidden == 0 || d->ffn == 0 || d->n_layers == 0 || d->n_layers > LFM_EXEC_MAX_LAYERS)
        return LFM_EXEC_ERR_ARGS;
    if ((d->hidden % LFM_QGROUP) != 0 || (d->ffn % LFM_QGROUP) != 0)
        return LFM_EXEC_ERR_ARGS;
    if (d->vocab == 0 || d->ctx == 0 || d->max_seqs == 0 || d->kv_pool == 0)
        return LFM_EXEC_ERR_ARGS;
    if (d->kv_layers < d->n_layers)
        return LFM_EXEC_ERR_ARGS;
    if (d->kv_stage < LFM_PAGE_TOKENS || (d->kv_stage % LFM_PAGE_TOKENS) != 0)
        return LFM_EXEC_ERR_ARGS;
    if (d->n_heads == 0 || d->kv_heads == 0 || d->head_dim == 0)
        return LFM_EXEC_ERR_ARGS;
    if (d->n_heads % d->kv_heads)
        return LFM_EXEC_ERR_ARGS;
    if (d->kv_dim != d->kv_heads * d->head_dim)
        return LFM_EXEC_ERR_ARGS;
    if (d->kv_format != LFM_KV_F16 && d->kv_format != LFM_KVARN_K4V2 &&
        d->kv_format != LFM_KVARN_K4V4)
        return LFM_EXEC_ERR_ARGS;
    if (d->kv_format != LFM_KV_F16 && d->head_dim != LFM_KVARN_HEAD_DIM)
        return LFM_EXEC_ERR_ARGS;
    if (m->n_layers != d->n_layers)
        return LFM_EXEC_ERR_MODEL;
    if (m->embed == NULL || m->out_norm == NULL || m->lm_head == NULL)
        return LFM_EXEC_ERR_MODEL;
    return LFM_EXEC_OK;
}

static int layer_valid(const lfm_exec_model *m, uint32_t layer)
{
    const lfm_exec_layer *w = &m->layer[layer];
    uint32_t kind = m->layer_kind[layer];
    if (w->attn_norm == NULL || w->post_attn_norm == NULL || w->ffn_norm == NULL ||
        w->post_ffw_norm == NULL)
        return LFM_EXEC_ERR_MODEL;
    if (w->ffn_gate == NULL || w->ffn_up == NULL || w->ffn_down == NULL)
        return LFM_EXEC_ERR_MODEL;
    if (kind == LFM_LAYER_CONV) {
        if (w->conv_in == NULL || w->conv_w == NULL || w->conv_out == NULL)
            return LFM_EXEC_ERR_MODEL;
    } else if (kind == LFM_LAYER_FULL || kind == LFM_LAYER_SLIDING) {
        if (w->attn_q == NULL || w->attn_k == NULL || w->attn_v == NULL ||
            w->attn_gate == NULL || w->attn_out == NULL)
            return LFM_EXEC_ERR_MODEL;
    } else {
        return LFM_EXEC_ERR_MODEL;
    }
    return LFM_EXEC_OK;
}

static int ws_reserve(const lfm_model_desc *d, uint32_t tokens)
{
    size_t h = d->hidden;
    size_t f = d->ffn;
    size_t qd = (size_t)d->n_heads * d->head_dim;
    size_t kd = (size_t)d->kv_heads * d->head_dim;
    size_t kmax = h > f ? h : f;
    size_t floats = (size_t)tokens * (11u * h + 2u * qd + 2u * kd + d->n_heads + 3u * f);
    size_t acts = (size_t)tokens * kmax;
    size_t scales = (size_t)tokens * (kmax / LFM_QGROUP);
    size_t probs = (size_t)d->ctx + 1u;
    size_t attn = 2u * ((size_t)d->ctx + LFM_EXEC_CHUNK) * d->head_dim +
                  8u * (size_t)d->n_heads * d->head_dim + 4u * d->ctx + 256u;

    if (floats > g_ws.cap) {
        float *p = (float *)realloc(g_ws.buf, floats * sizeof(float));
        if (p == NULL)
            return LFM_EXEC_ERR_MEM;
        g_ws.buf = p;
        g_ws.cap = floats;
    }
    if (acts > g_ws.act_cap) {
        int8_t *p = (int8_t *)realloc(g_ws.act, acts);
        if (p == NULL)
            return LFM_EXEC_ERR_MEM;
        g_ws.act = p;
        g_ws.act_cap = acts;
    }
    if (scales > g_ws.scale_cap) {
        float *p = (float *)realloc(g_ws.scale, scales * sizeof(float));
        if (p == NULL)
            return LFM_EXEC_ERR_MEM;
        g_ws.scale = p;
        g_ws.scale_cap = scales;
    }
    if (probs > g_ws.prob_cap) {
        float *p = (float *)realloc(g_ws.prob, probs * sizeof(float));
        if (p == NULL)
            return LFM_EXEC_ERR_MEM;
        g_ws.prob = p;
        g_ws.prob_cap = probs;
    }
    if (attn > g_ws.attn_cap) {
        float *p = (float *)realloc(g_ws.attn, attn * sizeof(float));
        if (p == NULL)
            return LFM_EXEC_ERR_MEM;
        g_ws.attn = p;
        g_ws.attn_cap = attn;
    }
    return LFM_EXEC_OK;
}

static void rms_norm_f32(const float *x, const float *w, int n, float *y)
{
    float acc = 0.0f;
    float inv;
    int i;
    for (i = 0; i < n; i++)
        acc += x[i] * x[i];
    inv = 1.0f / sqrtf(acc / (float)n + LFM_EXEC_EPS);
    if (w == NULL) {
        for (i = 0; i < n; i++)
            y[i] = x[i] * inv;
    } else {
        for (i = 0; i < n; i++)
            y[i] = x[i] * inv * w[i];
    }
}

static void rope_apply(float *v, uint32_t pos, int dim, int pairs, double theta)
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

static void head_rot_fwd(float *v, int dim)
{
    if (dim == LFM_EXEC_ROT_BLOCK)
        lfm_rot_fwd_f32(v, dim);
}

static void head_rot_inv(float *v, int dim)
{
    if (dim == LFM_EXEC_ROT_BLOCK)
        lfm_rot_inv_f32(v, dim);
}

static int project_i4(const uint8_t *img, int n, int k, const float *x, uint32_t m, int8_t *aq,
                      float *as, float *y)
{
    int groups = k / LFM_QGROUP;
    uint32_t r;
    for (r = 0; r < m; r++) {
        if (lfm_act_i8_rot(x + (size_t)r * (size_t)k, k, LFM_EXEC_ROT_BLOCK,
                           aq + (size_t)r * (size_t)k, as + (size_t)r * (size_t)groups) != 0)
            return LFM_EXEC_ERR_ARGS;
    }
    if (m == 1)
        lfm_hvx_gemv_i4(img, n, k, aq, as, y);
    else
        lfm_hvx_gemm_i4(img, n, k, aq, m, as, y);
    return LFM_EXEC_OK;
}

typedef struct {
    float *x;
    float *res;
    float *nrm;
    float *blk;
    float *mrg;
    float *tmp;
    float *q;
    float *k;
    float *v;
    float *gat;
    float *att;
    float *mix;
    float *fa;
    float *fu;
    float *mid;
    float *cin;
    float *cvout;
} exec_ptrs;

static void ws_split(const lfm_model_desc *d, exec_ptrs *p)
{
    size_t h = d->hidden;
    size_t f = d->ffn;
    size_t qd = (size_t)d->n_heads * d->head_dim;
    size_t kd = (size_t)d->kv_heads * d->head_dim;
    size_t t = LFM_EXEC_CHUNK;
    float *cur = g_ws.buf;

    p->x = cur;
    cur += t * h;
    p->res = cur;
    cur += t * h;
    p->nrm = cur;
    cur += t * h;
    p->blk = cur;
    cur += t * h;
    p->mrg = cur;
    cur += t * h;
    p->tmp = cur;
    cur += t * h;
    p->q = cur;
    cur += t * qd;
    p->k = cur;
    cur += t * kd;
    p->v = cur;
    cur += t * kd;
    p->gat = cur;
    cur += t * d->n_heads;
    p->att = cur;
    cur += t * qd;
    p->mix = cur;
    cur += t * 3u * h;
    p->fa = cur;
    cur += t * f;
    p->fu = cur;
    cur += t * f;
    p->mid = cur;
    cur += t * f;
    p->cin = cur;
    cur += h;
    p->cvout = cur;
}

static int ffn_apply(const lfm_model_desc *d, const lfm_exec_layer *lw, uint32_t m, exec_ptrs *p,
                     float *y)
{
    const int h = (int)d->hidden;
    const int f = (int)d->ffn;
    int rc;
    uint32_t t;
    int r;

    if (m == 1) {
        if (lfm_act_i8_rot(p->nrm, h, LFM_EXEC_ROT_BLOCK, g_ws.act, g_ws.scale) != 0)
            return LFM_EXEC_ERR_ARGS;
        lfm_hvx_ffn_i4(lw->ffn_gate, f, h, lw->ffn_up, lw->ffn_down, g_ws.act, g_ws.scale, y);
        return LFM_EXEC_OK;
    }
    rc = project_i4(lw->ffn_gate, f, h, p->nrm, m, g_ws.act, g_ws.scale, p->fa);
    if (rc != LFM_EXEC_OK)
        return rc;
    rc = project_i4(lw->ffn_up, f, h, p->nrm, m, g_ws.act, g_ws.scale, p->fu);
    if (rc != LFM_EXEC_OK)
        return rc;
    for (t = 0; t < m; t++) {
        for (r = 0; r < f; r++) {
            float g = p->fa[(size_t)t * f + r];
            float u = p->fu[(size_t)t * f + r];
            p->mid[(size_t)t * f + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
    return project_i4(lw->ffn_down, h, f, p->mid, m, g_ws.act, g_ws.scale, y);
}

static int conv_apply(const lfm_model_desc *d, const lfm_exec_layer *lw, uint32_t seq,
                      void *arena, uint32_t m, exec_ptrs *p, float *y)
{
    const int h = (int)d->hidden;
    float *hist = lfm_exec_conv_state(d, arena, seq);
    int rc;
    int i;
    int j;
    uint32_t t;

    if (hist == NULL)
        return LFM_EXEC_ERR_ARGS;
    rc = project_i4(lw->conv_in, 3 * h, h, p->nrm, m, g_ws.act, g_ws.scale, p->mix);
    if (rc != LFM_EXEC_OK)
        return rc;
    for (t = 0; t < m; t++) {
        const float *mix = p->mix + (size_t)t * 3u * h;
        float *cvg = p->mid + (size_t)t * h;
        for (i = 0; i < h; i++)
            p->cin[i] = mix[i] * mix[2 * h + i];
        for (i = 0; i < h; i++) {
            float acc = lw->conv_w[(size_t)i * LFM_EXEC_CONV_TAPS + 6] * p->cin[i];
            for (j = 0; j < LFM_EXEC_CONV_HISTORY; j++)
                acc += lw->conv_w[(size_t)i * LFM_EXEC_CONV_TAPS + j] * hist[(size_t)j * h + i];
            p->cvout[i] = acc;
        }
        memmove(hist, hist + h, (size_t)(LFM_EXEC_CONV_HISTORY - 1) * h * sizeof(float));
        memcpy(hist + (size_t)(LFM_EXEC_CONV_HISTORY - 1) * h, p->cin, (size_t)h * sizeof(float));
        for (i = 0; i < h; i++)
            cvg[i] = mix[h + i] * p->cvout[i];
    }
    return project_i4(lw->conv_out, h, h, p->mid, m, g_ws.act, g_ws.scale, y);
}

static void apply_head_norm(float *v, const float *w, int dim)
{
    float acc = 0.0f;
    float inv;
    int i;
    for (i = 0; i < dim; i++)
        acc += v[i] * v[i];
    inv = 1.0f / sqrtf(acc / (float)dim + LFM_EXEC_EPS);
    for (i = 0; i < dim; i++)
        v[i] = v[i] * inv * w[i];
}

static int attn_apply(const lfm_model_desc *d, const lfm_exec_layer *lw, uint32_t seq,
                      uint32_t layer, uint32_t kind, const uint32_t *tbl, uint8_t *kv_pages,
                      void *arena, uint32_t pos, uint32_t m, uint32_t win, exec_ptrs *p,
                      float *y)
{
    const int h = (int)d->hidden;
    const int dim = (int)d->head_dim;
    const int hq = (int)d->n_heads;
    const int hkv = (int)d->kv_heads;
    const int qd = hq * dim;
    const int kd = hkv * dim;
    const size_t page_bytes = exec_page_bytes(d);
    const int pairs = kind == LFM_LAYER_FULL ? dim / 4 : dim / 2;
    const double theta = kind == LFM_LAYER_FULL ? 1e6 : 1e4;
    int rc;
    int hh;
    int i;
    uint32_t t;

    rc = project_i4(lw->attn_q, qd, h, p->nrm, m, g_ws.act, g_ws.scale, p->q);
    if (rc != LFM_EXEC_OK)
        return rc;
    rc = project_i4(lw->attn_k, kd, h, p->nrm, m, g_ws.act, g_ws.scale, p->k);
    if (rc != LFM_EXEC_OK)
        return rc;
    rc = project_i4(lw->attn_v, kd, h, p->nrm, m, g_ws.act, g_ws.scale, p->v);
    if (rc != LFM_EXEC_OK)
        return rc;
    rc = project_i4(lw->attn_gate, hq, h, p->nrm, m, g_ws.act, g_ws.scale, p->gat);
    if (rc != LFM_EXEC_OK)
        return rc;

    for (t = 0; t < m; t++) {
        uint32_t qpos = pos + t;
        for (hh = 0; hh < hq; hh++) {
            float *qh = p->q + (size_t)t * qd + (size_t)hh * dim;
            if (lw->q_norm != NULL)
                apply_head_norm(qh, lw->q_norm, dim);
            rope_apply(qh, qpos, dim, pairs, theta);
            head_rot_fwd(qh, dim);
        }
        for (hh = 0; hh < hkv; hh++) {
            float *kh = p->k + (size_t)t * kd + (size_t)hh * dim;
            float *vh = p->v + (size_t)t * kd + (size_t)hh * dim;
            if (lw->k_norm != NULL)
                apply_head_norm(kh, lw->k_norm, dim);
            rope_apply(kh, qpos, dim, pairs, theta);
            head_rot_fwd(kh, dim);
            head_rot_fwd(vh, dim);
        }
        for (hh = 0; hh < hkv; hh++) {
            uint32_t slot = qpos & 31u;
            uint32_t lp = qpos >> 5;
            uint32_t phys = tbl[lp];
            uint8_t *page = kv_pages + (size_t)phys * page_bytes;
            uint8_t *stage = lfm_exec_kv_stage(d, arena, seq, layer, (uint32_t)hh);
            float *sbuf;
            if (stage == NULL)
                return LFM_EXEC_ERR_ARGS;
            sbuf = (float *)stage;
            if (slot == 0u)
                memset(sbuf, 0, 2u * LFM_PAGE_TOKENS * (size_t)dim * sizeof(float));
            memcpy(sbuf + (size_t)slot * dim, p->k + (size_t)t * kd + (size_t)hh * dim,
                   (size_t)dim * sizeof(float));
            memcpy(sbuf + (size_t)LFM_PAGE_TOKENS * dim + (size_t)slot * dim,
                   p->v + (size_t)t * kd + (size_t)hh * dim, (size_t)dim * sizeof(float));
            lfm_hvx_kvcache_commit((lfm_kv_format)d->kv_format, page, sbuf,
                                   sbuf + (size_t)LFM_PAGE_TOKENS * dim, hh, dim);
        }
    }

    lfm_hvx_attn_flash((lfm_kv_format)d->kv_format, tbl, kv_pages, hkv, hq, dim, p->q, m, pos,
                       win, p->att, g_ws.attn);

    for (t = 0; t < m; t++) {
        for (hh = 0; hh < hq; hh++) {
            float *oh = p->att + (size_t)t * qd + (size_t)hh * dim;
            float g = 1.0f / (1.0f + expf(-p->gat[(size_t)t * hq + hh]));
            for (i = 0; i < dim; i++)
                oh[i] *= g;
            head_rot_inv(oh, dim);
        }
    }
    return project_i4(lw->attn_out, h, qd, p->att, m, g_ws.act, g_ws.scale, y);
}

static int page_ready(const lfm_model_desc *d, const lfm_row *row, uint32_t *tbl, uint32_t pos,
                      uint32_t *out)
{
    size_t pages = exec_pages(d);
    size_t span = ((size_t)d->ctx + LFM_PAGE_TOKENS - 1u) / LFM_PAGE_TOKENS;
    size_t base = (size_t)row->kv_off / LFM_PAGE_TOKENS;
    uint32_t lp;
    uint32_t phys;

    if (pos >= d->ctx)
        return LFM_EXEC_ERR_ARGS;
    lp = pos >> 5;
    if ((size_t)lp >= pages || (size_t)lp >= span)
        return LFM_EXEC_ERR_ARGS;
    if (base >= pages)
        return LFM_EXEC_ERR_STATE;
    phys = tbl[lp];
    if (phys == LFM_EXEC_PAGE_NONE) {
        if (base % span != 0u || base + span > pages)
            return LFM_EXEC_ERR_STATE;
        phys = (uint32_t)(base + (size_t)lp);
        tbl[lp] = phys;
    } else if ((size_t)phys >= pages) {
        return LFM_EXEC_ERR_STATE;
    }
    *out = phys;
    return LFM_EXEC_OK;
}

static int exec_logits(const lfm_model_desc *d, const float *x, float *logits)
{
    const int h = (int)d->hidden;
    int id;
    int i;
    if (logits == NULL)
        return LFM_EXEC_OK;
    for (id = 0; id < (int)d->vocab; id++) {
        const float *row = g_model.lm_head + (size_t)id * h;
        float acc = 0.0f;
        for (i = 0; i < h; i++)
            acc += row[i] * x[i];
        logits[id] = acc;
    }
    return LFM_EXEC_OK;
}

static void seq_reset(const lfm_model_desc *d, void *arena, uint32_t seq)
{
    float *hist = lfm_exec_conv_state(d, arena, seq);
    uint32_t l;
    uint32_t hh;
    if (hist != NULL)
        memset(hist, 0, exec_conv_slots(d) * (size_t)d->hidden * sizeof(float));
    for (l = 0; l < d->n_layers; l++) {
        for (hh = 0; hh < d->kv_heads; hh++) {
            uint8_t *st = lfm_exec_kv_stage(d, arena, seq, l, hh);
            if (st != NULL)
                memset(st, 0, 2u * LFM_PAGE_TOKENS * (size_t)d->head_dim * sizeof(float));
        }
    }
}

static int exec_row(const lfm_model_desc *d, void *arena, uint32_t row_index, const lfm_row *row,
                    uint32_t window, float *logits)
{
    const int h = (int)d->hidden;
    uint32_t ntok = row->ntok;
    uint32_t *tbl;
    uint8_t *kv_pages;
    exec_ptrs p;
    uint32_t off;
    uint32_t c;
    uint32_t t;
    uint32_t l;

    if (ntok == 0u || ntok > d->ctx)
        return LFM_EXEC_ERR_ARGS;
    if (row->seq >= d->max_seqs)
        return LFM_EXEC_ERR_ARGS;
    if ((uint64_t)row->pos + (uint64_t)ntok > (uint64_t)d->ctx)
        return LFM_EXEC_ERR_ARGS;
    if (ntok > 1u && row_index >= d->max_seqs)
        return LFM_EXEC_ERR_ARGS;
    tbl = lfm_exec_table(d, arena, row->seq);
    if (tbl == NULL)
        return LFM_EXEC_ERR_ARGS;
    kv_pages = lfm_exec_kv_pages(d, arena, 0);
    if (kv_pages == NULL)
        return LFM_EXEC_ERR_ARGS;
    if (row->pos == 0u)
        seq_reset(d, arena, row->seq);
    ws_split(d, &p);

    for (off = 0; off < ntok; off += c) {
        uint32_t pos0 = row->pos + off;
        const uint32_t *src = lfm_exec_tokens(d, arena, row_index);
        c = ntok - off;
        if (c > LFM_EXEC_CHUNK)
            c = LFM_EXEC_CHUNK;
        if (src == NULL)
            return LFM_EXEC_ERR_ARGS;
        for (t = 0; t < c; t++) {
            uint32_t phys;
            uint32_t id;
            int i;
            int rc = page_ready(d, row, tbl, pos0 + t, &phys);
            (void)phys;
            if (rc != LFM_EXEC_OK)
                return rc;
            id = ntok == 1u ? row->token : src[off + t];
            if (id >= d->vocab)
                return LFM_EXEC_ERR_ARGS;
            for (i = 0; i < h; i++)
                p.x[(size_t)t * h + i] = g_model.embed[(size_t)id * h + i];
        }

        for (l = 0; l < d->n_layers; l++) {
            const lfm_exec_layer *w = &g_model.layer[l];
            uint32_t kind = g_model.layer_kind[l];
            int rc;
            if (layer_valid(&g_model, l) != LFM_EXEC_OK)
                return LFM_EXEC_ERR_MODEL;
            for (t = 0; t < c; t++) {
                memcpy(p.res + (size_t)t * h, p.x + (size_t)t * h, (size_t)h * sizeof(float));
                rms_norm_f32(p.x + (size_t)t * h, w->attn_norm, h, p.nrm + (size_t)t * h);
            }
            if (kind == LFM_LAYER_CONV) {
                rc = conv_apply(d, w, row->seq, arena, c, &p, p.blk);
            } else {
                uint32_t win = window;
                if (win == 0u && kind == LFM_LAYER_SLIDING)
                    win = d->sliding_window;
                kv_pages = lfm_exec_kv_pages(d, arena, l);
                rc = attn_apply(d, w, row->seq, l, kind, tbl, kv_pages, arena, pos0, c, win, &p,
                                p.blk);
            }
            if (rc != LFM_EXEC_OK)
                return rc;
            for (t = 0; t < c; t++) {
                int i;
                rms_norm_f32(p.blk + (size_t)t * h, w->post_attn_norm, h, p.tmp + (size_t)t * h);
                for (i = 0; i < h; i++)
                    p.mrg[(size_t)t * h + i] = p.res[(size_t)t * h + i] + p.tmp[(size_t)t * h + i];
                rms_norm_f32(p.mrg + (size_t)t * h, w->ffn_norm, h, p.nrm + (size_t)t * h);
            }
            rc = ffn_apply(d, w, c, &p, p.blk);
            if (rc != LFM_EXEC_OK)
                return rc;
            for (t = 0; t < c; t++) {
                int i;
                rms_norm_f32(p.blk + (size_t)t * h, w->post_ffw_norm, h, p.tmp + (size_t)t * h);
                for (i = 0; i < h; i++)
                    p.x[(size_t)t * h + i] = p.mrg[(size_t)t * h + i] + p.tmp[(size_t)t * h + i];
            }
        }

        if (off + c == ntok && logits != NULL) {
            int i;
            rms_norm_f32(p.x + (size_t)(c - 1) * h, g_model.out_norm, h, p.tmp);
            (void)i;
            if (exec_logits(d, p.tmp, logits) != LFM_EXEC_OK)
                return LFM_EXEC_ERR_MODEL;
        }
    }
    return LFM_EXEC_OK;
}

int lfm_hvx_exec(const lfm_model_desc *desc, const lfm_step *step, const lfm_row *rows,
                 void *arena, float *logits)
{
    uint32_t i;
    int rc;

    if (desc == NULL || step == NULL || rows == NULL || arena == NULL)
        return LFM_EXEC_ERR_ARGS;
    if (!g_bound)
        return LFM_EXEC_ERR_MODEL;
    rc = desc_valid(desc, &g_model);
    if (rc != LFM_EXEC_OK)
        return rc;
    if (step->n_rows > desc->max_seqs)
        return LFM_EXEC_ERR_ARGS;
    if (step->max_rows != 0u && step->n_rows > step->max_rows)
        return LFM_EXEC_ERR_ARGS;
    if (lfm_exec_arena_bytes(desc) == 0u)
        return LFM_EXEC_ERR_ARGS;
    rc = ws_reserve(desc, LFM_EXEC_CHUNK);
    if (rc != LFM_EXEC_OK)
        return rc;
    for (i = 0; i < step->n_rows; i++) {
        float *row_logits = logits != NULL ? logits + (size_t)i * desc->vocab : NULL;
        rc = exec_row(desc, arena, i, &rows[i], step->window, row_logits);
        if (rc != LFM_EXEC_OK)
            return rc;
    }
    return LFM_EXEC_OK;
}
