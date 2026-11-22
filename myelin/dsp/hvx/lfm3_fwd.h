#ifndef LFM3_FWD_H
#define LFM3_FWD_H
/*
 * Minimal f32 forward pass of LFM3-1.7B, used only to obtain *real* FFN inputs
 * for the deeper layers of the DSP studies (the layer-0 path alone cannot
 * reach them).  Mirrors myelin/crates/myelin-core/src/session.rs and
 * backend/cpu.rs operation for operation:
 *
 *   x = embed(tok)
 *   per layer:  residual = x
 *               n      = rms_norm(x, attn_norm)
 *               block  = shortconv(n) | attention(n)          (layer type)
 *               merged = residual + rms_norm(block, post_attention_norm)
 *               f      = rms_norm(merged, ffn_norm)           <- captured
 *               merged = merged + rms_norm(ffn(f), post_ffw_norm)
 *               x      = merged
 *   output_norm(x) -> tied lm_head (top-1 only, sanity check)
 *
 * Attention: q/k/v/o + per-head gate, q/k RMSNorm per head, partial rope on
 * full-attention layers (rotated_dim = head_dim/2), full rope on sliding ones,
 * GQA 16/4, window 256 for sliding layers.  Conv: 7-tap depthwise with
 * 6-slot history.  Weights are dequantised from the GGUF layer by layer.
 */
#include "gguf_f32.h"
#include "ffn_ref.h"

#define LFM_MAX_LAYERS 32

typedef struct {
    int H, FFN, NL, HQ, HKV, D, WIN, VOCAB;
    unsigned char kind[LFM_MAX_LAYERS];   /* 0 conv, 1 full attn, 2 sliding */
} lfm_t;

/* LFM3 layer pattern: 0,1 conv; even L: L%4==2 full, L%4==0 sliding; odd conv */
static void lfm_init(lfm_t *s, int nl, int hidden, int ffn, int heads, int kv_heads,
                     int head_dim, int window, int vocab)
{
    int L;
    memset(s, 0, sizeof(*s));
    s->NL = nl; s->H = hidden; s->FFN = ffn; s->HQ = heads; s->HKV = kv_heads;
    s->D = head_dim; s->WIN = window; s->VOCAB = vocab;
    for (L = 0; L < nl; L++) {
        if (L < 2 || (L & 1)) s->kind[L] = 0;
        else s->kind[L] = (L % 4 == 2) ? 1 : 2;
    }
}

/* per-layer weights, loaded on demand */
typedef struct {
    float *an, *pan, *fnn, *pfn;
    float *fg, *fu, *fd;
    float *cin, *cw, *cout;
    float *wq, *wk, *wv, *wg, *wo, *qn, *kn;
} lw_t;

static void lw_free(lw_t *w)
{
    free(w->an); free(w->pan); free(w->fnn); free(w->pfn);
    free(w->fg); free(w->fu); free(w->fd);
    free(w->cin); free(w->cw); free(w->cout);
    free(w->wq); free(w->wk); free(w->wv); free(w->wg); free(w->wo);
    free(w->qn); free(w->kn);
    memset(w, 0, sizeof(*w));
}

static int lw_load(lw_t *w, const lfm_t *s, int L)
{
    char nm[64];
    const tinfo_t *t;
    memset(w, 0, sizeof(*w));
    snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", L);  w->an  = gguf_vec_alloc(nm, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", L);
    w->pan = gguf_vec_alloc(nm, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", L);   w->fnn = gguf_vec_alloc(nm, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.post_ffw_norm.weight", L);
    w->pfn = gguf_vec_alloc(nm, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", L);
    w->fg = gguf_matrix(gguf_find(nm), s->FFN, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", L);
    w->fu = gguf_matrix(gguf_find(nm), s->FFN, s->H);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", L);
    w->fd = gguf_matrix(gguf_find(nm), s->H, s->FFN);
    if (s->kind[L] == 0) {
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.in_proj.weight", L);
        w->cin = gguf_matrix(gguf_find(nm), 3 * s->H, s->H);
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.conv.weight", L);
        t = gguf_find(nm);
        w->cw = gguf_matrix(t, s->H, 7);
        snprintf(nm, sizeof(nm), "blk.%d.shortconv.out_proj.weight", L);
        w->cout = gguf_matrix(gguf_find(nm), s->H, s->H);
    } else {
        int qd = s->HQ * s->D, kd = s->HKV * s->D;
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", L);
        w->wq = gguf_matrix(gguf_find(nm), qd, s->H);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", L);
        w->wk = gguf_matrix(gguf_find(nm), kd, s->H);
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", L);
        w->wv = gguf_matrix(gguf_find(nm), kd, s->H);
        snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", L);
        w->wg = gguf_matrix(gguf_find(nm), s->HQ, s->H);
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", L);
        w->wo = gguf_matrix(gguf_find(nm), s->H, qd);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", L);
        w->qn = gguf_vec_alloc(nm, s->D);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", L);
        w->kn = gguf_vec_alloc(nm, s->D);
    }
    return (w->an && w->pan && w->fnn && w->pfn && w->fg && w->fu && w->fd) ? 0 : -1;
}

static void norm_head(float *v, const float *w, int n, float eps)
{
    double s = 0.0;
    int i;
    float inv;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    inv = (float)(1.0 / sqrt(s / n + (double)eps));
    for (i = 0; i < n; i++) v[i] = v[i] * inv * w[i];
}

/* rope: pair i rotates dims i and i+half; pairs >= rotated_pairs untouched */
static void rope_apply(float *v, int pos, int dim, int rotated_pairs, double theta)
{
    int half = dim / 2, i;
    for (i = 0; i < rotated_pairs; i++) {
        double inv = 1.0 / pow(theta, (2.0 * i) / (double)dim);
        double a = (double)pos * inv;
        double sn = sin(a), cs = cos(a);
        double x0 = v[i], x1 = v[i + half];
        v[i] = (float)(x0 * cs - x1 * sn);
        v[i + half] = (float)(x1 * cs + x0 * sn);
    }
}

static void softmax_f32(double *v, int n)
{
    double m = v[0], s = 0.0;
    int i;
    for (i = 1; i < n; i++) if (v[i] > m) m = v[i];
    for (i = 0; i < n; i++) { v[i] = exp(v[i] - m); s += v[i]; }
    for (i = 0; i < n; i++) v[i] /= s;
}

/*
 * Run `ntok` tokens through the stack; capture the FFN input of every layer L
 * into cap[L*ntok + t][H].  top1 (optional) gets the lm_head argmax of the last
 * token (ties/NaN not handled: sanity check only).
 */
static int lfm_forward(const lfm_t *s, const uint32_t *tok, int ntok,
                       float *cap, int *top1)
{
    int L, t, i, h;
    const int H = s->H, FFN = s->FFN, D = s->D;
    const int QD = s->HQ * D, KD = s->HKV * D;
    const float eps = 1e-5f;
    float *x = calloc((size_t)H, 4), *res = calloc((size_t)H, 4);
    float *nrm = calloc((size_t)H, 4), *blk = calloc((size_t)H, 4);
    float *mrg = calloc((size_t)H, 4), *tmp = calloc((size_t)H, 4);
    float *q = calloc((size_t)QD, 4), *k = calloc((size_t)KD, 4), *v = calloc((size_t)KD, 4);
    float *gat = calloc((size_t)s->HQ, 4), *att = calloc((size_t)QD, 4);
    float *mix = calloc((size_t)3 * H, 4), *cin = calloc((size_t)H, 4);
    float *cvout = calloc((size_t)H, 4), *cvg = calloc((size_t)H, 4);
    float *fa = calloc((size_t)FFN, 4), *fu = calloc((size_t)FFN, 4), *fg = calloc((size_t)FFN, 4);
    float *kvk = calloc((size_t)s->NL * (ntok > 0 ? ntok : 1) * KD, 4);
    float *kvv = calloc((size_t)s->NL * (ntok > 0 ? ntok : 1) * KD, 4);
    float *cst = calloc((size_t)s->NL * 6 * H, 4);
    double *sc = calloc((size_t)(ntok > 0 ? ntok : 1), 8);
    const tinfo_t *temb = gguf_find("token_embd.weight");
    const tinfo_t *ton = gguf_find("output_norm.weight");
    int64_t erb = gguf_bytes(temb) / s->VOCAB;
    float *x0 = calloc((size_t)H, 4);
    float *onorm = calloc((size_t)H, 4);
    float *logit = NULL;
    int rc = -1;

    if (!x || !res || !nrm || !blk || !mrg || !tmp || !q || !k || !v || !gat ||
        !att || !mix || !cin || !cvout || !cvg || !fa || !fu || !fg ||
        !kvk || !kvv || !cst || !sc || !x0 || !onorm) goto out;

    for (t = 0; t < ntok; t++) {
        gguf_row_f32(g_data + (int64_t)temb->off + (int64_t)tok[t] * erb, H,
                     (int)temb->type, x);
        for (L = 0; L < s->NL; L++) {
            lw_t w;
            if (lw_load(&w, s, L) != 0) { lw_free(&w); goto out; }
            memcpy(res, x, (size_t)H * 4);
            f32_rmsnorm(x, w.an, H, eps, nrm);

            if (s->kind[L] == 0) {
                /* shortconv: b*x -> 7-tap depthwise -> c*conv -> out_proj */
                float *st = cst + (size_t)L * 6 * H;
                f32_matvec(w.cin, 3 * H, H, nrm, mix);
                for (i = 0; i < H; i++) cin[i] = mix[i] * mix[2 * H + i];
                for (i = 0; i < H; i++) {
                    double acc = (double)w.cw[(size_t)i * 7 + 6] * cin[i];
                    int j;
                    for (j = 0; j < 6; j++) acc += (double)w.cw[(size_t)i * 7 + j] * st[j * H + i];
                    cvout[i] = (float)acc;
                }
                memmove(st, st + H, (size_t)5 * H * 4);
                memcpy(st + 5 * H, cin, (size_t)H * 4);
                for (i = 0; i < H; i++) cvg[i] = mix[H + i] * cvout[i];
                f32_matvec(w.cout, H, H, cvg, blk);
            } else {
                int start = (s->kind[L] == 2) ? (t + 1 > s->WIN ? t + 1 - s->WIN : 0) : 0;
                int n = t + 1 - start;
                f32_matvec(w.wq, QD, H, nrm, q);
                f32_matvec(w.wk, KD, H, nrm, k);
                f32_matvec(w.wv, KD, H, nrm, v);
                f32_matvec(w.wg, s->HQ, H, nrm, gat);
                for (h = 0; h < s->HQ; h++) {
                    norm_head(q + h * D, w.qn, D, eps);
                    rope_apply(q + h * D, t, D, s->kind[L] == 1 ? D / 4 : D / 2,
                               s->kind[L] == 1 ? 1e6 : 1e4);
                }
                for (h = 0; h < s->HKV; h++) {
                    norm_head(k + h * D, w.kn, D, eps);
                    rope_apply(k + h * D, t, D, s->kind[L] == 1 ? D / 4 : D / 2,
                               s->kind[L] == 1 ? 1e6 : 1e4);
                }
                memcpy(kvk + ((size_t)L * ntok + t) * KD, k, (size_t)KD * 4);
                memcpy(kvv + ((size_t)L * ntok + t) * KD, v, (size_t)KD * 4);
                memset(att, 0, (size_t)QD * 4);
                {
                    int grp = s->HQ / s->HKV;
                    for (h = 0; h < s->HQ; h++) {
                        int kh = h / grp, j;
                        double scale = 1.0 / sqrt((double)D);
                        for (j = 0; j < n; j++) {
                            const float *kk = kvk + ((size_t)L * ntok + start + j) * KD + kh * D;
                            const float *qq = q + h * D;
                            double d = 0.0;
                            int z;
                            for (z = 0; z < D; z++) d += (double)qq[z] * kk[z];
                            sc[j] = d * scale;
                        }
                        softmax_f32(sc, n);
                        for (j = 0; j < n; j++) {
                            const float *vv = kvv + ((size_t)L * ntok + start + j) * KD + kh * D;
                            float *o = att + h * D;
                            int z;
                            for (z = 0; z < D; z++) o[z] += (float)(sc[j] * vv[z]);
                        }
                    }
                }
                for (h = 0; h < s->HQ; h++) {
                    float g = (float)(1.0 / (1.0 + exp(-(double)gat[h])));
                    for (i = 0; i < D; i++) att[h * D + i] *= g;
                }
                f32_matvec(w.wo, H, QD, att, blk);
            }

            f32_rmsnorm(blk, w.pan, H, eps, tmp);
            for (i = 0; i < H; i++) mrg[i] = res[i] + tmp[i];
            f32_rmsnorm(mrg, w.fnn, H, eps, nrm);
            if (cap)
                memcpy(cap + ((size_t)L * ntok + t) * H, nrm, (size_t)H * 4);

            f32_matvec(w.fu, FFN, H, nrm, fu);
            f32_matvec(w.fg, FFN, H, nrm, fg);
            for (i = 0; i < FFN; i++) fa[i] = (float)(ffn_silu_f32((double)fg[i]) * (double)fu[i]);
            f32_matvec(w.fd, H, FFN, fa, blk);
            f32_rmsnorm(blk, w.pfn, H, eps, tmp);
            for (i = 0; i < H; i++) x[i] = mrg[i] + tmp[i];
            lw_free(&w);
        }
        if (t == ntok - 1 && top1) {
            gguf_vec_f32(ton, onorm, H);
            f32_rmsnorm(x, onorm, H, eps, x0);
            logit = malloc((size_t)s->VOCAB * 4);
            if (!logit) goto out;
            {
                float best = -1e38f;
                int bi = 0, id;
                for (id = 0; id < s->VOCAB; id++) {
                    float *row = malloc((size_t)H * 4);
                    double d = 0.0;
                    int z;
                    gguf_row_f32(g_data + (int64_t)temb->off + (int64_t)id * erb, H,
                                 (int)temb->type, row);
                    for (z = 0; z < H; z++) d += (double)row[z] * x0[z];
                    if ((float)d > best) { best = (float)d; bi = id; }
                    free(row);
                }
                *top1 = bi;
            }
            free(logit);
        }
    }
    rc = 0;
out:
    free(x); free(res); free(nrm); free(blk); free(mrg); free(tmp);
    free(q); free(k); free(v); free(gat); free(att);
    free(mix); free(cin); free(cvout); free(cvg);
    free(fa); free(fu); free(fg); free(kvk); free(kvv); free(cst); free(sc);
    free(x0); free(onorm);
    return rc;
}

#endif /* LFM3_FWD_H */
