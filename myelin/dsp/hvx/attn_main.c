#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../common/fp16.h"
#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/rot.h"
#include "hvx_ops.h"

#define D 128
#define SCRATCH_FLOATS (2 * LFM_KVARN_PAGE_TOKENS * D)

#if defined(__HEXAGON__)
#define LFM_ATTN_NTOK 32u
#define LFM_ATTN_NTOK_LONG 64u
#define LFM_ATTN_NTOK_TWO 32u
#define LFM_ATTN_NTOK_F16 32u
#else
#define LFM_ATTN_NTOK 96u
#define LFM_ATTN_NTOK_LONG 192u
#define LFM_ATTN_NTOK_TWO 64u
#define LFM_ATTN_NTOK_F16 64u
#endif

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

typedef struct {
    uint32_t s;
} rng_t;

static float rng_f(rng_t *r)
{
    r->s = r->s * 1664525u + 1013904223u;
    return ((float)(r->s >> 8) / 16777216.0f - 0.5f) * 3.0f;
}

typedef struct {
    int n_heads;
    int kv_heads;
    uint32_t ntok;
    uint32_t start;
    uint32_t window;
} attn_cfg;

typedef struct {
    float *q;
    float *qr;
    float *k;
    float *kr;
    float *v;
    float *vr;
} attn_tensors;

static void *xalloc(size_t n)
{
    void *p = calloc(1, n);
    if (p == NULL) {
        printf("FAIL: alloc %zu bytes\n", n);
        exit(2);
    }
    return p;
}

static const char *fmt_name(lfm_kv_format fmt)
{
    switch (fmt) {
    case LFM_KV_F16: return "f16";
    case LFM_KVARN_K4V2: return "k4v2";
    case LFM_KVARN_K4V4: return "k4v4";
    default: return "?";
    }
}

static size_t head_bytes_of(lfm_kv_format fmt)
{
    if (fmt == LFM_KV_F16)
        return 2u * (size_t)LFM_KVARN_PAGE_TOKENS * D * 2u;
    return lfm_kvarn_head_bytes(fmt);
}

static size_t page_bytes_of(lfm_kv_format fmt, int kv_heads)
{
    return (size_t)lfm_page_bytes((uint32_t)fmt, (uint32_t)kv_heads, (uint32_t)D);
}

static void tensors_init(attn_tensors *ts, const attn_cfg *c, uint32_t seed)
{
    size_t qn = (size_t)c->ntok * (size_t)c->n_heads * D;
    size_t kn = (size_t)(c->start + c->ntok) * (size_t)c->kv_heads * D;
    size_t i;
    uint32_t t;
    int h;
    rng_t r;

    r.s = seed;
    ts->q = xalloc(qn * sizeof(float));
    ts->qr = xalloc(qn * sizeof(float));
    ts->k = xalloc(kn * sizeof(float));
    ts->kr = xalloc(kn * sizeof(float));
    ts->v = xalloc(kn * sizeof(float));
    ts->vr = xalloc(kn * sizeof(float));

    for (i = 0; i < qn; i++)
        ts->q[i] = rng_f(&r);
    for (i = 0; i < kn; i++) {
        ts->k[i] = rng_f(&r);
        ts->v[i] = rng_f(&r);
    }

    for (t = 0; t < c->ntok; t++) {
        for (h = 0; h < c->n_heads; h++) {
            size_t off = ((size_t)t * (size_t)c->n_heads + (size_t)h) * D;
            memcpy(ts->qr + off, ts->q + off, sizeof(float) * D);
            lfm_rot_fwd_f32(ts->qr + off, D);
        }
    }
    for (t = 0; t < c->start + c->ntok; t++) {
        int kh;
        for (kh = 0; kh < c->kv_heads; kh++) {
            size_t off = ((size_t)t * (size_t)c->kv_heads + (size_t)kh) * D;
            memcpy(ts->kr + off, ts->k + off, sizeof(float) * D);
            memcpy(ts->vr + off, ts->v + off, sizeof(float) * D);
            lfm_rot_fwd_f32(ts->kr + off, D);
            lfm_rot_fwd_f32(ts->vr + off, D);
        }
    }
}

static void tensors_free(attn_tensors *ts)
{
    free(ts->q);
    free(ts->qr);
    free(ts->k);
    free(ts->kr);
    free(ts->v);
    free(ts->vr);
}

static void commit_pages(lfm_kv_format fmt, const attn_cfg *c, const attn_tensors *ts,
                         uint8_t *arena, const uint32_t *tbl)
{
    size_t pb = page_bytes_of(fmt, c->kv_heads);
    uint32_t npages = (c->start + c->ntok) / LFM_KVARN_PAGE_TOKENS;
    float khead[LFM_KVARN_PAGE_TOKENS * D];
    float vhead[LFM_KVARN_PAGE_TOKENS * D];
    uint32_t lp;
    int kh;
    int j;

    for (lp = 0; lp < npages; lp++) {
        uint8_t *page = arena + (size_t)tbl[lp] * pb;
        for (kh = 0; kh < c->kv_heads; kh++) {
            for (j = 0; j < LFM_KVARN_PAGE_TOKENS; j++) {
                size_t src = ((size_t)(lp * LFM_KVARN_PAGE_TOKENS + (uint32_t)j) *
                                  (size_t)c->kv_heads +
                              (size_t)kh) *
                             D;
                memcpy(khead + (size_t)j * D, ts->kr + src, sizeof(float) * D);
                memcpy(vhead + (size_t)j * D, ts->vr + src, sizeof(float) * D);
            }
            lfm_hvx_kvcache_commit(fmt, page, khead, vhead, kh, D);
        }
    }
}

static void decode_pages(lfm_kv_format fmt, const attn_cfg *c, const uint8_t *arena,
                         const uint32_t *tbl, float *kd, float *vd)
{
    size_t pb = page_bytes_of(fmt, c->kv_heads);
    size_t hb = head_bytes_of(fmt);
    uint32_t npages = (c->start + c->ntok) / LFM_KVARN_PAGE_TOKENS;
    uint32_t lp;
    int kh;
    int j;
    int d;

    for (lp = 0; lp < npages; lp++) {
        const uint8_t *base = arena + (size_t)tbl[lp] * pb;
        for (kh = 0; kh < c->kv_heads; kh++) {
            const uint8_t *page = base + (size_t)kh * hb;
            for (j = 0; j < LFM_KVARN_PAGE_TOKENS; j++) {
                uint32_t pos = lp * LFM_KVARN_PAGE_TOKENS + (uint32_t)j;
                size_t off = ((size_t)pos * (size_t)c->kv_heads + (size_t)kh) * D;
                if (fmt == LFM_KV_F16) {
                    for (d = 0; d < D; d++) {
                        kd[off + (size_t)d] = lfm_f16_to_f32(
                            lfm_f16_load(page + (size_t)j * D * 2u + (size_t)d * 2u));
                        vd[off + (size_t)d] = lfm_f16_to_f32(lfm_f16_load(
                            page + (size_t)LFM_KVARN_PAGE_TOKENS * D * 2u +
                            (size_t)j * D * 2u + (size_t)d * 2u));
                    }
                } else {
                    lfm_kvarn_decode_k(page, j, kd + off);
                    lfm_kvarn_decode_v(fmt, page, j, vd + off);
                }
            }
        }
    }
}

static double cos_head(const float *a, const float *b)
{
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    int i;

    for (i = 0; i < D; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0)
        return 0.0;
    return dot / sqrt(na * nb);
}

static void ref_attention(const attn_cfg *c, const float *q, const float *k, const float *v,
                          float *out)
{
    int group = c->n_heads / c->kv_heads;
    uint32_t t;
    int h;
    int d;

    for (t = 0; t < c->ntok; t++) {
        uint32_t pos = c->start + t;
        uint32_t w0 = lfm_window_start(pos, c->window);
        for (h = 0; h < c->n_heads; h++) {
            int kh = h / group;
            const float *qrow = q + ((size_t)t * (size_t)c->n_heads + (size_t)h) * D;
            float *orow = out + ((size_t)t * (size_t)c->n_heads + (size_t)h) * D;
            float mx = -FLT_MAX;
            float sum = 0.0f;
            uint32_t j;

            for (j = w0; j <= pos; j++) {
                const float *krow = k + ((size_t)j * (size_t)c->kv_heads + (size_t)kh) * D;
                float s = 0.0f;
                for (d = 0; d < D; d++)
                    s += qrow[d] * krow[d];
                s *= 1.0f / sqrtf((float)D);
                if (s > mx)
                    mx = s;
            }
            memset(orow, 0, sizeof(float) * D);
            for (j = w0; j <= pos; j++) {
                const float *krow = k + ((size_t)j * (size_t)c->kv_heads + (size_t)kh) * D;
                const float *vrow = v + ((size_t)j * (size_t)c->kv_heads + (size_t)kh) * D;
                float s = 0.0f;
                float p;
                for (d = 0; d < D; d++)
                    s += qrow[d] * krow[d];
                s *= 1.0f / sqrtf((float)D);
                p = expf(s - mx);
                sum += p;
                for (d = 0; d < D; d++)
                    orow[d] += p * vrow[d];
            }
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (d = 0; d < D; d++)
                    orow[d] *= inv;
            }
        }
    }
}

static void run_case(const char *tag, lfm_kv_format fmt, int n_heads, int kv_heads, uint32_t ntok,
                     uint32_t start, uint32_t window, uint8_t *arena, uint32_t arena_pages,
                     const uint32_t *tbl, uint32_t seed, double gate, double qual_gate,
                     int trunc_check)
{
    attn_cfg c;
    attn_tensors ts;
    uint32_t total = start + ntok;
    uint32_t npages = total / LFM_KVARN_PAGE_TOKENS;
    size_t on;
    size_t kn;
    float *scratch;
    float *out;
    float *ref;
    float *refq;
    float *kd;
    float *vd;
    clock_t c0;
    clock_t c1;
    double mn = 1.0;
    double sum = 0.0;
    double qmn = 1.0;
    uint32_t t;
    int h;
    uint32_t lp;

    c.n_heads = n_heads;
    c.kv_heads = kv_heads;
    c.ntok = ntok;
    c.start = start;
    c.window = window;

    CHECK(total % LFM_KVARN_PAGE_TOKENS == 0, "%s: total %u not page aligned", tag, total);
    for (lp = 0; lp < npages; lp++)
        CHECK(tbl[lp] < arena_pages, "%s: table entry %u out of arena %u", tag, tbl[lp],
              arena_pages);

    tensors_init(&ts, &c, seed);
    commit_pages(fmt, &c, &ts, arena, tbl);

    on = (size_t)ntok * (size_t)n_heads * D;
    kn = (size_t)total * (size_t)kv_heads * D;
    kd = xalloc(kn * sizeof(float));
    vd = xalloc(kn * sizeof(float));
    decode_pages(fmt, &c, arena, tbl, kd, vd);

    scratch = xalloc((size_t)SCRATCH_FLOATS * sizeof(float));
    out = xalloc(on * sizeof(float));

    c0 = clock();
    lfm_hvx_attn_flash(fmt, tbl, arena, kv_heads, n_heads, D, ts.qr, ntok, start, window, out,
                       scratch);
    c1 = clock();

    ref = xalloc(on * sizeof(float));
    ref_attention(&c, ts.qr, kd, vd, ref);
    for (t = 0; t < ntok; t++) {
        for (h = 0; h < n_heads; h++) {
            size_t off = ((size_t)t * (size_t)n_heads + (size_t)h) * D;
            double cc = cos_head(out + off, ref + off);
            if (cc < mn)
                mn = cc;
            sum += cc;
        }
    }

    refq = xalloc(on * sizeof(float));
    ref_attention(&c, ts.qr, ts.kr, ts.vr, refq);
    for (t = 0; t < ntok; t++) {
        for (h = 0; h < n_heads; h++) {
            size_t off = ((size_t)t * (size_t)n_heads + (size_t)h) * D;
            double cc = cos_head(out + off, refq + off);
            if (cc < qmn)
                qmn = cc;
        }
    }

    printf("[%s] fmt=%s heads=%d/%d ntok=%u start=%u window=%u cos_min=%.7f cos_mean=%.7f "
           "qual_min=%.6f time=%.2f ms\n",
           tag, fmt_name(fmt), n_heads, kv_heads, ntok, start, window, mn,
           sum / ((double)ntok * (double)n_heads), qmn,
           1000.0 * (double)(c1 - c0) / (double)CLOCKS_PER_SEC);

    CHECK(mn >= gate, "%s: cos_min %.7f < %.4f", tag, mn, gate);
    CHECK(qmn >= qual_gate, "%s: qual_min %.6f < %.4f", tag, qmn, qual_gate);

    if (trunc_check) {
        attn_cfg cf = c;
        float *reffull = xalloc(on * sizeof(float));
        float maxdiff = 0.0f;
        cf.window = 0;
        ref_attention(&cf, ts.qr, kd, vd, reffull);
        for (t = 0; t < ntok; t++) {
            for (h = 0; h < n_heads; h++) {
                size_t off = ((size_t)t * (size_t)n_heads + (size_t)h) * D;
                int d;
                for (d = 0; d < D; d++) {
                    float diff = fabsf(out[off + (size_t)d] - reffull[off + (size_t)d]);
                    if (diff > maxdiff)
                        maxdiff = diff;
                }
            }
        }
        printf("[%s] window truncation vs full context: maxdiff=%.6f\n", tag, (double)maxdiff);
        CHECK(maxdiff > 1e-2f, "%s: window did not truncate (maxdiff %.6f)", tag,
              (double)maxdiff);
        free(reffull);
    }

    free(scratch);
    free(out);
    free(ref);
    free(refq);
    free(kd);
    free(vd);
    tensors_free(&ts);
}

static void test_commit_layout(lfm_kv_format fmt)
{
    size_t hb = lfm_kvarn_head_bytes(fmt);
    size_t voff = lfm_kvarn_vcode_off(fmt);
    uint8_t *page = xalloc(hb * 3u);
    uint8_t *want = xalloc(hb);
    uint8_t *vpage = xalloc(hb);
    float *k = xalloc((size_t)LFM_KVARN_PAGE_TOKENS * D * sizeof(float));
    float *v = xalloc((size_t)LFM_KVARN_PAGE_TOKENS * D * sizeof(float));
    uint16_t *f = xalloc((size_t)LFM_KVARN_PAGE_TOKENS * D * sizeof(uint16_t));
    float *scr = xalloc((size_t)SCRATCH_FLOATS * sizeof(float));
    rng_t r;
    int i;
    size_t z;
    int zero = 1;

    r.s = 99u;
    for (i = 0; i < LFM_KVARN_PAGE_TOKENS * D; i++) {
        k[i] = rng_f(&r);
        v[i] = rng_f(&r);
    }
    printf("  commit-layout %s: start\n", fmt_name(fmt));
    lfm_hvx_kvcache_commit(fmt, page, k, v, 1, D);
    printf("  commit-layout %s: committed\n", fmt_name(fmt));

    for (i = 0; i < LFM_KVARN_PAGE_TOKENS * D; i++)
        f[i] = lfm_f32_to_f16(k[i]);
    lfm_kvarn_commit_work(fmt, f, want, scr);
    for (i = 0; i < LFM_KVARN_PAGE_TOKENS * D; i++)
        f[i] = lfm_f32_to_f16(v[i]);
    lfm_kvarn_commit_work(fmt, f, vpage, scr);
    memcpy(want + voff, vpage + voff, hb - voff);

    CHECK(memcmp(page + hb, want, hb) == 0, "%s: head 1 page does not match K|V merge",
          fmt_name(fmt));
    for (z = 0; z < hb; z++) {
        if (page[hb + z] != want[z]) {
            printf("  %s first diff at %zu (%s): %02x != %02x\n", fmt_name(fmt), z,
                   z < voff ? "K" : "V", page[hb + z], want[z]);
            break;
        }
    }
    for (z = 0; z < hb; z++) {
        if (page[z] != 0 || page[2u * hb + z] != 0)
            zero = 0;
    }
    CHECK(zero, "%s: commit touched neighbour heads", fmt_name(fmt));

    free(page);
    free(want);
    free(vpage);
    free(scr);
}

static void test_two_sequences(void)
{
    enum { NTOK = (int)LFM_ATTN_NTOK_TWO, PAGES = 4 };
    static const uint32_t tbl_a[2] = {3u, 0u};
    static const uint32_t tbl_b[2] = {2u, 1u};
    int kv_heads = 4;
    uint8_t *arena = xalloc(page_bytes_of(LFM_KVARN_K4V4, kv_heads) * PAGES);

    run_case("seqA", LFM_KVARN_K4V4, 16, kv_heads, NTOK, 0u, 0u, arena, PAGES, tbl_a, 11u, 0.999,
             0.95, 0);
    run_case("seqB", LFM_KVARN_K4V4, 16, kv_heads, NTOK, 0u, 0u, arena, PAGES, tbl_b, 77u, 0.999,
             0.95, 0);

    free(arena);
}

static void test_f16(void)
{
    enum { NTOK = (int)LFM_ATTN_NTOK_F16, PAGES = 2 };
    static const uint32_t tbl[2] = {1u, 0u};
    int kv_heads = 4;
    uint8_t *arena = xalloc(page_bytes_of(LFM_KV_F16, kv_heads) * PAGES);

    run_case("f16", LFM_KV_F16, 16, kv_heads, NTOK, 0u, 0u, arena, PAGES, tbl, 5u, 0.9999, 0.999,
             0);

    free(arena);
}

int main(void)
{
    enum { NTOK = (int)LFM_ATTN_NTOK, PAGES = 3, KVH = 4, NH = 16 };
    static const uint32_t tbl_a[3] = {2u, 0u, 1u};
    static const uint32_t tbl_b[3] = {1u, 2u, 0u};
    static const uint32_t tbl_w[3] = {2u, 0u, 1u};
    static const uint32_t tbl_s[4] = {1u, 3u, 0u, 2u};
    static const uint32_t tbl_d[3] = {1u, 2u, 0u};
    static const uint32_t tbl_l[6] = {4u, 1u, 5u, 0u, 3u, 2u};
#if defined(__HEXAGON__)
    clock_t t0 = 0;
#else
    clock_t t0 = clock();
#endif
    uint8_t *arena;
    size_t page_k4v4 = page_bytes_of(LFM_KVARN_K4V4, KVH);
    size_t page_k4v2 = page_bytes_of(LFM_KVARN_K4V2, KVH);

    printf("attn: host test, head_dim=%d scratch=%d floats\n", D, SCRATCH_FLOATS);

    test_commit_layout(LFM_KVARN_K4V4);
    test_commit_layout(LFM_KVARN_K4V2);

#if defined(__HEXAGON__)
    goto done;
#endif

    arena = xalloc(page_k4v4 * PAGES);
    run_case("full-k4v4", LFM_KVARN_K4V4, NH, KVH, NTOK, 0u, 0u, arena, PAGES, tbl_a, 1234u,
             0.999, 0.95, 0);
    free(arena);

    arena = xalloc(page_k4v2 * PAGES);
    run_case("full-k4v2", LFM_KVARN_K4V2, NH, KVH, NTOK, 0u, 0u, arena, PAGES, tbl_b, 4321u,
             0.995, 0.80, 0);
    free(arena);

    arena = xalloc(page_k4v4 * PAGES);
    run_case("win64", LFM_KVARN_K4V4, NH, KVH, NTOK, 0u, 64u, arena, PAGES, tbl_w, 2468u, 0.999,
             0.95, 1);
    free(arena);

    arena = xalloc(page_k4v4 * 4u);
    run_case("start120-win32", LFM_KVARN_K4V4, NH, KVH, 8u, 120u, 32u, arena, 4u, tbl_s, 1357u,
             0.999, 0.95, 0);
    free(arena);

    arena = xalloc(page_k4v2 * PAGES);
    run_case("decode-k4v2", LFM_KVARN_K4V2, NH, KVH, 1u, 95u, 0u, arena, PAGES, tbl_d, 8642u,
             0.995, 0.80, 0);
    free(arena);

    arena = xalloc(page_k4v4 * 6u);
    run_case("long-win128", LFM_KVARN_K4V4, NH, KVH, LFM_ATTN_NTOK_LONG, 0u, 128u, arena, 6u, tbl_l, 9753u,
             0.999, 0.95, 1);
    free(arena);

    test_two_sequences();
    test_f16();

done:
    if (failures) {
        printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
#if defined(__HEXAGON__)
    printf("PASS: all checks passed\n");
#else
    printf("PASS: all checks passed in %.2f ms\n",
           1000.0 * (double)(clock() - t0) / (double)CLOCKS_PER_SEC);
#endif
    return 0;
}
