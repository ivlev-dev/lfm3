/* silu_err_study.c -- host measurement of the FFN mid-buffer requant.
 *
 * Question: silu_range() in lfmgemv_impl.c computes
 *     pr = mulq16(silu_q16(zg), zu);        // Q16 of silu(g)*u  = P<<16
 *     v  = mulq16(pr >> 12, mq_mid);        // mq_mid = 4096/s_mid
 * mulq16() resolves its first operand with a step of 256 (see the comment in
 * lfmgemv_impl.c): mulq16(a,b) has absolute error (a & 0xff)*b/65536, i.e. up
 * to b/257 in the units of the result.  Here a = pr>>12 = 16*P, so every lane
 * with P < 16 collapses to 0 and the rest is quantised to a step of 16 in P
 * before the multiplier is applied.
 *
 * Measured on the real LFM3-1.7B Q4_K_M weights:
 *   - layer 0 is a "conv" layer (no attention), so its FFN input is produced
 *     exactly by the GGUF path token_embd -> attn_norm -> shortconv ->
 *     post_attention_norm -> ffn_norm, for a sequence of real token ids;
 *   - gate/up/down are dequantised in f32; P = silu(gate)*up over all 7680
 *     lanes of every token;
 *   - s_mid = max|P|/127 (absmax calibration), and the mid buffer is requantised
 *     four ways: (a) literal mulq16, (b) int64, (c) 32-bit split of the
 *     multiplier, (d) ideal round(P/s_mid);
 *   - the FFN output is y = Wd*(mid8*s_mid) with exact f32 weights, so the
 *     measurement isolates the requant from the weight quantisation.
 *
 * Build: cc -O2 silu_err_study.c -lm -o silu_err_study
 * Run:   ./silu_err_study ../../../models/LFM3-1.7B-Q4_K_M.gguf 16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define HID 2048
#define INT 7680

/* ------------------------------------------------------------------ GGUF */
static FILE *gf;
static int64_t g_data;
static int g_align = 32;

typedef struct { char name[64]; uint32_t nd; uint64_t dim[4]; uint32_t type; uint64_t off; } tinfo_t;
static tinfo_t g_t[300];
static int g_nt;

static void grd(void *p, size_t n)
{
    if (fread(p, 1, n, gf) != n) { fprintf(stderr, "gguf: short read\n"); exit(1); }
}
static uint32_t gu32(void) { uint32_t v; grd(&v, 4); return v; }
static uint64_t gu64(void) { uint64_t v; grd(&v, 8); return v; }
static char *gstr(void)
{
    uint64_t n = gu64();
    char *s = malloc(n + 1);
    grd(s, n); s[n] = 0;
    return s;
}
static void skip_val(uint32_t t, uint32_t *out_u32)
{
    switch (t) {
    case 0: case 1: case 7: { uint8_t v; grd(&v, 1); break; }
    case 2: case 3: { uint16_t v; grd(&v, 2); break; }
    case 4: { uint32_t v; grd(&v, 4); if (out_u32) *out_u32 = v; break; }
    case 5: case 6: { uint32_t v; grd(&v, 4); break; }
    case 10: case 11: case 12: { uint64_t v; grd(&v, 8); break; }
    case 8: { char *s = gstr(); free(s); break; }
    case 9: {
        uint32_t et = gu32(); uint64_t n = gu64(), i;
        for (i = 0; i < n; i++) skip_val(et, NULL);
        break;
    }
    default: fprintf(stderr, "gguf: kv type %u\n", t); exit(1);
    }
}

static const tinfo_t *tfind(const char *name)
{
    int i;
    for (i = 0; i < g_nt; i++) if (!strcmp(g_t[i].name, name)) return &g_t[i];
    fprintf(stderr, "gguf: no tensor %s\n", name);
    exit(1);
}

static void gguf_open(const char *path)
{
    char magic[4];
    uint64_t nkv; int i;
    gf = fopen(path, "rb");
    if (!gf) { perror(path); exit(1); }
    grd(magic, 4);
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); exit(1); }
    gu32();                                   /* version */
    g_nt = (int)gu64();
    nkv = gu64();
    for (i = 0; i < (int)nkv; i++) {
        char *k = gstr();
        uint32_t t = gu32();
        if (!strcmp(k, "general.alignment")) skip_val(t, (uint32_t *)&g_align);
        else skip_val(t, NULL);
        free(k);
    }
    for (i = 0; i < g_nt; i++) {
        char *n = gstr();
        uint32_t d;
        snprintf(g_t[i].name, sizeof(g_t[i].name), "%s", n);
        free(n);
        g_t[i].nd = gu32();
        for (d = 0; d < g_t[i].nd && d < 4; d++) g_t[i].dim[d] = gu64();
        g_t[i].type = gu32();
        g_t[i].off = gu64();
    }
    g_data = (int64_t)((ftell(gf) + g_align - 1) / g_align * g_align);
}

static int64_t tensor_bytes(const tinfo_t *t)
{
    int64_t cnt = 1; uint32_t i;
    for (i = 0; i < t->nd; i++) cnt *= (int64_t)t->dim[i];
    switch (t->type) {
    case 0:  return cnt * 4;
    case 1:  return cnt * 2;
    case 12: return cnt / 256 * 144;          /* Q4_K */
    case 14: return cnt / 256 * 210;          /* Q6_K */
    default: fprintf(stderr, "type %u unsupported\n", t->type); exit(1);
    }
}

/* ------------------------------------------------------------------ dequant */
static float f16(uint16_t h)
{
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3ffu;
            f = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    { float r; memcpy(&r, &f, 4); return r; }
}

static void deq_q4k(const uint8_t *b, float *y)
{
    float d = f16((uint16_t)(b[0] | (b[1] << 8)));
    float dmin = f16((uint16_t)(b[2] | (b[3] << 8)));
    const uint8_t *sc = b + 4, *qs = b + 16;
    int g, half, l;
    for (g = 0; g < 4; g++) for (half = 0; half < 2; half++) {
        int j = 2 * g + half, s, m;
        if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
        else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
               m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
        for (l = 0; l < 32; l++) {
            int v = qs[g * 32 + l];
            int q = half ? (v >> 4) : (v & 0xF);
            y[j * 32 + l] = d * (float)s * (float)q - dmin * (float)m;
        }
    }
}

static void deq_q6k(const uint8_t *b, float *y)
{
    float d = f16((uint16_t)(b[208] | (b[209] << 8)));
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    int n, l;
    for (n = 0; n < 2; n++) for (l = 0; l < 32; l++) {
        int base = n * 64, hb = n * 32, isb = l / 16;
        int q1 = ((ql[base + l] & 0xF) | (((qh[hb + l] >> 0) & 3) << 4)) - 32;
        int q2 = ((ql[base + l + 32] & 0xF) | (((qh[hb + l] >> 2) & 3) << 4)) - 32;
        int q3 = ((ql[base + l] >> 4) | (((qh[hb + l] >> 4) & 3) << 4)) - 32;
        int q4 = ((ql[base + l + 32] >> 4) | (((qh[hb + l] >> 6) & 3) << 4)) - 32;
        y[n * 128 + l +  0] = d * (float)sc[n * 8 + isb + 0] * (float)q1;
        y[n * 128 + l + 32] = d * (float)sc[n * 8 + isb + 2] * (float)q2;
        y[n * 128 + l + 64] = d * (float)sc[n * 8 + isb + 4] * (float)q3;
        y[n * 128 + l + 96] = d * (float)sc[n * 8 + isb + 6] * (float)q4;
    }
}

/* dequantise one row of `n` values (n % 256 == 0) at absolute file offset */
static void row_f32(int64_t off, int n, int type, float *out)
{
    int nb = n / 256, i, bs = (type == 12) ? 144 : 210;
    size_t rb = (size_t)nb * bs;
    uint8_t *buf = malloc(rb);
    fseek(gf, (long)off, SEEK_SET);
    grd(buf, rb);
    for (i = 0; i < nb; i++) {
        if (type == 12) deq_q4k(buf + (size_t)i * 144, out + i * 256);
        else            deq_q6k(buf + (size_t)i * 210, out + i * 256);
    }
    free(buf);
}

/* --------------------------------------------------------------- arithmetic */
static inline int32_t mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((uint32_t)lo * (uint32_t)b) >> 8);
}

static void rmsnorm(const float *v, const float *w, int n, float eps, float *out)
{
    double s = 0.0; int i; float inv;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    inv = (float)(1.0 / sqrt(s / n + eps));
    for (i = 0; i < n; i++) out[i] = v[i] * inv * w[i];
}

/* dequantise a whole [m][k] tensor into f32 (row-major, ggml ne order) */
static float *load_matrix(const tinfo_t *t, int m, int k)
{
    float *w = malloc((size_t)m * k * sizeof(float));
    int64_t rb = tensor_bytes(t) / m;
    int r;
    if (!w) { fprintf(stderr, "oom\n"); exit(1); }
    for (r = 0; r < m; r++)
        row_f32(g_data + (int64_t)t->off + (int64_t)r * rb, k, (int)t->type, w + (size_t)r * k);
    return w;
}

static void matvec(const float *w, int m, int k, const float *x, float *out)
{
    int r, i;
    for (r = 0; r < m; r++) {
        const float *wr = w + (size_t)r * k;
        double s = 0.0;
        for (i = 0; i < k; i++) s += (double)wr[i] * x[i];
        out[r] = (float)s;
    }
}

static double cosine(const float *a, const float *b, int n)
{
    double d = 0.0, na = 0.0, nb = 0.0; int i;
    for (i = 0; i < n; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return d / (sqrt(na) * sqrt(nb) + 1e-30);
}

static int cmpf(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* one mid-buffer requant decision, variant 0..3 */
static int requant(double P, int32_t mq, double s_mid, int v)
{
    int32_t pr = (int32_t)llround(P * 65536.0);   /* Q16, as the kernel */
    int32_t a = pr >> 12;
    int32_t got;
    switch (v) {
    case 0: got = mulq16(a, mq); break;
    case 1: got = (int32_t)(((int64_t)a * mq + 32768) >> 16); break;
    case 2: {
        int32_t mhi = mq >> 8, mlo = mq & 0xff;
        got = (int32_t)(((int64_t)a * mhi + (((int64_t)a * mlo) >> 8)) >> 8);
        break;
    }
    case 3: got = (int32_t)lround(P / s_mid); break;
    default: {
        /* full precision: shift AFTER the 64-bit product */
        int64_t t64 = ((int64_t)pr * (int64_t)mq + ((int64_t)1 << 27)) >> 28;
        got = (int32_t)t64;
        break;
    }
    }
    if (got > 127) got = 127;
    if (got < -127) got = -127;
    return got;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "../../models/LFM3-1.7B-Q4_K_M.gguf";
    int ntok = argc > 2 ? atoi(argv[2]) : 16;
    const float eps = 1e-5f;
    const tinfo_t *temb, *tan, *tpan, *tffn, *tin, *tconv, *tout, *tg, *tu, *td;
    float *x0 = malloc(HID * 4), *h = malloc(HID * 4), *x1 = malloc(HID * 4);
    float *ffn_in = malloc(HID * 4);
    float *bcx = malloc(3 * HID * 4), *op = malloc(HID * 4), *cprod = malloc(HID * 4);
    float *hist = calloc((size_t)7 * HID, 4);     /* shortconv ring: 6 old + cur */
    float *nw = malloc(HID * 4), *pan = malloc(HID * 4), *fnw = malloc(HID * 4);
    float *gate = malloc(INT * 4), *up = malloc(INT * 4), *yq = malloc(HID * 4);
    float *gate_all, *up_all;
    float *Win, *Wout, *Wg, *Wu, *Wd, *cw;
    float *mid_all, *yref_all;
    int t, i;
    double pmax = 0.0, prms = 0.0, p999 = 0.0, p50 = 0.0;
    int64_t pn = 0;
    uint32_t st = 20240517u;
    int64_t erb;

    /* ---- self-test of the fixed requant (same expression as the DSP) ----
     * code = clamp(round(pr*mq/2^28)), pr = Q16 of P, mq = 4096/S.
     * Checks: agreement with round(P/S) within 1 LSB, saturation, and that the
     * 64-bit product cannot overflow for pr,mq < 2^31. */
    {
        static const double Ss[] = { 0.001953125, 0.0127, 0.0625, 0.5, 4.0, 100.0 };
        long bad = 0, n = 0, sat = 0;
        double mqerr = 0.0;
        int si, pi;
        for (si = 0; si < 6; si++) {
            double S = Ss[si];
            int32_t mq = (int32_t)(4096.0f / (float)S);
            for (pi = -400; pi <= 400; pi++) {
                double P = (double)pi * (S / 8.0);        /* -50..50 LSB */
                int64_t pr = (int64_t)llround(P * 65536.0);
                int64_t v = (pr * (int64_t)mq + ((int64_t)1 << 27)) >> 28;
                long ideal = lround(P / S);               /* exact S */
                long kern = lround((double)pr * (double)mq / 268435456.0);
                long d;
                if (v > 127) { v = 127; sat++; }
                if (v < -127) { v = -127; sat++; }
                d = (long)v - kern;                       /* requant only */
                if (d < 0) d = -d;
                if (d > 1) bad++;
                /* extra error from the integer mq_mid = 4096/S */
                {
                    double e = fabs((double)v - (double)ideal);
                    if (e > mqerr) mqerr = e;
                }
                n++;
            }
        }
        printf("requant self-test: %ld values, %ld with |req err|>1 LSB, %ld sat; "
               "max |v-round(P/S)| incl. int32 mq truncation = %.2f LSB\n",
               n, bad, sat, mqerr);
        printf("  max |pr*mq| = %.3e (int64 limit 9.2e18)\n",
               (double)((int64_t)0x7fffffffLL * (int64_t)0x7fffffffLL));
    }

    gguf_open(path);
    temb = tfind("token_embd.weight");
    tan  = tfind("blk.0.attn_norm.weight");
    tpan = tfind("blk.0.post_attention_norm.weight");
    tffn = tfind("blk.0.ffn_norm.weight");
    tin  = tfind("blk.0.shortconv.in_proj.weight");
    tconv= tfind("blk.0.shortconv.conv.weight");
    tout = tfind("blk.0.shortconv.out_proj.weight");
    tg   = tfind("blk.0.ffn_gate.weight");
    tu   = tfind("blk.0.ffn_up.weight");
    td   = tfind("blk.0.ffn_down.weight");

    printf("loading layer-0 weights (gate/up/down/in_proj/out_proj)...\n"); fflush(stdout);
    Win  = load_matrix(tin, 3 * HID, HID);
    Wout = load_matrix(tout, HID, HID);
    Wg   = load_matrix(tg, INT, HID);
    Wu   = load_matrix(tu, INT, HID);
    Wd   = load_matrix(td, HID, INT);
    cw = malloc((size_t)HID * 7 * 4);              /* f32 [k][tap] */
    fseek(gf, (long)(g_data + (int64_t)tconv->off), SEEK_SET);
    grd(cw, (size_t)HID * 7 * 4);
    fseek(gf, (long)(g_data + (int64_t)tan->off), SEEK_SET);  grd(nw, HID * 4);
    fseek(gf, (long)(g_data + (int64_t)tpan->off), SEEK_SET); grd(pan, HID * 4);
    fseek(gf, (long)(g_data + (int64_t)tffn->off), SEEK_SET); grd(fnw, HID * 4);

    mid_all = malloc((size_t)ntok * INT * 4);
    yref_all = malloc((size_t)ntok * HID * 4);
    gate_all = malloc((size_t)ntok * INT * 4);
    up_all = malloc((size_t)ntok * INT * 4);
    if (!mid_all || !yref_all) { fprintf(stderr, "oom\n"); return 1; }

    erb = tensor_bytes(temb) / 128000;
    for (t = 0; t < ntok; t++) {
        float *mid = mid_all + (size_t)t * INT;
        st = st * 1664525u + 1013904223u;
        {
            int id = (int)(st % 127999u);
            /* real token embedding row (Q6_K) */
            row_f32(g_data + (int64_t)temb->off + (int64_t)id * erb, HID, (int)temb->type, x0);
        }
        rmsnorm(x0, nw, HID, eps, h);
        matvec(Win, 3 * HID, HID, h, bcx);
        for (i = 0; i < HID; i++) cprod[i] = bcx[i] * bcx[2 * HID + i];
        /* causal 7-tap depthwise conv: window = [t-6 .. t] */
        memmove(hist, hist + HID, (size_t)6 * HID * 4);
        memcpy(hist + 6 * HID, cprod, (size_t)HID * 4);
        for (i = 0; i < HID; i++) {
            double s = 0.0; int j;
            for (j = 0; j < 7; j++) s += (double)cw[i * 7 + j] * hist[j * HID + i];
            op[i] = (float)s;
        }
        for (i = 0; i < HID; i++) op[i] = bcx[HID + i] * op[i];
        matvec(Wout, HID, HID, op, x1);                /* out_proj @ (c*conv) */
        rmsnorm(x1, pan, HID, eps, x1);
        for (i = 0; i < HID; i++) x1[i] += x0[i];      /* residual = embedding */
        rmsnorm(x1, fnw, HID, eps, ffn_in);

        {
            double se = 0.0, sf = 0.0;
            for (i = 0; i < HID; i++) { se += (double)x0[i] * x0[i]; sf += (double)ffn_in[i] * ffn_in[i]; }
            if (t == 0)
                printf("  emb rms %.4f  ffn_in rms %.4f\n",
                       sqrt(se / HID), sqrt(sf / HID));
        }
        matvec(Wg, INT, HID, ffn_in, gate);
        matvec(Wu, INT, HID, ffn_in, up);
        memcpy(gate_all + (size_t)t * INT, gate, (size_t)INT * 4);
        memcpy(up_all + (size_t)t * INT, up, (size_t)INT * 4);
        {
            double sg = 0.0, su = 0.0; float ag = 0, au = 0;
            for (i = 0; i < INT; i++) {
                sg += (double)gate[i] * gate[i]; su += (double)up[i] * up[i];
                if (fabsf(gate[i]) > ag) ag = fabsf(gate[i]);
                if (fabsf(up[i]) > au) au = fabsf(up[i]);
            }
            if (t == 0)
                printf("  gate rms %.3f absmax %.3f   up rms %.3f absmax %.3f\n",
                       sqrt(sg / INT), ag, sqrt(su / INT), au);
        }
        for (i = 0; i < INT; i++) {
            double g = gate[i];
            double p = (g / (1.0 + exp(-g))) * (double)up[i];
            double ap = fabs(p);
            mid[i] = (float)p;
            if (ap > pmax) pmax = ap;
            prms += p * p; pn++;
        }
        matvec(Wd, HID, INT, mid, yref_all + (size_t)t * HID);
    }

    prms = sqrt(prms / (double)pn);
    {
        float *a = malloc((size_t)pn * 4);
        int64_t k = 0;
        for (t = 0; t < ntok; t++) for (i = 0; i < INT; i++)
            a[k++] = (float)fabs(mid_all[(size_t)t * INT + i]);
        qsort(a, (size_t)pn, 4, cmpf);
        p999 = a[(int64_t)(0.999 * (double)(pn - 1))];
        p50 = a[pn / 2];
        printf("\nP = silu(gate)*up, %d tokens x %d lanes:\n", ntok, INT);
        printf("  rms %.4f  absmax %.4f  p99.9 %.4f  p50 %.4f\n", prms, pmax, p999, p50);
        free(a);
    }

    {
        const char *nm[5] = { "a mulq16 (as is)", "b int64 on pr>>12", "c shift/split", "d ideal round", "e int64 on Q16 pr" };
        double s_mid = pmax / 127.0;
        int32_t mq = (int32_t)(4096.0f / (float)s_mid);
        int v;
        double mid_cos[5] = { 0 }, out_worst[5] = { 0 }, out_mean[5] = { 0 };
        double lsb_rms[5] = { 0 };
        int maxlsb[5] = { 0 };

        printf("\ns_mid = absmax/127 = %.6f   mq_mid = 4096/s_mid = %d\n", s_mid, mq);
        for (v = 0; v < 5; v++) {
            double sum2 = 0.0;
            int n = 0;
            for (t = 0; t < ntok; t++) {
                float hq[INT];
                float *mid = mid_all + (size_t)t * INT;
                double dmid = 0.0, na = 0.0, nb = 0.0;
                for (i = 0; i < INT; i++) {
                    int ideal = (int)lround(mid[i] / s_mid);
                    int got = requant(mid[i], mq, s_mid, v);
                    int d = got - ideal;
                    if (ideal > 127) ideal = 127;
                    if (ideal < -127) ideal = -127;
                    if (d < 0) d = -d;
                    if (d > maxlsb[v]) maxlsb[v] = d;
                    sum2 += (double)d * d; n++;
                    hq[i] = (float)got * (float)s_mid;
                    dmid += (double)hq[i] * mid[i];
                    na += (double)hq[i] * hq[i];
                    nb += (double)mid[i] * mid[i];
                }
                {
                    double c = dmid / (sqrt(na) * sqrt(nb) + 1e-30);
                    if (t == 0 || c < mid_cos[v]) mid_cos[v] = c;
                }
                matvec(Wd, HID, INT, hq, yq);
                {
                    double c = cosine(yq, yref_all + (size_t)t * HID, HID);
                    if (t == 0 || c < out_worst[v]) out_worst[v] = c;
                    out_mean[v] += c;
                }
            }
            out_mean[v] /= ntok;
            lsb_rms[v] = sqrt(sum2 / (double)n);
        }
        printf("\n  %-20s %10s %10s %10s %10s\n", "variant", "max|LSB|", "rms LSB", "cos(mid)", "cos(out)");
        for (v = 0; v < 5; v++)
            printf("  %-20s %10d %10.3f %10.6f %10.6f  (mean %.6f)\n",
                   nm[v], maxlsb[v], lsb_rms[v], mid_cos[v], out_worst[v], out_mean[v]);

        {
            int zero = 0, n = 0;
            for (t = 0; t < ntok; t++) for (i = 0; i < INT; i++) {
                int32_t a = ((int32_t)llround(mid_all[(size_t)t * INT + i] * 65536.0)) >> 12;
                if (a < 256) zero++;
                n++;
            }
            printf("\n  literal path: %d/%d lanes (%.1f%%) have pr>>12 < 256 => "
                   "mulq16 zeroes them\n", zero, n, 100.0 * zero / n);
        }

        printf("\n  s_mid sweep, cos(out) worst token (a=mulq16, b=int64 on pr>>12, e=full precision):\n");
        printf("  %10s %9s %9s %9s\n", "s_mid", "a", "b", "e");
        {
            double mult[] = { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0 };
            int mi;
            for (mi = 0; mi < 8; mi++) {
                double S = s_mid * mult[mi];
                int32_t m2 = (int32_t)(4096.0f / (float)S);
                double cc[3] = { 2.0, 2.0, 2.0 };
                int vv;
                for (vv = 0; vv < 3; vv++) {
                    int vidx = (vv == 0) ? 0 : (vv == 1 ? 1 : 4);
                    for (t = 0; t < ntok; t++) {
                        float hq[INT];
                        float *mid = mid_all + (size_t)t * INT;
                        for (i = 0; i < INT; i++)
                            hq[i] = (float)requant(mid[i], m2, S, vidx) * (float)S;
                        matvec(Wd, HID, INT, hq, yq);
                        {
                            double c = cosine(yq, yref_all + (size_t)t * HID, HID);
                            if (c < cc[vv]) cc[vv] = c;
                        }
                    }
                }
                printf("  %10.4f %9.6f %9.6f %9.6f%s\n", S, cc[0], cc[1], cc[2],
                       (fabs(S - s_mid) < 1e-9) ? "  <- measured" : "");
            }
        }
    }

    /* ---- activation magnitude sensitivity ------------------------------
     * Layer 0 sees only the token embedding (residual rms 0.023); deeper
     * layers see a grown residual stream.  Scale gate/up by k (so P grows
     * roughly as k^2) and re-derive s_mid per k, to check whether the requant
     * verdict is a layer-0 artefact. */
    {
        int ks[] = { 1, 2, 4, 8, 16 };
        int ki;
        int nk = ntok < 8 ? ntok : 8;
        printf("\n  activation scale sensitivity (gate,up * k; s_mid re-derived):\n");
        printf("  %3s %10s %10s %9s %9s %9s %9s\n",
               "k", "P absmax", "s_mid", "a", "b", "e", "d");
        for (ki = 0; ki < 5; ki++) {
            int k = ks[ki], v;
            double pmx = 0.0;
            for (t = 0; t < nk; t++) for (i = 0; i < INT; i++) {
                double g = (double)k * gate_all[(size_t)t * INT + i];
                double u = (double)k * up_all[(size_t)t * INT + i];
                double p = fabs((g / (1.0 + exp(-g))) * u);
                if (p > pmx) pmx = p;
            }
            {
                double S = pmx / 127.0;
                int32_t m2 = (int32_t)(4096.0f / (float)S);
                double cc[5];
                for (v = 0; v < 5; v++) cc[v] = 2.0;
                for (t = 0; t < nk; t++) {
                    float hq[INT], yref_k[HID];
                    float *mid = mid_all + (size_t)t * INT;
                    /* rebuild mid and its own f32 reference for this k */
                    if (k != 1) {
                        for (i = 0; i < INT; i++) {
                            double g = (double)k * gate_all[(size_t)t * INT + i];
                            double u = (double)k * up_all[(size_t)t * INT + i];
                            mid[i] = (float)((g / (1.0 + exp(-g))) * u);
                        }
                        matvec(Wd, HID, INT, mid, yref_k);
                    } else {
                        memcpy(yref_k, yref_all + (size_t)t * HID, (size_t)HID * 4);
                    }
                    for (v = 0; v < 5; v++) {
                        double s2 = 0.0, na = 0.0, nb = 0.0, dd = 0.0;
                        for (i = 0; i < INT; i++) {
                            int got = requant(mid[i], m2, S, v);
                            int ideal = (int)lround(mid[i] / S);
                            int d;
                            if (ideal > 127) ideal = 127;
                            if (ideal < -127) ideal = -127;
                            d = got - ideal;
                            s2 += (double)d * d;
                            hq[i] = (float)got * (float)S;
                            dd += (double)hq[i] * mid[i];
                            na += (double)hq[i] * hq[i];
                            nb += (double)mid[i] * mid[i];
                        }
                        (void)s2;
                        matvec(Wd, HID, INT, hq, yq);
                        {
                            double c = cosine(yq, yref_k, HID);
                            (void)dd; (void)na; (void)nb;
                            if (c < cc[v]) cc[v] = c;
                        }
                    }
                }
                printf("  %3d %10.3f %10.5f %9.6f %9.6f %9.6f %9.6f\n",
                       k, pmx, S, cc[0], cc[1], cc[4], cc[3]);
            }
        }
        printf("\n  (a=mulq16 as is, b=int64 on pr>>12, e=int64 on Q16 pr,"
               " d=ideal; cos(out), worst token)\n");
    }

    (void)h;
    return 0;
}
