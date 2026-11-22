#ifndef GGUF_F32_H
#define GGUF_F32_H
/*
 * Minimal read-only GGUF access for the offline studies: tensor directory,
 * Q4_K/Q6_K dequantisation to f32, row-at-a-time loading.
 *
 * Extracted from silu_err_study.c (same code, same semantics).  The studies
 * only read weights; nothing here writes the model.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static FILE *gf;
static int64_t g_data;
static int g_align = 32;

typedef struct { char name[64]; uint32_t nd; uint64_t dim[4]; uint32_t type; uint64_t off; } tinfo_t;
static tinfo_t g_t[400];
static int g_nt;

static void gguf_grd(void *p, size_t n)
{
    if (fread(p, 1, n, gf) != n) { fprintf(stderr, "gguf: short read\n"); exit(1); }
}
static uint32_t gguf_u32(void) { uint32_t v; gguf_grd(&v, 4); return v; }
static uint64_t gguf_u64(void) { uint64_t v; gguf_grd(&v, 8); return v; }
static char *gguf_str(void)
{
    uint64_t n = gguf_u64();
    char *s = malloc(n + 1);
    gguf_grd(s, n); s[n] = 0;
    return s;
}
static void gguf_skip(uint32_t t, uint32_t *out_u32)
{
    switch (t) {
    case 0: case 1: case 7: { uint8_t v; gguf_grd(&v, 1); break; }
    case 2: case 3: { uint16_t v; gguf_grd(&v, 2); break; }
    case 4: { uint32_t v; gguf_grd(&v, 4); if (out_u32) *out_u32 = v; break; }
    case 5: case 6: { uint32_t v; gguf_grd(&v, 4); break; }
    case 10: case 11: case 12: { uint64_t v; gguf_grd(&v, 8); break; }
    case 8: { char *s = gguf_str(); free(s); break; }
    case 9: {
        uint32_t et = gguf_u32(); uint64_t n = gguf_u64(), i;
        for (i = 0; i < n; i++) gguf_skip(et, NULL);
        break;
    }
    default: fprintf(stderr, "gguf: kv type %u\n", t); exit(1);
    }
}

static const tinfo_t *gguf_find(const char *name)
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
    gguf_grd(magic, 4);
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); exit(1); }
    gguf_u32();                                   /* version */
    g_nt = (int)gguf_u64();
    if (g_nt > 400) { fprintf(stderr, "gguf: too many tensors\n"); exit(1); }
    nkv = gguf_u64();
    for (i = 0; i < (int)nkv; i++) {
        char *k = gguf_str();
        uint32_t t = gguf_u32();
        if (!strcmp(k, "general.alignment")) gguf_skip(t, (uint32_t *)&g_align);
        else gguf_skip(t, NULL);
        free(k);
    }
    for (i = 0; i < g_nt; i++) {
        char *n = gguf_str();
        uint32_t d;
        snprintf(g_t[i].name, sizeof(g_t[i].name), "%s", n);
        free(n);
        g_t[i].nd = gguf_u32();
        for (d = 0; d < g_t[i].nd && d < 4; d++) g_t[i].dim[d] = gguf_u64();
        g_t[i].type = gguf_u32();
        g_t[i].off = gguf_u64();
    }
    g_data = (int64_t)((ftell(gf) + g_align - 1) / g_align * g_align);
}

static int64_t gguf_bytes(const tinfo_t *t)
{
    int64_t cnt = 1; uint32_t i;
    for (i = 0; i < t->nd; i++) cnt *= (int64_t)t->dim[i];
    switch (t->type) {
    case 0:  return cnt * 4;                  /* F32  */
    case 1:  return cnt * 2;                  /* F16  */
    case 12: return cnt / 256 * 144;          /* Q4_K */
    case 14: return cnt / 256 * 210;          /* Q6_K */
    default: fprintf(stderr, "type %u unsupported\n", t->type); exit(1);
    }
}

static float gguf_f16(uint16_t h)
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

static void gguf_deq_q4k(const uint8_t *b, float *y)
{
    float d = gguf_f16((uint16_t)(b[0] | (b[1] << 8)));
    float dmin = gguf_f16((uint16_t)(b[2] | (b[3] << 8)));
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

static void gguf_deq_q6k(const uint8_t *b, float *y)
{
    float d = gguf_f16((uint16_t)(b[208] | (b[209] << 8)));
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

/* dequantise one row of n values (n % 256 == 0) at an absolute file offset */
static void gguf_row_f32(int64_t off, int n, int type, float *out)
{
    int nb, i, bs;
    size_t rb;
    uint8_t *buf;

    if (type == 0) {                       /* F32 row */
        fseek(gf, (long)off, SEEK_SET);
        gguf_grd(out, (size_t)n * 4);
        return;
    }
    if (type == 1) {                       /* F16 row */
        uint16_t *h = malloc((size_t)n * 2);
        fseek(gf, (long)off, SEEK_SET);
        gguf_grd(h, (size_t)n * 2);
        for (i = 0; i < n; i++) out[i] = gguf_f16(h[i]);
        free(h);
        return;
    }
    nb = n / 256;
    bs = (type == 12) ? 144 : 210;
    rb = (size_t)nb * bs;
    buf = malloc(rb);
    fseek(gf, (long)off, SEEK_SET);
    gguf_grd(buf, rb);
    for (i = 0; i < nb; i++) {
        if (type == 12) gguf_deq_q4k(buf + (size_t)i * 144, out + i * 256);
        else            gguf_deq_q6k(buf + (size_t)i * 210, out + i * 256);
    }
    free(buf);
}

/* whole [m][k] tensor, row-major, dequantised to f32 */
static float *gguf_matrix(const tinfo_t *t, int m, int k)
{
    float *w = malloc((size_t)m * k * sizeof(float));
    int64_t rb = gguf_bytes(t) / m;
    int r;
    if (!w) { fprintf(stderr, "oom\n"); exit(1); }
    for (r = 0; r < m; r++)
        gguf_row_f32(g_data + (int64_t)t->off + (int64_t)r * rb, k, (int)t->type,
                     w + (size_t)r * k);
    return w;
}

/* raw f32 tensor (e.g. a norm weight) */
static void gguf_vec_f32(const tinfo_t *t, float *out, int n)
{
    fseek(gf, (long)(g_data + (int64_t)t->off), SEEK_SET);
    gguf_grd(out, (size_t)n * 4);
}

/* allocate + load a raw f32 vector tensor by name */
static float *gguf_vec_alloc(const char *name, int n)
{
    const tinfo_t *t = gguf_find(name);
    float *v = malloc((size_t)n * 4);
    if (!v) { fprintf(stderr, "oom\n"); exit(1); }
    gguf_vec_f32(t, v, n);
    return v;
}

static void f32_matvec(const float *w, int m, int k, const float *x, float *out)
{
    int r, i;
    for (r = 0; r < m; r++) {
        const float *wr = w + (size_t)r * k;
        double s = 0.0;
        for (i = 0; i < k; i++) s += (double)wr[i] * x[i];
        out[r] = (float)s;
    }
}

static void f32_rmsnorm(const float *v, const float *w, int n, float eps, float *out)
{
    double s = 0.0; int i; float inv;
    for (i = 0; i < n; i++) s += (double)v[i] * v[i];
    inv = (float)(1.0 / sqrt(s / n + eps));
    for (i = 0; i < n; i++) out[i] = v[i] * inv * w[i];
}

static double f32_cosine(const float *a, const float *b, int n)
{
    double d = 0.0, na = 0.0, nb = 0.0; int i;
    for (i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return d / (sqrt(na) * sqrt(nb) + 1e-30);
}

__attribute__((unused)) static int f32_cmp(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

#endif /* GGUF_F32_H */
