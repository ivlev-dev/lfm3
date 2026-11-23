/*
 * hmxtest.c - HMX int4/32 GEMM with uint8 (zero-point) activations.
 *
 * hexagon-sim only (HMX + VTCM at 0xd8400000, no QuRT).  Reads the case file
 * produced by host/mkcase.c and runs one 32x32 int4 crouton per (group, tile).
 *
 * v68 store reality (measured, probe_bias / probe_round):
 *   - `mxmem(..):after:sat.ub = acc` writes exactly one u8 per cell, sat 0..255;
 *   - the "16-bit" `:after:sat.uh = acc:2x1` form also saturates at 255 in
 *     hexagon-sim (values >255 come back 0xFFFF), i.e. it is not a 16-bit
 *     readout on v68; `cvt.uh = acc` is arch >= 73 only.
 * So the per-group integer dot is observed through a power-of-two scale:
 *
 *   out_u8 = trunc((dot + tbias[n][g]) * 2^-tshift[n][g])
 *
 * with a per-(row,group) power-of-two calibration chosen to use 0..255 of the
 * observed range (host/mkcase.c).  The scale is an exact fp16 power of two,
 * so the conversion is exact truncation with no saturation.  `in_bias` in the
 * bias block carries
 *     tbias[n][g] - 128*sum_k(w[n][k])
 * so the accumulator value before the scale is the signed dot plus tbias.
 *
 * Two dumps are produced per 32-row tile, one byte per cell, order
 * [group][row m][channel c]:
 *   fixed: u = q + 128, in_bias = B - 128*rowsum   (the fix)
 *   raw:   u = (uint8)q, in_bias = B               (the old sign bug)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VTCM_BASE 0xd8400000ul
#define MAGIC "LFMQ25"

#define OFF_ACT_F 0x00000u   /* 64 groups * 2 KB = 128 KB */
#define OFF_ACT_R 0x40000u
#define OFF_W     0x80000u   /* 32 KB weight strip */
#define OFF_OUT   0x88000u   /* 2 KB */
#define OFF_BIAS  0x89000u   /* 256 B */

/* |dot| <= 32 * 127 * 7 = 28448: B makes it non-negative, >>8 makes it u8 */
/* fp16 value 512*2^-sh -> word (24-sh)<<10 (sh <= 8) */
#define LFM_HMX_SCALE(sh) ((uint32_t)(24u - (unsigned)(sh)) << 10)

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}
static inline void hmx_clr(void) { __asm__ __volatile__("mxclracc" ::: "memory"); }
static inline void hmx_mul(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
           "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}
static inline void hmx_bias(const void *b)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
        :: "r"((uint32_t)(uintptr_t)b) : "memory");
}
static inline void hmx_st_ub(void *o)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
        :: "r"((uint32_t)(uintptr_t)o), "r"(0u) : "memory");
}
static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

static uint8_t *g_actf, *g_actr, *g_w, *g_out, *g_bias;

/* one group of activations into a 2 KB HMX slot: u8 value, row m -> s = m */
static void pack_act(uint8_t *dst, const int8_t *qa, int M, int K, int g, int zp)
{
    int m, l;
    memset(dst, 0, 2048);
    for (m = 0; m < M; m++)
        for (l = 0; l < 32; l++)
            dst[128u * (unsigned)(m >> 2) + 4u * (unsigned)l + (unsigned)(m & 3)] =
                (uint8_t)((int)qa[(size_t)m * K + g * 32 + l] + zp);
}

static int32_t group_rowsum(const uint8_t *tile, int n)
{
    int32_t s = 0;
    int kk;
    for (kk = 0; kk < 32; kk++) {
        unsigned v = tile[128u * (unsigned)(kk >> 3) + 4u * (unsigned)n
                         + (unsigned)(kk & 3)];
        int nib = ((kk >> 2) & 1) ? (int)(v >> 4) : (int)(v & 0xF);
        s += (nib & 8) ? nib - 16 : nib;
    }
    return s;
}

struct case_hdr { uint32_t ver, M, N, K, rot; };

int main(int argc, char **argv)
{
    const char *casef, *fixedf, *rawf;
    FILE *f, *ff, *fr;
    char magic[8];
    struct case_hdr h;
    int32_t M, N, K, G, tiles, rot;
    size_t off_qa, off_blob;
    int8_t *qa;
    uint8_t *blob, *dump, *tshift;
    int32_t *tbias;
    long filebytes;
    int t, g, m, c;

    if (argc < 4) {
        fprintf(stderr, "hmxtest <case.bin> <fixed.bin> <raw.bin>\n");
        return 2;
    }
    casef = argv[1]; fixedf = argv[2]; rawf = argv[3];

    f = fopen(casef, "rb");
    if (!f) { perror(casef); return 1; }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MAGIC, 8)) {
        printf("bad magic\n"); return 1;
    }
    if (fread(&h, sizeof(h), 1, f) != 1) return 1;
    M = (int32_t)h.M; N = (int32_t)h.N; K = (int32_t)h.K; rot = (int32_t)h.rot;
    G = K / 32; tiles = N / 32;
    off_qa = 8 + sizeof(h);
    {
        size_t off_sa = off_qa + (size_t)M * K;
        size_t off_sw = off_sa + (size_t)(M * K / 32) * 4;
        size_t off_x  = off_sw + (size_t)(N * K / 32) * 4;
        off_blob = off_x + (size_t)M * K * 4 + (size_t)(N * K / 32) * 4
                 + (size_t)(N * K / 32);
        tbias = malloc((size_t)(N * K / 32) * 4);
        tshift = malloc((size_t)(N * K / 32));
        if (!tbias || !tshift) { printf("oom calib\n"); return 1; }
        fseek(f, (long)(off_x + (size_t)M * K * 4), SEEK_SET);
        if (fread(tbias, 4, (size_t)(N * K / 32), f) != (size_t)(N * K / 32)) return 1;
        if (fread(tshift, 1, (size_t)(N * K / 32), f) != (size_t)(N * K / 32)) return 1;
    }
    fseek(f, 0, SEEK_END);
    filebytes = ftell(f);
    printf("case M=%d N=%d K=%d rot=%d tiles=%d groups=%d file=%ld B\n",
           (int)M, (int)N, (int)K, (int)rot, (int)tiles, (int)G, filebytes);
    qa = malloc((size_t)M * K);
    blob = malloc((size_t)(N / 32) * (K / 32) * 512u);
    dump = malloc((size_t)G * 32 * 32);
    if (!qa || !blob || !dump) { printf("oom\n"); return 1; }
    fseek(f, (long)off_qa, SEEK_SET);
    if (fread(qa, 1, (size_t)M * K, f) != (size_t)M * K) { printf("short qa\n"); return 1; }
    fseek(f, (long)off_blob, SEEK_SET);
    if (fread(blob, 1, (size_t)(N / 32) * (K / 32) * 512u, f)
        != (size_t)(N / 32) * (K / 32) * 512u) { printf("short blob\n"); return 1; }
    fclose(f);

    set_xe2();
    g_actf = (uint8_t *)(uintptr_t)(VTCM_BASE + OFF_ACT_F);
    g_actr = (uint8_t *)(uintptr_t)(VTCM_BASE + OFF_ACT_R);
    g_w    = (uint8_t *)(uintptr_t)(VTCM_BASE + OFF_W);
    g_out  = (uint8_t *)(uintptr_t)(VTCM_BASE + OFF_OUT);
    g_bias = (uint8_t *)(uintptr_t)(VTCM_BASE + OFF_BIAS);

    for (g = 0; g < G; g++) {
        pack_act(g_actf + (size_t)g * 2048u, qa, M, K, g, 128);
        pack_act(g_actr + (size_t)g * 2048u, qa, M, K, g, 0);
    }
    printf("packed activations: fixed(q+128), raw(int8 as u8)\n");

    ff = fopen(fixedf, "wb");
    fr = fopen(rawf, "wb");
    if (!ff || !fr) { perror("dump"); return 1; }

    for (t = 0; t < tiles; t++) {
        const uint8_t *strip = blob + (size_t)t * (size_t)G * 512u;
        memcpy(g_w, strip, (size_t)G * 512u);
        __asm__ __volatile__("syncht" ::: "memory");
        __asm__ __volatile__("barrier" ::: "memory");

        for (g = 0; g < G; g++) {
            uint32_t *bl = (uint32_t *)g_bias;
            int i;
            for (i = 0; i < 32; i++) {
                int nrow = t * 32 + i, sh = tshift[(size_t)nrow * G + g];
                bl[i] = LFM_HMX_SCALE(sh);
                bl[32 + i] = (uint32_t)(int32_t)
                    (tbias[(size_t)nrow * G + g]
                     - 128 * group_rowsum(g_w + (size_t)g * 512u, i));
            }
            hmx_clr();
            hmx_mul(g_actf + (size_t)g * 2048u, 0x3FFu, g_w + (size_t)g * 512u, 511u);
            hmx_bias(g_bias);
            hmx_st_ub(g_out);
            hmx_sync();
            for (m = 0; m < 32; m++)
                for (c = 0; c < 32; c++)
                    dump[((size_t)g * 32 + m) * 32 + c] =
                        g_out[128u * (unsigned)(m >> 2) + 4u * (unsigned)c
                              + (unsigned)(m & 3)];
        }
        fwrite(dump, 1, (size_t)G * 32 * 32, ff);

        for (g = 0; g < G; g++) {
            uint32_t *bl = (uint32_t *)g_bias;
            int i;
            for (i = 0; i < 32; i++) {
                int nrow = t * 32 + i, sh = tshift[(size_t)nrow * G + g];
                bl[i] = LFM_HMX_SCALE(sh);
                bl[32 + i] = (uint32_t)(int32_t)tbias[(size_t)nrow * G + g];
            }
            hmx_clr();
            hmx_mul(g_actr + (size_t)g * 2048u, 0x3FFu, g_w + (size_t)g * 512u, 511u);
            hmx_bias(g_bias);
            hmx_st_ub(g_out);
            hmx_sync();
            for (m = 0; m < 32; m++)
                for (c = 0; c < 32; c++)
                    dump[((size_t)g * 32 + m) * 32 + c] =
                        g_out[128u * (unsigned)(m >> 2) + 4u * (unsigned)c
                              + (unsigned)(m & 3)];
        }
        fwrite(dump, 1, (size_t)G * 32 * 32, fr);
    }
    fclose(ff);
    fclose(fr);

    printf("dumps: %s %s\n", fixedf, rawf);
    printf("RESULT: OK\n");
    return 0;
}
