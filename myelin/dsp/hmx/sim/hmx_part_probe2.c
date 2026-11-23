/*
 * hmx_part_probe2 — signed partial-sum store for REPORT24 block scales.
 *
 * Established so far (hexagon-sim v68): the HMX int4 recipe computes
 * 32 K x 32 N x 64 spatial; `.ub` writes 2 KB unsigned 8-bit, `.uh` writes
 * 8 KB (two 4 KB banks) unsigned 16-bit; the bias word scale is fixed point
 * (`.ub`: fp16(word)/512, `.uh`: fp16(word)/2) and the out_bias half is not
 * applied on the paths tried.
 *
 * Q6 `.hf` (fp16) store: multiplier, sign, byte count, bank structure
 * Q7 bias words 32..63 / 96..127: is out_bias there?
 * Q8 `weight.sc` / `weight.sm` with a controlled block: native per-block scale?
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define VTCM_BASE 0xd8400000ul
#define OFF_ACT   0x00000u
#define OFF_W4    0x04000u
#define OFF_OUT   0x08000u
#define OFF_BIAS  0x10000u

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}
static inline void hmx_clr(void) { __asm__ __volatile__("mxclracc" ::: "memory"); }
static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}
static inline void hmx_bias2(const void *p)
{
    __asm__ __volatile__("{ bias = mxmem2(%0) }"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}
#define MK(NAME, WT)                                                         \
static inline void NAME(const void *a, uint32_t ac, const void *w, uint32_t wc) \
{                                                                            \
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t"            \
                         "weight." #WT " = mxmem(%2,%3)\n\t}"               \
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),            \
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");\
}
MK(mul_n, n)
MK(mul_sc, sc)
MK(mul_sm, sm)
static inline void st_hf(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}
static inline void st_uh(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}
static inline void st_ub(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}

/* bias block: 512 B, four 32-word groups */
static void bias_grp(uint8_t *b, int grp, uint32_t word)
{
    int i;
    memset(b, 0, 512);
    for (i = 0; i < 32; i++) ((uint32_t *)b)[grp * 32 + i] = word;
}

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w4 = p + OFF_W4;
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int i, s, n;

    set_xe2();
    memset(p, 0, 0x20000);

    /* ---------- Q6: .hf store ---------- */
    printf("Q6  .hf store (act=1, q=1, 32 K -> acc=32; word scan)\n");
    memset(act, 1, 4096);
    memset(w4, 0x11, 512);
    {
        static const uint32_t w[] = { 0x3C00u, 0x6000u, 0x4000u, 0x4400u };
        for (i = 0; i < 4; i++) {
            const __fp16 *f;
            int nz = 0, last = -1;
            bias_grp(bias, 0, w[i]);
            memset(out, 0, 16384);
            hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_hf(out, 0);
            hmx_sync();
            f = (const __fp16 *)out;
            for (n = 0; n < 8192; n++) if (f[n] != 0) { nz++; last = n; }
            printf("    word=0x%04x -> f[0]=%.4f f[1]=%.4f  nz=%d last=%d (bytes=%d)\n",
                   w[i], (double)f[0], (double)f[1], nz, last, last < 0 ? 0 : (last + 1) * 2);
        }
    }
    /* signed accumulator through .hf: q = -1 */
    printf("    signed: q=-1 (nibble 0xF), acc=-32, word 0x6000\n");
    memset(w4, 0xFF, 512);
    bias_grp(bias, 0, 0x6000u);
    memset(out, 0, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_hf(out, 0);
    hmx_sync();
    printf("    .hf f[0]=%.4f  (negative => signed fp16 store works)\n",
           (double)((const __fp16 *)out)[0]);
    printf("    .uh  ");
    memset(out, 0, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    printf("h[0]=%d (unsigned 16-bit clamps)\n", ((const int16_t *)out)[0]);

    /* ---------- Q7: out_bias in groups 1 and 3 ---------- */
    printf("\nQ7  out_bias location: word=(100<<16)|scale in group g\n");
    memset(w4, 0x11, 512);
    for (i = 0; i < 4; i++) {
        uint32_t w = (100u << 16) | 0x6000u;
        bias_grp(bias, i, w);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    .ub group %d -> out[0]=%d\n", i, out[0]);
        memset(out, 0, 16384);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_hf(out, 0);
        hmx_sync();
        printf("    .hf group %d -> f[0]=%.3f\n", i, (double)((const __fp16 *)out)[0]);
    }

    /* ---------- Q8: weight.sc / weight.sm ---------- */
    printf("\nQ8  weight.sc / weight.sm with a controlled 512 B block\n");
    bias_grp(bias, 0, 0x4000u);          /* .uh multiplier 1.0 */
    memset(act, 1, 2048);
    {
        uint8_t *wb = p + OFF_W4;
        int pat;
        for (pat = 0; pat < 4; pat++) {
            int16_t rn, rsc, rsm;
            memset(wb, 0, 1024);
            switch (pat) {
            case 0: memset(wb, 0x11, 512); break;               /* all nibbles 1 */
            case 1: memset(wb, 0x00, 512); wb[0] = 0x01; break;  /* one nibble */
            case 2: memset(wb, 0x00, 512); wb[0] = 0xF0; break;  /* high nibble */
            case 3: memset(wb, 0x00, 512); wb[1] = 0x01; break;  /* next byte */
            }
            memset(out, 0, 16384); hmx_clr();
            mul_n(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
            hmx_sync(); rn = ((const int16_t *)out)[0];
            memset(out, 0, 16384); hmx_clr();
            mul_sc(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
            hmx_sync(); rsc = ((const int16_t *)out)[0];
            memset(out, 0, 16384); hmx_clr();
            mul_sm(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
            hmx_sync(); rsm = ((const int16_t *)out)[0];
            printf("    pat%d: weight.n -> %d   weight.sc -> %d   weight.sm -> %d\n",
                   pat, rn, rsc, rsm);
        }
    }
    /* how many bytes of the block does weight.sc consume? */
    printf("    weight.sc with 0x11 at [0..511] but 0x22 at [512..1023]:\n");
    {
        uint8_t *wb = p + OFF_W4;
        int16_t a, b;
        memset(wb, 0x11, 512); memset(wb + 512, 0x22, 512);
        memset(out, 0, 16384); hmx_clr();
        mul_sc(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync(); a = ((const int16_t *)out)[0];
        memset(wb, 0x22, 512);
        memset(out, 0, 16384); hmx_clr();
        mul_sc(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync(); b = ((const int16_t *)out)[0];
        printf("      neighbour 0x22 -> %d ; whole block 0x22 -> %d\n", a, b);
    }
    /* spatial map for .hf, to place the combine reads */
    printf("\n.hf spatial/byte map (act(s,0)=s+1, q=1 at k==0, word 0x4000 x1)\n");
    memset(act, 0, 2048);
    for (s = 0; s < 64; s++) act[128 * (s >> 2) + (s & 3)] = (uint8_t)(s + 1);
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++)
        for (s = 0; s < 4; s++) w4[4 * n + s] |= 1u;
    bias_grp(bias, 0, 0x4000u);
    memset(out, 0, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_hf(out, 0);
    hmx_sync();
    {
        const __fp16 *f = (const __fp16 *)out;
        int nz = 0, last = -1;
        for (n = 0; n < 8192; n++) if (f[n] != 0) { nz++; last = n; }
        printf("   f[0..7]: ");
        for (i = 0; i < 8; i++) printf("%.0f ", (double)f[i]);
        printf("\n   f[32..35]: %.0f %.0f %.0f %.0f ; f[64]=%.0f f[128]=%.0f\n",
               (double)f[32], (double)f[33], (double)f[34], (double)f[35],
               (double)f[64], (double)f[128]);
        printf("   nz=%d last=%d -> %d bytes\n", nz, last, (last + 1) * 2);
    }
    return 0;
}
