/*
 * hmx_part_probe3 — the two remaining unknowns for the REPORT24 design:
 *   Q9  does `weight.ubit` decode the 4-bit field as unsigned 0..15
 *       (key: then a per-block partial is non-negative and needs no bias)?
 *   Q10 exactly how many bytes does `.uh = acc:2x1` write, and where
 *       (poison-fill the destination).
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
MK(mul_ubit, ubit)
MK(mul_sbit, sbit)
MK(mul_b, b)
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

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w4 = p + OFF_W4;
    uint8_t *wb = p + OFF_W4 + 2048;
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int v, i, s, n;

    set_xe2();
    memset(p, 0, 0x20000);
    memset(act, 1, 4096);
    memset(bias, 0, 512);
    for (i = 0; i < 32; i++) ((uint32_t *)bias)[i] = 0x4000u;   /* .uh mult 1.0 */

    printf("Q9  4-bit decode: act=1 over 32 K, all nibbles = v, .uh mult 1.0\n");
    printf("    v :    weight.n   weight.ubit   weight.sbit\n");
    for (v = 0; v < 16; v++) {
        int16_t rn, ru, rs;
        memset(w4, (v & 0xF) | ((v & 0xF) << 4), 512);
        memset(out, 0, 16384); hmx_clr();
        mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync(); rn = ((const int16_t *)out)[0];
        memset(out, 0, 16384); hmx_clr();
        mul_ubit(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync(); ru = ((const int16_t *)out)[0];
        memset(out, 0, 16384); hmx_clr();
        mul_sbit(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync(); rs = ((const int16_t *)out)[0];
        printf("   %2d : %10d %13d %13d\n", v, rn, ru, rs);
    }

    printf("\nQ10 store byte count (dest poisoned 0xA5)\n");
    memset(w4, 0x11, 512);
    memset(out, 0xA5, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    {
        int changed = 0, first = -1, last = -1;
        for (i = 0; i < 16384; i++)
            if (out[i] != 0xA5) { changed++; if (first < 0) first = i; last = i; }
        printf("   .uh : changed %d bytes, first=%d last=%d\n", changed, first, last);
    }
    memset(out, 0xA5, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    {
        int changed = 0, first = -1, last = -1;
        for (i = 0; i < 16384; i++)
            if (out[i] != 0xA5) { changed++; if (first < 0) first = i; last = i; }
        printf("   .ub : changed %d bytes, first=%d last=%d\n", changed, first, last);
    }

    printf("\nQ11 .uh word map: act(s,0)=s+1, q[k==0]=1, mult 1.0\n");
    memset(act, 0, 2048);
    for (s = 0; s < 64; s++) act[128 * (s >> 2) + (s & 3)] = (uint8_t)(s + 1);
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++)
        for (s = 0; s < 4; s++) w4[4 * n + s] |= 1u;
    memset(out, 0, 16384);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    {
        uint32_t *w32 = (uint32_t *)out;
        const int16_t *h16 = (const int16_t *)out;
        printf("   words[0..3] = %08x %08x %08x %08x\n",
               w32[0], w32[1], w32[2], w32[3]);
        printf("   words[31] = %08x  words[32] = %08x  words[64] = %08x\n",
               w32[31], w32[32], w32[64]);
        /* find (s,n) placement: value s+1 in either half of each word */
        for (i = 0; i < 6; i++)
            printf("   word%02d: lo=%d hi=%d\n", i, h16[2 * i], h16[2 * i + 1]);
        printf("   word31: lo=%d hi=%d ; word32: lo=%d hi=%d\n",
               h16[62], h16[63], h16[64], h16[65]);
    }
    return 0;
}
