/*
 * hmx_u16map2 — is there a store form that writes the *other* 32 spatial rows
 * in 16 bits?  `.uh:after:sat.uh = acc:2x1` covers only the odd rows (REPORT24).
 * Try the :before / no-2x1 variants.  One activation row is lit at a time and
 * the stored value 17 marks where it lands.
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
static inline void mul_n(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}
#define ST(NAME, ASM)                                                        \
static inline void NAME(void *o, uint32_t rt)                                \
{                                                                            \
    __asm__ __volatile__(ASM :: "r"((uint32_t)(uintptr_t)o), "r"(rt)         \
                         : "memory");                                        \
}
ST(s_a_uh,  "{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}")
ST(s_b_uh,  "{\n\t mxmem(%0,%1):before:sat.uh = acc:2x1\n\t}")
ST(s_a_uh1, "{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}")  /* dup */
ST(s_b_uh1, "{\n\t mxmem(%0,%1):before:sat.uh = acc:2x1\n\t}")
ST(s_a_ub,  "{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}")

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w4 = p + OFF_W4;
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int i, s, n;
    static void (*fn[5])(void *, uint32_t) = { s_a_uh, s_b_uh, s_b_uh1, s_a_ub, s_a_ub };
    static const char *nm[5] = { "after:uh:2x1", "before:uh:2x1",
                                 "before:uh:2x1 (2)", "after:ub", "after:ub (2)" };

    set_xe2();
    memset(p, 0, 0x20000);
    memset(bias, 0, 512);
    for (i = 0; i < 32; i++) ((uint32_t *)bias)[i] = 0x4000u;   /* mult 1.0 */
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++) w4[4 * n] = 0x01;                  /* k = 0, all n */

    for (int f = 0; f < 5; f++) {
        printf("\n%s:\n", nm[f]);
        for (s = 0; s < 8; s++) {
            int nb = -1, cnt = 0;
            memset(act, 0, 2048);
            act[128 * (s >> 2) + 0 + (s & 3)] = 17;
            memset(out, 0, 8192);
            hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias);
            fn[f](out, 0);
            hmx_sync();
            for (i = 0; i < 8192; i++) if (out[i] == 17) { cnt++; if (nb < 0) nb = i; }
            printf("   s=%d -> first=%d count=%d\n", s, nb, cnt);
        }
        /* total bytes touched, all rows lit */
        memset(act, 0, 2048);
        for (s = 0; s < 64; s++) act[128 * (s >> 2) + (s & 3)] = 17;
        memset(out, 0xA5, 16384);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias);
        fn[f](out, 0);
        hmx_sync();
        {
            int ch = 0, first = -1, last = -1;
            for (i = 0; i < 16384; i++)
                if (out[i] != 0xA5) { ch++; if (first < 0) first = i; last = i; }
            printf("   all rows lit: changed=%d first=%d last=%d\n", ch, first, last);
        }
    }
    return 0;
}
