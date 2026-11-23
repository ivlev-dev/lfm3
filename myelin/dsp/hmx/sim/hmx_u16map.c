/*
 * hmx_u16map — exact `.uh = acc:2x1` output map.
 *
 * One activation row s is lit at a time: act(s,0) = 17, everything else 0,
 * weight nibble 1 only at k==0, bias word 0x4000 (multiplier 1.0).
 * The stored value 17 therefore marks (s = lit row, n = any).
 * We print, for every lit row, the byte offset of the first nonzero int16.
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
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int i, s, n;

    set_xe2();
    memset(p, 0, 0x20000);
    memset(bias, 0, 512);
    for (i = 0; i < 32; i++) ((uint32_t *)bias)[i] = 0x4000u;   /* mult 1.0 */
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++) w4[4 * n] = 0x01;                  /* k=0 low nibble 1, all n */

    printf("u16 map: for each lit activation row, byte offset of the value 17\n");
    printf("row :  bytes (first 4 offsets)\n");
    for (s = 0; s < 64; s++) {
        int found[4] = { -1, -1, -1, -1 }, nf = 0;
        memset(act, 0, 2048);
        act[128 * (s >> 2) + 0 + (s & 3)] = 17;
        memset(out, 0, 8192);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync();
        for (i = 0; i < 2048 && nf < 4; i += 2) {
            int16_t v;
            memcpy(&v, out + i, 2);
            if (v == 17) found[nf++] = i;
        }
        printf("%3d : %5d %5d %5d %5d  (count%c)\n", s,
               found[0], found[1], found[2], found[3], nf > 3 ? '+' : ' ');
    }
    /* the same through .ub for comparison */
    printf("\nsame through .ub (byte offset of 17):\n");
    for (s = 0; s < 8; s++) {
        int found = -1;
        memset(act, 0, 2048);
        act[128 * (s >> 2) + 0 + (s & 3)] = 17;
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        for (i = 0; i < 2048; i++) if (out[i] == 17) { found = i; break; }
        printf("  s=%d -> %d\n", s, found);
    }
    return 0;
}
