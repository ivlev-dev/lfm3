/*
 * probe_round.c - how does the int8 store convert acc*scale?
 * acc = 32 (act=1, w=1, K=32), in_bias shifts it to any small integer.
 * Prints sat.ub(acc) for values that land exactly on .5 at the given scale.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VTCM_BASE 0xd8400000ul
#define OFF_ACT 0x0000u
#define OFF_W8  0x2000u
#define OFF_OUT 0x4000u
#define OFF_BIAS 0x6000u

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}
static inline void clr(void) { __asm__ __volatile__("mxclracc" ::: "memory"); }
static inline void mul(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
           "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}
static inline void bload(const void *b)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
        :: "r"((uint32_t)(uintptr_t)b) : "memory");
}
static inline void st_ub(void *o)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
        :: "r"((uint32_t)(uintptr_t)o), "r"(0u) : "memory");
}
static inline void sync_(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w = p + OFF_W8, *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    static const uint32_t scales[] = { 0x6000u, 0x5800u, 0x5000u, 0x4800u };
    static const char *names[] = { "1.0", "0.25", "0.0625", "0.015625" };
    int si, ib;
    set_xe2();
    memset(act, 1, 2048);
    memset(w, 1, 1024);
    memset(bias, 0, 256);
    printf("acc = 32 + in_bias, sat.ub, different scales (no saturation):\n");
    for (si = 0; si < 4; si++) {
        printf("scale=%s (0x%04lx):\n", names[si], (unsigned long)scales[si]);
        for (ib = -32; ib <= 8; ib++) {
            uint32_t *bl = (uint32_t *)bias;
            int i;
            for (i = 0; i < 32; i++) bl[i] = scales[si];
            for (i = 0; i < 32; i++) bl[32 + i] = (uint32_t)(int32_t)ib;
            memset(out, 0, 2048);
            clr(); mul(act, 0x3FFu, w, 1023u); bload(bias); st_ub(out); sync_();
            if (out[0] <= 255) {
                int acc = 32 + ib;
                printf("  acc=%3d -> %3u   (exact %.6f)\n", acc, out[0],
                       (double)acc * (si == 0 ? 1.0 : si == 1 ? 0.25 : si == 2 ? 0.0625 : 0.015625));
            }
        }
    }
    return 0;
}
