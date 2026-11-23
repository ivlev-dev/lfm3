/*
 * hmx_ctrl_probe — control-word / bias-scale / store-form semantics of the
 * int4 HMX recipe, hexagon-sim v68.
 *
 * The known-good baseline is sim/hmx_vtcm_probe.c (REPORT22): all-ones data,
 * bias words 0..31 = 0x6000, `mxmem(out,0):after:sat.ub = acc` -> 32.
 * This probe starts from that exact baseline and changes one thing at a time.
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
static inline void st_ub_bank(void *o, uint32_t rt)   /* bank 1 */
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc.bank1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}

static void bias_set(uint8_t *b, uint32_t word, int half /*0 ok block*/)
{
    memset(b, 0, 256);
    if (half == 0) {
        for (int i = 0; i < 32; i++) ((uint32_t *)b)[i] = word;
        for (int i = 0; i < 32; i++) ((uint32_t *)b)[64 + i] = word;
    } else {
        for (int i = 0; i < 32; i++) ((uint32_t *)b)[32 + i] = word;
        for (int i = 0; i < 32; i++) ((uint32_t *)b)[96 + i] = word;
    }
}

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w4 = p + OFF_W4;
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int16_t h[4096];
    int i, s, n;

    set_xe2();
    memset(p, 0, 0x20000);
    memset(act, 1, 4096);
    memset(w4, 0x11, 2048);

    /* ---- baseline exactly as REPORT22 ---- */
    bias_set(bias, 0x6000u, 0);
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("baseline (.ub, bias 0x6000, act_ctl 0x3FF): out[0]=%d  [want 32]\n", out[0]);

    /* ---- scale word format: one acc value, several scale words ---- */
    printf("\nscale-word scan: acc=32 (all ones, 32 K), .ub store\n");
    {
        static const uint32_t w[] = { 0x6000u, 0x3C00u, 0x3800u, 0x4000u,
                                      0x7000u, 0x5000u, 0x2000u, 0x6C00u };
        for (i = 0; i < 8; i++) {
            bias_set(bias, w[i], 0);
            memset(out, 0, 2048);
            hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
            hmx_sync();
            printf("   word=0x%04x -> out[0]=%d\n", w[i], out[0]);
        }
    }

    /* ---- K channel count with the known-good bias: .uh too ---- */
    printf("\nchannel count / store form (bias 0x6000)\n");
    bias_set(bias, 0x6000u, 0);
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("   .ub : out[0]=%d out[1024]=%d out[2047]=%d\n", out[0], out[1024], out[2047]);
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    memcpy(h, out, sizeof(h));
    printf("   .uh : h[0]=%d h[1]=%d h[2]=%d h[3]=%d  h[2047]=%d h[2048]=%d\n",
           h[0], h[1], h[2], h[3], h[2047], h[2048]);

    /* ---- spatial identity map through .ub (64 rows, act(s,0)=s+1) ---- */
    printf("\nspatial identity through .ub (act(s,0)=s+1, w k==0 only)\n");
    memset(act, 0, 4096);
    for (s = 0; s < 64; s++) act[128 * (s >> 2) + (s & 3)] = (uint8_t)(s + 1);
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++)
        for (s = 0; s < 4; s++) w4[4 * n + s] |= 1u;
    bias_set(bias, 0x6000u, 0);
    memset(out, 0, 2048);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    {
        int live = 0;
        for (s = 0; s < 64; s++)
            if (out[128 * (s >> 2) + (s & 3)] == s + 1) live++;
        printf("   rows matching = %2d/64;  raw out[0..7]= %d %d %d %d %d %d %d %d\n",
               live, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
        printf("   out[128..131]= %d %d %d %d ; out[256..259]= %d %d %d %d\n",
               out[128], out[129], out[130], out[131],
               out[256], out[257], out[258], out[259]);
    }

    /* ---- per-channel scale index (which word n uses) ---- */
    printf("\nper-channel scale: word[n]=0x6000 except word[5]=0x5000 (0.5?)\n");
    bias_set(bias, 0x6000u, 0);
    ((uint32_t *)bias)[5] = 0x5000u;
    memset(act, 1, 4096);
    memset(w4, 0x11, 512);
    memset(out, 0, 2048);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("   n=0..7: %d %d %d %d %d %d %d %d (n=5 differs => word index == n)\n",
           out[0], out[4], out[8], out[12], out[16], out[20], out[24], out[28]);

    /* ---- saturation / negative with the known-good bias ---- */
    printf("\nnegative accumulator with .ub (w nibble 0xF = -1)\n");
    memset(w4, 0xFF, 512);
    memset(out, 0, 2048);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("   acc=-32 -> out[0]=%d\n", out[0]);
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    memcpy(h, out, sizeof(h));
    printf("   acc=-32 -> .uh h[0]=%d h[1]=%d\n", h[0], h[1]);

    /* ---- out_bias: is it added in the high 16 bits of the word? ---- */
    printf("\nout_bias test: scale word 0x6000, out_bias 100 at n=0\n");
    memset(w4, 0x11, 512);
    bias_set(bias, 0x6000u, 0);
    ((uint32_t *)bias)[0] = (100u << 16) | 0x6000u;
    memset(out, 0, 2048);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("   n=0 -> %d (32 or 132), n=4 -> %d\n", out[0], out[16]);

    /* ---- control word effect at the known-good baseline ---- */
    printf("\ncontrol word vs channel count (bias 0x6000, .ub, out[0])\n");
    {
        static const uint32_t c[] = { 0x07Cu, 0x87Cu, 0x3FFu, 0x47Cu, 0xFFCu };
        memset(w4, 0x11, 512);
        for (i = 0; i < 5; i++) {
            memset(out, 0, 2048);
            hmx_clr(); mul_n(act, c[i], w4, 511u); hmx_bias2(bias); st_ub(out, 0);
            hmx_sync();
            printf("   act_ctl=0x%03x -> %d\n", c[i], out[0]);
        }
    }
    return 0;
}
