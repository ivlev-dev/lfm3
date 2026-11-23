/*
 * hmx_part_probe — the exact recipe for *partial-sum* stores, which is what
 * REPORT24's block-scale schemes need.
 *
 * Findings carried in from hmx_ctrl_probe (hexagon-sim v68, SDK 6.7):
 *   - `activation.ub = mxmem(...,0x3FF)` + `weight.n = mxmem(...,511)` really
 *     computes 32 K x 32 N x 64 spatial, control word irrelevant;
 *   - the bias word low half is a scale in *fixed point*, not raw fp16:
 *     `.ub` multiplies by fp16(word)/512, `.uh` by fp16(word)/2;
 *   - `mxmem(...):after:sat.ub = acc` writes 2 KB, `.uh = acc:2x1` writes 4 KB;
 *   - both stores are unsigned (negative accumulator -> 0).
 *
 * Questions here:
 *   Q1 `.uh` byte layout for 64 spatial x 32 N, 16-bit
 *   Q2 scale multiplier scan for `.uh` (which word gives an exact power of 2)
 *   Q3 does out_bias (high half of the word) work?  (needed for signed partials)
 *   Q4 exactness of a += 1 / -= 1 accumulator through `.uh`
 *   Q5 can two partials (2 K blocks, different scales) be stored from the two
 *      accumulator banks in one go, or must they be two stores?
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
static inline void hmx_swap(void) { __asm__ __volatile__("mxswapacc" ::: "memory"); }
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
static uint16_t f16(float f)
{
    union { float f; uint32_t u; } v;
    uint32_t s, e, m;
    v.f = f; s = (v.u >> 31) & 1u;
    if (v.u == 0) return 0;
    e = ((v.u >> 23) & 0xFFu) - 127 + 15;
    m = v.u & 0x7FFFFFu;
    return (uint16_t)((s << 15) | ((e & 0x1Fu) << 10) | (m >> 13));
}
static void bias_set(uint8_t *b, uint32_t word)
{
    int i;
    memset(b, 0, 512);
    for (i = 0; i < 32; i++) ((uint32_t *)b)[i] = word;
    for (i = 0; i < 32; i++) ((uint32_t *)b)[64 + i] = word;
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
    printf("Q2 scale scan: acc=32 (act=1, q=1, 32 K), .uh store, fp16(word)/2\n");
    {
        static const uint32_t w[] = { 0x6000u, 0x6400u, 0x5800u, 0x6800u, 0x6600u };
        memset(act, 1, 4096);
        memset(w4, 0x11, 512);
        for (i = 0; i < 5; i++) {
            bias_set(bias, w[i]);
            memset(out, 0, 8192);
            hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
            hmx_sync();
            memcpy(h, out, sizeof(h));
            printf("   word=0x%04x -> h[0]=%d   (acc*mult, mult=fp16/2=%.1f)\n",
                   w[i], h[0], (double)({
                       union { uint16_t hb; __fp16 f; } u;
                       u.hb = (uint16_t)w[i]; (float)u.f; }) / 2.0);
        }
    }

    /* Q1: .uh layout with act(s,0)=s+1 and q=1 only at k==0 */
    printf("\nQ1 .uh byte layout (act(s,0)=s+1, q[k==0]=1, word=0x6000 => x256)\n");
    memset(act, 0, 4096);
    for (s = 0; s < 64; s++) act[128 * (s >> 2) + (s & 3)] = (uint8_t)(s + 1);
    memset(w4, 0, 512);
    for (n = 0; n < 32; n++)
        for (s = 0; s < 4; s++) w4[4 * n + s] |= 1u;
    bias_set(bias, 0x6000u);
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    memcpy(h, out, sizeof(h));
    printf("   h[0..15]: ");
    for (i = 0; i < 16; i++) printf("%d ", h[i]);
    printf("\n   intending 256*(s+1); find the stride: h[0]=%d h[64]=%d h[128]=%d "
           "h[256]=%d h[1024]=%d\n", h[0], h[64], h[128], h[256], h[1024]);
    printf("   h[2048]=%d h[4095]=%d (0 => 4 KB written)\n", h[2048], h[4095]);

    /* Q3: out_bias in the high 16 bits of the word */
    printf("\nQ3 out_bias (high half of the bias word)\n");
    memset(act, 1, 4096);
    memset(w4, 0x11, 512);
    {
        uint32_t wu = (100u << 16) | 0x6000u;
        bias_set(bias, wu);
        memset(out, 0, 8192);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync();
        memcpy(h, out, sizeof(h));
        printf("   .uh acc=32, out_bias=100 -> h[0]=%d  (8192 = no bias, "
               "8192+100*x = bias applied)\n", h[0]);
        bias_set(bias, wu);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("   .ub acc=32, out_bias=100 -> out[0]=%d\n", out[0]);
        /* negative accumulator with a +128 offset */
        memset(w4, 0xFF, 512);                /* q = -1 -> acc = -32 */
        bias_set(bias, (128u << 16) | 0x6000u);
        memset(out, 0, 8192);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync();
        memcpy(h, out, sizeof(h));
        printf("   .uh acc=-32, out_bias=128 -> h[0]=%d (0 = sat, 96 = offset works\n"
               "        as (acc+128) = 96 * ...)\n", h[0]);
    }

    /* Q4: exact accumulator granularity through .uh */
    printf("\nQ4 .uh granularity: acc = 1 (act 1, q 1 on one channel only)\n");
    memset(w4, 0, 512);
    w4[0] = 0x01;                              /* k=0,n=0 low nibble = 1 */
    bias_set(bias, 0x6400u);                   /* x512 -> out = 1*512 */
    memset(out, 0, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    memcpy(h, out, sizeof(h));
    printf("   acc=1 x512 -> h[0]=%d (want 512)\n", h[0]);

    /* Q5: two banks: accumulate block A into bank0, block B into bank1 */
    printf("\nQ5 two accumulator banks (weight.n:deep switches banks)\n");
    {
        uint8_t *a1 = act + 2048;
        memset(act, 1, 2048);
        memset(a1, 1, 2048);
        memset(w4, 0x22, 1024);               /* q = 2 */
        bias_set(bias, 0x6000u);
        memset(out, 0, 8192);
        hmx_clr();
        hmx_bias2(bias);
        mul_n(act, 0x3FFu, w4, 511u);
        __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t"
                             "weight.n = mxmem(%2,%3):deep\n\t}"
                             :: "r"((uint32_t)(uintptr_t)a1), "r"(0x3FFu),
                                "r"((uint32_t)(uintptr_t)(w4 + 512)), "r"(511u)
                             : "memory");
        st_uh(out, 0);
        hmx_sync();
        memcpy(h, out, sizeof(h));
        printf("   after two blocks: h[0]=%d h[2048]=%d  (0/4096 = separate banks?)\n",
               h[0], h[2048]);
        memset(out, 0, 8192);
        hmx_clr();
        hmx_bias2(bias);
        mul_n(act, 0x3FFu, w4, 511u);
        st_uh(out, 0);
        hmx_sync();
        memcpy(h, out, sizeof(h));
        printf("   single block .uh: h[0]=%d (64 = 32 K x 2)\n", h[0]);
        memset(out, 0, 8192);
        hmx_clr();
        hmx_bias2(bias);
        mul_n(act, 0x3FFu, w4, 511u);         /* bank0 */
        hmx_swap();
        st_uh(out, 0);
        hmx_sync();
        memcpy(h, out, sizeof(h));
        printf("   after mxswapacc: h[0]=%d (0 => bank1 was empty)\n", h[0]);
    }
    return 0;
}
