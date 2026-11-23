/*
 * probe_bias.c - what exactly does the HMX `bias` register block do?
 *
 * Context: `activation.ub` reads UNsigned bytes, so int8 activations must be
 * stored as u8 = a + 128 (zp = 128) and the term zp*sum_k(w[n][k]) has to be
 * removed from every output row.  The HMX PRM calls the per-channel field of
 * the 256-byte bias block "input_bias" (PRM 2.3.1), but the exact encoding
 * (int32? 16.16? signed 16-bit?) is not documented in the SDK we have, and
 * myelin/docs/hmx.md only records `scale` (low word, fp16*512).
 *
 * This probe answers it on hexagon-sim: accumulator is a known small constant
 * (act = 1, w = 1, K = 32 -> acc = 32, or act = 2 -> 64), the low word carries
 * only the identity scale, and the high word is swept.  Output is read back
 * through `:after:sat.ub`, so it is clipped to 0..255; a handful of cases
 * around 0 and 255 disambiguate sign and shift.
 *
 * Build/run: ./build.sh && ./run.sh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VTCM_BASE 0xd8400000ul

#define OFF_ACT   0x0000u   /* 2 KB */
#define OFF_W8    0x2000u   /* 1 KB */
#define OFF_OUT   0x4000u   /* 2 KB */
#define OFF_BIAS  0x5000u   /* 256 B */

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}

static inline void hmx_clracc(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
}

static inline void hmx_mul(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
           "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}

static inline void hmx_bias_load(const void *b)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
        :: "r"((uint32_t)(uintptr_t)b) : "memory");
}

static inline void hmx_store_ub(void *o)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
        :: "r"((uint32_t)(uintptr_t)o), "r"(0u) : "memory");
}

static inline void hmx_store_uh(void *o)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}"
        :: "r"((uint32_t)(uintptr_t)o), "r"(0u) : "memory");
}

static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

static uint8_t *g_act, *g_w8, *g_out, *g_bias;

static void fill_ones(int av, int wv)
{
    memset(g_act, av, 2048);
    memset(g_w8,  wv, 1024);
}

/* one K=32 multiply, identity scale, the given low/high bias words */
static unsigned run_case(uint32_t low_ch0, uint32_t hi_ch0, int uh)
{
    uint32_t *w = (uint32_t *)g_bias;
    int i;
    memset(g_bias, 0, 256);
    for (i = 0; i < 32; i++) w[i] = 0x00006000u;   /* scale = 1.0, out_bias = 0 */
    for (i = 0; i < 32; i++) w[32 + i] = 0u;
    w[0] = low_ch0;
    w[32 + 0] = hi_ch0;

    memset(g_out, 0, 2048);
    hmx_clracc();
    hmx_mul(g_act, 0x3FFu, g_w8, 1023u);
    hmx_bias_load(g_bias);
    if (uh) hmx_store_uh(g_out); else hmx_store_ub(g_out);
    hmx_sync();
    if (uh) return (unsigned)g_out[0] | ((unsigned)g_out[1] << 8);
    return g_out[0];
}

int main(int argc, char **argv)
{
    unsigned long base = argc > 1 ? strtoul(argv[1], 0, 0) : VTCM_BASE;
    uint8_t *p = (uint8_t *)(uintptr_t)base;
    const uint32_t ID = 0x00006000u;
    int fails = 0;

    set_xe2();
    g_act = p + OFF_ACT;
    g_w8  = p + OFF_W8;
    g_out = p + OFF_OUT;
    g_bias = p + OFF_BIAS;

    printf("VTCM base=0x%08lx\n", base);

    /* baseline: act=1, w=1, K=32 -> acc=32 */
    fill_ones(1, 1);

    printf("\n-- accumulator baseline (identity bias) --\n");
    printf("ub(hi=0)          = %u   (expect 32)\n", run_case(ID, 0u, 0));
    printf("uh(hi=0)          = %u   (expect 32)\n", run_case(ID, 0u, 1));
    fails += run_case(ID, 0u, 0) != 32;

    printf("\n-- high word: integer candidate (acc=32) --\n");
    {
        struct { uint32_t hi; const char *what; } t[] = {
            { 100u,                 "hi=+100 (plain int)" },
            { 0xFFFFFF9Cu,          "hi=-100 (int32 twos complement)" },
            { 100u << 16,           "hi=100<<16" },
            { 0x00000100u,          "hi=256" },
            { 0x00010000u,          "hi=65536" },
            { 0x00008000u,          "hi=32768" },
            { 0x7FFFFFFFu,          "hi=INT32_MAX" },
            { 0x80000000u,          "hi=INT32_MIN" },
            { 0xFFFFFFFFu,          "hi=-1" },
        };
        unsigned i;
        for (i = 0; i < sizeof(t) / sizeof(t[0]); i++)
            printf("  %-34s -> ub=%3u  uh=%5u\n", t[i].what,
                   run_case(ID, t[i].hi, 0), run_case(ID, t[i].hi, 1));
    }

    printf("\n-- low word upper half (out_bias candidate), scale=identity --\n");
    {
        uint32_t lows[] = {
            ID | (16u << 16),
            ID | (100u << 16),
            ID | (0xFFFFu << 16),
            ID | (0x8000u << 16),
            ID | (0x0100u << 16),
        };
        unsigned i;
        for (i = 0; i < sizeof(lows) / sizeof(lows[0]); i++)
            printf("  low=0x%08lx -> ub=%3u  uh=%5u\n", (unsigned long)lows[i],
                   run_case(lows[i], 0u, 0), run_case(lows[i], 0u, 1));
    }

    printf("\n-- scale sweep, scale word = fp16*512 --\n");
    {
        struct { uint32_t s; const char *what; } t[] = {
            { 0x6000u, "1.0" },
            { 0x6200u, "1.5" },
            { 0x6400u, "2.0" },
            { 0x5800u, "0.5" },
            { 0x5000u, "0.25" },
            { 0x3C00u, "fp16 1.0 without the 1/512 convention" },
            { 0x0000u, "0" },
        };
        unsigned i;
        for (i = 0; i < sizeof(t) / sizeof(t[0]); i++)
            printf("  scale=0x%04lx (%-36s) ub=%3u uh=%5u\n",
                   (unsigned long)t[i].s, t[i].what,
                   run_case(t[i].s, 0u, 0), run_case(t[i].s, 0u, 1));
    }

    /* sign check: acc = 1*1*32 - 100 = -68 -> sat.ub must clip to 0 */
    printf("\n-- large negative accumulator under sat.ub --\n");
    fill_ones(1, 1);
    printf("  acc=32, hi=0xFFFFFF00 (-256) -> ub=%3u (expect 0)\n",
           run_case(ID, 0xFFFFFF00u, 0));
    fails += run_case(ID, 0xFFFFFF00u, 0) != 0;

    /* the actual use case: acc = 128*sum(w) = 128*32 = 4096, hi = -4096 */
    printf("\n-- zero-point use case: act=128, w=1, K=32 --\n");
    fill_ones(128, 1);
    printf("  no compensation          -> ub=%3u (expect 255, clipped)\n",
           run_case(ID, 0u, 0));
    printf("  hi=-4096                 -> ub=%3u (expect 0)\n",
           run_case(ID, (uint32_t)(int32_t)-4096, 0));
    fails += run_case(ID, (uint32_t)(int32_t)-4096, 0) != 0;

    printf("\n%s\n", fails ? "RESULT: FAIL" : "RESULT: OK (bias probe)");
    return fails ? 1 : 0;
}
