/*
 * HMX (mxmem) + VTCM probe for hexagon-sim, SDK 6.7 / Hexagon v68.
 *
 * Purpose: prove that HMX VTCM instructions work in hexagon-sim and measure
 * how many K channels one `activation.ub = mxmem(...)` + `weight.b/n = mxmem(...)`
 * packet really accumulates.
 *
 * Build:
 *   hexagon-clang -mv68 -mhmx -mhvx -mhvx-length=128b -O2 hmx_vtcm_probe.c -o hmx_probe
 * Run:
 *   hexagon-sim -mv68 --mhmx 1 ./hmx_probe -- 0xd8400000
 *
 * VTCM in hexagon-sim is ON by default:
 *   low = 0xd8000000 (TCM region base) + QDSP6_TCM_REGION_VTCM_OFFSET(KB) << 10
 *       = 0xd8400000 with the default offset 4096 KB
 *   size = 4 MB  (hi = 0xd87fffff)
 * An address outside that window raises 0x26 "Coprocessor VMEM address error".
 *
 * Three things are mandatory for a non-zero result:
 *   1. SSR.XE2 (bit 26) set, otherwise mxclracc -> 0x18.
 *   2. hexagon-sim --mhmx 1.
 *   3. A bias/scale register block loaded with `bias = mxmem2(addr)` where the
 *      low word of each of the 32 entries carries the fp16 scale.
 *      Without it (all-zero bias) the fxp convert/store path writes zeros
 *      even though the accumulator itself holds the right numbers.
 *      0x6000 = int8/int4 identity scale.
 *
 * What the numbers mean: all activations and weights are 1, so the result is
 * exactly the number of multiply-accumulates = number of accumulated K
 * channels.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VTCM_BASE 0xd8400000ul

/* VTCM layout used here (all 2 KB aligned, well inside the 4 MB window) */
#define OFF_ACT   0x0000u          /* 4 * 2048 B */
#define OFF_W8    0x4000u          /* 4 * 2048 B int8 weights */
#define OFF_W4    0x6000u          /* 4 *  512 B int4 weights */
#define OFF_OUT   0x8000u          /* 2 KB result tile */
#define OFF_BIAS  0x9000u          /* 256 B bias/scale block */

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

/* one crouton: 32 K channels x 32 N, 8-bit activation + 8-bit weight */
static inline void hmx_mul_i8(const void *a, uint32_t ac, const void *w, uint32_t wc, int deep)
{
    if (deep)
        __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.b = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
               "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
    else
        __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
               "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}

/* one crouton: 32 K channels x 32 N, 8-bit activation + 4-bit (nibble) weight */
static inline void hmx_mul_i4(const void *a, uint32_t ac, const void *w, uint32_t wc, int deep)
{
    if (deep)
        __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.n = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
               "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
    else
        __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
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

static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

/* ACT ctrl = (croutons << 11) | spatial mask 0x7C ; WGT ctrl = bytes-1 */
#define ACT_CTRL(ncr)   (((uint32_t)(ncr) << 11) | 0x7Cu)
#define WGT8_ALL(ncr)   (((uint32_t)(ncr) * 2048u) - 1u)   /* 2 KB per crouton */
#define WGT4_ALL(ncr)   (((uint32_t)(ncr) * 512u) - 1u)    /* 512 B per crouton */

int main(int argc, char **argv)
{
    unsigned long base = argc > 1 ? strtoul(argv[1], 0, 0) : VTCM_BASE;
    int skip_xe2 = (argc > 2 && argv[2][0] == '0');
    uint8_t *p    = (uint8_t *)(uintptr_t)base;
    uint8_t *act  = p + OFF_ACT;
    uint8_t *w8   = p + OFF_W8;
    uint8_t *w4   = p + OFF_W4;
    uint8_t *out  = p + OFF_OUT;
    uint8_t *bias = p + OFF_BIAS;
    int fails = 0;

    if (!skip_xe2) set_xe2();

    /* all-ones data: the result equals the number of MACs = number of K channels */
    memset(act, 1, 4 * 2048);
    memset(w8,  1, 4 * 2048);
    memset(w4, 0x11, 4 * 512);          /* both nibbles = 1 */

    /* bias/scale: identity. low word = {out_bias[31:16], scale_fp16[15:0]} */
    memset(bias, 0, 256);
    {
        uint32_t *w = (uint32_t *)bias;
        for (int i = 0; i < 32; i++) w[i] = 0x00006000u;   /* identity scale */
    }

    printf("VTCM base=0x%08lx  (SSR.XE2 %s)\n", base, skip_xe2 ? "NOT set" : "set");

    for (int idx = 0; idx < 3; idx++) {
        int ncr = 1 << idx;                     /* 1, 2, 4 croutons */
        int expect = 32 * ncr;                  /* K = 32, 64, 128 */
        unsigned v;

        /* A: one mxmem packet per crouton, weight ctrl covers one crouton */
        memset(out, 0, 2048);
        hmx_clracc();
        for (int i = 0; i < ncr; i++)
            hmx_mul_i8(act + i * 2048, ACT_CTRL(1), w8 + i * 2048, 2047u, 0);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        v = out[0];
        printf("int8 loop per-crouton   K=%3d -> %3u  %s\n", expect, v, v == (unsigned)expect ? "PASS" : "FAIL");
        if (v != (unsigned)expect) fails++;

        /* B: one :deep packet, weight ctrl covers all croutons
         * (informational: the simulator's :deep recipe is quirky, see REPORT22) */
        memset(out, 0, 2048);
        hmx_clracc();
        hmx_mul_i8(act, ACT_CTRL(ncr), w8, WGT8_ALL(ncr), 1);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        printf("int8 :deep full-range    K=%3d -> %3u  (informational)\n", expect, out[0]);

        /* C: one :deep packet, weight ctrl covers ONE crouton -> truncated */
        memset(out, 0, 2048);
        hmx_clracc();
        hmx_mul_i8(act, ACT_CTRL(ncr), w8, 2047u, 1);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        printf("int8 :deep short-weight K=%3d -> %3u  (weight range clamps the block count)\n",
               expect, out[0]);

        /* D: int4 nibble weights, one packet per crouton */
        memset(out, 0, 2048);
        hmx_clracc();
        for (int i = 0; i < ncr; i++)
            hmx_mul_i4(act + i * 2048, ACT_CTRL(1), w4 + i * 512, 511u, 0);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        v = out[0];
        printf("int4 loop per-crouton   K=%3d -> %3u  %s\n", expect, v, v == (unsigned)expect ? "PASS" : "FAIL");
        if (v != (unsigned)expect) fails++;

        /* E: int4, one :deep packet, weight ctrl covers all croutons */
        memset(out, 0, 2048);
        hmx_clracc();
        hmx_mul_i4(act, ACT_CTRL(ncr), w4, WGT4_ALL(ncr), 1);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        v = out[0];
        printf("int4 :deep full-range    K=%3d -> %3u  %s\n", expect, v, v == (unsigned)expect ? "PASS" : "FAIL");
        if (v != (unsigned)expect) fails++;

        /* F: int4, one :deep packet, weight ctrl covers ONE crouton */
        memset(out, 0, 2048);
        hmx_clracc();
        hmx_mul_i4(act, ACT_CTRL(ncr), w4, 511u, 1);
        hmx_bias_load(bias); hmx_store_ub(out); hmx_sync();
        printf("int4 :deep short-weight K=%3d -> %3u  (weight range clamps the block count)\n",
               expect, out[0]);
        printf("\n");
    }

    printf("%s\n", fails ? "RESULT: FAIL" : "RESULT: OK (VTCM + HMX functional)");
    return fails ? 1 : 0;
}
