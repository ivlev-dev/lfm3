/*
 * hmx_blk_probe — semantics of the HMX paths REPORT24 needs, measured in
 * hexagon-sim (Hexagon v68, SDK 6.7).
 *
 * Questions:
 *  T1 how many spatial rows does the int4 recipe
 *     `activation.ub = mxmem(...,0x3FF)` + `weight.n = mxmem(...,511)` fill?
 *  T2 does `weight.ubit` decode the 4-bit field as unsigned 0..15 while
 *     `weight.n` decodes it as signed -8..7?  (Q4_K nibbles are unsigned.)
 *  T3 what do the store forms write, and how many bytes?
 *     (.ub 8-bit, .uh 16-bit :2x1, .hf fp16)
 *  T4 bias/scale word block (`bias = mxmem2`): per-channel indexing, rounding,
 *     out_bias, saturation, and which 128 B half the store reads.
 *  T5 can `weight.sc` / `weight.sm` carry a per-block scale/minimum, i.e. is
 *     there native Q4_K support in the HMX?
 *
 * Build: ./build.sh blk
 * Run:   ./run.sh blk
 *
 * Layouts (identical math to mx4_impl.c / mkblob.c):
 *   activation crouton 2 KB: byte(s,k) = 128*(s>>2) + 4*(k&31) + (s&3)
 *   int4 weight crouton 512 B: byte(k,n) = 128*((k&31)>>3) + 4*n + (k&3),
 *                              high nibble iff ((k>>2)&1)
 *   .ub store tile 2 KB:      out(s,n) = 128*(s>>2) + 4*n + (s&3)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VTCM_BASE 0xd8400000ul
#define OFF_ACT   0x00000u
#define OFF_W4    0x04000u
#define OFF_OUT   0x08000u          /* 8 KB of store space */
#define OFF_BIAS  0x10000u

static void set_xe2(void)
{
    unsigned s;
    __asm__ __volatile__("r0 = ssr\n\t%0 = r0" : "=r"(s) :: "r0");
    s |= 1u << 26;
    __asm__ __volatile__("ssr = %0" :: "r"(s) : "memory");
}
static inline void hmx_clr(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
}
static inline void hmx_bias2(const void *p)
{
    __asm__ __volatile__("{ bias = mxmem2(%0) }"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}
static inline void hmx_sync(void)
{
    __asm__ __volatile__("mxclracc" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

#define MUL(WT)                                                              \
static inline void mul_##WT(const void *a, uint32_t ac, const void *w,        \
                            uint32_t wc)                                     \
{                                                                            \
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t"            \
                         "weight." #WT " = mxmem(%2,%3)\n\t}"               \
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),            \
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");\
}
MUL(n)
MUL(ubit)
MUL(sbit)
MUL(b)

/* weight.sc / weight.sm are alternative weight *encodings*: one activation +
 * one weight per packet, so they need the activation in the same packet. */
static inline void mul_sc(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t"
                         "weight.sc = mxmem(%2,%3)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}
static inline void mul_sm(const void *a, uint32_t ac, const void *w, uint32_t wc)
{
    __asm__ __volatile__("{\n\t activation.ub = mxmem(%0,%1)\n\t"
                         "weight.sm = mxmem(%2,%3)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)a), "r"(ac),
                            "r"((uint32_t)(uintptr_t)w), "r"(wc) : "memory");
}

static inline void st_ub(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}
static inline void st_uh(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}
static inline void st_hf(void *o, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)o), "r"(rt) : "memory");
}

static uint16_t f16(float f)
{
    union { float f; uint32_t u; } v;
    uint32_t s, e, m, r;
    v.f = f;
    s = (v.u >> 31) & 1u;
    if (v.u == 0) return 0;
    e = ((v.u >> 23) & 0xFFu) - 127 + 15;
    m = v.u & 0x7FFFFFu;
    r = (s << 15) | ((e & 0x1Fu) << 10) | (m >> 13);
    return (uint16_t)r;
}

static void bias_fill(uint8_t *b, const float *scale, const int *ob,
                      const float *scale1)
{
    uint32_t *w = (uint32_t *)b;
    int i;
    memset(b, 0, 256);
    for (i = 0; i < 32; i++)
        w[i] = ((uint32_t)(uint16_t)(ob ? ob[i] : 0) << 16)
             | (uint32_t)f16(scale[i]);
    for (i = 0; i < 32; i++)
        w[64 + i] = (uint32_t)f16(scale1 ? scale1[i] : 1.0f);
}

/* activation crouton: 64 spatial x 32 K, 8-bit codes */
static void act_ones(uint8_t *a) { memset(a, 1, 2048); }
static void act_row(uint8_t *a)                   /* act(s,0)=s+1, rest 0 */
{
    int s;
    memset(a, 0, 2048);
    for (s = 0; s < 64; s++)
        a[128 * (s >> 2) + 0 + (s & 3)] = (uint8_t)(s + 1);
}
/* weight crouton: all 32 K of all 32 N = nibble value v */
static void w4_flat(uint8_t *w, int v)
{
    memset(w, (v & 0xF) | ((v & 0xF) << 4), 512);
}
/* weight crouton: nibble v only at k==0, zero elsewhere */
static void w4_k0(uint8_t *w, int v)
{
    int k, n;
    memset(w, 0, 512);
    for (k = 0; k < 32; k++)
        for (n = 0; n < 32; n++) {
            unsigned q = (k == 0) ? (unsigned)v : 0u;
            unsigned off = 128u * (unsigned)((k & 31) >> 3)
                         + 4u * (unsigned)n + (unsigned)(k & 3);
            if ((k >> 2) & 1) w[off] |= (uint8_t)(q << 4);
            else              w[off] |= (uint8_t)q;
        }
}
static int ub_at(const uint8_t *o, int s, int n)
{
    return o[128 * (s >> 2) + 4 * n + (s & 3)];
}

static int fails;

int main(void)
{
    uint8_t *p = (uint8_t *)(uintptr_t)VTCM_BASE;
    uint8_t *act = p + OFF_ACT, *w4 = p + OFF_W4;
    uint8_t *out = p + OFF_OUT, *bias = p + OFF_BIAS;
    int s, i;
    int16_t hbuf[4096];
    float one[32], qtr[32];
    int zero[32];

    set_xe2();
    memset(p, 0, 0x20000);
    for (i = 0; i < 32; i++) { one[i] = 1.0f; qtr[i] = 0.25f; zero[i] = 0; }

    printf("VTCM base=0x%08lx\n\n", VTCM_BASE);

    /* ---------------- T1: spatial row coverage ---------------- */
    act_row(act);
    w4_k0(w4, 1);
    bias_fill(bias, one, zero, 0);
    memset(out, 0, 4096);
    hmx_clr();
    mul_n(act, 0x3FFu, w4, 511u);
    hmx_bias2(bias);
    st_ub(out, 0);
    hmx_sync();
    {
        int live = 0, firstbad = -1;
        for (s = 0; s < 64; s++) {
            int got = ub_at(out, s, 0), want = s + 1;
            if (got == want) live++;
            else if (firstbad < 0) firstbad = s;
            if (s < 4 || (s >= 30 && s <= 34) || s == 63)
                printf("T1 s=%2d out=%3d want=%3d%s\n", s, got, want,
                       got == want ? "" : "   <-- MISMATCH");
        }
        printf("T1 live spatial rows = %d/64, first mismatch s=%d\n", live, firstbad);
        if (live != 32 && live != 64) fails++;
    }

    /* ---------------- T2: nibble decode, signed vs unsigned ---------------- */
    printf("\nT2  nibble decode: acc = 32 * decode(v), scale 1/4, .uh store\n");
    printf("    v : ");
    {
        int vs[4] = { 7, 8, 15, 1 };
        for (i = 0; i < 4; i++) printf("%10d", vs[i]);
        printf("\n");
        for (i = 0; i < 4; i++) {
            int v = vs[i];
            int32_t r;
            act_ones(act);
            w4_flat(w4, v);
            bias_fill(bias, qtr, zero, 0);
            memset(out, 0, 8192);
            hmx_clr();
            mul_ubit(act, 0x3FFu, w4, 511u);
            hmx_bias2(bias);
            st_uh(out, 0);
            hmx_sync();
            memcpy(hbuf, out, sizeof(hbuf));
            r = hbuf[0];
            printf("    ubit -> %7d", r);
            memset(out, 0, 8192);
            hmx_clr();
            mul_n(act, 0x3FFu, w4, 511u);
            hmx_bias2(bias);
            st_uh(out, 0);
            hmx_sync();
            memcpy(hbuf, out, sizeof(hbuf));
            printf("   (weight.n -> %d)\n", hbuf[0]);
        }
    }

    /* ---------------- T3: store forms ---------------- */
    printf("\nT3  store forms, acc = 32 for every (s,n), scale 1/4\n");
    act_ones(act);
    w4_flat(w4, 1);
    bias_fill(bias, qtr, zero, 0);
    memset(out, 0xA5, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
    hmx_sync();
    printf("    .ub  out(0,0)=%d out(63,31)=%d  [2048]=%02x [4095]=%02x\n",
           ub_at(out, 0, 0), ub_at(out, 63, 31), out[2048], out[4095]);
    memset(out, 0xA5, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_uh(out, 0);
    hmx_sync();
    memcpy(hbuf, out, sizeof(hbuf));
    {
        int nz = 0, last = -1;
        for (i = 0; i < 4096; i++) if (hbuf[i]) { nz++; last = i; }
        printf("    .uh  h[0]=%d nz16=%d last=%d  [4096]=%02x\n",
               hbuf[0], nz, last, out[4096]);
    }
    memset(out, 0xA5, 8192);
    hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_hf(out, 0);
    hmx_sync();
    {
        const __fp16 *f = (const __fp16 *)out;
        int nz = 0, last = -1;
        for (i = 0; i < 4096; i++) if (f[i]) { nz++; last = i; }
        printf("    .hf  f[0]=%.3f nz16=%d last=%d  [4096]=%02x\n",
               (double)f[0], nz, last, out[4096]);
    }

    /* ---------------- T4: bias/scale block semantics ---------------- */
    printf("\nT4  bias/scale block\n");
    {
        float sc[32];
        int ob[32];
        for (i = 0; i < 32; i++) { sc[i] = 1.0f; ob[i] = 0; }
        sc[5] = 0.25f;
        bias_fill(bias, sc, ob, 0);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    per-channel: n0=%d(want 32) n5=%d(want 8) n6=%d(want 32)\n",
               ub_at(out, 0, 0), ub_at(out, 0, 5), ub_at(out, 0, 6));
        for (i = 0; i < 32; i++) { sc[i] = 0.1f; ob[i] = 0; }
        bias_fill(bias, sc, ob, 0);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    rounding: 32*0.1=3.2 -> %d (3 trunc / 4 round)\n", ub_at(out, 0, 0));
        for (i = 0; i < 32; i++) { sc[i] = 0.25f; ob[i] = 100; }
        bias_fill(bias, sc, ob, 0);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    out_bias: 32*0.25 + 100 -> %d (want 108)\n", ub_at(out, 0, 0));
        /* saturation of the .ub store */
        for (i = 0; i < 32; i++) { sc[i] = 100.0f; ob[i] = 0; }
        bias_fill(bias, sc, ob, 0);
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    .ub saturation: 3200 -> %d (want 255)\n", ub_at(out, 0, 0));
        /* negative accumulator through .ub with a 128 offset */
        for (i = 0; i < 32; i++) { sc[i] = 0.25f; ob[i] = 128; }
        bias_fill(bias, sc, ob, 0);
        w4_flat(w4, 0xF);                     /* -1 signed */
        memset(out, 0, 2048);
        hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
        hmx_sync();
        printf("    signed -32 * .25 + 128 -> %d (want 120)\n", ub_at(out, 0, 0));
        /* which half of mbias does the plain store use? */
        {
            float h0[32];
            for (i = 0; i < 32; i++) { h0[i] = 1.0f; sc[i] = 1.0f; }
            bias_fill(bias, h0, zero, h0);
            ((uint32_t *)bias)[32] = 0x3800u;   /* second 128 B half = 0.5 */
            memset(out, 0, 2048);
            hmx_clr(); mul_n(act, 0x3FFu, w4, 511u); hmx_bias2(bias); st_ub(out, 0);
            hmx_sync();
            printf("    second 128 B half set to 0.5 -> %d (32 = to be ignored)\n",
                   ub_at(out, 0, 0));
        }
    }

    /* ---------------- T5: weight.sc / weight.sm ---------------- */
    printf("\nT5  weight.sc / weight.sm: alternative 4-bit weight encodings?\n");
    act_ones(act);
    bias_fill(bias, one, zero, 0);
    for (i = 0; i < 3; i++) {
        uint8_t *wb = p + 0x14000 + i * 2048;
        int16_t r0 = -9999, r1 = -9999;
        memset(wb, 0x11, 2048);                    /* nibble 1 everywhere */
        memset(out, 0, 8192);
        hmx_clr(); mul_n(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
        hmx_sync();
        memcpy(hbuf, out, sizeof(hbuf)); r0 = hbuf[0];
        memset(out, 0, 8192);
        hmx_clr();
        if (i == 0)      mul_sc(act, 0x3FFu, wb, 511u);
        else if (i == 1) mul_sm(act, 0x3FFu, wb, 511u);
        else             mul_n(act, 0x3FFu, wb, 511u);
        hmx_bias2(bias);
        st_uh(out, 0);
        hmx_sync();
        memcpy(hbuf, out, sizeof(hbuf)); r1 = hbuf[0];
        printf("    %-10s nibbles=1, range 511 -> out16[0]=%d   (weight.n -> %d)\n",
               i == 0 ? "weight.sc" : i == 1 ? "weight.sm" : "weight.n", r1, r0);
    }
    /* does the encoding read extra bytes outside the 512 B window? */
    {
        uint8_t *wb = p + 0x14000;
        int v;
        for (v = 0; v < 2; v++) {
            int16_t r;
            memset(wb, 0x11, 2048);
            if (v) memset(wb + 512, 0xFF, 512);   /* just past the weight block */
            memset(out, 0, 8192);
            hmx_clr(); mul_sc(act, 0x3FFu, wb, 511u); hmx_bias2(bias); st_uh(out, 0);
            hmx_sync();
            memcpy(hbuf, out, sizeof(hbuf)); r = hbuf[0];
            printf("    weight.sc, bytes[512..1023]=%s -> %d\n",
                   v ? "0xFF" : "0x11", r);
        }
    }

    printf("\n%s\n", fails ? "RESULT: CHECK" : "RESULT: OK");
    return 0;
}
