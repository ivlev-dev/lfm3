/*
 * int4 GEMV on Hexagon HVX (v68), M=1..4
 *
 * Weight format (per row, row-major [N][K]):
 *   K/2 bytes, two signed int4 per byte: byte = lo | (hi << 4),
 *   stored nibble = (w_signed + 8) in [0,15]  (offset-8 form).
 *   lo -> k even, hi -> k odd.
 *   Dot correction: sum(w_s*a) = sum(u*a) - 8*sum(a); sum(a) is row independent.
 *
 * Activations: int8 [M][K], K % 256 == 0.
 * Output: int8 [M][N] (per-row Q16 dequant) or int32 [M][N] raw.
 */
#include "lfmgemv.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include <hexagon_types.h>
#include "hvx_hexagon_protos.h"
#include "hexagon_protos.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <qurt_thread.h>
#include <qurt_sem.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>

#include "silu_lut.h"
#include "exptab.h"
#include "apps_mem.h"

typedef void *Address;

/* Weight handles are cheap metadata: the 24-layer LFM3 graph needs 72 FFN
 * handles + 44 attention handles (K/KT/V/block table) + 39 conv-layer handles
 * = 155, and the stand adds 72 more for the arena check.  256 covers a full
 * model with headroom; weights themselves live in ION arenas. */
#define MAXBUF 256

/* one host mapping, many views (alloc_view): 0 = model arena, 1 = spare */
#define MAXARENA 160

typedef struct {
    uint8_t *base;      /* DSP virtual address of the mapping */
    int      bytes;
    int      fd;
    int      used;
} arena_t;

static arena_t g_arena[MAXARENA];

typedef struct {
    int      used;
    int      N, K;
    int      nvec;      /* K/256 : 128-byte weight vectors per row */
    uint8_t *w;         /* N*K/2 bytes, 128B aligned */
    int32_t *qp;        /* N Q16 dequant multipliers (low16), or NULL -> int32 out */
    int32_t *qm;        /* N Q16.16 multipliers m/2^k (ffn gate/up dequant) */
    int      qshift;    /* k: dequantised gate/up is Q(16-k), see deq_wide() */
    int32_t *acc;       /* 4*N int32 scratch (M<=4) */
    int8_t  *ae, *ao;   /* 8*(K/2+128) scratch, 128B aligned */
    size_t   wbytes;
    int      w8;        /* 1 = int8 weights (N*K bytes, stored offset-128) */
    int8_t  *out_ptr;   /* host ION buffer mapped via apps_mem_share_map */
    int      out_bytes;
    uint8_t *w_heap;    /* weight copy allocated on the DSP heap (alloc_buf) */
    int8_t  *in_ptr;    /* host ION buffer mapped as the weights (attach_in) */
    int      in_bytes;
    int      in_owned;  /* 1 = in_ptr is this handle's own mapping */
    int      arena;     /* arena index for an alloc_view handle, else -1 */
    float    sa, so;    /* activation / output scales last given to set_scale */
} gbuf_t;

static gbuf_t g_buf[MAXBUF];

typedef struct {
    int32_t *acc2;
    int8_t  *h;
    int      n;
} ffn_t;
static ffn_t g_ffn;

/* SiLU path A/B switch (probe kind 45): 1 = vector, 0 = scalar chain. */
static int g_silu_vec = 1;

/* Buffers of the last FFN call, for the SiLU profiling probe (kind 40/42). */
static gbuf_t *g_ffn_last_g, *g_ffn_last_u;
static int32_t g_ffn_last_mq;
static volatile int g_prof_sink;

/* tuning */
/* tuning: measured optimum on this target (same as myelin/dsp/conv).  The
 * worker team is started lazily by the first split job. */
static int g_pf_kb   = 32;  /* l2fetch lookahead in KB (0 = off) */
static int g_rows    = 2;   /* rows per iteration in the specialised kernel */
static int g_usepf   = 1;
static int g_threads = 2;   /* row-range split across threads */
static int g_attn_tm = 0;
static const int32_t *g_dbg_scores;   /* for probe 28/29 */
static int g_dbg_n;   /* 0 = head-grouped (correct), 1 = token-major (experimental) */

/* ---------------- HVX helpers ---------------- */

static inline HVX_Vector vand_mask0f(HVX_Vector v) {
    return Q6_V_vand_VV(v, Q6_V_vsplat_R(0x0f0f0f0fu));
}
static inline HVX_Vector vlsr4b(HVX_Vector v) {
    return Q6_Vub_vlsr_VubR(v, 4);
}
static inline void pf_window(const void *p, int bytes) {
    uint64 ctl = ((uint64)128 << 32) | ((uint64)128 << 16) | (uint64)(bytes / 128);
    Q6_l2fetch_AP((Address)p, ctl);
}
/* prefetch `cnt` rows starting at row `r`, row size `rb`, clamped to `nrows` */
static inline void pf_rows(const uint8_t *base, int r, int cnt, int rb, int nrows)
{
    if (r + cnt > nrows) cnt = nrows - r;
    if (cnt <= 0) return;
    pf_window(base + (size_t)r * rb, cnt * rb);
}

static inline int32_t vsum_w(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    union { HVX_Vector v; int32_t s[32]; } u;
    u.v = v;
    return u.s[0];
}
/* (a*b)>>16 without 64-bit helpers: |a| < 2^23, |b| < 2^23.
 * Splits a into high/low 16 bits and truncates the low part to 8 bits,
 * error ~= (a & 0xff)*b/65536.  Negligible in the SiLU/exp path (b ~ 2^16),
 * but NOT for int8 requantisation with a small multiplier: b ~ 0.015 gives
 * ~4 LSB of 127.  pack_out_i8() and requant_mid() below take the product in
 * 64 bits.
 * The low partial product must be signed: the old unsigned form returned
 * ~2^24 for a negative b and put every such lane at +127 in the mid buffer.
 * The gate/up dequant no longer uses this helper (deq_wide/prod_p take the
 * product in 64 bits); exp/attention still do, and the regression gate in
 * ffn_sim keeps the old unsigned form. */
static inline int32_t mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((int64_t)lo * (int64_t)b) >> 8);
}

/* Gate/up dequantiser with a dynamic range wider than Q16.
 *
 * m = sw*sa/so is the real per-row multiplier before the fixed-point scale:
 * for the LFM3 int4 per-row absmax/7 format m = sw*s_in*2^16 measured at
 * 6..215, and for int8 absmax/127 at 0.3..12.  A Q16 multiplier clamps at
 * 1.0 (the old bug: every int4 row lost the factor), and a Q30 int32 field
 * cannot hold m > 2 either.
 *
 * The kernel stores m/2^k as Q16.16 (int32, so m/2^k < 32768) and takes the
 * product in 64 bits:
 *     z = (acc*qm + 2^15) >> 16 = acc*m/2^k,   scale 2^(16-k).
 * k is picked per buffer by set_scale() from the worst-case accumulator
 * |acc| <= 127*|w|max*K and the largest m, such that z <= 2^30 (2x int32
 * headroom) -- see deq_shift_for().  On the LFM3 tensors k = 0 (int4:
 * 2.1e6 * 216 = 4.5e8 < 2^30), so no resolution is spent; silu_qk() still
 * accepts Q(16-k) for shapes that need a widening shift. */
static inline int32_t deq_wide(int32_t acc, int32_t qm)
{
    return (int32_t)(((int64_t)acc * (int64_t)qm + 32768) >> 16);
}

/* Smallest k with 127*|w|max*K*mmax/2^k <= 2^30 and mmax/2^k < 2^15. */
static inline int deq_shift_for(double mmax, int K, int wbits)
{
    double bound = (double)(1 << wbits) * 127.0 * (double)K;
    int k = 0;
    while (k < 8 &&
           (bound * mmax / (double)(1u << k) > 1073741824.0 ||
            mmax / (double)(1u << k) > 32767.0))
        k++;
    return k;
}

/* product of a Q16 SiLU value and a Q(16-k) up value -> Q16 P */
static inline int32_t prod_p(int32_t sv, int32_t zu, int k)
{
    int64_t v = (int64_t)sv * (int64_t)zu + ((int64_t)1 << (15 - k));
    v >>= (16 - k);
    if (v > 0x7fffffffLL) v = 0x7fffffffLL;
    if (v < -0x7fffffffLL) v = -0x7fffffffLL;
    return (int32_t)v;
}

static int32_t sum_act_i8(const int8_t *a, int K) {
    int32_t s = 0;
    int i;
    for (i = 0; i < K; i++) s += a[i];
    return s;
}
static void deinterleave(const int8_t *a, int8_t *ae, int8_t *ao, int K) {
    int i, j;
    for (i = 0, j = 0; i + 1 < K; i += 2, j++) {
        ae[j] = a[i];
        ao[j] = a[i + 1];
    }
}

/* ---------------- generic M=1 kernel (any nvec) ---------------- */

static void gemv_rows_m1_generic(const gbuf_t *b, const int8_t *ae, const int8_t *ao,
                                 int32_t rowbias, int32_t *acc)
{
    const int nvec = b->nvec;
    const HVX_Vector *Ae = (const HVX_Vector *)ae;
    const HVX_Vector *Ao = (const HVX_Vector *)ao;
    const size_t rowbytes = (size_t)nvec * 128u;
    int r;

    for (r = 0; r < b->N; r++) {
        const HVX_Vector *wp = (const HVX_Vector *)(b->w + (size_t)r * rowbytes);
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        int c;
        for (c = 0; c + 3 < nvec; c += 4) {
            HVX_Vector v0 = wp[c + 0], v1 = wp[c + 1];
            HVX_Vector v2 = wp[c + 2], v3 = wp[c + 3];
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, vand_mask0f(v0), Ae[c + 0]);
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, vlsr4b(v0),      Ao[c + 0]);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, vand_mask0f(v1), Ae[c + 1]);
            a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, vlsr4b(v1),      Ao[c + 1]);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, vand_mask0f(v2), Ae[c + 2]);
            a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, vlsr4b(v2),      Ao[c + 2]);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, vand_mask0f(v3), Ae[c + 3]);
            a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, vlsr4b(v3),      Ao[c + 3]);
        }
        for (; c < nvec; c++) {
            HVX_Vector v0 = wp[c];
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, vand_mask0f(v0), Ae[c]);
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, vlsr4b(v0),      Ao[c]);
        }
        acc[r] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1),
                                        Q6_Vw_vadd_VwVw(a2, a3))) + rowbias;
    }
}

/* ---------------- specialised M=1 kernel (any nvec, prefetch) ----------------
 * ROW8P issues 8 independent 128B loads then 16 MACs.
 * g_rows = 1|2 rows per iteration, g_pf_kb = l2fetch lookahead window in KB.
 */

#define ROW8P(pw, pe, po, A0, A1, A2, A3) do {                                \
        HVX_Vector q0_ = (pw)[0], q1_ = (pw)[1], q2_ = (pw)[2], q3_ = (pw)[3]; \
        HVX_Vector q4_ = (pw)[4], q5_ = (pw)[5], q6_ = (pw)[6], q7_ = (pw)[7]; \
        A0 = Q6_Vw_vrmpyacc_VwVubVb(A0, vand_mask0f(q0_), (pe)[0]);            \
        A1 = Q6_Vw_vrmpyacc_VwVubVb(A1, vlsr4b(q0_),      (po)[0]);            \
        A2 = Q6_Vw_vrmpyacc_VwVubVb(A2, vand_mask0f(q1_), (pe)[1]);            \
        A3 = Q6_Vw_vrmpyacc_VwVubVb(A3, vlsr4b(q1_),      (po)[1]);            \
        A0 = Q6_Vw_vrmpyacc_VwVubVb(A0, vand_mask0f(q2_), (pe)[2]);            \
        A1 = Q6_Vw_vrmpyacc_VwVubVb(A1, vlsr4b(q2_),      (po)[2]);            \
        A2 = Q6_Vw_vrmpyacc_VwVubVb(A2, vand_mask0f(q3_), (pe)[3]);            \
        A3 = Q6_Vw_vrmpyacc_VwVubVb(A3, vlsr4b(q3_),      (po)[3]);            \
        A0 = Q6_Vw_vrmpyacc_VwVubVb(A0, vand_mask0f(q4_), (pe)[4]);            \
        A1 = Q6_Vw_vrmpyacc_VwVubVb(A1, vlsr4b(q4_),      (po)[4]);            \
        A2 = Q6_Vw_vrmpyacc_VwVubVb(A2, vand_mask0f(q5_), (pe)[5]);            \
        A3 = Q6_Vw_vrmpyacc_VwVubVb(A3, vlsr4b(q5_),      (po)[5]);            \
        A0 = Q6_Vw_vrmpyacc_VwVubVb(A0, vand_mask0f(q6_), (pe)[6]);            \
        A1 = Q6_Vw_vrmpyacc_VwVubVb(A1, vlsr4b(q6_),      (po)[6]);            \
        A2 = Q6_Vw_vrmpyacc_VwVubVb(A2, vand_mask0f(q7_), (pe)[7]);            \
        A3 = Q6_Vw_vrmpyacc_VwVubVb(A3, vlsr4b(q7_),      (po)[7]);            \
    } while (0)

static void gemv_rows_m1_k2048_range(const gbuf_t *b, const int8_t *ae, const int8_t *ao,
                                     int32_t rowbias, int r0, int r1, int32_t *acc)
{
    const size_t ROW = (size_t)(b->nvec * 128);     /* K/2 bytes */
    const HVX_Vector *Ae = (const HVX_Vector *)ae;
    const HVX_Vector *Ao = (const HVX_Vector *)ao;
    const uint8_t *base = b->w;
    const int nvec = b->nvec;
    const int N = r1;
    const int pf_rows = g_usepf ? (g_pf_kb * 1024) / (int)ROW : 0;
    int r;

    if (pf_rows > 0) {
        int p;
        for (p = r0; p < r0 + pf_rows && p + 8 <= N; p += 8)
            pf_window(base + (size_t)p * ROW, 8 * (int)ROW);
    }

    r = r0;
    for (; r + g_rows <= N; r += g_rows) {
        const HVX_Vector *p0 = (const HVX_Vector *)(base + (size_t)r * ROW);
        HVX_Vector e0 = Q6_V_vzero(), e1 = Q6_V_vzero();
        HVX_Vector e2 = Q6_V_vzero(), e3 = Q6_V_vzero();
        int c = 0;

        if (pf_rows > 0 && (r & 7) == 0 && r + pf_rows + 8 <= N)
            pf_window(base + (size_t)(r + pf_rows) * ROW, 8 * (int)ROW);

        if (g_rows == 1) {
            for (; c + 8 <= nvec; c += 8)
                ROW8P(p0 + c, Ae + c, Ao + c, e0, e1, e2, e3);
            for (; c < nvec; c++) {
                HVX_Vector v = p0[c];
                e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, vand_mask0f(v), Ae[c]);
                e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, vlsr4b(v),      Ao[c]);
            }
            acc[r] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e0, e1),
                                            Q6_Vw_vadd_VwVw(e2, e3))) + rowbias;
        } else {
            const HVX_Vector *p1 = (const HVX_Vector *)(base + (size_t)(r + 1) * ROW);
            HVX_Vector f0 = Q6_V_vzero(), f1 = Q6_V_vzero();
            HVX_Vector f2 = Q6_V_vzero(), f3 = Q6_V_vzero();
            for (; c + 8 <= nvec; c += 8) {
                ROW8P(p0 + c, Ae + c, Ao + c, e0, e1, e2, e3);
                ROW8P(p1 + c, Ae + c, Ao + c, f0, f1, f2, f3);
            }
            for (; c < nvec; c++) {
                HVX_Vector v0 = p0[c], v1 = p1[c];
                e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, vand_mask0f(v0), Ae[c]);
                e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, vlsr4b(v0),      Ao[c]);
                f0 = Q6_Vw_vrmpyacc_VwVubVb(f0, vand_mask0f(v1), Ae[c]);
                f1 = Q6_Vw_vrmpyacc_VwVubVb(f1, vlsr4b(v1),      Ao[c]);
            }
            acc[r]     = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e0, e1),
                                                Q6_Vw_vadd_VwVw(e2, e3))) + rowbias;
            acc[r + 1] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(f0, f1),
                                                Q6_Vw_vadd_VwVw(f2, f3))) + rowbias;
        }
    }
    for (; r < N; r++) {
        const HVX_Vector *p0 = (const HVX_Vector *)(base + (size_t)r * ROW);
        HVX_Vector e0 = Q6_V_vzero(), e1 = Q6_V_vzero();
        HVX_Vector e2 = Q6_V_vzero(), e3 = Q6_V_vzero();
        int c;
        for (c = 0; c + 8 <= nvec; c += 8)
            ROW8P(p0 + c, Ae + c, Ao + c, e0, e1, e2, e3);
        for (; c < nvec; c++) {
            HVX_Vector v = p0[c];
            e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, vand_mask0f(v), Ae[c]);
            e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, vlsr4b(v),      Ao[c]);
        }
        acc[r] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e0, e1),
                                        Q6_Vw_vadd_VwVw(e2, e3))) + rowbias;
    }
}

/* ---------------- M=1, int8 weights (offset-128, no unpack) ---------------- */

#define ROW8I(ptr, A0, A1, A2, A3, Av) do {                                   \
        HVX_Vector t0_ = (ptr)[0], t1_ = (ptr)[1], t2_ = (ptr)[2], t3_ = (ptr)[3]; \
        A0 = Q6_Vw_vrmpyacc_VwVubVb(A0, t0_, (Av)[0]);                        \
        A1 = Q6_Vw_vrmpyacc_VwVubVb(A1, t1_, (Av)[1]);                        \
        A2 = Q6_Vw_vrmpyacc_VwVubVb(A2, t2_, (Av)[2]);                        \
        A3 = Q6_Vw_vrmpyacc_VwVubVb(A3, t3_, (Av)[3]);                        \
    } while (0)

static void gemv_rows_m1_i8(const gbuf_t *b, const int8_t *act, int32_t rowbias,
                            int32_t *acc)
{
    const size_t ROW = (size_t)b->K;            /* K bytes per row */
    const HVX_Vector *A = (const HVX_Vector *)act;
    const uint8_t *base = b->w;
    const int N = b->N;
    const int nv = b->K / 128;                  /* 128 int8 per vector */
    const int pf_bytes = g_usepf ? g_pf_kb * 1024 : 0;
    int pf_rows = pf_bytes / (int)ROW;
    int r;

    if (pf_rows > 0) {
        int p;
        for (p = 0; p < pf_rows && p + 4 <= N; p += 4)
            pf_window(base + (size_t)p * ROW, 4 * (int)ROW);
    }
    for (r = 0; r < N; r++) {
        const HVX_Vector *wp = (const HVX_Vector *)(base + (size_t)r * ROW);
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        int c;
        if (pf_rows > 0 && (r & 3) == 0) {
            int a = r + pf_rows;
            if (a + 4 <= N) pf_window(base + (size_t)a * ROW, 4 * (int)ROW);
        }
        for (c = 0; c + 4 <= nv; c += 4) {
            ROW8I(wp + c, a0, a1, a2, a3, A + c);
        }
        for (; c < nv; c++) {
            a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, wp[c], A[c]);
        }
        acc[r] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1),
                                        Q6_Vw_vadd_VwVw(a2, a3))) + rowbias;
    }
}

/* ---------------- M=2 (constant-index unroll) ---------------- */

#define M2_STEP(k) do {                                                       \
        HVX_Vector v_ = wp[k];                                                \
        HVX_Vector lo_ = vand_mask0f(v_);                                     \
        HVX_Vector hi_ = vlsr4b(v_);                                          \
        e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, lo_, Ae0[k]);                         \
        o0 = Q6_Vw_vrmpyacc_VwVubVb(o0, hi_, Ao0[k]);                         \
        e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, lo_, Ae1[k]);                         \
        o1 = Q6_Vw_vrmpyacc_VwVubVb(o1, hi_, Ao1[k]);                         \
    } while (0)

#define M2_STEP8(c) do { M2_STEP((c)+0); M2_STEP((c)+1); M2_STEP((c)+2); M2_STEP((c)+3); \
                         M2_STEP((c)+4); M2_STEP((c)+5); M2_STEP((c)+6); M2_STEP((c)+7); } while (0)

static void gemv_rows_m2(const gbuf_t *b, const int8_t *ae, const int8_t *ao,
                         int32_t bias0, int32_t bias1, int32_t *acc)
{
    const int nvec = b->nvec;
    const size_t half = (size_t)nvec * 128u;
    const HVX_Vector *Ae0 = (const HVX_Vector *)ae;
    const HVX_Vector *Ao0 = (const HVX_Vector *)ao;
    const HVX_Vector *Ae1 = (const HVX_Vector *)(ae + half);
    const HVX_Vector *Ao1 = (const HVX_Vector *)(ao + half);
    const int pfr = g_usepf ? (g_pf_kb * 1024) / (int)half : 0;
    int r;

    if (pfr) pf_rows(b->w, 0, pfr, (int)half, b->N);

    for (r = 0; r < b->N; r++) {
        const HVX_Vector *wp = (const HVX_Vector *)(b->w + (size_t)r * half);
        HVX_Vector e0 = Q6_V_vzero(), o0 = Q6_V_vzero();
        HVX_Vector e1 = Q6_V_vzero(), o1 = Q6_V_vzero();
        int c;
        if (pfr && (r & 7) == 0) pf_rows(b->w, r + pfr, 8, (int)half, b->N);
        if (nvec == 8) {
            M2_STEP8(0);
        } else {
            for (c = 0; c < nvec; c++) {
                HVX_Vector v  = wp[c];
                HVX_Vector lo = vand_mask0f(v);
                HVX_Vector hi = vlsr4b(v);
                e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, lo, Ae0[c]);
                o0 = Q6_Vw_vrmpyacc_VwVubVb(o0, hi, Ao0[c]);
                e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, lo, Ae1[c]);
                o1 = Q6_Vw_vrmpyacc_VwVubVb(o1, hi, Ao1[c]);
            }
        }
        acc[r]        = vsum_w(Q6_Vw_vadd_VwVw(e0, o0)) + bias0;
        acc[b->N + r] = vsum_w(Q6_Vw_vadd_VwVw(e1, o1)) + bias1;
    }
}

/* ---------------- M=4 ---------------- */

static void gemv_rows_m4(const gbuf_t *b, const int8_t *ae, const int8_t *ao,
                         const int32_t *bias, int32_t *acc)
{
    const int nvec = b->nvec;
    const size_t half = (size_t)nvec * 128u;
    const HVX_Vector *Ae0 = (const HVX_Vector *)ae;
    const HVX_Vector *Ao0 = (const HVX_Vector *)ao;
    const HVX_Vector *Ae1 = (const HVX_Vector *)(ae + half);
    const HVX_Vector *Ao1 = (const HVX_Vector *)(ao + half);
    const HVX_Vector *Ae2 = (const HVX_Vector *)(ae + 2 * half);
    const HVX_Vector *Ao2 = (const HVX_Vector *)(ao + 2 * half);
    const HVX_Vector *Ae3 = (const HVX_Vector *)(ae + 3 * half);
    const HVX_Vector *Ao3 = (const HVX_Vector *)(ao + 3 * half);
    const int pf4 = g_usepf ? (g_pf_kb * 1024) / (int)half : 0;
    int r;

    if (pf4) pf_rows(b->w, 0, pf4, (int)half, b->N);

    for (r = 0; r < b->N; r++) {
        const HVX_Vector *wp = (const HVX_Vector *)(b->w + (size_t)r * half);
        if (pf4 && (r & 7) == 0) pf_rows(b->w, r + pf4, 8, (int)half, b->N);
        HVX_Vector e0 = Q6_V_vzero(), o0 = Q6_V_vzero();
        HVX_Vector e1 = Q6_V_vzero(), o1 = Q6_V_vzero();
        HVX_Vector e2 = Q6_V_vzero(), o2 = Q6_V_vzero();
        HVX_Vector e3 = Q6_V_vzero(), o3 = Q6_V_vzero();
        int c;
        for (c = 0; c < nvec; c++) {
            HVX_Vector v  = wp[c];
            HVX_Vector lo = vand_mask0f(v);
            HVX_Vector hi = vlsr4b(v);
            e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, lo, Ae0[c]);
            o0 = Q6_Vw_vrmpyacc_VwVubVb(o0, hi, Ao0[c]);
            e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, lo, Ae1[c]);
            o1 = Q6_Vw_vrmpyacc_VwVubVb(o1, hi, Ao1[c]);
            e2 = Q6_Vw_vrmpyacc_VwVubVb(e2, lo, Ae2[c]);
            o2 = Q6_Vw_vrmpyacc_VwVubVb(o2, hi, Ao2[c]);
            e3 = Q6_Vw_vrmpyacc_VwVubVb(e3, lo, Ae3[c]);
            o3 = Q6_Vw_vrmpyacc_VwVubVb(o3, hi, Ao3[c]);
        }
        acc[r]            = vsum_w(Q6_Vw_vadd_VwVw(e0, o0)) + bias[0];
        acc[b->N + r]     = vsum_w(Q6_Vw_vadd_VwVw(e1, o1)) + bias[1];
        acc[2 * b->N + r] = vsum_w(Q6_Vw_vadd_VwVw(e2, o2)) + bias[2];
        acc[3 * b->N + r] = vsum_w(Q6_Vw_vadd_VwVw(e3, o3)) + bias[3];
    }
}

/* dequant int32 -> int8. qv[i] holds the Q16 multiplier (any halfword).
 * The reference expression is 64-bit: ((int64)acc*q + 32768) >> 16, clamped to
 * +-127 (conv_pack_i8() in myelin/dsp/conv/conv_math.h, and the host models).
 * mulq16(acc,q) resolves acc with a step of 256, i.e. ~4 LSB of 127 at
 * m ~ 0.015, so that shortcut is not allowed.
 *
 * Vector form: q <= 65535 fits an unsigned halfword, so
 *   vmpyie(Vu.w, Vv.uh) ->  acc*q  (exact 32-bit product)
 * is bit-identical to the 64-bit product as long as |acc|*q < 2^30; the
 * rounding add and the arithmetic shift then give the same value as the
 * scalar code.  The guard is computed once per call from max|acc| and max q
 * (both cheap: N/32 vector ops); when it fails (m near 1 with a large
 * accumulator, e.g. some FFN down matrices) the scalar 64-bit loop runs.
 * Same code as in myelin/dsp/conv/lfmgemv_impl.c, where 196 us became 16. */
static void pack_out_i8(const int32_t *acc, const int32_t *qv, int8_t *out, int n)
{
    HVX_Vector qmask = Q6_V_vsplat_R(0xffff);
    HVX_Vector vrnd  = Q6_V_vsplat_R(32768);
    HVX_Vector vlo   = Q6_V_vsplat_R(-127);
    HVX_Vector vhi   = Q6_V_vsplat_R(127);
    int32_t amax = 0, qmax = 0, t32[32] __attribute__((aligned(128)));
    int i, j, fast;
    {
        HVX_Vector vmax = Q6_V_vzero(), vmin = Q6_V_vzero(), vq = Q6_V_vzero();
        for (i = 0; i + 32 <= n; i += 32) {
            HVX_Vector a = *(const HVX_Vector *)(acc + i);
            HVX_Vector q = Q6_V_vand_VV(*(const HVX_Vector *)(qv + i), qmask);
            vmax = Q6_Vw_vmax_VwVw(vmax, a);
            vmin = Q6_Vw_vmin_VwVw(vmin, a);
            vq   = Q6_Vw_vmax_VwVw(vq, q);
        }
        *(HVX_Vector *)t32 = vmax;
        for (j = 0; j < 32; j++) if (t32[j] > amax) amax = t32[j];
        *(HVX_Vector *)t32 = vmin;
        for (j = 0; j < 32; j++) if (-t32[j] > amax) amax = -t32[j];
        *(HVX_Vector *)t32 = vq;
        for (j = 0; j < 32; j++) if (t32[j] > qmax) qmax = t32[j];
    }
    /* acc/qv are 128 B aligned by set_shape(); the output pointer is not
     * (FastRPC hands us an arbitrary byte offset), so it is checked here. */
    fast = ((int64_t)amax * (int64_t)qmax <= (int64_t)1 << 30)
        && (((uintptr_t)out & 127u) == 0);

    i = 0;
    if (fast) {
        for (; i + 128 <= n; i += 128) {
            HVX_Vector r0, r1, r2, r3;
            /* qv holds q in both halfwords, so the odd-halfword form is exact */
            r0 = Q6_Vw_vmpyie_VwVuh(*(const HVX_Vector *)(acc + i + 0),
                                     *(const HVX_Vector *)(qv + i + 0));
            r1 = Q6_Vw_vmpyie_VwVuh(*(const HVX_Vector *)(acc + i + 32),
                                     *(const HVX_Vector *)(qv + i + 32));
            r2 = Q6_Vw_vmpyie_VwVuh(*(const HVX_Vector *)(acc + i + 64),
                                     *(const HVX_Vector *)(qv + i + 64));
            r3 = Q6_Vw_vmpyie_VwVuh(*(const HVX_Vector *)(acc + i + 96),
                                     *(const HVX_Vector *)(qv + i + 96));
            r0 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(r0, vrnd), 16), vlo), vhi);
            r1 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(r1, vrnd), 16), vlo), vhi);
            r2 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(r2, vrnd), 16), vlo), vhi);
            r3 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(r3, vrnd), 16), vlo), vhi);
            /* vpack writes its *second* operand into the low half of the
             * result, i.e. vpack(Vhi, Vlo) -- verified in hexagon-sim. */
            *(HVX_Vector *)(out + i) =
                Q6_Vb_vpack_VhVh_sat(Q6_Vh_vpack_VwVw_sat(r3, r2),
                                     Q6_Vh_vpack_VwVw_sat(r1, r0));
        }
    }
    for (; i < n; i++) {
        int32_t q = (int32_t)((uint32_t)qv[i] & 0xffffu);
        int64_t v = ((int64_t)acc[i] * (int64_t)q + 32768) >> 16;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }
}

#define ATTN_HQ   16
#define ATTN_KV   4
#define ATTN_D    128
#define ATTN_PAGE 32

typedef struct {
    const gbuf_t *k, *kt, *v;
    const int8_t *q;
    int8_t  *qe, *qo;             /* deinterleaved q, 16 x 64 */
    int32_t *sumq;
    int32_t *scores;              /* 16 x n */
    int32_t  mx[ATTN_HQ], ls[ATTN_HQ];
    int32_t *acc;                 /* 16 x 128, natural d order */
    const int *tbl;
    int       t0, n;
    int       sk, sv, skt;        /* K / V / K-scale row strides in bytes */
    int32_t   uq[ATTN_HQ][16][2]; /* per-head, per-dim-group multiplier words */
} actx_t;

/* ---------------- attention step trace ----------------
 * The worker used to die silently on a second HVX context (rc=78, no clue
 * where).  Job boundaries now leave a small ring of (stage, arg) words that
 * the host reads only when a call fails: overhead is a dozen integer stores
 * per attention call, nothing in the inner loops. */
#define ATR_PUB   1   /* main: jobs published  (arg = threads)   */
#define ATR_WAKE  2   /* worker: sem_down       (arg = idx)      */
#define ATR_LOCK  3   /* worker: hvx_lock       (arg = idx)      */
#define ATR_P1    4   /* pass 1 done            (arg = k-group)  */
#define ATR_P2    5   /* pass 2 done            (arg = k-group)  */
#define ATR_SYN   6   /* worker: mem_syncht     (arg = idx)      */
#define ATR_UNL   7   /* worker: hvx_unlock     (arg = idx)      */
#define ATR_DONE  8   /* worker: done_seq       (arg = idx)      */
#define ATR_SEM   9   /* worker: sem_up(done)   (arg = idx)      */
#define ATR_JOIN 10   /* main: join ok          (arg = idx)      */
#define ATR_FB   11   /* main: range recomputed (arg = idx)      */

static volatile int g_atr_stage[64];
static volatile int g_atr_arg[64];
static volatile int g_atr_i;

static inline void atr(int stage, int arg)
{
    int i = g_atr_i & 63;
    g_atr_stage[i] = stage;
    g_atr_arg[i] = arg;
    g_atr_i = (g_atr_i + 1) & 1023;
}

/* ---------------- HVX worker team ---------------- */

typedef struct {
    qurt_thread_t tid;
    qurt_sem_t    go;
    qurt_sem_t    done;
    int           idx;
    volatile int  shutdown;
    void         *stack;    /* must be supplied explicitly, else create fails */
} thw_t;

typedef struct {
    volatile unsigned seq;   /* publication handshake, see team_kick()/team_join() */
    int           kind;      /* 0 = gemv rows, 1 = silu range, 2 = attention groups */
    const gbuf_t *b;
    const int8_t *ae, *ao;
    int32_t       bias;
    int           r0, r1;
    volatile int  done_seq;
    int32_t      *acc;
    void         *ctx;       /* attention context (kind 2) */
} m1job_t;

typedef struct {
    int     state;        /* 0 = not started, 1 = running */
    int     nwork;        /* worker threads (main thread not counted) */
    int     hvx_rc;
    int     thr_rc;
    int     ncreated;
    thw_t   w[8];
    m1job_t job[8];
} team_t;

static team_t g_team;
static int    g_team_warm;   /* workers have run one (empty) job */

static void worker_body(thw_t *w) __attribute__((noinline));
static void silu_range(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                       int32_t mq_mid);
static void attn_group(int kh);
static void attn_group_tm(actx_t *ctx, int kh);

/* Publish the jobs written by the caller and kick the workers.  The extra
 * sequence bump makes the handshake in worker_body() independent of any
 * ordering assumption about qurt_sem_up(). */
static void team_kick(int total)
{
    int i;
    for (i = 1; i < total; i++) g_team.job[i].seq++;
    if (total > 1 && g_team.job[1].kind == 2) atr(ATR_PUB, total);
    __sync_synchronize();
    for (i = 1; i < total; i++) qurt_sem_up(&g_team.w[i].go);
}

static int g_team_fallbacks;   /* jobs the team did not publish (see below) */

/* Wait until worker i has published the *result* of the job we kicked.  The
 * done_seq store happens in the worker after the kernel and its syncht, so
 * this also catches a premature qurt_sem_down() wakeup (observed on this
 * target: the caller started reading acc[] with the worker's range still in
 * progress and got ~960 stale rows).  If a worker still has not published
 * after a bounded spin, its rows are recomputed on the main thread: the
 * result stays correct and the event is counted (team_info/times). */
static void team_join(int total)
{
    int i;
    for (i = 1; i < total; i++) {
        m1job_t *j = &g_team.job[i];
        unsigned spin = 0;
        qurt_sem_down(&g_team.w[i].done);
        while (j->done_seq != j->seq && ++spin < (1u << 20)) __sync_synchronize();
        if (j->kind == 2 && j->done_seq == j->seq) atr(ATR_JOIN, i);
        if (j->done_seq != j->seq) {
            g_team_fallbacks++;
            if (j->kind == 2) atr(ATR_FB, i);
            if (j->kind == 0)
                gemv_rows_m1_k2048_range(j->b, j->ae, j->ao, j->bias, j->r0, j->r1, j->acc);
            else if (j->kind == 1)
                silu_range(j->r0, j->r1, j->b, (const gbuf_t *)j->ae, j->bias);
            else if (j->kind == 2) {
                int kg;
                qurt_hvx_lock(QURT_HVX_MODE_128B);
                for (kg = j->r0; kg < j->r1; kg++) {
                    if (j->ctx) attn_group_tm((actx_t *)j->ctx, kg);
                    else        attn_group(kg);
                }
                atr(ATR_SYN, i);
                qurt_mem_syncht();
                qurt_hvx_unlock();
                atr(ATR_UNL, i);
            }
        }
    }
}

/* thin, non-HVX entry: creating a thread whose entry prologue touches HVX
 * registers makes qurt_thread_create fail with -1 on this target */
void team_worker(void *a)
{
    worker_body((thw_t *)a);
}

static void worker_body(thw_t *w)
{
    for (;;) {
        m1job_t *j;
        qurt_sem_down(&w->go);
        __sync_synchronize();          /* publish the job written by the caller */
        if (w->shutdown) break;
        /* Wait for the sequence number of *this* job.  Without the explicit
         * handshake the first job after team_start() was sometimes not
         * visible to the worker on this target: it then ran with the stale
         * (zero) range and the caller's rows stayed untouched. */
        j = &g_team.job[w->idx];
        while (j->seq == j->done_seq) __sync_synchronize();
        /* lock per job: holding a context for the thread's lifetime blocks
         * other workers when there are fewer contexts than threads */
        {
            if (j->kind == 2) {
                int kg;
                atr(ATR_WAKE, w->idx);
                qurt_hvx_lock(QURT_HVX_MODE_128B);
                atr(ATR_LOCK, w->idx);
                if (j->ctx) {
                    /* fully local context: nothing but the job is shared */
                    for (kg = j->r0; kg < j->r1; kg++) attn_group_tm((actx_t *)j->ctx, kg);
                } else {
                    for (kg = j->r0; kg < j->r1; kg++) attn_group(kg);
                }
                qurt_mem_syncht();
                qurt_hvx_unlock();
            } else if (j->kind == 1) {
                silu_range(j->r0, j->r1, j->b, (const gbuf_t *)j->ae, j->bias);
            } else {
                qurt_hvx_lock(QURT_HVX_MODE_128B);
                gemv_rows_m1_k2048_range(j->b, j->ae, j->ao, j->bias, j->r0, j->r1, j->acc);
                /* drain the HVX store queue before publishing the result:
                 * qurt_hvx_unlock() alone does not, and a reader on another
                 * thread can then see stale acc[] rows (measured: the tail of
                 * the worker's range was 0/old on the fast vectorised pack). */
                qurt_mem_syncht();
                qurt_hvx_unlock();
            }
        }
        __sync_synchronize();
        j->done_seq = j->seq;          /* result published: see team_join() */
        if (j->kind == 2) atr(ATR_DONE, w->idx);
        __sync_synchronize();
        qurt_sem_up(&w->done);
        if (j->kind == 2) atr(ATR_SEM, w->idx);
    }
}

/* ntotal = main thread + workers; returns workers actually created */
static int team_start(int ntotal)
{
    int i, made = 0;

    if (ntotal < 2) return 0;
    {
        int units = (int)((qurt_hvx_get_units() >> 8) & 0xFF);
        /* leave one HVX context for the main thread, and never create more
         * worker threads than contexts: measured 3-4 threads on 2 contexts
         * are *slower* (13.8-14.6 GB/s vs 20.0) */
        if (units > 1 && ntotal - 1 > units - 1) ntotal = units;
        if (units >= 1 && ntotal > units + 1) ntotal = units + 1;
        if (ntotal > 4) ntotal = 4;
    }
    if (ntotal > 8) ntotal = 8;
    if (ntotal < 2) return 0;
    if (g_team.state) return g_team.nwork;

    g_team.hvx_rc = 0x7e57;   /* reserve skipped: fails (-1) and appears to break create */

    for (i = 1; i < ntotal; i++) {
        qurt_thread_attr_t attr;
        thw_t *w = &g_team.w[i];
        qurt_sem_init(&w->go);
        qurt_sem_init(&w->done);
        w->idx = i;
        w->shutdown = 0;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "gemvhvx");
        if (!w->stack) w->stack = memalign(128, 96 * 1024);
        qurt_thread_attr_set_stack_addr(&attr, w->stack);
        qurt_thread_attr_set_stack_size(&attr, 96 * 1024);
        {
            int prio = qurt_thread_get_priority(qurt_thread_get_id());
            if (prio < 1) prio = 1;
            qurt_thread_attr_set_priority(&attr, (unsigned short)prio);
        }
        {
            int crc = qurt_thread_create(&w->tid, &attr, team_worker, w);
            g_team.thr_rc = crc;
            if (crc == 0) made++;
            else break;
        }
    }
    g_team.nwork = made;
    g_team.ncreated = made;
    g_team.state = 1;
    return made;
}

/* split rows [0,N) over (nwork+1) threads; rows aligned to 4 */
static void run_split_m1(const gbuf_t *b, const int8_t *ae, const int8_t *ao,
                         int32_t bias, int32_t *acc)
{
    int total = g_team.nwork + 1;
    int i;

    if (g_threads > 1 && g_team.state == 0) team_start(g_threads);
    /* Let every worker run once (empty job, no stores) before the first real
     * job, so its HVX context exists before we depend on it.  Correctness does
     * not rely on this: team_join() verifies each result and recomputes the
     * range if the worker did not publish it. */
    if (!g_team_warm && g_team.nwork > 0) {
        int i;
        g_team_warm = 1;
        for (i = 1; i <= g_team.nwork; i++) {
            m1job_t *jw = &g_team.job[i];
            jw->kind = 0; jw->b = b; jw->ae = ae; jw->ao = ao;
            jw->bias = 0; jw->r0 = 0; jw->r1 = 0; jw->acc = acc;
        }
        team_kick(g_team.nwork + 1);
        team_join(g_team.nwork + 1);
    }
    total = g_team.nwork + 1;
    if (g_threads < total) total = g_threads;
    if (total <= 1 || b->N < 512) {
        gemv_rows_m1_k2048_range(b, ae, ao, bias, 0, b->N, acc);
        return;
    }
    for (i = 0; i < total; i++) {
        int r0 = (int)((int64_t)b->N * i / total) & ~3;
        int r1 = (int)((int64_t)b->N * (i + 1) / total) & ~3;
        g_team.job[i].kind = 0;
        g_team.job[i].b = b; g_team.job[i].ae = ae; g_team.job[i].ao = ao;
        g_team.job[i].bias = bias; g_team.job[i].r0 = r0; g_team.job[i].r1 = r1;
        g_team.job[i].acc = acc;
    }
    g_team.job[total - 1].r1 = b->N;
    team_kick(total);
    {
        m1job_t *j = &g_team.job[0];
        gemv_rows_m1_k2048_range(j->b, j->ae, j->ao, j->bias, j->r0, j->r1, j->acc);
    }
    team_join(total);
    __sync_synchronize();
}

/* ---------------- SiLU (integer, LUT + linear interpolation) ---------------- */

static inline int32_t silu_q16(int32_t zq)
{
    int32_t off, idx, frac, a, b;

    if (zq > 8 * 65536) zq = 8 * 65536;
    if (zq < -8 * 65536) zq = -8 * 65536;
    off = zq + 8 * 65536;          /* 0 .. 16*65536 */
    idx = off >> 10;               /* step 1/64 = 1024 in Q16 */
    frac = off & 1023;
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> 10);   /* |b-a|<2^13, frac<2^10 -> int32 */
}

/* SiLU for an argument in Q(16-k), k in [0,8]; k = 0 is exactly silu_q16().
 * The LUT argument grid is 1/64, i.e. 1024 in Q16 and 1024>>k in Q(16-k),
 * so idx/frac just move to a coarser fixed-point scale. */
static inline int32_t silu_qk(int32_t zq, int k)
{
    int32_t lim, step, off, idx, frac, a, b;

    if (k <= 0) return silu_q16(zq);
    lim  = 8 << (16 - k);
    step = 1024 >> k;
    if (zq > lim) zq = lim;
    if (zq < -lim) zq = -lim;
    off = zq + lim;
    idx = off >> (10 - k);
    frac = off & (step - 1);
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> (10 - k));
}

/* Mid-buffer requant: pr is Q16 of the real product P = silu(gate)*up, and
 * P/s_mid = pr*mq_mid/2^28 (mq_mid = 4096/s_mid).  The product must be taken
 * in 64 bits.  The former `mulq16(pr >> 12, mq_mid)` resolves pr with a step
 * of 2^12, i.e. P with a step of 1/16, which is 1/(16*s_mid) of a mid-buffer
 * LSB: 5 LSB at the measured layer-0 scale s_mid = 0.0127, and it zeroes
 * every lane with P < 16 outright.  Measured on the real LFM3-1.7B weights
 * (myelin/dsp/hvx/silu_err_study.c, 24 tokens x 7680 lanes): literal mulq16
 * leaves 100% of the lanes at 0, cos(y_ffn) = 0.47 vs 0.995 for this form;
 * int64 on the already-shifted product is still 3-5 LSB / cos = 0.73.
 * Same precision rule as pack_out_i8(). */
static inline int32_t requant_mid(int32_t pr, int32_t mq_mid)
{
    int64_t v = ((int64_t)pr * (int64_t)mq_mid + ((int64_t)1 << 27)) >> 28;
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int32_t)v;
}

/* ---------------- SiLU vector path (HVX, v68) ----------------
 *
 * The scalar chain is bit-exact only with int64 products:
 *   deq_wide   (acc*qm  + 2^15) >> 16
 *   prod_p     (sv *zu  + 2^15) >> 16
 *   requant_mid(pr *mq  + 2^27) >> 28
 * v68 has no 32x32->64 vector multiply and no Q6_Vd_* ops at all, so each
 * product is split into four 16x16 partial products that stay in int32:
 *
 *   a = ahi*2^16 + alo,  b = bhi*2^16 + blo   (ahi/bhi signed, alo/blo unsigned)
 *   a*b = ahi*bhi*2^32 + (ahi*blo + alo*bhi)*2^16 + alo*blo
 *
 * ahi*bhi is small, alo*blo is exact as an *unsigned* 32-bit value (the
 * rounding add never wraps), and the cross terms fit int32 while
 * |a*b| <= 2^45 -- the per-call guard (silu_vec_guard) checks that bound on
 * the live buffers; outside it the scalar loop takes over.
 *
 * The LUT part stays scalar: vgather raises a precise exception in this PD
 * and v68 has no vector float, so there is no vector lookup to be had; it
 * still costs ~27 of the 71 cycles/lane of the scalar chain (measured), the
 * rest is the three int64 products.
 */
#define SILU_ZLIM   (8 * 65536)

typedef struct { HVX_Vector hi, cross, low; } vsplit_t;

static inline vsplit_t vsplit_mul(HVX_Vector a, HVX_Vector b)
{
    const HVX_Vector z   = Q6_V_vzero();
    const HVX_Vector m16 = Q6_V_vsplat_R(0xffff);
    HVX_Vector ahi = Q6_Vw_vasr_VwR(a, 16);
    HVX_Vector alo = Q6_V_vand_VV(a, m16);
    HVX_Vector bhi = Q6_Vw_vasr_VwR(b, 16);
    HVX_Vector blo = Q6_V_vand_VV(b, m16);
    vsplit_t s;
    /* signed x signed (low halfword), unsigned x unsigned -> one vmpy each */
    s.hi    = Q6_Vw_vmpyieacc_VwVwVh(z, ahi, bhi);
    s.cross = Q6_Vw_vadd_VwVw(Q6_Vw_vmpyie_VwVuh(ahi, blo),
                              Q6_Vw_vmpyieacc_VwVwVh(z, alo, bhi));
    s.low   = Q6_Vw_vmpyie_VwVuh(alo, blo);      /* exact as unsigned */
    return s;
}

/* (a*b + 32768) >> 16, floor semantics of the scalar code */
static inline HVX_Vector vmul_q16_rnd(HVX_Vector a, HVX_Vector b)
{
    vsplit_t s = vsplit_mul(a, b);
    HVX_Vector lo = Q6_Vw_vadd_VwVw(s.low, Q6_V_vsplat_R(32768));
    HVX_Vector v  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(s.hi, 16), s.cross);
    return Q6_Vw_vadd_VwVw(v, Q6_Vuw_vlsr_VuwR(lo, 16));
}

/* (a*b) >> 16 (floor) */
static inline HVX_Vector vmul_q16_floor(HVX_Vector a, HVX_Vector b)
{
    vsplit_t s = vsplit_mul(a, b);
    HVX_Vector v = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(s.hi, 16), s.cross);
    return Q6_Vw_vadd_VwVw(v, Q6_Vuw_vlsr_VuwR(s.low, 16));
}

/* Branch-free silu over the clamped argument; the LUT copy carries a sentinel
 * in entry N so idx+1 is always in range (idx == N-1 returns LUT[N-1] because
 * sentinel == LUT[N-1], exactly like the scalar early return). */
static int32_t silu_lut_pad[SILU_LUT_N + 1] __attribute__((aligned(128)));

static int silu_lut_ready;

/* Called on the main thread before the workers enter silu_range(). */
static void silu_lut_prep(void)
{
    if (silu_lut_ready) return;
    memcpy(silu_lut_pad, silu_lut, sizeof(silu_lut));
    silu_lut_pad[SILU_LUT_N] = silu_lut[SILU_LUT_N - 1];
    silu_lut_ready = 1;
}

static inline int32_t silu_q16_off(int32_t off)
{
    int32_t idx = off >> 10;
    int32_t frac = off & 1023;
    int32_t a = silu_lut_pad[idx];
    int32_t b = silu_lut_pad[idx + 1];
    return a + (((b - a) * frac) >> 10);
}

/* silu_q16() lane-exact.  The 1025-entry LUT lookup stays scalar: v68 vgather
 * raises a precise exception in this PD with its own base address in Bad VA
 * (reproduced from a writable static table and from rodata, both 128-byte
 * aligned), and v68 has no vector float.  The clamp and the offset are
 * vectorised, the lookup is 2 L1 loads + 1 imul per lane. */
static inline HVX_Vector vsilu_q16(HVX_Vector zq)
{
    union { HVX_Vector v; int32_t s[32]; } zin __attribute__((aligned(128)));
    union { HVX_Vector v; int32_t s[32]; } zout __attribute__((aligned(128)));
    const HVX_Vector v8 = Q6_V_vsplat_R(SILU_ZLIM);
    int k;
    zq = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(zq, Q6_Vw_vsub_VwVw(Q6_V_vzero(), v8)), v8);
    zin.v = Q6_Vw_vadd_VwVw(zq, v8);            /* 0 .. 16*65536 */
    for (k = 0; k < 32; k++) zout.s[k] = silu_q16_off(zin.s[k]);
    return zout.v;
}

/* requant_mid() lane-exact: (pr*mq + 2^27) >> 28, clamp +-127 */
static inline HVX_Vector vrequant_mid(HVX_Vector pr, HVX_Vector mq)
{
    HVX_Vector v = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(vmul_q16_floor(pr, mq),
                                                  Q6_V_vsplat_R(2048)), 12);
    v = Q6_Vw_vmax_VwVw(v, Q6_V_vsplat_R(-127));
    return Q6_Vw_vmin_VwVw(v, Q6_V_vsplat_R(127));
}

/* max |x| over 32-bit lanes (i0..i1, unaligned ends handled scalar) */
static int32_t range_absmax(const int32_t *p, int i0, int i1)
{
    const HVX_Vector z = Q6_V_vzero();
    HVX_Vector va = z;
    int32_t t[32] __attribute__((aligned(128)));
    int32_t m = 0;
    int i = i0, j;
    for (; i < i1 && (i & 31); i++) { int32_t v = p[i] < 0 ? -p[i] : p[i]; if (v > m) m = v; }
    for (; i + 32 <= i1; i += 32) {
        HVX_Vector v = *(const HVX_Vector *)(p + i);
        va = Q6_Vw_vmax_VwVw(va, Q6_Vw_vmax_VwVw(v, Q6_Vw_vsub_VwVw(z, v)));
    }
    *(HVX_Vector *)t = va;
    for (j = 0; j < 32; j++) if (t[j] > m) m = t[j];
    for (; i < i1; i++) { int32_t v = p[i] < 0 ? -p[i] : p[i]; if (v > m) m = v; }
    return m;
}

/* min and max over 32-bit lanes */
static void range_minmax(const int32_t *p, int i0, int i1, int32_t *mn, int32_t *mx)
{
    HVX_Vector vx = Q6_V_vsplat_R(-0x7fffffff), vn = Q6_V_vsplat_R(0x7fffffff);
    int32_t t[32] __attribute__((aligned(128)));
    int i = i0, j;
    for (; i < i1 && (i & 31); i++) {
        if (p[i] > *mx) *mx = p[i];
        if (p[i] < *mn) *mn = p[i];
    }
    for (; i + 32 <= i1; i += 32) {
        HVX_Vector v = *(const HVX_Vector *)(p + i);
        vx = Q6_Vw_vmax_VwVw(vx, v);
        vn = Q6_Vw_vmin_VwVw(vn, v);
    }
    *(HVX_Vector *)t = vx;
    for (j = 0; j < 32; j++) if (t[j] > *mx) *mx = t[j];
    *(HVX_Vector *)t = vn;
    for (j = 0; j < 32; j++) if (t[j] < *mn) *mn = t[j];
    for (; i < i1; i++) {
        if (p[i] > *mx) *mx = p[i];
        if (p[i] < *mn) *mn = p[i];
    }
}

/* One lane of the scalar chain, for heads/tails of a split range. */
static inline int8_t silu_lane(int k, const gbuf_t *g, const gbuf_t *u, int32_t mq_mid)
{
    int32_t zg = deq_wide(g->acc[k], g->qm[k]);
    int32_t zu = deq_wide(g_ffn.acc2[k], u->qm[k]);
    int32_t pr = prod_p(silu_qk(zg, g->qshift), zu, u->qshift);
    return (int8_t)requant_mid(pr, mq_mid);
}

/* Guards for the split multiply: the scalar chain is reproduced exactly while
 * every product stays inside the int32 partial-product bounds. */
static int silu_vec_guard(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                          int32_t mq_mid)
{
    int32_t ag = 0, qg = 0, qgn = 0, au = 0, qu = 0, qun = 0;
    int64_t bound_zu, bound_pr;

    if (!g_silu_vec) return 0;
    if (g->qshift != 0 || u->qshift != 0) return 0;
    if (mq_mid <= 0) return 0;
    ag = range_absmax(g->acc, i0, i1);
    au = range_absmax(g_ffn.acc2, i0, i1);
    qgn = 0; qg = 0;
    range_minmax(g->qm, i0, i1, &qgn, &qg);
    qun = 0; qu = 0;
    range_minmax(u->qm, i0, i1, &qun, &qu);
    if (qgn < 0 || qun < 0) return 0;                 /* qm is a positive scale */
    if ((int64_t)ag * (int64_t)qg > ((int64_t)1 << 45)) return 0;
    if ((int64_t)au * (int64_t)qu > ((int64_t)1 << 45)) return 0;
    bound_zu = ((int64_t)au * (int64_t)qu + 32768) >> 16;
    bound_pr = ((int64_t)524271 * bound_zu + 32768) >> 16;
    if (bound_pr * (int64_t)mq_mid > ((int64_t)1 << 46)) return 0;
    return 1;
}

/* 32 lanes of the whole chain, word lanes; qshift == 0 only */
static inline HVX_Vector silu_vec32(const int32_t *ag, const int32_t *qg,
                                    const int32_t *au, const int32_t *qu,
                                    HVX_Vector mq)
{
    HVX_Vector zg = vmul_q16_rnd(*(const HVX_Vector *)ag, *(const HVX_Vector *)qg);
    HVX_Vector zu = vmul_q16_rnd(*(const HVX_Vector *)au, *(const HVX_Vector *)qu);
    HVX_Vector pr = vmul_q16_rnd(vsilu_q16(zg), zu);
    pr = Q6_Vw_vmax_VwVw(pr, Q6_V_vsplat_R(-0x7fffffff));
    pr = Q6_Vw_vmin_VwVw(pr, Q6_V_vsplat_R(0x7fffffff));
    return vrequant_mid(pr, mq);
}

/* Vector SiLU pass; returns 0 if the range must stay scalar. */
static int silu_range_vec(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                          int32_t mq_mid)
{
    HVX_Vector mq = Q6_V_vsplat_R(mq_mid);
    int i;

    if (!silu_vec_guard(i0, i1, g, u, mq_mid)) return 0;

    i = i0;
    for (; i < i1 && (i & 31); i++) g_ffn.h[i] = silu_lane(i, g, u, mq_mid);

    for (; i + 128 <= i1; i += 128) {
        HVX_Vector r0 = silu_vec32(g->acc + i,      g->qm + i,
                                   g_ffn.acc2 + i,      u->qm + i, mq);
        HVX_Vector r1 = silu_vec32(g->acc + i + 32, g->qm + i + 32,
                                   g_ffn.acc2 + i + 32, u->qm + i + 32, mq);
        HVX_Vector r2 = silu_vec32(g->acc + i + 64, g->qm + i + 64,
                                   g_ffn.acc2 + i + 64, u->qm + i + 64, mq);
        HVX_Vector r3 = silu_vec32(g->acc + i + 96, g->qm + i + 96,
                                   g_ffn.acc2 + i + 96, u->qm + i + 96, mq);
        *(HVX_Vector *)(g_ffn.h + i) =
            Q6_Vb_vpack_VhVh_sat(Q6_Vh_vpack_VwVw_sat(r3, r2),
                                 Q6_Vh_vpack_VwVw_sat(r1, r0));
    }
    for (; i + 32 <= i1; i += 32) {
        union { HVX_Vector v; int8_t b[128]; } p;
        HVX_Vector r = silu_vec32(g->acc + i, g->qm + i,
                                  g_ffn.acc2 + i, u->qm + i, mq);
        p.v = Q6_Vb_vpack_VhVh_sat(Q6_V_vzero(),
                                   Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(), r));
        memcpy(g_ffn.h + i, p.b, 32);
    }
    for (; i < i1; i++) g_ffn.h[i] = silu_lane(i, g, u, mq_mid);
    return 1;
}

/* Skip-based scalar stage profile (probe kind 40): every variant runs the same
 * store loop, one chain step is replaced by a trivial op.  Cost(step) is the
 * difference from the full chain, so nothing can be optimised away. */
#define SILU_PROF_LOOP(NAME, ZGE, ZUE, ZE, PRE, VE)                     \
static void NAME(const gbuf_t *g, const gbuf_t *u, int32_t mq, int n)   \
    __attribute__((noinline));                                          \
static void NAME(const gbuf_t *g, const gbuf_t *u, int32_t mq, int n)   \
{                                                                       \
    volatile int32_t s = 0;                                             \
    int k;                                                              \
    for (k = 0; k < n; k++) {                                           \
        int32_t zg = (ZGE), zu = (ZUE), z = (ZE), pr = (PRE);           \
        int32_t v = (VE);                                               \
        g_ffn.h[k] = (int8_t)v;                                         \
        s += v;                                                         \
    }                                                                   \
    g_prof_sink = s;                                                    \
}

SILU_PROF_LOOP(silu_p_all,
               deq_wide(g->acc[k], g->qm[k]),
               deq_wide(g_ffn.acc2[k], u->qm[k]),
               silu_qk(zg, 0),
               prod_p(z, zu, 0),
               requant_mid(pr, mq))
SILU_PROF_LOOP(silu_p_nolut,
               deq_wide(g->acc[k], g->qm[k]),
               deq_wide(g_ffn.acc2[k], u->qm[k]),
               zg,
               prod_p(z, zu, 0),
               requant_mid(pr, mq))
SILU_PROF_LOOP(silu_p_noprod,
               deq_wide(g->acc[k], g->qm[k]),
               deq_wide(g_ffn.acc2[k], u->qm[k]),
               silu_qk(zg, 0),
               (z >> 1) + (zu >> 20),   /* both operands stay live */
               requant_mid(pr, mq))
SILU_PROF_LOOP(silu_p_noreq,
               deq_wide(g->acc[k], g->qm[k]),
               deq_wide(g_ffn.acc2[k], u->qm[k]),
               silu_qk(zg, 0),
               prod_p(z, zu, 0),
               pr >> 10)
SILU_PROF_LOOP(silu_p_nodeq,
               g->acc[k],
               g_ffn.acc2[k],
               silu_qk(zg, 0),
               prod_p(z, zu, 0),
               requant_mid(pr, mq))

static void silu_prof_run(int variant, const gbuf_t *g, const gbuf_t *u,
                          int32_t mq, int n)
{
    switch (variant) {
    case 0: silu_p_all(g, u, mq, n);   break;
    case 1: silu_p_nolut(g, u, mq, n); break;
    case 2: silu_p_noprod(g, u, mq, n); break;
    case 3: silu_p_noreq(g, u, mq, n); break;
    case 4: silu_p_nodeq(g, u, mq, n); break;
    default: break;
    }
}

/* SiLU + int8 quantisation over an index range (scalar fallback, 4x unrolled) */
static void silu_range(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                       int32_t mq_mid)
{
    int i4;

    if (silu_range_vec(i0, i1, g, u, mq_mid)) return;

    i4 = i0;
    for (; i4 + 4 <= i1; i4 += 4) {
        int32_t hv[4];
        int t;
        for (t = 0; t < 4; t++) {
            int k = i4 + t;
            int32_t zg = deq_wide(g->acc[k], g->qm[k]);
            int32_t zu = deq_wide(g_ffn.acc2[k], u->qm[k]);
            int32_t pr = prod_p(silu_qk(zg, g->qshift), zu, u->qshift);
            hv[t] = requant_mid(pr, mq_mid);
        }
        g_ffn.h[i4 + 0] = (int8_t)hv[0];
        g_ffn.h[i4 + 1] = (int8_t)hv[1];
        g_ffn.h[i4 + 2] = (int8_t)hv[2];
        g_ffn.h[i4 + 3] = (int8_t)hv[3];
    }
    for (; i4 < i1; i4++) {
        int32_t zg = deq_wide(g->acc[i4], g->qm[i4]);
        int32_t zu = deq_wide(g_ffn.acc2[i4], u->qm[i4]);
        int32_t pr = prod_p(silu_qk(zg, g->qshift), zu, u->qshift);
        g_ffn.h[i4] = (int8_t)requant_mid(pr, mq_mid);
    }
}

/* split the SiLU pass across the team (it is the largest non-GEMM phase) */
static void run_split_silu(const gbuf_t *g, const gbuf_t *u, int32_t mq_mid)
{
    int total = g_team.nwork + 1;
    int i;
    silu_lut_prep();      /* before the workers can enter silu_range */
    if (g_threads < total) total = g_threads;
    if (total <= 1 || g->N < 2048) {
        silu_range(0, g->N, g, u, mq_mid);
        return;
    }
    for (i = 0; i < total; i++) {
        int i0 = (int)((int64_t)g->N * i / total);
        int i1 = (int)((int64_t)g->N * (i + 1) / total);
        m1job_t *j = &g_team.job[i];
        j->kind = 1;
        j->b = g;
        j->ae = (const int8_t *)u;      /* pass the "up" matrix through */
        j->bias = mq_mid;
        j->r0 = i0;
        j->r1 = i1;
        j->acc = 0;
    }
    team_kick(total);
    silu_range(0, (int)((int64_t)g->N / total), g, u, mq_mid);
    team_join(total);
}


/* ---------------- power ---------------- */

static int g_perf_set;
static void set_perf(void)
{
    HAP_power_request_t req;

    if (g_perf_set) return;
    g_perf_set = 1;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = TRUE;
    HAP_power_set(0, &req);

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    HAP_power_set(0, &req);

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_DUTY_CYCLE_MODE;
    req.dcvs_v3.set_core_params = TRUE;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_SVS2;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.set_bus_params = TRUE;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_SVS2;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
    HAP_power_set(0, &req);

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_mips_bw;
    req.mips_bw.set_mips = TRUE;
    req.mips_bw.mipsTotal = 0x7fffffff;
    req.mips_bw.set_bus_bw = TRUE;
    req.mips_bw.bwBytePerSec = 34000000000ull;
    req.mips_bw.busbwUsagePercentage = 100;
    HAP_power_set(0, &req);
}

int lfmgemv_attach_out(int h, int fd, int nbytes)
{
    gbuf_t *b;
    uint64 vapps = 0, vadsp = 0;
    int rc;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    b = &g_buf[h];
    if (fd < 0 || nbytes <= 0) return AEE_EBADPARM;
    if (b->out_ptr) { apps_mem_share_unmap((uint64)(size_t)b->out_ptr, b->out_bytes); b->out_ptr = 0; }
    rc = apps_mem_share_map(fd, nbytes, &vapps, &vadsp);
    if (rc != 0 || vadsp == 0) return rc ? rc : AEE_EBADPARM;
    b->out_ptr = (int8_t *)(size_t)vadsp;
    b->out_bytes = nbytes;
    return 0;
}

int lfmgemv_detach_out(int h)
{
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    if (g_buf[h].out_ptr) {
        apps_mem_share_unmap((uint64)(size_t)g_buf[h].out_ptr, g_buf[h].out_bytes);
        g_buf[h].out_ptr = 0;
        g_buf[h].out_bytes = 0;
    }
    return 0;
}

/* ---------------- zero-copy weights (attach_in) ----------------
 * The host owns the weights in an rpcmem/ION buffer; the DSP only needs the
 * mapping.  apps_mem_share_map() returns the DSP virtual address of the same
 * pages, so b->w can point straight at DDR: gemv/ffn/gemv_resident/
 * ffn_resident read it exactly as they read the heap copy.
 *
 * Cache rules measured on the device (XQ-BC72, see attach_main.c):
 *   - data written by the host before attach_in is visible to the DSP
 *     without an explicit flush, for cached and uncached rpcmem alike
 *     (the map itself does the AP-side cache maintenance);
 *   - rewriting the pages in place afterwards is NOT visible, even with a
 *     host DMA_BUF_IOCTL_SYNC flush: the DSP keeps its own cached view.  The
 *     only refresh path is detach_in + attach_in.
 *
 * detach_in/release_buf drop the mapping.  The DSP keeps no copy of its own,
 * so nothing is written back on the way out (weights are read-only here).
 */
static int attach_in_locked(gbuf_t *b, int fd, int nbytes)
{
    uint64 vapps = 0, vadsp = 0;
    int rc;
    if (fd < 0 || nbytes <= 0 || (nbytes & 127) != 0) return AEE_EBADPARM;
    rc = apps_mem_share_map(fd, nbytes, &vapps, &vadsp);
    if (rc != 0 || vadsp == 0) return rc ? rc : AEE_EBADPARM;
    if (((uint64)vadsp & 127) != 0) {   /* HVX loads need 128 B alignment */
        apps_mem_share_unmap(vadsp, nbytes);
        return AEE_EBADPARM;
    }
    /* replace the previous mapping / heap copy only once the new map worked */
    if (b->in_ptr) {
        apps_mem_share_unmap((uint64)(size_t)b->in_ptr, b->in_bytes);
        b->in_ptr = 0;
        b->in_bytes = 0;
    }
    free(b->w_heap);
    b->w_heap = 0;
    b->in_ptr = (int8_t *)(size_t)vadsp;
    b->in_bytes = nbytes;
    b->in_owned = 1;
    b->arena = -1;
    b->w = (uint8_t *)b->in_ptr;
    b->wbytes = (size_t)nbytes;
    return 0;
}

int lfmgemv_attach_in(int h, int fd, int nbytes)
{
    gbuf_t *b;
    int rc;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    b = &g_buf[h];
    if (b->N && (size_t)b->N * (size_t)(b->K / 2) > (size_t)nbytes)
        return AEE_EBADPARM;   /* weights already described do not fit */
    rc = attach_in_locked(b, fd, nbytes);
    if (rc == 0) set_perf();
    return rc;
}

int lfmgemv_detach_in(int h)
{
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    if (g_buf[h].in_ptr) {
        if (g_buf[h].in_owned)
            apps_mem_share_unmap((uint64)(size_t)g_buf[h].in_ptr, g_buf[h].in_bytes);
        g_buf[h].in_ptr = 0;
        g_buf[h].in_bytes = 0;
        g_buf[h].in_owned = 0;
        g_buf[h].arena = -1;
        g_buf[h].w = 0;
        g_buf[h].wbytes = 0;
    }
    return 0;
}

int lfmgemv_alloc_in(int fd, int nbytes, int *h)
{
    int i, rc;
    if (h == 0 || fd < 0 || nbytes <= 0) return AEE_EBADPARM;
    set_perf();
    for (i = 0; i < MAXBUF; i++) if (!g_buf[i].used) break;
    if (i == MAXBUF) return AEE_EBADPARM;
    memset(&g_buf[i], 0, sizeof(g_buf[i]));
    rc = attach_in_locked(&g_buf[i], fd, nbytes);
    if (rc != 0) return rc;
    g_buf[i].used = 1;
    *h = i;
    return 0;
}

/* ---------------- arenas: one mapping, many tensor views ----------------
 * A production model does not fit in MAXBUF separate mappings well: every
 * rpcmem buffer is its own fd and every attach_in is a driver round trip.
 * The host instead allocates one arena for the whole model, maps it once and
 * registers views (offset, length) that behave exactly like attach_in
 * handles: w = arena_base + off, wbytes = nbytes.  Bounds are checked against
 * the mapping, so a view can never point outside the buffer.
 */
int lfmgemv_map_arena(int fd, int nbytes, int *ah)
{
    int i, rc;
    uint64 vapps = 0, vadsp = 0;

    if (ah == 0 || fd < 0 || nbytes <= 0 || (nbytes & 127) != 0) return AEE_EBADPARM;
    set_perf();
    for (i = 0; i < MAXARENA; i++) if (!g_arena[i].used) break;
    if (i == MAXARENA) return AEE_EBADPARM;
    rc = apps_mem_share_map(fd, nbytes, &vapps, &vadsp);
    if (rc != 0 || vadsp == 0) return rc ? rc : AEE_EBADPARM;
    if ((vadsp & 127) != 0) {
        apps_mem_share_unmap(vadsp, nbytes);
        return AEE_EBADPARM;
    }
    g_arena[i].base = (uint8_t *)(size_t)vadsp;
    g_arena[i].bytes = nbytes;
    g_arena[i].fd = fd;
    g_arena[i].used = 1;
    *ah = i;
    return 0;
}

int lfmgemv_unmap_arena(int ah)
{
    int i;
    if (ah < 0 || ah >= MAXARENA || !g_arena[ah].used) return AEE_EBADPARM;
    for (i = 0; i < MAXBUF; i++)
        if (g_buf[i].used && g_buf[i].arena == ah) return AEE_EBADPARM;
    apps_mem_share_unmap((uint64)(size_t)g_arena[ah].base, g_arena[ah].bytes);
    memset(&g_arena[ah], 0, sizeof(g_arena[ah]));
    return 0;
}

int lfmgemv_alloc_view(int ah, int off, int nbytes, int *h)
{
    gbuf_t *b;
    int i;

    if (h == 0 || ah < 0 || ah >= MAXARENA || !g_arena[ah].used) return AEE_EBADPARM;
    if (off < 0 || nbytes <= 0 || (off & 127) != 0) return AEE_EBADPARM;
    if ((size_t)off + (size_t)nbytes > (size_t)g_arena[ah].bytes) return AEE_EBADPARM;
    for (i = 0; i < MAXBUF; i++) if (!g_buf[i].used) break;
    if (i == MAXBUF) return AEE_EBADPARM;
    b = &g_buf[i];
    memset(b, 0, sizeof(*b));
    b->used = 1;
    b->arena = ah;
    b->in_owned = 0;
    b->in_ptr = (int8_t *)(g_arena[ah].base + off);
    b->in_bytes = nbytes;
    b->w = (uint8_t *)b->in_ptr;
    b->wbytes = (size_t)nbytes;
    *h = i;
    return 0;
}

/* tiny thread probes */
void tiny_worker(void *a) { *(volatile int *)a = 42; }
void sem_worker(void *a) { qurt_sem_up((qurt_sem_t *)a); }
void hvx_worker(void *a)
{
    volatile int *p = (int *)a;
    qurt_sem_t *sem = (qurt_sem_t *)(p + 8);
    HVX_Vector v = Q6_V_vsplat_R(7);
    union { HVX_Vector v; int32_t s[32]; } u;
    p[0] = qurt_hvx_lock(QURT_HVX_MODE_128B);
    p[1] = qurt_hvx_get_mode();
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vsplat_R(35));
    u.v = v;
    p[2] = u.s[0];
    qurt_hvx_unlock();
    p[3] = 1;
    qurt_sem_up(sem);
}

/* ---------------- paged attention (M=1, GQA 16/4, head_dim 128) ----------------
 * K : int4 per token, offset-8 nibbles (nibble = k+8), lo nibble = even dim.
 *     Per-channel scales are folded into q by the host; the per-token scale
 *     kt[kh][slot] (Q16) is applied to the score here.
 * V : int2 per token, dequantised to signed int8 [kh][slot][128] by the host
 *     (test mode; the on-DSP 2-bit expansion needs a halfword-interleave idiom
 *     that this intrinsic version does not expose).
 * q : int8 [16][128].  out : int32 [16][128] in Q12, already divided by sum(p).
 */

static int g_exp_mode = 1;      /* 0 = bit-trick exp2, 1 = 256-entry table */

/* exp(x) for x <= 0 -> Q8 [0,255]: integer 2^f (3 terms) + variable shift */
static inline int exp_q7_bit(int32_t xq)
{
    int32_t y, n, f, p;
    if (xq < -16 * 65536) return 0;
    if (xq > 0) xq = 0;
    y = mulq16(xq, 94548);                          /* x*log2(e), Q16, no int64 */
    n = y >> 16;
    f = (y - (n << 16)) >> 2;                       /* Q14 fraction, < 2^14 */
    p = 3637;
    p = ((f * p) >> 14) + 15742;                    /* f<2^14, p<2^16 -> int32 ok */
    p = ((f * p) >> 14) + 45426;
    p = ((f * p) >> 14) + 65536;
    n = -n;
    if (n >= 15) return 0;
    {
        int32_t r = (int32_t)((p >> n) >> 8);   /* Q16 -> Q8 */
        return r > 255 ? 255 : r;               /* p must fit in one byte */
    }
}

/* piecewise-linear exp on [-16,0], 32 segments of 0.5 (Q8 out) */
static const int16_t g_expline[33] = {
    255, 155, 94, 57, 35, 21, 13, 8, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
/* table-based: 2^f from a 256-entry int16 table, integer part by shift */
static inline int exp_q7_pwl(int32_t xq)
{
    int32_t y, n, f, r;
    if (xq >= 0) return 255;
    if (xq < -16 * 65536) return 0;
    y = mulq16(xq, 94548);           /* x*log2(e), Q16 */
    n = y >> 16;                     /* floor, <= 0 */
    f = (y >> 8) & 0xff;             /* Q8 fraction */
    r = g_exp2tab[f];                /* 256*2^f, Q8 */
    n = -n;
    if (n >= 9) return 0;
    r >>= n;
    return r > 255 ? 255 : r;
}
static inline int exp_q7(int32_t xq)
{
    return g_exp_mode ? exp_q7_pwl(xq) : exp_q7_bit(xq);
}


static actx_t g_at;
static uint64 g_pc_qk, g_pc_pv;   /* per-phase cycle counters */
static int g_pc_on = 1;

/* dot(int8 q[128], int4 k[64B]) minus the offset-8 term */
static inline int32_t dot_q8_k4(const HVX_Vector *qa, const HVX_Vector *krow, int32_t sumq)
{
    HVX_Vector k, lo, hi, acc;
    union { HVX_Vector v; int32_t s[32]; } u;
    k = *krow;                                  /* 128B aligned, upper 64B are 0 */
    lo = Q6_V_vand_VV(k, Q6_V_vsplat_R(0x0f0f0f0fu));
    hi = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(k, 4), Q6_V_vsplat_R(0x0f0f0f0fu));
    /* qa[0] = even-dim q in bytes 0..63, qa[1] = odd-dim q */
    acc = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[0]);
    acc = Q6_Vw_vrmpyacc_VwVubVb(acc, hi, qa[1]);
    return vsum_w(acc) - 8 * sumq;
}

/* tree-reduce a 32-lane int32 vector, result in lane 0 */
#define VSUM5(v) do {                                    \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 64)); \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 32)); \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 16)); \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 8));  \
        (v) = Q6_Vw_vadd_VwVw((v), Q6_V_vror_VR((v), 4));  \
    } while (0)

/* Per-head, per-dim-group multiplier words for the token-major kernel:
 * lane byte j carries q[8g+2j] (lo nibbles) or q[8g+2j+1] (hi nibbles). */
static void attn_build_uq(actx_t *c)
{
    int h, g, j;
    for (h = 0; h < ATTN_HQ; h++) {
        const int8_t *qp = c->q + (size_t)h * ATTN_D;
        for (g = 0; g < 16; g++) {
            uint32_t lo = 0, hi = 0;
            for (j = 0; j < 4; j++) {
                lo |= (uint32_t)(uint8_t)qp[8 * g + 2 * j] << (8 * j);
                hi |= (uint32_t)(uint8_t)qp[8 * g + 2 * j + 1] << (8 * j);
            }
            c->uq[h][g][0] = (int32_t)lo;
            c->uq[h][g][1] = (int32_t)hi;
        }
    }
}

/* ---- token-major: one page (32 tokens) per reduction ----
 * Per head: a single accumulator chain over the 16 dim-groups, multiplier
 * words copied into stack locals, K page re-read per head (L2-hot). */
static void attn_group_tm(actx_t *ctx, int kh)
{
    const int q0 = kh * 4;
    const uint8_t *kb = ctx->k->w + (size_t)kh * ctx->sk;
    const int32_t *ktp = (const int32_t *)(ctx->kt->w + (size_t)kh * ctx->skt);
    const int tok_first = ctx->t0;
    const int tok_last = ctx->t0 + ctx->n - 1;
    const int pg0 = tok_first >> 5, pg1 = tok_last >> 5;
    const int PAGEB = 16 * 128;
    int32_t best[4], lsum[4];
    int i, t, k, hd, pg, g;
    uint64 pc0 = 0, pc1;

    for (i = 0; i < 4; i++) { best[i] = (int32_t)0x80000000; lsum[i] = 0; }
    if (g_pc_on) pc0 = HAP_perf_get_pcycles();

    /* ---- pass 1: one K page read serves all 4 heads (g outer, heads inner) ---- */
    {
        int32_t ul[4][16], uh[4][16];
        int32_t besth[4];
        int h2;

        for (h2 = 0; h2 < 4; h2++) {
            besth[h2] = (int32_t)0x80000000;
            for (g = 0; g < 16; g++) {
                ul[h2][g] = ctx->uq[q0 + h2][g][0];
                uh[h2][g] = ctx->uq[q0 + h2][g][1];
            }
        }
        for (pg = pg0; pg <= pg1; pg++) {
            const int spg = ctx->tbl[pg];
            const uint8_t *pb = kb + (size_t)spg * PAGEB;
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
            HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
            int32_t l0[32], l1[32], l2[32], l3[32];

            if (pg + 2 <= pg1)
                pf_window(kb + (size_t)ctx->tbl[pg + 2] * PAGEB, 2 * PAGEB);

            for (g = 0; g < 16; g++) {
                HVX_Vector v  = *(const HVX_Vector *)(pb + (size_t)g * 128);
                HVX_Vector lo = Q6_V_vand_VV(v, Q6_V_vsplat_R(0x0f0f0f0fu));
                HVX_Vector hi = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v, 4), Q6_V_vsplat_R(0x0f0f0f0fu));
                a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, lo, Q6_V_vsplat_R((unsigned)ul[0][g]));
                a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, hi, Q6_V_vsplat_R((unsigned)uh[0][g]));
                a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, lo, Q6_V_vsplat_R((unsigned)ul[1][g]));
                a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, hi, Q6_V_vsplat_R((unsigned)uh[1][g]));
                a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, lo, Q6_V_vsplat_R((unsigned)ul[2][g]));
                a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, hi, Q6_V_vsplat_R((unsigned)uh[2][g]));
                a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, lo, Q6_V_vsplat_R((unsigned)ul[3][g]));
                a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, hi, Q6_V_vsplat_R((unsigned)uh[3][g]));
            }
            memcpy(l0, &a0, 128); memcpy(l1, &a1, 128);
            memcpy(l2, &a2, 128); memcpy(l3, &a3, 128);
            for (h2 = 0; h2 < 4; h2++) {
                const int qh = q0 + h2;
                const int32_t *src = (h2 == 0) ? l0 : (h2 == 1) ? l1 : (h2 == 2) ? l2 : l3;
                const int32_t sumq = ctx->sumq[qh];
                int32_t *sp = ctx->scores + (size_t)qh * ctx->n;
                int32_t bh = besth[h2];
                for (t = 0; t < 32; t++) {
                    int tok = pg * 32 + t;
                    int32_t sc;
                    if (tok < tok_first || tok > tok_last) continue;
                    sc = mulq16((src[t] - 8 * sumq) << 8, ktp[(spg << 5) | t]);
                    sp[tok - tok_first] = sc;
                    if (sc > bh) bh = sc;
                }
                besth[h2] = bh;
            }
        }
        for (h2 = 0; h2 < 4; h2++) best[h2] = besth[h2];
    }
    atr(ATR_P1, kh);
    if (g_pc_on) { pc1 = HAP_perf_get_pcycles(); g_pc_qk += pc1 - pc0; pc0 = pc1; }

    /* ---- pass 2: token-major PV.  V is stored transposed per 32-token page:
     * vector (g,c) = 32 dims x 4 tokens, byte 4*d'+j = V[32g+d'][32c+j].
     * One vrmpy per (dim-group, 4 tokens); lane d' accumulates its column. ---- */
    for (hd = 0; hd < 4; hd++) {
        const int qh = q0 + hd;
        const int32_t *sc = ctx->scores + (size_t)qh * ctx->n;
        const int32_t mx = best[hd];
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        int32_t lsum = 0;

        for (pg = pg0; pg <= pg1; pg++) {
            const int spg = ctx->tbl[pg];
            const uint8_t *vb = ctx->v->w + (size_t)kh * ctx->sv + (size_t)spg * 4096;
            int32_t pw[8];
            int c, g, j;

            if (pg + 1 <= pg1)
                pf_window(ctx->v->w + (size_t)kh * ctx->sv
                          + (size_t)ctx->tbl[pg + 1] * 4096, 4096);

            for (c = 0; c < 8; c++) {
                uint32_t w = 0;
                for (j = 0; j < 4; j++) {
                    int32_t p = 0;
                    int tok = pg * 32 + c * 4 + j;
                    if (tok >= tok_first && tok <= tok_last) {
                        p = exp_q7((sc[tok - tok_first] - mx) << 8);
                        lsum += p;
                    }
                    w |= (uint32_t)p << (8 * j);
                }
                pw[c] = (int32_t)w;
            }
            for (g = 0; g < 4; g++) {
                HVX_Vector v0 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 0);
                HVX_Vector v1 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 128);
                HVX_Vector v2 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 256);
                HVX_Vector v3 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 384);
                HVX_Vector v4 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 512);
                HVX_Vector v5 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 640);
                HVX_Vector v6 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 768);
                HVX_Vector v7 = *(const HVX_Vector *)(vb + (size_t)g * 1024 + 896);
                if (g == 0) {
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[0]), v0);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[1]), v1);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[2]), v2);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[3]), v3);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[4]), v4);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[5]), v5);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[6]), v6);
                    a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, Q6_V_vsplat_R((unsigned)pw[7]), v7);
                } else if (g == 1) {
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[0]), v0);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[1]), v1);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[2]), v2);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[3]), v3);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[4]), v4);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[5]), v5);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[6]), v6);
                    a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, Q6_V_vsplat_R((unsigned)pw[7]), v7);
                } else if (g == 2) {
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[0]), v0);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[1]), v1);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[2]), v2);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[3]), v3);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[4]), v4);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[5]), v5);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[6]), v6);
                    a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, Q6_V_vsplat_R((unsigned)pw[7]), v7);
                } else {
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[0]), v0);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[1]), v1);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[2]), v2);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[3]), v3);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[4]), v4);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[5]), v5);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[6]), v6);
                    a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, Q6_V_vsplat_R((unsigned)pw[7]), v7);
                }
            }
        }
        /* lane d' of acc[g] is out[32g + d']: natural order */
        memcpy(ctx->acc + (size_t)qh * ATTN_D + 0,   &a0, 128);
        memcpy(ctx->acc + (size_t)qh * ATTN_D + 32,  &a1, 128);
        memcpy(ctx->acc + (size_t)qh * ATTN_D + 64,  &a2, 128);
        memcpy(ctx->acc + (size_t)qh * ATTN_D + 96,  &a3, 128);
        ctx->ls[qh] = lsum;
        ctx->mx[qh] = mx;
    }
    atr(ATR_P2, kh);
    if (g_pc_on) { pc1 = HAP_perf_get_pcycles(); g_pc_pv += pc1 - pc0; }
}

/* one KV group: the 4 q-heads sharing it are processed in a single pass over
 * the tokens, so each K and V row is read exactly once and the four reduce
 * chains overlap. */
static void attn_group(int kh)
{
    const int q0 = kh * 4;
    const uint8_t *kb = g_at.k->w + (size_t)kh * g_at.sk;
    const int32_t *ktp = (const int32_t *)(g_at.kt->w + (size_t)kh * g_at.skt);
    HVX_Vector qa[4][2];
    int32_t best[4], lsum[4];
    int i, t, k;
    uint64 pc0 = 0, pc1;

    for (i = 0; i < 4; i++) {
        qa[i][0] = *(const HVX_Vector *)(g_at.qe + (size_t)(q0 + i) * 128);
        qa[i][1] = *(const HVX_Vector *)(g_at.qo + (size_t)(q0 + i) * 128);
        best[i] = (int32_t)0x80000000;
        lsum[i] = 0;
    }

    if (g_pc_on) pc0 = HAP_perf_get_pcycles();

    /* ---- pass 1: scores for 4 heads per token, one K row load ---- */
    for (t = 0; t < g_at.n; t++) {
        int tok = g_at.t0 + t;
        int slot = (g_at.tbl[tok >> 5] << 5) | (tok & 31);
        HVX_Vector krow, lo, hi, a0, a1, a2, a3, c;
        union { HVX_Vector v; int32_t w[32]; } u;
        int32_t d[4];

        if ((t & 7) == 0 && t + 24 <= g_at.n) {
            int tk = g_at.t0 + t + 16;
            int sl = (g_at.tbl[tk >> 5] << 5) | (tk & 31);
            if (sl + 8 <= g_at.sk / 128) pf_window(kb + (size_t)sl * 128, 8 * 128);
        }
        krow = *(const HVX_Vector *)(kb + (size_t)slot * 128);
        lo = Q6_V_vand_VV(krow, Q6_V_vsplat_R(0x0f0f0f0fu));
        hi = Q6_V_vand_VV(Q6_Vub_vlsr_VubR(krow, 4), Q6_V_vsplat_R(0x0f0f0f0fu));

        a0 = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[0][0]);
        a0 = Q6_Vw_vrmpyacc_VwVubVb(a0, hi, qa[0][1]);
        a1 = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[1][0]);
        a1 = Q6_Vw_vrmpyacc_VwVubVb(a1, hi, qa[1][1]);
        a2 = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[2][0]);
        a2 = Q6_Vw_vrmpyacc_VwVubVb(a2, hi, qa[2][1]);
        a3 = Q6_Vw_vrmpyacc_VwVubVb(Q6_V_vzero(), lo, qa[3][0]);
        a3 = Q6_Vw_vrmpyacc_VwVubVb(a3, hi, qa[3][1]);

        VSUM5(a0); VSUM5(a1); VSUM5(a2); VSUM5(a3);
        /* after VSUM5 every lane holds its total: rotate into lanes 0..3,
           keeping whole 32-bit lanes */
        {
            static const int32_t lmask_src[4][32] = {
                { -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
                { 0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
                { 0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
                { 0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }
            };
            HVX_Vector L0 = *(const HVX_Vector *)lmask_src[0];
            HVX_Vector L1 = *(const HVX_Vector *)lmask_src[1];
            HVX_Vector L2 = *(const HVX_Vector *)lmask_src[2];
            HVX_Vector L3 = *(const HVX_Vector *)lmask_src[3];
            c = Q6_V_vor_VV(
                  Q6_V_vor_VV(Q6_V_vand_VV(a0, L0), Q6_V_vand_VV(Q6_V_vror_VR(a1, 124), L1)),
                  Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(a2, 120), L2),
                              Q6_V_vand_VV(Q6_V_vror_VR(a3, 116), L3)));
        }
        u.v = c;
        for (i = 0; i < 4; i++) d[i] = u.w[i];

        for (i = 0; i < 4; i++) {
            int32_t sc16 = mulq16((d[i] - 8 * g_at.sumq[q0 + i]) << 8, ktp[slot]);
            g_at.scores[(size_t)(q0 + i) * g_at.n + t] = sc16;
            if (sc16 > best[i]) best[i] = sc16;
        }
    }

    if (g_pc_on) { pc1 = HAP_perf_get_pcycles(); g_pc_qk += pc1 - pc0; pc0 = pc1; }

    /* ---- pass 2: two heads at a time (8 accumulators -> no register spills) ---- */
    {
        const HVX_Vector m0 = Q6_V_vsplat_R(0x000000ffu);
        const HVX_Vector m1 = Q6_V_vsplat_R(0x0000ff00u);
        const HVX_Vector m2 = Q6_V_vsplat_R(0x00ff0000u);
        const HVX_Vector m3 = Q6_V_vsplat_R(0xff000000u);
        int sub, k;

        for (sub = 0; sub < 2; sub++) {
            const int h0 = sub * 2;
            HVX_Vector e0 = Q6_V_vzero(), e1 = Q6_V_vzero();
            HVX_Vector e2 = Q6_V_vzero(), e3 = Q6_V_vzero();
            HVX_Vector f0 = Q6_V_vzero(), f1 = Q6_V_vzero();
            HVX_Vector f2 = Q6_V_vzero(), f3 = Q6_V_vzero();
            int32_t l0 = 0, l1 = 0;

            for (t = 0; t < g_at.n; t++) {
                int tok = g_at.t0 + t;
                int slot = (g_at.tbl[tok >> 5] << 5) | (tok & 31);
                HVX_Vector vv, pa, pb;
                int32_t qa_, qb_;

                if (sub == 0 && (t & 7) == 0 && t + 24 <= g_at.n) {
                    int tk = g_at.t0 + t + 16;
                    int sl = (g_at.tbl[tk >> 5] << 5) | (tk & 31);
                    if (sl + 8 <= g_at.sv / 128)
                        pf_window(g_at.v->w + (size_t)kh * g_at.sv + (size_t)sl * 128, 8 * 128);
                }
                qa_ = exp_q7((g_at.scores[(size_t)(q0 + h0) * g_at.n + t] - best[h0]) << 8);
                qb_ = exp_q7((g_at.scores[(size_t)(q0 + h0 + 1) * g_at.n + t] - best[h0 + 1]) << 8);
                l0 += qa_; l1 += qb_;
                if (!qa_ && !qb_) continue;
                vv = *(const HVX_Vector *)(g_at.v->w + (size_t)kh * g_at.sv + (size_t)slot * 128);
                pa = Q6_V_vsplat_R((unsigned)(0x01010101u * (unsigned)qa_));
                pb = Q6_V_vsplat_R((unsigned)(0x01010101u * (unsigned)qb_));
                e0 = Q6_Vw_vrmpyacc_VwVubVb(e0, Q6_V_vand_VV(pa, m0), vv);
                e1 = Q6_Vw_vrmpyacc_VwVubVb(e1, Q6_V_vand_VV(pa, m1), vv);
                e2 = Q6_Vw_vrmpyacc_VwVubVb(e2, Q6_V_vand_VV(pa, m2), vv);
                e3 = Q6_Vw_vrmpyacc_VwVubVb(e3, Q6_V_vand_VV(pa, m3), vv);
                f0 = Q6_Vw_vrmpyacc_VwVubVb(f0, Q6_V_vand_VV(pb, m0), vv);
                f1 = Q6_Vw_vrmpyacc_VwVubVb(f1, Q6_V_vand_VV(pb, m1), vv);
                f2 = Q6_Vw_vrmpyacc_VwVubVb(f2, Q6_V_vand_VV(pb, m2), vv);
                f3 = Q6_Vw_vrmpyacc_VwVubVb(f3, Q6_V_vand_VV(pb, m3), vv);
            }
            g_at.ls[q0 + h0] = l0;
            g_at.ls[q0 + h0 + 1] = l1;

            {
                union { HVX_Vector v; int32_t s[32]; } ua[4], ub[4];
                int32_t *da = g_at.acc + (size_t)(q0 + h0) * ATTN_D;
                int32_t *db = g_at.acc + (size_t)(q0 + h0 + 1) * ATTN_D;
                ua[0].v = e0; ua[1].v = e1; ua[2].v = e2; ua[3].v = e3;
                ub[0].v = f0; ub[1].v = f1; ub[2].v = f2; ub[3].v = f3;
                for (k = 0; k < 32; k++) {
                    da[4 * k + 0] = ua[0].s[k]; da[4 * k + 1] = ua[1].s[k];
                    da[4 * k + 2] = ua[2].s[k]; da[4 * k + 3] = ua[3].s[k];
                    db[4 * k + 0] = ub[0].s[k]; db[4 * k + 1] = ub[1].s[k];
                    db[4 * k + 2] = ub[2].s[k]; db[4 * k + 3] = ub[3].s[k];
                }
            }
        }
        for (i = 0; i < 4; i++) g_at.mx[q0 + i] = best[i];
    }
    if (g_pc_on) { pc1 = HAP_perf_get_pcycles(); g_pc_pv += pc1 - pc0; }
}

/* ---------------- one-RPC FFN layer ---------------- */



static int ffn_impl(int hg, int hu, int hd, const unsigned char *act, int actLen, int reps,
                    unsigned char *dst, int dstLen, unsigned char *mid, int midLen,
                    int *times, int timesLen, int use_resident)
{
    gbuf_t *g, *u, *d;
    const int8_t *x = (const int8_t *)act;
    int64_t t0, t1, ph[5] = { 0, 0, 0, 0, 0 };
    int32_t bias, biasd, mq_mid;
    int i, rep;

    if (hg < 0 || hu < 0 || hd < 0 || hg >= MAXBUF || hu >= MAXBUF || hd >= MAXBUF)
        return AEE_EBADPARM;
    g = &g_buf[hg]; u = &g_buf[hu]; d = &g_buf[hd];
    if (!g->used || !u->used || !d->used) return AEE_EBADPARM;
    if (!g->w || !u->w || !d->w) return AEE_EBADPARM;
    if (g->N != u->N || g->K != u->K || d->N != u->K || d->K != g->N) return AEE_EBADPARM;
    if (actLen != g->K) return AEE_EBADPARM;
    if (use_resident) {
        if (!d->out_ptr || d->out_bytes < d->N) return AEE_EBADPARM;
    } else if (dstLen != d->N) return AEE_EBADPARM;
    if (!g->qp || !u->qp || !d->qp || !g->qm || !u->qm || !d->sa) return AEE_EBADPARM;
    if (mid && midLen != g->N) return AEE_EBADPARM;
    if (reps < 1) reps = 1;

    set_perf();
    if (g_ffn.n != g->N) {
        free(g_ffn.acc2); free(g_ffn.h);
        g_ffn.acc2 = (int32_t *)memalign(128, (size_t)g->N * sizeof(int32_t));
        g_ffn.h    = (int8_t *)memalign(128, (size_t)g->N + 128);
        g_ffn.n = g->N;
        if (!g_ffn.acc2 || !g_ffn.h) return AEE_ENOMEMORY;
    }
    mq_mid = (int32_t)(4096.0f / d->sa);
    g_ffn_last_g = g;
    g_ffn_last_u = u;
    g_ffn_last_mq = mq_mid;

    for (rep = 0; rep < reps; rep++) {
        /* 1: deinterleave + gate GEMV */
        t0 = HAP_perf_get_time_us();
        deinterleave(x, g->ae, g->ao, g->K);
        bias = -8 * sum_act_i8(x, g->K);
        run_split_m1(g, g->ae, g->ao, bias, g->acc);
        t1 = HAP_perf_get_time_us(); ph[0] += t1 - t0;

        /* 2: up GEMV (same activations) */
        t0 = HAP_perf_get_time_us();
        run_split_m1(u, g->ae, g->ao, bias, g_ffn.acc2);
        t1 = HAP_perf_get_time_us(); ph[1] += t1 - t0;

        /* 3: silu(gate) * up -> int8, split across the worker team */
        t0 = HAP_perf_get_time_us();
        run_split_silu(g, u, mq_mid);
        t1 = HAP_perf_get_time_us(); ph[2] += t1 - t0;

        /* 4: down GEMV */
        t0 = HAP_perf_get_time_us();
        deinterleave(g_ffn.h, d->ae, d->ao, d->K);
        biasd = -8 * sum_act_i8(g_ffn.h, d->K);
        run_split_m1(d, d->ae, d->ao, biasd, d->acc);
        t1 = HAP_perf_get_time_us(); ph[3] += t1 - t0;

        /* 5: dequant + pack */
        t0 = HAP_perf_get_time_us();
        pack_out_i8(d->acc, d->qp, use_resident ? d->out_ptr : (int8_t *)dst, d->N);
        t1 = HAP_perf_get_time_us(); ph[4] += t1 - t0;
    }

    if (mid) memcpy(mid, g_ffn.h, (size_t)g->N);
    if (times && timesLen >= 8) {
        times[0] = (int)((ph[0] + ph[1] + ph[2] + ph[3] + ph[4]) / reps);
        times[1] = (int)(ph[0] / reps);   /* deint + gate */
        times[2] = (int)(ph[1] / reps);   /* up          */
        times[3] = (int)(ph[2] / reps);   /* silu        */
        times[4] = (int)(ph[3] / reps);   /* down        */
        times[5] = (int)(ph[4] / reps);   /* pack        */
        times[6] = g->N;
        times[7] = d->N;
        if (timesLen >= 9) times[8] = g_team_fallbacks;
    }
    return 0;
}

int lfmgemv_ffn(int hg, int hu, int hd, const unsigned char *act, int actLen, int reps,
                unsigned char *dst, int dstLen, unsigned char *mid, int midLen,
                int *times, int timesLen)
{
    return ffn_impl(hg, hu, hd, act, actLen, reps, dst, dstLen, mid, midLen,
                    times, timesLen, 0);
}

int lfmgemv_ffn_resident(int hg, int hu, int hd, const unsigned char *act, int actLen,
                         int reps, unsigned char *mid, int midLen,
                         int *times, int timesLen)
{
    return ffn_impl(hg, hu, hd, act, actLen, reps, 0, 0, mid, midLen,
                    times, timesLen, 1);
}

static int attn_core(int hk, int hkt, int hv, int hvs, const unsigned char *q, int qLen,
                     const int *tbl, int tblLen, int pos, int seq, int window, int reps,
                     unsigned char *outp, int outLen, int *times, int timesLen)
{
    static int8_t *qe, *qo;
    static int32_t *sumq, *scores, *acc;
    static int cap_n = 0;
    int64_t t0, ph[6] = { 0, 0, 0, 0, 0, 0 };
    int i, r, qh, n, tok0, tok1;

    if (hk < 0 || hkt < 0 || hv < 0 || hk >= MAXBUF || hkt >= MAXBUF || hv >= MAXBUF)
        return AEE_EBADPARM;
    if (!g_buf[hk].used || !g_buf[hkt].used || !g_buf[hv].used) return AEE_EBADPARM;
    if (!q || qLen != ATTN_HQ * ATTN_D) return AEE_EBADPARM;
    if (!tbl || tblLen < 1 || outLen != ATTN_HQ * ATTN_D * 4) return AEE_EBADPARM;
    if (pos < 0 || seq < 1 || pos >= seq) return AEE_EBADPARM;
    if ((pos >> 5) >= tblLen) return AEE_EBADPARM;
    if (reps < 1) reps = 1;
    set_perf();

    g_at.k  = &g_buf[hk];
    g_at.kt = &g_buf[hkt];
    g_at.v  = &g_buf[hv];
    /* row-major layout: slots*128 (64 useful + 64 pad);
       token-major layout: slots*64 (pure int4, 2048 B per 32-token page) */
    g_at.sk = (int)(g_at.k->wbytes / ATTN_KV) / (g_attn_tm ? 2 : 1);
    g_at.sv = g_at.v->wbytes / ATTN_KV;      /* slots*128 */
    g_at.skt = g_at.kt->wbytes / ATTN_KV;    /* slots*4   */
    g_at.q  = (const int8_t *)q;
    g_at.tbl = tbl;

    tok1 = pos + 1;
    tok0 = (window > 0 && tok1 - window > 0) ? tok1 - window : 0;
    n = tok1 - tok0;
    g_at.t0 = tok0;
    g_at.n = n;

    if (n > cap_n) {
        free(qe); free(qo); free(sumq); free(scores); free(acc);
        qe     = (int8_t *)memalign(128, ATTN_HQ * 128);
        qo     = (int8_t *)memalign(128, ATTN_HQ * 128);
        sumq   = (int32_t *)memalign(128, ATTN_HQ * 4);
        scores = (int32_t *)memalign(128, (size_t)ATTN_HQ * n * 4);
        acc    = (int32_t *)memalign(128, ATTN_HQ * ATTN_D * 4);
        if (!qe) return AEE_EBADPARM + 1000;
        if (!qo) return AEE_EBADPARM + 1001;
        if (!sumq) return AEE_EBADPARM + 1002;
        if (!scores) return AEE_EBADPARM + 1003;
        if (!acc) return AEE_EBADPARM + 1004;
        memset(qe, 0, ATTN_HQ * 128);   /* upper 64B of every head vector stays 0 */
        memset(qo, 0, ATTN_HQ * 128);
        cap_n = n;
    }
    g_at.qe = qe; g_at.qo = qo; g_at.sumq = sumq;
    g_at.scores = scores; g_at.acc = acc;

    /* create the worker team only after g_at is fully consistent */
    if (g_threads > 1 && g_team.state == 0) team_start(g_threads);

    /* local attention context: the workers get its address, they never read
       the g_at global */
    actx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.k = g_at.k; ctx.kt = g_at.kt; ctx.v = g_at.v; ctx.q = g_at.q;
    ctx.qe = qe; ctx.qo = qo; ctx.sumq = sumq;
    ctx.scores = scores; ctx.acc = acc; ctx.tbl = tbl;
    ctx.t0 = g_at.t0; ctx.n = g_at.n;
    ctx.sk = g_at.sk; ctx.sv = g_at.sv; ctx.skt = g_at.skt;
    attn_build_uq(&ctx);

    g_pc_qk = 0; g_pc_pv = 0;
    for (r = 0; r < reps; r++) {
        t0 = HAP_perf_get_time_us();
        for (qh = 0; qh < ATTN_HQ; qh++) {
            int j;
            int32_t sq = 0;
            const int8_t *qp = g_at.q + (size_t)qh * ATTN_D;
            for (j = 0; j < ATTN_D; j++) sq += qp[j];
            for (j = 0; j < 64; j++) {
                qe[qh * 128 + j] = qp[2 * j];
                qo[qh * 128 + j] = qp[2 * j + 1];
            }
            sumq[qh] = sq;
        }
        { int64_t tt = HAP_perf_get_time_us(); ph[0] += tt - t0; t0 = tt; }

        /* heads: main thread + workers */
        {
            int total = g_team.nwork + 1;
            if (g_threads < total) total = g_threads;
            if (total <= 1) {
                for (i = 0; i < ATTN_KV; i++) {
                    if (g_attn_tm) attn_group_tm(&ctx, i); else attn_group(i);
                }
            } else {
                g_pc_on = 0;      /* cycle counters are shared: off when threaded */
                for (i = 0; i < total; i++) {
                    m1job_t *j = &g_team.job[i];
                    j->kind = 2;
                    j->r0 = (int)((int64_t)ATTN_KV * i / total);
                    j->r1 = (int)((int64_t)ATTN_KV * (i + 1) / total);
                    j->ctx = g_attn_tm ? (void *)&ctx : 0;
                }
                team_kick(total);
                for (i = g_team.job[0].r0; i < g_team.job[0].r1; i++) {
                    if (g_attn_tm) attn_group_tm(&ctx, i); else attn_group(i);
                }
                team_join(total);
                g_pc_on = 1;
            }
        }
        if (g_attn_tm) {   /* results live in the local context */
            memcpy(g_at.ls, ctx.ls, sizeof(g_at.ls));
            memcpy(g_at.mx, ctx.mx, sizeof(g_at.mx));
        }
        { int64_t tt = HAP_perf_get_time_us(); ph[1] += tt - t0; t0 = tt; }

        /* normalise into int32 Q12 */
        {
            int32_t *o32 = (int32_t *)outp;
            for (qh = 0; qh < ATTN_HQ; qh++) {
                int32_t l = g_at.ls[qh];
                const int32_t *a = acc + (size_t)qh * ATTN_D;
                int32_t inv = l ? (int32_t)((1LL << 28) / l) : 0;   /* 2^28 / L, once */
                for (i = 0; i < ATTN_D; i++)
                    o32[qh * ATTN_D + i] = (int32_t)(((int64_t)a[i] * inv) >> 16);
            }
        }
        { int64_t tt = HAP_perf_get_time_us(); ph[2] += tt - t0; }
    }

    g_dbg_scores = scores;
    g_dbg_n = n;
    if (times && timesLen >= 8) {
        times[0] = (int)((ph[0] + ph[1] + ph[2]) / reps);
        times[1] = (int)(ph[0] / reps);      /* q prep      */
        times[2] = (int)(ph[1] / reps);      /* heads (all) */
        times[3] = (int)(ph[2] / reps);      /* normalise   */
        times[4] = n;                        /* tokens      */
        times[5] = (int)((g_pc_qk / reps) / 1420);   /* QK^T phase, us */
        times[6] = (int)((g_pc_pv / reps) / 1420);   /* exp + PV, us */
        times[7] = reps;
        if (timesLen >= 9) times[8] = g_team_fallbacks;
    }
    return 0;
}

int lfmgemv_attn(int hk, int hkt, int hv, int hvs, const unsigned char *q, int qLen,
                 const int *tbl, int tblLen, int pos, int seq, int window, int reps,
                 unsigned char *outp, int outLen, int *times, int timesLen)
{
    return attn_core(hk, hkt, hv, hvs, q, qLen, tbl, tblLen, pos, seq, window, reps,
                     outp, outLen, times, timesLen);
}

int lfmgemv_team_info(int *res, int resLen)
{
    if (!res || resLen < 8) return AEE_EBADPARM;
    res[0] = g_team.state;
    res[1] = g_team.hvx_rc;
    res[2] = qurt_hvx_get_mode();
    res[3] = g_team.ncreated;
    res[4] = g_threads;
    res[5] = g_team.thr_rc;
    res[6] = (int)((qurt_hvx_get_units() >> 8) & 0xFF);
    res[7] = g_rows;
    return 0;
}

/* ---------------- RPC entry points ---------------- */

static int team_start(int ntotal);

int lfmgemv_alloc_buf(int nbytes, int *h)
{
    int i;
    if (h == 0 || nbytes <= 0) return AEE_EBADPARM;
    set_perf();
    for (i = 0; i < MAXBUF; i++) if (!g_buf[i].used) break;
    if (i == MAXBUF) return AEE_EBADPARM;
    memset(&g_buf[i], 0, sizeof(g_buf[i]));
    g_buf[i].w = (uint8_t *)memalign(128, (size_t)nbytes + 128);
    if (g_buf[i].w == 0) return AEE_ENOMEMORY;
    g_buf[i].w_heap = g_buf[i].w;
    g_buf[i].used = 1;
    g_buf[i].arena = -1;
    g_buf[i].wbytes = (size_t)nbytes;
    *h = i;
    return 0;
}

int lfmgemv_set_shape(int h, int N, int K, int with_scale)
{
    gbuf_t *b;
    size_t half;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    if (N < 1 || K < 256 || (K & 255) != 0) return AEE_EBADPARM;
    b = &g_buf[h];
    if ((size_t)N * (size_t)(K / 2) > b->wbytes) return AEE_EBADPARM;
    b->w8 = (with_scale & 2) ? 1 : 0;
    if (b->w8 && (size_t)N * (size_t)K > b->wbytes) return AEE_EBADPARM;
    b->N = N;
    b->K = K;
    b->nvec = K / 256;
    half = (size_t)(K / 2) + 128;
    free(b->acc); free(b->ae); free(b->ao); free(b->qp); free(b->qm);
    b->acc  = (int32_t *)memalign(128, (size_t)4 * N * sizeof(int32_t));
    b->ae   = (int8_t *)memalign(128, 8 * half);
    b->ao   = (int8_t *)memalign(128, 8 * half);
    b->qp   = (with_scale & 1) ? (int32_t *)memalign(128, (size_t)N * sizeof(int32_t)) : 0;
    b->qm   = (with_scale & 1) ? (int32_t *)memalign(128, (size_t)N * sizeof(int32_t)) : 0;
    if (!b->acc || !b->ae || !b->ao || ((with_scale & 1) && (!b->qp || !b->qm))) return AEE_ENOMEMORY;
    set_perf();
    return 0;
}

int lfmgemv_load_buf(int h, int off, const unsigned char *data, int dataLen)
{
    gbuf_t *b;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used || !data || dataLen < 0) return AEE_EBADPARM;
    b = &g_buf[h];
    if (!b->w) return AEE_EBADPARM;
    if ((size_t)off + (size_t)dataLen > b->wbytes) return AEE_EBADPARM;
    memcpy(b->w + off, data, (size_t)dataLen);
    return 0;
}

/* One token into all kv heads: 12 load_buf calls collapse into one RPC.
 *
 * Row layout of the attention kernel (see attn_group):
 *   K  [head][slot][128]  - 64 useful int4 bytes, upper 64 bytes zero;
 *   KT [head][slot][4]    - int32 Q16 scale;
 *   V  [head][slot][128]  - signed int8.
 * k/kt/v carry `heads` rows each, in head order, so the host builds three
 * contiguous vectors and the DSP scatters them at head * slots + slot. */
int lfmgemv_kv_write(int hk, int hkt, int hv, int slots, int slot,
                     const unsigned char *k, int kLen,
                     const unsigned char *kt, int ktLen,
                     const unsigned char *v, int vLen)
{
    gbuf_t *bk, *bt, *bv;
    int heads, i;

    if (hk < 0 || hkt < 0 || hv < 0 || hk >= MAXBUF || hkt >= MAXBUF || hv >= MAXBUF)
        return AEE_EBADPARM;
    if (!k || !kt || !v) return AEE_EBADPARM;
    bk = &g_buf[hk];
    bt = &g_buf[hkt];
    bv = &g_buf[hv];
    if (!bk->used || !bt->used || !bv->used) return AEE_EBADPARM;
    if (!bk->w || !bt->w || !bv->w) return AEE_EBADPARM;
    if (slots < 1 || slot < 0 || slot >= 32 || slot >= slots) return AEE_EBADPARM;
    if (kLen <= 0 || (kLen & 63) != 0 || ktLen < 0 || (ktLen & 3) != 0 ||
        vLen < 0 || (vLen & 127) != 0)
        return AEE_EBADPARM;
    heads = kLen / 64;
    if (heads != ktLen / 4 || heads != vLen / 128) return AEE_EBADPARM;
    for (i = 0; i < heads; i++) {
        size_t row = (size_t)i * (size_t)slots + (size_t)slot;
        size_t ko = row * 128, to = row * 4, vo = row * 128;
        if (ko + 128 > bk->wbytes || to + 4 > bt->wbytes || vo + 128 > bv->wbytes)
            return AEE_EBADPARM;
        memcpy(bk->w + ko, k + (size_t)i * 64, 64);
        memset(bk->w + ko + 64, 0, 64);
        memcpy(bt->w + to, kt + (size_t)i * 4, 4);
        memcpy(bv->w + vo, v + (size_t)i * 128, 128);
    }
    return 0;
}

int lfmgemv_set_scale(int h, const unsigned char *scales, int scalesLen)
{
    gbuf_t *b;
    int N, K, i;
    float sa, so;
    const float *sw;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used || !scales) return AEE_EBADPARM;
    b = &g_buf[h];
    if (scalesLen < 16) return AEE_EBADPARM;
    memcpy(&N, scales + 0, 4);
    memcpy(&K, scales + 4, 4);
    memcpy(&sa, scales + 8, 4);
    memcpy(&so, scales + 12, 4);
    if (N != b->N || K != b->K || scalesLen < 16 + 4 * N || !b->qp || !b->qm) return AEE_EBADPARM;
    b->sa = sa;
    b->so = so;
    sw = (const float *)(scales + 16);
    {
        double mmax = 0.0;
        for (i = 0; i < N; i++) {
            double m = (double)sw[i] * (double)sa / (double)so;
            if (m > mmax) mmax = m;
        }
        /* gate/up need the wide (int64) dequant; down stays below 1 and uses
         * qp.  k is 0 whenever the worst-case product fits 2^30. */
        b->qshift = deq_shift_for(mmax, K, b->w8 ? 7 : 3);
    }
    for (i = 0; i < N; i++) {
        float m = sw[i] * sa / so;          /* exactly the old pack_out field */
        double md = (double)sw[i] * (double)sa / (double)so;
        unsigned q;
        int64_t qm;
        qm = (int64_t)((md / (double)(1u << b->qshift)) * 65536.0 + 0.5);
        if (qm > 0x7fffffffLL) qm = 0x7fffffffLL;
        if (qm < 0) qm = 0;
        b->qm[i] = (int32_t)qm;
        if (m < 0.0f) m = 0.0f;
        if (m > 0.99997f) m = 0.99997f;
        q = (unsigned)(m * 65536.0f + 0.5f);
        if (q > 65535u) q = 65535u;
        b->qp[i] = (int32_t)(q | (q << 16));
    }
    return 0;
}

static int gemv_impl(int h, int M, const unsigned char *act, int actLen, int reps,
                     unsigned char *dst, int dstLen, int *times, int timesLen,
                     int use_resident)
{
    gbuf_t *b;
    const int8_t *a;
    int64_t t0, t1;
    int32_t bias[4];
    int64_t best = 0, worst = 0, total = 0;
    int64_t tdeint = 0, tpack = 0;
    uint64 best_pc = 0;
    int i, m;

    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    b = &g_buf[h];
    if (!b->w) return AEE_EBADPARM;
    if (M != 1 && M != 2 && M != 4) return AEE_EBADPARM;
    if (actLen != M * b->K) return AEE_EBADPARM;
    if (b->qp) {
        if (use_resident) {
            if (!b->out_ptr || b->out_bytes < M * b->N) return AEE_EBADPARM;
        } else if (dstLen != M * b->N) return AEE_EBADPARM;
    } else {
        if (dstLen != M * b->N * (int)sizeof(int32_t)) return AEE_EBADPARM;
    }
    if (reps < 1) reps = 1;
    a = (const int8_t *)act;
    set_perf();

    for (i = 0; i < reps; i++) {
        t0 = HAP_perf_get_time_us();
        for (m = 0; m < M; m++) {
            const int8_t *am = a + (size_t)m * b->K;
            deinterleave(am, b->ae + (size_t)m * (b->K / 2),
                         b->ao + (size_t)m * (b->K / 2), b->K);
            bias[m] = (b->w8 ? -128 : -8) * sum_act_i8(am, b->K);
        }
        t1 = HAP_perf_get_time_us();
        tdeint += t1 - t0;

        {
            uint64 pc0 = HAP_perf_get_pcycles();
            t0 = HAP_perf_get_time_us();
            if (M == 1) {
                if (b->w8) {
                    gemv_rows_m1_i8(b, a, bias[0], b->acc);
                } else if (b->nvec == 8) {
                    run_split_m1(b, b->ae, b->ao, bias[0], b->acc);
                } else {
                    gemv_rows_m1_generic(b, b->ae, b->ao, bias[0], b->acc);
                }
            } else if (M == 2) {
                gemv_rows_m2(b, b->ae, b->ao, bias[0], bias[1], b->acc);
            } else {
                gemv_rows_m4(b, b->ae, b->ao, bias, b->acc);
            }
            t1 = HAP_perf_get_time_us();
            {
                uint64 pc = HAP_perf_get_pcycles() - pc0;
                int64_t dt = t1 - t0;
                if (i == 0 || dt < best) { best = dt; best_pc = pc; }
                if (dt > worst) worst = dt;
                total += dt;
            }
        }

        t0 = HAP_perf_get_time_us();
        if (b->qp) {
            int8_t *o = use_resident ? b->out_ptr : (int8_t *)dst;
            for (m = 0; m < M; m++) {
                pack_out_i8(b->acc + (size_t)m * b->N, b->qp,
                            o + (size_t)m * b->N, b->N);
            }
        } else {
            memcpy(dst, b->acc, (size_t)M * b->N * sizeof(int32_t));
        }
        t1 = HAP_perf_get_time_us();
        tpack += t1 - t0;
    }

    if (times != 0 && timesLen >= 8) {
        times[0] = (int)best;
        times[1] = (int)worst;
        times[2] = (int)(total / reps);
        times[3] = (int)(tdeint / reps);
        times[4] = (int)(tpack / reps);
        times[5] = (int)(best_pc & 0x7fffffff);
        times[6] = b->N;
        times[7] = b->K;
    }
    return 0;
}

int lfmgemv_gemv(int h, int M, const unsigned char *act, int actLen, int reps,
                 unsigned char *dst, int dstLen, int *times, int timesLen)
{
    return gemv_impl(h, M, act, actLen, reps, dst, dstLen, times, timesLen, 0);
}

int lfmgemv_gemv_resident(int h, int M, const unsigned char *act, int actLen,
                          int reps, int *times, int timesLen)
{
    return gemv_impl(h, M, act, actLen, reps, 0, 0, times, timesLen, 1);
}

/* ---------------- micro benchmarks ---------------- */

int lfmgemv_probe(int h, int kind, int arg, int *res, int resLen)
{
    gbuf_t *b = (h >= 0 && h < MAXBUF && g_buf[h].used) ? &g_buf[h] : 0;
    uint64 pc0 = 0, pc1 = 0;
    int64_t t0 = 0, t1 = 0;
    int i;
    uint32_t sum = 0;

    if (resLen < 8 || !res) return AEE_EBADPARM;
    for (i = 0; i < 8; i++) res[i] = 0;
    set_perf();

    switch (kind) {
    case 0: { /* HVX ALU dependency chain */
        HVX_Vector a = Q6_V_vsplat_R((unsigned)arg);
        HVX_Vector m = Q6_V_vsplat_R(0x0f0f0f0fu);
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        for (i = 0; i < arg; i++) a = Q6_V_vand_VV(a, m);
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        { union { HVX_Vector v; uint32_t w[32]; } u; u.v = a; res[2] = (int)(u.w[0] & 0xffff); }
        break;
    }
    case 1: { /* HVX ALU throughput, 4 chains */
        HVX_Vector a = Q6_V_vsplat_R(1u), c = Q6_V_vsplat_R(1u);
        HVX_Vector e = Q6_V_vsplat_R(2u), f = Q6_V_vsplat_R(2u);
        HVX_Vector m = Q6_V_vsplat_R(0x0f0f0f0fu);
        int it = arg / 4;
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        for (i = 0; i < it; i++) {
            a = Q6_V_vand_VV(a, m); c = Q6_V_vand_VV(c, m);
            e = Q6_V_vand_VV(e, m); f = Q6_V_vand_VV(f, m);
        }
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        { union { HVX_Vector v; uint32_t w[32]; } u;
          u.v = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a, c), Q6_Vw_vadd_VwVw(e, f));
          res[2] = (int)(u.w[0] & 0xff); }
        break;
    }
    case 2: { /* plain HVX loads over the whole weight buffer */
        const int n = b ? (int)(b->wbytes / 128) : 0;
        const HVX_Vector *p = b ? (const HVX_Vector *)b->w : 0;
        HVX_Vector acc = Q6_V_vzero();
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        for (i = 0; i < n; i++) acc = Q6_Vw_vadd_VwVw(acc, p[i]);
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = vsum_w(acc);
        break;
    }
    case 3: { /* loads + unpack + vrmpy (kernel pattern, dependent) */
        const int n = b ? (int)(b->wbytes / 128) : 0;
        const HVX_Vector *p = b ? (const HVX_Vector *)b->w : 0;
        HVX_Vector acc = Q6_V_vzero();
        HVX_Vector av = Q6_V_vsplat_R(0x01020304u);
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        for (i = 0; i < n; i++) {
            HVX_Vector v = p[i];
            acc = Q6_Vw_vrmpyacc_VwVubVb(acc, vand_mask0f(v), av);
            acc = Q6_Vw_vrmpyacc_VwVubVb(acc, vlsr4b(v), av);
        }
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = vsum_w(acc);
        break;
    }
    case 4: { /* scalar volatile byte reads */
        const volatile uint8_t *p = b ? (const volatile uint8_t *)b->w : 0;
        int n = b ? (int)b->wbytes : 0;
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        for (i = 0; i < n; i++) sum += p[i];
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = (int)sum;
        break;
    }
    case 5: { /* environment */
        int clkmax = 0, busmax = 0;
        HAP_power_response_t resp;
        HAP_power_get_max_speed(&clkmax, &busmax);
        memset(&resp, 0, sizeof(resp));
        resp.type = HAP_power_get_clk_Freq;
        HAP_power_get(0, &resp);
        res[2] = clkmax;
        res[3] = (int)resp.clkFreqHz;
        res[6] = busmax;
        break;
    }
    case 6: { /* linear streaming read, arg = MiB to read, 512 KiB blocks */
        const int blk = 512 * 1024;
        const int nblk = b ? (int)(b->wbytes / blk) : 0;
        int want = (arg > 0 ? arg : 1024) * 1024 * 1024;
        HVX_Vector acc = Q6_V_vzero();
        int done = 0;
        uint8_t *p = b ? b->w : 0;
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        while (done + blk <= want && nblk > 0) {
            int bi = (done / blk) % nblk;
            const HVX_Vector *q = (const HVX_Vector *)(p + (size_t)bi * blk);
            int j;
            for (j = 0; j < blk / 128; j++) acc = Q6_Vw_vadd_VwVw(acc, q[j]);
            done += blk;
        }
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = vsum_w(acc);
        res[6] = done;
        break;
    }
    case 7: { /* linear streaming read with l2fetch, arg = MiB */
        const int blk = 512 * 1024;
        const int nblk = b ? (int)(b->wbytes / blk) : 0;
        int want = (arg > 0 ? arg : 1024) * 1024 * 1024;
        HVX_Vector acc = Q6_V_vzero();
        int done = 0;
        uint8_t *p = b ? b->w : 0;
        const int pf_kb = g_pf_kb > 0 ? g_pf_kb : 16;
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        while (done + blk <= want && nblk > 0) {
            int bi = (done / blk) % nblk;
            const HVX_Vector *q = (const HVX_Vector *)(p + (size_t)bi * blk);
            int j;
            for (j = 0; j < blk / 128; j++) {
                if ((j & 15) == 0 && (j + pf_kb * 8) < blk / 128)
                    pf_window(&q[j + pf_kb * 8], 8 * 128);
                acc = Q6_Vw_vadd_VwVw(acc, q[j]);
            }
            done += blk;
        }
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = vsum_w(acc);
        res[6] = done;
        break;
    }
    case 15: { /* create with struct-held tid/sem, sem_worker entry */
        qurt_thread_attr_t attr;
        qurt_sem_init(&g_team.w[1].go);
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "probe15");
        qurt_thread_attr_set_stack_size(&attr, 8 * 1024);
        res[5] = qurt_thread_create(&g_team.w[1].tid, &attr, team_worker, &g_team.w[1]);
        break;
    }
    case 20: { /* HVX entry + explicit stack + caller priority (worker_pool style) */
        static qurt_sem_t s20;
        static void *stack20;
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        int prio = qurt_thread_get_priority(qurt_thread_get_id());
        qurt_sem_init(&s20);
        free(stack20);
        stack20 = memalign(128, 32 * 1024);
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "p20");
        qurt_thread_attr_set_stack_addr(&attr, stack20);
        qurt_thread_attr_set_stack_size(&attr, 32 * 1024);
        qurt_thread_attr_set_priority(&attr, (unsigned short)(prio < 1 ? 1 : prio));
        res[5] = qurt_thread_create(&tid, &attr, team_worker, &s20);
        res[6] = (int)tid;
        res[7] = prio;
        break;
    }
    case 19: {
        static qurt_sem_t s19;
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        qurt_sem_init(&s19);
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "p19");
        qurt_thread_attr_set_stack_size(&attr, 16 * 1024);
        res[5] = qurt_thread_create(&tid, &attr, team_worker, &s19);
        res[6] = (int)tid;
        break;
    }
    case 18: { /* create with team_worker entry */
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        static thw_t dummy;
        qurt_sem_init(&dummy.go); qurt_sem_init(&dummy.done);
        dummy.idx = 1; dummy.shutdown = 0;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "p18");
        qurt_thread_attr_set_stack_size(&attr, 16 * 1024);
        res[5] = qurt_thread_create(&tid, &attr, team_worker, &dummy);
        res[6] = (int)tid;
        break;
    }
    case 17: { /* worker_main with configurable stack: arg = KB */
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        thw_t dummy;
        qurt_sem_init(&dummy.go); qurt_sem_init(&dummy.done);
        dummy.idx = 1; dummy.shutdown = 0;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "p17");
        qurt_thread_attr_set_stack_size(&attr, (unsigned)(arg > 0 ? arg : 8) * 1024);
        dummy.go.X.val = 0;
        res[5] = qurt_thread_create(&tid, &attr, team_worker, &dummy);
        res[6] = (int)tid;
        if (res[5] == 0) {
            qurt_sem_up(&dummy.go);
            qurt_sem_down(&dummy.done);
            res[7] = 1;
        }
        break;
    }
    case 16: { /* create with local tid, worker_main entry */
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        thw_t dummy;
        qurt_sem_init(&dummy.go); qurt_sem_init(&dummy.done);
        dummy.idx = 1; dummy.shutdown = 0;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "probe16");
        qurt_thread_attr_set_stack_size(&attr, 8 * 1024);
        res[5] = qurt_thread_create(&tid, &attr, team_worker, &dummy);
        res[6] = (int)tid;
        break;
    }
    case 14: { /* call team_start directly */
        int made = team_start(8);
        res[5] = made;
        res[6] = g_team.thr_rc;
        res[7] = g_team.hvx_rc;
        break;
    }
    case 30: { /* retired (multipliers now live in the local context) */
        return AEE_EBADPARM;
    }
    case 31: { /* attention step trace: arg = first slot (0..63), 8 events */
        int b0 = arg & 63;
        for (i = 0; i < 8; i++) {
            int k2 = (b0 + i) & 63;
            res[i] = (g_atr_stage[k2] << 8) | (g_atr_arg[k2] & 0xff);
        }
        return 0;    /* skip the epilogue that would clobber res[0..4] */
    }
    case 32: { /* trace state: [0]=written, [1]=team fallbacks, [2]=workers */
        res[0] = g_atr_i & 1023;
        res[1] = g_team_fallbacks;
        res[2] = g_team.nwork;
        return 0;    /* skip the epilogue that would clobber res[0..4] */
    }
    case 29: { /* dump 8 scores: arg = head*4096 + t (after an attn call) */
        int hd2 = arg / 4096, tt = arg % 4096;
        if (!g_dbg_scores) return AEE_EBADPARM;
        for (h = 0; h < 8; h++) res[h] = g_dbg_scores[(size_t)hd2 * g_dbg_n + tt + h];
        return 0;    /* skip the epilogue that would clobber res[0..4] */
    }
    case 28: { /* dump the first 8 scores of head 0 (after an attn call) */
        int h;
        if (!g_at.scores) return AEE_EBADPARM;
        for (h = 0; h < 8; h++) res[h] = g_at.scores[arg + h];
        break;
    }
    case 27: { /* attention kernel: 1 = token-major, 0 = head-grouped */
        g_attn_tm = arg ? 1 : 0;
        res[2] = g_attn_tm;
        break;
    }
    case 26: { /* exp mode: 0 = bit-trick, 1 = piecewise linear */
        g_exp_mode = arg ? 1 : 0;
        res[2] = g_exp_mode;
        break;
    }
    case 13: { /* create + sync via semaphore; arg = stack KB, priority */
        static qurt_sem_t sem;
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        int rc;
        extern void sem_worker(void *);
        qurt_sem_init(&sem);
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "probe13");
        qurt_thread_attr_set_stack_size(&attr, (unsigned)((arg & 0xff) ? (arg & 0xff) : 8) * 1024);
        if (arg >> 8) qurt_thread_attr_set_priority(&attr, (unsigned short)(arg >> 8));
        rc = qurt_thread_create(&tid, &attr, sem_worker, (void *)&sem);
        res[0] = rc;
        if (rc == 0) {
            res[1] = qurt_sem_down(&sem);
            res[2] = 1;
        }
        break;
    }
    case 11: { /* thread create diagnostic (no HVX) */
        static volatile int flag;
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        int rc;
        extern void tiny_worker(void *);
        flag = 0;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "probe");
        qurt_thread_attr_set_stack_size(&attr, 8192);
        rc = qurt_thread_create(&tid, &attr, tiny_worker, (void *)&flag);
        res[0] = rc;
        for (i = 0; i < 2000000 && !flag; i++) ;
        res[1] = flag;
        res[2] = (int)tid;
        break;
    }
    case 12: { /* thread create + HVX lock diagnostic */
        static volatile int flag2[16];
        static qurt_sem_t hsem;
        qurt_sem_t *ps = &hsem;
        qurt_thread_t tid;
        qurt_thread_attr_t attr;
        int rc;
        extern void hvx_worker(void *);
        qurt_sem_init(&hsem);
        flag2[0] = -1; flag2[1] = -1; flag2[2] = -1; flag2[3] = 0;
        flag2[8] = (int)(size_t)&hsem;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "probehvx");
        qurt_thread_attr_set_stack_size(&attr, 16384);
        rc = qurt_thread_create(&tid, &attr, hvx_worker, (void *)flag2);
        res[0] = rc;
        if (rc == 0) qurt_sem_down(&hsem);
        res[1] = flag2[0];   /* lock rc */
        res[2] = flag2[1];   /* mode */
        res[3] = flag2[2];   /* hvx op result */
        res[4] = flag2[3];   /* done flag */
        break;
    }
    case 22: { /* vmpyieacc 32x16 semantics, scalar read-back */
        static int32_t ta[32], tq[32];
        HVX_Vector va, vq, vr;
        union { HVX_Vector v; int32_t w[32]; } u;
        for (i = 0; i < 32; i++) {
            ta[i] = (i + 1) * 1000;
            tq[i] = 0x00008000;          /* 0.5 in the low halfword */
        }
        va = *(HVX_Vector *)ta;
        vq = *(HVX_Vector *)tq;
        vr = Q6_Vw_vmpyieacc_VwVwVuh(Q6_V_vzero(), va, vq);
        u.v = vr;
        for (i = 0; i < 8; i++) res[i] = u.w[i];
        break;
    }
    case 23: { /* same with the multiplier in both halfwords */
        static int32_t ta[32], tq[32];
        HVX_Vector va, vq, vr;
        union { HVX_Vector v; int32_t w[32]; } u;
        for (i = 0; i < 32; i++) { ta[i] = (i + 1) * 1000; tq[i] = 0x80008000; }
        va = *(HVX_Vector *)ta;
        vq = *(HVX_Vector *)tq;
        vr = Q6_Vw_vmpyieacc_VwVwVuh(Q6_V_vzero(), va, vq);
        u.v = vr;
        for (i = 0; i < 8; i++) res[i] = u.w[i];
        break;
    }
    case 21: { /* pack_out_i8 numeric test */
        static int32_t tacc[32], tqp[32];
        static int8_t tout[32];
        for (i = 0; i < 32; i++) {
            tacc[i] = i * 4;
            tqp[i] = 0x80008000;            /* 0.5 in both halfwords */
        }
        pack_out_i8(tacc, tqp, tout, 32);
        for (i = 0; i < 8; i++) res[i] = (int)tout[i];
        break;
    }
    case 9: { /* streaming read, 8 independent chains, optional l2fetch */
        const int blk = 512 * 1024;
        const int nblk = b ? (int)(b->wbytes / blk) : 0;
        int want = (arg > 0 ? arg : 128) * 1024 * 1024;
        HVX_Vector c0 = Q6_V_vzero(), c1 = Q6_V_vzero(), c2 = Q6_V_vzero(), c3 = Q6_V_vzero();
        HVX_Vector c4 = Q6_V_vzero(), c5 = Q6_V_vzero(), c6 = Q6_V_vzero(), c7 = Q6_V_vzero();
        int done = 0;
        uint8_t *p = b ? b->w : 0;
        const int pf_kb = g_usepf ? (g_pf_kb > 0 ? g_pf_kb : 8) : 0;
        t0 = HAP_perf_get_time_us(); pc0 = HAP_perf_get_pcycles();
        while (done + blk <= want && nblk > 0) {
            int bi = (done / blk) % nblk;
            const HVX_Vector *q = (const HVX_Vector *)(p + (size_t)bi * blk);
            const int nv = blk / 128;
            int j;
            for (j = 0; j + 8 <= nv; j += 8) {
                if (pf_kb && (j & 127) == 0 && (j + pf_kb * 8 + 8) <= nv)
                    pf_window(&q[j + pf_kb * 8], 8 * 128);
                c0 = Q6_Vw_vadd_VwVw(c0, q[j + 0]);
                c1 = Q6_Vw_vadd_VwVw(c1, q[j + 1]);
                c2 = Q6_Vw_vadd_VwVw(c2, q[j + 2]);
                c3 = Q6_Vw_vadd_VwVw(c3, q[j + 3]);
                c4 = Q6_Vw_vadd_VwVw(c4, q[j + 4]);
                c5 = Q6_Vw_vadd_VwVw(c5, q[j + 5]);
                c6 = Q6_Vw_vadd_VwVw(c6, q[j + 6]);
                c7 = Q6_Vw_vadd_VwVw(c7, q[j + 7]);
            }
            done += blk;
        }
        pc1 = HAP_perf_get_pcycles(); t1 = HAP_perf_get_time_us();
        res[2] = vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(c0, c1), Q6_Vw_vadd_VwVw(c2, c3)))
               + vsum_w(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(c4, c5), Q6_Vw_vadd_VwVw(c6, c7)));
        res[6] = done;
        break;
    }
    case 8: { /* set tuning: arg = pf_kb | (rows<<8) | (usepf<<16) | (threads<<24) */
        g_pf_kb = arg & 0xff;
        g_rows  = (arg >> 8) & 0xff;
        g_usepf = (arg >> 16) & 0xff;
        g_threads = (arg >> 24) & 0xff;
        if (g_threads < 1) g_threads = 1;
        if (g_rows < 1) g_rows = 1;
        if (g_rows > 2) g_rows = 2;      /* 3+ rows are not implemented */
        if (g_threads > 4) g_threads = 4;
        res[2] = g_pf_kb;
        res[3] = g_rows;
        res[5] = g_usepf;
        res[6] = g_threads;
        break;
    }
    case 40: { /* scalar SiLU step profile on the last FFN buffers; arg = iters
                 res[2..6] = pcycles/pass for full, no-LUT, no-prod, no-requant,
                 no-deq; res[7] = full pass again (repeat) */
        gbuf_t *g = g_ffn_last_g, *u = g_ffn_last_u;
        int iters = arg > 0 ? arg : 16, st;
        int n = g_ffn.n;
        if (!g || !u || n <= 0 || !g_ffn.acc2 || !g_ffn.h) return AEE_EBADPARM;
        for (st = 0; st < 5; st++) {
            static const int slot[5] = { 2, 3, 5, 6, 7 };
            uint64 c0 = HAP_perf_get_pcycles();
            for (i = 0; i < iters; i++)
                silu_prof_run(st, g, u, g_ffn_last_mq, n);
            res[slot[st]] = (int)((HAP_perf_get_pcycles() - c0) / (unsigned)iters);
        }
        break;
    }
    case 42: { /* vector SiLU path: guard and full pass; arg = iters
                 res[2] guard, res[3] pass, res[5] pass rc (1 = vector used) */
        gbuf_t *g = g_ffn_last_g, *u = g_ffn_last_u;
        int iters = arg > 0 ? arg : 16;
        int n = g_ffn.n, rc = 0;
        if (!g || !u || n <= 0) return AEE_EBADPARM;
        {
            uint64 c0 = HAP_perf_get_pcycles();
            for (i = 0; i < iters; i++) {
                sum += (uint32_t)silu_vec_guard(0, n, g, u, g_ffn_last_mq);
                __asm__ __volatile__("" ::: "memory");
            }
            res[2] = (int)((HAP_perf_get_pcycles() - c0) / (unsigned)iters);
        }
        {
            uint64 c0 = HAP_perf_get_pcycles();
            for (i = 0; i < iters; i++)
                rc = silu_range_vec(i == 0 ? 0 : 128, n, g, u, g_ffn_last_mq);
            res[3] = (int)((HAP_perf_get_pcycles() - c0) / (unsigned)iters);
        }
        res[5] = rc;
        break;
    }
    case 44: { /* SiLU vector A/B: split multiply, LUT grid, whole chain on
                 adversarial buffers.  res[2] mul_rnd, res[3] mul_floor,
                 res[4] LUT, res[5] chain, res[6] guard fallbacks, res[7] lanes */
        gbuf_t *g = g_ffn_last_g, *u = g_ffn_last_u;
        static const int32_t aval[18] = {
            0, 1, -1, 32767, -32768, 65535, -65536, 65536, -65537,
            1 << 20, -(1 << 20), (1 << 22), -(1 << 22),
            12345, -54321, 999999, -999999, 0x3fffffff
        };
        int32_t va[32] __attribute__((aligned(128)));
        int32_t vb[32] __attribute__((aligned(128)));
        int32_t vo[32] __attribute__((aligned(128)));
        int bad_rnd = 0, bad_flr = 0, bad_lut = 0, t, q;
        uint32_t seed = 0x1234u;
        int n = g_ffn.n;
        int bad_chain = 0, fell = 0, lanes = 0;
        int8_t *ref;

        /* 1. the two split multiplies vs the int64 formulas, 64 x 32 lanes */
        for (t = 0; t < 64; t++) {
            for (i = 0; i < 32; i++) {
                int32_t a = aval[(i + t) % 18];
                seed = seed * 1103515245u + 12345u;
                vb[i] = (int32_t)(seed >> 8) - (1 << 21);
                if (a == 0x3fffffff) a = 1 << 22;
                va[i] = a;
            }
            *(HVX_Vector *)vo = vmul_q16_rnd(*(HVX_Vector *)va, *(HVX_Vector *)vb);
            for (i = 0; i < 32; i++) {
                int32_t ex = (int32_t)(((int64_t)va[i] * vb[i] + 32768) >> 16);
                if (vo[i] != ex) bad_rnd++;
            }
            *(HVX_Vector *)vo = vmul_q16_floor(*(HVX_Vector *)va, *(HVX_Vector *)vb);
            for (i = 0; i < 32; i++) {
                int32_t ex = (int32_t)(((int64_t)va[i] * vb[i]) >> 16);
                if (vo[i] != ex) bad_flr++;
            }
        }
        res[2] = bad_rnd;
        res[3] = bad_flr;

        /* 2. vsilu_q16 against silu_q16 on and around every LUT node */
        for (t = 0; t < SILU_LUT_N; t++) {
            for (q = 0; q < 4; q++) {
                int32_t in[32] __attribute__((aligned(128)));
                union { HVX_Vector v; int32_t s[32]; } out __attribute__((aligned(128)));
                for (i = 0; i < 32; i++) {
                    int idx = (t + i) % SILU_LUT_N;
                    int32_t z = -SILU_ZLIM + idx * 1024;
                    if (q == 1) z -= 1;
                    if (q == 2) z += 1;
                    if (q == 3) z += 1023;
                    if (z > SILU_ZLIM) z = SILU_ZLIM;
                    if (z < -SILU_ZLIM) z = -SILU_ZLIM;
                    in[i] = z;
                }
                out.v = vsilu_q16(*(HVX_Vector *)in);
                for (i = 0; i < 32; i++)
                    if (out.s[i] != silu_q16(in[i])) bad_lut++;
            }
        }
        res[5] = bad_lut;

        /* 3. whole chain: scalar vs vector on adversarial live buffers */
        ref = (ref = (int8_t *)memalign(128, (size_t)n + 128)) ? ref : 0;
        if (!g || !u || n <= 0 || !ref) return AEE_EBADPARM;
        for (t = 0; t < 6; t++) {
            for (i = 0; i < n; i++) {
                int32_t amax = 127 * 7 * 2048;
                seed = seed * 1103515245u + 12345u;
                switch (t) {
                case 0:   /* everything saturates in both directions */
                    g->acc[i] = (i & 1) ? amax : -amax;
                    g_ffn.acc2[i] = (i & 2) ? amax : -amax;
                    break;
                case 1:   /* up < 0 everywhere -> mid at -127 */
                    g->acc[i] = (int32_t)(seed >> 9);
                    g_ffn.acc2[i] = -amax;
                    break;
                case 2:   /* near-zero gate and up */
                    g->acc[i] = (int32_t)(seed >> 20) - 4;
                    g_ffn.acc2[i] = (int32_t)(seed >> 22) - 2;
                    break;
                case 3:   /* gate on the LUT clamp and node boundaries */
                    g->acc[i] = ((i % 5) == 0) ? (8 << 16) : -(8 << 16);
                    g_ffn.acc2[i] = (int32_t)(seed >> 10) - (1 << 20);
                    break;
                case 4:   /* realistic full range, both signs */
                    g->acc[i] = (int32_t)(seed >> 8) - (1 << 21);
                    seed = seed * 1103515245u + 12345u;
                    g_ffn.acc2[i] = (int32_t)(seed >> 8) - (1 << 21);
                    break;
                default:  /* small up, large gate (P near the clamp edges) */
                    g->acc[i] = (int32_t)(seed >> 7) - (1 << 22);
                    g_ffn.acc2[i] = ((i & 3) == 0) ? 0 : (int32_t)(seed >> 24) - 64;
                    break;
                }
                g->qm[i] = 65536 + (int32_t)(seed & 1023);
                u->qm[i] = 65536 + (int32_t)((seed >> 10) & 1023);
            }
            g_silu_vec = 0;
            silu_range(0, n, g, u, g_ffn_last_mq);
            memcpy(ref, g_ffn.h, (size_t)n);
            g_silu_vec = 1;
            if (!silu_range_vec(0, n, g, u, g_ffn_last_mq)) { fell++; continue; }
            for (i = 0; i < n; i++) if (g_ffn.h[i] != ref[i]) bad_chain++;
            lanes += n;
        }
        free(ref);
        res[6] = bad_chain;
        res[7] = fell;
        (void)lanes;
        break;
    }
    case 45: { /* SiLU path switch: arg 0 = scalar chain, 1 = vector */
        g_silu_vec = arg ? 1 : 0;
        res[2] = g_silu_vec;
        break;
    }
    default:
        return AEE_EBADPARM;
    }
    res[0] = (int)(pc1 - pc0);
    res[1] = (int)(t1 - t0);
    res[4] = arg;
    return 0;
}

int lfmgemv_release_buf(int h)
{
    if (h < 0 || h >= MAXBUF || !g_buf[h].used) return AEE_EBADPARM;
    if (g_buf[h].in_ptr) {
        if (g_buf[h].in_owned)
            apps_mem_share_unmap((uint64)(size_t)g_buf[h].in_ptr, g_buf[h].in_bytes);
        g_buf[h].in_ptr = 0;
        g_buf[h].in_bytes = 0;
    }
    if (g_buf[h].out_ptr)
        apps_mem_share_unmap((uint64)(size_t)g_buf[h].out_ptr, g_buf[h].out_bytes);
    free(g_buf[h].w_heap);
    free(g_buf[h].qp);
    free(g_buf[h].qm);
    free(g_buf[h].acc);
    free(g_buf[h].ae);
    free(g_buf[h].ao);
    memset(&g_buf[h], 0, sizeof(g_buf[h]));
    return 0;
}

/* ======================================================================
 * One-RPC token pipeline: a versioned command list living in ION/DSP memory.
 *
 * Why a list and not 48 calls: one FastRPC round trip costs ~145 us on this
 * device and does not go lower (rpc_test), so a 48-op token spends ~7 ms in
 * the driver.  The host already knows the graph for the next token, so the
 * whole sequence is published once and executed on the DSP in one call.
 *
 * Memory model
 *   - command list: host ION arena (map_arena) at byte offset cmd_off; it is
 *     written before the mapping is established and must not be rewritten
 *     while mapped (the DSP caches it, see attach_in notes);
 *   - weights: views of the same or another arena (alloc_view), read in
 *     place by the existing kernels;
 *   - activations: one DSP-heap scratch (reserve_scratch), addressed by byte
 *     offsets.  Nothing intermediate crosses the RPC boundary.
 *
 * Every length and offset is checked against its arena before use: this code
 * runs in the DSP process, where an out-of-bounds read is an SSR, not a
 * signal.  A malformed list returns AEE_EBADPARM with times[1] = index of the
 * offending command.
 * ====================================================================== */

#include "../conv/conv_math.h"

#define LFM_EXEC_MAGIC    0x454D464Cu   /* "LFME" */
#define LFM_EXEC_VERSION  1u
#define LFM_CMD_MAX       256

enum { LFM_CMD_QUANT = 0, LFM_CMD_GEMV = 1, LFM_CMD_FFN = 2,
       LFM_CMD_ATTN = 3, LFM_CMD_CONV = 4, LFM_CMD_DUMP = 5 };
enum { LFM_SRC_SCRATCH = 0, LFM_SRC_OUT = 1 };
enum { LFM_DST_SCRATCH = 0, LFM_DST_OUT = 1, LFM_DST_ATTACHED = 2 };

/* Header of the list.  magic/version/cmd_size are the first three checks; a
 * list built for another struct size is rejected instead of misread. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_cmds;        /* must equal the n_cmds argument */
    uint32_t cmd_size;      /* sizeof(lfm_cmd) */
    uint32_t list_bytes;    /* header + n_cmds * cmd_size */
    uint32_t scratch_bytes; /* scratch the list needs (<= reserve_scratch) */
    uint32_t flags;         /* 0 */
    uint32_t hdr_size;      /* sizeof(lfm_exec_hdr) */
    uint64_t reserved[4];
} lfm_exec_hdr;

/* One command, 96 B.  Field use per type:
 *
 *   QUANT h0=-1  src int32[n] -> dst int8[n], a0 = Q16 multiplier
 *   GEMV  h0=weight; a0=M (1|2|4), a1=reps; src int8[M*K] -> dst int8[M*N]
 *   FFN   h0/h1/h2=gate/up/down; a0=reps; src int8[K] -> dst int8[N]
 *   ATTN  h0/h1/h2=K/KT/V, h3=value-scale handle (reserved, pass 0);
 *         a0..a3 = pos,seq,window,reps, or rt[rt_base..rt_base+3] when
 *         rt_base >= 0; src int8[16*128] -> dst int32[16*128];
 *         block table at tbl_h (weights view, int32) or scratch[tbl_off]
 *   CONV  h0=quantised shortconv blob (8 phases wq8, then kq[H]); a0=H,
 *         a1=ksh, a2=slot, a3=token counter (or rt[rt_base] when
 *         rt_base >= 0); src int8[3H] -> dst int8[H];
 *         ring state at scratch[tbl_off], 2*H*8 bytes per slot
 */
typedef struct {
    uint32_t type;
    uint32_t src;       /* LFM_SRC_* */
    uint32_t dst;       /* LFM_DST_* */
    uint32_t pad;
    int32_t  h0, h1, h2, h3;
    int32_t  dst_h;     /* LFM_DST_ATTACHED: handle whose attach_out is used */
    int32_t  src_off, src_len;
    int32_t  dst_off, dst_len;
    int32_t  tbl_h;     /* ATTN: table view handle (-1: scratch[tbl_off]) */
    int32_t  tbl_len;   /* ATTN: table entries */
    int32_t  tbl_off;   /* ATTN table / CONV ring state offset in scratch */
    int32_t  a0, a1, a2, a3;
    int32_t  rt_base;   /* >= 0: index into the rt[] argument */
    int32_t  reserved[3];
} lfm_cmd;

typedef char lfm_exec_hdr_size_check[(sizeof(lfm_exec_hdr) == 64) ? 1 : -1];
typedef char lfm_cmd_size_check[(sizeof(lfm_cmd) == 96) ? 1 : -1];

static uint8_t *g_scratch;
static size_t   g_scratch_bytes;

/* shared int32 result buffer of the depthwise conv dot (2*H words) */
static int32_t *g_conv_res;
static int      g_conv_res_words;

/* ---------------- shortconv step (shared math with ../conv) ---------------- */

static void xconv_bx_write(int8_t *hi, int8_t *lo, int p,
                           const int8_t *b, const int8_t *x, int H)
{
    int ch;
    for (ch = 0; ch < H; ch++) {
        int32_t t = (int32_t)b[ch] * (int32_t)x[ch];
        hi[(size_t)ch * 8 + p] = (int8_t)(t >> 8);
        lo[(size_t)ch * 8 + p] = (int8_t)(t & 0xff);
    }
}

static void xconv_dot(const int8_t *hi, const int8_t *lo, const int8_t *wq,
                      int32_t *res, int ng)
{
    const HVX_Vector *W  = (const HVX_Vector *)wq;
    const HVX_Vector *PH = (const HVX_Vector *)hi;
    const HVX_Vector *PL = (const HVX_Vector *)lo;
    HVX_Vector z = Q6_V_vzero();
    int g;

    for (g = 0; g < ng; g++) {
        HVX_Vector w  = W[g];
        HVX_Vector ah = Q6_Vw_vrmpyacc_VwVbVb(z, PH[g], w);
        HVX_Vector al = Q6_Vw_vrmpyacc_VwVubVb(z, PL[g], w);
        HVX_Vector r  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(ah, 8), al);
        *(HVX_Vector *)(res + (size_t)g * 32) = r;
    }
}

static void xconv_pack_y(const int32_t *res, const int8_t *c, const int32_t *kq,
                         int ksh, int8_t *y, int H)
{
    int ch;
    for (ch = 0; ch < H; ch++)
        y[ch] = conv_glue(res[2 * ch] + res[2 * ch + 1], c[ch], kq[ch], ksh);
}

/* ---------------- bounds helpers ---------------- */

static int lfm_range_ok(int off, int len, size_t total)
{
    if (off < 0 || len < 0) return 0;
    if ((size_t)off > total) return 0;
    return (size_t)len <= total - (size_t)off;
}

static int lfm_src_ptr(const lfm_cmd *c, const unsigned char *out, int outLen,
                       const uint8_t **p)
{
    if (c->src == LFM_SRC_SCRATCH) {
        if (!lfm_range_ok(c->src_off, c->src_len, g_scratch_bytes)) return AEE_EBADPARM;
        *p = g_scratch + c->src_off;
        return 0;
    }
    if (c->src == LFM_SRC_OUT) {
        if (!out || !lfm_range_ok(c->src_off, c->src_len, (size_t)outLen)) return AEE_EBADPARM;
        *p = out + c->src_off;
        return 0;
    }
    return AEE_EBADPARM;
}

static int lfm_dst_ptr(const lfm_cmd *c, unsigned char *out, int outLen,
                       uint8_t **p)
{
    if (c->dst == LFM_DST_SCRATCH) {
        if (!lfm_range_ok(c->dst_off, c->dst_len, g_scratch_bytes)) return AEE_EBADPARM;
        *p = g_scratch + c->dst_off;
        return 0;
    }
    if (c->dst == LFM_DST_OUT) {
        if (!out || !lfm_range_ok(c->dst_off, c->dst_len, (size_t)outLen)) return AEE_EBADPARM;
        *p = out + c->dst_off;
        return 0;
    }
    if (c->dst == LFM_DST_ATTACHED) {
        gbuf_t *b;
        if (c->dst_h < 0 || c->dst_h >= MAXBUF) return AEE_EBADPARM;
        b = &g_buf[c->dst_h];
        if (!b->used || !b->out_ptr) return AEE_EBADPARM;
        if (!lfm_range_ok(c->dst_off, c->dst_len, (size_t)b->out_bytes)) return AEE_EBADPARM;
        *p = (uint8_t *)b->out_ptr + c->dst_off;
        return 0;
    }
    return AEE_EBADPARM;
}

/* ---------------- scratch ---------------- */

int lfmgemv_reserve_scratch(int nbytes)
{
    free(g_scratch);
    g_scratch = 0;
    g_scratch_bytes = 0;
    if (nbytes <= 0) return 0;
    g_scratch = (uint8_t *)memalign(128, (size_t)nbytes);
    if (!g_scratch) return AEE_ENOMEMORY;
    memset(g_scratch, 0, (size_t)nbytes);
    g_scratch_bytes = (size_t)nbytes;
    return 0;
}

static int lfm_conv_res_ensure(int words)
{
    if (g_conv_res_words >= words && g_conv_res) return 0;
    free(g_conv_res);
    g_conv_res = (int32_t *)memalign(128, (size_t)words * sizeof(int32_t));
    g_conv_res_words = g_conv_res ? words : 0;
    return g_conv_res ? 0 : AEE_ENOMEMORY;
}

int lfmgemv_reset_state(int flags)
{
    int f = (flags == 0) ? 3 : flags;

    if (f & ~3) return AEE_EBADPARM;
    /* The CONV ring states live in g_scratch; zeroing it is exactly the
       state a fresh reserve_scratch() leaves behind.  Arena mappings and
       resident weights are deliberately untouched: the reactive lists stay
       mapped and the next token starts from a clean sequence. */
    if ((f & 1) && g_scratch && g_scratch_bytes)
        memset(g_scratch, 0, g_scratch_bytes);
    if (f & 2) g_team_fallbacks = 0;
    return 0;
}

/* ---------------- the DAG executor ---------------- */

static int lfm_run_cmd(const lfm_cmd *c, const int *rt, int rt_n,
                       unsigned char *out, int outLen)
{
    const uint8_t *sp = 0;
    uint8_t *dp = 0;
    int rc, reps, n;

    if (c->type > LFM_CMD_DUMP) return AEE_EBADPARM;
    rc = lfm_src_ptr(c, out, outLen, &sp);
    if (rc != 0) return rc;
    rc = lfm_dst_ptr(c, out, outLen, &dp);
    if (rc != 0) return rc;

    switch (c->type) {
    case LFM_CMD_QUANT: {
        /* int32 Q12 -> int8 with one Q16 multiplier.  64-bit product, same
           rounding as pack_out_i8/conv_pack_i8. */
        const int32_t *s = (const int32_t *)sp;
        int8_t *d = (int8_t *)dp;
        int32_t m = c->a0;
        int i;
        if (c->src_len < 0 || (c->src_len & 3) != 0) return AEE_EBADPARM;
        n = c->src_len / 4;
        if (c->dst_len != n) return AEE_EBADPARM;
        for (i = 0; i < n; i++) {
            int64_t v = ((int64_t)s[i] * (int64_t)m + 32768) >> 16;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            d[i] = (int8_t)v;
        }
        return 0;
    }
    case LFM_CMD_GEMV:
        reps = c->a1 < 1 ? 1 : c->a1;
        return gemv_impl(c->h0, c->a0, sp, c->src_len, reps, dp, c->dst_len,
                         0, 0, 0);
    case LFM_CMD_FFN:
        reps = c->a0 < 1 ? 1 : c->a0;
        return ffn_impl(c->h0, c->h1, c->h2, sp, c->src_len, reps, dp,
                        c->dst_len, 0, 0, 0, 0, 0);
    case LFM_CMD_ATTN: {
        const int *tbl;
        int tbl_n = c->tbl_len, pos, seq, window, areps;
        if (tbl_n < 1 || tbl_n > (1 << 20)) return AEE_EBADPARM;
        if (c->tbl_h >= 0) {
            gbuf_t *b;
            if (c->tbl_h >= MAXBUF) return AEE_EBADPARM;
            b = &g_buf[c->tbl_h];
            if (!b->used || !b->w) return AEE_EBADPARM;
            if ((size_t)tbl_n * sizeof(int32_t) > b->wbytes) return AEE_EBADPARM;
            tbl = (const int *)b->w;
        } else {
            if (!lfm_range_ok(c->tbl_off, tbl_n * 4, g_scratch_bytes)) return AEE_EBADPARM;
            tbl = (const int *)(g_scratch + c->tbl_off);
        }
        if (c->rt_base >= 0) {
            if (rt == 0 || c->rt_base + 3 >= rt_n) return AEE_EBADPARM;
            pos = rt[c->rt_base + 0];
            seq = rt[c->rt_base + 1];
            window = rt[c->rt_base + 2];
            areps = rt[c->rt_base + 3];
        } else {
            pos = c->a0; seq = c->a1; window = c->a2; areps = c->a3;
        }
        return attn_core(c->h0, c->h1, c->h2, c->h3, sp, c->src_len, tbl, tbl_n,
                         pos, seq, window, areps, dp, c->dst_len, 0, 0);
    }
    case LFM_CMD_DUMP:
        /* Debug only: copy a scratch range into the OUT sequence so the host
           can compare kernel intermediates with its own model. */
        if (c->src != LFM_SRC_SCRATCH) return AEE_EBADPARM;
        if (!lfm_range_ok(c->src_off, c->src_len, g_scratch_bytes)) return AEE_EBADPARM;
        if (!lfm_range_ok(c->dst_off, c->src_len, (size_t)outLen)) return AEE_EBADPARM;
        memcpy(out + c->dst_off, g_scratch + c->src_off, (size_t)c->src_len);
        return 0;
    case LFM_CMD_CONV: {
        gbuf_t *b;
        int H = c->a0, ksh = c->a1, slot = c->a2, ctr = c->a3;
        const int8_t *wq8;
        const int32_t *kq;
        int8_t *st, *hi, *lo;
        size_t need, st_bytes;
        int p;
        if (c->rt_base >= 0) {
            /* The token counter is the only per-token field of CONV; taking
               it from rt[] keeps the command list byte-identical between
               tokens, which is what lets the host keep one mapped arena. */
            if (rt == 0 || c->rt_base >= rt_n) return AEE_EBADPARM;
            ctr = rt[c->rt_base];
        }
        if (c->h0 < 0 || c->h0 >= MAXBUF) return AEE_EBADPARM;
        b = &g_buf[c->h0];
        if (!b->used || !b->w) return AEE_EBADPARM;
        if (H < 16 || (H & 15) != 0 || H > 16384) return AEE_EBADPARM;
        if (ksh < 1 || ksh > 40) return AEE_EBADPARM;
        if (slot < 0 || slot >= 8) return AEE_EBADPARM;
        need = (size_t)8 * H * 8 + (size_t)4 * H;      /* 8 phases + kq */
        if (b->wbytes < need) return AEE_EBADPARM;
        if (c->src_len != 3 * H || c->dst_len != H) return AEE_EBADPARM;
        st_bytes = (size_t)2 * H * 8 * (slot + 1);
        if (!lfm_range_ok(c->tbl_off, (int)st_bytes, g_scratch_bytes)) return AEE_EBADPARM;
        rc = lfm_conv_res_ensure(2 * H);
        if (rc != 0) return rc;
        wq8 = (const int8_t *)b->w;
        kq = (const int32_t *)(b->w + (size_t)8 * H * 8);
        st = (int8_t *)g_scratch + c->tbl_off + (size_t)slot * 2 * H * 8;
        hi = st;
        lo = st + (size_t)H * 8;
        p = ctr & 7;
        xconv_bx_write(hi, lo, p, (const int8_t *)sp, (const int8_t *)sp + 2 * H, H);
        xconv_dot(hi, lo, wq8 + (size_t)p * (H / 16) * 128, g_conv_res, H / 16);
        xconv_pack_y(g_conv_res, (const int8_t *)sp + H, kq, ksh, (int8_t *)dp, H);
        return 0;
    }
    default:
        return AEE_EBADPARM;
    }
}

int lfmgemv_exec(int ah, int cmd_off, int n_cmds, const unsigned char *act, int actLen,
                 const int *rt, int rtLen, unsigned char *out, int outLen,
                 int *times, int timesLen)
{
    const lfm_exec_hdr *hdr;
    const lfm_cmd *cmds;
    int64_t t0, total = 0;
    int i, rc, done = 0;

    if (times && timesLen >= 4) { times[0] = 0; times[1] = -1; times[2] = 0; times[3] = 0; }
    if (ah < 0 || ah >= MAXARENA || !g_arena[ah].used) return AEE_EBADPARM;
    if (n_cmds < 1 || n_cmds > LFM_CMD_MAX) return AEE_EBADPARM;
    if (actLen < 0 || (actLen > 0 && !act)) return AEE_EBADPARM;
    if (rtLen < 0 || (rtLen > 0 && !rt)) return AEE_EBADPARM;
    if (outLen < 0 || (outLen > 0 && !out)) return AEE_EBADPARM;
    if (!lfm_range_ok(cmd_off, (int)sizeof(lfm_exec_hdr), (size_t)g_arena[ah].bytes))
        return AEE_EBADPARM;
    hdr = (const lfm_exec_hdr *)(g_arena[ah].base + cmd_off);
    if (hdr->magic != LFM_EXEC_MAGIC || hdr->version != LFM_EXEC_VERSION) return AEE_EBADPARM;
    if (hdr->hdr_size != (uint32_t)sizeof(lfm_exec_hdr) ||
        hdr->cmd_size != (uint32_t)sizeof(lfm_cmd)) return AEE_EBADPARM;
    if (hdr->n_cmds != (uint32_t)n_cmds) return AEE_EBADPARM;
    if (hdr->list_bytes > (1u << 20)) return AEE_EBADPARM;
    if (hdr->list_bytes < (uint32_t)(sizeof(lfm_exec_hdr) + (size_t)n_cmds * sizeof(lfm_cmd)))
        return AEE_EBADPARM;
    if (!lfm_range_ok(cmd_off, (int)hdr->list_bytes, (size_t)g_arena[ah].bytes))
        return AEE_EBADPARM;
    if ((size_t)hdr->scratch_bytes > g_scratch_bytes) return AEE_EBADPARM;
    if ((size_t)actLen > (size_t)hdr->scratch_bytes) return AEE_EBADPARM;
    set_perf();
    if (actLen) memcpy(g_scratch, act, (size_t)actLen);

    cmds = (const lfm_cmd *)((const uint8_t *)hdr + sizeof(lfm_exec_hdr));
    for (i = 0; i < n_cmds; i++) {
        t0 = HAP_perf_get_time_us();
        rc = lfm_run_cmd(&cmds[i], rt, rtLen, out, outLen);
        total += HAP_perf_get_time_us() - t0;
        if (rc != 0) {
            if (times && timesLen >= 4) {
                times[0] = (int)total;
                times[1] = i;
                times[2] = done;
            }
            return rc;
        }
        done++;
    }
    if (times && timesLen >= 4) {
        times[0] = (int)total;
        times[1] = -1;
        times[2] = done;
        times[3] = g_team_fallbacks;
    }
    return 0;
}
