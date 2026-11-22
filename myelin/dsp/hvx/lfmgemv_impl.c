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

#include "silu_lut.h"
#include "exptab.h"
#include "apps_mem.h"

typedef void *Address;

#define MAXBUF 8

typedef struct {
    int      used;
    int      N, K;
    int      nvec;      /* K/256 : 128-byte weight vectors per row */
    uint8_t *w;         /* N*K/2 bytes, 128B aligned */
    int32_t *qp;        /* N Q16 dequant multipliers (low16), or NULL -> int32 out */
    int32_t *qp30;      /* N Q30 multipliers (ffn gate/up dequant, scalar path) */
    int32_t *acc;       /* 4*N int32 scratch (M<=4) */
    int8_t  *ae, *ao;   /* 8*(K/2+128) scratch, 128B aligned */
    size_t   wbytes;
    int      w8;        /* 1 = int8 weights (N*K bytes, stored offset-128) */
    int8_t  *out_ptr;   /* host ION buffer mapped via apps_mem_share_map */
    int      out_bytes;
    float    sa, so;    /* activation / output scales last given to set_scale */
} gbuf_t;

static gbuf_t g_buf[MAXBUF];

typedef struct {
    int32_t *acc2;
    int8_t  *h;
    int      n;
} ffn_t;
static ffn_t g_ffn;

/* tuning */
static int g_pf_kb   = 0;   /* l2fetch lookahead in KB (0 = off) */
static int g_rows    = 1;   /* rows per iteration in the specialised kernel */
static int g_usepf   = 0;
static int g_threads = 1;   /* row-range split across threads */
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
 * ~4 LSB of 127.  pack_out_i8() below takes the product in 64 bits. */
static inline int32_t mulq16(int32_t a, int32_t b)
{
    int32_t hi = a >> 16;
    uint32_t lo = ((uint32_t)a >> 8) & 0xFFu;
    return hi * b + (int32_t)(((uint32_t)lo * (uint32_t)b) >> 8);
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
 * The product is taken in 64 bits: mulq16(acc,q) resolves acc with a step of
 * 256, so its int8 error is (acc & 0xff)*m -- fine when m ~ 1, but ~4 LSB of
 * 127 when m ~ 0.015 (the FFN down / out_proj layers).  Kept bit-exact with
 * conv_pack_i8() in myelin/dsp/conv/conv_math.h (the host reference must
 * match the DSP).
 * Scalar on purpose: N is 2k..8k, ~8 cycles/element = 15..45 us, and the
 * vector vmpyiewh form turned out to be numerically unreliable here. */
static void pack_out_i8(const int32_t *acc, const int32_t *qv, int8_t *out, int n)
{
    int i;
    for (i = 0; i < n; i++) {
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
    int           kind;      /* 0 = gemv rows, 1 = silu range, 2 = attention groups */
    const gbuf_t *b;
    const int8_t *ae, *ao;
    int32_t       bias;
    int           r0, r1;
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

static void worker_body(thw_t *w) __attribute__((noinline));
static void silu_range(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                       int32_t mq_mid);
static void attn_group(int kh);
static void attn_group_tm(actx_t *ctx, int kh);

/* thin, non-HVX entry: creating a thread whose entry prologue touches HVX
 * registers makes qurt_thread_create fail with -1 on this target */
void team_worker(void *a)
{
    worker_body((thw_t *)a);
}

static void worker_body(thw_t *w)
{
    for (;;) {
        qurt_sem_down(&w->go);
        if (w->shutdown) break;
        /* lock per job: holding a context for the thread's lifetime blocks
         * other workers when there are fewer contexts than threads */
        {
            m1job_t *j = &g_team.job[w->idx];
            if (j->kind == 2) {
                int kg;
                qurt_hvx_lock(QURT_HVX_MODE_128B);
                if (j->ctx) {
                    /* fully local context: nothing but the job is shared */
                    for (kg = j->r0; kg < j->r1; kg++) attn_group_tm((actx_t *)j->ctx, kg);
                } else {
                    for (kg = j->r0; kg < j->r1; kg++) attn_group(kg);
                }
                qurt_hvx_unlock();
            } else if (j->kind == 1) {
                silu_range(j->r0, j->r1, j->b, (const gbuf_t *)j->ae, j->bias);
            } else {
                qurt_hvx_lock(QURT_HVX_MODE_128B);
                gemv_rows_m1_k2048_range(j->b, j->ae, j->ao, j->bias, j->r0, j->r1, j->acc);
                qurt_hvx_unlock();
            }
        }
        qurt_sem_up(&w->done);
    }
}

/* ntotal = main thread + workers; returns workers actually created */
static int team_start(int ntotal)
{
    int i, made = 0;

    if (ntotal < 2) return 0;
    {
        int units = (int)((qurt_hvx_get_units() >> 8) & 0xFF);
        /* leave one HVX context for the main thread */
        if (units > 1 && ntotal - 1 > units - 1) ntotal = units;
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
    for (i = 1; i < total; i++) qurt_sem_up(&g_team.w[i].go);
    {
        m1job_t *j = &g_team.job[0];
        gemv_rows_m1_k2048_range(j->b, j->ae, j->ao, j->bias, j->r0, j->r1, j->acc);
    }
    for (i = 1; i < total; i++) qurt_sem_down(&g_team.w[i].done);
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

/* SiLU + int8 quantisation over an index range (scalar, 4x unrolled) */
static void silu_range(int i0, int i1, const gbuf_t *g, const gbuf_t *u,
                       int32_t mq_mid)
{
    int i4 = i0;
    for (; i4 + 4 <= i1; i4 += 4) {
        int32_t hv[4];
        int t;
        for (t = 0; t < 4; t++) {
            int k = i4 + t;
            int32_t zg = mulq16(g->acc[k], g->qp30[k] >> 14);
            int32_t zu = mulq16(g_ffn.acc2[k], u->qp30[k] >> 14);
            int32_t pr = mulq16(silu_q16(zg), zu);
            int32_t v = mulq16(pr >> 12, mq_mid);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            hv[t] = v;
        }
        g_ffn.h[i4 + 0] = (int8_t)hv[0];
        g_ffn.h[i4 + 1] = (int8_t)hv[1];
        g_ffn.h[i4 + 2] = (int8_t)hv[2];
        g_ffn.h[i4 + 3] = (int8_t)hv[3];
    }
    for (; i4 < i1; i4++) {
        int32_t zg = mulq16(g->acc[i4], g->qp30[i4] >> 14);
        int32_t zu = mulq16(g_ffn.acc2[i4], u->qp30[i4] >> 14);
        int32_t pr = mulq16(silu_q16(zg), zu);
        int32_t v = mulq16(pr >> 12, mq_mid);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        g_ffn.h[i4] = (int8_t)v;
    }
}

/* split the SiLU pass across the team (it is the largest non-GEMM phase) */
static void run_split_silu(const gbuf_t *g, const gbuf_t *u, int32_t mq_mid)
{
    int total = g_team.nwork + 1;
    int i;
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
    for (i = 1; i < total; i++) qurt_sem_up(&g_team.w[i].go);
    silu_range(0, (int)((int64_t)g->N / total), g, u, mq_mid);
    for (i = 1; i < total; i++) qurt_sem_down(&g_team.w[i].done);
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
    if (g->N != u->N || g->K != u->K || d->N != u->K || d->K != g->N) return AEE_EBADPARM;
    if (actLen != g->K) return AEE_EBADPARM;
    if (use_resident) {
        if (!d->out_ptr || d->out_bytes < d->N) return AEE_EBADPARM;
    } else if (dstLen != d->N) return AEE_EBADPARM;
    if (!g->qp || !u->qp || !d->qp || !d->sa) return AEE_EBADPARM;
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

int lfmgemv_attn(int hk, int hkt, int hv, int hvs, const unsigned char *q, int qLen,
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
                for (i = 1; i < total; i++) qurt_sem_up(&g_team.w[i].go);
                for (i = g_team.job[0].r0; i < g_team.job[0].r1; i++) {
                    if (g_attn_tm) attn_group_tm(&ctx, i); else attn_group(i);
                }
                for (i = 1; i < total; i++) qurt_sem_down(&g_team.w[i].done);
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
    }
    return 0;
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
    res[6] = g_rows;
    res[7] = g_usepf;
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
    g_buf[i].used = 1;
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
    free(b->acc); free(b->ae); free(b->ao); free(b->qp); free(b->qp30);
    b->acc  = (int32_t *)memalign(128, (size_t)4 * N * sizeof(int32_t));
    b->ae   = (int8_t *)memalign(128, 8 * half);
    b->ao   = (int8_t *)memalign(128, 8 * half);
    b->qp   = (with_scale & 1) ? (int32_t *)memalign(128, (size_t)N * sizeof(int32_t)) : 0;
    b->qp30 = (with_scale & 1) ? (int32_t *)memalign(128, (size_t)N * sizeof(int32_t)) : 0;
    if (!b->acc || !b->ae || !b->ao || ((with_scale & 1) && (!b->qp || !b->qp30))) return AEE_ENOMEMORY;
    set_perf();
    return 0;
}

int lfmgemv_load_buf(int h, int off, const unsigned char *data, int dataLen)
{
    gbuf_t *b;
    if (h < 0 || h >= MAXBUF || !g_buf[h].used || !data || dataLen < 0) return AEE_EBADPARM;
    b = &g_buf[h];
    if ((size_t)off + (size_t)dataLen > b->wbytes) return AEE_EBADPARM;
    memcpy(b->w + off, data, (size_t)dataLen);
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
    if (N != b->N || K != b->K || scalesLen < 16 + 4 * N || !b->qp) return AEE_EBADPARM;
    b->sa = sa;
    b->so = so;
    sw = (const float *)(scales + 16);
    for (i = 0; i < N; i++) {
        float m = sw[i] * sa / so;
        unsigned q;
        if (m < 0.0f) m = 0.0f;
        if (m > 0.99997f) m = 0.99997f;
        q = (unsigned)(m * 65536.0f + 0.5f);
        if (q > 65535u) q = 65535u;
        b->qp[i] = (int32_t)(q | (q << 16));
        {
            int64_t q30 = (int64_t)((double)m * 1073741824.0 + 0.5);
            if (q30 > 0x7fffffffLL) q30 = 0x7fffffffLL;
            b->qp30[i] = (int32_t)q30;
        }
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
        res[2] = g_pf_kb;
        res[3] = g_rows;
        res[5] = g_usepf;
        res[6] = g_threads;
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
    free(g_buf[h].w);
    free(g_buf[h].qp);
    free(g_buf[h].qp30);
    free(g_buf[h].acc);
    free(g_buf[h].ae);
    free(g_buf[h].ao);
    memset(&g_buf[h], 0, sizeof(g_buf[h]));
    return 0;
}
