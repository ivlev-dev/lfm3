/*
 * HMX int4 prefill with per-group scales, ported verbatim from
 * `dsp/hmx/mx4_impl.c` (`mx4_blkpref`, REPORT24).  Only the RPC plumbing is
 * dropped: operands are pointers into the caller's resident buffers.
 */
#include "prefill_impl.h"
#include "emit_impl.h"

#include <string.h>
#include <stdlib.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include <qurt_hvx.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_power.h>

#define PREF_MAXK 8192
#define PREF_MAXG 64

typedef struct {
    unsigned ctx;
    uint8_t *base;
    unsigned size;
} pref_res_t;

static int pref_res_get(unsigned vtcm_bytes, pref_res_t *r)
{
    compute_res_attr_t attr;
    unsigned ctx;
    void *p;

    r->ctx = 0;
    r->base = 0;
    r->size = 0;
    if (!compute_resource_attr_init) return -1;
    if (compute_resource_attr_init(&attr) != 0) return -2;
    if (compute_resource_attr_set_hmx_param &&
        compute_resource_attr_set_hmx_param(&attr, 1) != 0) return -3;
    if (compute_resource_attr_set_vtcm_param &&
        compute_resource_attr_set_vtcm_param(&attr, vtcm_bytes, 0) != 0) return -4;

    ctx = HAP_compute_res_acquire(&attr, 1000000);
    if (ctx == 0) return -5;
    p = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (p == 0) {
        HAP_compute_res_release(ctx);
        return -6;
    }
    r->ctx = ctx;
    r->base = (uint8_t *)p;
    r->size = vtcm_bytes;
    return 0;
}

static void pref_res_put(pref_res_t *r)
{
    if (r->ctx) {
        HAP_compute_res_release(r->ctx);
        r->ctx = 0;
    }
}

static int pref_hmx_power(int up)
{
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = up ? TRUE : FALSE;
    return HAP_power_set(0, &req);
}

static inline unsigned pref_align(unsigned v, unsigned a)
{
    return (v + a - 1u) & ~(a - 1u);
}

static inline void pref_hmx_clr(void)
{
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
}

static inline void pref_hmx_bias(const void *p)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}

static inline void pref_hmx_st_ub(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void pref_hmx_st_ub_raw(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void pref_hmx_st_uh(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void pref_hmx_st_uh_raw(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.uh = acc:2x1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void pref_hmx_sync(void)
{
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

static inline void pref_i4_mul(const void *act, uint32_t art,
                               const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(art),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
}

static inline void pref_copy_v2048(void *dst, const void *src)
{
    const HVX_Vector *s = (const HVX_Vector *)src;
    HVX_Vector *d = (HVX_Vector *)dst;
    int i;
    for (i = 0; i < 16; i++) d[i] = s[i];
}

static void pref_copy(uint8_t *dst, const uint8_t *src, unsigned bytes)
{
    unsigned i;
    for (i = 0; i < bytes; i += 2048u)
        pref_copy_v2048(dst + i, src + i);
    __asm__ __volatile__("syncht" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

static void pref_pack_act(uint8_t *dst, const int8_t *A, int M, int K, int ncr)
{
    int g, c;
    for (g = 0; g < M; g++)
        for (c = 0; c < K; c++)
            dst[(unsigned)(g >> 6) * (unsigned)ncr * 2048u
                + 2048u * (unsigned)(c >> 5)
                + 128u * (unsigned)((g & 63) >> 2)
                + 4u * (unsigned)(c & 31) + (unsigned)(g & 3)] =
                (uint8_t)A[(size_t)g * K + c];
}

static void pref_emit16(const uint8_t *part, int ng, const float *S,
                        const float *Q, float *C, int ldc, int rows, int n0)
{
    int s, n, g;
    for (s = 0; s < rows; s++)
        for (n = 0; n < 32; n++) {
            float acc = 0.0f;
            for (g = 0; g < ng; g++) {
                size_t off = (size_t)g * 4096u
                           + 2u * (128u * (unsigned)(s >> 2) + 4u * (unsigned)n
                                   + (unsigned)(s & 3));
                short p16;
                memcpy(&p16, part + off, 2);
                acc += S[(size_t)(n0 + n) * ng + g] *
                       (2.0f * (float)p16 - 128.0f * Q[(size_t)(n0 + n) * ng + g]);
            }
            C[(size_t)s * ldc + n0 + n] = acc;
        }
}

static float g_svec[128 * 64] __attribute__((aligned(128)));

int lfm_prefill(const int8_t *A, const uint8_t *B, const float *S,
                const float *Q, int M, int N, int K, int G, int C0, int mode,
                int reps, float *C, int *info, int info_len)
{
    pref_res_t r;
    uint8_t *base, *vact, *vwgt, *vpart, *vbias, *voact, *vowgt;
    int ncr, ng, mt, nt, rc, i, rep, m, c, g, j;
    unsigned aregion, wstrip, nbuf, need, chunk;
    uint64_t t0, t1, c0, c1, tc;
    int copycheck = 0, Aq, st16;
    float mstore;

    st16 = ((mode >> 8) & 3) == 3;

    if (info_len < 16) return -1;
    memset(info, 0, sizeof(int) * info_len);
    if (!A || !B || !S) return -2;
    if (K < 32 || K > PREF_MAXK || (K & 31)) return -2;
    if (N < 32 || (N & 31)) return -3;
    if (M < 1 || M > 256) return -4;
    if (G < 1 || G > PREF_MAXG) return -5;
    if (K % (32 * G)) return -6;
    if (C0 < 448 || (C0 % 448) != 0) return -10;
    Aq = C0 / 448;                 /* packet contributes 448*Aq (REPORT25 s.3) */
    if (Aq > 255) return -11;
    if ((mode & 8) && !C) return -12;
    if (reps < 1 || reps > 100) reps = 1;

    ncr = K / 32;
    ng = ncr / G;
    mt = (M + 63) / 64;
    nt = N / 32;
    wstrip = 512u * (unsigned)ncr;
    chunk = wstrip;
    nbuf = (mode & 1) ? 2u : 1u;
    mstore = 255.0f / (2.0f * (float)C0);

    aregion = 2048u * (unsigned)ncr * (unsigned)mt;
    need = aregion
         + nbuf * pref_align(chunk, 2048u)
         + (unsigned)ng * (st16 ? 4096u : 2048u) + 2048u
         + 512u + 2048u + 2048u + 2048u + 512u;
    rc = pref_res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    info[7] = (int)r.size;

    base = (uint8_t *)(((uintptr_t)r.base + 2047u) & ~(uintptr_t)2047u);
    vact = base;
    vwgt = vact + aregion;
    vpart = vwgt + nbuf * pref_align(chunk, 2048u);
    vbias = vpart + (unsigned)ng * (st16 ? 4096u : 2048u);
    voact = vbias + 2048u;          /* 2 KB granularity: see REPORT25 s.0.3 */
    vowgt = voact + 2048u;
    memset(base, 0, need);
    info[4] = nt;
    info[5] = mt;
    info[6] = ncr;
    info[9] = mode;
    info[13] = reps;
    info[15] = ng;

    t0 = HAP_perf_get_time_us();
    pref_pack_act(vact, A, M, K, ncr);
    info[10] = (int)(HAP_perf_get_time_us() - t0);
    if (!(mode & 1)) pref_copy(vwgt, B, wstrip);

    {
        int s, k, n;
        for (s = 0; s < 64; s++)
            for (k = 0; k < 32; k++)
                voact[128u * (unsigned)(s >> 2) + 4u * (unsigned)k
                      + (unsigned)(s & 3)] = (uint8_t)Aq;
        for (k = 0; k < 32; k++)
            for (n = 0; n < 32; n++) {
                unsigned off = 128u * (unsigned)(k >> 3) + 4u * (unsigned)n
                             + (unsigned)(k & 3);
                if ((k >> 2) & 1) vowgt[off] |= 0x70u;
                else              vowgt[off] |= 0x07u;
            }
    }
    {
        uint32_t word;
        __fp16 h = (__fp16)(mstore * 512.0f);
        unsigned short bits;
        __asm__ __volatile__("" ::: "memory");
        memcpy(&bits, &h, 2);
        word = (uint32_t)bits;
        if (((mode >> 8) & 3) != 0) word = 0x6000u;
        for (j = 0; j < 32; j++) {
            ((uint32_t *)vbias)[j] = word;
            ((uint32_t *)vbias)[64 + j] = word;
        }
    }

    if (!(mode & 4)) {
        qurt_hvx_lock(QURT_HVX_MODE_128B);
        pref_hmx_power(1);
        if (compute_resource_hmx_lock) compute_resource_hmx_lock(r.ctx);
    }

    if ((mode & 1) && !(mode & 2)) {
        tc = HAP_perf_get_time_us();
        pref_copy(vwgt, B, wstrip);
        info[2] += (int)(HAP_perf_get_time_us() - tc);
        info[14] += (int)wstrip;
        {
            int q, bad = 0;
            for (q = 0; q < (int)wstrip; q++) if (vwgt[q] != B[q]) bad++;
            copycheck = bad;
        }
    }

    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        for (i = 0; i < nt; i++) {
            uint8_t *slot = (mode & 1)
                ? vwgt + (unsigned)(i & 1) * pref_align(chunk, 2048u) : vwgt;
            if (mode & 2) {
                tc = HAP_perf_get_time_us();
                pref_copy(vwgt + (unsigned)(i & 1) * pref_align(chunk, 2048u),
                          B + (size_t)i * wstrip, wstrip);
                info[2] += (int)(HAP_perf_get_time_us() - tc);
                info[14] += (int)wstrip;
                continue;
            }
            if (rep == 0 && i == 0) {
                int q, bad = 0;
                for (q = 0; q < (int)wstrip; q++)
                    if (slot[q] != B[q]) bad++;
                copycheck = bad;
            }
            for (m = 0; m < mt; m++) {
                int rows = (M - m * 64) < 64 ? (M - m * 64) : 64;
                for (g = 0; g < ng; g++) {
                    int aq = Q ? (int)Q[g] : Aq;
                    if (aq < 0) aq = 0;
                    if (aq > 255) aq = 255;
                    if (Q) {
                        int ss2, kk2;
                        for (ss2 = 0; ss2 < 64; ss2++)
                            for (kk2 = 0; kk2 < 32; kk2++)
                                voact[128u * (unsigned)(ss2 >> 2)
                                      + 4u * (unsigned)kk2
                                      + (unsigned)(ss2 & 3)] = (uint8_t)aq;
                    }
                    if (Q) {
                        float mg = 145.7142857f / (float)(aq < 1 ? 1 : aq);
                        __fp16 hh = (__fp16)mg;
                        unsigned short bb;
                        uint32_t ww;
                        int jj;
                        memcpy(&bb, &hh, 2);
                        ww = (uint32_t)bb;
                        for (jj = 0; jj < 32; jj++) {
                            ((uint32_t *)vbias)[jj] = ww;
                            ((uint32_t *)vbias)[64 + jj] = ww;
                        }
                    }
                    pref_hmx_clr();
                    pref_i4_mul(voact, 0x3FFu, vowgt, 511u);
                    for (c = 0; c < G; c++)
                        pref_i4_mul(vact + (unsigned)(m * ncr + g * G + c) * 2048u,
                                    0x3FFu,
                                    slot + (unsigned)(g * G + c) * 512u, 511u);
                    pref_hmx_bias(vbias);
                    switch ((mode >> 8) & 3) {
                    case 0: pref_hmx_st_ub(vpart + (unsigned)g * 2048u, 0); break;
                    case 1: pref_hmx_st_ub(vpart + (unsigned)g * 2048u, 0); break;
                    case 2: pref_hmx_st_ub_raw(vpart + (unsigned)g * 2048u, 0); break;
                    default: pref_hmx_st_uh(vpart + (unsigned)g * 4096u, 0); break;
                    }
                }
                if (m + 1 == mt && (mode & 1) && i + 1 < nt) {
                    uint8_t *nslot = vwgt
                        + (unsigned)((i + 1) & 1) * pref_align(chunk, 2048u);
                    tc = HAP_perf_get_time_us();
                    pref_copy(nslot, B + (size_t)(i + 1) * wstrip, wstrip);
                    info[2] += (int)(HAP_perf_get_time_us() - tc);
                    info[14] += (int)wstrip;
                }
                pref_hmx_sync();
                if ((mode & 16) && (mode & 8)) {
                    int q2;
                    if (!st16 && i == 0 && m == 0) {
                        for (q2 = 0; q2 < ng && q2 < M * N; q2++)
                            C[q2] = (float)vpart[(size_t)q2 * 2048u];
                    }
                    if (st16) {
                        for (q2 = 0; q2 < rows * 16 && q2 < M * N; q2++) {
                            short p16;
                            memcpy(&p16, vpart + 2u * (unsigned)q2, 2);
                            C[q2] = (float)p16;
                        }
                    } else {
                        for (q2 = 0; q2 < rows * 32 && q2 < M * N; q2++)
                            C[q2] = (float)vpart[q2];
                    }
                } else if (mode & 8) {
                    tc = HAP_perf_get_time_us();
                    if (st16) {
                        if (!Q) return -13;
                        pref_emit16(vpart, ng, S, Q, C, N, rows, i * 32);
                    } else {
                        pref_emit(vpart, ng, S, Q, g_svec, C, N, rows, m * 64,
                                  i * 32);
                    }
                    info[11] += (int)(HAP_perf_get_time_us() - tc);
                }
            }
        }
    }
    pref_hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)(t1 - t0);
    info[3] = (int)(c1 - c0);
    info[12] = copycheck;

    if (!(mode & 4)) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        pref_hmx_power(0);
        qurt_hvx_unlock();
    }
    pref_res_put(&r);
    info[0] = 0;
    return 0;
}
