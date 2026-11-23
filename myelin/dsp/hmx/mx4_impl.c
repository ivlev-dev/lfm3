/*
 * mx4 - minimal int8 HMX GEMM, safety first.
 *
 * Design rules (derived from the HMX PRM + QNN v68 kernels):
 *   - every 32x32 int8 crouton lives in its own 2 KB aligned, 2 KB sized,
 *     zero padded VTCM slot (1 KB payload + 1 KB zeros).  The activation
 *     address field in the HMX instruction is Rs[31:11] (2 KB aligned) and
 *     Rs[10:2] carry spatial/channel fields, so a 1 KB-offset activation
 *     pointer would silently corrupt those fields -> undefined behaviour was
 *     the most likely reason the previous attempt took down the CDSP.
 *   - zero padding makes the result immune to whether the hardware reads
 *     1 KB or 2 KB per crouton, and keeps any over-read inside our own
 *     VTCM block (a VTCM access outside the allocation is an exception).
 *   - weight dW (wc) covers the whole 2 KB slot, so HMX range checking never
 *     sees an address beyond the slot.
 *   - no bias register load, no :deep, no :cm, no :above in the first steps:
 *     the minimal instruction stream is mxclracc + act/wgt pair + acc store.
 *
 * VTCM tile layouts (payload = first 1 KB of a 2 KB slot):
 *   activation  off = (s>>2)*128 + c*4 + (s&3)
 *   weight      off = (k>>2)*128 + n*4 + (k&3)
 *   output      off = (s>>2)*128 + n*4 + (s&3)
 */
#include "mx4.h"
#include "HAP_farf.h"
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_power.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#define SLOT      2048u
#define PAYLOAD   1024u
#define TILE      32
#define ACT_CTRL  0x0000007Cu
#define WGT_CTRL  1023u

/* ------------------------------------------------------------------ */
/* resource handling (recipe verified on this device)                   */
/* ------------------------------------------------------------------ */

typedef struct {
    unsigned ctx;
    uint8_t *base;
    unsigned size;
} res_t;

static int res_get(unsigned vtcm_bytes, res_t *r)
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

    r->ctx  = ctx;
    r->base = (uint8_t *)p;
    r->size = vtcm_bytes;

    FARF(ALWAYS, "mx4: vtcm %u bytes at %p ctx=%u", vtcm_bytes, p, ctx);
    return 0;
}

static void res_put(res_t *r)
{
    if (r->ctx) {
        HAP_compute_res_release(r->ctx);
        r->ctx = 0;
    }
}

/* HMX must be powered on before any VTCM access from HMX.  QNN does this
 * through HAP_power_set(HAP_power_set_HMX); without it mxclracc "works" but
 * MXMEM loads return garbage and stores never retire (watchdog -> SSR). */
static int hmx_power(int up)
{
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = up ? TRUE : FALSE;
    return HAP_power_set(0, &req);
}

static int hmx_clock(void)
{
    HAP_power_response_t resp;
    memset(&resp, 0, sizeof(resp));
    resp.type = HAP_power_get_hmx_core_clk_Freq;
    if (HAP_power_get(0, &resp) != 0) return -1;
    return (int)resp.clkFreqHz;
}

static inline unsigned align_up(unsigned v, unsigned a)
{
    return (v + a - 1u) & ~(a - 1u);
}

/* ------------------------------------------------------------------ */
/* HMX instruction helpers                                             */
/* ------------------------------------------------------------------ */

static inline void hmx_clr(void)
{
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
}

static inline void hmx_bias(const void *p)
{
    __asm__ __volatile__("{\n\t bias = mxmem2(%0)\n\t}"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}

/* activation + weight must be in one packet: together they form one multiply */
static inline void hmx_mul(const void *act, uint32_t ac,
                           const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(ac),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc)
        : "memory");
}

/* compact (8-bit packed) activation form, as used by the QNN v68 int8 kernels */
static inline void hmx_mul_cm(const void *act, uint32_t ac,
                              const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.ub = mxmem(%0,%1):cm\n\t weight.b = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(ac),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc)
        : "memory");
}

static inline void hmx_st_ub(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_st_ub_cm(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:cm:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_st_uh(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:sat.uh = acc:2x1\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_st_ub_nosat(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_st_ub_cm_nosat(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:cm.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_st_ub_cm_before(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):before:cm:sat.ub = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_sync(void)
{
    /* mxclracc retires after the :after store; barrier keeps the scalar
     * VTCM reads below from being hoisted above it. */
    __asm__ __volatile__("{\n\t mxclracc\n\t}" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

/* one 32 (spatial) x 32 (out ch) uint8 tile -> row-major C[ldc] */
static void unpack_tile(const uint8_t *src, int rows, int cols,
                        uint8_t *C, int ldc)
{
    int s, n;
    for (s = 0; s < rows; s++) {
        const uint8_t *line = src + (unsigned)(s >> 2) * 128u + (unsigned)(s & 3);
        uint8_t *dst = C + (size_t)s * ldc;
        for (n = 0; n < cols; n++)
            dst[n] = line[(unsigned)n * 4u];
    }
}

/* ------------------------------------------------------------------ */
/* info / recipe                                                       */
/* ------------------------------------------------------------------ */

int mx4_info(int *r, int rLen)
{
    unsigned page_size = 0, page_count = 0, avail = 0, max_page = 0, npages = 0;
    int rc1, rc2;

    if (rLen < 8) return -1;
    memset(r, 0, sizeof(int) * rLen);

    rc1 = HAP_query_total_VTCM(&page_size, &page_count);
    rc2 = HAP_query_avail_VTCM(&avail, &max_page, &npages);

    r[0] = rc1;
    r[1] = (int)page_size;
    r[2] = (int)page_count;
    r[3] = rc2;
    r[4] = (int)avail;
    r[5] = (int)max_page;
    r[6] = (int)npages;
    r[7] = (compute_resource_hmx_lock != 0) | ((compute_resource_release != 0) << 1);

    FARF(ALWAYS, "mx4_info: total rc=%d page=%u pages=%u avail rc=%d %u bytes",
         rc1, page_size, page_count, rc2, avail);
    return 0;
}

int mx4_recipe(int *r, int rLen)
{
    res_t res;
    int rc;

    if (rLen < 8) return -1;
    memset(r, 0, sizeof(int) * rLen);

    rc = res_get(64u * 1024u, &res);
    r[0] = rc;
    if (rc != 0) return 0;

    r[3] = (int)res.size;
    r[4] = (int)res.ctx;

    r[1] = compute_resource_hmx_lock ? compute_resource_hmx_lock(res.ctx) : -100;
    if (r[1] == 0) {
        hmx_clr();
        __asm__ __volatile__("barrier" ::: "memory");
        r[2] = compute_resource_hmx_unlock ? compute_resource_hmx_unlock(res.ctx) : -101;
    }
    res_put(&res);
    r[7] = 0x4D58;
    return 0;
}

/* ------------------------------------------------------------------ */
/* one tile stream: ncr multiplies + one acc store                      */
/* ------------------------------------------------------------------ */

int mx4_tile(const unsigned char *act, int actLen,
             const unsigned char *wgt, int wgtLen,
             int ac, int wc, int srt, int variant, int ops,
             unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need;
    int rc, i, ncr;

    if (infoLen < 6) return -1;
    memset(info, 0, sizeof(int) * 6);
    if (rawLen < (int)SLOT) return -3;
    if (actLen <= 0 || (actLen % (int)SLOT) || wgtLen != actLen) return -2;

    ncr = actLen / (int)SLOT;
    need = (unsigned)actLen + (unsigned)wgtLen + SLOT + 2048u;

    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    info[5] = (int)need;
    info[1] = ac;
    info[2] = wc;
    info[3] = variant;
    info[4] = ncr;

    vact  = r.base;
    vwgt  = vact + (unsigned)actLen;
    vout  = vwgt + (unsigned)wgtLen;
    vbias = vout + SLOT;

    memset(r.base, 0, need);
    memcpy(vact, act, (size_t)actLen);
    memcpy(vwgt, wgt, (size_t)wgtLen);
    if (ops & 16) {
        /* leave the bias block zeroed */
    } else {
        /* 32 x 64-bit bias entries: identity (input bias 0, shape 0,
         * scale 1.0 fp16, output bias 0) */
        /* PRM: 256 B = vector1 (low 32 bits of bias 0..31) then vector2
         * (high 32 bits). low word = {out bias[31:16], scale fp16[15:0]}. */
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;      /* scale = 1.0 */
        for (i = 0; i < 32; i++) w[32 + i] = 0u;          /* rest zero */
    }

    if ((uintptr_t)vact & (SLOT - 1u)) { info[0] = -30; res_put(&r); return 0; }
    if ((uintptr_t)vwgt & (SLOT - 1u)) { info[0] = -31; res_put(&r); return 0; }
    if ((uintptr_t)vout & (SLOT - 1u)) { info[0] = -32; res_put(&r); return 0; }
    if ((uintptr_t)vbias & 255u)       { info[0] = -33; res_put(&r); return 0; }

    if (compute_resource_hmx_lock == 0) { info[0] = -40; res_put(&r); return 0; }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    if (compute_resource_hmx_lock(r.ctx) != 0) { info[0] = -20; res_put(&r); return 0; }

    hmx_clr();
    if (ops & 1) {
        for (i = 0; i < ncr; i++) {
            /* 8-bit crouton = 32 spatial x 16 channels x 2 B = 1 KB, so a
             * 32-channel K block needs two multiplies 1 KB apart */
            if (ops & 32) {
                if (variant & 1) {
                    hmx_mul_cm(vact + (unsigned)i * SLOT, (uint32_t)ac,
                               vwgt + (unsigned)i * SLOT, (uint32_t)wc);
                    hmx_mul_cm(vact + (unsigned)i * SLOT + 1024u, (uint32_t)ac,
                               vwgt + (unsigned)i * SLOT + 1024u, (uint32_t)wc);
                } else {
                    hmx_mul(vact + (unsigned)i * SLOT, (uint32_t)ac,
                            vwgt + (unsigned)i * SLOT, (uint32_t)wc);
                    hmx_mul(vact + (unsigned)i * SLOT + 1024u, (uint32_t)ac,
                            vwgt + (unsigned)i * SLOT + 1024u, (uint32_t)wc);
                }
            } else if (variant & 1) {
                hmx_mul_cm(vact + (unsigned)i * SLOT, (uint32_t)ac,
                           vwgt + (unsigned)i * SLOT, (uint32_t)wc);
            } else {
                hmx_mul(vact + (unsigned)i * SLOT, (uint32_t)ac,
                        vwgt + (unsigned)i * SLOT, (uint32_t)wc);
            }
        }
    }
    if (ops & 2) {
        /* QNN v68 order: multiply -> bias load -> convert transfer */
        if (!(ops & 8)) hmx_bias(vbias);
        switch ((variant >> 1) & 7) {
        case 1: hmx_st_ub_cm(vout, (uint32_t)srt); break;
        case 2: hmx_st_ub_cm_before(vout, (uint32_t)srt); break;
        case 3: hmx_st_uh(vout, (uint32_t)srt); break;
        case 4: hmx_st_ub_nosat(vout, (uint32_t)srt); break;
        case 5: hmx_st_ub_cm_nosat(vout, (uint32_t)srt); break;
        default: hmx_st_ub(vout, (uint32_t)srt); break;
        }
    }
    hmx_sync();

    memcpy(raw, vout, SLOT);
    info[0] = 0;

    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

/* ------------------------------------------------------------------ */
/* VTCM diagnostics                                                    */
/* ------------------------------------------------------------------ */

static inline void hmx_bias_store(const void *p)
{
    __asm__ __volatile__("{\n\t mxmem2(%0) = bias\n\t}"
                         :: "r"((uint32_t)(uintptr_t)p) : "memory");
}

static res_t  g_res;
static int    g_have;
static uint8_t *g_src, *g_dst;

static int g_get(void)
{
    int rc;
    if (g_have) return 0;
    rc = res_get(64u * 1024u, &g_res);
    if (rc != 0) return rc;
    g_src = g_res.base;
    g_dst = g_src + 32768u;
    g_have = 1;
    return 0;
}

int mx4_pos(int side, int q, int ac, int wc, int *r, int rLen)
{
    res_t res;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * SLOT + 2048u;
    int rc, i;

    if (rLen < 5) return -1;
    memset(r, 0, sizeof(int) * rLen);
    if (side < 3 && (q < 0 || q >= (int)SLOT)) return -2;

    rc = res_get(need, &res);
    r[0] = rc;
    if (rc != 0) return 0;

    vact = res.base;
    vwgt = vact + SLOT;
    vout = vwgt + SLOT;
    vbias = vout + SLOT;

    memset(res.base, 0, need);
    if (side == 4) {
        /* activation range probe: q = (start << 11) | len */
        int start = q >> 11, len = q & 0x7ff, k;
        if (start + len > (int)SLOT) len = (int)SLOT - start;
        memset(vwgt, 1, SLOT);
        for (k = 0; k < len; k++) vact[start + k] = 1;
    } else if (side == 3) {
        /* range probe: q = (start << 11) | len ; weight bytes [start,start+len)=1 */
        int start = q >> 11, len = q & 0x7ff, k;
        if (start + len > (int)SLOT) len = (int)SLOT - start;
        memset(vact, 1, SLOT);
        for (k = 0; k < len; k++) vwgt[start + k] = 1;
    } else if (side == 2) {
        memset(vact, 1, SLOT);
        memset(vwgt, 1, SLOT);
    } else if (side) {
        /* probe the weight tile: everything 0, single 1 at byte q; act all 1 */
        memset(vact, 1, SLOT);
        vwgt[q] = 1;
    } else {
        /* probe the activation tile: everything 0, single 1; weights all 1 */
        vact[q] = 1;
        memset(vwgt, 1, SLOT);
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
        for (i = 0; i < 32; i++) w[32 + i] = 0u;
    }

    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(res.ctx) != 0) {
        r[0] = -20; res_put(&res); return 0;
    }
    if (hmx_power(1) != 0) { r[0] = -50; res_put(&res); return 0; }
    hmx_clr();
    hmx_mul(vact, (uint32_t)ac, vwgt, (uint32_t)wc);
    hmx_bias(vbias);
    hmx_st_uh(vout, 0);
    hmx_sync();
    /* rows 0..3 of output channel 0 (spatial cells are contiguous, 2 B each) */
    r[1] = vout[4] | (vout[5] << 8);
    r[2] = vout[8] | (vout[9] << 8);
    r[3] = vout[12] | (vout[13] << 8);
    r[4] = vout[16] | (vout[17] << 8);
    compute_resource_hmx_unlock(res.ctx);
    hmx_power(0);
    res_put(&res);
    return 0;
}

int mx4_probe(int mode, int *r, int rLen)
{
    res_t res;
    uint8_t *src, *dst;
    unsigned avail = 0, max_page = 0, npages = 0;
    unsigned sum = 0;
    int rc, i;

    if (rLen < 8) return -1;
    memset(r, 0, sizeof(int) * rLen);

    if (mode & 0x18) {
        /* persistent block, split across RPC calls */
        rc = g_get();
        r[0] = rc;
        if (rc != 0) return 0;
        r[1] = (int)(uintptr_t)g_res.base;
        r[2] = (int)g_res.size;
        if (mode & 8) {
            /* write: bias load from g_src, bias store to g_dst, NO read of g_dst */
            int i;
            for (i = 0; i < 256; i++) g_src[i] = (uint8_t)(0x20 + (i & 0x3f));
            for (i = 0; i < 256; i++) g_dst[i] = 0;
            if (!compute_resource_hmx_lock || compute_resource_hmx_lock(g_res.ctx) != 0) {
                r[0] = -20; return 0;
            }
            hmx_clr();
            hmx_bias(g_src);
            hmx_bias_store(g_dst);
            hmx_clr();
            __asm__ __volatile__("barrier" ::: "memory");
            compute_resource_hmx_unlock(g_res.ctx);
            r[5] = 0x5157;
            return 0;
        }
        if (mode & 16) {
            /* read back the persistent block (separate RPC call later) */
            unsigned sum = 0;
            int i;
            for (i = 0; i < 256; i++) sum += g_dst[i];
            r[4] = (int)sum;
            r[3] = (g_dst[0] << 8) | g_dst[255];
            r[5] = 0x5244;
            return 0;
        }
    }

    rc = res_get(16u * 1024u, &res);
    r[0] = rc;
    if (rc != 0) return 0;

    r[1] = (int)(uintptr_t)res.base;
    r[2] = (int)res.size;
    HAP_query_avail_VTCM(&avail, &max_page, &npages);
    r[5] = (int)avail;
    r[6] = (int)max_page;

    src = res.base;
    dst = src + 8192u;

    /* mode bit1: hand HMX VTCM-relative offsets instead of the MMU-mapped
     * virtual pointer returned by HAP (the user PD maps VTCM at 0xfe......) */
    if (mode & 2) {
        uintptr_t b = (uintptr_t)res.base;
        r[7] = (int)(uintptr_t)(((uintptr_t)src - b) | (((uintptr_t)dst - b) << 16));
    }

    /* scalar write/read */
    for (i = 0; i < 64; i++) src[i] = (uint8_t)(0x40 + i);
    r[3] = (src[7] << 8) | src[63];

    if (mode & 64) {
        /* power on HMX, then bias load+store roundtrip with readback */
        unsigned ssum = 0, dsum = 0;
        r[6] = hmx_power(1);
        r[7] = hmx_clock();
        for (i = 0; i < 256; i++) src[i] = (uint8_t)(0x20 + (i & 0x3f));
        for (i = 0; i < 256; i++) { ssum += src[i]; dst[i] = 0; }
        r[5] = (int)ssum;
        if (!compute_resource_hmx_lock || compute_resource_hmx_lock(res.ctx) != 0) {
            r[0] = -20; res_put(&res); return 0;
        }
        hmx_clr();
        hmx_bias(src);
        hmx_bias_store(dst);
        hmx_clr();
        __asm__ __volatile__("barrier" ::: "memory");
        compute_resource_hmx_unlock(res.ctx);
        __asm__ __volatile__("barrier" ::: "memory");
        for (i = 0; i < 256; i++) dsum += dst[i];
        r[4] = (int)dsum;
        r[3] = (dst[0] << 8) | dst[255];
        res_put(&res);
        return 0;
    }

    if (mode & 32) {
        /* HMX write, then unlock HMX, then read VTCM: does the read still stall? */
        unsigned ssum = 0, dsum = 0;
        for (i = 0; i < 256; i++) src[i] = (uint8_t)(0x20 + (i & 0x3f));
        for (i = 0; i < 256; i++) { dsum += src[i]; dst[i] = 0; }
        r[6] = (int)dsum;               /* expected */
        ssum = 0;
        if (!compute_resource_hmx_lock || compute_resource_hmx_lock(res.ctx) != 0) {
            r[0] = -20; res_put(&res); return 0;
        }
        hmx_clr();
        hmx_bias(src);
        hmx_bias_store(dst);
        hmx_clr();
        __asm__ __volatile__("barrier" ::: "memory");
        compute_resource_hmx_unlock(res.ctx);      /* unlock BEFORE reading */
        __asm__ __volatile__("barrier" ::: "memory");
        for (i = 0; i < 256; i++) ssum += dst[i];
        r[4] = (int)ssum;
        r[3] = (dst[0] << 8) | dst[255];
        r[5] = 0x5731;
        res_put(&res);
        return 0;
    }

    if (!(mode & 1)) { res_put(&res); return 0; }

    memset(src, 0x11, 256);
    memset(dst, 0, 256);

    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(res.ctx) != 0) {
        r[0] = -20; res_put(&res); return 0;
    }
    hmx_clr();
    if (mode & 2) {
        uintptr_t b = (uintptr_t)res.base;
        hmx_bias((void *)((uintptr_t)src - b));
        hmx_bias_store((void *)((uintptr_t)dst - b));
    } else {
        hmx_bias(src);
        hmx_bias_store(dst);
    }
    hmx_clr();
    __asm__ __volatile__("barrier" ::: "memory");

    for (i = 0; i < 256; i++) sum += dst[i];
    r[4] = (int)sum;

    compute_resource_hmx_unlock(res.ctx);
    res_put(&res);
    return 0;
}

/* ------------------------------------------------------------------ */
/* fp16 reference tile (documented path)                                */
/* ------------------------------------------------------------------ */

static inline void hmx_clr_hf(void)
{
    __asm__ __volatile__("{\n\t mxclracc.hf\n\t}" ::: "memory");
}

static inline void hmx_mul_hf(const void *act, uint32_t ac,
                              const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.hf = mxmem(%0,%1):deep\n\t weight.hf = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(ac),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc)
        : "memory");
}

static inline void hmx_st_hf(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static inline void hmx_mul_hf_m(const void *act, uint32_t ac,
                                const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.hf = mxmem(%0,%1)\n\t weight.hf = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(ac),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc)
        : "memory");
}

/* multi-crouton multiply: activation deep (dC in Rt[31:11]), weights continuous */
static inline void hmx_mul_hf_deep(const void *act, uint32_t ac,
                                   const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.hf = mxmem(%0,%1):deep\n\t weight.hf = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(ac),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc)
        : "memory");
}

static inline void hmx_st_hf_m(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

/* PRM 3.2 example: A[s][c] = s + 0.01c, B[n][k] = n if k==n else 0, bias scale 1. */
static inline void hmx_st_hf_at(void *out, uint32_t rt)   /* :after */
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}
static inline void hmx_st_hf_before(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):before.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}
static inline void hmx_st_hf_retain(void *out, uint32_t rt)
{
    __asm__ __volatile__("{\n\t mxmem(%0,%1):after:retain.hf = acc\n\t}"
                         :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
}

static int g_pack, g_wrt = 2047, g_art = 0x7C;

int mx4_s1f16(const unsigned char *Adata, int ALen,
              const unsigned char *Bdata, int BLen,
              unsigned char *Cdata, int CLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    int rc, s, c, i;
    const __fp16 *A = (const __fp16 *)Adata, *B = (const __fp16 *)Bdata;
    __fp16 *C = (__fp16 *)Cdata;

    if (infoLen < 2) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (ALen < 2048 || BLen < 2048 || CLen < 2048) return -2;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (s = 0; s < 32; s++)
            for (c = 0; c < 32; c++) {
                int ae = (g_pack == 1 || g_pack == 2 || g_pack == 3)
                    ? ((s >> 1) * 64 + c * 2 + (s & 1))
                    : (512 * (c >> 4) + (s >> 1) * 32 + (c & 15) * 2 + (s & 1));
                a[ae] = A[s * 32 + c];
                if (g_pack == 3)
                    b[(s >> 1) * 64 + c * 2 + (s & 1)] = B[s * 32 + c];  /* n/k swapped */
                else
                    b[(c >> 1) * 64 + s * 2 + (c & 1)] = B[s * 32 + c];  /* B[n][k] */
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    /* PRM 4.2: one deep multiply over 2 continuous croutons (32 channels) */
    if (g_pack == 3)
        hmx_mul_hf_m(vact, 0x7Cu, vwgt, 2047u);
    else
        hmx_mul_hf_deep(vact, (1u << 11) | ((g_pack == 2) ? 0x700u : 0u) | 0x7Cu,
                        vwgt, (uint32_t)g_wrt);
    hmx_bias(vbias);
    hmx_st_hf_at(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    for (s = 0; s < 32; s++)
        for (c = 0; c < 32; c++)
            C[s * 32 + c] = *(__fp16 *)(vout + ((s >> 1) * 64 + c * 2 + (s & 1)) * 2);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

/* Exact QNN form (hmx_convhbh1x1deep_stride1 @0x371ec0): :deep on the WEIGHT,
 * activation pointer += 1024 per crouton, weight pointer += wstride. */
static const unsigned char *g_qa, *g_qb;

int mx4_qnn8(int ncr, int wstride, int wgt_rt, int astep,
             unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 2048u + 2048u + 2048u + 512u;
    int rc, i, s, c;

    if (infoLen < 3) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base;
    vwgt = vact + 2048u;
    vout = vwgt + 4096u;
    vbias = vout + 2048u;
    memset(r.base, 0, need);

    for (s = 0; s < 32; s++)
        for (c = 0; c < 32; c++) {
            /* 1 byte per cell, channel-major: 1 KB = 32 ch x 32 spatial */
            unsigned ao = (unsigned)c * 32u + (unsigned)s;
            unsigned wo = (unsigned)c * 32u + (unsigned)s;
            unsigned char va = g_qa ? g_qa[s * 32 + c] : 1u;
            unsigned char vb = g_qb ? g_qb[s * 32 + c] : 1u;
            if (ao < 2048u) vact[ao] = va;
            if (wo < 2048u) vwgt[wo] = vb;
        }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr();
    for (i = 0; i < ncr; i++) {
        /* pair 0: channel start 0, stop 15 ; pair 1: channel start 16 (Rs+64), stop 31 */
        unsigned aoff = (unsigned)i * 512u;
        unsigned woff = (unsigned)i * 512u;
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3):deep\n\t}"
            :: "r"((uint32_t)(uintptr_t)(vact + aoff)), "r"((uint32_t)g_art),
               "r"((uint32_t)(uintptr_t)(vwgt + woff)), "r"((uint32_t)g_wrt)
            : "memory");
    }
    hmx_bias(vbias);
    hmx_st_uh(vout, 0);
    hmx_sync();
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_s1q(const unsigned char *A, int ALen, const unsigned char *B, int BLen,
            int wstride, int wgt_rt, unsigned char *raw, int rawLen,
            int *info, int infoLen)
{
    int rc;
    (void)ALen; (void)BLen; (void)wstride;
    g_qa = A; g_qb = B;
    rc = mx4_qnn8(2, 512, 1023, 512, raw, rawLen, info, infoLen);
    g_qa = g_qb = 0;
    return rc;
}

int mx4_psum(int cnt, unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    int rc, s, c, i;

    if (infoLen < 2) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (i = 0; i < 1024; i++) a[i] = (__fp16)1.0f;
        for (s = 0; s < 32; s++)                    /* s = output channel n */
            for (c = 0; c < 32; c++) {              /* c = input channel k  */
                int welem = (c >> 1) * 64 + s * 2 + (c & 1);
                b[welem] = (c < cnt) ? (__fp16)1.0f : (__fp16)0.0f;
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf_m(vact, (uint32_t)g_art, vwgt, (uint32_t)g_wrt);
    hmx_bias(vbias);
    if (g_pack == 1) {
        __asm__ __volatile__("{\n\t mxswapacc\n\t}" ::: "memory");   /* read secondary */
        __asm__ __volatile__("barrier" ::: "memory");
    }
    hmx_st_hf_at(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_kscan(int k0, unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    int rc, s, c, i;

    if (infoLen < 2) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (i = 0; i < 1024; i++) b[i] = (__fp16)1.0f;
        for (s = 0; s < 32; s++)
            for (c = 0; c < 32; c++) {
                int ae = (s >> 1) * 64 + c * 2 + (s & 1);
                a[ae] = (c == k0) ? (__fp16)1.0f : (__fp16)0.0f;
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf_m(vact, 0x7Cu, vwgt, 2047u);
    hmx_bias(vbias);
    hmx_st_hf_at(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_dlprobe(int *r, int rLen)
{
    void *h;
    void *sym;

    if (rLen < 5) return -1;
    memset(r, 0, sizeof(int) * rLen);
    h = dlopen("/data/local/tmp/hmx3/libQnnHtpV68Skel.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("libQnnHtpV68Skel.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
        /* try the copy that fastrpc uses for the unsigned shell */
        h = dlopen("/vendor/dsp/cdsp/libQnnHtpV68Skel.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    r[0] = h ? 1 : 0;
    if (!h) return 0;
    sym = dlsym(h, "hmx_convhbh1x1deep_stride1");
    r[1] = (int)(uintptr_t)h;
    r[2] = (int)(uintptr_t)sym;
    r[3] = sym ? 1 : 0;
    {
        void *s2 = dlsym(h, "hmx_convbbb1x1_stride1");
        r[4] = (int)(uintptr_t)s2;
    }
    return 0;
}

int mx4_cfg(int pack, int wrt) { g_pack = pack; g_wrt = wrt; return 0; }
int mx4_cfg2(int art, int wrt) { g_art = art; g_wrt = wrt; return 0; }

int mx4_amap2(int mode, unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    int rc, s, c, i;

    if (infoLen < 2) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (s = 0; s < 32; s++)
            for (c = 0; c < 32; c++) {
                /* doc layout: 2 spatials per 32-bit vector, channel-major */
                int elem = (s >> 1) * 64 + c * 2 + (s & 1);
                a[elem] = (mode == 0) ? (__fp16)2.0f
                        : (mode == 2) ? (__fp16)1.0f
                                      : (__fp16)((float)s + (float)c / 32.0f);
                {
                    int welem = (c >> 1) * 64 + s * 2 + (c & 1);
                    b[welem] = (mode == 2) ? (__fp16)1.0f
                              : ((c == s) ? (__fp16)1.0f : (__fp16)0.0f);
                }
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf_m(vact, 0x7Cu, vwgt, 2047u);
    hmx_bias(vbias);
    hmx_st_hf_at(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_idmap(int mode, unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    int rc, s, c, i;

    if (infoLen < 4) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (s = 0; s < 32; s++)
            for (c = 0; c < 32; c++) {
                int elem = (mode >= 2) ? (s * 32 + c)
                                       : ((s >> 1) * 64 + c * 2 + (s & 1));
                a[elem] = (__fp16)(float)(mode ? s : c);
                /* weights, doc layout */
                {
                    int welem = (c >> 1) * 64 + s * 2 + (c & 1);
                    int keep = mode ? (c == 0) : (c == s);
                    b[welem] = keep ? (__fp16)1.0f : (__fp16)0.0f;
                }
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf_m(vact, 0x7Cu, vwgt, 2047u);
    hmx_bias(vbias);
    hmx_st_hf_at(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_cov(int off0, int step, int nst, int rt, int form,
            unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    uint32_t act_rt = 0x7Cu, wgt_rt = 2047u;
    int rc, i, j, filled = 0;

    if (infoLen < 6) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;
    if (nst < 1) nst = 1;

    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;
    vact = r.base; vwgt = vact + 2048u; vout = vwgt + 2048u; vbias = vout + 2048u;
    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (i = 0; i < 1024; i++) a[i] = (__fp16)1.0f;
        for (i = 0; i < 1024; i++) b[i] = (__fp16)0.0f;
        for (i = 0; i < 32; i++)
            b[(i >> 1) * 64 + i * 2 + (i & 1)] = (__fp16)1.0f;   /* identity */
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;
    }
    info[2] = (int)act_rt; info[3] = (int)wgt_rt;

    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    if (form == 3) {
        /* int8: act/wgt all 1 -> out = 16 per cell; store form :sat.uh = acc:2x1 */
        memset(vact, 1, 2048);
        memset(vwgt, 1, 2048);
        hmx_clr();
        hmx_mul(vact, 0x7Cu, vwgt, 2047u);
        hmx_bias(vbias);
        for (i = 0; i < nst; i++) {
            unsigned off = (unsigned)off0 + (unsigned)i * (unsigned)step;
            hmx_st_uh(vout + off, (uint32_t)rt);
        }
        hmx_sync();
        memcpy(raw, vout, 2048);
        for (j = 0; j < 1024; j++) {
            unsigned short h = (unsigned short)(raw[j * 2] | (raw[j * 2 + 1] << 8));
            if (h == 16) filled++;
        }
        info[1] = filled;
        info[0] = 0;
        compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        res_put(&r);
        return 0;
    }

    hmx_clr_hf();
    hmx_mul_hf_m(vact, act_rt, vwgt, wgt_rt);
    hmx_bias(vbias);
    for (i = 0; i < nst; i++) {
        unsigned off = (unsigned)off0 + (unsigned)i * (unsigned)step;
        void *dst = vout + off;
        if (form == 1)      hmx_st_hf_before(dst, (uint32_t)rt);
        else if (form == 2) hmx_st_hf_retain(dst, (uint32_t)rt);
        else                hmx_st_hf_at(dst, (uint32_t)rt);
    }
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    for (j = 0; j < 1024; j++) {
        unsigned short h = (unsigned short)(raw[j * 2] | (raw[j * 2 + 1] << 8));
        if (h == 0x3C00) filled++;
    }
    info[1] = filled;
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_fp16ex(int mask, unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 3u * 2048u + 2048u;
    uint32_t act_rt, wgt_rt, wr_rt;
    int rc, vec, c, n, i;

    if (infoLen < 6) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (rawLen < 2048) return -3;

    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;

    vact = r.base;
    vwgt = vact + 2048u;
    vout = vwgt + 2048u;
    vbias = vout + 2048u;

    memset(r.base, 0, need);
    {
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        __fp16 *pa = a, *pb = b;
        /* write_act_vtcm(): 2 spatials per vector, ch-major */
        for (vec = 0; vec < 16; vec++)
            for (c = 0; c < 32; c++)
                for (i = 0; i < 2; i++) {
                    int sp = vec * 2 + i;
                    *pa++ = (__fp16)((float)sp + (float)c * 0.01f);
                }
        /* write_wgt_vtcm(): 2 input channels per vector, out-major */
        for (vec = 0; vec < 16; vec++)
            for (n = 0; n < 32; n++)
                for (i = 0; i < 2; i++) {
                    int in_ch = vec * 2 + i;
                    *pb++ = (in_ch == n) ? (__fp16)(float)n : (__fp16)0.0f;
                }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (i = 0; i < 32; i++) w[i] = 0x00003C00u;   /* scale 1.0, rest 0 */
    }

    act_rt = (((uint32_t)mask >> 1) << 7) | (31u << 2) | (((uint32_t)mask & 1u) << 1);
    wgt_rt = 2047u;
    wr_rt  = (((uint32_t)mask >> 1) << 7) | (((uint32_t)mask & 1u) << 1);

    info[1] = (int)act_rt;
    info[2] = (int)wgt_rt;
    info[3] = (int)wr_rt;
    info[4] = mask;

    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf_m(vact, act_rt, vwgt, wgt_rt);
    hmx_bias(vbias);
    hmx_st_hf_m(vout, wr_rt);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");
    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

int mx4_fp16(const unsigned char *unused, int ulen,
             unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned need = 4u * 2048u;
    int rc, s, n;

    (void)unused; (void)ulen;
    if (infoLen < 6) return -1;
    memset(info, 0, sizeof(int) * 6);
    if (rawLen < 2048) return -3;

    rc = res_get(need, &r);
    info[0] = rc;
    if (rc != 0) return 0;

    vact  = r.base;
    vwgt  = vact + 2048u;
    vout  = vwgt + 2048u;
    vbias = vout + 2048u;

    memset(r.base, 0, need);
    {
        /* PRM layout: cell (spatial s, channel c) at (s>>1)*128 + c*4 + (s&1)*2 */
        __fp16 *a = (__fp16 *)vact, *b = (__fp16 *)vwgt;
        for (s = 0; s < TILE; s++)
            for (n = 0; n < TILE; n++) {
                /* ramp mode: A[s][k] = s+1 at k=0 ; B = identity  => out[s][0] = s+1 */
                a[((size_t)(s >> 1) * 128u + (size_t)n * 4u + (size_t)(s & 1) * 2u) >> 1] =
                    (__fp16)((n == 0) ? (float)(s + 1) : 0.0f);
                b[((size_t)(n >> 1) * 128u + (size_t)s * 4u + (size_t)(n & 1) * 2u) >> 1] =
                    (s == n) ? (__fp16)1.0f : (__fp16)0.0f;
            }
    }
    {
        uint32_t *w = (uint32_t *)vbias;
        for (s = 0; s < 32; s++) w[s] = 0x00003C00u;
        for (s = 0; s < 32; s++) w[32 + s] = 0u;
    }

    info[3] = (int)(uintptr_t)r.base;
    info[5] = (int)need;

    if ((uintptr_t)vact & 2047u) { info[0] = -30; res_put(&r); return 0; }

    if (!compute_resource_hmx_lock || compute_resource_hmx_lock(r.ctx) != 0) {
        info[0] = -20; res_put(&r); return 0;
    }
    if (hmx_power(1) != 0) { info[0] = -50; res_put(&r); return 0; }
    hmx_clr_hf();
    hmx_mul_hf(vact, (1u << 11) | ACT_CTRL, vwgt, 2047u);
    hmx_bias(vbias);
    hmx_st_hf(vout, 0);
    hmx_clr_hf();
    __asm__ __volatile__("barrier" ::: "memory");

    memcpy(raw, vout, 2048);
    info[0] = 0;
    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);
    return 0;
}

/* ------------------------------------------------------------------ */
/* tiled GEMM                                                          */
/* ------------------------------------------------------------------ */

int mx4_gemm(const unsigned char *A, int ALen,
             const unsigned char *B, int BLen,
             int M, int N, int K, int reps, int mode,
             unsigned char *C, int CLen, int *res, int resLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout;
    unsigned m32, n32, ncr, need, act_bytes, wgt_bytes;
    unsigned per_mt;
    int rc, mt, nt, rr, i;
    int stage_mode, cm;
    uint64_t t0, t1, t2, t3, t4, t5;
    int lock_rc = 0;

    if (resLen < 12) return -1;
    memset(res, 0, sizeof(int) * resLen);
    if (M <= 0 || N <= 0 || K <= 0 || (K % TILE) || reps <= 0) return -2;

    m32 = (unsigned)((M + TILE - 1) / TILE);
    n32 = (unsigned)((N + TILE - 1) / TILE);
    ncr = (unsigned)K / TILE;

    act_bytes = m32 * ncr * SLOT;
    wgt_bytes = ncr * SLOT;
    per_mt    = ncr * SLOT;

    if (ALen < (int)(m32 * ncr * SLOT) || BLen < (int)(n32 * ncr * SLOT)) return -3;
    if (CLen < (int)((size_t)M * N)) return -4;

    stage_mode = mode & 1;
    cm = (mode & 2) != 0;
    need = (stage_mode ? per_mt : act_bytes) + wgt_bytes + SLOT + 2048u;

    rc = res_get(need, &r);
    if (rc != 0) {
        /* fall back to single M tile staging */
        stage_mode = 1;
        need = per_mt + wgt_bytes + SLOT + 2048u;
        rc = res_get(need, &r);
    }
    res[0] = rc;
    if (rc != 0) return 0;

    res[8] = (int)need;
    res[9] = (int)act_bytes;

    vact = r.base;
    vwgt = vact + (stage_mode ? per_mt : act_bytes);
    vout = vwgt + wgt_bytes;

    if ((uintptr_t)vact & (SLOT - 1u)) { res[0] = -30; res_put(&r); return 0; }
    if ((uintptr_t)vwgt & (SLOT - 1u)) { res[0] = -31; res_put(&r); return 0; }
    if ((uintptr_t)vout & (SLOT - 1u)) { res[0] = -32; res_put(&r); return 0; }

    memset(r.base, 0, need);

    if (!compute_resource_hmx_lock) { res[0] = -40; res_put(&r); return 0; }
    if (hmx_power(1) != 0) { res[0] = -50; res_put(&r); return 0; }
    lock_rc = compute_resource_hmx_lock(r.ctx);
    if (lock_rc != 0) { res[0] = -20; res_put(&r); return 0; }

    /* ---- stage activations ---- */
    t0 = HAP_perf_get_time_us();
    if (!stage_mode) {
        memcpy(vact, A, (size_t)act_bytes);
    }
    t1 = HAP_perf_get_time_us();

    /* ---- pure HMX throughput: resident weight tile, compute + store ---- */
    if (stage_mode)
        memcpy(vact, A, (size_t)per_mt);
    memcpy(vwgt, B, (size_t)wgt_bytes);
    __asm__ __volatile__("barrier" ::: "memory");

    t2 = HAP_perf_get_time_us();
    for (rr = 0; rr < reps; rr++) {
        if (stage_mode) {
            for (mt = 0; mt < (int)m32; mt++) {
                if (mt) memcpy(vact, A + (size_t)mt * per_mt, per_mt);
                for (nt = 0; nt < (int)n32; nt++) {
                    hmx_clr();
                    for (i = 0; i < (int)ncr; i++)
                        hmx_mul(vact + (unsigned)i * SLOT, ACT_CTRL,
                                vwgt + (unsigned)i * SLOT, WGT_CTRL);
                    hmx_st_ub(vout, 0);
                }
            }
        } else {
            for (nt = 0; nt < (int)n32; nt++)
                for (mt = 0; mt < (int)m32; mt++) {
                    hmx_clr();
                    for (i = 0; i < (int)ncr; i++)
                        hmx_mul(vact + (size_t)(mt * ncr + i) * SLOT, ACT_CTRL,
                                vwgt + (unsigned)i * SLOT, WGT_CTRL);
                    hmx_st_ub(vout, 0);
                }
        }
    }
    hmx_sync();
    t3 = HAP_perf_get_time_us();

    /* ---- full pass: stream weights from DDR, compute, unpack to DDR ---- */
    t4 = HAP_perf_get_time_us();
    for (rr = 0; rr < reps; rr++) {
        if (stage_mode) {
            for (mt = 0; mt < (int)m32; mt++) {
                memcpy(vact, A + (size_t)mt * per_mt, per_mt);
                for (nt = 0; nt < (int)n32; nt++) {
                    memcpy(vwgt, B + (size_t)nt * wgt_bytes, wgt_bytes);
                    hmx_clr();
                    for (i = 0; i < (int)ncr; i++) {
                        if (cm) hmx_mul_cm(vact + (unsigned)i * SLOT, ACT_CTRL,
                                           vwgt + (unsigned)i * SLOT, WGT_CTRL);
                        else    hmx_mul(vact + (unsigned)i * SLOT, ACT_CTRL,
                                        vwgt + (unsigned)i * SLOT, WGT_CTRL);
                    }
                    if (cm) hmx_st_ub_cm(vout, 0); else hmx_st_ub(vout, 0);
                    hmx_sync();
                    {
                        int rows = M - mt * TILE;
                        int cols = N - nt * TILE;
                        if (rows > TILE) rows = TILE;
                        if (cols > TILE) cols = TILE;
                        unpack_tile(vout, rows, cols,
                                    C + (size_t)mt * TILE * N + (size_t)nt * TILE, N);
                    }
                }
            }
        } else {
            for (nt = 0; nt < (int)n32; nt++) {
                memcpy(vwgt, B + (size_t)nt * wgt_bytes, wgt_bytes);
                for (mt = 0; mt < (int)m32; mt++) {
                    hmx_clr();
                    for (i = 0; i < (int)ncr; i++) {
                        if (cm) hmx_mul_cm(vact + (size_t)(mt * ncr + i) * SLOT, ACT_CTRL,
                                           vwgt + (unsigned)i * SLOT, WGT_CTRL);
                        else    hmx_mul(vact + (size_t)(mt * ncr + i) * SLOT, ACT_CTRL,
                                        vwgt + (unsigned)i * SLOT, WGT_CTRL);
                    }
                    if (cm) hmx_st_ub_cm(vout, 0); else hmx_st_ub(vout, 0);
                    hmx_sync();
                    {
                        int rows = M - mt * TILE;
                        int cols = N - nt * TILE;
                        if (rows > TILE) rows = TILE;
                        if (cols > TILE) cols = TILE;
                        unpack_tile(vout, rows, cols,
                                    C + (size_t)mt * TILE * N + (size_t)nt * TILE, N);
                    }
                }
            }
        }
    }
    __asm__ __volatile__("barrier" ::: "memory");
    t5 = HAP_perf_get_time_us();

    compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    res_put(&r);

    res[0] = 0;
    res[1] = (int)(t3 - t2);
    res[2] = (int)(t5 - t4);
    res[3] = (int)(t1 - t0);
    res[4] = (int)(m32 * n32 * (unsigned)reps);
    res[5] = (int)m32;
    res[6] = (int)n32;
    res[7] = (int)ncr;
    res[10] = (int)wgt_bytes;
    res[11] = stage_mode;

    res[9] = (int)(wgt_bytes ? 1 : 0); /* keep act_bytes in slot 9 */
    res[9] = (int)act_bytes;
    return 0;
}

/* ================================================================== */
/* qhl_hmx SDK reference path (v68)                                    */
/*                                                                     */
/* Straight port of $SDK/libs/qhl_hmx:                                 */
/*   qhblas_hmx_transfer_row_ahf     (act 32xK -> VRAM croutons)        */
/*   qhblas_hmx_transfer_column_ahf  (wgt Kx64 -> VRAM croutons)        */
/*   qhblas_hmx_matrix_matrix_mpy_ahf(32xK x KxN -> 32xN)              */
/* The .S kernel takes the __HEXAGON_ARCH__ < 73 branch on this device  */
/* (no cvt:  mxmem():after.hf = acc).                                  */
/* ================================================================== */

#include "qurt_hmx.h"
#include "qurt_hvx.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

int32_t qhblas_hmx_transfer_row_ahf(__fp16 *input_mtx, uint8_t *vtcm_addr, uint32_t K);
int32_t qhblas_hmx_transfer_column_ahf(__fp16 *input_mtx, uint8_t *vtcm_addr,
                                       uint32_t K, uint32_t N);
int32_t qhblas_hmx_matrix_matrix_mpy_ahf(uint8_t *vtcm_mtx1, uint8_t *vtcm_mtx2,
                                         uint8_t *vtcm_out, __fp16 *output_mtx,
                                         uint32_t M, uint32_t K, uint32_t N);

#define REF_MAXK   256
#define REF_NPAD   64

/* sources and the row-major result of the HVX permute must be 128B aligned.
 * g_refguard keeps qhblas_hmx_transfer_column_ahf's l2fetch alive: it
 * prefetches `input + (1+i)*4096` bytes for every k-group, i.e. up to
 * ~7 KB past a [Kx64] tile for K=256 (the SDK block size ignores the real
 * row stride), and a TLBMISS there kills the PD. */
static __fp16 g_refA[32 * REF_MAXK] __attribute__((aligned(128)));
static __fp16 g_refB[REF_MAXK * REF_NPAD] __attribute__((aligned(128)));
static __fp16 g_refC[32 * REF_NPAD] __attribute__((aligned(128)));
static uint8_t g_refguard[65536] __attribute__((used, aligned(128)));

/* VTCM block: [act][wgt][out 2 tiles][2 KB spare], every part 2 KB aligned */
static int ref_alloc(int K, int npad, res_t *r, uint8_t **pa, uint8_t **pw, uint8_t **po)
{
    unsigned abytes = (unsigned)(32 * K * 2);
    unsigned wbytes = (unsigned)(K * npad * 2);
    unsigned aoff = align_up(abytes, 2048u);
    unsigned woff = align_up(wbytes, 2048u);
    unsigned need = aoff + woff + 2u * 2048u + 2048u;
    int rc = res_get(need, r);

    if (rc) return rc;
    memset(r->base, 0, need);
    *pa = r->base;
    *pw = r->base + aoff;
    *po = r->base + aoff + woff;
    return 0;
}

/* mode bit0: skip HAP_power_set(HMX)      bit1: skip mxclracc.hf
 *      bit2: skip HMX lock                bit3: skip HVX lock
 *      bit4: native N (no 64-wide pad)    bit5: qurt_hmx_lock2 instead of
 *                                               compute_resource_hmx_lock
 *      bit6: hold the HVX lock across the HMX call
 *      bit7: activation Rt mask 0x3C instead of 0x7C
 * info = {status, us/rep, lock_rc, act_ctrl, wgt_ctrl, ncr, power_rc, mpy_rc,
 *         hvx_rc, pack_row_rc, vtcm_bytes, npad, pack_col_rc, unlock_rc}     */
int mx4_hmxref(const unsigned char *Adata, int ALen,
               const unsigned char *Bdata, int BLen,
               int N, int K, int mode, int reps,
               unsigned char *Cdata, int CLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact = 0, *vwgt = 0, *vout = 0;
    uint32_t ncr, act_ctrl, wgt_ctrl;
    int npad, hold_hvx, hvx_ok = 0, s, k, n, i, rc;
    uint64_t t0, t1;
    const __fp16 *A = (const __fp16 *)Adata;
    const __fp16 *B = (const __fp16 *)Bdata;
    __fp16 *C = (__fp16 *)Cdata;

    if (infoLen < 14) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > REF_MAXK || (K & 31)) return -2;
    if (N < 1 || N > 64) return -3;
    if (ALen < 32 * K * 2) return -4;
    if (BLen < K * N * 2) return -5;
    if (CLen < 32 * N * 2) return -6;
    if (reps < 1 || reps > 200) reps = 1;

    /* qhblas_hmx_transfer_column_ahf only fills K*2 elements for N<=32
     * (row_iters = K>>5 vectors), so the 32-wide weight tile it produces is
     * incomplete; pad N up to 64 unless the caller asks for the native path. */
    npad = (mode & 16) ? (N <= 32 ? 32 : 64) : 64;

    for (s = 0; s < 32; s++)
        for (k = 0; k < K; k++)
            g_refA[s * K + k] = A[s * K + k];
    memset(g_refB, 0, sizeof(__fp16) * (size_t)K * REF_NPAD);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++)
            g_refB[k * REF_NPAD + n] = B[k * N + n];

    rc = ref_alloc(K, npad, &r, &vact, &vwgt, &vout);
    info[0] = rc;
    if (rc) return 0;
    info[10] = (int)r.size;
    info[11] = npad;

    hold_hvx = (mode & 64) != 0;
    if (!(mode & 8)) {
        info[8] = qurt_hvx_lock(QURT_HVX_MODE_128B);
        hvx_ok = (info[8] == 0);
    }

    info[9]  = qhblas_hmx_transfer_row_ahf(g_refA, vact, (uint32_t)K);
    info[12] = qhblas_hmx_transfer_column_ahf(g_refB, vwgt, (uint32_t)K, (uint32_t)npad);

    if (hvx_ok && !hold_hvx) {
        qurt_hvx_unlock();
        hvx_ok = 0;
    }

    if (!(mode & 1)) { info[6] = hmx_power(1); info[15] = hmx_clock(); }
    if (!(mode & 4)) {
        if (mode & 32)
            info[2] = qurt_hmx_lock2(QURT_HMX_SHARED_LOCK);
        else
            info[2] = compute_resource_hmx_lock ? compute_resource_hmx_lock(r.ctx) : -100;
    }

    ncr      = (uint32_t)K >> 5;
    act_ctrl = (ncr << 11) | ((mode & 128) ? 0x3Cu : 0x7Cu);
    wgt_ctrl = (ncr << 11) - 1u;
    info[3] = (int)act_ctrl;
    info[4] = (int)wgt_ctrl;
    info[5] = (int)ncr;

    t0 = HAP_perf_get_time_us();
    {
        uint64_t c0 = HAP_perf_get_pcycles(), c1;
        for (i = 0; i < reps; i++) {
            if (!(mode & 2)) hmx_clr_hf();
            info[7] = qhblas_hmx_matrix_matrix_mpy_ahf(vact, vwgt, vout, g_refC,
                                                       32u, (uint32_t)K, (uint32_t)npad);
        }
        c1 = HAP_perf_get_pcycles();
        info[14] = (int)((c1 - c0) / (uint64_t)reps);
    }
    t1 = HAP_perf_get_time_us();
    info[1] = (int)((t1 - t0) / (uint64_t)reps);


    if (!(mode & 4)) {
        if (mode & 32)
            info[13] = qurt_hmx_unlock2(QURT_HMX_SHARED_UNLOCK);
        else if (compute_resource_hmx_unlock)
            info[13] = compute_resource_hmx_unlock(r.ctx);
    }
    if (!(mode & 1)) hmx_power(0);
    if (hvx_ok) qurt_hvx_unlock();
    res_put(&r);

    for (s = 0; s < 32; s++)
        for (n = 0; n < N; n++)
            C[s * N + n] = g_refC[s * npad + n];
    return 0;
}

/* Layout dump: store element *indices* as raw fp16 bit patterns, run the SDK
 * packers, hand the packed bytes back.  side 0 = activation (s*K+k),
 * side 1 = weights (k*64+n, first 32 columns of the padded 64-wide tile).  */
int mx4_refdump(int K, int side, unsigned char *out, int outLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact = 0, *vwgt = 0, *vout = 0;
    uint16_t *pa = (uint16_t *)g_refA, *pb = (uint16_t *)g_refB;
    int rc, s, k, n, hvx;
    unsigned bytes;

    if (infoLen < 4) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > REF_MAXK || (K & 31)) return -2;

    rc = ref_alloc(K, 64, &r, &vact, &vwgt, &vout);
    info[0] = rc;
    if (rc) return 0;

    for (k = 0; k < K; k++)
        for (n = 0; n < 64; n++)
            pb[k * 64 + n] = (uint16_t)(k * 64 + n);
    for (k = 0; k < 32 * K; k++)
        pa[k] = (uint16_t)k;

    hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    info[1] = hvx;
    if (side == 0) {
        info[2] = qhblas_hmx_transfer_row_ahf(g_refA, vact, (uint32_t)K);
        bytes = (unsigned)(32 * K * 2);
        if (outLen < (int)bytes) { if (hvx == 0) qurt_hvx_unlock(); res_put(&r); return -3; }
        memcpy(out, vact, bytes);
    } else {
        info[2] = qhblas_hmx_transfer_column_ahf(g_refB, vwgt, (uint32_t)K, 64u);
        bytes = (unsigned)(K * 32 * 2);
        if (outLen < (int)bytes) { if (hvx == 0) qurt_hvx_unlock(); res_put(&r); return -3; }
        memcpy(out, vwgt, bytes);
    }
    if (hvx == 0) qurt_hvx_unlock();
    res_put(&r);
    info[3] = (int)bytes;
    (void)s;
    return 0;
}

/* ================================================================== */
/* int8 probe (v68): one act/wgt pair, one 2 KB store                   */
/* ================================================================== */

static inline void i8_mul(const void *act, uint32_t art,
                          const void *wgt, uint32_t wc, int mul)
{
    switch (mul & 3) {
    case 1:  /* :deep on the activation (multi-crouton K accumulation) */
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.b = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
        break;
    case 2:  /* :deep on the weight */
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3):deep\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
        break;
    case 3:
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.b = mxmem(%2,%3):deep\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
        break;
    default:
        hmx_mul(act, art, wgt, wc);
        break;
    }
}

static inline void i8_store(void *out, uint32_t rt, int srt)
{
    switch (srt & 3) {
    case 1:  hmx_st_uh(out, rt);          break;
    case 2:  hmx_st_ub_nosat(out, rt);    break;
    case 3:
        __asm__ __volatile__("{\n\t mxmem(%0,%1):after.uh = acc:2x1\n\t}"
                             :: "r"((uint32_t)(uintptr_t)out), "r"(rt) : "memory");
        break;
    default: hmx_st_ub(out, rt);          break;
    }
}

#define I8_MAXK 2048

static void i8_pack_act(uint8_t *dst, const int8_t *A, int K, int lay, int M)
{
    int s, c;
    if (lay != 3 && M > 32) M = 32;   /* only the hw int8 layout is 64-wide */
    for (s = 0; s < M; s++)
        for (c = 0; c < K; c++) {
            unsigned off;
            if (lay == 3)       /* hardware int8: 4 spatial per 128 B vector */
                off = 2048u * (unsigned)(c >> 5) + 128u * (unsigned)(s >> 2)
                    + 4u * (unsigned)(c & 31) + (unsigned)(s & 3);
            else if (lay == 4)  /* int4 probe: 1 KB activation croutons */
                off = 1024u * (unsigned)(c >> 5) + 128u * (unsigned)(s >> 2)
                    + 4u * (unsigned)(c & 31) + (unsigned)(s & 3);
            else if (lay == 0)  /* 64 ch/crouton, rows s,s+1 interleaved 1:1 */
                off = 2048u * (unsigned)(c >> 6) + 128u * (unsigned)(s >> 1)
                    + 2u * (unsigned)(c & 63) + (unsigned)(s & 1);
            else if (lay == 1)  /* 64 ch/crouton, 64-byte row blocks */
                off = 2048u * (unsigned)(c >> 6) + 128u * (unsigned)(s >> 1)
                    + 64u * (unsigned)(s & 1) + (unsigned)(c & 63);
            else                /* 32 ch/crouton, 1 KB payload + 1 KB zero pad */
                off = 2048u * (unsigned)(c >> 5) + 64u * (unsigned)(s >> 1)
                    + 2u * (unsigned)(c & 31) + (unsigned)(s & 1);
            dst[off] = (uint8_t)A[s * K + c];
        }
}

/* weight, PRM Figure 6/7: 32 K x 32 N = 1 KB, (k>>2)*128 + 4*n + (k&3).
 * wl 4 / N>32: the tile is 2 KB, one 1 KB sub-block per output-channel group. */
static void i8_pack_wgt(uint8_t *dst, const int8_t *B, int N, int K, int wl)
{
    int k, n;
    int wide = (wl == 4);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++) {
            unsigned off;
            if (wl == 3 || wl == 4)
                off = (wide ? 2048u : 1024u) * (unsigned)(k >> 5)
                    + (wide ? 1024u * (unsigned)(n >> 5) : 0u)
                    + 128u * (unsigned)((k & 31) >> 2) + 4u * (unsigned)(n & 31)
                    + (unsigned)(k & 3);
            else if (wl == 0)
                off = 2048u * (unsigned)(k >> 5)
                    + 128u * (unsigned)((k & 31) >> 1) + 2u * (unsigned)n
                    + (unsigned)(k & 1);
            else if (wl == 1)
                off = 2048u * (unsigned)(k >> 5)
                    + 128u * (unsigned)((k & 31) >> 1)
                    + 64u * (unsigned)(k & 1) + (unsigned)n;
            else
                off = 2048u * (unsigned)(k >> 5)
                    + 64u * (unsigned)((k & 31) >> 1) + 2u * (unsigned)n
                    + (unsigned)(k & 1);
            dst[off] = (uint8_t)B[n * K + k];
        }
}

/* deferrable-store: one act/wgt pair, int32 accumulator, sat.ub store */
static inline void i8_mul_deepact(const void *act, uint32_t art,
                                  const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.b = mxmem(%2,%3)\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(art),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
}

static int i8_core(const int8_t *A, int ALen, const int8_t *B, int BLen,
                   int M, int N, int K, int lay, int wl, int art, int wc,
                   int srt, int mul, int npair, int bscale, int mode, int reps,
                   unsigned char *raw, int rawLen, int *info, int infoLen);

int mx4_i8(const unsigned char *Adata, int ALen,
           const unsigned char *Bdata, int BLen,
           int N, int K, int lay, int wl, int art, int wc, int srt,
           int mul, int npair, int bscale, int mode, int reps,
           unsigned char *raw, int rawLen, int *info, int infoLen)
{
    return i8_core((const int8_t *)Adata, ALen, (const int8_t *)Bdata, BLen,
                   32, N, K, lay, wl, art, wc, srt, mul, npair, bscale, mode,
                   reps, raw, rawLen, info, infoLen);
}

/* int8 GEMM with configurable spatial rows (M = 32 or 64).  One HMX int8
 * crouton is 2048 B = 64 spatial x 32 K, so M=64 fills the whole array and
 * costs exactly the same instruction count as M=32. */
int mx4_i8m(const unsigned char *Adata, int ALen,
            const unsigned char *Bdata, int BLen,
            int M, int N, int K, int lay, int wl, int art, int wc, int srt,
            int mul, int npair, int bscale, int mode, int reps,
            unsigned char *raw, int rawLen, int *info, int infoLen)
{
    return i8_core((const int8_t *)Adata, ALen, (const int8_t *)Bdata, BLen,
                   M, N, K, lay, wl, art, wc, srt, mul, npair, bscale, mode,
                   reps, raw, rawLen, info, infoLen);
}

static int i8_core(const int8_t *A, int ALen, const int8_t *B, int BLen,
                   int M, int N, int K, int lay, int wl, int art, int wc,
                   int srt, int mul, int npair, int bscale, int mode, int reps,
                   unsigned char *raw, int rawLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    unsigned aregion, wregion, wstep, need;
    int rc, hvx_ok = 0, i, p, rep;
    uint64_t t0, t1, c0, c1;
    int ncr = (K + 31) / 32;

    if (infoLen < 14) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > I8_MAXK || (K & 31)) return -2;
    if (N < 1 || N > 64) return -3;
    if (M < 1 || M > 64) return -3;
    if (ALen < M * K) return -4;
    if (BLen < K * N) return -5;
    if (rawLen < 2048) return -6;
    if (npair <= 0 || npair > ncr) npair = ncr;
    if (reps < 1 || reps > 10000) reps = 1;
    if (lay != 3) M = 32;

    aregion = 2048u * (unsigned)ncr;
    wstep   = (N > 32 || wl == 4) ? 2048u : 1024u;
    wregion = wstep * (unsigned)ncr;
    need = align_up(aregion + wregion, 2048u) + 4096u + 512u + 2048u;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    /* HMX store/load addresses are 2 KB granular (Rs[31:11]); a 1 KB-offset
     * output pointer silently writes only the first KB (= 32 spatial rows). */
    vact = r.base;
    vwgt = vact + aregion;
    vout = vact + align_up(aregion + wregion, 2048u);
    vbias = vout + 4096u;
    memset(r.base, 0, need);
    info[10] = (int)r.size;
    info[12] = (int)aregion;
    info[13] = (int)wregion;

    i8_pack_act(vact, A, K, lay, M);
    i8_pack_wgt(vwgt, B, N, K, wl);

    /* bias/scale: 256 B per 32 output channels, low words first, then high.
     * scale is an fp16 field ([15:0] + [35:32]); 0x6000 (=512.0) is the int8
     * identity because the convert carries an implicit 1/512. */
    if (bscale) {
        uint32_t *bl = (uint32_t *)vbias;
        int g;
        for (i = 0; i < 128; i++) bl[i] = 0;
        for (g = 0; g < 2; g++)
            for (i = 0; i < 32; i++)
                bl[g * 64 + i] = (uint32_t)bscale;  /* low words, then zeros */
    }

    if (!(mode & 4)) {
        info[8] = qurt_hvx_lock(QURT_HVX_MODE_128B);
        hvx_ok = (info[8] == 0);
    }
    if (!(mode & 1)) info[6] = hmx_power(1);
    if (!(mode & 4)) {
        if (mode & 8) info[7] = qurt_hmx_lock2(QURT_HMX_SHARED_LOCK);
        else          info[7] = compute_resource_hmx_lock ? compute_resource_hmx_lock(r.ctx) : -100;
    }

    info[3] = art;
    info[4] = wc;
    info[5] = ncr;
    info[9] = srt;
    info[11] = mul;

    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
    if (!(mode & 2)) hmx_clr();
    if (mul & 4) {
        /* chunked :deep over the activation: dC <= 31 per instruction */
        int rem = ncr, off = 0;
        while (rem > 0) {
            int ch = rem > 32 ? 32 : rem;
            uint32_t a2 = (uint32_t)((ch - 1) << 11) | ((uint32_t)art & 0x7FFu);
            uint32_t w2 = (uint32_t)(ch * 1024 - 1);
            i8_mul_deepact(vact + (unsigned)off * 2048u, a2,
                           vwgt + (unsigned)off * 1024u, w2);
            rem -= ch;
            off += ch;
        }
    } else if (mul & 1)
        i8_mul_deepact(vact, (uint32_t)art, vwgt, (uint32_t)wc);
    else if (N > 32)
        for (p = 0; p < npair; p++) {
            __asm__ __volatile__(
                "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3):deep\n\t}"
                :: "r"((uint32_t)(uintptr_t)(vact + (unsigned)p * 2048u)),
                   "r"((uint32_t)art),
                   "r"((uint32_t)(uintptr_t)(vwgt + (unsigned)p * 2048u)),
                   "r"((uint32_t)wc) : "memory");
        }
    else
        for (p = 0; p < npair; p++)
            hmx_mul(vact + (unsigned)p * 2048u, (uint32_t)art,
                    vwgt + (unsigned)p * wstep, (uint32_t)wc);

    if (bscale) hmx_bias(vbias);
    i8_store(vout, 0, srt);
    if (N > 32) {
        if (bscale) hmx_bias(vbias + 256);
        i8_store(vout + 2048, 0, srt);
    }
    }
    hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)((t1 - t0) / (uint64_t)reps);
    info[2] = (int)((c1 - c0) / (uint64_t)reps);

    if (!(mode & 4)) {
        if (mode & 8) qurt_hmx_unlock2(QURT_HMX_SHARED_UNLOCK);
        else if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
    }
    if (!(mode & 1)) hmx_power(0);
    if (hvx_ok) qurt_hvx_unlock();

    info[1] = (int)(t1 - t0);   /* total over all reps, us */
    info[2] = (int)((c1 - c0) / (uint64_t)reps);
    for (i = 0; i < (rawLen >= 4096 ? 4096 : 2048); i++) raw[i] = vout[i];
    res_put(&r);
    info[0] = 0;
    return 0;
}


/* ================================================================== */
/* prefill / batched int8 GEMM with real VTCM staging                  */
/* ================================================================== */
/* Shapes: N is a multiple of 64, K a multiple of 32, M any (padded to 64
 * spatial rows per tile).  Every HMX operand lives in VTCM, so a real FFN
 * layer must stream the weights DDR -> VTCM; mode bit0 selects that path.
 *
 *   mode bit0 : stage the weights from DDR for every N tile (honest cost)
 *        bit1 : skip the HMX instruction stream (staging only)
 *        bit2 : keep HMX power / locks across reps (timing isolation)
 *        bit3 : collect C (slow scalar unpack, correctness runs only) */

#define PREF_MAXK 2048

static inline void i8_mul_wdeep(const void *act, uint32_t art,
                                const void *wgt, uint32_t wc)
{
    __asm__ __volatile__(
        "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.b = mxmem(%2,%3):deep\n\t}"
        :: "r"((uint32_t)(uintptr_t)act), "r"(art),
           "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
}

/* DDR -> VTCM copy: 2048 B = 16 x 128 B HVX vectors */
static inline void copy_v128(void *dst, const void *src)
{
    int i;
    const uint8_t *s = (const uint8_t *)src;
    uint8_t *d = (uint8_t *)dst;
    for (i = 0; i < 2048; i += 128) {
        HVX_Vector v = *(const HVX_Vector *)(s + i);
        *(HVX_Vector *)(d + i) = v;
    }
}

/* whole 2 KB crouton, N loads in flight then N stores (U-way unroll).
 * The plain `*d++ = *s++` loop serialises load/store (store-load aliasing),
 * which caps the copy at ~2 GB/s (one 128 B load per DDR latency). */
static inline void copy_v2048(void *dst, const void *src)
{
    const HVX_Vector *s = (const HVX_Vector *)src;
    HVX_Vector *d = (HVX_Vector *)dst;
    int i;
    for (i = 0; i < 16; i++) d[i] = s[i];
}

static void copy_mem(void *dst, const void *src)
{
    memcpy(dst, src, 2048);
}

/* 2048 B with 64-bit scalar transfers */
static void copy64(void *dst, const void *src)
{
    const uint64_t *s = (const uint64_t *)src;
    uint64_t *d = (uint64_t *)dst;
    uint64_t v0, v1, v2, v3;
    int i;
    for (i = 0; i < 256; i += 4) {
        v0 = s[i]; v1 = s[i + 1]; v2 = s[i + 2]; v3 = s[i + 3];
        d[i] = v0; d[i + 1] = v1; d[i + 2] = v2; d[i + 3] = v3;
    }
}

__attribute__((noinline))
static void stage_plain(uint8_t *__restrict dst, const uint8_t *__restrict src,
                        unsigned bytes)
{
    const HVX_Vector *sp = (const HVX_Vector *)src;
    HVX_Vector *dp = (HVX_Vector *)dst;
    unsigned n = bytes >> 7, i;
    for (i = 0; i < n; i++) dp[i] = sp[i];
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

/* weight K x N, wl=4 style: one 2 KB crouton per 32 K x 64 N tile */
static void pref_pack_wgt(uint8_t *dst, const int8_t *B, int N, int K, int ncr)
{
    int k, n, nt;
    for (nt = 0; nt < N / 64; nt++)
        for (k = 0; k < K; k++)
            for (n = 0; n < 64; n++)
                dst[(unsigned)nt * (unsigned)ncr * 2048u
                    + 2048u * (unsigned)(k >> 5) + 1024u * (unsigned)(n >> 5)
                    + 128u * (unsigned)((k & 31) >> 2)
                    + 4u * (unsigned)(n & 31) + (unsigned)(k & 3)] =
                    (uint8_t)B[(size_t)(nt * 64 + n) * K + k];
}

static void pref_unpack(const uint8_t *tile, uint8_t *C, int ldc, int s0, int n0,
                        int rows, int cols)
{
    int s, n;
    for (s = 0; s < rows; s++)
        for (n = 0; n < cols; n++)
            C[(size_t)(s0 + s) * ldc + n0 + n] =
                tile[(unsigned)(s >> 2) * 128u + 4u * (unsigned)n + (unsigned)(s & 3)];
}

int mx4_pref(const unsigned char *Adata, int ALen,
             const unsigned char *Bdata, int BLen,
             int M, int N, int K, int mode, int reps,
             unsigned char *Cdata, int CLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias, *wpack = 0, *dscratch = 0;
    const int8_t *A = (const int8_t *)Adata;
    const int8_t *B = (const int8_t *)Bdata;
    int ncr, mt, nt, rc, i, rep, k, m, c, ch;
    unsigned aregion, wregion, need;
    uint64_t t0, t1, c0, c1;
    uint32_t art, wc;

    if (infoLen < 16) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > PREF_MAXK || (K & 31)) return -2;
    if (N < 64 || (N & 63)) return -3;
    if (M < 1 || M > 512) return -4;
    if (ALen < M * K) return -5;
    if (BLen < N * K) return -6;
    if (reps < 1 || reps > 100) reps = 1;

    ncr = K / 32;
    mt  = (M + 63) / 64;
    nt  = N / 64;

    aregion = 2048u * (unsigned)ncr * (unsigned)mt;
    wregion = 2048u * (unsigned)ncr;          /* one N tile of 64 channels */
    need = aregion + 2u * align_up(wregion, 2048u) + 4096u + 512u + 2048u + 2048u;
    if (mode & 4096) need = 2u << 20;      /* VTCM allocation size diagnostic */
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    info[10] = (int)r.size;

    vact  = r.base;
    vwgt  = vact + aregion;
    vout  = vwgt + 2u * align_up(wregion, 2048u);
    vbias = vout + 4096u;
    memset(r.base, 0, need);

    t0 = HAP_perf_get_time_us();
    pref_pack_act(vact, A, M, K, ncr);
    /* all N tiles packed into DDR once: the staged variant streams from here */
    wpack = (uint8_t *)memalign(128, (size_t)nt * (size_t)ncr * 2048u + 128u);
    if (!wpack) { res_put(&r); info[0] = -7; return 0; }
    if (mode & 64) {   /* staging destination = DDR, to separate DDR vs VTCM cost */
        dscratch = (uint8_t *)memalign(128, wregion + 128u);
        if (!dscratch) { free(wpack); res_put(&r); info[0] = -8; return 0; }
    }
    pref_pack_wgt(wpack, B, N, K, ncr);
    memcpy(vwgt, wpack, wregion);              /* resident set = first N tile */
    info[13] = (int)(HAP_perf_get_time_us() - t0);   /* layout transform, us */
    {
        uint32_t *bl = (uint32_t *)vbias;
        for (i = 0; i < 128; i++) bl[i] = 0;
        /* scale 0x6000 (= 512.0) is the int8 identity: the converter carries
         * an implicit 1/512.  Low words of both channel groups. */
        for (i = 0; i < 32; i++) { bl[i] = 0x6000u; bl[64 + i] = 0x6000u; }
    }

    if (!(mode & 4)) info[8] = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (!(mode & 4)) info[6] = hmx_power(1);
    if (!(mode & 4)) info[7] = compute_resource_hmx_lock ? compute_resource_hmx_lock(r.ctx) : -100;

    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        hmx_clr();
        if ((mode & 1) && (mode & 2048)) {
            for (k = 0; k < ncr; k++)
                copy_v2048(vwgt + (unsigned)k * 2048u,
                           ((const uint8_t *)wpack) + (size_t)k * 2048u);
            __asm__ __volatile__("syncht" ::: "memory");
            __asm__ __volatile__("barrier" ::: "memory");
        }
        for (i = 0; i < nt; i++) {
            if (mode & 1) {
                int j = (mode & 2048) ? (i + 1) : i;
                if (j < nt) {
                    uint64_t tc0 = HAP_perf_get_time_us();
                    const uint8_t *wsrc = ((mode & 512) ? (const uint8_t *)Bdata
                                                        : wpack)
                                          + (size_t)((mode & 8192) ? 0 : j)
                                            * (size_t)ncr * 2048u;
                    uint8_t *wdst = dscratch ? dscratch
                                  : (vwgt + (unsigned)((mode & 2048)
                                                       ? (j & 1) * (int)wregion : 0));
                    if ((mode & 2048) && i > 0)
                        hmx_sync();      /* previous tile must be done with its slot */
                    if (mode & 32) {
                        memcpy(wdst, wsrc, wregion);
                    } else if (mode & 128) {
                        for (k = 0; k < ncr; k++)
                            copy64(wdst + (unsigned)k * 2048u,
                                   wsrc + (size_t)k * 2048u);
                    } else if (mode & 256) {
                        stage_plain(wdst, wsrc, wregion);
                    } else {
                        for (k = 0; k < ncr; k++)
                            copy_v2048(wdst + (unsigned)k * 2048u,
                                       wsrc + (size_t)k * 2048u);
                    }
                    if (!(mode & 1024)) {
                        __asm__ __volatile__("syncht" ::: "memory");
                        __asm__ __volatile__("barrier" ::: "memory");
                    }
                    info[15] += (int)(HAP_perf_get_time_us() - tc0);
                    if (i == 0 && j == 0 && rep == 0 && !(mode & 16384)) {
                        int q, bad = 0;
                        for (q = 0; q < (int)wregion; q++)
                            if (vwgt[q] != wsrc[q]) bad++;
                        info[12] = bad;
                    }
                }
            }
            if (mode & 2) continue;
            {
                uint8_t *wuse = (mode & 2048)
                    ? (vwgt + (unsigned)(i & 1) * wregion) : vwgt;
            for (m = 0; m < mt; m++) {
                if (mode & 16) {
                    /* one :deep activation load per 32 K croutons */
                    for (c = 0; c < ncr; c += 32) {
                        ch = (ncr - c) > 32 ? 32 : (ncr - c);
                        art = ((uint32_t)(ch - 1) << 11) | 0x3FFu;
                        wc  = (uint32_t)ch * 2048u - 1u;
                        i8_mul_deepact(vact + (unsigned)m * (unsigned)ncr * 2048u
                                       + (unsigned)c * 2048u, art,
                                       wuse + (unsigned)c * 2048u, wc);
                    }
                } else {
                    /* verified N=64 recipe: one pair per K crouton, :deep on
                     * the weight so the acc defers until the store */
                    for (c = 0; c < ncr; c++)
                        i8_mul_wdeep(vact + (unsigned)m * (unsigned)ncr * 2048u
                                     + (unsigned)c * 2048u, 0x3FFu,
                                     wuse + (unsigned)c * 2048u, 2047u);
                }
                hmx_bias(vbias);
                i8_store(vout, 0, 0);
                hmx_bias(vbias + 256);
                i8_store(vout + 2048, 0, 0);
                if (mode & 8) {
                    int rows = (M - m * 64) < 64 ? (M - m * 64) : 64;
                    pref_unpack(vout, Cdata, N, m * 64, i * 64, rows, 32);
                    pref_unpack(vout + 2048, Cdata, N, m * 64, i * 64 + 32, rows, 32);
                }
            }
            }
        }
    }
    hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)(t1 - t0);
    info[2] = (int)(c1 - c0);
    info[3] = nt;
    info[4] = mt;
    info[5] = ncr;
    info[9] = mode;
    info[11] = (int)((unsigned long long)nt * (unsigned)ncr * 2048u);
    info[12] = 0;

    if (!(mode & 4)) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        qurt_hvx_unlock();
    }
    free(dscratch);
    free(wpack);
    res_put(&r);
    info[0] = 0;
    return 0;
}


/* ================================================================== */
/* int4 prefill: real packed int4 weights staged DDR -> VTCM           */
/* ================================================================== */
/* Same shape rules as `pref` but 32 output channels per tile (int4 weight
 * crouton = 512 B = 32 K x 32 N) and half the weight traffic.
 *
 *   A : M x K int8 row major, M <= 256, spatial padded to 64 rows per tile
 *   B : packed int4, tile t = N/32 block, (K/32) croutons of 512 B each:
 *         byte(t,k,n) = t*(K/32)*512 + 512*(k>>5) + 128*((k&31)>>3)
 *                       + 4*n + (k&3)
 *       nibble high iff ((k>>2)&1), 4-bit two's complement
 *
 *   mode bit0 : stage weights DDR->VTCM per tile (honest cost); off = the
 *               first strip stays resident (compute-only reference)
 *        bit1 : skip the HMX stream (staging only)
 *        bit2 : keep HMX power / HVX lock across reps
 *        bit3 : collect C (adds a sync per tile, correctness runs only)
 *        bit4 : double-buffered staging (copy tile i+1 into the other slot
 *               while the HMX works on tile i)
 *        bit5 : stage straight from the FastRPC buffer (else the weights are
 *               copied once into a DSP-heap buffer first, like `pref`)
 *        bit6 : evict L2 (16 MB read) before the timed loop, cold DDR baseline
 *        bits 12..15 : staging depth - 1 (32-N tiles copied per DDR->VTCM
 *               step; 1 tile = K*512/32 bytes per step)
 *
 * Staging uses HVX 128 B vector copies: DSP libc memcpy into VTCM runs at
 * ~0.9 GB/s, the vector loop at ~5 GB/s on the same buffers.
 *
 * info = {status, us_total, us_copy, pcyc_total, ntiles, mtiles, ncr,
 *         vtcm_bytes, staged_bytes, mode, act_pack_us, wheap_copy_us,
 *         copycheck, reps, copy_bytes, sat}
 */
#define I4P_MAXK 8192

static inline void i4_mul(const void *act, uint32_t art,
                          const void *wgt, uint32_t wc, int deep);

static void i4p_copy(uint8_t *dst, const uint8_t *src, unsigned bytes, int hvx)
{
    if (hvx && !(bytes & 2047u)) {
        unsigned i;
        for (i = 0; i < bytes; i += 2048u)
            copy_v2048(dst + i, src + i);
    } else {
        memcpy(dst, src, bytes);
    }
    /* VTCM visibility: same barrier sequence as `pref` */
    __asm__ __volatile__("syncht" ::: "memory");
    __asm__ __volatile__("barrier" ::: "memory");
}

int mx4_i4pref(const unsigned char *Adata, int ALen,
               const unsigned char *Bdata, int BLen,
               int M, int N, int K, int mode, int reps,
               unsigned char *Cdata, int CLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *base, *vact, *vwgt, *vout, *vbias, *wheap = 0;
    const int8_t *A = (const int8_t *)Adata;
    int ncr, mt, nt, rc, i, rep, m, c, g, depth, ngroups;
    unsigned aregion, wstrip, nbuf, need, wbytes, chunk, tbytes;
    uint64_t t0, t1, c0, c1, tc;
    int sat = 0;
    const uint8_t *wsrc;

    if (infoLen < 16) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > I4P_MAXK || (K & 31)) return -2;
    if (N < 32 || (N & 31)) return -3;
    if (M < 1 || M > 256) return -4;
    if (ALen < M * K) return -5;
    if (BLen < (N / 32) * (K / 32) * 512) return -6;
    if (mode & 8) {
        if (CLen < M * N) return -7;
    }
    if (reps < 1 || reps > 100) reps = 1;

    ncr = K / 32;
    mt = (M + 63) / 64;
    nt = N / 32;
    wstrip = 512u * (unsigned)ncr;          /* one 32-N tile, all K */
    depth = ((mode >> 12) & 15) + 1;        /* 32-N tiles per staging step */
    if (depth > nt) depth = nt;
    ngroups = (nt + depth - 1) / depth;
    chunk = (unsigned)depth * wstrip;
    nbuf = (mode & 16) ? 2u : 1u;
    wbytes = (unsigned)nt * wstrip;

    aregion = 2048u * (unsigned)ncr * (unsigned)mt;
    need = aregion + nbuf * align_up(chunk, 2048u) + 2048u + 2048u + 512u + 2048u;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    info[7] = (int)r.size;

    base = (uint8_t *)(((uintptr_t)r.base + 2047u) & ~(uintptr_t)2047u);
    vact = base;
    vwgt = vact + aregion;
    vout = vwgt + nbuf * align_up(chunk, 2048u);
    vbias = vout + 2048u + 2048u;           /* store slot 2 KB aligned */
    memset(base, 0, need);
    info[4] = nt; info[5] = mt; info[6] = ncr;
    info[8] = (int)wbytes;
    info[9] = mode;
    info[13] = reps;

    t0 = HAP_perf_get_time_us();
    pref_pack_act(vact, A, M, K, ncr);
    {
        uint32_t *bl = (uint32_t *)vbias;
        for (i = 0; i < 128; i++) bl[i] = 0;
        for (i = 0; i < 32; i++) { bl[i] = 0x6000u; bl[64 + i] = 0x6000u; }
    }
    info[10] = (int)(HAP_perf_get_time_us() - t0);
    if (!(mode & 32)) {
        /* prod-like: weights resident in a DSP-heap (DDR) buffer */
        wheap = (uint8_t *)memalign(128, wbytes + 128u);
        if (!wheap) { res_put(&r); info[0] = -8; return 0; }
        t0 = HAP_perf_get_time_us();
        memcpy(wheap, Bdata, wbytes);
        info[11] = (int)(HAP_perf_get_time_us() - t0);
        wsrc = wheap;
    } else {
        wsrc = Bdata;
    }

    if (!(mode & 1)) {
        /* resident mode: stage the first chunk once, outside the timer */
        i4p_copy(vwgt, wsrc, chunk < wbytes ? chunk : wbytes, 1);
    }

    if (!(mode & 4)) {
        qurt_hvx_lock(QURT_HVX_MODE_128B);
        hmx_power(1);
        if (compute_resource_hmx_lock) compute_resource_hmx_lock(r.ctx);
    }

    if (mode & 64) {
        /* evict L2 so the staged read really comes from DDR (cold baseline) */
        unsigned i, n = (16u << 20) / 128u;
        uint8_t *scr = (uint8_t *)memalign(128, 16u << 20);
        if (scr) {
            const HVX_Vector *p = (const HVX_Vector *)scr;
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
            memset(scr, 0x33, 16u << 20);
            for (i = 0; i + 1 < n; i += 2) {
                a0 = Q6_Vw_vadd_VwVw(a0, p[i]);
                a1 = Q6_Vw_vadd_VwVw(a1, p[i + 1]);
            }
            ((HVX_Vector *)vwgt)[0] = Q6_V_vxor_VV(a0, a1);
            __asm__ __volatile__("syncht" ::: "memory");
            __asm__ __volatile__("barrier" ::: "memory");
            free(scr);
        }
    }

    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        if ((mode & 1) && (mode & 16)) {
            /* prime slot 0 of the double buffer (inside the honest timer) */
            tbytes = chunk < wbytes ? chunk : wbytes;
            tc = HAP_perf_get_time_us();
            i4p_copy(vwgt, wsrc, tbytes, 1);
            info[2] += (int)(HAP_perf_get_time_us() - tc);
            info[14] += (int)tbytes;
        }
        for (g = 0; g < ngroups; g++) {
            int i0 = g * depth;
            int ncell = (nt - i0) < depth ? (nt - i0) : depth;
            uint8_t *slot = (mode & 16)
                ? (vwgt + (unsigned)(g & 1) * align_up(chunk, 2048u)) : vwgt;
            tbytes = (unsigned)ncell * wstrip;
            if ((mode & 1) && !(mode & 16)) {
                tc = HAP_perf_get_time_us();
                i4p_copy(slot, wsrc + (size_t)i0 * wstrip, tbytes, 1);
                info[2] += (int)(HAP_perf_get_time_us() - tc);
                info[14] += (int)tbytes;
                if (rep == 0 && g == 0) {
                    int q, bad = 0;
                    for (q = 0; q < (int)tbytes; q++)
                        if (slot[q] != wsrc[q]) bad++;
                    info[12] = bad;             /* reused as copycheck on rep 0 */
                }
            }
            if (mode & 2) {
                /* staging only: still pipeline the copies */
                if ((mode & 16) && (mode & 1) && g + 1 < ngroups) {
                    int i1 = (g + 1) * depth;
                    unsigned nb = (unsigned)((nt - i1) < depth ? (nt - i1) : depth) * wstrip;
                    tc = HAP_perf_get_time_us();
                    i4p_copy(vwgt + (unsigned)((g + 1) & 1) * align_up(chunk, 2048u),
                             wsrc + (size_t)i1 * wstrip, nb, 1);
                    info[2] += (int)(HAP_perf_get_time_us() - tc);
                    info[14] += (int)nb;
                }
                continue;
            }
            for (i = 0; i < ncell; i++) {
                int tile = i0 + i;
                for (m = 0; m < mt; m++) {
                    if (!(mode & 128)) hmx_clr();
                    for (c = 0; c < ncr; c++)
                        i4_mul(vact + (unsigned)(m * ncr + c) * 2048u, 0x3FFu,
                               slot + (unsigned)i * wstrip + (unsigned)c * 512u,
                               511u, (mode & 128) ? 2 : 0);
                    hmx_bias(vbias);
                    i8_store(vout, 0, 0);
                    if (mode & 8) {
                        int rows = (M - m * 64) < 64 ? (M - m * 64) : 64;
                        int s, n;
                        hmx_sync();
                        pref_unpack(vout, Cdata, N, m * 64, tile * 32, rows, 32);
                        for (s = 0; s < rows; s++)
                            for (n = 0; n < 32; n++)
                                if (Cdata[(size_t)(m * 64 + s) * N + tile * 32 + n] == 255)
                                    sat++;
                    }
                }
            }
            if ((mode & 1) && (mode & 16) && g + 1 < ngroups) {
                /* the next chunk goes into the slot of group g-1: wait for that
                 * compute only, then copy while group g is still running */
                int i1 = (g + 1) * depth;
                unsigned nb = (unsigned)((nt - i1) < depth ? (nt - i1) : depth) * wstrip;
                if (mode & 2) { /* staging-only: nothing to wait for */ }
                else if (g > 0) hmx_sync();
                tc = HAP_perf_get_time_us();
                i4p_copy(vwgt + (unsigned)((g + 1) & 1) * align_up(chunk, 2048u),
                         wsrc + (size_t)i1 * wstrip, nb, 1);
                info[2] += (int)(HAP_perf_get_time_us() - tc);
                info[14] += (int)nb;
            }
        }
    }
    hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)(t1 - t0);
    info[3] = (int)(c1 - c0);
    info[15] = sat;

    if (!(mode & 4)) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        qurt_hvx_unlock();
    }
    free(wheap);
    res_put(&r);
    info[0] = 0;
    return 0;
}


/* ================================================================== */
/* int4 prefill with per-block scales (REPORT24)                       */
/* ================================================================== */
/* Same operand shape as `i4pref`, but the weights carry one scale per
 * G*32 values of K and the result is the real (float) GEMM output, so the
 * block scales are actually applied.
 *
 *   A : M x K uint8 row major (activations, spatial padded to 64)
 *   B : packed int4, tile t = N/32, (K/32) croutons of 512 B, same layout
 *       as `i4pref` but quantised with one scale per 32*G values of K
 *   S : float[N * ng], ng = K/(32*G) - the block scales themselves
 *   C : float[M * N]
 *
 * How the scales are applied (measured in hexagon-sim, see REPORT24):
 *   - one int4 packet = 32 K x 32 N x 64 spatial, so G packets accumulate
 *     one partial sum per output element;
 *   - the only partial store that covers all 64 spatial rows is
 *     `:after:sat.ub = acc`: 2 KB, unsigned 8-bit, multiplied by
 *     fp16(bias word)/512.  `.uh` writes only the odd 32 of the 64 rows, so
 *     16-bit partials would cost twice the packets and twice the traffic;
 *   - one extra all-constant packet adds the offset C0 to every lane, which
 *     keeps the partial positive; the store multiplier is the fixed
 *     m = 255/(2*C0), i.e. every partial uses the whole 0..255 range;
 *   - the per-(n, group) block scale is applied in the combine: one
 *     multiply-add per element per group.  That is the price of block-32
 *     accuracy: ng multiply-adds and ng*2 KB of partial traffic per tile.
 *
 * mode bit0 : stage weights DDR->VTCM (double buffered, overlaps the HMX
 *             stream), bit1 staging only, bit2 keep power/locks,
 *      bit8 : write C, bit6 evict L2.
 * info = {status, us_total, us_copy, pcyc, ntiles, mtiles, ncr, vtcm_bytes,
 *         staged_bytes, mode, act_pack_us, combine_us, copycheck, reps,
 *         copy_bytes, ng, 0, 0, 0}
 */
#define BLK_MAXK 8192
#define BLK_MAXG 64

/* .ub partial layout: out(s,n) = 128*(s>>2) + 4*n + (s&3).
 * C[s][n] = (sum_g S[n][g] * (u + 0.5)) / m - C0 * sum_g S[n][g]
 * The +0.5 removes the store's truncation bias (verified in hexagon-sim). */
static void blk_emit(const uint8_t *part, int ng, const float *S, int sng,
                     float m, float C0, float *C, int ldc, int rows, int n0)
{
    int s, n, g;
    for (s = 0; s < rows; s++)
        for (n = 0; n < 32; n++) {
            float acc = 0, scs = 0;
            for (g = 0; g < ng; g++) {
                float sc = S[(size_t)(n0 + n) * sng + g];
                acc += sc * ((float)part[(size_t)g * 2048u
                              + 128u * (unsigned)(s >> 2) + 4u * (unsigned)n
                              + (unsigned)(s & 3)] + 0.5f);
                scs += sc;
            }
            C[(size_t)s * ldc + n0 + n] = acc / m - C0 * scs;
        }
}

int mx4_blkpref(const unsigned char *Adata, int ALen,
                const unsigned char *Bdata, int BLen,
                const unsigned char *Sdata, int SLen,
                int M, int N, int K, int G, int C0, int mode, int reps,
                unsigned char *Cdata, int CLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *base, *vact, *vwgt, *vpart, *vbias, *voact, *vowgt;
    const int8_t *A = (const int8_t *)Adata;
    const float *S = (const float *)Sdata;
    float *C = (float *)Cdata;
    int ncr, ng, mt, nt, rc, i, rep, m, c, g, j;
    unsigned aregion, wstrip, nbuf, need, wbytes, chunk;
    uint64_t t0, t1, c0, c1, tc;
    int copycheck = 0, Aq;
    float mstore;

    if (infoLen < 16) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > BLK_MAXK || (K & 31)) return -2;
    if (N < 32 || (N & 31)) return -3;
    if (M < 1 || M > 256) return -4;
    if (G < 1 || G > BLK_MAXG) return -5;
    if (K % (32 * G)) return -6;
    if (ALen < M * K) return -7;
    if (BLen < (N / 32) * (K / 32) * 512) return -8;
    ncr = K / 32;
    ng = ncr / G;
    if (SLen < (int)((size_t)N * ng * sizeof(float))) return -9;
    if (C0 < 224 || (C0 % 224) != 0) return -10;   /* offset = 224*Aq */
    Aq = C0 / 224;
    if (Aq > 255) return -11;
    if (mode & 256) {
        if (CLen < M * N * (int)sizeof(float)) return -12;
    }
    if (reps < 1 || reps > 100) reps = 1;

    mt = (M + 63) / 64;
    nt = N / 32;
    wstrip = 512u * (unsigned)ncr;
    chunk = wstrip;
    nbuf = (mode & 1) ? 2u : 1u;
    wbytes = (unsigned)nt * wstrip;
    mstore = 255.0f / (2.0f * (float)C0);

    aregion = 2048u * (unsigned)ncr * (unsigned)mt;
    need = aregion
         + nbuf * align_up(chunk, 2048u)
         + (unsigned)ng * 2048u + 2048u
         + 512u + 2048u + 2048u + 512u;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    info[7] = (int)r.size;

    base = (uint8_t *)(((uintptr_t)r.base + 2047u) & ~(uintptr_t)2047u);
    vact = base;
    vwgt = vact + aregion;
    vpart = vwgt + nbuf * align_up(chunk, 2048u);
    vbias = vpart + (unsigned)ng * 2048u;
    voact = vbias + 512u + 2048u;
    vowgt = voact + 2048u;
    memset(base, 0, need);
    info[4] = nt; info[5] = mt; info[6] = ncr;
    info[8] = (int)wbytes;
    info[9] = mode;
    info[13] = reps;
    info[15] = ng;

    t0 = HAP_perf_get_time_us();
    pref_pack_act(vact, A, M, K, ncr);
    info[10] = (int)(HAP_perf_get_time_us() - t0);
    if (!(mode & 1))
        i4p_copy(vwgt, Bdata, wstrip, 1);   /* resident reference strip */

    /* constant packets: the offset adds C0 to every (s,n) lane; the bias
     * block normalises the 8-bit store to the full range */
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
        for (j = 0; j < 32; j++) {
            ((uint32_t *)vbias)[j] = word;
            ((uint32_t *)vbias)[64 + j] = word;
        }
    }

    if (!(mode & 4)) {
        qurt_hvx_lock(QURT_HVX_MODE_128B);
        hmx_power(1);
        if (compute_resource_hmx_lock) compute_resource_hmx_lock(r.ctx);
    }
    if (mode & 64) {
        unsigned q, n = (16u << 20) / 128u;
        uint8_t *scr = (uint8_t *)memalign(128, 16u << 20);
        if (scr) {
            const HVX_Vector *p = (const HVX_Vector *)scr;
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
            memset(scr, 0x33, 16u << 20);
            for (q = 0; q + 1 < n; q += 2) {
                a0 = Q6_Vw_vadd_VwVw(a0, p[q]);
                a1 = Q6_Vw_vadd_VwVw(a1, p[q + 1]);
            }
            ((HVX_Vector *)vwgt)[0] = Q6_V_vxor_VV(a0, a1);
            __asm__ __volatile__("syncht" ::: "memory");
            __asm__ __volatile__("barrier" ::: "memory");
            free(scr);
        }
    }

    if ((mode & 1) && !(mode & 2)) {
        /* prime slot 0 inside the honest timer */
        tc = HAP_perf_get_time_us();
        i4p_copy(vwgt, Bdata, wstrip, 1);
        info[2] += (int)(HAP_perf_get_time_us() - tc);
        info[14] += (int)wstrip;
        if (!(mode & 256)) {
            int q, bad = 0;
            for (q = 0; q < (int)wstrip; q++) if (vwgt[q] != Bdata[q]) bad++;
            copycheck = bad;
        }
    }

    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        for (i = 0; i < nt; i++) {
            uint8_t *slot = (mode & 1)
                ? vwgt + (unsigned)(i & 1) * align_up(chunk, 2048u) : vwgt;
            if (mode & 2) {
                /* staging only: no HMX stream, just the copies */
                tc = HAP_perf_get_time_us();
                i4p_copy(vwgt + (unsigned)(i & 1) * align_up(chunk, 2048u),
                         Bdata + (size_t)i * wstrip, wstrip, 1);
                info[2] += (int)(HAP_perf_get_time_us() - tc);
                info[14] += (int)wstrip;
                continue;
            }
            if (rep == 0 && i == 0) {
                int q, bad = 0;
                for (q = 0; q < (int)wstrip; q++)
                    if (slot[q] != Bdata[q]) bad++;
                copycheck = bad;
            }
            for (m = 0; m < mt; m++) {
                int rows = (M - m * 64) < 64 ? (M - m * 64) : 64;
                /* the group chain is deferred in the HMX queue, so the
                 * staging copy below runs on HVX while the array works */
                for (g = 0; g < ng; g++) {
                    hmx_clr();
                    i4_mul(voact, 0x3FFu, vowgt, 511u, 0);      /* +C0 */
                    for (c = 0; c < G; c++)
                        i4_mul(vact + (unsigned)(m * ncr + g * G + c) * 2048u,
                               0x3FFu,
                               slot + (unsigned)(g * G + c) * 512u, 511u, 0);
                    hmx_bias(vbias);
                    hmx_st_ub(vpart + (unsigned)g * 2048u, 0);
                }
                if (m + 1 == mt && (mode & 1) && i + 1 < nt) {
                    uint8_t *nslot = vwgt
                        + (unsigned)((i + 1) & 1) * align_up(chunk, 2048u);
                    tc = HAP_perf_get_time_us();
                    i4p_copy(nslot, Bdata + (size_t)(i + 1) * wstrip, wstrip, 1);
                    info[2] += (int)(HAP_perf_get_time_us() - tc);
                    info[14] += (int)wstrip;
                }
                hmx_sync();                    /* drain this tile's stores */
                if (mode & 256) {
                    tc = HAP_perf_get_time_us();
                    blk_emit(vpart, ng, S, ng, mstore, (float)C0, C, N,
                             rows, i * 32);
                    info[11] += (int)(HAP_perf_get_time_us() - tc);
                }
            }
        }
    }
    hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)(t1 - t0);
    info[3] = (int)(c1 - c0);
    info[12] = copycheck;

    if (!(mode & 4)) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        qurt_hvx_unlock();
    }
    res_put(&r);
    info[0] = 0;
    return 0;
}

/* DDR / VTCM copy bandwidth probe.
 *   mode 0 : HVX read-sum of a DDR buffer            (read only)
 *   mode 1 : HVX read from DDR, store to DDR buffer  (read+write)
 *   mode 2 : HVX read from DDR, store to VTCM        (staging path)
 * r = {status, us, bytes, pcyc, GB/s*100, align_ddr, 0, 0} */
int mx4_bw(int mb, int mode, int reps, int *r, int rLen)
{
    res_t res;
    uint8_t *src, *dst;
    unsigned bytes = (unsigned)mb << 20;
    unsigned i, n = bytes / 128u;
    int rc, rep;
    uint64_t t0, t1, c0, c1;
    HVX_Vector acc;
    int hvx;

    if (rLen < 8) return -1;
    memset(r, 0, sizeof(int) * rLen);
    if (mb < 1 || mb > 64 || reps < 1 || reps > 100) return -2;

    src = (uint8_t *)memalign(128, bytes);
    dst = (uint8_t *)memalign(128, 1u << 20);
    if (!src || !dst) { r[0] = -3; return 0; }
    memset(src, 1, bytes);
    memset(dst, 0, 1u << 20);
    r[5] = (int)(((uintptr_t)src) & 127);

    rc = res_get(1u << 20, &res);
    r[0] = rc;
    if (rc) { free(src); free(dst); return 0; }

    hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hvx) { r[0] = -4; res_put(&res); free(src); free(dst); return 0; }

    acc = Q6_V_vzero();
    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        if (mode == 0) {
            const HVX_Vector *p = (const HVX_Vector *)src;
            HVX_Vector a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
            for (i = 0; i < n; i += 4) {
                acc = Q6_Vw_vadd_VwVw(acc, p[i]);
                a1  = Q6_Vw_vadd_VwVw(a1,  p[i + 1]);
                a2  = Q6_Vw_vadd_VwVw(a2,  p[i + 2]);
                a3  = Q6_Vw_vadd_VwVw(a3,  p[i + 3]);
            }
            acc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(acc, a1),
                                  Q6_Vw_vadd_VwVw(a2, a3));
        } else if (mode == 1) {
            const HVX_Vector *p = (const HVX_Vector *)src;
            HVX_Vector *q = (HVX_Vector *)dst;
            unsigned m = (1u << 20) / 128u;
            unsigned j = 0;
            for (i = 0; i < n; i++) { q[j] = p[i]; if (++j == m) j = 0; }
        } else {
            const HVX_Vector *p = (const HVX_Vector *)src;
            HVX_Vector *q = (HVX_Vector *)res.base;
            unsigned m = (1u << 20) / 128u;
            unsigned j = 0;
            for (i = 0; i < n; i++) { q[j] = p[i]; if (++j == m) j = 0; }
        }
    }
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    ((HVX_Vector *)dst)[0] = acc;
    r[1] = (int)(t1 - t0);
    r[2] = (int)bytes * reps;
    r[3] = (int)(c1 - c0);
    r[4] = (t1 > t0) ? (int)((double)bytes * reps / (double)(t1 - t0) * 100.0 / 1000.0) : 0;
    r[6] = 0;
    r[7] = 0;

    qurt_hvx_unlock();
    res_put(&res);
    free(src);
    free(dst);
    return 0;
}


/* ================================================================== */
/* bw2: where the 4.29 GB/s ceiling comes from                         */
/* ================================================================== */
/* kind:
 *   0 read-sum, 4 accumulators      (same shape as mx4_bw mode 0)
 *   1 read-sum, 8 accumulators
 *   2 copy DDR->VTCM per 2 KB crouton (staging baseline)
 *   3 copy DDR->VTCM, contiguous chunkKb block
 *   4 copy DDR->VTCM, l2fetch pipelined (depth DEPTH chunks)
 *   5 read-sum, l2fetch pipelined (prefetch + consume, no VTCM store)
 *   6 libc memcpy DDR->VTCM per chunkKb block
 * mode bit0: source = host FastRPC/ION buffer, else DSP-heap memalign
 * mode bit1: write VTCM (kinds 2..4,6); else VTCM region untouched
 * r = {status, us, bytes, pcyc, GBps*100, src_align, sink, chunk_kb}
 */

#define BW2_VTCM (2u << 20)
#define BW2_DEPTH 4

static inline void l2f(void *p, unsigned len)
{
    while (len) {
        unsigned w = len > 32768u ? 32768u : len;
        uint64_t d = ((uint64_t)w << 16) | 1u;   /* height=1, stride=0, dir=0 */
        Q6_l2fetch_AP(p, d);
        p = (char *)p + w;
        len -= w;
    }
}

/* issue `ns` independent l2fetch streams covering one chunk */
static inline void l2f_multi(void *p, unsigned len, int ns)
{
    unsigned sub = (len / (unsigned)ns) & ~127u;
    int i;
    for (i = 0; i < ns; i++)
        l2f((char *)p + (size_t)i * sub, sub);
}

static void bw2_body(uint8_t *src, int kind, unsigned chunk, unsigned nchunks,
                     unsigned nvec, uint8_t *vdst, int reps, HVX_Vector *accp)
{
    unsigned i, ci, rep;
    HVX_Vector acc = *accp;

    for (rep = 0; rep < reps; rep++) {
        switch (kind) {
        case 0: {
            const HVX_Vector *p = (const HVX_Vector *)src;
            HVX_Vector a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
            unsigned n = nchunks * nvec;
            for (i = 0; i + 3 < n; i += 4) {
                acc = Q6_Vw_vadd_VwVw(acc, p[i]);
                a1  = Q6_Vw_vadd_VwVw(a1,  p[i + 1]);
                a2  = Q6_Vw_vadd_VwVw(a2,  p[i + 2]);
                a3  = Q6_Vw_vadd_VwVw(a3,  p[i + 3]);
            }
            acc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(acc, a1),
                                  Q6_Vw_vadd_VwVw(a2, a3));
            break;
        }
        case 1: {   /* batched loads: 16 vectors in flight, then consume */
            const HVX_Vector *p = (const HVX_Vector *)src;
            unsigned n = nchunks * nvec;
            for (i = 0; i + 15 < n; i += 16) {
                HVX_Vector b0 = p[i + 0],  b1 = p[i + 1],
                           b2 = p[i + 2],  b3 = p[i + 3],
                           b4 = p[i + 4],  b5 = p[i + 5],
                           b6 = p[i + 6],  b7 = p[i + 7],
                           b8 = p[i + 8],  b9 = p[i + 9],
                           b10 = p[i + 10], b11 = p[i + 11],
                           b12 = p[i + 12], b13 = p[i + 13],
                           b14 = p[i + 14], b15 = p[i + 15];
                acc = Q6_Vw_vadd_VwVw(acc, b0);
                b1 = Q6_Vw_vadd_VwVw(b1, b2);
                b3 = Q6_Vw_vadd_VwVw(b3, b4);
                b5 = Q6_Vw_vadd_VwVw(b5, b6);
                b7 = Q6_Vw_vadd_VwVw(b7, b8);
                b9 = Q6_Vw_vadd_VwVw(b9, b10);
                b11 = Q6_Vw_vadd_VwVw(b11, b12);
                b13 = Q6_Vw_vadd_VwVw(b13, b14);
                b15 = Q6_Vw_vadd_VwVw(b15, b1);
                b3 = Q6_Vw_vadd_VwVw(b3, b5);
                b7 = Q6_Vw_vadd_VwVw(b7, b9);
                b11 = Q6_Vw_vadd_VwVw(b11, b13);
                b15 = Q6_Vw_vadd_VwVw(b15, b3);
                b15 = Q6_Vw_vadd_VwVw(b15, b7);
                acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(b11, b15));
            }
            break;
        }
        case 2: {   /* per-2 KB crouton, 16 loads then 16 stores */
            unsigned nc = BW2_VTCM / 2048u;
            if (chunk < 2048u) break;
            for (ci = 0; ci < nchunks; ci++) {
                const uint8_t *s = src + (size_t)ci * chunk;
                uint8_t *d = vdst + (size_t)(ci % nc) * 2048u;
                unsigned k, ncr = chunk / 2048u;
                for (k = 0; k < ncr; k++) {
                    const HVX_Vector *sp = (const HVX_Vector *)(s + k * 2048u);
                    HVX_Vector *dp = (HVX_Vector *)(d + k * 2048u);
                    HVX_Vector b0 = sp[0],  b1 = sp[1],  b2 = sp[2],
                               b3 = sp[3],  b4 = sp[4],  b5 = sp[5],
                               b6 = sp[6],  b7 = sp[7],  b8 = sp[8],
                               b9 = sp[9],  b10 = sp[10], b11 = sp[11],
                               b12 = sp[12], b13 = sp[13], b14 = sp[14],
                               b15 = sp[15];
                    dp[0] = b0; dp[1] = b1; dp[2] = b2;  dp[3] = b3;
                    dp[4] = b4; dp[5] = b5; dp[6] = b6;  dp[7] = b7;
                    dp[8] = b8; dp[9] = b9; dp[10] = b10; dp[11] = b11;
                    dp[12] = b12; dp[13] = b13; dp[14] = b14; dp[15] = b15;
                }
            }
            break;
        }
        case 3: {   /* contiguous chunk, plain load/store loop */
            unsigned nc = BW2_VTCM / chunk;
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *s = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *d = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                for (i = 0; i < nvec; i++) d[i] = s[i];
            }
            break;
        }
        case 4: {   /* contiguous chunk + l2fetch pipeline */
            unsigned nc = BW2_VTCM / chunk;
            for (i = 0; i < BW2_DEPTH && i < nchunks; i++)
                l2f(src + (size_t)i * chunk, chunk);
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *s = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *d = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                if (ci + BW2_DEPTH < nchunks)
                    l2f(src + (size_t)(ci + BW2_DEPTH) * chunk, chunk);
                for (i = 0; i < nvec; i++) d[i] = s[i];
            }
            break;
        }
        case 5: {   /* l2fetch + consume (read only) */
            for (i = 0; i < BW2_DEPTH && i < nchunks; i++)
                l2f(src + (size_t)i * chunk, chunk);
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *s = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector a1 = Q6_V_vzero(), a2 = Q6_V_vzero(),
                           a3 = Q6_V_vzero();
                if (ci + BW2_DEPTH < nchunks)
                    l2f(src + (size_t)(ci + BW2_DEPTH) * chunk, chunk);
                for (i = 0; i + 3 < nvec; i += 4) {
                    acc = Q6_Vw_vadd_VwVw(acc, s[i]);
                    a1  = Q6_Vw_vadd_VwVw(a1,  s[i + 1]);
                    a2  = Q6_Vw_vadd_VwVw(a2,  s[i + 2]);
                    a3  = Q6_Vw_vadd_VwVw(a3,  s[i + 3]);
                }
                acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(a1,
                                      Q6_Vw_vadd_VwVw(a2, a3)));
            }
            break;
        }
        case 6: {   /* libc memcpy per chunk */
            unsigned nc = BW2_VTCM / chunk;
            for (ci = 0; ci < nchunks; ci++)
                memcpy(vdst + (size_t)(ci % nc) * chunk,
                       src + (size_t)ci * chunk, chunk);
            break;
        }
        case 7: {   /* 2 KB crouton copy: 8 loads, 8 stores, 2x per crouton */
            unsigned nc = BW2_VTCM / 2048u;
            if (chunk < 2048u) break;
            for (ci = 0; ci < nchunks; ci++) {
                const uint8_t *s = src + (size_t)ci * chunk;
                uint8_t *d = vdst + (size_t)(ci % nc) * 2048u;
                unsigned k, ncr = chunk / 2048u;
                for (k = 0; k < ncr; k++) {
                    const HVX_Vector *sp = (const HVX_Vector *)(s + k * 2048u);
                    HVX_Vector *dp = (HVX_Vector *)(d + k * 2048u);
                    int h;
                    for (h = 0; h < 16; h += 8) {
                        HVX_Vector b0 = sp[h], b1 = sp[h + 1],
                                   b2 = sp[h + 2], b3 = sp[h + 3],
                                   b4 = sp[h + 4], b5 = sp[h + 5],
                                   b6 = sp[h + 6], b7 = sp[h + 7];
                        dp[h] = b0; dp[h + 1] = b1;
                        dp[h + 2] = b2; dp[h + 3] = b3;
                        dp[h + 4] = b4; dp[h + 5] = b5;
                        dp[h + 6] = b6; dp[h + 7] = b7;
                    }
                }
            }
            break;
        }
        case 8: case 9: case 10: case 11: {
            /* N independent sequential streams spread over the buffer.
             * One 128 B vector per stream per iteration, step = ns. */
            int ns = 1 << (kind - 7);      /* 2, 4, 8, 16 */
            unsigned seg = (nchunks * nvec) / (unsigned)ns;
            const HVX_Vector *p = (const HVX_Vector *)src;
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero(),
                       a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
            HVX_Vector a4 = Q6_V_vzero(), a5 = Q6_V_vzero(),
                       a6 = Q6_V_vzero(), a7 = Q6_V_vzero();
            seg &= ~7u;
            for (i = 0; i < seg; i += (unsigned)ns) {
                a0 = Q6_Vw_vadd_VwVw(a0, p[i]);
                if (ns > 1) a1 = Q6_Vw_vadd_VwVw(a1, p[seg + i]);
                if (ns > 2) a2 = Q6_Vw_vadd_VwVw(a2, p[2 * seg + i]);
                if (ns > 3) a3 = Q6_Vw_vadd_VwVw(a3, p[3 * seg + i]);
                if (ns > 4) {
                    a4 = Q6_Vw_vadd_VwVw(a4, p[4 * seg + i]);
                    a5 = Q6_Vw_vadd_VwVw(a5, p[5 * seg + i]);
                    a6 = Q6_Vw_vadd_VwVw(a6, p[6 * seg + i]);
                    a7 = Q6_Vw_vadd_VwVw(a7, p[7 * seg + i]);
                }
            }
            if (ns == 2)
                a0 = Q6_Vw_vadd_VwVw(a0, a1);
            else if (ns == 4)
                a0 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1),
                                     Q6_Vw_vadd_VwVw(a2, a3));
            else if (ns > 4)
                a0 = Q6_Vw_vadd_VwVw(a0, Q6_Vw_vadd_VwVw(
                        Q6_Vw_vadd_VwVw(a4, a5), Q6_Vw_vadd_VwVw(a6, a7)));
            acc = Q6_Vw_vadd_VwVw(acc, a0);
            break;
        }
        case 12: case 13: case 14: {
            /* copy the chunk as 2/4/8 independent sequential streams */
            int ns = 1 << (kind - 11);
            unsigned sub = chunk / (unsigned)ns;      /* bytes per stream */
            unsigned sn = sub / 128u;                 /* vectors per stream */
            unsigned nc = BW2_VTCM / chunk;
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *sp = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *dp = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                unsigned j;
                for (j = 0; j < sn; j++) {
                    HVX_Vector b0, b1, b2, b3, b4, b5, b6, b7;
                    b0 = sp[j];
                    dp[j] = b0;
                    if (ns > 1) {
                        b1 = sp[sn + j]; dp[sn + j] = b1;
                    }
                    if (ns > 2) {
                        b2 = sp[2 * sn + j]; dp[2 * sn + j] = b2;
                        b3 = sp[3 * sn + j]; dp[3 * sn + j] = b3;
                    }
                    if (ns > 4) {
                        b4 = sp[4 * sn + j]; dp[4 * sn + j] = b4;
                        b5 = sp[5 * sn + j]; dp[5 * sn + j] = b5;
                        b6 = sp[6 * sn + j]; dp[6 * sn + j] = b6;
                        b7 = sp[7 * sn + j]; dp[7 * sn + j] = b7;
                    }
                }
            }
            break;
        }
        case 15: case 16: {   /* multi-stream l2fetch pipeline + plain copy */
            int ns = (kind == 15) ? 8 : 4;
            unsigned nc = BW2_VTCM / chunk;
            for (i = 0; i < BW2_DEPTH && i < nchunks; i++)
                l2f_multi(src + (size_t)i * chunk, chunk, ns);
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *sp = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *dp = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                if (ci + BW2_DEPTH < nchunks)
                    l2f_multi(src + (size_t)(ci + BW2_DEPTH) * chunk, chunk, ns);
                for (i = 0; i < nvec; i++) dp[i] = sp[i];
            }
            break;
        }
        case 17: {   /* L2-resident copy: 512 KB window copied over and over */
            unsigned win = 512u << 10;
            unsigned wn = win / 128u;
            unsigned passes = (nchunks * chunk) / win;
            const HVX_Vector *sp = (const HVX_Vector *)src;
            HVX_Vector *dp = (HVX_Vector *)vdst;
            unsigned q;
            if (passes < 1) passes = 1;
            for (q = 0; q < passes; q++)
                for (i = 0; i < wn; i++) dp[i] = sp[i];
            break;
        }
        case 18: case 19: {
            /* copy with batched loads from 4 or 8 streams: 16 loads, then
             * 16 stores (the crouton pattern, but spread over the chunk) */
            int ns = (kind == 18) ? 4 : 8;
            unsigned per = (kind == 18) ? 4u : 2u;
            unsigned nc = BW2_VTCM / chunk;
            unsigned sub = (chunk / (unsigned)ns) & ~127u;
            unsigned sn = sub / 128u;
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *sp = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *dp = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                unsigned j;
                for (j = 0; j + per <= sn; j += per) {
                    HVX_Vector b[16];
                    int q;
                    for (q = 0; q < ns; q++)
                        for (i = 0; i < per; i++)
                            b[q * per + i] = sp[(size_t)q * sn + j + i];
                    for (q = 0; q < ns; q++)
                        for (i = 0; i < per; i++)
                            dp[(size_t)q * sn + j + i] = b[q * per + i];
                }
            }
            break;
        }
        case 20: {   /* two-pass staging: 8-stream read fills L2, then the
                      * chunk is copied L2 -> VTCM at cache speed */
            unsigned nc = BW2_VTCM / chunk;
            unsigned seg = nvec / 8u;
            for (ci = 0; ci < nchunks; ci++) {
                const HVX_Vector *s = (const HVX_Vector *)(src + (size_t)ci * chunk);
                HVX_Vector *d = (HVX_Vector *)(vdst + (size_t)(ci % nc) * chunk);
                HVX_Vector a1 = Q6_V_vzero(), a2 = Q6_V_vzero(),
                           a3 = Q6_V_vzero(), a4 = Q6_V_vzero(),
                           a5 = Q6_V_vzero(), a6 = Q6_V_vzero(),
                           a7 = Q6_V_vzero();
                for (i = 0; i < seg; i++) {
                    acc = Q6_Vw_vadd_VwVw(acc, s[i]);
                    a1 = Q6_Vw_vadd_VwVw(a1, s[seg + i]);
                    a2 = Q6_Vw_vadd_VwVw(a2, s[2 * seg + i]);
                    a3 = Q6_Vw_vadd_VwVw(a3, s[3 * seg + i]);
                    a4 = Q6_Vw_vadd_VwVw(a4, s[4 * seg + i]);
                    a5 = Q6_Vw_vadd_VwVw(a5, s[5 * seg + i]);
                    a6 = Q6_Vw_vadd_VwVw(a6, s[6 * seg + i]);
                    a7 = Q6_Vw_vadd_VwVw(a7, s[7 * seg + i]);
                }
                acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(
                        Q6_Vw_vadd_VwVw(a1, a2), Q6_Vw_vadd_VwVw(a3, a4)));
                acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(
                        Q6_Vw_vadd_VwVw(a5, a6), a7));
                for (i = 0; i < nvec; i++) d[i] = s[i];
            }
            break;
        }
        default:
            break;
        }
    }
    *accp = acc;
}

/* bus bandwidth vote: the DSP client's own DDR vote (mips_bw request) */
static int hmx_bus_vote(uint64_t bytes_per_sec, int on)
{
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_mips_bw;
    req.mips_bw.set_bus_bw = TRUE;
    req.mips_bw.bwBytePerSec = bytes_per_sec;
    req.mips_bw.busbwUsagePercentage = on ? 100 : 0;
    req.mips_bw.set_mips = FALSE;
    return HAP_power_set(0, &req);
}

/* DCVS v3: performance mode + bus corner TURBO (or back to defaults) */
static int hmx_bus_corner(int corner)
{
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_enable = TRUE;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = TRUE;
    req.dcvs_v3.bus_params.target_corner = (HAP_dcvs_voltage_corner_t)corner;
    req.dcvs_v3.bus_params.min_corner = (HAP_dcvs_voltage_corner_t)corner;
    req.dcvs_v3.bus_params.max_corner = (HAP_dcvs_voltage_corner_t)corner;
    return HAP_power_set(0, &req);
}

static void bw2_timed(uint8_t *src, int kind, unsigned chunk, unsigned nchunks,
                      unsigned nvec, uint8_t *vdst, int reps, HVX_Vector *acc,
                      int *us, int *pcyc, int *rate100)
{
    uint64_t t0, t1, c0, c1;
    unsigned bytes = chunk * nchunks * (unsigned)reps;
    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    bw2_body(src, kind, chunk, nchunks, nvec, vdst, reps, acc);
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    *us = (int)(t1 - t0);
    *pcyc = (int)(c1 - c0);
    *rate100 = (*us > 0) ? (int)((double)bytes / (double)*us * 100.0 / 1000.0) : 0;
}

/* r = {status, usA, bytes, pcycA, rateA*100, align, vote_rc, chunk_kb,
 *      usB, pcycB, rateB*100, src_align, usC, pcycC, rateC*100, 0} */
int mx4_bw2(const unsigned char *srcData, int srcLen,
            int kind, int chunkKb, int reps, int mode, int *r, int rLen)
{
    res_t res;
    uint8_t *src, *heap = 0, *host = (uint8_t *)srcData;
    unsigned bytes, chunk, nchunks;
    unsigned nvec;
    int rc;
    HVX_Vector acc;

    if (rLen < 16) return -1;
    memset(r, 0, sizeof(int) * rLen);
    if (srcLen < (1 << 20) || reps < 1 || reps > 50) { r[0] = -2; return 0; }
    if (chunkKb < 1 || chunkKb > 1024) chunkKb = 32;

    bytes = (unsigned)srcLen & ~127u;
    chunk = (unsigned)chunkKb << 10;
    if (chunk > bytes) chunk = bytes;
    nchunks = bytes / chunk;
    nvec = chunk / 128u;

    heap = (uint8_t *)memalign(4096, bytes + (1u << 18));
    if (!heap) { r[0] = -3; return 0; }
    memset(heap, 3, bytes + (1u << 18));

    rc = res_get(BW2_VTCM, &res);
    r[0] = rc;
    if (rc) { free(heap); return 0; }

    rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (rc) { r[0] = -4; res_put(&res); free(heap); return 0; }

    src = (mode & 1) ? host : heap;
    if (mode & 0x40) {          /* HMX powered + locked, as during prefill */
        r[6] = hmx_power(1);
        r[6] |= (compute_resource_hmx_lock ? compute_resource_hmx_lock(res.ctx) : -100) << 8;
    }
    r[5] = (int)(((uintptr_t)src) & 4095);
    r[7] = chunkKb;
    r[2] = (int)bytes * reps;

    acc = Q6_V_vzero();
    bw2_timed(src, kind, chunk, nchunks, nvec, res.base, reps, &acc,
              &r[1], &r[3], &r[4]);

    if (mode & 6) {         /* phase B: with a bus vote */
        int rc1 = 0;
        if (mode & 2) rc1 |= hmx_bus_vote((uint64_t)12 << 30, 1) & 0xff;
        if (mode & 4) rc1 |= (hmx_bus_corner(HAP_DCVS_VCORNER_TURBO) & 0xff) << 8;
        r[6] = rc1;
        bw2_timed(src, kind, chunk, nchunks, nvec, res.base, reps, &acc,
                  &r[8], &r[9], &r[10]);
        if (mode & 2) hmx_bus_vote(0, 0);
        if (mode & 4) hmx_bus_corner(HAP_DCVS_VCORNER_NOM);
        bw2_timed(src, kind, chunk, nchunks, nvec, res.base, reps, &acc,
                  &r[12], &r[13], &r[14]);
    }

    ((HVX_Vector *)res.base)[0] = acc;

    if (mode & 0x40) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(res.ctx);
        hmx_power(0);
    }
    qurt_hvx_unlock();
    res_put(&res);
    free(heap);
    return 0;
}


/* L2 fill test: cold read, warm read, l2fetch-then-read (single / 8 stream).
 * r = {status, bytes, cold_us, warm_us, pref1_us, pref8_us, cold_pc, warm_pc} */
int mx4_l2t(const unsigned char *srcData, int srcLen, int kb, int reps,
            int *r, int rLen)
{
    res_t res;
    uint8_t *heap, *src;
    unsigned bytes, i, rep;
    int rc;
    HVX_Vector acc;
    uint64_t t0, t1;
    int us[4], pc[4];

    if (rLen < 8) return -1;
    memset(r, 0, sizeof(int) * rLen);
    if (srcLen < kb * 1024 || kb < 64 || kb > 4096) { r[0] = -2; return 0; }
    bytes = (unsigned)kb * 1024u;
    heap = (uint8_t *)memalign(4096, bytes + (1u << 18));
    if (!heap) { r[0] = -3; return 0; }
    memset(heap, 5, bytes + (1u << 18));
    src = heap;
    rc = res_get(1u << 20, &res);
    r[0] = rc;
    if (rc) { free(heap); return 0; }
    rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (rc) { r[0] = -4; res_put(&res); free(heap); return 0; }

    for (i = 0; i < 4; i++) { us[i] = 0; pc[i] = 0; }
    acc = Q6_V_vzero();
    for (rep = 0; rep < reps; rep++) {
        const HVX_Vector *p = (const HVX_Vector *)src;
        unsigned n = bytes / 128u;
        unsigned j;

        /* cold: drop the region from L2 first */
        for (j = 0; j < bytes; j += 128)
            Q6_dccleaninva_A((void *)(src + j));

        t0 = HAP_perf_get_time_us();
        { uint64_t c = HAP_perf_get_pcycles();
          for (i = 0; i + 3 < n; i += 4) {
              acc = Q6_Vw_vadd_VwVw(acc, p[i]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 1]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 2]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 3]);
          }
          pc[0] += (int)(HAP_perf_get_pcycles() - c); }
        t1 = HAP_perf_get_time_us(); us[0] += (int)(t1 - t0);

        t0 = HAP_perf_get_time_us();
        { uint64_t c = HAP_perf_get_pcycles();
          for (i = 0; i + 3 < n; i += 4) {
              acc = Q6_Vw_vadd_VwVw(acc, p[i]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 1]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 2]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 3]);
          }
          pc[1] += (int)(HAP_perf_get_pcycles() - c); }
        t1 = HAP_perf_get_time_us(); us[1] += (int)(t1 - t0);

        for (j = 0; j < bytes; j += 128)
            Q6_dccleaninva_A((void *)(src + j));
        t0 = HAP_perf_get_time_us();
        { uint64_t c = HAP_perf_get_pcycles();
          l2f(src, bytes);                         /* single stream */
          __asm__ __volatile__("syncht" ::: "memory");
          __asm__ __volatile__("barrier" ::: "memory");
          for (i = 0; i + 3 < n; i += 4) {
              acc = Q6_Vw_vadd_VwVw(acc, p[i]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 1]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 2]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 3]);
          }
          pc[2] += (int)(HAP_perf_get_pcycles() - c); }
        t1 = HAP_perf_get_time_us(); us[2] += (int)(t1 - t0);

        for (j = 0; j < bytes; j += 128)
            Q6_dccleaninva_A((void *)(src + j));
        t0 = HAP_perf_get_time_us();
        { uint64_t c = HAP_perf_get_pcycles();
          l2f_multi(src, bytes, 8);                /* 8 streams */
          __asm__ __volatile__("syncht" ::: "memory");
          __asm__ __volatile__("barrier" ::: "memory");
          for (i = 0; i + 3 < n; i += 4) {
              acc = Q6_Vw_vadd_VwVw(acc, p[i]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 1]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 2]);
              acc = Q6_Vw_vadd_VwVw(acc, p[i + 3]);
          }
          pc[3] += (int)(HAP_perf_get_pcycles() - c); }
        t1 = HAP_perf_get_time_us(); us[3] += (int)(t1 - t0);
    }
    ((HVX_Vector *)res.base)[0] = acc;

    r[1] = (int)bytes * reps;
    r[2] = us[0] / reps;
    r[3] = us[1] / reps;
    r[4] = us[2] / reps;
    r[5] = us[3] / reps;
    r[6] = pc[0] / reps;
    r[7] = pc[1] / reps;
    qurt_hvx_unlock();
    res_put(&res);
    free(heap);
    return 0;
}


/* Interleaved tournament: every kind is measured in the same process, in
 * forward and reverse order, so DVFS drift hits all kinds alike.
 * r = {status, bytes_per_kind, reps, chunk_bytes,
 *      best rate*100 per kind 0..7 (8 entries), last rate*100 0..2} */
int mx4_bwt(const unsigned char *srcData, int srcLen, int chunkKb, int reps,
            int rounds, int *r, int rLen)
{
    res_t res;
    uint8_t *heap;
    unsigned bytes, chunk, nchunks, nvec;
    int rc, kind, rnd, idx;
    HVX_Vector acc;
    int best[24], last[24];

    if (rLen < 16) return -1;
    memset(r, 0, sizeof(int) * rLen);
    if (srcLen < (1 << 20) || reps < 1 || reps > 8 || rounds < 1 || rounds > 8)
        { r[0] = -2; return 0; }
    if (chunkKb < 1 || chunkKb > 1024) chunkKb = 32;

    bytes = (unsigned)srcLen & ~127u;
    chunk = (unsigned)chunkKb << 10;
    if (chunk > bytes) chunk = bytes;
    nchunks = bytes / chunk;
    nvec = chunk / 128u;

    heap = (uint8_t *)memalign(4096, bytes + (1u << 18));
    if (!heap) { r[0] = -3; return 0; }
    memset(heap, 3, bytes + (1u << 18));
    (void)srcData;

    rc = res_get(BW2_VTCM, &res);
    r[0] = rc;
    if (rc) { free(heap); return 0; }
    rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (rc) { r[0] = -4; res_put(&res); free(heap); return 0; }

    for (kind = 0; kind < 21; kind++) { best[kind] = 0; last[kind] = 0; }
    acc = Q6_V_vzero();
    for (rnd = 0; rnd < rounds; rnd++) {
        for (idx = 0; idx < 21; idx++) {
            int us, pcyc, rate;
            kind = (rnd & 1) ? (20 - idx) : idx;
            bw2_timed(heap, kind, chunk, nchunks, nvec, res.base, reps, &acc,
                      &us, &pcyc, &rate);
            last[kind] = rate;
            if (rate > best[kind]) best[kind] = rate;
        }
    }
    ((HVX_Vector *)res.base)[0] = acc;

    r[1] = (int)bytes * reps;
    r[2] = reps;
    r[3] = (int)chunk;
    for (kind = 0; kind < 21; kind++) {
        if (28 + kind < rLen) r[28 + kind] = best[kind];
        if (4 + kind < rLen) r[4 + kind] = last[kind];
    }
    qurt_hvx_unlock();
    res_put(&res);
    free(heap);
    return 0;
}


/* ================================================================== */
/* int4 weights: weight.n = mxmem(w, 512-1)                            */
/* ================================================================== */
/* 4-bit tile: 32 K x 32 N = 512 B.  The nibble address is a halved
 * int8 crouton; `wl` selects the candidate permutation. */
static void i4_pack_wgt(uint8_t *dst, const int8_t *B, int N, int K, int wl,
                        unsigned wstride)
{
    int k, n, ncr = (K + 31) / 32;
    memset(dst, 0, (size_t)wstride * (size_t)ncr);
    for (k = 0; k < K; k++)
        for (n = 0; n < N; n++) {
            unsigned g = ((unsigned)k & 31u) >> 2, byte, nib;
            int odd = k & 1;
            if (wl == 4) {          /* verified v68 int4 layout: 8 K per 128 B */
                unsigned bb = wstride * ((unsigned)k >> 5)
                            + 128u * (((unsigned)k & 31u) >> 3)
                            + 4u * (unsigned)n + ((unsigned)k & 3u);
                int hi = ((unsigned)k >> 2) & 1u;
                if (hi) dst[bb] = (uint8_t)((dst[bb] & 0x0Fu) | ((B[n * K + k] & 0x0F) << 4));
                else    dst[bb] = (uint8_t)((dst[bb] & 0xF0u) | (B[n * K + k] & 0x0F));
                continue;
            }
            switch (wl & 3) {
            case 0: byte = 64u * g + 2u * (unsigned)n + (((unsigned)k & 3u) >> 1);
                    nib = (unsigned)(odd == 0); break;
            case 1: byte = 64u * g + 2u * (unsigned)n + (((unsigned)k & 3u) >> 1);
                    nib = (unsigned)(odd != 0); break;
            case 2: byte = 64u * g + (unsigned)n + 32u * (((unsigned)k & 3u) >> 1);
                    nib = (unsigned)(odd == 0); break;
            default: byte = 64u * g + (unsigned)n + 32u * (((unsigned)k & 3u) >> 1);
                    nib = (unsigned)(odd != 0); break;
            }
            byte += wstride * ((unsigned)k >> 5);
            if (nib)
                dst[byte] = (uint8_t)((dst[byte] & 0x0Fu) | ((B[n * K + k] & 0x0F) << 4));
            else
                dst[byte] = (uint8_t)((dst[byte] & 0xF0u) | (B[n * K + k] & 0x0F));
        }
}

static inline void i4_mul(const void *act, uint32_t art,
                          const void *wgt, uint32_t wc, int deep)
{
    if (deep == 2)
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3):deep\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
    else if (deep == 3)
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.n = mxmem(%2,%3):deep\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
    else if (deep)
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1):deep\n\t weight.n = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
    else
        __asm__ __volatile__(
            "{\n\t activation.ub = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
}

/* int4 weights with 16-bit (fp16) activations - QNN native_hmx_a16w4. */
static inline void i4_mul_hf(const void *act, uint32_t art,
                             const void *wgt, uint32_t wc, int deep)
{
    if (deep)
        __asm__ __volatile__(
            "{\n\t activation.hf = mxmem(%0,%1):deep\n\t weight.n = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
    else
        __asm__ __volatile__(
            "{\n\t activation.hf = mxmem(%0,%1)\n\t weight.n = mxmem(%2,%3)\n\t}"
            :: "r"((uint32_t)(uintptr_t)act), "r"(art),
               "r"((uint32_t)(uintptr_t)wgt), "r"(wc) : "memory");
}

/* fp16 activation packing: same spatial/K ordering as the int8 lay=3 crouton
 * but two bytes per element and a 4 KB stride per 32 K. */
static void i4_pack_act_hf(uint8_t *dst, const int8_t *A, int K, int M)
{
    int s, c;
    for (s = 0; s < M && s < 64; s++)
        for (c = 0; c < K; c++) {
            unsigned off = 4096u * (unsigned)(c >> 5)
                         + 128u * (unsigned)(s >> 2)
                         + 4u * (unsigned)(c & 31) + (unsigned)(s & 3);
            __fp16 v = (__fp16)A[s * K + c];
            memcpy(dst + 2u * off, &v, 2);
        }
}

/* H16: 16-K activation crouton = 1 KB (32 spatial x 16 K). */
static void i4_pack_act16(uint8_t *dst, const int8_t *A, int K, int M)
{
    int s, c;
    for (s = 0; s < M && s < 64; s++)
        for (c = 0; c < K; c++) {
            unsigned off = 1024u * (unsigned)(c >> 4)
                         + 128u * (unsigned)(s >> 2)
                         + 4u * (unsigned)(c & 15) + (unsigned)(s & 3);
            dst[off] = (uint8_t)A[s * K + c];
        }
}

/* A: M x K uint8 row major, B: N x K int8 row major with values in [-8,7].
 *   wl  : nibble permutation candidate 0..3
 *   bs  : bias/scale low word (0x6000 = int8 identity)
 *   mode bit0: :deep on the activation (chunked), bit1: skip power/lock
 *              bit2: no bias load, bit3: M=64 spatial
 *   raw = 2 KB output tile, info = {status, us, pcyc, ncr, art, wc, 0,0,0,0,
 *   vtcm, bs, wl, 0} */
int mx4_i4(const unsigned char *Adata, int ALen,
           const unsigned char *Bdata, int BLen,
           int M, int N, int K, int wl, int art, int wc, int bs,
           int mode, int reps, int astep, int wstep, int aoff, int woff,
           unsigned char *raw, int rawLen,
           int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    const int8_t *A = (const int8_t *)Adata;
    const int8_t *B = (const int8_t *)Bdata;
    int ncr = (K + 31) / 32, rc, i, rep, c, ch;
    unsigned aregion, wregion, need, wstride, aspan;
    int estep, nmul, loopdeep;
    uint32_t eart;
    uint64_t t0, t1, c0, c1;
    int hvx_ok = 0;

    if (infoLen < 14) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    if (K < 32 || K > 2048 || (K & 31)) return -2;
    if (N < 1 || N > 32) return -3;
    if (M < 1 || M > 64) return -3;
    if (ALen < M * K || BLen < N * K) return -4;
    if (rawLen < 2048) return -5;
    if (reps < 1 || reps > 20000) reps = 1;
    if (wc <= 0) wc = (wl == 4) ? 511 : 512 * ncr - 1;
    M = (mode & 8) ? 64 : 32;

    wstride = (mode & 4) ? 1024u : (mode & 16) ? 2048u : 512u;

    /* --- probe controls (REPORT21).  mode=0 is the fixed int4 path. --- */
    eart = (uint32_t)art;
    estep  = (mode & 1024)    ? 4096            /* H1: two act croutons/insn */
           : (mode & 16384)   ? 0               /* same act every insn */
           : (mode & 65536)   ? 1024            /* half act crouton */
           : 2048;
    if (astep > 0) estep = astep;
    if (wstep < 0) {
        wstep = (mode & 32768)   ? 0            /* same weight every insn */
              : (mode & 131072)  ? (int)wstride / 2
              : (int)wstride;
    }
    nmul   = (mode & 4096)    ? 1 : ncr;        /* H: one instruction only */
    loopdeep = (mode & 262144) ? 2 : (mode & 524288) ? 3 : 0;
    if (mode & 2097152) {                       /* H16: 16-K croutons */
        wstride = 256u;
        if (astep <= 0) estep = 1024;
        if (wstep < 0) wstep = 256;
        if (!(mode & 4096)) nmul = 2 * ncr;
    }
    if (mode & 1048576) {                       /* native_hmx_a16w4: hf acts */
        if (astep <= 0) estep = 4096;
    }

    aregion = ((mode & 1048576) ? 4096u : 2048u) * (unsigned)ncr;
    aspan = aregion;
    if (nmul > 1) {
        unsigned need_a = (unsigned)(nmul - 1) * (unsigned)(estep > 0 ? estep : 0) + 2048u;
        if (need_a > aspan) aspan = need_a;
    }
    wregion = wstride * (unsigned)ncr;
    if (mode & 2097152) wregion += 512u;   /* 16-K probe: spare room */
    if (nmul > 1 && wstep > 0) {
        unsigned need_w = (unsigned)(nmul - 1) * (unsigned)wstep + wstride;
        if (need_w > wregion) wregion = need_w;
    }
    need = aspan + align_up(wregion, 2048u) + 8192u + 512u + 2048u;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    info[10] = (int)r.size;
    vact = r.base;
    vwgt = vact + aspan;
    vout = vact + align_up(aspan + wregion, 2048u);
    vbias = vout + 4096u;
    memset(r.base, 0, need);
    if (mode & 1048576)      i4_pack_act_hf(vact, A, K, M);
    else if (mode & 2097152) i4_pack_act16(vact, A, K, M);
    else                     i8_pack_act(vact, A, K, (mode & 128) ? 4 : 3, M);
    i4_pack_wgt(vwgt, B, N, K, wl, wstride);
    if (bs) {
        uint32_t *bl = (uint32_t *)vbias;
        for (i = 0; i < 128; i++) bl[i] = 0;
        for (i = 0; i < 32; i++) { bl[i] = (uint32_t)bs; bl[64 + i] = (uint32_t)bs; }
    }

    info[3] = ncr; info[4] = art; info[5] = wc;
    info[11] = bs; info[12] = wl;
    if (!(mode & 2)) {
        info[8] = qurt_hvx_lock(QURT_HVX_MODE_128B);
        hvx_ok = (info[8] == 0);
        info[6] = hmx_power(1);
        info[7] = compute_resource_hmx_lock ? compute_resource_hmx_lock(r.ctx) : -100;
    }
    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < reps; rep++) {
        hmx_clr();
        if (mode & 64) {
            i4_mul(vact, ((uint32_t)(ncr - 1) << 11) | 0x3FFu,
                   vwgt, (uint32_t)wstride * (uint32_t)ncr - 1u, 2);
        } else if (mode & 32) {
            i4_mul(vact, ((uint32_t)(ncr - 1) << 11) | 0x3FFu,
                   vwgt, (uint32_t)wstride * (uint32_t)ncr - 1u, 3);
        } else if (mode & 1) {
            for (c = 0; c < ncr; c += 32) {
                ch = (ncr - c) > 32 ? 32 : (ncr - c);
                i4_mul(vact + (unsigned)c * 2048u,
                       ((uint32_t)(ch - 1) << 11) | 0x3FFu,
                       vwgt + (unsigned)c * wstride,
                       (uint32_t)ch * wstride - 1u, 1);
            }
        } else {
            for (c = 0; c < nmul; c++) {
                uint32_t a = (mode & 8192)
                    ? ((((uint32_t)c & 0x1Fu) << 11) | 0x3FFu) : eart;
                const void *ap = vact + (unsigned)aoff
                               + (unsigned)c * (unsigned)estep;
                const void *wp = vwgt + (unsigned)woff
                               + (unsigned)c * (unsigned)wstep;
                if (mode & 1048576) i4_mul_hf(ap, a, wp, (uint32_t)wc, loopdeep);
                else                i4_mul(ap, a, wp, (uint32_t)wc, loopdeep);
            }
        }
        if (bs) hmx_bias(vbias);
        i8_store(vout, 0, (mode & 2048) ? 3 : 0);
        if (mode & 512) {
            if (bs) hmx_bias(vbias + 256);
            i8_store(vout + 2048, 0, (mode & 2048) ? 3 : 0);
        }
    }
    hmx_sync();
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    info[1] = (int)(t1 - t0);
    info[2] = (int)((c1 - c0) / (uint64_t)reps);
    if (!(mode & 2)) {
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        if (hvx_ok) qurt_hvx_unlock();
    }
    for (i = 0; i < 4096 && i < rawLen; i++) raw[i] = vout[i];
    res_put(&r);
    info[0] = 0;
    return 0;
}

/* int4 impulse map: one nibble at a time at physical byte nb of the 512 B
 * weight tile.  A[s][k] = k+1, so a responding output column n carrying the
 * value v decodes to (k = v-1, n).  map[nb] = (k<<5)|n, 0xFF = no response,
 * 0xFE = response outside 1..32.  info = {status, max_v, hits, nibhi, 0,0,0,0} */
int mx4_i4map(int nibhi, int art, int wc, int bs, int amode, int k0,
              unsigned char *map, int mapLen, int *info, int infoLen)
{
    res_t r;
    uint8_t *vact, *vwgt, *vout, *vbias;
    int rc, nb, n, i, hits = 0, maxv = 0, maxhit = 0;
    int8_t A[32 * 32];
    unsigned need = 2048u + 4096u + 4096u + 512u + 2048u;

    if (infoLen < 8 || mapLen < 512) return -1;
    memset(info, 0, sizeof(int) * infoLen);
    memset(map, 0xFF, 512);
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    vact  = r.base;
    vwgt  = vact + 2048u;
    vout  = vwgt + 4096u;
    vbias = vout + 4096u;
    memset(r.base, 0, need);

    for (i = 0; i < 32; i++)
        for (n = 0; n < 32; n++) {
            if (amode == 0 || amode == 4 || amode == 5 || amode == 6 ||
                amode == 7 || amode == 8 || amode == 9 || amode == 10)
                A[i * 32 + n] = 1;
            else if (amode == 1)
                A[i * 32 + n] = (int8_t)(n + 1);
            else
                A[i * 32 + n] = (int8_t)(n == k0 ? 1 : 0);
        }
    i8_pack_act(vact, A, 32, 3, 32);
    {
        uint32_t *bl = (uint32_t *)vbias;
        for (i = 0; i < 128; i++) bl[i] = 0;
        for (i = 0; i < 32; i++) { bl[i] = (uint32_t)bs; bl[64 + i] = (uint32_t)bs; }
    }

    info[6] = qurt_hvx_lock(QURT_HVX_MODE_128B);
    info[7] = hmx_power(1);
    info[5] = compute_resource_hmx_lock ? compute_resource_hmx_lock(r.ctx) : -100;

    if (amode == 5) {
        /* one nibble at byte k0, A = ones: dump out[s][0] for s = 0..63 */
        memset(vwgt, 0, 2048);
        vwgt[k0 & 511] = nibhi ? 0x10u : 0x01u;
        memset(vout, 0, 2048);
        hmx_clr();
        i4_mul(vact, (uint32_t)art, vwgt, (uint32_t)wc, 0);
        hmx_bias(vbias);
        i8_store(vout, 0, 0);
        hmx_sync();
        for (nb = 0; nb < 64; nb++)
            map[nb] = vout[(unsigned)(nb >> 2) * 128u + (unsigned)(nb & 3)];
        info[1] = 0;
        for (nb = 0; nb < 2048; nb++) if (vout[nb]) info[1]++;
        if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
        hmx_power(0);
        qurt_hvx_unlock();
        res_put(&r);
        info[0] = 0;
        return 0;
    }

    for (nb = 0; nb < 512; nb++) {
        int nresp = 0, first = -1;
        memset(vwgt, 0, 2048);
        if (amode == 8) {
            memset(vwgt, 0x01, 2048);          /* low nibbles only */
        } else if (amode == 9) {
            memset(vwgt, 0x10, 2048);          /* high nibbles only */
        } else if (amode == 10) {
            memset(vwgt, 0x11, 2048);          /* both nibbles */
        } else if (amode == 7) {
            vwgt[0] = 0x01u; vwgt[128] = (unsigned)(k0 & 15);
        } else if (amode == 4) {
            memset(vwgt, nibhi ? ((k0 & 15) << 4) : (k0 & 15), 2048);
        } else if (amode == 6) {
            memset(vwgt, nibhi ? ((k0 & 15) << 4) : (k0 & 15), 2048);
            memset(vwgt, 0, (k0 & 3) * 128u);
            memset(vwgt + (unsigned)((k0 & 3) + 1) * 128u, 0,
                   512u - (unsigned)((k0 & 3) + 1) * 128u);
        } else if (amode == 3) {
            vwgt[nb] = nibhi ? (unsigned)((k0 & 15) << 4) : (unsigned)(k0 & 15);
        } else {
            vwgt[nb] = nibhi ? 0x10u : 0x01u;
        }
        memset(vout, 0, 2048);
        hmx_clr();
        i4_mul(vact, (uint32_t)art, vwgt, (uint32_t)wc, 0);
        hmx_bias(vbias);
        i8_store(vout, 0, 0);
        hmx_sync();
        for (n = 0; n < 32; n++) {
            unsigned v = vout[4u * (unsigned)n];
            if (v) {
                if ((int)v > maxv) maxv = (int)v;
                nresp++;
                if (first < 0) first = n;
            }
        }
        if (nresp > maxhit) maxhit = nresp;
        if (nresp) { map[nb] = (unsigned char)first; hits++; }
        if (amode == 6 || (amode >= 7 && amode <= 10)) break;
    }
    if (compute_resource_hmx_unlock) compute_resource_hmx_unlock(r.ctx);
    hmx_power(0);
    qurt_hvx_unlock();
    res_put(&r);
    info[1] = maxv;
    info[2] = hits;
    info[3] = nibhi;
    info[4] = maxhit;
    info[5] = k0;
    info[0] = 0;
    return 0;
}

int mx4_ping0(int x, int *r, int rLen)
{
    if (rLen < 1) return -1;
    r[0] = x + 1;
    return 0;
}

int mx4_pingbuf(const unsigned char *data, int len, int *r, int rLen)
{
    if (rLen < 2) return -1;
    r[0] = len;
    r[1] = (len > 0) ? (int)data[0] + (int)data[len - 1] : 0;
    return 0;
}

/* pcycles vs microseconds: what the DSP cycle counter really runs at */
int mx4_clk(int *r, int rLen)
{
    uint64_t t0, t1, c0, c1;
    int i;
    volatile uint32_t x = 1;

    if (rLen < 3) return -1;
    t0 = HAP_perf_get_time_us();
    c0 = HAP_perf_get_pcycles();
    for (i = 0; i < 300000; i++)
        x = x * 1664525u + 1013904223u;
    c1 = HAP_perf_get_pcycles();
    t1 = HAP_perf_get_time_us();
    r[0] = (int)(c1 - c0);
    r[1] = (int)(t1 - t0);
    r[2] = (t1 > t0) ? (int)((c1 - c0) / (t1 - t0)) : 0;
    return 0;
}
