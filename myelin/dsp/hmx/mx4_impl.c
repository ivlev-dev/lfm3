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
            int mul, int bscale, int mode, int reps,
            unsigned char *raw, int rawLen, int *info, int infoLen)
{
    return i8_core((const int8_t *)Adata, ALen, (const int8_t *)Bdata, BLen,
                   M, N, K, lay, wl, art, wc, srt, mul, 0, bscale, mode,
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
    need = aregion + wregion + 4096u + 512u + 2048u;
    rc = res_get(need, &r);
    info[0] = rc;
    if (rc) return 0;
    vact = r.base;
    vwgt = vact + aregion;
    vout = vwgt + wregion;
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
