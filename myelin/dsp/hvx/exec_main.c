/*
 * Host stand for the one-RPC token pipeline (`exec`).
 *
 * The token graph is published once as a versioned command list in ION and a
 * single FastRPC call walks it on the DSP: 24 FFN + 11 attention + 13
 * shortconv layers, intermediate activations in one DSP-heap scratch.
 *
 * Checks (default run):
 *   1. bit-exactness, one op per RPC vs the same op inside `exec`:
 *        FFN layer   : lfmgemv_ffn(...)        vs exec(FFN)
 *        attention   : lfmgemv_attn(...)       vs exec(ATTN)
 *        shortconv   : host ring model          vs exec(CONV)
 *        quantise    : host int64 formula      vs exec(QUANT)
 *   2. weight handles: 24 layers x 3 tensors = 72 simultaneously mapped views
 *      of ONE arena, each verified against the host int4 dot (raw int32 GEMV).
 *   3. full graph: 85 commands (24 FFN + 11 ATTN + 11 QUANT + 13 x
 *      (in_proj GEMV + CONV + out_proj GEMV)), real LFM3 shapes, weights in
 *      ION.  Run (a) as one exec per token and (b) one exec per command
 *      (85 round trips).  Both must produce the same final int8 (0 mismatch).
 *      Prints DSP microseconds, wall and calls per token for both.
 *
 * Run on device: ./exec_test [tokens]        (default 5 tokens, full graph)
 *                ./exec_test 5 small        (one shared FFN layer, ~130 MB)
 *
 * Exit code: 0 pass, 2 bit-exactness failure, 10.. RPC failure.
 * Build: see build.sh.
 */
#include "lfmgemv.h"
#include "ffn_ref.h"
#include "../conv/conv_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include "rpcmem.h"

#define HID       2048
#define FFN_W     7680
#define NL        24
#define N_ATTN    11
#define N_CONV    13

#define HQ        16
#define HKV       4
#define HD        128
#define CTX       2048            /* attention context for the timing graph */
#define PAGES     (CTX / 32)

#define CONV_TAPS 7
#define CONV_RING 8

/* ---- exec ABI mirror (must match lfmgemv_impl.c) ---- */
#define LFM_EXEC_MAGIC 0x454D464Cu
#define LFM_EXEC_VERSION 1u
#define LFM_CMD_MAX 256
enum { LFM_CMD_QUANT = 0, LFM_CMD_GEMV = 1, LFM_CMD_FFN = 2,
       LFM_CMD_ATTN = 3, LFM_CMD_CONV = 4 };
enum { LFM_SRC_SCRATCH = 0, LFM_SRC_OUT = 1 };
enum { LFM_DST_SCRATCH = 0, LFM_DST_OUT = 1, LFM_DST_ATTACHED = 2 };

typedef struct {
    uint32_t magic, version, n_cmds, cmd_size, list_bytes, scratch_bytes;
    uint32_t flags, hdr_size;
    uint64_t reserved[4];
} exec_hdr_t;

typedef struct {
    uint32_t type, src, dst, pad;
    int32_t h0, h1, h2, h3, dst_h;
    int32_t src_off, src_len, dst_off, dst_len;
    int32_t tbl_h, tbl_len, tbl_off;
    int32_t a0, a1, a2, a3, rt_base;
    int32_t reserved[3];
} exec_cmd_t;

typedef char exec_hdr_size_check[(sizeof(exec_hdr_t) == 64) ? 1 : -1];
typedef char exec_cmd_size_check[(sizeof(exec_cmd_t) == 96) ? 1 : -1];

/* ---- scratch layout (byte offsets, 128 B aligned) ---- */
#define S_ACT     0
#define S_Q       4096
#define S_ATTN32  8192                    /* 16*128*4 */
#define S_Y8      16384                   /* HID */
#define S_BCX     20480                   /* 3*HID */
#define S_CY      32768                   /* HID */
#define S_STATE   34816                   /* 14 * 2*HID*8 */
#define SCRATCH   (S_STATE + 16 * 2 * HID * 8 + 4096)

/* ---- command buffer layout (one small ION mapping) ---- */
#define OFF_FFN    256
#define OFF_ATTN   1024
#define OFF_CONV   2048
#define OFF_QUANT  3072
#define OFF_ALL    4096
#define OFF_SINGLE 16384
#define SINGLE_STRIDE 160
#define CMD_BYTES  (65536)

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                  (uint32_t)sizeof(um));
}

/* ---------------- host ION buffer ---------------- */

typedef struct {
    void *p;
    int   fd;
    int   bytes;
} ionbuf_t;

static int ion_alloc(ionbuf_t *b, size_t bytes)
{
    size_t n = (bytes + 4095) & ~(size_t)4095;
    b->p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n);
    if (!b->p) { b->fd = -1; b->bytes = 0; return -1; }
    b->fd = rpcmem_to_fd(b->p);
    b->bytes = (int)n;
    return 0;
}

static void ion_free(ionbuf_t *b)
{
    if (b->p) rpcmem_free(b->p);
    b->p = 0; b->fd = -1; b->bytes = 0;
}

/* ---------------- arena builder ---------------- */

typedef struct {
    uint8_t *base;
    size_t   bytes;
    size_t   off;
} arena_build_t;

static size_t ab_take(arena_build_t *a, size_t bytes)
{
    size_t at = (a->off + 127) & ~(size_t)127;
    if (at + bytes > a->bytes) { printf("arena overflow\n"); exit(1); }
    a->off = at + bytes;
    return at;
}

static void ab_put(arena_build_t *a, size_t off, const void *src, size_t bytes)
{
    memcpy(a->base + off, src, bytes);
}

static void fill_i4(uint8_t *dst, size_t bytes, uint32_t seed)
{
    size_t i;
    for (i = 0; i < bytes; i++) {
        seed = seed * 1103515245u + 12345u;
        dst[i] = (uint8_t)(seed >> 24);
    }
}

static void fill_i8(int8_t *dst, size_t n, uint32_t seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        dst[i] = (int8_t)((seed >> 24) % 255 - 127);
    }
}

static int dma_sync(int fd, uint64_t flags)
{
    struct dma_buf_sync arg;
    arg.flags = flags;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &arg);
}

/* ---------------- weight setup helper ---------------- */

static int set_scale_of(int h, int N, int K, float sa, float so, const float *sw)
{
    unsigned char *sbuf = (unsigned char *)malloc(16 + (size_t)N * sizeof(float));
    int rc;
    if (!sbuf) return AEE_ENOMEMORY;
    memcpy(sbuf + 0, &N, 4);
    memcpy(sbuf + 4, &K, 4);
    memcpy(sbuf + 8, &sa, 4);
    memcpy(sbuf + 12, &so, 4);
    memcpy(sbuf + 16, sw, (size_t)N * sizeof(float));
    rc = lfmgemv_set_scale(h, sbuf, 16 + N * (int)sizeof(float));
    free(sbuf);
    return rc;
}

/* ---------------- host int4 reference (same recipe as the DSP) ---------------- */

static int32_t host_i4_dot(const uint8_t *row, const int8_t *act, int K)
{
    int32_t acc = 0;
    int k;
    for (k = 0; k < K; k++) {
        uint8_t byte = row[k >> 1];
        int u = (k & 1) ? (int)(byte >> 4) : (int)(byte & 0xf);
        acc += u * (int32_t)act[k];
    }
    return acc - 8 * (int32_t)ffn_sum_i8(act, K);
}

/* ---------------- command lists ---------------- */

typedef struct {
    exec_hdr_t *hdr;
    exec_cmd_t *cmd;
    int         n;
} list_t;

static void list_init(uint8_t *buf, size_t off, int n, list_t *l)
{
    size_t bytes = sizeof(exec_hdr_t) + (size_t)n * sizeof(exec_cmd_t);
    l->hdr = (exec_hdr_t *)(buf + off);
    l->cmd = (exec_cmd_t *)(buf + off + sizeof(exec_hdr_t));
    l->n = n;
    memset(l->hdr, 0, bytes);
    l->hdr->magic = LFM_EXEC_MAGIC;
    l->hdr->version = LFM_EXEC_VERSION;
    l->hdr->n_cmds = (uint32_t)n;
    l->hdr->cmd_size = sizeof(exec_cmd_t);
    l->hdr->list_bytes = (uint32_t)bytes;
    l->hdr->scratch_bytes = SCRATCH;
    l->hdr->hdr_size = sizeof(exec_hdr_t);
}

static exec_cmd_t *cmd_at(list_t *l, int index)
{
    exec_cmd_t *c = &l->cmd[index];
    memset(c, 0, sizeof(*c));
    c->h0 = c->h1 = c->h2 = c->h3 = -1;
    c->dst_h = -1;
    c->tbl_h = -1;
    c->rt_base = -1;
    return c;
}

static void mk_quant(exec_cmd_t *c, int32_t src_off, int32_t dst_off, int32_t n,
                     int32_t m, int32_t dst_kind)
{
    c->type = LFM_CMD_QUANT;
    c->src = LFM_SRC_SCRATCH; c->dst = dst_kind;
    c->src_off = src_off; c->src_len = n * 4;
    c->dst_off = dst_off; c->dst_len = n;
    c->a0 = m;
}

static void mk_gemv(exec_cmd_t *c, int32_t h, int32_t src_off, int32_t dst_off,
                    int32_t dst_kind, int32_t dst_len)
{
    c->type = LFM_CMD_GEMV;
    c->src = LFM_SRC_SCRATCH; c->dst = dst_kind;
    c->h0 = h; c->a0 = 1; c->a1 = 1;
    c->src_off = src_off; c->src_len = HID;
    c->dst_off = dst_off; c->dst_len = dst_len;
}

static void mk_ffn(exec_cmd_t *c, int32_t hg, int32_t hu, int32_t hd,
                   int32_t src_off, int32_t dst_kind, int32_t dst_off)
{
    c->type = LFM_CMD_FFN;
    c->src = LFM_SRC_SCRATCH; c->dst = dst_kind;
    c->h0 = hg; c->h1 = hu; c->h2 = hd;
    c->src_off = src_off; c->src_len = HID;
    c->dst_off = dst_off; c->dst_len = HID;
}

static void mk_attn(exec_cmd_t *c, int32_t hk, int32_t hkt, int32_t hv,
                    int32_t tbl_h, int32_t ntok, int32_t rt_base,
                    int32_t src_off, int32_t dst_off, int32_t dst_kind)
{
    c->type = LFM_CMD_ATTN;
    c->src = LFM_SRC_SCRATCH; c->dst = dst_kind;
    c->h0 = hk; c->h1 = hkt; c->h2 = hv; c->h3 = 0;
    c->src_off = src_off; c->src_len = HQ * HD;
    c->dst_off = dst_off; c->dst_len = HQ * HD * 4;
    c->tbl_h = tbl_h; c->tbl_len = ntok;
    c->rt_base = rt_base;
}

static void mk_conv(exec_cmd_t *c, int32_t blob, int32_t H, int32_t ksh,
                    int32_t state_off, int32_t src_off, int32_t dst_off,
                    int32_t dst_kind)
{
    c->type = LFM_CMD_CONV;
    c->src = LFM_SRC_SCRATCH; c->dst = dst_kind;
    c->h0 = blob;
    c->src_off = src_off; c->src_len = 3 * H;
    c->dst_off = dst_off; c->dst_len = H;
    c->tbl_off = state_off;
    c->a0 = H; c->a1 = ksh; c->a2 = 0; c->a3 = 100;
}

static int exec_list(int ah_cmd, size_t off, const list_t *l, const int8_t *act,
                     int act_len, const int32_t *rt, int rt_n,
                     unsigned char *out, int out_len, int *times)
{
    return lfmgemv_exec(ah_cmd, (int)off, l->n, (const unsigned char *)act, act_len,
                        (const int *)rt, rt_n, out, out_len, times, 8);
}

/* ---------------- shortconv blob builder (mirrors ../conv/conv_init) ---------------- */

typedef struct {
    int8_t *blob;      /* 8*H*8 rotated weights + kq[H] */
    int8_t *wq1;       /* [H][7] unrotated, for the host reference */
    int     H, ksh;
} conv_blob_t;

static int conv_blob_build(conv_blob_t *b, int H, float s_bc, float s_y,
                           const float *wf)
{
    double kmax = 0.0;
    int ch, j, s, p, g;

    b->H = H;
    b->blob = (int8_t *)malloc((size_t)8 * H * 8 + (size_t)4 * H);
    b->wq1 = (int8_t *)malloc((size_t)H * CONV_TAPS);
    if (!b->blob || !b->wq1) return -1;
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * CONV_TAPS;
        float m = 0.0f, sw;
        for (j = 0; j < CONV_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        sw = m / 127.0f;
        for (j = 0; j < CONV_TAPS; j++)
            b->wq1[(size_t)ch * CONV_TAPS + j] = (int8_t)conv_wq(w[j], sw);
        {
            double K = conv_K((double)m, (double)s_bc, (double)s_y);
            if (K > kmax) kmax = K;
        }
    }
    b->ksh = conv_pick_ksh(kmax);
    for (ch = 0; ch < H; ch++) {
        const float *w = wf + (size_t)ch * CONV_TAPS;
        float m = 0.0f;
        for (j = 0; j < CONV_TAPS; j++) {
            float a = w[j] < 0.0f ? -w[j] : w[j];
            if (a > m) m = a;
        }
        if (!(m > 0.0f)) m = 1.0f;
        ((int32_t *)(b->blob + (size_t)8 * H * 8))[ch] =
            conv_make_kq(conv_K((double)m, (double)s_bc, (double)s_y), b->ksh);
    }
    for (p = 0; p < CONV_RING; p++) {
        for (g = 0; g < H / 16; g++) {
            for (j = 0; j < 16; j++) {
                int c = g * 16 + j;
                int8_t *dst = b->blob + ((size_t)p * (H / 16) + g) * 128 + (size_t)j * 8;
                for (s = 0; s < CONV_RING; s++) {
                    int tap = conv_slot_tap(p, s);
                    dst[s] = (tap < CONV_TAPS) ? b->wq1[(size_t)c * CONV_TAPS + tap] : 0;
                }
            }
        }
    }
    return 0;
}

/* Host ring model on a zero-initialised history: only tap 6 (current token)
 * is non-zero, every other ring position stays 0, exactly as the fresh DSP
 * scratch state.  Same operations as xconv_bx_write/dot/pack_y. */
static void conv_ref_first(const conv_blob_t *b, const int8_t *bcx, int8_t *y)
{
    int H = b->H, ch;
    for (ch = 0; ch < H; ch++) {
        int32_t out32 = (int32_t)b->wq1[(size_t)ch * CONV_TAPS + CONV_TAPS - 1] *
                        (int32_t)((int32_t)bcx[ch] * (int32_t)bcx[2 * H + ch]);
        y[ch] = conv_glue(out32, bcx[H + ch],
                          ((const int32_t *)(b->blob + (size_t)8 * H * 8))[ch],
                          b->ksh);
    }
}

/* ---------------- main ---------------- */

typedef struct {
    int ffn[NL][3];
    int k[N_ATTN], v[N_ATTN], kt[N_ATTN], tbl[N_ATTN];
    int ci[N_CONV], co[N_CONV], cb[N_CONV];
} handles_t;

int main(int argc, char **argv)
{
    static ionbuf_t arena, cmd_ion;
    static handles_t H;
    static list_t all, one_ffn, one_attn, one_conv, one_quant;
    static list_t singles[LFM_CMD_MAX];
    static size_t single_off[LFM_CMD_MAX];
    int32_t rt[4];
    int ah_model = -1, ah_cmd = -1, rc, i, l, k, bad, c;
    int tokens = (argc > 1) ? atoi(argv[1]) : 5;
    int small = (argc > 2 && strcmp(argv[2], "small") == 0);
    int times[8];
    int failures = 0;
    int n_ffn_layers;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (tokens < 1) tokens = 1;
    if (tokens > 12) tokens = 12;

    const size_t ffn_row_g = (size_t)FFN_W * (HID / 2);     /* [FFN_W][HID] */
    const size_t ffn_row_d = (size_t)HID * (FFN_W / 2);     /* [HID][FFN_W] */
    const size_t kv_k  = (size_t)HKV * CTX * 128;
    const size_t kv_v  = (size_t)HKV * CTX * 128;
    const size_t kv_kt = (size_t)HKV * CTX * 4;
    const size_t conv_in  = (size_t)(3 * HID) * (HID / 2);
    const size_t conv_out = (size_t)HID * (HID / 2);
    const size_t conv_blob_sz = (size_t)8 * HID * 8 + (size_t)4 * HID;

    n_ffn_layers = small ? 1 : NL;

    /* view table, allocated before the arena is sized */
    typedef struct { size_t off; int bytes; } view_t;
    static view_t v_ffn[NL][3];
    static view_t v_k[N_ATTN], v_v[N_ATTN], v_kt[N_ATTN], v_tbl[N_ATTN];
    static view_t v_ci[N_CONV], v_co[N_CONV], v_cb[N_CONV];
    static view_t v_small[72];
    static conv_blob_t blobs[N_CONV];

    size_t need = 0;
    need += 128;
    need += (size_t)n_ffn_layers * 3 * ffn_row_g;
    need += (size_t)N_ATTN * (kv_k + kv_v + kv_kt) + 512;
    need += (size_t)N_CONV * (conv_in + conv_out + conv_blob_sz);
    need += 72 * (size_t)256 * 128;
    need += 1 << 20;                             /* alignment + slack */

    printf("exec stand: tokens=%d %s, arena %.1f MiB\n", tokens,
           small ? "(shared FFN layer)" : "(24 distinct FFN layers)",
           (double)need / (1024.0 * 1024.0));
    if (enable_unsigned() != 0) { printf("unsigned PD failed\n"); return 10; }
    if (ion_alloc(&arena, need) != 0) { printf("ION alloc failed\n"); return 1; }
    memset(arena.p, 0x88, (size_t)arena.bytes);  /* valid int4 default */
    arena_build_t ab = { (uint8_t *)arena.p, (size_t)arena.bytes, 128 };

    float *swbuf = (float *)malloc((size_t)3 * HID * sizeof(float));
    float *cwf = (float *)malloc((size_t)HID * CONV_TAPS * sizeof(float));
    int8_t *act = (int8_t *)malloc(HID);
    int8_t *dst_ref = (int8_t *)malloc(HID);
    int8_t *out_all = (int8_t *)malloc(HID);
    int8_t *out_one = (int8_t *)malloc(HID);
    int8_t *out_exec = (int8_t *)malloc(HID);
    int32_t *tbl = (int32_t *)malloc((size_t)PAGES * 4);
    int32_t *attn_ref = (int32_t *)malloc((size_t)HQ * HD * 4);
    int32_t *attn_dev = (int32_t *)malloc((size_t)HQ * HD * 4);
    int8_t *q = (int8_t *)malloc(HQ * HD);
    int8_t *bcx = (int8_t *)malloc(3 * HID);
    int8_t *y_ref = (int8_t *)malloc(HID);
    int8_t *y_dev = (int8_t *)malloc(HID);
    int8_t *quant_dev = (int8_t *)malloc(HQ * HD);
    int8_t *quant_src = (int8_t *)malloc(HQ * HD * 4);
    uint32_t seed = 0x1234567u;

    if (!swbuf || !cwf || !act || !dst_ref || !out_all || !out_one || !out_exec ||
        !tbl || !attn_ref || !attn_dev || !q || !bcx || !y_ref || !y_dev ||
        !quant_dev || !quant_src) { printf("host alloc failed\n"); return 1; }

    /* ---------- 1. lay out and fill the model arena (before mapping) -------- */
    for (l = 0; l < n_ffn_layers; l++)
        for (k = 0; k < 3; k++) {
            size_t sz = (k == 2) ? ffn_row_d : ffn_row_g;
            v_ffn[l][k].off = ab_take(&ab, sz);
            v_ffn[l][k].bytes = (int)sz;
        }
    for (l = 0; l < N_ATTN; l++) {
        v_k[l].off = ab_take(&ab, kv_k);   v_k[l].bytes = (int)kv_k;
        v_v[l].off = ab_take(&ab, kv_v);   v_v[l].bytes = (int)kv_v;
        v_kt[l].off = ab_take(&ab, kv_kt); v_kt[l].bytes = (int)kv_kt;
        v_tbl[l].off = ab_take(&ab, (size_t)PAGES * 4);
        v_tbl[l].bytes = PAGES * 4;
    }
    for (l = 0; l < N_CONV; l++) {
        v_ci[l].off = ab_take(&ab, conv_in);      v_ci[l].bytes = (int)conv_in;
        v_co[l].off = ab_take(&ab, conv_out);     v_co[l].bytes = (int)conv_out;
        v_cb[l].off = ab_take(&ab, conv_blob_sz); v_cb[l].bytes = (int)conv_blob_sz;
    }
    for (i = 0; i < 72; i++) {
        v_small[i].off = ab_take(&ab, (size_t)256 * 128);
        v_small[i].bytes = 256 * 128;
    }

    {
        uint8_t *base = (uint8_t *)arena.p;
        for (l = 0; l < N_ATTN; l++) {
            fill_i4(base + v_k[l].off, kv_k, seed ^ (uint32_t)l);
            fill_i8((int8_t *)(base + v_v[l].off), kv_v, seed + 77u + (uint32_t)l);
            for (i = 0; i < HKV * CTX; i++)
                ((int32_t *)(base + v_kt[l].off))[i] = 30000 + (int32_t)(i % 4096);
            for (i = 0; i < PAGES; i++) ((int32_t *)(base + v_tbl[l].off))[i] = i;
        }
        fill_i8(act, HID, seed);
        fill_i8(q, HQ * HD, seed ^ 0x55u);
        for (i = 0; i < HQ * HD; i++)
            ((int32_t *)quant_src)[i] = (int32_t)((i * 2654435761u) & 0x3ffff) - 0x1ffff;
        fill_i8(bcx, 3 * HID, 0x77u);
        for (l = 0; l < N_CONV; l++) {
            for (i = 0; i < HID * CONV_TAPS; i++) {
                seed = seed * 1103515245u + 12345u;
                cwf[i] = ((float)((int)(seed >> 20) % 2001) - 1000.0f) * 1.0e-3f;
            }
            if (conv_blob_build(&blobs[l], HID, 0.01f, 0.05f, cwf) != 0) {
                printf("conv blob failed\n"); return 1;
            }
            ab_put(&ab, v_cb[l].off, blobs[l].blob, conv_blob_sz);
        }
        /* real int4 for layer 0 (the FFN bit-exactness check) */
        fill_i4(base + v_ffn[0][0].off, ffn_row_g, 0xa1u);
        fill_i4(base + v_ffn[0][1].off, ffn_row_g, 0xa2u);
        fill_i4(base + v_ffn[0][2].off, ffn_row_d, 0xa3u);
    }

    /* ---------- 2. map once, register views and shapes ---------- */
    if ((rc = lfmgemv_map_arena(arena.fd, arena.bytes, &ah_model)) != 0) {
        printf("map_arena rc=%d (%.1f MiB)\n", rc, (double)arena.bytes / 1048576.0);
        return 10;
    }
    printf("arena %d: %.1f MiB mapped once\n", ah_model, (double)arena.bytes / 1048576.0);

    {
        int h_ffn[NL][3];
        for (l = 0; l < n_ffn_layers; l++) {
            for (k = 0; k < 3; k++) {
                if ((rc = lfmgemv_alloc_view(ah_model, (int)v_ffn[l][k].off,
                                             v_ffn[l][k].bytes, &h_ffn[l][k])) != 0) {
                    printf("view ffn rc=%d\n", rc); return 10;
                }
                if (k == 2) {
                    if ((rc = lfmgemv_set_shape(h_ffn[l][k], HID, FFN_W, 1)) != 0) {
                        printf("shape down rc=%d\n", rc); return 10;
                    }
                    for (i = 0; i < HID; i++) swbuf[i] = 1.1e-3f;
                    if ((rc = set_scale_of(h_ffn[l][k], HID, FFN_W, 0.02f, 0.03f,
                                           swbuf)) != 0) {
                        printf("scale down rc=%d\n", rc); return 10;
                    }
                } else {
                    if ((rc = lfmgemv_set_shape(h_ffn[l][k], FFN_W, HID, 1)) != 0) {
                        printf("shape gate/up rc=%d\n", rc); return 10;
                    }
                    for (i = 0; i < FFN_W; i++) swbuf[i] = 1.0e-3f;
                    if ((rc = set_scale_of(h_ffn[l][k], FFN_W, HID, 0.005f, 1.0f,
                                           swbuf)) != 0) {
                        printf("scale gate/up rc=%d\n", rc); return 10;
                    }
                }
            }
        }
        for (l = 0; l < NL; l++)
            for (k = 0; k < 3; k++)
                H.ffn[l][k] = h_ffn[small ? 0 : l][k];

        for (l = 0; l < N_ATTN; l++) {
            if ((rc = lfmgemv_alloc_view(ah_model, (int)v_k[l].off, v_k[l].bytes, &H.k[l])) != 0 ||
                (rc = lfmgemv_alloc_view(ah_model, (int)v_v[l].off, v_v[l].bytes, &H.v[l])) != 0 ||
                (rc = lfmgemv_alloc_view(ah_model, (int)v_kt[l].off, v_kt[l].bytes, &H.kt[l])) != 0 ||
                (rc = lfmgemv_alloc_view(ah_model, (int)v_tbl[l].off, v_tbl[l].bytes, &H.tbl[l])) != 0) {
                printf("view attn rc=%d\n", rc); return 10;
            }
        }
        for (l = 0; l < N_CONV; l++) {
            if ((rc = lfmgemv_alloc_view(ah_model, (int)v_ci[l].off, v_ci[l].bytes, &H.ci[l])) != 0 ||
                (rc = lfmgemv_alloc_view(ah_model, (int)v_co[l].off, v_co[l].bytes, &H.co[l])) != 0 ||
                (rc = lfmgemv_alloc_view(ah_model, (int)v_cb[l].off, v_cb[l].bytes, &H.cb[l])) != 0) {
                printf("view conv rc=%d\n", rc); return 10;
            }
            if ((rc = lfmgemv_set_shape(H.ci[l], 3 * HID, HID, 1)) != 0 ||
                (rc = lfmgemv_set_shape(H.co[l], HID, HID, 1)) != 0) {
                printf("shape conv rc=%d\n", rc); return 10;
            }
            for (i = 0; i < 3 * HID; i++) swbuf[i] = 1.2e-3f;
            if ((rc = set_scale_of(H.ci[l], 3 * HID, HID, 0.01f, 0.05f, swbuf)) != 0) {
                printf("scale in_proj rc=%d\n", rc); return 10;
            }
            for (i = 0; i < HID; i++) swbuf[i] = 1.4e-3f;
            if ((rc = set_scale_of(H.co[l], HID, HID, 0.01f, 0.05f, swbuf)) != 0) {
                printf("scale out_proj rc=%d\n", rc); return 10;
            }
        }
        printf("handles: %d views of arena %d (one mapping for the model)\n",
               3 * n_ffn_layers + 4 * N_ATTN + 3 * N_CONV, ah_model);
    }
    if ((rc = lfmgemv_reserve_scratch(SCRATCH)) != 0) {
        printf("reserve_scratch rc=%d\n", rc); return 10;
    }

    /* ---------- 3. build the command lists (host memory, before mapping) ---- */
    {
        uint8_t *cb = (uint8_t *)calloc(1, CMD_BYTES);
        int n_cmd = NL + 2 * N_ATTN + 3 * N_CONV;
        if (!cb) { printf("cmd buffer alloc failed\n"); return 1; }

        /* FFN: RPC-equivalent single command, result to the out sequence */
        list_init(cb, OFF_FFN, 1, &one_ffn);
        mk_ffn(cmd_at(&one_ffn, 0), H.ffn[0][0], H.ffn[0][1], H.ffn[0][2],
               S_ACT, LFM_DST_OUT, 0);

        /* attention: q comes through act -> scratch[0] */
        list_init(cb, OFF_ATTN, 1, &one_attn);
        mk_attn(cmd_at(&one_attn, 0), H.k[0], H.kt[0], H.v[0], H.tbl[0],
                PAGES, -1, S_ACT, 0, LFM_DST_OUT);
        one_attn.cmd[0].a0 = CTX - 1; one_attn.cmd[0].a1 = CTX;
        one_attn.cmd[0].a2 = 0;       one_attn.cmd[0].a3 = 1;

        /* shortconv: bcx through act -> scratch[0], fresh state at slot 15 */
        list_init(cb, OFF_CONV, 1, &one_conv);
        mk_conv(cmd_at(&one_conv, 0), H.cb[0], HID, blobs[0].ksh,
                S_STATE + (size_t)15 * 2 * HID * 8, S_ACT, 0, LFM_DST_OUT);

        /* quantise: int32 pattern through act -> scratch[0] */
        list_init(cb, OFF_QUANT, 1, &one_quant);
        mk_quant(cmd_at(&one_quant, 0), S_ACT, 0, HQ * HD, 4096, LFM_DST_OUT);

        /* the full token graph */
        list_init(cb, OFF_ALL, n_cmd, &all);
        c = 0;
        for (l = 0; l < NL; l++)
            mk_ffn(cmd_at(&all, c++), H.ffn[l][0], H.ffn[l][1], H.ffn[l][2],
                   S_ACT, (l == NL - 1) ? LFM_DST_OUT : LFM_DST_SCRATCH,
                   (l == NL - 1) ? 0 : S_Y8);
        for (l = 0; l < N_ATTN; l++) {
            mk_attn(cmd_at(&all, c++), H.k[l], H.kt[l], H.v[l], H.tbl[l],
                    PAGES, 0, S_Q, S_ATTN32, LFM_DST_SCRATCH);
            mk_quant(cmd_at(&all, c++), S_ATTN32, S_Y8, HQ * HD, 4096,
                     LFM_DST_SCRATCH);
        }
        for (l = 0; l < N_CONV; l++) {
            mk_gemv(cmd_at(&all, c++), H.ci[l], S_ACT, S_BCX,
                    LFM_DST_SCRATCH, 3 * HID);
            mk_conv(cmd_at(&all, c++), H.cb[l], HID, blobs[l].ksh,
                    S_STATE + (size_t)l * 2 * HID * 8, S_BCX, S_CY,
                    LFM_DST_SCRATCH);
            mk_gemv(cmd_at(&all, c++), H.co[l], S_CY, S_Y8,
                    LFM_DST_SCRATCH, HID);
        }
        if (c != n_cmd) { printf("cmd count %d != %d\n", c, n_cmd); return 1; }

        /* one single-command list per graph command, for the 85-RPC run */
        for (i = 0; i < n_cmd; i++) {
            size_t off = OFF_SINGLE + (size_t)i * SINGLE_STRIDE;
            list_init(cb, off, 1, &singles[i]);
            singles[i].cmd[0] = all.cmd[i];
            single_off[i] = off;
            if (off + SINGLE_STRIDE > (size_t)CMD_BYTES) {
                printf("single list overflow\n"); return 1;
            }
        }

        if (ion_alloc(&cmd_ion, CMD_BYTES) != 0) {
            printf("cmd ION alloc failed\n"); return 1;
        }
        memcpy(cmd_ion.p, cb, CMD_BYTES);
        free(cb);
        if ((rc = lfmgemv_map_arena(cmd_ion.fd, cmd_ion.bytes, &ah_cmd)) != 0) {
            printf("map cmd arena rc=%d\n", rc); return 10;
        }
        printf("graph: %d commands (%d FFN, %d ATTN+QUANT, %d conv x 3), "
               "list %.1f KiB\n", n_cmd, NL, N_ATTN, N_CONV,
               (double)(sizeof(exec_hdr_t) + (size_t)n_cmd * sizeof(exec_cmd_t)) / 1024.0);

        /* wq1 for the host conv reference */

        /* ---------- 4. bit-exact: FFN RPC vs exec ---------- */
        {
            int8_t mid[FFN_W];
            static ionbuf_t obuf;
            rc = lfmgemv_ffn(H.ffn[0][0], H.ffn[0][1], H.ffn[0][2],
                             (const unsigned char *)act, HID, 1,
                             (unsigned char *)dst_ref, HID,
                             (unsigned char *)mid, FFN_W, times, 8);
            if (rc != 0) { printf("ffn rc=%d\n", rc); return 10; }
            rc = exec_list(ah_cmd, OFF_FFN, &one_ffn, act, HID, 0, 0,
                           (unsigned char *)out_exec, HID, times);
            if (rc != 0) { printf("exec ffn rc=%d cmd=%d\n", rc, times[1]); return 10; }
            bad = 0;
            for (i = 0; i < HID; i++) if (dst_ref[i] != out_exec[i]) bad++;
            printf("bit-exact FFN : rpc vs exec mismatches=%d/%d dsp=%d us\n",
                   bad, HID, times[0]);
            failures += bad != 0;

            /* same FFN written straight into an attach_out ION buffer */
            if (ion_alloc(&obuf, 8192) == 0) {
                list_t att;
                uint8_t *cb = (uint8_t *)calloc(1, 512);
                list_init(cb, 0, 1, &att);
                mk_ffn(cmd_at(&att, 0), H.ffn[0][0], H.ffn[0][1], H.ffn[0][2],
                       S_ACT, LFM_DST_ATTACHED, 0);
                att.cmd[0].dst_h = H.ffn[0][0];
                /* the attached buffer is listed on the same arena as the
                   command list: map a tiny second ION arena for the command */
                {
                    ionbuf_t cl;
                    int ah2 = -1;
                    ion_alloc(&cl, 512);
                    memcpy(cl.p, cb, 512);
                    free(cb);
                    rc = lfmgemv_map_arena(cl.fd, cl.bytes, &ah2);
                    if (rc == 0)
                        rc = lfmgemv_attach_out(H.ffn[0][0], obuf.fd, obuf.bytes);
                    if (rc == 0)
                        rc = exec_list(ah2, 0, &att, act, HID, 0, 0, 0, 0, times);
                    if (rc == 0) {
                        dma_sync(obuf.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
                        bad = memcmp(obuf.p, dst_ref, HID) != 0;
                        dma_sync(obuf.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
                        printf("bit-exact FFN : attach_out dst mismatches=%d/%d\n",
                               bad ? 1 : 0, HID);
                        failures += bad;
                    } else {
                        printf("attach_out dst rc=%d (skipped)\n", rc);
                        failures += 1;
                    }
                    lfmgemv_detach_out(H.ffn[0][0]);
                    if (ah2 >= 0) lfmgemv_unmap_arena(ah2);
                    ion_free(&cl);
                }
                ion_free(&obuf);
            }
        }

        /* ---------- 5. bit-exact: attention RPC vs exec ---------- */
        {
            for (i = 0; i < PAGES; i++) tbl[i] = i;
            rc = lfmgemv_attn(H.k[0], H.kt[0], H.v[0], 0,
                              (const unsigned char *)q, HQ * HD,
                              (const int *)tbl, PAGES, CTX - 1, CTX, 0, 1,
                              (unsigned char *)attn_ref, HQ * HD * 4, times, 8);
            if (rc != 0) { printf("attn rc=%d\n", rc); return 10; }
            rc = exec_list(ah_cmd, OFF_ATTN, &one_attn, q, HQ * HD, 0, 0,
                           (unsigned char *)attn_dev, HQ * HD * 4, times);
            if (rc != 0) { printf("exec attn rc=%d cmd=%d\n", rc, times[1]); return 10; }
            bad = memcmp(attn_ref, attn_dev, (size_t)HQ * HD * 4) != 0;
            printf("bit-exact ATTN: rpc vs exec mismatches=%d dsp=%d us\n",
                   bad ? 1 : 0, times[0]);
            failures += bad;
        }

        /* ---------- 6. bit-exact: shortconv host model vs exec ---------- */
        {
            rc = exec_list(ah_cmd, OFF_CONV, &one_conv, bcx, 3 * HID, 0, 0,
                           (unsigned char *)y_dev, HID, times);
            if (rc != 0) { printf("exec conv rc=%d cmd=%d\n", rc, times[1]); return 10; }
            conv_ref_first(&blobs[0], bcx, y_ref);
            bad = 0;
            for (i = 0; i < HID; i++) if (y_ref[i] != y_dev[i]) bad++;
            printf("bit-exact CONV: ring model vs exec mismatches=%d/%d dsp=%d us\n",
                   bad, HID, times[0]);
            failures += bad != 0;
        }

        /* ---------- 7. bit-exact: QUANT formula ---------- */
        {
            rc = exec_list(ah_cmd, OFF_QUANT, &one_quant, quant_src, HQ * HD * 4,
                           0, 0, (unsigned char *)quant_dev, HQ * HD, times);
            if (rc != 0) { printf("exec quant rc=%d\n", rc); return 10; }
            bad = 0;
            for (i = 0; i < HQ * HD; i++) {
                int64_t v = ((int64_t)((int32_t *)quant_src)[i] * 4096 + 32768) >> 16;
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                if ((int8_t)v != quant_dev[i]) bad++;
            }
            printf("bit-exact QUANT: formula vs exec mismatches=%d/%d\n",
                   bad, HQ * HD);
            failures += bad != 0;
        }

        /* ---------- 8. 24 layers x 3 views, raw int32 GEMV ---------- */
        {
            int8_t a256[256];
            int32_t acc[256];
            int bad_rows = 0, bad_layers = 0, vi = 0;
            fill_i8(a256, 256, 0x999u);
            for (l = 0; l < NL; l++) {
                for (k = 0; k < 3; k++) {
                    int h;
                    rc = lfmgemv_alloc_view(ah_model, (int)v_small[vi].off,
                                            v_small[vi].bytes, &h);
                    if (rc != 0) { printf("small view rc=%d\n", rc); return 10; }
                    rc = lfmgemv_set_shape(h, 256, 256, 0);
                    if (rc != 0) { printf("small shape rc=%d\n", rc); return 10; }
                    rc = lfmgemv_gemv(h, 1, (const unsigned char *)a256, 256, 1,
                                      (unsigned char *)acc, 256 * 4, times, 8);
                    if (rc != 0) { printf("small gemv rc=%d\n", rc); return 10; }
                    {
                        const uint8_t *w = (const uint8_t *)arena.p + v_small[vi].off;
                        int r, lb = 0;
                        for (r = 0; r < 256; r++)
                            if (acc[r] != host_i4_dot(w + (size_t)r * 128, a256, 256))
                                lb++;
                        if (lb) { bad_layers++; bad_rows += lb; }
                    }
                    vi++;
                }
            }
            printf("handles/arena: 24 layers x 3 views read: layers_bad=%d rows_bad=%d\n",
                   bad_layers, bad_rows);
            failures += bad_layers != 0;
        }

        /* ---------- 9. full graph: one exec vs one exec per command -------- */
        {
            double wall_all[16], wall_one[16], dsp_all[16], dsp_one[16];
            int n_all = 0, n_one = 0, t;
            int one_runs = (tokens < 3 ? tokens : 3);

            rt[0] = CTX - 1; rt[1] = CTX; rt[2] = 0; rt[3] = 1;

            /* walk the singles once, in order: reports the first bad command
               by index and type before the timing loops hide it */
            {
                double slow_wall[5] = { 0 }, slow_dsp[5] = { 0 };
                int slow_idx[5] = { -1, -1, -1, -1, -1 };
                double type_dsp[5] = { 0 };
                int type_n[5] = { 0 };
                for (i = 0; i < n_cmd; i++) {
                    double w0 = now_us(), w;
                    unsigned ty;
                    rc = exec_list(ah_cmd, single_off[i], &singles[i],
                                   (i == 0) ? act : 0, (i == 0) ? HID : 0,
                                   rt, 4, (unsigned char *)out_exec, HID, times);
                    w = now_us() - w0;
                    if (rc != 0) {
                        printf("single cmd %d (type %u) rc=0x%x\n", i,
                               singles[i].cmd[0].type, rc);
                        return 10;
                    }
                    ty = singles[i].cmd[0].type;
                    if (ty <= LFM_CMD_CONV) { type_dsp[ty] += times[0]; type_n[ty]++; }
                    {
                        int j, worst = 4;
                        if (w > slow_wall[worst]) {
                            for (j = 3; j >= 0 && w > slow_wall[j]; j--) {
                                slow_wall[j + 1] = slow_wall[j];
                                slow_dsp[j + 1] = slow_dsp[j];
                                slow_idx[j + 1] = slow_idx[j];
                            }
                            j++;
                            slow_wall[j] = w;
                            slow_dsp[j] = times[0];
                            slow_idx[j] = i;
                        }
                    }
                }
                for (i = 0; i < 5; i++)
                    if (slow_idx[i] >= 0)
                        printf("  slow cmd %3d type %u: wall=%.0f dsp=%d us\n",
                               slow_idx[i], singles[slow_idx[i]].cmd[0].type,
                               slow_wall[i], (int)slow_dsp[i]);
                printf("DSP by type: FFN %.1f us x%d | ATTN %.0f us x%d | "
                       "QUANT %.0f us x%d | GEMV %.0f us x%d | CONV %.0f us x%d\n",
                       type_dsp[LFM_CMD_FFN] / (type_n[LFM_CMD_FFN] ? type_n[LFM_CMD_FFN] : 1),
                       type_n[LFM_CMD_FFN],
                       type_dsp[LFM_CMD_ATTN] / (type_n[LFM_CMD_ATTN] ? type_n[LFM_CMD_ATTN] : 1),
                       type_n[LFM_CMD_ATTN],
                       type_dsp[LFM_CMD_QUANT] / (type_n[LFM_CMD_QUANT] ? type_n[LFM_CMD_QUANT] : 1),
                       type_n[LFM_CMD_QUANT],
                       type_dsp[LFM_CMD_GEMV] / (type_n[LFM_CMD_GEMV] ? type_n[LFM_CMD_GEMV] : 1),
                       type_n[LFM_CMD_GEMV],
                       type_dsp[LFM_CMD_CONV] / (type_n[LFM_CMD_CONV] ? type_n[LFM_CMD_CONV] : 1),
                       type_n[LFM_CMD_CONV]);
            }

            for (t = 0; t <= tokens; t++) {
                int8_t *o = (t == 0) ? out_exec : out_all;
                double w0 = now_us();
                rc = exec_list(ah_cmd, OFF_ALL, &all, act, HID, rt, 4,
                               (unsigned char *)o, HID, times);
                if (rc != 0) { printf("exec all rc=%d cmd=%d\n", rc, times[1]); return 10; }
                if (t > 0) { dsp_all[n_all] = times[0]; wall_all[n_all] = now_us() - w0; n_all++; }
            }
            memcpy(out_all, out_exec, HID);

            for (t = 0; t <= one_runs; t++) {
                int8_t *o = (t == 0) ? out_exec : out_one;
                double w0 = now_us(), dsum = 0;
                for (i = 0; i < n_cmd; i++) {
                    rc = exec_list(ah_cmd, single_off[i], &singles[i],
                                   (i == 0) ? act : 0, (i == 0) ? HID : 0,
                                   rt, 4, (unsigned char *)o, HID, times);
                    if (rc != 0) { printf("exec one rc=%d cmd=%d\n", rc, times[1]); return 10; }
                    dsum += times[0];
                }
                if (t > 0) { dsp_one[n_one] = dsum; wall_one[n_one] = now_us() - w0; n_one++; }
            }

            bad = memcmp(out_all, out_one, HID) != 0;
            printf("graph bit-exact: 1 exec vs %d execs mismatches=%d\n",
                   n_cmd, bad ? 1 : 0);
            failures += bad;

            {
                double ma = 0, mo = 0, da = 0, d1 = 0;
                double wmin_a = 1e18, wmin_o = 1e18, dmin_a = 1e18, dmin_o = 1e18;
                int j;
                for (j = 0; j < n_all; j++) {
                    ma += wall_all[j];
                    da += dsp_all[j];
                    if (wall_all[j] < wmin_a) wmin_a = wall_all[j];
                    if (dsp_all[j] < dmin_a) dmin_a = dsp_all[j];
                }
                for (j = 0; j < n_one; j++) {
                    mo += wall_one[j];
                    d1 += dsp_one[j];
                    if (wall_one[j] < wmin_o) wmin_o = wall_one[j];
                    if (dsp_one[j] < dmin_o) dmin_o = dsp_one[j];
                }
                ma /= n_all; mo /= n_one; da /= n_all; d1 /= n_one;
                printf("\nper token:            1 exec      %d execs\n", n_cmd);
                printf("  DSP (sum of cmds) : %7.0f us  %7.0f us\n", da, d1);
                printf("  wall (mean)      : %7.0f us  %7.0f us\n", ma, mo);
                printf("  wall (min)       : %7.0f us  %7.0f us\n", wmin_a, wmin_o);
                printf("  RPC calls        : %7d     %7d\n", 1, n_cmd);
                printf("  RPC overhead(min): %7.0f us  %7.0f us\n",
                       wmin_a - dmin_a, wmin_o - dmin_o);
                printf("  tok/s (wall min) : %7.2f     %7.2f\n",
                       1e6 / wmin_a, 1e6 / wmin_o);
                printf("  vs 54.3 ms report: %+6.1f %% (wall min)\n",
                       (wmin_a / 54300.0 - 1.0) * 100.0);
            }
        }
    }

    lfmgemv_reserve_scratch(0);
    printf("\n%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 2 : 0;
}
