/*
 * prefill_test: HMX prefill through the product runtime (`prefill` RPC).
 *
 * The weights are built the way the product will build them: synthetic GGUF
 * Q4_K blocks -> dequant to f32 (q4k_dequant_gguf) -> requantise to int4 with
 * one scale per 32*G channels -> pack into the HMX crouton blob.
 *
 * HMX `activation.ub` is unsigned, so the signed int8 activations are shifted
 * by +128 (same convention as the dsp/hmx stand).  The shift adds
 * 128 * sum_k(q) per (n, g); the host knows the weight codes and removes it:
 *   C_signed = C_hmx - 128 * sum_g S[n][g] * Q[n][g],  Q = sum_k q.
 * The result is compared with the exact integer reference in signed space
 * (REPORT24 gate: the 8-bit partial store is not exact, cos must stay above
 * 0.9995).
 *
 * Run on device: ./prefill_test [M] [N] [K] [G]
 * Exit code: 0 pass, 2 numerics, 10.. RPC failure.
 */
#include "lfmgemv.h"
#include "q4k_impl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rpcmem.h"

#define PASS_COS 0.9995

static uint32_t g_rng = 0x51ed270bu;

static uint32_t rnd(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

static float frnd(void)
{
    return (float)(rnd() >> 8) / 16777216.0f - 0.5f;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                  (uint32_t)sizeof(um));
}

static int upload(const uint8_t *p, int bytes, int *h)
{
    size_t off = 0, chunk = 256 * 1024;
    int rc;
    if ((rc = lfmgemv_alloc_buf(bytes, h)) != 0)
        return rc;
    while (off < (size_t)bytes) {
        size_t n = (size_t)bytes - off < chunk ? (size_t)bytes - off : chunk;
        if ((rc = lfmgemv_load_buf(*h, (int)off, p + off, (int)n)) != 0)
            return rc;
        off += n;
    }
    return 0;
}

static void make_block(uint8_t *b)
{
    uint16_t db = (uint16_t)(0x0b00u | (rnd() & 0x3ffu));
    uint16_t mb = (uint16_t)(0x0900u | (rnd() & 0x3ffu));
    int i;
    b[0] = (uint8_t)(db & 0xff);
    b[1] = (uint8_t)(db >> 8);
    b[2] = (uint8_t)(mb & 0xff);
    b[3] = (uint8_t)(mb >> 8);
    for (i = 4; i < Q4K_GGUF_BLOCK; i++)
        b[i] = (uint8_t)rnd();
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 128;
    int N = argc > 2 ? atoi(argv[2]) : 7680;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    int G = argc > 4 ? atoi(argv[4]) : 4;
    int store = argc > 5 ? atoi(argv[5]) : 0;
    int dump = argc > 6 ? atoi(argv[6]) : 0;
    int plain = argc > 7 ? atoi(argv[7]) : 0;
    int cact = argc > 8 ? atoi(argv[8]) : -1;
    int hw = -1, hs = -1, hq = -1, rc, t, k, n, g, i, s;
    int ng = K / (32 * G);
    size_t wbytes = (size_t)(N / 32) * (K / 32) * 512u;
    size_t sbytes = (size_t)N * ng * sizeof(float);
    uint8_t *blob = NULL;
    float *S = NULL, *W = NULL, *C = NULL, *Q = NULL;
    int8_t *A = NULL, *q = NULL;
    unsigned char *Au = NULL;
    int info[20], times[8];
    int C0, Aq;
    double cos, dot = 0, na = 0, nb = 0, mx = 0;
    double dot2 = 0, na2 = 0, nb2 = 0;
    long cells = 0, bad = 0;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (K % (32 * G) || N % 32 || K % 256) {
        printf("bad shape M=%d N=%d K=%d G=%d\n", M, N, K, G);
        return 1;
    }
    if ((rc = enable_unsigned()) != 0)
        printf("unsigned PD rc=%d\n", rc);

    blob = (uint8_t *)calloc(1, wbytes);
    S = (float *)malloc(sbytes);
    Q = (float *)malloc(sbytes);
    W = (float *)malloc((size_t)N * K * sizeof(float));
    C = (float *)malloc(((size_t)M * N * sizeof(float) > (size_t)ng * 4096u)
                        ? (size_t)M * N * sizeof(float) : (size_t)ng * 4096u);
    A = (int8_t *)malloc((size_t)M * K);
    q = (int8_t *)malloc((size_t)N * K);
    Au = (unsigned char *)malloc((size_t)M * K);

    if (!blob || !S || !W || !C || !A || !q || !Au || !Q) {
        printf("host alloc failed\n");
        return 1;
    }

    /* synthetic Q4_K tensor -> f32 (the product starts from these blocks) */
    {
        int blocks = K / 256;
        uint8_t *gguf = (uint8_t *)malloc((size_t)N * blocks * Q4K_GGUF_BLOCK);
        if (!gguf) return 1;
        for (n = 0; n < N * blocks; n++)
            make_block(gguf + (size_t)n * Q4K_GGUF_BLOCK);
        for (n = 0; n < N; n++)
            for (t = 0; t < blocks; t++)
                q4k_dequant_gguf(gguf + ((size_t)n * blocks + t) * Q4K_GGUF_BLOCK,
                                 W + (size_t)n * K + t * 256);
        free(gguf);
    }

    /* f32 -> int4 with one scale per 32*G channels, packed for HMX */
    for (n = 0; n < N; n++) {
        const float *row = W + (size_t)n * K;
        for (g = 0; g < ng; g++) {
            const float *wg = row + g * 32 * G;
            float amax = 1e-8f, sc;
            for (i = 0; i < 32 * G; i++)
                if (fabsf(wg[i]) > amax) amax = fabsf(wg[i]);
            sc = amax / 7.0f;
            S[(size_t)n * ng + g] = sc;
            for (i = 0; i < 32 * G; i++) {
                int v = (int)roundf(wg[i] / sc);
                if (v > 7) v = 7;
                if (v < -7) v = -7;
                q[(size_t)n * K + g * 32 * G + i] = (int8_t)v;
            }
        }
    }
    /* zero-mean codes per (n,g): removes the 128*sum(q) term of the unsigned
     * activation shift, so the 8-bit partial carries only the useful signal */
    for (n = 0; n < N; n++)
        for (g = 0; g < ng; g++) {
            int8_t *row = q + (size_t)n * K + g * 32 * G;
            int len = 32 * G, sum = 0, it;
            for (it = 0; it < len; it++) sum += row[it];
            while (sum != 0) {
                int best = -1, bv = 0, it2;
                for (it2 = 0; it2 < len; it2++) {
                    if (sum > 0 && row[it2] > 0 && row[it2] > bv) { bv = row[it2]; best = it2; }
                    if (sum < 0 && row[it2] < 0 && row[it2] < bv) { bv = row[it2]; best = it2; }
                }
                if (best < 0) break;
                row[best] = (int8_t)(row[best] + (sum > 0 ? -1 : 1));
                sum += (sum > 0 ? -1 : 1);
            }
        }
    for (n = 0; n < N; n++) {
        int tile = n / 32, nn = n & 31;
        for (k = 0; k < K; k++) {
            unsigned byte = (unsigned)tile * (unsigned)(K / 32) * 512u
                          + 512u * (unsigned)(k >> 5) + 128u * (unsigned)((k & 31) >> 3)
                          + 4u * (unsigned)nn + (unsigned)(k & 3);
            int nib = q[(size_t)n * K + k] & 15;
            if ((k >> 2) & 1) blob[byte] |= (uint8_t)(nib << 4);
            else              blob[byte] |= (uint8_t)nib;
        }
    }

    if (plain) {
        for (i = 0; i < N * K; i++) q[i] = (plain == 2) ? 0 : 7;
        for (i = 0; i < N * ng; i++) S[i] = 1.0f;
    }
    for (i = 0; i < M * K; i++) {
        if (cact >= 0) {
            Au[i] = (unsigned char)cact;
            A[i] = (int8_t)(cact - 128);
        } else {
            A[i] = (int8_t)((int)(rnd() >> 24) - 128);
            Au[i] = (unsigned char)(A[i] + 128);
        }
    }
    for (g = 0; g < ng; g++) {
        double pm = 0;
        for (n = 0; n < N; n++)
            for (s = 0; s < (M < 8 ? M : 8); s++) {
                long acc = 0;
                for (k = 0; k < 32 * G; k++)
                    acc += (long)Au[(size_t)s * K + g * 32 * G + k]
                         * (long)q[(size_t)n * K + g * 32 * G + k];
                if (fabs((double)acc) > pm) pm = fabs((double)acc);
            }
        {
            int aq = (int)(pm * 1.25 / 448.0) + 1;
            if (aq < 1) aq = 1;
            if (aq > 255) aq = 255;
            Q[g] = (float)aq;
        }
    }


    /* C0 = 224*Aq from a sample of partial sums (same rule as the stand) */
    {
        double mxs = 0;
        int cols = N;
        for (n = 0; n < cols; n++)
            for (s = 0; s < (M < 8 ? M : 8); s++)
                for (g = 0; g < ng; g++) {
                    long pacc = 0;
                    for (k = 0; k < 32 * G; k++)
                        pacc += (long)Au[(size_t)s * K + g * 32 * G + k]
                              * (long)q[(size_t)n * K + g * 32 * G + k];
                    if (fabs((double)pacc) > mxs) mxs = fabs((double)pacc);
                }
        Aq = (int)(mxs * 1.25 / 448.0) + 1;
        if (Aq < 1) Aq = 1;
        if (Aq > 255) Aq = 255;
        C0 = 448 * Aq;
    }

    rc = upload(blob, (int)wbytes, &hw);
    if (!rc) rc = upload((const uint8_t *)S, (int)sbytes, &hs);
    if (!rc) rc = upload((const uint8_t *)Q, ng * (int)sizeof(float), &hq);
    if (rc != 0) {
        printf("upload rc=%d\n", rc);
        return 10 + (rc & 7);
    }
    rc = lfmgemv_prefill(hw, hs, hq, (const unsigned char *)Au, M * K, M, N, K,
                         G, C0, 1 | 8 | (dump ? 16 : 0) | (store << 8), 1,
                         (unsigned char *)C, M * N * 4, info, 20);
    if (rc != 0) {
        printf("prefill rc=%d info0=%d info1=%d\n", rc, info[0], info[1]);
        return 10 + (rc & 7);
    }
    lfmgemv_release_buf(hw);
    lfmgemv_release_buf(hs);
    lfmgemv_release_buf(hq);

    printf("PREFILL M=%d N=%d K=%d G=%d (B=%d K) C0=%d store=%d\n", M, N, K, G,
           32 * G, C0, store);
    if (dump) {
        const unsigned char *pt = (const unsigned char *)C;
        const short *p16 = (const short *)C;
        int qi;
        long accu = 0;
        for (k = 0; k < 32 * G; k++)
            accu += (long)Au[k] * (long)q[k];
        int nz = -1, nz2 = -1, gg;
        printf("  u per group:");
        for (gg = 0; gg < ng && gg < 16; gg++) printf(" %.0f", C[gg]);
        printf("\n");
        for (qi = 0; qi < 2048; qi++) if (C[qi] != 0) { nz = qi; break; }
        for (qi = 0; qi < 1024; qi++) if (C[qi] != 0) { nz2 = qi; break; }
        printf("  ref acc(n=0,g=0)=%ld  first nz idx=%d val=%.0f (first 1k=%d)\n",
               accu, nz, nz >= 0 ? C[nz] : 0.0f, nz2);
        printf("  C[0..7]:");
        for (qi = 0; qi < 8; qi++) printf(" %.0f", C[qi]);
        printf("\n  C[2040..2047]:");
        for (qi = 2040; qi < 2048; qi++) printf(" %.0f", C[qi]);
        printf("\n");
        (void)pt; (void)p16;
    }
    printf("  total=%d us copy=%d us combine=%d us tiles nt=%d mt=%d ncr=%d "
           "ng=%d vtcm=%d copycheck=%d\n",
           info[1], info[2], info[11], info[4], info[5], info[6], info[15],
           info[7], info[12]);
    {
        double scs = 0;
        for (k = 0; k < ng; k++) scs += S[(size_t)0 * ng + k];
        {
            double m = 255.0 / (2.0 * (double)C0);
            double utot = ((double)C[0] + (double)C0 * scs) * m;
            printf("  probe: C[0]=%.3f C[1]=%.3f u_avg=%.4f  (acc_expected=%.1f)\n",
                   C[0], C[1], utot / (double)ng,
                   32.0 * (cact >= 0 ? cact : 0) * 7.0);
        }
    }
    if (info[1] > 0)
        printf("  TMAC/s=%.2f  one matrix %.1f us/token at batch %d -> %.0f tok/s\n",
               (double)M * N * K / info[1] * 1e-6, (double)info[1] / M, M,
               M * 1e6 / (double)info[1]);
    (void)times;

    {
        int chk = (M < 4) ? M : 4;
        for (s = 0; s < chk; s++)
            for (n = 0; n < N; n++) {
                double ref = 0, got, e;
                for (g = 0; g < ng; g++) {
                    long pacc = 0;
                    for (k = 0; k < 32 * G; k++)
                        pacc += (long)((int)Au[(size_t)s * K + g * 32 * G + k] - 128)
                              * (long)q[(size_t)n * K + g * 32 * G + k];
                    ref += (double)S[(size_t)n * ng + g] * (double)pacc;
                }
                got = C[(size_t)s * N + n];
                if (s == 0 && n == 0)
                    printf("  cell[0][0]: got=%.2f ref=%.2f\n", got, ref);
                e = fabs(got - ref);
                if (e > mx) mx = e;
                if (e > 0.02 * (fabs(ref) + 1e-9)) bad++;
                dot += got * ref;
                na += got * got;
                nb += ref * ref;
                dot2 += got * (-ref);
                na2 += got * got;
                nb2 += ref * ref;
                cells++;
            }
        cos = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0.0;
        printf("  cos(+ref)=%.7f cos(-ref)=%.7f\n", cos,
               (na2 > 0 && nb2 > 0) ? dot2 / sqrt(na2 * nb2) : 0.0);
        printf("  vs integer reference: cells=%ld maxabs=%.3f bad>2%%=%ld cos=%.7f %s\n",
               cells, mx, bad, cos, cos >= PASS_COS ? "OK" : "FAIL");
    }
    if (cos < PASS_COS) {
        printf("PREFILL RESULT: FAIL\n");
        return 2;
    }
    printf("PREFILL RESULT: PASS\n");
    return 0;
}
