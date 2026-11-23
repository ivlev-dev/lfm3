/*
 * emit_main: stand for the HVX-friendly combine `pref_emit` (emit_impl.h).
 *
 * Fills a synthetic partial tile set (ng groups of 64 spatial x 32 outputs)
 * the way the HMX store does, then checks the combine against a plain scalar
 * triple loop and prints pcycles for both.  Run under hexagon-sim:
 *
 *   ./emit_build.sh
 */
#include "emit_impl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef EMIT_HOST
#include <time.h>
static uint64_t HAP_perf_get_pcycles(void) { return (uint64_t)clock(); }
#else
#include <qurt_alloc.h>
#include "HAP_perf.h"
QURT_DECLARE_STATIC_HEAP(8u * 1024u * 1024u);
#endif

#define ROWS 64
#define NG   16
#define NCOL 32

static uint8_t g_part[NG * 2048] __attribute__((aligned(128)));
static float g_S[NG * NCOL * 4] __attribute__((aligned(128)));
static float g_Q[64] __attribute__((aligned(128)));
static float g_svec[128 * 64] __attribute__((aligned(128)));
static float g_fast[ROWS * NCOL] __attribute__((aligned(128)));
static float g_ref[ROWS * NCOL] __attribute__((aligned(128)));
static float g_tmp[ROWS * NCOL] __attribute__((aligned(128)));

static uint32_t g_rng = 0x1234abcd;

static uint32_t rnd(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

/* the old product loop: S is re-read per output element inside the g loop */
static void old_emit(const uint8_t *part, int ng, const float *S,
                     const float *Q, float *C, int ldc, int rows, int n0)
{
    int s, n, g;
    for (s = 0; s < rows; s++)
        for (n = 0; n < NCOL; n++) {
            float acc = 0.0f;
            for (g = 0; g < ng; g++) {
                float u = (float)part[(size_t)g * 2048u
                            + 128u * (unsigned)(s >> 2) + 4u * (unsigned)n
                            + (unsigned)(s & 3)];
                acc += S[(size_t)(n0 + n) * ng + g] * (Q ? Q[g] : 1.0f) *
                       (7.02745f * (u + 0.5f) - 448.0f);
            }
            C[(size_t)s * ldc + n0 + n] = acc;
        }
}

/* straightforward reference: one output at a time */
static void ref_emit(const uint8_t *part, int ng, const float *S,
                     const float *Q, float *C, int ldc, int rows, int n0)
{
    int s, n, g;
    for (s = 0; s < rows; s++)
        for (n = 0; n < NCOL; n++) {
            float acc = 0.0f;
            for (g = 0; g < ng; g++) {
                float u = (float)part[(size_t)g * 2048u
                            + 128u * (unsigned)(s >> 2) + 4u * (unsigned)n
                            + (unsigned)(s & 3)];
                acc += S[(size_t)(n0 + n) * ng + g] * (Q ? Q[g] : 1.0f) *
                       (7.02745f * (u + 0.5f) - 448.0f);
            }
            C[(size_t)s * ldc + n0 + n] = acc;
        }
}

int main(void)
{
    int i, bad = 0;
    float maxd = 0.0f;
    uint64_t t0, t1, t2;
    int rep;

    for (i = 0; i < NG * 2048; i++)
        g_part[i] = (uint8_t)(rnd() >> 24);
    for (i = 0; i < NG * NCOL; i++)
        g_S[i] = (float)(rnd() >> 8) / 16777216.0f;
    for (i = 0; i < NG; i++)
        g_Q[i] = (float)(1 + (rnd() % 60));

    t0 = HAP_perf_get_pcycles();
    for (rep = 0; rep < 4; rep++)
        old_emit(g_part, NG, g_S, g_Q, g_ref, NCOL, ROWS, 0);
    t1 = HAP_perf_get_pcycles();
    {
        uint64_t tc = t1;
        for (rep = 0; rep < 4; rep++)
            ref_emit(g_part, NG, g_S, g_Q, g_tmp, NCOL, ROWS, 0);
        printf("EMIT pcycles old=%llu ref=%llu\n",
               (unsigned long long)((t1 - t0) / 4),
               (unsigned long long)((HAP_perf_get_pcycles() - tc) / 4));
    }
    t1 = HAP_perf_get_pcycles();
    for (rep = 0; rep < 4; rep++)
        pref_emit(g_part, NG, g_S, g_Q, g_svec, g_fast, NCOL, ROWS, 0, 0);
    t2 = HAP_perf_get_pcycles();

    for (i = 0; i < ROWS * NCOL; i++) {
        float d = fabsf(g_fast[i] - g_ref[i]);
        if (d > maxd) maxd = d;
        if (d > 1e-3f * (fabsf(g_ref[i]) + 1.0f)) bad++;
    }
    printf("EMIT rows=%d ng=%d: mismatches=%d max|d|=%.3e  %s\n", ROWS, NG, bad,
           maxd, bad == 0 ? "OK" : "FAIL");
    printf("EMIT pcycles: scalar=%llu fast=%llu (x%.2f)\n",
           (unsigned long long)((t1 - t0) / 4), (unsigned long long)((t2 - t1) / 4),
           (double)(t1 - t0) / (double)(t2 - t1));
    printf("EMIT RESULT: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad != 0;
}
