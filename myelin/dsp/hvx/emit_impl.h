#ifndef LFM_EMIT_IMPL_H
#define LFM_EMIT_IMPL_H

#include <stdint.h>
#include <stddef.h>

/*
 * Combine of one HMX partial tile (64 spatial rows x 32 outputs) with the
 * per-group scales, shared by the product prefill and the stand.
 *
 * Partial layout (REPORT25 s.3): byte at 128*(s>>2) + 4*n + (s&3) holds
 * u = trunc((P + 448*Aq_g) * m_g / 2), with m_g = 255/(2*448*Aq_g).
 * The combine is
 *
 *   C[s][n] = sum_g S[n][g] * Aq_g * (7.02745*(u+0.5) - 448)
 *
 * (7.02745 = 2*448*255^-1 * 2 = 4*448/255).  The group loop is written as a
 * contiguous 128-element run over one phase block (4 spatial rows x 32 n), so
 * the compiler vectorises it with HVX; the 4x32 transpose is done once per
 * block.
 *
 * `svec` must hold at least 128 * ng floats (scratch owned by the caller).
 */
static void pref_emit(const uint8_t *part, int ng, const float *S,
                      const float *Q, float *svec, float *C, int ldc,
                      int rows, int s0, int n0)
{
    float acc[128] __attribute__((aligned(128)));
    int nblk = (rows + 3) >> 2;
    int blk, g, j, t, n;

    for (g = 0; g < ng; g++) {
        float aq = Q ? Q[g] : 0.0f;
        float *sv = svec + (size_t)g * 128u;
        for (n = 0; n < 32; n++) {
            float v = S[(size_t)(n0 + n) * ng + g] * aq;
            sv[4 * n + 0] = v;
            sv[4 * n + 1] = v;
            sv[4 * n + 2] = v;
            sv[4 * n + 3] = v;
        }
    }

    for (blk = 0; blk < nblk; blk++) {
        for (j = 0; j < 128; j++)
            acc[j] = 0.0f;
        for (g = 0; g < ng; g++) {
            const uint8_t *p = part + (size_t)g * 2048u + 128u * (unsigned)blk;
            const float *sv = svec + (size_t)g * 128u;
            for (j = 0; j < 128; j++)
                acc[j] += sv[j] * (7.02745f * ((float)p[j] + 0.5f) - 448.0f);
        }
        for (t = 0; t < 4; t++) {
            int s = blk * 4 + t;
            float *out;
            if (s >= rows)
                break;
            out = C + (size_t)(s0 + s) * (size_t)ldc + n0;
            for (n = 0; n < 32; n++)
                out[n] = acc[4 * n + t];
        }
    }
}

#endif /* LFM_EMIT_IMPL_H */
