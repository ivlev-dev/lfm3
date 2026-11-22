/* Offline numeric study for the int8 SiLU stage (no DSP needed). */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "silu_lut.h"

static double silu_exact(double x) { return x / (1.0 + exp(-x)); }

/* current DSP implementation: LUT + linear interpolation, Q16 */
static int32_t silu_q16_lut(int32_t zq)
{
    int32_t off, idx, frac, a, b;
    if (zq > 8 * 65536) zq = 8 * 65536;
    if (zq < -8 * 65536) zq = -8 * 65536;
    off = zq + 8 * 65536;
    idx = off >> 10;
    frac = off & 1023;
    if (idx >= SILU_LUT_N - 1) return silu_lut[SILU_LUT_N - 1];
    a = silu_lut[idx];
    b = silu_lut[idx + 1];
    return a + (((b - a) * frac) >> 10);
}

/* piecewise fit helpers */
typedef struct { double a, b, c; } mypoly_t;   /* value = a + b*u + c*u^2, u = x - x0 */

int main(void)
{
    int i;
    double maxerr_lut = 0.0;
    double arg = 0;

    /* 1. LUT accuracy */
    for (i = 0; i <= 200000; i++) {
        double x = -8.0 + 16.0 * i / 200000.0;
        int32_t zq = (int32_t)llround(x * 65536.0);
        double got = silu_q16_lut(zq) / 65536.0;
        double e = fabs(got - silu_exact(x));
        if (e > maxerr_lut) { maxerr_lut = e; arg = x; }
    }
    printf("LUT-1025 + lerp      : max|err| = %.3e  (at x=%+.3f)\n", maxerr_lut, arg);

    /* 2. piecewise linear on N equal segments over [-8,8] */
    {
        int Ns[] = { 32, 64, 128, 256 };
        int k;
        for (k = 0; k < 4; k++) {
            int n = Ns[k], s;
            double w = 16.0 / n, maxe = 0.0;
            double *v = malloc((n + 1) * sizeof(double));
            for (s = 0; s <= n; s++) v[s] = silu_exact(-8.0 + s * w);
            for (i = 0; i <= 200000; i++) {
                double x = -8.0 + 16.0 * i / 200000.0;
                double t = (x + 8.0) / w;
                int j = (int)floor(t);
                double f;
                if (j >= n) { j = n - 1; }
                f = t - j;
                {
                    double got = v[j] * (1 - f) + v[j + 1] * f;
                    double e = fabs(got - silu_exact(x));
                    if (e > maxe) maxe = e;
                }
            }
            printf("pwl  %4d segments  : max|err| = %.3e  (slope table %d B q16)\n",
                   n, maxe, (n + 1) * 4);
            free(v);
        }
    }

    /* 3. piecewise quadratic, least squares on 3 nodes per segment */
    {
        int Ns[] = { 4, 8, 16, 32 };
        int k;
        for (k = 0; k < 4; k++) {
            int n = Ns[k], s;
            double w = 16.0 / n, maxe = 0.0;
            mypoly_t *q = malloc(n * sizeof(mypoly_t));
            for (s = 0; s < n; s++) {
                double x0 = -8.0 + s * w;
                double u0 = 0.0, u1 = w * 0.5, u2 = w;   /* nodes */
                double y0 = silu_exact(x0 + u0), y1 = silu_exact(x0 + u1), y2 = silu_exact(x0 + u2);
                /* solve for a,b,c through the 3 points */
                double det = (u1 - u0) * (u2 * u2 - u0 * u0) - (u2 - u0) * (u1 * u1 - u0 * u0);
                double a, b, c;
                b = ((y1 - y0) * (u2 * u2 - u0 * u0) - (y2 - y0) * (u1 * u1 - u0 * u0)) / det;
                c = ((y2 - y0) - b * (u2 - u0)) / (u2 * u2 - u0 * u0);
                a = y0 - b * u0 - c * u0 * u0;
                q[s].a = a; q[s].b = b; q[s].c = c;
            }
            for (i = 0; i <= 200000; i++) {
                double x = -8.0 + 16.0 * i / 200000.0;
                double t = (x + 8.0) / w;
                int j = (int)floor(t);
                double u, got, e;
                if (j >= n) j = n - 1;
                u = x - (-8.0 + j * w);
                got = q[j].a + q[j].b * u + q[j].c * u * u;
                e = fabs(got - silu_exact(x));
                if (e > maxe) maxe = e;
            }
            printf("quad %4d segments  : max|err| = %.3e  (coeff table %d B q14)\n",
                   n, maxe, n * 3 * 2);
            free(q);
        }
    }

    /* 3b. integer Q14 implementation of the 32-segment quadratic
       (the shape a HVX version would take: u in Q14, coeffs in Q14,
        products via 16x16 -> 32 followed by >>14) */
    {
        const int n = 32;
        double w = 16.0 / n;
        int32_t ca[32], cb[32], cc[32];
        int s2;
        double maxe_q = 0.0;
        for (s2 = 0; s2 < n; s2++) {
            double x0 = -8.0 + s2 * w;
            double u0 = 0.0, u1 = w * 0.5, u2 = w;
            double y0 = silu_exact(x0 + u0), y1 = silu_exact(x0 + u1), y2 = silu_exact(x0 + u2);
            double det = (u1 - u0) * (u2 * u2 - u0 * u0) - (u2 - u0) * (u1 * u1 - u0 * u0);
            double b = ((y1 - y0) * (u2 * u2 - u0 * u0) - (y2 - y0) * (u1 * u1 - u0 * u0)) / det;
            double c = ((y2 - y0) - b * (u2 - u0)) / (u2 * u2 - u0 * u0);
            double a = y0 - b * u0 - c * u0 * u0;
            ca[s2] = (int32_t)llround(a * 16384.0);
            cb[s2] = (int32_t)llround(b * 16384.0 * w);       /* b*u, u = w*t, t in Q14 */
            cc[s2] = (int32_t)llround(c * 16384.0 * w * w);
        }
        for (i = 0; i <= 200000; i++) {
            double x = -8.0 + 16.0 * i / 200000.0;
            int32_t tq = (int32_t)llround((x + 8.0) / w * 16384.0); /* segment coord, Q14 */
            int32_t j = tq >> 14;
            int32_t t;
            int32_t got;
            if (j >= n) j = n - 1;
            if (j < 0) j = 0;
            t = tq - (j << 14);                       /* 0..16383, Q14 */
            {
                int32_t t2 = (int32_t)(((int64_t)t * t) >> 14);
                got = ca[j] + (int32_t)(((int64_t)cb[j] * t) >> 14)
                            + (int32_t)(((int64_t)cc[j] * t2) >> 14);
            }
            {
                double e = fabs(got / 16384.0 - silu_exact(x));
                if (e > maxe_q) maxe_q = e;
            }
        }
        printf("quad 32 seg, int Q14: max|err| = %.3e\n", maxe_q);
    }

    /* 4. error propagation: SiLU error -> int8 LSB of the middle activation
       h = silu(gate)*up/s_mid,  s_mid = 0.0625, |up| <= 4 */
    printf("\nbudget: 1 LSB of h = s_mid = 0.0625\n");
    printf("  SiLU err 1e-3 with |up|=4  -> %.2f LSB\n", 1e-3 * 4 / 0.0625);
    printf("  SiLU err 1e-2 with |up|=4  -> %.2f LSB\n", 1e-2 * 4 / 0.0625);
    return 0;
}
