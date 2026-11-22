/* Chebyshev fit of silu(8z), z in [-1,1]; then exact integer Q12/Q14 Horner sim. */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
static double silu(double x){ return x/(1.0+exp(-x)); }

/* solve normal equations for least squares on Chebyshev nodes (simple approach:
   fit by least squares on a dense grid using monomial basis with scaling) */
static void fit(int deg, double *c)
{
    int n = 4000, i, j, k;
    double *A = calloc((deg+1)*(deg+1), sizeof(double));
    double *b = calloc(deg+1, sizeof(double));
    for (k = 0; k < n; k++) {
        double z = -1.0 + 2.0*k/(n-1);
        double y = silu(8.0*z);
        double pw[32];
        pw[0] = 1.0;
        for (j = 1; j <= deg; j++) pw[j] = pw[j-1]*z;
        for (i = 0; i <= deg; i++) {
            b[i] += pw[i]*y;
            for (j = 0; j <= deg; j++) A[i*(deg+1)+j] += pw[i]*pw[j];
        }
    }
    /* gaussian elimination */
    for (i = 0; i <= deg; i++) {
        double piv = A[i*(deg+1)+i];
        for (j = i; j <= deg; j++) A[i*(deg+1)+j] /= piv;
        b[i] /= piv;
        for (k = 0; k <= deg; k++) {
            if (k == i) continue;
            {
                double f = A[k*(deg+1)+i];
                for (j = i; j <= deg; j++) A[k*(deg+1)+j] -= f*A[i*(deg+1)+j];
                b[k] -= f*b[i];
            }
        }
    }
    for (i = 0; i <= deg; i++) c[i] = b[i];
    free(A); free(b);
}

int main(void)
{
    int degs[] = {9, 11, 13, 15, 17};
    int t;
    for (t = 0; t < 5; t++) {
        int deg = degs[t];
        double c[32];
        double maxe = 0, maxe_int = 0;
        int i, k;
        int32_t q[32];
        int32_t cmax = 0;
        fit(deg, c);
        for (k = 0; k <= 200000; k++) {
            double z = -1.0 + 2.0*k/200000.0;
            double p = c[deg];
            for (i = deg-1; i >= 0; i--) p = p*z + c[i];
            {
                double e = fabs(p - silu(8.0*z));
                if (e > maxe) maxe = e;
            }
        }
        /* exact Q12 value, z in Q14 */
        for (i = 0; i <= deg; i++) {
            q[i] = (int32_t)llround(c[i]*4096.0);
            if (abs(q[i]) > cmax) cmax = abs(q[i]);
        }
        for (k = 0; k <= 200000; k++) {
            double z = -1.0 + 2.0*k/200000.0;
            int32_t zq = (int32_t)llround(z*16384.0);
            int32_t p = q[deg];
            for (i = deg-1; i >= 0; i--) p = (int32_t)(((int64_t)p*zq) >> 14) + q[i];
            {
                double e = fabs(p/4096.0 - silu(8.0*z));
                if (e > maxe_int) maxe_int = e;
            }
        }
        printf("deg %2d: double max|err|=%.3e  Q12/Q14 int max|err|=%.3e  max|coef|=%d\n",
               deg, maxe, maxe_int, cmax);
    }
    return 0;
}
