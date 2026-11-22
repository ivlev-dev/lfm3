/* How few segments/coefficients give <=1e-3 for SiLU on [-8,8]? */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
static double silu(double x){ return x/(1.0+exp(-x)); }

/* least squares poly fit of degree d on [lo,hi] via normal equations */
static int fitpoly(double lo, double hi, int d, double *c)
{
    int n = 200, i, j, k, m = d + 1;
    double *A = calloc(m*m, sizeof(double)), *b = calloc(m, sizeof(double));
    for (k = 0; k < n; k++) {
        double x = lo + (hi-lo)*k/(n-1), y = silu(x), pw[16];
        pw[0] = 1; for (j = 1; j < m; j++) pw[j] = pw[j-1]*x;
        for (i = 0; i < m; i++) { b[i] += pw[i]*y; for (j = 0; j < m; j++) A[i*m+j] += pw[i]*pw[j]; }
    }
    for (i = 0; i < m; i++) {
        double p = A[i*m+i]; int j2;
        if (fabs(p) < 1e-12) { free(A); free(b); return -1; }
        for (j2 = i; j2 < m; j2++) A[i*m+j2] /= p;
        b[i] /= p;
        for (k = 0; k < m; k++) if (k != i) {
            double f = A[k*m+i]; int j2;
            for (j2 = i; j2 < m; j2++) A[k*m+j2] -= f*A[i*m+j2];
            b[k] -= f*b[i];
        }
    }
    for (i = 0; i < m; i++) c[i] = b[i];
    free(A); free(b);
    return 0;
}

int main(void)
{
    int segs[] = {2, 4, 8};
    int degs[] = {4, 5, 6, 7, 8};
    int si, di;
    printf("%-6s %-5s %-10s %-10s\n", "segs", "deg", "max|err|", "coefmax");
    for (si = 0; si < 3; si++) {
        for (di = 0; di < 5; di++) {
            int n = segs[si], d = degs[di], s, i;
            double w = 16.0/n, maxe = 0, cmax = 0;
            double c[16];
            for (s = 0; s < n; s++) {
                double lo = -8.0 + s*w, hi = lo + w;
                if (fitpoly(lo, hi, d, c) != 0) { printf("%d %d FAIL\n", n, d); break; }
                for (i = 0; i <= d; i++) if (fabs(c[i]) > cmax) cmax = fabs(c[i]);
                for (i = 0; i <= 4000; i++) {
                    double x = lo + w*i/4000.0, p = c[d]; int jj;
                    for (jj = d-1; jj >= 0; jj--) p = p*x + c[jj];
                    { double e = fabs(p - silu(x)); if (e > maxe) maxe = e; }
                }
            }
            printf("%-6d %-5d %-10.3e %-10.1f\n", n, d, maxe, cmax);
        }
    }
    return 0;
}
