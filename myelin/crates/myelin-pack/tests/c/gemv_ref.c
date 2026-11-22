/*
 * Bit-exact reference packer for the int4 GEMV weights.
 *
 * `ffn_pack_i4_row()` is included *from the stand header* by name, so a drift
 * in the quantisation recipe (`max|w|/7`, `lrintf`, clamp to [-8, 7],
 * offset-8 nibbles) fails the comparison immediately.
 *
 * Usage: gemv_ref <N> <K> <weights.f32> <weights.i4> <scales.f32>
 *   weights.f32: N * K little-endian f32, row-major [N][K]
 *   weights.i4 : N * (K / 2) bytes, row-major
 *   scales.f32 : N little-endian f32
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ffn_ref.h"

int main(int argc, char **argv)
{
    int n, k, r, i;
    FILE *f;
    float *w;
    uint8_t *packed;
    float *scales;
    size_t count;

    if (argc != 6) {
        fprintf(stderr, "usage: %s N K weights.f32 out.i4 scales.f32\n", argv[0]);
        return 2;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    if (n <= 0 || k <= 0 || (k & 1) != 0) {
        fprintf(stderr, "bad shape %d x %d\n", n, k);
        return 2;
    }
    count = (size_t)n * (size_t)k;
    w = (float *)malloc(count * sizeof(float));
    packed = (uint8_t *)malloc((size_t)n * (size_t)(k / 2));
    scales = (float *)malloc((size_t)n * sizeof(float));
    if (!w || !packed || !scales) {
        fprintf(stderr, "alloc failed\n");
        return 2;
    }
    f = fopen(argv[3], "rb");
    if (!f || fread(w, sizeof(float), count, f) != count) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 2;
    }
    fclose(f);

    for (r = 0; r < n; r++) {
        scales[r] = ffn_pack_i4_row(packed + (size_t)r * (size_t)(k / 2),
                                    w + (size_t)r * (size_t)k, k);
    }

    f = fopen(argv[4], "wb");
    if (!f || fwrite(packed, 1, (size_t)n * (size_t)(k / 2), f) !=
                  (size_t)n * (size_t)(k / 2)) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 2;
    }
    fclose(f);
    f = fopen(argv[5], "wb");
    if (!f || fwrite(scales, sizeof(float), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 2;
    }
    fclose(f);

    printf("packed %d rows x %d cols -> %zu bytes\n", n, k,
           (size_t)n * (size_t)(k / 2));
    free(w);
    free(packed);
    free(scales);
    return 0;
}
