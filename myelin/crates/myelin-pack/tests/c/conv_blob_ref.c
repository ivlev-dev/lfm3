/*
 * Bit-exact reference builder for the DSP shortconv blob.
 *
 * `conv_blob_build()` below is copied verbatim from
 * `myelin/dsp/hvx/exec_main.c` (the function the `exec` stand uses to fill the
 * `CONV` weight view).  The integer recipes come from
 * `myelin/dsp/conv/conv_math.h`, which is included by name; the test compiles
 * this file with `-I <repo>/myelin/dsp/conv` and compares the output with
 * `myelin-pack conv` byte for byte.
 *
 * Usage: conv_blob_ref <H> <s_bc> <s_y> <taps.f32> <out.bin>
 *   taps.f32: H * 7 little-endian f32, row-major [channel][tap]
 *   out.bin : 8*H*8 + 4*H bytes
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "conv_math.h"

#define CONV_TAPS 7
#define CONV_RING 8

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

int main(int argc, char **argv)
{
    conv_blob_t blob;
    float *taps;
    FILE *file;
    int H;
    int written;

    if (argc != 6) {
        fprintf(stderr, "usage: %s H s_bc s_y taps.f32 out.bin\n", argv[0]);
        return 2;
    }
    H = atoi(argv[1]);
    if (H < 16 || (H & 15) != 0) {
        fprintf(stderr, "H must be a positive multiple of 16\n");
        return 2;
    }
    taps = (float *)malloc((size_t)H * CONV_TAPS * sizeof(float));
    if (!taps) return 3;

    file = fopen(argv[4], "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", argv[4]);
        return 3;
    }
    if (fread(taps, sizeof(float), (size_t)H * CONV_TAPS, file) !=
        (size_t)H * CONV_TAPS) {
        fprintf(stderr, "short read on %s\n", argv[4]);
        fclose(file);
        return 3;
    }
    fclose(file);

    if (conv_blob_build(&blob, H, strtof(argv[2], NULL), strtof(argv[3], NULL),
                        taps) != 0) {
        fprintf(stderr, "conv_blob_build failed\n");
        return 3;
    }

    file = fopen(argv[5], "wb");
    if (!file) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 3;
    }
    written = (int)fwrite(blob.blob, 1, (size_t)8 * H * 8 + (size_t)4 * H, file);
    fclose(file);
    if (written != (int)((size_t)8 * H * 8 + (size_t)4 * H)) {
        fprintf(stderr, "short write on %s\n", argv[5]);
        return 3;
    }

    fprintf(stdout, "ksh %d\n", blob.ksh);
    free(blob.blob);
    free(blob.wq1);
    free(taps);
    return 0;
}
