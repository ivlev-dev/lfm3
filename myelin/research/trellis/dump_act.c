/* dump_act.c - dump REAL FFN inputs of LFM3-1.7B (dev machine only).
 *
 * Reuses the existing f32 reference forward (myelin/dsp/hvx/lfm3_fwd.h), which
 * mirrors backend/cpu.rs operation for operation, and writes the captured
 * "f = rms_norm(merged, ffn_norm)" vectors of the requested layers to a raw
 * f32 file.
 *
 *   ./dump_act <model.gguf> <out.bin> <layer-list> <ntok> [tok ...]
 *
 * out.bin layout: int32 ntok, int32 H, int32 nlayer, int32 pad, then
 *                 float32[nlayer][ntok][H].
 */
#include "ffn_ref.h"
#include "gguf_f32.h"
#include "lfm3_fwd.h"

#define HID 2048
#define INT 7680
#define NL  24

int main(int argc, char **argv)
{
    const char *path, *out;
    lfm_t spec;
    uint32_t *tok;
    float *cap;
    int top1 = -1;
    int ntok, i, t, L;
    int layers[64], nlay = 0;
    char *ls;
    FILE *f;

    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model.gguf out.bin layerlist ntokens [tok...]\n"
                "  layerlist: comma separated, e.g. 0,12,23\n", argv[0]);
        return 2;
    }
    path = argv[1];
    out = argv[2];
    ls = argv[3];
    ntok = atoi(argv[4]);
    if (ntok < 1) ntok = 1;
    if (ntok > 32) ntok = 32;

    for (char *p = ls; *p && nlay < 64;) {
        int v = (int)strtol(p, &p, 10);
        if (v < 0 || v >= NL) { fprintf(stderr, "bad layer %d\n", v); return 2; }
        layers[nlay++] = v;
        if (*p == ',') p++;
    }

    tok = malloc((size_t)ntok * 4);
    if (argc > 5) {
        int n = argc - 5;
        if (n > ntok) n = ntok;
        for (i = 0; i < n; i++) tok[i] = (uint32_t)strtoul(argv[5 + i], NULL, 10);
        ntok = n;
    } else {
        uint32_t st = 20240517u;              /* same LCG as ffn_real_study.c */
        for (t = 0; t < ntok; t++) {
            st = st * 1664525u + 1013904223u;
            tok[t] = st % 127999u;
        }
    }

    gguf_open(path);
    lfm_init(&spec, NL, HID, INT, 16, 4, 128, 256, 128000);
    cap = malloc((size_t)NL * ntok * HID * 4);
    if (!cap) { fprintf(stderr, "oom\n"); return 1; }

    fprintf(stderr, "forward: %d tokens, %d layers captured\n", ntok, nlay);
    if (lfm_forward(&spec, tok, ntok, cap, &top1) != 0) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    fprintf(stderr, "lm_head argmax(last token) = %d (sanity)\n", top1);

    f = fopen(out, "wb");
    if (!f) { perror(out); return 1; }
    int hdr[4] = { ntok, HID, nlay, 0 };
    fwrite(hdr, 4, 4, f);
    for (i = 0; i < nlay; i++) {
        L = layers[i];
        fwrite(cap + (size_t)L * ntok * HID, 4, (size_t)ntok * HID, f);
        fprintf(stderr, "  wrote layer %d FFN input (%d x %d f32)\n", L, ntok, HID);
    }
    fclose(f);
    free(cap);
    free(tok);
    return 0;
}
