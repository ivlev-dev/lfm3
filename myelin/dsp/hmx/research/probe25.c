/* probe25 — семантика HMX-стора на реальном v68 (REPORT25).
 *
 * Стенд: dsp/hmx, неизменённые libmx4_skel.so / mx4_test. Все опыты идут
 * через существующие RPC (mx4_i4, mx4_i8m, mx4_blkpref), поэтому DSP-код
 * не требуется менять.
 *
 * Сборка (dev-машина):
 *   S=/opt/hexagon-sdk-6.7.0.0
 *   HCC=/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang
 *   $HCC -O2 -w -I$S/incs -I$S/incs/stddef -I. -o probe25 probe25.c mx4_stub.c \
 *        -L$S/ipc/fastrpc/remote/ship/android_aarch64 -lcdsprpc -llog -ldl -lm \
 *        -Wl,-rpath,'$ORIGIN'
 *
 * Запуск на устройстве:
 *   adb push probe25 /data/local/tmp/hmx4/ && adb shell \
 *     'cd /data/local/tmp/hmx4 && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=$PWD ./probe25'
 */
#include "mx4.h"
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int info[24];
static unsigned char raw[4096];

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                  (uint32_t)sizeof(um));
}

static float fp16_of(unsigned w)
{
    __fp16 h;
    unsigned short s = (unsigned short)w;
    memcpy(&h, &s, 2);
    return (float)h;
}

/* ================================================================== */
/* 1. blkpref как есть: константный пакет смещён на 2 КБ + 512 Б      */
/*    (mx4_impl.c: voact = vbias + 512 + 2048).                        */
/*    Один масштаб S=1 и одна группа: u = (C+C0)*m - 0.5.              */
/* ================================================================== */
static void blk_offset_map(void)
{
    unsigned char A[64 * 32], B[512];
    float S[32], C[64 * 32];
    int C0 = 6496, rc, s, n, umin = 9999, umax = -1, hi = 0, miss = 0;
    float m = 255.0f / (2.0f * (float)C0);

    memset(A, 1, sizeof(A));
    memset(B, 0x11, sizeof(B));              /* все нибблы +1: P = 32 */
    for (n = 0; n < 32; n++) S[n] = 1.0f;
    memset(C, 0x7f, sizeof(C));
    rc = mx4_blkpref(A, sizeof(A), B, sizeof(B), (unsigned char *)S,
                     sizeof(S), 64, 32, 32, 1, C0, 257, 1,
                     (unsigned char *)C, sizeof(C), info, 24);
    if (rc || info[0]) { printf("blk: rc=%d st=%d\n", rc, info[0]); return; }
    for (s = 0; s < 64; s++)
        for (n = 0; n < 32; n++) {
            int u = (int)lroundf((C[s * 32 + n] + (float)C0) * m - 0.5f);
            if (u < umin) umin = u;
            if (u > umax) umax = u;
            if (u >= 127) hi++;
            else miss++;
        }
    printf("blk C0=%d m=%.6f P=32: u in [%d..%d]; u=127 (со смещением) %d ячеек,"
           " u=0..1 (без смещения) %d ячеек\n", C0, m, umin, umax, hi, miss);
    printf("    без смещения строки:");
    for (s = 0; s < 64; s++) {
        int bad = 0;
        for (n = 0; n < 32; n++) {
            int u = (int)lroundf((C[s * 32 + n] + (float)C0) * m - 0.5f);
            if (u < 127) bad++;
        }
        if (bad) printf(" s=%d(%d/32)", s, bad);
    }
    printf("\n");
}

/* ================================================================== */
/* 2. Маскирование адреса активации до 2 КБ: aoff=512 не действует.    */
/*    Два int4-пакета (K=64): константа Aq -> P0=224*Aq, реальный q.    */
/* ================================================================== */
static int i4_two_packets(int aoff, int Aq, int q, int word,
                          unsigned char *out)
{
    static unsigned char A[64 * 64], B[32 * 64];
    int s, n, k, rc;
    for (s = 0; s < 64; s++) {
        for (k = 0; k < 32; k++) A[s * 64 + k] = (unsigned char)Aq;
        for (k = 32; k < 64; k++) A[s * 64 + k] = (unsigned char)(s * 3 + 1);
    }
    for (n = 0; n < 32; n++) {
        for (k = 0; k < 32; k++) B[n * 64 + k] = 7;
        for (k = 32; k < 64; k++) B[n * 64 + k] = (unsigned char)(char)q;
    }
    memset(raw, 0, sizeof(raw));
    rc = mx4_i4(A, sizeof(A), B, sizeof(B), 64, 32, 64, 4, 0x3FF, 511,
                word, 8 /* M=64, .ub */, 1, 0, 512 /* wstep */, aoff, 0,
                raw, 4096, info, 24);
    if (rc || info[0]) { printf("i4: rc=%d st=%d\n", rc, info[0]); return 1; }
    memcpy(out, raw, 2048);
    return 0;
}

static void mask_probe(void)
{
    unsigned char r0[2048], r1[2048];
    if (i4_two_packets(0, 29, 1, 0x4906, r0)) return;
    if (i4_two_packets(512, 29, 1, 0x4906, r1)) return;
    printf("mask aoff=0   vs aoff=512: %s  (первые 8:",
           memcmp(r0, r1, 2048) ? "ОТЛИЧАЮТСЯ" : "байт-в-байт РАВНЫ");
    for (int s = 0; s < 8; s++) printf(" %u", r0[128 * (s >> 2) + (s & 3)]);
    printf(")\n");
}

/* ================================================================== */
/* 3. Правильный 8-битный рецепт: 2 пакета по 2 КБ-выровненным адресам */
/* ================================================================== */
static void rec8(int Aq, int q, int word)
{
    unsigned char r[2048];
    int s, nbad = 0, nclip = 0;
    int C0 = 224 * Aq;
    float m = fp16_of((unsigned)word) / 512.0f;

    if (i4_two_packets(0, Aq, q, word, r)) return;
    for (s = 0; s < 64; s++) {
        int want = (int)((C0 + 32 * (s * 3 + 1) * q) * m);
        int got = r[128 * (s >> 2) + (s & 3)];
        if (want > 255) { want = 255; nclip++; }
        if (got != want) nbad++;
    }
    printf("rec8 int4 Aq=%2d q=%2d C0=%4d word=%#04x m=%.6f -> расхождений "
           "%d/64 (клип %d), первые 8:", Aq, q, C0, word, m, nbad, nclip);
    for (s = 0; s < 8; s++) printf(" %u", r[128 * (s >> 2) + (s & 3)]);
    printf("\n");
}

/* ================================================================== */
/* 4. .ub: trunc + clamp(:sat) / wrap(без :sat), покрытие 64 строк     */
/*    acc = 32 * ncr * w  (A=1, B=w: +1 или -1)                        */
/* ================================================================== */
static int ub_probe(int ncr, int w, int srt, int word)
{
    static unsigned char A[64 * 2048], B[32 * 2048];
    int rc, K = 32 * ncr;
    memset(A, 1, sizeof(A));
    memset(B, (unsigned char)(w < 0 ? 0xFF : w), sizeof(B));
    memset(raw, 0, sizeof(raw));
    rc = mx4_i8m(A, sizeof(A), B, sizeof(B), 64, 32, K, 3, 3, 0x3FF, 1023,
                 srt, 0, 0, word, 0, 1, raw, 4096, info, 24);
    if (rc || info[0]) return -1;
    return raw[0];
}

static void ub_table(int word)
{
    static const int ncr[] = {1, 4, 8, 9, 1, 2};
    static const int wv[]  = {1, 1, 1, 1, -1, -3};
    float m = fp16_of((unsigned)word) / 512.0f;
    int i, k, nz = 0;
    for (i = 0; i < 6; i++) {
        int acc = 32 * ncr[i] * wv[i];
        int sat = ub_probe(ncr[i], wv[i], 0, word);
        int nos = ub_probe(ncr[i], wv[i], 2, word);
        printf("ub word=%#04x m=%.4f acc=%4d trunc=%7.1f -> :sat %3d | nosat %3d\n",
               word, m, acc, (float)acc * m, sat, nos);
    }
    /* покрытие: все единицы -> все 64 строки x 32 n */
    ub_probe(1, 1, 0, 0x6000);
    for (k = 0; k < 2048; k++) if (raw[k]) nz++;
    printf("   покрытие .ub при A=B=1: непустых байт %d/2048 = %d строк x 32 n,"
           " строка s читается по байту 128*(s>>2)+(s&3)\n", nz, nz / 32);
}

/* ================================================================== */
/* 5. .uh: какие строки, какой layout, множитель, знак                */
/* ================================================================== */
static void uh_probe(int mode, int val, int word)
{
    static unsigned char A[64 * 32], B[32 * 32];
    int rc, s, n, k, cnt = 0;
    float m = fp16_of((unsigned)word) / 2.0f;
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    for (s = 0; s < 64; s++) A[s * 32] = (unsigned char)(mode ? val : s + 1);
    for (n = 0; n < 32; n++) B[n * 32] = 1;
    memset(raw, 0, sizeof(raw));
    rc = mx4_i8m(A, sizeof(A), B, sizeof(B), 64, 32, 32, 3, 3, 0x3FF, 1023,
                 1, 0, 0, word, 0, 1, raw, 4096, info, 24);
    if (rc || info[0]) { printf("uh: rc=%d st=%d\n", rc, info[0]); return; }
    for (k = 0; k < 2048; k++) if (raw[k]) cnt++;
    printf("uh word=%#04x (%.1f на единицу acc): u16[0..5] =", word, m);
    for (k = 0; k < 6; k++) printf(" %d", raw[2 * k] | (raw[2 * k + 1] << 8));
    printf("  (byte 128: %d, byte 256: %d), непустых байт %d/2048\n",
           raw[128] | (raw[129] << 8), raw[256] | (raw[257] << 8), cnt);
    printf("   строка -> первый байт:");
    for (s = 0; s < 64; s++) {
        int want = (int)((float)(s + 1) * m), off = -1;
        for (k = 0; k < 1024; k++)
            if ((raw[2 * k] | (raw[2 * k + 1] << 8)) == want) { off = 2 * k; break; }
        if (off >= 0) printf(" %d:%d", s, off);
    }
    printf("\n");
}

static void uh_model(void)
{
    static const int accs[] = {1, 3, 100, 128, 255, 1000, 2000, -1, -128};
    int i, k, nz = 0;
    printf("uh модель (row s=1, word=0x6000):");
    for (i = 0; i < (int)(sizeof(accs) / sizeof(accs[0])); i++) {
        static unsigned char A[64 * 32], B[32 * 32];
        int rc, n;
        memset(A, 0, sizeof(A));
        memset(B, 0, sizeof(B));
        A[32] = (unsigned char)(accs[i] < 0 ? -accs[i] : accs[i]);
        for (n = 0; n < 32; n++) B[n * 32] = (unsigned char)(accs[i] < 0 ? 0xFF : 1);
        memset(raw, 0, sizeof(raw));
        rc = mx4_i8m(A, sizeof(A), B, sizeof(B), 64, 32, 32, 3, 3, 0x3FF, 1023,
                     1, 0, 0, 0x6000, 0, 1, raw, 4096, info, 24);
        if (rc || info[0]) continue;
        printf(" %d->%d", accs[i], raw[0] | (raw[1] << 8));
    }
    printf("\n");
}

/* ================================================================== */
/* 6. Bias-слово: множитель стора = fp16(word)/512                    */
/* ================================================================== */
static void bias_table(void)
{
    static const unsigned words[] = {0x6000, 0x7000, 0x5800, 0x5000, 0x3C00,
                                     0x4906, 0x5C00, 0x6200};
    int i;
    printf("bias (acc=32, .ub):");
    for (i = 0; i < (int)(sizeof(words) / sizeof(words[0])); i++)
        printf(" %#06x:fp16=%.3f->%d", words[i], fp16_of(words[i]),
               ub_probe(1, 1, 0, (int)words[i]) * 1);
    printf("\n");
}

int main(void)
{
    printf("unsigned PD: rc=%d\n", enable_unsigned());
    printf("== 1. blkpref: константный пакет по адресу 2 КБ+512 ==\n");
    blk_offset_map();
    printf("== 2. маскирование адреса активации ==\n");
    mask_probe();
    printf("== 3. правильный 8-битный рецепт (int4, 2 КБ-выравнивание) ==\n");
    rec8(29, 1, 0x4906);
    rec8(29, -1, 0x4906);
    rec8(2, 7, 0x5c8e);
    rec8(1, 7, 0x5c8e);
    printf("== 4. .ub: диапазон, sat/wrap, покрытие строк ==\n");
    ub_table(0x6000);
    ub_table(0x5C00);
    printf("== 5. .uh: layout, покрытие, множитель, знак ==\n");
    uh_probe(0, 0, 0x6000);
    uh_model();
    printf("== 6. bias-слово ==\n");
    bias_table();
    return 0;
}
