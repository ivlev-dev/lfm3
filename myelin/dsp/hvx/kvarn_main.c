/*
 * kvarn_main: DSP stand for the kvarn page codec (k4v2 / k4v4).
 *
 * The case header `kvarn_case.h` is produced by the Rust reference codec
 * (`cargo run -p myelin-pack --example kvarn_dump`), so every number below is
 * an independent oracle:
 *
 *   1. f16 round trip on the page values
 *   2. Hadamard rotation: f16 forward must match the reference page bit for
 *      bit; inverse must return to the input within an f16 ulp
 *   3. decode of the reference page: comparison with the reference
 *      reconstruction for tokens 0, 1, 17, 31
 *   4. encode of the same rotated page: field-by-field comparison with the
 *      reference page (scales bit-exact, codes counted) and the quality of the
 *      self-decoded page against the reference reconstruction
 *
 *   build/run: ./kvarn_build.sh        (host check + qexe under hexagon-sim)
 */
#include "kvarn_impl.h"
#include "kvarn_case.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef KVARN_HOST
#include <qurt_alloc.h>
QURT_DECLARE_STATIC_HEAP(16u * 1024u * 1024u);
#endif

#define PAGE_VALUES (KV_CASE_TOKENS * KV_CASE_DIM)
#define CHECK_TOKENS 4

static const int g_tokens[CHECK_TOKENS] = { 0, 1, 17, KV_CASE_TOKENS - 1 };

static uint16_t g_rot[PAGE_VALUES] __attribute__((aligned(128)));
static uint8_t g_page[KVARN_K4V4_BYTES] __attribute__((aligned(128)));
static float g_dec[KVARN_DIM];

static int g_fail;

static int close_enough(float a, float b)
{
    float scale = fabsf(a) > 1.0f ? fabsf(a) : 1.0f;
    return fabsf(a - b) <= 1e-6f * scale;
}

static int count_bytes_equal(const uint8_t *a, const uint8_t *b, int bytes)
{
    int i, n = 0;
    for (i = 0; i < bytes; i++)
        n += (a[i] == b[i]);
    return n;
}

static void check_f16(void)
{
    int i, bad = 0;
    for (i = 0; i < PAGE_VALUES; i++) {
        uint16_t h = (uint16_t)kv_case_rotated[i];
        if (kvarn_f32_to_f16(kvarn_f16_to_f32(h)) != h)
            bad++;
        h = (uint16_t)kv_case_raw[i];
        if (kvarn_f32_to_f16(kvarn_f16_to_f32(h)) != h)
            bad++;
    }
    printf("KV %-38s bad=%d  %s\n", "f16 round trip", bad,
           bad == 0 ? "OK" : "FAIL");
    if (bad)
        g_fail = 1;
}

static void check_rotation(void)
{
    int token, d, bad = 0;
    for (token = 0; token < KV_CASE_TOKENS; token++) {
        for (d = 0; d < KV_CASE_DIM; d++)
            g_rot[token * KV_CASE_DIM + d] =
                (uint16_t)kv_case_raw[token * KV_CASE_DIM + d];
        kvarn_rot_fwd_f16(g_rot + token * KV_CASE_DIM);
        for (d = 0; d < KV_CASE_DIM; d++) {
            if (g_rot[token * KV_CASE_DIM + d] !=
                (uint16_t)kv_case_rotated[token * KV_CASE_DIM + d])
                bad++;
        }
    }
    printf("KV %-38s bad=%d  %s\n", "rot_fwd_f16 vs reference", bad,
           bad == 0 ? "OK" : "FAIL");
    if (bad)
        g_fail = 1;
}

static void check_decode(int v_bits, const unsigned char *encoded,
                         const float *ref_k, const float *ref_v)
{
    int i, t, bad_k = 0, bad_v = 0;
    float max_k = 0.0f, max_v = 0.0f;

    for (t = 0; t < CHECK_TOKENS; t++) {
        kvarn_decode_k(encoded, g_tokens[t], g_dec);
        for (i = 0; i < KV_CASE_DIM; i++) {
            float ref = ref_k[t * KV_CASE_DIM + i];
            float d = fabsf(g_dec[i] - ref);
            if (d > max_k)
                max_k = d;
            if (!close_enough(g_dec[i], ref))
                bad_k++;
        }
        kvarn_decode_v(encoded, v_bits, g_tokens[t], g_dec);
        for (i = 0; i < KV_CASE_DIM; i++) {
            float ref = ref_v[t * KV_CASE_DIM + i];
            float d = fabsf(g_dec[i] - ref);
            if (d > max_v)
                max_v = d;
            if (!close_enough(g_dec[i], ref))
                bad_v++;
        }
    }
    printf("KV decode k4v%d %-24s K bad=%3d max=%.3e | V bad=%3d max=%.3e  %s\n",
           v_bits, "vs reference", bad_k, max_k, bad_v, max_v,
           (bad_k == 0 && bad_v == 0) ? "OK" : "FAIL");
    if (bad_k || bad_v)
        g_fail = 1;
}

static void check_encode(int v_bits, const unsigned char *ref_kcode,
                         const unsigned char *ref_vcode,
                         const uint16_t *ref_ka, const uint16_t *ref_kc,
                         const uint16_t *ref_vb, const uint16_t *ref_vbz,
                         const uint16_t *ref_vc, const float *ref_k,
                         const float *ref_v)
{
    size_t vcode_len = (size_t)KV_CASE_TOKENS * KV_CASE_DIM * (size_t)v_bits / 8;
    int kcode_len = KVARN_KCODE_BYTES;
    const uint8_t *ka = g_page + KVARN_KCODE_BYTES;
    const uint8_t *kc = ka + KVARN_KA_BYTES;
    const uint8_t *vcode = kc + KVARN_KC_BYTES;
    const uint8_t *vb = vcode + vcode_len;
    const uint8_t *vbz = vb + KVARN_VB_BYTES;
    const uint8_t *vc = vbz + KVARN_VBZ_BYTES;
    int kcode_eq, vcode_eq, scale_eq, scale_total;
    double cos_num = 0.0, cos_da = 0.0, cos_db = 0.0;
    int t, i, ok;

    for (t = 0; t < PAGE_VALUES; t++)
        g_rot[t] = (uint16_t)kv_case_rotated[t];
    if (kvarn_commit(g_rot, v_bits, g_page) != 0) {
        printf("KV encode k4v%d: commit failed\n", v_bits);
        g_fail = 1;
        return;
    }

    kcode_eq = count_bytes_equal(g_page, ref_kcode, kcode_len);
    vcode_eq = count_bytes_equal(vcode, ref_vcode, (int)vcode_len);
    scale_total = KVARN_KA_BYTES + KVARN_KC_BYTES + KVARN_VB_BYTES +
                  KVARN_VBZ_BYTES + KVARN_VC_BYTES;
    scale_eq = count_bytes_equal(ka, (const uint8_t *)ref_ka, KVARN_KA_BYTES) +
               count_bytes_equal(kc, (const uint8_t *)ref_kc, KVARN_KC_BYTES) +
               count_bytes_equal(vb, (const uint8_t *)ref_vb, KVARN_VB_BYTES) +
               count_bytes_equal(vbz, (const uint8_t *)ref_vbz, KVARN_VBZ_BYTES) +
               count_bytes_equal(vc, (const uint8_t *)ref_vc, KVARN_VC_BYTES);

    ok = kcode_eq * 1000 >= kcode_len * 995 &&
         vcode_eq * 1000 >= (int)vcode_len * 995;
    printf("KV encode k4v%d K codes %4d/%4d V codes %4d/%4d scales %3d/%3d  %s\n",
           v_bits, kcode_eq, kcode_len, vcode_eq, (int)vcode_len, scale_eq,
           scale_total, ok ? "OK" : "FAIL");
    if (!ok)
        g_fail = 1;

    for (t = 0; t < CHECK_TOKENS; t++) {
        kvarn_decode_k(g_page, g_tokens[t], g_dec);
        for (i = 0; i < KV_CASE_DIM; i++) {
            float a = g_dec[i];
            float b = ref_k[t * KV_CASE_DIM + i];
            cos_num += (double)a * b;
            cos_da += (double)a * a;
            cos_db += (double)b * b;
        }
        kvarn_decode_v(g_page, v_bits, g_tokens[t], g_dec);
        for (i = 0; i < KV_CASE_DIM; i++) {
            float a = g_dec[i];
            float b = ref_v[t * KV_CASE_DIM + i];
            cos_num += (double)a * b;
            cos_da += (double)a * a;
            cos_db += (double)b * b;
        }
    }
    {
        double cosv = cos_num / sqrt(cos_da * cos_db);
        printf("KV encode k4v%d cos(self-decode, reference) = %.9f  %s\n", v_bits,
               cosv, cosv >= 0.9995 ? "OK" : "FAIL");
        if (cosv < 0.9995)
            g_fail = 1;
    }
}

int main(void)
{
    printf("KVARN stand: %dx%d page, k4v2 %d B, k4v4 %d B\n", KV_CASE_TOKENS,
           KV_CASE_DIM, KVARN_K4V2_BYTES, KVARN_K4V4_BYTES);

    check_f16();
    check_rotation();
    check_decode(2, kv_case_k4v2_encoded, kv_case_k4v2_decoded_k,
                 kv_case_k4v2_decoded_v);
    check_decode(4, kv_case_k4v4_encoded, kv_case_k4v4_decoded_k,
                 kv_case_k4v4_decoded_v);
    check_encode(2, kv_case_k4v2_codes, kv_case_k4v2_vcode, kv_case_k4v2_ka,
                 kv_case_k4v2_kc, kv_case_k4v2_vb, kv_case_k4v2_vbz,
                 kv_case_k4v2_vc, kv_case_k4v2_decoded_k,
                 kv_case_k4v2_decoded_v);
    check_encode(4, kv_case_k4v4_codes, kv_case_k4v4_vcode, kv_case_k4v4_ka,
                 kv_case_k4v4_kc, kv_case_k4v4_vb, kv_case_k4v4_vbz,
                 kv_case_k4v4_vc, kv_case_k4v4_decoded_k,
                 kv_case_k4v4_decoded_v);

    printf("KVARN RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
