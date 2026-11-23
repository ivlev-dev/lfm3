#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/fp16.h"
#include "../common/kvarn.h"
#include "../common/page.h"
#include "../common/q4.h"
#include "../common/quant.h"
#include "../common/rot.h"
#include "kvarn_case.h"

static int failures;

#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                            \
            printf(__VA_ARGS__);                                                                   \
            printf("\n");                                                                          \
        }                                                                                          \
    } while (0)

static void test_fp16(void)
{
    int i;
    for (i = 0; i < KV_CASE_TOKENS * KV_CASE_DIM; i++) {
        float f = lfm_f16_to_f32(kv_case_raw[i]);
        uint16_t back = lfm_f32_to_f16(f);
        CHECK(back == kv_case_raw[i], "fp16 roundtrip %d: %04x -> %04x", i, kv_case_raw[i], back);
    }
    CHECK(lfm_f32_to_f16(1.0f) == 0x3C00u, "one");
    CHECK(lfm_f32_to_f16(-2.0f) == 0xC000u, "-2");
    CHECK(lfm_f32_to_f16(65504.0f) == 0x7BFFu, "max");
    CHECK(lfm_f32_to_f16(1e9f) == 0x7C00u, "inf");
    CHECK(lfm_f32_to_f16(1e-9f) == 0x0000u, "underflow");
}

static void test_rot(void)
{
    uint16_t buf[KV_CASE_TOKENS * KV_CASE_DIM];
    int t;
    int d;
    memcpy(buf, kv_case_raw, sizeof buf);
    lfm_rot_fwd_page_f16(buf, KV_CASE_TOKENS, KV_CASE_DIM);
    t = memcmp(buf, kv_case_rotated, sizeof buf);
    CHECK(t == 0, "rotated page differs");
    if (t != 0) {
        for (d = 0; d < KV_CASE_TOKENS * KV_CASE_DIM; d++) {
            if (buf[d] != kv_case_rotated[d]) {
                printf("  first diff %d: %04x != %04x\n", d, buf[d], kv_case_rotated[d]);
                break;
            }
        }
    }
}

static void check_page(const char *tag, lfm_kv_format fmt, const unsigned char *encoded,
                       size_t bytes, const float *gk, const float *gv)
{
    uint8_t out[8192];
    float k[LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM];
    float v[LFM_KVARN_PAGE_TOKENS * LFM_KVARN_HEAD_DIM];
    static const int tokens[4] = {0, 1, 17, LFM_KVARN_PAGE_TOKENS - 1};
    int i;
    size_t want = lfm_kvarn_head_bytes(fmt);
    CHECK(want == bytes, "%s head bytes %zu != %zu", tag, want, bytes);
    if (want > sizeof out)
        return;
    memset(out, 0, sizeof out);
    lfm_kvarn_commit(fmt, kv_case_rotated, out);
    CHECK(memcmp(out, encoded, bytes) == 0, "%s encoded page differs", tag);
    if (memcmp(out, encoded, bytes) != 0) {
        for (i = 0; i < (int)bytes; i++) {
            if (out[i] != encoded[i]) {
                printf("  first diff %d: %02x != %02x\n", i, out[i], encoded[i]);
                break;
            }
        }
    }
    lfm_kvarn_decode_page(fmt, encoded, k, v);
    for (i = 0; i < 4; i++) {
        int t = tokens[i];
        int d;
        for (d = 0; d < LFM_KVARN_HEAD_DIM; d++) {
            CHECK(fabsf(k[t * LFM_KVARN_HEAD_DIM + d] - gk[i * LFM_KVARN_HEAD_DIM + d]) <= 1e-6f,
                  "%s decode_k token %d dim %d", tag, t, d);
            CHECK(fabsf(v[t * LFM_KVARN_HEAD_DIM + d] - gv[i * LFM_KVARN_HEAD_DIM + d]) <= 1e-6f,
                  "%s decode_v token %d dim %d", tag, t, d);
        }
    }
}

static void test_kvarn(void)
{
    CHECK(lfm_kvarn_page_bytes(LFM_KVARN_K4V2, 4) == 15104, "k4v2 page bytes");
    CHECK(lfm_kvarn_page_bytes(LFM_KVARN_K4V4, 4) == 19200, "k4v4 page bytes");
    check_page("k4v2", LFM_KVARN_K4V2, kv_case_k4v2_encoded, KV_CASE_K4V2_BYTES,
               kv_case_k4v2_decoded_k, kv_case_k4v2_decoded_v);
    check_page("k4v4", LFM_KVARN_K4V4, kv_case_k4v4_encoded, KV_CASE_K4V4_BYTES,
               kv_case_k4v4_decoded_k, kv_case_k4v4_decoded_v);
}

static float dot_f32(const float *a, const float *b, int n)
{
    float s = 0.0f;
    int i;
    for (i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static float cos_f32(const float *a, const float *b, int n)
{
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f)
        return 0.0f;
    return dot / sqrtf(na * nb);
}

static void test_q4(void)
{
    enum { N = 64, K = 512, M = 4 };
    float *w = malloc((size_t)N * K * sizeof(float));
    float *x = malloc((size_t)M * K * sizeof(float));
    float *xr = malloc((size_t)M * K * sizeof(float));
    float *y_ref = malloc((size_t)M * N * sizeof(float));
    float *y = malloc((size_t)M * N * sizeof(float));
    float *y_gemm = malloc((size_t)M * N * sizeof(float));
    uint8_t *img = malloc(lfm_q4_image_bytes(N, K));
    int8_t *q = malloc((size_t)M * K);
    float *ascale = malloc((size_t)M * (K / 32) * sizeof(float));
    unsigned seed = 12345u;
    int r;
    int t;
    int i;
    CHECK(w && x && xr && y_ref && y && y_gemm && img && q && ascale, "alloc");
    if (!(w && x && xr && y_ref && y && y_gemm && img && q && ascale))
        return;
    for (i = 0; i < N * K; i++) {
        seed = seed * 1664525u + 1013904223u;
        w[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 0.7f;
    }
    for (i = 0; i < M * K; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 1.5f;
        if (i % K == 7)
            x[i] += 4.0f;
    }
    memset(img, 0, lfm_q4_image_bytes(N, K));
    lfm_q4_pack_matrix(w, N, K, img, 128);
    for (t = 0; t < M; t++) {
        for (i = 0; i < K; i++)
            xr[t * K + i] = x[t * K + i];
        lfm_rot_fwd_blocks_f32(xr + t * K, K, 128);
        lfm_quant_i8_group32(xr + t * K, K, q + t * K, ascale + t * (K / 32));
        for (r = 0; r < N; r++)
            y_ref[t * N + r] = lfm_q4_row_dot_i8(
                lfm_q4_row_qs(img, N, K, r), lfm_q4_row_scales(img, N, K, r), q + t * K,
                ascale + t * (K / 32), K);
    }
    lfm_q4_gemv_i8(img, N, K, q, ascale, y);
    lfm_q4_gemm_i8(img, N, K, q, M, ascale, y_gemm);
    for (r = 0; r < N; r++)
        CHECK(cos_f32(y + r, y_ref + r, 1) == 1.0f, "gemv matches row dot %d", r);
    for (t = 0; t < M; t++)
        for (r = 0; r < N; r++)
            CHECK(y_gemm[t * N + r] == y_ref[t * N + r], "gemm/gemv %d %d", t, r);
    {
        float *ref = malloc((size_t)M * N * sizeof(float));
        if (ref) {
            for (t = 0; t < M; t++)
                for (r = 0; r < N; r++)
                    ref[t * N + r] = dot_f32(w + (size_t)r * K, x + (size_t)t * K, K);
            for (t = 0; t < M; t++) {
                float c = cos_f32(y_gemm + t * N, ref + t * N, N);
                CHECK(c >= 0.99f, "int4+hadamard cos %.6f (token %d)", c, t);
            }
            free(ref);
        }
    }
    free(w);
    free(x);
    free(xr);
    free(y_ref);
    free(y);
    free(y_gemm);
    free(img);
    free(q);
    free(ascale);
}

static void test_rot_orthogonal(void)
{
    float a[128];
    float b[128];
    float a0[128];
    float b0[128];
    int i;
    for (i = 0; i < 128; i++) {
        a[i] = sinf(0.11f * (float)i) + 0.3f * cosf(0.37f * (float)i);
        b[i] = cosf(0.07f * (float)i) - 0.2f * sinf(0.53f * (float)i);
        a0[i] = a[i];
        b0[i] = b[i];
    }
    lfm_rot_fwd_f32(a, 128);
    lfm_rot_fwd_f32(b, 128);
    CHECK(fabsf(dot_f32(a0, b0, 128) - dot_f32(a, b, 128)) < 1e-4f, "rotation dot invariant");
    lfm_rot_inv_f32(a, 128);
    for (i = 0; i < 128; i++)
        CHECK(fabsf(a[i] - a0[i]) < 1e-5f, "rotation inverse %d", i);
}

static void test_page(void)
{
    CHECK(lfm_max_pages(2048, 2) == 128, "max_pages");
    CHECK(lfm_kv_stage(100) == 128, "kv_stage");
    CHECK(lfm_kv_stage(10) == 64, "kv_stage min");
    CHECK(lfm_page_of(65) == 2, "page_of");
    CHECK(lfm_slot_of(3, 65) == 3 * 32 + 1, "slot_of");
    CHECK(lfm_window_start(1000, 256) == 745, "window start");
    CHECK(lfm_window_start(10, 256) == 0, "window start clamp");
    CHECK(lfm_page_bytes(LFM_KV_F16, 4, 128) == 65536, "f16 page");
    CHECK(lfm_page_bytes(LFM_KVARN_K4V2, 4, 128) == 15104, "kvarn page");
    CHECK(lfm_kv_arena_bytes(11, 128, LFM_KVARN_K4V2, 4, 128) == 21266432ull, "kv arena");
}

static void test_rot_blob(void)
{
    float row[128];
    float row_rot[128];
    float x[128];
    float x_rot[128];
    int i;
    float lhs;
    float rhs;
    for (i = 0; i < 128; i++) {
        row[i] = sinf(0.021f * (float)i) + (i == 3 ? 3.0f : 0.0f);
        x[i] = cosf(0.043f * (float)i) - (i == 9 ? 1.5f : 0.0f);
        row_rot[i] = row[i];
        x_rot[i] = x[i];
    }
    lfm_rot_fwd_f32(row_rot, 128);
    lfm_rot_fwd_f32(x_rot, 128);
    lhs = dot_f32(row, x, 128);
    rhs = dot_f32(row_rot, x_rot, 128);
    CHECK(fabsf(lhs - rhs) < 1e-4f, "rot dot %f != %f", lhs, rhs);
}

int main(void)
{
    test_fp16();
    test_rot();
    test_kvarn();
    test_q4();
    test_rot_orthogonal();
    test_rot_blob();
    test_page();
    if (failures) {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
    printf("common: all checks passed\n");
    return 0;
}
