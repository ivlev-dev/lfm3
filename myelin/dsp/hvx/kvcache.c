#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_ops.h"

#define LFM_COMMIT_DIM LFM_KVARN_HEAD_DIM
#define LFM_COMMIT_TOKENS LFM_KVARN_PAGE_TOKENS
#define LFM_COMMIT_VALUES (LFM_COMMIT_TOKENS * LFM_COMMIT_DIM)
#define LFM_COMMIT_F16_HEAD (2u * LFM_COMMIT_VALUES * 2u)

void lfm_hvx_kvcache_commit(lfm_kv_format fmt, uint8_t *page, const float *k_rot,
                            const float *v_rot, int head, int head_dim)
{
    uint16_t *src;
    float *scratch;
    uint8_t *tmp;
    size_t head_bytes;
    size_t voff;
    uint8_t *dst;
    int i;

    if (page == NULL || k_rot == NULL || v_rot == NULL || head < 0 || head_dim != LFM_COMMIT_DIM)
        return;

    if (fmt == LFM_KV_F16) {
        dst = page + (size_t)head * LFM_COMMIT_F16_HEAD;
        for (i = 0; i < LFM_COMMIT_VALUES; i++)
            lfm_f16_store(dst + (size_t)i * 2u, lfm_f32_to_f16(k_rot[i]));
        for (i = 0; i < LFM_COMMIT_VALUES; i++)
            lfm_f16_store(dst + (size_t)LFM_COMMIT_VALUES * 2u + (size_t)i * 2u,
                          lfm_f32_to_f16(v_rot[i]));
        return;
    }

    if (fmt != LFM_KVARN_K4V2 && fmt != LFM_KVARN_K4V4)
        return;

    head_bytes = lfm_kvarn_head_bytes(fmt);
    voff = lfm_kvarn_vcode_off(fmt);
    dst = page + (size_t)head * head_bytes;
    src = (uint16_t *)malloc((size_t)LFM_COMMIT_VALUES * sizeof(uint16_t));
    scratch = (float *)malloc((size_t)LFM_COMMIT_VALUES * sizeof(float));
    tmp = (uint8_t *)malloc(head_bytes);
    if (src == NULL || scratch == NULL || tmp == NULL) {
        free(src);
        free(scratch);
        free(tmp);
        return;
    }

    for (i = 0; i < LFM_COMMIT_VALUES; i++)
        src[i] = lfm_f32_to_f16(k_rot[i]);
    lfm_kvarn_commit_work(fmt, src, dst, scratch);

    for (i = 0; i < LFM_COMMIT_VALUES; i++)
        src[i] = lfm_f32_to_f16(v_rot[i]);
    lfm_kvarn_commit_work(fmt, src, tmp, scratch);

    memcpy(dst + voff, tmp + voff, head_bytes - voff);

    free(src);
    free(scratch);
    free(tmp);
}
