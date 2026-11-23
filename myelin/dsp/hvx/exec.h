#ifndef LFM_EXEC_H
#define LFM_EXEC_H

#include <stddef.h>
#include <stdint.h>

#include "../common/page.h"

#define LFM_EXEC_MAX_LAYERS 64
#define LFM_EXEC_CONV_TAPS 7
#define LFM_EXEC_CONV_HISTORY 6
#define LFM_EXEC_CHUNK 32
#define LFM_EXEC_PAGE_NONE 0xFFFFFFFFu

#define LFM_EXEC_OK 0
#define LFM_EXEC_ERR_ARGS (-1)
#define LFM_EXEC_ERR_MODEL (-2)
#define LFM_EXEC_ERR_MEM (-3)
#define LFM_EXEC_ERR_STATE (-4)

typedef enum {
    LFM_LAYER_CONV = 0,
    LFM_LAYER_FULL = 1,
    LFM_LAYER_SLIDING = 2
} lfm_layer_kind;

typedef enum {
    LFM_T_ATTN_NORM = 0,
    LFM_T_POST_ATTN_NORM = 1,
    LFM_T_FFN_NORM = 2,
    LFM_T_POST_FFW_NORM = 3,
    LFM_T_FFN_GATE = 4,
    LFM_T_FFN_UP = 5,
    LFM_T_FFN_DOWN = 6,
    LFM_T_ATTN_Q = 7,
    LFM_T_ATTN_K = 8,
    LFM_T_ATTN_V = 9,
    LFM_T_ATTN_GATE = 10,
    LFM_T_ATTN_OUT = 11,
    LFM_T_Q_NORM = 12,
    LFM_T_K_NORM = 13,
    LFM_T_CONV_IN = 14,
    LFM_T_CONV_W = 15,
    LFM_T_CONV_OUT = 16,
    LFM_T_COUNT = 17
} lfm_tensor_kind;

typedef struct {
    const float *attn_norm;
    const float *post_attn_norm;
    const float *ffn_norm;
    const float *post_ffw_norm;
    const uint8_t *ffn_gate;
    const uint8_t *ffn_up;
    const uint8_t *ffn_down;
    const uint8_t *attn_q;
    const uint8_t *attn_k;
    const uint8_t *attn_v;
    const uint8_t *attn_gate;
    const uint8_t *attn_out;
    const float *q_norm;
    const float *k_norm;
    const uint8_t *conv_in;
    const float *conv_w;
    const uint8_t *conv_out;
} lfm_exec_layer;

typedef struct {
    const float *embed;
    const float *out_norm;
    const float *lm_head;
    uint32_t n_layers;
    uint32_t layer_kind[LFM_EXEC_MAX_LAYERS];
    lfm_exec_layer layer[LFM_EXEC_MAX_LAYERS];
} lfm_exec_model;

int lfm_hvx_exec_bind(const lfm_exec_model *model);
void lfm_hvx_exec_unbind(void);
void lfm_hvx_exec_release(void);

size_t lfm_exec_arena_bytes(const lfm_model_desc *desc);
void lfm_exec_init_tables(const lfm_model_desc *desc, void *arena);
uint8_t *lfm_exec_kv_pages(const lfm_model_desc *desc, void *arena, uint32_t layer);
uint32_t *lfm_exec_table(const lfm_model_desc *desc, void *arena, uint32_t seq);
uint32_t *lfm_exec_tokens(const lfm_model_desc *desc, void *arena, uint32_t row);
float *lfm_exec_conv_state(const lfm_model_desc *desc, void *arena, uint32_t seq);
uint8_t *lfm_exec_kv_stage(const lfm_model_desc *desc, void *arena, uint32_t seq, uint32_t layer,
                           uint32_t head);

#endif
