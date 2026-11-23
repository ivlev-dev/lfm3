#include "mx4.h"

#include <stdlib.h>
#include <string.h>

#include "hmx_ops.h"
#include "hmx_util.h"

#define MX4_MAX_BUF 32
#define MX4_OK 0

typedef struct {
    uint8_t *data;
    size_t bytes;
} mx4_buf;

static mx4_buf g_buf[MX4_MAX_BUF];
static lfm_hmx_vtcm g_vtcm;
static int g_ready;

static void buf_clear(int id)
{
    free(g_buf[id].data);
    g_buf[id].data = NULL;
    g_buf[id].bytes = 0;
}

int mx4_init(int vtcm_bytes)
{
    int rc;
    if (g_ready)
        return MX4_OK;
    if (vtcm_bytes <= 0)
        vtcm_bytes = (int)lfm_hmx_prefill_vtcm_bytes();
    rc = lfm_hmx_vtcm_alloc(&g_vtcm, (size_t)vtcm_bytes);
    if (rc != 0)
        return rc;
    rc = lfm_hmx_power(1);
    if (rc != 0) {
        lfm_hmx_vtcm_free(&g_vtcm);
        return rc;
    }
    g_ready = 1;
    return MX4_OK;
}

int mx4_load_buf(int id, const unsigned char *data, int dataLen)
{
    if (id < 0 || id >= MX4_MAX_BUF || data == NULL || dataLen <= 0)
        return -1;
    buf_clear(id);
    g_buf[id].data = (uint8_t *)malloc((size_t)dataLen);
    if (g_buf[id].data == NULL)
        return -2;
    memcpy(g_buf[id].data, data, (size_t)dataLen);
    g_buf[id].bytes = (size_t)dataLen;
    return MX4_OK;
}

int mx4_gemm_i4(int wid, const unsigned char *act, int actLen, const float *ascale,
                int ascaleLen, int m, int n, int k, float *y, int yLen)
{
    int rc;
    if (!g_ready)
        return -1;
    if (wid < 0 || wid >= MX4_MAX_BUF || g_buf[wid].data == NULL)
        return -2;
    if (act == NULL || ascale == NULL || y == NULL)
        return -3;
    if (m <= 0 || n <= 0 || k <= 0 || (k % LFM_HMX_KC) != 0)
        return -4;
    if ((size_t)actLen < (size_t)m * (size_t)k)
        return -5;
    if ((size_t)ascaleLen < (size_t)m * (size_t)(k / LFM_HMX_KC))
        return -6;
    if ((size_t)yLen < (size_t)m * (size_t)n)
        return -7;
    rc = lfm_hmx_lock();
    if (rc != 0)
        return rc;
    lfm_hmx_gemm_i4(g_buf[wid].data, n, k, (const int8_t *)act, ascale, m, y);
    lfm_hmx_unlock();
    return MX4_OK;
}

int mx4_gemm_i8(int wid, const float *wscale, int wscaleLen, const unsigned char *act,
                int actLen, const float *ascale, int ascaleLen, int m, int n, int k,
                float *y, int yLen)
{
    int rc;
    if (!g_ready)
        return -1;
    if (wid < 0 || wid >= MX4_MAX_BUF || g_buf[wid].data == NULL)
        return -2;
    if (wscale == NULL || act == NULL || ascale == NULL || y == NULL)
        return -3;
    if (m <= 0 || n <= 0 || k <= 0 || (k % LFM_HMX_KC) != 0)
        return -4;
    if ((size_t)wscaleLen < (size_t)n * (size_t)(k / LFM_HMX_KC))
        return -5;
    if ((size_t)actLen < (size_t)m * (size_t)k)
        return -6;
    if ((size_t)ascaleLen < (size_t)m * (size_t)(k / LFM_HMX_KC))
        return -7;
    if ((size_t)yLen < (size_t)m * (size_t)n)
        return -8;
    rc = lfm_hmx_lock();
    if (rc != 0)
        return rc;
    lfm_hmx_gemm_i8((const int8_t *)g_buf[wid].data, wscale, n, k, (const int8_t *)act, ascale,
                    m, y);
    lfm_hmx_unlock();
    return MX4_OK;
}

int mx4_prefill(int wid, const unsigned char *act, int actLen, const float *ascale,
                int ascaleLen, int m, int n, int k, float *y, int yLen)
{
    lfm_hmx_layer_i4 layer;
    int rc;
    if (!g_ready)
        return -1;
    if (wid < 0 || wid >= MX4_MAX_BUF || g_buf[wid].data == NULL)
        return -2;
    if (act == NULL || ascale == NULL || y == NULL)
        return -3;
    if (m <= 0 || n <= 0 || k <= 0 || (k % LFM_HMX_KC) != 0)
        return -4;
    if ((size_t)actLen < (size_t)m * (size_t)k)
        return -5;
    if ((size_t)ascaleLen < (size_t)m * (size_t)(k / LFM_HMX_KC))
        return -6;
    if ((size_t)yLen < (size_t)m * (size_t)n)
        return -7;
    layer.weights = g_buf[wid].data;
    layer.ascale = ascale;
    layer.n = n;
    layer.k = k;
    rc = lfm_hmx_lock();
    if (rc != 0)
        return rc;
    rc = lfm_hmx_prefill_i4(&layer, (const int8_t *)act, m, y, &g_vtcm);
    lfm_hmx_unlock();
    return rc;
}

int mx4_reset(void)
{
    int i;
    for (i = 0; i < MX4_MAX_BUF; i++)
        buf_clear(i);
    if (g_ready) {
        lfm_hmx_power(0);
        lfm_hmx_vtcm_free(&g_vtcm);
        g_ready = 0;
    }
    return MX4_OK;
}
