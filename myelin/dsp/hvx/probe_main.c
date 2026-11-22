#include "lfmgemv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}

int main(int argc, char **argv)
{
    int h = -1, rc, k;
    int res[8];
    int N = 7680, K = 2048;
    size_t wbytes = (size_t)N * (K / 2);

    if (enable_unsigned() != 0) return 10;
    rc = lfmgemv_alloc_buf((int)wbytes, &h);
    if (rc) { printf("alloc rc=%d\n", rc); return 11; }
    lfmgemv_set_shape(h, N, K, 0);

    for (k = 0; k < 8; k++) {
        int arg = (k == 0 || k == 1) ? 100000 : ((k == 6) ? 64 : ((k == 7) ? 8 : 0));
        rc = lfmgemv_probe(h, k, arg, res, 8);
        printf("probe %d rc=%d pcycles=%d us=%d sum=%d extra=%d arg=%d\n",
               k, rc, res[0], res[1], res[2], res[3], res[4]);
    }
    /* kind 0/1: cycles per HVX op */
    printf("\nnote: kind0/1 arg=100000 iterations\n");
    lfmgemv_release_buf(h);
    return 0;
}
