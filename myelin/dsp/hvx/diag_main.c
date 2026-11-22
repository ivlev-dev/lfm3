#include "lfmgemv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID; um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, (uint32_t)sizeof(um));
}
int main(void)
{
    int h = -1, rc, res[8];
    if (enable_unsigned() != 0) return 10;
    if (lfmgemv_alloc_buf(1024 * 1024, &h) != 0) { printf("alloc fail\n"); return 11; }
    rc = lfmgemv_probe(h, 11, 0, res, 8);
    printf("thread create: rc=%d(%d) flag=%d tid=%d\n", rc, res[0], res[1], res[2]);
    rc = lfmgemv_probe(h, 12, 0, res, 8);
    printf("hvx worker: rc=%d create_rc=%d lock_rc=%d mode=%d op=%d done=%d\n", rc, res[0], res[1], res[2], res[3], res[4]);
    rc = lfmgemv_probe(h, 13, 8, res, 8);
    printf("sem thread (8KB,defprio): rc=%d create=%d down=%d\n", rc, res[0], res[1]);
    rc = lfmgemv_probe(h, 13, 32 | (120 << 8), res, 8);
    printf("sem thread (32KB,prio120): rc=%d create=%d down=%d\n", rc, res[0], res[1]);
    rc = lfmgemv_probe(h, 20, 0, res, 8);
    printf("HVX thread + explicit stack: create=%d tid=%d prio=%d\n", res[5], res[6], res[7]);
    rc = lfmgemv_probe(h, 19, 0, res, 8);
    printf("create(team_worker, &sem): create=%d tid=%d\n", res[5], res[6]);
    rc = lfmgemv_probe(h, 18, 0, res, 8);
    printf("create(team_worker): rc=%d res5=%d tid=%d\n", rc, res[5], res[6]);
    if (0) {
        static const int kb[] = { 8, 16, 24, 32, 48, 64 };
        int z;
        for (z = 0; z < 6; z++) {
            rc = lfmgemv_probe(h, 17, kb[z], res, 8);
            printf("worker_main stack %2d KB: create=%d tid=%d ran=%d\n", kb[z], res[5], res[6], res[7]);
        }
    }
    rc = lfmgemv_probe(h, 15, 0, res, 8);
    printf("create(struct,worker_main): rc=%d res5=%d\n", rc, res[5]);
    rc = lfmgemv_probe(h, 16, 0, res, 8);
    printf("create(local,worker_main): rc=%d res5=%d tid=%d\n", rc, res[5], res[6]);
    rc = lfmgemv_probe(h, 14, 0, res, 8);
    printf("team_start: rc=%d made=%d thr_rc=%d hvx_rc=%d\n", rc, res[5], res[6], res[7]);
    rc = lfmgemv_probe(h, 24, 0, res, 8);
    printf("vmpyiacc pos: %d,%d,%d,%d,%d (expect 250,500,750,1000,1250)\n",
           res[0], res[1], res[2], res[3], res[4]);
    rc = lfmgemv_probe(h, 25, 0, res, 8);
    printf("vmpyiacc neg: %d,%d,%d,%d,%d (expect -250,-500,...)\n",
           res[0], res[1], res[2], res[3], res[4]);
    rc = lfmgemv_probe(h, 22, 0, res, 8);
    printf("vmpyieacc (q in low half): %d,%d,%d,%d,%d,%d  (expect 500,1000,1500..)\n",
           res[0], res[1], res[2], res[3], res[4], res[5]);
    rc = lfmgemv_probe(h, 23, 0, res, 8);
    printf("vmpyieacc (q in both):     %d,%d,%d,%d,%d,%d\n",
           res[0], res[1], res[2], res[3], res[4], res[5]);
    rc = lfmgemv_probe(h, 21, 0, res, 8);
    printf("pack test: rc=%d out[5..7]=%d,%d,%d\n", rc, res[5], res[6], res[7]);
    /* expected: acc = -15000,-14000,... * 0.5 -> -7500,-7000 -> int8 saturates to -127 */
    lfmgemv_release_buf(h);
    return 0;
}
