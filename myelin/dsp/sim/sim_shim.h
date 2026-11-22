#ifndef LFM_SIM_SHIM_H
#define LFM_SIM_SHIM_H

#include <stddef.h>

/*
 * QEMU/QuRT has no FastRPC and no ION: an rpcmem fd cannot be passed to the
 * DSP and there is no apps_mem driver.  The DSP image is flat-memory, though,
 * so the stand emulates the fd part of the API: the test registers a normal
 * heap buffer and gets a fake fd back; apps_mem_share_map() resolves that fd
 * to the same pointer.  This lets lfmgemv_map_arena()/alloc_view() and the
 * exec command lists run unmodified.
 *
 * Returns a non-negative fake fd, or -1 when the table is full.
 */
int sim_ion_register(void *p, size_t bytes);

#endif /* LFM_SIM_SHIM_H */
