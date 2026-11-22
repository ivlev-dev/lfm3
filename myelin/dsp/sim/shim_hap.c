/*
 * HAP / apps_mem shims for the local QEMU stand.
 *
 * lfmgemv_impl.c only needs HAP_power_* and apps_mem_share_map/unmap at link
 * time; HAP_perf_get_time_us()/get_pcycles() are static inline (C31:30 and
 * C15:14 reads), and the file does not call FARF at all, so no logging shim
 * is required.  Under QuRT in QEMU there is no HAP power driver and no
 * apps_mem/ION driver: the functions below keep the kernel code path intact
 * and report plausible constants.
 */
#include "sim_shim.h"

#include "HAP_power.h"
#include "apps_mem.h"
#include "AEEStdErr.h"

#include <stdint.h>
#include <string.h>

#define SIM_MAX_MAPS 16

typedef struct {
    void  *base;
    size_t bytes;
} sim_map_t;

static sim_map_t g_map[SIM_MAX_MAPS];

int sim_ion_register(void *p, size_t bytes)
{
    int i;

    if (p == 0 || bytes == 0) return -1;
    for (i = 0; i < SIM_MAX_MAPS; i++) {
        if (g_map[i].base == 0) {
            g_map[i].base = p;
            g_map[i].bytes = bytes;
            return i;
        }
    }
    return -1;
}

static const sim_map_t *sim_map_find(int fd, int size)
{
    const sim_map_t *m;

    if (fd < 0 || fd >= SIM_MAX_MAPS || g_map[fd].base == 0) return 0;
    m = &g_map[fd];
    if (size < 0 || (size_t)size > m->bytes) return 0;
    return m;
}

/* Fake ION mapping: the buffer is already DSP-accessible (flat memory). */
int apps_mem_share_map(int fd, int size, uint64 *vapps, uint64 *vadsp)
{
    const sim_map_t *m = sim_map_find(fd, size);

    if (m == 0 || vapps == 0 || vadsp == 0) return AEE_EBADPARM;
    *vapps = (uint64)(uintptr_t)m->base;
    *vadsp = (uint64)(uintptr_t)m->base;
    return 0;
}

int apps_mem_share_unmap(uint64 vadsp, int size)
{
    int i;

    (void)size;
    for (i = 0; i < SIM_MAX_MAPS; i++)
        if (g_map[i].base != 0 && (uint64)(uintptr_t)g_map[i].base == vadsp) return 0;
    return AEE_EBADPARM;
}

/* HAP power: success without touching hardware. */
int HAP_power_set(void *context, HAP_power_request_t *request)
{
    (void)context;
    (void)request;
    return 0;
}

int HAP_power_get(void *context, HAP_power_response_t *response)
{
    (void)context;
    if (response == 0) return AEE_EBADPARM;
    switch (response->type) {
    case HAP_power_get_clk_Freq:      response->clkFreqHz = 660000000u; break;
    case HAP_power_get_max_mips:      response->max_mips = 660; break;
    case HAP_power_get_max_bus_bw:    response->max_bus_bw = 34000000000ull; break;
    case HAP_power_get_client_class:  response->client_class = HAP_POWER_COMPUTE_CLIENT_CLASS; break;
    case HAP_power_get_dcvsEnabled:   response->dcvsEnabled = 0; break;
    default: break;
    }
    return 0;
}

int HAP_power_get_max_speed(int *clock_max, int *bus_max)
{
    if (clock_max) *clock_max = 660000000;
    if (bus_max) *bus_max = 0x7fffffff;
    return 0;
}
