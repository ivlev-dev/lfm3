#ifndef LFM_SIM_SHIM_H
#define LFM_SIM_SHIM_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define LFM_SIM_WEAK __attribute__((weak))
#else
#define LFM_SIM_WEAK
#endif

typedef int (*lfm_sim_fn)(int argc, char **argv);

typedef struct {
    const char *name;
    lfm_sim_fn fn;
} lfm_sim_case;

int sim_ion_register(void *p, size_t bytes);

int sim_hvx_lock(int mode);
int sim_hvx_unlock(void);
int sim_hvx_units(void);

unsigned long long sim_ticks(void);
unsigned long long sim_time_us(void);

#if defined(__hexagon__)
typedef struct {
    unsigned int tid;
    void *stack;
    unsigned int stack_bytes;
} sim_thread_t;
#else
#include <pthread.h>
typedef struct {
    pthread_t tid;
    void (*entry)(void *);
    void *arg;
} sim_thread_t;
#endif

int sim_thread_start(sim_thread_t *t, void (*entry)(void *), void *arg,
                     unsigned int stack_bytes);
int sim_thread_join(sim_thread_t *t);

extern int lfm_case_common(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_gemv(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_gemm(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_ffn(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_attn(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_kvcache(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_exec(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_quant(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_hmx_gemm(int argc, char **argv) LFM_SIM_WEAK;
extern int lfm_case_hmx_gemm_i8(int argc, char **argv) LFM_SIM_WEAK;

#endif
