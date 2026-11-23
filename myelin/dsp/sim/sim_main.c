#define _POSIX_C_SOURCE 200809L

#include "sim_shim.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__hexagon__)
#include <qurt.h>
#include <qurt_alloc.h>
#include <qurt_hvx.h>
#include <qurt_sclk.h>
#else
#include <time.h>
#endif

#ifndef LFM_SIM_HEAP_MB
#define LFM_SIM_HEAP_MB 56
#endif

#if defined(__hexagon__) && !defined(LFM_SIM_NO_STATIC_HEAP)
QURT_DECLARE_STATIC_HEAP(LFM_SIM_HEAP_MB * 1024u * 1024u);
#endif

#ifndef LFM_SIM_ONLY_SLUG
#define LFM_SIM_ONLY_SLUG ""
#endif

#define SIM_TICKS_PER_US 19200000ULL
#define SIM_CASE_STACK (1024u * 1024u)

typedef struct {
    lfm_sim_fn fn;
    int argc;
    char **argv;
    int rc;
} sim_run_ctx;

static const lfm_sim_case g_cases[] = {
    {"common", lfm_case_common},
    {"gemv", lfm_case_gemv},
    {"gemm", lfm_case_gemm},
    {"ffn", lfm_case_ffn},
    {"attn", lfm_case_attn},
    {"kvcache", lfm_case_kvcache},
    {"exec", lfm_case_exec},
    {"quant", lfm_case_quant},
    {"hmx_gemm", lfm_case_hmx_gemm},
    {"hmx_gemm_i8", lfm_case_hmx_gemm_i8},
};

LFM_SIM_WEAK int sim_ion_register(void *p, size_t bytes)
{
    (void)p;
    (void)bytes;
    return -1;
}

int sim_hvx_lock(int mode)
{
#if defined(__hexagon__)
    return qurt_hvx_lock((qurt_hvx_mode_t)mode);
#else
    (void)mode;
    return 0;
#endif
}

int sim_hvx_unlock(void)
{
#if defined(__hexagon__)
    return qurt_hvx_unlock();
#else
    return 0;
#endif
}

int sim_hvx_units(void)
{
#if defined(__hexagon__)
    return qurt_hvx_get_units();
#else
    return 0;
#endif
}

unsigned long long sim_ticks(void)
{
#if defined(__hexagon__)
    return (unsigned long long)qurt_sysclock_get_hw_ticks();
#else
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (unsigned long long)ts.tv_sec * 1000000000ULL +
           (unsigned long long)ts.tv_nsec;
#endif
}

unsigned long long sim_time_us(void)
{
#if defined(__hexagon__)
    return sim_ticks() / SIM_TICKS_PER_US;
#else
    return sim_ticks() / 1000ULL;
#endif
}

#if !defined(__hexagon__)
static void *sim_thread_entry(void *p)
{
    sim_thread_t *t = (sim_thread_t *)p;

    t->entry(t->arg);
    return NULL;
}
#endif

int sim_thread_start(sim_thread_t *t, void (*entry)(void *), void *arg,
                     unsigned int stack_bytes)
{
    if (t == NULL || entry == NULL)
        return -1;
#if defined(__hexagon__)
    {
        qurt_thread_attr_t attr;
        void *stack;
        unsigned int bytes = stack_bytes != 0 ? stack_bytes : 96u * 1024u;

        stack = malloc(bytes);
        if (stack == NULL)
            return -1;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stack);
        qurt_thread_attr_set_stack_size(&attr, bytes);
        qurt_thread_attr_set_priority(&attr, 100);
        if (qurt_thread_create(&t->tid, &attr, entry, arg) != 0) {
            free(stack);
            return -1;
        }
        t->stack = stack;
        t->stack_bytes = bytes;
        return 0;
    }
#else
    (void)stack_bytes;
    t->entry = entry;
    t->arg = arg;
    if (pthread_create(&t->tid, NULL, sim_thread_entry, t) != 0)
        return -1;
    return 0;
#endif
}

int sim_thread_join(sim_thread_t *t)
{
    if (t == NULL)
        return -1;
#if defined(__hexagon__)
    {
        int status = 0;

        if (qurt_thread_join(t->tid, &status) != 0)
            return -1;
        free(t->stack);
        t->stack = NULL;
        return status;
    }
#else
    {
        void *ret = NULL;

        if (pthread_join(t->tid, &ret) != 0)
            return -1;
        return 0;
    }
#endif
}

static lfm_sim_fn sim_case_entry(const lfm_sim_case *c)
{
    lfm_sim_fn fn = NULL;

    memcpy(&fn, &c->fn, sizeof fn);
    return fn;
}

static int sim_case_selected(const lfm_sim_case *c)
{
    if (LFM_SIM_ONLY_SLUG[0] == '\0')
        return 1;
    return strcmp(c->name, LFM_SIM_ONLY_SLUG) == 0;
}

static void sim_case_thread(void *p)
{
    sim_run_ctx *ctx = (sim_run_ctx *)p;
    int locked = sim_hvx_lock(QURT_HVX_MODE_128B) == 0;

    ctx->rc = ctx->fn(ctx->argc, ctx->argv);
    if (locked)
        sim_hvx_unlock();
}

static int sim_run_case(lfm_sim_fn fn, int argc, char **argv)
{
    sim_run_ctx ctx;
    sim_thread_t thread;

    ctx.fn = fn;
    ctx.argc = argc;
    ctx.argv = argv;
    ctx.rc = -1;
    if (sim_thread_start(&thread, sim_case_thread, &ctx, SIM_CASE_STACK) != 0)
        return -1;
    if (sim_thread_join(&thread) < 0)
        return -1;
    return ctx.rc;
}

int main(int argc, char **argv)
{
    static char *case_argv[] = {(char *)"lfm_sim", NULL};
    unsigned int i;
    int pass = 0;
    int skip = 0;
    int fail = 0;
    const unsigned int count = (unsigned int)(sizeof g_cases / sizeof g_cases[0]);

    (void)argc;
    (void)argv;

    setvbuf(stdout, NULL, _IONBF, 0);

#if defined(__hexagon__)
    printf("SIM platform: hexagon hvx_units=%d\n", sim_hvx_units());
#else
    printf("SIM platform: host\n");
#endif
    fflush(stdout);

    for (i = 0; i < count; i++) {
        lfm_sim_fn fn;
        int rc;

        if (!sim_case_selected(&g_cases[i]))
            continue;
        fn = sim_case_entry(&g_cases[i]);
        if (fn == NULL) {
            printf("SIM case %s SKIP (symbol not linked)\n", g_cases[i].name);
            fflush(stdout);
            skip++;
            continue;
        }
        printf("SIM case %s RUN\n", g_cases[i].name);
        fflush(stdout);
        rc = sim_run_case(fn, 1, case_argv);
        if (rc == 0) {
            printf("SIM case %s PASS\n", g_cases[i].name);
            pass++;
        } else {
            printf("SIM case %s FAIL (rc=%d)\n", g_cases[i].name, rc);
            fail++;
        }
        fflush(stdout);
    }

    printf("SIM RESULT: %s (pass=%d skip=%d fail=%d)\n",
           fail == 0 ? "PASS" : "FAIL", pass, skip, fail);
    fflush(stdout);
    return fail == 0 ? 0 : 1;
}
