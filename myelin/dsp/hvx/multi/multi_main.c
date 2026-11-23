/*
 * Multi-PD driver: N independent FastRPC user PDs (one host process each),
 * each holding a row slice of the FFN weight matrix and running the same
 * int4 GEMV kernel simultaneously.
 *
 * Why processes and not N handles in one process: in FastRPC a "session 0"
 * belongs to the calling process, so N processes => N user PDs.  N handles in
 * one process all land in the same PD (and on this SDK only 2 reserved
 * sessions exist, see the `sessions` probe).  The `sessions` mode measures
 * that limit on the device instead of assuming it.
 *
 * Modes:
 *   mode=main n=<1..4> reps=20 rounds=5 hold=0
 *       ref run (1 PD, full 7680x2048 matrix) then the shard run (N PDs,
 *       7680/N rows each); prints per-PD timings, aggregate GB/s, exactness.
 *   mode=base2 gap=30
 *       two identical N=1 runs separated by `gap` seconds, to classify noise
 *       and DVFS drift before the series.
 *   mode=heap
 *       one PD, ascending alloc_buf until failure: per-PD heap ceiling.
 *   mode=sessions
 *       reserve sessions 1..3, open handles by session URI, plus 4 plain
 *       handle opens in one process: how many independent sessions exist.
 *
 * Every kernel number comes from the DSP (times[] = best_us, best_pc,
 * deint_us, pack_us); the pcycles give the effective clock so DVFS drift is
 * visible instead of being silently mixed into microseconds.
 */
#include "lfmgemv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <errno.h>

#define ROWS_TOTAL 7680
#define K_DIM      2048
#define ROW_BYTES  (K_DIM / 2)
#define MAXW       4
#define MAXR       32

typedef struct {
    int ok, rc;                 /* ok=0 when the worker finished */
    int pid, row0, rows;
    int heap_bytes;
    double t_enable_us, t_open_us, t_shape_us, t_load_us;
    int team[8];
    int clk_hz, clk_max_hz;
    int k_us[MAXR], k_pc[MAXR], w_us[MAXR], deint_us[MAXR], pack_us[MAXR];
    double ts[MAXR], te[MAXR];  /* CLOCK_MONOTONIC, comparable across procs */
    int alive_rc;
    double alive_t0, alive_t1;
    long long checksum;
} wres_t;

typedef struct {
    volatile int gen, arrive, nw;
    wres_t w[MAXW];
    int32_t part[MAXW][ROWS_TOTAL];
} shared_t;

/* ------------------------------------------------------------------ utils */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static double median_i(int *v, int n)
{
    int t[MAXR];
    if (n <= 0) return -1;
    memcpy(t, v, (size_t)n * sizeof(int));
    qsort(t, (size_t)n, sizeof(int), cmp_int);
    return (n & 1) ? (double)t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

static inline uint32_t lcg(uint32_t *st)
{
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static int enable_unsigned(void)
{
    struct remote_rpc_control_unsigned_module um;
    um.domain = CDSP_DOMAIN_ID;
    um.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um,
                                  (uint32_t)sizeof(um));
}

/* packed int4: byte = (lo+8) | ((hi+8) << 4), lo = even k, hi = odd k */
static void build_weights(uint8_t *w)
{
    int r, k;
    uint32_t st = 0x12345678u;
    for (r = 0; r < ROWS_TOTAL; r++) {
        uint8_t *row = w + (size_t)r * ROW_BYTES;
        for (k = 0; k < K_DIM; k += 2) {
            int lo = (int)(lcg(&st) % 16) - 8;
            int hi = (int)(lcg(&st) % 16) - 8;
            row[k / 2] = (uint8_t)(((lo + 8) & 0xf) | (((hi + 8) & 0xf) << 4));
        }
    }
}

static int32_t dot_ref(const uint8_t *packed, const int8_t *act, int k)
{
    int32_t s = 0;
    int i;
    for (i = 0; i < k; i += 2) {
        int lo = (int)(packed[i / 2] & 0xf) - 8;
        int hi = (int)(packed[i / 2] >> 4) - 8;
        s += lo * act[i] + hi * act[i + 1];
    }
    return s;
}

/* barrier over the worker processes (no parent participation) */
static void bar_wait(shared_t *s)
{
    int nw = s->nw;
    int gen = __atomic_load_n(&s->gen, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&s->arrive, 1, __ATOMIC_ACQ_REL) == nw) {
        __atomic_store_n(&s->arrive, 0, __ATOMIC_RELEASE);
        __atomic_add_fetch(&s->gen, 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&s->gen, __ATOMIC_ACQUIRE) == gen)
            usleep(50);
    }
}

/* ---------------------------------------------------------------- worker */

static int run_worker(shared_t *s, int idx, int nw, const uint8_t *wfull,
                      const int8_t *act, int row0, int rows,
                      int reps, int rounds, int hold_ms, int threads,
                      int stream)
{
    wres_t *r = &s->w[idx];
    int h = -1, rc, i;
    size_t bytes = (size_t)rows * ROW_BYTES;
    double t0;

    memset(r, 0, sizeof(*r));
    r->pid = getpid();
    r->row0 = row0;
    r->rows = rows;

    t0 = now_us();
    rc = enable_unsigned();
    r->t_enable_us = now_us() - t0;
    if (rc) { r->rc = rc; r->ok = -1; return 1; }

    /* first RPC: opens the module in this process' PD + PD boot + heap alloc */
    t0 = now_us();
    rc = lfmgemv_alloc_buf((int)bytes, &h);
    r->t_open_us = now_us() - t0;
    if (rc) { r->rc = rc; r->ok = -2; return 2; }
    r->heap_bytes = (int)bytes;

    t0 = now_us();
    rc = lfmgemv_set_shape(h, rows, K_DIM, 0);   /* int32 accumulator mode */
    r->t_shape_us = now_us() - t0;
    if (rc) { r->rc = rc; r->ok = -3; return 3; }

    t0 = now_us();
    {
        const size_t chunk = 256 * 1024;
        size_t off;
        const uint8_t *src = wfull + (size_t)row0 * ROW_BYTES;
        for (off = 0; off < bytes; off += chunk) {
            size_t n = (bytes - off < chunk) ? bytes - off : chunk;
            rc = lfmgemv_load_buf(h, (int)off, src + off, (int)n);
            if (rc) break;
        }
    }
    r->t_load_us = now_us() - t0;
    if (rc) { r->rc = rc; r->ok = -4; return 4; }

    if (lfmgemv_team_info(r->team, 8) != 0)
        memset(r->team, -1, sizeof(r->team));
    /* worker-team size per PD: probe(8) sets pf_kb|rows|usepf|threads.
     * Must run before the first gemv, because the team is started lazily by
     * the first split job.  The other three knobs keep their defaults. */
    {
        int p[8];
        int arg = 32 | (2 << 8) | (1 << 16) | ((threads & 0xff) << 24);
        if (lfmgemv_probe(h, 8, arg, p, 8) == 0) {
            r->team[4] = p[6];
            r->team[6] = threads;
        }
    }
    {
        int p[8];
        if (lfmgemv_probe(h, 5, 0, p, 8) == 0) {
            r->clk_max_hz = p[2];
            r->clk_hz = p[3];
        }
    }

    bar_wait(s);   /* all PDs loaded -> start the timed rounds together */

    for (i = 0; i < rounds; i++) {
        int times[8];
        double a, b;
        memset(times, 0, sizeof(times));
        bar_wait(s);
        a = now_us();
        if (stream) {
            int p[8];
            rc = lfmgemv_probe(h, 6, 256, p, 8);   /* 256 MiB linear read */
            times[0] = p[1];   /* us   */
            times[5] = p[0];   /* pcycles */
            times[3] = p[6];   /* bytes actually read */
        } else {
            rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, K_DIM, reps,
                              (unsigned char *)s->part[idx],
                              rows * (int)sizeof(int32_t), times, 8);
        }
        b = now_us();
        r->ts[i] = a;
        r->te[i] = b;
        if (!rc) {
            r->k_us[i]    = times[0];   /* best GEMV-only, us (DSP clock) */
            r->k_pc[i]    = times[5];   /* best GEMV-only, pcycles          */
            r->deint_us[i] = times[3];
            r->pack_us[i]  = times[4];
            r->w_us[i]    = (int)(b - a);
        }
        bar_wait(s);
        if (rc) { r->rc = rc; r->ok = -5; return 5; }
    }

    /* liveness: every PD runs one more GEMV while all the others are alive */
    bar_wait(s);
    {
        int times[8];
        double a = now_us();
        rc = lfmgemv_gemv(h, 1, (const unsigned char *)act, K_DIM, 1,
                          (unsigned char *)s->part[idx],
                          rows * (int)sizeof(int32_t), times, 8);
        r->alive_rc = rc;
        r->alive_t0 = a;
        r->alive_t1 = now_us();
    }
    bar_wait(s);

    {
        long long sum = 0;
        int32_t *p = s->part[idx];
        for (i = 0; i < rows; i++) sum += p[i];
        r->checksum = sum;
    }

    if (hold_ms > 0) usleep((useconds_t)hold_ms * 1000);

    lfmgemv_release_buf(h);
    r->ok = 0;
    return 0;
}

static pid_t spawn_worker(shared_t *s, int idx, int nw, const uint8_t *w,
                          const int8_t *act, int row0, int rows,
                          int reps, int rounds, int hold_ms, int threads,
                      int stream)
{
    pid_t p = fork();
    if (p == 0) {
        int rc = run_worker(s, idx, nw, w, act, row0, rows, reps, rounds,
                            hold_ms, threads, stream);
        _exit(rc ? 1 : 0);
    }
    return p;
}

/* ---------------------------------------------------------------- report */

static double agg_kernel_us(shared_t *s, int nw, int rounds, double *mhz_out,
                            double *wall_out, int *overlap_out)
{
    double agg = 0, aggwall = 0, mhz = 0;
    int overlap = 0, i, w;

    for (w = 0; w < nw; w++) {
        int mhz_i[MAXR];
        double m = median_i(s->w[w].k_us, rounds);
        double wm = median_i(s->w[w].w_us, rounds);
        if (m > agg) agg = m;
        if (wm > aggwall) aggwall = wm;
        for (i = 0; i < rounds; i++)
            mhz_i[i] = s->w[w].k_us[i] > 0
                       ? (int)((double)s->w[w].k_pc[i] / s->w[w].k_us[i]) : 0;
        { double mm = median_i(mhz_i, rounds); if (mm > mhz) mhz = mm; }
    }
    for (i = 0; i < rounds; i++) {
        double mx = 0, mn = 1e18;
        for (w = 0; w < nw; w++) {
            if (s->w[w].ts[i] > mx) mx = s->w[w].ts[i];
            if (s->w[w].te[i] < mn) mn = s->w[w].te[i];
        }
        if (mx < mn) overlap++;
    }
    *mhz_out = mhz;
    *wall_out = aggwall;
    *overlap_out = overlap;
    return agg;
}

static void print_worker(shared_t *s, int w, int rounds)
{
    wres_t *r = &s->w[w];
    int i;
    printf("W w=%d pid=%d row0=%d rows=%d heap=%d enable_us=%.0f open_us=%.0f"
           " shape_us=%.0f load_us=%.0f team=[state=%d mode=%d created=%d"
           " threads=%d] hvx_units=%d clk_mhz=%.0f\n",
           w, r->pid, r->row0, r->rows, r->heap_bytes, r->t_enable_us,
           r->t_open_us, r->t_shape_us, r->t_load_us,
           r->team[0], r->team[2], r->team[3], r->team[4], r->team[6],
           r->clk_hz / 1e6);
    for (i = 0; i < rounds; i++)
        printf("RUN w=%d r=%d k_us=%d k_pc=%d mhz=%.1f deint_us=%d pack_us=%d"
               " wall_us=%d\n", w, i, r->k_us[i], r->k_pc[i],
               r->k_us[i] ? (double)r->k_pc[i] / r->k_us[i] : 0.0,
               r->deint_us[i], r->pack_us[i], r->w_us[i]);
    printf("ALIVE w=%d rc=%d t0=%.0f t1=%.0f csum=%lld\n",
           w, r->alive_rc, r->alive_t0, r->alive_t1, r->checksum);
}

/* --------------------------------------------------------------- parent */

int main(int argc, char **argv)
{
    const char *mode = "main";
    int n = 1, reps = 20, rounds = 5, hold_ms = 0, gap = 30, threads = 2;
    int stream = 0;
    int i;
    uint8_t *w;
    int8_t *act;
    shared_t *s;
    int32_t *ref = NULL;
    double ref_kernel = 0;
    int rc = 0;

    for (i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "mode=", 5))       mode = argv[i] + 5;
        else if (!strncmp(argv[i], "n=", 2))     n = atoi(argv[i] + 2);
        else if (!strncmp(argv[i], "reps=", 5))  reps = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "rounds=", 7)) rounds = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "hold=", 5))  hold_ms = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "gap=", 4))   gap = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "threads=", 8)) threads = atoi(argv[i] + 8);
    }
    stream = !strcmp(mode, "stream");
    if (rounds < 1) rounds = 1;
    if (rounds > MAXR) rounds = MAXR;
    if (reps < 1) reps = 1;

    /* ---- session-limit probe (no kernel needed) ---- */
    if (!strcmp(mode, "sessions")) {
        struct remote_rpc_reserve_new_session rs;
        struct remote_rpc_get_uri gu;
        int k;
        printf("PROBE sessions\n");
        if (enable_unsigned()) { printf("FAIL enable_unsigned\n"); return 10; }
        for (k = 0; k < 4; k++) {
            char name[16], uri[160];
            uint32_t sid;
            remote_handle h = (remote_handle)-1;
            memset(&rs, 0, sizeof(rs));
            snprintf(name, sizeof(name), "m%d", k + 1);
            rs.domain_name = (char *)"cdsp";
            rs.domain_name_len = 4;
            rs.session_name = name;
            rs.session_name_len = (uint32_t)strlen(name);
            rc = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, &rs,
                                        (uint32_t)sizeof(rs));
            printf("RESERVE k=%d rc=%d eff_domain=%u session_id=%u\n",
                   k, rc, rs.effective_domain_id, rs.session_id);
            if (rc) continue;
            sid = rs.session_id;
            memset(&gu, 0, sizeof(gu));
            snprintf(uri, sizeof(uri), "lfmgemv");
            gu.domain_name = (char *)"cdsp";
            gu.domain_name_len = 4;
            gu.session_id = sid;
            gu.module_uri = uri;
            gu.module_uri_len = (uint32_t)strlen(uri);
            {
                char out[160];
                memset(out, 0, sizeof(out));
                gu.uri = out;
                gu.uri_len = sizeof(out) - 1;
                rc = remote_session_control(FASTRPC_GET_URI, &gu,
                                            (uint32_t)sizeof(gu));
                printf("GETURI k=%d rc=%d uri=%s\n", k, rc, out);
                if (!rc) {
                    int orc = remote_handle_open(out, &h);
                    printf("OPEN_BY_URI k=%d rc=%d h=%u\n", k, orc, h);
                    if (!orc && h != (remote_handle)-1) remote_handle_close(h);
                }
            }
        }
        {   /* plain opens in one process: same session 0, same PD */
            remote_handle hs[4];
            int ok = 0;
            for (k = 0; k < 4; k++) {
                hs[k] = (remote_handle)-1;
                rc = remote_handle_open("lfmgemv", &hs[k]);
                printf("OPEN_PLAIN k=%d rc=%d h=%u\n", k, rc, hs[k]);
                if (!rc) ok++;
            }
            for (k = 0; k < 4; k++)
                if (hs[k] != (remote_handle)-1) remote_handle_close(hs[k]);
            printf("OPEN_PLAIN_TOTAL ok=%d of 4 (all in session 0 = one PD)\n", ok);
        }
        return 0;
    }

    /* ---- per-PD heap ceiling ---- */
    if (!strcmp(mode, "heap")) {
        static const int mb[] = {8, 16, 32, 64, 128, 192, 256, 384, 512, 600};
        int hs[16], cnt = 0, h;
        long long total = 0;
        printf("PROBE heap\n");
        if (enable_unsigned()) { printf("FAIL enable_unsigned\n"); return 10; }
        for (i = 0; i < (int)(sizeof(mb) / sizeof(mb[0])); i++) {
            size_t bytes = (size_t)mb[i] << 20;
            double t0 = now_us();
            rc = lfmgemv_alloc_buf((int)bytes, &h);
            printf("ALLOC mb=%d rc=%d h=%d us=%.0f total_mb=%.0f\n", mb[i], rc,
                   h, now_us() - t0, (double)total / 1048576.0);
            if (rc) break;
            hs[cnt++] = h;
            total += (long long)bytes;
            printf("ALLOC_OK mb=%d cumulative_mb=%.0f\n", mb[i],
                   (double)total / 1048576.0);
            if (!lfmgemv_set_shape(h, 2560, K_DIM, 0))
                printf("SHAPE_OK mb=%d h=%d\n", mb[i], h);
        }
        for (i = 0; i < cnt; i++) lfmgemv_release_buf(hs[i]);
        printf("HEAP max_single_mb=%d cumulative_mb=%.0f\n",
               cnt ? mb[cnt - 1] : 0, (double)total / 1048576.0);
        return 0;
    }

    s = mmap(NULL, sizeof(*s), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (s == MAP_FAILED) { perror("mmap"); return 11; }
    memset(s, 0, sizeof(*s));

    w = malloc((size_t)ROWS_TOTAL * ROW_BYTES);
    act = malloc(K_DIM);
    ref = malloc((size_t)ROWS_TOTAL * sizeof(int32_t));
    if (!w || !act || !ref) { printf("FAIL host malloc\n"); return 12; }
    build_weights(w);
    {
        uint32_t st = 0xdeadbeefu;
        for (i = 0; i < K_DIM; i++) act[i] = (int8_t)((int)(lcg(&st) % 255) - 127);
    }

    /* ---- noise baseline: two N=1 runs, `gap` seconds apart ---- */
    if (!strcmp(mode, "base2")) {
        int pass;
        printf("BASE2 gap=%d reps=%d rounds=%d\n", gap, reps, rounds);
        for (pass = 0; pass < 2; pass++) {
            double mhz, wall;
            int ov, ww;
            pid_t p;
            memset(&s->w[0], 0, sizeof(s->w[0]));
            s->nw = 1;
            p = spawn_worker(s, 0, 1, w, act, 0, ROWS_TOTAL, reps, rounds, 0,
                             threads, stream);
            waitpid(p, &rc, 0);
            print_worker(s, 0, rounds);
            ref_kernel = agg_kernel_us(s, 1, rounds, &mhz, &wall, &ov);
            printf("BASE2 pass=%d pid=%d kernel_us=%.0f wall_us=%.0f mhz=%.0f"
                   " gbs=%.2f ok=%d exit=%d\n", pass, s->w[0].pid, ref_kernel,
                   wall, mhz, (double)ROWS_TOTAL * ROW_BYTES / (ref_kernel * 1e3),
                   s->w[0].ok, WEXITSTATUS(rc));
            for (ww = 0; ww < 1; ww++)
                if (s->w[ww].ok != 0) printf("FAIL worker ok=%d rc=%d\n",
                                             s->w[ww].ok, s->w[ww].rc);
            if (pass == 0) sleep((unsigned)gap);
        }
        return 0;
    }

    /* ---- default: ref (1 PD, full matrix) + target (N PDs, slices) ---- */
    printf("CFG mode=main n=%d reps=%d rounds=%d threads=%d rows_total=%d K=%d"
           " weight_mb=%.2f\n", n, reps, rounds, threads, ROWS_TOTAL, K_DIM,
           (double)ROWS_TOTAL * ROW_BYTES / 1e6);
    {
        double mhz, wall;
        int ov;
        pid_t p;
        s->nw = 1;
        memset(&s->w[0], 0, sizeof(s->w[0]));
        p = spawn_worker(s, 0, 1, w, act, 0, ROWS_TOTAL, reps, rounds, 0,
                         threads, stream);
        waitpid(p, &rc, 0);
        print_worker(s, 0, rounds);
        if (s->w[0].ok != 0) {
            printf("FAIL ref ok=%d rc=%d\n", s->w[0].ok, s->w[0].rc);
            return 20;
        }
        memcpy(ref, s->part[0], (size_t)ROWS_TOTAL * sizeof(int32_t));
        ref_kernel = agg_kernel_us(s, 1, rounds, &mhz, &wall, &ov);
        printf("REF kernel_us=%.0f wall_us=%.0f mhz=%.0f gbs=%.2f"
               " open_us=%.0f load_us=%.0f\n", ref_kernel, wall, mhz,
               (double)ROWS_TOTAL * ROW_BYTES / (ref_kernel * 1e3),
               s->w[0].t_open_us, s->w[0].t_load_us);
    }
    if (stream) { ref_kernel = ref_kernel > 0 ? ref_kernel : 1; }

    if (n < 1) n = 1;
    if (n > MAXW) n = MAXW;

    /* two configurations:
     *  slice - the requested one: rows split 7680/N, aggregate 7.86 MB
     *  full  - every PD holds the whole matrix: aggregate 7.86*N MB, i.e.
     *          a working set that cannot fit in the DSP L2 (honest DDR test)
     */
    {
        const char *cfg;
        int variants;
        for (variants = stream ? 1 : 0; variants < 2; variants++) {
            cfg = variants ? "full" : "slice";
            {
                int rows = variants ? ROWS_TOTAL : ROWS_TOTAL / n;
                pid_t pid[MAXW];
                int status[MAXW], ww;
                double mhz, wall, agg;
                int ov, bad = 0;

                /* clear previous run's results but keep the ref copy */
                for (ww = 0; ww < MAXW; ww++) memset(&s->w[ww], 0, sizeof(s->w[ww]));
                memset(s->part, 0, sizeof(s->part));
                s->gen = 0;
                s->arrive = 0;
                s->nw = n;
                __sync_synchronize();

                {
                    double t0 = now_us();
                    for (ww = 0; ww < n; ww++)
                        pid[ww] = spawn_worker(s, ww, n, w, act,
                                               variants ? 0 : ww * rows, rows,
                                               reps, rounds, hold_ms, threads,
                                               stream);
                    for (ww = 0; ww < n; ww++)
                        waitpid(pid[ww], &status[ww], 0);
                    printf("SPAWN cfg=%s n=%d wall_to_done_us=%.0f\n",
                           cfg, n, now_us() - t0);
                }

                for (ww = 0; ww < n; ww++) {
                    print_worker(s, ww, rounds);
                    if (s->w[ww].ok != 0 || WIFSIGNALED(status[ww])) {
                        printf("FAIL cfg=%s w=%d ok=%d rc=%d signaled=%d\n",
                               cfg, ww, s->w[ww].ok, s->w[ww].rc,
                               WIFSIGNALED(status[ww]));
                        bad = 1;
                    }
                    if (s->w[ww].alive_rc != 0) {
                        printf("FAIL cfg=%s w=%d alive_rc=%d\n", cfg, ww,
                               s->w[ww].alive_rc);
                        bad = 1;
                    }
                }
                if (bad) continue;

                agg = agg_kernel_us(s, n, rounds, &mhz, &wall, &ov);
                {
                    double bytes = stream
                        ? (double)n * (256.0 * 1048576.0)
                        : (double)ROWS_TOTAL * ROW_BYTES * (variants ? n : 1);
                    double gbs = bytes / (agg * 1e3);
                    printf("AGG cfg=%s n=%d rows_per_pd=%d kernel_us=%.0f"
                           " wall_us=%.0f gbs=%.2f mhz=%.0f overlap=%d/%d"
                           " speedup=%.2f tok_s_880MB=%.1f\n",
                           cfg, n, rows, agg, wall, gbs, mhz, ov, rounds,
                           ref_kernel / agg, gbs * 1e9 / 880e6);
                }

                /* exactness: concatenated shards vs the 1-PD result */
                if (!stream) {
                    int32_t *cat = calloc((size_t)ROWS_TOTAL, sizeof(int32_t));
                    int mism = 0, maxerr = 0, host_bad = 0, hm = 0;
                    long long cs = 0;
                    for (ww = 0; ww < n; ww++) {
                        int rows_w = s->w[ww].rows;
                        int row0 = variants ? 0 : s->w[ww].row0;
                        int j;
                        for (j = 0; j < rows_w && row0 + j < ROWS_TOTAL; j++) {
                            int32_t v = s->part[ww][j];
                            if (variants) {
                                int32_t e = v - ref[row0 + j];
                                if (e < 0) e = -e;
                                if (e > maxerr) maxerr = e;
                                if (e) mism++;
                                cs += v;
                            } else {
                                cat[row0 + j] = v;
                            }
                        }
                    }
                    if (!variants) {
                        for (i = 0; i < ROWS_TOTAL; i++) {
                            int32_t e = cat[i] - ref[i];
                            if (e < 0) e = -e;
                            if (e > maxerr) maxerr = e;
                            if (e) mism++;
                            cs += cat[i];
                        }
                    }
                    /* independent host reference on every row */
                    for (i = 0; i < ROWS_TOTAL; i++) {
                        const int32_t *src = variants ? s->part[0] : cat;
                        int32_t h = dot_ref(w + (size_t)i * ROW_BYTES, act, K_DIM);
                        int32_t e = src[i] - h;
                        if (e < 0) e = -e;
                        if (e > hm) hm = e;
                        if (e) host_bad++;
                    }
                    printf("CHK cfg=%s n=%d rows_total=%d vs_ref_mismatches=%d"
                           " vs_ref_maxerr=%d vs_host_mismatches=%d"
                           " vs_host_maxerr=%d csum=%lld\n", cfg, n, ROWS_TOTAL,
                           mism, maxerr, host_bad, hm, cs);
                    free(cat);
                }

                /* host-side reduction of the N partials (the price of this
                 * design at every layer boundary) */
                if (n > 1 && !variants && !stream) {
                    const int rreps = 200;
                    int32_t *out = calloc((size_t)ROWS_TOTAL, sizeof(int32_t));
                    int rep;
                    double t0 = now_us();
                    for (rep = 0; rep < rreps; rep++) {
                        for (i = 0; i < ROWS_TOTAL; i++) {
                            int32_t v = 0;
                            for (ww = 0; ww < n; ww++) v += s->part[ww][i];
                            out[i] = v;
                        }
                    }
                    printf("RED n=%d us_per_token=%.2f reps=%d csum=%d\n", n,
                           (now_us() - t0) / rreps, rreps, out[0]);
                    free(out);
                }
            }
        }
    }
    return 0;
}
