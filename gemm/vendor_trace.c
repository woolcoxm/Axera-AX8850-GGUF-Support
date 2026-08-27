// vendor_trace.so — LD_PRELOAD shim logging the vendor runtime's engine IO
// usage: axclrtEngineExecute (group), Set*BufferByIndex (io, idx, ptr, size),
// and dumps bound device buffers at execute time.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include "axcl.h"
#include "axcl_rt.h"

static int in_hook = 0;
static pthread_mutex_t hook_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE * lg = NULL;
static void ensure_log() { // caller holds hook_mu
    if (lg == NULL) { lg = fopen("/tmp/vendor_trace.log", "w"); setbuf(lg, NULL); }
}

typedef axclError (*exec_fn)(uint64_t, uint64_t, uint32_t, axclrtEngineIO);
typedef axclError (*setin_fn)(axclrtEngineIO, uint32_t, const void *, uint64_t);
typedef axclError (*setout_fn)(axclrtEngineIO, uint32_t, const void *, uint64_t);
typedef axclError (*memcpy_fn)(void *, const void *, size_t, axclrtMemcpyKind);
typedef axclError (*createio_fn)(axclrtEngineIOInfo, axclrtEngineIO *);

static exec_fn real_exec;
static setin_fn real_setin;
static setout_fn real_setout;
static memcpy_fn real_memcpy;
static createio_fn real_createio;

// track the most recent bindings per io pointer (io addresses are unique)
#define MAX_IO 64
struct bindrec { void * io; int idx; const void * ptr; unsigned long long size; int is_out; };
static struct bindrec binds[4096];
static int n_binds = 0;
static void * known_io[MAX_IO];
static int n_known = 0;

axclError axclrtEngineCreateIO_disabled(axclrtEngineIOInfo info, axclrtEngineIO * io) {
    if (!real_createio) real_createio = (createio_fn) dlsym(RTLD_NEXT, "axclrtEngineCreateIO");
    axclError r = real_createio(info, io);
    ensure_log();
    in_hook = 1;
    fprintf(lg, "[createio] info=%p -> io=%p\n", (void*)info, (void*)*io);
    if (n_known < MAX_IO) known_io[n_known++] = *io;
    in_hook = 0;
    return r;
}

axclError axclrtEngineSetInputBufferByIndex_disabled(axclrtEngineIO io, uint32_t idx, const void * ptr, uint64_t size) {
    if (!real_setin) real_setin = (setin_fn) dlsym(RTLD_NEXT, "axclrtEngineSetInputBufferByIndex");
    axclError r = real_setin(io, idx, ptr, size);
    pthread_mutex_lock(&hook_mu);
    ensure_log();
    if (n_binds < 4096) {
        binds[n_binds].io = io; binds[n_binds].idx = idx;
        binds[n_binds].ptr = ptr; binds[n_binds].size = (unsigned long long) size; binds[n_binds].is_out = 0;
        n_binds++;
    }
    return r;
}

axclError axclrtEngineSetOutputBufferByIndex_disabled(axclrtEngineIO io, uint32_t idx, const void * ptr, uint64_t size) {
    if (!real_setout) real_setout = (setout_fn) dlsym(RTLD_NEXT, "axclrtEngineSetOutputBufferByIndex");
    axclError r = real_setout(io, idx, ptr, size);
    pthread_mutex_lock(&hook_mu);
    ensure_log();
    if (n_binds < 4096) {
        binds[n_binds].io = io; binds[n_binds].idx = idx;
        binds[n_binds].ptr = ptr; binds[n_binds].size = (unsigned long long) size; binds[n_binds].is_out = 1;
        n_binds++;
    }
    return r;
}

static void dump_binds(void * io) {
    // dump small INPUT buffers content (D2H through the real memcpy)
    for (int i = 0; i < n_binds; i++) {
        if (binds[i].io != io || binds[i].is_out) continue;
        size_t n = binds[i].size > 1024 ? 1024 : binds[i].size; // cap dumps
        static char buf[1024];
        if (real_memcpy(buf, binds[i].ptr, n, AXCL_MEMCPY_DEVICE_TO_HOST) == AXCL_SUCC) {
            fprintf(lg, "    in[%d] ptr=%p size=%zu head:", binds[i].idx, binds[i].ptr, binds[i].size);
            if (binds[i].size == 4 || binds[i].size == 512) { // indices: u32s
                unsigned int * u = (unsigned int *) buf;
                for (int q = 0; q < (int)(n / 4) && q < 8; q++) fprintf(lg, " %u", u[q]);
            } else { // bf16 values
                unsigned short * h = (unsigned short *) buf;
                for (int q = 0; q < (int)(n / 2) && q < 8; q++) fprintf(lg, " %04x", h[q]);
            }
            fprintf(lg, "\n");
        }
    }
}

axclError axclrtEngineExecute(uint64_t model, uint64_t ctx, uint32_t group, axclrtEngineIO io) {
    if (!real_exec) real_exec = (exec_fn) dlsym(RTLD_NEXT, "axclrtEngineExecute");
    ensure_log();
    if (!real_memcpy) real_memcpy = (memcpy_fn) dlsym(RTLD_NEXT, "axclrtMemcpy");
    pthread_mutex_lock(&hook_mu);
    in_hook = 1;
    fprintf(lg, "[exec] model=%llx ctx=%llx GROUP=%u io=%p\n",
             (unsigned long long) model, (unsigned long long) ctx, group, (void*) io);
    if (group != 0) dump_binds(io);
    in_hook = 0;
    pthread_mutex_unlock(&hook_mu);
    return real_exec(model, ctx, group, io);
}
