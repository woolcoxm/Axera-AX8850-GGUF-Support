// minimal per-layer engine timing: load, allocate IO by dims, run N times
// usage: bench_layer <model.axmodel> [iters]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"
#include "axcl_rt.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static size_t alloc_io(axclrtEngineIOInfo info, axclrtEngineIO io,
                       uint32_t n, int is_in) {
    size_t tot = 0;
    for (uint32_t i = 0; i < n; i++) {
        axclrtEngineIODims d;
        if (is_in) axclrtEngineGetInputDims(info, 0, i, &d);
        else       axclrtEngineGetOutputDims(info, 0, i, &d);
        size_t sz = 4;
        for (int k = 0; k < d.dimCount; k++) sz *= (size_t)d.dims[k];
        const char *nm = is_in ? axclrtEngineGetInputNameByIndex(info, i)
                               : axclrtEngineGetOutputNameByIndex(info, i);
        // bf16/f16 = 2B; dims product is element count for these engines
        void *p = NULL; axclrtMalloc(&p, sz * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMemset(p, 0, sz * 2);
        if (is_in) axclrtEngineSetInputBufferByIndex(io, i, p, sz * 2);
        else       axclrtEngineSetOutputBufferByIndex(io, i, p, sz * 2);
        printf("  %s[%u] %s: %d dims, %zu el (%zu B)\n",
               is_in ? "in" : "out", i, nm ? nm : "?", d.dimCount, sz, sz * 2);
        tot += sz * 2;
    }
    return tot;
}

int main(int argc, char **argv) {
    const char *path = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 50;
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext c; axclrtCreateContext(&c, dl.devices[0]);
    axclrtSetCurrentContext(c);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    uint64_t model = 0, ectx = 0, io = 0; axclrtEngineIOInfo info = 0;
    if (axclrtEngineLoadFromFile(path, &model)) { printf("LOAD FAIL\n"); return 1; }
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    printf("inputs=%u outputs=%u\n", ni, no);
    alloc_io(info, io, ni, 1);
    alloc_io(info, io, no, 0);
    if (axclrtEngineExecute(model, ectx, 0, io)) { printf("EXEC FAIL (warmup)\n"); return 1; }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        if (axclrtEngineExecute(model, ectx, 0, io)) { printf("EXEC FAIL @%d\n", i); return 1; }
    double dt = now_ms() - t0;
    printf("%s: %d iters, mean %.3f ms/layer (%.1f layers/s)\n",
           path, iters, dt / iters, iters / (dt / 1000.0));
    return 0;
}
