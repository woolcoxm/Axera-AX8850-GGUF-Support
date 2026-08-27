// phase_c_timing.c — generic whole-layer engine latency microbench (Phase C).
// Loads any .axmodel, dumps shape groups/IO sizes, binds correctly-sized
// buffers, and times each shape group at m=its own declared shape.
// build: gcc phase_c_timing.c -laxclrt -laxcl -o phase_c_timing
// usage: ./phase_c_timing <engine.axmodel> [iters] [group]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

static double now_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <axmodel> [iters=200] [group=all]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    const int only_g = argc > 3 ? atoi(argv[3]) : -1;
    if (axclInit(NULL) != AXCL_SUCC) { fprintf(stderr, "axclInit failed\n"); return 1; }
    axclrtDeviceList dl; memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC || dl.num == 0) { fprintf(stderr, "no devices\n"); return 1; }
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx = 0;
    axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    if (axclrtEngineInit(AXCL_VNPU_DISABLE) != AXCL_SUCC) { fprintf(stderr, "engine init failed\n"); return 1; }

    uint64_t model = 0;
    if (axclrtEngineLoadFromFile(argv[1], &model)) { printf("LOAD FAIL %s\n", argv[1]); return 1; }
    axclrtEngineIOInfo info = NULL;
    axclrtEngineGetIOInfo(model, &info);
    int32_t ngroups = 0;
    axclrtEngineGetShapeGroupsCount(info, &ngroups);
    uint32_t n_in = axclrtEngineGetNumInputs(info);
    uint32_t n_out = axclrtEngineGetNumOutputs(info);
    printf("engine=%s groups=%d in=%u out=%u\n", argv[1], ngroups, n_in, n_out);

    axclrtEngineIO io = 0;
    axclrtEngineCreateIO(info, &io);
    uint64_t ectx = 0;
    axclrtEngineCreateContext(model, &ectx);

    // index buffers by (group, io-index); allocate per group since sizes differ
    enum { MAXIO = 16, MAXG = 12 };
    static void * din[MAXG][MAXIO], * dout[MAXG][MAXIO];
    for (int32_t g = 0; g < ngroups && g < MAXG; g++) {
        if (only_g >= 0 && g != only_g) continue;
        printf("group %d:", g);
        for (uint32_t i = 0; i < n_in && i < MAXIO; i++) {
            const char * nm = axclrtEngineGetInputNameByIndex(info, i);
            uint64_t sz = axclrtEngineGetInputSizeByIndex(info, g, i);
            axclrtEngineIODims d; memset(&d, 0, sizeof(d));
            axclrtEngineGetInputDims(info, g, i, &d);
            printf(" %s=%llub", nm ? nm : "?", (unsigned long long) sz);
            printf("[");
            for (uint32_t j = 0; j < d.dimCount; j++) printf("%u%s", d.dims[j], j+1<d.dimCount?",":"");
            printf("]");
            if (axclrtMalloc(&din[g][i], sz ? sz : 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
                printf("\nMALLOC FAIL g%d in%u\n", g, i); return 1;
            }
            // zero-fill inputs (notes: uninit caches NaN-poison through masks; zeros are safe)
            static char z[4096]; memset(z, 0, sizeof(z));
            for (uint64_t off = 0; off < sz; off += sizeof(z))
                axclrtMemcpy((char *) din[g][i] + off, z, (sz - off < sizeof(z)) ? sz - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtEngineSetInputBufferByIndex(io, i, din[g][i], sz);
        }
        for (uint32_t i = 0; i < n_out && i < MAXIO; i++) {
            const char * nm = axclrtEngineGetOutputNameByIndex(info, i);
            uint64_t sz = axclrtEngineGetOutputSizeByIndex(info, g, i);
            printf(" out:%s=%llub", nm ? nm : "?", (unsigned long long) sz);
            if (axclrtMalloc(&dout[g][i], sz ? sz : 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
                printf("\nMALLOC FAIL g%d out%u\n", g, i); return 1;
            }
            axclrtEngineSetOutputBufferByIndex(io, i, dout[g][i], sz);
        }
        printf("\n");
        // timing: warmup 5, then iters synchronous executes
        for (int w = 0; w < 5; w++)
            axclrtEngineExecute(model, ectx, (uint32_t) g, io);
        double t0 = now_us();
        for (int it = 0; it < iters; it++) {
            if (axclrtEngineExecute(model, ectx, (uint32_t) g, io) != AXCL_SUCC) {
                printf("EXEC FAIL g%d iter %d\n", g, it); break;
            }
        }
        double dt = now_us() - t0;
        printf("RESULT engine=%s group=%d iters=%d avg=%.1f us/call\n",
               argv[1], g, iters, dt / iters);
    }
    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
