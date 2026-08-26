// vendor whole-layer engine microbenchmark: decode-group latency
// qwen3_p128_l0_together.axmodel: inputs K_cache[1,2048,1024]bf16 V_cache[...] indices[1,1]u32
//                                input[1,1,1024]bf16 mask[1,1,2049]bf16
// outputs: K_cache_out V_cache_out output
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt.h"

static double now_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/home/kram/Qwen3-0.6B/qwen3_p128_l0_together.axmodel";
    const int iters = argc > 2 ? atoi(argv[2]) : 100;
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile(path, &model)) { printf("LOAD FAIL\n"); return 1; }
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    int32_t ngroups = 0; axclrtEngineGetShapeGroupsCount(info, &ngroups);
    printf("groups=%d\n", ngroups);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

    // sizes: K/V 2048*1024*2B, indices 4B, input 2048B, mask 2049*2B
    const size_t kv_bytes = (size_t) 2048 * 1024 * 2;
    void *dk, *dv, *di, *dx, *dm, *dko, *dvo, *dy;
    axclrtMalloc(&dk, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dv, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&di, 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dx, 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm, (size_t) 2049 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dko, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvo, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dy, 2048, AXCL_MEM_MALLOC_HUGE_FIRST);

    // init host data
    unsigned short * kv = malloc(kv_bytes);
    for (size_t i = 0; i < kv_bytes / 2; i++) kv[i] = 0x3f80; // bf16 1.0
    unsigned int idx = 7;
    unsigned short * mask = malloc((size_t) 2049 * 2);
    for (int i = 0; i < 2049; i++) mask[i] = 0x3f80;
    axclrtMemcpy(dk, kv, kv_bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dv, kv, kv_bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(di, &idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dx, kv, 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dm, mask, (size_t) 2049 * 2, AXCL_MEMCPY_HOST_TO_DEVICE);

    int ik = axclrtEngineGetInputIndexByName(info, "K_cache");
    int iv = axclrtEngineGetInputIndexByName(info, "V_cache");
    int ii = axclrtEngineGetInputIndexByName(info, "indices");
    int ix = axclrtEngineGetInputIndexByName(info, "input");
    int im = axclrtEngineGetInputIndexByName(info, "mask");
    int iko = axclrtEngineGetOutputIndexByName(info, "K_cache_out");
    int ivo = axclrtEngineGetOutputIndexByName(info, "V_cache_out");
    int iyo = axclrtEngineGetOutputIndexByName(info, "output");
    printf("idx: K=%d V=%d i=%d x=%d m=%d Ko=%d Vo=%d y=%d\n", ik, iv, ii, ix, im, iko, ivo, iyo);

    axclrtEngineSetInputBufferByIndex(io, ik, dk, kv_bytes);
    axclrtEngineSetInputBufferByIndex(io, iv, dv, kv_bytes);
    axclrtEngineSetInputBufferByIndex(io, ii, di, 4);
    axclrtEngineSetInputBufferByIndex(io, ix, dx, 2048);
    axclrtEngineSetInputBufferByIndex(io, im, dm, (size_t) 2049 * 2);
    axclrtEngineSetOutputBufferByIndex(io, iko, dko, kv_bytes);
    axclrtEngineSetOutputBufferByIndex(io, ivo, dvo, kv_bytes);
    axclrtEngineSetOutputBufferByIndex(io, iyo, dy, 2048);

    double t0 = now_us();
    int rc = axclrtEngineExecute(model, ectx, 0, io);
    double t1 = now_us();
    if (rc) { printf("EXEC FAIL rc=%d\n", rc); return 1; }
    printf("first exec: %.0fus\n", t1 - t0);

    unsigned short * y = malloc(2048);
    axclrtMemcpy(y, dy, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    int nfin = 0; double mx = 0, sum = 0;
    for (int i = 0; i < 1024; i++) {
        // bf16 -> f32
        unsigned int b = ((unsigned int) y[i]) << 16;
        float f; memcpy(&f, &b, 4);
        if (isfinite(f)) nfin++;
        double a = fabs((double) f); if (a > mx) mx = a; sum += a;
    }
    printf("out[0..1024]: finite=%d maxabs=%.4f meanabs=%.4f\n", nfin, mx, sum / 1024);

    t0 = now_us();
    for (int i = 0; i < iters; i++) axclrtEngineExecute(model, ectx, 0, io);
    t1 = now_us();
    printf("bench: %.0fus/exec (%d iters) => x28 layers = %.2fms/token (layers only)\n",
           (t1 - t0) / iters, iters, (t1 - t0) / iters * 28 / 1000.0);
    return 0;
}
