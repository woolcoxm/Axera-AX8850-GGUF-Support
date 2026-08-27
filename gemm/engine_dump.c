// dump whole-layer engine outputs for A/B comparison of patched engines
// deterministic input (same convention as test_vendor_layer.c)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "axcl.h"
#include "axcl_rt.h"

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "/home/kram/Qwen3-0.6B/qwen3_p128_l0_together.axmodel";
    const char * outp = argc > 2 ? argv[2] : "/tmp/edump.bin";
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
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

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

    // seeded generic inputs (all-ones makes attention scores exactly tied ->
    // softmax amplifies tiny weight diffs chaotically)
    unsigned int seed = 12345;
    unsigned short * kv = malloc(kv_bytes);
    for (size_t i = 0; i < kv_bytes / 2; i++) {
        seed = seed * 1664525u + 1013904223u;
        float v = ((seed >> 8) % 2000 - 1000) / 1000.0f * 0.05f;
        unsigned int b; memcpy(&b, &v, 4);
        kv[i] = (unsigned short)(b >> 16);
    }
    unsigned int idx = 0;
    unsigned short * mask = malloc((size_t) 2049 * 2);
    for (int i = 0; i < 2049; i++) {
        // row 0: allow nothing before + the engine's internal self slot (2048)
        mask[i] = (i == 2048) ? 0x3f80 : 0xBF80;
    }
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
    axclrtEngineSetInputBufferByIndex(io, ik, dk, kv_bytes);
    axclrtEngineSetInputBufferByIndex(io, iv, dv, kv_bytes);
    axclrtEngineSetInputBufferByIndex(io, ii, di, 4);
    axclrtEngineSetInputBufferByIndex(io, ix, dx, 2048);
    axclrtEngineSetInputBufferByIndex(io, im, dm, (size_t) 2049 * 2);
    axclrtEngineSetOutputBufferByIndex(io, iko, dko, kv_bytes);
    axclrtEngineSetOutputBufferByIndex(io, ivo, dvo, kv_bytes);
    axclrtEngineSetOutputBufferByIndex(io, iyo, dy, 2048);

    if (axclrtEngineExecute(model, ectx, 0, io)) { printf("EXEC FAIL\n"); return 1; }

    // dump: y[1024] bf16 + K_out row 7 [1024] + V_out row 7 [1024]
    FILE * f = fopen(outp, "wb");
    unsigned short * y = malloc(2048);
    axclrtMemcpy(y, dy, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    fwrite(y, 2, 1024, f);
    unsigned short * krow = malloc(2048);
    axclrtMemcpy(krow, dko, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    fwrite(krow, 2, 1024, f);
    unsigned short * vrow = malloc(2048);
    axclrtMemcpy(vrow, dvo, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    fwrite(vrow, 2, 1024, f);
    fclose(f);
    printf("dumped %s (y + K7 + V7, 3x1024 bf16)\n", outp);
    fflush(stdout);
    _exit(0);
    return 0;
}
