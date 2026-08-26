// probe_slots.c: determine the whole-layer engine's attention slot layout.
// Runs the decode group with a mask allowing exactly ONE slot; if the
// output tracks V_cache[j], slots are cache-major; if it tracks the engine's
// internal self slot, we learn the +1 convention. V[t] filled with t-scaled
// patterns so the winner is identifiable per head.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt.h"

#define CTX 2048
#define HID 1024

static unsigned seed = 777;
static float frand(void) { seed = seed * 1103515245 + 12345; return ((seed >> 8) & 0xFFFF) / 65535.0f * 2 - 1; }
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u >> 16) + ((u >> 16) & 1)); }
static float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = argc > 1 ? argv[1] : "/usr/local/share/ggml-axcl/layer/qwen3_p128_l0_together.axmodel";
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext c; axclrtCreateContext(&c, dl.devices[0]); axclrtSetCurrentContext(c);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    axclSetLogLevel(3);
    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile(path, &model)) { printf("load fail\n"); return 1; }
    axclrtEngineIOInfo info; axclrtEngineIO io;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);
    int ik = axclrtEngineGetInputIndexByName(info, "K_cache");
    int iv = axclrtEngineGetInputIndexByName(info, "V_cache");
    int ii = axclrtEngineGetInputIndexByName(info, "indices");
    int ix = axclrtEngineGetInputIndexByName(info, "input");
    int im = axclrtEngineGetInputIndexByName(info, "mask");
    int iko = axclrtEngineGetOutputIndexByName(info, "K_cache_out");
    int ivo = axclrtEngineGetOutputIndexByName(info, "V_cache_out");
    int iyo = axclrtEngineGetOutputIndexByName(info, "output");
    printf("idx %d %d %d %d %d | %d %d %d\n", ik, iv, ii, ix, im, iko, ivo, iyo);

    const size_t kvb = (size_t)CTX * HID * 2;
    void *dk, *dv, *di, *dx, *dm, *dko, *dvo, *dyo;
    axclrtMalloc(&dk, kvb, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dv, kvb, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&di, 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dx, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm, (CTX + 1) * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dko, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvo, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dyo, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);

    // K/V: per-slot constant pattern so the attended slot is identifiable:
    // v[t*1024 + d] = t + d/1024; k similar (q·k picks a slot via mask anyway)
    uint16_t *kb = malloc(kvb), *vb = malloc(kvb);
    for (int t = 0; t < CTX; t++)
        for (int d = 0; d < HID; d++) {
            kb[(size_t)t * HID + d] = f2b(0.01f * (t % 100) + 0.0001f * d);
            vb[(size_t)t * HID + d] = f2b(0.02f * (t % 100) + 0.0001f * d);
        }
    uint16_t *xb = malloc(HID * 2);
    for (int d = 0; d < HID; d++) xb[d] = f2b(frand());
    uint16_t *mb = malloc((CTX + 1) * 2);
    uint16_t *out = malloc(HID * 2), *krow = malloc(HID * 2);

    struct { int slot; uint32_t idx; } probes[] = {
        {0, 0}, {1, 0}, {5, 0}, {100, 0}, {100, 50}, {2047, 0},
        {-1, 0},   // -1 = the SELF slot (index 2048 in mask)
        {-1, 5},
    };
    for (unsigned p = 0; p < sizeof(probes) / sizeof(probes[0]); p++) {
        int slot = probes[p].slot;
        uint32_t idx = probes[p].idx;
        for (int t = 0; t <= CTX; t++) {
            float v = (t == CTX) ? ((slot == -1) ? 0.0f : -1e9f)
                                 : ((t == slot) ? 0.0f : -1e9f);
            mb[t] = f2b(v);
        }
        axclrtMemcpy(dk, kb, kvb, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dv, vb, kvb, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(di, &idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dx, xb, HID * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dm, mb, (CTX + 1) * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, ik, dk, kvb);
        axclrtEngineSetInputBufferByIndex(io, iv, dv, kvb);
        axclrtEngineSetInputBufferByIndex(io, ii, di, 4);
        axclrtEngineSetInputBufferByIndex(io, ix, dx, HID * 2);
        axclrtEngineSetInputBufferByIndex(io, im, dm, (CTX + 1) * 2);
        axclrtEngineSetOutputBufferByIndex(io, iko, dko, HID * 2);
        axclrtEngineSetOutputBufferByIndex(io, ivo, dvo, HID * 2);
        axclrtEngineSetOutputBufferByIndex(io, iyo, dyo, HID * 2);
        if (axclrtEngineExecute(model, ectx, 0, io)) { printf("exec fail\n"); return 1; }
        axclrtMemcpy(out, dyo, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
        axclrtMemcpy(krow, dko, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
        // report: mean of out over all dims + first 4 head-0 values
        double s = 0;
        for (int d = 0; d < HID; d++) s += b2f(out[d]);
        printf("slot=%4d idx=%u: out mean=%9.5f  h0[0..3]=%.4f %.4f %.4f %.4f  krow[0..1]=%.4f %.4f\n",
               slot, idx, s / HID, b2f(out[0]), b2f(out[1]), b2f(out[2]), b2f(out[3]),
               b2f(krow[0]), b2f(krow[1]));
    }
    return 0;
}
