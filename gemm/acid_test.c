// acid_test.c: does a nibble+scale-only patched engine compute identically
// to the fully-baked reference engine (decode shape group)?
//
// usage: acid_test <patched.axmodel> <reference.axmodel> [iterations]
// Both engines get identical random inputs; outputs compared element-wise.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt.h"

#define CTX 256
#define HID 1024

static unsigned int seed = 12345;
static float frand(void) {
    seed = seed * 1103515245 + 12345;
    return ((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t rounded = (u >> 16) & 1;
    return (uint16_t)((u >> 16) + rounded);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct eng {
    uint64_t model, ectx;
    axclrtEngineIOInfo info;
    axclrtEngineIO io;
    void *dk, *dv, *di, *dx, *dm, *dko, *dvo, *dyo;
    int ik, iv, ii, ix, im, iko, ivo, iyo;
};

static int load_engine(struct eng * e, const char * path) {
    memset(e, 0, sizeof(*e));
    FILE * f = fopen(path, "rb");
    if (!f) { printf("open fail %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void * buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { printf("read fail\n"); return -1; }
    fclose(f);
    printf("LoadFromMem %s (%ld bytes)...\n", path, sz);
    fflush(stdout);
    if (axclrtEngineLoadFromMem(buf, sz, &e->model)) {
        printf("LoadFromMem FAIL %s\n", path);
        return -1;
    }
    printf("LoadFromMem ok -> model=%llx\n", (unsigned long long)e->model);
    if (axclrtEngineGetIOInfo(e->model, &e->info)) return -1;
    if (axclrtEngineCreateIO(e->info, &e->io)) return -1;
    if (axclrtEngineCreateContext(e->model, &e->ectx)) return -1;
    printf("loaded %s (%ld bytes, model=%llx)\n", path, sz, (unsigned long long)e->model);

    e->ik = axclrtEngineGetInputIndexByName(e->info, "K_cache");
    e->iv = axclrtEngineGetInputIndexByName(e->info, "V_cache");
    e->ii = axclrtEngineGetInputIndexByName(e->info, "indices");
    e->ix = axclrtEngineGetInputIndexByName(e->info, "input");
    e->im = axclrtEngineGetInputIndexByName(e->info, "mask");
    e->iko = axclrtEngineGetOutputIndexByName(e->info, "K_cache_out");
    e->ivo = axclrtEngineGetOutputIndexByName(e->info, "V_cache_out");
    e->iyo = axclrtEngineGetOutputIndexByName(e->info, "output");
    printf("  idx: k=%d v=%d i=%d x=%d m=%d | ko=%d vo=%d yo=%d\n",
           e->ik, e->iv, e->ii, e->ix, e->im, e->iko, e->ivo, e->iyo);
    if (e->ik < 0 || e->iv < 0 || e->ii < 0 || e->ix < 0 || e->im < 0 ||
        e->iko < 0 || e->ivo < 0 || e->iyo < 0) {
        printf("  IO index lookup FAILED\n");
        return -1;
    }
    const size_t kv = (size_t)CTX * HID * 2;
    axclrtMalloc(&e->dk, kv, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dv, kv, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->di, 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dx, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dm, (CTX + 1) * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dko, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dvo, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dyo, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    return 0;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) { printf("usage: acid_test <patched> <reference> [iters]\n"); return 1; }
    const int iters = argc > 3 ? atoi(argv[3]) : 1;
    printf("init axcl...\n");
    axclInit(NULL);
    printf("axclInit ok\n");
    axclrtDeviceList dl;
    axclrtGetDeviceList(&dl);
    printf("devicelist num=%u first=%u\n", dl.num, dl.num ? dl.devices[0] : 9999);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx;
    axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    printf("context ok, engine init...\n");
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    printf("engine init ok\n");
    axclSetLogLevel(3);

    struct eng A, B;
    if (load_engine(&A, argv[1])) return 1;
    if (load_engine(&B, argv[2])) return 1;
    printf("engines loaded\n");

    // shared inputs
    const size_t kv_elems = (size_t)CTX * HID;
    uint16_t * kb = malloc(kv_elems * 2), * vb = malloc(kv_elems * 2);
    uint16_t * xb = malloc(HID * 2), * mb = malloc((CTX + 1) * 2);
    uint32_t idx = 5;
    for (size_t i = 0; i < kv_elems; i++) {
        kb[i] = f32_to_bf16(frand() * 0.5f);
        vb[i] = f32_to_bf16(frand() * 0.5f);
    }
    for (int i = 0; i < HID; i++) xb[i] = f32_to_bf16(frand());
    for (int t = 0; t <= CTX; t++) mb[t] = f32_to_bf16(t <= (int)idx ? 0.0f : -1e9f);

    uint16_t * outA = malloc(HID * 2), * outB = malloc(HID * 2);
    uint16_t * krowA = malloc(HID * 2), * krowB = malloc(HID * 2);

    double worst = 0, worstk = 0;
    int nonzero = 0;
    for (int it = 0; it < iters; it++) {
        // fresh random inputs each iter
        for (size_t i = 0; i < kv_elems; i++) {
            kb[i] = f32_to_bf16(frand() * 0.5f);
            vb[i] = f32_to_bf16(frand() * 0.5f);
        }
        for (int i = 0; i < HID; i++) xb[i] = f32_to_bf16(frand());
        struct eng * es[2] = {&A, &B};
        uint16_t * outs[2] = {outA, outB};
        uint16_t * krs[2] = {krowA, krowB};
        for (int e = 0; e < 2; e++) {
            axclrtMemcpy(es[e]->dk, kb, kv_elems * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(es[e]->dv, vb, kv_elems * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(es[e]->di, &idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(es[e]->dx, xb, HID * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(es[e]->dm, mb, (CTX + 1) * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtEngineIO io = es[e]->io;
            axclrtEngineSetInputBufferByIndex(io, es[e]->ik, es[e]->dk, (size_t)CTX * HID * 2);
            axclrtEngineSetInputBufferByIndex(io, es[e]->iv, es[e]->dv, (size_t)CTX * HID * 2);
            axclrtEngineSetInputBufferByIndex(io, es[e]->ii, es[e]->di, 4);
            axclrtEngineSetInputBufferByIndex(io, es[e]->ix, es[e]->dx, HID * 2);
            axclrtEngineSetInputBufferByIndex(io, es[e]->im, es[e]->dm, (CTX + 1) * 2);
            axclrtEngineSetOutputBufferByIndex(io, es[e]->iko, es[e]->dko, HID * 2);
            axclrtEngineSetOutputBufferByIndex(io, es[e]->ivo, es[e]->dvo, HID * 2);
            axclrtEngineSetOutputBufferByIndex(io, es[e]->iyo, es[e]->dyo, HID * 2);
            if (axclrtEngineExecute(es[e]->model, es[e]->ectx, 0, io)) {
                printf("EXEC FAIL engine %d\n", e);
                return 1;
            }
            axclrtMemcpy(outs[e], es[e]->dyo, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
            axclrtMemcpy(krs[e], es[e]->dko, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
        }
        for (int i = 0; i < HID; i++) {
            double d = fabs(bf16_to_f32(outA[i]) - bf16_to_f32(outB[i]));
            if (d > worst) worst = d;
            if (fabs(bf16_to_f32(outA[i])) > 1e-3) nonzero++;
            d = fabs(bf16_to_f32(krowA[i]) - bf16_to_f32(krowB[i]));
            if (d > worstk) worstk = d;
        }
    }
    printf("RESULT: max |outA-outB| = %.6f, max |krowA-krowB| = %.6f (nonzero out elems: %d/%d)\n",
           worst, worstk, nonzero, HID * iters);
    printf(worst < 1e-2 ? "ACID TEST: PASS (patched == reference)\n"
                        : "ACID TEST: DIFFER (unpatched regions affect compute)\n");
    return 0;
}
