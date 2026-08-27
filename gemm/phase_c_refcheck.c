// phase_c_refcheck.c — chunk-group correctness gate.
// Invariant: K_cache_out row i from an m-token chunk group must equal the
// decode group's K_cache_out for the same input row at position i (K is a
// pure function of x and rope pos; attention plays no role).
// PASS => the m>1 group computes real per-token projections from the bound
//         input => usable for batched prefill / speculative verification.
// build: gcc phase_c_refcheck.c -I/usr/include/axcl -L/usr/lib/axcl \
//        -laxcl_rt -laxcl_sys -Wl,-rpath,/usr/lib/axcl -o phase_c_refcheck -lm
// usage: ./phase_c_refcheck <axmodel> <chunk_group>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
static float f16(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <axmodel> <chunk_group>\n", argv[0]); return 2; }
    const int g = atoi(argv[2]);
    if (axclInit(NULL) != AXCL_SUCC) return 2;
    axclrtDeviceList dl; memset(&dl, 0, sizeof(dl));
    axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx = 0; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    uint64_t model = 0;
    if (axclrtEngineLoadFromFile(argv[1], &model)) { printf("LOAD FAIL\n"); return 2; }
    axclrtEngineIOInfo info = NULL;
    axclrtEngineGetIOInfo(model, &info);
    uint32_t n_in = axclrtEngineGetNumInputs(info);
    uint32_t n_out = axclrtEngineGetNumOutputs(info);
    int ix = -1, iiko = -1, iivo = -1, ioy = -1, iidx = -1, imask = -1;
    uint64_t x_sz = 0;
    for (uint32_t i = 0; i < n_in; i++) {
        const char * nm = axclrtEngineGetInputNameByIndex(info, i);
        if (nm && strcmp(nm, "input") == 0) ix = i;
        if (nm && strcmp(nm, "indices") == 0) iidx = i;
        if (nm && strcmp(nm, "mask") == 0) imask = i;
    }
    for (uint32_t i = 0; i < n_out; i++) {
        const char * nm = axclrtEngineGetOutputNameByIndex(info, i);
        if (nm && strcmp(nm, "K_cache_out") == 0) iiko = i;
        if (nm && strcmp(nm, "V_cache_out") == 0) iivo = i;
        if (nm && strcmp(nm, "output") == 0) ioy = i;
    }
    if (ix < 0 || iiko < 0) { printf("io not found\n"); return 2; }
    x_sz = axclrtEngineGetInputSizeByIndex(info, g, ix);
    const int m = (int) ((x_sz / 2) / 1024);

    // ---- dedicated IO handles: one for decode group 0, one for chunk group g
    axclrtEngineIO io0 = 0, iog = 0;
    axclrtEngineCreateIO(info, &io0);
    axclrtEngineCreateIO(info, &iog);
    uint64_t ectx = 0; axclrtEngineCreateContext(model, &ectx);

    // dedicated EXACT-size buffers per group (the runtime mishandles
    // sub-buffer/offset bindings — never bind a slice of a larger buffer)
    uint64_t kv0 = axclrtEngineGetInputSizeByIndex(info, 0, 0);   // decode K_cache
    uint64_t m0 = axclrtEngineGetInputSizeByIndex(info, 0, imask);
    uint64_t kvo0 = axclrtEngineGetOutputSizeByIndex(info, 0, iiko);
    uint64_t kvog = axclrtEngineGetOutputSizeByIndex(info, g, iiko);
    uint64_t kvo0v = iivo >= 0 ? axclrtEngineGetOutputSizeByIndex(info, 0, iivo) : 0;
    uint64_t kvogv = iivo >= 0 ? axclrtEngineGetOutputSizeByIndex(info, g, iivo) : 0;
    void *dkv0, *dvv0, *didx0, *dx0, *dm0, *dko0, *dvo0, *dy0;
    void *dkvg, *dvvg, *didxg, *dxg, *dmg, *dkog, *dvog, *dyg;
    uint64_t idx0_sz = axclrtEngineGetInputSizeByIndex(info, 0, iidx);
    uint64_t x0_sz = axclrtEngineGetInputSizeByIndex(info, 0, ix);
    uint64_t y0_sz = axclrtEngineGetOutputSizeByIndex(info, 0, ioy);
    uint64_t idxg_sz = axclrtEngineGetInputSizeByIndex(info, g, iidx);
    uint64_t yg_sz = axclrtEngineGetOutputSizeByIndex(info, g, ioy);
    uint64_t kvg_kv = axclrtEngineGetInputSizeByIndex(info, g, 0);
    uint64_t mkg2 = axclrtEngineGetInputSizeByIndex(info, g, imask);
    axclrtMalloc(&dkv0, kv0, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvv0, kv0, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&didx0, idx0_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dx0, x0_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm0, m0, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dko0, kvo0, AXCL_MEM_MALLOC_HUGE_FIRST);
    if (kvo0v) axclrtMalloc(&dvo0, kvo0v, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dy0, y0_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dkvg, kvg_kv, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvvg, kvg_kv, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&didxg, idxg_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dxg, x_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dmg, mkg2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dkog, kvog, AXCL_MEM_MALLOC_HUGE_FIRST);
    if (kvogv) axclrtMalloc(&dvog, kvogv, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dyg, yg_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    // zero K/V caches and masks
    static char z[4096]; memset(z, 0, sizeof(z));
    for (uint64_t off = 0; off < kv0; off += sizeof(z))
        axclrtMemcpy((char *) dkv0 + off, z, (kv0 - off < sizeof(z)) ? kv0 - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
    for (uint64_t off = 0; off < m0; off += sizeof(z))
        axclrtMemcpy((char *) dm0 + off, z, (m0 - off < sizeof(z)) ? m0 - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
    for (uint64_t off = 0; off < kv0; off += sizeof(z))
        axclrtMemcpy((char *) dvv0 + off, z, (kv0 - off < sizeof(z)) ? kv0 - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
    for (uint64_t off = 0; off < kvg_kv; off += sizeof(z))
        axclrtMemcpy((char *) dkvg + off, z, (kvg_kv - off < sizeof(z)) ? kvg_kv - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
    for (uint64_t off = 0; off < mkg2; off += sizeof(z))
        axclrtMemcpy((char *) dmg + off, z, (mkg2 - off < sizeof(z)) ? mkg2 - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
    for (uint64_t off = 0; off < kvg_kv; off += sizeof(z))
        axclrtMemcpy((char *) dvvg + off, z, (kvg_kv - off < sizeof(z)) ? kvg_kv - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);

    // seeded input rows (distinct per row so K rows differ)
    uint16_t * hx = malloc(x_sz);
    unsigned s = 12345;
    for (uint64_t i = 0; i < (uint64_t) m * 1024; i++) {
        s = s * 1103515245u + 12345u;
        hx[i] = bf16(((float) ((s >> 16) % 2000) - 1000.0f) / 1000.0f * 0.5f);
    }

    // ---- decode group 0 reference: one row at a time, pos = 0..m-1
    axclrtEngineSetInputBufferByIndex(io0, axclrtEngineGetInputIndexByName(info, "K_cache"), dkv0, kv0);
    axclrtEngineSetInputBufferByIndex(io0, axclrtEngineGetInputIndexByName(info, "V_cache"), dvv0, kv0);
    axclrtEngineSetInputBufferByIndex(io0, iidx, didx0, idx0_sz);
    axclrtEngineSetInputBufferByIndex(io0, ix, dx0, x0_sz);
    axclrtEngineSetInputBufferByIndex(io0, imask, dm0, m0);
    axclrtEngineSetOutputBufferByIndex(io0, iiko, dko0, kvo0);
    if (kvo0v) axclrtEngineSetOutputBufferByIndex(io0, iivo, dvo0, kvo0v);
    axclrtEngineSetOutputBufferByIndex(io0, ioy, dy0, y0_sz);
    uint16_t * kref = malloc((size_t) m * 2048);
    for (int p = 0; p < m; p++) {
        uint32_t pos = (uint32_t) p;
        axclrtMemcpy(didx0, &pos, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dx0, hx + (size_t) p * 1024, 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
        if (axclrtEngineExecute(model, ectx, 0, io0) != AXCL_SUCC) { printf("exec g0 p=%d fail\n", p); return 2; }
        axclrtMemcpy(kref + (size_t) p * 1024, dko0, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    }

    // ---- chunk group g: all m rows, indices 0..m-1, causal mask (j<=i visible)
    {
        uint32_t * idx = malloc((size_t) m * 4);
        for (int i = 0; i < m; i++) idx[i] = (uint32_t) i;
        axclrtMemcpy(didxg, idx, (size_t) m * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dxg, hx, x_sz, AXCL_MEMCPY_HOST_TO_DEVICE);
        // mask [1, m, w]: the group's declared mask width w comes from size
        int w = (int) (mkg2 / 2 / m);
        uint16_t * mk = malloc(mkg2);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < w; j++)
                mk[(size_t) i * w + j] = (j <= i) ? bf16(0.0f) : bf16(-1e9f);
        axclrtMemcpy(dmg, mk, mkg2, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(iog, axclrtEngineGetInputIndexByName(info, "K_cache"), dkvg, kvg_kv);
        axclrtEngineSetInputBufferByIndex(iog, axclrtEngineGetInputIndexByName(info, "V_cache"), dvvg, kvg_kv);
        axclrtEngineSetInputBufferByIndex(iog, iidx, didxg, idxg_sz);
        axclrtEngineSetInputBufferByIndex(iog, ix, dxg, x_sz);
        axclrtEngineSetInputBufferByIndex(iog, imask, dmg, mkg2);
        axclrtEngineSetOutputBufferByIndex(iog, iiko, dkog, kvog);
        if (kvogv) axclrtEngineSetOutputBufferByIndex(iog, iivo, dvog, kvogv);
        axclrtEngineSetOutputBufferByIndex(iog, ioy, dyg, yg_sz);
        if (axclrtEngineExecute(model, ectx, g, iog) != AXCL_SUCC) { printf("exec g%d fail\n", g); return 2; }
    }
    uint16_t * kchunk = malloc((size_t) m * 2048);
    axclrtMemcpy(kchunk, dkog, (size_t) m * 2048, AXCL_MEMCPY_DEVICE_TO_HOST);

    // ---- compare
    double worst = 0; int worst_row = -1; double rel_worst = 0;
    for (int p = 0; p < m; p++) {
        double dmax = 0, denom = 1e-6;
        for (int k = 0; k < 1024; k++) {
            double a = f16(kchunk[(size_t) p * 1024 + k]);
            double b = f16(kref[(size_t) p * 1024 + k]);
            double d = fabs(a - b);
            denom = fabs(b) > denom ? fabs(b) : denom;
            if (d > dmax) dmax = d;
        }
        double rel = dmax / denom;
        if (rel > rel_worst) { rel_worst = rel; worst = dmax; worst_row = p; }
    }
    printf("engine=%s chunk_group=%d m=%d\n", argv[1], g, m);
    printf("K MATCH: worst row %d abs=%.4g rel=%.4g -> %s\n", worst_row, worst, rel_worst,
           rel_worst < 1e-2 ? "MATCH (chunk groups compute real per-token projections)" : "MISMATCH");
    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
