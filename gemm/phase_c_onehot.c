// phase_c_onehot.c — do m>1 shape groups READ the bound input, causally?
// For a chosen group: fill all m input rows with bf16(0.01), execute, snapshot
// output; then set row (m-1) to bf16(2.0), execute again, diff outputs.
// PASS = output rows >= changed row move, rows < changed row identical (causal
// input binding). FAIL-IGNORED = zero output movement (the vendor ladder bug).
// build: gcc phase_c_onehot.c -I/usr/include/axcl -L/usr/lib/axcl -laxcl_rt \
//          -laxcl_sys -Wl,-rpath,/usr/lib/axcl -o phase_c_onehot
// usage: ./phase_c_onehot <axmodel> <group> [flip_row]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

static uint16_t f32_to_bf16_bits(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
static float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <axmodel> <group> [flip_row]\n", argv[0]); return 2; }
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
    axclrtEngineIO io = 0; axclrtEngineCreateIO(info, &io);
    uint64_t ectx = 0; axclrtEngineCreateContext(model, &ectx);

    // find "input" and "output" by name; bind all IO with zeroed buffers
    int ix = -1, oy = -1;
    static void * din[16], * dout[16];
    uint64_t x_sz = 0, y_sz = 0;
    for (uint32_t i = 0; i < n_in && i < 16; i++) {
        const char * nm = axclrtEngineGetInputNameByIndex(info, i);
        uint64_t sz = axclrtEngineGetInputSizeByIndex(info, g, i);
        axclrtMalloc(&din[i], sz ? sz : 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        static char z[4096]; memset(z, 0, sizeof(z));
        for (uint64_t off = 0; off < sz; off += sizeof(z))
            axclrtMemcpy((char *) din[i] + off, z, (sz - off < sizeof(z)) ? sz - off : sizeof(z), AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, i, din[i], sz);
        printf("in[%u] %s = %llu B\n", i, nm ? nm : "?", (unsigned long long) sz);
        if (nm && strcmp(nm, "input") == 0) { ix = i; x_sz = sz; }
        if (nm && strcmp(nm, "indices") == 0) {
            // rope positions 0..m-1, deduced from the index tensor size
            uint32_t m_idx = (uint32_t) (sz / 4);
            uint32_t * idx = malloc(sz);
            for (uint32_t k = 0; k < m_idx; k++) idx[k] = k;
            axclrtMemcpy(din[i], idx, sz, AXCL_MEMCPY_HOST_TO_DEVICE);
            free(idx);
        }
    }
    for (uint32_t i = 0; i < n_out && i < 16; i++) {
        const char * nm = axclrtEngineGetOutputNameByIndex(info, i);
        uint64_t sz = axclrtEngineGetOutputSizeByIndex(info, g, i);
        axclrtMalloc(&dout[i], sz ? sz : 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtEngineSetOutputBufferByIndex(io, i, dout[i], sz);
        if (nm && strcmp(nm, "output") == 0) { oy = i; y_sz = sz; }
    }
    if (ix < 0 || oy < 0) { printf("no input/output tensors\n"); return 2; }
    const uint64_t n_elem = x_sz / 2;          // bf16 elements
    const int m = (int) (n_elem / 1024);        // rows of 1024
    const int flip = argc > 3 ? atoi(argv[3]) : m - 1;
    printf("group=%d m=%d x=%lluB y=%lluB flip_row=%d\n", g, m,
           (unsigned long long) x_sz, (unsigned long long) y_sz, flip);

    // fill input rows: 0.01 everywhere
    uint16_t * hx = malloc(x_sz);
    for (uint64_t i = 0; i < n_elem; i++) hx[i] = f32_to_bf16_bits(0.01f);
    axclrtMemcpy(din[ix], hx, x_sz, AXCL_MEMCPY_HOST_TO_DEVICE);
    if (axclrtEngineExecute(model, ectx, g, io) != AXCL_SUCC) { printf("EXEC FAIL\n"); return 2; }
    uint16_t * y1 = malloc(y_sz ? y_sz : 4);
    axclrtMemcpy(y1, dout[oy], y_sz, AXCL_MEMCPY_DEVICE_TO_HOST);

    // flip one row to 2.0 (large contrast)
    for (int k = 0; k < 1024; k++) hx[(size_t) flip * 1024 + k] = f32_to_bf16_bits(2.0f);
    axclrtMemcpy(din[ix], hx, x_sz, AXCL_MEMCPY_HOST_TO_DEVICE);
    if (axclrtEngineExecute(model, ectx, g, io) != AXCL_SUCC) { printf("EXEC FAIL 2\n"); return 2; }
    uint16_t * y2 = malloc(y_sz ? y_sz : 4);
    axclrtMemcpy(y2, dout[oy], y_sz, AXCL_MEMCPY_DEVICE_TO_HOST);

    // compare per output row (rows of 1024 bf16), y rows correspond to m inputs
    const int ym = (int) (y_sz / 2 / 1024);
    double max_below = 0, max_atabove = 0;
    int rows_moved = 0;
    for (int r = 0; r < ym; r++) {
        double dmax = 0;
        for (int k = 0; k < 1024; k++) {
            double d = fabsf(bf16_bits_to_f32(y2[(size_t) r * 1024 + k]) - bf16_bits_to_f32(y1[(size_t) r * 1024 + k]));
            if (d > dmax) dmax = d;
        }
        if (dmax > 1e-6) rows_moved++;
        if (r < flip && dmax > max_below) max_below = dmax;
        if (r >= flip && dmax > max_atabove) max_atabove = dmax;
    }
    printf("VERDICT: rows_moved=%d/%d  maxdiff_below_flip=%.3g  maxdiff_at_above_flip=%.3g  ->  %s\n",
           rows_moved, ym, max_below, max_atabove,
           (max_atabove > 1e-3 && max_below < 1e-3) ? "CAUSAL-INPUT-BOUND (USABLE)"
           : (max_atabove > 1e-3) ? "INPUT-BOUND-NONCAUSAL"
           : "INPUT-IGNORED (vendor ladder bug)");
    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
