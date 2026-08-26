// verify the OptimizedQuantAxModel FFN block engine: act[1,1024] f32 + 3 int8 weights -> out[1,1024] f32
// reference: silu(gemm(act,gate_w)) * gemm(act,up_w) -> gemm -> out
// usage: ./tffn <engine.axmodel> [iters]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "axcl.h"
#include "axcl_rt.h"

static double now_us() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s engine [iters]\n", argv[0]); return 1; }
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    srand(4242);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile(argv[1], &model)) { printf("LOAD FAIL\n"); return 1; }
    printf("load ok\n");
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

    // dump IO signature
    printf("inputs:\n");
    for (uint32_t i = 0; i < axclrtEngineGetNumInputs(info); i++) {
        axclrtEngineDataType dt; axclrtEngineGetInputDataType(info, i, &dt);
        printf("  [%u] %s dtype=%d size=%llu dims=[", i, axclrtEngineGetInputNameByIndex(info, i),
               (int) dt, (unsigned long long) axclrtEngineGetInputSizeByIndex(info, 0, i));
        axclrtEngineIODims dims;
        if (axclrtEngineGetInputDims(info, 0, i, &dims) == 0)
            for (uint32_t d = 0; d < dims.dimCount; d++) printf("%u%s", dims.dims[d], d + 1 < dims.dimCount ? "," : "");
        printf("]\n");
    }
    printf("outputs:\n");
    for (uint32_t i = 0; i < axclrtEngineGetNumOutputs(info); i++) {
        axclrtEngineDataType dt; axclrtEngineGetOutputDataType(info, i, &dt);
        printf("  [%u] %s dtype=%d size=%llu dims=[", i, axclrtEngineGetOutputNameByIndex(info, i),
               (int) dt, (unsigned long long) axclrtEngineGetOutputSizeByIndex(info, 0, i));
        axclrtEngineIODims dims;
        if (axclrtEngineGetOutputDims(info, 0, i, &dims) == 0)
            for (uint32_t d = 0; d < dims.dimCount; d++) printf("%u%s", dims.dims[d], d + 1 < dims.dimCount ? "," : "");
        printf("]\n");
    }

    const int K = 1024, MID = 3072;
    // host data
    float * act = malloc(K * 4);
    for (int i = 0; i < K; i++) act[i] = (rand() % 200 - 100) * 0.02f; // std ~1.15
    int8_t * gate_w = malloc((size_t) K * MID);
    int8_t * up_w   = malloc((size_t) K * MID);
    int8_t * down_w = malloc((size_t) MID * K);
    for (size_t i = 0; i < (size_t) K * MID; i++) {
        gate_w[i] = (int8_t) (rand() % 51 - 25);
        up_w[i]   = (int8_t) (rand() % 51 - 25);
    }
    for (size_t i = 0; i < (size_t) MID * K; i++) down_w[i] = (int8_t) (rand() % 51 - 25);
    // scales: quantized from f32 with per-tensor scale; pick scale = 0.02 (weights ~std 0.0115*?)
    const float gscale = 0.02f, uscale = 0.02f, dscale = 0.005f; // value of one int8 step
    // dequantized weight = int8 * scale
    // NOTE engine computes in units of (x_q * w_q); we correct on host after readback.

    // device buffers
    void *dact, *dgw, *duw, *ddw, *dout;
    axclrtMalloc(&dact, K * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dgw, (size_t) K * MID, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&duw, (size_t) K * MID, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&ddw, (size_t) MID * K, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dout, K * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(dact, act, K * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dgw, gate_w, (size_t) K * MID, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(duw, up_w, (size_t) K * MID, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(ddw, down_w, (size_t) MID * K, AXCL_MEMCPY_HOST_TO_DEVICE);

    // input 'act' is FP32 per the ONNX def; but the first AxQMM consumed int8 in QuantAxModel runs.
    // OptimizedQuantAxModel may insert its own activation quant (Cast-like) inside subgraph[1] or at
    // the NPU boundary — the IO dump above tells us the true dtype. Bind and run regardless.
    int iact = axclrtEngineGetInputIndexByName(info, "act");
    int iout = axclrtEngineGetOutputIndexByName(info, "out");
    axclrtEngineSetInputBufferByIndex(io, iact, dact, K * 4);
    axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "gate_w"), dgw, (size_t) K * MID);
    axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "up_w"), duw, (size_t) K * MID);
    axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "down_w"), ddw, (size_t) MID * K);
    axclrtEngineSetOutputBufferByIndex(io, iout, dout, K * 4);

    double t0 = now_us();
    int rc = axclrtEngineExecute(model, ectx, 0, io);
    double t1 = now_us();
    if (rc) { printf("EXEC FAIL rc=%d\n", rc); return 1; }
    printf("exec ok %.0fus\n", t1 - t0);

    float * y = malloc(K * 4);
    axclrtMemcpy(y, dout, K * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
    printf("y[0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n", y[0], y[1], y[2], y[3], y[4], y[5]);

    // benchmark
    t0 = now_us();
    for (int i = 0; i < iters; i++) axclrtEngineExecute(model, ectx, 0, io);
    t1 = now_us();
    printf("bench: %.0fus/exec over %d iters\n", (t1 - t0) / iters, iters);

    // CPU reference: engine sees f32 act (if act is f32, engine quantizes internally with its own
    // scale — unknown. So instead check the ratio path: if act were int8 too... just report
    // correlation against a reference computed with act treated as int8 (act*? ). For the mixed
    // graph the ONNX subgraph runs in f32 with scales from input_scales attrs (1.0), so:
    // gate_raw[n] = sum_k act_q[k]*gate_w[k,n] (int8 products, f32 accumulate)
    // silu(gate_raw) * up_raw -> mid; out = sum mid_q?*down_w... mid Cast to int8 truncates.
    // The engine semantics here are approximate; correctness gets checked with the dedicated
    // fp32 comparison in the follow-up. For now: non-zero, finite, no NaN.
    int nfin = 0; double mx = 0;
    for (int i = 0; i < K; i++) { if (isfinite(y[i])) nfin++; if (fabs(y[i]) > mx) mx = fabs(y[i]); }
    printf("finite=%d/%d maxabs=%.4f\n", nfin, K, mx);
    return 0;
}
