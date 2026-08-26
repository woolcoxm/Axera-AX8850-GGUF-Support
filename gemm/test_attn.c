// Fused attention engine correctness test: NPU vs CPU reference
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_memory.h>

#define H 16
#define D 64
#define T 512
#define SEQ 30   // actual sequence length; mask hides the rest

int main() {
    setbuf(stdout, NULL);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    axclError e = axclrtEngineLoadFromFile("/home/kram/matmul/attn_test.axmodel", &model);
    printf("load=%d\n", (int) e); if (e) _exit(1);
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

    // host reference data
    static float qh[H * D], kh[H * T * D], vh[H * T * D], maskh[T];
    srand(42);
    for (int i = 0; i < H * D; i++) qh[i] = (rand() % 200 - 100) * 0.01f;
    for (int i = 0; i < H * T * D; i++) kh[i] = (rand() % 200 - 100) * 0.01f;
    for (int i = 0; i < H * T * D; i++) vh[i] = (rand() % 200 - 100) * 0.01f;
    for (int t = 0; t < T; t++) maskh[t] = (t < SEQ) ? 0.0f : -100000.0f;

    // CPU reference: softmax(q.k*scale + mask).v per head
    static float ref[H * D];
    for (int h = 0; h < H; h++) {
        float scores[T], mx = -1e30f, sum = 0;
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qh[h * D + d] * kh[(h * T + t) * D + d];
            scores[t] = s / sqrtf((float) D) + maskh[t];
            if (scores[t] > mx) mx = scores[t];
        }
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        for (int d = 0; d < D; d++) {
            float acc = 0;
            for (int t = 0; t < T; t++) acc += (scores[t] / sum) * vh[(h * T + t) * D + d];
            ref[h * D + d] = acc;
        }
    }

    // device buffers + H2D
    void *dq, *dk, *dv, *dm, *do_;
    axclrtMalloc(&dq, H * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dk, (size_t) H * T * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dv, (size_t) H * T * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm, T * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&do_, H * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(dq, qh, H * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dk, kh, (size_t) H * T * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dv, vh, (size_t) H * T * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dm, maskh, T * 4, AXCL_MEMCPY_HOST_TO_DEVICE);

    int iq = axclrtEngineGetInputIndexByName(info, "Q");
    int ik = axclrtEngineGetInputIndexByName(info, "K");
    int iv = axclrtEngineGetInputIndexByName(info, "V");
    int im = axclrtEngineGetInputIndexByName(info, "mask");
    int io_ = axclrtEngineGetOutputIndexByName(info, "out");
    axclrtEngineSetInputBufferByIndex(io, iq, dq, H * D * 4);
    axclrtEngineSetInputBufferByIndex(io, ik, dk, (size_t) H * T * D * 4);
    axclrtEngineSetInputBufferByIndex(io, iv, dv, (size_t) H * T * D * 4);
    axclrtEngineSetInputBufferByIndex(io, im, dm, T * 4);
    axclrtEngineSetOutputBufferByIndex(io, io_, do_, H * D * 4);

    e = axclrtEngineExecute(model, ectx, 0, io);
    printf("execute=%d\n", (int) e); if (e) _exit(1);

    static float out[H * D];
    axclrtMemcpy(out, do_, H * D * 4, AXCL_MEMCPY_DEVICE_TO_HOST);

    double mx = 0, sum = 0;
    for (int i = 0; i < H * D; i++) {
        double d = fabs((double) out[i] - ref[i]);
        if (d > mx) mx = d; sum += d;
    }
    printf("RESULT: max_err=%.5f mean_err=%.6f (int8 noise expected)\n", mx, sum / (H * D));
    printf("out[0..3]:  %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
    printf("ref[0..3]:  %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    fflush(stdout);
    _exit(0);
}
