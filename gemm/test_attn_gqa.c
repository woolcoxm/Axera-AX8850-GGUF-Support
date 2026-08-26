// GQA attention via 16-head engine: host repacks 8 KV heads to 16 (repeat 2x)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_memory.h>

#define HQ 16
#define HKV 8
#define G   (HQ / HKV)  // 2
#define D   64
#define T   512
#define SEQ 30

int main() {
    setbuf(stdout, NULL);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    axclError e = axclrtEngineLoadFromFile("/usr/local/share/ggml-axcl/attn_h16_d64_t512.axmodel", &model);
    printf("load=%d\n", (int) e); if (e) _exit(1);
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

    // GQA reference data: Q has 16 heads, K/V have 8
    static float qh[HQ * D], kv_h[HKV * T * D], vv_h[HKV * T * D], maskh[T];
    srand(42);
    for (int i = 0; i < HQ * D; i++) qh[i] = (rand() % 200 - 100) * 0.01f;
    for (int i = 0; i < HKV * T * D; i++) kv_h[i] = (rand() % 200 - 100) * 0.01f;
    for (int i = 0; i < HKV * T * D; i++) vv_h[i] = (rand() % 200 - 100) * 0.01f;
    for (int t = 0; t < T; t++) maskh[t] = (t < SEQ) ? 0.0f : -100000.0f;

    // repack KV from 8 heads to 16 (repeat each head G times)
    static float kh16[HQ * T * D], vh16[HQ * T * D];
    for (int hq = 0; hq < HQ; hq++) {
        int hk = hq / G;
        memcpy(kh16 + (size_t) hq * T * D, kv_h + (size_t) hk * T * D, (size_t) T * D * 4);
        memcpy(vh16 + (size_t) hq * T * D, vv_h + (size_t) hk * T * D, (size_t) T * D * 4);
    }

    // CPU reference with GQA
    static float ref[HQ * D];
    for (int h = 0; h < HQ; h++) {
        int hk = h / G;
        float scores[T], mx = -1e30f, sum = 0;
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qh[h * D + d] * kv_h[(hk * T + t) * D + d];
            scores[t] = s / sqrtf((float) D) + maskh[t];
            if (scores[t] > mx) mx = scores[t];
        }
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
        for (int d = 0; d < D; d++) {
            float acc = 0;
            for (int t = 0; t < T; t++) acc += (scores[t] / sum) * vv_h[(hk * T + t) * D + d];
            ref[h * D + d] = acc;
        }
    }

    // device buffers
    void *dq, *dk, *dv, *dm, *do_;
    axclrtMalloc(&dq, HQ * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dk, (size_t) HQ * T * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dv, (size_t) HQ * T * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm, T * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&do_, HQ * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(dq, qh, HQ * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dk, kh16, (size_t) HQ * T * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dv, vh16, (size_t) HQ * T * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(dm, maskh, T * 4, AXCL_MEMCPY_HOST_TO_DEVICE);

    int iq = axclrtEngineGetInputIndexByName(info, "Q");
    int ik = axclrtEngineGetInputIndexByName(info, "K");
    int iv = axclrtEngineGetInputIndexByName(info, "V");
    int im = axclrtEngineGetInputIndexByName(info, "mask");
    int io_ = axclrtEngineGetOutputIndexByName(info, "out");
    axclrtEngineSetInputBufferByIndex(io, iq, dq, HQ * D * 4);
    axclrtEngineSetInputBufferByIndex(io, ik, dk, (size_t) HQ * T * D * 4);
    axclrtEngineSetInputBufferByIndex(io, iv, dv, (size_t) HQ * T * D * 4);
    axclrtEngineSetInputBufferByIndex(io, im, dm, T * 4);
    axclrtEngineSetOutputBufferByIndex(io, io_, do_, HQ * D * 4);

    e = axclrtEngineExecute(model, ectx, 0, io);
    printf("execute=%d\n", (int) e); if (e) _exit(1);

    static float out[HQ * D];
    axclrtMemcpy(out, do_, HQ * D * 4, AXCL_MEMCPY_DEVICE_TO_HOST);

    double mx = 0, sum = 0;
    for (int i = 0; i < HQ * D; i++) {
        double d = fabs((double) out[i] - ref[i]);
        if (d > mx) mx = d; sum += d;
    }
    printf("GQA RESULT: max_err=%.5f mean_err=%.6f\n", mx, sum / (HQ * D));
    printf("out[0..3]:  %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3]);
    printf("ref[0..3]:  %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    fflush(stdout);
    _exit(0);
}
