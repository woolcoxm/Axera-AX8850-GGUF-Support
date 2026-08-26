// NPU matmul correctness test — canonical axcl-samples IO pattern:
// axclrtMalloc device buffers + axclrtMemcpy, device pointers bound to IO.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_memory.h>

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff, f;
    if (exp == 0) f = (sign << 31) | (frac << 13);
    else if (exp == 31) f = (sign << 31) | 0x7f800000 | (frac << 13);
    else { exp -= 15; f = (sign << 31) | ((exp + 127) << 23) | (frac << 13); }
    float out; memcpy(&out, &f, 4); return out;
}

int main() {
    setbuf(stdout, NULL);
    printf("init=%d\n", (int) axclInit(NULL));
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    int dev = dl.devices[0];
    printf("slot=%d setdev=%d\n", dev, (int) axclrtSetDevice(dev));

    axclrtContext ctx = 0;
    printf("createctx=%d\n", (int) axclrtCreateContext(&ctx, dev));
    axclrtSetCurrentContext(ctx);
    printf("engineInit=%d\n", (int) axclrtEngineInit(AXCL_VNPU_DISABLE));

    uint64_t model = 0;
    axclError e = axclrtEngineLoadFromFile("/home/kram/matmul/matmul_m1_k2048_n2048.axmodel", &model);
    printf("load=%d\n", (int) e);
    if (e) return 1;

    axclrtEngineIOInfo info = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineIO io = 0;
    axclrtEngineCreateIO(info, &io);

    uint64_t ectx = 0;
    axclrtEngineCreateContext(model, &ectx);

    // ---- reference inputs (host) ----
#define K 2048
#define N 2048
    static float xh[K], wh[K * N], yref[N];
    FILE *f = fopen("/home/kram/matmul/x.f32", "rb"); fread(xh, 4, K, f); fclose(f);
    f = fopen("/home/kram/matmul/w.f32", "rb");       fread(wh, 4, K * N, f); fclose(f);
    f = fopen("/home/kram/matmul/y_ref.f32", "rb");   fread(yref, 4, N, f); fclose(f);

    // ---- device buffers + H2D ----
    void *dx = 0, *dw = 0, *dy = 0;
    axclrtMalloc(&dx, K * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dw, (size_t) K * N * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dy, N * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    printf("malloc dx=%p dw=%p dy=%p\n", dx, dw, dy);
    printf("h2d x=%d w=%d\n",
        (int) axclrtMemcpy(dx, xh, K * 4, AXCL_MEMCPY_HOST_TO_DEVICE),
        (int) axclrtMemcpy(dw, wh, (size_t) K * N * 4, AXCL_MEMCPY_HOST_TO_DEVICE));

    int xi = axclrtEngineGetInputIndexByName(info, "X");
    int wi = axclrtEngineGetInputIndexByName(info, "W");
    int yi = axclrtEngineGetOutputIndexByName(info, "Y");
    printf("idx X=%d W=%d Y=%d\n", xi, wi, yi);

    printf("bindX=%d bindW=%d bindY=%d\n",
        (int) axclrtEngineSetInputBufferByIndex(io, xi, dx, K * 4),
        (int) axclrtEngineSetInputBufferByIndex(io, wi, dw, (size_t) K * N * 4),
        (int) axclrtEngineSetOutputBufferByIndex(io, yi, dy, N * 4));

    e = axclrtEngineExecute(model, ectx, 0, io);
    printf("execute=%d\n", (int) e);
    if (e) { fflush(stdout); _exit(1); }

    static float yh[N];
    printf("d2h=%d\n", (int) axclrtMemcpy(yh, dy, N * 4, AXCL_MEMCPY_DEVICE_TO_HOST));

    double maxerr = 0, sum = 0;
    for (int i = 0; i < N; i++) {
        double d = fabs((double) yh[i] - yref[i]);
        if (d > maxerr) maxerr = d;
        sum += d;
    }
    printf("RESULT: max_err=%.5f mean_err=%.6f\n", maxerr, sum / N);
    printf("y[0..3]:  %.4f %.4f %.4f %.4f\n", yh[0], yh[1], yh[2], yh[3]);
    printf("ref[0..3]: %.4f %.4f %.4f %.4f\n", yref[0], yref[1], yref[2], yref[3]);
    fflush(stdout);
    _exit(0);
}
