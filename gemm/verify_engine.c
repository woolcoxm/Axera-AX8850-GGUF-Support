// generic engine verifier: fresh-seed realistic inputs vs CPU reference
// usage: ./ve <engine.axmodel> <gemm|qkv|gateup> <seed>
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

// transposed matmul: y[n] = sum_k w_t[k*n+n'] * x[k]  (w_t is [k,n] engine layout)
static void ref_mm(const float * w_t, const float * x, float * y, int k, int n) {
    for (int nn = 0; nn < n; nn++) y[nn] = 0;
    for (int kk = 0; kk < k; kk++) {
        const float xv = x[kk];
        const float * row = w_t + (size_t) kk * n;
        for (int nn = 0; nn < n; nn++) y[nn] += xv * row[nn];
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s engine mode seed\n", argv[0]); return 1; }
    const char * engine = argv[1], * mode = argv[2];
    const unsigned seed = (unsigned) atoi(argv[3]);
    srand(seed);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile(engine, &model)) { printf("load FAIL\n"); return 1; }
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);

    // fresh realistic data (different from any calibration seed)
    float * h  = malloc(1024 * 4);
    for (int i = 0; i < 1024; i++) h[i] = (rand() % 200 - 100) * 0.02f; // std ~1.15
    const int K = 1024;
    int outs[3]; size_t on[3];
    float * wq, * wk, * wv;
    void *dh, *dw1, *dw2, *dw3, *douts[3];
    if (!strcmp(mode, "gemm")) {
        const int N = atoi(argv[4]);
        wq = malloc((size_t) K * N * 4);
        for (size_t i = 0; i < (size_t) K * N; i++) wq[i] = (rand() % 200 - 100) * 0.0002f; // std ~0.0115
        axclrtMalloc(&dh, K * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&dw1, (size_t) K * N * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&douts[0], (size_t) N * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMemcpy(dh, h, K * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dw1, wq, (size_t) K * N * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "X"), dh, K * 4);
        axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "W"), dw1, (size_t) K * N * 4);
        axclrtEngineSetOutputBufferByIndex(io, axclrtEngineGetOutputIndexByName(info, "Y"), douts[0], (size_t) N * 4);
        float * yref = malloc((size_t) N * 4), * y = malloc((size_t) N * 4);
        ref_mm(wq, h, yref, K, N);
        double t0 = now_us();
        if (axclrtEngineExecute(model, ectx, 0, io)) { printf("exec FAIL\n"); return 1; }
        double t1 = now_us();
        axclrtMemcpy(y, douts[0], (size_t) N * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
        double mx = 0, sum = 0; size_t bad = 0;
        for (int i = 0; i < N; i++) {
            double d = fabs((double) y[i] - yref[i]);
            if (d > mx) mx = d; sum += d;
            if (d > 0.05) bad++;
        }
        printf("[%s gemm n=%d] exec=%.0fus max_err=%.5f mean_err=%.6f bad(>0.05)=%zu/%d\n",
               engine, N, t1 - t0, mx, sum / N, bad, N);
        return 0;
    }
    // qkv / gateup: h + up to 3 weights, parallel outputs
    int nw, onum;
    const char *wn[3], *onm[3];
    int nn[3];
    if (!strcmp(mode, "qkv")) {
        wn[0]="q_w"; wn[1]="k_w"; wn[2]="v_w"; onm[0]="q"; onm[1]="k"; onm[2]="v";
        nn[0]=2048; nn[1]=1024; nn[2]=1024; nw=3; onum=3;
    } else {
        wn[0]="gate_w"; wn[1]="up_w"; wn[2]=NULL; onm[0]="gate"; onm[1]="up"; onm[2]=NULL;
        nn[0]=3072; nn[1]=3072; nw=2; onum=2;
    }
    axclrtMalloc(&dh, K * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(dh, h, K * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, "h"), dh, K * 4);
    float * ws[3]; void * dws[3];
    for (int j = 0; j < nw; j++) {
        ws[j] = malloc((size_t) K * nn[j] * 4);
        for (size_t i = 0; i < (size_t) K * nn[j]; i++) ws[j][i] = (rand() % 200 - 100) * 0.0002f;
        axclrtMalloc(&dws[j], (size_t) K * nn[j] * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMemcpy(dws[j], ws[j], (size_t) K * nn[j] * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, axclrtEngineGetInputIndexByName(info, wn[j]), dws[j], (size_t) K * nn[j] * 4);
    }
    for (int j = 0; j < onum; j++) {
        axclrtMalloc(&douts[j], (size_t) nn[j] * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtEngineSetOutputBufferByIndex(io, axclrtEngineGetOutputIndexByName(info, onm[j]), douts[j], (size_t) nn[j] * 4);
    }
    double t0 = now_us();
    if (axclrtEngineExecute(model, ectx, 0, io)) { printf("exec FAIL\n"); return 1; }
    double t1 = now_us();
    for (int j = 0; j < onum; j++) {
        float * yref = malloc((size_t) nn[j] * 4), * y = malloc((size_t) nn[j] * 4);
        ref_mm(ws[j], h, yref, K, nn[j]);
        axclrtMemcpy(y, douts[j], (size_t) nn[j] * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
        double mx = 0, sum = 0;
        for (int i = 0; i < nn[j]; i++) {
            double d = fabs((double) y[i] - yref[i]);
            if (d > mx) mx = d; sum += d;
        }
        printf("[%s %s] out=%s exec=%.0fus max_err=%.5f mean_err=%.6f\n",
               engine, mode, onm[j], t1 - t0, mx, sum / nn[j]);
    }
    return 0;
}
