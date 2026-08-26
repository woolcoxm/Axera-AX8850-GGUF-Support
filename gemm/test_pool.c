// rehearsal for the weight-pool design: ONE big CMM malloc, carve sub-buffers,
// upload weights into them, interleave engine executes - the exact traffic
// pattern the backend will produce
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_memory.h>

int main() {
    setbuf(stdout, NULL);
    printf("init=%d\n", (int) axclInit(NULL));
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    printf("engineInit=%d\n", (int) axclrtEngineInit(AXCL_VNPU_DISABLE));

    // load the 1024x1024 engine for interleaved executes
    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile("/usr/local/share/ggml-axcl/matmul/matmul_m1_k1024_n1024.axmodel", &model)) {
        printf("no test engine\n"); _exit(1);
    }
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);
    int xi = axclrtEngineGetInputIndexByName(info, "X");
    int wi = axclrtEngineGetInputIndexByName(info, "W");
    int yi = axclrtEngineGetOutputIndexByName(info, "Y");
    void *dx, *dy; axclrtMalloc(&dx, 4096, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dy, 4096, AXCL_MEM_MALLOC_HUGE_FIRST);

    // THE design under test: one big pool allocation
    const size_t POOL = (size_t) 2560 * 1024 * 1024; // 2.5 GB
    void * pool = NULL;
    axclError e = axclrtMalloc(&pool, POOL, AXCL_MEM_MALLOC_HUGE_FIRST);
    printf("pool malloc(2.5GB) = %d ptr=%p\n", (int) e, pool);
    if (e || !pool) { printf("POOL FAILED\n"); _exit(1); }

    // carve 24 weight slots of 12MB (aligned 4K), upload, execute between carves
    static float wsrc[1024 * 1024 * 3]; // 12MB source
    memset(wsrc, 0, sizeof wsrc);
    char * bump = (char *) pool;
    for (int i = 0; i < 24; i++) {
        void * slot = bump; bump += ((size_t) 12 * 1024 * 1024 + 4095) & ~4095;
        e = axclrtMemcpy(slot, wsrc, (size_t) 12 * 1024 * 1024, AXCL_MEMCPY_HOST_TO_DEVICE);
        if (e) { printf("carve %d H2D fail %d\n", i, (int) e); _exit(1); }

        // interleave: run the engine against this slot as W (zero weights -> y=0)
        float x = 1.0f;
        axclrtMemcpy(dx, &x, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, xi, dx, 4096);
        axclrtEngineSetInputBufferByIndex(io, wi, slot, (size_t) 1024 * 1024 * 4);
        axclrtEngineSetOutputBufferByIndex(io, yi, dy, 4096);
        e = axclrtEngineExecute(model, ectx, 0, io);
        if (e) { printf("carve %d EXEC fail %d\n", i, (int) e); _exit(1); }
        float y = -1; axclrtMemcpy(&y, dy, 4, AXCL_MEMCPY_DEVICE_TO_HOST);
        if (y != 0.0f) { printf("carve %d BAD RESULT y=%f\n", i, y); _exit(1); }
        if (i % 6 == 5) printf("carve %d ok (execute interleaved, y=0)\n", i);
    }
    printf("POOL DESIGN PASSES\n");
    fflush(stdout);
    _exit(0);
}
