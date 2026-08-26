// interleaved multi-engine execute stress: mimics generation-phase
// layer cycling (different shapes back-to-back on the same card session)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>
#include <axcl_rt_memory.h>
#include <pthread.h>

typedef struct {
    uint64_t model, ectx;
    axclrtEngineIOInfo info;
    axclrtEngineIO io;
    void * dx, * dw, * dy;
    const char * path;
    int k, n;
    int xi, wi, yi;
} eng_t;

static int setup(eng_t * e, const char * path, int k, int n) {
    e->path = path; e->k = k; e->n = n;
    if (axclrtEngineLoadFromFile(path, &e->model)) { printf("load fail %s\n", path); return 1; }
    axclrtEngineGetIOInfo(e->model, &e->info);
    axclrtEngineCreateIO(e->info, &e->io);
    axclrtEngineCreateContext(e->model, &e->ectx);
    e->xi = axclrtEngineGetInputIndexByName(e->info, "X");
    e->wi = axclrtEngineGetInputIndexByName(e->info, "W");
    e->yi = axclrtEngineGetOutputIndexByName(e->info, "Y");
    axclrtMalloc(&e->dx, k * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dw, (size_t) k * n * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&e->dy, n * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    printf("loaded %s (idx %d,%d->%d)\n", path, e->xi, e->wi, e->yi);
    return 0;
}

static float spinner_a[512*512], spinner_b[512*512], spinner_c[512*512];
static volatile int spin = 1;
static void * cpu_hog(void * arg) {
    (void) arg;
    while (spin) {
        for (int i = 0; i < 512*512; i++) spinner_c[i] = spinner_a[i] * spinner_b[i] + 1.0f;
    }
    return NULL;
}

int main() {
    setbuf(stdout, NULL);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    const char * dir = "/usr/local/share/ggml-axcl/matmul";
    static char p0[256], p1[256], p2[256], p3[256];
    snprintf(p0, sizeof p0, "%s/matmul_m1_k1024_n1024.axmodel", dir);
    snprintf(p1, sizeof p1, "%s/matmul_m1_k1024_n3072.axmodel", dir);
    snprintf(p2, sizeof p2, "%s/matmul_m1_k3072_n1024.axmodel", dir);
    snprintf(p3, sizeof p3, "%s/matmul_m1_k1024_n2048.axmodel", dir);

    eng_t e[4]; int ne = 0;
    if (!setup(&e[ne], p0, 1024, 1024)) ne++;
    if (!setup(&e[ne], p1, 1024, 3072)) ne++;
    if (!setup(&e[ne], p2, 3072, 1024)) ne++;
    FILE * f = fopen(p3, "r");
    if (f) { fclose(f); if (!setup(&e[ne], p3, 1024, 2048)) ne++; }

    static float xbuf[4096], wbuf[3072 * 3072], ybuf[4096];
    for (int i = 0; i < 4096; i++) xbuf[i] = 0.01f * i;

    pthread_t th;
    pthread_create(&th, NULL, cpu_hog, NULL);
    printf("=== interleaved stress w/ cpu hog: %d engines x 50 rounds ===\n", ne);
    for (int r = 0; r < 50; r++) {
        for (int i = 0; i < ne; i++) {
            eng_t * g = &e[i];
            memset(wbuf, 0, (size_t) g->k * g->n * 4); // zero weights: y must be 0
            axclrtMemcpy(g->dx, xbuf, g->k * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(g->dw, wbuf, (size_t) g->k * g->n * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtEngineSetInputBufferByIndex(g->io, g->xi, g->dx, g->k * 4);
            axclrtEngineSetInputBufferByIndex(g->io, g->wi, g->dw, (size_t) g->k * g->n * 4);
            axclrtEngineSetOutputBufferByIndex(g->io, g->yi, g->dy, g->n * 4);
            axclError err = axclrtEngineExecute(g->model, g->ectx, 0, g->io);
            if (err) { printf("round %d eng %d EXECUTE FAIL %d\n", r, i, (int) err); fflush(stdout); _exit(1); }
            axclrtMemcpy(ybuf, g->dy, g->n * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
            if (ybuf[0] != 0.0f || ybuf[g->n - 1] != 0.0f) { printf("round %d eng %d BAD RESULT\n", r, i); }
        }
        if (r % 10 == 9) { printf("round %d ok\n", r); fflush(stdout); }
    }
    spin = 0; pthread_join(th, NULL);
    printf("ALL ROUNDS PASSED (with concurrent cpu load)\n");
    fflush(stdout);
    _exit(0);
}
