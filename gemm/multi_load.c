// multi_load.c — which engine-load combination fails? Loads each file in
// order, prints rc + a marker; no executes. Usage: ./multi_load f1 f2 ...
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

int main(int argc, char ** argv) {
    if (axclInit(NULL) != AXCL_SUCC) { printf("axclInit FAIL\n"); return 1; }
    axclrtDeviceList dl; memset(&dl, 0, sizeof(dl));
    axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx = 0; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    if (axclrtEngineInit(AXCL_VNPU_DISABLE) != AXCL_SUCC) { printf("engineInit FAIL\n"); return 1; }
    uint64_t models[64] = {0};
    for (int i = 1; i < argc && i < 64; i++) {
        printf("loading [%d] %s ... ", i - 1, argv[i]); fflush(stdout);
        if (axclrtEngineLoadFromFile(argv[i], &models[i-1]) == AXCL_SUCC) printf("OK\n");
        else { printf("FAIL\n"); return 2; }
        fflush(stdout);
    }
    printf("ALL %d LOADED OK\n", argc - 1);
    for (int i = 1; i < argc && i < 64; i++) if (models[i-1]) axclrtEngineUnload(models[i-1]);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
