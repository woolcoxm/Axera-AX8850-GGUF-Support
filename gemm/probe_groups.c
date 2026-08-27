// probe_groups.c — dump shape groups of a whole-layer axmodel.
// build: gcc probe_groups.c -laxclrt -laxcl -o probe_groups
// usage: ./probe_groups <engine.axmodel>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <axmodel>\n", argv[0]); return 1; }
    if (axclInit(NULL) != AXCL_SUCC) { fprintf(stderr, "axclInit failed\n"); return 1; }
    // activation sequence per ggml-axcl: device list probe activates the card
    axclrtDeviceList dl;
    memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC || dl.num == 0) { fprintf(stderr, "no devices\n"); return 1; }
    int32_t dev = dl.devices[0];
    axclrtSetDevice(dev);
    axclrtContext ctx = 0;
    if (axclrtCreateContext(&ctx, dev) != AXCL_SUCC) { fprintf(stderr, "ctx failed\n"); return 1; }
    axclrtSetCurrentContext(ctx);
    if (axclrtEngineInit(AXCL_VNPU_DISABLE) != AXCL_SUCC) { fprintf(stderr, "engine init failed\n"); return 1; }
    uint64_t model = 0;
    if (axclrtEngineLoadFromFile(argv[1], &model) != AXCL_SUCC) { fprintf(stderr, "load failed\n"); return 1; }
    axclrtEngineIOInfo info = NULL;
    if (axclrtEngineGetIOInfo(model, &info) != AXCL_SUCC) { fprintf(stderr, "ioinfo failed\n"); return 1; }
    int32_t ngroups = 0;
    axclrtEngineGetShapeGroupsCount(info, &ngroups);
    printf("shape groups: %d\n", ngroups);
    uint32_t n_in = axclrtEngineGetNumInputs(info);
    uint32_t n_out = axclrtEngineGetNumOutputs(info);
    printf("num inputs: %u  num outputs: %u\n", n_in, n_out);
    for (int32_t g = 0; g < ngroups && g < 2; g++) {
        printf("group %d:\n", g);
        for (uint32_t i = 0; i < n_in; i++) {
            const char * nm = axclrtEngineGetInputNameByIndex(info, i);
            uint64_t sz = axclrtEngineGetInputSizeByIndex(info, g, i);
            axclrtEngineIODims d;
            memset(&d, 0, sizeof(d));
            axclrtEngineGetInputDims(info, g, i, &d);
            uint64_t prod = 1;
            for (uint32_t j = 0; j < d.dimCount; j++) prod *= d.dims[j];
            printf("  in  [%u] %-12s sizeAPI=%llu prod=%llu (%llux2B) dims=[", i, nm ? nm : "?",
                   (unsigned long long) sz, (unsigned long long) prod * 2, (unsigned long long) prod);
            for (uint32_t j = 0; j < d.dimCount; j++) printf("%u%s", d.dims[j], j+1<d.dimCount?",":"");
            printf("]\n");
        }
        for (uint32_t i = 0; i < n_out; i++) {
            const char * nm = axclrtEngineGetOutputNameByIndex(info, i);
            uint64_t sz = axclrtEngineGetOutputSizeByIndex(info, g, i);
            axclrtEngineIODims d;
            memset(&d, 0, sizeof(d));
            axclrtEngineGetOutputDims(info, g, i, &d);
            uint64_t prod = 1;
            for (uint32_t j = 0; j < d.dimCount; j++) prod *= d.dims[j];
            printf("  out [%u] %-12s sizeAPI=%llu prod=%llu dims=[", i, nm ? nm : "?",
                   (unsigned long long) sz, (unsigned long long) prod * 2);
            for (uint32_t j = 0; j < d.dimCount; j++) printf("%u%s", d.dims[j], j+1<d.dimCount?",":"");
            printf("]\n");
        }
    }
    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
