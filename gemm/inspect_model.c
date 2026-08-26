// dump the full IO structure of any axmodel: names, shapes, types, groups
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <axcl.h>
#include <axcl_rt_context.h>
#include <axcl_rt_engine.h>

static void dump(const char * path) {
    printf("=== %s ===\n", path);
    uint64_t model = 0;
    axclError e = axclrtEngineLoadFromFile(path, &model);
    if (e) { printf("  load failed: %d\n", (int) e); return; }
    axclrtEngineIOInfo info = 0;
    axclrtEngineGetIOInfo(model, &info);
    uint32_t nin = axclrtEngineGetNumInputs(info), nout = axclrtEngineGetNumOutputs(info);
    uint32_t ngroups = axclrtEngineGetShapeGroupsCount(info);
    printf("  inputs=%u outputs=%u shape_groups=%u\n", nin, nout, ngroups);
    for (uint32_t g = 0; g < (ngroups ? ngroups : 1); g++) {
        if (ngroups > 1) printf("  -- group %u --\n", g);
        for (uint32_t i = 0; i < nin; i++) {
            const char * n = axclrtEngineGetInputNameByIndex(info, i);
            axclrtEngineDataType t; axclrtEngineGetInputDataType(info, i, &t);
            axclrtEngineIODims d;
            if (axclrtEngineGetInputDims(info, g < ngroups ? g : 0, i, &d) == AXCL_SUCC) {
                printf("  in[%u] %-24s type=%d dims=[%u", i, n ? n : "?", (int) t, d.count ? d.dims[0] : 0);
                for (uint32_t j = 1; j < d.count && j < 6; j++) printf(",%u", d.dims[j]);
                printf("]\n");
            }
        }
        for (uint32_t i = 0; i < nout; i++) {
            const char * n = axclrtEngineGetOutputNameByIndex(info, i);
            axclrtEngineDataType t; axclrtEngineGetOutputDataType(info, i, &t);
            printf("  out[%u] %-24s type=%d\n", i, n ? n : "?", (int) t);
        }
    }
    axclrtEngineUnload(model);
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    for (int i = 1; i < argc; i++) dump(argv[i]);
    fflush(stdout);
    _exit(0);
}
