// chain_test.c: run the full 28-engine whole-layer chain on a fixed 6-token
// input, dumping every (pos, layer) in/out. Buffer discipline mirrors the
// backend: per-pos persistent hidden slots refreshed after each layer,
// alternating yout buffer per call.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt.h"

#define CTX 2048
#define HID 1024
#define NL 28
#define NT 6

struct eng {
    uint64_t model, ectx;
    axclrtEngineIOInfo info;
    axclrtEngineIO io;
    int ik, iv, ii, ix, im, iko, ivo, iyo;
    void *dk, *dv;
};

static unsigned seed = 4242;
static float frand(void) { seed = seed * 1103515245 + 12345; return ((seed >> 8) & 0xFFFF) / 65535.0f * 2 - 1; }
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u >> 16) + ((u >> 16) & 1)); }
static float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext c; axclrtCreateContext(&c, dl.devices[0]); axclrtSetCurrentContext(c);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    axclSetLogLevel(3);

    struct eng E[NL];
    const size_t kvb = (size_t)CTX * HID * 2;
    for (int l = 0; l < NL; l++) {
        char p[256];
        snprintf(p, sizeof(p), "/usr/local/share/ggml-axcl/layer/qwen3_p128_l%d_together.axmodel", l);
        struct eng * e = &E[l];
        if (axclrtEngineLoadFromFile(p, &e->model)) { printf("load fail %s\n", p); return 1; }
        axclrtEngineGetIOInfo(e->model, &e->info);
        axclrtEngineCreateIO(e->info, &e->io);
        axclrtEngineCreateContext(e->model, &e->ectx);
        e->ik = axclrtEngineGetInputIndexByName(e->info, "K_cache");
        e->iv = axclrtEngineGetInputIndexByName(e->info, "V_cache");
        e->ii = axclrtEngineGetInputIndexByName(e->info, "indices");
        e->ix = axclrtEngineGetInputIndexByName(e->info, "input");
        e->im = axclrtEngineGetInputIndexByName(e->info, "mask");
        e->iko = axclrtEngineGetOutputIndexByName(e->info, "K_cache_out");
        e->ivo = axclrtEngineGetOutputIndexByName(e->info, "V_cache_out");
        e->iyo = axclrtEngineGetOutputIndexByName(e->info, "output");
        axclrtMalloc(&e->dk, kvb, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&e->dv, kvb, AXCL_MEM_MALLOC_HUGE_FIRST);
        // zero the caches (NaN-garbage poisons masked softmax)
        {
            static char z[1 << 20];
            memset(z, 0, sizeof(z));
            for (size_t off = 0; off < kvb; off += sizeof(z)) {
                size_t n = kvb - off < sizeof(z) ? kvb - off : sizeof(z);
                axclrtMemcpy((char *)e->dk + off, z, n, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtMemcpy((char *)e->dv + off, z, n, AXCL_MEMCPY_HOST_TO_DEVICE);
            }
        }
    }

    // shared IO
    void *d_idx, *d_mask, *d_kout, *d_vout, *d_ya, *d_yb;
    void *dh[NT];
    axclrtMalloc(&d_idx, 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    const size_t rowb = ((size_t)(CTX + 1) * 2 + 7) & ~(size_t)7;
    axclrtMalloc(&d_mask, (size_t)CTX * rowb, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&d_kout, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&d_vout, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&d_ya, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&d_yb, HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    for (int t = 0; t < NT; t++) axclrtMalloc(&dh[t], HID * 2, AXCL_MEM_MALLOC_HUGE_FIRST);

    // mask rows: allow t < p (cache) and t == CTX (engine self slot)
    {
        char * m = malloc((size_t)CTX * rowb);
        memset(m, 0, (size_t)CTX * rowb);
        for (int p = 0; p < CTX; p++) {
            for (int t = 0; t <= CTX; t++) {
                float v = ((t < p) || (t == CTX)) ? 0.0f : -1e9f;
                uint16_t b = f2b(v);
                memcpy(m + (size_t)p * rowb + (size_t)t * 2, &b, 2);
            }
        }
        axclrtMemcpy(d_mask, m, (size_t)CTX * rowb, AXCL_MEMCPY_HOST_TO_DEVICE);
        free(m);
    }
    void * d_mask_row;
    axclrtMalloc(&d_mask_row, rowb, AXCL_MEM_MALLOC_HUGE_FIRST);

    // load input embeddings
    uint16_t * in = malloc(NT * HID * 2);
    FILE * f = fopen("/tmp/chain_in.bin", "rb");
    if (!f || fread(in, 2, NT * HID, f) != NT * HID) { printf("input read fail\n"); return 1; }
    fclose(f);
    for (int t = 0; t < NT; t++) axclrtMemcpy(dh[t], in + (size_t)t * HID, HID * 2, AXCL_MEMCPY_HOST_TO_DEVICE);

    FILE * log = fopen("/tmp/chain_engine.log", "w");
    uint16_t hout[HID], krow[HID];

    for (int pass = 0; pass < NT; pass++) {
        const int pos = pass;
        // NOTE: within one pass, tokens are processed sequentially at
        // increasing positions; here each pass IS one token.
        for (int l = 0; l < NL; l++) {
            struct eng * e = &E[l];
            uint32_t idx = (uint32_t)pos;
            axclrtMemcpy(d_idx, &idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy(d_mask_row, (char *)d_mask + (size_t)pos * rowb, (size_t)(CTX + 1) * 2,
                         AXCL_MEMCPY_DEVICE_TO_DEVICE);
            void * ybuf = (l % 2) ? d_yb : d_ya;
            axclrtEngineSetInputBufferByIndex(e->io, e->ik, e->dk, kvb);
            axclrtEngineSetInputBufferByIndex(e->io, e->iv, e->dv, kvb);
            axclrtEngineSetInputBufferByIndex(e->io, e->ii, d_idx, 4);
            axclrtEngineSetInputBufferByIndex(e->io, e->ix, dh[pass], HID * 2);
            axclrtEngineSetInputBufferByIndex(e->io, e->im, d_mask_row, (size_t)(CTX + 1) * 2);
            axclrtEngineSetOutputBufferByIndex(e->io, e->iko, d_kout, HID * 2);
            axclrtEngineSetOutputBufferByIndex(e->io, e->ivo, d_vout, HID * 2);
            axclrtEngineSetOutputBufferByIndex(e->io, e->iyo, ybuf, HID * 2);
            if (axclrtEngineExecute(e->model, e->ectx, 0, e->io)) { printf("exec fail l=%d\n", l); return 1; }
            // scatter the new K/V row into this layer's cache
            axclrtMemcpy((char *)e->dk + (size_t)pos * HID * 2, d_kout, HID * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
            axclrtMemcpy((char *)e->dv + (size_t)pos * HID * 2, d_vout, HID * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
            // stage the hidden into this pass's slot
            axclrtMemcpy(dh[pass], ybuf, HID * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
            // log
            axclrtMemcpy(hout, ybuf, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
            axclrtMemcpy(krow, d_kout, HID * 2, AXCL_MEMCPY_DEVICE_TO_HOST);
            double cs = 0;
            for (int i = 0; i < 8; i++) cs += b2f(hout[i]);
            fprintf(log, "pos=%d L%d cs8=%.6f out0-3=%g %g %g %g k0-1=%g %g\n",
                    pos, l, cs, b2f(hout[0]), b2f(hout[1]), b2f(hout[2]), b2f(hout[3]),
                    b2f(krow[0]), b2f(krow[1]));
        }
    }
    fclose(log);
    printf("chain done; log at /tmp/chain_engine.log\n");
    return 0;
}
