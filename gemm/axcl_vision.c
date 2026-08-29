// axcl_vision.c — drive the vendor's compiled Qwen3.5 vision tower
// (qwen3_5_vision.axmodel) on the AX8850 NPU.
//
// IO (probed on-card): input  hidden_states [1,576,512,3] u8 = one image,
//   384x384, patchified Qwen3-VL style (24x24 patches of 16x16 px x 2
//   temporal frames x 3 channels, vendor repack order); output
//   pooler_output [144,1024] = 144 merged vision-token embeddings.
//
// Packing recipe ported verbatim from ax-llm's Qwen2VideoProcessor (the
// reference reshape+transpose loop): resize to 384x384 RGB u8, duplicate
// the frame (temporal_patch_size=2), then gather in the order
// d2,d5 (2x2 merge blocks), d3,d6 (in-merge patch), d1 (temporal),
// d4,d7 (in-patch pixel), d8 (channel).
//
// usage: ./axcl_vision <image> <engine.axmodel> [out.bin]
//   writes 144*1024 floats (pooler_output) to out.bin (default /tmp/vis_emb.bin)
// build: gcc axcl_vision.c -I/usr/include/axcl -I<stb-dir> -lstb? no —
//   stb is header-only: see STB_IMPLEMENT below; link -l:libaxcl_rt.so etc.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// bilinear RGB u8 resize (no stb_resize dependency)
static void bilinear_resize(const unsigned char * src, int sw, int sh,
                             unsigned char * dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        const float fy = (y + 0.5f) * sh / dh - 0.5f;
        const int y0 = (int) fy; const float ty = fy - y0;
        const int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
        const int yy0 = y0 > 0 ? y0 : 0;
        for (int x = 0; x < dw; x++) {
            const float fx = (x + 0.5f) * sw / dw - 0.5f;
            const int x0 = (int) fx; const float tx = fx - x0;
            const int x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
            const int xx0 = x0 > 0 ? x0 : 0;
            for (int c = 0; c < 3; c++) {
                const float a = src[(size_t)(yy0 * sw + xx0) * 3 + c];
                const float b = src[(size_t)(yy0 * sw + x1) * 3 + c];
                const float d = src[(size_t)(y1 * sw + xx0) * 3 + c];
                const float e = src[(size_t)(y1 * sw + x1) * 3 + c];
                float v = a * (1 - tx) * (1 - ty) + b * tx * (1 - ty)
                        + d * (1 - tx) * ty + e * tx * ty;
                dst[(size_t)(y * dw + x) * 3 + c] = (unsigned char) (v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
            }
        }
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "axcl.h"
#include "axcl_rt_engine.h"
#include "axcl_rt_context.h"
#include "axcl_rt_device.h"

#define TGT 384           // engine's fixed resolution
#define PS  16            // patch size
#define TP  2             // temporal patch (frames)
#define MS  2             // spatial merge
#define GH  (TGT / PS)    // 24
#define GW  (TGT / PS)    // 24
#define NOUT_TOKENS ((GH / MS) * (GW / MS))  // 144
#define EMB 1024

// vendor repack: patches[t][y][x][c] -> out[], order d2 d5 d3 d6 d1 d4 d7 d8
static void repack(const unsigned char * frames /* TP * TGT * TGT * 3 */,
                   unsigned char * out /* 576*512*3 */) {
    size_t w = 0;
    for (int d2 = 0; d2 < GH / MS; d2++)
    for (int d5 = 0; d5 < GW / MS; d5++)
    for (int d3 = 0; d3 < MS; d3++)
    for (int d6 = 0; d6 < MS; d6++)
    for (int d1 = 0; d1 < TP; d1++)
    for (int d4 = 0; d4 < PS; d4++)
    for (int d7 = 0; d7 < PS; d7++)
    for (int d8 = 0; d8 < 3; d8++) {
        size_t idx = (size_t)d1 * GH * PS * GW * PS * 3
                   + (size_t)(d2 * MS + d3) * PS * GW * PS * 3
                   + (size_t)d4 * GW * PS * 3
                   + (size_t)(d5 * MS + d6) * PS * 3
                   + (size_t)d7 * 3
                   + (size_t)d8;
        out[w++] = frames[idx];
    }
    (void) 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <image> <engine.axmodel> [out.bin]\n", argv[0]); return 1; }
    setvbuf(stdout, NULL, _IONBF, 0);

    // ---- load + resize to 384x384 RGB
    int iw, ih, ic;
    unsigned char * img = stbi_load(argv[1], &iw, &ih, &ic, 3);
    if (!img) { fprintf(stderr, "stbi_load failed: %s\n", argv[1]); return 1; }
    unsigned char * res = malloc((size_t) TGT * TGT * 3);
    bilinear_resize(img, iw, ih, res, TGT, TGT);
    stbi_image_free(img);
    printf("image %dx%d -> %dx%d RGB\n", iw, ih, TGT, TGT);

    // ---- frame-duplicate (temporal=2) + repack to engine layout
    static unsigned char frames[TP * TGT * TGT * 3];
    memcpy(frames, res, (size_t) TGT * TGT * 3);
    memcpy(frames + (size_t) TGT * TGT * 3, res, (size_t) TGT * TGT * 3);
    free(res);
    static unsigned char packed[GH * GW * TP * PS * PS * 3]; // 884736
    repack(frames, packed);
    printf("packed %zu bytes (576 patches x 512 x 3)\n", sizeof(packed));

    // ---- card init + engine load
    if (axclInit(NULL) != AXCL_SUCC) { fprintf(stderr, "axclInit failed\n"); return 1; }
    axclrtDeviceList dl; memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC || dl.num == 0) { fprintf(stderr, "no devices\n"); return 1; }
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx = 0;
    axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    if (axclrtEngineInit(AXCL_VNPU_DISABLE) != AXCL_SUCC) { fprintf(stderr, "engine init failed\n"); return 1; }
    uint64_t model = 0;
    if (axclrtEngineLoadFromFile(argv[2], &model) != AXCL_SUCC) { fprintf(stderr, "load failed\n"); return 1; }
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0; uint64_t ectx = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    axclrtEngineCreateContext(model, &ectx);
    const int ix = axclrtEngineGetInputIndexByName(info, "hidden_states");
    const int iyo = axclrtEngineGetOutputIndexByName(info, "pooler_output");
    printf("io: in=%d out=%d\n", ix, iyo);
    if (ix < 0 || iyo < 0) { fprintf(stderr, "io names missing\n"); return 1; }
    const uint64_t in_sz = axclrtEngineGetInputSizeByIndex(info, 0, (uint32_t) ix);
    const uint64_t out_sz = axclrtEngineGetOutputSizeByIndex(info, 0, (uint32_t) iyo);
    printf("sizes: in=%llu out=%llu (expect 884736 / 589824)\n",
           (unsigned long long) in_sz, (unsigned long long) out_sz);

    void * din = NULL, * dout = NULL;
    axclrtMalloc(&din, in_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dout, out_sz, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(din, packed, in_sz, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtEngineSetInputBufferByIndex(io, ix, din, in_sz);
    axclrtEngineSetOutputBufferByIndex(io, iyo, dout, out_sz);

    printf("executing vision tower...\n");
    uint64_t t0 = 0; struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec*1000000ull + ts.tv_nsec/1000;
    if (axclrtEngineExecute(model, ectx, 0, io) != AXCL_SUCC) { fprintf(stderr, "execute failed\n"); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("execute ok in %lld ms\n", (long long)((ts.tv_sec*1000000ull + ts.tv_nsec/1000) - t0) / 1000);

    static float out[NOUT_TOKENS * EMB];
    axclrtMemcpy(out, dout, out_sz < sizeof(out) ? out_sz : sizeof(out), AXCL_MEMCPY_DEVICE_TO_HOST);
    double cs = 0; float mx = -1e30f, mn = 1e30f;
    for (int i = 0; i < NOUT_TOKENS * EMB; i++) { cs += out[i]; if (out[i] > mx) mx = out[i]; if (out[i] < mn) mn = out[i]; }
    printf("embeddings: %d x %d, checksum=%.3f, range=[%.4f, %.4f], first=%.6f\n",
           NOUT_TOKENS, EMB, cs, mn, mx, out[0]);

    const char * outp = argc > 3 ? argv[3] : "/tmp/vis_emb.bin";
    FILE * f = fopen(outp, "wb");
    if (f) { fwrite(out, 4, NOUT_TOKENS * EMB, f); fclose(f); printf("wrote %s\n", outp); }

    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
