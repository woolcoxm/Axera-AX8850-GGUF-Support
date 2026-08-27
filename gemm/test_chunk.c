// test_chunk.c — validate the 128-token chunk shape group against per-token
// decode on a vendor whole-layer engine.
// Reference: group 0 run 128 times (proper causal masks, accumulating cache).
// Candidate: group 1 run once (128 rows, ladder conventions).
// Usage: ./test_chunk [engine] [mask_variant: incl|strict] [bind_io: shared|dedic]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "axcl.h"
#include "axcl_rt.h"

static unsigned short f2bf(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    unsigned int r = u & 0xFFFF, b = u >> 16;
    unsigned int ru = (r > 0x8000) | ((r == 0x8000) & ((b & 1) == 1));
    return (unsigned short) (b + ru);
}
static float bf2f(unsigned short h) {
    unsigned int u = (unsigned int) h << 16; float f; memcpy(&f, &u, 4); return f;
}

int main(int argc, char ** argv) {
    setbuf(stdout, NULL);
    const char * path = argc > 1 ? argv[1] : "/home/kram/Qwen3-0.6B/qwen3_p128_l0_together.axmodel";
    const char * mvar = argc > 2 ? argv[2] : "incl";     // incl | strict
    const int dedic = (argc > 3 && strcmp(argv[3], "dedic") == 0); // dedicated chunk IO
    const int NT = 128;

    axclInit(NULL);
    axclrtDeviceList dl; axclrtGetDeviceList(&dl);
    axclrtSetDevice(dl.devices[0]);
    axclrtContext ctx; axclrtCreateContext(&ctx, dl.devices[0]);
    axclrtSetCurrentContext(ctx);
    axclrtEngineInit(AXCL_VNPU_DISABLE);

    uint64_t model = 0, ectx = 0;
    if (axclrtEngineLoadFromFile(path, &model)) { printf("LOAD FAIL\n"); return 1; }
    axclrtEngineIOInfo info = 0; axclrtEngineIO io = 0, io2 = 0;
    axclrtEngineGetIOInfo(model, &info);
    axclrtEngineCreateIO(info, &io);
    if (dedic) axclrtEngineCreateIO(info, &io2); else io2 = io;
    axclrtEngineCreateContext(model, &ectx);
    int ik = axclrtEngineGetInputIndexByName(info, "K_cache");
    int iv = axclrtEngineGetInputIndexByName(info, "V_cache");
    int ii = axclrtEngineGetInputIndexByName(info, "indices");
    int ix = axclrtEngineGetInputIndexByName(info, "input");
    int im = axclrtEngineGetInputIndexByName(info, "mask");
    int iko = axclrtEngineGetOutputIndexByName(info, "K_cache_out");
    int ivo = axclrtEngineGetOutputIndexByName(info, "V_cache_out");
    int iyo = axclrtEngineGetOutputIndexByName(info, "output");

    const size_t kv_bytes = (size_t) 2048 * 1024 * 2;
    void *dk, *dv, *di, *dx, *dm, *dko, *dvo, *dy;
    axclrtMalloc(&dk, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dv, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&di, 2048 * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dx, 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dm, (size_t) 2049 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dko, (size_t) 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvo, (size_t) 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dy, 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    // zero caches
    { char * z = calloc(1, 1 << 20);
      for (size_t o = 0; o < kv_bytes; o += (1 << 20)) axclrtMemcpy((char*)dk + o, z, 1 << 20, AXCL_MEMCPY_HOST_TO_DEVICE);
      for (size_t o = 0; o < kv_bytes; o += (1 << 20)) axclrtMemcpy((char*)dv + o, z, 1 << 20, AXCL_MEMCPY_HOST_TO_DEVICE);
      free(z); }

    // deterministic hiddens: x[t][j] = sin(t*0.37 + j*0.11) — or shaped modes
    const int diag = (argc > 4 && strcmp(argv[4], "diag") == 0);
    const int onehot = (argc > 4 && strcmp(argv[4], "onehot") == 0);
    float (*X)[1024] = malloc(sizeof(float) * NT * 1024);
    unsigned short (*Xb)[1024] = malloc(sizeof(unsigned short) * NT * 1024);
    memset(X, 0, sizeof(float) * NT * 1024);
    for (int t = 0; t < NT; t++)
        for (int j = 0; j < 1024; j++) {
            if (onehot) X[t][j] = (t == 0) ? (0.5f * sinf(j * 0.11f)) : 0.0f;
            else if (diag) X[t][j] = 0.5f * sinf(j * 0.11f);
            else X[t][j] = 0.5f * sinf(t * 0.37f + j * 0.11f);
            Xb[t][j] = f2bf(X[t][j]);
        }

    // ---- reference: group-0 per-token ----
    unsigned short * mask_d = malloc((size_t) 2049 * 2); // decode mask row
    float * yref = malloc(sizeof(float) * NT * 1024);
    unsigned short * kref = malloc((size_t) NT * 2048);
    for (int p = 0; p < NT; p++) {
        unsigned int idx = (unsigned int) p;
        axclrtMemcpy(di, &idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(dx, Xb[p], 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
        for (int t = 0; t < 2049; t++) {
            float v = (t < p || t == 2048) ? 0.0f : -1e9f;
            mask_d[t] = f2bf(v);
        }
        axclrtMemcpy(dm, mask_d, (size_t) 2049 * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtEngineSetInputBufferByIndex(io, ik, dk, kv_bytes);
        axclrtEngineSetInputBufferByIndex(io, iv, dv, kv_bytes);
        axclrtEngineSetInputBufferByIndex(io, ii, di, 4);
        axclrtEngineSetInputBufferByIndex(io, ix, dx, 2048);
        axclrtEngineSetInputBufferByIndex(io, im, dm, (size_t) 2049 * 2);
        axclrtEngineSetOutputBufferByIndex(io, iko, dko, 1024 * 2);
        axclrtEngineSetOutputBufferByIndex(io, ivo, dvo, 1024 * 2);
        axclrtEngineSetOutputBufferByIndex(io, iyo, dy, 2048);
        if (axclrtEngineExecute(model, ectx, 0, io)) { printf("ref exec fail p=%d\n", p); return 1; }
        // scatter k/v back into the cache (like the backend does)
        axclrtMemcpy((char *) dk + (size_t) p * 2048, dko, 2048, AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy((char *) dv + (size_t) p * 2048, dvo, 2048, AXCL_MEMCPY_DEVICE_TO_DEVICE);
        unsigned short yb[1024];
        axclrtMemcpy(yb, dy, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
        for (int j = 0; j < 1024; j++) yref[(size_t) p * 1024 + j] = bf2f(yb[j]);
        if (p < NT) axclrtMemcpy(kref + (size_t) p * 1024, dko, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    }
    printf("reference done (%s mask)\n", mvar);

    // ---- candidate: group-1 single chunk ----
    void *dxm, *dim, *dmm, *dkom, *dvom, *dym;
    axclrtMalloc(&dxm, (size_t) NT * 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dim, (size_t) NT * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dmm, (size_t) NT * NT * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dkom, (size_t) NT * 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dvom, (size_t) NT * 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&dym, (size_t) NT * 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMemcpy(dxm, Xb, (size_t) NT * 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
    if (argc > 4 && strcmp(argv[4], "xpose") == 0) {
        // feature-major staging: xT[j][t] = x[t][j]
        unsigned short * xt = malloc((size_t) NT * 2048);
        for (int t = 0; t < NT; t++)
            for (int j = 0; j < 1024; j++) xt[(size_t) j * NT + t] = Xb[t][j];
        axclrtMemcpy(dxm, xt, (size_t) NT * 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
        free(xt);
    }
    unsigned int * idxm = malloc((size_t) NT * 4);
    const int diag0 = (argc > 4 && strcmp(argv[4], "diag0") == 0);
    for (int i = 0; i < NT; i++) idxm[i] = (diag0 || onehot) ? 0u : (unsigned int) i;
    axclrtMemcpy(dim, idxm, (size_t) NT * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    // chunk mask 128x128: row i allows q<=i (incl) or q<i (strict)
    unsigned short * cm = malloc((size_t) NT * NT * 2);
    for (int i = 0; i < NT; i++)
        for (int q = 0; q < NT; q++) {
            int allow = (strcmp(mvar, "strict") == 0) ? (q < i) : (q <= i);
            cm[(size_t) i * NT + q] = f2bf(allow ? 0.0f : -1e9f);
        }
    axclrtMemcpy(dmm, cm, (size_t) NT * NT * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
    // K/V binding for group 1: dims [1,1,1024] -> one zero row
    axclrtEngineSetInputBufferByIndex(io2, ik, dk, 1024 * 2);
    axclrtEngineSetInputBufferByIndex(io2, iv, dv, 1024 * 2);
    axclrtEngineSetInputBufferByIndex(io2, ii, dim, (size_t) NT * 4);
    axclrtEngineSetInputBufferByIndex(io2, ix, dxm, (size_t) NT * 2048);
    axclrtEngineSetInputBufferByIndex(io2, im, dmm, (size_t) NT * NT * 2);
    axclrtEngineSetOutputBufferByIndex(io2, iko, dkom, (size_t) NT * 2048);
    axclrtEngineSetOutputBufferByIndex(io2, ivo, dvom, (size_t) NT * 2048);
    axclrtEngineSetOutputBufferByIndex(io2, iyo, dym, (size_t) NT * 2048);
    if (axclrtEngineExecute(model, ectx, 1, io2)) { printf("chunk exec fail\n"); return 1; }
    unsigned short * ych = malloc((size_t) NT * 2048);
    unsigned short * kch = malloc((size_t) NT * 2048);
    axclrtMemcpy(ych, dym, (size_t) NT * 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
    axclrtMemcpy(kch, dkom, (size_t) NT * 2048, AXCL_MEMCPY_DEVICE_TO_HOST);

    // ---- compare ----
    if (argc > 4 && strncmp(argv[4], "dump", 4) == 0) {
        FILE * f = fopen("/tmp/chunk_dump.bin", "wb");
        fwrite(kref, 2048, NT, f);     // reference K rows (per-token)
        fwrite(kch, 2048, NT, f);      // chunk K rows
        fwrite(yref, 4 * 1024, NT, f); // reference y (f32)
        fwrite(ych, 2048, NT, f);      // chunk y (bf16)
        fwrite(Xb, 2048, NT, f);       // inputs (bf16)
        fclose(f);
        printf("dumped /tmp/chunk_dump.bin\n");
    }
    if (argc > 4 && strcmp(argv[4], "diag") == 0) {
        // all-equal inputs: kout rows must be identical; find which ref row
        // each chunk row matches
        double maxrow = 0;
        for (int p = 1; p < NT; p++)
            for (int j = 0; j < 1024; j++) {
                double d = fabs(bf2f(kch[(size_t) p * 1024 + j]) - bf2f(kch[j]));
                if (d > maxrow) maxrow = d;
            }
        printf("diag: kout row self-consistency maxdiff=%g\n", maxrow);
        // match chunk kout row 0 against every reference k row
        int best = -1; double bestd = 1e30;
        for (int p = 0; p < NT; p++) {
            double s = 0;
            for (int j = 0; j < 1024; j++)
                s += fabs(bf2f(kch[j]) - bf2f(kref[(size_t) p * 1024 + j]));
            if (s < bestd) { bestd = s; best = p; }
        }
        printf("diag: chunk kout[0] best-matches ref k row %d (L1=%g)\n", best, bestd);
        // and the reverse: ref k[0] vs all chunk rows
        best = -1; bestd = 1e30;
        for (int p = 0; p < NT; p++) {
            double s = 0;
            for (int j = 0; j < 1024; j++)
                s += fabs(bf2f(kref[j]) - bf2f(kch[(size_t) p * 1024 + j]));
            if (s < bestd) { bestd = s; best = p; }
        }
        printf("diag: ref k[0] best-matches chunk kout row %d (L1=%g)\n", best, bestd);
    }
    double maxdy = 0; int worst = -1;
    for (int p = 0; p < NT; p++)
        for (int j = 0; j < 1024; j++) {
            double d = fabs(bf2f(ych[(size_t) p * 1024 + j]) - yref[(size_t) p * 1024 + j]);
            if (d > maxdy) { maxdy = d; worst = p; }
        }
    double maxdk = 0;
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 1024; j++) {
            double d = fabs(bf2f(kch[(size_t) p * 1024 + j]) - bf2f(kref[(size_t) p * 1024 + j]));
            if (d > maxdk) maxdk = d;
        }
    printf("mask=%s io=%s: max|dy|=%g (worst p=%d)  max|dk|(first4)=%g\n",
           mvar, dedic ? "dedic" : "shared", maxdy, worst, maxdk);
    printf("sample y[0][0..3]: ref %g %g %g %g  chunk %g %g %g %g\n",
           yref[0], yref[1], yref[2], yref[3],
           bf2f(ych[0]), bf2f(ych[1]), bf2f(ych[2]), bf2f(ych[3]));
    axclrtSetCurrentContext(ctx);
    axclrtEngineUnload(model);
    axclrtEngineFinalize();
    axclFinalize();
    return 0;
}
