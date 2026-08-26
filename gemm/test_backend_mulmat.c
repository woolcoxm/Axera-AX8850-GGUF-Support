// ggml-level test: MUL_MAT through the AXCL backend vs the CPU backend.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include "ggml.h"
#include "ggml-allocr.h"
#include "ggml-backend.h"

int main() {
    setbuf(stdout, NULL);
    const int K = 2048, N = 2048;

    FILE *f = fopen("/home/kram/matmul/x.f32", "rb");
    static float xh[K]; fread(xh, 4, K, f); fclose(f);
    f = fopen("/home/kram/matmul/w.f32", "rb");
    static float wh[K * N]; fread(wh, 4, (size_t) K * N, f); fclose(f);
    f = fopen("/home/kram/matmul/y_ref.f32", "rb");
    static float yref[N]; fread(yref, 4, N, f); fclose(f);

    ggml_backend_t cpu = ggml_backend_cpu_init();

    // find the AXCL device
    ggml_backend_dev_t axcl_dev = ggml_backend_dev_by_name("AXCL0");
    if (!axcl_dev) { printf("AXCL0 not found\n"); _exit(1); }
    ggml_backend_t npu = ggml_backend_dev_init(axcl_dev, nullptr);
    if (!npu) { printf("AXCL backend init failed\n"); _exit(1); }
    printf("backends up (cpu=%s npu=%s)\n", ggml_backend_name(cpu), ggml_backend_name(npu));

    auto run = [&](ggml_backend_t bk, float * out) {
        ggml_init_params ip = { 0, nullptr, GGML_SCHED_OP_EVAL_TRUE };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        memcpy(X->data, xh, (size_t) K * 4);
        memcpy(W->data, wh, (size_t) K * N * 4);

        // simple no-alloc schedule: compute on the given backend directly
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, Y);
        if (!ggml_backend_supports_op(bk, Y)) {
            printf("  backend %s does not support MUL_MAT\n", ggml_backend_name(bk));
            ggml_free(ctx); return false;
        }
        if (ggml_backend_graph_compute(bk, graph) != GGML_STATUS_SUCCESS) {
            printf("  compute failed on %s\n", ggml_backend_name(bk));
            ggml_free(ctx); return false;
        }
        memcpy(out, Y->data, (size_t) N * 4);
        ggml_free(ctx);
        return true;
    };

    static float y_cpu[N], y_npu[N];
    bool ok1 = run(cpu, y_cpu);
    bool ok2 = run(npu, y_npu);
    printf("cpu=%d npu=%d\n", ok1, ok2);
    if (!ok2) _exit(1);

    auto cmp = [&](const char * tag, float * y) {
        double mx = 0, sum = 0;
        for (int i = 0; i < N; i++) { double d = fabs(y[i] - yref[i]); if (d > mx) mx = d; sum += d; }
        printf("%s vs ref: max=%.4f mean=%.5f | y[0..2]: %.3f %.3f %.3f (ref %.3f %.3f %.3f)\n",
               tag, mx, sum / N, y[0], y[1], y[2], yref[0], yref[1], yref[2]);
    };
    if (ok1) cmp("CPU", y_cpu);
    cmp("NPU", y_npu);
    fflush(stdout);
    _exit(0);
}
