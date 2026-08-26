// minimal backend enumeration test: link against the built libggml
// g++ test_axcl_backend.c -o test_axcl -I llama.cpp/ggml/include -L build-axcl/bin -lggml
#include <ggml.h>
#include <ggml-backend.h>
#include <cstdio>

int main() {
    size_t n_reg = ggml_backend_reg_count();
    printf("backends registered: %zu\n", n_reg);
    for (size_t r = 0; r < n_reg; r++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        printf("  reg[%zu] name=%s devices=%zu\n", r, ggml_backend_reg_name(reg),
               ggml_backend_reg_dev_count(reg));
        for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); d++) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            printf("    dev[%zu] %-10s type=%d free=%.2fGiB total=%.2fGiB id=%s\n", d,
                   props.name, (int) props.type, props.memory_free / 1073741824.0,
                   props.memory_total / 1073741824.0, props.device_id ? props.device_id : "n/a");
        }
    }
    // try to init the AXCL backend explicitly
    ggml_backend_dev_t axcl = ggml_backend_dev_by_name("AXCL0");
    if (axcl) {
        ggml_backend_t b = ggml_backend_dev_init(axcl, nullptr);
        printf("AXCL backend init: %s\n", b ? "OK" : "FAILED");
        if (b) ggml_backend_free(b);
    } else {
        printf("AXCL0 device not found\n");
    }
    return 0;
}
