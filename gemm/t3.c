#include <stdio.h>
#include <axcl.h>
#include <axcl_rt_device.h>

int main() {
    setbuf(stdout, NULL);
    axclError e = axclInit(NULL);
    printf("axclInit -> %d\n", (int) e);
    uint32_t n = 0;
    e = axclrtGetDeviceCount(&n);
    printf("GetDeviceCount -> %d count=%u\n", (int) e, n);
    e = axclrtSetDevice(0);
    printf("SetDevice(0) -> %d\n", (int) e);
    axclrtDeviceProperties p;
    e = axclrtGetDeviceProperties(0, &p);
    printf("GetProps -> %d sw=%s cmm_total=%u KB temp=%u\n", (int) e, p.swVersion, p.totalCmmSize, p.temperature);
    printf("done\n");
    return 0;
}
