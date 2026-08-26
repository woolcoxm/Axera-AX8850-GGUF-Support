// dump the AXCL device list ioctl response for state debugging
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

struct device_list_t {
    unsigned int number;
    unsigned int rsv;
    unsigned int dev[64][2]; // bus device pairs (best effort layout)
};

#define IOC_AXCL_MAGIC 'A'
#define IOC_AXCL_DEVICE_LIST _IOWR(IOC_AXCL_MAGIC, 4, struct device_list_t)

int main() {
    setbuf(stdout, NULL);
    int fd = open("/dev/axcl_host", O_RDWR);
    if (fd < 0) { printf("open failed\n"); return 1; }
    struct device_list_t dl; memset(&dl, 0, sizeof(dl));
    int r = ioctl(fd, IOC_AXCL_DEVICE_LIST, &dl);
    printf("ioctl ret=%d\n", r);
    unsigned char *p = (unsigned char *)&dl;
    for (int i = 0; i < 64; i++) {
        printf("%02x", p[i]);
        if (i % 32 == 31) printf("\n");
    }
    printf("\n");
    return 0;
}
