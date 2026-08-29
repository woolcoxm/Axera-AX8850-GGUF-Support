// card_reboot — software card reset for AXERA NPU cards via the axcl host
// runtime, replacing the wall-power cycle that deep wedges otherwise need
// (FINDINGS.md "Card stability rules" #3).
//
// Two mechanisms, weakest first:
//   default        axclrtRebootDevice(dev)  — EP firmware reboot (the vendor
//                                              header documents this as the
//                                              response to DEVICE_OFFLINE)
//   --soft         axclrtResetDevice(dev)   — device reset without EP reboot
//
// Build ON THE PI (aarch64) — see build.sh. Usage: sudo ./card_reboot [--soft]
// Exit codes: 0 rebooted, 1 axclInit failed, 2 no devices visible, 3 reboot call failed.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "axcl.h"
#include "axcl_rt_device.h"

static void dump_devices(void) {
    axclrtDeviceList dl;
    memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC) {
        fprintf(stderr, "card_reboot: GetDeviceList failed\n");
        return;
    }
    printf("card_reboot: %u device(s) visible:", dl.num);
    for (uint32_t i = 0; i < dl.num; i++) {
        printf(" %u", dl.devices[i]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    int soft = (argc > 1 && strcmp(argv[1], "--soft") == 0);

    axclError e = axclInit(NULL);
    if (e != AXCL_SUCC) {
        fprintf(stderr, "card_reboot: axclInit failed (0x%x) — is the axcl driver loaded?\n", e);
        return 1;
    }

    axclrtDeviceList dl;
    memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC || dl.num == 0) {
        fprintf(stderr, "card_reboot: no devices visible to the runtime — "
                        "the PCIe link is down below the driver level.\n"
                        "Try: card_reset.sh (pci remove/rescan + secondary bus reset) first.\n");
        dump_devices();
        axclFinalize();
        return 2;
    }

    int rc = 0;
    for (uint32_t i = 0; i < dl.num; i++) {
        int32_t dev = dl.devices[i];
        if (soft) {
            printf("card_reboot: axclrtResetDevice(%d) ... ", dev);
            fflush(stdout);
            e = axclrtResetDevice(dev);
        } else {
            printf("card_reboot: axclrtRebootDevice(%d) ... ", dev);
            fflush(stdout);
            e = axclrtRebootDevice(dev);
        }
        printf("%s (0x%x)\n", e == AXCL_SUCC ? "OK" : "FAIL", e);
        if (e != AXCL_SUCC) {
            rc = 3;
        }
    }
    axclFinalize();
    return rc;
}
