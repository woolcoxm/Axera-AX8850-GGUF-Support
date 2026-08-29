#!/bin/sh
# wait_card — block until the NPU card is visible to axcl-smi (enumeration
# settle after boot, driver reload, EP reboot, or PCIe rescan).
#
# usage: wait_card [timeout-sec]     (default 30; exit 0 = visible, 1 = not)
#
# Uses axcl-smi's AX650N line as the presence probe (same check the
# regression suite's cmm_now uses). No sudo needed for the probe itself on
# most setups; AXCL_SMI can override the command.
AXCL_SMI="${AXCL_SMI:-sudo /usr/bin/axcl/axcl-smi}"
TMO="${1:-30}"

i=0
while [ "$i" -lt "$TMO" ]; do
    if $AXCL_SMI 2>/dev/null | grep -q AX650N; then
        [ "$i" -gt 0 ] && echo "wait_card: card visible after ${i}s"
        exit 0
    fi
    sleep 1
    i=$((i + 1))
done
echo "wait_card: card NOT visible after ${TMO}s" >&2
exit 1
