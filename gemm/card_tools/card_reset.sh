#!/bin/sh
# card_reset — recover a wedged/dropped AXERA NPU card WITHOUT wall power.
#
# Replaces the "unplug the Pi" step of FINDINGS.md "Card stability rules" #3.
# The vendor runtime (axclhost 3.6.5-m5stack1, already installed) exports
# axclrtRebootDevice() — an EP firmware reboot from software — which the
# axcl_rt_device.h header documents as THE response to DEVICE_OFFLINE.
# Below that sit PCIe-level resets (remove/rescan, FLR, secondary bus reset)
# for wedges where the link is down below the driver.
#
# Ladder, weakest to strongest; stops as soon as the card answers:
#   0. status report
#   1. zombie sweep (SIGTERM -> grace -> SIGKILL)
#   2. EP software reboot        (card_reboot: axclrtRebootDevice)
#   3. driver reload + settle    (host side only)
#   4. EP software reboot again  (device probe-able post-reload)
#   5. PCIe remove + rescan
#   6. PCIe FLR / secondary-bus reset (PERST#) + rescan
#   7. POWER_CMD if configured, else wall-power instruction
#
# usage: sudo ./card_reset.sh [-y]
#   -y  skip the "press enter" confirm before destructive steps (5/6)
# Env: AXCL_SMI, CARD_REBOOT (path), POWER_CMD (smart-plug command), WAIT (sec)

AXCL_SMI="${AXCL_SMI:-sudo /usr/bin/axcl/axcl-smi}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
CARD_REBOOT="${CARD_REBOOT:-$TOOLS/card_reboot}"
WAIT="${WAIT:-30}"
ASSUME_YES=0
[ "${1:-}" = "-y" ] && ASSUME_YES=1

# needs root for modprobe / setpci / sysfs; re-exec under sudo if necessary
if [ "$(id -u)" != 0 ]; then
    exec sudo sh "$TOOLS/card_reset.sh" "$@"
fi

log() { printf '%s | %s\n' "$(date +%H:%M:%S)" "$*"; }

card_visible() { $AXCL_SMI 2>/dev/null | grep -q AX650N; }

card_healthy() {  # visible AND (engine exec if EXECPROBE provided)
    card_visible || return 1
    if [ -n "${EXECPROBE:-}" ] && [ -x "$EXECPROBE" ]; then
        timeout 40 "$EXECPROBE" "$1" 0 1 2>/dev/null | grep -q "execute rc=0"
    else
        true
    fi
}

CANARY="${CANARY_MODEL:-}"
check() {
    if [ -n "$CANARY" ] && [ -n "${EXECPROBE:-}" ] && [ -x "$EXECPROBE" ]; then
        card_healthy "$CANARY"
    else
        card_visible
    fi
}

zombie_sweep() {
    for p in llama-simple llama-server llama-cli axcl_vision; do
        if pgrep -x "$p" >/dev/null 2>&1; then
            log "state: killing stale $p (SIGTERM, 10s grace)"
            pkill -x "$p" 2>/dev/null; sleep 10
            pkill -9 -x "$p" 2>/dev/null; sleep 2
        fi
    done
}

ep_reboot() {
    if [ ! -x "$CARD_REBOOT" ]; then
        log "state: card_reboot not built — run $TOOLS/build.sh on the Pi"
        return 1
    fi
    log "state: EP software reboot ($CARD_REBOOT)"
    "$CARD_REBOOT"
}

driver_reload() {
    log "state: ordered driver reload (resets the HOST side only)"
    for m in axcl_host ax_pcie_host_dev ax_pcie_p2p_rc ax_pcie_mmb ax_pcie_msg; do
        modprobe -r "$m" 2>/dev/null
    done
    sleep 5
    modprobe ax_pcie_msg; modprobe ax_pcie_mmb; modprobe ax_pcie_p2p_rc
    modprobe ax_pcie_host_dev; modprobe axcl_host
    "$TOOLS/wait_card.sh" "$WAIT" && log "state: card enumerated after reload"
}

pci_bdf() {  # prints e.g. 0001:03:00.0 (domain included!) or nothing
    local bdf
    # -D prints the PCI domain: this card sits at 0001:03:00.0, not 0000:...
    bdf="$(lspci -D 2>/dev/null | grep -i axera | awk '{print $1}' | head -1)"
    if [ -z "$bdf" ]; then
        # fallback: whatever device the ax_pcie driver is bound to
        # (BDF names contain colons; bind/unbind/module files don't)
        local d
        for d in /sys/bus/pci/drivers/ax_pcie_host_dev/*; do
            d="$(basename "$d")"
            case "$d" in *:*) bdf="$d"; break;; esac
        done
    fi
    printf '%s\n' "$bdf"
}

pci_rescan() {
    local bdf; bdf="$(pci_bdf)"
    if [ -n "$bdf" ]; then
        log "state: PCIe remove/rescan of $bdf"
        echo 1 > "/sys/bus/pci/devices/$bdf/remove" 2>/dev/null
    else
        log "state: card not on the PCI tree — plain rescan"
    fi
    echo 1 > /sys/bus/pci/rescan 2>/dev/null
    sleep 3
    "$TOOLS/wait_card.sh" "$WAIT" && log "state: card enumerated after rescan"
}

pci_hard_reset() {  # FLR if available, else secondary bus reset (PERST#)
    local bdf; bdf="$(pci_bdf)"
    [ -z "$bdf" ] && { log "state: no PCI device to reset"; return 1; }
    if [ -f "/sys/bus/pci/devices/$bdf/reset" ]; then
        log "state: PCIe FLR on $bdf"
        echo 1 > "/sys/bus/pci/devices/$bdf/reset" 2>/dev/null || log "state: FLR failed, trying SBR"
    fi
    if ! card_visible; then
        # secondary bus reset: assert PERST# via the parent bridge's Bridge
        # Control register (bit 6 at config offset 0x3e) — a real hardware
        # reset of the card, delivered in software
        local sys bridge
        sys="$(readlink -f "/sys/bus/pci/devices/$bdf")"
        bridge="$(basename "$(dirname "$sys")")"
        log "state: PCIe secondary bus reset via bridge $bridge"
        setpci -s "$bridge" 3E.w=40:40  2>/dev/null   # assert PERST#
        sleep 1
        setpci -s "$bridge" 3E.w=00:40  2>/dev/null   # release
        sleep 3
        echo 1 > /sys/bus/pci/rescan 2>/dev/null
    fi
    "$TOOLS/wait_card.sh" "$WAIT" && log "state: card enumerated after hard reset"
}

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    printf 'card_reset: %s — press Enter to continue (Ctrl-C to abort): ' "$1"
    read _ || exit 2
}

# ---------------------------------------------------------------- ladder ----

log "card_reset: start (AXCL card recovery, no wall power)"
$AXCL_SMI 2>/dev/null | sed -n '1,4p' | sed 's/^/smi: /'

if check; then
    log "state: card already healthy — nothing to do"
    exit 0
fi

zombie_sweep
if check; then log "state: GREEN after zombie sweep"; exit 0; fi

ep_reboot
sleep 3
if check; then log "state: GREEN after EP software reboot"; exit 0; fi

driver_reload
if check; then log "state: GREEN after driver reload"; exit 0; fi

ep_reboot
sleep 3
if check; then log "state: GREEN after reload + EP reboot"; exit 0; fi

confirm "PCIe remove/rescan of the card"
pci_rescan
if check; then log "state: GREEN after PCIe rescan"; exit 0; fi

confirm "PCIe FLR / secondary bus reset (hardware card reset via PERST#)"
pci_hard_reset
if check; then log "state: GREEN after PCIe hard reset"; exit 0; fi

if [ -n "${POWER_CMD:-}" ] && [ "${NO_POWER_CMD:-0}" != 1 ]; then
    log "state: invoking POWER_CMD for wall-power cycle: $POWER_CMD"
    sh -c "$POWER_CMD"
    sleep "${POWER_SETTLE:-20}"
    if check; then log "state: GREEN after wall-power cycle"; exit 0; fi
fi

log "FAIL  card unrecoverable in software — WALL-POWER CYCLE required (unplug, 10s, replug)"
log "      then verify: $AXCL_SMI | grep AX650N"
exit 1
