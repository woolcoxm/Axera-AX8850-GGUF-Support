# card_tools — killing the wedge-until-power-cycle problem

**The problem**: when a host process dies with NPU work in flight (SIGKILL
mid-execute, segfault, engine-load failure), the card wedges or drops off
the PCIe link. Every later process sees `device 0 is not connected` /
`recv dma size 0`, driver reloads don't help, and the card stays dead until
someone removes wall power (FINDINGS.md "Card stability rules", Axera bug
report #3).

**The discovery (2026-08-29)**: the axcl runtime we already ship —
`axclhost 3.6.5-m5stack1` — contains a software card reset the whole time:

- `axclrtRebootDevice(dev)` — EP firmware reboot from the host (exported
  symbol in `libaxcl_rt.so`; header comment in `axcl_rt_device.h`
  documents exactly this: *on `AXCL_DEVICE_STATUS_OFFLINE` →
  `axclrtRebootDevice(device)`*). This is the same "EP-panic card reset"
  that V3.10.2's axcl-smi exposes — it exists in our stack as an API.
- `axclrtRegisterDeviceStatusCallback` — tells you the moment the device
  goes OFFLINE, instead of finding out from a zero-byte DMA.
- `axclrtResetDevice` — soft device reset (kept as `--soft` option).

## The fix (three layers)

### 1. Backend never causes a wedge — `llama.cpp/ggml/src/ggml-axcl/ggml-axcl.cpp`

- **Fail-fast**: once the device reports OFFLINE, graph compute refuses to
  submit NPU work (hammering a dead channel is what turns a transient drop
  into a deep wedge). The process exits; systemd/ops restarts it into a
  healthy card.
- **Auto-reboot**: on OFFLINE the backend calls `axclrtRebootDevice` itself
  (vendor-documented pattern), so a dropped card self-heals in software.
  Disable with `GGML_AXCL_AUTO_REBOOT=0`.
- **Signal guards**: SIGTERM no longer kills the process mid-execute —
  compute stops between engine calls and the process unwinds to a clean
  exit (atexit runtime finalize). A 10s alarm preserves stop semantics for
  apps that never exit. SIGSEGV/SIGABRT/SIGBUS/SIGFPE do a best-effort
  `axclrtEngineFinalize` + `axclFinalize` before dying (alarm(3) watchdog
  against hanging inside the dead runtime). Disable with
  `GGML_AXCL_SIGNAL_GUARD=0`. Handlers are only installed where the app
  hasn't set its own.
- **Enumeration race absorbed**: activation and device-count now wait up
  to `GGML_AXCL_CONNECT_TIMEOUT` (default 15s) for the card to appear
  after boot/driver-reload, instead of failing instantly with
  `device 0 is not connected` or silently registering zero NPU devices.
- **Partial-activation leak fixed**: failed engine init no longer leaks a
  live context + runtime worker threads past `main()` (which aborted during
  exit handling *with the device held* — a wedge vector of its own).

### 2. Recovery without wall power — this directory

```
build.sh        build card_reboot ON THE PI (aarch64 host libs)
card_reboot.c   axclrtRebootDevice / axclrtResetDevice wrapper
card_reset.sh   full ladder: zombies -> EP reboot -> driver reload ->
                EP reboot -> PCI remove/rescan -> FLR / secondary-bus
                reset (PERST#) -> POWER_CMD -> wall-power instruction
wait_card.sh    block until axcl-smi sees the AX650N again
```

`sudo ./card_reset.sh` interactively, `-y` for automation. The
regression suite's `recover()` now runs this ladder before declaring the
card BLACK, so deep wedges that used to force a reboot+wall-power are
attempted in software first.

### 3. Deployments self-heal

For llama-server (or any long-running service) on "people's devices":

```ini
# /etc/systemd/system/llama-axcl.service
[Service]
ExecStart=/usr/local/bin/llama-server ...
Restart=on-failure
RestartSec=15
# the OOM killer SIGKILLs — the ONE death no code can intercept. Make it
# never choose us, and cap memory so the kernel never needs to:
OOMScoreAdjust=-1000
MemoryHigh=3G          # throttle before reclaim, don't kill
# card drop -> backend reboots EP + exits -> systemd restarts into a
# healthy card; worst case add OnFailure=card-reset.service
```

The three SIGKILL sources and their closures: our timeouts (closed —
SIGINT/SIGTERM-first everywhere + 12s zombie grace), our crashes (closed at
the source — signal-guard finalize; remaining crash bugs like the
llama-lookup SIGSEGV still need fixing), and the OOM killer (closed by
deployment config above). There is no fourth kind: SIGKILL cannot be caught
by any process, ever, by OS design.

## Verification on the Pi (this desktop can't reach the card)

1. `cd card_tools && ./build.sh && sudo ./card_reboot` on a HEALTHY card —
   expect `axclrtRebootDevice(n) ... OK` and the card back in `axcl-smi`
   within a few seconds. This proves the API works on our stack.
2. Rebuild the backend (`ggml-axcl.cpp` changed): rebuild llama-simple,
   run `./regression_suite.sh quick` — the suite exercises the new init
   path end to end.
3. Wedge drill (optional, you know the recipe): reproduce a wedge the old
   way (kill -9 mid-execute), then `sudo card_tools/card_reset.sh -y` —
   it should recover at the EP-reboot or PCI steps instead of requiring
   wall power. If the EP reboot alone fixes it, consider setting
   `OnFailure=` in the service to skip even the reboot.
4. Boot-race check: `sudo reboot`, immediately start llama-simple —
   expect the `waiting up to 15s (enumeration race)` log once, then a
   normal run (no `device 0 is not connected`, no CPU fallback).

## Notes / limits

- `axclrtRebootDevice` reaches the card through the driver — if the PCIe
  link is fully down (`card_reboot: no devices visible`), the PCI
  remove/rescan and PERST# secondary-bus-reset steps take over; those are
  the last software options before wall power.
- The PERST# reset asserts the parent bridge's Secondary Bus Reset bit
  (`setpci 3E.w=40:40`). On the Pi 5 the card hangs off the root port —
  test the SBR step on a healthy card once before trusting it in anger.
- The vendor engine-load-order bug (pulsar2-build heads must load FIRST)
  still exists; the backend's ordering workaround stays. The new defenses
  only stop its blast radius from being a power cycle.
