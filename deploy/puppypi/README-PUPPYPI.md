# Incorporating the LLM system into the PuppyPi

This kit deploys **ggml-axcl** (Qwen3.5-0.8B running on the M5Stack LLM-8850 /
Axera AX8850 NPU card via your llama.cpp fork) onto the PuppyPi robot at
`10.0.0.211` (user `pi`). It is self-contained: driver `.deb`, NPU engine set,
GGUF + mmproj, fork source, installer, systemd service, and robot-facing
client tools. Nothing on the robot's existing (Hiwonder) software is modified.

## What ends up on the robot

| Piece | Path |
|---|---|
| NPU engines (24 layers + post + vision) | `/opt/llm/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4` |
| GGUF + mmproj | `/opt/llm/models` |
| llama.cpp build (`llama-server`, `llama-simple`, `llama-mtmd-cli`) | `/opt/llm/build/bin` |
| `axcl_vision` (image encoder on the NPU, ~45 ms) | `/opt/llm/bin` |
| axclhost driver 3.6.5-m5stack1 | system (deb) |
| HTTP service (OpenAI-compatible, port 8080) | `systemctl status llm-serve` |
| `llm-ask` / `llm-look` / `llm-doctor` | `/usr/local/bin` |
| Python client for the Hiwonder apps | `/opt/llm/robot/llm.py` |

## Deploying (from the x86 box)

```bash
tar xf puppypi-deploy.tar.gz
scp -r puppypi-deploy pi@10.0.0.211:            # ~1.8 GB, a few min on LAN
ssh pi@10.0.0.211
cd puppypi-deploy && sudo ./install.sh          # auto-detects; ~15 min build on a Pi 5
llm-ask "Say something cute."
```

The installer auto-detects and branches:

- **NPU mode** — aarch64 OS and the AX8850 card present: full stack, ~27 t/s
  decode, ~900 MB card memory, Pi CPU stays ~idle for the robot's own code.
- **CPU mode** — aarch64 but no card: same llama.cpp build without the AXCL
  backend, running the Q4_K_M GGUF on the Pi's cores (slow; fine for testing).
- **CLIENT mode** — 32-bit OS or `--mode client`: no local LLM. Installs only
  `llm-ask`/`llm.py` pointed at a llama-server elsewhere
  (`--server-host 10.0.0.81:8080`, the Pi 5 that currently hosts the card).

## Moving the NPU card into the robot (when it's a Pi 5)

1. Power off **both** Pis (the card is not hot-pluggable; wall-power-cycle if
   the PCIe link was wedged — a Pi reboot alone does not reset the card).
2. Move the M.2 card + adapter/FPC into the PuppyPi's Pi 5.
3. Boot the robot and run `sudo ./install.sh`. The bundled driver deb ships the
   matching `ax650_card.pac` firmware; since the card already runs the M5Stack
   firmware, **no reflash is triggered** — do not force one (repeated flashes
   can brick the card until a matched-pair reinstall).

## Using it from the robot's software

```python
import sys; sys.path.insert(0, "/opt/llm/robot")
from llm import LLM
llm = LLM()                                   # http://127.0.0.1:8080
reply = llm.ask("The user said: sit. Reply with the action id.")
for tok in llm.ask_stream("Tell me what you are"):   # streaming
    print(tok, end="", flush=True)
```

- CLI chat: `llm-ask "..."` or just `llm-ask` for an interactive session.
- Camera vision: `libcamera-still -o /tmp/s.jpg -n && llm-look /tmp/s.jpg "What do you see?"`.
- Health / recovery: `llm-doctor`, and `llm-doctor --recover` to reload the
  driver module stack if the card wedges (early-wedge fix; deep wedges need a
  full power-off).

## Card stability rules (from the main README — they apply doubly on a robot)

1. Never kill `llm-serve` while engines are loading — the service unit already
   uses `KillMode=mixed` + `TimeoutStopSec=300` so a routine
   `systemctl restart` is safe.
2. Early wedge → `llm-doctor --recover`. Deep wedge → power the robot off
   completely and back on.
3. The driver/firmware are a matched pair — use only the bundled
   `3.6.5-m5stack1` deb with the card's M5Stack firmware.
4. Context is capped at 2048 (`-c 2048`) to match the CTX2047 engines.

## Regenerating the bundle

From the LLMTest repo root on the x86 box: `deploy/puppypi/pack.sh` →
`puppypi-deploy.tar.gz` next to the repo.
