#!/usr/bin/env bash
# install.sh — deploy the ggml-axcl LLM system (Qwen3.5-0.8B on the AX8850 NPU
# card) onto a PuppyPi / any aarch64 Raspberry Pi.
#
# Usage:
#   sudo ./install.sh                       # auto-detect
#   sudo ./install.sh --mode npu            # full NPU stack (card must be present)
#   sudo ./install.sh --mode cpu            # CPU-only llama.cpp (no card)
#   sudo ./install.sh --mode client --server-host 10.0.0.81:8080
#                                           # robot queries a llama-server elsewhere
#
# What it installs (nothing of the robot's existing software is touched):
#   /opt/llm/engines  NPU engine set (24 layer + post + vision .axmodels)
#   /opt/llm/models   Qwen3.5-0.8B Q4_K_M GGUF + mmproj
#   /opt/llm/src      llama.cpp fork (branch Axera-8850-GGUF-support-PoC-...)
#   /opt/llm/build    llama-server, llama-simple, llama-mtmd-cli
#   /opt/llm/bin      axcl_vision (image encoder driven on the NPU)
#   /opt/llm/robot    llm.py client library
#   /usr/local/bin    llm-ask, llm-look, llm-doctor
#   axclhost driver deb, systemd service llm-serve (port 8080)
set -euo pipefail

MODE=auto
SERVER_HOST=""
PREFIX=/opt/llm
PORT=8080

while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --server-host) SERVER_HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ "$EUID" -ne 0 ]; then exec sudo "$0" --mode "$MODE" ${SERVER_HOST:+--server-host "$SERVER_HOST"} --port "$PORT"; fi

KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$KIT_DIR/assets"
ENGINES_SRC="$ASSETS/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4"
GGUF="$ASSETS/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
MMPROJ="$ASSETS/gguf/mmproj-BF16.gguf"
DEB="$(ls "$ASSETS"/axclhost_*_arm64.deb 2>/dev/null | head -1 || true)"

step() { echo; echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

[ -f "$GGUF" ]      || die "assets not found — run from the unpacked puppypi-deploy dir (got: $KIT_DIR)"

# ---------------------------------------------------------------- detection
ARCH="$(uname -m)"
MODEL_PI="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "host: $MODEL_PI  arch=$ARCH  kernel=$(uname -r)"

if [ "$MODE" = auto ]; then
    if [ "$ARCH" != aarch64 ]; then
        echo "32-bit OS detected — the AXCL driver and engines are aarch64-only."
        echo "Falling back to CLIENT mode (query a llama-server on the network)."
        MODE=client
        SERVER_HOST="${SERVER_HOST:-10.0.0.81:$PORT}"
    else
        MODE=npu   # card presence is re-checked after the driver install
    fi
fi

# disk space: engines 0.9G + gguf 0.75G + src/build ~2.5G staged
AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "$MODE" = npu ] || [ "$MODE" = cpu ]; then
    [ "$AVAIL_GB" -ge 5 ] || die "only ${AVAIL_GB}GB free on / — need ~5GB (engines + models + build). Free space or use --mode client."
fi

# ---------------------------------------------------------------- client mode
if [ "$MODE" = client ]; then
    SERVER_HOST="${SERVER_HOST:-10.0.0.81:$PORT}"
    "$KIT_DIR/install-client.sh" --server-host "$SERVER_HOST"
    exit 0
fi

# ---------------------------------------------------------------- packages
step "installing build prerequisites (apt)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || echo "WARNING: apt update failed — continuing"
apt-get install -y -qq build-essential cmake libcurl4-openssl-dev file >/dev/null

IS_RPI=false; grep -qiE 'raspberry pi' <<<"$MODEL_PI" && IS_RPI=true
if $IS_RPI; then
    step "installing Raspberry Pi kernel headers (driver builds its modules)"
    if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
        apt-get install -y -qq raspberrypi-kernel-headers >/dev/null || true
    fi
    [ -d "/lib/modules/$(uname -r)/build" ] || {
        echo " kernel headers for $(uname -r) are not available."
        echo " Fix with:  sudo apt update && sudo apt install --reinstall raspberrypi-kernel raspberrypi-kernel-headers && sudo reboot"
        echo " then re-run this installer."
        if [ "$MODE" = npu ]; then
            echo " continuing in CPU mode for now."
            MODE=cpu
        fi
    }
fi

# ---------------------------------------------------------------- driver + card
HAVE_CARD=false
AXCL_LIB=/usr/lib/axcl/libaxcl_rt.so
if [ "$MODE" = npu ]; then
    step "installing axclhost driver (M5Stack 3.6.5-m5stack1, matched to the card firmware)"
    [ -n "$DEB" ] || die "axclhost deb missing from assets/"
    DOWNGRADE=""
    if dpkg-query -W -f'${Version}\n' axclhost 2>/dev/null | grep -qE '^3\.(6\.[6-9]|[7-9])'; then
        echo " newer axclhost ($(dpkg-query -W -f'${Version}' axclhost)) installed — downgrading to the verified matched pair"
        DOWNGRADE="--allow-downgrades"
    fi
    apt-get install -y -qq $DOWNGRADE "$DEB" >/dev/null || {
        echo "driver install failed (module build?) — see /var/log/axclhost-install.log"
        echo " the LLM keeps working in CPU mode; re-run this installer after"
        echo " installing kernel headers for $(uname -r)."
        exit 1
    }
    if [ -e /dev/axcl_host ]; then
        HAVE_CARD=true
        echo "card: PRESENT (/dev/axcl_host)"
    else
        echo "card: NOT FOUND (axcl-smi reports no device connected)."
        echo " The stack is installed and NPU-ready; to use it, power off both"
        echo " Pis, move the AX8850 M.2 card into this Pi 5, boot, and re-run"
        echo " this installer (it flips the service to NPU mode)."
        echo " Meanwhile the LLM runs on the CPU."
        HAVE_CARD=false
    fi
fi
if $HAVE_CARD; then
    /usr/bin/axcl/axcl-smi || true   # informational
fi
# build against the AXCL backend whenever its libs exist (works without a card)
BUILD_AXCL=false; [ -f "$AXCL_LIB" ] && BUILD_AXCL=true

# ---------------------------------------------------------------- assets
step "installing engines + models into $PREFIX"
mkdir -p "$PREFIX"/{engines,models,bin,robot,src}
if $BUILD_AXCL; then
    mkdir -p "$PREFIX/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4"
    cp -u "$ENGINES_SRC"/*.axmodel "$PREFIX/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4/" 2>/dev/null || true
    cp -u "$ENGINES_SRC"/config.json "$ENGINES_SRC"/post_config.json "$PREFIX/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4/" 2>/dev/null || true
fi
cp -u "$GGUF" "$PREFIX/models/"
[ -f "$MMPROJ" ] && cp -u "$MMPROJ" "$PREFIX/models/"

# ---------------------------------------------------------------- build llama.cpp
step "building llama.cpp (this is the slow part: ~10-15 min on a Pi 5)"
if [ ! -d "$PREFIX/src/llama.cpp" ]; then
    mkdir -p "$PREFIX/src"
    tar xf "$ASSETS/llama.cpp-src.tar.gz" -C "$PREFIX/src"
fi
AXCL_FLAG=OFF; $BUILD_AXCL && AXCL_FLAG=ON
cmake -S "$PREFIX/src/llama.cpp" -B "$PREFIX/build" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_AXCL=$AXCL_FLAG \
      -DBUILD_SHARED_LIBS=OFF > /dev/null
cmake --build "$PREFIX/build" --target llama-server llama-simple llama-mtmd-cli \
      -j"$(nproc)" > /tmp/llm-build.log 2>&1 \
    || { tail -30 /tmp/llm-build.log; die "build failed (full log: /tmp/llm-build.log)"; }
echo "built: $(ls "$PREFIX/build/bin/" | tr '\n' ' ')  (GGML_AXCL=$AXCL_FLAG)"

# ---------------------------------------------------------------- vision tool
if $BUILD_AXCL; then
    step "building axcl_vision (NPU image encoder)"
    gcc -O2 -o "$PREFIX/bin/axcl_vision" "$KIT_DIR/axcl_vision.c" \
        -I"$PREFIX/src/llama.cpp/vendor/stb" -I/usr/include/axcl -L/usr/lib/axcl -laxcl_rt -laxcl \
        || echo "WARNING: axcl_vision build failed — vision (llm-look) will use the CPU path"
fi

# ---------------------------------------------------------------- verify on metal
if $HAVE_CARD; then
    step "verification: one short generation on the NPU (first run loads ~1GB of engines, be patient)"
    OUT=$(timeout 600 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
        GGML_AXCL_LAYER_DIR="$PREFIX/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4" \
        "$PREFIX/build/bin/llama-simple" -m "$PREFIX/models/Qwen3.5-0.8B-Q4_K_M.gguf" \
        -n 24 "The capital of France is" 2>/tmp/llm-verify.log || true)
    echo "$OUT"
    if ! grep -qE 'Paris|capital' <<<"$OUT"; then
        echo "WARNING: generation looked wrong — tail of the run log:"
        tail -15 /tmp/llm-verify.log
    fi
fi

# ---------------------------------------------------------------- service + clients
step "installing llm-serve systemd service (port $PORT)"
# pick a free port if the default is taken (Hiwonder apps sometimes use 8080)
if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT$"; then
    for P in 8081 8082 8083 8084; do
        ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$P$" || { PORT=$P; break; }
    done
    echo " port 8080 busy — using $PORT"
fi
EDIR="$PREFIX/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4"
if $HAVE_CARD; then
    cat > /etc/systemd/system/llm-serve.service <<EOF
[Unit]
Description=ggml-axcl llama-server — Qwen3.5-0.8B on AX8850 NPU
After=network.target

[Service]
Environment=GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1
Environment=GGML_AXCL_LAYER_DIR=$EDIR
ExecStart=$PREFIX/build/bin/llama-server -m $PREFIX/models/Qwen3.5-0.8B-Q4_K_M.gguf --host 0.0.0.0 --port $PORT -c 2048 --jinja
Restart=on-failure
RestartSec=15
# card rule: never SIGKILL while engines are loading (can wedge PCIe)
KillMode=mixed
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
EOF
else
    # no card (yet): CPU service; the AXCL build + staged engines are ready for
    # the day the card moves in — then re-run this installer to switch over.
    cat > /etc/systemd/system/llm-serve.service <<EOF
[Unit]
Description=llama-server — Qwen3.5-0.8B (CPU; NPU-ready, card not present)
After=network.target

[Service]
ExecStart=$PREFIX/build/bin/llama-server -m $PREFIX/models/Qwen3.5-0.8B-Q4_K_M.gguf --host 0.0.0.0 --port $PORT -c 2048 --jinja -t $(nproc)
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF
fi
systemctl daemon-reload
systemctl enable llm-serve.service >/dev/null 2>&1
systemctl restart llm-serve.service

step "installing client tools"
cp "$KIT_DIR/llm.py"        "$PREFIX/robot/llm.py"
install -m 755 "$KIT_DIR/llm-ask"   /usr/local/bin/llm-ask
install -m 755 "$KIT_DIR/llm-look"  /usr/local/bin/llm-look
install -m 755 "$KIT_DIR/llm-doctor" /usr/local/bin/llm-doctor
cat > /etc/profile.d/llm.sh <<EOF
export LLM_SERVER_HOST=127.0.0.1:$PORT
EOF

step "waiting for the server to come up (engine load can take ~2 min on cold start)"
for i in $(seq 1 120); do
    if curl -fs -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "service healthy after ~${i}x2s"; break
    fi
    sleep 2
    [ "$i" = 120 ] && { echo "service did not become healthy — check: journalctl -u llm-serve -e"; }
done

echo
echo "=========================== DONE ==========================="
if $HAVE_CARD; then
    echo " NPU mode. Try:   llm-ask \"Name three planets.\""
    echo " Camera:          llm-look /tmp/snap.jpg \"What do you see?\""
    echo " Service:         systemctl status llm-serve   (port $PORT)"
else
    echo " CPU mode (no AX8850 card detected)."
    echo " Remote option:   sudo $KIT_DIR/install-client.sh --server-host 10.0.0.81:$PORT"
fi
echo " Health:          llm-doctor"
echo "============================================================"
