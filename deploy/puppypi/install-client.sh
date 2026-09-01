#!/usr/bin/env bash
# install-client.sh — client-only install for robots that don't run the LLM
# themselves (32-bit OS, no NPU card, or the server lives on another Pi).
# The robot gets llm-ask / llm.py pointed at a llama-server on the network.
#
#   sudo ./install-client.sh [--server-host 10.0.0.81:8080]
set -euo pipefail
HOST="10.0.0.81:8080"
[ "${1:-}" = --server-host ] && [ -n "${2:-}" ] && HOST="$2"

if [ "$EUID" -ne 0 ]; then exec sudo "$0" --server-host "$HOST"; fi
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /opt/llm/robot
cp "$KIT_DIR/llm.py" /opt/llm/robot/llm.py
install -m 755 "$KIT_DIR/llm-ask" /usr/local/bin/llm-ask
cat > /etc/profile.d/llm.sh <<EOF
export LLM_SERVER_HOST=$HOST
EOF

echo
echo "client installed — LLM server: $HOST"
echo "test with:   LLM_SERVER_HOST=$HOST llm-ask \"hello\""
echo "(source /etc/profile.d/llm.sh or re-login to pick up the default)"
