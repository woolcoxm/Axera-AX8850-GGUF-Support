#!/usr/bin/env bash
# pack.sh — assemble puppypi-deploy.tar.gz from the LLMTest repo's assets.
# Run from anywhere; output lands next to the repo (NOT inside it — it's 1.8GB).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
KIT="$REPO/deploy/puppypi"
STAGE="$(mktemp -d /tmp/puppypi-pack.XXXX)"
OUT="$REPO/../puppypi-deploy.tar.gz"

EDIR="$REPO/models/Qwen3.5-0.8B-AX650-GPTQ-Int4"
GGUFDIR="$REPO/models/qwen3.5-0.8b-gguf"

echo "==> staging into $STAGE/puppypi-deploy"
mkdir -p "$STAGE/puppypi-deploy/assets/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4" \
         "$STAGE/puppypi-deploy/assets/gguf"

# scripts + docs
install -m 755 "$KIT"/install.sh "$KIT"/install-client.sh "$KIT"/llm-ask \
               "$KIT"/llm-look "$KIT"/llm-doctor "$STAGE/puppypi-deploy/"
install -m 644 "$KIT"/llm.py "$KIT"/README-PUPPYPI.md "$STAGE/puppypi-deploy/"
install -m 644 "$REPO/gemm/axcl_vision.c" "$STAGE/puppypi-deploy/axcl_vision.c"

# driver (matched pair with the card's M5Stack firmware)
cp "$REPO"/driver-good/axclhost_3.6.5-m5stack1_arm64.deb "$STAGE/puppypi-deploy/assets/"

# engine set — .axmodels + configs only; the backend never reads the vendor's
# embed_tokens bf16 bin (GGUF supplies embeddings) → saves 508 MB
cp "$EDIR"/*.axmodel "$STAGE/puppypi-deploy/assets/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4/"
cp "$EDIR"/config.json "$EDIR"/post_config.json \
      "$STAGE/puppypi-deploy/assets/engines/Qwen3.5-0.8B-AX650-GPTQ-Int4/" 2>/dev/null || true

# model artifacts
cp "$GGUFDIR/Qwen3.5-0.8B-Q4_K_M.gguf" "$GGUFDIR/mmproj-BF16.gguf" \
      "$STAGE/puppypi-deploy/assets/gguf/"

# llama.cpp fork source (exact tested tree, no .git), wrapped in a llama.cpp/ dir
echo "==> archiving llama.cpp fork ($(git -C "$REPO/llama.cpp" describe --always --dirty))"
mkdir -p "$STAGE/puppypi-deploy/assets/llama.cpp"
git -C "$REPO/llama.cpp" archive --format=tar HEAD \
    | tar x -C "$STAGE/puppypi-deploy/assets/llama.cpp"
(cd "$STAGE/puppypi-deploy/assets" && tar czf llama.cpp-src.tar.gz llama.cpp)
rm -rf "$STAGE/puppypi-deploy/assets/llama.cpp"

echo "==> writing $OUT (plain tar; ~1.8 GB, models barely compress)"
rm -f "$OUT"
tar cf "$OUT" -C "$STAGE" puppypi-deploy
du -h "$OUT"
echo "done: $OUT"
rm -rf "$STAGE"
