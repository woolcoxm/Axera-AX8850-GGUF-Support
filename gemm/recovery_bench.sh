#!/bin/bash
# recovery_bench.sh — full post-power-cycle test queue (run on the Pi).
# Order: wedge canary -> s4-GPTQ TG -> speculative end-to-end -> kv1024 TG.
set -u
cd ~/phasec
E="GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1"
COMMON="$E GGML_AXCL_POST_MODEL"

echo "=== [0] wedge canary: vocab64 solo load ==="
if ! LD_LIBRARY_PATH=/usr/lib/axcl timeout 60 ./multi_load vocab_m64.axmodel 2>/dev/null | grep -q "ALL 1 LOADED OK"; then
  echo "CARD STILL WEDGED — do a full power cycle (unplug wall power)"; exit 1
fi
echo "card OK"

echo "=== [1] s4-GPTQ TG + coherence ==="
env $COMMON=$HOME/s4-gptq/qwen3_post.axmodel GGML_AXCL_LAYER_DIR=$HOME/s4-gptq \
  ~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 96 \
  "The capital of France is" > /tmp/b_s4.log 2>&1
grep -aE "eval time" /tmp/b_s4.log | tail -1
grep -ao "Paris[^\"]*" /tmp/b_s4.log | head -1

echo "=== [2] speculative end-to-end (ngram draft, chunk verify, vocab64 head) ==="
env $E GGML_AXCL_BATCH=1 GGML_AXCL_VOCAB64=$HOME/phasec/vocab_m64.axmodel \
  GGML_AXCL_LAYER_DIR=$HOME/Qwen3-0.6B GGML_AXCL_POST_MODEL=$HOME/Qwen3-0.6B/qwen3_post.axmodel \
  ~/build-axcl/bin/llama-lookup -m ~/models/qwen3-q8.gguf -t 2 --spec-draft-n-max 16 -n 96 \
  -p "Write a short essay about the history of computing, starting with Charles Babbage." > /tmp/b_spec.log 2>&1
echo "exit=$?"
grep -aE "eval time|tokens per second" /tmp/b_spec.log | tail -2
echo "errors: $(grep -ac 'nil pointer\|dma size' /tmp/b_spec.log)"

echo "=== [3] kv1024 s4 TG ==="
env $COMMON=/tmp/kv1024/qwen3_post.axmodel GGML_AXCL_LAYER_DIR=/tmp/kv1024 \
  ~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 96 \
  "The capital of France is" > /tmp/b_kv1024.log 2>&1
grep -aE "eval time" /tmp/b_kv1024.log | tail -1
grep -ao "Paris[^\"]*" /tmp/b_kv1024.log | head -1
echo "=== DONE ==="
