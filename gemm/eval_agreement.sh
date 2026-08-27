#!/bin/bash
# Backend-fidelity eval: token agreement between a candidate mode and the
# legacy CPU reference (greedy decode is deterministic per mode; bf16-engine
# vs q8-GGUF weights give slightly different logits, so near-tie tokens may
# flip — agreement rate quantifies fidelity).
# Usage: eval_agreement.sh "<mode-env-string>" [model]
MODE=${1:-"GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1"}
MODEL=${2:-qwen3-q8}
BIN=~/build-axcl/bin/llama-simple
M=~/models/$MODEL.gguf
N=${N:-24}

PROMPTS=(
  "The capital of France is"
  "Q: What is 2+2? A:"
  "The quick brown fox jumps over the"
  "def fibonacci(n):"
  "Once upon a time there was a"
  "Water boils at a temperature of"
  "The largest planet in the solar system is"
  "Q: What is the capital of Japan? A:"
  "In programming, a loop is used to"
  "The three primary colors are"
)
agree_total=0; tok_total=0; full_match=0; cnt=0
for p in "${PROMPTS[@]}"; do
    a=$(env $MODE timeout 600 $BIN -m $M -n $N "$p" 2>/dev/null | grep -avE '^\[2026|^$')
    b=$(timeout 600 $BIN -m $M -n $N "$p" 2>/dev/null | grep -avE '^\[2026|^$')
    # token-level prefix agreement
    read -r _ rest_a <<< "$a"; read -r _ rest_b <<< "$b"   # skip echoed prompt word count noise
    aa=$(echo "$a" | tr ' ' '\n'); bb=$(echo "$b" | tr ' ' '\n')
    n=0; same=0
    while IFS= read -r ta && IFS= read -r tb <&3; do
        n=$((n+1))
        [ "$ta" = "$tb" ] && same=$((same+1)) || break
    done < <(echo "$aa") 3< <(echo "$bb")
    agree_total=$((agree_total + same)); tok_total=$((tok_total + n))
    [ "$a" = "$b" ] && full_match=$((full_match + 1))
    cnt=$((cnt + 1))
    echo "  [${same}/${n} prefix] $p"
done
echo "===== AGREEMENT: $agree_total/$tok_total prefix tokens ($(( 100 * agree_total / (tok_total > 0 ? tok_total : 1) ))%), exact-match $full_match/$cnt prompts ====="
