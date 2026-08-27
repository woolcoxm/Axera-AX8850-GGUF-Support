#!/bin/bash
# final_suite.sh — e2e + evals + agreement + bench for the s4-GPTQ mode.
# Detached runner; writes progress to /tmp/final_tests.log.
L=/tmp/final_tests.log
cd ~/phasec
S4="GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 GGML_AXCL_LAYER_DIR=$HOME/s4-gptq GGML_AXCL_POST_MODEL=$HOME/s4-gptq/qwen3_post.axmodel"

echo "=== E2E (s4 + batched prefill) $(date +%T) ===" >> $L
env $S4 GGML_AXCL_BATCH=1 bash ~/Desktop/Projects/LLMTest/gemm/e2e_test.sh >> $L 2>&1 || true
# e2e lives in repo + Pi copy; fall back
if [ ! -f ~/Desktop/Projects/LLMTest/gemm/e2e_test.sh ]; then
  env $S4 GGML_AXCL_BATCH=1 bash /tmp/e2e_test.sh >> $L 2>&1 || true
fi
echo "=== EVAL SUITE (s4) $(date +%T) ===" >> $L
bash /tmp/eval_suite.sh "$S4" >> $L 2>&1 || true
echo "=== AGREEMENT (s4) $(date +%T) ===" >> $L
bash /tmp/eval_agreement.sh "$S4" >> $L 2>&1 || bash /tmp/eval_agreement.sh $HOME/s4-gptq >> $L 2>&1 || true
echo "=== BENCH s4 decode x3 $(date +%T) ===" >> $L
for i in 1 2 3; do
  env $S4 timeout 600 ~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 48 "Bench run $i: the quick brown fox" 2>&1 >/dev/null | grep -a "eval time" | tail -1 >> $L
done
echo "=== BENCH s4+batch prefill x3 $(date +%T) ===" >> $L
P=$(python3 -c "print(' '.join(f'note {i}: bandwidth and latency measurement on the accelerator card during sustained transfer tests.' for i in range(14)))")
for i in 1 2 3; do
  env $S4 GGML_AXCL_BATCH=1 timeout 600 ~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 48 "$P" 2>&1 >/dev/null | grep -aE "prompt eval|eval time" | tail -2 >> $L
done
echo "SUITE_DONE $(date +%T)" >> $L
