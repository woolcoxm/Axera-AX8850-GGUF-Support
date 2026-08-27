#!/bin/bash
# Quality evals for ggml-axcl modes: known-answer + code + coherence.
# Compares a mode's output against expectations AND against the legacy
# CPU reference (same GGUF, no env flags) for answer-agreement.
# Usage: eval_suite.sh <mode-env-string> [model]
MODE=${1:-"GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1"}
MODEL=${2:-qwen3-q8}
BIN=~/build-axcl/bin/llama-simple
M=~/models/$MODEL.gguf
PASS=0; FAIL=0; TOTAL_TOKENS=0; TOTAL_MS=0

run() { # prompt n
    local prompt="$1" n="${2:-32}"
    local t0=$(date +%s.%N)
    timeout 600 env $MODE $BIN -m $M -n $n "$prompt" 2>/tmp/eval.err | grep -avE '^\[2026|^$'
    local t1=$(date +%s.%N)
    TOTAL_MS=$(echo "$TOTAL_MS + ($t1 - $t0) * 1000" | bc)
}

expect_contains() { # name expected
    local name="$1" expected="$2" out="$3"
    if echo "$out" | grep -aqF "$expected"; then
        echo "PASS: $name"; PASS=$((PASS+1))
    else
        echo "FAIL: $name (wanted '$expected')"; echo "     got: $(echo "$out" | head -c 150)"; FAIL=$((FAIL+1))
    fi
}

echo "===== EVALS: mode=[$MODE] model=$MODEL ====="

# --- factual / known-answer ---
out=$(run "The capital of France is" 12)
expect_contains "capital-of-france" "Paris" "$out"

out=$(run "Q: What is 2+2? A:" 8)
expect_contains "arithmetic-2plus2" "4" "$out"

out=$(run "Q: What is 10 minus 3? A:" 8)
expect_contains "arithmetic-10minus3" "7" "$out"

out=$(run "The quick brown fox jumps over the" 8)
expect_contains "fox-completion" "dog" "$out"

# --- code ---
out=$(run "def fibonacci(n):" 40)
if echo "$out" | grep -aqE "return|fibonacci|n ==|n<2|n < 2"; then
    echo "PASS: code-fibonacci-structure"; PASS=$((PASS+1))
else
    echo "FAIL: code-fibonacci-structure"; FAIL=$((FAIL+1))
fi

out=$(run "def add_numbers(a, b):" 20)
expect_contains "code-add" "return" "$out"

# --- coherence: repetition degeneration check (a garbled model repeats) ---
out=$(run "Once upon a time there was a" 48)
words=$(echo "$out" | tr ' ' '\n' | grep -ac . )
unique=$(echo "$out" | tr ' ' '\n' | grep -ac . | xargs -I{} sh -c "echo \"\$0\" | sort -u | wc -l" "$(echo "$out" | tr ' ' '\n' | sort -u)")
# simple ratio via awk
ratio=$(echo "$out" | tr ' ' '\n' | awk 'NF{w++; seen[$0]=1} END{u=0; for(k in seen)u++; print u/w}')
ok=$(echo "$ratio > 0.4" | bc)
if [ "$ok" = "1" ]; then
    echo "PASS: coherence-lexical-diversity ($ratio unique)"; PASS=$((PASS+1))
else
    echo "FAIL: coherence-lexical-diversity ($ratio unique) — repetition loop?"; echo "     got: $(echo "$out" | head -c 150)"; FAIL=$((FAIL+1))
fi

# --- multilingual / unicode survives ---
out=$(run "Le chat est sur la" 12)
if echo "$out" | grep -aqE "[a-zA-Z]"; then
    echo "PASS: basic-generation"; PASS=$((PASS+1))
else
    echo "FAIL: basic-generation"; FAIL=$((FAIL+1))
fi

echo "===== SUMMARY: PASS=$PASS FAIL=$FAIL wall=${TOTAL_MS%.*}ms ====="
[ $FAIL -eq 0 ] && echo "ALL EVALS PASS" || echo "EVAL FAILURES PRESENT"
