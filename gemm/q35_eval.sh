#!/bin/bash
# q35_eval.sh — Qwen3.5-0.8B quality eval on the Pi + LLM-8850 (vendor w4a16
# engines). Same style as the 0.6B eval_suite: known-answer factuality,
# arithmetic, instruction following, coherence — scored pass/fail with
# substring/regex checks, run on BOTH quants (the engine carries the weights,
# so scores should be identical; this documents exactly that).
#
# usage: ./q35_eval.sh <gguf> [label]

set -u
RUNNER="${RUNNER_BIN:-$HOME/build-axcl/bin/llama-simple}"
EDIR="${Q35_DIR:-$HOME/Qwen3.5-0.8B-int4}"
GGUF="${1:?usage: q35_eval.sh <gguf> [label]}"
LABEL="${2:-$(basename "$GGUF" .gguf)}"
N=64
export LD_LIBRARY_PATH="/usr/lib/axcl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PASS=0; FAIL=0

ask() { # prompt -> stdout (cleaned)
    timeout 200 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
        GGML_AXCL_LAYER_DIR="$EDIR" "$RUNNER" -m "$GGUF" -n "$N" "$1" 2>/dev/null \
        | awk '!/^\[2026/ && !/: 0x/ && !/^allocated device/'
}

check() { # name prompt pattern
    local name="$1" prompt="$2" pat="$3" out
    out=$(ask "$prompt")
    if echo "$out" | grep -qE "$pat"; then
        PASS=$((PASS+1)); echo "PASS  $name"
    else
        FAIL=$((FAIL+1)); echo "FAIL  $name — got: $(echo "$out" | tail -2 | head -c 120)"
    fi
}

# --- factuality ---
check "capital-france"  "What is the capital of France? Answer with just the city name." 'Paris'
check "primes-five"     "What are the first five prime numbers?" '2,? 3,? 5,? 7,? and 11|2, 3, 5, 7(,| and) 11'
check "planets-order"   "Which planet is closest to the sun? One word." 'Mercury'
check "water-formula"   "What is the chemical formula for water?" 'H2O|H₂O'
check "author-moby"     "Who wrote the novel Moby Dick? Surname only." 'Melville'

# --- arithmetic ---
check "arith-12div3"    "Compute 12 divided by 3. Answer with only the number." '4'
check "arith-17plus25"  "Compute 17 plus 25. Answer with only the number." '42'
check "arith-3items"    "Alyssa, Ben, and Carmen split 12 apples evenly. How many each? Number only." '4'

# --- instruction following ---
check "oneline"         "In exactly one short sentence, explain why the sky is blue." '[^.]+\.[^.]*(scatter|blue|light|molecule|Rayleigh)' 
check "list-three"      "List exactly three colors, one per line, nothing else." 'red|blue|green|yellow|black|white'

# --- coherence (story with named entity carry) ---
check "story-entity"    "Write a two-sentence story about a cat named Whiskers who pilots a submarine." 'Whiskers'
check "translation-fr"  "Translate to French: the cat is black. Give only the translation." 'chat est noir|le chat'

echo "=== $LABEL: $PASS pass, $FAIL fail ==="
exit $([ "$FAIL" = 0 ])
