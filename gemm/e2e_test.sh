#!/bin/bash
# E2E test matrix for ggml-axcl on the Pi.
# Usage: ./e2e_test.sh [model]   (default qwen3-q8)
MODEL=${1:-qwen3-q8}
BIN=~/build-axcl/bin/llama-simple
M=~/models/$MODEL.gguf
PASS=0; FAIL=0

run_test() {
    local name="$1"; local prompt="$2"; local n="${3:-48}"; local tmo="${4:-420}"
    echo "=========================================="
    echo "TEST: $name  (n=$n, timeout=${tmo}s)"
    local out=/tmp/e2e_$(echo "$name" | tr ' /' '__').out
    local err=${out%.out}.err
    local t0=$(date +%s)
    timeout $tmo $BIN -m $M -n $n "$prompt" > $out 2> $err
    local rc=$?
    local t1=$(date +%s)
    echo "exit=$rc time=$((t1-t0))s"
    # generation = stdout minus the device-dump lines
    local gen
    gen=$(grep -vE '^\[2026|^$' $out | head -c 400)
    echo "OUT[:400]: $gen"
    # crash detection
    if grep -aqE 'Segmentation fault|SIGSEGV|Aborted|core dumped|terminate called|double free|corrupt' $err; then
        echo "RESULT: CRASH"
        grep -aE 'Segmentation|SIGSEGV|Aborted|core dumped|terminate|double free|corrupt' $err | head -3
        FAIL=$((FAIL+1)); return
    fi
    if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then
        echo "RESULT: FAIL(rc=$rc)"
        tail -5 $err
        FAIL=$((FAIL+1)); return
    fi
    # perf from stderr
    grep -a 'total time' $err | tail -1
    echo "RESULT: PASS"
    PASS=$((PASS+1))
}

# --- normal sizes ---
run_test "small" "The capital of France is" 32
run_test "small_q" "Q: What is 2+2? A:" 32
run_test "medium" "$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 12)")Now summarize this sentence in one word:" 48
run_test "code" "def fibonacci(n):" 64
run_test "long_gen" "Once upon a time" 160 600

# --- large prompts ---
LARGE=$(python3 -c "print('In a shocking finding, scientists discovered a herd of unicorns living in a remote valley. ' * 40)")
run_test "large_~600tok" "$LARGE Continue the story: the unicorns decided to" 32 600
HUGE=$(python3 -c "print('The industrial revolution transformed manufacturing and society across Europe and the world in many profound ways. ' * 110)")
run_test "huge_~1800tok" "$HUGE In summary:" 32 900

# --- adversarial: weird inputs ---
run_test "empty_prompt" "" 16
run_test "single_char" "a" 16
run_test "unicode" "你好，请用一句话介绍大语言模型。" 32
run_test "emoji" "🦄🚀 Explain what this emoji means:" 24
run_test "special_chars" 'printf("%s\n", "hello & <world> `whoami` $(rm -rf ~)");' 32
run_test "newlines" "$(python3 -c "print('\n'.join(['line %d' % i for i in range(30)])))" 24
run_test "very_long_word" "$(python3 -c "print('x' * 2000)")" 16

echo "=========================================="
echo "SUMMARY: PASS=$PASS FAIL=$FAIL"
