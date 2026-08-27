#!/bin/bash
# Benchmark suite: throughput, memory, startup, stress for ggml-axcl modes.
BIN=~/build-axcl/bin/llama-simple
res() { echo "$1"; }

echo "===== 1) THROUGHPUT (48-token gen, 3 runs each) ====="
bench() { # label env-string model
    local label="$1" envs="$2" model="${3:-qwen3-q8}"
    for i in 1 2 3; do
        t=$(env $envs timeout 900 $BIN -m ~/models/$model.gguf -n 48 "Bench mark run number $i:" 2>&1 >/dev/null | grep -a "eval time" | tail -1 | grep -aoE "\(\s*[0-9.]+ tokens per second" | grep -aoE "[0-9.]+")
        p=$(env $envs timeout 900 $BIN -m ~/models/$model.gguf -n 48 "Bench mark run number $i:" 2>&1 >/dev/null | grep -a "prompt eval" | tail -1 | grep -aoE "\(\s*[0-9.]+ tokens per second" | grep -aoE "[0-9.]+")
        echo "  $label run$i: decode=${t}t/s prefill=${p}t/s"
    done
}
bench "legacy     " ""
bench "layer-baked" "GGML_AXCL_LAYER=1 GGML_AXCL_FA=1"
bench "gguf-dyn   " "GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1"
bench "gguf-q4    " "GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1" qwen3-q4km

echo "===== 2) MEMORY during decode (gguf-dyn) ====="
(env GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 timeout 120 $BIN -m ~/models/qwen3-q8.gguf -n 2000 "x" >/dev/null 2>&1 &)
sleep 75
PID=$(pgrep -x llama-simple | head -1)
if [ -n "$PID" ]; then
    J1=$(awk '{print $14+$15}' /proc/$PID/stat); sleep 3; J2=$(awk '{print $14+$15}' /proc/$PID/stat)
    CPU_PCT=$(echo "scale=1; ($J2 - $J1) / 3" | bc)
    RSS=$(ps -o rss= -p $PID | awk '{print $1/1024}')
    echo "  CPU: ${CPU_PCT}% of one core   host RSS: ${RSS}MB"
fi
/usr/bin/axcl/axcl-smi | grep "MiB /" | tail -1 | sed 's/^/  /'
echo "  NPU util:"; /usr/bin/axcl/axcl-smi | grep -E "^\|.*%" | head -1 | sed 's/^/  /'
pkill -x llama-simple; sleep 3

echo "===== 3) CMM release after exit ====="
/usr/bin/axcl/axcl-smi | grep "MiB /" | tail -1 | sed 's/^/  /'

echo "===== 4) STRESS: 5 back-to-back runs (leak check) ====="
for i in 1 2 3 4 5; do
    out=$(env GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 timeout 300 $BIN -m ~/models/qwen3-q8.gguf -n 8 "leak test $i" 2>/dev/null | grep -avE '^\[2026|^$' | head -c 30)
    cmm=$(/usr/bin/axcl/axcl-smi | grep "7040" | grep -oE "[0-9]+ MiB / 7040" | grep -oE "^[0-9]+")
    echo "  run$i: ok='$out' CMM=${cmm}MiB"
done

echo "===== 5) STRESS: SIGINT clean shutdown ====="
(env GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 timeout 120 $BIN -m ~/models/qwen3-q8.gguf -n 2000 "interrupt me" >/dev/null 2>/tmp/sig.err &)
sleep 50; pkill -INT -x llama-simple; sleep 5
if pgrep -x llama-simple >/dev/null; then echo "  FAIL: process survived SIGINT"; pkill -9 -x llama-simple; else echo "  PASS: exited on SIGINT"; fi
sleep 3; /usr/bin/axcl/axcl-smi | grep "MiB /" | tail -1 | sed 's/^/  post-exit CMM: /'

echo "===== 6) STARTUP time (warm cache) ====="
t0=$(date +%s.%N)
env GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 timeout 300 $BIN -m ~/models/qwen3-q8.gguf -n 1 "hi" >/dev/null 2>&1
t1=$(date +%s.%N)
echo "  warm-cache total: $(echo "($t1-$t0)" | bc)s"
