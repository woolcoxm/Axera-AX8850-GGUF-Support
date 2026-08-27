#!/bin/bash
# Serial code-marker builds for the vendor-w8a16 (Pulsar2 5.2) layout crack.
# Vendor's exact flags (from the AXERA-TECH/Qwen3-0.6B README), -c 1 -> -c 0
# (marker weights fail the checker; layout is check-independent).
set -e
cd /home/kram/Desktop/Projects/LLMTest/gemm
P52=/home/kram/Desktop/Projects/LLMTest/pulsar2/5.2/5.2/ax_pulsar2_5.2_lite_package
export PATH=$P52/bin:$PATH
PY=/usr/bin/python3

build_one() {
    local tag=$1 mode=$2
    if [ -f /tmp/v52_${tag}_l0_params.bin ]; then
        echo "== $tag already done"
        return
    fi
    echo "== $tag: generate + build ($(date +%H:%M:%S))"
    $PY mk_code_marker.py /tmp/v52_${tag} $mode
    env FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
        --input_path /tmp/v52_${tag} --output_path /tmp/v52_out_${tag} \
        --hidden_state_type bf16 --kv_cache_len 2048 --prefill_len 128 \
        --chip AX650 -c 0 --parallel 32 \
        --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 \
        --last_kv_cache_len 512 --last_kv_cache_len 640 --last_kv_cache_len 768 \
        --last_kv_cache_len 896 --last_kv_cache_len 1024 -w s8 \
        > /tmp/v52_build_${tag}.log 2>&1
    grep -q "build llm model done" /tmp/v52_build_${tag}.log || {
        echo "!! $tag build FAILED"; tail -5 /tmp/v52_build_${tag}.log; return 1; }
    $PY extract_npu_params.py /tmp/v52_out_${tag}/qwen3_p128_l0_together.axmodel \
        /tmp/v52_${tag}_l0_params.bin
    ls -la /tmp/v52_${tag}_l0_params.bin
}

for spec in "cmd2 d2" "cm3 3" "cm4 4" "cm5 5" "cm6 6" "cm7 7" "cmmc mc"; do
    build_one $spec
done
echo "ALL MARKER BUILDS DONE ($(date +%H:%M:%S))"
