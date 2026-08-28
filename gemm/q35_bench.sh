#!/bin/bash
# q35_bench.sh — Qwen3.5-0.8B benchmark matrix on the Pi + LLM-8850.
# Produces the README numbers: decode t/s (short + ~2k ctx), ladder prefill
# t/s, card memory (CMM) during steady decode, engine-load time — for BOTH
# Q4_K_M and Q8_0 GGUFs (same w4a16 engine set; the GGUF only supplies
# tokenizer/graph/sampling, so any quant works — this documents that).
#
# usage: ./q35_bench.sh [gguf...]     (default: both quants)
# output: markdown fragments in /tmp (and stdout) — paste into README.

set -u
RUNNER="${RUNNER_BIN:-$HOME/build-axcl/bin/llama-simple}"
EDIR="${Q35_DIR:-$HOME/Qwen3.5-0.8B-int4}"
AXCL_SMI="sudo /usr/bin/axcl/axcl-smi"
export LD_LIBRARY_PATH="/usr/lib/axcl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmm() { $AXCL_SMI 2>/dev/null | grep -A1 AX650N | tail -1 | grep -oE '[0-9]+ MiB /' | head -1 | grep -oE '^[0-9]+'; }

bench_one() { # label gguf
    local label="$1" gguf="$2"
    local out="/tmp/bench_$label"
    echo "| $label |" >&2
    # warm caches
    cat "$EDIR"/*.axmodel "$gguf" >/dev/null 2>&1 || true
    local base; base=$(cmm); echo "  cmm baseline: $base MiB"

    # decode @ short ctx (48-token prompt, 256 tokens out)
    local t0 t1
    t0=$(date +%s%N)
    timeout 300 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
        GGML_AXCL_LAYER_DIR="$EDIR" "$RUNNER" -m "$gguf" -n 256 \
        "Write a detailed story about a robot exploring Mars." >/dev/null 2>"$out.short.dbg"
    t1=$(date +%s%N)
    local short_tps short_ct
    short_tps=$(grep -E "^\s*llama_perf_context_print:\s+eval time" "$out.short.dbg" | grep -oE "[0-9.]+ tokens per second" | grep -oE "[0-9.]+" | head -1)
    short_ct=$(awk "BEGIN{print ($t1-$t0)/1000000000}")

    # decode @ ~2k ctx: 2000-token generation (ctx grows to ~2030)
    timeout 300 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
        GGML_AXCL_LAYER_DIR="$EDIR" "$RUNNER" -m "$gguf" -n 2000 \
        "count upward from one and never stop" >/dev/null 2>"$out.long.dbg"
    local long_tps
    long_tps=$(grep -E "^\s*llama_perf_context_print:\s+eval time" "$out.long.dbg" | grep -oE "[0-9.]+ tokens per second" | grep -oE "[0-9.]+" | head -1)

    # CMM during steady decode of a long run (sample 6x5s from a 1500-tok run)
    timeout 120 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
        GGML_AXCL_LAYER_DIR="$EDIR" "$RUNNER" -m "$gguf" -n 1500 \
        "keep writing numbers without stopping" >/dev/null 2>/dev/null &
    local pid=$! cmm_max=0 i v
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        sleep 5; v=$(cmm); [ -n "$v" ] && [ "$v" -gt "$cmm_max" ] && cmm_max=$v
        kill -0 $pid 2>/dev/null || break
    done
    wait $pid 2>/dev/null

    # ladder prefill (847-token prompt)
    local P; P=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 60)")
    timeout 300 env GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 GGML_AXCL_BATCH=1 \
        GGML_AXCL_LAYER_DIR="$EDIR" "$RUNNER" -m "$gguf" -n 8 \
        "$P Summarize in one sentence." >/dev/null 2>"$out.pf.dbg"
    local pf_tps pf_tok
    pf_tps=$(grep -E "prompt eval time" "$out.pf.dbg" | grep -oE "[0-9.]+ tokens per second" | grep -oE "[0-9.]+" | head -1)
    pf_tok=$(grep -E "prompt eval time" "$out.pf.dbg" | grep -oE "/ +[0-9]+ tokens" | grep -oE "[0-9]+" | head -1)

    echo "| $label | ${short_tps:-?} | ${long_tps:-?} | ${pf_tps:-?} (${pf_tok:-?} tok) | ${cmm_max:-?} MiB |"
}

echo "| quant (GGUF) | decode t/s @~300 ctx | decode t/s @~2k ctx | prefill t/s (ladder) | peak card CMM |"
echo "|---|---|---|---|---|"
for g in "$@"; do
    [ -z "$g" ] && g=""
    bench_one "$(basename "$g" .gguf | sed 's/Qwen3.5-0.8B-//')" "$g"
done
