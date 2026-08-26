#!/bin/bash
# Build one differential marker and collect its QuantAxModel dump.
# Usage: run_diff_build.sh <tag> <variant>
set -e
TAG=$1
VARIANT=$2
GEMM=/home/kram/Desktop/Projects/LLMTest/gemm
export PATH=/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package/bin:$PATH

# snapshot existing dumps so parallel runs never steal each other's files
BEFORE=$(ls /tmp/qam_dump_*.onnx 2>/dev/null || true)

case "$VARIANT" in
  cm*) GEN=mk_code_marker.py; VARG=$(echo $VARIANT | tr -d 'cm') ;;
  *)   GEN=mk_diff_marker.py; VARG=$VARIANT ;;
esac
/usr/bin/python3 $GEMM/$GEN /tmp/${TAG} ${VARG}
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path /tmp/${TAG} --output_path /tmp/${TAG}_out \
  --hidden_state_type bf16 --kv_cache_len ${CTX:-256} --prefill_len 128 \
  --last_kv_cache_len 128 --chip AX650 -c 0 --parallel 8 -w ${WT:-s8} \
  > /tmp/${TAG}.log 2>&1

/usr/bin/python3 - $TAG "$BEFORE" <<'EOF'
import glob
import os
import shutil
import sys
import onnx

tag, before = sys.argv[1], set(sys.argv[2].split())
cands = []
for f in sorted(glob.glob('/tmp/qam_dump_*.onnx')):
    if f in before:
        continue
    try:
        m = onnx.load(f, load_external_data=False)
    except Exception:
        continue
    for init in m.graph.initializer:
        if init.name == 'npu_params' and 17000000 < len(init.raw_data):
            cands.append((os.path.getmtime(f), f))
if not cands:
    print(f'{tag}: NO LAYER DUMP FOUND')
    sys.exit(0)
cands.sort()
# archive all new layer dumps for manual association, use the newest
for mt, f in cands:
    idx = cands.index((mt, f))
    shutil.copy(f, f'/tmp/cmdumps/{tag}_cand{idx}_{os.path.basename(f)}')
mt, f = cands[-1]
out = f'/tmp/cmdumps/{tag}_layer_dump.onnx'
shutil.copy(f, out)
print(f'{tag}: {len(cands)} layer dumps seen, newest ({f}) -> {out}')
EOF
ls -la /tmp/${TAG}_out/ || true
