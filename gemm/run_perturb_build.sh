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

/usr/bin/python3 $GEMM/mk_perturb.py /tmp/${TAG} ${VARIANT}
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
  --input_path /tmp/${TAG} --output_path /tmp/${TAG}_out \
  --hidden_state_type bf16 --kv_cache_len 256 --prefill_len 128 \
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
# content check: perturbation blobs differ from cm0 by < 100KB
import numpy as np, onnx as _o
from onnx import numpy_helper as _nh
def _np(p):
    mm = _o.load(p, load_external_data=False)
    for ii in mm.graph.initializer:
        if ii.name == 'npu_params':
            return _nh.to_array(ii).astype(np.uint8)[:17346568]
_cmf = sorted(glob.glob('/tmp/cmdumps/cm0_layer_*.onnx'))
if _cmf:
    _a, _b = _np(_cmf[0]), _np(out)
    _d = int((_a != _b).sum())
    print(f'{tag}: diff-vs-cm0 = {_d} bytes')
    import os as _os
    if _d > 100000 and _os.environ.get("WT") != "bf16":
        print(f'{tag}: CONTENT CHECK FAILED (expected small perturbation diff)')
        sys.exit(3)
EOF
ls -la /tmp/${TAG}_out/ || true
