#!/bin/bash
# Claims-decode marker series for llm_build2 s4 layout (x86 only, no card).
# NOTE: never export the package LD_LIBRARY_PATH globally - it breaks nested
# bash wrappers (glibc mismatch). runpulsar2.sh scopes its own env.
ulimit -c 0
R=/home/kram/Desktop/Projects/LLMTest/int4lab/runpulsar2.sh
S=/home/kram/Desktop/Projects/LLMTest/int4lab/scratch
PKG=/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package
mkdir -p $S/claims
for mode in 2 3 4 5 6 7 d2 mc mixamp; do
  echo "=== marker $mode ==="
  $R llm_build2 --input_path $S/mk_$mode --output_path ${S}/mk_${mode}_s4 \
      --max_context 64 -c 0 -w s4 --parallel 2 --chip AX650 > $S/mk_${mode}_build.log 2>&1 \
    && echo "build $mode OK" || echo "build $mode FAILED"
done
S=$S $PKG/lib/ld-linux-x86-64.so.2 $PKG/python3/bin/python3 /home/kram/Desktop/Projects/LLMTest/int4lab/extract_claims.py
echo ALL_DONE
