#!/bin/bash
P52=/home/kram/Desktop/Projects/LLMTest/pulsar2/5.2/5.2/ax_pulsar2_5.2_lite_package
export LD_LIBRARY_PATH=$P52/lib
export PYTHONHOME=$P52/python3
export PYTHONPATH=$P52/pulsar2/tools:$P52/pulsar2
exec $P52/lib/ld-linux-x86-64.so.2 $P52/python3/bin/python3 $P52/pulsar2/tools/axcli/convert_onnx_to_4w8f.py "$@"
