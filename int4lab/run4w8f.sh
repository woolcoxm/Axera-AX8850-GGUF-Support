#!/bin/bash
# Runner for Axera's obfuscated 4w8f converter (Pulsar2 7.0-patch1 lite package)
PKG=/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package
export LD_LIBRARY_PATH=$PKG/lib
export PYTHONHOME=$PKG/python3
export PYTHONPATH=$PKG/pulsar2/axnn:$PKG/pulsar2/axnn/axnn/tools:$PKG/pulsar2
export PATH=$PKG/bin:$PATH
exec $PKG/lib/ld-linux-x86-64.so.2 $PKG/python3/bin/python3 $PKG/pulsar2/axnn/tools/axcli/convert_onnx_to_4w8f.py "$@"
