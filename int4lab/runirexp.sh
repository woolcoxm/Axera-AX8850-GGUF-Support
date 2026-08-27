#!/bin/bash
PKG=/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package
export LD_LIBRARY_PATH=$PKG/lib PYTHONHOME=$PKG/python3
export PYTHONPATH=$PKG/pulsar2/axnn:$PKG/pulsar2/axnn/axnn/tools:$PKG/pulsar2
exec $PKG/lib/ld-linux-x86-64.so.2 $PKG/python3/bin/python3 $PKG/pulsar2/ir_exporter "$@"
