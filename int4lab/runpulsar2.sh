#!/bin/bash
PKG=/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package
if [ ! -e "$HOME/.hasplm/installed/32434/Unlocked_20230901_perpetual.v2c" ]; then
    mkdir -p "$HOME/.hasplm/installed/32434" && cp "$PKG/install/Unlocked_20230901_perpetual.v2c" "$HOME/.hasplm/installed/32434/" 2>/dev/null || true
fi
export LD_LIBRARY_PATH=$PKG/lib
export PYTHONHOME=$PKG/python3
export PYTHONPATH=$PKG/pulsar2/axnn:$PKG/pulsar2/axnn/axnn/tools:$PKG/pulsar2
export PATH=$PKG/bin:$PATH
ulimit -c 0
exec $PKG/lib/ld-linux-x86-64.so.2 $PKG/python3/bin/python3 $PKG/pulsar2/axnn/axnn/yamain/main.py "$@"
