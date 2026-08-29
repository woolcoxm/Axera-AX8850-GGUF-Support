#!/bin/sh
# build.sh — build card_reboot ON THE PI (aarch64). The axcl host libs are
# arm64; cross-building elsewhere is not worth the trouble for one file.
#
# usage: ./build.sh [path-to-axcl-include]   (default: ../vendor/axcl-include)
set -e
cd "$(dirname "$0")"
INC="${1:-../vendor/axcl-include}"
[ -d "$INC" ] || INC=/usr/include/axcl
cc -O2 -Wall -o card_reboot card_reboot.c -I"$INC" \
   -L/usr/lib/axcl -laxcl_rt -Wl,-rpath,/usr/lib/axcl
echo "built: $(pwd)/card_reboot"
echo "install (optional): sudo install -m755 card_reboot /usr/local/sbin/"
