#!/usr/bin/env python3
"""Diff amp-1.0 vs amp-0.5 marker layer blobs -> exact scale-table positions.

The amplitude marker (seed 20260825) used identical weight patterns in every
layer; even layers amp 1.0, odd 0.5. Weight BYTES are amplitude-invariant
(normalized quant) so the only per-layer differences should be scale entries
(and any reciprocals/derived tables).
"""
import glob
import zlib

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568


def load_blob(path):
    m = onnx.load(path, load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params':
            a = numpy_helper.to_array(init)
            return a.astype(np.uint8)[:BLOB], len(a)
    return None, 0


def main():
    files = sorted(glob.glob('/tmp/qam_dump_*.onnx'))
    print(f'{len(files)} dumps')
    blobs = {}
    sizes = {}
    for f in files:
        b, sz = load_blob(f)
        if b is None:
            print('no npu_params:', f)
            continue
        h = (zlib.crc32(b.tobytes()), sz)
        blobs.setdefault(h, []).append(f)
        sizes[f] = sz
    print(f'{len(blobs)} distinct (crc,size) groups:')
    for h, fs in blobs.items():
        print(f'  crc={h[0]:08x} size={h[1]:,} n={len(fs)} files={fs[:3]}')

    # pick a pair of equal-size groups with different crc
    groups = [g for g in blobs.items() if g[0][1] == BLOB]
    groups.sort(key=lambda kv: -len(kv[1]))
    if len(groups) < 2:
        print('need 2 distinct full-size layer groups')
        return
    (h1, fs1), (h2, fs2) = groups[0], groups[1]
    b1, _ = load_blob(fs1[0])
    b2, _ = load_blob(fs2[0])
    print('comparing', fs1[0], 'vs', fs2[0])
    diff = b1 != b2
    print('differing bytes:', int(diff.sum()))
    pos = np.flatnonzero(diff)
    if len(pos) == 0:
        return
    print('diff range:', pos.min(), '-', pos.max())
    d = np.diff(pos)
    big = np.flatnonzero(d > 64)
    # cluster boundaries
    starts = np.r_[0, big + 1]
    ends = np.r_[big + 1, len(pos)]
    lens = pos[ends - 1] - pos[starts] + 1
    print(f'{len(starts)} clusters, lens:', np.bincount(lens)[:40])


if __name__ == '__main__':
    main()
