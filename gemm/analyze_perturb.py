#!/usr/bin/env python3
"""Analyze single-group perturbation diffs: where do a group's elements live?

pq_100_0: q row 100 kgroup 0 (k=0..255) perturbed by +7/127 on V.
Diff positions vs cm0 = every byte that depends on those elements.
"""
import glob
import pickle
import sys

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568


def load(p):
    m = onnx.load(p, load_external_data=False)
    for i in m.graph.initializer:
        if i.name == 'npu_params' and len(i.raw_data) > 1000000:
            return numpy_helper.to_array(i).astype(np.uint8)[:BLOB]
    raise RuntimeError(p)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'pq_100_0'
    mat, row, grp = tag.split('_')[0][1:], int(tag.split('_')[1]), int(tag.split('_')[2])
    base = load(sorted(glob.glob('/tmp/cmdumps/cm0_layer_*.onnx'))[0])
    pert = load(f'/tmp/cmdumps/{tag}_layer_dump.onnx')
    diff = np.flatnonzero(base != pert)
    print(f'{tag}: {len(diff)} differing bytes, range {diff.min()}..{diff.max()}')

    tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
    v3pos = {}
    for (m, r, k), (p, half) in tab.items():
        if m == mat and r == row and grp * 256 <= k < (grp + 1) * 256:
            v3pos[k] = p  # first-claim only
    print(f'v3 table positions for this group: {len(v3pos)}')

    dset = set(diff.tolist())
    in_v3 = sum(1 for k, p in v3pos.items() if p in dset)
    print(f'diff positions that are v3 positions: {in_v3}/{len(v3pos)}')
    extra = [p for p in diff.tolist() if p not in set(v3pos.values())]
    print(f'EXTRA diff positions (other copies / scales / derived): {len(extra)}')
    extra_a = np.array(sorted(extra))
    # cluster the extras
    if len(extra_a):
        d = np.diff(extra_a)
        cuts = np.flatnonzero(d > 8)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts + 1, len(extra_a)]
        print(f'{len(starts)} clusters of extras:')
        for s, e in list(zip(starts, ends))[:12]:
            p0 = extra_a[s]
            print(f'  @{p0} len={extra_a[e-1]-p0+1} n={e-s}',
                  'base:', [int(x) for x in base[p0:p0+8]],
                  'pert:', [int(x) for x in pert[p0:p0+8]])
    # scale entries for this (row, group)?
    # values changed at 4B-aligned [i16][bf16] slots?
    b16 = base.view(np.uint16)
    p16 = pert.view(np.uint16)
    wch = np.flatnonzero(b16 != p16)
    print(f'changed u16 words: {len(wch)}')
    for w in wch[:20]:
        bv = (np.uint32(b16[w]) << 16).view if False else np.array([np.uint32(b16[w]) << 16], np.uint32).view(np.float32)[0]
        pv = np.array([np.uint32(p16[w]) << 16], np.uint32).view(np.float32)[0]
        print(f'  word {w} (byte {2*w}): bf16 {bv:.6e} -> {pv:.6e}  (x{pv/bv if bv else 0:.4f})')


if __name__ == '__main__':
    main()
