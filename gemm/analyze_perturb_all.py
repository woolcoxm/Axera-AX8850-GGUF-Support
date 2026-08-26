#!/usr/bin/env python3
"""Batch perturbation analysis: for each +7 probe, classify diff positions
against the v3 table claims for the perturbed group AND against claims for
OTHER (row, group) combos — finds the true mapping permutation.
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


PROBES = [
    ('q', 100, 0, 'pq16_100_0'),
    ('q', 100, 1, 'pq16_100_1'),
    ('q', 2000, 0, 'pq16_2000_0'),
    ('k', 500, 2, 'pk16_500_2'),
    ('q', 100, 0, 'pq_100_0'),
    ('q', 100, 1, 'pq_100_1'),
    ('q', 2000, 3, 'pq_2000_3'),
    ('k', 500, 2, 'pk_500_2'),
    ('o', 700, 7, 'po_700_7'),
    ('gate', 1500, 1, 'pgate_1500_1'),
    ('down', 300, 11, 'pdown_300_11'),
]


def main():
    base = load(sorted(glob.glob('/tmp/cmdumps/cm0_layer_*.onnx'))[0])
    print('loading v3...')
    tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
    # position -> (m, r, k) for claims (first-claim only)
    claim = {}
    for (m, r, k), (p, half) in tab.items():
        claim.setdefault(p, (m, r, k))
    # per (m, r, kgroup) -> sorted positions
    grp_pos = {}
    for (m, r, k), (p, half) in tab.items():
        g = k // 256
        grp_pos.setdefault((m, r, g), []).append(p)

    for mat, row, grp, tag in PROBES:
        try:
            pert = load(f'/tmp/cmdumps/{tag}_layer_dump.onnx')
        except Exception:
            continue
        diff = set(np.flatnonzero(base != pert).tolist())
        # claims of the perturbed group
        own = grp_pos.get((mat, row, grp), [])
        hit = sum(1 for p in own if p in diff)
        # which groups DO the diff positions claim?
        from collections import Counter
        c = Counter()
        for p in diff:
            if p in claim:
                m2, r2, k2 = claim[p]
                c[(m2, r2, k2 // 256)] += 1
        top = c.most_common(6)
        print(f'{tag}: {len(diff)} diff bytes | own-claims hit {hit}/{len(own)} | top claimed groups: {top}')
    print()
    print('NOTE: if own-claims ~0 but one claimed group dominates, the table')
    print('row/group mapping is permuted: the dominant claim = true location.')


if __name__ == '__main__':
    main()
