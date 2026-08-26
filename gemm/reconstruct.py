#!/usr/bin/env python3
"""Reconstruction test: predict sdA blob from cm0 + decoded tables.

Known so far:
- weight nibbles are M-invariant (normalized per-(row,kgroup) quant)
- scale entries: [i16 516][bf16 s] at 4B stride; sdA should carry
  bf16(round_f32(base * 2^mid))

Residual diff = structures still unexplained (meta region, region-0 tables).
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']


def load(path):
    m = onnx.load(path, load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params' and len(init.raw_data) > 1000000:
            return numpy_helper.to_array(init).astype(np.uint8)[:BLOB]


def load_tag(tag):
    for pat in (f'/tmp/cmdumps/{tag}_layer_dump.onnx', f'/tmp/cmdumps/{tag}_layer_*.onnx'):
        fs = sorted(glob.glob(pat))
        if fs:
            return load(fs[0])
    raise RuntimeError(tag)


def bf16_round(f32):
    u = f32.astype(np.float32).view(np.uint32)
    bits = u >> 16
    rem = u & 0xFFFF
    round_up = (rem > 0x8000) | ((rem == 0x8000) & ((bits & 1) == 1))
    return (bits + round_up).astype(np.uint16)


def main():
    base = load_tag('cm0')
    sdA = load_tag('sdA')
    ent = pickle.load(open('/tmp/scale_entries.pkl', 'rb'))
    wpos, mat = ent['wpos'], ent['mat']

    pred = base.copy()
    b16 = pred.view(np.uint16)
    ok = 0
    bad = []
    for mi, m in enumerate(MATS):
        sel = np.flatnonzero(mat == mi)
        for w in wpos[sel]:
            v = np.uint32(b16[w]) << 16
            f = np.array([v], np.uint32).view(np.float32)[0]
            nb = bf16_round(np.float32(f * (2.0 ** mi)))
            b16[w] = nb
            actual = sdA.view(np.uint16)[w]
            if nb == actual:
                ok += 1
            else:
                bad.append((int(w), int(b16[w]), int(nb), int(actual)))
    print(f'scale words: {ok} exact, {len(bad)} mismatched of {len(wpos)}')
    for b in bad[:10]:
        print('  mismatch word', b)

    # full residual
    resid = np.flatnonzero(pred != sdA)
    print(f'residual differing bytes: {len(resid)} of {int((base != sdA).sum())} total')
    # cluster the residual
    if len(resid):
        d = np.diff(resid)
        cuts = np.flatnonzero(d > 8)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts + 1, len(resid)]
        print(f'{len(starts)} residual clusters:')
        lens = resid[ends - 1] - resid[starts] + 1
        import collections
        c = collections.Counter(lens.tolist())
        print('  lengths:', c.most_common(10))
        for s, e in list(zip(starts, ends))[:5]:
            p0 = resid[s]
            print(f'  cluster @{p0}: len={resid[e-1]-p0+1} first bytes',
                  [int(x) for x in base[p0:p0+8]], '->', [int(x) for x in sdA[p0:p0+8]])
        np.save('/tmp/resid_pos.npy', resid)


if __name__ == '__main__':
    main()
