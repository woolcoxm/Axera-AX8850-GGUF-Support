#!/usr/bin/env python3
"""Complete scale-table decode.

Entries structurally: [u16 516][bf16 scale] at even byte offsets (4B stride).
Matrix id: sdA value ratio = 2^mid (M=2^mid per matrix).
Row: sdc1/2/3 exponent deltas = row bits [0:4], [4:8], [8:12] (M=2^bits).
Group index: within a row, the entry with bf16 == bf16(1/127)... in markers
group 0 carries the k=0 anchor; other groups ordered by position.
Emits: scale_slots.pkl {(m, r, g): word_index}
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
NGRP = {m: SHAPES[m][1] // 256 for m in MATS}
EXP = {m: SHAPES[m][0] * NGRP[m] for m in MATS}


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


def f32_of(u16):
    return np.array([np.uint32(u16) << 16], np.uint32).view(np.float32)


def main():
    base = load_tag('cm0')
    A = load_tag('sdA')
    C1 = load_tag('sdc1')
    C2 = load_tag('sdc2')
    C3 = load_tag('sdc3')
    b16 = base.view(np.uint16)
    A16, c16_1, c16_2, c16_3 = A.view(np.uint16), C1.view(np.uint16), C2.view(np.uint16), C3.view(np.uint16)

    # structural entry scan: word w-1 == 516, word w = plausible scale
    n = len(b16) - 1
    cand = np.flatnonzero(b16[:n] == 516) + 1  # scale word index
    val = (b16[cand].astype(np.uint32) << 16).view(np.float32)
    exp = (b16[cand] >> 7) & 0xFF
    plausible = (val > 1e-7) & (val < 100) & (exp > 80) & (exp < 142)
    cand = cand[plausible]
    print('structural entries:', len(cand), '(expect', sum(EXP.values()), ')')

    # matrix from sdA: exponent delta (exact bit arithmetic)
    def exp_delta(other):
        d = ((other[cand] >> 7) & 0xFF).astype(int) - ((b16[cand] >> 7) & 0xFF).astype(int)
        return d
    dA = exp_delta(A16)
    mat = np.full(len(cand), -1, np.int8)
    for mi, m in enumerate(MATS):
        mat[dA == mi] = mi  # q (mi=0): delta 0 — ambiguous with unchanged
    # separate q from unchanged-by-A via sdc diffs: q rows with r&15==0 have
    # M=1 in sdc1 -> unchanged in ALL builds; assign leftover by count later
    print('sdA matrix assign counts:', np.bincount(mat[mat >= 0], minlength=8)[:8])

    # row bits from sdc exponent deltas
    d1 = exp_delta(c16_1)
    d2 = exp_delta(c16_2)
    d3 = exp_delta(c16_3)
    row = (d1 & 15) | ((d2 & 15) << 4) | ((d3 & 15) << 8)

    out = {}
    counts = {}
    for mi, m in enumerate(MATS):
        if m == 'q':
            continue
        sel = np.flatnonzero(mat == mi)
        rows = row[sel]
        ok = (rows >= 0) & (rows < SHAPES[m][0])
        ur, uc = np.unique(rows[ok], return_counts=True)
        full = np.sum(uc == NGRP[m])
        print(f'{m}: entries={len(sel)}/{EXP[m]} rows-ok={int(ok.sum())} '
              f'uniq={len(ur)}/{SHAPES[m][0]} full-rows={full}')
        bad = dict(zip(ur[uc != NGRP[m]].tolist(), uc[uc != NGRP[m]].tolist()))
        if bad:
            print(f'   partial rows: {len(bad)} e.g. {list(bad.items())[:6]}')
    pickle.dump({'cand': cand, 'mat': mat, 'row': row,
                 'd1': d1, 'd2': d2, 'd3': d3}, open('/tmp/scale_full.pkl', 'wb'))
    print('saved /tmp/scale_full.pkl')


if __name__ == '__main__':
    main()
