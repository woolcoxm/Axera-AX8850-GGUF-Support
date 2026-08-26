#!/usr/bin/env python3
"""Definitive scale-table decode using the [bf16 s][i16 v] 4B entry structure.

Scale entries live in the weight region as 4-byte entries: LE u16 = bf16
scale, next u16 = constant-per-matrix value (516 for k). Find all entries by
the sdB diff (every M(r) != 1 row changes), decode row from the value ratio,
matrix from sdA ratio / cluster grouping, kgroup from intra-row ordering.
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
EXP_SLOTS = {m: SHAPES[m][0] * NGRP[m] for m in MATS}


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


def bf16_f(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    base = load_tag('cm0')
    sdA = load_tag('sdA')
    sdB = load_tag('sdB')
    b16, A16, B16 = base.view(np.uint16), sdA.view(np.uint16), sdB.view(np.uint16)
    bv, Av, Bv = bf16_f(b16), bf16_f(A16), bf16_f(B16)

    # candidate entries: 4B stride, bf16 word W even offset, partner word W+1
    # Use the sdB diff to find them, then extend by structure (partner const).
    diffB_words = b16 != B16
    wpos = np.flatnonzero(diffB_words)
    print('changed words:', len(wpos))

    # For each changed word, try [i16 partner][bf16 scale] with partner at
    # w-1 (observed layout: bytes [04 02][01 3c] = i16 516, bf16 0x3C01)
    partner = b16[np.clip(wpos - 1, 0, len(b16) - 1)].astype(np.int32)
    val = bv[wpos]
    plausible = (partner > 0) & (partner < 8192) & (val > 1e-7) & (val < 100)
    wpos = wpos[plausible]
    print('structured scale-like words:', len(wpos), 'expected ~61440')

    # i16 partner values histogram
    pv, pc = np.unique(partner[plausible], return_counts=True)
    top = np.argsort(-pc)[:10]
    print('partner i16 values:', [(int(pv[t]), int(pc[t])) for t in top])

    # ratio decode: sdB/base
    with np.errstate(all='ignore'):
        rB = Bv[wpos] / bv[wpos]
        rA = Av[wpos] / bv[wpos]
    lg = np.log2(rB)
    e = np.floor(lg + 1e-9)
    frac = lg - e
    mrow = np.rint(frac * 128)
    row = np.where((e >= 0) & (e < 64) & (mrow >= 0) & (mrow <= 127),
                   e * 128 + mrow, -1).astype(np.int64)

    # matrix from sdA ratio: 2^mid, or ratio==1 -> q or row0
    lgA = np.log2(rA)
    mat = np.full(len(wpos), -1, np.int8)
    for mi, m in enumerate(MATS):
        hit = np.isclose(lgA, mi, atol=0.01)
        mat[hit] = mi
    print('matrix assignment from sdA:', np.bincount(mat[mat >= 0], minlength=8)[:8])
    un = mat < 0
    print('unassigned (expect q + row-0 rows):', int(un.sum()), '(q needs', EXP_SLOTS['q'], ')')

    # for assigned: rows valid?
    for mi, m in enumerate(MATS):
        sel = mat == mi
        if sel.sum() == 0:
            continue
        rows = row[sel]
        okr = (rows >= 0) & (rows < SHAPES[m][0])
        ur = np.unique(rows[okr])
        print(f'{m}: slots={int(sel.sum())}/{EXP_SLOTS[m]} rows-ok={int(okr.sum())} uniq-rows={len(ur)}/{SHAPES[m][0]}',
              'slots/row:', np.bincount(rows[okr], minlength=SHAPES[m][0])[:8], '...')

    pickle.dump({'wpos': wpos, 'row': row, 'mat': mat}, open('/tmp/scale_entries.pkl', 'wb'))
    print('saved /tmp/scale_entries.pkl')


if __name__ == '__main__':
    main()
