#!/usr/bin/env python3
"""Exact-match scale slot decode: for each scale entry, find the row r whose
M(r) reproduces the observed sdB bf16 value exactly (f32 mul + RNE bf16).
Then group index from base value ordering. Emits (m, r, g) -> word pos map.
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


def bf16_round_u16(f):
    u = np.float32(f).view(np.uint32)
    bits = u >> 16
    rem = u & 0xFFFF
    ru = (rem > 0x8000) | ((rem == 0x8000) & ((bits & 1) == 1))
    return np.uint16(bits + ru)


def main():
    base = load_tag('cm0')
    sdB = load_tag('sdB')
    ent = pickle.load(open('/tmp/scale_entries.pkl', 'rb'))
    wpos, mat, row0 = ent['wpos'], ent['mat'], ent['row']
    b16, B16 = base.view(np.uint16), sdB.view(np.uint16)
    bv = (b16.astype(np.uint32) << 16).view(np.float32)
    Bv = (B16.astype(np.uint32) << 16).view(np.float32)

    # candidate M values per row
    out = {}
    stats = {}
    for mi, m in enumerate(MATS):
        n = SHAPES[m][0]
        r = np.arange(n, dtype=np.float32)
        M = (2.0 ** (r // 128)) * (1.0 + (np.mod(r, 128)) / 128.0)
        Mf = M.astype(np.float32)
        sel = np.flatnonzero(mat == mi)
        rows = np.full(len(sel), -1, np.int64)
        for idx, w in enumerate(sel):
            target = B16[w]
            # predicted = bf16(f32(base_f32 * M(r))); vector over r
            prod = (bv[w] * Mf).astype(np.float32)
            u = prod.view(np.uint32)
            bits = u >> 16
            rem = u & 0xFFFF
            ru = (rem > 0x8000) | ((rem == 0x8000) & ((bits & 1) == 1))
            pred = (bits + ru).astype(np.uint16)
            hits = np.flatnonzero(pred == target)
            if len(hits) == 1:
                rows[idx] = hits[0]
            elif len(hits) > 1:
                # prefer the previously-ratio-decoded row if among hits
                if row0[idx] in hits:
                    rows[idx] = row0[idx]
                else:
                    rows[idx] = hits[0]  # ambiguity flagged by count below
        okr = rows >= 0
        ur, uc = np.unique(rows[okr], return_counts=True)
        stats[m] = (len(sel), int(okr.sum()), len(ur), dict(zip(ur.tolist(), uc.tolist())))
        print(f'{m}: entries={len(sel)} decoded={int(okr.sum())} uniq_rows={len(ur)}/{n}')
        bad = {r: c for r, c in zip(ur.tolist(), uc.tolist()) if c != NGRP[m]}
        if bad:
            few = sum(1 for c in bad.values() if c < NGRP[m])
            print(f'   rows with wrong entry count: {len(bad)} (of which under-filled: {few})')
        for idx, w in enumerate(sel):
            if rows[idx] >= 0:
                out[(m, int(rows[idx]), w)] = int(w)
    pickle.dump(out, open('/tmp/scale_rowpos.pkl', 'wb'))
    print('saved', len(out), 'entries')


if __name__ == '__main__':
    main()
