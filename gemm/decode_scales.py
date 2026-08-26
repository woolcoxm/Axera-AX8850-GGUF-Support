#!/usr/bin/env python3
"""Full scale-table decode: (matrix, row, kgroup) -> blob offset + formula.

sdA ratios (2^mid) give matrix identity per slot; sdB ratios give M(r) ->
row identity. q slots = diffB & ~diffA. Base cm0 values give the constant.
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
MID = {m: i for i, m in enumerate(MATS)}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}


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


def bf16_words(b):
    return b.view(np.uint16)


def bf16_f(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    base = load_tag('cm0')
    sdA = load_tag('sdA')
    sdB = load_tag('sdB')
    b16, A16, B16 = bf16_words(base), bf16_words(sdA), bf16_words(sdB)
    bv, Av, Bv = bf16_f(b16), bf16_f(A16), bf16_f(B16)

    diffA = b16 != A16
    diffB = b16 != B16
    print(f'diffA words={int(diffA.sum())} diffB words={int(diffB.sum())}')

    # slot matrix id from sdA ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        rA = np.where(bv != 0, Av / bv, np.nan)
    mat_of = np.full(len(b16), -1, np.int8)
    for m in MATS:
        if m == 'q':
            continue
        want = 2.0 ** MID[m]
        hit = diffA & np.isclose(rA, want, rtol=1e-3)
        mat_of[hit] = MATS.index(m)
        print(f'{m}: ratio 2^{MID[m]} slots={int(hit.sum())}')
    # q = diffB & ~diffA & plausible scale value & not already assigned
    cand_q = diffB & ~diffA & (mat_of < 0)
    print(f'q candidates (diffB-only): {int(cand_q.sum())}')

    # row decode from sdB ratio: ratio = M(r) = 2^(r>>7)*(1+(r&127)/128)
    with np.errstate(divide='ignore', invalid='ignore'):
        rB = np.where(bv != 0, Bv / bv, np.nan)
    lg = np.log2(np.where(np.isfinite(rB) & (rB > 0), rB, np.nan))
    e = np.floor(lg + 1e-9).astype(int)
    frac = lg - e
    mrow = np.rint(frac * 128).astype(int)
    row = np.where((e >= 0) & (mrow >= 0) & (mrow <= 127), e * 128 + mrow, -1)

    # report: for each matrix, rows decoded, kgroup slots per row
    for mi, m in enumerate(MATS):
        if m == 'q':
            slots = cand_q
        else:
            slots = mat_of == mi
        w = np.flatnonzero(slots)
        if len(w) == 0:
            continue
        rows = row[w]
        ok = rows >= 0
        print(f'== {m}: {len(w)} slots, rows decoded for {int(ok.sum())} ==')
        if ok.sum():
            ur, uc = np.unique(rows[ok], return_counts=True)
            print(f'  unique rows={len(ur)} counts[min/med/max]={uc.min()}/{int(np.median(uc))}/{uc.max()} row range={ur.min()}..{ur.max()} (n={SHAPES[m][0]})')
            bad = ur[ur >= SHAPES[m][0]]
            if len(bad):
                print(f'  OUT-OF-RANGE rows: {len(bad)}')
            # values: base bf16 for this matrix's slots
            vals = bv[w]
            print(f'  base value max={bv[w].max():.6f} min={bv[w].min():.6e}')
            top = np.argsort(-np.abs(bv[w]))[:3]
            print('  top values:', bv[w][top], 'at words', w[top])
            # group structure per row: value ordering within a row
            r0 = ur[0]
            wsel = w[rows == r0]
            print(f'  row {r0}: words={wsel[:16]} values={bv[wsel][:16]}')
    # int16 partner: word after each slot
    allw = np.flatnonzero(diffA | cand_q)
    nxt = b16[allw + 1] if len(allw) else np.array([], np.uint16)
    print('partner u16 word values:', np.unique(nxt)[:10], 'counts:', len(nxt))
    # entry stride check: are slots 2 words (4 bytes) apart mostly?
    wA = np.flatnonzero(diffA)
    dw = np.diff(wA)
    vals, cnts = np.unique(dw, return_counts=True)
    top = np.argsort(-cnts)[:6]
    print('word-gaps top:', [(int(vals[t]), int(cnts[t])) for t in top])

    np.savez('/tmp/scale_slots.npz', mat_of=mat_of, cand_q=cand_q, row=row,
             diffA=diffA, diffB=diffB)
    print('saved /tmp/scale_slots.npz')


if __name__ == '__main__':
    main()
