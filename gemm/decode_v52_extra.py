#!/usr/bin/env python3
"""Decode the 8-bit weight positions and k=0 anchor columns (5.2 markers).

8-bit positions: cm0 != cmd2 (dither-dependent), hi nibble = code+8 stable
across builds, lo = dither (validated against both dither families).
byte = 128 + 16*code + dither (offset-binary int8 slot).
Anchors: nibble slots with value 15 in every build (V[:,0]=127 -> q8=127).
Anchor->row assignment by modal row of nearby same-matrix claims.
Merges into v52_claims.npz as separate arrays (eight_*, anchor_*).
"""
import numpy as np
from pathlib import Path
from collections import Counter

DIR = Path(__file__).resolve().parent / 'baked' / 'v52_markers'
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
MATS = list(SHAPES)
SEEDS = {m: 101 + i for i, m in enumerate(MATS)}
SEEDS2 = {m: 151 + i for i, m in enumerate(MATS)}
TAGS = ['cm0', 'cm1', 'cmB2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']
BLOB = 19_212_296
CDT = np.dtype([('mat', np.uint8), ('r', np.uint32), ('k', np.uint32),
                ('pos', np.uint32), ('half', np.uint8)])


def load(t):
    return np.frombuffer((DIR / f'v52_{t}_l0_params.bin').read_bytes(), np.uint8)


def main():
    B = {t: load(t) for t in TAGS + ['cmd2', 'cmmc']}
    d1 = {m: np.random.default_rng(SEEDS[m]).integers(0, 13, SHAPES[m]).reshape(-1) for m in MATS}
    d2 = {m: np.random.default_rng(SEEDS2[m]).integers(0, 13, SHAPES[m]).reshape(-1) for m in MATS}

    # --- 8-bit positions
    cand = (B['cm0'] != B['cmd2']) & (B['cm0'] >= 128) & (B['cm0'] < 256)
    keys = np.zeros(BLOB, np.int32)
    ok = cand.copy()
    for i, t in enumerate(TAGS):
        hi = (B[t] >> 4).astype(np.int32) - 8
        ok &= (hi >= 0) & (hi <= 7)
        keys |= hi << (3 * i)
    mc = (B['cmmc'] >> 4).astype(np.int32) - 8
    ok &= (mc >= 0) & (mc <= 6)
    dith_lo0 = (B['cm0'] & 15).astype(np.int32)
    dith_lo2 = (B['cmd2'] & 15).astype(np.int32)
    print(f'8-bit candidates: {int(cand.sum()):,}; code-consistent: {int(ok.sum()):,}')

    out8 = []
    for mi, m in enumerate(MATS):
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        sel = np.flatnonzero(ok & (mc == mi))
        if not len(sel):
            print(f'{m:5s}: 0')
            continue
        kv = keys[sel]
        r = (kv & ((1 << rnb) - 1)).astype(np.int64)
        k = ((kv >> rnb) & ((1 << knb) - 1)).astype(np.int64)
        good = ((kv >> (rnb + knb)) == 0) & (r < n) & (k < kk)
        sel = sel[good]; r = r[good]; k = k[good]
        flat = r * kk + k
        good = (dith_lo0[sel] == d1[m][flat]) & (dith_lo2[sel] == d2[m][flat])
        sel = sel[good]; r = r[good]; k = k[good]
        arr = np.empty(len(sel), CDT)
        arr['mat'] = mi; arr['r'] = r; arr['k'] = k
        arr['pos'] = sel.astype(np.uint32); arr['half'] = 2  # kind=8bit
        out8.append(arr)
        print(f'{m:5s}: 8-bit positions {len(sel):,}')
    eight = np.concatenate(out8) if out8 else np.empty(0, CDT)
    print(f'total 8-bit: {len(eight):,}')

    # --- anchor columns (all builds nibble 15, byte-stable cm0==cmd2)
    same02 = B['cm0'] == B['cmd2']
    anch = same02.copy()
    for t in TAGS + ['cmmc']:
        anch &= (B[t] >> 4) == 15
        anch &= (B[t] & 15) == 15
    print(f'all-15 stable bytes: {int(anch.sum()):,} (expect ~13,312 + few all-ones coords)')
    apos = np.flatnonzero(anch)

    # assign rows by neighborhood modality
    claims = np.load(DIR / 'v52_claims.npz')['claims']
    cpos = claims['pos'].astype(np.int64)
    order = np.argsort(cpos)
    cpos_s = cpos[order]
    cmat = claims['mat'][order]; cr = claims['r'][order]
    anchors = []
    W_IN = 144
    for p in apos:
        lo = np.searchsorted(cpos_s, p - W_IN)
        hi = np.searchsorted(cpos_s, p + W_IN)
        if hi <= lo:
            anchors.append((7, 0xFFFFFFFF, int(p)))   # unassigned
            continue
        cnt = Counter(cr[lo:hi].tolist())
        row, best = cnt.most_common(1)[0]
        anchors.append((0, row, int(p)))
    # build anchor claims: (m, r, k=0) for the modal row's matrix
    a_out = []
    used = Counter()
    for _, row, p in anchors:
        if row == 0xFFFFFFFF:
            continue
        # matrix from neighbors
        lo = np.searchsorted(cpos_s, p - W_IN); hi = np.searchsorted(cpos_s, p + W_IN)
        if hi <= lo:
            continue
        mm = Counter(cmat[lo:hi].tolist()).most_common(1)[0][0]
        used[(mm, row)] += 1
        a_out.append((mm, row, 0, p))
    arr = np.empty(len(a_out), CDT)
    for i, (mm, rr, kk_, pp) in enumerate(a_out):
        arr[i] = (mm, rr, kk_, pp, 3)
    print(f'anchors assigned: {len(a_out):,}; distinct (m,row): {len(used):,}; '
          f'dupes: {sum(1 for v in used.values() if v>1)}')
    for mi, m in enumerate(MATS):
        print(f'  {m:5s}: rows covered {len(set(r for (mm, r) in used if mm==mi))}/{SHAPES[m][0]}')
    np.savez_compressed(DIR / 'v52_extra_claims.npz', eight=eight, anchors=arr)
    print('saved', DIR / 'v52_extra_claims.npz')


if __name__ == '__main__':
    main()
