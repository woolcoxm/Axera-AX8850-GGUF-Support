#!/usr/bin/env python3
"""Memory-frugal decode of the 5.2 npu_params layout from code markers.

Claims stored as numpy structured arrays (no Python dicts):
  mat u8, r u32, k u32, pos u32, half u8
Fine plane (low nibbles of q8) mapped by lag discovery + dither verification.
Inputs: gemm/baked/v52_markers/{cm0,cm1,cmB2,cm3..cm7,cmd2,cmmc}_l0_params.bin
Outputs (same dir): v52_claims.npz, v52_fine.npz   Peak RSS target: < 2 GB.
"""
import collections
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent / 'baked' / 'v52_markers'
SHAPES = {
    'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
    'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072),
}
MATS = list(SHAPES)
SEEDS = {m: 101 + i for i, m in enumerate(MATS)}
SEEDS2 = {m: 151 + i for i, m in enumerate(MATS)}
TAGS = ['cm0', 'cm1', 'cmB2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']
BLOB = 19_212_296
CLAIM_DTYPE = np.dtype([('mat', np.uint8), ('r', np.uint32), ('k', np.uint32),
                        ('pos', np.uint32), ('half', np.uint8)])


def load(tag):
    a = np.frombuffer((DIR / f'v52_{tag}_l0_params.bin').read_bytes(), np.uint8)
    assert len(a) == BLOB, f'{tag}: {len(a)}'
    return a


def main():
    B = {t: load(t) for t in TAGS + ['cmd2', 'cmmc']}
    print('blobs loaded')

    same02 = B['cm0'] == B['cmd2']
    nib_hi = same02.copy()
    nib_lo = same02.copy()
    for t in TAGS:
        nib_hi &= (B[t] >> 4) >= 8
        nib_lo &= (B[t] & 15) >= 8
    mc_hi = (B['cmmc'] >> 4).astype(np.int16) - 8
    mc_lo = (B['cmmc'] & 15).astype(np.int16) - 8
    nib_hi &= (mc_hi >= 0) & (mc_hi <= 6)
    nib_lo &= (mc_lo >= 0) & (mc_lo <= 6)
    print(f'nibble slots: hi {int(nib_hi.sum()):,} lo {int(nib_lo.sum()):,}')

    def keys_for(half):
        keys = np.zeros(BLOB, np.int32)
        for i, t in enumerate(TAGS):
            v = (B[t] >> 4) if half else (B[t] & 15)
            keys |= (v.astype(np.int32) - 8) << (3 * i)
        return keys
    hi_keys = keys_for(True)
    lo_keys = keys_for(False)
    del B['cmmc']

    out = []
    for mi, m in enumerate(MATS):
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        for half, keys, slots, matc in ((1, hi_keys, nib_hi, mc_hi),
                                        (0, lo_keys, nib_lo, mc_lo)):
            sel = np.flatnonzero(slots & (matc == mi))
            if not len(sel):
                continue
            kv = keys[sel]
            r = (kv & ((1 << rnb) - 1)).astype(np.uint32)
            k = ((kv >> rnb) & ((1 << knb) - 1)).astype(np.uint32)
            ok = ((kv >> (rnb + knb)) == 0) & (r < n) & (k < kk)
            sel = sel[ok]; r = r[ok]; k = k[ok]
            arr = np.empty(len(sel), CLAIM_DTYPE)
            arr['mat'] = mi; arr['r'] = r; arr['k'] = k
            arr['pos'] = sel.astype(np.uint32); arr['half'] = half
            out.append(arr)
            del kv
    claims = np.concatenate(out)
    # drop the (m, 0, 0) degenerate keys: all-zero codes land there (the 7
    # genuine (m,0,0) elements are k=0 anchors anyway — recovered by elimination)
    claims = claims[(claims['r'] + claims['k']) > 0]
    order = np.lexsort((claims['half'], claims['pos'], claims['k'],
                        claims['r'], claims['mat']))
    claims = claims[order]
    same = (claims['mat'][1:] == claims['mat'][:-1]) & (claims['r'][1:] == claims['r'][:-1]) \
        & (claims['k'][1:] == claims['k'][:-1]) & (claims['pos'][1:] == claims['pos'][:-1]) \
        & (claims['half'][1:] == claims['half'][:-1])
    claims = claims[np.concatenate(([True], ~same))]
    print(f'claims: {len(claims):,} entries')
    for mi, m in enumerate(MATS):
        sel = claims['mat'] == mi
        els = len(np.unique(claims['r'][sel].astype(np.int64) * SHAPES[m][1]
                            + claims['k'][sel]))
        print(f'  {m:5s}: {int(sel.sum()):,} claims, {els:,}/{SHAPES[m][0]*SHAPES[m][1]} elements')
    np.savez_compressed(DIR / 'v52_claims.npz', claims=claims)
    print('saved', DIR / 'v52_claims.npz')

    # ---- coarse pairs: (m, r, k even) at lo half and (m, r, k+1) at hi, same byte
    lo_cl = claims[claims['half'] == 0]
    hi_cl = claims[claims['half'] == 1]

    def jkey(a):
        return (a['mat'].astype(np.uint64) << 48) | (a['r'].astype(np.uint64) << 32) \
            | ((a['k'] >> 1).astype(np.uint64) << 16) | a['pos']
    jhi = jkey(hi_cl)
    o2 = np.argsort(jhi)
    hi_s = hi_cl[o2]; jhi_s = jhi[o2]
    idx = np.clip(np.searchsorted(jhi_s, jkey(lo_cl)), 0, len(jhi_s) - 1)
    match = (jhi_s[idx] == jkey(lo_cl)) & (hi_s['k'][idx] == lo_cl['k'] + 1)
    pairs_lo, pairs_hi = lo_cl[match], hi_s[idx][match]
    print(f'coarse pairs: {len(pairs_lo):,}')

    # ---- fine plane: lag discovery on a sample, then full assignment
    fine = (~same02) & ((B['cm0'] & 15) <= 12) & ((B['cmd2'] & 15) <= 12) \
           & ((B['cm0'] >> 4) <= 12) & ((B['cmd2'] >> 4) <= 12)
    print(f'fine-plane candidate bytes: {int(fine.sum()):,}')

    d1f = {m: np.random.default_rng(SEEDS[m]).integers(0, 13, SHAPES[m]).reshape(-1)
           for m in MATS}
    d2f = {m: np.random.default_rng(SEEDS2[m]).integers(0, 13, SHAPES[m]).reshape(-1)
           for m in MATS}

    def want_for(pl, ph):
        want = np.empty(len(pl), np.uint8)
        want2 = np.empty(len(pl), np.uint8)
        for mi, m in enumerate(MATS):
            sel = pl['mat'] == mi
            if not sel.any():
                continue
            fl = pl['r'][sel].astype(np.int64) * SHAPES[m][1] + pl['k'][sel]
            fh = ph['r'][sel].astype(np.int64) * SHAPES[m][1] + ph['k'][sel]
            want[sel] = ((d1f[m][fh].astype(np.uint16) << 4) | d1f[m][fl]).astype(np.uint8)
            want2[sel] = ((d2f[m][fh].astype(np.uint16) << 4) | d2f[m][fl]).astype(np.uint8)
        return want, want2

    rng = np.random.default_rng(0)
    samp = rng.choice(len(pairs_lo), min(120_000, len(pairs_lo)), replace=False)
    w, w2 = want_for(pairs_lo[samp], pairs_hi[samp])
    cp = pairs_lo['pos'][samp].astype(np.int64)
    lag_hist = collections.Counter()
    W = 96
    for L in range(-W, W + 1):
        q = cp + L
        valid = (q >= 0) & (q < BLOB) & fine[q]
        if not valid.any():
            continue
        qv = q[valid]
        hit = (B['cm0'][qv] == w[valid]) & (B['cmd2'][qv] == w2[valid])
        lag_hist[L] = int(hit.sum())
    top = lag_hist.most_common(8)
    print('lag histogram top:', top)
    base = sum(c for _, c in lag_hist.items())
    print(f'(sample match mass {base:,} of {len(samp):,} pairs; window ±{W})')

    # full assignment over the top lags (nearest-lag preference)
    LAGS = [L for L, _ in top if _ > 0][:8]
    fpos = np.full(len(pairs_lo), -1, np.int64)
    w, w2 = want_for(pairs_lo, pairs_hi)
    cp = pairs_lo['pos'].astype(np.int64)
    assigned = fpos == -1
    for L in sorted(LAGS, key=abs):
        q = cp + L
        valid = (q >= 0) & (q < BLOB) & fine[q] & assigned
        if not valid.any():
            continue
        qv = q[valid]
        hit = valid.copy()
        hit[valid] = (B['cm0'][qv] == w[valid]) & (B['cmd2'][qv] == w2[valid])
        fpos[hit] = q[hit]
        assigned = fpos == -1
    print(f'fine assigned: {int((fpos >= 0).sum()):,} / {len(pairs_lo):,}')
    np.savez_compressed(DIR / 'v52_fine.npz',
                        pair_pos=pairs_lo['pos'].astype(np.int64),
                        pair_mat=pairs_lo['mat'], pair_r=pairs_lo['r'],
                        pair_k=pairs_lo['k'], fine_pos=fpos)
    print('saved', DIR / 'v52_fine.npz')


if __name__ == '__main__':
    main()
