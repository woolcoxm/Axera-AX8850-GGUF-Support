#!/usr/bin/env python3
"""Decode the vendor-w8a16 (Pulsar2 5.2) npu_params layout from code markers.

Builds (3 coordinate bits each, per-matrix bit plan: bits<rnb=row, then k):
  cm0:0-2  cm1:3-5  cmB2:6-8  cm3:9-11  cm4:12-14  cm5:15-17  cm6:18-20  cm7:21
  cmd2: dither probe (codes=cm0, second dither family)
  cmmc: matrix code (V = 16*matrix_index + dither)

Storage model (validated on cm0/cm1): int4 nibble pairs, byte=(v_odd<<4)|v_even,
v=q4+8 offset-binary; ~34% of elements stored twice (decode+prefill groups).
"""
import sys
import pickle
import numpy as np

SHAPES = {
    'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
    'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072),
}
MATS = list(SHAPES)
SEEDS = {m: 101 + i for i, m in enumerate(MATS)}
SEEDS2 = {m: 151 + i for i, m in enumerate(MATS)}
MAXBITS = 22
BLOB = 19_212_296

TAGS = ['cm0', 'cm1', 'cmB2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']


def load(tag):
    return np.frombuffer(open(f'/tmp/v52_{tag}_l0_params.bin', 'rb').read(), np.uint8)


def main():
    B = {t: load(t) for t in TAGS + ['cmd2', 'cmmc']}
    for t in TAGS + ['cmd2', 'cmmc']:
        assert len(B[t]) == BLOB, f'{t}: {len(B[t])}'
    print('all builds loaded:', ', '.join(B))

    # --- nibble slots: hi and lo of every byte; code-consistent across builds
    # and dither-invariant (cm0 vs cmd2 identical bytes where dither is dropped)
    same02 = B['cm0'] == B['cmd2']
    nib_hi = same02.copy()
    nib_lo = same02.copy()
    for t in TAGS:
        nib_hi &= (B[t] >> 4) >= 8
        nib_lo &= (B[t] & 15) >= 8
    nib_hi &= (B['cmmc'] >> 4) <= 14  # matrix code 0..6 (+anchor 15 ok)
    nib_lo &= (B['cmmc'] & 15) <= 14
    print(f'nibble slots: hi {int(nib_hi.sum()):,}  lo {int(nib_lo.sum()):,}')

    # accumulate coordinate keys per slot
    def keys_for(slots):
        keys = np.zeros(BLOB, np.int64)
        ok = slots.copy()
        for i, t in enumerate(TAGS):
            code = (B[t] >> 4).astype(np.int64) if slots is nib_hi else (B[t] & 15).astype(np.int64)
            keys |= (code - 8) << (3 * i)
            ok &= ((code - 8) >= 0) & ((code - 8) <= 7)
        return keys, ok

    hi_keys, hi_ok = keys_for(nib_hi)
    lo_keys, lo_ok = keys_for(nib_lo)

    mat_hi = (B['cmmc'] >> 4).astype(np.int64) - 8
    mat_lo = (B['cmmc'] & 15).astype(np.int64) - 8

    claims = {}   # (m, r, k) -> list of (pos, 'hi'/'lo')
    for mi, m in enumerate(MATS):
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        cnt = 0
        for half, keys, ok, matc in (('hi', hi_keys, hi_ok, mat_hi),
                                     ('lo', lo_keys, lo_ok, mat_lo)):
            sel = ok & (matc == mi)
            r = keys & ((1 << rnb) - 1)
            k = (keys >> rnb) & ((1 << knb) - 1)
            sel &= (r < n) & (k < kk)
            # zero high k-bits must be zero (else key has stray bits)
            sel &= (keys >> (rnb + knb)) == 0
            for rr, kkk, pp in zip(r[sel], k[sel], np.flatnonzero(sel)):
                claims.setdefault((m, int(rr), int(kkk)), []).append((int(pp), half))
                cnt += 1
        print(f'{m:5s}: {cnt:,} claims, {sum(1 for c in claims if c[0]==m):,} elements')
    tot = sum(len(v) for v in claims.values())
    print(f'total: {tot:,} claims over {len(claims):,} elements '
          f'(dup factor {tot/15_728_640:.3f}x of 15.73M)')

    # pair adjacency: (m,r,2j) hi and (m,r,2j+1) lo in same byte?
    paired = same_byte = 0
    for (m, r, k), lst in claims.items():
        if k % 2 == 0:
            other = claims.get((m, r, k + 1))
            if other:
                paired += 1
                for (p1, h1) in lst:
                    if h1 == 'hi':
                        for (p2, h2) in other:
                            if h2 == 'lo' and p1 == p2:
                                same_byte += 1
    print(f'even-k elements with odd-k partner: {paired:,}; same-byte hi/lo pairs: {same_byte:,}')

    pickle.dump(claims, open('/tmp/v52_layout_claims.pkl', 'wb'))
    np.savez('/tmp/v52_layout_masks.npz', nib_hi=nib_hi, nib_lo=nib_lo)
    print('saved /tmp/v52_layout_claims.pkl + masks')

    # --- fine plane: bytes carrying the LOW nibbles of q8 (dither pairs).
    # Candidates: cm0 != cmd2 (dither-dependent) with both lo-nibbles <= 12.
    fine = (~same02) & ((B['cm0'] & 15) <= 12) & ((B['cmd2'] & 15) <= 12) \
           & ((B['cm0'] >> 4) <= 12) & ((B['cmd2'] >> 4) <= 12)
    print(f'fine-plane candidate bytes: {int(fine.sum()):,}')

    # map fine bytes to elements via known dither, searching near each
    # coarse pair position (unit-structure locality)
    dith = {m: np.random.default_rng(SEEDS[m]).integers(0, 13, SHAPES[m]) for m in MATS}
    dith2 = {m: np.random.default_rng(SEEDS2[m]).integers(0, 13, SHAPES[m]) for m in MATS}
    finepos = np.flatnonzero(fine)
    fset = set(finepos.tolist())
    from collections import defaultdict
    fine_claims = defaultdict(list)   # (m, r, k) -> fine byte pos
    fine_of_pair = {}                 # coarse byte pos -> fine byte pos
    W_IN = 80   # search half-window around the coarse byte
    nver = 0
    for (m, r, k), lst in claims.items():
        if k % 2:
            continue
        d1 = dith[m][r, k]; d2 = dith[m][r, k + 1] if k + 1 < SHAPES[m][1] else -1
        e1 = dith2[m][r, k]; e2 = dith2[m][r, k + 1] if k + 1 < SHAPES[m][1] else -1
        for (p, half) in lst:
            if half != 'hi':
                continue
            # fine byte packs (odd, even) = (d2<<4)|d1 in cm0 and (e2<<4)|e1 in cmd2
            want0 = (int(d2) << 4) | int(d1)
            wantD = (int(e2) << 4) | int(e1)
            for q in range(max(0, p - W_IN), min(BLOB, p + W_IN)):
                if q in fset and B['cm0'][q] == want0 and B['cmd2'][q] == wantD:
                    fine_claims[(m, r, k)].append(q)
                    fine_claims[(m, r, k + 1)].append(q)
                    fine_of_pair[p] = q
                    nver += 1
                    break
    print(f'fine bytes matched (dither-verified): pairs {nver:,} '
          f'(of {sum(1 for c in claims if c[2]%2==0)} even-k pairs)')
    # fine-coarse offset histogram
    from collections import Counter
    offs = Counter(p2 - p1 for p1, p2 in fine_of_pair.items())
    print('top fine-minus-coarse offsets:', offs.most_common(12))
    pickle.dump({'fine_claims': dict(fine_claims), 'fine_of_pair': fine_of_pair},
                open('/tmp/v52_fine_claims.pkl', 'wb'))
    print('saved /tmp/v52_fine_claims.pkl')


if __name__ == '__main__':
    main()
