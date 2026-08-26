#!/usr/bin/env python3
"""Full joint decode of layer npu_params layout from code-marker builds.

Builds (bits encoded, 3 per build): cm0=0-2, cm1=3-5, cmb2=6-8, cm3=9-11,
cm4=12-14, cm5=15-17, cm6=18-20, cm7=21. cm2 = dither probe (codes = cm0).
Per matrix bit plan: bits 0..rnb-1 = r, bits rnb.. = k (kb = bit - rnb).

Storage: nibble-packed positions (2 elements/byte, q4+8) and 8-bit positions
(V or V+128 per position). Dither cross-check validates assignments.
"""
import glob
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SHAPES = {
    'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
    'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072),
}
MATS = list(SHAPES)
SEEDS = {n: 101 + i for i, n in enumerate(SHAPES)}
SEEDS2 = {n: 151 + i for i, n in enumerate(SHAPES)}
MAXBITS = 22
BLOB = 17346568


def load(tag):
    fs = sorted(glob.glob(f'/tmp/cmdumps/{tag}_layer_*.onnx'))
    if not fs:
        return None
    m = onnx.load(fs[0], load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params':
            return numpy_helper.to_array(init).astype(np.int32)[:BLOB]
    return None


def main():
    order = ['cm0', 'cm1', 'cmb2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']
    B = {t: load(t) for t in order + ['cm2']}
    for t in order + ['cm2']:
        assert B[t] is not None, f'missing {t}'
    L = BLOB
    print('all builds loaded')

    # nibble positions: identical cm0/cm2 bytes, doubled nibble
    b0, b2 = B['cm0'], B['cm2']
    nib = (b0 == b2) & ((b0 >> 4) == (b0 & 15)) & ((b0 >> 4) >= 8)
    print(f'nibble positions: {nib.sum():,}')

    # dither tables per matrix (flattened to (r,k) -> dither), and reverse maps
    dith1 = {m: np.random.default_rng(SEEDS[m]).integers(0, 13, SHAPES[m]) for m in MATS}
    dith2 = {m: np.random.default_rng(SEEDS2[m]).integers(0, 13, SHAPES[m]) for m in MATS}

    # bit-plan tables: for matrix m, build (r,k) grid -> 22-bit key
    key_of = {}
    for m in MATS:
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        R, K = np.meshgrid(np.arange(n), np.arange(kk), indexing='ij')
        key = np.zeros((n, kk), np.int64)
        for bit in range(MAXBITS):
            if bit < rnb:
                v = (R >> bit) & 1
            else:
                kb = bit - rnb
                v = (K >> kb) & 1 if kb < knb else np.zeros_like(R)
            key |= v.astype(np.int64) << bit
        key_of[m] = key
        # forward map key -> (r,k); keys may repeat for k bits beyond knb (zeroed) - fine

    # decode 8-bit positions
    eight = ~nib
    # codes per build: V = byte or byte-128; flag = byte>=128 (validate consistent)
    flag = B['cm0'] >= 128
    keys8 = np.zeros(L, np.int64)
    ok8 = eight.copy()
    for i, t in enumerate(order):
        byte = B[t]
        v = byte - np.where(byte >= 128, 128, 0)
        # flag consistency check for later builds
        fl = byte >= 128
        code = v >> 4
        d = v & 15
        keys8 |= code.astype(np.int64) << (3 * i)
        ok8 &= (code <= 7) & (d <= 12) & (fl == flag) & (byte != 0) | (~eight)
    # per matrix validation with dither
    assign_m = np.full(L, -1, np.int8)
    assign_r = np.zeros(L, np.int32)
    assign_k = np.zeros(L, np.int32)
    solved8 = np.zeros(L, bool)
    for m in MATS:
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        r = (keys8 >> 0)
        r = np.where(rnb < 64, r & ((1 << rnb) - 1), r)
        k = (keys8 >> rnb) & ((1 << knb) - 1)
        inrange = ok8 & (r < n) & (k < kk) & ~solved8
        # dither check: build dithers at (r,k) must equal decoded low nibbles
        idx = (r.astype(np.int64) * kk + k)
        d1v = dith1[m].flatten()[np.clip(idx, 0, n * kk - 1)]
        d2v = dith2[m].flatten()[np.clip(idx, 0, n * kk - 1)]
        # decoded dithers per build
        good = inrange.copy()
        for i, t in enumerate(order):
            byte = B[t]
            v = byte - np.where(byte >= 128, 128, 0)
            dexp = dith1[m].flatten()[np.clip(idx, 0, n * kk - 1)]
            good &= (v & 15) == dexp
            good &= ~nib  # 8-bit only
        # cm2 dither2 check
        v2 = B['cm2'] - np.where(B['cm2'] >= 128, 128, 0)
        good &= (v2 & 15) == d2v
        assign_m[good] = MATS.index(m)
        assign_r[good] = r[good]
        assign_k[good] = k[good]
        solved8 |= good
        print(f'{m}: 8-bit positions {int(good.sum()):,}')
    print(f'8-bit solved total: {int(solved8.sum()):,}')

    # nibble positions: hi/lo elements
    hi_keys = np.zeros(L, np.int64)
    lo_keys = np.zeros(L, np.int64)
    for i, t in enumerate(order):
        code_h = (B[t] >> 4) - 8
        code_l = (B[t] & 15) - 8
        hi_keys |= code_h.astype(np.int64) << (3 * i)
        lo_keys |= code_l.astype(np.int64) << (3 * i)

    nib_elem = {}  # (m, r, k) -> (pos, 'hi'/'lo')
    for m in MATS:
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        for tagname, keys in (('hi', hi_keys), ('lo', lo_keys)):
            r = keys & ((1 << rnb) - 1)
            k = (keys >> rnb) & ((1 << knb) - 1)
            # consecutive pairing check: hi->even k, lo->odd k expected
            good = nib & (r < n) & (k < kk)
            if tagname == 'hi':
                good &= (k & 1) == 0
            else:
                good &= (k & 1) == 1
            for (rr, kk_, pp) in zip(r[good], k[good], np.flatnonzero(good)):
                nib_elem[(m, int(rr), int(kk_))] = (int(pp), tagname)
        print(f'{m}: nibble elems {sum(1 for key in nib_elem if key[0]==m):,}')
    print(f'nibble elements decoded: {len(nib_elem):,}')
    # pair adjacency check
    paired = sum(1 for (m, r, k), (p, half) in nib_elem.items()
                 if (m, r, k ^ 1) in nib_elem and nib_elem[(m, r, k ^ 1)][0] == p)
    print(f'adjacent-pair elements: {paired:,} (expect ~2x elements/2)')

    np.savez('/tmp/layer_layout.npz',
             m=assign_m, r=assign_r, k=assign_k, solved8=solved8, nib=nib)
    import pickle
    pickle.dump(nib_elem, open('/tmp/layer_nib_elems.pkl', 'wb'))
    print('saved')


if __name__ == '__main__':
    main()
