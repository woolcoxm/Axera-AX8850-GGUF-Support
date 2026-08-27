#!/usr/bin/env python3
"""Decode the 5.2 scale-entry table: map each entry to (matrix, row, kgroup).

Entries live in 960 clusters x 63 = 60,480 4-byte slots [i16 A][u16 B].
A = C - 128*log2(group_scale) per build, so A differences between builds
are C-free integers -> trajectory matching against computed marker group
scales identifies each entry's (m, r, kgroup).
Outputs: v52_scale_entries.npz (entry slot -> mat, r, g + C constants).
"""
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent / 'baked' / 'v52_markers'
TAGS = ['cm0', 'cm1', 'cmB2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
MATS = list(SHAPES)
SEEDS = {m: 101 + i for i, m in enumerate(MATS)}


def groupmax_V(mi, build_idx):
    """max V per (row, kgroup) for a code build; anchor col = 127."""
    n, kk = SHAPES[MATS[mi]]
    rng = np.random.default_rng(SEEDS[MATS[mi]])
    dith = rng.integers(0, 13, (n, kk))
    rnb = int(np.ceil(np.log2(n)))
    R, K = np.meshgrid(np.arange(n), np.arange(kk), indexing='ij')
    code = np.zeros((n, kk), np.int16)
    for j in range(3):
        bit = build_idx * 3 + j
        if bit < rnb:
            code |= ((R >> bit) & 1) << j
        else:
            kb = bit - rnb
            code |= ((K >> kb) & 1) << j
    V = 16 * code + dith
    V[:, 0] = 127
    return V.reshape(n, kk // 256, 256).max(2).astype(np.float64)  # (n, kk/256)


def main():
    blobs = {t: np.frombuffer((DIR / f'v52_{t}_l0_params.bin').read_bytes(), np.uint8)
             for t in TAGS + ['cmmc']}
    mx = np.frombuffer((DIR / 'v52_mixamp_l1_params.bin').read_bytes(), np.uint8)
    c0 = blobs['cm0']
    d = c0 != mx
    pos = np.flatnonzero(d)
    gaps = np.diff(pos)
    cuts = np.flatnonzero(gaps > 16)
    st = np.concatenate(([0], cuts + 1)); en = np.concatenate((cuts, [len(pos) - 1]))
    clusters = [(int(pos[s]), int(pos[e])) for s, e in zip(st, en)]
    print(f'{len(clusters)} clusters')

    Aent = {t: [] for t in TAGS}
    Bent = []
    slot_pos = []
    for s, e in clusters:
        v0 = c0[s:s + 4 * 63]
        Aent_entry = v0.view(np.int16).reshape(-1, 2)[:, 0]
        Bent.append(v0.view(np.uint16).reshape(-1, 2)[:, 1])
        slot_pos.append(np.arange(s, s + 4 * 63, 4))
        for t in TAGS:
            Aent[t].append(blobs[t][s:s + 4 * 63].view(np.int16).reshape(-1, 2)[:, 0])
    A = {t: np.concatenate(Aent[t]) for t in TAGS}
    B0 = np.concatenate(Bent)
    SLOTS = np.concatenate(slot_pos)
    print(f'{len(SLOTS)} entries')

    # predicted -128*log2(scale) per (m, r, g) per build, flattened per matrix
    pred = {}   # (mi, tag) -> flattened -128*log2(gmax/127)
    for mi in range(len(MATS)):
        for bi, t in enumerate(TAGS):
            gm = groupmax_V(mi, bi) / 127.0
            pred[(mi, t)] = (-128.0 * np.log2(gm)).reshape(-1)

    # entry dA vs cm0 (integers); candidate dP vs cm0 (rounded)
    dA = np.stack([A[t] - A['cm0'] for t in TAGS[1:]], 1)   # (E, 7)
    from collections import defaultdict
    cand_map = defaultdict(list)
    for mi in range(len(MATS)):
        base = pred[(mi, 'cm0')]
        diff = np.stack([pred[(mi, t)] - base for t in TAGS[1:]], 1)  # (N, 7)
        diff_r = np.round(diff).astype(np.int64)
        for idx in range(len(base)):
            cand_map[tuple(diff_r[idx].tolist())].append((mi, idx))
    print('candidate hash size:', len(cand_map))

    matched = 0
    out_mat = np.full(len(SLOTS), -1, np.int8)
    out_idx = np.full(len(SLOTS), -1, np.int64)
    Cvals = np.zeros(len(SLOTS))
    dA_r = np.round(dA).astype(np.int64)
    for e in range(len(SLOTS)):
        key = tuple(dA_r[e])
        cands = cand_map.get(key, [])
        if len(cands) == 1:
            mi, idx = cands[0]
            out_mat[e], out_idx[e] = mi, idx
            # C from cm0: A + (-128 log2 s)?? A = C - 128 log2 s -> C = A + 128 log2 s
            Cvals[e] = A['cm0'][e] - pred[(mi, 'cm0')][idx]
            matched += 1
    print(f'uniquely matched: {matched:,} / {len(SLOTS):,}')
    for mi, m in enumerate(MATS):
        n, kk = SHAPES[m]
        sel = out_mat == mi
        print(f'  {m:5s}: {int(sel.sum()):,}/{n * kk // 256} entries')
    np.savez_compressed(DIR / 'v52_scale_entries.npz', slot=SLOTS, mat=out_mat,
                        idx=out_idx, C=Cvals, B=B0)
    print('saved', DIR / 'v52_scale_entries.npz')


if __name__ == '__main__':
    main()
