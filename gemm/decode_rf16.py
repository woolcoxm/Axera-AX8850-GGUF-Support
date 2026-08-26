#!/usr/bin/env python3
"""Decode the complete BF16-template weight layout from cm0..cm7+cmb2 markers.

Markers: W = (16*code + dither)/127; each element stored as its 2-byte bf16.
q8 = rint(v * 127) recovers 16*code + dither DIRECTLY from any candidate
position — no bit-twiddling. Cross-build codes give (r, k); the dither
validates. Emits {(matrix, r, k) -> byte_pos} for ALL 15.7M elements.
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
MATS = list(SHAPES)
NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}
SEEDS = {n: 101 + i for i, n in enumerate(NAMES.values())}
ORDER = [f'rfcm{i}' for i in range(8)]


def load(tag):
    fs = sorted(glob.glob(f'/tmp/cmdumps/{tag}_layer_dump.onnx'))
    assert fs, tag
    m = onnx.load(fs[0], load_external_data=False)
    for i in m.graph.initializer:
        if i.name == 'npu_params' and len(i.raw_data) > 1000000:
            return numpy_helper.to_array(i).astype(np.uint8)
    raise RuntimeError(tag)


def main():
    B = {t: load(t) for t in ORDER}
    L = len(B['rfcm0'])
    n2 = L // 2
    print('builds loaded, blob', L)

    # dither tables
    dith = {m: np.random.default_rng(SEEDS[NAMES[m]]).integers(0, 13, SHAPES[m]) for m in MATS}
    dith2 = {m: np.random.default_rng(SEEDS[NAMES[m]] + 50).integers(0, 13, SHAPES[m]) for m in MATS}

    # q8 per build at every 2-byte slot
    Q = {}
    cand = np.ones(n2, bool)
    for t in ORDER:
        u = B[t].view(np.uint16)
        v = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        q = np.rint(v * 127.0)
        q = np.where(np.isfinite(q), q, -1)
        ok = (q >= 0) & (q <= 127)
        cand &= ok
        Q[t] = q.astype(np.int16)
    wpos = np.flatnonzero(cand)
    print('candidate slots:', len(wpos), 'of', n2)

    # code accumulation: bits 3*build+j (same plan as the generator)
    # build i covers bits 3i..3i+2; r-bits first then k-bits per matrix
    key = np.zeros(len(wpos), np.int64)
    for bi, t in enumerate(ORDER[:8]):
        q = (Q[t][wpos].astype(np.int64) >> 4)  # code 0..7
        for j in range(3):
            key |= ((q >> j) & 1) << (3 * bi + j)
    d_lo = Q['rfcm0'][wpos].astype(np.int64) & 15

    # try matrix assignment: for each matrix, decode (r, k) from key with
    # that matrix's rnb and check the dither
    out = {}
    total = 0
    wpos64 = wpos.astype(np.int64)
    for mi, m in enumerate(MATS):
        n, kk = SHAPES[m]
        rnb = int(np.ceil(np.log2(n)))
        knb = int(np.ceil(np.log2(kk)))
        r = key & ((1 << rnb) - 1)
        k = (key >> rnb) & ((1 << knb) - 1)
        inr = (r < n) & (k < kk)
        idx = r.astype(np.int64) * kk + k
        dexp = dith[m].flatten()[np.clip(idx, 0, n * kk - 1)]
        good = inr & (d_lo == dexp)
        # second dither (cmb2 seed+50... NOTE generator uses SEEDS2? mk_code_marker uses same seeds per build; dither identical per matrix across builds. Use rebuild-consistency instead: cross-build dither equal.)
        if good.sum():
            ur, uc = np.unique(r[good], return_counts=True)
            print(f'{m}: {int(good.sum()):,} slots, {len(ur)} uniq rows (n={n})')
            total += int(good.sum())
        out[m] = (wpos[good].astype(np.int64), r[good].astype(np.int32), k[good].astype(np.int32))
    print('total decoded:', total, '/ expected', sum(n * kk for n, kk in SHAPES.values()))
    np.savez('/tmp/bf16_layout.npz',
             **{f'{m}_{f}': out[m][i] for m in out for i, f in enumerate(('pos', 'r', 'k'))})
    print('saved /tmp/bf16_layout.npz')


if __name__ == '__main__':
    main()
