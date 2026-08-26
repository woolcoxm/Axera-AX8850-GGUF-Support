#!/usr/bin/env python3
"""Decode the BF16-template weight layout from code markers.

cm builds at -w bf16: weights W = (16*code + dither)/127, codes carry 3
coordinate bits per build (bit plan identical to the s8 crack). Storage:
each element = its bf16 (2 bytes, position unknown). Decoding is DIRECT:
read candidate 2-byte positions, interpret as bf16 value v; then
q8 = round(v * 127) gives 16*code + dither (code = q8>>4, dither = q8&15).
Cross-build consistency + dither validation pins (r, k).

Emits: {(matrix, r, k) -> byte_pos}
"""
import glob
import pickle
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
SEEDS = {m: 101 + i for i, m in enumerate(['self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                                           'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                                           'mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                           'mlp.down_proj.weight'])}
MID = {n: i for i, n in enumerate(SEEDS)}
NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}


def load(tag):
    for pat in (f'/tmp/cmdumps/{tag}_layer_dump.onnx',):
        fs = sorted(glob.glob(pat))
        if fs:
            m = onnx.load(fs[0], load_external_data=False)
            for i in m.graph.initializer:
                if i.name == 'npu_params' and len(i.raw_data) > 1000000:
                    return numpy_helper.to_array(i).astype(np.uint8)
    raise RuntimeError(tag)


def bf16_val(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    order = [f'bfcm{i}' for i in range(8)] + ['bfcmb2']
    B = {t: load(t) for t in order}
    print('loaded', len(B), 'builds; len', len(B['bfcm0']))

    # candidate weight positions: 2-byte aligned slots where ALL builds give
    # values that decode to valid q8 (0..127) with matching dither
    L = len(B['bfcm0'])
    n2 = L // 2
    cand = np.ones(n2, bool)
    q8 = {}
    for t in order:
        u = B[t].view(np.uint16)
        v = bf16_val(u).astype(np.float64)
        q = np.rint(v * 127.0)
        ok = (q >= 0) & (q <= 127)
        # q8 = 16*code + dither with dither 0..12 for real elements (anchor
        # k=0 = 127 exactly); accept 0..15 to include anchors' neighbors
        cand &= ok
        q8[t] = q.astype(np.int16)
    wpos = np.flatnonzero(cand)
    print('candidate element slots:', len(wpos), 'of', n2)
    np.save('/tmp/bf_wpos.npy', wpos)
    np.savez('/tmp/bf_q8.npz', **{t: q8[t] for t in order})
    print('saved candidates + q8 values')


if __name__ == '__main__':
    main()
