#!/usr/bin/env python3
"""Find (bf16, i16) pair clusters structurally in code-marker blobs.

In code-marker builds every row maxabs = 1.0 and q8max = 127 (amplitude
invariant). If the scale clusters are (bf16 scale, int16 q8max) pairs at
period 4, the int16 word must read 127 (0x7f00 LE) repeatedly.
"""
import glob
import sys

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568


def load_blob(tag):
    fs = sorted(glob.glob(f'/tmp/cmdumps/{tag}_layer_*.onnx'))
    m = onnx.load(fs[0], load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params':
            return numpy_helper.to_array(init).astype(np.uint8)[:BLOB]
    raise RuntimeError


def runs_of_true(mask, min_len):
    # yields (start, end) of runs
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            yield s, e


def main():
    b = load_blob('cm0')
    b16 = b.view(np.uint16)
    i16 = b.astype(np.int16)

    # candidate pair phase 0: bytes [bf16][i16] -> u16[0]=bf16, u16[1]=127
    for phase in (0, 1):
        m = np.zeros(len(b16) - 1, bool)
        # i16 word == 127 at odd u16 slot (phase 0) or even (phase 1)
        w = i16.astype(np.int32)
        tgt = (w == 127)
        if phase == 0:
            m = tgt[1::2]
        else:
            m = tgt[0::2]
        # runs of >= 16 consecutive (i.e. >= 8 pairs adjacent)
        total = 0
        clusters = []
        for s, e in runs_of_true(m, 16):
            total += e - s
            clusters.append((s, e))
        print(f'phase {phase}: {len(clusters)} runs>=16words, mass={total}')
        for s, e in clusters[:10]:
            byte_s = s * 2 + (0 if phase == 0 else -1)
            print(f'  run u16[{s}:{e}] bytes~{s*2}-{e*2} len={(e-s)*2}B')

    # Also try: i16 == 127 anywhere, then look at spacing histogram
    hits = np.flatnonzero(i16 == 127)
    print('total i16==127 words:', len(hits))
    # look at the word before each hit (candidate bf16)
    if len(hits):
        prev = b16[np.clip(hits - 1, 0, len(b16) - 1)]
        vals, cnts = np.unique(prev, return_counts=True)
        top = np.argsort(-cnts)[:10]
        print('top preceding u16 values:', [(hex(int(vals[t])), int(cnts[t])) for t in top])


if __name__ == '__main__':
    main()
