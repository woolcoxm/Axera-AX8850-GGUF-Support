#!/usr/bin/env python3
"""Decode scale-table layout from sdA/sdB differential builds.

sdA: per-matrix M = 2^mid   -> scale slots carry c*2^mid (c unknown const)
sdB: per-row M(r) = 2^(r>>7)*(1+(r&127)/128) -> scale slots carry c*M(r)

Base = cm0 (M=1). Weight BYTES are M-invariant (normalized quant), so diffs
isolate scale/derived entries. For each differing byte position we try bf16
interpretation at both phases and compute the value ratio vs cm0.
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


def load_blob(path):
    m = onnx.load(path, load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params' and len(init.raw_data) > 1000000:
            return numpy_helper.to_array(init).astype(np.uint8)[:BLOB]
    raise RuntimeError(f'no npu_params in {path}')


def load_tag(tag):
    for pat in (f'/tmp/cmdumps/{tag}_layer_dump.onnx', f'/tmp/cmdumps/{tag}_layer_*.onnx'):
        fs = sorted(glob.glob(pat))
        if fs:
            return load_blob(fs[0])
    raise RuntimeError(f'missing dump for {tag}')


def bf16_to_f32(u16):
    u = u16.astype(np.uint32) << 16
    return u.view(np.float32) if hasattr(u, 'view') else np.frombuffer(u.tobytes(), np.float32)


def main():
    base = load_tag('cm0')
    sdA = load_tag('sdA')
    sdB = load_tag('sdB')
    print('loaded')

    for tag, other in (('sdA', sdA), ('sdB', sdB)):
        diff = base != other
        print(f'== {tag}: {int(diff.sum()):,} differing bytes ==')
        pos = np.flatnonzero(diff)
        print('range:', pos.min(), pos.max())
        d = np.diff(pos)
        # cluster view
        for thr in (8, 64):
            cuts = np.flatnonzero(d > thr)
            print(f'  clusters(gap>{thr}): {len(cuts) + 1}')
        # try bf16 at phase 0 and 1
        for phase in (0, 1):
            b16 = base[phase:].view(np.uint16) if len(base[phase:]) % 2 == 0 else base[phase:-1].view(np.uint16)
            o16 = other[phase:].view(np.uint16) if len(other[phase:]) % 2 == 0 else other[phase:-1].view(np.uint16)
            # word index w covers bytes [2w+phase, 2w+phase+2)
            changed_words = b16 != o16
            print(f'  phase {phase}: {int(changed_words.sum()):,} differing u16 words')
            wpos = np.flatnonzero(changed_words)
            if len(wpos) == 0:
                continue
            bv = bf16_to_f32(b16[wpos])
            ov = bf16_to_f32(o16[wpos])
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(bv != 0, ov / bv, np.nan)
            fin = np.isfinite(ratio) & (np.abs(ratio) > 1e-6)
            if fin.sum():
                rv = ratio[fin]
                vals, cnts = np.unique(np.round(np.log2(np.abs(rv)), 3), return_counts=True)
                top = np.argsort(-cnts)[:12]
                print('    log2 ratio histogram:', [(float(vals[t]), int(cnts[t])) for t in top])

    np.savez('/tmp/diff_positions.npz',
             diffA=np.asarray(base != sdA, dtype=bool),
             diffB=np.asarray(base != sdB, dtype=bool))
    print('saved diff masks')


if __name__ == '__main__':
    main()
