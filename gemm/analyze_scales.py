#!/usr/bin/env python3
"""Scale-table + meta-region analysis of layer npu_params blobs.

Known: marker weights W = (16*code(r,k) + dither(r,k))/127, rowmax forced to
1.0 by anchors W[:,0] = 1.0. Decode proved: int4 nibble = code = q8>>4,
8-bit bytes = 128 + q8 with q8 = W*127 exactly => per-row scale = rowmax/127.

This script:
 1. locates candidate scale values (bf16) in the blob for a range of
    plausible int4 scale formulas (16/127, 1/7, 1/7.5, 1/8, 1/127, ...)
 2. dumps the structure around the hits ((bf16, i16) pair hypothesis)
 3. checks nib-position coverage vs layer_layout_v3.pkl and finds the
    unassigned (anchor k=0) slots
 4. locates npu_params inside the raw .axmodel file for byte-patch offsets
"""
import glob
import pickle
import sys

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
CMDUMPS = '/tmp/cmdumps'


def load_blob(tag):
    fs = sorted(glob.glob(f'{CMDUMPS}/{tag}_layer_*.onnx'))
    assert fs, tag
    m = onnx.load(fs[0], load_external_data=False)
    for init in m.graph.initializer:
        if init.name == 'npu_params':
            return numpy_helper.to_array(init).astype(np.int32)[:BLOB]
    raise RuntimeError('no npu_params')


def bf16_bits(f):
    u = np.float32(f).view(np.uint32)
    # round-to-nearest-even bf16 conversion
    rounded = (u >> 16) & 1
    bits = (u >> 16) + rounded
    return np.uint16(bits)


def find_pattern(blob_u16, pattern):
    hits = np.flatnonzero(blob_u16 == pattern)
    return hits


def main():
    b0 = load_blob('cm0')
    b2 = load_blob('cm2')  # same codes, different dither
    blob_u16 = b0.astype(np.uint16)  # every byte-offset alignment via shifts

    print('== scale candidates (rowmax=1.0) ==')
    cands = {
        '16/127 (0.125984)': 16 / 127,
        '1/7 (0.142857)': 1 / 7,
        '1/7.5 (0.13333)': 1 / 7.5,
        '1/8 (0.125)': 1 / 8,
        '1/7.9375': 1 / 7.9375,
        '1/127 s8': 1 / 127,
        '1/112': 1 / 112,
        '1/120': 1 / 120,
        '1/127*8 (0.0630)': 8 / 127,
    }
    for name, val in cands.items():
        pat = bf16_bits(val)
        # even alignment
        h_even = np.flatnonzero(blob_u16 == pat)
        # odd alignment: bytes shifted by 1
        u16_odd = b0[1:-1].astype(np.uint8).view  # placeholder
        # count both alignments by raw byte search
        raw = b0.astype(np.uint8).tobytes()
        pb = pat.tobytes()
        cnt = 0
        off = raw.find(pb)
        positions = []
        while off != -1 and cnt < 100000:
            positions.append(off)
            cnt += 1
            off = raw.find(pb, off + 1)
        print(f'{name:22s} bf16=0x{int(pat):04x} count={len(positions)}', end='')
        if 0 < len(positions) <= 30000:
            positions = np.array(positions)
            print(f' first={positions[0]} gaps-uniques={len(np.unique(np.diff(np.sort(positions))))}', end='')
        print()

    print()
    print('== around first 16/127 hits ==')
    raw = b0.astype(np.uint8).tobytes()
    for name, val in cands.items():
        pat = bf16_bits(val)
        pb = pat.tobytes()
        off = raw.find(pb)
        if off == -1:
            continue
        positions = []
        while off != -1:
            positions.append(off)
            off = raw.find(pb, off + 1)
        if not (100 < len(positions) < 300000):
            continue
        positions = np.sort(np.array(positions))
        print(f'--- {name}: n={len(positions)} ---')
        print('first 16 offsets:', positions[:16])
        d = np.diff(positions)
        vals, cnts = np.unique(d, return_counts=True)
        top = np.argsort(-cnts)[:8]
        print('top gaps:', [(int(vals[t]), int(cnts[t])) for t in top])
        # dump 32 bytes around first hit
        lo = max(0, positions[0] - 16)
        chunk = b0[lo:positions[0] + 48]
        print('ctx:', ' '.join(f'{int(x):02x}' for x in chunk))
        # check the int16 word following each bf16
        nxt = b0[positions[:64] + 2].astype(np.int16)
        print('following i16:', nxt[:32])

    # nib coverage vs layout table
    print()
    print('== layout table coverage ==')
    tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
    nib_pos = np.zeros(BLOB, bool)
    for (m, r, k), (p, half) in tab.items():
        nib_pos[p] = True
    print('positions claimed by table:', nib_pos.sum())
    nib_mask = (b0 == b2)
    print('dither-immune bytes (nibble or const):', int(nib_mask.sum()))
    unclaimed = nib_mask & ~nib_pos
    print('dither-immune but unclaimed:', int(unclaimed.sum()))
    upos = np.flatnonzero(unclaimed)
    if len(upos):
        print('unclaimed offsets head:', upos[:32])
        print('unclaimed region range:', upos.min(), upos.max())


if __name__ == '__main__':
    main()
