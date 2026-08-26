#!/usr/bin/env python3
"""Decode npu_params layout from the marker build dumps.

Marker design: every layer's matrix of type T holds the SAME seeded uniform
[-1,1] pattern, scaled 1.0 (even layers) / 0.5 (odd layers). Under any
per-row-symmetric quantization the int8 bytes are amplitude-invariant:
q = clip(rint(127 * w / maxabs(row)), -127, 127).
So:  - all 28 layer blobs must be byte-identical in weight regions
     - even/odd blobs differ ONLY in scale tables (2x amplitude)
     - each matrix's bytes = known pattern -> search finds offsets + interleave.
"""
import glob
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SHAPES = {
    'self_attn.q_proj.weight': (2048, 1024),
    'self_attn.k_proj.weight': (1024, 1024),
    'self_attn.v_proj.weight': (1024, 1024),
    'self_attn.o_proj.weight': (1024, 2048),
    'mlp.gate_proj.weight': (3072, 1024),
    'mlp.up_proj.weight': (3072, 1024),
    'mlp.down_proj.weight': (1024, 3072),
}


def marker_patterns():
    rng = np.random.default_rng(20260825)
    # generation order must match mk checkpoint script: dict order = SHAPES order
    return {k: rng.uniform(-1, 1, s).astype(np.float32) for k, s in SHAPES.items()}


def quant_expected(pat):
    """per-row symmetric -> int8 pattern (amplitude invariant)"""
    m = np.abs(pat).max(axis=1, keepdims=True)
    return np.clip(np.rint(127.0 * pat / m), -127, 127).astype(np.int8)


def load_blobs():
    blobs = []
    for fp in sorted(glob.glob('/tmp/qam_dump_*.onnx')):
        m = onnx.load(fp, load_external_data=False)
        for init in m.graph.initializer:
            if init.name == 'npu_params':
                arr = numpy_helper.to_array(init)
                blobs.append((fp, arr))
                break
    return blobs


def variants(q):
    """layout hypotheses for how an [out,in] int8 matrix is stored"""
    out, inn = q.shape
    v = {}
    v['rowmajor'] = q
    v['colmajor'] = np.ascontiguousarray(q.T)
    # 4-lane n-major interleave: groups of 4 output rows, walk input dim
    for lane in (4, 8, 16, 32):
        if out % lane == 0:
            t = q.reshape(out // lane, lane, inn).transpose(0, 2, 1).reshape(-1)
            v[f'ilv{lane}_out'] = np.ascontiguousarray(t)
        if inn % lane == 0:
            t = q.reshape(out, inn // lane, lane).transpose(1, 0, 2).reshape(-1)
            v[f'ilv{lane}_in'] = np.ascontiguousarray(t)
    return v


def search(blob, needle):
    nd = needle.astype(np.uint8).tobytes()
    probe = nd[:64]
    p = blob.find(probe)
    while p >= 0:
        if blob[p:p + len(nd)] == nd:
            return p
        p = blob.find(probe, p + 1)
    return None


def main():
    blobs = load_blobs()
    layer_blobs = [(fp, b) for fp, b in blobs if b.size == 17360392]
    print(f'{len(layer_blobs)} layer blobs')
    if len(layer_blobs) < 2:
        print('need at least 2 layer blobs (rerun marker build)')
        return

    # 1. even vs odd amplitude -> which blobs are even/odd unknown; group by equality
    sigs = {}
    for fp, b in layer_blobs:
        h = b[::97].tobytes()  # cheap signature
        sigs.setdefault(h, []).append(fp)
    groups = list(sigs.values())
    print(f'{len(groups)} distinct blob signatures (expect 2: amplitude 1.0 / 0.5)')
    if len(groups) == 2:
        a = layer_blobs[[f for f, _ in layer_blobs].index(groups[0][0])][1].tobytes()
        b = layer_blobs[[f for f, _ in layer_blobs].index(groups[1][0])][1].tobytes()
        same = np.frombuffer(a, np.uint8) == np.frombuffer(b, np.uint8)
        CH = 256
        n = len(a) // CH
        frac = same[:n * CH].reshape(n, CH).mean(axis=1)
        runs, cur, start = [], frac[0] > 0.99, 0
        for i in range(1, n):
            v = frac[i] > 0.99
            if v != cur:
                runs.append((cur, start * CH, i * CH)); cur, start = v, i
        runs.append((cur, start * CH, n * CH))
        print('even-vs-odd diff map (SCALE tables differ, weight bytes equal):')
        for iseq, s, e in runs:
            if e - s > 512:
                print(f'  {"SAME" if iseq else "DIFF"}: {s:>10,}-{e:>10,} ({e-s:>7,} B)')

    # 2. find matrix offsets via pattern probes (use one blob)
    blob = layer_blobs[0][1].tobytes()
    pats = marker_patterns()
    results = {}
    for name, pat in pats.items():
        q = quant_expected(pat)
        found = False
        for vname, arr in variants(q).items():
            off = search(blob, arr[:8192].copy() if arr.size >= 8192 else arr)
            if off is not None:
                print(f'{name}: {vname} at offset {off:,} (len {arr.size:,})')
                results[name] = (vname, off, arr.size)
                found = True
                break
        if not found:
            print(f'{name}: NO variant matched — need deeper interleave search')
    json.dump({k: [v[0], v[1], v[2]] for k, v in results.items()},
              open('/tmp/blob_layout.json', 'w'), indent=1)
    print('layout saved to /tmp/blob_layout.json')


if __name__ == '__main__':
    main()
