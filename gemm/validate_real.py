#!/usr/bin/env python3
"""Validate the v3 nibble layout + scale formula against the real-weight
2048-ctx build (blob from /tmp/qam_dump_301753_1.onnx).

Model: q8 = clamp(round(W[r,k] * 127 / gmax), -128, 127) with
gmax = max|W[r, g*256:(g+1)*256]|; nibble = (q8 >> 4) + 8 (arithmetic).
Scale entry = bf16(gmax / 127) at the [i16 516][bf16] slots.
"""
import glob
import json
import pickle
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']


def read_safetensors(path, names):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    out = {}
    with open(path, 'rb') as f:
        f.seek(8 + n)
        for name, meta in hdr.items():
            if name == '__metadata__' or name not in names:
                continue
            off, end = meta['data_offsets']
            f.seek(8 + n + off)
            raw = f.read(end - off)
            dt = meta['dtype']
            if dt == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32)
            elif dt == 'BF16':
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif dt == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                raise ValueError(dt)
            out[name] = arr.reshape(meta['shape'])
    return out


def main():
    real_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/qam_dump_301753_1.onnx'
    m = onnx.load(real_path, load_external_data=False)
    blob = None
    for init in m.graph.initializer:
        if init.name == 'npu_params':
            blob = numpy_helper.to_array(init).astype(np.uint8)[:BLOB]
    assert blob is not None
    print('real blob loaded')

    print('loading v3 table...')
    tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
    print('table loaded:', len(tab))

    names = {
        'q': 'model.layers.0.self_attn.q_proj.weight',
        'k': 'model.layers.0.self_attn.k_proj.weight',
        'v': 'model.layers.0.self_attn.v_proj.weight',
        'o': 'model.layers.0.self_attn.o_proj.weight',
        'gate': 'model.layers.0.mlp.gate_proj.weight',
        'up': 'model.layers.0.mlp.up_proj.weight',
        'down': 'model.layers.0.mlp.down_proj.weight',
    }
    W = read_safetensors('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                         set(names.values()))
    for tag, name in names.items():
        if name not in W:
            print(f'{tag}: MISSING {name}')
            return
    rng = np.random.default_rng(1234)
    for tag in MATS:
        w = W[names[tag]]  # [n, k] rows=out
        n, kk = w.shape
        rows = rng.choice(n, size=min(16, n), replace=False)
        match = mism = miss = 0
        for r in rows:
            for g in range(kk // 256):
                seg = w[r, g * 256:(g + 1) * 256]
                gmax = float(np.abs(seg).max())
                if gmax == 0:
                    continue
                s = gmax / 127.0
                q8 = np.clip(np.round(seg / s), -128, 127).astype(np.int32)
                q4 = (q8 >> 4) + 8  # arithmetic shift, offset +8
                for i in range(0, 256, 2):
                    k0 = g * 256 + i
                    e0 = (tag, int(r), k0)
                    e1 = (tag, int(r), k0 + 1)
                    if e0 in tab and e1 in tab and tab[e0][0] == tab[e1][0]:
                        p, _ = tab[e0]
                        byte = int(blob[p])
                        got_hi = byte >> 4
                        got_lo = byte & 15
                        if got_hi == q4[i]:
                            match += 1
                        else:
                            mism += 1
                            if mism <= 3:
                                print(f'  {tag} r={r} k={k0}: pred hi={q4[i]} got {got_hi} (w={seg[i]:.5f} gmax={gmax:.5f} q8={q8[i]})')
                        if got_lo == q4[i + 1]:
                            match += 1
                        else:
                            mism += 1
                            if mism <= 6:
                                print(f'  {tag} r={r} k={k0+1}: pred lo={q4[i+1]} got {got_lo} (w={seg[i+1]:.5f} q8={q8[i+1]})')
                    else:
                        miss += 1
        print(f'{tag}: match={match} mism={mism} miss={miss} (rate {match/(match+mism+1e-9):.4f})')


if __name__ == '__main__':
    main()
