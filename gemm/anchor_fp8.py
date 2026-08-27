#!/usr/bin/env python3
"""Anchor the fp8_e4m3 layer engine's weight layout by value.

Adapts the bf16 value-anchored search (anchor_real_layout.py) to 1-byte
e4m3 storage. Tries scale hypotheses:
  A) raw e4m3(w)            (no scale folded)
  B) e4m3(w / row_scale)    row_scale = maxabs/448  (per-output-channel)
  C) e4m3(w / 16)           (fixed scale)
For each: search q_proj row prefixes at geometric strides, report hits.
Usage: anchor_fp8.py [engine_path]
"""
import sys

import numpy as np
import json
import struct


def read_st(path, wanted):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    out = {}
    with open(path, 'rb') as f:
        f.seek(8 + n)
        for name, meta in hdr.items():
            if name == '__metadata__' or name not in wanted:
                continue
            off, end = meta['data_offsets']
            f.seek(8 + n + off)
            raw = f.read(end - off)
            if meta['dtype'] == 'BF16':
                u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif meta['dtype'] == 'F16':
                arr = np.frombuffer(raw, np.float16).astype(np.float32)
            else:
                arr = np.frombuffer(raw, np.float32)
            out[name] = arr.reshape(meta['shape'])
    return out


def f32_to_e4m3_rne(x):
    """Round-nearest-even f32 -> e4m3 byte via ml_dtypes (e4m3fn: no inf, max 448)."""
    import ml_dtypes
    return np.asarray(x, np.float32).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)


NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}


def try_anchor(eng, pat_rows, strides=(8, 16, 32, 64, 128), n_probe=6):
    """pat_rows: list of (label, uint8[6] pattern). Return hits per label."""
    res = {}
    for label, pat in pat_rows:
        firsts = np.flatnonzero(eng == pat[0])
        hits = []
        for c in firsts:
            for s in strides:
                if c + s * (len(pat) - 1) < len(eng):
                    ok = all(eng[c + s * i] == pat[i] for i in range(1, len(pat)))
                    if ok:
                        hits.append((int(c), s))
                        break
            if len(hits) >= 3:
                break
        res[label] = hits
    return res


def main():
    eng_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fp8build/qwen3_p128_l0_together.axmodel'
    L = 'model.layers.0.'
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {L + NAMES[m] for m in NAMES})
    raw = open(eng_path, 'rb').read()
    eng = np.frombuffer(raw, np.uint8)
    print('engine bytes:', len(eng))

    w = W[L + NAMES['q']]  # [2048, 1024]
    r = 0
    row = w[r]
    variants = {
        'A_raw': f32_to_e4m3_rne(row[:8]),
        'B_rowmax448': f32_to_e4m3_rne(row[:8] / (np.abs(row).max() / 448.0)),
        'C_div16': f32_to_e4m3_rne(row[:8] / 16.0),
        'D_rowmax240': f32_to_e4m3_rne(row[:8] / (np.abs(row).max() / 240.0)),
    }
    for label, pat in variants.items():
        print(label, 'bytes:', ' '.join(f'{b:02x}' for b in pat))
    pats = [(k, v[:6]) for k, v in variants.items()]
    hits = try_anchor(eng, pats)
    for label, h in hits.items():
        print(f'{label}: {len(h)} hits -> {h[:3]}')

    # extra: sanity search for ANY single e4m3 byte run of q row0 anywhere
    # (contiguous, 4+ bytes) — catches unstrided storage
    pat = variants['A_raw']
    run = 0
    for i in range(len(eng) - 4):
        if (eng[i:i + 4] == pat[:4]).all():
            print('contiguous 4-run at', i)
            run += 1
            if run > 5:
                break


if __name__ == '__main__':
    main()
