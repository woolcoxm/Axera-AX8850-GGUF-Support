#!/usr/bin/env python3
"""Anchor the vendor w8a16 npu_params int8 layout by value search.

Quantizes HF layer weights under candidate schemes and searches the
npu_params blob for exact byte-pattern matches (row-start and mid-row
windows). Reports hits per (matrix, scheme).
"""
import sys
import json
import struct
import numpy as np


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


NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}


def q_rowsym(w, denom=127.0):
    """Per-row symmetric int8, round-to-nearest-even."""
    m = np.abs(w).max(axis=1, keepdims=True)
    q = np.clip(np.round(w * (denom / m)), -128, 127).astype(np.int8)
    return q


def q_group(w, g=256, denom=127.0):
    """Per-(row, kgroup) symmetric int8."""
    n, k = w.shape
    pad = (-k) % g
    wp = np.concatenate([w, np.zeros((n, pad), np.float32)], 1)
    wg = wp.reshape(n, -1, g)
    m = np.abs(wg).max(axis=2, keepdims=True)
    q = np.clip(np.round(wg * (denom / np.maximum(m, 1e-30))), -128, 127).astype(np.int8)
    return q.reshape(n, -1)[:, :k]


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    blob = np.frombuffer(open('/tmp/vendor_l0_params.bin' if layer == 0 else
                              '/tmp/vendor_l1_params.bin', 'rb').read(), np.uint8)
    L = f'model.layers.{layer}.'
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {L + NAMES[m] for m in NAMES} |
                {L + 'input_layernorm.weight', L + 'post_attention_layernorm.weight'})
    in_ln = W[L + 'input_layernorm.weight']
    post_ln = W[L + 'post_attention_layernorm.weight']

    # precompute sorted rolling 8-byte windows of the blob as uint64 keys
    win = np.lib.stride_tricks.sliding_window_view(blob, 8)
    keys = win.view(np.uint64).reshape(-1)
    order = np.argsort(keys, kind='stable')
    skeys = keys[order]

    def find(pat8):
        k = np.frombuffer(pat8.tobytes(), np.uint64)[0]
        i = np.searchsorted(skeys, k, 'left')
        j = np.searchsorted(skeys, k, 'right')
        return order[i:j] if j > i else np.array([], np.int64)

    schemes = {
        'rowsym+128': lambda w: q_rowsym(w).astype(np.int16) + 128,
        'rowsym tc': lambda w: q_rowsym(w).astype(np.uint8),
        'grp256+128': lambda w: q_group(w).astype(np.int16) + 128,
        'grp256 tc': lambda w: q_group(w).astype(np.uint8),
    }
    variants = {'raw': lambda m, w: w,
                'fold': lambda m, w: w * (in_ln[None, :] if m in ('q', 'k', 'v')
                                          else post_ln[None, :] if m in ('gate', 'up') else 1.0)}
    for mat in ['q', 'o', 'gate', 'down']:
        w0 = W[L + NAMES[mat]]
        for vname, vf in variants.items():
            w = vf(mat, w0)
            for sname, sf in schemes.items():
                q = sf(w).astype(np.uint8)
                hits_rs = hits_mid = rows_rs = rows_mid = 0
                for r in list(range(0, q.shape[0], max(1, q.shape[0] // 48))):
                    if find(q[r, :8]).size:
                        rows_rs += 1
                    if find(q[r, q.shape[1] // 2: q.shape[1] // 2 + 8]).size:
                        rows_mid += 1
                    hits_rs = rows_rs
                    hits_mid = rows_mid
                nsamp = len(list(range(0, q.shape[0], max(1, q.shape[0] // 48))))
                if rows_rs or rows_mid:
                    print(f'{mat:5s} {vname:5s} {sname:11s}: rowstart {rows_rs}/{nsamp}  mid {rows_mid}/{nsamp}')
    print('done')


if __name__ == '__main__':
    main()
