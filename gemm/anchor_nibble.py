#!/usr/bin/env python3
"""Search vendor npu_params for int4 nibble-packed weight patterns.

Tries {raw, norm-folded} x {row, group64..512} x {denom 7/7.5/8} x {rne/floor}
quantization; patterns are 8 consecutive k-elements packed as
byte = (v_odd << 4) | v_even, v = q + 8 (offset binary), per the 7.0 crack.
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


def pack(qrow):
    """q elements (int) -> packed bytes, v=q+8, byte=(v_odd<<4)|v_even."""
    v = (qrow + 8).astype(np.uint8)
    return (v[1::2] << 4) | v[0::2]


def quant(w, g, denom, floor):
    n, k = w.shape
    if g is None:
        m = np.abs(w).max(axis=1, keepdims=True)
    else:
        pad = (-k) % g
        wp = np.concatenate([w, np.zeros((n, pad), np.float32)], 1) if pad else w
        m = np.abs(wp.reshape(n, -1, g)).max(axis=2, keepdims=True)
        m = np.repeat(m, g, 2).reshape(n, -1)[:, :k]
    s = np.maximum(m, 1e-30) / denom
    q = w / s
    if floor:
        q = np.floor(q)
    else:
        q = np.round(q)
    return np.clip(q, -8, 7).astype(np.int64)


def main():
    blob = np.frombuffer(open('/tmp/vendor_l0_params.bin', 'rb').read(), np.uint8)
    win = np.lib.stride_tricks.sliding_window_view(blob, 4)
    keys = win.view(np.uint32).reshape(-1)
    order = np.argsort(keys, kind='stable')
    skeys = keys[order]

    def find(pat4):
        k = np.frombuffer(pat4.tobytes(), np.uint32)[0]
        i = np.searchsorted(skeys, k, 'left')
        j = np.searchsorted(skeys, k, 'right')
        return order[i:j] if j > i else np.array([], np.int64)

    L = 'model.layers.0.'
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {L + NAMES[m] for m in NAMES} |
                {L + 'input_layernorm.weight', L + 'post_attention_layernorm.weight'})
    in_ln = W[L + 'input_layernorm.weight']
    post_ln = W[L + 'post_attention_layernorm.weight']

    results = []
    for mat in ['q', 'k', 'o', 'down', 'gate']:
        w0 = W[L + NAMES[mat]]
        for vname, w in [('raw', w0),
                         ('fold', w0 * (in_ln[None, :] if mat in ('q', 'k', 'v')
                                        else post_ln[None, :] if mat in ('gate', 'up') else 1.0))]:
            for g in [None, 512, 256, 128, 64]:
                for denom in [7.0, 7.5, 8.0]:
                    for floor in [False, True]:
                        q = quant(w, g, denom, floor)
                        n = q.shape[0]
                        rows = list(range(0, n, max(1, n // 32)))
                        hits = 0
                        for r in rows:
                            if find(pack(q[r, :8])).size or \
                               find(pack(q[r, q.shape[1] // 2: q.shape[1] // 2 + 8])).size:
                                hits += 1
                        if hits:
                            gname = 'row' if g is None else f'g{g}'
                            rname = 'rne' if not floor else 'flr'
                            results.append((mat, vname, gname, denom, rname, hits, len(rows)))
                            print(f'{mat:5s} {vname:4s} {gname:5s} d{denom:<4} {rname}: {hits}/{len(rows)} rows')
    if not results:
        print('no hits in any scheme')


if __name__ == '__main__':
    main()
