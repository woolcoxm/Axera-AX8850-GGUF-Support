#!/usr/bin/env python3
"""Anchor the REAL bf16 engine's weight layout directly by value.

For every matrix row, search the engine for its first N bf16 values at
stride 64 (the observed interleave). Builds {(matrix, r, k) -> u16 pos}
for the real templates. Handles dual copies (multiple hits).
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


def f2b(f):
    u = f.astype(np.float32).view(np.uint32)
    bits = u >> 16
    rem = u & 0xFFFF
    ru = (rem > 0x8000) | ((rem == 0x8000) & ((bits & 1) == 1))
    return (bits + ru).astype(np.uint16)


NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    eng_path = sys.argv[2] if len(sys.argv) > 2 else \
        '/home/kram/Desktop/Projects/LLMTest/gemm/baked/real2048_bf16/qwen3_p128_l0_together.axmodel'
    L = f'model.layers.{layer}.'
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {L + NAMES[m] for m in NAMES} |
                {L + 'input_layernorm.weight', L + 'post_attention_layernorm.weight'})
    raw = open(eng_path, 'rb').read()
    # npu_params at 1570 (verified by marker signature); blob = 65,023,112
    eng = np.frombuffer(raw, np.uint8, 65023112, 1570).view(np.uint16)
    print('engine u16 slots:', len(eng))

    # value -> positions index (only for values that appear in row prefixes)
    in_ln = W[L + 'input_layernorm.weight']
    post_ln = W[L + 'post_attention_layernorm.weight']

    results = {}
    for mat in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
        n, kk = SHAPES[mat]
        w = W[L + NAMES[mat]]
        # fold candidates per matrix (input-side norm)
        if mat in ('q', 'k', 'v'):
            wf = w * in_ln[None, :]
        elif mat in ('gate', 'up'):
            wf = w * post_ln[None, :]
        else:
            wf = w
        found_rows = 0
        pos_list, r_list, k_list = [], [], []
        maxk = np.int64(0)
        for r in range(n):
            pat = f2b(w[r, :6])
            firsts = np.flatnonzero(eng == pat[0])
            hits = []
            for c in firsts:
                if c + 64 * 5 < len(eng) and np.array_equal(eng[c:c + 64 * 6:64], pat):
                    hits.append(int(c))
                    if len(hits) >= 3:
                        break
            if hits:
                found_rows += 1
                for hit in hits:
                    for k in range(kk):
                        p = hit + 64 * k
                        if p < len(eng):
                            pos_list.append(p)
                            r_list.append(r)
                            k_list.append(k)
        results[mat] = (np.array(pos_list, np.int64), np.array(r_list), np.array(k_list))
        print(f'{mat}: anchored {found_rows}/{n} rows, {len(pos_list)} elements')
    np.savez('/tmp/real_layout.npz',
             **{f'{m}_{f}': results[m][i] for m in results for i, f in enumerate(('pos', 'r', 'k'))})
    print('saved /tmp/real_layout.npz')


if __name__ == '__main__':
    main()
