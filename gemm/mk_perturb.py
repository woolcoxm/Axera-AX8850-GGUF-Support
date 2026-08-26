#!/usr/bin/env python3
"""Perturbation markers: change ONE (row, kgroup) of ONE matrix so the blob
diff isolates every storage position of that group (nibbles, int8 copies,
scales, anything).

Variant string: p<mat>_<row>_<group>  e.g. pq_100_0 = q row 100 group 0.
Perturbation: W[row, g*256:(g+1)*256] = (V + 7)/127 (shifts dither pattern,
keeps codes and group max).
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

LAYERS = 2
VOCAB = 4096
HIDDEN = 1024
QH, KVH, HDIM = 16, 8, 128
INTER = 3072

SHAPES = {
    'self_attn.q_proj.weight': (QH * HDIM, HIDDEN),
    'self_attn.k_proj.weight': (KVH * HDIM, HIDDEN),
    'self_attn.v_proj.weight': (KVH * HDIM, HIDDEN),
    'self_attn.o_proj.weight': (HIDDEN, QH * HDIM),
    'mlp.gate_proj.weight': (INTER, HIDDEN),
    'mlp.up_proj.weight': (INTER, HIDDEN),
    'mlp.down_proj.weight': (HIDDEN, INTER),
}
SEEDS = {n: 101 + i for i, n in enumerate(SHAPES)}
SHORT = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
         'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
         'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}


def base_V(matrix_name):
    shape = SHAPES[matrix_name]
    n, kk = shape
    rng = np.random.default_rng(SEEDS[matrix_name])
    dither = rng.integers(0, 13, shape)
    rnb = int(np.ceil(np.log2(n)))
    R, K = np.meshgrid(np.arange(n), np.arange(kk), indexing='ij')
    code = np.zeros(shape, np.int16)
    for j in range(3):
        if j < rnb:
            code |= ((R >> j) & 1) << j
    V = 16 * code + dither
    V[:, 0] = 127
    return V.astype(np.float32)


def make_W(matrix_name, variant):
    V = base_V(matrix_name)
    W = (V / 127.0).astype(np.float32)
    if variant.startswith('p'):
        parts = variant.split('_')
        mat, row, grp = parts[0][1:].replace('16', ''), int(parts[1]), int(parts[2])
        if SHORT[mat] == matrix_name:
            n, kk = SHAPES[matrix_name]
            seg = slice(grp * 256, (grp + 1) * 256)
            shift = 16.0 if variant.startswith('pq16') or variant.startswith('pk16') else 7.0
            V2 = V[row, seg] + shift
            if grp == 0:
                V2[0] = 127.0  # keep anchor: same group max
            W[row, seg] = V2 / 127.0
    return W


def save_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes()); offset += arr.nbytes
    hdr = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hdr)) % 8:
        hdr += b" "
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr))); f.write(hdr)
        for b in blobs:
            f.write(b)


def main():
    out = Path(sys.argv[1])
    variant = sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        shutil.copy(f'/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/{f}', out / f)
    cfg = json.load(open('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/config.json'))
    cfg.update({'num_hidden_layers': LAYERS, 'vocab_size': VOCAB})
    (out / 'config.json').write_text(json.dumps(cfg))
    rng = np.random.default_rng(7)
    tensors = {
        'model.embed_tokens.weight': (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(np.float32),
        'model.norm.weight': np.ones(HIDDEN, np.float32),
        'lm_head.weight': (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(np.float32),
    }
    for L in range(LAYERS):
        p = f'model.layers.{L}.'
        tensors[p + 'input_layernorm.weight'] = np.ones(HIDDEN, np.float32)
        tensors[p + 'post_attention_layernorm.weight'] = np.ones(HIDDEN, np.float32)
        tensors[p + 'self_attn.q_norm.weight'] = np.ones(HDIM, np.float32)
        tensors[p + 'self_attn.k_norm.weight'] = np.ones(HDIM, np.float32)
        for mname in SHAPES:
            tensors[p + mname] = make_W(mname, variant)
    save_safetensors(out / 'model.safetensors', tensors)
    print(f'{variant}: checkpoint at {out}')


if __name__ == '__main__':
    main()

# pq16 variant: V+16 on the target group (every element's code +1 -> all
# nibbles flip, int8 +16, sub-group maxes +16 -> scales shift predictably)
