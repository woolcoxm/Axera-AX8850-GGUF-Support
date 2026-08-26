#!/usr/bin/env python3
"""Generate a code-encoded marker checkpoint for layer-layout extraction.

Every weight matrix carries V = 16*code(r,k) + dither(r,k) where code encodes
3 coordinate bits per build (bit plan), dither is per-matrix seeded.
Anchors V[:,0] = 127 force uniform rowmax -> symmetric quant preserves V.
Usage: mk_code_marker.py <outdir> <build_index>
"""
import json
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

# bit plan: per matrix, concatenated (r bits then k bits) LSB-first
def nbits(shape):
    n, k = shape
    return int(np.ceil(np.log2(n))) + int(np.ceil(np.log2(k)))

MAXBITS = max(nbits(s) for s in SHAPES.values())  # 12+12=24 for gate/up? computed


def make_V(matrix_name, build_idx):
    shape = SHAPES[matrix_name]
    n, kk = shape
    rng = np.random.default_rng(SEEDS[matrix_name])
    dither = rng.integers(0, 13, shape)
    rnb = int(np.ceil(np.log2(n)))
    R, K = np.meshgrid(np.arange(n), np.arange(kk), indexing='ij')
    code = np.zeros(shape, np.int16)
    for j in range(3):
        bit = build_idx * 3 + j
        if bit < rnb:
            code |= ((R >> bit) & 1) << j
        elif bit < MAXBITS:
            kb = bit - rnb
            code |= ((K >> kb) & 1) << j
    V = 16 * code + dither
    V[:, 0] = 127
    return V.astype(np.float32) / 127.0


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
    build_idx = int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    import shutil
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
        for mname, shape in SHAPES.items():
            tensors[p + mname] = make_V(mname, build_idx)
    save_safetensors(out / 'model.safetensors', tensors)
    print(f'build {build_idx}: checkpoint at {out} ({len(tensors)} tensors, maxbits={MAXBITS})')


if __name__ == '__main__':
    main()
