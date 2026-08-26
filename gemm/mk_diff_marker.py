#!/usr/bin/env python3
"""Differential marker checkpoints for blob decode (same codes/dither as cm0).

Base (identical to mk_code_marker build 0): W = (16*code + dither)/127,
rowmax forced 1.0 by W[:,0]=1.0 anchors, so q8 = 16*code+dither and
int4 = code exactly (verified by the v3 layout decode).

Variants (only ONE thing changes vs the base):
  sdA : per-matrix maxabs M = 2^mid  -> only per-matrix scale entries move
  sdB : per-row maxabs M(r) = 2^(r>>7) * (1 + (r&127)/128)  (unique bf16 per
        row, collision-safe: adjacent M are ~0.78% apart vs bf16 ULP 0.39%)
        -> every scale entry becomes a row-identifiable value
  sn  : W *= -1 on odd k  -> only sign handling differs (byte/nibble/scales)
  nw  : norm weights coded 1..16 (per layer/type offset) -> norm slots move

Usage: mk_diff_marker.py <outdir> <variant>
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
MID = {n: i for i, n in enumerate(SHAPES)}


def base_V(matrix_name):
    """cm0 codes: bits 0,1,2 are row bits; dither per-matrix seeded."""
    shape = SHAPES[matrix_name]
    n, kk = shape
    rng = np.random.default_rng(SEEDS[matrix_name])
    dither = rng.integers(0, 13, shape)
    rnb = int(np.ceil(np.log2(n)))
    R, K = np.meshgrid(np.arange(n), np.arange(kk), indexing='ij')
    code = np.zeros(shape, np.int16)
    for j in range(3):
        bit = j
        if bit < rnb:
            code |= ((R >> bit) & 1) << j
    V = 16 * code + dither
    V[:, 0] = 127
    return V.astype(np.float32)


def make_W(matrix_name, variant):
    V = base_V(matrix_name)
    n, kk = SHAPES[matrix_name]
    W = V / 127.0
    if variant == 'sdA':
        W = W * (2.0 ** MID[matrix_name])
    elif variant == 'sdB':
        r = np.arange(n, dtype=np.float32)[:, None]
        M = (2.0 ** (r // 128)) * (1.0 + (np.mod(r, 128)) / 128.0)
        W = W * M
    elif variant.startswith('sdc'):
        # exponent-only row coding: M = 2^((r >> 4*idx) & 15). The bf16
        # scale shifts exactly by that exponent (no rounding ambiguity).
        idx = int(variant[3])
        r = np.arange(n, dtype=np.int64)[:, None]
        M = 2.0 ** ((r >> (4 * idx)) & 15).astype(np.float32)
        W = W * M
    elif variant == 'sn':
        s = np.where((np.arange(kk) % 2) == 1, -1.0, 1.0).astype(np.float32)
        W = W * s[None, :]
    # base variant 'cm0re' returns W unchanged (regression base)
    return W.astype(np.float32)


def make_norm(L, name, variant):
    size = HIDDEN if ('layernorm' in name) else HDIM
    if variant != 'nw':
        return np.ones(size, np.float32)
    off = {'input_layernorm': 1.0, 'post_attention_layernorm': 5.0,
           'q_norm': 9.0, 'k_norm': 13.0}[name.split('.')[-2] if 'norm' in name else name]
    off += L * 0.25  # layer offset: layer1 = base + 0.25
    return (off + np.arange(size) / (4.0 * size)).astype(np.float32)


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
    assert variant in ('sdA', 'sdB', 'sn', 'nw', 'cm0re') or variant.startswith('sdc')
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
        tensors[p + 'input_layernorm.weight'] = make_norm(L, 'input_layernorm.weight', variant)
        tensors[p + 'post_attention_layernorm.weight'] = make_norm(L, 'post_attention_layernorm.weight', variant)
        tensors[p + 'self_attn.q_norm.weight'] = make_norm(L, 'q_norm.weight', variant)
        tensors[p + 'self_attn.k_norm.weight'] = make_norm(L, 'k_norm.weight', variant)
        for mname in SHAPES:
            tensors[p + mname] = make_W(mname, variant)
    save_safetensors(out / 'model.safetensors', tensors)
    print(f'{variant}: checkpoint at {out}')


if __name__ == '__main__':
    main()
