#!/usr/bin/env python3
"""Numpy reference of Qwen3 layer-0 forward for the dumped pass-1 state.

Inputs (from the backend dumps):
  p1_in: bf16 [1024] hidden at pos 1
  p0 k/v rows: bf16 [1024] cache row 0 (already rope'd by the engine at pos 0)
Weights: HF safetensors layer 0.
Computes the true layer-0 output and prints its first values + checksum8,
to compare against the engine mask variants.
"""
import json
import struct
import sys

import numpy as np


def bf16_to_f32(path):
    b = np.fromfile(path, np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32).copy()


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
            if dt == 'BF16':
                u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif dt == 'F16':
                arr = np.frombuffer(raw, np.float16).astype(np.float32)
            else:
                arr = np.frombuffer(raw, np.float32)
            out[name] = arr.reshape(meta['shape'])
    return out


def rope(x, pos, dim=128, theta=1000000.0):
    # neo pairing: (i, i+dim/2)... qwen3 uses neox-style half split
    y = x.copy()
    half = dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half) / half))
    for h in range(len(x) // dim):
        seg = x[h * dim:(h + 1) * dim]
        for i in range(half):
            t = pos * inv[i]
            c, s = np.cos(t), np.sin(t)
            y[h * dim + i] = seg[i] * c - seg[i + half] * s
            y[h * dim + i + half] = seg[i] * s + seg[i + half] * c
    return y


def main():
    names = {f'model.layers.0.{k}': v for k, v in {
        'input_layernorm.weight': 1, 'post_attention_layernorm.weight': 1,
        'self_attn.q_proj.weight': 1, 'self_attn.k_proj.weight': 1,
        'self_attn.v_proj.weight': 1, 'self_attn.o_proj.weight': 1,
        'self_attn.q_norm.weight': 1, 'self_attn.k_norm.weight': 1,
        'mlp.gate_proj.weight': 1, 'mlp.up_proj.weight': 1, 'mlp.down_proj.weight': 1,
    }.items()}
    W = read_safetensors('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors', set(names))
    L = 'model.layers.0.'
    eps = 1e-6

    x1 = bf16_to_f32('/tmp/lstate/lstate_p1_L0_in.bin')
    k0 = bf16_to_f32('/tmp/lstate/lstate_p0_L0_kout.bin')
    v0 = bf16_to_f32('/tmp/lstate/lstate_p0_L0_vout.bin')

    def rms(v, w):
        r = 1.0 / np.sqrt((v.astype(np.float64) ** 2).mean() + eps)
        return v * r * w

    h = rms(x1, W[L + 'input_layernorm.weight'])
    q = h @ W[L + 'self_attn.q_proj.weight'].T
    k1 = h @ W[L + 'self_attn.k_proj.weight'].T
    v1 = h @ W[L + 'self_attn.v_proj.weight'].T
    HQ, HKV, D = 16, 8, 128
    qn = W[L + 'self_attn.q_norm.weight']
    kn = W[L + 'self_attn.k_norm.weight']
    q = q.reshape(HQ, D)
    k1 = k1.reshape(HKV, D)
    q = np.concatenate([rms(q[i], qn)[None] for i in range(HQ)])
    k1 = np.concatenate([rms(k1[i], kn)[None] for i in range(HKV)])
    q = np.concatenate([rope(q[i][None], 1)[0][None] for i in range(HQ)])
    k1r = np.concatenate([rope(k1[i][None], 1)[0][None] for i in range(HKV)])
    v1 = v1.reshape(HKV, D)

    # attention: heads attend [k0/v0 (pos0, GQA expand), k1/v1 (pos1/self)]
    Kall = np.stack([k0.reshape(HKV, D), k1r])   # [2, HKV, D]
    Vall = np.stack([v0.reshape(HKV, D), v1])
    out = np.zeros(HQ * D, np.float32)
    for hq in range(HQ):
        hk = hq // 2
        sc = np.array([q[hq] @ Kall[t, hk] / np.sqrt(D) for t in range(2)])
        p = np.exp(sc - sc.max()); p /= p.sum()
        o = sum(p[t] * Vall[t, hk] for t in range(2))
        out[hq * D:(hq + 1) * D] = o
    attn = out @ W[L + 'self_attn.o_proj.weight'].T
    x2 = x1 + attn
    h2 = rms(x2, W[L + 'post_attention_layernorm.weight'])
    g = h2 @ W[L + 'mlp.gate_proj.weight'].T
    u = h2 @ W[L + 'mlp.up_proj.weight'].T
    act = g / (1 + np.exp(-g)) * u
    x3 = x2 + act @ W[L + 'mlp.down_proj.weight'].T
    cs8 = float(x3[:8].sum())
    print(f'reference layer0(pos=1, attend [cache0, self]) out[:4] = {x3[:4]}')
    print(f'cs8 = {cs8:.6f}')


if __name__ == '__main__':
    main()
