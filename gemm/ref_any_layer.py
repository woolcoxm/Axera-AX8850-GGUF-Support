#!/usr/bin/env python3
"""Numpy reference of Qwen3 layer-L forward for the dumped pass-1 state
(corrected RMS/rope math — validated against the engine at L0)."""
import json
import struct
import sys

import numpy as np


def bf16(p):
    b = np.fromfile(p, np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32).copy()


def read_st(path, names):
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
            if meta['dtype'] == 'BF16':
                u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif meta['dtype'] == 'F16':
                arr = np.frombuffer(raw, np.float16).astype(np.float32)
            else:
                arr = np.frombuffer(raw, np.float32)
            out[name] = arr.reshape(meta['shape'])
    return out


def rope(x, pos, theta=1000000.0):
    y = x.copy()
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    for i in range(half):
        t = pos * inv[i]
        c, s = np.cos(t), np.sin(t)
        y[..., i] = x[..., i] * c - x[..., i + half] * s
        y[..., i + half] = x[..., i] * s + x[..., i + half] * c
    return y


def main():
    Lidx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    L = f'model.layers.{Lidx}.'
    names = {L + x for x in ['input_layernorm.weight', 'post_attention_layernorm.weight',
                             'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                             'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                             'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
                             'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']}
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors', names)
    eps = 1e-6
    rms = lambda t, w: (t / np.sqrt((t.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)) * w

    x1 = bf16(f'/tmp/lstate/lstate_p1_L{Lidx}_in.bin')
    k0 = bf16(f'/tmp/lstate/lstate_p0_L{Lidx}_kout.bin')
    v0 = bf16(f'/tmp/lstate/lstate_p0_L{Lidx}_vout.bin')

    h = rms(x1[None], W[L + 'input_layernorm.weight'])[0]
    q = (h @ W[L + 'self_attn.q_proj.weight'].T).reshape(16, 128)
    k1 = (h @ W[L + 'self_attn.k_proj.weight'].T).reshape(8, 128)
    v1 = (h @ W[L + 'self_attn.v_proj.weight'].T).reshape(8, 128)
    q = rms(q, W[L + 'self_attn.q_norm.weight'])
    k1 = rms(k1, W[L + 'self_attn.k_norm.weight'])
    q1 = rope(q, 1)
    k1r = rope(k1, 1)

    K = np.stack([k0.reshape(8, 128), k1r])
    V = np.stack([v0.reshape(8, 128), v1])
    o = np.zeros((16, 128), np.float32)
    for hq in range(16):
        hk = hq // 2
        sc = np.array([q1[hq] @ K[t, hk] for t in range(2)]) / np.sqrt(128)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        o[hq] = sum(p[t] * V[t, hk] for t in range(2))
    attn = o.reshape(-1) @ W[L + 'self_attn.o_proj.weight'].T
    x2 = x1 + attn
    h2 = rms(x2[None], W[L + 'post_attention_layernorm.weight'])[0]
    g = h2 @ W[L + 'mlp.gate_proj.weight'].T
    u = h2 @ W[L + 'mlp.up_proj.weight'].T
    x3 = x2 + (g / (1 + np.exp(-g)) * u) @ W[L + 'mlp.down_proj.weight'].T
    print(f'ref L{Lidx}(pos=1): out[:4] = {x3[:4]}')
    print(f'cs8 = {x3[:8].sum():.6f}')


if __name__ == '__main__':
    main()
