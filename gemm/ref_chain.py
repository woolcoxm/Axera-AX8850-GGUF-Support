#!/usr/bin/env python3
"""Full-model numpy reference for the chain harness (autoregressive, 6 tokens).

Mirrors chain_test.c exactly: same embeddings (bf16 round-tripped), same
cache update order, prints the same (pos, layer) checksums for diffing.
"""
import json
import struct
import sys

import numpy as np

NL = 28
NT = 6
EPS = 1e-6


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


def rope(x, pos, theta=1000000.0):
    y = x.copy()
    half = x.shape[-1] // 2
    inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    for i in range(half):
        t = pos * inv[i]
        y[..., i] = x[..., i] * np.cos(t) - x[..., i + half] * np.sin(t)
        y[..., i + half] = x[..., i] * np.sin(t) + x[..., i + half] * np.cos(t)
    return y


def main():
    wanted = set()
    for l in range(NL):
        for k in ['input_layernorm.weight', 'post_attention_layernorm.weight',
                  'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
                  'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
                  'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
                  'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']:
            wanted.add(f'model.layers.{l}.{k}')
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors', wanted)

    emb = np.load('/tmp/chain_emb_rt.npy')  # [6, 1024] f32 (bf16 round-tripped)

    rms = lambda t, w: (t / np.sqrt((t.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)) * w

    Kc = [[None] * NL for _ in range(NT)]  # Kc[pos][layer] = k row [8,128]
    Vc = [[None] * NL for _ in range(NT)]
    log = open('/tmp/chain_ref.log', 'w')

    for pos in range(NT):
        x = emb[pos].copy()
        for l in range(NL):
            L = f'model.layers.{l}.'
            h = rms(x[None], W[L + 'input_layernorm.weight'])[0]
            q = (h @ W[L + 'self_attn.q_proj.weight'].T).reshape(16, 128)
            k = (h @ W[L + 'self_attn.k_proj.weight'].T).reshape(8, 128)
            v = (h @ W[L + 'self_attn.v_proj.weight'].T).reshape(8, 128)
            q = rms(q, W[L + 'self_attn.q_norm.weight'])
            k = rms(k, W[L + 'self_attn.k_norm.weight'])
            q1 = rope(q, pos)
            k1 = rope(k, pos)
            Kc[pos][l] = k1
            Vc[pos][l] = v
            o = np.zeros((16, 128), np.float32)
            for hq in range(16):
                hk = hq // 2
                sc = np.array([q1[hq] @ Kc[t][l][hk] for t in range(pos + 1)])
                sc = sc / np.sqrt(128)
                p = np.exp(sc - sc.max())
                p /= p.sum()
                ov = np.zeros(128, np.float32)
                for t in range(pos + 1):
                    ov += p[t] * Vc[t][l][hk]
                o[hq] = ov
            attn = o.reshape(-1) @ W[L + 'self_attn.o_proj.weight'].T
            x2 = x + attn
            h2 = rms(x2[None], W[L + 'post_attention_layernorm.weight'])[0]
            g = h2 @ W[L + 'mlp.gate_proj.weight'].T
            u = h2 @ W[L + 'mlp.up_proj.weight'].T
            x = (x2 + (g / (1 + np.exp(-g)) * u) @ W[L + 'mlp.down_proj.weight'].T).astype(np.float32)
            cs8 = float(x[:8].sum())
            log.write(f'pos={pos} L={l} cs8={cs8:.6f} out0-3={x[0]:.6g} {x[1]:.6g} {x[2]:.6g} {x[3]:.6g} '
                      f'k0-1={k1.flatten()[0]:.6g} {k1.flatten()[1]:.6g}\n')
    log.close()
    print('ref done; log at /tmp/chain_ref.log')


if __name__ == '__main__':
    # patch: rope q inside main loop — redefine cleanly below
    main()
