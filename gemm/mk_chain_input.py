#!/usr/bin/env python3
"""Make a fixed 6-token test input for the chain harness.

Random token ids (seeded); embeddings from HF; converted to bf16 and back to
f32 so the harness and the numpy reference see bit-identical inputs.
Outputs: /tmp/chain_in.bin (6 x 1024 bf16), /tmp/chain_ids.txt
"""
import json
import struct

import numpy as np


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


def f2b(f):
    u = f.astype(np.float32).view(np.uint32)
    bits = u >> 16
    rem = u & 0xFFFF
    ru = (rem > 0x8000) | ((rem == 0x8000) & ((bits & 1) == 1))
    return (bits + ru).astype(np.uint16)


def main():
    rng = np.random.default_rng(20260826)
    ids = rng.integers(0, 151936, size=6)
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {'model.embed_tokens.weight'})
    emb = W['model.embed_tokens.weight'][ids]          # [6, 1024] f32
    b = f2b(emb)                                        # bf16 round trip
    emb_rt = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
    b.tofile('/tmp/chain_in.bin')
    np.save('/tmp/chain_emb_rt.npy', emb_rt)
    with open('/tmp/chain_ids.txt', 'w') as f:
        f.write(' '.join(map(str, ids.tolist())))
    print('ids:', ids.tolist())
    print('emb[0][:4]:', emb_rt[0][:4])
    # also save full model weights needed by the reference in f32 for speed
    print('written /tmp/chain_in.bin (6x1024 bf16)')


if __name__ == '__main__':
    main()
