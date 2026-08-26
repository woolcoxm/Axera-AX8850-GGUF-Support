#!/usr/bin/env python3
"""Emit the compact loader sidecar: layout_v4.bin.

Sections: 7 matrices (u64 byte-offsets per element, 0xFFFFFFFFFFFFFFFF =
unplaced) + 4 norms (u32 byte-offsets per element) + down-tail (u64
byte-offsets by element index). File offsets are ABSOLUTE in the engine
file (npu_params at 1570).
"""
import struct

import numpy as np

NAMES = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
NORMS = ['input_layernorm.weight', 'post_attention_layernorm.weight',
         'self_attn.q_norm.weight', 'self_attn.k_norm.weight']

tab = np.load('/tmp/real_layout.npz')
norms = np.load('/tmp/norm_slots.npz')
tail = np.load('/tmp/down_tail.npz')

out = bytearray()
out += b'AXL4'
out += struct.pack('<I', 1)          # version
out += struct.pack('<I', 7)          # matrices
out += struct.pack('<I', 4)          # norms

mats = {}
for m in NAMES:
    n, kk = SHAPES[m]
    arr = np.full(n * kk, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
    p = tab[f'{m}_pos'].astype(np.int64) * 2 + 1570
    idx = tab[f'{m}_r'].astype(np.int64) * kk + tab[f'{m}_k']
    arr[idx] = p
    mats[m] = arr

# merge down tail
bp = tail['bpos'].astype(np.int64)
el = tail['elem'].astype(np.int64)
mats['down'][el] = bp
unplaced = {m: int((mats[m] == np.uint64(0xFFFFFFFFFFFFFFFF)).sum()) for m in NAMES}
print('unplaced per matrix:', unplaced, 'total', sum(unplaced.values()))

for m in NAMES:
    n, kk = SHAPES[m]
    out += m.encode().ljust(8, b'\0') + struct.pack('<II', n, kk)
    out += mats[m].tobytes()

for nm in NORMS:
    key = nm.replace('.', '_').replace('-', '_')
    bp = norms[f'{key}_bpos'].astype(np.int64)
    el = norms[f'{key}_elem'].astype(np.int64)
    n = len(el)
    arr = np.full(n, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
    arr[el] = bp
    out += nm.encode().ljust(32, b'\0') + struct.pack('<I', n)
    out += arr.tobytes()

open('/home/kram/Desktop/Projects/LLMTest/gemm/layout_v4.bin', 'wb').write(bytes(out))
print('wrote layout_v4.bin', len(out), 'bytes')
