#!/usr/bin/env python3
"""Validate sign semantics using the sn build (odd-k weights negated).

Base W = (16c+d)/127 >= 0; sn: odd k negated. q8_base = 16c+d (0..127).
q8_sn = -(16c+d) for odd k. Predictions:
  nibble pos: nib = (q8 >> 4) + 8 (arithmetic shift, offset binary)
  8-bit pos:  byte = 128 + q8
Even k must be byte-identical to cm0.
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']


def load(p):
    m = onnx.load(p, load_external_data=False)
    for i in m.graph.initializer:
        if i.name == 'npu_params':
            return numpy_helper.to_array(i).astype(np.uint8)[:BLOB]


base = load(sorted(glob.glob('/tmp/cmdumps/cm0_layer_*.onnx'))[0])
sn = load('/tmp/cmdumps/sn_layer_dump.onnx')
print('blobs loaded, diff bytes:', int((base != sn).sum()))

tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
print('table loaded')

rng = np.random.default_rng(7)
stat = {'hi_same': 0, 'hi_diff': 0, 'lo_pred_ok': 0, 'lo_pred_bad': 0, 'lo_base_ok': 0}
bad_samples = []
for tag in MATS:
    keys = [k for k in tab if k[0] == tag]
    sample = rng.choice(len(keys), size=min(40000, len(keys)), replace=False)
    for si in sample:
        m_, r, k = keys[si]
        pos, half = tab[(m_, r, k)]
        c = r & 7  # cm0 code for bits 0-2 (valid for all matrices, rnb>=10)
        # dither unknown here (not needed: only even/odd k behavior tested)
        if k % 2 == 0:
            continue
        # odd k: negated in sn
        # nibble predicted: q8_sn in [-124,0]: nib = (q8>>4)+8, q8 in
        # [-128,-113]->nib 0 ... [-16,-1]->nib 7..? we test BOTH formulas
        pass
    # vector approach below

# vectorized: build arrays for odd-k table entries
pos_arr, r_arr, k_arr = [], [], []
for (m_, r, k), (p, half) in tab.items():
    if m_ == 'q' and k % 2 == 1:
        pos_arr.append(p)
        r_arr.append(r)
        k_arr.append(k)
pos_arr = np.array(pos_arr); r_arr = np.array(r_arr); k_arr = np.array(k_arr)
print('odd-k q entries:', len(pos_arr))

got_lo = sn[pos_arr] & 15
base_lo = base[pos_arr] & 15
# q8_sn = -(16*(r&7) + d) for dither d in 0..12: possible q8 values per row
r7 = r_arr & 7
# nibble = ((-q) >> 4) + 8 for q = 16c + d:
#   d>0: (-q)>>4 = -(c+1)  -> nib = 7 - c
#   d=0: (-q)>>4 = -c      -> nib = 8 - c
pred_lo_posd = 7 - r7
pred_lo_d0 = 8 - r7
ok_posd = (got_lo == pred_lo_posd)
ok_d0 = (got_lo == pred_lo_d0)
print(f'lo nibble matches 7-c (d>0 rows): {int(ok_posd.sum())}/{len(pos_arr)}')
print(f'lo nibble matches 8-c (d=0 rows): {int(ok_d0.sum())}/{len(pos_arr)}')
print('base lo (positive): ', np.bincount(base_lo, minlength=16))
print('sn lo distribution:', np.bincount(got_lo, minlength=16))
# check what fraction is explained by sign-magnitude alternative: nib = 8-|q4|?
# |q4| for -q: ceil(q/16)... test nib = 8 + ((q8_sn - 1) >> 4)?? just report a crosstab
ct = np.zeros((16, 16), int)
for b7, gl in zip(np.clip(7 - r7, 0, 15), got_lo):
    ct[b7, gl] += 1
print('crosstab pred(7-c) x got (nonzero rows):')
for i in range(16):
    if ct[i].sum() > 100:
        print(f'  pred {i}: ', ct[i].tolist())
