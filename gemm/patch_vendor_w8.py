#!/usr/bin/env python3
"""Patch HF weights into a vendor w8a16 engine (identity/validation path).

Layout: marker claims + anchors mapped to vendor npu_params positions by
the piecewise shift map. Weights are UNFOLDED (5.2 keeps norm ops at
runtime), per-row symmetric scales; scales least-squares-fit from the
engine's own int8s (== rowmax/127 to 0.02%).
Storage per element pair (k even, k+1 odd): coarse byte at p holds the
two top nibbles (v=(q8>>4)+8), fine byte at p-18 the two low nibbles.

Usage: patch_vendor_w8.py <engine.axmodel> <layer> <out.axmodel>
"""
import json
import os
import pickle
import struct
import sys
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baked', 'v52_markers') + os.sep
NAMES = {'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight', 'v': 'self_attn.v_proj.weight',
         'o': 'self_attn.o_proj.weight', 'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
         'down': 'mlp.down_proj.weight'}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
MATS = list(NAMES)
NP_OFF = 5035
NP_LEN = 19_226_120


def vendor_pos(p):
    # exact map derived from claim-order monotonicity: three insertion events
    # (+8192 at R4 start, +9216 at R6 start, +512 @4,523,255, +4096 @8,936,951)
    p = np.asarray(p, np.int64)
    return np.where(p < 1_289_094, p + 8192,
                    np.where(p < 3_386_411, p,
                             np.where(p < 4_523_255, p + 9216,
                                      np.where(p < 8_936_951, p + 9728, p + 13824))))


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
            else:
                arr = np.frombuffer(raw, np.float32)
            out[name] = arr.reshape(meta['shape'])
    return out


def scatter_nibbles(out, pos, half, hi_nib, lo_nib, fine_pos=None):
    mask = np.where(half == 1, np.uint8(0xF0), np.uint8(0x0F))
    val = np.where(half == 1, (hi_nib << 4).astype(np.uint8), hi_nib)
    np.bitwise_and.at(out, pos, (~mask).astype(np.uint8))
    np.bitwise_or.at(out, pos, val)
    if fine_pos is not None:
        fval = np.where(half == 1, (lo_nib << 4).astype(np.uint8), lo_nib)
        np.bitwise_and.at(out, fine_pos, (~mask).astype(np.uint8))
        np.bitwise_or.at(out, fine_pos, fval)


def main():
    eng_path, layer, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    raw = bytearray(open(eng_path, 'rb').read())
    ven = np.frombuffer(bytes(raw[NP_OFF:NP_OFF + NP_LEN]), np.uint8)

    L = f'model.layers.{layer}.'
    W = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                {L + NAMES[m] for m in MATS})
    claims = np.load(D + 'v52_claims.npz')['claims']
    anchors = pickle.load(open(D + 'v52_anchor_rows.pkl', 'rb'))
    fine = np.load(D + 'v52_fine.npz')

    # exact scale-entry mask from the real mixamp pair: real_l1 (== vendor l1)
    # vs realmix_l1 (same weights x0.5 -> int8s identical, scales halved).
    # Clusters dilated +/-8B for safety. Claims targeting these bytes are
    # excluded (writing scale entries corrupts whole-group dequant).
    ra = np.frombuffer(open(D + 'real_l1_params.bin', 'rb').read(), np.uint8)
    rb = np.frombuffer(open(D + 'realmix_l1_params.bin', 'rb').read(), np.uint8)
    dd = np.flatnonzero(ra != rb)
    scale_mask = np.zeros(NP_LEN, bool)
    if len(dd):
        gaps = np.diff(dd)
        cuts = np.flatnonzero(gaps > 16)
        st = np.concatenate(([0], cuts + 1)); en = np.concatenate((cuts, [len(dd) - 1]))
        for s, e in zip(st, en):
            scale_mask[max(0, int(dd[s]) - 8):min(NP_LEN, int(dd[e]) + 9)] = True
    print(f'scale-entry mask: {int(scale_mask.sum()):,} bytes '
          f'({len(st)} clusters)')
    drop = scale_mask[vendor_pos(claims['pos'])]
    # insertion-boundary claims: the first pair past each inserted scale-block
    # region maps onto the block edge (a scale entry) -> detonates the engine.
    posm_all = claims['pos'].astype(np.int64)
    drop |= (posm_all == 4_523_255) | (posm_all == 8_936_951)
    # inactive-slot guard: 0x00 bytes are unused weight slots; activating one
    # corrupts the engine
    drop |= ven[vendor_pos(posm_all)] == 0
    # insertion-neighborhood guard: claim pairs mapping near the two inserted
    # scale-block regions land on structural edge bytes (empirically 3 detonators
    # found there) — drop writes into a +-2KB window around each insertion.
    vend_all = vendor_pos(posm_all)
    drop |= ((vend_all >= 4_530_400) & (vend_all <= 4_535_000)) \
         | ((vend_all >= 8_945_600) & (vend_all <= 8_952_800))
    print(f'claims dropped (scale/inactive/boundary): {int(drop.sum()):,}')
    keep = ~drop
    claims = claims[keep]

    # per-byte fine target (marker space) for dither-verified pairs only
    fine_map = {}
    okf = fine['fine_pos'] >= 0
    for pp, fp in zip(fine['pair_pos'][okf].tolist(), fine['fine_pos'][okf].tolist()):
        fine_map[pp] = fp
    print(f'fine targets known for {len(fine_map):,} pairs '
          f'({len(fine["pair_pos"]) - len(fine_map):,} pairs left untouched)')

    pv = vendor_pos(claims['pos'])
    b = ven[pv]
    coarse = np.where(claims['half'] == 1, b >> 4, b & 15).astype(np.int16) - 8
    fb = ven[pv - 18]
    finen = np.where(claims['half'] == 1, fb >> 4, fb & 15).astype(np.int16)
    q8v = (coarse << 4) | finen
    q8v = np.where(q8v > 127, q8v - 256, q8v).astype(np.float64)

    # read-verification: drop claims whose stored top nibble differs from the
    # RTN quantization of the reference weights by >1 step. Verified claims
    # are genuine weight slots; failures are mis-mapped targets (scale
    # entries, inactive slots, insertion-edge bytes) or GPTQ-corrected
    # elements (~0.03%) — both safely left unpatched.
    keep_ver = np.ones(len(claims), bool)
    for mi, m in enumerate(MATS):
        selm = np.flatnonzero(claims['mat'] == mi)
        if not len(selm):
            continue
        w = W[L + NAMES[m]].astype(np.float64)
        s = np.abs(w).max(1) / 127.0
        rr = claims['r'][selm].astype(np.int64)
        want = (np.clip(np.round(w[rr, claims['k'][selm]] / s[rr]), -127, 127).astype(np.int64) >> 4).astype(np.int16)
        got = coarse[selm]
        keep_ver[selm] = np.abs(want - got) <= 1
    print(f'read-verified claims: {int(keep_ver.sum()):,} / {len(claims):,} '
          f'({100*keep_ver.mean():.3f}%)')
    claims = claims[keep_ver]
    pv = pv[keep_ver]
    b = ven[pv]
    coarse = coarse[keep_ver]
    fb = ven[pv - 18]
    finen = finen[keep_ver]
    q8v = q8v[keep_ver]

    # anchor (m, r) -> marker position, grouped by matrix for fast lookup
    anch_by_mat = {mi: {} for mi in range(len(MATS))}
    for p, (m, r) in anchors.items():
        anch_by_mat[MATS.index(m)][r] = p

    out = ven.copy()
    agree = tot = 0
    for mi, m in enumerate(MATS):
        sel = claims['mat'] == mi
        n, kk = SHAPES[m]
        w = W[L + NAMES[m]].astype(np.float64)
        rr = claims['r'][sel].astype(np.int64)
        wv = w.reshape(-1)[rr * kk + claims['k'][sel]]
        num = np.bincount(rr, weights=wv * q8v[sel], minlength=n)
        den = np.bincount(rr, weights=q8v[sel] ** 2, minlength=n)
        s = num / np.maximum(den, 1e-30)

        q8n = np.clip(np.round(wv / (np.abs(w).max(1) / 127.0)[rr]), -127, 127).astype(np.int32)
        agree += int((((q8n >> 4) == (q8v[sel].astype(np.int32) >> 4)).sum()))
        tot += int(sel.sum())
        # fine writes only where the pair's fine position is dither-verified
        # AND the fine target is not inside a scale block
        posm = claims['pos'][sel].astype(np.int64)
        fpv = np.array([fine_map.get(int(q), -1) for q in posm], np.int64)
        fpos_v = np.full(len(posm), -1, np.int64)
        okf = fpv >= 0
        fpos_v[okf] = vendor_pos(fpv[okf])
        okf &= ~scale_mask[np.where(fpos_v >= 0, fpos_v, 0)]
        okf &= ven[np.where(fpos_v >= 0, fpos_v, 0)] != 0
        okf &= ~(((fpos_v >= 4_530_400) & (fpos_v <= 4_535_000))
                 | ((fpos_v >= 8_945_600) & (fpos_v <= 8_952_800))
                 | (fpos_v < 0))
        sel_pos = pv[sel]
        scatter_nibbles(out, sel_pos, claims['half'][sel],
                        ((q8n >> 4) + 8).astype(np.uint8), (q8n & 15).astype(np.uint8),
                        fine_pos=None)
        sub = np.flatnonzero(okf)
        if len(sub):
            scatter_nibbles(out, sel_pos[sub], claims['half'][sel][sub],
                            ((q8n >> 4) + 8).astype(np.uint8)[sub], (q8n & 15).astype(np.uint8)[sub],
                            fine_pos=fpos_v[sub])

        # anchors: k=0 of each row, coarse only, skip scale-block targets
        rows = np.fromiter(anch_by_mat[mi].keys(), np.int64)
        pmark = np.fromiter(anch_by_mat[mi].values(), np.int64)
        if len(rows):
            pvA = vendor_pos(pmark)
            keepA = ~scale_mask[pvA]
            q8a = np.clip(np.round(w[rows, 0] / (np.abs(w).max(1) / 127.0)[rows]), -127, 127).astype(np.int32)
            # anchor sits in the LO nibble of its byte (k=0 is even)
            np.bitwise_and.at(out, pvA[keepA], np.uint8(0xF0))
            np.bitwise_or.at(out, pvA[keepA], ((q8a >> 4) + 8).astype(np.uint8)[keepA])
    print(f'int8 top-nibble agreement (RTN vs engine): {agree/tot:.4f} over {tot:,} elements')

    diff = out != ven
    print(f'bytes changed: {int(diff.sum()):,} / {NP_LEN:,} ({100*diff.mean():.2f}%)')
    raw[NP_OFF:NP_OFF + NP_LEN] = out.tobytes()
    open(out_path, 'wb').write(raw)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
