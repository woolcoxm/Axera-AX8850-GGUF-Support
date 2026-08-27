#!/usr/bin/env python3
"""Patch GGUF weights into vendor w8a16 int8 engines (same-architecture mode).

Dequantizes GGUF tensors (Q8_0/Q4_K/Q6_K, validated against the ggml
reference dequant), quantizes per-row symmetric int8 against the ENGINE'S
OWN scales (least-squares vs the engine's stored int8s — correct for
same-model GGUFs; scale entries untouched), scatters via the decoded
claims/anchor/fine tables with structural guards, and emits a complete
GGML_AXCL_LAYER_DIR engine set.

Usage: gguf_patch_w8.py <model.gguf> <vendor_engine_dir> <out_dir> [first_layer] [last_layer]
"""
import os
import pickle
import shutil
import sys
import time
import numpy as np

sys.path.insert(0, '/home/kram/Desktop/Projects/LLMTest/llama.cpp/gguf-py')
from gguf import GGUFReader

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'baked', 'v52_markers') + os.sep
NAMES = {'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output',
         'gate': 'ffn_gate', 'up': 'ffn_up', 'down': 'ffn_down'}
SHAPES = {'q': (2048, 1024), 'k': (1024, 1024), 'v': (1024, 1024), 'o': (1024, 2048),
          'gate': (3072, 1024), 'up': (3072, 1024), 'down': (1024, 3072)}
MATS = list(NAMES)
NP_OFF = 5035
NP_LEN = 19_226_120
NLAYERS = 28


def vendor_pos(p):
    p = np.asarray(p, np.int64)
    return np.where(p < 1_289_094, p + 8192,
                    np.where(p < 3_386_411, p,
                             np.where(p < 4_523_255, p + 9216,
                                      np.where(p < 8_936_951, p + 9728, p + 13824))))


# ---------------- dequant (ggml reference semantics) ----------------
def fp16(h):
    h = np.asarray(h, np.uint16).astype(np.uint32)
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    out = np.where(exp == 0, (frac / 1024.0) * 2.0 ** -14,
                   (1.0 + frac / 1024.0) * 2.0 ** (exp.astype(np.int32) - 15))
    return np.where(sign == 1, -out, out).astype(np.float32)


def deq_q8_0(data, n_out, n_in):
    b = np.frombuffer(data, np.uint8).reshape(n_out, n_in // 32, 34)
    d = fp16(b[:, :, :2].reshape(-1, 2).copy().view(np.uint16)).reshape(n_out, -1)
    q = b[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (d[:, :, None] * q).reshape(n_out, n_in)


def get_scale_min_k4(scales):
    """(n, 12) -> sc, m each (n, 8); exact port of ggml helper."""
    n = scales.shape[0]
    sc = np.zeros((n, 8), np.float32)
    m = np.zeros((n, 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            m[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, m


def deq_q4_k(data, n_out, n_in):
    b = np.frombuffer(data, np.uint8).reshape(n_out, n_in // 256, 144)
    d = fp16(b[:, :, 0:2].reshape(-1, 2).copy().view(np.uint16)).reshape(n_out, -1)
    dmin = fp16(b[:, :, 2:4].reshape(-1, 2).copy().view(np.uint16)).reshape(n_out, -1)
    sc, m = get_scale_min_k4(b[:, :, 4:16].reshape(-1, 12))
    sc = sc.reshape(n_out, -1, 8)
    m = m.reshape(n_out, -1, 8)
    qs = b[:, :, 16:144]
    out = np.zeros((n_out, n_in // 256, 256), np.float32)
    for j in range(4):                      # 64-weight groups
        q = qs[:, :, j * 32:(j + 1) * 32]
        lo = (q & 0xF).astype(np.float32)
        hi = (q >> 4).astype(np.float32)
        out[:, :, j*64:j*64+32] = (d * sc[:, :, 2*j])[:, :, None] * lo - (dmin * m[:, :, 2*j])[:, :, None]
        out[:, :, j*64+32:j*64+64] = (d * sc[:, :, 2*j+1])[:, :, None] * hi - (dmin * m[:, :, 2*j+1])[:, :, None]
    return out.reshape(n_out, n_in)


def deq_q6_k(data, n_out, n_in):
    b = np.frombuffer(data, np.uint8).reshape(n_out, n_in // 256, 210)
    ql = b[:, :, 0:128]
    qh = b[:, :, 128:192]
    sc = b[:, :, 192:208].copy().view(np.int8).astype(np.float32)
    d = fp16(b[:, :, 208:210].reshape(-1, 2).copy().view(np.uint16)).reshape(n_out, -1)
    out = np.zeros((n_out, n_in // 256, 256), np.float32)
    for n in (0, 128):
        nb = n // 128                       # block half index
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[:, :, nb*64 + l] & 0xF) | (((qh[:, :, nb*32 + l] >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((ql[:, :, nb*64 + l + 32] & 0xF) | (((qh[:, :, nb*32 + l] >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((ql[:, :, nb*64 + l] >> 4) | (((qh[:, :, nb*32 + l] >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((ql[:, :, nb*64 + l + 32] >> 4) | (((qh[:, :, nb*32 + l] >> 6) & 3) << 4)).astype(np.int32) - 32
            s = nb * 8
            out[:, :, n + l] = d * sc[:, :, s + is_ + 0] * q1
            out[:, :, n + l + 32] = d * sc[:, :, s + is_ + 2] * q2
            out[:, :, n + l + 64] = d * sc[:, :, s + is_ + 4] * q3
            out[:, :, n + l + 96] = d * sc[:, :, s + is_ + 6] * q4
    return out.reshape(n_out, n_in)


def dequant_tensor(t):
    n_in, n_out = int(t.shape[0]), int(t.shape[1])
    data = bytes(t.data)
    tt = int(t.tensor_type)
    if tt == 8:
        return deq_q8_0(data, n_out, n_in)
    if tt == 12:
        return deq_q4_k(data, n_out, n_in)
    if tt == 14:
        return deq_q6_k(data, n_out, n_in)
    raise ValueError(f'tensor type {tt} not supported')


def read_st(path, wanted):
    import json as _json
    import struct as _struct
    with open(path, 'rb') as f:
        n = _struct.unpack('<Q', f.read(8))[0]
        hdr = _json.loads(f.read(n))
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


# ---------------- patch machinery ----------------
def load_tables():
    claims = np.load(D + 'v52_claims.npz')['claims']
    anchors = pickle.load(open(D + 'v52_anchor_rows.pkl', 'rb'))
    fine = np.load(D + 'v52_fine.npz')
    fine_map = {}
    okf = fine['fine_pos'] >= 0
    for pp, fp in zip(fine['pair_pos'][okf].tolist(), fine['fine_pos'][okf].tolist()):
        fine_map[pp] = fp
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
    anch_by_mat = {mi: {} for mi in range(len(MATS))}
    for p, (m, r) in anchors.items():
        anch_by_mat[MATS.index(m)][r] = p
    return claims, anch_by_mat, fine_map, scale_mask


def scatter_nibbles(out, pos, half, hi_nib, lo_nib, fine_pos=None):
    mask = np.where(half == 1, np.uint8(0xF0), np.uint8(0x0F))
    val = np.where(half == 1, (hi_nib << 4).astype(np.uint8), hi_nib)
    np.bitwise_and.at(out, pos, (~mask).astype(np.uint8))
    np.bitwise_or.at(out, pos, val)
    if fine_pos is not None:
        fval = np.where(half == 1, (lo_nib << 4).astype(np.uint8), lo_nib)
        np.bitwise_and.at(out, fine_pos, (~mask).astype(np.uint8))
        np.bitwise_or.at(out, fine_pos, fval)


def patch_layer(eng_bytes, Wg, Sref, Sref_whf, claims, anch_by_mat, fine_map, scale_mask):
    ven = np.frombuffer(bytes(eng_bytes), np.uint8).copy()
    pv = vendor_pos(claims['pos'])
    b = ven[pv]
    coarse = np.where(claims['half'] == 1, b >> 4, b & 15).astype(np.int16) - 8
    finen = np.where(claims['half'] == 1, ven[pv - 18] >> 4, ven[pv - 18] & 15).astype(np.int16)
    q8v = (coarse << 4) | finen
    q8v = np.where(q8v > 127, q8v - 256, q8v).astype(np.float64)

    out = ven.copy()
    stats = [0, 0]
    for mi, m in enumerate(MATS):
        sel = np.flatnonzero(claims['mat'] == mi)
        n, kk = SHAPES[m]
        w = Wg[m]
        srow = Sref[m]
        rr = claims['r'][sel].astype(np.int64)
        wv = w.reshape(-1)[rr * kk + claims['k'][sel]]
        whf = Sref_whf[m].reshape(-1)[rr * kk + claims['k'][sel]]
        # the engine dequantizes with ITS stored per-row scale = rowmax(HF)/127;
        # an LS-fit scale is systematically ~0.2% low and that bias compounds
        # across 28 layers -> degenerate output. Use the engine's scale.
        q8n = np.clip(np.round(wv / srow[rr]), -127, 127).astype(np.int32)
        q8ref = np.clip(np.round(whf / srow[rr]), -127, 127).astype(np.int32)
        # read-verification against engine's stored int8 (tolerance 3 steps)
        ver = (np.abs(q8n - q8v[sel].astype(np.int32)) >= 2) & (np.abs(q8ref - q8v[sel].astype(np.int32)) <= 1)
        stats[0] += int(ver.sum()); stats[1] += len(ver)
        sel = sel[ver]
        rr = rr[ver]
        posm = claims['pos'][sel].astype(np.int64)
        pvv = pv[sel]
        fpv = np.array([fine_map.get(int(q), -1) for q in posm], np.int64)
        okf = fpv >= 0
        fpos_v = np.full(len(posm), -1, np.int64)
        fpos_v[okf] = vendor_pos(fpv[okf])
        okf &= ~scale_mask[np.where(fpos_v >= 0, fpos_v, 0)]
        okf &= ven[np.where(fpos_v >= 0, fpos_v, 0)] != 0
        scatter_nibbles(out, pvv, claims['half'][sel],
                        ((q8n[ver] >> 4) + 8).astype(np.uint8), (q8n[ver] & 15).astype(np.uint8))
        sub = np.flatnonzero(okf)
        if len(sub):
            scatter_nibbles(out, pvv[sub], claims['half'][sel][sub],
                            ((q8n[ver] >> 4) + 8).astype(np.uint8)[sub],
                            (q8n[ver] & 15).astype(np.uint8)[sub], fine_pos=fpos_v[sub])
        rows = np.fromiter(anch_by_mat[mi].keys(), np.int64)
        pmark = np.fromiter(anch_by_mat[mi].values(), np.int64)
        if len(rows):
            pvA = vendor_pos(pmark)
            q8a = np.clip(np.round(w[rows, 0] / srow[rows]), -127, 127).astype(np.int32)
            stored_top = (ven[pvA] & 15).astype(np.int16) - 8
            keepA = ~scale_mask[pvA] & (ven[pvA] != 0) & (np.abs((q8a >> 4) - stored_top) >= 1)
            np.bitwise_and.at(out, pvA[keepA], np.uint8(0xF0))
            np.bitwise_or.at(out, pvA[keepA], ((q8a >> 4) + 8).astype(np.uint8)[keepA])
    return out.tobytes(), stats


def main():
    gguf_path, vendor_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    l0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    l1 = int(sys.argv[5]) if len(sys.argv) > 5 else NLAYERS - 1
    os.makedirs(out_dir, exist_ok=True)
    reader = GGUFReader(gguf_path)
    tmap = {str(t.name): t for t in reader.tensors}
    claims, anch_by_mat, fine_map, scale_mask = load_tables()
    HFREF = {  # engine scale = rowmax(HF reference)/127 (engines are byte-equal
               # to a 5.2 build of this checkpoint)
        'q': 'self_attn.q_proj.weight', 'k': 'self_attn.k_proj.weight',
        'v': 'self_attn.v_proj.weight', 'o': 'self_attn.o_proj.weight',
        'gate': 'mlp.gate_proj.weight', 'up': 'mlp.up_proj.weight',
        'down': 'mlp.down_proj.weight'}

    t_start = time.time()
    for L in range(l0, l1 + 1):
        Wg = {}
        for m, gname in NAMES.items():
            t = tmap[f'blk.{L}.{gname}.weight']
            n, kk = SHAPES[m]
            w = dequant_tensor(t)
            assert w.shape == (n, kk), (m, w.shape)
            Wg[m] = w.astype(np.float64)
        WHF = read_st('/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B/model.safetensors',
                      {f'model.layers.{L}.{n}' for n in HFREF.values()})
        Sref = {m: np.abs(WHF[f'model.layers.{L}.{n}'].astype(np.float64)).max(1) / 127.0
                for m, n in HFREF.items()}
        WHF_W = {m: WHF[f'model.layers.{L}.{n}'].astype(np.float64) for m, n in HFREF.items()}
        eng = open(os.path.join(vendor_dir, f'qwen3_p128_l{L}_together.axmodel'), 'rb').read()
        patched, (nv, nt) = patch_layer(eng[NP_OFF:NP_OFF + NP_LEN], Wg, Sref, WHF_W,
                                        claims, anch_by_mat, fine_map, scale_mask)
        raw = bytearray(eng)
        raw[NP_OFF:NP_OFF + NP_LEN] = patched
        open(os.path.join(out_dir, f'qwen3_p128_l{L}_together.axmodel'), 'wb').write(raw)
        ch = int((np.frombuffer(patched[:1000], np.uint8) != 0).sum() and
                 sum(a != b for a, b in zip(patched[:2000], eng[NP_OFF:NP_OFF+2000])))
        print(f'layer {L:2d}: verified {nv:,}/{nt:,} ({100*nv/nt:.2f}%), '
              f'{time.time()-t_start:.0f}s', flush=True)
    # copy the post engine and package files if layer 0 was in range
    if l0 == 0:
        for f in os.listdir(vendor_dir):
            if 'post' in f or f.endswith(('.json', '.txt', '.bin')):
                shutil.copy(os.path.join(vendor_dir, f), os.path.join(out_dir, f))
    print('done ->', out_dir)


if __name__ == '__main__':
    main()
