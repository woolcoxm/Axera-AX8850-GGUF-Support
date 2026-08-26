#!/usr/bin/env python3
"""Build the acid-test patched axmodel.

Take cm0's axmodel file; replace ONLY the decoded positions (v3 nibble
positions, with corrected hi/lo parity) + scale entry words with the sn
build's bytes. Everything else stays cm0. If this patched engine computes
the same as the true sn engine, the unpatched residual regions (meta etc.)
do not affect the decode shape group -> the loader is viable as-is.
"""
import glob
import pickle

import numpy as np
import onnx
from onnx import numpy_helper

BLOB = 17346568
BLOB_OFF = 1570  # npu_params offset inside the axmodel file (verified)


def load_blob(p):
    m = onnx.load(p, load_external_data=False)
    for i in m.graph.initializer:
        if i.name == 'npu_params':
            return numpy_helper.to_array(i).astype(np.uint8)[:BLOB]
    raise RuntimeError


def main():
    cm0_ax = open('/tmp/cm0_out/qwen3_p128_l0_together.axmodel', 'rb').read()
    ref = load_blob('/tmp/cmdumps/sn_layer_dump.onnx')  # sn blob = target bytes
    base = np.frombuffer(cm0_ax, np.uint8, BLOB, BLOB_OFF)

    print('loading v3 table...')
    tab = pickle.load(open('/tmp/layer_layout_v3.pkl', 'rb'))
    # corrected parity: hi nibble = odd k, lo nibble = even k
    patched = np.frombuffer(cm0_ax, np.uint8).copy()  # writable copy
    n_hi = n_lo = 0
    for (m, r, k), (p, _half) in tab.items():
        byte_off = BLOB_OFF + p
        if k % 2 == 1:
            hi = ref[p] >> 4
            patched[byte_off] = (patched[byte_off] & 0x0F) | (hi << 4)
            n_hi += 1
        else:
            lo = ref[p] & 0x0F
            patched[byte_off] = (patched[byte_off] & 0xF0) | lo
            n_lo += 1
    print(f'nibbles patched: hi(odd k)={n_hi} lo(even k)={n_lo}')

    # scale entries: [i16][bf16] word positions from the sdA diff decode
    ent = pickle.load(open('/tmp/scale_entries.pkl', 'rb'))
    wpos = ent['wpos']
    b16ref = ref.view(np.uint16)
    for w in wpos:
        wb = np.frombuffer(b16ref[w].tobytes(), np.uint8)
        patched[BLOB_OFF + 2 * w:BLOB_OFF + 2 * w + 2] = wb
    print(f'scale words patched: {len(wpos)}')

    out = patched.tobytes()
    open('/tmp/acid_patched.axmodel', 'wb').write(out)
    # stats: how close is patched to the true sn file?
    sn_ax = bytearray(out)
    diff_ref = int((np.frombuffer(out, np.uint8, BLOB, BLOB_OFF) != ref).sum())
    diff_base = int((np.frombuffer(out, np.uint8, BLOB, BLOB_OFF) != base).sum())
    tot = int((base != ref).sum())
    print(f'patched-vs-sn remaining diff: {diff_ref} bytes (was {tot}); changed vs cm0: {diff_base}')
    print('wrote /tmp/acid_patched.axmodel')


if __name__ == '__main__':
    main()
