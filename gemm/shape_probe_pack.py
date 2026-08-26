#!/usr/bin/env python3
"""Generate + compile 6 code-encoding builds for a given (N,K) matmul shape.
Usage: shape_probe_pack.py N K   (logical weight [N,K])
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

P7 = Path('/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package')


def gen(N, K, tag, rbits, kbits):
    rng = np.random.default_rng(31337)
    dither = rng.integers(0, 13, (N, K))
    R, Kx = np.meshgrid(np.arange(N), np.arange(K), indexing='ij')
    code = np.zeros((N, K), np.int16)
    for j, (w, i) in enumerate(rbits + kbits):
        code += ((((R if w == 'r' else Kx) >> i) & 1) << j)
    V = 16 * code + dither
    V[:, 0] = 127
    W = V.astype(np.float32) / 127.0
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, N])
    g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['Y'], name='mm')], 'mm', [X], [Y],
                          initializer=[numpy_helper.from_array(np.ascontiguousarray(W.T), 'W')])
    p = f'/tmp/shape_{N}x{K}_{tag}.onnx'
    onnx.save(helper.make_model(g, opset_imports=[helper.make_operatorsetid('', 16)]), p)
    return p


def main():
    N, K = int(sys.argv[1]), int(sys.argv[2])
    rbits_total = int(np.ceil(np.log2(N)))
    kbits_total = int(np.ceil(np.log2(K)))
    bits = ([('r', i) for i in range(rbits_total)] + [('k', i) for i in range(kbits_total)])
    nbuilds = int(np.ceil(len(bits) / 3))
    print(f'{N}x{K}: {rbits_total}+{kbits_total} bits, {nbuilds} builds')
    for b in range(nbuilds):
        sel = bits[b*3:(b+1)*3]
        rb = [x for x in sel if x[0] == 'r']
        kb = [x for x in sel if x[0] == 'k']
        p = gen(N, K, f'b{b}', rb, kb)
        out = f'/tmp/shape_out_{N}x{K}_{b}'
        Path(out).mkdir(exist_ok=True)
        r = subprocess.run([f'{P7}/bin/pulsar2', 'build', '--input', p, '--config', f'/tmp/layout_cfg_{K}.json',
                            '--output_dir', out], capture_output=True, text=True, timeout=600,
                           env={'PATH': f'{P7}/bin:/usr/bin:/bin', 'FLOAT_MATMUL_USE_CONV_EU': '1'})
        ok = Path(out, 'compiled.axmodel').exists()
        print(f'  b{b} bits {sel}: {"OK" if ok else "FAIL"} {r.stderr[-200:] if not ok else ""}')


if __name__ == '__main__':
    main()
