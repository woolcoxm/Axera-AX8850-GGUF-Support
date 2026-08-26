#!/usr/bin/env python3
"""Generate a GEMM ONNX graph with BOTH operands as runtime inputs.

This is the critical experiment for the ggml-axcl backend: Axera's own LLM
axmodels bake weights into per-layer graphs, but a ggml backend needs the
weight matrix as a dynamic input (every layer has different weights).

Graph:  Y[M, N] = X[M, K] @ W[K, N]
        X: float32 input  (activation, M=1 for decode)
        W: float32 input  (weight, transposed already: ggml MUL_MAT uses
                           src0[W, K] row-major with ne0=K, i.e. W^T in
                           BLAS terms)
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--out", type=str, default="gemm.onnx")
    p.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16"])
    args = p.parse_args()

    etype = TensorProto.FLOAT if args.dtype == "f32" else TensorProto.FLOAT16

    x = helper.make_tensor_value_info("X", etype, [args.m, args.k])
    w = helper.make_tensor_value_info("W", etype, [args.k, args.n])
    y = helper.make_tensor_value_info("Y", etype, [args.m, args.n])

    node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="gemm")

    graph = helper.make_graph(
        [node],
        "gemm",
        [x, w],
        [y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # keep compatible with older toolchains
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}: Y[{args.m},{args.n}] = X[{args.m},{args.k}] @ W[{args.k},{args.n}] ({args.dtype})")

    # reference data for correctness checking on device
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((args.m, args.k)).astype(np.float32)
    w_np = rng.standard_normal((args.k, args.n)).astype(np.float32)
    np.savez(args.out + ".npz", X=x_np, W=w_np, Y=x_np @ w_np)
    print(f"wrote {args.out}.npz reference inputs/outputs")


if __name__ == "__main__":
    main()
