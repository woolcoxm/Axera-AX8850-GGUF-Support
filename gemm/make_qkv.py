#!/usr/bin/env python3
"""QKV projection engine: RMS norm + 3 parallel matmuls in ONE graph.

hidden [1,H] → rms_norm(hidden, norm_w) → h_normed
h_normed @ q_w [H,H]  → q [1,H]
h_normed @ k_w [H,H/2] → k [1,H/2]
h_normed @ v_w [H,H/2] → v [1,H/2]

The parallel structure works on the NPU (proven); sequential chains don't.
"""
import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--kv", type=int, default=512)  # kv dim (GQA)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    H, KV = args.hidden, args.kv
    out = args.out or f"qkv_h{H}_kv{KV}.onnx"

    h = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, H])
    nw = helper.make_tensor_value_info("norm_w", TensorProto.FLOAT, [H])
    qw = helper.make_tensor_value_info("q_w", TensorProto.FLOAT, [H, H])
    kw = helper.make_tensor_value_info("k_w", TensorProto.FLOAT, [H, KV])
    vw = helper.make_tensor_value_info("v_w", TensorProto.FLOAT, [H, KV])

    q_out = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, H])
    k_out = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, KV])
    v_out = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, KV])

    nodes = []
    # RMS norm: rsqrt(mean(x^2) + eps) * x * w
    one_over_h = numpy_helper.from_array(np.array([1.0 / H], dtype=np.float32), name="inv_h")
    eps = numpy_helper.from_array(np.array([1e-5], dtype=np.float32), name="eps")

    nodes.append(helper.make_node("Mul", ["hidden", "hidden"], ["x2"]))       # x^2
    nodes.append(helper.make_node("ReduceMean", ["x2"], ["x2m"], axes=[-1]))  # mean(x^2)
    nodes.append(helper.make_node("Add", ["x2m", "eps"], ["x2me"]))
    nodes.append(helper.make_node("Sqrt", ["x2me"], ["rms"]))
    nodes.append(helper.make_node("Div", ["hidden", "rms"], ["x_norm"]))       # x/rms
    nodes.append(helper.make_node("Mul", ["x_norm", "norm_w"], ["h_normed"])) # * w

    # 3 parallel projections from h_normed
    nodes.append(helper.make_node("MatMul", ["h_normed", "q_w"], ["q"], name="q_proj"))
    nodes.append(helper.make_node("MatMul", ["h_normed", "k_w"], ["k"], name="k_proj"))
    nodes.append(helper.make_node("MatMul", ["h_normed", "v_w"], ["v"], name="v_proj"))

    inits = [one_over_h, eps]
    # actually need axes for ReduceMean as an input in opset 13
    axes = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes_rm")
    inits.append(axes)

    graph = helper.make_graph(nodes, "qkv_block", [h, nw, qw, kw, vw], [q_out, k_out, v_out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"wrote {out}: QKV+norm h={H} kv={KV} (rms_norm + 3 parallel matmuls)")


if __name__ == "__main__":
    main()
