#!/usr/bin/env python3
"""Full FFN block with dynamic weights for the AX8850 NPU.

Computes: out = silu(h @ gate_w) * (h @ up_w) @ down_w

This is the proof-of-concept for Option B: a multi-matmul sub-graph
with weights as runtime inputs, compiled with mixed precision.
"""
import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=1024)   # n_embd
    p.add_argument("--inter", type=int, default=3072)    # intermediate
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    H, I = args.hidden, args.inter
    out = args.out or f"ffn_h{H}_i{I}.onnx"

    h = helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, H])
    gw = helper.make_tensor_value_info("gate_w", TensorProto.FLOAT, [H, I])
    uw = helper.make_tensor_value_info("up_w", TensorProto.FLOAT, [H, I])
    dw = helper.make_tensor_value_info("down_w", TensorProto.FLOAT, [I, H])
    out_vi = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, H])

    nodes = []
    # gate = h @ gate_w  -> [1, I]
    nodes.append(helper.make_node("MatMul", ["h", "gate_w"], ["gate"], name="gate_proj"))
    # up = h @ up_w      -> [1, I]
    nodes.append(helper.make_node("MatMul", ["h", "up_w"], ["up"], name="up_proj"))
    # silu(gate) = gate * sigmoid(gate)
    nodes.append(helper.make_node("Sigmoid", ["gate"], ["gate_sig"]))
    nodes.append(helper.make_node("Mul", ["gate", "gate_sig"], ["gate_silu"]))
    # act = gate_silu * up
    nodes.append(helper.make_node("Mul", ["gate_silu", "up"], ["act"]))
    # out = act @ down_w -> [1, H]
    nodes.append(helper.make_node("MatMul", ["act", "down_w"], ["out"], name="down_proj"))

    graph = helper.make_graph(nodes, "ffn_block", [h, gw, uw, dw], [out_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"wrote {out}: FFN block h={H} i={I} (3 matmuls + silu, all weights dynamic)")


if __name__ == "__main__":
    main()
