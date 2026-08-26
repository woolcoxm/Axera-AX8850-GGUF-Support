#!/usr/bin/env python3
"""Fused multi-head attention graph for the AX8850 NPU.

Computes: out[h, d] = sum_t softmax(q[h,:] . K[h,t,:] * mask[t]) * V[h,t,:]

Compiled at FIXED max context; unused positions are masked to -inf so the
softmax ignores them. KV slices are runtime inputs (DMA'd from host per call)
so llama.cpp's host-side KV cache works unchanged.

Shapes:
  Q:    [H, D]        queries for all heads (flattened from [D*H, 1])
  K:    [H, T, D]     key cache slice per head
  V:    [H, T, D]     value cache slice per head
  mask: [T]            0 = keep, large negative = ignore
  out:  [H, D]         attention output for all heads
"""
import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)     # per-head dim
    p.add_argument("--ctx", type=int, default=512)    # max context for this engine
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    H, D, T = args.heads, args.dim, args.ctx
    out = args.out or f"attn_h{H}_d{D}_t{T}.onnx"

    # inputs
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [H, D])
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [H, T, D])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [H, T, D])
    m = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [T])
    out_vi = helper.make_tensor_value_info("out", TensorProto.FLOAT, [H, D])

    import numpy as np
    from onnx import numpy_helper
    ax1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ax1")
    ax01 = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="ax01")
    nodes = []
    # scores = Q @ K^T : [H, D] x [H, T, D] -> need [H, 1, D] @ [H, D, T]
    # ONNX MatMul broadcasts batch dims: [H, 1, D] @ [H, D, T] = [H, 1, T]
    nodes.append(helper.make_node("Unsqueeze", ["Q", "ax1"], ["Q3"]))
    nodes.append(helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 2, 1]))  # [H, D, T]
    nodes.append(helper.make_node("MatMul", ["Q3", "Kt"], ["scores"], name="qk"))  # [H, 1, T]

    # scale scores by 1/sqrt(D) then add mask
    scale = numpy_helper.from_array(
        np.array([1.0 / np.sqrt(D)], dtype=np.float32), name="scale")
    nodes.append(helper.make_node("Mul", ["scores", "scale"], ["scores_scaled"]))
    # mask: [T] -> [1, 1, T] to broadcast over heads
    nodes.append(helper.make_node("Unsqueeze", ["mask", "ax01"], ["mask3"]))
    nodes.append(helper.make_node("Add", ["scores_scaled", "mask3"], ["scores_masked"]))

    # softmax over T
    nodes.append(helper.make_node("Softmax", ["scores_masked"], ["probs"], axis=-1))

    # out = probs @ V : [H, 1, T] @ [H, T, D] = [H, 1, D]
    nodes.append(helper.make_node("MatMul", ["probs", "V"], ["out3"], name="sv"))
    nodes.append(helper.make_node("Squeeze", ["out3", "ax1"], ["out"]))

    graph = helper.make_graph(nodes, "attn", [q, k, v, m], [out_vi], [scale, ax1, ax01])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f"wrote {out}: heads={H} dim={D} ctx={T} (mask-based dynamic length)")


if __name__ == "__main__":
    main()
