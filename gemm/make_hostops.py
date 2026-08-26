#!/usr/bin/env python3
"""Small per-op engines for the device-resident decode chain.

Each graph has dynamic tensor inputs (weights/values flow in per call) so the
GGUF model stays the source of truth. All compiled with highest_mix_precision
+ precision_analysis(EndToEnd) -> fp16 dataflow, matching the verified-exact
matmul engines.

Ops:
  rope_q:  q[2048] + pos -> q'[2048]   (NEOX pairing, freq_base 1e6, Qwen3)
  rope_k:  k[1024] + pos -> k'[1024]
  add:     a[1024] + b[1024] -> out    (residual)
  rms_norm: x[1024] + w[1024] -> out   (eps inside)
  glu:     x[6144] -> out[3072]        (silu(x[:3072]) * x[3072:])
  vocab_fp16: x[1024] + W[1024,151936](f16) -> logits[151936]
"""
import argparse, numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

FREQ_BASE = 1e6  # Qwen3

def mk_out(path, graph):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print(f"wrote {path}")

def rope(name, dim, out):
    # x[dim], pos (scalar f32) -> rotated x
    half = dim // 2
    inv_freq = 1.0 / (FREQ_BASE ** (np.arange(0, dim, 2, dtype=np.float64) / dim))  # [half]
    theta_scale = numpy_helper.from_array(inv_freq.astype(np.float32), name="theta_scale")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [dim])
    p = helper.make_tensor_value_info("pos", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [dim])
    inits_slice = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="st0"),
        numpy_helper.from_array(np.array([half], dtype=np.int64), name="st1"),
        numpy_helper.from_array(np.array([dim], dtype=np.int64), name="st2"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="ax0i"),
    ]
    nodes = [
        helper.make_node("Mul", ["pos", "theta_scale"], ["theta"]),          # [half]
        helper.make_node("Cos", ["theta"], ["cos_t"]),
        helper.make_node("Sin", ["theta"], ["sin_t"]),
        helper.make_node("Slice", ["x", "st0", "st1", "ax0i"], ["x1"]),
        helper.make_node("Slice", ["x", "st1", "st2", "ax0i"], ["x2"]),
        helper.make_node("Mul", ["x1", "cos_t"], ["a"]),
        helper.make_node("Mul", ["x2", "sin_t"], ["b"]),
        helper.make_node("Sub", ["a", "b"], ["y1"]),
        helper.make_node("Mul", ["x1", "sin_t"], ["c"]),
        helper.make_node("Mul", ["x2", "cos_t"], ["d"]),
        helper.make_node("Add", ["c", "d"], ["y2"]),
        helper.make_node("Concat", ["y1", "y2"], ["y"], axis=0),
    ]
    inits = [theta_scale] + inits_slice
    g = helper.make_graph(nodes, name, [x, p], [y], inits)
    mk_out(out, g)

def add(out):
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1024])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1024])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])
    g = helper.make_graph([helper.make_node("Add", ["a", "b"], ["y"])], "add", [a, b], [y])
    mk_out(out, g)

def rms_norm(out):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1024])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1024])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])
    eps = numpy_helper.from_array(np.array([1e-6], dtype=np.float32), name="eps")
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["xx"]),
        helper.make_node("ReduceMean", ["xx"], ["ms"], keepdims=0),
        helper.make_node("Add", ["ms", "eps"], ["mse"]),
        helper.make_node("Sqrt", ["mse"], ["rms"]),
        helper.make_node("Div", ["x", "rms"], ["xn"]),
        helper.make_node("Mul", ["xn", "w"], ["y"]),
    ]
    g = helper.make_graph(nodes, "rms_norm", [x, w], [y], [eps])
    mk_out(out, g)

def glu(out):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6144])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3072])
    nodes = [
        helper.make_node("Slice", ["x", "c0", "c1", "axz"], ["g"]),
        helper.make_node("Slice", ["x", "c1", "c2", "axz"], ["u"]),
        helper.make_node("Sigmoid", ["g"], ["sig"]),
        helper.make_node("Mul", ["g", "sig"], ["silu"]),
        helper.make_node("Mul", ["silu", "u"], ["y"]),
    ]
    inits = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="c0"),
        numpy_helper.from_array(np.array([3072], dtype=np.int64), name="c1"),
        numpy_helper.from_array(np.array([6144], dtype=np.int64), name="c2"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="axz"),
    ]
    g = helper.make_graph(nodes, "glu", [x], [y], inits)
    mk_out(out, g)

def vocab_fp16(out):
    # W arrives as fp16 (halves the per-token CMM read of 622MB -> 311MB)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1024])
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT16, [1024, 151936])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 151936])
    nodes = [
        helper.make_node("Cast", ["W"], ["Wf"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["x", "Wf"], ["y"]),
    ]
    g = helper.make_graph(nodes, "vocab", [x, w], [y])
    mk_out(out, g)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="hostops")
    args = ap.parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    rope("rope_q", 2048, f"{args.outdir}/rope_q_d2048.onnx")
    rope("rope_k", 1024, f"{args.outdir}/rope_k_d1024.onnx")
    add(f"{args.outdir}/add_h1024.onnx")
    rms_norm(f"{args.outdir}/rmsnorm_h1024.onnx")
    glu(f"{args.outdir}/glu_h3072.onnx")
    vocab_fp16(f"{args.outdir}/vocab_fp16.onnx")
