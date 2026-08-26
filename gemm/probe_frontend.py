#!/usr/bin/env python3
"""Probe which (model_type, op) combinations produce all-NPU axmodels.

Builds: X(int8/f32) + W1 -> AxQMM -> [candidate op chain] -> AxQMM -> out
and reports the subgraph partition for each config.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def qmm(name, x, w, out, n_out, k, m=1):
    return helper.make_node(
        op_type="AxQuantizedMatMul", name=name,
        inputs=[x, w], outputs=[out],
        input_scales=(1.0, 1.0), input_zeropoints=(0, 0),
        output_dtype="FP32", output_scales=(1.0,), output_zeropoints=(0,),
        quant_method=0, transposed=1, domain="ax.matmul_gen")


def build_ffn(path, model_type_want):
    """gate/up AxQMM + SiLU/Sigmoid/Mul + down AxQMM, all standard-op middle."""
    K, MID = 1024, 3072
    act = helper.make_tensor_value_info("act", TensorProto.INT8, [1, K])
    gw = helper.make_tensor_value_info("gate_w", TensorProto.INT8, [MID, K])
    uw = helper.make_tensor_value_info("up_w", TensorProto.INT8, [MID, K])
    dw = helper.make_tensor_value_info("down_w", TensorProto.INT8, [K, MID])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, K])
    nodes = [
        qmm("gate", "act", "gate_w", "gate_raw", MID, K),
        qmm("up", "act", "up_w", "up_raw", MID, K),
        helper.make_node("Sigmoid", ["gate_raw"], ["gate_sig"], name="gs"),
        helper.make_node("Mul", ["gate_raw", "gate_sig"], ["gate_silu"], name="silu"),
        helper.make_node("Mul", ["gate_silu", "up_raw"], ["mid_f"], name="glu"),
        helper.make_node("Cast", ["mid_f"], ["mid_q"], to=TensorProto.INT8, name="castq"),
        qmm("down", "mid_q", "down_w", "out", K, MID),
    ]
    graph = helper.make_graph(nodes, "ffn", [act, gw, uw, dw], [out], initializer=[])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 16),
                       helper.make_operatorsetid("ax.matmul_gen", 1)],
        producer_name="probe")
    onnx.save(model, path)


def run_build(model_path, model_type, out_dir, tag):
    cfg = {
        "model_type": model_type,
        "npu_mode": "NPU1",
        "debug": {"stride_ios": ["out"], "single_eu_io": True},
        "compiler": {"check": 0, "ddr_bw_limit": 20},
    }
    if model_type == "ONNX":
        cfg["quant"] = {"input_configs": [{
            "tensor_name": "DEFAULT",
            "calibration_dataset": "/tmp/cal2.tar",
            "calibration_format": "Numpy",
            "calibration_size": 1,
        }]}
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        r = subprocess.run(
            ["/home/kram/Desktop/Projects/LLMTest/pulsar2/p7p/ax_pulsar2_7.0_patch1_lite_package/bin/pulsar2", "build", "--input", str(model_path), "--config", str(cfg_path),
             "--output_dir", str(out_dir)],
            capture_output=True, text=True, timeout=600)
    log = r.stdout + r.stderr
    sg_lines = [l for l in log.splitlines() if "subgraph [" in l and "type:" in l]
    n_npu = sum("GraphType.NPU" in l for l in sg_lines)
    n_onnx = sum("GraphType.ONNX" in l for l in sg_lines)
    ok = r.returncode == 0 and (Path(out_dir) / "compiled.axmodel").exists()
    print(f"[{tag}] rc={r.returncode} axmodel={ok} subgraphs: NPU={n_npu} ONNX={n_onnx}")
    for l in sg_lines:
        print("   ", l.split(" - ", 1)[-1])
    if not ok:
        errs = [l for l in log.splitlines() if "rror" in l or "not support" in l.lower()][:5]
        for e in errs:
            print("   ERR:", e[:160])
    return ok, n_npu, n_onnx


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "ONNX"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/probe_{model_type}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mpath = Path(out_dir) / "ffn.onnx"
    build_ffn(mpath, model_type)
    run_build(mpath, model_type, out_dir, model_type)
