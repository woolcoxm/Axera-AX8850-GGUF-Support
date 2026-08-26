#!/usr/bin/env python3
"""Rebuild all ggml-axcl engines with distribution-independent precision.

The original engines baked int8 quantization scales from synthetic std-1.0
calibration data; real model weights (std ~0.02) quantized to noise. The fix:
highest_mix_precision + precision_analysis(EndToEnd) keeps the matmuls in fp16,
verified exact on fresh-seed realistic data (max_err 1e-5).

Each engine gets calibration matching REAL distributions: activations std ~2,
weights std ~0.02 — needed so the precision analysis makes sensible choices.
"""
import io, json, os, subprocess, sys, tarfile
import numpy as np

GEMM = "/home/kram/Desktop/Projects/LLMTest/gemm"
PULSAR = "/home/kram/Desktop/Projects/LLMTest/pulsar2/5.2/5.2/ax_pulsar2_5.2_lite_package/bin/pulsar2"
OUT = os.path.join(GEMM, "mixall")

def tar_of(arrays, path, name):
    with tarfile.open(path, "w") as tf:
        for i, a in enumerate(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a, dtype=np.float32), allow_pickle=False)
            d = buf.getvalue()
            ti = tarfile.TarInfo(name=f"{name}/{i}.npy")
            ti.size = len(d)
            tf.addfile(ti, io.BytesIO(d))

def build(tag, onnx_path, inputs):
    """inputs: list of (tensor_name, shape, std, n_samples)"""
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    cfg_inputs = []
    for name, shape, std, n in inputs:
        rng = np.random.default_rng(hash(tag + name) % (2**32))
        arrs = [rng.standard_normal(shape) * std for _ in range(n)]
        tar_of(arrs, os.path.join(d, f"cal_{name}.tar"), name)
        cfg_inputs.append({
            "tensor_name": name,
            "calibration_dataset": f"./cal_{name}.tar",
            "calibration_size": n,
            "calibration_format": "Numpy",
        })
    cfg = {
        "model_type": "ONNX", "npu_mode": "NPU1",
        "quant": {
            "input_configs": cfg_inputs,
            "calibration_method": "MinMax",
            "highest_mix_precision": True,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "transformer_opt_level": 2,
        },
        "compiler": {"check": 0},
    }
    cfgp = os.path.join(d, "cfg.json")
    json.dump(cfg, open(cfgp, "w"), indent=1)
    r = subprocess.run([PULSAR, "build", "--config", "cfg.json", "--input", onnx_path,
                        "--output_dir", "out", "--output_name", "compiled.axmodel"],
                       cwd=d, capture_output=True, text=True)
    ok = os.path.exists(os.path.join(d, "out", "compiled.axmodel"))
    print(f"[{tag}] {'OK' if ok else 'FAIL'} ({len(r.stderr)}B stderr tail: {r.stderr[-200:] if not ok else ''})", flush=True)
    return ok

def main():
    jobs = []
    # matmul engines: (k, n, w_samples)
    for k, n, ws in [(1024,512,2), (1024,1024,2), (1024,2048,2), (1024,3072,2),
                     (2048,1024,2), (2048,2048,2), (3072,1024,2), (1024,151936,1)]:
        tag = f"mm_k{k}_n{n}"
        onnx = os.path.join(d_ := os.path.join(OUT, tag), f"gemm_{k}_{n}.onnx")
        os.makedirs(d_, exist_ok=True)
        subprocess.run([sys.executable, os.path.join(GEMM, "make_gemm.py"),
                        "--k", str(k), "--n", str(n), "--out", onnx],
                       cwd=GEMM, capture_output=True)
        jobs.append((tag, onnx, [("X", (1, k), 2.0, 8), ("W", (k, n), 0.02, ws)]))
    # qkv no-norm: h + 3 weights
    jobs.append(("qkv_nn", os.path.join(GEMM, "qkv_nn_h1024_q2048_kv1024.onnx"),
                 [("h", (1, 1024), 2.0, 8), ("q_w", (1024, 2048), 0.02, 2),
                  ("k_w", (1024, 1024), 0.02, 2), ("v_w", (1024, 1024), 0.02, 2)]))
    # gate+up
    jobs.append(("gate_up", os.path.join(GEMM, "gate_up_h1024_i3072.onnx"),
                 [("h", (1, 1024), 2.0, 8), ("gate_w", (1024, 3072), 0.02, 2),
                  ("up_w", (1024, 3072), 0.02, 2)]))
    # attention t512 (mixed already, but recalibrate with realistic spreads)
    jobs.append(("attn512", os.path.join(GEMM, "attn_h16_d128_t512.onnx"),
                 [("Q", (16, 128), 4.0, 8), ("K", (16, 512, 128), 4.0, 4),
                  ("V", (16, 512, 128), 1.0, 4), ("mask", (512,), 1.0, 4)]))

    good = bad = 0
    for tag, onnx, inputs in jobs:
        if not os.path.exists(onnx):
            print(f"[{tag}] SKIP: no onnx {onnx}", flush=True)
            bad += 1
            continue
        if build(tag, onnx, inputs):
            good += 1
        else:
            bad += 1
    print(f"DONE good={good} bad={bad}", flush=True)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
