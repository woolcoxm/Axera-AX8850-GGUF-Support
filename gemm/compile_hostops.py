#!/usr/bin/env python3
"""Compile the host-op chain engines with the verified mixed-precision config."""
import io, json, os, subprocess, tarfile
import numpy as np

GEMM = "/home/kram/Desktop/Projects/LLMTest/gemm"
PULSAR = "/home/kram/Desktop/Projects/LLMTest/pulsar2/5.2/5.2/ax_pulsar2_5.2_lite_package/bin/pulsar2"
OUT = os.path.join(GEMM, "hostops")

def tar_of(arrays, path, name):
    with tarfile.open(path, "w") as tf:
        for i, a in enumerate(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a, dtype=a.dtype), allow_pickle=False)
            d = buf.getvalue()
            ti = tarfile.TarInfo(name=f"{name}/{i}.npy"); ti.size = len(d)
            tf.addfile(ti, io.BytesIO(d))

def build(tag, onnx, inputs, dtypes=None):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    cfg_inputs = []
    for (name, shape, std, n) in inputs:
        rng = np.random.default_rng(abs(hash(tag + name)) % (2**31))
        dt = np.float16 if (dtypes and name in dtypes) else np.float32
        arrs = [(rng.standard_normal(shape) * std).astype(dt) for _ in range(n)]
        tar_of(arrs, os.path.join(d, f"cal_{name}.tar"), name)
        cfg_inputs.append({"tensor_name": name, "calibration_dataset": f"./cal_{name}.tar",
                           "calibration_size": n, "calibration_format": "Numpy"})
    cfg = {"model_type": "ONNX", "npu_mode": "NPU1",
           "quant": {"input_configs": cfg_inputs, "calibration_method": "MinMax",
                     "highest_mix_precision": True, "precision_analysis": True,
                     "precision_analysis_method": "EndToEnd", "transformer_opt_level": 2},
           "compiler": {"check": 0}}
    json.dump(cfg, open(os.path.join(d, "cfg.json"), "w"), indent=1)
    r = subprocess.run([PULSAR, "build", "--config", "cfg.json", "--input", os.path.abspath(onnx),
                        "--output_dir", "out", "--output_name", "compiled.axmodel"],
                       cwd=d, capture_output=True, text=True)
    ok = os.path.exists(os.path.join(d, "out", "compiled.axmodel"))
    print(f"[{tag}] {'OK' if ok else 'FAIL ' + r.stderr[-300:]}", flush=True)
    return ok

def main():
    os.chdir(OUT)
    jobs = [
        ("rope_q",  "rope_q_d2048.onnx",  [("x", (2048,), 4.0, 8), ("pos", (1,), 500.0, 8)]),
        ("rope_k",  "rope_k_d1024.onnx",  [("x", (1024,), 4.0, 8), ("pos", (1,), 500.0, 8)]),
        ("add",     "add_h1024.onnx",     [("a", (1024,), 8.0, 8), ("b", (1024,), 8.0, 8)]),
        ("rmsnorm", "rmsnorm_h1024.onnx", [("x", (1024,), 8.0, 8), ("w", (1024,), 1.0, 4)]),
        ("glu",     "glu_h3072.onnx",     [("x", (6144,), 8.0, 8)]),
        # vocab: W fp16 input, values in the ±0.2 range typical of lm_head rows
        ("vocab16", "vocab_fp16.onnx",    [("x", (1, 1024), 4.0, 8), ("W", (1024, 151936), 0.02, 1)]),
        # qkv WITH norm inside (recompile of the original with mixed precision)
        ("qkv_norm", "qkv_norm_h1024_q2048_kv1024.onnx",
         [("hidden", (1, 1024), 8.0, 8), ("norm_w", (1024,), 1.0, 4),
          ("q_w", (1024, 2048), 0.02, 2), ("k_w", (1024, 1024), 0.02, 2), ("v_w", (1024, 1024), 0.02, 2)]),
    ]
    good = bad = 0
    for tag, onnx, inputs in jobs:
        dtypes = {"W": np.float16} if tag == "vocab16" else None
        if not os.path.exists(onnx):
            print(f"[{tag}] SKIP no onnx {onnx}", flush=True); bad += 1; continue
        if build(tag, onnx, inputs, dtypes):
            good += 1
        else:
            bad += 1
    print(f"DONE good={good} bad={bad}", flush=True)

if __name__ == "__main__":
    main()
