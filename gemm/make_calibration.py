#!/usr/bin/env python3
"""Prepare calibration data + build config for the GEMM axmodel.

Pulsar2's build config wants, per input tensor, a calibration_dataset tar
of raw files. For our non-image graph we emit raw .bin files (f32 bytes)
and reference them per-input.
"""
import argparse
import io
import json
import os
import tarfile

import numpy as np


def tar_of(arrays, path):
    with tarfile.open(path, "w") as tf:
        for i, a in enumerate(arrays):
            data = a.astype(np.float32).tobytes()
            info = tarfile.TarInfo(name=f"{i}.bin")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--outdir", type=str, default=".")
    args = p.parse_args()

    rng = np.random.default_rng(7)

    # calibration values spanning a sane activation/weight range
    xs = [rng.standard_normal((args.m, args.k)).astype(np.float32) * 0.5 for _ in range(args.samples)]
    ws = [rng.standard_normal((args.k, args.n)).astype(np.float32) * 0.05 for _ in range(args.samples)]
    tar_of(xs, os.path.join(args.outdir, "calib_x.tar"))
    tar_of(ws, os.path.join(args.outdir, "calib_w.tar"))

    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "X",
                    "calibration_dataset": "./calib_x.tar",
                },
                {
                    "tensor_name": "W",
                    "calibration_dataset": "./calib_w.tar",
                },
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "input_processors": [
            {
                "tensor_name": "X",
                "tensor_format": "RAW",
                "src_format": "RAW",
                "src_dtype": "F32",
                "src_layout": "NCHW",
                "csc_mode": "NoCSC",
            },
            {
                "tensor_name": "W",
                "tensor_format": "RAW",
                "src_format": "RAW",
                "src_dtype": "F32",
                "src_layout": "NCHW",
                "csc_mode": "NoCSC",
            },
        ],
        "compiler": {
            "check": 0,
        },
    }
    cfg_path = os.path.join(args.outdir, "gemm_build_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"wrote {cfg_path}, calib_x.tar, calib_w.tar")


if __name__ == "__main__":
    main()
