#!/usr/bin/env python3
"""Generate a random-weight HuggingFace-style Qwen3 checkpoint (safetensors) for
llm_build dissection. Tiny vocab for fast post build."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3_rand")
VOCAB = 4096
HIDDEN = 1024
LAYERS = 28
QHEADS = 16
KVHEADS = 8
HEADDIM = 128
INTER = 3072

cfg = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "num_hidden_layers": LAYERS,
    "num_attention_heads": QHEADS,
    "num_key_value_heads": KVHEADS,
    "head_dim": HEADDIM,
    "hidden_size": HIDDEN,
    "intermediate_size": INTER,
    "vocab_size": VOCAB,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))

rng = np.random.default_rng(7)


def save_safetensors(path, tensors):
    # safetensors format: u64 header_len, JSON header (padded with spaces), raw data
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + nbytes]}
        blobs.append(arr.tobytes())
        offset += nbytes
    hdr = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 + len(hdr)) % 8  # keep alignment
    hdr += b" " * ((8 - pad) % 8 if pad else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


def w(std, *shape):
    return (rng.standard_normal(shape) * std).astype(np.float32)


tensors = {"model.embed_tokens.weight": w(0.02, VOCAB, HIDDEN),
           "model.norm.weight": (1.0 + 0.05 * rng.standard_normal(HIDDEN)).astype(np.float32),
           "lm_head.weight": w(0.02, VOCAB, HIDDEN)}
shard = {}
idx = 0
for i in range(LAYERS):
    p = f"model.layers.{i}."
    shard.update({
        p + "input_layernorm.weight": (1.0 + 0.05 * rng.standard_normal(HIDDEN)).astype(np.float32),
        p + "self_attn.q_proj.weight": w(0.02, QHEADS * HEADDIM, HIDDEN),
        p + "self_attn.k_proj.weight": w(0.02, KVHEADS * HEADDIM, HIDDEN),
        p + "self_attn.v_proj.weight": w(0.02, KVHEADS * HEADDIM, HIDDEN),
        p + "self_attn.o_proj.weight": w(0.02, HIDDEN, QHEADS * HEADDIM),
        p + "post_attention_layernorm.weight": (1.0 + 0.05 * rng.standard_normal(HIDDEN)).astype(np.float32),
        p + "mlp.gate_proj.weight": w(0.02, INTER, HIDDEN),
        p + "mlp.up_proj.weight": w(0.02, INTER, HIDDEN),
        p + "mlp.down_proj.weight": w(0.02, HIDDEN, INTER),
    })
    # shard every 8 layers
    if (i + 1) % 8 == 0 or i == LAYERS - 1:
        save_safetensors(OUT / f"model-{idx:05d}-of-{(LAYERS + 7) // 8:05d}.safetensors", shard)
        shard = {}
        idx += 1

(OUT / "model.safetensors.index.json").write_text(json.dumps({
    "metadata": {"total_size": 1},
    "weight_map": {n: f"model-00000-of-00004.safetensors" for n in []},
}))
print(f"checkpoint written to {OUT}")
