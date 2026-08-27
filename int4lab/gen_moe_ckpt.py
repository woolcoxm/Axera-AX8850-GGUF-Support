"""Tiny qwen3_moe checkpoint for the llm_build2 MoE acceptance probe (R6)."""
import json, shutil, struct
from pathlib import Path
import numpy as np

LAYERS, VOCAB, HIDDEN = 2, 4096, 256
QH, KVH, HDIM = 2, 2, 128
NEXP, TOPK, MINTER = 8, 2, 256

def save_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes()); offset += arr.nbytes
    hdr = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hdr)) % 8: hdr += b" "
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr))); f.write(hdr)
        [f.write(b) for b in blobs]

out = Path('/tmp/int4lb2/moe_tiny'); out.mkdir(parents=True, exist_ok=True)
src = '/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B'
for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
    shutil.copy(f'{src}/{f}', out / f)

cfg = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "hidden_size": HIDDEN,
    "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HDIM,
    "num_experts": NEXP, "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MINTER, "decoder_sparse_step": 1,
    "norm_topk_prob": True, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
    "max_position_embeddings": 4096, "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}
(out / 'config.json').write_text(json.dumps(cfg, indent=1))

rng = np.random.default_rng(7)
tensors = {
    'model.embed_tokens.weight': (rng.standard_normal((VOCAB, HIDDEN))*0.02).astype(np.float32),
    'model.norm.weight': np.ones(HIDDEN, np.float32),
    'lm_head.weight': (rng.standard_normal((VOCAB, HIDDEN))*0.02).astype(np.float32),
}
for L in range(LAYERS):
    p = f'model.layers.{L}.'
    tensors[p+'input_layernorm.weight'] = np.ones(HIDDEN, np.float32)
    tensors[p+'post_attention_layernorm.weight'] = np.ones(HIDDEN, np.float32)
    tensors[p+'self_attn.q_norm.weight'] = np.ones(HDIM, np.float32)
    tensors[p+'self_attn.k_norm.weight'] = np.ones(HDIM, np.float32)
    tensors[p+'self_attn.q_proj.weight'] = (rng.standard_normal((QH*HDIM, HIDDEN))*0.05).astype(np.float32)
    tensors[p+'self_attn.k_proj.weight'] = (rng.standard_normal((KVH*HDIM, HIDDEN))*0.05).astype(np.float32)
    tensors[p+'self_attn.v_proj.weight'] = (rng.standard_normal((KVH*HDIM, HIDDEN))*0.05).astype(np.float32)
    tensors[p+'self_attn.o_proj.weight'] = (rng.standard_normal((HIDDEN, QH*HDIM))*0.05).astype(np.float32)
    tensors[p+'mlp.gate.weight'] = (rng.standard_normal((NEXP, HIDDEN))*0.05).astype(np.float32)  # router
    for e in range(NEXP):
        ep = p + f'mlp.experts.{e}.'
        tensors[ep+'gate_proj.weight'] = (rng.standard_normal((MINTER, HIDDEN))*0.05).astype(np.float32)
        tensors[ep+'up_proj.weight']   = (rng.standard_normal((MINTER, HIDDEN))*0.05).astype(np.float32)
        tensors[ep+'down_proj.weight'] = (rng.standard_normal((HIDDEN, MINTER))*0.05).astype(np.float32)
save_safetensors(out/'model.safetensors', tensors)
print(f"qwen3_moe tiny ckpt: {len(tensors)} tensors at {out}")
