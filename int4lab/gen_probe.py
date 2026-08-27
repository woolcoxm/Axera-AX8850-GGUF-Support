"""Uniform-sea probe checkpoints for s4 npu_params element-layout recovery.

All weights V=8 -> every group amax=8, scale=8/7, q=7 (v=15, 0xff sea).
Darks V=1 -> q=round(7/8)=1 (v=9) - single distinguishable nibbles.
Sections without darks should collapse (mc evidence); dark positions in the
surviving sections map directly to element order.
Usage: gen_probe.py <outdir> <kf|rf>
"""
import json, shutil, struct, sys
from pathlib import Path
import numpy as np

LAYERS = 2
VOCAB = 4096
HIDDEN = 1024
QH, KVH, HDIM = 16, 8, 128
INTER = 3072

SHAPES = {
    'self_attn.q_proj.weight': (QH * HDIM, HIDDEN),
    'self_attn.k_proj.weight': (KVH * HDIM, HIDDEN),
    'self_attn.v_proj.weight': (KVH * HDIM, HIDDEN),
    'self_attn.o_proj.weight': (HIDDEN, QH * HDIM),
    'mlp.gate_proj.weight': (INTER, HIDDEN),
    'mlp.up_proj.weight': (INTER, HIDDEN),
    'mlp.down_proj.weight': (HIDDEN, INTER),
}


def save_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes()); offset += arr.nbytes
    hdr = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hdr)) % 8:
        hdr += b" "
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr))); f.write(hdr)
        [f.write(b) for b in blobs]


def main():
    out = Path(sys.argv[1])
    mode = sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    src = '/home/kram/Desktop/Projects/LLMTest/Qwen3-0.6B'
    for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        shutil.copy(f'{src}/{f}', out / f)
    cfg = json.load(open(f'{src}/config.json'))
    cfg.update({'num_hidden_layers': LAYERS, 'vocab_size': VOCAB})
    (out / 'config.json').write_text(json.dumps(cfg))

    rng = np.random.default_rng(7)
    tensors = {
        'model.embed_tokens.weight': np.full((VOCAB, HIDDEN), 8.0, np.float32),
        'model.norm.weight': np.ones(HIDDEN, np.float32),
        'lm_head.weight': np.full((VOCAB, HIDDEN), 8.0, np.float32),
    }
    for L in range(LAYERS):
        p = f'model.layers.{L}.'
        tensors[p + 'input_layernorm.weight'] = np.ones(HIDDEN, np.float32)
        tensors[p + 'post_attention_layernorm.weight'] = np.ones(HIDDEN, np.float32)
        tensors[p + 'self_attn.q_norm.weight'] = np.ones(HDIM, np.float32)
        tensors[p + 'self_attn.k_norm.weight'] = np.ones(HDIM, np.float32)
        for mname, (n, k) in SHAPES.items():
            W = np.full((n, k), 8.0, np.float32)
            if mode == 'kf':
                # q_proj row 0: darks at k = 1,2,4,...,512
                if mname == 'self_attn.q_proj.weight':
                    for j in range(10):
                        W[0, 1 << j] = 1.0
                else:
                    W[0, 1] = 1.0     # one locator dark per other matrix
            elif mode == 'rf':
                # q_proj col 1: darks at r = 1,2,4,...,1024
                W[:, 0] = 8.0
                if mname == 'self_attn.q_proj.weight':
                    for j in range(11):
                        W[1 << j, 1] = 1.0
                else:
                    W[0, 1] = 1.0
            tensors[p + mname] = W
    save_safetensors(out / 'model.safetensors', tensors)
    print(f'probe {mode}: {out} ({len(tensors)} tensors)')


if __name__ == '__main__':
    main()
