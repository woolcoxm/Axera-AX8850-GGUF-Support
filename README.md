# ggml-axcl — llama.cpp Axera NPU backend

A custom [llama.cpp](https://github.com/ggml-org/llama.cpp) backend (`ggml-axcl`) that runs
Qwen3-0.6B **directly from GGUF** on an Axera AX8850 NPU accelerator card
(M5Stack LLM-8850: 24 TOPS INT8, 8 GB LPDDR4x) hosted on a Raspberry Pi 5.

**The GGUF is the only model artifact.** At load time the GGUF's weights are
dequantized and patched into pre-compiled whole-layer NPU engines — no model
conversion, no per-model compile step. Q8_0 and Q4_K_M quants work from the
same code path.

| | decode | prefill | CPU load | card CMM |
|---|---|---|---|---|
| **Dynamic-GGUF mode** (flagship) | **7.9 t/s** | **7.9 t/s** | **~8%** of one core | 2.4 GB |
| Baked-weights mode | 8.0 t/s | 7.9 t/s | ~8% | 2.4 GB |
| Legacy per-op mode | 2.2 t/s | 1.6 t/s | ~100% | 5.5 GB |
| Vendor reference (closed runtime) | 13.5-16.9 t/s | — | — | — |

All model compute — 28 whole-layer engines (attention + FFN + norms + RoPE)
plus the post engine (final norm + 151936-wide lm_head) — executes on the
AX8850 with device-resident bf16 activations; the CPU only orchestrates.

Fidelity: greedy-decode token agreement vs the CPU reference is 94%
(dynamic-GGUF) / 95% (baked) — divergence is bf16-engine vs q8-GGUF weight
numerics on near-tie tokens. E2E suite: 12/12 pass (1→3000 token prompts,
unicode, emoji, shell metacharacters, empty prompts, SIGINT, 5× back-to-back
leak check).

- Code: branch `Axera-8850-GGUF-support-PoC-qwen3-0.9b-Q4KM-Q8` on
  `github.com/woolcoxm/llama.cpp`; research lab in
  [LLMTest](https://github.com/woolcoxm/LLMTest) (`gemm/`, this repo)

## Quick start

On the **Pi** (kram@10.0.0.81 in this setup; any aarch64 host with the AXCL
driver works):

```bash
# build llama.cpp with the backend
git clone -b Axera-8850-GGUF-support-PoC-qwen3-0.9b-Q4KM-Q8 \
    https://github.com/woolcoxm/llama.cpp
cmake -B build-axcl -S llama.cpp -DGGML_AXCL=ON
cmake --build build-axcl -j4

# install the engine set (templates + post + layout sidecar)
sudo mkdir -p /usr/local/share/ggml-axcl/layer
# from the LLMTest repo (built on the x86 box, see below):
scp gemm/baked/real2048_bf16/qwen3_p128_l*_together.axmodel pi:/usr/local/share/ggml-axcl/layer/
scp gemm/baked/real2048_bf16/qwen3_post.axmodel       pi:/usr/local/share/ggml-axcl/layer/
scp gemm/layout_v4.bin                                pi:/usr/local/share/ggml-axcl/layer/

# run: the GGUF's weights flow into the engines at load
GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 \
    ~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 48 "Your prompt"
```

Notes:
- llama-simple takes the prompt as a **positional argument** (not `-p`), and
  `-n` must come *before* the prompt.
- First run per GGUF patches 28 engines (~30s, cached afterwards in
  `/tmp/axcl-gguf`, keyed by a hash of the weights). Warm starts take ~60s
  to load 28×65MB engines into card memory — the same cost the vendor
  runtime pays.
- `GGML_AXCL_LAYER=1 GGML_AXCL_FA=1` alone runs the baked template weights
  (HF f32-derived); add `GGML_AXCL_GGUF=1` to patch in the GGUF's weights.

### Building the engine templates (only needed once per architecture)

On an **x86_64 Linux box** with the Pulsar2 toolchain (Axera's compiler,
7.0-patch1 in `LLMTest/pulsar2/`):

```bash
export PATH=$PULSAR2/bin:$PATH
# whole-layer engines: 28 files, bf16 weights (the -w s8 default stores
# int4 weights whose accumulated drift garbles generation — use bf16)
FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build \
    --input_path Qwen3-0.6B \
    --output_path out \
    --hidden_state_type bf16 --kv_cache_len 2048 --prefill_len 128 \
    --last_kv_cache_len 128 --chip AX650 -c 0 --parallel 8 -w bf16
# produces qwen3_p128_l{0..27}_together.axmodel + qwen3_post.axmodel
```

The `layout_v4.bin` sidecar maps every weight to its byte offset inside a
template engine. It is derived once from the baked engines by value-anchored
search (`gemm/anchor_real_layout.py` + `gemm/emit_layout_v4.py`) and is
re-derivable if the build config changes. It is only tied to the engine
build, not to any particular GGUF.

## Modes

| Env | What it does |
|---|---|
| `GGML_AXCL_GGUF=1` | patch engines from the GGUF's weights at load (with the two below) |
| `GGML_AXCL_LAYER=1` | whole-layer engine mode (1 call/layer/token, device-resident hidden) |
| `GGML_AXCL_FA=1` | claim flash-attention so decode graphs arrive unsplit |
| `GGML_AXCL_GGUF_DIR` | cache dir for patched engines (default /tmp/axcl-gguf) |
| `GGML_AXCL_LAYOUT` | layout sidecar path (default .../layer/layout_v4.bin) |
| `GGML_AXCL_POST_MODEL` | post-engine path override |
| `GGML_AXCL_CHAIN=1` | legacy device-resident chain mode (superseded) |
| `GGML_AXCL_CHAIN_OPS` | gate chain routes (`norm,add,glu`) |
| `GGML_AXCL_WPOOL_MB` | legacy weight pool size (unused in layer mode) |
| `GGML_AXCL_NO_OVERRIDE` | disable activation-source override |
| `GGML_AXCL_NO_FUSION` | disable all fusions |
| `GGML_AXCL_ASYNC` | async engine execute + stream sync (legacy) |
| `GGML_AXCL_LAYER_DEBUG` / `GGML_AXCL_CHECKSUM` / `GGML_AXCL_DUMPSTATE` | diagnostics |

## How it works

1. **Whole-layer engines**: pulsar2's `llm_build` compiles one NPU graph per
   transformer layer (RMS norms, Q/K/V with q/k-norm, RoPE, attention over
   the on-card KV cache, FFN with SwiGLU). Per token: 28 engine calls +
   1 post-engine call (final norm + lm_head). Hidden state is bf16 and
   never leaves the card.
2. **Weight patching**: the engine files store weights as **raw bf16 at
   deterministic file offsets** (reverse-engineered; byte-exact validation:
   patching layer-0's template with layer-1's weights reproduces the baked
   layer-1 engine to 3 bytes — layer-index microcode — and computes
   identically on-card). The loader dequantizes each GGUF tensor row, bf16
   rounds it, and scatters it into a copy of the template via the sidecar
   table; patched files are cached by a hash of the weights.
3. **Graph interception**: the backend claims the whole compute graph so
   llama.cpp's scheduler delivers it unsplit; the first armed graph's
   prescan registers all weight tensors, patches + loads the engines
   (swap happens BEFORE any node executes — swapping later mixes
   template-weights KV state with GGUF compute), then each layer's
   q_proj anchor runs one engine call.

Full research log: `NOTES-DYNAMIC-WEIGHTS.md`.

## Test / eval / bench tooling

- `gemm/e2e_test.sh` — the E2E matrix (run on the Pi)
- `gemm/eval_suite.sh` — factuality/code/coherence scoring
- `gemm/eval_agreement.sh` — token-agreement vs the CPU reference
- `gemm/bench_suite.sh` — throughput, memory, stress, startup
- `gemm/chain_test.c` + `gemm/ref_chain.py` — engine-chain vs numpy reference

## Troubleshooting

- Engines fail to load → `sudo chmod 777 /tmp/axcl` (runtime log sink), check
  driver. Transient load failures retry automatically (~10s window).
- Garbage output → ensure all three env flags together; unset old experiment
  flags (`GGML_AXCL_CHAIN`, `GGML_AXCL_QKV_X`).
- Card busy → check `axcl-smi` for stale processes; CMM baseline is ~18 MiB.
- Periodic `memory api ... return fail` log lines from the card runtime are
  non-fatal.

## Repository layout (LLMTest)

```
LLMTest/
├── README.md                   this file
├── NOTES-DYNAMIC-WEIGHTS.md    research log: whole-layer engines (weight layout cracked)
├── llama.cpp/                  llama.cpp fork with the ggml-axcl backend
│   └── ggml/src/ggml-axcl/ggml-axcl.cpp   THE backend
├── gemm/                       NPU engine lab (harnesses, layout research, test scripts)
│   ├── layout_v4.bin           weight-position sidecar for the loader
│   ├── baked/real2048_bf16/    compiled engine templates (28 layers + post)
│   ├── e2e_test.sh             the E2E test matrix (run on the Pi)
│   └── eval_*.sh, bench_suite.sh, chain_test.c, ref_chain.py
├── pulsar2/                    Axera compiler toolchain (x86_64)
├── Qwen3-0.6B/                 HF checkpoint (ground truth for builds)
└── vendor/                     vendor reference packages
```
