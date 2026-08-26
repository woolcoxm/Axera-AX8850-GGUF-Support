# ggml-axcl — llama.cpp Axera NPU backend

A custom [llama.cpp](https://github.com/ggml-org/llama.cpp) backend (`ggml-axcl`) that runs
Qwen3-0.6B **directly from GGUF** on an Axera AX8850 NPU accelerator card
(M5Stack LLM-8850: 24 TOPS INT8, 8 GB LPDDR4x) hosted on a Raspberry Pi 5.

**The GGUF is the only model artifact** — weights stream from the GGUF into NPU engines
at load time. Q8_0 and Q4_K_M quants both work from the same code path.

- Current speed: ~1.3-2.7 t/s decode (see "Architecture" for why)
- Vendor reference, same card + model: 13.5-16.9 t/s (baked weights, closed runtime)
- Code: branch `Axera-8850-GGUF-support-PoC-qwen3-0.9b-Q4KM-Q8` on
  `github.com/woolcoxm/llama.cpp`

## Repository layout

```
LLMTest/
├── README.md                   this file
├── NOTES-DYNAMIC-WEIGHTS.md    research log: whole-layer engines (weight layout cracked)
├── llama.cpp/                  llama.cpp fork with the ggml-axcl backend
│   └── ggml/src/ggml-axcl/ggml-axcl.cpp   THE backend
├── gemm/                       NPU engine lab (harnesses, layout research, test scripts)
│   ├── layout_artifacts/       captured QuantAxModel builds
│   ├── mk_code_marker.py       code-encoded checkpoints (layout extraction)
│   └── e2e_test.sh             the E2E test matrix (run on the Pi)
├── pulsar2/                    Axera compiler toolchain (x86_64)
├── ax-llm-build/               vendor LLM-builder configs
├── Qwen3-0.6B/                 HF checkpoint (ground truth for builds)
└── vendor/                     vendor reference packages
```

On the **Pi** (kram@10.0.0.81): `~/build-axcl/` (build), `~/models/` (qwen3-q8.gguf,
qwen3-q4km.gguf), `/usr/local/share/ggml-axcl/` (compiled NPU engines),
`/usr/lib/axcl/` (card runtime, `axcl-smi` at `/usr/bin/axcl/axcl-smi`).

## Architecture — and the honest performance story

Each generated token executes ~120-140 NPU engine calls (7 matmuls × 28 layers + vocab
head). Between every call the Pi's CPU does glue work: RMSNorm, RoPE, softmax, masking,
adds, GLU. Per call: ~0.6 ms NPU exec wrapped in ~2-3 ms of host staging + PCIe DMA +
scheduler overhead — llama.cpp's scheduler splits the graph into per-op fragments, so
**the NPU idles ~80% of every token waiting for the host** (visible as ~21% NPU
utilization with CMM at 7 GB).

The vendor's engine fuses the *entire layer* into one NPU call (norm, qkv, rope,
attention with on-card KV cache, FFN, GLU): 28 calls/token, no host round-trips,
1.5 ms/layer. That's the 5-10× gap. Our path there is staged (see "Offload roadmap").

The weight path is already solved: Pulsar2's `AxQuantizedMatMul` custom op accepts
**int8 weights as runtime tensor inputs**, so GGUF weights are quantized once at load,
uploaded, and bound per call — no conversion step, 4× less traffic than f32.

## Test status (2026-08-26 E2E pass)

13/14 automated tests PASS: prompt sizes 1 → 3000 tokens, unicode, emoji, shell
metacharacters, 2000-char word, single char, long generation. Fixed during the pass:
two empty-prompt crashes in llama-simple (`n_batch=0` assert; zero-token batch → BOS
fallback). Verified: SIGINT shutdown clean with full CMM release; 5 back-to-back runs
leak-free; 2 concurrent runs correct; NPU activity confirmed live via axcl-smi.

Device-resident **chain mode** (`GGML_AXCL_CHAIN=1`): verified coherent including
180-token prompts (the old corruption is fixed) — norm/add/glu run as NPU engines on
device-resident activations.

Cross-fragment QKV fusion: implemented, engine output verified bit-exact vs CPU
reference, 3 calls → 1 per layer — but default-off because the scheduler interleaves
q_norm/rope CPU fragments before the fusion's writeback (corrupts KV cache). Enable
for experiments with `GGML_AXCL_QKV_X=1`; needs op-claiming/scheduler work to be safe.

## Build and run

Pi:
```bash
cmake -B build-axcl -S llama.cpp -DGGML_AXCL=ON
cmake --build build-axcl -j4
~/build-axcl/bin/llama-simple -m ~/models/qwen3-q8.gguf -n 48 "Your prompt here"
```
NOTE: llama-simple takes the prompt as a **positional argument** (not `-p`), and `-n`
must come *before* the prompt.

Engines (dev machine, x86): compiled with Pulsar2 from `gemm/` ONNX sources
(`pulsar2/p7p/.../bin` on PATH), installed to the Pi under
`/usr/local/share/ggml-axcl/`.

## Environment variables

| Var | Effect |
|---|---|
| `GGML_AXCL_CHAIN` | device-resident chain mode (norm/add/glu on NPU) |
| `GGML_AXCL_CHAIN_OPS` | gate chain routes (`norm,add,glu`) |
| `GGML_AXCL_QKV_X` | cross-fragment QKV fusion (default off — see above) |
| `GGML_AXCL_QKV_SWAP` | swap k/v bindings (diagnostics) |
| `GGML_AXCL_WPOOL_MB` | device weight pool size (default 2560) |
| `GGML_AXCL_NO_OVERRIDE` | disable activation-source override |
| `GGML_AXCL_NO_FUSION` | disable all fusions |
| `GGML_AXCL_ASYNC` | async engine execute + stream sync |
| `GGML_AXCL_ATTN_MODEL` | attention engine path override |
| `GGML_AXCL_DEBUG` | verbose engine load/bind logging |

## NPU3 / multi-core findings

- NPU3-compiled engines measured **neutral** vs NPU1 for our shapes (DRAM-bound);
  the vendor ships NPU1-default builds for this model class.
- Multi-core opportunity: VNPU partitioning (`axclrtEngineInit` VNPU kinds) +
  concurrent engine execution — untapped.
- Full card utilization comes from the whole-layer engines, not multi-core compilation.

## Offload roadmap (everything-on-NPU)

1. **Chain mode everywhere** (done, default off): elementwise ops as NPU engines.
2. **Matmul device-pipelining**: bind X from device-resident chain buffers (the
   `g_chain_x_override` mechanism) for all projections — removes H2D per call.
3. **Attention engine for all context lengths**: device KV cache with watermark
   uploads (exists; activation gate needs fixing — currently requires seq > 128).
4. **Whole-layer engines**: vendor-class 28 calls/token. Weight layout fully
   reverse-engineered and verified (int4 nibble pairs, complete position table —
   `gemm/layer_layout_v3.pkl`); remaining: scale-table mapping, weight-patching
   loader via `axclrtEngineLoadFromMem`, backend integration. Full log:
   `NOTES-DYNAMIC-WEIGHTS.md`.

## Troubleshooting

- Engines fail to load → `sudo chmod 777 /tmp/axcl` (runtime log sink), check driver.
- Garbage output → unset `GGML_AXCL_CHAIN`/`GGML_AXCL_QKV_X` experiment flags.
- Card busy → check `axcl-smi` for stale processes; CMM baseline is ~18 MiB.
- Periodic `memory api ... return fail` log lines from the card runtime are non-fatal.
