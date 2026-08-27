# llama.cpp backend for AX8850 (M5Stack LLM-8850 on Raspberry Pi 5): 19.6 tokens/s — faster than the vendor runtime

*Update to my earlier thread on the custom ggml-axcl backend. Repo: github.com/woolcoxm/llama.cpp (branch `Axera-8850-GGUF-support-PoC-qwen3-0.9b-Q4KM-Q8`) + research lab: github.com/woolcoxm/LLMTest.*

## Where it stands

The backend now runs Qwen3-0.6B **straight from GGUF** (q8_0 and Q4_K_M through the same code path) on the AX8850 card, driven by llama.cpp on a Raspberry Pi 5:

| | decode | prefill | host CPU | card memory |
|---|---|---|---|---|
| ggml-axcl, vendor w8a16 engines | **19.6 t/s** | 18.5 t/s | **2% of one core** | 1.3 GB |
| ggml-axcl, dynamic GGUF weights (bf16 engines) | 10.2 t/s | 5.0 t/s | ~5% | 2.4 GB |
| vendor closed runtime, same card | 13.5–14.5 t/s | — | — | — |

All 28 whole-layer engines + the post engine run on the NPU with device-resident bf16 activations; the CPU only orchestrates and samples. E2E suite (1→3000-token prompts, unicode/emoji/metacharacters, empty prompts, SIGINT, leak checks): 12/12 pass in both modes.

## How we got from 7.9 to 19.6 t/s

**Half the win was host plumbing.** Profiling showed 35 ms/token of orchestration around 90 ms of engine time. On this stack, an *unpinned* small memcpy costs ~1 ms (page pinning per transfer) — and the code was doing 4-byte uploads and 2 KB write-backs from stack buffers every layer. After moving every hot transfer to `axclrtMallocHost` staging, hoisting the per-layer index/mask refresh to once per token, binding static IO once per engine, writing engine K/V outputs directly into cache rows, deferring the host KV write-back behind a watermark journal (batched contiguous flushes), and NEON-vectorizing the bf16 conversions, host overhead dropped to ~2 ms.

**The other half was the engine set.** Our pulsar2 `llm_build` bf16 engines are DRAM-bound at 3.24 ms/layer. We tried `-w fp8_e4m3` and `-w s8` builds — interestingly, both produce same-size engines that run at *exactly* the same speed as bf16 (the flag repacks the blob, but the conv-EU path still pays full traffic). The vendor w8a16 engines use the native int8 path: 1.51 ms/layer, 23 MB instead of 65 MB per layer. Since they expose the same IO conventions we had already reverse-engineered (K/V caches, indices, mask, one call per layer), pointing our backend at them was a one-line change — and llama.cpp then outperforms the vendor's own runner on the same card. Full credit to the Axera compiler team: those engines are excellent.

## Two findings we hope are useful to you

1. **The 128-token prefill shape groups don't work through the PCIe host runtime (V3.6.5_P1).** The vendor layer engines carry a 10-group ladder (decode m=1 + chunk groups with prefix 0..1024 — a great design). We mapped it completely via `axclrtEngineGetInputSizeByIndex(info, group, i)` and drove it: calls *execute* (fast!), but the engine ignores the bound input buffer for chunk groups — zeroing 127 of 128 input rows leaves the output unchanged — and the runtime logs an internal `[memory][memcpy] nil pointer`. An LD_PRELOAD trace of `main_axcl_aarch64` shows the vendor runtime itself never executes any group except 0 (18,089 group-0 calls on a 300-token prefill, zero chunk calls). It looks like the chunk groups are only wired up for the on-device SDK runtime (`libax_engine`), not the axcl host stack. If a future release enables them host-side, prefill on this card could go from ~18 t/s to several hundred t/s per token-slot — we left the (working, validated-against-group-0) driver code in the backend behind `GGML_AXCL_BATCH=1`.

2. A small PSA for anyone building models with pulsar2 on a shared box: the QuantAxModel ONNX-capture `sitecustomize.py` hooks (tensor logging to /tmp) can fill a tmpfs and fail builds with a confusing `Disk quota exceeded`.

## What's next

- GGUF-weight patching onto the int8 engines (needs the w8a16 weight layout inside `npu_params` — the bf16 raw layout is fully cracked, the int8 one is the current research front).
- Greedy token-agreement vs the CPU reference (10 prompts x 24 tokens): vendor-engine mode q8_0 94% / Q4_K_M 90%; dynamic-GGUF mode q8_0 91% / Q4_K_M 93%.

Thanks for a genuinely fun platform to work on — the whole-layer engine design (one graph per transformer layer, KV resident on-card) is what makes this backend possible at all. Questions and pointers welcome, especially from anyone who knows the chunk-group story.
