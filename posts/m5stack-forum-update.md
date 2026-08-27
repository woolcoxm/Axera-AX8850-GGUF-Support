# LLM-8850 on Raspberry Pi 5: llama.cpp now runs GGUF models at 19.6 tok/s — faster than the bundled runtime

*Update to my LLM-8850 project thread. Open source: github.com/woolcoxm/llama.cpp (branch `Axera-8850-GGUF-support-PoC-qwen3-0.9b-Q4KM-Q8`) + the research repo: github.com/woolcoxm/LLMTest.*

Quick summary for fellow LLM-8850 owners: I've been building a custom llama.cpp backend (`ggml-axcl`) that drives the card's AX8850 NPU directly — Qwen3-0.6B runs **straight from a GGUF file** (q8_0 and Q4_K_M both work, no model conversion), with every transformer layer compiled as a single NPU graph and the KV cache living on the card.

## The numbers (Pi 5 host)

| | decode | prefill | Pi CPU load | card memory |
|---|---|---|---|---|
| **llama.cpp + int8 engines** | **19.6 tok/s** | 18.5 tok/s | **2% of one core** | 1.3 GB |
| llama.cpp, dynamic GGUF weights | 10.2 tok/s | 5.0 tok/s | ~5% | 2.4 GB |
| bundled closed runtime (same card) | 13.5–14.5 tok/s | — | — | — |

So the open stack now beats the card's own bundled runtime by ~40%, while leaving the Pi's CPU essentially free (it only tokenizes and samples). That headroom means you could run the Pi's CPU cores for other work while decoding. Numbers are for Qwen3-0.6B at 2048 context; identical between q8_0 and Q4_K_M GGUFs in the fastest mode (the NPU engines carry the weights there; the GGUF drives tokenizer/sampling).

## What made the difference

1. **Killing per-token host overhead.** The card's runtime charges ~1 ms for any small unpinned memory transfer. The backend now pins everything, refreshes per-layer state once per token instead of 28 times, binds engine IO buffers once, writes K/V rows directly into the on-card cache, and defers syncing the KV cache back to host memory until something actually needs it. Host cost per token: 35 ms → ~2 ms.
2. **The int8 engines.** The AXERA-TECH w8a16 engine set for Qwen3-0.6B (same file layout the vendor runtime uses) runs the NPU's native int8 path — half the time and a third of the memory of the bf16 engines I compiled myself. The backend speaks the same IO protocol to both.
3. **Pipelining** the 28 per-layer engine calls on one async stream.

There's also a fully "GGUF-native" mode where the GGUF's own weights are patched into the engine files at load (that's the 10.2 tok/s row) — that one needed reverse-engineering the engine weight layout, which is documented in the repo.

## Try it

Build llama.cpp with `-DGGML_AXCL=ON`, install the engine set, and:

```bash
GGML_AXCL_LAYER=1 GGML_AXCL_FA=1 GGML_AXCL_STREAM=1 \
GGML_AXCL_LAYER_DIR=$HOME/Qwen3-0.6B \
GGML_AXCL_POST_MODEL=$HOME/Qwen3-0.6B/qwen3_post.axmodel \
./llama-simple -m qwen3-q8.gguf -n 128 "your prompt"
```

Full instructions, the benchmark matrix (both quants), E2E test results (12/12), and the whole research log are in the LLMTest README.

## One heads-up for M5Stack/Axera

The card's engines contain a 128-token prefill "ladder" that would make long prompts process many times faster — but it doesn't function through the PCIe host runtime (the engine ignores the input binding for those shape groups; the bundled runtime never uses them either, which we verified by tracing it). Reported to Axera with full details. And a practical tip: if a run gets killed mid-flight, the card driver can wedge (next process hangs loading engines) — a Pi reboot fixes it.

Happy to answer setup questions — and thanks M5Stack for making this little card available; it's been a great NPU sandbox.
