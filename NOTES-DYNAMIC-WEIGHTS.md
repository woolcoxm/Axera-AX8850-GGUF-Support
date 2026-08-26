# Whole-layer engine + dynamic GGUF weights — session state (2026-08-25)

## Goal
1 engine call per transformer layer with weights sourced from GGUF at load time
(pure dynamic GGUF; axmodels may exist only as architecture templates).

## Established facts
- Card: AX8850 (identifies via axcl-smi), accepts AX650A-target axmodels. Driver V3.6.5_P1.
- Vendor Qwen3-0.6B w8a16 package on Pi: ~/Qwen3-0.6B (28 layer axmodels + post + bf16 embed bin).
  Vendor layer engine = ONE call: IO {K_cache[1,2048,1024]bf16, V_cache, indices u32 [1,1],
  input[1,1,1024]bf16, mask[1,1,2049]bf16} -> {K_cache_out, V_cache_out, output}.
  10 shape groups (decode m=1 + prefill ladder 128..1024). 1499us/layer decode (2048 ctx).
  Vendor runtime (main_axcl_aarch64 from Qwen2.5-1.5B pkg + tokenizer HTTP service):
  13.48 t/s on our card. README claims 16.9.
- Our local llm_build WORKS: pulsar2 llm_build on real HF Qwen3-0.6B -> 28 layers + post,
  kv 256 ctx: 1067us/layer on card. Output dir /tmp/llmbuild_out (real), /tmp/llmbuild_out2 (rerun),
  /tmp/llmbuild_marker (marker patterns, running).
  Command: FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build --input_path <hf_dir> --output_path <out>
  --hidden_state_type bf16 --kv_cache_len 256 --prefill_len 128 --last_kv_cache_len 128
  --chip AX650 -c 0 --parallel 8 -w s8
- QuantAxModel layer ONNX captured via sitecustomize hook (onnx.ModelProto.SerializeToString
  patch at $P7/python3/.../site-packages/sitecustomize.py) -> /tmp/qam_dump_*.onnx.
  Structure: per shape group a `neu` custom-op node (empty domain); attrs npu_graph_info JSON:
  {"dotneus":[{"neu_key":"subgraph_npu_X_b1_neu","batch":1,"extra_inputs":[{"name":"params",
  "const_data_key":"npu_params"}]}]}. Initializers: npu_params UINT8 (17,360,392 B layer,
  168,711,940 B post) + 3 neu microcode blobs (~190-370KB each, DIFFER per layer by ~224B —
  scales or layer consts embedded?).
- npu_params internal: 0-1,294,336 differ-per-layer; 1,294,336-1,552,384 IDENTICAL across
  layers (rope table? 252KB); 1,552,384-17,358,848 differ (15.4MB weights).
  Expected int8 weight mass 14.36MB (q 2M k 1M v 1M o 2M? NOTE o_proj [1024,2048]=2M,
  gate 3.1M up 3.1M down 3.1M = 15.4MB — matches region 2 exactly!).
  Region 1 (1.26MB) = probably q/k/v (2M+1M+1M=4M? no... TBD via marker analysis).
- Making npu_params a graph INPUT: pulsar2 build fails "backend not support yet"
  (native_parser, obfuscated). Both QuantAxModel and OptimizedQuantAxModel.
- OptimizedQuantAxModel compiles AxQMM+standard-ops mixed graphs but embeds ONNX CPU
  subgraph -> axclrt LOAD FAIL. Pure-AxQMM (QuantAxModel) loads fine (verified, 248us 1024^2).
- axclrt API: axclrtEngineLoadFromMem(model bytes, size, &id) EXISTS -> runtime patch plan.
- bf16 hidden states end-to-end (embed bin bf16, IO bf16).
- M5Stack apt repo (repo.llm.m5stack.com) has axclhost only. HF: AXERA-TECH org
  (Qwen3-0.6B w8a16, GPTQ variants, Pulsar2, AXCL repos).

## Current plan: runtime weight patch
1. Marker build: /tmp/qwen3_marker checkpoint (pattern weights from seed 20260825,
   even layers amp 1.0, odd 0.5; norms real; shapes list in gemm/analyze_marker.py).
2. gemm/analyze_marker.py decodes layout: matrix offsets via pattern search
   (per-row-sym quant is amplitude invariant), interleave variant search,
   even-vs-odd blob diff = scale tables.
3. Loader: GGUF -> int8 quantize (scheme per marker findings) -> scatter into npu_params
   blob -> patch axmodel file bytes -> axclrtEngineLoadFromMem. Zero compile at load.
4. Integrate in ggml-axcl.cpp: bf16 activations, device KV cache, indices tensor,
   mask, 1 call/layer, post engine for logits.

## Fallbacks if patch plan fails
- AxQMM chained engines (verified working int8 dynamic) + device-resident elementwise.
- llm_build at GGUF load (5-6 min, cacheable) — user prefers not.

## Key files
- gemm/test_vendor_layer.c (tvl on Pi): vendor/our layer bench harness.
- gemm/analyze_marker.py: layout decoder.
- gemm/probe_frontend.py: (model_type, ops) partition prober.
- /tmp/qam_dump_*.onnx: captured QuantAxModels (28 layers + 1 post).
- Pi: ~/matmul/{im,tvl,tffn}, /usr/local/share/ggml-axcl engines, models in ~/models.

## Layout crack session (late 2026-08-25) — FC path SOLVED

### Tiny FC (model_type=ONNX, MatMul [K,N], W=[N,K] f32) — COMPLETE
- Quant: per-output-channel symmetric int8 (weight_scales in quant_axmodel.onnx),
  W_trans initializer int8 [N,K] row-major, EXACT (verified 100% on controlled weights).
- Compiled storage: pos(r,k) via TABLE /tmp/layout_table_512x256.npy [512,256] -> byte offset.
  stored byte = V + 128 (offset-binary uint8).
  Structure: 144-byte chunks = 4 rows x 36 k-bytes; chunk = k//36, sub = row%4 (block-specific);
  blocks of 8 chunks (1152B) for K=256; row->block interleaving (rows {2b,2b+1}+{2b+32,2b+33}
  for N=512) + k-chunk ROTATIONS for rows >= N/4 (empirical table captures all).
- Extraction method: 6 builds with V = 16*code + dither(seed 31337), V[:,0]=127 anchor
  (uniform rowmax=127 -> scale 1/127 -> stored=V+128 exactly). code bits encode r (builds 0-2)
  and k (builds 3-5), 3 bits each. Dither cross-validation kills false positives.
  Reconstruction verified 99.97%. Scripts: gemm/shape_probe_pack.py (per-shape pack generator,
  needs per-K calibration tar /tmp/calk{K}.tar + config /tmp/layout_cfg_{K}.json).
- 37 shape-probe builds compiled OK for shapes: 2048x1024, 1024x1024, 1024x2048,
  3072x1024, 1024x3072 (outputs /tmp/shape_out_{N}x{K}_{b}/).

### Layer (llm_build) path — NOT YET CRACKED
- Marker blob (identical patterns, 1.0/0.5 amplitude): weight bytes amplitude-INVARIANT
  (same under any sym per-row/group scheme), so stored = normalized quant under SOME scheme.
- Direct searches FAILED for: per-row sym (127 and 128 scale), per-group sym (g=32/64/128/256
  along N or K), asymmetric per-row, row/col orientation, raw/+128. => transform is deeper
  (scrambling, int4 packing, or folded scales).
- Scale tables: (bf16 scale, int16 val) 4-byte pairs in 126B clusters ~17KB apart inside blob;
  bf16 exactly 2x between 1.0/0.5 amplitude layers; int16 amplitude-invariant.
- npu_params regions: 0-1,294,336 per-layer data (weights? attn structs?), 1,294,336-1,552,384
  IDENTICAL across layers (rope?), 1,552,384-17,358,848 per-layer (15.4MB weights).
- np.concatenate/packbits/tobytes hooks: NO large-array calls (assembly in torch or C++).
  torch hooks added (sitecustomize): torch.cat / Tensor.numpy intercepts + array dumps.
- NEXT: analyze torch traces from marker rebuild; OR differential marker builds
  (one matrix changed per build) to locate matrices, then code-encoded marker builds
  (V=16*code+dither per matrix, anchors V[:,0]=127) -> decode layer layout table directly.

### Key insight for final loader
Whatever the layer layout is, it is DETERMINISTIC (same arch -> same table) and
amplitude-invariant for weight bytes. Loader = quantize GGUF -> scatter via table -> patch
scale tables (format TBD) -> axclrtEngineLoadFromMem.

## LAYER LAYOUT CRACKED (2026-08-26 early AM) — nibble weights 100% verified
- Method: 9 tiny llm_build runs (2-layer, 4k-vocab checkpoints, ~1 min each; generator
  gemm/mk_code_marker.py): V = 16*code(r,k) + dither per matrix (V[:,0]=127 anchors),
  builds cm0/cmb2/cm1/cm3..cm7 = 22 coordinate bits (3/build), cm2 = dither-seed probe,
  cmx = per-matrix bit-plane build (offsets q:0 k:3 v:6 o:9 gate:12 up:15 down:18) to
  disambiguate same-shape matrices (k/v, gate/up). Dumps in /tmp/cmdumps/ + gemm/layout_artifacts/.
- FINDINGS:
  * ALL weights stored as INT4 nibble pairs: byte = (q4_hi << 4) | q4_lo, q4 = code + 8
    (unsigned offset 8). 8.42M nibble bytes = 15.4M elements. Dither discarded (4-bit!).
  * ~40% of elements stored TWICE (decode-group + prefill-group weight copies).
  * 72-byte repeating unit in weight region: 6 meta | 18 nibble | 18 meta | 18 nibble | 12 meta
    (36 nibble bytes = 72 elements + 36 metadata bytes). Metadata bytes: dither-sensitive,
    values ~uniform 0..204, UNCORRELATED with adjacent pair values => NOT per-pair scales;
    likely second weight representation (int8 copy for some subgraph) or other tables. TBD.
  * Scale tables: (bf16 scale, int16) 4-byte clusters ~126B every ~17KB; bf16 halves exactly
    between amplitude-1.0/0.5 marker layers; i16 amplitude-invariant. Likely per-channel
    scales (12,288 channels x2 copies ~ 24.6K entries ~ matches cluster mass).
- VERIFIED: nibble decode = 100% match on all complete pairs (5.16M pairs + 12.4M elements,
  zero mismatches). Coverage 99.93-99.98% per matrix. Table: gemm/layer_layout_v3.pkl
  {(matrix, r, k) -> (pos, 'hi'/'lo')} (first-claim only; multi-claims = duplicated elements,
  full claim lists recoverable by rerunning decode with claims dict).
- REMAINING for the loader:
  1. Scale table mapping: locate per-channel bf16 scale entries (channel -> blob offset),
     decode exact scale semantics (likely w = (q4-8) * scale_ch; loader recomputes both
     q4 and scale from GGUF row maxabs: scale_ch = maxabs/7-ish, verify sign handling
     via a controlled build with negative weights!).
  2. The 36 meta bytes per 72B unit: identify (differential builds with same codes+dither
     but permuted... or zero-weight builds) — may need patching too if weight-dependent.
  3. k=0 anchor elements (12,288) map to nibble=15 slots; recover via elimination.
  4. Loader + on-card verify vs baked engine, then ggml-axcl integration.
- IMPORTANT: weights are 4-BIT in these engines (even with -w s8?!). Accuracy check needed
  vs vendor w8a16 build (their blob had 15.4MB int8-looking mass; ours 7.7M nibble+7.7 meta).
  Reconsider: maybe -w s8 got ignored for layers, or 4-bit + 8-bit-meta = w8 effective.
