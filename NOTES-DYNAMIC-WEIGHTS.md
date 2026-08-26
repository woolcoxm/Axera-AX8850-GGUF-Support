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

## WHOLE-LAYER ENGINES WORKING (2026-08-26 mid-day) — 7.6-8.3 t/s coherent

`GGML_AXCL_LAYER=1 GGML_AXCL_FA=1` on the Pi: 28 engine calls/token,
device-resident bf16 hidden chain, per-layer device KV caches. Prefill runs
the engine per token (autoregressive loop) — 8.1-8.3 t/s, ~3.5x faster than
the legacy per-op prefill. Committed: llama.cpp 5e2b590.

### Engine conventions (all validated vs numpy, layer by layer)
- mask [1,1,2049] bf16: slots [0..2047] = caller cache, slot 2048 = the
  engine's internal SELF row. Row p: allow t<p plus t==T.
- indices u32: rope position + cache write slot.
- K_cache_out/V_cache_out [1,1,1024]: the new token's rows — caller scatters
  (D2D) into its cache copy and write-backs to the host cache.
- The runtime MISHANDLES sub-buffer bindings (offset into a table) for both
  mask and indices: refresh dedicated buffers' contents per call instead.
- Uninitialized device caches: NaN-shaped garbage poisons the softmax
  THROUGH the -inf mask (NaN + -inf = NaN). Zero-fill at alloc.
- The engine must never read its input from the buffer it writes
  (double-buffer the hidden yout).
- axclrtEngineLoadFromMem hangs (unmodified bytes!) in V3.6.5 — use temp
  file + LoadFromFile.

### llama.cpp integration shape (the hard-won parts)
- The scheduler only delivers the decode graph unsplit when (a) metadata
  ops are claimed AND (b) FLASH_ATTN_EXT is claimed (any claim flips
  llama.cpp onto its FA graph path; unclaimed = manual attention which the
  scheduler splits at every attention matmul because CPU owns the KV buft).
- THE FA HOST-OP IS STILL WRONG (engine route AND scalar fallback produce
  wrong outputs on real graphs) — claiming FA is safe ONLY because armed
  graphs skip the node entirely (subsumed by the layer engine).
- Output delivery: the graph's LAST residual ADD is the fragment output;
  backfill the engine hidden there and disarm IN-LOOP so the final
  RMS_NORM + weight MUL (in the same fragment!) compute via host ops; the
  vocab matmul stays on CPU.
- First-RMS_NORM-after-anchors = layer 27's post-attention norm (trap!).
- Cache-base detection for resync: [1024, >=ctx, <=8192] excludes the
  embedding/lm_head tables which match the same shape filter.

### Weight dtype (the garbage-output root cause)
- `-w s8` templates store INT4 weights: 3%/layer drift vs f32 reference,
  garbling generation. The chain-vs-numpy harness quantified it cleanly.
- `-w bf16` templates: coherent everywhere (28x66MB, DRAM-bound 3.2ms/layer).
- fp8_e4m3 under test (would halve weight reads; quality borderline).

### Layout research state (dynamic GGUF loader)
- layer_layout_v3.pkl VINDICATED: pq16 (+16 code shift) probes put exactly
  128/128 nibble bytes on the probed group's claimed positions. Earlier
  zero-hit results were dump-collection races — serial builds + content
  checks added to run_perturb_build.sh.
- Sign semantics: nibble = (q8>>4)+8 ARITHMETIC shift; hi nibble = odd k,
  lo = even k (NOTES' earlier claim was inverted).
- Scales: [i16 516][bf16 s] entries, per (row, kgroup-256); value =
  bf16(groupmax/127); 30,700 slots reconstruct EXACTLY (RNE) vs sdA.
- Norm folding: llm_build FOLDS norm gains into the projection weights
  (nw marker changed 16M blob bytes; templates must be built with the
  target model's norms or the loader must fold GGUF norms before quant).
- REMAINING for patch-at-load: the second weight representation (the
  non-nibble region is load-bearing per acid test 2), scale slot map
  completion (sdc2/sdc3 dumps were collection-poisoned; rebuild serially).

## DYNAMIC GGUF WORKING (2026-08-26 evening) — commits ce07a2f + 567a973

`GGML_AXCL_GGUF=1 GGML_AXCL_LAYER=1 GGML_AXCL_FA=1`: the GGUF's own weights
run through the whole-layer engines. 12/12 E2E pass. q8_0 + Q4_K_M verified.

### The bf16-template crack (completely different from the s8 nibble maze)
- `-w bf16` templates store weights as RAW bf16: no norm folding, no quant
  transform, no scale entries. l0+l1-weights repatch = baked l1 minus 3
  layer-index microcode bytes (65164243/65355556/65607129) — and patched
  engines compute IDENTICALLY to baked (on-card acid test, 0.0 diff).
- Layout decoded by VALUE-ANCHORED SEARCH (anchor_real_layout.py): for every
  matrix row, find its first-6-values bf16 sequence at stride 64 in the
  engine blob -> 100% of rows anchored for all 7 matrices (15.73M elements).
- Norm slots (input/post/q_norm/k_norm, 2304 entries) + down-tail (2687)
  via l0-vs-l1 diff windows + value-triple matching.
- Sidecar: gemm/layout_v4.bin (AXL4: u64 byte offsets per element).

### Backend loader
- Registry: prescan stashes leaf tensors per layer. GOTCHAS: node order is
  q,v,k (k's norm/rope chain longer) -> k = lower address of the 1024x1024
  pair; norm gains matched via RMS_NORM-linked MULs; o/down by shape.
- Swap timing is CRITICAL: patch+swap inside the FIRST armed graph's
  prescan, before any node runs. Swapping one graph later = prefill on
  template weights = mixed-model KV caches = degenerate output.
- The 28 template engines are unloaded (axclrtEngineUnload) before the
  patched set loads; patched files cached in /tmp/axcl-gguf.
