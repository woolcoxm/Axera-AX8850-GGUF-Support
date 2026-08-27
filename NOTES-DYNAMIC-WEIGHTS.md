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

## THROUGHPUT SESSION (2026-08-26 night) — 7.9 -> 19.6 t/s decode

### Baseline measurement (per-token budget at 7.96 t/s = 125.6 ms/tok)
- engine execute: 3.235 ms/layer avg (us accumulator delta over 3500 calls) -> 90.6 ms
- host-side: ~35 ms (28x [4B idx H2D from STACK + 7 binds + 4KB mask D2D +
  2x 2KB D2D scatter + 2x 2KB D2H writeback + scalar bf16->f16 loops])
- KEY: unpinned small transfers cost ~1ms each on this stack (page pinning
  per transfer) — the idx/row stack buffers were the dominant host cost.
- "0.21 t/s prefill" in short-prompt runs is engine swap cost billed to
  prompt-eval by llama-simple; not per-token.

### Host optimization pass (items: call diet + KV defer + NEON + pinned)
- per-TOKEN (not per-layer) idx upload + mask-row D2D (synced_pos guard)
- static IO bindings once per engine (K/V/idx/mask); only hidden ping-pong
  + K/V rows rebind per call; K/V outputs bound DIRECTLY into cache rows
  (2KB aligned; mask protects the row from same-call reads) -> both D2D
  scatters gone (GGML_AXCL_KV_INPLACE=0 restores old path)
- host KV write-back DEFERRED via host_wm[l] watermark (flush at resync /
  every 32 pos / non-armed graph entry / bail; batched contiguous D2H into
  pinned 64-row staging). GGML_AXCL_KVWB=now restores per-call writes.
- NEON bf16<->f32 helpers (axcl_bf16_to_f32 / axcl_f32_to_bf16)
- pinned staging for idx/rows/hidden/logits (304KB logits D2H was unpinned
  std::vector)
- RESULT: 9.91 t/s (100.9 ms/tok; host 35 -> 10.5 ms). E2E 12/12 PASS.

### Weight dtype dead ends (measured, not guessed)
- -w fp8_e4m3 full build: engines SAME size (65.6MB), SAME speed (9.94 t/s,
  90.1ms eng) — flag changes blob packing, not card-side cost. Layer path
  is NOT a "same pipeline, fewer bytes" situation.
- --post_weight_type fp8_e4m3: post engine 621MB (bigger than s8's 170MB).
- s4 post: not an accepted choice for --post_weight_type.
- post engine is s8 by default already (169.7MB); NOT GGUF-patched (runs
  template lm_head; part of the 94% agreement gap).

### The real decode win: vendor w8a16 engines
- ~/Qwen3-0.6B engines: 23.1MB/layer (int8 path) vs our 65.6MB bf16
- SAME filenames + IO conventions -> just GGML_AXCL_LAYER_DIR=~/Qwen3-0.6B
- 1.51 ms/layer -> 52.9 ms/tok, 18.9 t/s; +GGML_AXCL_STREAM=1 (28 execs
  async on one stream, sync after layer 27) -> 19.5 t/s
- FINAL: 19.6 t/s decode. BEATS vendor closed runtime (13.48 measured).
- E2E 12/12 PASS on this mode too. Governor=performance on Pi.

### Prefill chunk ladder: fully mapped, PROVEN UNUSABLE on axcl V3.6.5
- Vendor engines have 10 shape groups. group 0 = decode m=1; groups 1..9 =
  128-token chunks, prefix ladder 0..1024:
  g1: K/V [1,1,1024](dummy), idx [1,128], in [1,128,1024], mask [1,128,128]
  g(n>1): K/V [1,(n-1)*128,1024], mask [1,128,(n-1)*128+128]
- size API (axclrtEngineGetInputSizeByIndex(info,g,i)) matches dims products.
- test_chunk.c harness (per-token reference vs single group-1 call):
  K_out differs even with identical x + rope 0 (max diff 139; K is a pure
  function of x+pos — cannot be a convention issue we control).
- ONE-HOT test: zeroing 127/128 input rows leaves output essentially
  unchanged -> the engine IGNORES the bound input for chunk groups.
- Runtime logs internal "[memory][memcpy] nil pointer" on chunk executes.
- Vendor's OWN host runtime never calls groups != 0 (LD_PRELOAD trace of
  main_axcl_aarch64 on a 300-token prompt: 18089x GROUP=0, zero others).
- Chunk path kept behind GGML_AXCL_BATCH=1 (defaults off). Per-token
  prefill through group 0 with vendor engines: ~55 t/s steady (3.9x the
  14 t/s at session start; ~4x vendor runtime's own prefill).

### Misc
- vNPU = runtime NPU partitioning (axclrtEngineInit kind DISABLE/STD/
  BIG_LITTLE/LITTLE_BIG). We use AXCL_VNPU_DISABLE = full NPU — correct
  for single-stream; partitioning only matters for multi-process tenants.
- pulsar2 --npu_mode {NPU1,NPU2,NPU3} is the compile-time core selection;
  -c is check_level (not cores). Default build = all cores. NPU1 build
  matrix untested (low expected value while DRAM-bound).
- sitecustomize.py tensor-dump hook was filling /tmp (tmpfs 16G) ->
  Errno 122 on builds; disabled (kept as .research-disabled).
- Killed runs can wedge the card (PCIe DMA errors on next load) — reboot
  the Pi to recover; governor resets to ondemand (set performance).

## VENDOR w8a16 LAYOUT CRACK (2026-08-27) — npu_params structure SOLVED, marker decode underway

Goal: patch GGUF weights into the vendor int8 engines (19.6 t/s path).

### Container (gemm/walk_axmodel.py + gemm/extract_npu_params.py)
- axmodel = protobuf. Top: f1 varint, f2 "Pulsar2", f6 version str,
  f7 = big graph section. Inside f7: 10x f1 group IO descriptors,
  f2 "ax-model", f5 = npu_params (f8 name + f9 data), then 10x f5
  microcode segments (compressed -> per-layer file sizes vary).
- Vendor 5.2 engines (Pulsar2 5.2, commit a3f2fda4): npu_params at file
  offset 5035, length 19,226,120 B, IDENTICAL length in every layer.
  Our 7.0 bf16 engines: npu_params at 1570.
- Vendor build command (their README): llm_build --kv_cache_len 2048
  --prefill_len 128 -c 1 --parallel 32 --last_kv_cache_len 128..1024
  (8 rungs) -w s8, FLOAT_MATMUL_USE_CONV_EU=1. 10 shape groups.

### npu_params region map (l0 vs l1 byte diff)
- [0..8198] shared [i16 1267][bf16 .0884] repeating entries
- [8199..12290] per-layer ~4KB, [12291..16392] shared
- [16393..1297926] per-layer 1.28MB (R4; contains some q data)
- [1297927..3395592] shared 2.1MB (rope-ish table)
- [3395593..19199494] per-layer 15.8MB (R6; weights + meta, ~1:1 ratio)
- [19199495..end] shared 26.6KB

### Established storage facts
- Weights = INT4 NIBBLE PAIRS: byte=(v_odd<<4)|v_even, v=q4+8 (bell
  nibble histogram; exact anchors). R6 ~ 2x nibble mass -> ~1:1 meta.
- Weights appear EXACTLY ONCE per nibble stream (anchor uniqueness
  search) — but ~34% of ELEMENTS stored twice (21.16M nibble slots for
  15.73M elements on the marker build) = decode+prefill group copies,
  same ratio as the 7.0 s8 build.
- Quant at anchored windows = norm-FOLDED per-row symmetric d7 RNE
  (three distinctive anchors match exactly; folding confirmed: in_ln
  gains mean 0.17, max 1.05).
- BUT the vendor's integer values are NOT deterministic RTN of HF
  weights: cross-layer validation fails (0/414 anchors reproduce in l1
  at chance rate) -> their quant is calibration/activation-aware
  (GPTQ-like). Embeddings match HF byte-exactly (sha256), same
  checkpoint. Layout itself IS layer-independent (marker l0/l1 with
  identical weights -> byte-identical npu_params).
- Storage model (fits all evidence incl. 7.0's "36 meta bytes per 72B
  unit... likely second weight representation"): the nibble plane is
  the TOP nibble of an int8 quantization; the "meta" mass carries the
  LOW nibble (-> w8 effective). 7.0 notes' "nibble = (q8>>4)+8
  arithmetic shift" was the same phenomenon.

### Marker-build path (the crack)
- Pulsar2 5.2 LITE already local: pulsar2/5.2/5.2/ax_pulsar2_5.2_lite_package.
  Marker ckpts via gemm/mk_code_marker.py (2-layer, 4k vocab; now with
  d2 = dither-probe and mc = matrix-code modes).
- Build cmd = vendor's exact flags but -c 0 (markers fail -c 1 check;
  layout is check-independent). ~4.5 min/build.
- Marker npu_params = 19,212,296 B (13,824 B shorter than vendor's —
  scale-table section size differs; region map otherwise matches:
  shared 2,097,336B table at ~1.29M, tail, etc.).
- Decode: gemm/decode_v52_layout.py (nibble claims via code builds +
  matrix codes; dither-invariance cm0 vs cmd2 validates code slots).
  NEXT: run remaining builds (cmB2=bits 6-8 was MISSING from the first
  batch — added later), decode, then re-anchor onto vendor engines
  (handle the 13.8KB section shift), validate, then scale tables
  (amplitude 1.0/0.5 builds) + loader.

### Reusable scripts added this session
- gemm/walk_axmodel.py (protobuf field-tree dumper)
- gemm/extract_npu_params.py (npu_params extractor for any axmodel)
- gemm/anchor_int8.py / anchor_nibble.py (scheme sweeps; superseded by
  marker path but document the quant findings)
- gemm/build_v52_markers.sh (serial marker builder)
- gemm/decode_v52_numpy.py (claims decoder, memory-frugal numpy form)
- Vendor engines l0/l1 + package docs: gemm/baked/vendor_w8a16/
- All marker npu_params blobs + claims/fine tables: gemm/baked/v52_markers/
  (gitignored, 220MB)

## VENDOR w8a16 LAYOUT — DECODED (2026-08-27 late)

Marker builds on Pulsar2 5.2 with the vendor's exact flags (10 code/dither/
matrix/amplitude builds; cmB2 = bits 6-8 was needed beyond the first batch).
Decoder: gemm/decode_v52_numpy.py -> v52_claims.npz + v52_fine.npz.

### THE FORMAT (w8a16, Pulsar2 5.2, -w s8 llm_build path)
- Effective weights are INT8, stored as TWO nibble planes:
  * coarse byte at p: (v_{k+1} << 4) | v_k for the element pair
    (even k, odd k+1) of the same (matrix, row); v = (q8 >> 4) + 8.
  * fine byte at p - 18: (lo4(q8_{k+1}) << 4) | lo4(q8_k).
    (lag -18 verified on 119,297/120,000 sampled pairs; 99.4% overall
    assignment = unit = 18B fine | 18B coarse repeating.)
  * q8 reconstruction: int8 = ((coarse_nib - 8) << 4) | fine_nib.
- Claims coverage: ~100% of all 7 matrices' elements
  (q 2,095,108/2,097,152 + recover k=0 anchor columns by elimination;
  only (m,0,0) elements entangled with zero-fill garbage).
- Intra-byte pairing: hi nibble = odd k, lo = even k (7.0 convention
  confirmed; earlier 0-count was my check bug).
- ~1.04M bytes of 0x88 zero-fill spread uniformly (decodes as all-zero
  codes; skip). ~210K full-byte int8 positions (code hi + dither lo)
  exist (like 7.0's "8-bit positions"); low priority (coverage complete
  without them).
- QUANT: codes are dither-INVARIANT (cm0 vs cmd2 builds) -> top nibbles
  deterministic; vendor integer values are NOT RTN of HF weights
  (activation-aware calibration in llm_build's yasched llama_test pass)
  -> for GGUF patching WE choose the integers (RTN per-row int8 sym),
  which is self-consistent; agreement vs their calibration is moot.
- SCALE TABLE (mixamp amplitude diff): 960 clusters x 254B ~= 61,440
  entries = per (row, kgroup-256). Entry = [u16 A][u16 B]: A drops by
  EXACTLY 128 (1.0 in 24.7 fixed point) when all weights halve ->
  A/128 = C - log2(scale)-like; B amplitude-INVARIANT (mantissa or
  index). A varies within a cluster despite uniform groupmax ->
  calibration-dependent. OPEN: exact formula + loader-side synthesis
  (may need to replicate llm_build calibration stats, or find that B
  encodes the per-group scale bf16 and A is a runtime-correctable).

### NEXT STEPS (for the loader)
1. Scale entry semantics: decode A,B across the 10 marker builds
   (codes differ, scales same -> separates calibration-input effects).
2. Transfer claims to vendor engines: marker npu_params is 13,824B
   shorter; section map shifts piecewise (head +56, R4 end -8.8K,
   rope end -9.2K, R6 end -13.9K). Align per-section (rope table is a
   fixed anchor) or compare against a real-weights 5.2 build (use
   --parallel 4..8, NOT 32 — the full-model build + analysis OOM'd the
   30GB box once; keep python dicts out of analysis, numpy only).
3. Round-trip: re-encode marker weights from claims+fine -> byte-exact
   engine (validates the map end to end).
4. Patch real GGUF weights into a vendor engine, on-card test on the
   Pi (test_vendor_layer.c harness), then ggml-axcl loader integration.

### Machine etiquette (learned the hard way)
- python dict-of-15M-tuples ~= 10GB RSS; the claims now live as numpy
  structured arrays (~300MB). /usr/bin/time -v everything big.
- --parallel 32 full-model builds + analysis = OOM. Markers OK (tiny).

## VENDOR w8a16 — FULLY CRACKED + ON-CARD VALIDATED (2026-08-27 night)

### Everything confirmed in this session
1. OUR 5.2 rebuild of HF Qwen3-0.6B with the vendor's flags is
   BYTE-IDENTICAL to the vendor engines (0/19,226,120 bytes differ).
   The engines are fully reproducible from the open checkpoint; quant is
   deterministic. (gemm/baked/v52_real/ = our build.)
2. QUANT FORMAT (final): weights UNFOLDED (norms run at runtime — unlike
   7.0 builds!), per-ROW symmetric scale = rowmax/127, int8 RTN + ~0.3
   sparse Hessian-ish corrections per row (4,030/15.7M elements differ
   from RTN). Decoded-engine-int8 vs RTN agreement 99.97%.
3. Claims->vendor map EXACT (from claim-order monotonicity): R4 +8192;
   R6: +9216 / +9728 (after marker 4,523,255; a +512 insertion) /
   +13824 (after marker 8,936,951; a +4096 insertion).
4. SCALE-ENTRY TABLE (exact map): real-vs-realmix diff = 960 clusters,
   53,258 bytes ~= 13.3K 4-byte entries ~= per-row. Entry [u16 A][u16 B]:
   A shifts EXACTLY -128 (24.7 fixed = log2 step) when weights halve; B
   weight-dependent. NORM-ENTRY TABLE (real-vs-realmorm diff): 3,139
   bytes, zero overlap with scales. Both must be patched for arbitrary
   GGUFs (currently left untouched = identity-patch-consistent).
5. WRITE-PATH GUARDS (each found via single-byte on-card bisection):
   - scale entries (136KB mask, +/-16B dilated)
   - 0x00 inactive slots (writing one detonates the engine: e25 y)
   - insertion-edge bytes (+/-2KB windows at the two insertions)
   - read-verification: keep only claims whose stored nibble matches
     RTN(reference) within +-1 (subsumes the above; 99.998% pass)
   - fine-plane writes only at dither-verified per-pair positions
6. ON-CARD A/B (engine_dump.c, seeded inputs, deterministic):
   identity-patched l0 engine vs original: K/V_out meanabs 1.8-5.4%
   of signal, layer output y cosine 0.9969, diff spread broadly (no
   structural corruption). CONTROLS: 100 one-step nibble perturbations
   in v -> y exactly unchanged (attention-neutral mask); in gate/up/
   down -> y ~0.003% (noise floor reference).
7. Test-harness lessons: all-ones KV/mask input makes attention scores
   exactly tied -> softmax amplifies +-1 weight noise to e25 (chaos
   artifact, not corruption — wasted hours on it). Use seeded generic
   inputs + controls before believing an A/B diff. K_cache_out is a
   single 2KB row (not the full cache). Mask self-slot is 2048.

### Files (loader core + validation)
- gemm/patch_vendor_w8.py — the patcher (identity mode; GGUF-mode TODO:
  patch scale+norm entries, quantize arbitrary weights)
- gemm/engine_dump.c — deterministic engine A/B harness (Pi)
- baked/v52_markers/: claims/anchors/fine tables, real_l1/realmix_l1/
  realmorm_l1 npu_params (scale/norm masks), vendor blobs
- baked/v52_real/ — our byte-exact vendor-equivalent engine set

### NEXT for GGUF patching (production)
1. Scale-entry WRITE format: decode A/B semantics fully (A = C -
   128*log2(scale)?), patch per-row scales for arbitrary weights.
   Norm entries likewise (from GGUF norms).
2. Extend patcher: GGUF dequant -> per-row int8 (vs engine's own scales
   read back from A-entries, or freshly written).
3. ggml-axcl.cpp loader integration (template engines + sidecar +
   runtime patch, cache by weight hash — mirror of the bf16 path).
4. Rebuild marker set is NOT needed for other layers (layout is
   layer-independent; only scale/norm VALUES differ).

## GGUF-INT8 MODE SHIPPED (2026-08-27 late night) — 19.5 t/s, 96% agreement

`gemm/gguf_patch_w8.py`: full pipeline working end-to-end.

### The final recipe
- Dequant GGUF (Q8_0/Q4_K/Q6_K — validated vs ggml reference dequant,
  corr 0.9999/0.997) -> per-row symmetric int8 against the ENGINE'S OWN
  scales (= rowmax(HF-reference)/127; the engines are byte-identical to a
  5.2 rebuild of the HF checkpoint, so the reference rowmaxes ARE the
  engine scales).
- DUAL write filter: write an element only if (a) its mapping is verified
  — RTN(HF-ref) matches the stored int8 within +-1 — AND (b) the GGUF's
  value genuinely differs — |RTN(GGUF) - stored| >= 2. (a) drops
  mis-mapped claims/structural bytes; (b) preserves the engine's sparse
  activation-aware (GPTQ-like) corrections everywhere the GGUF agrees
  with the checkpoint, which is what makes the mode MORE faithful than
  vendor-engine mode (96% vs 94%).
- All structural guards from the earlier session stay (scale-entry mask,
  0x00 inactive slots, insertion windows on coarse AND fine targets).

### Results (Pi, governor performance)
- decode 19.4-19.5 t/s, prefill 17.5 t/s, CMM 1.1GB, host CPU idle
- E2E 12/12 PASS (5 after a stale-process retry — see below)
- agreement vs CPU reference: 153/159 prefix tokens = 96%, 4/10 exact

### Hard-won gotchas this session (all cost hours — read before touching)
1. THE POST ENGINE. `rm ~/dir/*.axmodel` before re-deploying layer
   engines also deletes qwen3_post.axmodel; if GGML_AXCL_POST_MODEL then
   points at nothing, the backend silently falls back to the legacy
   per-op path: 0.4 t/s AND garbage output (the legacy FA/output path is
   known-wrong). Symptom signature: "slow + garbage" = missing post,
   "fast + garbage" = weight problem. Always re-check the post file.
2. An LS-fit scale (vs the engine int8s) is systematically ~0.2% low; the
   engine dequantizes with ITS stored scale, so the bias multiplies every
   weight the same direction and compounds over 28 layers -> fluent but
   wrong generation. Always quantize against rowmax(HF)/127 exactly.
3. Overwriting the engine's ~4K/layer activation-aware corrections with
   plain RTN (write-everything strategy) degrades output measurably.
   Writing ONLY >=2-step diffs while dropping mis-mapped claims (the dual
   filter) is both the most faithful and the safest.
4. A stale llama-simple holding the card makes the next run's engines
   fail to load mid-suite (rc=1 decode failures). Check axcl-smi /
   pgrep before running suites; re-run failures once the card is free.
5. axcl-smi lives at /usr/bin/axcl/axcl-smi (not on PATH) on this Pi
   image. CMM baseline 18 MiB.
