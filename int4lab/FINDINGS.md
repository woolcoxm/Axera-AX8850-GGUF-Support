# int4lab FINDINGS (session 2026-08-27, Phase R)

## The w4 landscape (measured, not guessed)
- `pulsar2 llm_build2 -w s4` = REAL 4-bit weights: Qwen3-0.6B layer engine
  s4 = 9,504,061 B vs s8 = 17,931,805 B (53.0%). Vendor 5.2 w8a16 = 23.1 MB.
- Docs (pulsar2-docs V7.0/6.0): llm_build2 verified models = Qwen3.5, Qwen3,
  Qwen2.5, DS-R1-Distill, MiniCPM4, InternVL2.5/3, ChatGLM3, OpenBuddy,
  SmolLM2, Llama3.2, Gemma2, Phi2/3, TinyLlama. AX8850 supported (SDK>=3.6.2).
- Community (nextgenredteam): s4 wants PRE-QUANTIZED GPTQ-Int4/AWQ group-128
  inputs (SRAM tiler groupN=128; raw fp16 falls back to groupN=1024 -> TileFail).
  They report MoE models FAILING on 6.0-era toolchain (mRoPE-related).
  AX8850 executes AX650-target axmodels natively; ~7040 MiB CMM visible.
- AXERA-TECH ships w4a16 packages since Pulsar2 6.0 (LlocateAnything-3B);
  Hojo-TTS upgraded s4->s8 in 7.0 for precision (quality note: s4 borderline
  for some tasks; fine as expert-weight format vs host Q4).

## convert_onnx_to_4w8f.py — CRACKED (obfuscated, Pyarmor 9)
- Trigger: node.op == DequantizeLinear (weight=inputs[0]) or Conv (inputs[1]),
  weight is Constant, 4D (Conv-shaped; [N,K,1,1] GEMM-form WORKS), and all
  values in int4 range [-8,7] (is_int4_range: 2**(4-1)).
- Output: weight TensorProto.INT4 packed (ORT pack_bytes_to_4bit): two's
  complement nibbles, LOW nibble = even element (288/288 verified on known
  pattern). zp retyped INT4 + renamed f"{node.name}_{tensor.name}". Scale
  untouched. Export via custom AxOnnxExporter (gs) — needs gs export_dtype
  guard bypass (our overlay pyoverlay/onnx_graphsurgeon).
- Conversion only; consumption blocked in stock build: PPQ executor can't run
  QDQ (ml_dtypes int4 arrays); QuantAxModel emits macs=0 hollow engines;
  INT4-typed graph INPUTS rejected ("Unsupported dtype int4 for randint").

## llm_build2 engine container — MAPPED (same family as llm_build1!)
- Top: f1/f2/f6 ver/f7 graph/f8 small/f14 extra_data (walk_axmodel.py).
- f7 (flat protobuf): N x [IO group: f1 inputs(K_cache,V_cache,indices,input,
  mask)+suffixes, f2 outputs, f3 'subgraph_npu_N', f4 'neu mode', f5 blobs],
  f2 'ax-model', f5 npu_params (f8 name 'npu_params', f9 DATA @+~1.5k),
  f5 microcode segments f8 'subgraph_npu_N_b1_neu' (~185KB each), f11 tail.
- s4 npu_params = 8,939,804 B. IO naming identical to vendor engines ->
  backend IO-by-name resolution transfers. extract_npu_params.py needs new
  offsets only.
- l0-vs-l1 (s4): 82% bytes differ; tiled per-layer structures ~18144 B stride,
  9471 B diff blocks; 12.4 KB diff region @3.74 MB (scale tables?).

## Runners / env
- int4lab/runpulsar2.sh, run4w8f.sh, run4w8f_52.sh (ld-linux loader dance,
  PYTHONHOME/PYTHONPATH set; quota: avoid home writes, builds -> /tmp).
- pyoverlay/onnx_graphsurgeon = patched gs (export_dtype guard removed).
- int4lab hooks/sitecustomize.py variants = proxy/profiler/excepthook
  harvesters (frame/code-object dumps WORK on pyarmor 9 group; dis/f_globals
  SEGFAULT — armor asserts).
- int4lab/scratch/out_s4 (28x9.5MB complete), out_s8 (finishing).

## Phase R status
- R1 (--model_type/--model_config) probe: RUNNING (bg).
- R5 ecosystem: DONE (above).
- R2 container: DONE (above).
- R4 (int4-input calib unblock): TODO.
- R6 (qwen3_moe through llm_build2): TODO — pivotal.
- R3 (marker decode of s4 npu_params): TODO — machinery transfers.

## Phase R RESULTS (2026-08-27)
- R6a: qwen3_moe DISPATCH-REJECTED ('model_type error qwen3_moe'). Mixtral/
  ernie_moe/deepseek_v3 also rejected. Path A (native qwen3_moe) CLOSED.
- R6b whitelist (dispatch-accepted): qwen3_5, qwen3, qwen2 (covers 2.5),
  gemma2, llama, phi3, chatglm, tinyllama. qwen3_5 ACCEPTED with wrong-tensor
  ckpt (failed later, bare AssertionError) -> config class exists; MoE-via-
  qwen3_5 needs faithful Qwen3.5 tensor naming (deltanet hybrid + MoE FFN) —
  TOP FOLLOW-UP: fetch Qwen3.5-35B-A3B safetensors index for names, fabricate
  tiny ckpt, retry.
- R4: int4 dynamic inputs: calib TARS bypass the randint gate (progress
  FrontendError->QuantError), but quant stage cannot process ml_dtypes.int4
  arrays ('can't convert np.ndarray of type ml_dtypes.int4'). DYNAMIC w4
  stays blocked; w4 is a STATIC engine format. MoE experts: int8 AxQMM
  dynamic (works) or static w4 engines.
- R3: marker crack TRANSFERS to llm_build2 s4. mk0/mk1 (repo mk_code_marker)
  built clean via llm_build2 -w s4 (files qwen3_lN.axmodel + post). npu_params
  per layer file @578, 8,727,812 B = 15.4M params @4bit + scales ✓. mk0-vs-mk1
  diff 79.3%; nibble hists: stored values 8-15 dominant (offset int4), zero-
  fill peaks. Full claims decode = mechanical continuation (9 builds).
- s8 twin set complete (int4lab/scratch/out_s8, 28x17.93MB + post).
- s4 quant wants GPTQ/AWQ group-128 prequantized inputs (fp raw -> groupN
  1024 fallback, can TileFail) — for REAL models feed GPTQ-Int4 ckpts.
- llm_build2 engine IO = K_cache/V_cache/indices/input/mask (+_N suffixes for
  shape groups) — SAME names as llm_build1 -> backend IO-by-name transfers.

## Phase C checklist (card window, ~15 min)
1. Load int4lab/scratch/out_s4 l0 + out_s8 l0 via axclrt (new container!).
2. Per-layer exec time s4 vs s8 vs vendor w8a16 (w4 speed thesis).
3. Marker-engine numerics vs numpy (ref_chain pattern).

## Phase C EXECUTED (2026-08-27 morning session) — s4 THESIS CONFIRMED + CHUNK LADDER UNBROKEN

Harnesses (new, on Pi as ~/phasec): `gemm/phase_c_timing.c` (generic group
dump + timed execute), `gemm/phase_c_refcheck.c` (chunk-group K rows vs
per-token decode-group K, same x/pos), `gemm/phase_c_onehot.c` (input-bind
probe). Build: `gcc x.c -I/usr/include/axcl -L/usr/lib/axcl -laxcl_rt
-laxcl_sys -Wl,-rpath,/usr/lib/axcl -lm`.

### BINDING RULE (the un-breaker — cost 3 failed variants to find)
axclrt executes correctly ONLY when: every input AND output tensor is bound
(including V_cache_out), each to its own exact-size dedicated buffer, one IO
handle per shape group, no offset/sub-buffer binds. Violations = silent exec
failure or stale reads. This was the real cause of the earlier "chunk ladder
broken" conclusion — not the engines.

### Timing (sync execute, 200 iters, card idle)
| engine | decode m=1 | chunk groups |
|---|---|---|
| s4 p64 kv255 (scratch/out_s4) | 772 µs | 64: 2148 µs, 64+px64: 3033 µs |
| s8 p64 kv255 (scratch/out_s8) | 1070 µs | 64: 2169 µs, 3031 µs |
| **s4 p64 kv2047 (rebuilt /tmp/int4lb2/out_s4_2048)** | **1166 µs** | same 2147/3032 |
| vendor w8a16 kv2048 (control) | 1500 µs | 128-ladder 2808→5220 µs |
| vendor post engine | 7917 µs — **1 group, m=1 only** | — |

- Two-point fit (s4 vs s8, same shape): marginal weight streaming ≈ **25 GB/s**
  (73% of the 34.1 GB/s LPDDR4x peak — DRAM is NOT the limiter) + **fixed
  ≈ 457 µs/call** (sync execute incl. PCIe round-trip). The w8a16 path's
  low effective GB/s was this fixed cost, not DRAM efficiency.
- s4 kv2047 decode = 1166 µs = **1.29× vs vendor w8a16**; at bench ctx (~470)
  ≈ 0.85 ms/layer → projected **~30 t/s decode** (vs 19.6) with unchanged
  post engine + host.
- s4 chunk64 == s8 chunk64 (m≥64 is compute-bound): prefill/verify passes do
  NOT get faster with s4; only decode does.

### Chunk groups were never broken
- refcheck: chunk K_cache_out rows are **BYTE-EXACT** vs per-token decode
  group K for m=8/16/32/64 (llm_build2) and m=128 (vendor) — input, indices,
  rope all flow correctly through axclrt.
- CAVEAT: the `output` (y) tensor is only written for **m ≥ 64** (m=8/16/32:
  K/V_out correct, y untouched). Batched prefill + speculative verify must
  use 64-token chunks until Axera fixes small-m y-write (bug-report fodder).
- Verify-pass economics at m=64: 28 × 2.15 ms = 60 ms covers up to 64
  candidates (one weight pass + one fixed cost). Multi-row logits need a
  custom lm_head engine — vendor AND llm_build2 post are m=1-only (7.9 ms
  per call at 152k vocab; ~19.6 GB/s, i.e. pure streaming).

### Context-length tax (llama-simple, vendor w8a16, governor performance)
- decode **19.64 t/s @ ctx≈60** vs **18.59 t/s @ ctx≈1580** → −5.6% only.
  KV reads are not a dense full-cache sweep; long-context decode is fine.
  (Vocab trim ≫ KV tricks at these lengths.)

### Builds this session (x86, /tmp/int4lb2, marker ckpt unless noted)
- mk_m8 (prefill 8), mk_p16, mk_p32, mk_base — group-probe engines.
- out_s4_2048: FULL 28-layer s4 set, kv 2047 (layer files verified on card;
  post was still building at session close) — deployment candidate for the
  s4 path.
- `--ld_param_opt` CRASHES llm_build2 7.0-patch1 (KeyError 'xxh128:..._ddr')
  on the marker ckpt — retry on the real ckpt / newer toolchain before
  writing it off.

### Backend work order to bank the wins

## Roadmap execution (2026-08-27 mid-day session) — items 1a + 2 BANKED





### SESSION CLOSE (2026-08-28 ~03:00) — EVERYTHING WORKING, ROOT CAUSES CLOSED
1. **Batch-prefill "refusal" was OUR bug, not the driver's**: the chunk
   ladder's depth is build-dependent (vendor 10 groups = 1152-token
   coverage; llm_build2 s4 sets = 3 groups = 256-token coverage). The
   dispatch assumed a bottomless ladder -> prompts >256 tokens aborted
   ("chunk ladder failed at layer 0 chunk 2"). Fix (fork 9db8049):
   capacity from n_groups + per-token fallback beyond it + graceful
   mid-ladder degradation. VERIFIED: 314-token prompt, 0 errors,
   **1276 t/s prefill** + 22.6 t/s decode on the M5Stack stack.
2. Final working stack: axclhost 3.6.5-m5stack1 (driver-good/ deb) +
   M5 3.6.6-deb card pac. Fan quiet. Decode ~23 t/s (s4), all modes OK.
3. The image ships NO driver (apt-installed post-boot) — the "P1" build
   IS the repo's 3.6.5-m5stack1. Card-firmware archaeology: three pac
   variants (3.6.5-deb 9f21f721, 3.6.6-deb 75712ac6, generic d636314).

### BACKEND-INIT DEBUG COMPLETE (2026-08-28 early)
1. VENDOR BUG (V3.10.2, clean repro): loading a `pulsar2 build` engine
   AFTER any llm_build engine drops the PCIe device ("recv dma size 0").
   Repro: multi_load <layer.axmodel> <post_trim.axmodel> -> instant
   zero-byte DMA. Reverse order fine; standalone fine. WORKAROUND in
   backend: heads load before the layer set. FILE WITH AXERA.
2. The "corrupt" trim engine: first rebuild was written during the quota
   crunch (rc=0 but garbage); ALSO built with the EndToEnd cfg = 372MB
   f32 weights. Correct build: cfg_simple (MinMax) = 99MB int8.
3. MY BUGS fixed (fork babca45): post IO names (input/output vs X/Y),
   f32 vs bf16 logits (trim map count = ground truth), per-token retry
   of failed post loads (4s/token), vocab64 stride guards.
4. **Trimmed post (90.9k vocab): 26.77 t/s decode, zero errors** —
   37.4ms = 32.7 layers + 4.7 post + host. Budget closes exactly.
5. OPEN (next session): llama-lookup SIGSEGV in graph_compute on
   all-logits batches (llama-simple immune; survives w/o vocab64 =>
   NOT the head branch). gdb frame: graph_compute <- sched <-
   process_ubatch <- llama_decode. Needs a -g build for the line.
   Spec e2e blocked on this alone — every primitive now works.

### DRIVER UPGRADE + FINAL NUMBERS (2026-08-27 evening)
- axcl V3.10.2 installed on the Pi (deb at ~/axcl_host_aarch64_V3.10.2*.deb;
  NOTE: new package ships UNVERSIONED libs — needed `for f in libaxcl_*.so;
  ln -s $f ${f%.so}.so.1` in /usr/lib/axcl + ldconfig for every built binary).
- Card firmware now V3.10.2. MEASURED ON IT:
  * s4-GPTQ @ kv2047: **24.55 t/s** (stable vs 24.32 on old driver)
  * **s4-GPTQ @ kv1023 (/tmp/kv1024): 29.90 t/s, coherent** — new record
    (+52% over shipped w8a16). 1k-ctx cap; redeploy after every reboot.
- The in-backend pulsar2-build load bug PARTIALLY persists: vocab64 loads
  standalone AND after other engines in multi_load on V3.10.2, but the
  trimmed post still fails inside llama-simple init (0.21 t/s cascade,
  spec run aborted). => the residual bug is in the BACKEND's init context
  (pinned-host allocations? second context?), not the driver. Next moves:
  axclrtEngineLoadFromMem, or trim the vocab64 head to ~55k (3x smaller
  transfer) per PERF-PLAN Phase 5.
- axcl-smi on V3.10.2 gains EP-panic card reset (untested — try it before
  driver reloads next wedge).

### GEMM TOPS LADDER (measured 2026-08-27, K=1024 N=3072 static int8, 30 iters)
m=128: 635.5us = 1.27 TOPS | m=256: 1019.5us = 1.58 | m=512: 1689us = 1.91
m=1024: 2488.8us = 2.59 | m=2048: 5342.3us = 2.41
=> transformer-shape ceiling ~2.6 TOPS (~11% of rated 24) on the w8a16
   dataflow; per-row marginal ~2.5us/row + ~310us fixed. The 24 TOPS rating
   is only reachable on pure-int8 conv paths (CV), not this toolchain's
   LLM engines.

### LOAD-BUG PATTERN CONFIRMED (2 engines, identical signature)
`pulsar2 build` (ONNX-path) engines — vocab_m64 (165MB) AND post_trim
(91MB) — fail to load INSIDE the backend process ("device 0 is not
connected" -> zero-byte DMAs), while loading fine standalone and while
same-size llm_build engines load in the same slot. After any such failure
the card needs a wall power cycle. => Axera bug report #3. Workaround
idea for next session: emit the verify head as an llm_build-style
container, or load via axclrtEngineLoadFromMem.

### kv1024 TG: UNMEASURED — engines redeployed to Pi /tmp/kv1024 (NOTE:
/tmp does NOT survive Pi reboots; use ~ or /usr/local/share next time).

### FINAL STATE 2026-08-27 (session close)
**MEASURED ON CARD (post power-cycle, healthy):**
- **s4-GPTQ (JunHowie/Qwen3-0.6B-GPTQ-Int4, g128) = 24.32 t/s TG, COHERENT**
  ("Paris... Versailles..." — the raw-fp repetition loop is GONE).
  Deployed: Pi ~/s4-gptq (29 files). Budget: 28×1.1665ms + 7.92 post + host.
- Chunked prefill with binding fix: **716.5 t/s** (530-token prompt, was
  18.4), greedy output byte-identical. GGML_AXCL_BATCH=1 safe.
- vocab64 verify head (static int8 [64,1024]@[1024,151936]): loads solo,
  **11.97ms/64 rows** (42× vs sequential post).

**CARD STABILITY RULES (learned the hard way, 4 wedges today):**
1. NEVER kill a process during engine loads (timeout wrappers around
   llama-* = wedge roulette). Run detached (setsid) + poll.
2. Engine load mid-graph corrupts the PCIe channel (vocab64 lazy-load).
   All loads at init — fixed in backend.
3. Recovery ladder: driver reload (modprobe -r/-v ax_pcie_host_dev stack)
   works for EARLY wedges; deep wedges need a full POWER CYCLE (wall
  unplug — Pi reboot does NOT reset the card). Symptom: "recv dma size 0
   is not equal to N" + zero-byte channel sends.
4. OPEN BUG: vocab64 (pulsar2-build product, 165MB) fails to LOAD inside
   the backend process (fails first, solo, after reloads — while the
   169MB llm_build post engine loads fine in the same slot). Root cause
   unresolved; needs a healthy card to debug. Spec e2e blocked on it.

**BUILT + DEPLOYED, UNTESTED (card wedged before bench):**
- kv1024 s4 set: Pi /tmp/kv1024 (target ~28 t/s, KV read is dense-full-len).
- GEMM TOPS ladder m=128..2048 (K1024×N3072 static int8): /tmp/int4lb2/
  gemmlab — the chip's transformer-shape ceiling experiment.
- Trimmed post (90.9k vocab) build: requeued (quota casualty) — check
  posttrim/status2.txt; backend remap support ALREADY shipped+rebuilt
  (GGML_AXCL_POST_TRIM env, auto-detects trimmed output size).

**BACKEND CHANGES SHIPPED (llama.cpp fork, rebuilt on Pi):**
- axcl_layer_run_chunk: per-group io_chunk[12], d_chunk_ko/vo staging +
  post-exec D2D scatter (THE chunk fix).
- vocab64 verify-head path (multi-row vocab matmul -> 64-row head call).
- trim-remap post support.
- recovery_bench.sh on Pi (~/phasec) runs the full queue.
1. **s4 raw-fp deployed end-to-end: 23.76 t/s decode** (vendor-engine mode,
   out_s4_2048 renamed p64→p128, GGML_AXCL_LAYER_DIR switch — zero backend
   changes; ctx_len auto-derives from K_cache dims). Budget closes exactly:
   28×1.1665ms + 7.92 post + ~1.5 host = 42.1ms. NOTE: the engine's KV read
   is DENSE over the full cache length regardless of position (context-
   independent 1166 µs) — a kv1024 build would run ~0.93 ms/layer → ~28 t/s
   if long context isn't needed.
   QUALITY: raw-fp s4 at g1024 GARBLES output (repetition loop) as the
   community warned — must feed GPTQ-g128. JunHowie/Qwen3-0.6B-GPTQ-Int4
   (gptqmodel 4.0.0, g128 sym) downloaded; s4 rebuild from it in flight.
2. **Chunk-binding fix SHIPPED + verified** (ggml-axcl.cpp: per-group
   io_chunk[12] handles, d_chunk_ko/vo staging + post-exec D2D scatter —
   no offset output binds):
   prefill 530 tokens: 18.43 t/s per-token → **716.5 t/s chunked = 38.9×**,
   and the greedy continuation is BYTE-IDENTICAL to per-token prefill.
   GGML_AXCL_BATCH=1 now safe to default-on (m=128 vendor ladder; the
   llm_build2 64-ladder needs the chunk-size parameterized first).
3. Vocab m=64 lm_head static-quant engine (X[64,1024]@W[1024,151936],
   final norm stays host-side) building — the verify-logits primitive.
1. s4: point GGML_AXCL_LAYER_DIR at out_s4_2048 (llm_build2 container, IO
   names identical) → expect ~28-30 t/s; then the s4 claims-decode GGUF
   patcher (R3 continuation).
2. Re-enable GGML_AXCL_BATCH chunked prefill with the binding fix (dedicated
   staging for the 64-row K/V_out + D2D scatter into cache rows; all outputs
   bound; own IO handle per group) → prefill ~4×.
3. Speculative decoding: n-gram draft on host (fork has speculative-simple/
   lookup ngram-*), verification via the m=64 chunk group, custom m=64
   lm_head (hostops vocab GEMM machinery) for all-position logits.
4. Vocab-trimmed post engine (152k → 50-60k zh/en+specials, host-side id
   remap): post 7.9 → ~2.7 ms → +11-15% decode, stacks with s4.

## Claims-decode series BUILT (2026-08-27 late) — state for next session
- 9 marker builds through llm_build2 s4, ALL OK (2..7, d2, mc, mixamp).
  npu_params extracted to int4lab/scratch/claims/*_l0.npy.
- KEY FINDING: npu_params size is CONTENT-DEPENDENT (5.3-8.7MB) =>
  the s4 weight mass is COMPRESSED in-engine (build_context compress_mcode).
  Claims decode must handle compression (or target the load-time
  decompressed image — check what axclrt does at load).
- Confirmations: mk_d2 (dither variant) same size as mk0; mk_mixamp
  (amp 0.5) same size => codes dither-invariant + amplitude-invariant
  (matches old s8 behavior).
- Tooling fixed: int4lab/build_claims_markers.sh (NEVER export pkg
  LD_LIBRARY_PATH globally - breaks nested bash), extract_claims.py
  (needs LD_LIBRARY_PATH/PYTHONHOME for bundled python).
- Card window still pending (busy with e2e): load out_s4/out_s8 layer
  engines + per-layer timing = go/no-go for the w4 speed thesis.

## Claims decode — session close (2026-08-27 night)
- ALL marker npu_params extracted for l0 AND l1 (scratch/claims/*_l{n}.npy).
- MIXAMP L1 RESULT (the key decode anchor): amp-0.5 changes EXACTLY
  61,440 B = 15,360 x 4B = full scale table @ groupN=1024 (15.73M/1024
  ~= 15,364). => nibble codes amplitude-INVARIANT; scales = 4B/group,
  ~15.4K entries, spread [8582..end]; markers built at groupN=1024
  (raw-fp fallback). For real models: feed GPTQ g128 -> different
  table size, decode must parametrize groupN.
- d2 vs mk0: 30.9% differ -> at g1024, dither DOES shift q codes
  (expected, V=16*code+dither); NOT evidence of compression.
- npu_params NOT zlib; starts/ends with repeated f32 patterns (raw
  tables); mk0-vs-mk_2 identical only to byte 278; SIZE VARIANCE
  (5.3-8.7MB) across same-shape builds STILL UNEXPLAINED — top open
  question next session (adaptive per-matrix structure? count the
  per-section sizes across all 11 blobs; no compress_mcode knob found
  in llm_build2 CLI).
- NEXT: (1) explain size variance via cross-build region fingerprint;
  (2) claims decode with dither-free codes (use mk builds, d2 only as
  xcheck); (3) card window (load out_s4/out_s8 + timing) when e2e
  finishes; (4) qwen3_5 MoE fabrication; (5) only then backend code.

## SIZE VARIANCE SOLVED + FORMAT ~CRACKED (2026-08-27 session 2, int4lab)

### Extraction bug fixed
- extract_claims_l1.py wrote {d}_l0.npy (both runs wrote _l0!) — claims/
  files were ALL l1 data. RE-EXTRACTED to scratch/claims2/ (correct l0/l1).
- mk0 l0==l1 byte-identical (marker weights same in all layers) =>
  LAYOUT IS LAYER-INDEPENDENT + builds are DETERMINISTIC (mixamp l0 ==
  mk0 l0 bit-exact from separate builds).

### npu_params container structure (llm_build2 s4, groupN=1024 raw-fp path)
- [0..256) 0.0884-f32 table; tail = 0.1426(=1/7)-f32 run; ~8.3KB fixed
  region; ONE 72,960B region (64,640B fixed rope-ish table + 8,320B
  varying) whose position is content-dependent.
- 960 sections (for full-density builds), each = [128B header: 16 groups
  x 8B = TWO live f32 scale copies per group] + data from a DISCRETE MENU
  {9600, 8448, 7296, 6144} (+the 72960). mk0: 9600x576, 8448x318,
  7296x32(o range), 6144x32(down range).
- Mixamp/amp-twin diff = THE entry map: 61,440B = 15,360 entries x 4B
  (2 changed bytes per f32; both scale copies live). All spacings match
  section sizes exactly (size-128+8). 960x16 = 15,360 = ALL groups.
- SCALE VALUE = 1/7 f32 (0x3E124925) for every anchored group
  (V[:,0]=127 anchors EVERY row's first group => all groups anchored at
  groupN=1024 k-major).

### Quantization mapping (VERIFIED)
- v_stored = 8 + round(7*V/127), V = 16*code + dither, scale=amax/7,
  amax=127 via col-0 anchor. RNE splits verified against mk_2 (77/23)
  and mc (86/7) section histograms. lo-nibble = EVEN k (same as 5.2).

### THE SIZE-VARIANCE MECHANISM: content-dependent compaction
- Uniform weights => npu_params COLLAPSES: probe (all V=8 sea + ~16
  dark V=1 elements) => 136KB total (64x smaller than 8.7MB!).
- Constant-code matrices in mc collapsed: v_proj 64->1 section,
  o 128->2, gate 192->2 (their 16-group scale headers OMITTED: mc has
  only 9,792 entries vs 15,360; missing 6,064 = 63x16+126x16+190x16
  EXACTLY). Two-valued matrices (q,k,up,down) keep all sections.
- mk_2 (codes constant per 64-row blocks): 612 sections vs mk0 960 —
  verified with a dedicated amp-0.5 twin build (mk2a; gen via
  mk_code_marker.py <dir> 2 0.5): 9,792 entries, sizes
  {9600x363, 8448x201, 7296x23, 6144x23, 72960x1}.
- mk0 vs mk_d2: size sequences IDENTICAL byte-for-byte => sizes depend
  on code/coarse structure, NOT dither.
- Section data modes: LITERAL v-nibbles (row-major, lo=even-k; verified
  by dark-probe: consecutive darks appear as 9f/99..99/f9 runs) + ELISION
  of uniform spans (12B 0x88-fill markers, 4B exception units
  00 00 7X f9, X=0/3 flags) + rows/sections omitted when fully uniform.
- 9600/8448 ALTERNATE per section from section 0 = two copies (shape
  groups prefill/decode); extra 1152B on 9600-class = copy-specific meta.
- NOT LZ4/zlib (ax650npu_cmodel.so exports ax_lz4_find_matches but
  probes parse invalid as LZ4; that's probably for microcode segments).

### Element layout status (the last 15%)
- 12-byte expected-row probes DO match literally (q/k_proj code-0 rows,
  packing lo=even) but at MULTIPLE positions: row 8 = 14 matches, row
  64/100 = 20+ (cap), rows 0/32/47 = 1 match => variable copy
  multiplicity per row + elision shifts. Rows 0/1 probes 18B apart at
  ~840,838 — sub-row chunk interleave (18B/2016B structure) unresolved.
- Uniform-sea probe methodology (gen_probe.py) WORKS and is FAST
  (~40s/build): darks readable by eye. Use consecutive-dark + fan probes
  to finish the chunk map next session.

### Next session plan
1. Finish chunk-interleave order via gen_probe.py fans (rows x k-grids).
2. Claims decode mk0..7 + mc against claims2 blobs (row-pattern voting
   across copies; elision-aware). d2 as xcheck.
3. Scale-table decode: amp-twin (0.5) of EACH marker build = exact scale
   entry map (proven with mixamp + mk2a).
4. Card window (Pi busy with e2e llama-simple at 08:27; recheck).
5. qwen3_5 MoE fabrication (qwen35_index.json already in scratch/).

### New files this session
- int4lab/fingerprint.py (blob structure scan)
- int4lab/reextract_claims.py -> scratch/claims2/ (CORRECT l0/l1 set)
- int4lab/gen_probe.py (uniform-sea dark probes kf/rf; cd variant inline)
- scratch/mk2a + mk2a_s4 (amp-0.5 twin of mk_2), probe_{kf,rf,cd}_s4

## W4 SPEED THESIS: CONFIRMED ON CARD (2026-08-27 session 2)

Pi card freed (~08:40). bench_layer.c (int4lab/, deployed /tmp on Pi,
gcc -I/usr/include/axcl -L/usr/lib/axcl -laxcl_rt; axclrt shutdown
aborts with 'terminate called without an active exception' AFTER
results — harmless, use _exit). M=1 decode, shape group 0, pos 0,
zero inputs, 50 iters, gov=performance:

| engine                                   | ms/layer | layers/s |
|------------------------------------------|----------|----------|
| vendor 5.2 w8a16 (kv2048, p128 l0)       | 1.488    | 672      |
| ours 7.0 s8 (marker shape, kv255, g1024) | 1.067    | 937      |
| ours 7.0 s4 (same shape/kv)              | 0.761-0.779 | 1283-1313 |

- s4 vs s8 (apples-to-apples): 1.39x faster per layer.
- s4 vs vendor w8a16: 1.93x faster.
- Implication for 19.6 t/s w8a16 decode: s4 engines could lift to
  ~25-27 t/s layer-bound (28 layers x 0.77ms = 21.6ms vs 30ms), before
  host-glue overheads. GO for the w4 path.
- NOTE: marker engines at groupN=1024; real GPTQ g128 builds may differ.

