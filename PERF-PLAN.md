# Efficiency plan — ggml-axcl decode/prefill (2026-08-27)

Built on PERF-REVIEW.md, cross-checked line-by-line against
`llama.cpp/ggml/src/ggml-axcl/ggml-axcl.cpp` (incl. the uncommitted chunk-ladder
diff, +168/−21) and the measured state in NOTES-DYNAMIC-WEIGHTS.md /
int4lab/FINDINGS.md.

## Baseline (measured, current)

s4-GPTQ mode, 24.32 t/s TG = 41.1 ms/token:

| component | cost | share |
|---|---|---|
| 28 layer engines (s4, 1.1665 ms ea) | 32.7 ms | 79.5% |
| post engine (m=1, 151936 vocab) | 7.92 ms | 19.3% |
| host (orchestration + getenv storm + flush spikes + I/O) | ~1.5 ms | ~3.6% |

Chunked prefill already banked at 716.5 t/s (byte-identical greedy).

## Review verdict

All 13 findings verified in code — none are wrong. Two corrections:

1. **vocab64 lazy-load (review #4) is already fixed** in the working tree:
   `axcl_vocab64_load()` runs at backend init BEFORE the engine set
   (ggml-axcl.cpp:4111), per the card-stability rule. The lazy call at 3712 is
   now a guarded fallback only. What remains open is FINDINGS' load-failure bug
   (165 MB engine dies inside the backend process even solo-after-set) —
   untested since the init-order fix on a healthy card.
2. **Several getenv sites are already static** (kv_inplace, wb_env, pad_tail,
   layer_env, skip/mm/uni/claim/chain_env, meta_env, ovr_k, tracemm, selfchk…).
   The remaining dynamic ones are enumerated below; the worst is real:
   supports_op:4317 re-reads `GGML_AXCL_LAYER` for every claimed node, and
   `offload_op` (4372) calls supports_op again, doubling it.

# The plan

Phase order is by (gain × certainty) ÷ effort, with measurement hygiene first
so every later A/B is trustworthy. Card-window items are parked in their own
phase since they need a healthy card + e2e queue.

## Phase 0 — bench hygiene (trivial, do first)

- `simple.cpp:203` — `fflush(stdout)` per token. Redirect to a file / `cat > /dev/null`
  for bench runs, or flush only at EOG. On an SSH console this can cost
  0.1–2 ms/token and may be inflating every number we've recorded. A/B once,
  keep whatever the runner change needs to be.

## Phase 1 — host-path diet (~0.5–1 ms/token, all trivial-to-small diffs)

Land as 1–2 commits; verify each session with `[layer] wall= vs eng=` deltas
plus `perf record` (getenv doesn't show in strace -c; the wall/eng gap is the
observable) and the E2E 12-suite.

1. **One-shot config struct for env flags.** All flags are process-start
   constants. Replace the remaining dynamic getenv sites (~0.3–2 ms/token,
   ~17k calls/s):
   - `supports_op` 4317 (`GGML_AXCL_LAYER` — worst offender, ~1k/token ×2 via
     offload_op), 4352 (FA), 4278 (vocab branch)
   - `axcl_layer_run` 2677–2678 (CHECKSUM/DUMPSTATE ×28)
   - dispatch 3216, 3220, 3262, 3305 (BATCH ×28), 3387, 3408
   - prescan 3134 (LAYER_DEBUG2 in the SET_ROWS scan), 3454, 3465, 3820
   Shape: `static axcl_cfg_t g_cfg = []{...}();` with one bool per flag.
2. **Precomputed claim table in supports_op** (4239–4363). The policy ladder is
   static after start; build `bool claim[GGML_OP_COUNT]` once, keep only the
   two shape-gated specials (vocab MUL_MAT, FA) as live branches. Hot path
   becomes one array lookup. (~15–20k calls/s through supports_op+offload_op.)
3. **Cache the device slot index.** graph_compute:2856 does
   `axcl_get_device_index()` → `axclrtGetDeviceList()` driver round trip every
   graph. Store the slot in `ggml_backend_axcl_context` at init (already
   computed at 4138). Keep `SetCurrentContext` (thread-local, needed).
4. **Spread the KV flush** (2646). `pos & 31 == 31` bills 28 layers × 2 sides
   to one token (~10 ms spike). Change to flush one layer per token
   (`axcl_layer_flush_kv(pos % n_layer)`), same average work, no spike, better
   overlap. Add `GGML_AXCL_KVFLUSH_EVERY=<n>` (0 = lazy-only) so the tradeoff
   is measurable. Flush-all already exists at resync/bail/save paths.
5. **NEON f16 in the KV write-back** (2554–2558). bf16→f32 is vectorized; the
   f32→f16 tail is scalar (57k convs/token spread across flushes; most of the
   every-32nd spike is this). `vcvt_f16_f32` matches ggml's native `__fp16`
   cast semantics (RNE) on AArch64. Composes with #4.
6. **Micro:** `std::unordered_set done` (2865/3201) → `std::vector<uint8_t>`;
   GET_ROWS `rowbuf` malloc per call (1460) → static scratch. Free µs.

Projected: −0.5…1 ms/token → ~24.8–25 t/s now; proportionally more valuable
after Phase 2/3 shrink the engine side.

## Phase 2 — post engine trim (biggest single decode lever)

1. **Land the 90.9k trimmed post** (build was requeued — check
   `gemm/posttrim/status2.txt`). Backend remap support is already shipped
   (n_out auto-detect + `GGML_AXCL_POST_TRIM` JSON map, 2055–2086, 3762–3770).
   7.92 → ~4.7 ms ⇒ ~26.5 t/s.
2. **Try a 50–60k kept set** (zh/en + specials + EOG) ⇒ ~2.7–3 ms ⇒ ~27.5 t/s.
   Kept-set rule: everything plausibly argmax-able — validate greedy agreement
   with the eval suite (divergence the moment the true winner is masked is the
   failure mode). Build the set as union(top-frequency over corpus, all
   specials/bytes); do not tune below 50k without eval evidence.
3. **Patch the kept lm_head rows from GGUF** while in there — the post still
   runs the template's lm_head (part of the 94% vs 96% agreement gap); trimming
   shrinks the patch surface ~10×. The w8 patcher (gemm/gguf_patch_w8.py)
   already proves row-patching on layer engines.
4. Accept the O(151936) −inf fill + argmax tail (~0.1–0.2 ms) — interface-bound,
   not worth surgery.

## Phase 3 — engine-side knobs (measured, near-zero code)

1. **Test the kv1024 s4 set** (already on the Pi, `/tmp/kv1024`): point
   `GGML_AXCL_LAYER_DIR` at it; ctx auto-derives. KV read is dense over the
   full cache length ⇒ ~0.93 ms/layer ⇒ ~28 t/s. Trade-off: 1024-token ctx cap
   — fine for chat-sized contexts, keep the 2048 set for long prompts.
2. s4 claims-decode GGUF patcher (int4lab roadmap: scale/norm entries for
   arbitrary GGUFs at s4 speed) — the "any GGUF, no prequant needed" endgame;
   follow int4lab FINDINGS' next-session plan, not this doc.
3. Retry `--ld_param_opt` on the real ckpt (crashed on markers) — only worth a
   slot in an existing build session.

## Phase 4 — prefill polish (before committing the chunk diff)

1. **Hoist the chunk mask/idx uploads** (2760–2778). Verified: idx + causal
   mask are rebuilt and H2D'd per LAYER (identical for all 28 within a chunk).
   Two-step fix:
   a. (tiny) precompute the two bf16 constants (0x0000, bf16(-1e9)) and write
      u16s directly — kills the float→memcpy→shift per element (~10× on the
      remaining builds).
   b. (small) per-chunk dedicated mask buffers filled once per graph at the
      h_all staging point + per-chunk idx buffers; run_chunk then only binds.
      Binding rule stays honored (base-pointer binds with exact sizes, no
      offset binds). Saves ~8.4 MB H2D + ~28 × mask-loop per chunk.
2. **m<64 y-guard**: run_chunk is ntok==128-only today (2729) so the
   "y written only for m≥64" caveat can't bite; add the assert now so the
   future 64-ladder parameterization inherits it.
3. Then **commit the uncommitted diff** (+168/−21) — it's the state already
   shipped on the Pi; the repo shouldn't drift further. Consider defaulting
   `GGML_AXCL_BATCH=1` on after 4a/4b.

## Phase 5 — speculative verification (the 2–3× endgame)

1. **Resolve the vocab64 load bug.** Init-order fix is in (4111) but unverified
   on a healthy card. Test at the next card window; if it still dies at the
   258048 B DMA, try (a) a trimmed-vocab64 build (64 × kept-set, also cuts the
   12 ms verify call to ~4 ms) or (b) loading it after engine unload/reload
   churn settles. Card rules apply: run detached, never kill mid-load.
2. **Wire speculative-simple** (fork already has it + ngram lookup): host n-gram
   draft, verify via one m=64 chunk pass (28 × 2.15 ms = 60 ms covers up to 64
   candidates) + vocab64 head (11.97 ms/64 rows measured). Step cost ~72 ms;
   at 1.5–2.5 accepted tokens/step ⇒ ~30–45 t/s effective at full ctx. The
   m>1 hidden staging (h_all) and chunk binding machinery all exist.
3. Quality gate: speculative output must be token-identical to greedy
   (verification is exact if the head is exact — validate against CPU logits).

## Projected end state (greedy decode, bench-scale ctx)

| step | ms/token | t/s |
|---|---|---|
| today (s4-GPTQ) | 41.1 | 24.3 |
| + Phase 1 host diet | ~40.2 | 24.9 |
| + post trim 90.9k | ~37.0 | 27.0 |
| + tighter trim (~55k) | ~35.3 | 28.3 |
| + kv1024 (ctx ≤1024 only) | ~28.7 | 34.8 |
| + spec verify (any ctx, acceptance-dependent) | — | ~30–45+ |

Post-trim and kv1024 stack multiplicatively with everything else; speculative
is the only path beyond ~35 t/s at 2048 ctx.

## Guardrails (every phase)

- Eval before/after: `eval_agreement.sh` + E2E 12/12 (stale-process retry rule
  from NOTES applies).
- Card etiquette: governor=performance, no kills during engine loads, recovery
  ladder (driver reload → power cycle), axcl-smi CMM baseline 18 MiB.
- Keep new env knobs A/B-able in the house style (default = old behavior until
  measured).
