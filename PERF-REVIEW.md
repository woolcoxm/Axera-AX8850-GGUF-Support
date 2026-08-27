# Performance review — ggml-axcl decode path (2026-08-27)

Scope: `llama.cpp/ggml/src/ggml-axcl/ggml-axcl.cpp` (4,480 lines, incl. the
uncommitted chunk-ladder/vocab64 diff), `llama.cpp/examples/simple/simple.cpp`
(the runner), cross-checked against the measured budgets in
`NOTES-DYNAMIC-WEIGHTS.md`. Everything below is framed by **how often it
fires**, because that's where the 0.1–0.5 ms savings live.

## Where the token time actually goes (your own measurements)

At **19.5 t/s = 51.3 ms/token** (GGUF-int8 mode):

| Component | Cost | Share | Frequency |
|---|---|---|---|
| 28 whole-layer engines (w8a16, 1.51 ms ea) | ~42.3 ms | 82% | 546 calls/s |
| …of which *fixed* per-call cost (two-point fit) | ~12.8 ms | 25% | 546 calls/s |
| Post engine (lm_head, s8, m=1 group) | **7.9 ms** | **15%** | 19.5 calls/s |
| Host orchestration + sampling + I/O | ~1–2 ms | ~3% | — |

Implication: a 0.5 ms saving anywhere in the per-token path ≈ **+0.2 t/s**;
the same 0.5 ms saved *per layer call* would be worth **+6 t/s**. The three
engines-side levers you already identified (s4 engines → 28–30 t/s, post
vocab trim, vocab64 speculative verify) dwarf anything I can find in code —
so this report focuses on the host side, which is where the remaining
addressable ~1–2 ms/token lives.

## Summary of findings

| # | Finding | Fires | Est. cost | Fix effort |
|---|---|---|---|---|
| 1 | `getenv()` storm on hot paths (~900–1,000 calls/token) | ~17k/s | 0.3–2 ms/token | trivial |
| 2 | Driver calls at every `graph_compute` entry (`axclrtGetDeviceList` etc.) | 19.5/s | 0.02–0.15 ms | trivial |
| 3 | KV flush spike: every 32nd token pays ~10 ms | every 32nd token | ~0.33 ms avg | small |
| 4 | Scalar f32→f16 in KV write-back | 57k conv/s | ~0.2 ms/token | small |
| 5 | Post-engine tail (7.9 ms engine + 0.3–0.5 ms D2H/convert) | 19.5/s | biggest single | structural (you have the infra) |
| 6 | Chunk path re-uploads idx+mask **per layer** (new uncommitted code) | prefill | ~0.1 ms/token + 27×288 KB H2D/chunk | small |
| 7 | `supports_op` re-branches the whole claim policy per node | ~15–20k/s | 0.1–0.4 ms/token | trivial (subsumes #1's worst offender) |
| 8 | Prescan does ~6 full node passes per token | 19.5/s × 6 | 0.05–0.15 ms | small |
| 9 | 4 IO binds per layer call; 2 of 4 are avoidable via ping-pong pre-bind | 2,184/s | 0.1–0.8 ms (A/B) | medium |
| 10 | `std::unordered_set<int> done` + `count()` per node | ~1k/token | ~10–20 µs | trivial |
| 11 | `fflush(stdout)` per token in `simple.cpp` | 19.5/s | 0–2 ms (terminal-dependent) | trivial |

---

## Tier 1 — structural (biggest wins, mostly already on your radar)

### 1. Post-engine vocab trim — 7.9 ms → ~1 ms projected (+3 t/s)
`ggml-axcl.cpp:3738-3777`, infra at `2040-2090`.

Your own note: post engine runs at ~19.6 GB/s streaming 170 MB of s8
lm_head weights — it is pure bandwidth, 15% of every token, and the
`n_out`/`trim_ids`/`GGML_AXCL_POST_TRIM` machinery is already written and
shipped in this diff. A trimmed post with ~16k kept ids would cut engine
time ~10×, plus the 304 KB D2H and the 151,936-wide bf16→f32 convert shrink
proportionally. What *doesn't* shrink: the `-INFINITY` scatter fill (608 KB
of f32 writes, ~60–100 µs) and llama.cpp's greedy argmax scan (another 608 KB
read, ~60–100 µs) — both stay O(151936) by interface.

Caveats to design for: the kept-id set must cover everything plausibly
argmax-able (greedy diverges the moment the true winner is masked — validate
with `eval_agreement.sh`), and remember the post engine still carries the
*template's* lm_head weights (your notes flag it as part of the 94% vs 96%
agreement gap; trimming is a natural moment to also GGUF-patch the kept rows,
which shrinks the patch surface 10×).

### 2. s4 engine set — your measurement, not a code finding
1,166 µs/layer vs 1,500 → ~28–30 t/s projected. Nothing to add except: the
per-call *fixed* 457 µs is ~39% of the s4 per-layer time, so once s4 lands,
everything in Tier 2 that touches per-call host work (findings 7, 9, 10)
gets proportionally *more* valuable, not less.

### 3. Finish the chunk-ladder prefill wiring (the uncommitted diff)
The phase-C result (dedicated output buffers, one IO handle per group, no
offset binds → byte-exact K for m=8..128) is the right fix, and the diff
implements it. Two things to carry across the finish line:
- **The y-output caveat**: y is only written for m ≥ 64 (your FINDINGS note).
  The current code binds `d_chunk_out` unconditionally — for a final chunk
  smaller than 64 the hidden rows would be garbage. Guard the tail chunk
  (zero-pad is fine since you already pad to 128 by default — with
  `pad_tail` on, m<64 can't happen mid-chunk, but keep the assert).
- See finding 6 below — the chunk path has a per-layer upload redundancy
  that's cheap to fix now, before it's baked in.

### 4. Speculative verification (vocab64)
The `axcl_vocab64_engine` in this diff is the seed of the only path to
2–3× beyond s4: one 64-row head call amortizes the 7.9 ms post problem
across multiple verified tokens. Flagging only that `axcl_vocab64_load()`
is lazily invoked from inside the node loop (`3712`) — your own comment at
`4113` says engine loads must never happen mid-graph (a load corrupted the
PCIe channel once). Move the load to backend init alongside
`axcl_post_load()`; the lazy call is an accident waiting for the first user
who sets `GGML_AXCL_VOCAB64` without the file present.

---

## Tier 2 — host-path code findings (the "0.5 ms × 500 Hz" class)

### 5. `getenv()` storm — ~900–1,000 calls per token (~17k/s)
glibc `getenv` is a linear scan of `environ` with `strcmp` per entry; a
*miss* (which all of these are in production) walks every entry — ~0.5–2 µs
each on a Pi 5 with a typical SSH environment. You already cache some in
statics (`kv_inplace`, `use_chain_stream`, `layer_env` — good), but these
are uncached on the hottest paths:

| Location | Call | Frequency |
|---|---|---|
| `ggml-axcl.cpp:4317` (`supports_op`) | `getenv("GGML_AXCL_LAYER")` | **once per claimed node ≈ 800–1,000/token** |
| `ggml-axcl.cpp:2677-2678` (`axcl_layer_run`) | `CHECKSUM` + `DUMPSTATE` | 2 × 28 = 56/token |
| `ggml-axcl.cpp:3287,3387,3408` (anchor dispatch) | `DUMPSTATE`/`CHECKSUM`/`LAYER_DEBUG` | 3 × 28 = 84/token |
| `ggml-axcl.cpp:3134` (SET_ROWS scan) | `LAYER_DEBUG2` | 56/token |
| `ggml-axcl.cpp:3216,3220,3262,3305,3465,3709,3820` | various | ~10/token |

Total ≈ 0.3–2 ms/token depending on environment size — i.e. up to ~4% of
the budget, spent asking the OS the same question 17,000 times a second.

**Fix (trivial):** one config struct resolved once, e.g.
```cpp
struct axcl_cfg_t {
    bool checksum, dumpstate, layer_debug, layer_debug2, gguf_debug;
    bool layer, gguf, batch, stream, vocab64, cnt, ...
};
static axcl_cfg_t g_cfg = []{ ...getenv each once... }();
```
`supports_op`'s line 4317 is the single worst offender and is a one-line
`static const bool` fix. Verify the win with `strace -c -p <pid>` (count of
`getenv`-adjacent page faults won't show; instead A/B the `[layer] wall=`
vs `eng=` debug numbers, or `perf record`).

### 6. New chunk path: idx + mask rebuilt and uploaded **per layer**
`ggml-axcl.cpp:2760-2778` (`axcl_layer_run_chunk`).

For a 128-token chunk at position p, the indices (512 B) and the causal
mask (up to 128×1152×2 = 288 KB, built with a scalar float→bf16 loop) are
**identical for all 28 layers**, but the function rebuilds both and does
two H2D transfers on *every* layer call. That's ~28 × 300 KB ≈ 8.4 MB of
redundant H2D per chunk plus ~28 × 0.5 ms of scalar mask-loop host time —
roughly 5–8% on top of the chunk engine time, and a visible prefill
regression vs what the ladder should deliver.

**Fix:** mirror decode's `synced_pos` guard — a `synced_chunk = (p<<16)|ntok`
check so idx/mask are staged once per chunk; while there, precompute the two
bf16 mask constants (`0.0f` → `0x0000`, `-1e9` → `0xF4D2`-ish, compute once)
and write `uint16_t`s directly instead of the float→memcpy→shift dance
(~10× faster on the remaining builds).

### 7. `supports_op` re-derives the claim policy per node
`ggml-axcl.cpp:4239-4363`.

Beyond the uncached `getenv` (finding 5), the whole function re-runs a
ladder of branches per node per token (the scheduler re-splits every
rebuild). The policy is 100% static after process start: precompute a
`bool claim[GGML_OP_COUNT]` table once at init (plus the two shape-gated
special cases: vocab matmul, FA) and make the hot path a single array
lookup. Expected 0.1–0.4 ms/token combined with the getenv fix — the
cleanest "fires 15,000×/s" win in the file.

### 8. Driver calls at every `graph_compute` entry
`ggml-axcl.cpp:2856-2859`.

Every token pays `axcl_get_device_index()` → `axclrtGetDeviceList()` (a real
driver round trip), then `axclrtSetDevice` + `axclrtSetCurrentContext`. The
slot index cannot change mid-run (your own comment: it's fixed by topology).
Cache it in `ggml_backend_axcl_context` at init and skip the list query
entirely; keep `SetCurrentContext` (thread-local, needed for worker
threads, and cheap). Est. 20–150 µs/token.

### 9. KV flush: a ~10 ms spike on every 32nd token
`ggml-axcl.cpp:2646` → `axcl_layer_flush_kv` at `2524-2567`.

`if (l == 0 && (pos & 31) == 31) axcl_layer_flush_kv_all();` bills
28 layers × 2 sides × (64 KB D2H + 32 rows of convert) to a *single token*
— a ~10 ms stall every 32nd token (≈ 0.33 ms/token average, and it lands
mid-chain where it also delays the async stream). Two independent fixes:
- **Spread it**: flush one or two layers per token (`l == pos % n_layer`)
  instead of all 28 at once — same average work, no spikes, better overlap
  with the async chain.
- **Make the interval/laziness tunable**: the periodic flush only bounds
  crash loss (resync / bail / save paths already flush). A
  `GGML_AXCL_KVFLUSH_EVERY=<n>` (0 = lazy-only) makes the tradeoff
  measurable instead of baked in.

### 10. Scalar f32→f16 conversion in the KV write-back
`ggml-axcl.cpp:2554-2558`.

The F16 branch (llama.cpp's default KV dtype — the live one) converts each
row with NEON bf16→f32 (good) and then a **scalar** `GGML_COMPUTE_FP32_TO_FP16`
loop over 1,024 elements. That's 57,344 scalar conversions per token in
steady state (~0.15–0.25 ms). Aarch64 has native vector fp16 conversion:
```cpp
// f32[4] -> f16[4], 8 elements per iteration with two q-regs
float16x4_t h = vcvt_f16_f32(vld1q_f32(f + i));
vst1_f16(dst + i, h);
```
Cuts the convert ~5–8×, composes with finding 9.

### 11. Prescan: ~6 full graph passes per token
`ggml-axcl.cpp:2877-3190`.

Anchor scan, final-norm scan, out_add scan, n_tok scan, SET_ROWS/KV scan,
cache-base detection — each walks all ~600–1,000 nodes touching scattered
tensor metadata (cache-miss dominated), ~50–150 µs/token total. They can be
one fused pass with one accumulator struct. Also worth short-circuiting:
for a decode graph the structure is identical every token; a cheap
fingerprint (n_nodes + first/last node ops + n_layer anchors) would let
steps like the GGUF registry scan (already gated) and cache-base detection
(a nested dedup loop) run once and be skipped thereafter.

### 12. IO binds: 4 per layer call → 2
`ggml-axcl.cpp:2664-2668`.

Per layer call: bind hidden input, K row out, V row out, y out. The K/V row
addresses must move with `pos` — unavoidable. But the (hidden-in, y-out)
pair only ever alternates between the same two states
(`d_yout ↔ d_yout_alt`). Creating **two pre-bound IO handles per layer**
(A: in=yout_a/out=yout_b; B: in=yout_b/out=yout_a — each with K/V/idx/mask
statically bound too) reduces per-call binds to just the two K/V rows.
Your own fused-engine measurements showed binds cost real descriptor
re-setup time, so this is plausibly 0.1–0.8 ms/token (56 handles total —
you already run 12 per layer for chunk groups). Worth an A/B behind an env
flag in your usual style.

### 13. Micro items (small, but they're on 500 Hz–1 kHz paths and free)
- `std::unordered_set<int> done` (`2865`) is constructed per token and
  `done.count(i)` is called for **every** node (`3201`) — hash+bucket walk
  for a set that's almost always empty. A `std::vector<uint8_t>` sized
  `n_nodes` (or an int watermark — inserts are always `< i`) is ~10–20 µs
  and removes an allocation.
- `ggml_axcl_host_op` GET_ROWS allocates `std::vector<float> rowbuf` per
  call (`1460`) — one malloc/free per token. Make it a member/static scratch.
- `axcl_us()` chrono calls in `axcl_layer_run` (28–56/token) feed only the
  debug stats — fine to keep, but gate them if you ever chase microseconds.
- `simple.cpp:203`: `fflush(stdout)` after every token. On a slow SSH
  console this can cost 0.1–2 ms/token; for benchmarking runs, redirect to
  a file/`void` pipe or drop to line-buffered and measure the difference —
  it may be polluting your t/s numbers today.

---

## What I checked and deliberately did *not* flag
- **Legacy per-op paths** (matmul engines, attn engine, chain, QKV/gate-up
  fusions): dead in layer mode (`layer_only` skips their loads at `4100`),
  and their hot-path branches are behind `g_*.model != 0` guards that cost
  a predictable branch. Not worth deleting for perf; worth deleting
  eventually for readability (~2,000 lines).
- **Pinned staging, per-token idx/mask hoisting, bind-once statics, in-place
  K/V rows, deferred write-back**: all already optimal or near-optimal —
  the 35 ms → ~2 ms host overhaul holds up under review.
- **Async chain** (`GGML_AXCL_STREAM`): correct shape (enqueue 28, sync
  once). The `SynchronizeStream` after layer 27 is the only sync point —
  good. Post engine executes synchronously after it; serializing on data,
  so no overlap available without speculative verify.
- **64 MB per-buffer slack** (`ggml_backend_axcl_buffer_type_alloc_buffer`):
  legacy-path only; `get_buffer_type` returns the CPU buffer type in layer
  mode, so this allocator is bypassed.
- **`axcl_post_load`'s 1 MB stack buffer** (`2066`): works (8 MB stacks) but
  it's a loaded gun on the scheduler thread — heap it while you're in there.

## Suggested verification order
1. static-cache all getenv + claim table (findings 5+7) → one rebuild, A/B
   `[layer] wall= vs eng=` and `perf record -- llama-simple`.
2. Cache device index (8) — same session.
3. Flush spreading + NEON f16 + tunable interval (9+10) — watch the
   every-32nd-token spike disappear in a per-token wall-time trace.
4. Chunk-path idx/mask hoist (6) before landing the uncommitted ladder.
5. Bind ping-pong A/B (12) behind `GGML_AXCL_BIND2=1`.
6. Then the structural items (post trim, s4, vocab64) carry the rest.
