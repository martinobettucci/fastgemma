# fastgemma engineering journal

Running log of experiments, measurements, dead ends and ideas. Newest entries at
the bottom of each section. Numbers here are measured on the target box unless
explicitly marked as an estimate.

**Target box:** Intel Xeon, Sapphire Rapids class, 4 cores @ 2.1 GHz, 15 GB RAM,
no GPU. **Target workload:** 8 concurrent requests, ~8k prompt in, ~2k out,
~12 tools × 4 params, tool-call exactness weighted equally with speed.

---

## Standing ideas / queued work

| idea | status | note |
|---|---|---|
| AMX INT8 tile GEMM | **done** | 12.7× AVX512-VNNI on this box; the core bet |
| Hadamard rotation (rotconv / QuaRot-style) | **done** | int4 err 0.194 → 0.078 on outlier weights |
| int4 group-size dial | **done** | converter `--group`; 64→256 = 2.2× for ~13% err |
| MSE-searched int4 scale | **done** | 0.107 → 0.0906 gaussian, conversion-time only |
| KV sharing + sliding-window KV | **done** | 32 MB/seq at 8k vs ~1.8 GB naive for 8 seqs |
| Batched decode (shared GEMM across seqs) | **done** | `forward_multi` |
| Persistent worker pool | **done** | GEMM split on N, attention on (token, head) |
| Flash-tiled attention | **rejected and deleted** | loses at *every* shape 128→8192 (−1.3% to −17.3%). §12 |
| VNNI-INT8 Q·Kᵀ and P·V | **done** | +34.6% prefill at 8192; attention 61.9% → 46.2% of prefill. §14 |
| AMX-INT8 attention | **open** | VNNI ≈ 580 G MAC/s ceiling vs AMX's 7.35 T; needs K in tile layout too (V already is) |
| Multithread the GEMM preamble (FWHT/quant/pack) | **done** | measured +25% at 2k, +20% at 4k, +9% at 8k prefill; bit-exact |
| K-blocking for L1-resident AMX operands | **open** | GEMM 3.7 of ~11.8 achievable TOPS |
| BF16 attention | **rejected** | KV is already int8 — bf16 *doubles* KV bandwidth, and AMX-BF16 is half AMX-INT8 (7.0 vs 14.69 TOPS). Only P·V is a genuine candidate |
| Dual-format weights (int8 prefill / int4 decode) | **done, measured** | int8 +10–14% prefill, int4 +0–6% decode — the split the design predicted. §14 |
| Prefix sharing (shared tool definitions) | **in progress** | `KvCache::fork_from`; 12 tools × 4 params is identical across the 8 requests |
| Chunked prefill + decode piggyback | **open** | amortise weight reads across both phases |
| Grammar-constrained tool decoding | **done (rewritten)** | Gemma 4 *native* call syntax, not JSON: 518-state DFA, 17 MB mask, 42% of steps forced. The JSON version could reach only 1 of 12 tools. §7 |
| Behavioural acceptance eval (tools + 8k retention) | **in progress** | replaces HF logit agreement as the accuracy gate |
| Activation-sparsity skipping | **n/a** | Gemma 3n had 95% sparsity; **Gemma 4 does not** |
| AMX tile-state guard (platform defect) | **done** | sentinel-biased accumulator + per-block retry, 1-2% cost |
| **Speculative multi-token prediction (MTP)** | **queued — after base perf** | https://ai.google.dev/gemma/docs/mtp/overview — explore an MTP drafter *only once base throughput is squeezed out*, per the brief. Decode here is DRAM-bound at batch 8, which is exactly the regime where speculation pays: extra tokens per weight read are ~free. Open questions: does Gemma 4 ship MTP heads for E2B/E4B, or do we train/distil a drafter? How does the accept rate interact with grammar constraints (a rejected draft inside a JSON tool call is cheap to re-mask, so acceptance may be *higher* under constraints)? |

---

## 0. Scoping — the model is Gemma 4, not Gemma 3n

Started against `gemma-3n-E2B` because "E2B/E4B" is Gemma 3n's naming and that
matched training knowledge. **Wrong.** Gemma 4 shipped 2026-03-02 with its own
E2B/E4B variants; the user corrected this. Killed the 3n download, verified on
the HF API, restarted on `google/gemma-4-E2B-it` (ungated, 200 — unlike
`google/gemma-3n-*` which is gated).

Lesson: when a model name is ambiguous and the knowledge cutoff is in play,
check the registry before committing 10 GB of download.

Wasted: ~4 GB of download, ~20 min. The 3n architecture study (AltUp, LAuReL)
was thrown away — Gemma 4 has neither.

---

## 1. Hardware ceilings (`bench/roofline`)

Measured first, because every later decision keys off these.

| | measured |
|---|---|
| AMX INT8 (`TDPBSSD`) | **14.69 TOPS** |
| AMX BF16 | 7.00 TFLOPS |
| AVX512-VNNI | 1.16 TOPS |
| L1 read | 393 GB/s |
| L2 read | 209 GB/s |
| L3 read (260 MB) | 58 GB/s |
| DRAM read | 33 GB/s |

**AMX is 12.7× VNNI.** llama.cpp and onnxruntime CPU paths are VNNI/AVX2. That
gap is the entire thesis of this project.

Second measurement, which turned out to matter more than the first — cost of one
2×2 AMX k-step (4 `tile_loadd` + 4 `tile_dpbssd`) by operand residency:

| operands in | cyc/k-step | TOPS (1 thread) |
|---|---|---|
| registers | 69 | 3.99 |
| L1 | 67 | 4.09 |
| L2 | 93 | 2.96 |
| L3 | 332 | 0.83 |
| DRAM | 388 | 0.71 |

Tile loads are **free from L1 and catastrophic from L3** (4.8× cliff). Blocking
for residency is worth more than any instruction-level tuning.

---

## 2. What Gemma 4 E2B actually is

Read off the checkpoint, not the model card. Several of these are exploitable:

- 35 layers, hidden 1536, MQA **1 KV head**, `head_dim` 256 on sliding layers but
  **512 on full-attention**, pattern `ssssF`×7.
- **20 of 35 layers share KV** and ship k/v projection weights the model never
  reads (`_keys_to_ignore_on_load_unexpected`). Dropped at conversion.
- Those same 20 layers get a **double-wide MLP** (12288 vs 6144).
- Only layers 0–14 store KV; only 4 need full length → **~32 MB/seq at 8k**.
- PLE table is 262144 × 8960 = **2.35 B params, larger than the whole compute
  network (1.86 B)** — but a pure gather.
- Full-attention RoPE is `proportional`, `partial_rotary_factor` 0.25: only 128
  of 512 head dims rotate.
- Attention `scaling = 1.0` (not 1/√d) — q_norm carries the magnitude.
- RMSNorm multiplies by `weight`, **not** `1 + weight` as in Gemma 2/3.

E4B: hidden 2560, 42 layers, 2 KV heads, 18 shared, no double-wide MLP.

---

## 3. Quantisation experiments

**int4 scale, full range + MSE search.** Symmetric int4 with `amax/7` wastes a
level. Switched to the Q4_0 convention (extreme element maps onto −8, all 16
levels) plus a per-group sweep over shrink factors keeping the lowest MSE.
Conversion-time only, no runtime change.

| | gaussian rel err |
|---|---|
| `amax/7`, 15 levels | 0.1074 |
| Q4_0 convention + MSE search | **0.0906** |

**Hadamard rotation (rotconv).** Fold `H` into the weight along K, apply the
matching FWHT to the activation at runtime. Orthogonal ⇒ mathematically
identical, but kills outliers in both weight groups and activation rows.

| weight | int4 rel err |
|---|---|
| gaussian | 0.0906 → 0.0777 |
| gaussian + outlier channels | **0.194 → 0.078** (2.5×) |

Decided to rotate **every** linear's K axis, not just `down_proj`. Runtime cost
is a 128-point FWHT (~10 K ops/token) against an 18.9 M-op GEMM — noise.
FWHT verified exact against an explicit Hadamard matrix (max abs err 1.2e-5)
and as an involution (7.6e-6).

**Group size dial.** int4 group scales force an int32→f32 accumulator drain
every `group/64` tile steps. Drain is O(MR·NR), the AMX work it amortises is
O(MR·NR·group) ⇒ group is a direct speed/accuracy knob.

| group | rel err | GEMM TOPS @ M=256 |
|---|---|---|
| 64 | 0.0902 | 0.98 |
| 128 | 0.0968 | 1.71 |
| 256 | 0.1018 | 2.19 |
| 512 | 0.1055 | 3.02 |

**End-to-end A/B, group 64 vs 256** (clean machine, 4 threads, E2B):

| | g64 | g256 | delta |
|---|---|---|---|
| prefill 256 tok | 117.3 tok/s | **142.4** | +21% |
| prefill 512 tok | 101.2 tok/s | **131.9** | +30% |
| decode, 1 seq | 12.72 tok/s | **14.47** | +14% |
| converted size | 2.83 GB | 2.78 GB | — |
| measured weight rel err | 0.0902 | 0.1016 | +13% |

Group 256 is the better operating point and is now the default recommendation.
The standalone GEMM bench predicted 2.2×; end-to-end delivers 1.2–1.3× because
the GEMM is only ~65–75% of prefill and the per-GEMM preamble (FWHT, activation
quantisation, A packing) is unchanged and still single-threaded.

Note the 64-token row for g256 read 32 tok/s on first run — cold page cache on a
freshly written 2.78 GB file, not a regression. Re-runs land at ~118 tok/s.

**Shipped E2B config:** FFN int4-g64, attention int8-per-channel, embed/LM-head
int4, PLE int4, norms f32. 10.2 GB bf16 → **2.83 GB** in 1076 s.

---

## 4. Kernel work (`crates/fgm-kernels/csrc/amx_gemm.c`)

**Trap 1 — GCC `-O3` miscompiles the int4 path.** `_tile_loadd` is opaque to
GCC's alias analysis, so at `-O3` it sinks or eliminates the AVX stores that fill
the int4 unpack buffer. Silent garbage at `-O3`, correct at `-O2`. Cost ~40 min
of chasing a "correctness bug" that was a codegen bug: the same shape passed in a
standalone sweep and failed in the bench, differing only by `-O2` vs `-O3`.
Fixed with an `asm volatile ... "memory"` barrier in `unpack_tile`; both paths
now verified bit-exact against scalar references across 19 shapes.

**Trap 2 — A must be tile-packed.** Loading a 16×64 A tile straight out of
row-major `A[M,K]` touches 16 cache lines K bytes apart (24 KB of stride at
K=1536). Those stalls cost more than the `dpbssd` they feed. Added `fgm_pack_a`
(O(M·K) against the GEMM's O(M·N·K)).

**Trap 3 — loop order.** Original nesting re-streamed all of B once per 32-row M
block. Added panel blocking so B panels stay L2-resident while M streams through.

Current GEMM throughput, 4 threads, N=6144 K=1536 (E2B FFN shape), TOPS:

| M | q8c | q4g/64 | g128 | g256 | g512 |
|---|---|---|---|---|---|
| 8 | 0.68 | 0.53 | 0.57 | 0.60 | 0.59 |
| 64 | 2.55 | 0.85 | 1.15 | 1.63 | 1.99 |
| 256 | 3.70 | 0.98 | 1.71 | 2.19 | 3.02 |
| 512 | 2.88 | 0.76 | 1.11 | 1.51 | 1.81 |

**Still ~3× short of the L2-resident ceiling (~11.8 TOPS aggregate).** The
residency table says tile loads are landing in L3. Next kernel pass: K-blocking
so both A and B micro-panels sit in L1 (Kc≈512 ⇒ 16 KB + 16 KB, fits the 48 KB
L1), draining to an f32 C accumulator per K-chunk.

---

## 5. Forward pass and validation

Validated layer-by-layer against `transformers` 5.14.1 (bf16 reference).
Layers 0–33 track HF at **cos > 0.99** with smoothly accumulating error and no
step change — the signature of quantisation noise, not a structural bug.

**Trap 4 — HF's `output_hidden_states` records layer *inputs*.** `hidden[0]` is
the embedding, `hidden[l+1]` is layer *l*'s output only for `l ≤ L-2`, and
`hidden[-1]` is `last_hidden_state`, i.e. already through the final norm.
Comparing our last layer against `hidden[-1]` showed a fake "explosion"
(cos 0.15, norm 189.8 → 53.8) that was pure misalignment. Cost ~20 min.

**Trap 5 — `_mm512_exp_ps` does not exist in GCC** (Intel-compiler SVML only).
Replaced with a Cephes range-reduction `exp512_ps`.

**Trap 6 — logits buffer sized for 16 rows, validation asked for 96.** Silent
out-of-bounds write. Now sized explicitly with an assertion.

### Measuring accuracy: three attempts, two of them wrong

1. *Single-token top-1 vs HF.* Useless — HF's top-5 sat within 0.4 logits and
   HF's own top-1 **changed between runs**. "top-1 disagrees" measured nothing.
2. *Greedy agreement over 96 **random** token ids.* Also bad: random ids give a
   near-uniform next-token distribution, so argmax is unstable by construction.
   HF-vs-HF came out at 88.5%, fastgemma 50.0% — uninterpretable.
3. *Greedy agreement on real tokenized text.* Better prompt, but the first run
   still reported an "HF vs HF noise floor" of 84.0%, which I nearly published
   as the baseline.

**Trap 7 — the noise floor was my own measurement bug.** torch's *multithreaded*
bf16 path is non-deterministic on this model. Measured directly:

| torch threads | HF vs HF greedy agreement | max abs logit diff |
|---|---|---|
| 1 | **100.0%** | 0.0000 |
| 4 | 86.4% | 10.7073 |
| 1 vs 4 | 87.7% | 10.4201 |

So the "floor" was an artifact, and comparing against it would have flattered
the engine by ~15 points. The reference is now pinned to `torch.set_num_threads(1)`
in `bench/validate/greedy_agree.py`, with a determinism assertion printed every
run. Lesson: when a baseline looks noisy, prove the noise is real before you
grade yourself against it.

### Honest accuracy, E2B int4-g64, real text, deterministic reference

| metric | value |
|---|---|
| greedy agreement | **82.7%** (67/81 positions) |
| HF's pick in our top-3 | 98.8% |
| HF's pick in our top-5 | 100.0% |
| mean cosine | 0.99901 |
| median top1–top2 logit gap | 3.79 overall, **1.04 where we disagree** |

Disagreements concentrate almost entirely on near-ties, which is what int4
weight error predicts. 82.7% is not yet good enough for the tool-exactness half
of the brief — the planned lever is int8 for the accuracy-critical tensors
(LM head first, then FFN) and measuring the agreement/throughput trade the same
way the group-size dial was measured.

---

## 6. Performance so far (E2B, 4 threads, group 64)

Prefill, single sequence:

| tokens | tok/s | ms/tok |
|---|---|---|
| 64 | 67.8 | 14.75 |
| 128 | 115.4 | 8.67 |
| 256 | 105.9 | 9.44 |
| 512 | 90.2 | 11.08 |

Decode, 1 sequence: **13.0 tok/s** (76.7 ms/step).

Phase profile (`FGM_PROFILE=1`) before the pool landed:

| tokens | ffn_gemm | attention | qkv+o | ple_inject |
|---|---|---|---|---|
| 64 | 66.5% | 4.5% | 17.3% | 5.3% |
| 256 | 65.2% | 16.6% | 10.8% | 4.2% |
| 512 | 54.2% | 30.8% | 9.4% | 3.5% |

Attention was single-threaded and O(M²) ⇒ moved onto the worker pool, split over
(token, head) pairs so batch-8 decode still uses 4 threads.

Prefill at 115 tok/s = 0.43 TOPS, against 3.70 TOPS measured for the same GEMM
shape standalone. **Most of the gap is not the GEMM** — it is the per-GEMM
preamble (FWHT + activation quantisation + A packing), which is still
single-threaded. That is the next target after K-blocking.

**Benchmarking hygiene note:** one sweep was silently contaminated by a
background conversion job stealing a core (prefill "dropped" 105 → 85 tok/s).
Always check for background load before trusting a delta.

### Batching (the point of the whole exercise)

`forward_multi` shares every GEMM across concurrent sequences. Measured at
8 concurrent, 512 in / 32 out, group 256:

| | tok/s |
|---|---|
| decode, 1 sequence | 14.47 |
| decode, 8 concurrent (aggregate) | **58.7** |
| per-sequence at batch 8 | 7.3 |

**4.05× aggregate throughput** from batching. Decode is DRAM-bound on the weight
read, which does not grow with batch size, so this is the expected shape — and
it is also exactly why MTP speculation should pay here.

### Long-context prefill: attention, not GEMM, is the wall

Running the real 8×8k target workload exposed the next bottleneck. Rough op
counts for one 8192-token prefill:

- GEMM: 8192 × 3.73 GOP/token ≈ **30.5 TOP**
- attention: 28 sliding layers are cheap (window 512) at ≈ 0.96 TOP total, but
  the **7 full-attention layers are O(M²)** at ≈ 3.85 TOP — and they run
  `head_dim` 512, double the sliding layers.

So attention is ~14% of the ops but a much larger share of the time, because the
attention kernel is naive: f32, one int8 KV element dequantised at a time, no
blocking, softmax over the whole row in one pass. The GEMM path has had three
rounds of tuning; attention has had none.

Next steps there, in order of expected payoff:
1. Flash-attention-style tiling so K/V stay in L1 and softmax is online.
2. Feed Q·Kᵀ through AMX as well — it is an int8 × int8 GEMM in disguise
   (`q_norm` already bounds Q's range), which is the same 12.7× lever.
3. Skip the sliding layers' out-of-window positions in the *scale* array too.

**Trap 10 — benchmarking a long run while doing anything else.** The first
attempt at the full 8x8k serve run died silently ~15 minutes in, with an empty
log after the KV line. I had been running cargo builds, a 3.35 GB download and
another bench alongside it. Worse, my liveness check was
`pgrep -f "fgm-bench serve"`, which matched Claude Code's own multi-kilobyte
command line and reported the run as healthy long after the process was gone.
Re-running with nothing else on the box and an explicit `pgrep -x fgm-bench`
plus a memory trace. Two lessons: match process names exactly when the
environment has enormous command lines, and treat "the long benchmark is still
running" as a claim needing evidence (load average was 0.00 the whole time).

---

## 6b. Two KV bugs and one open defect (found by running the real workload)

The full 8×8k serve run is what exposed all of this. Nothing under 512 tokens
could reach any of it.

**Bug A — ring-buffer KV read by absolute position. Segfault.**
Sliding layers that are not a shared-KV source store only `sliding_window`
positions in a ring. The *write* path mapped through `LayerKv::slot(pos)`; the
*read* path in attention indexed by absolute position. Past position 512 it read
hundreds of KB off the end of the allocation. Every sweep so far used ctx ≤ 640,
so it never fired. Fixed by threading the ring capacity into
`fgm_attend_q8_heads` and mapping reads through it. An earlier draft of the
forward pass literally had `debug_assert!(!lk.ring, "ring windows need
slot-mapped attention")` — I deleted the assert during a refactor instead of
implementing what it demanded, and release builds skip `debug_assert!` anyway.

**Bug B — a ring sized `window` is too small for a batched forward.**
Fixing A turned the segfault into wrong answers. A batched forward writes *all*
m rows before any attention runs, so with only `window` slots the later rows
overwrite history the earlier rows still need. A ring must hold
`sliding_window + max_batch`. Fixed, with a `debug_assert` that states the
requirement. This one produces silently wrong logits, not a crash — much worse.

**Partly-fixed defect — intermittent non-determinism.**
The regression test that caught B also caught this. Two identical `forward`
calls on freshly zeroed caches sometimes disagree.

*Cause 1, found and fixed: undefined behaviour in the `forward` wrapper.* It
took the scratch `seq`/`pos` vectors out of `self` with `mem::take`, called
`forward_multi`, kept a raw pointer into the returned slice, mutated `self` to
put the vectors back, then rebuilt the slice from that pointer. Holding a
pointer into `self.b.logits` across a mutation of `self` is UB, and the
optimiser is entitled to anything. Removing it (two small local `Vec`s instead —
free next to a forward pass) made several sizes go bit-deterministic. `forward.rs`
now contains no `unsafe` at all.

*Cause 2, still open.* A residue remains, and it is **intermittent** — the same
size flips between runs:

| tokens | run A | run B |
|---|---|---|
| 520 | 1.48 | — |
| 600 | 0.00 | 0.00 |
| 700 | 0.94 | — |
| 800 | 0.00 | — |
| 1024 | 0.00 | — |
| 1200 | 1.95 | **0.00** |

Ruled out: data race (1 thread reproduces), heap-address dependence (same
buffers both calls), the ring mapping (all these sizes make ring capacity equal
`max_len`, so ring and linear indexing coincide). Flakiness across *identical*
configurations points at an uninitialised read whose value happens to be stable
most of the time — the AMX kernels' stack `acc[4][256]` / `bt[4][1024]` and the
M-tail paths (`mr0`/`mr1` < 16) are the prime suspects, since the failing sizes
520 and 700 both hit tail paths.

Note the bisect harness (`FGM_ZERO=all|<buffer>`, zeroes scratch at the top of
each call) did **not** isolate it and in one run made things worse, which is
itself evidence the source is kernel-local stack rather than a Rust-side buffer.

Consequence to be honest about: **the 82.7% greedy-agreement figure has error
bars I did not measure.** argmax agrees between runs in every ring test, and the
differing logits sit in the same near-tie band quantisation already perturbs, so
the headline is probably close — but "probably" is not a measurement. This gets
fixed before any accuracy number is quoted as final and before the llama.cpp
comparison is worth running.

---

## 8. The non-determinism was a platform defect: AMX tile state is lost on context switch

The intermittent wrong answers were not a bug in this code. Chain of evidence:

1. **The GEMM kernels are bit-stable in isolation** — 30 repeats of an identical
   q4g/q8c call, byte-identical output, at every batch size that failed
   end-to-end. So the arithmetic is fine.
2. **Under CPU contention they fail every time** — same test, 6 competing
   spinners: 25/25 q4g mismatches, 25/25 q8c, 15/15 at M=1200. Quiet: 1/25.
3. **A minimal probe with none of this project's code reproduces it.** Load a
   known pattern into a tile with `_tile_loadd`, busy-wait, `_tile_stored`,
   compare (`bench/platform/amx_tilestate.c`). The tile comes back **zeroed**.

Corruption probability tracks how long the state is held, which is exactly the
signature of losing it across a context switch:

| hold | corrupted (400 trials, 6 spinners) |
|---|---|
| 0 | 0% |
| 10 µs | 0.25% |
| 100 µs | 3% |
| 1 ms | 16% |
| 10 ms | 89% |

That implies a context switch roughly every ~7 ms, each of which resets the
tile file to INIT. `XTILEDATA` permission is requested per thread
(`arch_prctl(ARCH_REQ_XCOMP_PERM)`) and the probe still fails, so this is the
kernel or the hypervisor not saving guest AMX state — not a missing opt-in.

**Why every earlier test passed.** The kernel correctness sweeps run in
milliseconds on an idle box; the end-to-end runs take minutes. The failures
appeared exactly when the machine got busy, which is why the first sighting was
"the engine is non-deterministic" rather than "the platform is broken".

**Why it looked like an uninitialized read.** Corruption zeroes tiles, so a
corrupted accumulator produces a *plausible smaller* number rather than garbage.
It clustered by run and moved between layers, which is what sent me looking for
stale buffers.

### What this means for the project

The AMX bet is sound — 14.69 TOPS vs 1.16 for VNNI is measured and real, and the
kernels compute the right answer. But on *this* VM, any tile-resident
accumulator held longer than a few hundred microseconds is at risk, and a full
GEMM holds thousands of them. Under load the engine will be intermittently
wrong.

Mitigation options, in order of preference:

1. **Sentinel row + per-block retry.** Configure accumulators with 16 rows but
   use only 15 for data, seeding row 15 with a distinctive constant and keeping
   the matching A row zeroed so nothing accumulates into it. After each block,
   check row 15; if it is not the constant, the tile file was reset — redo the
   block. Per-block windows are under a microsecond, so retries are rare and
   cheap. Costs ~6% throughput (M-blocking 30 instead of 32).
2. **Shrink the tile-resident window** by draining accumulators more often. Helps
   probabilistically, does not eliminate.
3. **Run on hardware that preserves XTILEDATA.** The right long-term answer, and
   the reason the throughput numbers here remain meaningful.

Until (1) lands, every accuracy number from this engine is provisional and every
throughput number should be read as "correct arithmetic, on an idle box".

Note this hits **any** AMX user on this VM, including llama.cpp built with
`GGML_AMX_*`. Its default AVX512/AVX2 path is unaffected, so the baseline
comparison is still valid — and it is the honest comparison anyway, since that
is what llama.cpp actually ships.

### Only pay for the guard where it is needed

The guard costs 1-2%, so it should be conditional on the platform actually being
broken. Detection took two attempts, and the failed one is the interesting part.

**Attempt 1 — hold tile state across `nanosleep`.** A sleep is a guaranteed
reschedule point, so this looked like a clean forced-preemption test. It reports
**zero corruption even at a 1.6 ms hold**, on the very box whose GEMMs
demonstrably corrupt. So a *voluntary* context switch preserves XTILEDATA
correctly; only *involuntary* preemption loses it. That is a useful clue about
where the fault lives, and it would have been a catastrophic false negative:
the guard would have switched itself off on a broken machine.

**Attempt 2 — make the probe generate real scheduling pressure.** Oversubscribe
the machine (2x cores in spinner threads), busy-wait while holding tile state.
This detects reliably, and escalates the hold time until it is conclusive
(2 ms → 8 ms → 16 ms), because the stakes are asymmetric: a false positive costs
1-2% throughput, a false negative means silently wrong answers under load.

Measured on this box, 5 consecutive warm-ups, 12 trials each:
detected every time at the first 2 ms stage (5-7 of 12 trials corrupted), with
end-to-end determinism 0.000000 in all five. Warm-up cost ~25 ms.

`FGM_TILE_GUARD=on|off` overrides; the default is auto-detect, and the fallback
if the probe never runs is guard-on rather than risk silent corruption.

### The guard works

Under 6-way CPU contention, same GEMM repeated:

| | q4g mismatches | q8c mismatches | tile resets recovered |
|---|---|---|---|
| guard **on**, M=700 ×25 | **0** | **0** | 67 |
| guard **on**, M=1200 ×15 | **0** | **0** | 123 |
| guard off, M=700 ×25 | 8 | 11 | — |

End-to-end `ringtest` self-determinism went to **0.000000 at every size**
(520 / 700 / 1200), and the ring-vs-linear comparison now passes.

Guard cost, measured with `FGM_NO_TILE_GUARD=1`:

| | guard on | guard off | cost |
|---|---|---|---|
| prefill 128 | 168.9 | 170.8 | 1.1% |
| prefill 256 | 164.5 | 168.0 | 2.1% |
| decode | 15.31 | 15.41 | 0.6% |

**1–2% for correctness under preemption.** It stays on by default.

---

## 9. Baseline: llama.cpp, same box, same model, same bit width

`google/gemma-4-E2B-it-qat-q4_0-gguf` (Google's own QAT Q4_0), llama.cpp
build 91f8c9c, `GGML_NATIVE=ON`, 4 threads. Both engines measured on an idle
machine — an earlier fastgemma sweep was contaminated by the llama.cpp build
running concurrently and read ~15% low.

**Prefill (tok/s):**

| tokens | fastgemma | llama.cpp | |
|---|---|---|---|
| 128 | **168.9** | 163.4 | fgm +3% |
| 256 | **164.5** | 161.0 | fgm +2% |
| 512 | 142.9 | **158.5** | llama +11% |

Our prefill degrades with length where llama.cpp's does not — that is the naive
O(M²) attention kernel (§6b), not the GEMM.

**Decode (tok/s aggregate), 512-token prompts, the target workload:**

| concurrency | fastgemma | llama.cpp | |
|---|---|---|---|
| 1 | 15.0 | **19.5** | llama +30% |
| 4 | **46.7** | 40.1 | **fgm +16%** |
| 8 | **67.3** | 59.7 | **fgm +13%** |

This is exactly the shape the architecture predicts. Single-stream decode is
pure memory bandwidth and llama.cpp's Q4_0 kernels are extremely well tuned;
AMX cannot help when M=1. As soon as requests batch, the same weight read serves
8 rows and the tile units have enough M to work with — and we pull ahead.

**Honest summary: at the concurrency this project was briefed for, fastgemma
beats llama.cpp on decode by 13%. It does not beat it on single-stream, and
loses on long prefill.** The remaining levers (K-blocking for L1-resident AMX
operands, flash-style attention, the still-single-threaded GEMM preamble) all
point the same way, and none of them are exhausted.

---

## 10. Is bit-determinism even the right goal? (challenged, and it held)

Fair challenge raised: torch isn't deterministic, flash-attention isn't
deterministic, and reduction-order differences producing different-but-correct
results are completely normal in ML. So is chasing determinism costing us
performance for a non-bug?

**The general point is right, and I measured it myself**: HF at 4 threads
disagreed with itself on 13.6% of token positions (max logit diff 10.7). That is
exactly the normal phenomenon, and I nearly published it as a noise floor.

**But "bit-identical?" is the wrong test.** The right one is *"is this delta
explainable by floating-point reduction order?"* Three independent reasons it
was not, here:

1. **The AMX accumulator is int32.** Integer addition is associative and exact —
   `a+b+c` is order-independent. A varying int32 dot product has no legitimate
   explanation.
2. **Magnitude is ~4 orders too large.** fp32 reduction-order effects on a
   K=1536 dot land near 1e-4 relative. Observed: absolute logit deltas of 1–7
   against a median top1–top2 gap of 3.79 — large enough to change the emitted
   token. Rounding noise does not change which token you emit.
3. **The minimal repro contains no arithmetic at all.** Store a byte pattern in a
   tile, wait, read it back, bytes differ. No FMA, no reduction, no order.

So: data loss, not numerical noise.

**Determinism here is free, not purchased.** fastgemma is deterministic *by
construction* — int32 accumulation, threads own disjoint output columns (no
cross-thread reduction anywhere), fixed loop order. That is precisely why it made
such a sensitive bug detector. If buying determinism ever costs throughput —
forcing reduction order, disabling split-k, serialising accumulation — it goes.

**Process lesson:** ask "is this near fp32 epsilon?" *first*. One minute of that
would have ruled out normal ML noise immediately and pointed at corruption,
instead of an hour spent hunting uninitialised buffers.

---

## 11. Prefill and decode are different problems (concurrency-1 curves)

Full target-range curve, single sequence, group 256, guard on:

| prefill tokens | tok/s | | decode from 8192 ctx | tok/s | ms/step |
|---|---|---|---|---|---|
| 128 | 156.0 | | 128 out | 13.1 | 76.5 |
| 256 | 166.8 | | 256 out | 13.0 | 77.1 |
| 512 | 149.8 | | 512 out | 13.1 | 76.3 |
| 1024 | 147.2 | | 1024 out | 12.9 | 77.3 |
| 2048 | 135.4 | | 2048 out | 12.7 | 78.5 |
| 4096 | 112.9 | | | | |
| 8192 | **90.5** | | | | |

Two things fall out, and they point in opposite directions:

- **Prefill collapses 42%** from 128 → 8192. That is the O(M²) attention kernel,
  not the GEMM.
- **Decode is flat** — 13.1 → 12.7 tok/s with context growing to 10k. The
  sliding-window + KV-sharing design means decode barely notices context length.
  Attention is *not* the decode bottleneck; the weight read is.

So the optimisation queue is really **two queues**:

**Prefill** (TTFT): flash-style tiling → AMX-INT8 for Q·Kᵀ → multithread the
GEMM preamble → K-blocking for L1 residency.

**Decode** (output throughput): none of the above move it. The levers are the
grammar-forced LM-head skip (36% of tool-call steps need no LM-head read at
all), int4 on attention projections, and then MTP.

### Why INT8 and not BF16 for attention

Proposed: bf16 attention to save bandwidth, flash-style over bf16. Both halves
need correcting.

- **Bandwidth: bf16 is a regression.** KV is already int8 (40 MB/seq at 10k);
  bf16 doubles it to 80 MB. That spends bandwidth rather than saving it.
- **Compute: INT8 is 2× BF16 on this box** — 14.69 TOPS vs 7.00 TFLOPS — and the
  operands are *already* int8. K sits in cache as int8, and Q is cheap to
  quantise because `q_norm` bounds its range before RoPE by construction. Going
  bf16 would mean converting int8 → bf16 to run at half rate.
- **Flash tiling is orthogonal to dtype.** Its win is L1 residency (67 cyc/k-step
  vs 332 from L3) plus online softmax avoiding the full attention matrix. Do it
  regardless.

The one genuine bf16 candidate is **P·V**: softmax probabilities live in [0,1]
with a long tail and quantise to int8 badly. Either keep the AVX-512 weighted sum
over int8 V, or use AMX-BF16 there. Worth benchmarking; it is the smaller half.

### "Multithread the GEMM" — right problem, wrong name

The GEMM has been multithreaded since the pool landed (split along N, disjoint
output columns). What is still single-threaded is the per-GEMM **preamble**:
FWHT rotation, activation quantisation, A tile-packing. That is why a standalone
GEMM bench shows ~2 TOPS while the same shape inside the forward pass delivers
~0.5 — roughly half of "ffn_gemm" time is not GEMM. Worth ~20–30% of prefill and
far easier than K-blocking.

---

## 7. Constrained tool calling

DFA over the tool schemas → per-state token bitset. Masking is one AND per step;
the compile is one-time per tool set.

### 7a. First attempt (wrong grammar, three bugs) — kept as the record

The first version constrained decoding to **JSON**: `{"name": "tool_0",
"arguments": {…}}`. On the real 262144-token vocab, 12 tools × 4 params it
reported 1011 states, 3.4 s compile, 33.1 MB of mask, a walk accepted in 47
steps, output that parsed with the right tool and 4/4 args, and 36% of steps
admitting exactly one token. All of those numbers were real. The feature was
still broken in three separate ways, and *every one of them passed the "it
emitted valid JSON" check*.

**Trap 8 — special tokens have literal byte forms.** `<pad>` decodes to the five
bytes `<pad>`, which satisfy the JSON string-body class, so the first
constrained walk emitted `{"p0": "<pad><pad><pad>…` forever *without ever
violating the grammar*. Added/special tokens are now returned as empty byte
strings, making them unreachable by the byte walk.

**Trap 9 — a weak checker confirms whatever you believe.** The number rule was
"digits and dots", which accepts `..` and the empty string; the walk emitted
`"p1": ..` and my bracket-counting "is valid JSON" check passed it. Replaced
with a real number DFA and a real parse. If the validity check is weaker than
the property being claimed, it is not evidence.

**Trap 10 — eleven of twelve tools were unreachable.** Each tool chained its own
literal from the start state, so the start state held twelve edges keyed on the
same first byte and `step` always took the first. Measured: `reachable 1/12`.
The constraint could only ever emit `tool_0` — *tool selection, the half of the
brief weighted equally with speed, did not work at all*. A walk that emits one
valid call cannot see this, because the one call it emits is the reachable one.
The test that finds it is boring: assert that every tool in the set can be
spelled out.

**Trap 11 — `false` dead-ended.** Bool and Enum alternatives ended in separate
states joined by a fake byte-0 edge instead of converging, so choosing `false`
landed in a state with no outgoing edges: empty mask, every logit −∞. Latent in
every schema with a boolean, which is most of them.

### 7b. The grammar was JSON. The model does not speak JSON.

`tokenizer_config.json` ships `response_template.fields.tool_calls`:

    open_pattern  <\|tool_call>call:(?P<name>\w+)
    close         <tool_call|>
    content       json, unquoted_keys=true, string_delims=[["<|"|>", "<|"|>"]]

Confirmed against the published format, which our builder now reproduces
byte-for-byte:

    <bos><|turn>system
    You are a helpful assistant.<|tool>declaration:get_current_weather{description:<|"|>…<|"|>,
    parameters:{properties:{…}},required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
    <|turn>user
    Hey, what's the weather in Tokyo right now?<turn|>
    <|turn>model
    <|tool_call>call:get_current_weather{location:<|"|>Tokyo, JP<|"|>}<tool_call|>

Three delimiters are **single vocabulary tokens**, not text: `<|tool_call>`=48,
`<tool_call|>`=49, `<|"|>`=52. Keys are bare. Strings are token-delimited, so a
string body may contain `"` freely and needs no escape machinery.

Constraining this model to JSON forbids the token it wants at every structural
position and forces one it has never emitted there. That is the worst thing a
constraint can do: it converts a model that knows the format into a model
fighting the mask. The measured "structural validity" was real and completely
beside the point.

Delimiters are now reachable only through explicit per-state **token edges** —
which keeps `<pad>` out (they have no byte form) while letting exactly the three
delimiters in exactly where the grammar wants them.

12 tools × 4 params on the real vocab:

| | states | compile | mask | tools reachable | forced steps |
|---|---|---|---|---|---|
| JSON grammar | 1011 | 3.4 s | 33.1 MB | **1/12** | 36% |
| native grammar | 518 | 1.6 s | 17.0 MB | **12/12** | 42% |

Smaller, faster, correct, and closer to the model's own distribution.

**42% of steps admit exactly one token.** Those steps skip the LM head
entirely — a 201 MB int4 read whose argmax is predetermined (`forward_nolm`).

`crates/fgm-core/tests/grammar.rs` has one test per bug above. Every one of them
fails against the previous grammar.

### 7c. What the checks should have been

Three checks passed while the feature was broken: "it compiled", "the walk was
accepted", "the output parsed". They share a flaw — each verifies a property of
*one* path through the grammar, and the grammar's whole job is to offer many.
The replacements assert coverage instead of instance: every tool spellable,
every alternative completable, every malformed string rejected.

---

## 12. Blocked attention: a negative result, and why the estimate was wrong

Flash-style blocking (online softmax, block over query positions × heads) was
supposed to be the prefill win. It is not. A/B on an idle box, single sequence,
prefill tok/s, g256 weights:

| ctx | blocked | per-head | delta |
|---|---|---|---|
| 1024 | 171.6 | 188.1 | −8.8% |
| 2048 | 149.5 | 169.8 | −12.0% |
| 4096 | 123.7 | 135.1 | −8.4% |
| 8192 | 91.6 | 98.6 | −7.1% |

Two mistakes, in order of embarrassment.

**The first was in the kernel.** It dequantised each K block into an f32 scratch
buffer before the dot products — a 4× expansion that turned a 1 MB int8 block
into 4 MB and spilled the very L2 the blocking existed to exploit. Measured
−9.3% at 1024 and −15.5% at 2048. Fixed by keeping K int8 in cache and widening
inline with `_mm512_cvtepi8_epi32` per 16 lanes. That recovered part of the gap.

**The second was in the estimate, and it was the real error.** The work was
motivated by "1924 GB of KV re-reads", computed assuming those re-reads hit
DRAM. They do not. With one KV head the int8 K set is 0.5 MB at 2k context and
2 MB at 8k — already L2-resident on the per-head path. There was no DRAM traffic
to save. Blocking therefore bought nothing and paid for the online-softmax
rescaling per block, plus a coarser parallel decomposition than the per-head
path's (row, head) pairs.

The lesson is narrow and reusable: **a traffic estimate that does not name the
cache level it assumes is not an estimate.** Bytes moved is not a cost until you
say where they move from. Both numbers — 1924 GB and 2 MB — are correct; only
one of them is about this machine.

Kept behind `FGM_BLOCKED_ATTN=1` rather than deleted. It is the path that wins
once K/V genuinely exceeds L2 (much longer context, or E4B's two KV heads), and
re-deriving it later costs more than carrying it. The A/B lives in
`bench/ab_blocked.sh` so the next person does not have to rebuild it — the first
version of that script was written in `/tmp` and lost.

### The parallel GEMM preamble, measured

The same runs pin down the preamble win (FWHT rotation, activation quantisation
and A tile-packing, previously single-threaded while the GEMM itself was not):

| ctx | before | after | gain |
|---|---|---|---|
| 2048 | 135.4 | 169.8 | +25% |
| 4096 | 112.9 | 135.1 | +20% |
| 8192 | 90.5 | 98.6 | +9% |

Bit-exact against the serial path, since the work is per-row and the split
preserves evaluation order — which is why bit-identity is a valid check *here*
and not for anything that reorders float accumulation.

---

## 13. Re-review: which rejections were about precision, and which were not

The acceptance bar changed. It used to be greedy agreement with HF logits; it is
now behavioural — right tool, right arguments, right answer at 8k. That is the
correct bar for this engine (we deliberately trade numerical precision for
speed, so matching another implementation's numbers was never going to happen),
but it also invalidates part of the reasoning behind earlier decisions. Going
back through them, honestly, including the ones the change does *not* rescue.

| idea | why it was shelved | does the new bar change it? |
|---|---|---|
| BF16 attention | bandwidth (KV is already int8, bf16 doubles it) and compute (AMX-BF16 is half AMX-INT8) | **No.** The objection was never precision. Still rejected. |
| int8 P·V | "softmax probabilities quantise badly to int8" | **Yes — promote.** That was a precision objection and precision is no longer the bar. And the premise is weak anyway: after softmax the row max is exactly 1, so a per-row int8 scale is well conditioned. V is *already* int8, so P·V becomes an int8×int8 GEMM eligible for AMX at 14.69 TOPS instead of an AVX-512 f32 weighted sum. |
| int4 group 512 | +3.6% weight error for +38% GEMM TOPS (3.02 vs 2.19), judged on error | **Yes — measure.** Convert at g512 and run the behavioural eval. If tool exactness and 8k retention hold, 38% on the dominant cost is the cheapest win on the board. |
| int4 KV cache | not attempted; precision | **Yes, conditionally.** Halves KV bytes. Worth it only if the profile says KV traffic is material — gate on measurement, not on appetite. |
| drop the Hadamard rotation | costs int4 error (0.078 → 0.194 on outlier weights) | **Yes — worth an A/B, expected to fail.** Saves an FWHT per GEMM. I expect the behavioural eval to reject it, which is a useful result: it would show the bar has teeth rather than rubber-stamping every approximation. |
| sparse attention | approximation was off the table | **Yes — see below.** |
| cheaper `exp` | Cephes range reduction is exact-ish and was never questioned | **Yes, conditionally.** Only worth it if softmax is a real share of attention time. |
| MTP | queued behind base performance, per the brief | **No.** Still last. |

### Sparse attention: which kinds are admissible, and why the retention test decides

28 of 35 layers are already sliding-window 512 — sparse in the strongest sense.
Only the **7 full-attention layers** scale with context, so they are the entire
target.

- **StreamingLLM (sinks + local window)** keeps the first few tokens and the
  last W, discarding the middle. It would fail the retention test at mid depths
  by construction. Reject — and note that this is exactly why the retention test
  sweeps depth instead of testing recall once.
- **H2O / heavy-hitter eviction** drops positions whose accumulated attention
  mass is low. Evicted means gone: a fact the query has not yet asked about can
  be evicted before it is needed. Same failure mode, less predictable.
- **Quest-style block top-k** is different in kind. Partition KV into blocks,
  keep per-block elementwise min/max of K, upper-bound each block's best score
  from the query, and attend only to the top blocks. Nothing is evicted — the
  whole cache stays addressable, so a planted fact at any depth remains
  *selectable* whenever the query actually attends to it. That is precisely the
  property the retention test measures, which makes it the one sparse scheme
  whose approximation is aligned with the acceptance criterion rather than at war
  with it.

Estimated ceiling before measuring, so the measurement can contradict it: a
first-principles MAC count puts full-attention layers at ~8.7% of prefill at
8192 and sliding layers at ~2.2%, so block top-k caps out around a 7% prefill
gain at 8192 and much less below. But prefill throughput measured 188 → 98.6
tok/s from 1024 to 8192, a 48% collapse that ~11% of attention cannot explain.
Either the attention kernel runs far below the f32 FMA rate I assumed, or
something else scales with context. **Profile first.** Picking an exotic
attention before knowing which of those is true would be the same mistake as the
1924 GB estimate in §12 — a number that is arithmetically correct and about the
wrong machine.

---

## 14. Attention was the whole problem, and the estimate said otherwise

### The profile that redirected the work

Measured per shape, concurrency 1, g256, idle box — attention as a share of
prefill:

| ctx | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| attention | 5.1% | 17.4% | 24.4% | 30.6% | 38.3% | 49.3% | **61.9%** |

Non-attention time per token is **flat at 3.6–3.7 ms** across that entire
range. The whole 188 → 104 tok/s collapse was attention and nothing else.

My first-principles MAC count had put attention at ~11% at 8192. It was off by
6×, and in the direction that would have caused real damage: I was about to
build block-sparse attention to reduce the *number* of positions, when the
actual problem was that the kernel processed each position at **49.6 G MAC/s —
18.5% of this box's f32 FMA peak**. Sparsity multiplies whatever the per-position
cost is; it does not fix it. Fixing the constant first was worth more and was
much less work.

Where the 81.5% went was visible once the share was known: per 16 elements the
dot product did `load + cvtepi8_epi32 + cvtepi32_ps + fmadd`. **Three of every
four issue slots were format conversion, on operands that were already int8 in
cache.**

### Q·Kᵀ in int8 (VNNI)

`vpdpbusd` does 64 MACs per instruction but wants (u8, i8). K is biased into u8
with `XOR 0x80` — on a two's-complement byte that is exactly +128 — and
corrected with

    sum_i (k_i + 128) q_i  =  sum_i k_i q_i  +  128 * sum_i q_i

where `sum_i q_i` is one scalar per (row, head), computed while quantising q.
**Biasing K rather than Q is what keeps this free of any KV-cache change**: no
stored per-position row sums, no second layout. Four positions per iteration
against four accumulators, combined with an unpack/hadd tree (~13 ops) rather
than four `_mm512_reduce_add_epi32` (~28) — without that the reduction would
have dominated a dot product that VNNI had just made eight times cheaper.

### P·V in int8, which needed a layout change

`out[j] = sum_t w_t v_t[j]` is a scaled accumulate, not a dot product, so it
cannot use an integer dot-product instruction while V is `[position][dim]`. V is
now stored transposed with 4-way interleave:

    v[((slot/4) * row_len + j) * 4 + (slot%4)]

so the four bytes `vpdpbusd` multiplies and adds inside each 32-bit lane are
four consecutive positions of one dim — the reduction over positions falls out
of the instruction. That is the VNNI/AMX B-tile layout. Writing costs a stride-4
scatter once per position per layer; reading is what prefill does O(context)
times, so this is the right side to make awkward.

| prefill tok/s | start | +Q·Kᵀ | +P·V | total |
|---|---|---|---|---|
| 1024 | 185.4 | 195.2 | **219.7** | +18.5% |
| 2048 | 169.6 | 184.0 | **207.7** | +22.5% |
| 4096 | 140.9 | 159.3 | **176.9** | +25.5% |
| 8192 | 104.2 | 124.9 | **140.3** | +34.6% |

Attention share at 8192: 61.9% → 53.4% → **46.2%**. `ffn_gemm` is now the
largest single cost at every context up to 4096.

### The acceptance bar earning its keep

Quantising anything in attention is an approximation, so the question is not
whether the kernel matches a float reference — it cannot, the KV cache is int8
by design — but **how much error it adds on top of the error the design already
accepts**. `bench/kernels/attn_test.c` measures exactly that ratio against an
exact double-precision reference, sweeping query magnitude because score error
is amplified through `exp()` and an error that is harmless on flat attention is
not harmless on sharp attention.

int8 query: **1.21–1.74×** the int8-cache error. Fine.

u8 softmax weights: **2.90×**, and worst on *flat* attention — precisely the
long-context regime this work exists to speed up. Before assuming quantisation,
I checked it was not a bug: a double-precision model of exactly what the kernel
claims to do matched it to **3.4e-08**. So the error was real. Fixed by
splitting the weight across two u8 passes for 16-bit resolution, reusing the
same loaded V vector — two extra `dpbusd` and one extra accumulator per 16 dims,
not a second pass over memory. Ratio returns to 1.21–1.74×.

**Trap 12 — an error number without a baseline is not a measurement.** The first
reading of the int8-query error was "13.8%", taken against an int8-K reference.
That isolates the query's contribution while hiding that K's contribution is the
same order, and it nearly bought a slower int16 path for no reason. The harness
now reports the ratio, so that framing is not available.

### Blocked attention, finished and deleted

The short-context end, which was the one open question:

| ctx | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| blocked | 186.8 | 201.6 | 189.4 | 159.5 |
| per-head | **189.3** | **218.5** | **203.4** | **192.9** |

It loses at every shape from 128 to 8192. At 128 the paths are within 1.3% for
an uninteresting reason — attention is 5.1% of prefill there, so neither can
matter. Deleted rather than carried: P·V needed the V transpose, and carrying a
second V reader through that is real complexity for a path that never wins.

---

## 15. Dual-format weights: the predicted split, measured

`--weights=both` stores an int8 twin of every int4 layer GEMM at `<name>.i8`
(+1.56 GB — an int8 twin is *twice* the int4 bytes, not the same; I had this
at +778 MB and was wrong). `FGM_WEIGHTS=int4|int8|auto:M`, default `auto:16`.

Median of 5 runs per cell, full range in brackets, concurrency 1, idle box.
Ranges are **disjoint in every cell**, which is the bar a claim has to clear
here: one sample per arm cannot resolve anything under ~10% (§16).

Prefill, tok/s:

| prompt | int4 | int8 | int8 gain |
|---|---|---|---|
| 1024 | 216.7 [210–234] | **280.3 [240–284]** | +29.4% |
| 8192 | 150.7 [144–151] | **168.5 [160–170]** | +11.8% |

Decode, tok/s:

| prompt | int4 | int8 | int4 gain |
|---|---|---|---|
| 1024 | **14.2 [14.0–15.0]** | 13.2 [12.8–13.9] | +7.6% |
| 8192 | **13.8 [13.1–14.5]** | 12.2 [11.1–12.8] | +13.1% |

Both effects are *larger* than the single-shot numbers first reported (+12.8%
/ +13.6% prefill, +0.7% / +6.4% decode). Withdrawing them in §16 was right —
they were 0.2–2.9σ from one sample each — and re-measuring restored both with
evidence that actually supports them. Note the direction of the correction: the
noisy measurements **understated** the real effect in three of four cells. Noise
does not only inflate results.

Exactly the split the design predicted, for the reason it predicted: a prefill
GEMM shares one weight read across hundreds of rows and is compute-bound, where
int4's per-group accumulator drain binds; a decode GEMM reads the weight for one
row per sequence and is DRAM-bound, where int4's halved bytes win. The `auto:16`
threshold is the AMX tile height — below it a GEMM cannot fill one tile of rows,
so it is decode-shaped whatever the caller calls it. Batched decode at
concurrency 8 has m=8, lands on int4, and int4 is indeed faster there.

### Trap 13 — loadavg cannot see a competitor that just started

The first run of this comparison was contaminated and I nearly published it. An
orphaned bench from a script killed by a tool timeout overlapped one arm:

| int4, c=1 | contaminated | clean |
|---|---|---|
| 1024 | 99.6 | **208.1** |
| 8192 | 84.9 | **149.6** |

2× low, on one side of a comparison only — the worst possible shape for an
error, because it looks like a result. The loadavg guard passed it because
**loadavg is a one-minute decaying average: a four-thread process one second old
barely moves it.** The guard now scans `/proc/<pid>/comm` for sibling benches
directly and refuses, and re-checks *after* the run — contention that starts
mid-run is invisible to any check made before it. The post-run warning prints to
stdout so it lands in the same log as the numbers it invalidates.

Matched on `comm`, not the command line, deliberately: a command-line match also
hits any watcher whose arguments mention the process it waits for. That is the
inverse bug, and it had already cost twenty minutes — `pgrep -f convert_gemma4`
matched the waiting loop's own command line, so a finished conversion was
reported as still running until I checked the log instead of the process table.


---

## 16. The noise floor, and two claims I have to withdraw

`auto:16` measured 272.3 tok/s prefill at 1024 where pure int8 measured 255.2 —
6.7% apart. That cannot be a real difference: under `auto:16` a 256-row prefill
chunk selects int8 *by construction*, so the two configurations execute
identical code. Something was wrong with the measurement, not the engine.

Five runs of one fixed configuration (int8, concurrency 1, 1024 prompt):

    prefill tok/s   259.3  279.5  270.9  286.7  256.7
    decode  tok/s    12.2   12.5   12.9   13.5   13.4

| | mean | std | full range |
|---|---|---|---|
| prefill | 270.6 | 12.8 (**4.7%**) | 256.7–286.7 (11.1%) |
| decode | 12.90 | 0.56 (**4.4%**) | 12.2–13.5 (10.1%) |

So `auto`'s 272.3 is the mean and pure int8's 255.2 sat at the bottom of the
range. There was no anomaly to explain — only single samples being read as
measurements.

### What survives, in units of the noise it has to clear

| claim | delta | σ |
|---|---|---|
| VNNI attention, 8192 | +34.6% | 7.4 |
| VNNI attention, 1024 | +18.5% | 3.9 |
| blocked attention loses, 1024 | −17.3% | 3.7 |
| int8 vs int4 prefill, 8192 | +13.6% | 2.9 |
| int8 vs int4 prefill, 1024 | +12.8% | 2.7 |
| blocked attention loses, 256 | −7.7% | 1.6 |
| **blocked attention loses, 512** | **−6.9%** | **1.5** |
| **int4 vs int8 decode, 8192** | **+6.4%** | **1.5** |
| **blocked attention loses, 128** | **−1.3%** | **0.3** |

**Withdrawn: "blocked attention loses at every shape from 128 to 8192."** At 128
and 512 the measured deltas are 0.3σ and 1.5σ from one sample per arm — that is
not a result, it is noise with a sign. The *decision* to delete the path still
stands on the 1024 and 2048 columns (3.7σ and ~2.5σ) and on it never once
winning, but the sentence claimed more than the data supports, and at 128 in
particular attention is only 5.1% of prefill so there was nothing there to
measure in the first place.

**Withdrawn, then restored on better evidence: the decode half of the dual-weight
claim.** +0.7% to +6.4% from single samples is 0.2σ to 1.5σ. First principles say
int4 should win decode — it is DRAM-bound and int4 halves the bytes — but "first
principles say so" is what the 1924 GB estimate in §12 and the 11%-attention
estimate in §14 also had going for them. Re-measured as a median of 5 with
disjoint ranges it holds at **+7.6% and +13.1%** — larger than first reported.
The lesson is not "the claim was wrong" but "the method could not tell", and a
method that cannot tell understates as often as it overstates: three of the four
cells came back bigger, not smaller.

### The fix

`FGM_REPEAT=n` runs each matrix cell n times and reports **median with full
range**, median because a contended run is an outlier rather than a shifted
sample, and the range printed so a reader can see whether a delta clears it.
`ab_weights.sh` alternates arms instead of running all of one then all of the
other, so drift over a long run is shared between arms rather than landing
entirely on whichever goes last.

**Trap 14 — a difference smaller than the noise floor is not a small result, it
is no result.** Every A/B in this project before this section was one sample per
arm. The ones that survive do so because they are large, not because the method
was sound.


---

## 17. The behavioural gate, finally run

Everything above this section was speed work validated only against
kernel-level error ratios. The gate that decides whether the engine still
*works* had never been executed. It has now.

25 requests against 12 tools x 4 parameters, prompts built in Gemma 4's real
declaration format (1628 tokens each), scored on whether the engine picks the
right tool and fills the right arguments.

| metric | unconstrained | grammar-constrained |
|---|---|---|
| well-formed call | 25/25 | 25/25 |
| correct tool | 25/25 | 25/25 |
| all arguments right | 25/25 | 25/25 |
| argument-level | 100/100 | 100/100 |

That is the whole accumulated approximation stack passing at once: int4
group-256 weights, int8 activations, int8 query quantisation, 16-bit softmax
weights, Hadamard rotation, and an attention kernel rewritten twice.

**The constraint contributes no measurable accuracy here.** Unconstrained
already scores 100%, so the grammar's value is a worst-case structural
guarantee rather than an average-case gain — plus 275 of 940 decode steps
(29%) whose LM-head read is skipped outright, which is a *speed* win. Saying
the constraint is what delivers exactness would be backwards on this evidence.
What it buys is that the 26th request cannot emit something unparseable, which
is a different and still worthwhile property.

**And 25/25 is a floor, not a precision instrument.** A regression to 95%
accuracy would still show 25/25 about 28% of the time (0.95^25 = 0.28). This
gate detects breakage, not drift. Its job starts now: any drop from 100% on a
future change is signal, and that is exactly what it is for when AMX attention
or sparsity lands.

### Long-context retention at 8192

A fact planted at seven depths in an 8184-token context, then asked for at the
end. Depth is swept because the failure modes are positional: a ring bug loses
the oldest positions, a sliding-window bug loses everything outside the last
window, a chunk-boundary bug loses whatever landed on a multiple of the prefill
chunk.

| depth | 0.02 | 0.15 | 0.35 | 0.50 | 0.65 | 0.85 | 0.98 |
|---|---|---|---|---|---|---|---|
| recalled | yes | yes | yes | yes | yes | yes | yes |

**7/7**, each answering with the exact planted code and nothing else. The 0.02
and 0.98 ends matter most: those are the ring-buffer wrap and the
just-before-the-question positions, the two places the KV bugs of §6b actually
lived.

This is also the measurement that would have killed StreamingLLM-style sink
attention and H2O eviction had they been built — both discard the middle of the
context, and depths 0.35–0.65 are exactly the middle. §13 predicted that; this
is the instrument that would have enforced it.

**Trap 15 — a gate everything passes tells you nothing until something fails.**
Worth stating because the temptation after a 100% result is to treat it as
proof of quality rather than as the absence of catastrophe. It is the second.


---

## 18. Prefix sharing, measured

The target profile is 8 concurrent requests carrying the same 12-tool
declaration block, so re-prefilling it per sequence is pure waste. Prefill the
shared span once, then copy the cache into every other sequence
(`KvCache::fork_from`).

8 sequences, 2048 shared + 6144 unique, chunk 256, g256:

| | tokens | time | effective |
|---|---|---|---|
| no sharing | 65536 | 470.83 s | 139.2 tok/s |
| sharing | 65536 | 395.26 s | **165.8 tok/s** |

**1.19×**, from 14336 prefill tokens avoided. Breakdown: 10.97 s of shared
prefill, **462 ms** to fork 238 MB into seven caches, 383.82 s of unique tails.

Note the gap between token count and time: 21.9% of the tokens disappear but
only 16% of the time does, because the avoided tokens are the *cheapest* — the
shared span sits at positions 0–2048 where attention has the least history to
scan. An estimate based on token count alone would have over-promised by a
third. (My pre-measurement estimate said ~1.14×, which was low for the same
reason in the other direction: it ignored that the tails also start from a
warm cache.)

**Correctness: `next-token identity after fork: 8/8`.** The fork is a copy, not
a recomputation, so nothing about float evaluation order changes and
bit-identity is the right check here — unlike anything that reorders
accumulation, where it is not.

### Trap 16 — my own guard cried wolf

The run printed `*** NUMBERS ABOVE MAY BE SUSPECT: load average rose 0.95 ->
3.76 on 4 cores ***`. It was wrong. The benchmark runs four busy threads, so it
drives loadavg to roughly ncpu *by itself*; any post-run threshold that catches
a real competitor also catches the bench measuring its own load. The sibling
scan — which names PIDs — stayed correctly silent, and the numbers were fine.

Removed rather than tuned, because there is no threshold that separates
self-load from competitor-load with this signal. **A guard that cries wolf is
worse than no guard**: the next real warning gets read as noise, which is
exactly the failure the guard exists to prevent.


---

## 19. The attention ceiling probe, and why AMX is the wrong tool

Before starting an AMX attention rewrite I measured what the kernel actually
achieves, because the previous two times I picked an attention optimisation from
a first-principles estimate I was wrong: once by assuming KV re-reads hit DRAM
when they were L2-resident (§12), once by putting attention at 11% of prefill
when it was 61.9% (§14).

`bench/kernels/attn_bench.c`, 16 query rows against a full context, 8 heads:

| head_dim | ctx | G MAC/s | % VNNI peak |
|---|---|---|---|
| 256 | 512 | 32.5 | 5.6% |
| 256 | 1024 | 36.7 | 6.3% |
| 256 | 2048 | 33.4 | 5.8% |
| 256 | 4096 | 22.5 | 3.9% |
| 256 | 8192 | 18.7 | 3.2% |
| 512 | 512 | 39.5 | 6.8% |
| 512 | 1024 | 41.3 | 7.1% |
| 512 | 2048 | 27.8 | 4.8% |
| 512 | 4096 | 21.1 | 3.6% |
| 512 | 8192 | 21.1 | 3.6% |

**AMX is the wrong tool, and the reason is arithmetic intensity.** With one KV
head every MAC consumes exactly one byte of K or V — intensity is *1 MAC per
byte*. That puts a hard ceiling of 33 G MAC/s from DRAM and 209 G MAC/s from L2,
regardless of how fast the multiplier is. AMX's 12.7× advantage over VNNI is an
advantage in **instructions**, and instructions are not what is scarce here: the
kernel runs at 3–7% of the VNNI ceiling it already has. Building AMX attention
would have been days of work against a ceiling that is not binding.

*Caveat on the probe, stated because it matters:* for this model `kv_heads *
head_dim == head_dim`, so bytes-touched and MACs are numerically identical and
the GB/s column is not independent evidence — it is the MAC column relabelled.
The conclusion rests on the %VNNI column and on the intensity argument, not on
two agreeing measurements.

### What the numbers actually point at

The kernel is neither instruction-bound (3–7% of VNNI) nor cleanly
bandwidth-bound at short context (39.5 GB/s against L2's 209). At 8192 it sits
at 21 G MAC/s, at or below the 33 G MAC/s DRAM ceiling — because K and V for one
full-attention layer are 4 MB each at that length and no longer fit L2, so the
eight per-head passes over the same range stop hitting cache.

So the levers are, in order:

1. **Fewer bytes.** int4 KV halves them. Block top-k sparsity cuts the number of
   positions read. Both attack the binding constraint directly.
2. **Less per-(row, head) overhead.** The u8 weight fill in P·V is a *scalar*
   loop over the whole score range — at ctx 512 it costs roughly as much as the
   entire Q·Kᵀ dot product it feeds. Vectorising it is cheap and clear.
3. **Fewer re-reads.** Eight query heads re-stream the same K range because the
   loop is `for head { for position }`. Inverting that is what blocked attention
   attempted; it lost for other reasons (§12), but the traffic argument was
   never the flawed part.

**Trap 17 — a speedup ratio is only as real as the ceiling it is measured
against.** "AMX is 12.7× VNNI" is true and was nearly decisive. It is also
irrelevant to a kernel that uses 5% of VNNI, and nothing about the ratio itself
says so. The number that mattered was one nobody quotes: bytes per MAC.


---

## 20. The weight-fill vectorisation: no end-to-end effect, and why

The attention ceiling probe (§19) found the P·V u8 weight fill running scalar,
costing about as much as the Q·Kᵀ it feeds. Vectorising it gave 6–40% in the
kernel microbenchmark. End to end it gives **nothing**.

Alternating arms inside one measurement window, 8192-token prefill, seconds:

| pass | vector ttft | scalar ttft | vector gen | scalar gen |
|---|---|---|---|---|
| 1 | 55.59 | 55.63 | 4.92 | 5.69 |
| 2 | 53.58 | 53.35 | 4.70 | 4.64 |
| 3 | 53.65 | 54.03 | 4.79 | 4.78 |
| median | **53.65** | **54.03** | **4.79** | **4.78** |

0.7% on prefill and 0.2% on decode — noise, in both directions.

**Why the kernel gain does not reach the model: 28 of 35 layers are sliding
with ring buffers, and the ring path is deliberately still scalar** (slot
indices jump at the wrap, so a vector store would need a scatter). Only the 7
full-attention layers take the vectorised branch — and the sliding layers'
fill was cheap anyway, since their range is capped at the 512-token window.
The microbenchmark measured 16 rows against a full contiguous context, which
is the full-attention shape exclusively. It answered a question the model does
not ask.

Kept rather than reverted: it is measurably faster in the kernel, measurably
neutral in the model, and carries no accuracy cost (error harness unchanged at
1.21–1.74×). But it is not a win, and recording it as one would be false.

### And the "regression" was the platform

The run that triggered all this read 160.1 tok/s prefill and 9.6 tok/s decode
against a 216.7 / 14.2 baseline. Median ttft in the controlled test is 53.65 s
at 8192 = **152.7 tok/s**, against the 150.7 baseline. There was never a
regression. The AMX tile-state corruption rate had drifted (5/12 → 7/12 trials
at warm-up; 2884 and 10182 guard retries logged in one run), and every
corrupted tile re-runs a whole GEMM block.

**Trap 18 — when the noise source is a platform defect, cross-run A/B is not
weak evidence, it is no evidence.** §16 established that one sample per arm
cannot resolve under ~10% here. This is worse: the confound is *not* symmetric
in time, so more repetitions of a cross-run comparison converge on the wrong
answer rather than on the right one with wider error bars. The only fix is to
put both arms in one binary and alternate them, so the drift is common-mode.
That is now how `FGM_SCALAR_WFILL` exists.


---

## 21. The target profile, end to end

The brief: 8 concurrent requests, ~8k prompt in, ~2k out, 12 tools x 4 params.
Run with everything current engaged — dual-format weights on `auto:16`, prefix
sharing over a 2048-token shared declaration block, VNNI int8 attention.

| | measured |
|---|---|
| prefill | 65536 tok in 342.07 s → **191.6 tok/s aggregate** |
| prefix sharing | 14336 prefill tokens avoided, 422 ms fork |
| decode | **46.8 tok/s aggregate** (170.9 ms/step, 5.9 tok/s/seq) |
| TTFT | first **70.1 s**, last 342.1 s |
| KV | **332 MB total, 41.4 MB/seq** at 10248 context |
| full request set | ~692 s extrapolated → **118 tok/s overall** |

81920 tokens of work (8 × 10240) in ~692 s. Decode was capped at 60 s and
extrapolated from 352 measured steps, which is stated rather than hidden: the
decode rate is measured, the total is arithmetic.

**This is a pessimistic reading.** The warm-up reported *8 of 12* tile-state
trials corrupted, the worst seen in this project (earlier runs: 5/12, 6/12,
7/12). Every corrupted tile re-runs a whole GEMM block, so this run carried
more guard overhead than any baseline it might be compared against.

### What it says about the shape of the problem

TTFT spread is 70 s to 342 s — the eighth request waits for all seven ahead of
it, because prefill is processed sequentially per sequence. That is the single
most user-visible number here and it is not a throughput problem: aggregate
prefill is 191.6 tok/s either way. Interleaving prefill chunks across sequences
would flatten TTFT without changing throughput at all, and is the obvious next
serving-layer change — it was never on the optimisation list because the list
was about tok/s, and TTFT is not tok/s.

Decode at 46.8 tok/s aggregate against 67.3 measured at 512 context earlier is
the cost of an 8-10k KV set: attention grows with context while the weight read
does not, so the batching win erodes as the context lengthens.


---

## 22. int4 group 512: the gate bites, and the constraint earns its keep

Group 512 halves the number of int32 accumulator drains the int4 kernel must
perform, at the cost of coarser scales. Converted: 2.77 GB, weight rel err
**0.1054** against g256's 0.1015 (+3.9%).

### Speed

Alternating the two models across three passes (Trap 18 — this box's AMX
corruption drift swamps effects of this size when arms are compared across
separate runs). 8192-token prefill:

| pass | g256 ttft | g512 ttft |
|---|---|---|
| 1 | 225.43 s (cold page cache) | 51.40 s |
| 2 | 53.69 s | 50.11 s |
| 3 | 54.17 s | 49.49 s |
| warm median | 53.9 s = 151.9 tok/s | **50.1 s = 163.5 tok/s** |

**+7.6% prefill**, decode unchanged at 13.3 tok/s. Not the +38% the standalone
GEMM benchmark predicted, and the reason is the same dilution as always:
`ffn_gemm` is ~40% of prefill at 8192, so a GEMM-local gain arrives at the
model divided by its share.

### The gate rejects it — the first time it has rejected anything

Unconstrained tool calling on g512: **24/25**. Retention held at 7/7, isolating
the regression to tool selection rather than context handling. The failing case:

    want: search_flights{origin: LHR, destination: JFK,
                         depart_date: 2026-08-03, passengers: 1}
    got:  "What is the IATA code for the destination airport (JFK) and the
           number of passengers? Also, what is the year for the departure date?"

The model declined to call the tool and asked a clarifying question instead.

This retroactively validates §17's own caveat. I wrote there that "a gate
everything passes tells you nothing until something fails" and that its job
would start when something did. Something did, on the first optimisation
submitted to it afterwards.

### And the constraint rescues it — overturning a claim I made two sections ago

Constrained tool calling on g512: **25/25, 100/100 argument-level.**

In §17 I wrote that the grammar constraint "contributes no measurable accuracy"
because unconstrained already scored 100%, and that its value was purely a
worst-case structural guarantee. That was true *for g256*, and I generalised it
one model too far. The g512 failure is exactly the worst case the constraint
exists for: the model emitting no call at all. Constraining forces one at that
position, and the arguments it then fills are correct.

**So the constraint is what makes the faster weights viable.** Correct
statement: on a model that does not need rescuing, the constraint buys only
structure and the 29% LM-head skip; on a model that does, it buys accuracy
outright. "No measurable benefit" was a measurement on one model, not a
property of constrained decoding.

### Verdict

g512 ships **conditionally**: +7.6% prefill, free *when tool calls are
grammar-constrained*, which the target profile always is (12 tools × 4 params
per request). Unconstrained serving keeps g256, because against a brief that
weights tool-call exactness equally with speed, 7.6% on one phase does not buy
a lost tool call.


---

## 23. llama.cpp, re-measured on an idle box

Parked earlier at the user's instruction ("stop the benchmark against llamacpp
until we finish all up on our side; don't say they are worse because you were
running a full bench at the same time"). That instruction was correct: the
original comparison was taken while a full benchmark ran on the same 4 cores.

Same box, 4 threads, same bit width — Google's own QAT Q4_0 GGUF (build
91f8c9c) against our int4 group-256. Engines alternated within the run.

| | fastgemma (median of 2) | llama.cpp | |
|---|---|---|---|
| prefill 512 | **208.5** [199.9–217.0] | 133.6 ± 13.3 | **+56%** |
| prefill 2048 | **192.7** [184.9–200.5] | 114.1 ± 2.7 | **+69%** |
| decode (tg64) | 13.6 [11.6–14.5] | 14.9–16.9 | **llama.cpp +10–24%** |

A reversal of the original reading, where llama.cpp led prefill at 512 (158.5
vs our 142.9). The attention rewrite is what moved it: prefill at 512 went
142.9 → 208.5 over this session.

### Three caveats that belong next to the numbers

**Decode still favours llama.cpp single-stream**, exactly as it did before. At
batch 1 decode is pure memory bandwidth, AMX has nothing to amortise, and their
Q4_0 kernels are extremely well tuned. Our decode advantage exists only under
batching — 46.8 tok/s aggregate at concurrency 8 in the target profile — and
claiming a decode win without that qualifier would be false.

**This is a speed comparison and nothing else.** llama.cpp runs a
quantisation-aware *trained* checkpoint; ours is post-training quantised from
bf16. Those are different accuracy starting points and no behavioural
comparison between the two engines exists. A throughput ratio does not license
a statement about which engine is better.

**Only fastgemma uses AMX**, and this box's tile-state corruption rate drifts
between runs. That shows up directly in the spread: our pp512 varies 199.9–217.0
(and hit 230.1/171.4 in an earlier pair, a 26% swing) against llama.cpp's
±13.3. Engines alternate for this reason, and ranges are quoted rather than
points.

### Trap 19 — a parser bug looks exactly like the other side losing

The first corrected run printed four populated fastgemma rows next to four
empty llama.cpp rows. That reads as the competitor failing to execute. It was
my `-o csv` column assumption not matching llama-bench's layout; the second
attempt then read the test name instead of the throughput because `-F"|"`
produces a leading empty field and I counted from the wrong end.

Both failures produce output that is *shaped* like a result and that favours
the side still reporting numbers — which was mine, both times. This belongs
with the contamination traps rather than with ordinary bugs: the danger is not
that it breaks, it is that it does not look broken.
