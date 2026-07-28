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
| Flash-tiled attention + AMX-INT8 Q·Kᵀ | **open** | biggest *prefill* win: 42% collapse 128→8192 is attention |
| Multithread the GEMM preamble (FWHT/quant/pack) | **open** | ~half of "ffn_gemm" time is not GEMM; ~20-30% of prefill |
| K-blocking for L1-resident AMX operands | **open** | GEMM 3.7 of ~11.8 achievable TOPS |
| BF16 attention | **rejected** | KV is already int8 — bf16 *doubles* KV bandwidth, and AMX-BF16 is half AMX-INT8 (7.0 vs 14.69 TOPS). Only P·V is a genuine candidate |
| Dual-format weights (int8 prefill / int4 decode) | **open** | prefill is compute-bound, decode DRAM-bound |
| Prefix radix cache (shared tool definitions) | **open** | 12 tools × 4 params is identical across the 8 requests |
| Chunked prefill + decode piggyback | **open** | amortise weight reads across both phases |
| Grammar-constrained tool decoding | **done** | 1011-state DFA, 33 MB mask, 36% of steps forced |
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

DFA over the tool schemas → per-state token bitset. On the real 262144-token
vocab, 12 tools × 4 params: **1011 states, 3.4 s one-time compile, 33.1 MB of
mask**, walk accepted in 47 steps emitting

    {"name": "tool_0", "arguments": {"p0": " ", "p1": 1.1, "p2": true, "p3": 1}}

which parses, with the right tool and 4/4 args. Structural validity is a
guarantee, not a hope.

**36% of steps admit exactly one token.** Those steps can skip the LM head
entirely — a 201 MB int4 read whose argmax is predetermined. Wiring that up is
the next decode win for tool-heavy traffic.

**Trap 8 — special tokens have literal byte forms.** `<pad>` decodes to the five
bytes `<pad>`, which satisfy the JSON string-body class, so the first
constrained walk emitted `{"p0": "<pad><pad><pad>…` forever *without ever
violating the grammar*. Added/special tokens are now returned as empty byte
strings, making them permanently illegal.

**Trap 9 — a weak checker confirms whatever you believe.** The number rule was
"digits and dots", which accepts `..` and the empty string; the walk emitted
`"p1": ..` and my bracket-counting "is valid JSON" check passed it. Replaced
with a real JSON number DFA and a real `serde_json` parse. If the validity check
is weaker than the property being claimed, it is not evidence.
