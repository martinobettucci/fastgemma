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
| K-blocking for L1-resident AMX operands | **open** | biggest known kernel win, see Kernel §4 |
| Dual-format weights (int8 prefill / int4 decode) | **open** | prefill is compute-bound, decode DRAM-bound |
| Prefix radix cache (shared tool definitions) | **open** | 12 tools × 4 params is identical across the 8 requests |
| Chunked prefill + decode piggyback | **open** | amortise weight reads across both phases |
| Grammar-constrained tool decoding | **open** | the exactness half of the brief |
| Activation-sparsity skipping | **n/a** | Gemma 3n had 95% sparsity; **Gemma 4 does not** |
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
