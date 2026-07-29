---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
  - gemma
  - gemma4
  - cpu
  - quantized
  - int4
  - int8
  - amx
  - avx512
  - fastgemma
library_name: fastgemma
pipeline_tag: text-generation
---

# fastgemma — Gemma 4 E2B, CPU-only, AMX INT8 / AVX-512 VNNI

Weights for [fastgemma](https://github.com/martinobettucci/fastgemma), a CPU-only
inference engine for Gemma 4 built around Intel **AMX INT8** tile matrix units,
with an **AVX-512 VNNI** path for hosts without them.

These are **not** safetensors and will not load in `transformers`. The `.fgm`
format stores weights already quantised, Hadamard-rotated and pre-packed into
AMX tile order, so the engine mmaps the file and executes with zero repacking —
a 4.34 GB file maps in under a millisecond.

## Licence

These weights are a derivative of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) and are
distributed under the **Gemma Terms of Use**. Use is subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
Quantising, rotating and repacking the tensors does not change that: the
restrictions travel with the derivative.

The fastgemma *engine* is separate and carries its own licence in its source
repository. The licence on this repo governs the weights.

## What is in the file

`g4e2b-dual.fgm`, 4.34 GB, containing both quantisations of every FFN weight so
the runtime can choose per GEMM:

| tensor class | format | why |
|---|---|---|
| FFN gate/up/down | int4, group 256 **+ int8 twin** | 84% of compute params |
| attention q/k/v/o | int8 per-channel | small, accuracy-critical |
| embed / LM head | int4, group 256 | 403 M params |
| PLE table | int4, group 64 | 2.35 B params, pure gather |
| norms, scalars | f32 | negligible |

Every linear weight is **Hadamard-rotated along K**. `H` is orthogonal, so
`W' = W Hᵀ` plus a runtime fast Walsh–Hadamard transform of the activation is
mathematically identical, but both weight groups and activation rows lose their
outliers — measured on outlier-heavy weights this takes int4 relative error from
**0.194 → 0.078**.

Carrying both formats costs +1.56 GB (an int8 twin is *twice* the int4 bytes)
and is what makes the prefill/decode split below possible.

## Hardware requirement

**Minimum: AVX-512 with VNNI** (Skylake-SP with VNNI / Cascade Lake / Ice Lake
and newer). **Recommended: AMX-INT8** (Sapphire Rapids or newer Xeon).

The engine picks its GEMM backend from CPUID at startup and prints which one it
took. Both read the same `.fgm` file — the weight layout is AMX tile order,
which *is* VNNI operand order chunked sixteen rows at a time, so there is no
second file and no runtime repacking on either path. `FGM_BACKEND=amx|vnni`
forces the choice on a host that has both.

The two backends are not close in speed, and the numbers below are reported
per backend rather than blended. AMX INT8 is 12.7× AVX512-VNNI on the same
silicon (14.69 vs 1.16 TOPS measured); expect prefill to fall by roughly half
without it, and single-stream decode — which is memory-bound, not
multiply-bound — to move much less.

Reference machines. **AMX numbers:** Intel Xeon (Sapphire Rapids class), 4
cores @ 2.1 GHz, 15 GB RAM. Measured ceilings AMX INT8 14.69 TOPS, AVX512-VNNI
1.16 TOPS, DRAM 33 GB/s. **AVX-512 numbers:** Intel Xeon (Cascade Lake class,
no AMX), 4 cores @ 2.8 GHz, 16 GB RAM, 1 MB L2 per core.

## Measured performance

All figures on an idle machine, 4 threads. Single-sequence, median of 5 runs
with the full range in brackets — the run-to-run noise floor on this box is
4.7% on prefill and 4.4% on decode, so ranges are quoted rather than points.

There is also an int4 **group-512** build (`g4e2b-g512.fgm`, 2.77 GB) giving
+7.6% prefill. It is shipped conditionally: unconstrained tool calling drops to
24/25 on the eval below, while grammar-constrained decoding restores 25/25. Use
it only when tool calls are constrained.

**Prefill, tok/s** (int8 weights, selected automatically for ≥16 rows):

| prompt | tok/s |
|---|---|
| 1024 | **280.3** [240–284] |
| 8192 | **168.5** [160–170] |

**Decode, tok/s** (int4 weights, selected automatically below 16 rows):

| context | tok/s |
|---|---|
| 1024 | **14.2** [14.0–15.0] |
| 8192 | **13.8** [13.1–14.5] |

**Target serving profile — 8 concurrent, 8192 in / 2048 out**, with a
2048-token shared tool-declaration prefix:

| | |
|---|---|
| prefill | 65536 tok in 342.1 s → **191.6 tok/s aggregate** |
| decode | **46.8 tok/s aggregate** (5.9 tok/s/seq) |
| TTFT | first 70.1 s, last 342.1 s |
| KV cache | **332 MB total**, 41.4 MB/seq at 10248 context |

Decode is measured over 352 steps and extrapolated to 2048; the rate is
measured, the total is arithmetic.

## vs llama.cpp

Same box, 4 threads, same bit width: Google's own QAT Q4_0 GGUF (llama.cpp
build 91f8c9c) against fastgemma. Engines alternated within each run so
platform drift is shared rather than landing on one side. Median of 3, full
range in brackets.

### On AMX — int4 group-256 both sides, no speculative decoding

| | fastgemma | llama.cpp | |
|---|---|---|---|
| prefill 512 | **208.5** [199.9–217.0] | 133.6 ± 13.3 | **+56%** |
| prefill 2048 | **192.7** [184.9–200.5] | 114.1 ± 2.7 | **+69%** |
| decode (tg64, batch 1) | 13.6 [11.6–14.5] | **14.9–16.9** | llama.cpp +10–24% |

### On AVX-512 VNNI — no AMX on either side

Both binaries target the same ISA class here: llama.cpp built with
`GGML_NATIVE=OFF` and an explicit AVX512+VNNI feature set, fastgemma on its
VNNI backend.

| | fastgemma int4 | fastgemma dual | llama.cpp | best |
|---|---|---|---|---|
| prefill 512 | 108.9 [108.1–110.8] | **124.6** [121.9–128.7] | 68.6 [68.3–69.0] | **+82%** |
| prefill 2048 | 96.2 [92.3–98.5] | **113.9** [106.1–115.9] | 60.8 [60.1–61.1] | **+87%** |
| decode (tg64, batch 1) | 7.9 | 7.4 | **12.3** [11.0–12.6] | llama.cpp +66% |

**The prefill margin is the same order with and without AMX** (+56/+69% there,
+82/+87% here), which is the useful thing this table says: the advantage comes
from the int8 activation path, the pre-packed weights and the absence of any
runtime repack — not from the tile units. AMX makes both sides' absolute
numbers larger; it is not where the ratio comes from.

### Speculative decoding, AVX-512 backend

Decode above is raw single-stream with no speculation, which is what
`llama-bench` measures. On the actual tool-calling workload — 25 requests,
grammar-constrained, decode timed separately from prefill — the engine's two
weight-free multi-token paths give:

| | decode tok/s | |
|---|---|---|
| baseline | 7.87 [7.83–8.15] | |
| grammar-forced batching | 8.41 [8.37–8.57] | +1.6% |
| prompt-lookup, k=4 | 9.06 [8.67–9.18] | +15.1% |
| **both** | **9.19** [9.08–9.22] | **+16.8%** |

All 25 generated sequences are **byte-identical to baseline** in every arm.
Grammar batching is exact by construction — a token the DFA forces is
determined without consulting the model — and prompt-lookup drafts are verified
against the model before acceptance (193/956 drafted tokens accepted, 20%).
Neither trades accuracy for speed.

This is not comparable to the `llama-bench` decode column: that measures free
generation from synthetic tokens, where prompt-lookup has nothing to copy. It
is reported here because it is the number that applies to the workload this
engine was built for. Even with it, llama.cpp is ahead on single-stream decode.

Three caveats that belong with these numbers, not under them:

- **Decode favours llama.cpp single-stream.** At batch 1 decode is pure memory
  bandwidth, there is nothing for a matrix unit to amortise, and their Q4_0
  kernels are very well tuned. fastgemma's decode advantage appears only under
  batching — 46.8 tok/s aggregate at concurrency 8 on AMX — and claiming a
  decode win without that qualifier would be false.
- **This is a speed comparison only.** llama.cpp runs a quantisation-aware
  *trained* checkpoint; these weights are post-training quantised from bf16.
  Different accuracy starting points, and no behavioural comparison between the
  two engines exists. A throughput ratio does not license a claim about which
  engine is better.
- **AMX and AVX-512 rows are from different machines** at different clocks and
  are not comparable to each other. Compare within a table, never across.

## Why AMX for GEMM but not for attention

AMX INT8 is 12.7× AVX512-VNNI on this box (14.69 vs 1.16 TOPS), and the engine
uses it for every weight GEMM. It is deliberately **not** used for attention,
and the reason is arithmetic intensity rather than any property of the
instruction:

| | MACs per byte moved | binding ceiling |
|---|---|---|
| weight GEMM, 256-row prefill chunk | 256 | compute — AMX wins |
| attention, 1 KV head | **1** | bandwidth — 33 G MAC/s (DRAM), 209 (L2) |

Every K or V byte in attention feeds exactly one multiply-accumulate and is
then done with, so N MACs require N bytes. Measured, the attention kernel runs
at 18–41 G MAC/s — that is 3–7% of the VNNI ceiling it already has, i.e. the
multiplier is 93–97% idle. Making an idle multiplier 12.7× faster buys nothing.
A weight GEMM reuses each byte across all M rows, which is why the same
instruction is transformative there.

## Accuracy

The engine trades numerical precision for speed deliberately, so it does not
match a reference implementation's logits and never will. It is graded
behaviourally instead.

**Tool calling** — 25 requests against 12 tools × 4 parameters, prompts in
Gemma 4's native declaration format:

| metric | unconstrained | grammar-constrained |
|---|---|---|
| well-formed call | 25/25 | 25/25 |
| correct tool | 25/25 | 25/25 |
| all arguments correct | 25/25 | 25/25 |
| argument-level | 100/100 | 100/100 |

**Long-context retention** — a fact planted at seven depths in an 8184-token
context and asked for at the end: **7/7**, depths 0.02 through 0.98.

Caveat worth stating: 25/25 has wide bounds. A regression to 95% accuracy would
still read 25/25 about 28% of the time. This is a floor that detects breakage,
not an instrument that resolves drift.

## Tool calling

Gemma 4 does **not** emit JSON tool calls. The native syntax is

```
<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>,country:<|"|>JP<|"|>}<tool_call|>
```

where `<|tool_call>`, `<tool_call|>` and `<|"|>` are single vocabulary tokens
(48, 49, 52) and keys are bare. The engine ships a DFA-based constrained decoder
for this syntax: 12 tools × 4 params compiles to 824 states, and **29% of decode
steps admit exactly one token**, letting those steps skip the 201 MB int4
LM-head read entirely.

## Usage

```sh
git clone https://github.com/martinobettucci/fastgemma && cd fastgemma
cargo build --release
huggingface-cli download P2Enjoy/fastgemma-gemma-4-E2B g4e2b-dual.fgm --local-dir models/
./target/release/fgm-bench serve models/g4e2b-dual.fgm
```

`FGM_WEIGHTS=int4|int8|auto:M` selects the quantisation (default `auto:16`, the
AMX tile height — below 16 rows a GEMM cannot fill one tile of rows, so it is
decode-shaped whatever the caller calls it).

## What was tried and rejected

Recorded because negative results are the expensive half of this work, and
because two of these look attractive on paper:

| change | result |
|---|---|
| **int4 KV cache** | **Rejected.** 0/7 retention at 8k, tool calling collapses into repetition loops. Weight precision and KV precision are not interchangeable: a weight error perturbs one matmul, a KV error perturbs every future score against that position, and softmax exponentiates it. |
| **Flash-style blocked attention** | Rejected. Loses at every context from 128 to 8192 (−1.3% to −17.3%). With 1 KV head the int8 K set is L2-resident at the contexts served, so there was no traffic to save and the online-softmax bookkeeping was pure cost. |
| **AMX for attention** | Rejected without building. The kernel runs at 3–7% of the VNNI ceiling it already has, so instructions are not scarce — see the section above. |
| **int4 group 512** | Conditionally accepted, +7.6% prefill. Ships only with constrained decoding. |

A **head-batched** attention kernel (one K line serving all 8 query heads,
raising intensity from 1 to 8 MACs/byte) measures **2.59× on Q·Kᵀ alone** at
head_dim 512 / ctx 8192, with a crossover exactly on the 2 MB L2 line. It is
integrated but **not yet verified end to end** — the benchmark host lost its
AMX units before that could be measured, so it is not reflected in any number
on this card.

## Known limitations

- **AMX required.** No AVX-512-only or ARM fallback.
- **TTFT scales with concurrency.** Prefill is sequential per sequence, so the
  eighth of eight requests waits 342 s while the first waits 70 s. Aggregate
  throughput is unaffected; interleaving prefill across sequences would flatten
  this and is not yet implemented.
- **Tile-state defect on some hosts.** This reference machine loses AMX tile
  state across context switches (5–8 of 12 warm-up trials corrupted, varying by
  run). The engine detects this at startup and enables a sentinel-based
  detect-and-retry guard costing 1–2%. Correctness is unaffected; throughput on
  such hosts varies by up to 25% run to run.
- E4B is not converted.
