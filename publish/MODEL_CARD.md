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
  - fastgemma
library_name: fastgemma
pipeline_tag: text-generation
---

# fastgemma — Gemma 4 E2B, CPU-only, AMX INT8

Weights for [fastgemma](https://github.com/martinobettucci/fastgemma), a CPU-only
inference engine for Gemma 4 built around Intel **AMX INT8** tile matrix units.

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

**Requires AMX-INT8** — Sapphire Rapids or newer Xeon. There is no fallback
path; the engine asserts on `XTILEDATA` permission at startup.

Reference machine for every number here: Intel Xeon (Sapphire Rapids class),
**4 cores @ 2.1 GHz**, 15 GB RAM, no GPU. Measured ceilings: AMX INT8 14.69
TOPS, AVX512-VNNI 1.16 TOPS, DRAM 33 GB/s.

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
build 91f8c9c) against fastgemma's int4 group-256. Engines alternated within
the run so platform drift is shared rather than landing on one side.

| | fastgemma | llama.cpp | |
|---|---|---|---|
| prefill 512 | **208.5** [199.9–217.0] | 133.6 ± 13.3 | **+56%** |
| prefill 2048 | **192.7** [184.9–200.5] | 114.1 ± 2.7 | **+69%** |
| decode (tg64, batch 1) | 13.6 [11.6–14.5] | **14.9–16.9** | llama.cpp +10–24% |

Three caveats that belong with these numbers, not under them:

- **Decode favours llama.cpp single-stream.** At batch 1 decode is pure memory
  bandwidth, AMX has nothing to amortise, and their Q4_0 kernels are very well
  tuned. fastgemma's decode advantage appears only under batching — 46.8 tok/s
  aggregate at concurrency 8 — and claiming a decode win without that qualifier
  would be false.
- **This is a speed comparison only.** llama.cpp runs a quantisation-aware
  *trained* checkpoint; these weights are post-training quantised from bf16.
  Different accuracy starting points, and no behavioural comparison between the
  two engines exists. A throughput ratio does not license a claim about which
  engine is better.
- **Only fastgemma uses AMX**, and this reference host's tile-state corruption
  rate drifts between runs — visible as the wider fastgemma spread. Ranges are
  quoted for that reason.

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
