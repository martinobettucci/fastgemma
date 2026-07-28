# fastgemma

An attempt to write a super fast CPU-only **Gemma 4** inference server.

Built for one specific serving profile:

> 8 concurrent requests · ~8k prompt tokens in · ~2k tokens out ·
> ~12 tools × 4 parameters per request · tool-call exactness weighted equally
> with speed.

Everything here is measured on the target machine, not extrapolated. Numbers in
commit messages are reproducible with the harnesses under `bench/`.

## Target machine

Intel Xeon (Sapphire Rapids class), 4 cores @ 2.1 GHz, 15 GB RAM.
Measured ceilings (`bench/roofline`):

| | measured |
|---|---|
| AMX INT8 (`TDPBSSD`) | **14.69 TOPS** |
| AMX BF16 | 7.00 TFLOPS |
| AVX512-VNNI | 1.16 TOPS |
| L1 read | 393 GB/s |
| L2 read | 209 GB/s |
| L3 read (260 MB) | 58 GB/s |
| DRAM read | 33 GB/s |

**AMX is 12.7× AVX512-VNNI.** llama.cpp and ONNX Runtime's CPU paths are
VNNI/AVX2-based; the AMX tile units are the headroom this project goes after.

AMX operand residency matters more than anything else — one 2×2 k-step
(4 `tile_loadd` + 4 `tile_dpbssd`), per `bench/kernels`:

| operands in | cycles / k-step | TOPS (1 thread) |
|---|---|---|
| registers (no loads) | 69 | 3.99 |
| L1 | 67 | 4.09 |
| L2 | 93 | 2.96 |
| L3 | 332 | 0.83 |
| DRAM | 388 | 0.71 |

Tile loads are effectively free out of L1 and catastrophic out of L3. Blocking
for L1/L2 residency is the whole game.

## What Gemma 4 E2B actually is

Read off the checkpoint, not the model card:

- 35 layers, hidden 1536, MQA with **1 KV head**, `head_dim` 256 on sliding
  layers but **512 on full-attention** layers, layer pattern `ssssF`×7.
- **20 of 35 layers share KV** and carry *no* k/v projection weights — the
  checkpoint ships them but `transformers` lists them in
  `_keys_to_ignore_on_load_unexpected`. We drop them.
- Those same 20 layers get a **double-wide MLP** (12288 vs 6144).
- Only layers 0–14 store KV, and only 4 of those need full length. Measured at
  10k context: **40 MB/seq, 320 MB for all 8 concurrent requests**, where a naive
  all-layers-full-length engine would need ~1.8 GB.
- Per-Layer Embeddings are a 262144 × 8960 lookup table — 2.35 B parameters,
  *larger than the entire compute network* (1.86 B), but pure gather.
- Full-attention layers use `proportional` RoPE with `partial_rotary_factor`
  0.25, so only 128 of 512 head dims rotate.

E4B differs: hidden 2560, 42 layers, 2 KV heads, 18 shared layers, no double-wide
MLP.

## Layout

```
convert/                   safetensors -> .fgm (quantise, rotate, AMX tile-pack)
crates/fgm-kernels/csrc/   AMX INT8 GEMM
bench/roofline/            hardware ceiling probe
bench/kernels/             GEMM correctness + throughput sweeps
```

## Quantisation

| tensor class | format | why |
|---|---|---|
| FFN gate/up/down | int4, group 256 | 84% of compute params; decode is DRAM-bound |
| attention q/k/v/o | int8 per-channel | small, and accuracy-critical |
| embed / LM head | int4, group 256 | 403 M params |
| PLE table | int4, group 64 | 2.35 B params, pure gather; no AMX drain so 64 is free |
| norms, scalars | f32 | negligible |

Group size is a measured dial, not a guess — it sets how often the int4 kernel
must drain its int32 accumulator (`--group`):

| group | weight rel err | prefill 512 | decode |
|---|---|---|---|
| 64 | 0.0902 | 101.2 tok/s | 12.7 tok/s |
| 256 | 0.1016 | **131.9 tok/s** | **14.5 tok/s** |

Every linear weight is **Hadamard-rotated along K** (`rotconv`). H is orthogonal,
so `W' = W Hᵀ` plus a runtime fast Walsh-Hadamard transform of the activation is
mathematically identical, but both weight groups and activation rows lose their
outliers. Measured on outlier-heavy weights this takes int4 relative error from
**0.194 → 0.078**. Runtime cost is a 128-point FWHT (~10 K ops/token) against an
18.9 M-op GEMM.

int4 scales use all 16 levels (Q4_0 convention: the extreme element maps onto −8)
refined by a per-group MSE search over shrink factors — conversion-time only, no
runtime change. Gaussian relative error 0.107 → 0.0906.

E2B converts to **2.78 GB** in ~1070 s; measured int4 FFN relative error ≈ 0.102
at group 256, 0.090 at group 64.

## Kernel notes

B is pre-packed by the converter into AMX tile order (`[n_block][k_block]`, each
tile 16 rows × 64 B holding `(K=64, N=16)` as `[k/4][n][k%4]` — the VNNI layout
`_tile_dpbssd` consumes), so the runtime mmaps and executes with zero repacking.
A is packed at call time into contiguous tiles: loading a 16×64 tile straight out
of row-major `A[M,K]` means 16 cache lines K bytes apart, and those stalls cost
more than the `dpbssd` they feed.

int4 group scales force an int32→f32 drain every `group/64` tile steps. The drain
is O(MR·NR) while the AMX work it amortises is O(MR·NR·group), so group size is a
direct speed/accuracy dial — `bench/kernels/gemm_bench` sweeps it.

> **`unpack_tile` contains a load-bearing `asm volatile` memory barrier.**
> `_tile_loadd` is opaque to GCC's alias analysis, so at `-O3` it sinks or
> eliminates the AVX stores that fill the unpack buffer. Without the barrier the
> int4 path silently returns garbage at `-O3` while passing at `-O2`. Both paths
> are verified against scalar references over a 19-shape sweep.

## Reproducing

```sh
gcc -O2 -march=sapphirerapids -mamx-int8 -mamx-bf16 -mamx-tile \
    -o bench/roofline/roofline bench/roofline/roofline.c -lpthread && \
    ./bench/roofline/roofline 4

gcc -O3 -march=sapphirerapids -mamx-int8 -mamx-tile -mavx512fp16 \
    -o bench/kernels/gemm_bench bench/kernels/gemm_bench.c \
    crates/fgm-kernels/csrc/amx_gemm.c -lpthread -lm && \
    ./bench/kernels/gemm_bench 4 6144 1536

python3 convert/convert_gemma4.py --src <hf-dir> --out model.fgm
```

## vs llama.cpp (same box, same model, same bit width)

Baseline is `google/gemma-4-E2B-it-qat-q4_0-gguf` — Google's own QAT Q4_0 GGUF —
on llama.cpp build 91f8c9c with `GGML_NATIVE=ON`, 4 threads, idle machine.

**Decode, aggregate tok/s, 512-token prompts:**

| concurrency | fastgemma | llama.cpp | |
|---|---|---|---|
| 1 | 15.0 | **19.5** | llama +30% |
| 4 | **46.7** | 40.1 | **fastgemma +16%** |
| 8 | **67.3** | 59.7 | **fastgemma +13%** |

**Prefill, tok/s:**

| tokens | fastgemma | llama.cpp | |
|---|---|---|---|
| 128 | **168.9** | 163.4 | fastgemma +3% |
| 256 | **164.5** | 161.0 | fastgemma +2% |
| 512 | 142.9 | **158.5** | llama +11% |

At the concurrency this project was briefed for — 8 requests — fastgemma is
**13% faster on decode**. It is *not* faster single-stream: at M=1 decode is pure
memory bandwidth and AMX has nothing to work with, while llama.cpp's Q4_0 kernels
are extremely well tuned. The advantage appears exactly when requests batch and
one weight read serves 8 rows. Long prefill still favours llama.cpp because our
attention kernel is naive and O(M²) — that is the next fix, not a ceiling.

## Results so far (E2B, 4 threads, int4 group 256)

Prefill, single sequence:

| tokens | tok/s |
|---|---|
| 128 | 168.9 |
| 256 | **164.5** |
| 512 | 142.9 |

Decode:

| | tok/s |
|---|---|
| 1 sequence | 15.3 |
| 8 concurrent, aggregate | **67.3** (4.5× batching win) |

KV cache at 8 concurrent × 10k context: **320 MB total, 40 MB/seq.**

Accuracy vs `transformers` bf16, deterministic (1-thread) reference, real text:

| metric | value |
|---|---|
| greedy agreement | 82.7% |
| HF's pick in our top-3 | 98.8% |
| HF's pick in our top-5 | 100.0% |
| mean cosine | 0.99901 |
| median top1–top2 gap | 3.79 overall, **1.04 where we disagree** |

Disagreements sit almost entirely on near-ties — the signature of int4 weight
error, not a structural fault. Layer-by-layer the engine tracks HF at cos > 0.99
through layer 33.

> The reference **must** be run single-threaded. torch's multithreaded bf16 path
> is non-deterministic on this model: HF agrees with itself on only 86.4% of
> positions at 4 threads (max logit diff 10.7) versus 100.0% at 1 thread.
> Grading against the 4-thread "noise floor" would have flattered this engine by
> ~15 points.

Constrained tool calling, 12 tools × 4 params on the real 262144-token vocab:
1011 DFA states, 3.4 s one-time compile, 33.1 MB of mask, output parses with the
right tool and all args. **36% of decode steps admit exactly one token** — those
can skip the 201 MB int4 LM-head read entirely.

## Status

Done and measured: hardware roofline, AMX INT8 kernels, quantizing converter,
validated Gemma 4 forward pass, worker pool, batched decode, int4 group-size
A/B, constrained tool-call decoding, accuracy harness.

Not done yet: llama.cpp / ONNX Runtime baselines on this box, E4B conversion,
prefix radix cache for the shared tool-definition prefix, K-blocking for
L1-resident AMX operands (the largest known remaining kernel win), and
speculative multi-token prediction. See `JOURNAL.md` for the running log,
including the traps that cost the most time.
