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
- Only layers 0–14 store KV, and only 4 of those need full length. At 8k context
  that is **~32 MB/seq — 258 MB for all 8 concurrent requests**, where a naive
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
| FFN gate/up/down | int4, group 64 | 84% of compute params; decode is DRAM-bound |
| attention q/k/v/o | int8 per-channel | small, and accuracy-critical |
| embed / LM head | int4 group 64 | 403 M params |
| PLE table | int4 group 64 | 2.35 B params, pure gather, mmap-friendly |
| norms, scalars | f32 | negligible |

Every linear weight is **Hadamard-rotated along K** (`rotconv`). H is orthogonal,
so `W' = W Hᵀ` plus a runtime fast Walsh-Hadamard transform of the activation is
mathematically identical, but both weight groups and activation rows lose their
outliers. Measured on outlier-heavy weights this takes int4 relative error from
**0.194 → 0.078**. Runtime cost is a 128-point FWHT (~10 K ops/token) against an
18.9 M-op GEMM.

int4 scales use all 16 levels (Q4_0 convention: the extreme element maps onto −8)
refined by a per-group MSE search over shrink factors — conversion-time only, no
runtime change. Gaussian relative error 0.107 → 0.0906.

E2B converts to **2.83 GB** in 1076 s; measured int4 FFN relative error ≈ 0.090.

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

## Status

Converter and AMX kernels done and verified. Model forward, serving runtime,
constrained tool-call decoding, and end-to-end baselines in progress.
