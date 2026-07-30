# fastgemma

A CPU-only inference engine for **Gemma 4**, written in Rust and C around Intel
**AMX INT8** tile units, with an **AVX-512 VNNI** path for hosts without them.

Weights, benchmarks and accuracy numbers live with the model:

### → **[P2Enjoy/fastgemma-gemma-4-E2B](https://huggingface.co/P2Enjoy/fastgemma-gemma-4-E2B)**

This repository is the engine. Below is how to build it and run it, and how to
tell in advance whether it will run on your machine at all.

---

## Will it run here?

Two hard requirements. Check both before building — the failure modes are a
compile error and a `SIGILL` respectively, and neither says what is wrong.

**1. x86-64 with AVX-512 and VNNI.**

```sh
grep -o -m1 'avx512f\|avx512_vnni' /proc/cpuinfo | sort -u
```

You need **both** lines. That is Skylake-SP-with-VNNI, Cascade Lake, Ice Lake,
Sapphire Rapids or newer. There is no AVX2, NEON or SVE fallback: the attention
kernels are written in 512-bit intrinsics throughout and the weight format is a
VNNI operand layout on disk. `fgm-serve` refuses to start with a message naming
what is missing rather than faulting inside a GEMM.

**2. AMX INT8, if you want the fast path.**

```sh
grep -o -m1 'amx_tile\|amx_int8' /proc/cpuinfo | sort -u
```

Optional. With it the engine runs the tile GEMM; without it, the VNNI GEMM.
Same weights file, chosen from CPUID at startup, printed on the first line of
output. AMX INT8 measures **12.7× AVX-512 VNNI** on the same silicon, so expect
prefill to fall by roughly half without it — decode much less, being
memory-bound rather than multiply-bound.

Under a hypervisor, `/proc/cpuinfo` can advertise AMX while the kernel refuses
the `XTILEDATA` grant. The engine checks CPUID *and* the grant, and falls back
rather than trusting either alone.

**Toolchain:** Rust 1.70+ and a C compiler that knows `-march=sapphirerapids`
(GCC 11+ or Clang 13+). The compiler only has to *know* the target; it does not
have to be running on one. The dependency tree is `serde`, `serde_json`,
`memmap2` and `cc` — no async runtime, no web framework, no HTTP client, no
tokenizer library.

**Memory:** the E2B weights are 4.34 GB, mapped rather than read, so ~6 GB of
RAM is comfortable and less will work at the cost of page faults.

---

## Build

```sh
git clone https://github.com/martinobettucci/fastgemma && cd fastgemma
cargo build --release
```

Three binaries land in `target/release/`:

| | |
|---|---|
| `fgm-serve` | the inference server — start here |
| `fgm-bench` | benchmark and evaluation driver |
| `fgm-tokcheck` | tokenizer differ, used by `bench/eval/tokenizer_check.py` |

## Run

```sh
./target/release/fgm-serve
```

First launch downloads `g4e2b-dual.fgm` (4.34 GB) and `tokenizer.json` from the
Hugging Face repo above into `~/.cache/fastgemma`, then listens on
`127.0.0.1:8080`. Interrupted downloads resume; a partial file is never
renamed into place, so a truncated 4 GB transfer cannot mmap cleanly and
produce nonsense.

```sh
curl localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

```json
{"choices":[{"text":" Paris.","index":0,"finish_reason":"stop"}],
 "usage":{"prompt_tokens":6,"completion_tokens":5,"total_tokens":11}}
```

### Options

```
--model FILE       .fgm weights (default: downloaded on first launch)
--tokenizer FILE   tokenizer.json (same)
--addr HOST:PORT   listen address (default 127.0.0.1:8080)
--threads N        worker threads (default: all cores)
--ctx N            max context in tokens (default 8192)
--batch N          sequences decoded together (default 8); one KV cache
                   is allocated per slot up front
--share-min N      reuse a shared token prefix at least N long across
                   requests (default 256; 0 disables)
--no-download      fail instead of fetching anything
```

| environment | |
|---|---|
| `FGM_HOME` | where downloaded files live |
| `FGM_HF_ENDPOINT` | Hugging Face mirror |
| `HF_TOKEN` | only for a gated or private mirror |
| `FGM_BACKEND` | `amx`\|`vnni`, overrides the CPUID choice |
| `FGM_WEIGHTS` | `int4`\|`int8`\|`auto:M` (default `auto:16`) |

### The API

`POST /v1/completions`, OpenAI-shaped, with `stream` supported via SSE. Also
`GET /v1/models` and `GET /health`.

**Text completions only.** No `/v1/chat/completions`: that would mean owning a
chat template, and getting Gemma 4's wrong is a silent accuracy loss rather than
an error. Send the formatted prompt yourself.

Concurrent requests are **decoded together**, up to `--batch` (default 8). This
is where the engine's throughput actually lives: a decode step reads ~1.4 GB of
int4 weights whether it computes one row or eight, so eight sequences cost about
what one costs. Measured on the same server, 8 concurrent clients × 64 tokens:

| | aggregate | per sequence |
|---|---|---|
| `--batch 1` | 7.6–8.0 tok/s | 0.95–1.00 |
| `--batch 8` | **36.0–40.7 tok/s** | 4.50–5.09 |

**4.8×**, and the output is byte-identical: eight identical prompts sent
concurrently produce one distinct completion, equal to the same prompt run
alone.

Prefill is not batched across sequences — it is compute-bound and already runs
256 rows per forward — so a request arriving mid-generation stalls the batch for
the length of its own prefill. That is a TTFT cost, not a throughput one.

**Shared prefixes are prefilled once** (`--share-min`, default 256). When
consecutive requests begin with the same token block — a common system prompt or
tool declaration — that block's KV is computed once and forked into every later
request, which then prefills only its unique suffix. Measured on one server, 8
requests sharing a 2304-token prefix: prefill **63.8 s vs 205.1 s** without
(3.2×), reused requests prefilling in ~2.4 s against ~24 s. The reused length is
floored to a whole prefill chunk so the arithmetic matches a full prefill
exactly; the generated text is byte-identical with sharing on or off.

These are rejected rather than ignored, because silently downgrading a request
returns wrong results instead of degraded ones:

```
n > 1 · temperature ≠ 0 · top_p ≠ 1 · logprobs · echo · best_of
suffix · logit_bias · token-id array as `prompt`
```

Sampling is greedy. `stop` (string or array) and `max_tokens` work.

## Benchmarks and evaluation

`fgm-bench` drives everything measured on this project:

```sh
./target/release/fgm-bench sweep  model.fgm      # prefill/decode vs shape
./target/release/fgm-bench matrix model.fgm      # (prompt x output) grid
./target/release/fgm-bench serve  model.fgm      # concurrency, KV, TTFT
./target/release/fgm-bench tools  model.fgm      # constrained tool calling

bash bench/run_eval.sh                           # behavioural acceptance gate
bash bench/ab_llamacpp.sh                        # vs llama.cpp, AMX host
bash bench/ab_llamacpp_vnni.sh                   # vs llama.cpp, AVX-512 host
python3 bench/eval/tokenizer_check.py            # tokenizer vs HF `tokenizers`
```

`fgm-bench` refuses to run when another benchmark is on the box or the load
average is high — contention has silently corrupted measurements on this
project five times. `FGM_IGNORE_LOAD=1` overrides it.

Kernel-level harnesses build standalone:

```sh
gcc -O2 -march=cascadelake -fno-strict-aliasing bench/kernels/vnni_test.c \
    crates/fgm-kernels/csrc/vnni_gemm.c crates/fgm-kernels/csrc/ops.c \
    -o bench/kernels/vnni_test -lm && ./bench/kernels/vnni_test bench

gcc -O2 -march=sapphirerapids -mamx-int8 -mamx-bf16 -mamx-tile \
    -o bench/roofline/roofline bench/roofline/roofline.c -lpthread && \
    ./bench/roofline/roofline 4
```

## Converting your own weights

```sh
python3 convert/convert_gemma4.py --src <hf-checkpoint-dir> --out model.fgm \
        --weights both --group 256
```

`--weights both` writes int8 twins of the FFN weights alongside the int4 ones
(+1.56 GB), which is what lets the runtime pick int8 for prefill and int4 for
decode. `--group` is the int4 group size, a measured speed/accuracy dial.

## Layout

```
crates/fgm-kernels/csrc/amx_gemm.c    AMX INT8 GEMM (tile path)
crates/fgm-kernels/csrc/vnni_gemm.c   AVX-512 VNNI GEMM (fallback path)
crates/fgm-kernels/csrc/ops.c         attention, norms, RoPE, quant, gathers
crates/fgm-core/                      forward pass, KV cache, worker pool, grammar
crates/fgm-serve/                     HTTP server, tokenizer, HF download
crates/fgm-bench/                     benchmark and evaluation driver
convert/                              safetensors -> .fgm
bench/                                roofline, kernel harnesses, eval gate
```

## JOURNAL.md

The running log: every experiment, every number, and every trap that cost time —
including the ones where the measurement was wrong rather than the code. If you
are wondering why something is built the way it is, the answer is usually there
with the measurement that forced it.

## Licence

The engine is in this repository under its own licence. The **weights** are a
derivative of `google/gemma-4-E2B-it` and are governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); see the model repo.
