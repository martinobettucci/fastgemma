#!/usr/bin/env bash
# Apples-to-apples baseline: llama.cpp on the same box, same model, same
# int4-class quantisation.
#
# Model: google/gemma-4-E2B-it-qat-q4_0-gguf (Google's own QAT Q4_0 GGUF), which
# is the fairest comparison available -- same weights, same bit width, and it is
# the build the llama.cpp ecosystem would actually deploy.
#
# The point of the comparison is the instruction path: llama.cpp's CPU backend
# uses AVX2/AVX512-VNNI, measured at 1.16 TOPS on this box, against 14.69 TOPS
# for the AMX tile units. Everything else (quantisation, KV layout, threading)
# is a second-order difference next to that 12.7x.
#
# Usage: run_llamacpp.sh <llama.cpp dir> <gguf> [threads]
set -euo pipefail

SRC=${1:-/home/user/llamacpp}
GGUF=${2:-/home/user/models/g4e2b-q4_0.gguf}
NT=${3:-4}

if [ ! -x "$SRC/build/bin/llama-bench" ]; then
  echo "== building llama.cpp (native, AMX+AVX512 enabled if detected) =="
  cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON \
    >/dev/null
  cmake --build "$SRC/build" -j"$NT" --target llama-bench llama-cli >/dev/null
fi

echo "== llama.cpp build flags actually compiled in =="
"$SRC/build/bin/llama-bench" --help >/dev/null 2>&1 || true
grep -oE "GGML_(AMX[A-Z_]*|AVX512[A-Z_]*|AVX2)" "$SRC/build/CMakeCache.txt" 2>/dev/null | sort -u || true

echo
echo "== prefill (pp) and decode (tg) throughput, $NT threads =="
# pp512 = prefill 512 tokens, tg128 = generate 128 tokens, -b batch size
"$SRC/build/bin/llama-bench" -m "$GGUF" -t "$NT" -p 128,256,512 -n 64 -r 2

echo
echo "== batched decode, the target concurrency =="
for B in 1 8; do
  echo "-- batch $B --"
  "$SRC/build/bin/llama-bench" -m "$GGUF" -t "$NT" -p 0 -n 64 -b "$B" -r 2 2>/dev/null || \
    echo "   (llama-bench in this version may not expose -b for tg; see llama-batched-bench)"
done
