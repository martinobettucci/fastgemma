#!/bin/bash
# Full measurement pass. Runs strictly serially and waits for an idle box before
# every step.
#
# Contention has silently corrupted measurements four times in this project: a
# concurrent conversion (-15%), a concurrent build (-40%), stray processes from a
# timed-out loop, and one `cargo build` fired off while a 20-minute prefix-sharing
# run was in its second arm. Build first, measure second, never both.
set -u
MODEL=${MODEL:-/home/user/models/g4e2b-g256.fgm}
DUAL=${DUAL:-/home/user/models/g4e2b-dual.fgm}
TOK=${TOK:-/home/user/models/g4e2b/tokenizer.json}
BIN=./target/release/fgm-bench

idle() {
  for _ in $(seq 120); do
    l=$(awk '{print $1}' /proc/loadavg)
    if (( $(echo "$l < 1.0" | bc -l) )); then return; fi
    sleep 5
  done
  echo "WARNING: box never went idle; numbers below are not trustworthy" >&2
}

step() { echo; echo "######## $* ########"; idle; }

step "grammar: structural smoke test"
$BIN tools "$MODEL" "$TOK"

step "behavioural: tool calling, unconstrained"
python3 bench/eval/tool_eval.py --model "$MODEL" --tokenizer "$TOK" --ngen 160

step "behavioural: tool calling, grammar-constrained"
python3 bench/eval/tool_eval.py --model "$MODEL" --tokenizer "$TOK" --constrained --ngen 160

step "behavioural: 8k long-context retention"
python3 bench/eval/retention.py --model "$MODEL" --tokenizer "$TOK" --ctx 8192

step "prefix sharing, 8 seq, 2048 shared + 6144 unique"
FGM_CONC=8 FGM_PREFIX=2048 FGM_SUFFIX=6144 $BIN share "$MODEL"

if [ -f "$DUAL" ]; then
  # Both extremes, at concurrency 1 and 8. `auto` picks int8 above 16 rows, so
  # the crossover claim needs the int4 and int8 endpoints measured at both a
  # prefill shape (m=256) and a batched-decode shape (m=8) to be worth anything.
  for c in 1 8; do
    for w in int4 int8; do
      step "dual weights: FGM_WEIGHTS=$w, concurrency $c"
      FGM_CONC=$c FGM_WEIGHTS=$w FGM_PP=512,2048,8192 FGM_TG=128 $BIN curve "$DUAL"
    done
  done
fi

step "target profile: 8 concurrent, 8k in / 2k out"
FGM_CONC=8 FGM_IN=8192 FGM_OUT=2048 FGM_SHARE=2048 $BIN serve "$MODEL"
