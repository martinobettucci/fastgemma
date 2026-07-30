#!/bin/bash
# Behavioural acceptance gate, run strictly serially on an idle box.
#
# This is the accuracy gate for the whole engine. It is deliberately not logit
# agreement with a reference: int4 weights, int8 activations, an int8 query, a
# Hadamard rotation and 16-bit softmax weights all trade numerical precision for
# speed on purpose, so the numbers were never going to match. What must hold is
# behavioural -- right tool, right arguments, and the start of an 8k context
# still recalled at the end of it.
set -u
M=${MODEL:-/home/user/models/g4e2b-g256.fgm}
TOK=${TOK:-/home/user/models/g4e2b/tokenizer.json}
idle() {
  for _ in $(seq 240); do
    if ! pgrep -x fgm-bench >/dev/null; then
      l=$(awk '{print $1}' /proc/loadavg)
      if (( $(echo "$l < 1.0" | bc -l) )); then return; fi
    fi
    sleep 5
  done
  echo "WARNING: box never went idle" >&2
}
step() { echo; echo "######## $* ########"; idle; }

step "smoke: 2 cases, unconstrained"
python3 bench/eval/tool_eval.py --model "$M" --tokenizer "$TOK" --limit 2 --ngen 160 || exit 1

step "tool calling, unconstrained (24 cases)"
python3 bench/eval/tool_eval.py --model "$M" --tokenizer "$TOK" --ngen 160

step "tool calling, grammar-constrained (24 cases)"
python3 bench/eval/tool_eval.py --model "$M" --tokenizer "$TOK" --constrained --ngen 160

step "long-context retention at 8192, 7 depths"
python3 bench/eval/retention.py --model "$M" --tokenizer "$TOK" --ctx 8192
