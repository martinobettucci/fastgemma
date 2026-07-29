#!/bin/bash
# 8 concurrent 8192-in / 2048-out requests against a running fgm-serve.
# Reports the same three numbers `fgm-bench serve` does, measured end to end.
set -u
N=${1:-8}; PORT=${2:-8081}
S=${PROMPT_DIR:-bench/eval}
tmp=$(mktemp -d)
t0=$(date +%s.%N)
for i in $(seq 1 "$N"); do
  ( curl -s --noproxy '*' -m 7200 "localhost:$PORT/v1/completions" \
      -H 'Content-Type: application/json' --data-binary "@$S/prompt8192.json" \
      > "$tmp/$i.json"; date +%s.%N > "$tmp/$i.done" ) &
done
wait
t1=$(date +%s.%N)
gen=$(cat "$tmp"/*.json | grep -o '"completion_tokens":[0-9]*' | cut -d: -f2 | paste -sd+ | bc)
pro=$(cat "$tmp"/*.json | grep -o '"prompt_tokens":[0-9]*' | cut -d: -f2 | paste -sd+ | bc)
el=$(echo "$t1 - $t0" | bc)
first=$(cat "$tmp"/*.done | sort -n | head -1); last=$(cat "$tmp"/*.done | sort -n | tail -1)
printf "  total %.1fs | %s prompt tok, %s gen tok | request %.1f tok/s overall\n" \
  "$el" "$pro" "$gen" "$(echo "($pro+$gen)/$el" | bc -l)"
printf "  first response %.1fs, last %.1fs\n" \
  "$(echo "$first - $t0" | bc)" "$(echo "$last - $t0" | bc)"
rm -rf "$tmp"
