#!/bin/bash
# N concurrent completions against a running fgm-serve, aggregate tok/s.
set -u
N=${1:-8}; MT=${2:-48}; PORT=${3:-8080}
tmp=$(mktemp -d)
t0=$(date +%s.%N)
for i in $(seq 1 "$N"); do
  ( curl -s --noproxy '*' -m 900 "localhost:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"Question $i: describe in detail how a bicycle works.\nAnswer:\",\"max_tokens\":$MT}" \
      > "$tmp/$i.json" ) &
done
wait
t1=$(date +%s.%N)
gen=$(cat "$tmp"/*.json | grep -o '"completion_tokens":[0-9]*' | cut -d: -f2 | paste -sd+ | bc)
el=$(echo "$t1 - $t0" | bc)
printf "concurrency %-2s  %s gen tok in %.2fs  ->  %.2f tok/s aggregate  (%.2f tok/s/seq)\n" \
  "$N" "$gen" "$el" "$(echo "$gen / $el" | bc -l)" "$(echo "$gen / $el / $N" | bc -l)"
rm -rf "$tmp"
