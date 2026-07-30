#!/bin/bash
# Prefix sharing on vs off, one server each, sequential (no oversubscription).
#
# Sends 8 requests that share a long system prefix and differ only in a short
# body. With --share-min set, the shared block is prefilled once and forked into
# every later request; without it, each request re-prefills the whole thing.
#
# Two things are checked, and the correctness one is the point: the generated
# text must be BYTE-IDENTICAL with sharing on and off. It only is because the
# cached prefix length is floored to a whole prefill chunk -- a ragged prefix
# reuses a different GEMM chunking than a full prefill, and in an int4/int8
# engine that flips the occasional greedy token (JOURNAL trap 29).
#
# Usage: bash bench/prefix_share.sh [model.fgm] [tokenizer.json]
set -u
M=${1:-/home/user/models/g4e2b-dual.fgm}
TOK=${2:-/home/user/models/g4e2b/tokenizer.json}
FG=./target/release/fgm-serve
PORT=8096
TMP=$(mktemp -d)

python3 - "$TOK" "$TMP" <<'PY'
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1]); tmp = sys.argv[2]
shared = ("You are an expert assistant with access to a large toolset. "
          "Follow instructions precisely and cite your reasoning. ") * 120
bodies = ["Now, what is the capital of Japan?", "Now, list the first three primes.",
          "Now, what colour is a clear sky?", "Now, how many continents are there?",
          "Now, the chemical symbol for gold?", "Now, name the largest ocean.",
          "Now, what year was the moon landing?", "Now, the square root of 64?"]
for i, b in enumerate(bodies):
    open(f"{tmp}/req{i}.json", "w").write(json.dumps({"prompt": shared+" "+b, "max_tokens": 16}))
print(f"shared prefix: {len(tok.encode(shared, add_special_tokens=False).ids)} tokens")
PY

run() {  # tag sharemin
  local tag=$1 sm=$2
  kill -9 $(pgrep -x fgm-serve) 2>/dev/null; while pgrep -x fgm-serve >/dev/null; do sleep 1; done
  $FG --model "$M" --tokenizer "$TOK" --ctx 3072 --batch 8 --share-min "$sm" \
      --addr "127.0.0.1:$PORT" > "$TMP/srv_$tag.log" 2>&1 &
  until curl -s --noproxy '*' -m 2 "localhost:$PORT/health" >/dev/null 2>&1; do sleep 1; done
  local t0=$(date +%s.%N)
  for i in $(seq 0 7); do
    ( curl -s --noproxy '*' -m 900 "localhost:$PORT/v1/completions" \
        -H 'Content-Type: application/json' --data-binary "@$TMP/req$i.json" \
        > "$TMP/out_${tag}_$i.json" ) &
  done
  wait
  local t1=$(date +%s.%N)
  local pf=$(awk '/prefill/{for(i=1;i<=NF;i++) if($i=="in"){v=$(i+1);gsub(/s/,"",v);s+=v}} END{printf "%.1f",s}' "$TMP/srv_$tag.log")
  printf "  [%-3s] wall %5.1fs | prefill sum %5ss | reuse events %s\n" \
    "$tag" "$(echo "$t1-$t0"|bc)" "$pf" "$(grep -c 'shared,' "$TMP/srv_$tag.log")"
}

echo "== prefix sharing A/B =="
run off 0
run on  256
kill -9 $(pgrep -x fgm-serve) 2>/dev/null

ok=1
for i in $(seq 0 7); do
  a=$(python3 -c "import json;print(json.load(open('$TMP/out_off_$i.json'))['choices'][0]['text'])")
  b=$(python3 -c "import json;print(json.load(open('$TMP/out_on_$i.json'))['choices'][0]['text'])")
  [ "$a" = "$b" ] || { echo "  req $i DIFFERS"; ok=1000; }
done
[ $ok -eq 1 ] && echo "  output: all 8 byte-identical, sharing on vs off" || echo "  output: MISMATCH -- sharing changed results"
rm -rf "$TMP"
