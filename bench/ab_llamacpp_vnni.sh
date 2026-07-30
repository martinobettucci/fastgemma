#!/bin/bash
# fastgemma (AVX-512 VNNI backend) vs llama.cpp, same box, same bit width.
# Engines alternate within each pass so any drift is shared.
set -u
cd /home/user/fastgemma
FG=./target/release/fgm-bench
LC=/home/user/llamacpp/build/bin/llama-bench
GGUF=/home/user/models/g4e2b-q4_0.gguf
export FGM_IGNORE_LOAD=1
idle() { for _ in $(seq 120); do
  pgrep -x fgm-bench >/dev/null || pgrep -x llama-bench >/dev/null || {
    l=$(awk '{print $1}' /proc/loadavg); (( $(echo "$l < 1.0"|bc -l) )) && return; }
  sleep 5; done; }

# Warm the page cache for every file first. A 2.8 GB mmap read cold off disk
# measured 2.5 tok/s prefill in the first attempt at this comparison, which
# looks exactly like the engine being 40x slower than it is.
for f in /home/user/models/g4e2b-g256.fgm /home/user/models/g4e2b-dual.fgm /home/user/models/g4e2b-q4_0.gguf; do
  cat "$f" > /dev/null; done

for pass in 1 2 3; do
  for pp in 512 2048; do
    for m in g256 dual; do
      idle
      printf "pass%s pp=%-5s fastgemma-%-5s " "$pass" "$pp" "$m"
      FGM_CONC=1 FGM_REPEAT=1 FGM_PP=$pp FGM_TG=64 $FG matrix "/home/user/models/g4e2b-$m.fgm" 2>/dev/null \
        | awk -v p=$pp '$1==p {printf "pp=%s tg=%s\n", $5, $6}'
    done
    idle
    printf "pass%s pp=%-5s llama.cpp        " "$pass" "$pp"
    $LC -m "$GGUF" -p $pp -n 64 -t 4 -r 1 2>/dev/null \
      | awk -F'|' '/\| *pp[0-9]/ {gsub(/ /,"",$8); split($8,a,"±"); p=a[1]}
                   /\| *tg[0-9]/ {gsub(/ /,"",$8); split($8,a,"±"); t=a[1]}
                   END {printf "pp=%s tg=%s\n", p, t}'
  done
done
