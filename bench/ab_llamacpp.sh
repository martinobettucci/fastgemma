#!/bin/bash
# fastgemma vs llama.cpp on the same box, same model family, same bit width.
#
# Parked earlier at the user's instruction until our side was finished, and for
# a good reason: the first comparison was taken while a full benchmark was
# running on the same 4 cores, which is exactly the contamination that has
# corrupted five measurements in this project.
#
# Engines ALTERNATE rather than running all of one then all of the other. This
# box's AMX tile-state corruption rate drifts between runs and only fastgemma
# uses AMX, so a block design would hand llama.cpp whichever half of the drift
# it happened to land in.
set -u
FG=./target/release/fgm-bench
LC=/home/user/llamacpp/build/bin/llama-bench
GGUF=/home/user/models/g4e2b-q4_0.gguf
FGM=${FGM:-/home/user/models/g4e2b-g256.fgm}
idle() { for _ in $(seq 240); do
  if ! pgrep -x fgm-bench >/dev/null && ! pgrep -x llama-bench >/dev/null; then
    l=$(awk '{print $1}' /proc/loadavg); (( $(echo "$l < 1.0"|bc -l) )) && return
  fi; sleep 5; done; }

for pass in 1 2; do
  for pp in 512 2048; do
    idle
    printf "pass%s pp=%s fastgemma " "$pass" "$pp"
    FGM_CONC=1 FGM_REPEAT=1 FGM_PP=$pp FGM_TG=64 $FG matrix "$FGM" 2>/dev/null \
      | awk -v p=$pp '$1==p {printf "pp=%s tg=%s\n", $5, $6}'
    idle
    printf "pass%s pp=%s llama.cpp " "$pass" "$pp"
    # Parse the markdown table llama-bench prints by default. The -o csv path
    # was tried first and its column layout did not match what I assumed, which
    # produced four empty rows that looked like llama.cpp failing rather than
    # like my parser failing.
    $LC -m "$GGUF" -p $pp -n 64 -t 4 -r 1 2>/dev/null \
      | awk -F'|' '/\| *pp[0-9]/ {gsub(/ /,"",$7); split($7,a,"±"); p=a[1]}
                   /\| *tg[0-9]/ {gsub(/ /,"",$7); split($7,a,"±"); t=a[1]}
                   END {printf "pp=%s tg=%s\n", p, t}'
  done
done
