#!/bin/bash
# A/B the blocked attention path against the per-head path across contexts.
#
# FGM_BLOCKED=<min_m>:<max_ctx> selects the path; 0:0 disables blocking. The
# sweep covers the short end too, because the two paths scale oppositely and a
# result measured only at 1024+ says nothing about 128.
#
# Waits for an idle box before each run -- four earlier measurement rounds in
# this project were corrupted by contention, one of them by a `cargo build`
# fired off while the bench was in its second arm.
set -u
M=${1:-/home/user/models/g4e2b-g256.fgm}
BIN=./target/release/fgm-bench
idle() {
  for _ in $(seq 60); do
    l=$(awk '{print $1}' /proc/loadavg)
    if (( $(echo "$l < 1.0" | bc -l) )); then return; fi
    sleep 5
  done
}
for pp in ${PPS:-128 256 512 1024 2048 4096 8192}; do
  for mode in blocked perhead; do
    idle
    if [ "$mode" = blocked ]; then export FGM_BLOCKED=16:999999999; else export FGM_BLOCKED=0:0; fi
    v=$(FGM_PP=$pp FGM_TG=8 $BIN curve "$M" 2>/dev/null \
        | awk -v p="$pp" '$1==p && NF==4 {print $3; exit}')
    echo "pp=$pp $mode ${v} tok/s"
  done
done
