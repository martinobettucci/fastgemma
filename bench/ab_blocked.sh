#!/bin/bash
# A/B the blocked attention path against the per-head path across contexts.
# Per-head is the default; FGM_BLOCKED_ATTN=1 opts into blocking. Waits for an
# idle box before
# each run -- three earlier measurement rounds were corrupted by contention.
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
for pp in 1024 2048 4096 8192; do
  for mode in blocked perhead; do
    idle
    if [ "$mode" = blocked ]; then export FGM_BLOCKED_ATTN=1; else unset FGM_BLOCKED_ATTN; fi
    v=$(FGM_PP=$pp FGM_TG=8 $BIN curve "$M" 2>/dev/null \
        | awk -v p="$pp" '$1==p && NF==4 {print $3; exit}')
    echo "pp=$pp $mode ${v} tok/s"
  done
done
