#!/bin/bash
# A/B int4 vs int8 weights on a dual-format model.
#
# The file is read into page cache first: it is 4.34 GB and a cold first touch
# reads as a 40% throughput regression that has nothing to do with the weights.
# Then wait for idle -- the previous run's threads take a few seconds to retire
# and the load guard will refuse the next one.
set -u
M=${1:-/home/user/models/g4e2b-dual.fgm}
BIN=./target/release/fgm-bench
# Wait for a genuinely free box. Checking loadavg alone is not enough: it is a
# one-minute decaying average, so a four-thread competitor one second old barely
# moves it. An orphaned bench from a killed run slipped past exactly that check
# and made one arm of this comparison read 2x low.
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
echo "warming page cache for $M"
dd if="$M" of=/dev/null bs=16M status=none
for w in int4 int8; do
  for c in 1 8; do
    idle
    echo "### FGM_WEIGHTS=$w concurrency=$c"
    FGM_WEIGHTS=$w FGM_CONC=$c FGM_PP=${PPS:-1024,8192} FGM_TG=${TGS:-128} \
      $BIN matrix "$M" 2>/dev/null | grep -E "^ +[0-9]+ +[0-9]+ "
  done
done
