#!/bin/bash
# A/B int4 vs int8 weights on a dual-format model.
#
# Median of FGM_REPEAT runs per cell. A single run of a fixed configuration on
# this box has std 4.7% on prefill and 4.4% on decode, so a one-sample-per-arm
# comparison cannot resolve anything under ~10% -- and two claims were published
# from exactly that before the noise floor was measured.
#
# The file is read into page cache first: it is 4.34 GB and a cold first touch
# reads as a ~40% regression that has nothing to do with the weights.
#
# Arms alternate rather than running all of one then all of the other, so any
# drift over the run (thermal, page cache, whatever else) is shared between them
# instead of landing entirely on whichever arm goes last.
set -u
M=${1:-/home/user/models/g4e2b-dual.fgm}
BIN=./target/release/fgm-bench
REPS=${REPS:-5}
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
for c in ${CONCS:-1 8}; do
  for w in int4 int8; do
    idle
    echo "### FGM_WEIGHTS=$w concurrency=$c reps=$REPS"
    FGM_WEIGHTS=$w FGM_CONC=$c FGM_REPEAT=$REPS FGM_PP=${PPS:-1024,8192} FGM_TG=${TGS:-128} \
      $BIN matrix "$M" 2>/dev/null | grep -E "^ +[0-9]+ +[0-9]+ |SUSPECT"
  done
done
