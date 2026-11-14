#!/bin/bash
# Perfect-foresight LP lower bound on co2_work_adjusted (sdc_lp_bound.py) on the shared protocol episodes, plus the
# replay of its plan (policy lp_replay) scored by scripts/eval_protocol.sh, so both go into the same paired table as
# the other methods. One location per call; runs in the repo root.
#
# Usage:
#   scripts/lp_bound.sh LOC OUT_DIR            (LOC: az, ny, tx or ca)
#   Slurm: sbatch -c 2 --mem 16G -t 04:00:00 scripts/lp_bound.sh tx $SDC_RUNS/oracles_v1/eval/tx/lp
# Writes OUT_DIR/bound/{bound.json,bound.csv} (per-episode bound and its paired % vs noop) and OUT_DIR/replay/
# (eval_compare output: noop + lp_replay; merge it with the other folders with eval_compare.py --combine).
# SKIP_REPLAY=1 computes the bound only.
set -euo pipefail
if [ $# -ne 2 ]; then sed -n '2,12p' "$0"; exit 1; fi
LOC=$1; OUT=$2
case $LOC in az|ny|tx|ca) ;; *) echo "LOC must be az, ny, tx or ca (got $LOC)"; exit 1 ;; esac
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
echo "lp_bound: $LOC | code $(git rev-parse --short HEAD 2>/dev/null || echo '?') | host $(hostname) | $(date)"
# Same scenario and seeds as scripts/eval_protocol.sh.
python -u sdc_lp_bound.py --location "$LOC" --month 6 --days_per_episode 30 --pv_capacity_kWp 0 \
    --n_episodes 10 --seed 1000 --out_dir "$OUT/bound"
if [ "${SKIP_REPLAY:-0}" != 1 ]; then
  scripts/eval_protocol.sh "$LOC" "$OUT/replay" "lp_replay=lp_replay"
fi
echo "done $(date)"
