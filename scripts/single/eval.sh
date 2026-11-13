#!/bin/bash
# Evaluate the single-agent PPO runs of one experiment / location on the shared protocol (scripts/eval_protocol.sh):
# noop + every seed of EXP (their best_model), 10 paired 30-day episodes. Seeds are labelled ppo_<EXP>_s<seed> and
# also pooled per method.
#
# Usage:  scripts/single/eval.sh EXP_NAME LOCATION
#   Slurm: sbatch -c 8 --mem 16G -t 04:00:00 scripts/single/eval.sh grid_v1 az
# SDC_RUNS: as for scripts/single/launch.sh. Results go to $SDC_RUNS/EXP_NAME/eval/LOCATION/single_ppo; merge with
# the other methods' folders: python eval_compare.py --combine <dirs> --out_dir <dir>
set -euo pipefail
if [ $# -ne 2 ]; then sed -n '2,9p' "$0"; exit 1; fi
EXP=$1; LOC=$2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(cd "${SDC_RUNS:-$REPO/results/marl_runs}" && pwd)
DIRS=("$RUNS/$EXP"/runs/sustaindc/"$LOC"/single_ppo/"$EXP"/seed-*)
[ -d "${DIRS[0]}" ] || { echo "No runs found under $RUNS/$EXP/runs/sustaindc/$LOC/single_ppo/$EXP/"; exit 1; }
SPECS=()
for d in "${DIRS[@]}"; do
  name=${d##*/}; seed=${name#seed-}; seed=$((10#${seed%%-*}))
  SPECS+=("sb3:$d=ppo_${EXP}_s$seed")
done
echo "${#DIRS[@]} run(s): ${DIRS[*]##*/}"
exec "$REPO/scripts/eval_protocol.sh" "$LOC" "$RUNS/$EXP/eval/$LOC/single_ppo" "${SPECS[@]}"
