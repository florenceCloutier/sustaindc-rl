#!/bin/bash
# Evaluate the learned-value MPC runs of one experiment / location on the shared protocol (scripts/eval_protocol.sh):
# noop + every run of EXP (its value.npz), 10 paired 30-day episodes. Runs are labelled mpc_<EXP>_s<k>, k = 1, 2, ...
# for training seeds 3000, 3100, ... (as in scripts/mpc/launch.sh), and also pooled per method.
# A run without value.npz (the naive terminal value was best) is evaluated with value=naive.
#
# Usage:  scripts/mpc/eval.sh EXP_NAME LOCATION
#   Slurm: sbatch -c 8 --mem 16G -t 04:00:00 scripts/mpc/eval.sh grid_v1 az
# SDC_RUNS: as for scripts/mpc/launch.sh. Results go to $SDC_RUNS/EXP_NAME/eval/LOCATION/mpc.
# PV_KWP / INCLUDE_PV: set them as for the training runs (see scripts/eval_protocol.sh); a value trained with PV
# refuses a scenario without PV.
# MODEL=final: evaluate the last policy-iteration round's value (value_iter<N>.npz, highest N) instead of the selected
#   best one (value.npz); labels and the output folder get a _final suffix.
set -euo pipefail
if [ $# -ne 2 ]; then sed -n '2,13p' "$0"; exit 1; fi
case ${MODEL:-best} in best) SFX='' ;; final) SFX=_final ;; *) echo "MODEL must be best or final"; exit 1 ;; esac
EXP=$1; LOC=$2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(cd "${SDC_RUNS:-$REPO/results/marl_runs}" && pwd)
DIRS=("$RUNS/$EXP"/runs/sustaindc/"$LOC"/mpc/"$EXP"/seed-*)
[ -d "${DIRS[0]}" ] || { echo "No runs found under $RUNS/$EXP/runs/sustaindc/$LOC/mpc/$EXP/"; exit 1; }
SPECS=()
for d in "${DIRS[@]}"; do
  name=${d##*/}; seed=${name#seed-}; seed=$((10#${seed%%-*}))
  k=$(( (seed - 3000) / 100 + 1 ))
  if [ -n "$SFX" ]; then
    value=$(ls "$d"/value_iter*.npz 2>/dev/null | sort -V | tail -1)
    [ -n "$value" ] || { echo "$name has no value_iter*.npz"; exit 1; }
  elif [ -f "$d/value.npz" ]; then
    value=$d/value.npz
  else
    echo "Note: $name has no value.npz (the naive value was best); evaluating it with value=naive"
    value=naive
  fi
  SPECS+=("mpc:H=16,forecast=realistic,value=$value=mpc${SFX}_${EXP}_s$k")
done
echo "${#DIRS[@]} run(s): ${DIRS[*]##*/}"
exec "$REPO/scripts/eval_protocol.sh" "$LOC" "$RUNS/$EXP/eval/$LOC/mpc$SFX" "${SPECS[@]}"
