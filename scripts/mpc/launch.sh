#!/bin/bash
# Submit learned-value MPC runs (train_mpc_value.py) to Slurm: one job per location x seed.
# Same interface and output layout as scripts/marl/launch.sh, so all methods of an experiment share one folder.
#
# Usage (from anywhere):
#   scripts/mpc/launch.sh EXP_NAME TIME "LOCATIONS" "SEEDS"
# e.g.
#   scripts/mpc/launch.sh grid_v1 04:00:00 az "1 2 3"
#
# Seed s uses the training episodes 3000 + 100 (s - 1), 3000 + 100 (s - 1) + 1, ... (MPC_TRAIN_SEED blocks), disjoint
# from the in-training evaluation seeds (10000+) and the final protocol seeds (1000+). The run directory is named
# after that training seed (seed-03000, seed-03100, ...).
#
# Settings (environment variables): CODE, PYTHON, SDC_RUNS, SBATCH_ARGS, DRY_RUN as for scripts/marl/launch.sh;
#   EXTRA       extra train_mpc_value.py arguments for every run (they override the settings below)
#   PV_KWP      on-site PV capacity in kWp (default 0: no PV). The learned value then has the PV context features
#               and must be evaluated with the same PV settings (scripts/mpc/eval.sh with the same PV_KWP/INCLUDE_PV).
#   INCLUDE_PV  1: PV actions, the MPC's battery planner picks among the 9 battery x PV actions (use / battery /
#               resell); needs PV_KWP > 0. Unset: the automatic PV rule.
#
# Each job: 16 CPUs (16 rollout workers), 16 GB, TIME. Policy iteration: 4 rounds of 32 training episodes (+ the last
# value's round), defaults otherwise (H=16, realistic forecast, per_unit value), fixed-seed evaluation each round.
# No PV unless PV_KWP is set. A requeued job restarts the run from scratch (it is short).
# Outputs: $SDC_RUNS/EXP_NAME/{jobs,slurm,runs}, manifest.tsv. Evaluate with scripts/mpc/eval.sh.
set -euo pipefail
if [ $# -ne 4 ]; then sed -n '2,22p' "$0"; exit 1; fi
EXP=$1; TIME=$2; LOCS=$3; SEEDS=$4

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "${CODE:-$SCRIPT_DIR/../..}" && pwd)
PY=${PYTHON:-$(command -v python || true)}
[ -n "$PY" ] || { echo "No python on PATH: activate your environment or set PYTHON=..."; exit 1; }
PY=$(command -v "$PY")
OUT=${SDC_RUNS:-$CODE/results/marl_runs}/$EXP
CPUS=16

PV_KWP=${PV_KWP:-0}
MPC_ARGS="--iters 4 --episodes 32 --workers $CPUS --pv_capacity_kWp $PV_KWP"
if [ "${INCLUDE_PV:-0}" = 1 ]; then
  awk -v k="$PV_KWP" 'BEGIN { exit !(k > 0) }' || { echo "INCLUDE_PV=1 needs PV_KWP > 0"; exit 1; }
  MPC_ARGS="$MPC_ARGS --include_pv"
fi

mkdir -p "$OUT/jobs" "$OUT/slurm"
echo "EXP $EXP | mpc | PV ${PV_KWP} kWp$([ "${INCLUDE_PV:-0}" = 1 ] && echo ', PV actions') | code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') | python $PY | out $OUT"
for loc in $LOCS; do for seed in $SEEDS; do
  name=mpc_${loc}_s${seed}
  train_seed=$((3000 + 100 * (seed - 1)))
  cat > "$OUT/jobs/$name.sh" <<JOB
#!/bin/bash
cd "$CODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host \$(hostname)  start \$(date)  code \$(git rev-parse --short HEAD 2>/dev/null)  restart \${SLURM_RESTART_COUNT:-0}"
"$PY" -u train_mpc_value.py --exp_name $EXP --out_root "$OUT/runs" --location $loc --seed $train_seed \\
    --run_id \$SLURM_JOB_ID $MPC_ARGS ${EXTRA:-}
echo "exit \$?  end \$(date)"
JOB
  chmod +x "$OUT/jobs/$name.sh"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry run) $OUT/jobs/$name.sh"; continue; fi
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable -J "sdc-$EXP-$name" --cpus-per-task=$CPUS --mem=16G --time=$TIME --requeue \
        --open-mode=append ${SBATCH_ARGS:-} -o "$OUT/slurm/$name.log" "$OUT/jobs/$name.sh")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "mpc" "$loc" "$train_seed" "-" "$(date -Is)" >> "$OUT/manifest.tsv"
  echo "$jid $name"
done; done
