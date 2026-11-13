#!/bin/bash
# Submit single-agent PPO runs (train_single_agent.py) to Slurm: one job per location x seed.
# Same interface and output layout as scripts/marl/launch.sh, so all methods of an experiment share one folder.
#
# Usage (from anywhere):
#   scripts/single/launch.sh EXP_NAME STEPS TIME "LOCATIONS" "SEEDS"
# e.g.
#   scripts/single/launch.sh grid_v1 20000000 08:00:00 az "1 2 3"
#
# Settings (environment variables, as for scripts/marl/launch.sh):
#   CODE         code directory the jobs run from (default: this checkout). Pin it if you keep editing:
#                git worktree add ../code-$(git rev-parse --short HEAD) HEAD; CODE=...
#   PYTHON       python of your environment (default: `python` on PATH when you submit)
#   SDC_RUNS     where runs go (default: $CODE/results/marl_runs); runs of EXP_NAME go to $SDC_RUNS/EXP_NAME
#   SBATCH_ARGS  extra sbatch options, e.g. "-p main-cpu,long-cpu"
#   EXTRA        extra train_single_agent.py arguments for every run (they override the settings below)
#   DRY_RUN=1    write the job scripts but do not submit
#
# Each job: 16 CPUs (15 env processes), 16 GB, TIME. Reward: the env yaml's (co2_team_reward). No PV.
# Hyperparameters: the best single-agent setting of the v2/v4 runs (lr 3e-4, entropy 0.001, clip 0.2, 10 epochs,
# VecNormalize on observations and rewards); evaluation and checkpoint every 250k env steps.
# Preemption and time limit: jobs use --run_id $SLURM_JOB_ID; a preempted job (--requeue) or one that reaches TIME
# (requeued by the USR1 trap 5 min before) resumes from its latest checkpoint.
# Outputs: $SDC_RUNS/EXP_NAME/{jobs,slurm,runs}, manifest.tsv. Evaluate with scripts/single/eval.sh.
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,25p' "$0"; exit 1; fi
EXP=$1; STEPS=$2; TIME=$3; LOCS=$4; SEEDS=$5

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "${CODE:-$SCRIPT_DIR/../..}" && pwd)
PY=${PYTHON:-$(command -v python || true)}
[ -n "$PY" ] || { echo "No python on PATH: activate your environment or set PYTHON=..."; exit 1; }
PY=$(command -v "$PY")
OUT=${SDC_RUNS:-$CODE/results/marl_runs}/$EXP
CPUS=16

PPO_ARGS="--algo ppo --n_envs $((CPUS - 1)) --learning_rate 3e-4 --ent_coef 0.001 --clip_range 0.2 --n_epochs 10"
RUN_ARGS='--pv_capacity_kWp 0 --no-include_pv_obs --eval_freq 250000 --n_eval_episodes 5 --checkpoint_freq 250000'

mkdir -p "$OUT/jobs" "$OUT/slurm"
echo "EXP $EXP | single_ppo | code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') | python $PY | out $OUT"
for loc in $LOCS; do for seed in $SEEDS; do
  name=single_ppo_${loc}_s${seed}
  cat > "$OUT/jobs/$name.sh" <<JOB
#!/bin/bash
cd "$CODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'echo "[\$(date)] time limit near, requeueing \$SLURM_JOB_ID"; scontrol requeue "\$SLURM_JOB_ID"' USR1
echo "host \$(hostname)  start \$(date)  code \$(git rev-parse --short HEAD 2>/dev/null)  restart \${SLURM_RESTART_COUNT:-0}"
"$PY" -u train_single_agent.py $PPO_ARGS --exp_name $EXP --out_root "$OUT/runs" \\
    --location $loc --seed $seed --total_timesteps $STEPS --run_id \$SLURM_JOB_ID $RUN_ARGS ${EXTRA:-} &
wait \$!
echo "exit \$?  end \$(date)"
JOB
  chmod +x "$OUT/jobs/$name.sh"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry run) $OUT/jobs/$name.sh"; continue; fi
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable -J "sdc-$EXP-$name" --cpus-per-task=$CPUS --mem=16G --time=$TIME --requeue \
        --signal=B:USR1@300 --open-mode=append ${SBATCH_ARGS:-} -o "$OUT/slurm/$name.log" "$OUT/jobs/$name.sh")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "single_ppo" "$loc" "$seed" "$STEPS" "$(date -Is)" >> "$OUT/manifest.tsv"
  echo "$jid $name"
done; done
