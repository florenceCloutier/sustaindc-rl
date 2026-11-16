#!/bin/bash
# Submit MPPI (mppi_policy.py) runs to Slurm, scored on the shared protocol (scripts/eval_protocol.sh: noop + the policy,
# by default seeds 1000..1009, 30 days, July, PV off), with ONE JOB PER EPISODE: each (location, config, forecast,
# seed) is a Slurm job array of N_EPISODES tasks, and a small dependent job merges them into the same folder layout as
# scripts/mpc/oracle_sweep.sh (episodes.csv + summary.json), so the results merge with every other method
# (eval_compare.py --combine). The results are identical to running the 10 episodes in one job (every episode is fixed
# by its seed, and the planner reseeds its RNG from it), but they arrive in about the time of a single episode, and a
# preempted or timed-out task only reruns its own episode.
#
# CPU only, on purpose: the planner rolls out exact clones of the simulator in CPU worker processes (numpy / Python,
# no torch), so a GPU would sit idle. Speed comes from the workers (WORKERS) and from the per-episode split.
#
# Usage (from anywhere):
#   scripts/mpc/mppi_launch.sh EXP_NAME "LOCATIONS" "CONFIGS" "FORECASTS" "SEEDS"
# e.g. MPPI next to the CEM oracle sweep (same configs and labels as scripts/mpc/oracle_sweep.sh with KIND=mppi):
#   scripts/mpc/mppi_launch.sh oracles_v1 "tx ca" "H48 H96" "oracle persistence" "1 2 3"
# a variant, labelled mppi_<forecast>_<config>_cat_s<k>:
#   EXTRA=sampling=categorical TAG=cat scripts/mpc/mppi_launch.sh oracles_v1 tx H48 oracle "1 2 3"
# choosing a variant on the tuning episodes first (labels get _sel, seeds 2000..):
#   MODE=select scripts/mpc/mppi_launch.sh oracles_v1 tx H48 oracle 1
#
# CONFIGS: as scripts/mpc/oracle_sweep.sh (scripts/mpc/configs.sh): H48 H96 H192 S H96S H96S_metric, run with the MPPI
#   defaults (gaussian sampling, ess=5, stochastic rounding, constant seed plans, execute=best; mppi_policy.py).
# FORECASTS: oracle (true future) and/or persistence (past data only).
# SEEDS: MPPI seeds k = 1, 2, ... (seed=k-1); runs are labelled mppi_<forecast>_<config>[_<TAG>]_s<k>, pooled per method.
#
# Settings (environment variables):
#   CODE, PYTHON, SDC_RUNS, SBATCH_ARGS, DRY_RUN=1   as for scripts/mpc/oracle_sweep.sh. Jobs read the code when they
#                START: pin it if you keep editing (git worktree add ../code-$(git rev-parse --short HEAD) HEAD; CODE=...)
#   EXTRA, TAG   more mppi parameters for every config (e.g. EXTRA=ess=10,noise=0.4), with a label suffix TAG
#   KIND         mppi (default) or mpc_joint: the CEM MPC with the same per-episode split
#   WORKERS      rollout worker processes per episode (default 16; the job gets WORKERS + 1 CPUs). An MPPI iteration
#                rolls out ~32 plans (~59 in the first, with the constant plans): 16 workers keep them busy, 32 is
#                ~1.8x faster at 2x the CPUs; past ~32 the extra CPUs mostly idle.
#   MEM          memory per episode job (default (WORKERS + 1) / 2 + 3 GB, 11G for 16 workers: each process measured
#                at ~0.3 GB)
#   TIME         per-episode time limit (default: per config, below)
#   ARRAY_MAX    at most this many episodes of one run at once (Slurm array throttle %N; default: no limit)
#   Scenario (as scripts/eval_protocol.sh): PV_KWP, INCLUDE_PV, CI_YEAR, WORKLOAD_FILE, MONTH (0..11 or cycle),
#                N_EPISODES (default 6 and 10)
#   MODE=select  the same scenario on the tuning episodes (seeds 2000..2004, 2000..2011 with MONTH=cycle)
#
# Per-episode time limits (30 days, 720 replans, WORKERS=16): measured rollout cost (H48 0.14 s, H96 0.27 s on a
# laptop core; ~124 rollouts per replan, ~9 rollouts on the critical path with 16 workers, ~43 for samples=128,iters=5)
# x2 for slower cluster cores x~2.5 margin. Short limits backfill sooner. x1.5 with INCLUDE_PV=1 (81 constant plans).
#   H48 01:30   H96 02:30   H192 05:00   S 06:00   H96S H96S_metric 12:00   (WORKERS=32: about 0.6x)
# Outputs: $SDC_RUNS/EXP_NAME/{jobs,slurm}, manifest.tsv, and per run $SDC_RUNS/EXP_NAME/eval/LOCATION/<label>/:
#   ep<seed>/ (one eval_compare folder per episode, with decisions.csv.gz), episodes.csv and summary.json (merged).
# Resubmitting the same command skips the episodes already done and only reruns the missing ones (then re-merges).
# Re-merge by hand: scripts/mpc/mppi_launch.sh --merge <eval_dir>
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Merge the per-episode folders of one run into episodes.csv + summary.json (the dependent job runs this).
if [ "${1:-}" = --merge ]; then
  DIR=$2; cd "${CODE:-$SCRIPT_DIR/../..}"
  shopt -s nullglob; parts=("$DIR"/ep*/episodes.csv); shopt -u nullglob
  [ ${#parts[@]} -gt 0 ] || { echo "No finished episode under $DIR"; exit 1; }
  echo "$DIR: ${#parts[@]} finished episode(s)${EXPECT:+ of $EXPECT}"
  [ -z "${EXPECT:-}" ] || [ ${#parts[@]} -eq "$EXPECT" ] || echo "WARNING: incomplete run, resubmit to finish it"
  awk 'FNR > 1 || NR == 1' "${parts[@]}" > "$DIR/episodes.csv"
  exec python -u eval_compare.py --combine "${parts[@]%/episodes.csv}" --out_dir "$DIR"
fi

if [ $# -ne 5 ]; then sed -n '2,50p' "$0"; exit 1; fi
EXP=$1; LOCS=$2; CONFIGS=$3; FORECASTS=$4; SEEDS=$5

CODE=$(cd "${CODE:-$SCRIPT_DIR/../..}" && pwd)
PY=${PYTHON:-$(command -v python || true)}
[ -n "$PY" ] || { echo "No python on PATH: activate your environment or set PYTHON=..."; exit 1; }
PY=$(command -v "$PY")
OUT=${SDC_RUNS:-$CODE/results/marl_runs}/$EXP
MODE=${MODE:-protocol}; KIND=${KIND:-mppi}; WORKERS=${WORKERS:-16}
case $KIND in mpc_joint|mppi) ;; *) echo "KIND must be mppi or mpc_joint (got $KIND)"; exit 1 ;; esac
case $MODE in protocol|select) ;; *) echo "MODE must be protocol or select (got $MODE)"; exit 1 ;; esac
if [ -n "${EXTRA:-}" ] && [ -z "${TAG:-}" ]; then echo "EXTRA needs a TAG (label suffix)"; exit 1; fi
[[ $WORKERS =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be a positive integer (got $WORKERS)"; exit 1; }
CPUS=$((WORKERS + 1))  # the rollout workers + the episode process
MEM=${MEM:-$((CPUS / 2 + 3))G}

# shellcheck source=configs.sh
source "$SCRIPT_DIR/configs.sh"
episode_time() {  # HH:MM:00, see the table above
  local m
  case $1 in H48) m=90 ;; H96) m=150 ;; H192) m=300 ;; S) m=360 ;; H96S|H96S_metric) m=720 ;; esac
  [ "${INCLUDE_PV:-0}" = 1 ] && m=$((m * 3 / 2))
  printf '%02d:%02d:00\n' $((m / 60)) $((m % 60))
}

PV_KWP=${PV_KWP:-0}; INCLUDE_PV=${INCLUDE_PV:-0}; MONTH=${MONTH:-6}; N_EPISODES=${N_EPISODES:-10}
if [ "$INCLUDE_PV" = 1 ]; then
  awk -v k="$PV_KWP" 'BEGIN { exit !(k > 0) }' || { echo "INCLUDE_PV=1 needs PV_KWP > 0"; exit 1; }
fi
case $MONTH in all) echo "MONTH=all is for training; use a fixed month or MONTH=cycle"; exit 1 ;; esac
if [ "$MODE" = protocol ]; then SEED0=1000; N=$N_EPISODES
else SEED0=2000; N=5; [ "$MONTH" = cycle ] && N=12; fi   # sdc_eval.constants.TUNE_SEED
# Everything but the month and the seed, which each array task sets.
SCEN_ARGS="--days_per_episode 30 --pv_capacity_kWp $PV_KWP$([ "$INCLUDE_PV" = 1 ] && echo ' --include_pv' || true)${CI_YEAR:+ --ci_year $CI_YEAR}${WORKLOAD_FILE:+ --workload_file $WORKLOAD_FILE}"

mkdir -p "$OUT/jobs" "$OUT/slurm"
echo "EXP $EXP | $KIND, one job per episode ($MODE: $N episodes from seed $SEED0, month $MONTH) | PV $PV_KWP kWp$([ "$INCLUDE_PV" = 1 ] && echo ', PV actions' || true) | CI ${CI_YEAR:-2022} | workload ${WORKLOAD_FILE:-yaml} | $CPUS CPUs, $MEM per episode | code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') | python $PY | out $OUT"
for loc in $LOCS; do
  case $loc in az|ny|tx|ca) ;; *) echo "LOCATION must be az, ny, tx or ca (got $loc)"; exit 1 ;; esac
  for cfg in $CONFIGS; do
    params=$(config_params "$cfg")${EXTRA:+,$EXTRA} || exit 1
    for fc in $FORECASTS; do
      case $fc in oracle|persistence) ;; *) echo "FORECAST must be oracle or persistence (got $fc)"; exit 1 ;; esac
      for k in $SEEDS; do
        label=${KIND}_${fc}_${cfg}${TAG:+_$TAG}$([ "$MODE" = select ] && echo _sel || true)_s$k
        spec="$KIND:$params,forecast=$fc,workers=$WORKERS,seed=$((k - 1))=$label"
        name=${label}_$loc
        eval_dir="$OUT/eval/$loc/$label"
        # Episodes still to run (resubmitting skips the finished ones).
        todo=()
        for ((i = 0; i < N; i++)); do [ -f "$eval_dir/ep$((SEED0 + i))/episodes.csv" ] || todo+=("$i"); done
        cat > "$OUT/jobs/$name.sh" <<JOB
#!/bin/bash
# Episode \$SLURM_ARRAY_TASK_ID of $label ($loc): seed $SEED0 + i, month $MONTH (cycle: i % 12).
cd "$CODE"
export PATH="$(dirname "$PY"):\$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
i=\$SLURM_ARRAY_TASK_ID; seed=\$(($SEED0 + i)); month=$MONTH
[ "\$month" = cycle ] && month=\$((i % 12))
dir="$eval_dir/ep\$seed"
echo "host \$(hostname)  start \$(date)  code \$(git rev-parse --short HEAD 2>/dev/null)  restart \${SLURM_RESTART_COUNT:-0}  episode \$i seed \$seed month \$month"
if [ -f "\$dir/episodes.csv" ]; then echo "already done: \$dir"; exit 0; fi
"$PY" -u eval_compare.py --policy noop --policy "$spec" --location $loc --month \$month \\
    $SCEN_ARGS --n_episodes 1 --seed \$seed --out_dir "\$dir.tmp"
rc=\$?
[ \$rc -eq 0 ] && rm -rf "\$dir" && mv "\$dir.tmp" "\$dir"
echo "exit \$rc  end \$(date)"
exit \$rc
JOB
        chmod +x "$OUT/jobs/$name.sh"
        if [ ${#todo[@]} -eq 0 ]; then
          echo "$name: all $N episodes done; re-merging only"; array=
        else
          array=$(IFS=,; echo "${todo[*]}")${ARRAY_MAX:+%$ARRAY_MAX}
        fi
        merge="EXPECT=$N CODE=\"$CODE\" PATH=\"$(dirname "$PY"):\$PATH\" \"$SCRIPT_DIR/mppi_launch.sh\" --merge \"$eval_dir\""
        if [ "${DRY_RUN:-0}" = 1 ]; then
          echo "(dry run) $OUT/jobs/$name.sh  array ${array:-none}  time ${TIME:-$(episode_time "$cfg")}"; continue
        fi
        dep=
        if [ -n "$array" ]; then
          # shellcheck disable=SC2086
          jid=$(sbatch --parsable -J "sdc-$EXP-$name" --array="$array" --cpus-per-task=$CPUS --mem=$MEM \
                --time=${TIME:-$(episode_time "$cfg")} --requeue --open-mode=append ${SBATCH_ARGS:-} \
                -o "$OUT/slurm/${name}_ep%a.log" "$OUT/jobs/$name.sh")
          dep="--dependency=afterany:$jid"
          printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "$KIND" "$loc" "$label" "$spec" "$(date -Is)" >> "$OUT/manifest.tsv"
        fi
        # shellcheck disable=SC2086
        mid=$(sbatch --parsable -J "sdc-$EXP-$name-merge" $dep --cpus-per-task=1 --mem=4G --time=00:20:00 \
              ${SBATCH_ARGS:-} -o "$OUT/slurm/${name}_merge.log" --wrap "$merge")
        echo "${jid:-} $name (${#todo[@]} episodes) -> merge $mid"
        jid=
      done
    done
  done
done
