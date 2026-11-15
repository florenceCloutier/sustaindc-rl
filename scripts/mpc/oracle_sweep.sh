#!/bin/bash
# Submit the joint-MPC (mpc_policy.py, CEM on a simulator clone) search / horizon sweep to Slurm: one job per
# location x config x forecast x CEM seed, each scored on the shared protocol (scripts/eval_protocol.sh: noop + the
# policy; by default seeds 1000..1009, 30 days, July, PV off, see Scenario below), so every result goes into the same paired table as the other
# methods (merge folders with eval_compare.py --combine).
#
# Usage (from anywhere):
#   scripts/mpc/oracle_sweep.sh EXP_NAME "LOCATIONS" "CONFIGS" "FORECASTS" "SEEDS"
# e.g. the full sweep with the perfect forecast, 3 CEM seeds, Texas and California:
#   scripts/mpc/oracle_sweep.sh oracles_v1 "tx ca" "H48 H96 H192 S H96S H96S_metric" oracle "1 2 3"
# and the same-effort persistence MPC for one config:
#   scripts/mpc/oracle_sweep.sh oracles_v1 "tx ca" "H96S_metric" persistence "1 2 3"
#
# CONFIGS (mpc_joint parameters; all with workers=16; H48 is the setting of the grid_v1 mpc_joint runs):
#   H48          horizon=48 (12 h), block=4, samples=32, iters=3                       est. ~2 h  (TIME 08:00:00)
#   H96          horizon=96, block=8, replan=4                                          ~4 h  (TIME 12:00:00)
#   H192         horizon=192, block=16, replan=4                                        ~8 h  (TIME 24:00:00)
#   S            horizon=48, samples=128, iters=5                                       ~10 h (TIME 24:00:00)
#   H96S         horizon=96, block=8, replan=4, samples=128, iters=5                    ~20 h (TIME 48:00:00)
#   H96S_metric  H96S planning on the evaluation metric: end_cost=metric, drop_w=0, overdue_w=0 (mpc_policy.py doc)
#                                                                                       ~20 h (TIME 48:00:00)
# FORECASTS: oracle (true future) and/or persistence (past data only).
# SEEDS: CEM seeds k = 1, 2, ... (mpc_joint seed=k-1, as the grid_v1 persist runs); runs are labelled
#   mpc_joint_<forecast>_<config>_s<k> and pooled per method in the paired table.
#
# Settings (environment variables, as for scripts/mpc/launch.sh):
#   CODE, PYTHON, SDC_RUNS, SBATCH_ARGS, DRY_RUN=1 (write the job scripts but do not submit)
#   TIME         override the per-config time limit for every job
#   Scenario (as scripts/eval_protocol.sh; written into every job script, so the jobs see them):
#   PV_KWP, INCLUDE_PV   on-site PV and PV actions (default: no PV)
#   CI_YEAR              carbon intensity year (2023: held-out test year, ca and tx)
#   WORKLOAD_FILE        workload trace (e.g. Alibaba_CPU_Data_Hourly_2.csv, held out)
#   MONTH, N_EPISODES    episode month (6 / cycle) and number of protocol episodes (default 6 and 10)
#   MODE=select  score on the tuning episodes instead (TUNE_SEED: seeds 2000..2004, otherwise the protocol's
#                scenario), to choose a config without looking at the protocol episodes. Labels get a _sel suffix.
# Outputs: $SDC_RUNS/EXP_NAME/{jobs,slurm}, $SDC_RUNS/EXP_NAME/eval/LOCATION/<label>, manifest.tsv.
set -euo pipefail
if [ $# -ne 5 ]; then sed -n '2,38p' "$0"; exit 1; fi
EXP=$1; LOCS=$2; CONFIGS=$3; FORECASTS=$4; SEEDS=$5

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "${CODE:-$SCRIPT_DIR/../..}" && pwd)
PY=${PYTHON:-$(command -v python || true)}
[ -n "$PY" ] || { echo "No python on PATH: activate your environment or set PYTHON=..."; exit 1; }
PY=$(command -v "$PY")
OUT=${SDC_RUNS:-$CODE/results/marl_runs}/$EXP
CPUS=17  # 16 rollout workers + the episode process
MODE=${MODE:-protocol}
case $MODE in protocol|select) ;; *) echo "MODE must be protocol or select (got $MODE)"; exit 1 ;; esac

config_params() {
  case $1 in
    H48)         echo "horizon=48,block=4" ;;
    H96)         echo "horizon=96,block=8,replan=4" ;;
    H192)        echo "horizon=192,block=16,replan=4" ;;
    S)           echo "horizon=48,block=4,samples=128,iters=5" ;;
    H96S)        echo "horizon=96,block=8,replan=4,samples=128,iters=5" ;;
    H96S_metric) echo "horizon=96,block=8,replan=4,samples=128,iters=5,end_cost=metric,drop_w=0,overdue_w=0" ;;
    *) echo "Unknown config $1 (valid: H48 H96 H192 S H96S H96S_metric)" >&2; exit 1 ;;
  esac
}
config_time() {
  case $1 in
    H48) echo 08:00:00 ;; H96) echo 12:00:00 ;; H192|S) echo 24:00:00 ;; H96S|H96S_metric) echo 48:00:00 ;;
  esac
}

mkdir -p "$OUT/jobs" "$OUT/slurm"
PV_KWP=${PV_KWP:-0}; INCLUDE_PV=${INCLUDE_PV:-0}; MONTH=${MONTH:-6}; N_EPISODES=${N_EPISODES:-10}
if [ "$INCLUDE_PV" = 1 ]; then
  awk -v k="$PV_KWP" 'BEGIN { exit !(k > 0) }' || { echo "INCLUDE_PV=1 needs PV_KWP > 0"; exit 1; }
fi
case $MONTH in all) echo "MONTH=all is for training; use a fixed month or MONTH=cycle"; exit 1 ;; esac
SCEN_ENV="PV_KWP=$PV_KWP INCLUDE_PV=$INCLUDE_PV MONTH=$MONTH N_EPISODES=$N_EPISODES${CI_YEAR:+ CI_YEAR=$CI_YEAR}${WORKLOAD_FILE:+ WORKLOAD_FILE=$WORKLOAD_FILE}"
# Select mode: the same scenario on the tuning seeds (one episode per month with cycle).
SEL_N=5; if [ "$MONTH" = cycle ]; then SEL_N=12; fi
SEL_ARGS="--month $MONTH --pv_capacity_kWp $PV_KWP$([ "$INCLUDE_PV" = 1 ] && echo ' --include_pv' || true)${CI_YEAR:+ --ci_year $CI_YEAR}${WORKLOAD_FILE:+ --workload_file $WORKLOAD_FILE}"
echo "EXP $EXP | mpc_joint sweep ($MODE) | $SCEN_ENV | code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') | python $PY | out $OUT"
for loc in $LOCS; do
  case $loc in az|ny|tx|ca) ;; *) echo "LOCATION must be az, ny, tx or ca (got $loc)"; exit 1 ;; esac
  for cfg in $CONFIGS; do
    params=$(config_params "$cfg")
    for fc in $FORECASTS; do
      case $fc in oracle|persistence) ;; *) echo "FORECAST must be oracle or persistence (got $fc)"; exit 1 ;; esac
      for k in $SEEDS; do
        label=mpc_joint_${fc}_${cfg}_s$k
        [ "$MODE" = select ] && label=mpc_joint_${fc}_${cfg}_sel_s$k
        spec="mpc_joint:$params,forecast=$fc,workers=16,seed=$((k - 1))=$label"
        name=${label}_$loc
        eval_dir="$OUT/eval/$loc/$label"
        if [ "$MODE" = protocol ]; then
          run="\"$CODE/scripts/eval_protocol.sh\" $loc \"$eval_dir\" \"$spec\""
        else  # the protocol's scenario on the tuning seeds (sdc_eval.constants.TUNE_SEED)
          run="\"$PY\" -u eval_compare.py --policy noop --policy \"$spec\" --location $loc --days_per_episode 30 \\
    $SEL_ARGS --n_episodes $SEL_N --seed 2000 --out_dir \"$eval_dir\""
        fi
        cat > "$OUT/jobs/$name.sh" <<JOB
#!/bin/bash
cd "$CODE"
export PATH="$(dirname "$PY"):\$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export $SCEN_ENV
echo "host \$(hostname)  start \$(date)  code \$(git rev-parse --short HEAD 2>/dev/null)  restart \${SLURM_RESTART_COUNT:-0}"
$run
echo "exit \$?  end \$(date)"
JOB
        chmod +x "$OUT/jobs/$name.sh"
        if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry run) $OUT/jobs/$name.sh"; continue; fi
        # shellcheck disable=SC2086
        jid=$(sbatch --parsable -J "sdc-$EXP-$name" --cpus-per-task=$CPUS --mem=32G --time=${TIME:-$(config_time "$cfg")} \
              --requeue ${SBATCH_ARGS:-} -o "$OUT/slurm/$name.log" "$OUT/jobs/$name.sh")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "mpc_joint" "$loc" "$label" "$spec" "$(date -Is)" >> "$OUT/manifest.tsv"
        echo "$jid $name"
      done
    done
  done
done
