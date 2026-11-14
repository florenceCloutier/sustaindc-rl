#!/bin/bash
# Score policies on the shared IFT 6162 evaluation protocol, so results from different methods (MPC, MARL,
# single-agent, rules) go into one paired table.
#
# Protocol (fixed here, do not change it per method):
#   10 episodes, seeds 1000..1009, 30 days each, July (month index 6), PV off, one location (az, ny, ca or tx),
#   carbon intensity of 2022 (the year every method trains on).
#   Every run also scores noop on the same episodes; results are paired % changes vs noop, metric = work-adjusted CO2.
#
# Held-out test year (opt-in): CI_YEAR=2023 replays July 2023 carbon intensity (locations ca and tx) for policies
#   trained on 2022; weather and workload are unchanged. Report it next to the 2022 results, not merged with them.
#
# PV scenario (opt-in; results are only comparable with runs of the same PV settings):
#   PV_KWP=<kWp>     on-site PV capacity (default 0 = the protocol above, no PV)
#   INCLUDE_PV=1     PV actions: the battery agent dispatches PV (use / battery / resell); needs PV_KWP > 0.
#                    Unset: with PV_KWP > 0, the automatic PV rule (PV serves the DC, the surplus charges the battery).
#
# Usage (from anywhere; runs in the repo root):
#   scripts/eval_protocol.sh LOC OUT_DIR SPEC [SPEC ...]
# e.g. a single-agent run (SB3, best_model by default):
#   scripts/eval_protocol.sh az results/eval_protocol/az/single_ppo "sb3:results/sustaindc/az/single_ppo/<exp>/seed-00001-<id>=single_ppo_s1"
# On Slurm:
#   sbatch -c 4 --mem 16G -t 12:00:00 scripts/eval_protocol.sh az <out_dir> "<spec>"
#
# SPEC is any eval_compare.py policy spec (sb3:, harl:, mpc:, rule:, ...); add =<label> to name it in the table.
# Name seeds <label>_s<seed> so they are also pooled per method.
# To merge folders from several runs (same LOC) into one table:
#   python eval_compare.py --combine <out_dir_1> <out_dir_2> ... --out_dir <combined_dir>
set -euo pipefail
if [ $# -lt 3 ]; then
    sed -n '2,21p' "$0"
    exit 1
fi
LOC=$1; OUT=$2; shift 2
case $LOC in az|ny|ca|tx) ;; *) echo "LOC must be az, ny, ca or tx (got $LOC)"; exit 1 ;; esac
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
PV_KWP=${PV_KWP:-0}
PV_ARGS=(--pv_capacity_kWp "$PV_KWP")
CI_ARGS=()
if [ -n "${CI_YEAR:-}" ]; then CI_ARGS=(--ci_year "$CI_YEAR"); fi
if [ "${INCLUDE_PV:-0}" = 1 ]; then
    awk -v k="$PV_KWP" 'BEGIN { exit !(k > 0) }' || { echo "INCLUDE_PV=1 needs PV_KWP > 0"; exit 1; }
    PV_ARGS+=(--include_pv)
fi
cd "$(dirname "$0")/.."
POLICIES=(--policy noop)
for spec in "$@"; do POLICIES+=(--policy "$spec"); done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
echo "eval_protocol: $LOC | CI ${CI_YEAR:-2022} | PV ${PV_KWP} kWp$([ "${INCLUDE_PV:-0}" = 1 ] && echo ', PV actions') | code $(git rev-parse --short HEAD 2>/dev/null || echo '?') | host $(hostname) | $(date)"
python -u eval_compare.py "${POLICIES[@]}" \
    --location "$LOC" --month 6 --days_per_episode 30 "${PV_ARGS[@]}" ${CI_ARGS[@]+"${CI_ARGS[@]}"} \
    --n_episodes 10 --seed 1000 --out_dir "$OUT"
