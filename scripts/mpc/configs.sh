# Joint-MPC configs shared by scripts/mpc/oracle_sweep.sh and scripts/mpc/mppi_launch.sh (sourced, not run).
# The parameters are the same for mpc_joint (CEM, mpc_policy.py) and mppi (mppi_policy.py); H48 is the setting of
# the grid_v1 mpc_joint runs.
MPC_CONFIGS="H48 H96 H192 S H96S H96S_metric"

config_params() {
  case $1 in
    H48)         echo "horizon=48,block=4" ;;
    H96)         echo "horizon=96,block=8,replan=4" ;;
    H192)        echo "horizon=192,block=16,replan=4" ;;
    S)           echo "horizon=48,block=4,samples=128,iters=5" ;;
    H96S)        echo "horizon=96,block=8,replan=4,samples=128,iters=5" ;;
    H96S_metric) echo "horizon=96,block=8,replan=4,samples=128,iters=5,end_cost=metric,drop_w=0,overdue_w=0" ;;
    *) echo "Unknown config $1 (valid: $MPC_CONFIGS)" >&2; return 1 ;;
  esac
}

# Time limit for the whole protocol (10 episodes in one job, scripts/mpc/oracle_sweep.sh).
config_time() {
  case $1 in
    H48) echo 08:00:00 ;; H96) echo 12:00:00 ;; H192|S) echo 24:00:00 ;; H96S|H96S_metric) echo 48:00:00 ;;
  esac
}
