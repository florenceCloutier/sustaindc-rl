"""Checks for full-year episodes and the held-out evaluation options: month 'all' (training) starts anywhere in the
year without running past Dec 31, fixed months keep their episodes (July unchanged), month 'cycle' in eval_compare
gives one episode per month, --workload_file changes the inputs, and every trainer runs with --month all.
Run from the repo root: python tests/verify_full_year.py
Optional: REF_CODE=<checkout of the previous code> also compares July episodes with it (default: skipped)."""
import json, os, random, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml

from sdc_eval.protocol import episode_months, parse_month
from sdc_eval.rollout import run_episode
from sdc_eval.policies import NoOpPolicy
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = {**yaml.safe_load(f), 'agents': list(AGENTS)}
TMP = tempfile.mkdtemp(prefix='verify_full_year_')
PY = sys.executable
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def start_day(sdc, seed):
    random.seed(seed); np.random.seed(seed)
    sdc.reset()
    return int(sdc.ls_env.current_day)


def run(cmd, timeout=3600):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout + out.stderr


def main():
    print("1. Month specs")
    check("parse_month", parse_month('6') == 6 and parse_month('all') == 'all' and parse_month('cycle') == 'cycle'
          and parse_month(None) is None)
    try:
        parse_month('13'); bad = False
    except ValueError:
        bad = True
    check("parse_month rejects 13", bad)
    check("cycle: episode i in month i % 12", episode_months('cycle', 14) == list(range(12)) + [0, 1])
    check("a fixed month repeats", episode_months(6, 3) == [6, 6, 6])

    print("2. month 'all': start days over the whole year, never past the data")
    sdc = SustainDC({**BASE, 'month': 'all', 'days_per_episode': 30})
    days = [start_day(sdc, s) for s in range(300)]
    last = 364 - 30 - 1
    check("start days span the year", min(days) < 30 and max(days) > last - 30, f"min {min(days)}, max {max(days)}")
    check("no start after the last day that fits a 30-day episode", max(days) <= last, f"max {max(days)} <= {last}")

    print("3. December episodes stay inside the year (no wrap of the managers)")
    sdc = SustainDC({**BASE, 'month': 11, 'days_per_episode': 30})
    ok = True
    for seed in (1, 2, 3):
        random.seed(seed); np.random.seed(seed)
        sdc.reset()
        prev = (sdc.ci_m.time_step, sdc.workload_m.time_step, sdc.weather_m.time_step)
        while True:
            _, _, _, truncs, _ = sdc.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
            now = (sdc.ci_m.time_step, sdc.workload_m.time_step, sdc.weather_m.time_step)
            ok &= all(b == a + 1 for a, b in zip(prev, now))
            prev = now
            if truncs['__all__']:
                break
    check("every manager advances by one step every step (seeds 1-3)", ok)

    print("4. July is unchanged")
    sdc = SustainDC({**BASE, 'month': 6, 'days_per_episode': 2})
    days = [start_day(sdc, s) for s in range(200)]
    check("July starts within +-7 days of July 1 (day 181)", min(days) == 174 and max(days) == 188,
          f"{min(days)}..{max(days)}")
    ref = os.environ.get('REF_CODE')
    if ref:
        code = ("import sys, os, json, random; sys.path.insert(0, os.getcwd()); import numpy as np, yaml; "
                "from sdc_eval.rollout import run_episode; from sdc_eval.policies import NoOpPolicy; "
                "from sustaindc_env import SustainDC; from sustaindc_single_env import AGENTS; "
                "c = {**yaml.safe_load(open('harl/configs/envs_cfgs/sustaindc.yaml')), 'agents': list(AGENTS), "
                "'location': 'tx', 'month': 6, 'days_per_episode': 2}; s = SustainDC(c); "
                "out = [run_episode(s, NoOpPolicy(), seed) for seed in (1000, 1001)]; "
                "print(json.dumps([[m['co2'], fp] for m, fp in out]))")
        rc_new, new = run([PY, '-W', 'ignore', '-c', code])
        out = subprocess.run([PY, '-W', 'ignore', '-c', code], capture_output=True, text=True, cwd=ref)
        same = rc_new == 0 and out.returncode == 0 and new.strip().splitlines()[-1] == out.stdout.strip().splitlines()[-1]
        check(f"July noop episodes identical to {ref}", same)

    print("5. eval_compare: month cycle and --workload_file")
    out_a, out_b = os.path.join(TMP, 'cycle_a'), os.path.join(TMP, 'cycle_b')
    base = [PY, '-W', 'ignore', 'eval_compare.py', '--policy', 'noop', '--policy', 'rule:ls,dc,bat=rule',
            '--location', 'tx', '--month', 'cycle', '--days_per_episode', '1', '--pv_capacity_kWp', '0',
            '--n_episodes', '12', '--seed', '1000', '--no-save_decisions']
    rc_a, log_a = run(base + ['--out_dir', out_a])
    rc_b, log_b = run(base + ['--workload_file', 'Alibaba_CPU_Data_Hourly_2.csv', '--ci_year', '2023',
                              '--out_dir', out_b])
    check("eval_compare --month cycle runs", rc_a == 0 and 'Exogenous inputs identical across policies: yes' in log_a,
          log_a.strip().splitlines()[-1] if log_a.strip() else '')
    if rc_a == 0 and rc_b == 0:
        import csv
        rows = lambda d: [r for r in csv.DictReader(open(os.path.join(d, 'episodes.csv'))) if r['policy'] == 'noop']
        a, b = rows(out_a), rows(out_b)
        check("12 distinct episodes (one per month)", len({r['inputs_fingerprint'] for r in a}) == 12)
        check("held-out workload + 2023 CI change every episode's inputs",
              all(x['inputs_fingerprint'] != y['inputs_fingerprint'] for x, y in zip(a, b)))
        s = json.load(open(os.path.join(out_b, 'summary.json')))
        check("summary records the scenario", s['scenario']['month'] == 'cycle'
              and s['scenario']['workload_file'] == 'Alibaba_CPU_Data_Hourly_2.csv' and s['scenario']['ci_year'] == 2023)
    else:
        check("eval_compare with --workload_file runs", False, log_b[-500:])

    print("6. Trainers run with --month all")
    common = ['--location', 'tx', '--month', 'all', '--days_per_episode', '1']
    rc, log = run([PY, '-W', 'ignore', 'train_single_agent.py', '--algo', 'ppo', '--n_envs', '2', '--total_timesteps',
                   '512', '--eval_freq', '512', '--n_eval_episodes', '1', '--checkpoint_freq', '100000',
                   '--exp_name', 'fy', '--run_id', 'fy', '--out_root', TMP, '--pv_capacity_kWp', '0'] + common)
    cfg = [os.path.join(r, 'config.json') for r, _, f in os.walk(TMP) if 'config.json' in f and 'single' in r]
    month_saved = json.load(open(cfg[0]))['env_args']['month'] if cfg else None
    check("train_single_agent.py --month all", rc == 0 and month_saved == 'all', log.strip().splitlines()[-1][:200]
          if rc else f"saved month {month_saved!r}")
    rc, log = run([PY, '-W', 'ignore', 'train_sustaindc.py', '--algo', 'mappo', '--exp_name', 'fy', '--run_id', 'fy',
                   '--log_dir', TMP, '--seed', '1', '--num_env_steps', '192', '--episode_length', '96',
                   '--n_rollout_threads', '2', '--n_eval_rollout_threads', '1', '--eval_episodes', '1',
                   '--eval_interval', '1', '--hidden_sizes', '[16,16]'] + common)
    cfg = [os.path.join(r, 'config.json') for r, _, f in os.walk(os.path.join(TMP, 'sustaindc')) if 'config.json' in f
           and 'mappo' in r]
    month_saved = json.load(open(cfg[0]))['env_args']['month'] if cfg else None
    check("train_sustaindc.py (MAPPO) --month all", rc == 0 and month_saved == 'all',
          log.strip().splitlines()[-1][:200] if rc else f"saved month {month_saved!r}")
    rc, log = run([PY, '-W', 'ignore', 'train_mpc_value.py', '--iters', '1', '--episodes', '2', '--workers', '2',
                   '--val_episodes', '1', '--n_eval_episodes', '1', '--fits', '1', '--exp_name', 'fy', '--run_id', 'fy',
                   '--out_root', TMP, '--pv_capacity_kWp', '0'] + common)
    check("train_mpc_value.py --month all", rc == 0, log.strip().splitlines()[-1][:200] if log.strip() else '')

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
