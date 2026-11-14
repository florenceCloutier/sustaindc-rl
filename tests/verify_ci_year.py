"""Checks for ci_year (held-out carbon intensity year): the default still loads the 2022 data, ci_year=2023 loads the
2023 files for ca/tx (aligned to local time like 2022), unknown years fail clearly, and episodes run on 2023 CI with
the same weather and workload as 2022.
Run from the repo root: python tests/verify_ci_year.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import yaml

from sdc_eval.policies import NoOpPolicy
from sdc_eval.rollout import run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = {**yaml.safe_load(f), 'days_per_episode': 2, 'agents': list(AGENTS)}
OFFSET = {'ca': -8, 'tx': -6}  # EPW UTC offsets of San Jose and Dallas-Fort Worth
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def ci_series(loc, year=None):
    env = SustainDC({**BASE, 'location': loc, 'ci_year': year})
    return env.ci_m.original_data if hasattr(env.ci_m, 'original_data') else env.ci_m.carbon_smooth, env


def expected(loc, year):
    name = {'ca': 'CA', 'tx': 'TX'}[loc] + ('' if year == 2022 else f'_{year}')
    v = pd.read_csv(f'data/CarbonIntensity/{name}_NG_&_avgCI.csv')['avg_CI'].values[:8760]
    v = np.roll(np.nan_to_num(v, nan=np.nanmean(v)), OFFSET[loc])
    return np.interp(np.linspace(0, len(v), len(v) * 4), range(len(v)), v)  # CI_Manager: hourly -> 15-min steps


print("1. Default and explicit 2022 load the 2022 files")
for loc in ('ca', 'tx'):
    s_default, env = ci_series(loc)
    s_2022, _ = ci_series(loc, 2022)
    check(f"{loc}: ci_year None == ci_year 2022 == the 2022 file (aligned to local time)",
          np.array_equal(s_default, s_2022) and np.allclose(s_default, expected(loc, 2022)) and env.ci_m.ci_shift_hours == OFFSET[loc])

print("2. ci_year=2023 loads the 2023 files")
for loc in ('ca', 'tx'):
    s23, env = ci_series(loc, 2023)
    s22, _ = ci_series(loc, 2022)
    jul = slice(181 * 96, 212 * 96)
    check(f"{loc}: 2023 file, same local-time alignment, different from 2022",
          np.allclose(s23, expected(loc, 2023)) and env.ci_m.ci_shift_hours == OFFSET[loc] and not np.allclose(s23, s22),
          f"July mean CI 2022 {s22[jul].mean():.1f} vs 2023 {s23[jul].mean():.1f} g/kWh")

print("3. A year without data fails clearly")
try:
    SustainDC({**BASE, 'location': 'az', 'ci_year': 2023}); ok = False; msg = 'no error'
except ValueError as e:
    ok, msg = 'does not exist' in str(e), str(e)[:90]
check("az 2023 -> ValueError naming the missing file", ok, msg)

print("4. Episodes on 2023 CI: same weather and workload, different CI")
for loc in ('ca', 'tx'):
    trace = {}
    for year in (2022, 2023):
        rows = []
        env = SustainDC({**BASE, 'location': loc, 'ci_year': year})
        m, _ = run_episode(env, NoOpPolicy(), 1000,
                           trace=lambda k, info, actions: rows.append((info['ls_original_workload'], info['outside_temp'], info['bat_avg_CI'])))
        trace[year] = (np.array(rows), m)
    a, b = trace[2022][0], trace[2023][0]
    check(f"{loc}: same workload and weather, different CI over a 2-day noop episode",
          len(a) == len(b) and np.allclose(a[:, :2], b[:, :2]) and not np.allclose(a[:, 2], b[:, 2]),
          f"CO2 2022 {trace[2022][1]['co2']:.0f} kg vs 2023 {trace[2023][1]['co2']:.0f} kg")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
