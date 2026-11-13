"""Checks for the index-policy pieces: bound with battery and cooling (geo_lp_oracle.site_flex), per-site queues
(GeoJobsDC(routing='site_queue')) and the index policy (geo_index.py). Run from the repo root:
    python tests/verify_geo_index.py
Needs the DCcluster-Opt data (scripts/import_dccluster_data.py). Takes ~10 minutes (1-day episodes)."""
import os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np

from geo_index import IndexPolicy, run_index_episode
from geo_job_baselines import JobDispatcher, JobRouter, run_job_episode, run_queue_episode
from geo_jobs import GeoJobsDC
from geo_lp_oracle import (FINE_LOADS, energy_curves, envelope_lines, exo_traces, sampled_lines, solve_bound,
                           tangent_lines)

REGIONS = ['US-CAL-CISO', 'DE-LU', 'CL-SIC', 'SG', 'AU-NSW']
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make(routing, util=0.5, days=1, regions=REGIONS):
    return GeoJobsDC({'sites': [{'location': f'dcc:{r}:2023'} for r in regions], 'month': 6,
                      'days_per_episode': days, 'target_util': util, 'routing': routing})


print("1. Energy lines stay below the curve, including kinks")
x = FINE_LOADS
f = 100 + 50 * x + 30 * x ** 2 - 4 * np.maximum(x - 0.6, 0)          # convex with a concave kink at 0.6
b, a = sampled_lines(f)
xx = np.linspace(0, 1, 2001)
ff = 100 + 50 * xx + 30 * xx ** 2 - 4 * np.maximum(xx - 0.6, 0)
check("sampled lines below a kinked curve everywhere", np.max(np.max(a[:, None] + b[:, None] * xx, axis=0) - ff) <= 1e-9)

print("2. Bound with site flexibility (1-day dccluster5)")
geo = make('pool')
sims = {}
for disp, spec in (('merit', 'rule:dc,bat'), ('forecast_greedy', 'oracle:dc,bat')):
    sims[(disp, spec)] = run_job_episode(geo, JobDispatcher(disp), 1, local_spec=spec)['co2_kg']
random.seed(1); np.random.seed(1); geo.reset()
tr = exo_traces(geo, geo.horizon)
cv = energy_curves(geo, tr)
env = envelope_lines(geo, tr)
# Validity: both line sets stay below the true energy curve at the site's own setpoint (the best-setpoint curve is
# lower still). Checked on a 201-point load grid for a sample of (site, step).
from baseline_policies import dc_power
from geo_sustaindc_env import STEP_HOURS
xs = np.linspace(0, 1, 201)
worst = {'fixed': -np.inf, 'envelope': -np.inf}
for i in range(geo.n_sites):
    for t in range(0, geo.horizon, 19):
        key = (round(tr[i]['amb'][t], 2), round(tr[i]['wb'][t], 2))
        true = np.array([sum(dc_power(geo.sites[i], u, tr[i]['stpt'], *key)) for u in xs]) / 1e3 * STEP_HOURS
        for name, (sl, ic) in (('fixed', tangent_lines(cv)), ('envelope', env[:2])):
            worst[name] = max(worst[name], float(np.max(np.max(ic[i, t][:, None] + sl[i, t][:, None] * xs, axis=0) - true)))
check("fixed and envelope lines stay below the true fixed-setpoint curve", max(worst.values()) <= 1e-6,
      {k: f'{v:.4f} kWh' for k, v in worst.items()})
bnd = {flex: solve_bound(geo, 'geo_time', traces=tr, curves=cv, verbose=False, site_flex=flex,
                         lines=env[:2] if 'cooling' in flex else None)['co2_kg']
       for flex in ('none', 'battery', 'cooling', 'battery+cooling')}
check("more flexibility, lower bound: battery+cooling <= battery, cooling <= none",
      bnd['battery+cooling'] <= min(bnd['battery'], bnd['cooling']) + 1e-6
      and max(bnd['battery'], bnd['cooling']) <= bnd['none'] + 1e-6, {k: round(v, 1) for k, v in bnd.items()})
check("full bound below runs whose sites use battery and cooling", all(bnd['battery+cooling'] <= v for v in sims.values()),
      {f'{k[0]} {k[1]}': round(v, 1) for k, v in sims.items()})

print("3. Per-site queues")
for util in (0.1, 0.2):
    gp = make('pool', util)
    a = run_job_episode(gp, JobDispatcher('local'), 3)
    moved = int(((gp.status >= 2) & (gp.site != gp.origin)).sum())
    b = run_queue_episode(make('site_queue', util), JobRouter('origin'), 3)
    check(f"util {util}: origin routing + EDF == pool `local` (no fallback moves)",
          moved == 0 and abs(a['co2_kg'] - b['co2_kg']) < 1e-6, f"{a['co2_kg']:.3f} vs {b['co2_kg']:.3f} kg")
gq = make('site_queue')
routed_at = {}
orig_route = gq.route


def route_and_record(route):
    arr = gq.arrivals()
    orig_route(route)
    for j in arr:
        routed_at[int(j)] = int(gq.site[j])


gq.route = route_and_record
m = run_queue_episode(gq, JobRouter('merit'), 1)
started = np.flatnonzero(gq.status >= 2)
check("a routed job runs at the site it was routed to", all(gq.site[j] == routed_at[j] for j in started))
check("every job starts (drain period) and load stays within capacity",
      m['jobs_unstarted'] == 0 and np.max(gq.util_history) <= gq.cap + 1e-9)
work = float(np.sum(gq.size[started, gq.site[started]] * gq.dur[started]))
check("utilization integrates to the jobs' work", abs(np.sum(gq.util_history) - work) < 1e-6 * max(work, 1),
      f"{np.sum(gq.util_history):.4f} vs {work:.4f} load-steps")

print("4. Index policy")
pol = IndexPolicy(cooling='fixed')
mi = run_index_episode(gq, pol, 1)
check("every job starts, load within capacity", mi['jobs_unstarted'] == 0 and np.max(gq.util_history) <= gq.cap + 1e-9)
check("cost curves convex in the accepted amount (indexability)", mi['curve_violations'] == 0,
      f"{mi['lp_solves']} LP solves")
check("SLA violations no worse than the merit router (+0.5 pt)", mi['sla_violation_rate'] <= m['sla_violation_rate'] + 0.005,
      f"{mi['sla_violation_rate']:.4f} vs {m['sla_violation_rate']:.4f}")
check("CO2 below the merit router", mi['co2_kg'] < m['co2_kg'], f"{mi['co2_kg']:.1f} vs {m['co2_kg']:.1f} kg")
check("decision time per step under 2 s (hourly steps included)", mi['decision_s_max'] < 2.0,
      f"mean {mi['decision_s_mean']:.2f}s, hourly {mi['replan_s_mean']:.2f}s, max {mi['decision_s_max']:.2f}s")
g1 = make('site_queue', regions=REGIONS[:1])
m1 = run_index_episode(g1, IndexPolicy(cooling='fixed'), 1)
check("one site: everything runs there, every job on time or nearly", m1['job_mwh_migrated'] == 0 and m1['jobs_unstarted'] == 0,
      f"sla {m1['sla_violation_rate']:.4f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
