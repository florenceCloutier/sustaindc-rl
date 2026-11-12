"""Checks for the job-level geo env (geo_jobs.py), its heuristics (geo_job_baselines.py) and the LP bound
(geo_lp_oracle.py), on DCcluster-Opt data (run scripts/import_dccluster_data.py first).
Run from the repo root: python tests/verify_geo_jobs.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from geo_jobs import GeoJobsDC
from geo_job_baselines import JOB_DISPATCHERS, JobDispatcher, run_job_episode
from geo_lp_oracle import energy_curves, exo_traces, model_co2_kg, solve_bound
from utils.utils_cf import obtain_paths

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def seeded(seed):
    random.seed(seed); np.random.seed(seed)


REGIONS = ['US-CAL-CISO', 'DE-LU', 'SG']


def make(days=2, **kw):
    return GeoJobsDC({'sites': [{'location': f'dcc:{r}'} for r in REGIONS], 'month': 6, 'days_per_episode': days,
                      'target_util': 0.5, **kw})


print("1. DCcluster-Opt data")
check("dcc locations resolve to converted files", obtain_paths('dcc:SG') == ['DCC_SG_2023', 'DCC_SG_2023.epw'])
geo = make()
seeded(1); geo.reset()
starts = [(s.t_m.day, s.t_m.hour) for s in geo.sites]
utc = [d * 24 + h - o for (d, h), o in zip(starts, geo.utc_offsets)]
check("all sites start at the same UTC instant", len(set(utc)) == 1, f"local starts {starts}")
check("dated weather is not rolled by random days", all(s.weather_m.last_roll_days == 0 for s in geo.sites))
ci_noon = []
for k in range(4 * 24):  # CA's solar dip: CI around local noon below CI around local midnight
    ci_noon.append(geo.sites[0].ci_m.carbon_smooth[geo.sites[0].ci_m.time_step + k])
h0 = geo.sites[0].t_m.hour
hours = (h0 + np.arange(96) / 4) % 24
check("CA CI dips at local noon (time zones aligned)",
      np.mean(np.array(ci_noon)[(hours > 11) & (hours < 15)]) < np.mean(np.array(ci_noon)[(hours < 3) | (hours > 22)]))

print("2. Job layer invariants (local and greedy dispatchers)")
for kind in ('local', 'greedy_ci'):
    geo = make()
    seeded(2); geo.reset()
    d = JobDispatcher(kind); d.reset(geo)
    site_seen = {}
    ok_cap, ok_stay, ok_util = True, True, True
    while True:
        v = geo.pending_view()
        _, _, _, tr, info = geo.step({'assign': d.assign(geo, v), 'sites': None})
        ok_cap &= bool(np.all(geo.committed_load() <= geo.cap + 1e-9))
        run = np.flatnonzero(geo.status >= 2)
        for j in run[np.isin(run, list(site_seen.keys()), invert=True)]:
            site_seen[j] = geo.site[j]
        ok_stay &= all(geo.site[j] == s for j, s in list(site_seen.items())[-200:])
        ok_util &= bool(np.all(info['utilization'] <= geo.cap + 1e-9))
        if tr:
            break
    m = geo.episode_metrics()
    check(f"{kind}: committed load never exceeds cap", ok_cap)
    check(f"{kind}: utilization never exceeds cap", ok_util)
    check(f"{kind}: a started job never changes site", ok_stay)
    check(f"{kind}: every job started (drain period)", m['jobs_unstarted'] == 0, f"{m['jobs_unstarted']} unstarted")
    work_util = np.sum(np.array(geo.util_history) * geo.cap_w[None, :]) * 0.25 / 1e6
    check(f"{kind}: utilization integrates to the jobs' work", abs(work_util - m['job_mwh']) < 1e-6 * m['job_mwh'],
          f"{work_util:.4f} vs {m['job_mwh']:.4f} MWh")

geo = make(rigid_fraction=1.0)
m = run_job_episode(geo, JobDispatcher('greedy_ci'), seed=3)
moved = (geo.status >= 2) & (geo.site != geo.origin)
check("rigid jobs only run at their origin", not moved.any(), f"{int(moved.sum())} moved")

print("3. Same seed, same exogenous inputs and jobs for every dispatcher")
geo = make()
fps = []
for kind in JOB_DISPATCHERS:
    seeded(4); geo.reset()
    fps.append((geo.n_jobs, float(geo.power_w.sum()), int(geo.origin.sum()),
                float(geo.sites[1].ci_m.carbon_smooth[geo.sites[1].ci_m.time_step])))
check("jobs and CI identical across resets with the same seed", len(set(fps)) == 1)

print("4. LP bounds and calibration")
geo = make()
m_local = run_job_episode(geo, JobDispatcher('local'), seed=5)
util_local = np.array(geo.util_history)
m_merit = run_job_episode(geo, JobDispatcher('merit'), seed=5)
seeded(5); geo.reset()
tr = exo_traces(geo, geo.horizon)
cv = energy_curves(geo, tr)
calib = model_co2_kg(geo, tr, util_local)
check("energy model reproduces the simulated CO2 of a schedule", abs(calib / m_local['co2_kg'] - 1) < 1e-6,
      f"{calib:.2f} vs {m_local['co2_kg']:.2f} kg")
lp = {v: solve_bound(geo, v, traces=tr, curves=cv, verbose=False) for v in ('time', 'geo', 'geo_time')}
check("all LPs solve", all(r['status'] == 0 for r in lp.values()), {v: r['message'][:40] for v, r in lp.items()})
b = {v: r['co2_kg'] for v, r in lp.items()}
check("bound ordering: geo_time <= geo and geo_time <= time", b['geo_time'] <= b['geo'] + 1e-6 and b['geo_time'] <= b['time'] + 1e-6,
      f"{ {k: round(v, 1) for k, v in b.items()} }")
check("bound below every simulated heuristic", b['geo_time'] <= min(m_local['co2_kg'], m_merit['co2_kg']),
      f"bound {b['geo_time']:.1f}, merit {m_merit['co2_kg']:.1f}, local {m_local['co2_kg']:.1f}")
exact = model_co2_kg(geo, tr, lp['geo_time']['util'])
check("tangent underestimation small (< 2%) and on the safe side", 0 <= exact / b['geo_time'] - 1 < 0.02,
      f"exact model at LP schedule {exact:.1f} vs LP {b['geo_time']:.1f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
