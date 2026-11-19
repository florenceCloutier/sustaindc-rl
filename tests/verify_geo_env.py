"""Checks for the geo-distributed engine (geo_sustaindc_env.py): one site matches SustainDC, shared clock, work
conservation, and sites seeing the work routed to them. Run from the repo root: python tests/verify_geo_env.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np

from geo_sustaindc_env import GEO_SITE_DEFAULTS, GeoSustainDC
from sustaindc_env import SustainDC

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def seeded(seed):
    random.seed(seed); np.random.seed(seed)


def random_site_actions(rng, n):
    return [{a: int(rng.integers(3)) for a in ('agent_ls', 'agent_dc', 'agent_bat')} for _ in range(n)]


print("1. One site reproduces SustainDC exactly (same seed, same actions)")
for cap in (1.0, 1.3):
    cfg = {**GEO_SITE_DEFAULTS, 'location': 'az', 'days_per_episode': 2, 'datacenter_capacity_mw': cap}
    sdc = SustainDC(dict(cfg)); geo = GeoSustainDC({'sites': [{'location': 'az', 'datacenter_capacity_mw': cap}],
                                                    'days_per_episode': 2})
    seeded(3); sdc.reset(); seeded(3); geo.reset()
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    same = True; n = 0
    while True:
        geo.route([1.0])  # one site: its observations must not change
        g0 = geo.sites[0]
        same &= all(np.array_equal(x, y) for x, y in ((sdc.ls_state, g0.ls_state), (sdc.dc_state, g0.dc_state),
                                                       (sdc.bat_state, g0.bat_state)))
        _, _, _, tr, info = sdc.step(random_site_actions(rng_a, 1)[0])
        _, _, _, gtr, ginfo = geo.step({'sites': random_site_actions(rng_b, 1)})
        a, b = info['__common__'], ginfo['sites'][0]
        for key in ('bat_CO2_footprint', 'ls_shifted_workload', 'ls_tasks_in_queue', 'ls_tasks_dropped', 'dc_crac_setpoint'):
            same &= a[key] == b[key]
        n += 1
        if tr['__all__']:
            break
    check(f"capacity {cap} MW: identical observations, CO2, utilization, queue, drops, setpoint over {n} steps",
          same and gtr)

print("2. Sites share one clock (same UTC instant every step)")
geo = GeoSustainDC({'sites': [{'location': 'ca'}, {'location': 'ny'}, {'location': 'wa'}], 'days_per_episode': 1})
seeded(1); geo.reset()
ok = True
while True:
    utc = [s.t_m.day * 24 + s.t_m.hour - off for s, off in zip(geo.sites, geo.utc_offsets)]
    ok &= max(utc) - min(utc) < 1e-9
    _, _, _, tr, _ = geo.step({'dispatch': [1, 1, 1]})
    if tr:
        break
check("ca / ny / wa local clocks stay at the same UTC time", ok, f"offsets {geo.utc_offsets.tolist()}")

print("3. Work is conserved (pool and each site's tasks), also with overflow and unequal capacities")
for caps, disp in (([1.0, 1.0, 1.0], [1, 0, 0]), ([1.0, 2.5, 0.7], [0.2, 0.5, 0.3]), ([1.0, 2.5, 0.7], 'random')):
    geo = GeoSustainDC({'sites': [{'location': l, 'datacenter_capacity_mw': c} for l, c in zip(('ca', 'ny', 'wa'), caps)],
                        'days_per_episode': 2, 'flexible_load': 0.8})
    seeded(2); geo.reset(); rng = np.random.default_rng(1)
    pool_ok = site_ok = carry_ok = True; overflow = 0
    prev_carry = np.zeros(3)
    while True:
        pool = geo.pool_units
        base = [(geo.non_shiftable[i] + geo.local_flexible[i], len(s.ls_env.tasks_queue)) for i, s in enumerate(geo.sites)]
        w = rng.dirichlet(np.ones(3)) if disp == 'random' else disp
        _, _, _, tr, info = geo.step({'dispatch': w, 'sites': random_site_actions(rng, 3)})
        g = info['geo']
        pool_ok &= abs(g['routed_in_units'].sum() + g['carry_units'].sum() - prev_carry.sum() - pool) < 1e-6
        carry_ok &= np.all(g['carry_units'] < geo.units_per_task + 1e-9) and np.all(g['carry_units'] > -1e-6)
        prev_carry = g['carry_units']
        for i, s in enumerate(geo.sites):
            si = info['sites'][i]
            inflow = base[i][0] + g['routed_tasks'][i] + base[i][1]
            outflow = round(si['ls_shifted_workload'] * 100) + si['ls_tasks_in_queue'] + si['ls_tasks_dropped']
            site_ok &= inflow == outflow
            overflow += si['ls_overflow_tasks']
        if tr:
            break
    check(f"capacities {caps}, dispatch {disp}: pool units conserved, carry < 1 task", pool_ok and carry_ok)
    check(f"capacities {caps}, dispatch {disp}: local + routed + queued = executed + queued + dropped", site_ok,
          f"{overflow} overflow tasks forced into queues")

print("4. No migration: migratable_fraction 0 == 'local' dispatcher")
sites = [{'location': 'ca'}, {'location': 'ny'}]
co2 = {}
for name, cfg, disp in (('mf0', {'migratable_fraction': 0.0}, [1, 0]), ('local', {}, None)):
    geo = GeoSustainDC({'sites': sites, 'days_per_episode': 2, **cfg})
    seeded(4); geo.reset(); rng = np.random.default_rng(2); total = 0.0; routed_zero = True
    while True:
        w = disp if disp is not None else geo.local_weights()
        _, _, _, tr, info = geo.step({'dispatch': w, 'sites': random_site_actions(rng, 2)})
        total += sum(i['bat_CO2_footprint'] for i in info['sites'])
        routed_zero &= name != 'mf0' or info['geo']['routed_tasks'].sum() == 0
        if tr:
            break
    co2[name] = total
check("pool is empty when migratable_fraction = 0", routed_zero)
check("same CO2 with no pool and with the 'local' dispatcher", abs(co2['mf0'] - co2['local']) < 1e-6,
      f"{co2['mf0'] / 1e3:.2f} vs {co2['local'] / 1e3:.2f} kg")

print("7. Sites see the work routed to them (observations and controller estimates)")
from baseline_policies import expected_utilization
from utils.joint_obs import BAT_WORKLOAD_IDX, DC_WORKLOAD_IDX, LS_WORKLOAD_IDX
geo = GeoSustainDC({'sites': [{'location': 'ca'}, {'location': 'ny'}, {'location': 'wa'}], 'days_per_episode': 1})
seeded(8); geo.reset()
obs_ok = est_ok = idem_ok = True; shown = 0
while True:
    info1 = geo.route([1, 0, 0]); carry1 = geo.carry.copy()
    info2 = geo.route([1, 0, 0])
    idem_ok &= np.array_equal(info1['routed_tasks'], info2['routed_tasks']) and np.allclose(carry1, geo.carry)
    expected = []
    for i, s in enumerate(geo.sites):
        ls = s.ls_env
        planned = float(ls.workload) + (geo.local_flexible[i] + info2['routed_tasks'][i] - ls.task_split()[1]) / 100
        obs_ok &= all(abs(float(x) - planned) < 1e-6 for x in (s.ls_state[LS_WORKLOAD_IDX], s.dc_state[DC_WORKLOAD_IDX],
                                                                s.bat_state[BAT_WORKLOAD_IDX]))
        expected.append(expected_utilization(s, 1))
    shown += info2['routed_tasks'][0] > geo.migratable[0]
    _, _, _, tr, info = geo.step({'sites': [{'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}] * 3})
    for i, si in enumerate(info['sites']):
        if si['ls_overdue_penalty'] == 0 and si['ls_overflow_tasks'] == 0:
            est_ok &= abs(si['ls_shifted_workload'] - expected[i]) < 0.011  # ceil/floor rounding of the local split
    if tr:
        break
check("ls / dc / bat workload entries = local + net routed-in work", obs_ok and shown > 0,
      f"{shown} steps with work routed into ca")
check("cooling controllers' utilization estimate matches what the site runs", est_ok)
check("route() is idempotent before step()", idem_ok)


print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
