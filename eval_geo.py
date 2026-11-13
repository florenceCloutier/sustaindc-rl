"""Compare dispatchers on GeoSustainDC (geo_sustaindc_env.py) on the same seeded episodes.

Every dispatcher is paired episode by episode with `local` (no migration), with the same per-site controllers, and
the table reports the mean % change and its 95% confidence interval. The exogenous inputs (workload, weather, carbon
intensity of every site) are checked to be identical across dispatchers.

Run from the repo root, e.g.:
    python eval_geo.py --sites ca,ny,wa --month 6 --days 7 --n_episodes 5 \\
        --dispatch local,capacity,greedy_ci,merit,sb3:results/geo_dispatcher/<run>:best_model=ppo --local rule:ls,dc,bat

Dispatchers decide every --dispatch_every steps (default 4 = hourly, as the learned dispatcher) and hold their weights
in between; `local` (no migration) is recomputed every step.

Work is counted in MWh of IT capacity (one site task = 1% of the site's capacity for 15 minutes).
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np

from geo_baselines import DISPATCHERS, GeoPolicy
from geo_sustaindc_env import STEP_HOURS, GeoSustainDC

METRICS = [  # (name, unit)
    ('co2', 'kgCO2'),
    ('total_energy', 'MWh'),
    ('it_energy', 'MWh'),
    ('hvac_energy', 'MWh'),
    ('water', 'L'),
    ('work_submitted', 'MWh-cap'),
    ('work_executed', 'MWh-cap'),
    ('work_dropped', 'MWh-cap'),
    ('work_queued_end', 'MWh-cap'),
    ('work_migrated', 'MWh-cap'),   # pool work routed to a site other than its origin (net, per step)
    ('co2_work_adjusted', 'kgCO2'),  # co2 + unexecuted work at the fleet's mean marginal energy x mean CI
]


def run_episode(geo, policy, seed):
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    policy.reset(seed, geo)
    n = geo.n_sites
    m = {name: 0.0 for name, _ in METRICS}
    share = np.zeros(n)
    ci_sum, amb_sum, util_sum = np.zeros(n), np.zeros(n), np.zeros(n)
    exo, steps = [], 0
    cap_mwh = geo.capacity_mw * STEP_HOURS  # MWh of one full step at each site
    while True:
        _, _, _, truncated, info = geo.step(policy.act(geo))
        g = info['geo']
        m['work_migrated'] += float(np.sum(np.maximum(g['routed_in_units'] - g['migrated_out_units'], 0))) \
            * 0.01 * STEP_HOURS
        for k, i in enumerate(info['sites']):
            m['co2'] += i['bat_CO2_footprint'] / 1e3
            m['total_energy'] += i['bat_total_energy_with_battery_KWh'] / 1e3
            m['it_energy'] += i['dc_ITE_total_power_kW'] * STEP_HOURS / 1e3
            m['hvac_energy'] += i['dc_HVAC_total_power_kW'] * STEP_HOURS / 1e3
            m['water'] += i['dc_water_usage']
            m['work_submitted'] += i['ls_original_workload'] * cap_mwh[k]
            m['work_executed'] += i['ls_shifted_workload'] * cap_mwh[k]
            m['work_dropped'] += i['ls_tasks_dropped'] / 100 * cap_mwh[k]
            share[k] += i['ls_shifted_workload'] * cap_mwh[k]
            ci_sum[k] += i['bat_avg_CI']
            amb_sum[k] += i['outside_temp']
            util_sum[k] += i['ls_original_workload']
            exo.append((i['ls_original_workload'], i['outside_temp'], i['norm_CI'], i['day'], i['hour']))
        steps += 1
        if truncated:
            break
    m['work_queued_end'] = float(sum(len(s.ls_env.tasks_queue) / 100 * cap_mwh[k] for k, s in enumerate(geo.sites))
                                 + np.sum(geo.carry * geo.units_per_task) * 0.01 * STEP_HOURS)
    # Unexecuted work (dropped or still queued) is charged at the fleet's mean marginal energy and mean CI, so a
    # dispatcher cannot look greener by not doing the work.
    from eval_compare import marginal_kwh_per_util_step
    kwh_per_mwh_cap = np.mean([marginal_kwh_per_util_step(s, util_sum[k] / steps, amb_sum[k] / steps) / cap_mwh[k]
                               for k, s in enumerate(geo.sites)])
    unexecuted = max(m['work_submitted'] - m['work_executed'], 0.0)
    m['co2_work_adjusted'] = m['co2'] + unexecuted * kwh_per_mwh_cap * np.mean(ci_sum / steps) / 1e3
    m['steps'] = steps
    m['share'] = (share / max(share.sum(), 1e-12)).tolist()
    fingerprint = hashlib.sha1(np.round(np.array(exo, dtype=np.float64), 9).tobytes()).hexdigest()[:12]
    return m, fingerprint


def paired(results, base='local'):
    from scipy import stats
    rows = {}
    for label, eps in results.items():
        if label == base:
            continue
        row = {}
        for key in ('co2', 'co2_work_adjusted', 'total_energy', 'hvac_energy', 'water'):
            d = np.array([100 * (e[key] / b[key] - 1) for e, b in zip(eps, results[base])])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
            row[key] = (float(d.mean()), float(half))
        rows[label] = row
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sites', default='ca,ny,wa', help='comma-separated locations (az, ca, ga, il, ny, tx, va, wa)')
    p.add_argument('--capacities', default='', help='comma-separated MW per site (default 1 each)')
    p.add_argument('--workloads', default='', help='comma-separated workload files per site (default: SustainDC default)')
    p.add_argument('--month', type=int, default=6)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--flexible_load', type=float, default=0.6)
    p.add_argument('--migratable_fraction', type=float, default=1.0)
    p.add_argument('--n_episodes', type=int, default=5)
    p.add_argument('--seed0', type=int, default=1000)
    p.add_argument('--dispatch', default='local,capacity,greedy_ci,merit',
                   help=f'comma-separated, from {DISPATCHERS} or sb3:<run_dir>[:<model>] (train_geo_dispatcher.py); '
                        'add =<label> to name one')
    p.add_argument('--dispatch_every', type=int, default=4, help='simulator steps per dispatch decision (4 = hourly)')
    p.add_argument('--local', default='noop', help='per-site controller spec (eval_compare.py syntax)')
    p.add_argument('--out_dir', default='')
    args = p.parse_args()

    locations = args.sites.split(',')
    caps = [float(c) for c in args.capacities.split(',')] if args.capacities else [1.0] * len(locations)
    wls = args.workloads.split(',') if args.workloads else [None] * len(locations)
    sites = []
    for loc, cap, wl in zip(locations, caps, wls):
        site = {'location': loc, 'datacenter_capacity_mw': cap}
        if wl:
            site['workload_file'] = wl
        sites.append(site)
    geo = GeoSustainDC({'sites': sites, 'month': args.month, 'days_per_episode': args.days,
                        'flexible_load': args.flexible_load, 'migratable_fraction': args.migratable_fraction})

    specs = {}
    for spec in args.dispatch.split(','):
        body, _, label = spec.partition('=')
        specs[label or body] = body
    if 'local' not in specs:
        specs = {'local': 'local', **specs}
    dispatchers = list(specs)
    seeds = [args.seed0 + k for k in range(args.n_episodes)]
    results, prints = {}, {}
    for d in dispatchers:
        policy = GeoPolicy(specs[d], args.local, args.dispatch_every)
        results[d], prints[d] = [], []
        for seed in seeds:
            m, fp = run_episode(geo, policy, seed)
            results[d].append(m)
            prints[d].append(fp)
            print(f"{d:>10} seed {seed}: co2 {m['co2']:.1f} kg, adjusted {m['co2_work_adjusted']:.1f} kg, "
                  f"migrated {m['work_migrated']:.2f} MWh-cap, share {np.round(m['share'], 2).tolist()}", flush=True)
    for d in dispatchers:
        if prints[d] != prints['local']:
            print(f"WARNING: exogenous inputs of {d!r} differ from 'local' (episodes are not paired)")

    print(f"\nSites {locations} (MW {caps}), month {args.month}, {args.days} days, {args.n_episodes} episodes, "
          f"local controllers {args.local!r}, migratable fraction {args.migratable_fraction}, "
          f"dispatch every {args.dispatch_every} steps")
    header = f"{'dispatcher':>10} " + ' '.join(f'{n:>17}' for n, _ in METRICS[:5] + METRICS[7:])
    print(header)
    for d in dispatchers:
        means = {n: np.mean([e[n] for e in results[d]]) for n, _ in METRICS}
        print(f"{d:>10} " + ' '.join(f'{means[n]:>17.3f}' for n, _ in METRICS[:5] + METRICS[7:]))
    rows = paired(results)
    print("\nChange vs local (no migration), mean % [95% CI half-width]:")
    for d, row in rows.items():
        print(f"{d:>10} " + '  '.join(f"{k} {v[0]:+.2f}±{v[1]:.2f}" for k, v in row.items()))

    out_dir = args.out_dir or os.path.join('results', 'eval_geo', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'args': vars(args), 'seeds': seeds, 'episodes': results, 'paired_vs_local': rows}, f, indent=1)
    print(f"\nSaved {out_dir}/summary.json")


if __name__ == '__main__':
    main()
