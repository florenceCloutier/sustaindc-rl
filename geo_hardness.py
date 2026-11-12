"""How hard is geo-distributed job dispatch? Heuristics (simulated) vs. perfect-foresight LP bounds, same episodes.

For each (region set, month, target utilization, seed):
  * simulate the job dispatchers of geo_job_baselines.py on GeoJobsDC (cooling and battery held by --local, noop by
    default, so only dispatch differs);
  * on the same episode (same seed: same jobs, CI, weather), solve the LP bounds of geo_lp_oracle.py:
        time      origin only, may wait within the deadline slack      value of moving work in time
        geo       any site, start on arrival                           value of moving work in space
        geo_time  any site, may wait                                   the bound
  * calibration: the LP's energy model evaluated on the simulated `local` schedule vs. the simulated CO2 (should
    match), and the exact model on the LP's own schedule vs. the LP value (tangent underestimation, should be small).

Headline: captured(h) = (local - h) / (local - geo_time bound), the share of the achievable CO2 saving that heuristic h
gets. High for `merit`/`greedy_ci` -> easy; low for them but high for `forecast_greedy` -> foresight is what matters;
low for both -> there is planning headroom for a learned dispatcher.

Run from the repo root, e.g.
    python geo_hardness.py --regions dccluster5 --months 6 --utils 0.5 --days 2 --seeds 1      # smoke
    python geo_hardness.py --regions dccluster5,us4 --months 0,6 --utils 0.5,0.7 --days 7 --seeds 1,2,3,4,5
One process per seed (e.g. a Slurm array) with the same --out_dir, then:
    python geo_hardness.py --combine <out_dir>
"""
import argparse
import datetime
import glob
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np

REGION_SETS = {
    'dccluster5': ['US-CAL-CISO', 'DE-LU', 'CL-SIC', 'SG', 'AU-NSW'],      # DCcluster-Opt's evaluation cluster
    'us4': ['US-CAL-CISO', 'US-NY-NYIS', 'US-TEX-ERCO', 'US-MIDA-PJM'],
    'eu_asia': ['DE-LU', 'GB', 'JP-TK', 'KR', 'SG'],
}
HEURISTICS = ('local', 'greedy_ci', 'merit', 'merit_defer', 'forecast_greedy')
BOUNDS = ('time', 'geo', 'geo_time')


def make_geo(regions, month, days, util, args):
    from geo_jobs import GeoJobsDC
    sites = [{'location': f'dcc:{r}:{args.year}'} for r in regions]
    return GeoJobsDC({'sites': sites, 'month': month, 'days_per_episode': days, 'target_util': util,
                      'sla_multiplier': args.sla_multiplier, 'rigid_fraction': args.rigid_fraction})


def run_case(regions_name, month, util, seed, args):
    from geo_job_baselines import JobDispatcher, run_job_episode
    from geo_lp_oracle import energy_curves, exo_traces, model_co2_kg, solve_bound
    geo = make_geo(REGION_SETS[regions_name], month, args.days, util, args)
    case = {'regions': regions_name, 'sites': REGION_SETS[regions_name], 'month': month, 'target_util': util,
            'seed': seed, 'days': args.days, 'sim': {}, 'lp': {}}
    local_util = None
    for kind in HEURISTICS:
        t0 = time.time()
        m = run_job_episode(geo, JobDispatcher(kind), seed, local_spec=args.local)
        m['seconds'] = time.time() - t0
        case['sim'][kind] = m
        if kind == 'local':
            local_util = np.array(geo.util_history)
        print(f"  [{regions_name} m{month} u{util} s{seed}] {kind:>15}: {m['co2_kg']:10.1f} kg, "
              f"sla {m['sla_violation_rate']:.4f}, migrated {m['job_mwh_migrated']:.1f}/{m['job_mwh']:.1f} MWh "
              f"({m['seconds']:.0f}s)", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    geo.reset()
    traces = exo_traces(geo, geo.horizon)
    curves = energy_curves(geo, traces)
    case['calibration'] = {'sim_local_kg': case['sim']['local']['co2_kg'],
                           'model_at_local_kg': model_co2_kg(geo, traces, local_util)}
    for v in BOUNDS:
        r = solve_bound(geo, v, traces=traces, curves=curves, verbose=False)
        if r['status'] == 0:
            r['exact_model_at_lp_kg'] = model_co2_kg(geo, traces, r['util'])
        r.pop('util', None)
        case['lp'][v] = r
        print(f"  [{regions_name} m{month} u{util} s{seed}] LP {v:>9}: {r['co2_kg']:10.1f} kg "
              f"(exact at LP schedule {r.get('exact_model_at_lp_kg', float('nan')):.1f}), late {r.get('late_job_mwh', 0):.2f} MWh, "
              f"{r['n_vars']} vars, {r['solve_s']:.0f}s", flush=True)
    return case


def summarize(cases):
    """Per (regions, month, util): mean over seeds of % change vs local and captured share; printed as a table."""
    groups = {}
    for c in cases:
        groups.setdefault((c['regions'], c['month'], c['target_util']), []).append(c)
    rows = []
    for (reg, month, util), cs in sorted(groups.items()):
        row = {'regions': reg, 'month': month, 'target_util': util, 'n_seeds': len(cs)}
        loc = np.array([c['sim']['local']['co2_kg'] for c in cs])
        bound = np.array([c['lp']['geo_time']['co2_kg'] for c in cs])
        for name in HEURISTICS[1:]:
            h = np.array([c['sim'][name]['co2_kg'] for c in cs])
            row[f'{name}_pct'] = float(np.mean(100 * (h / loc - 1)))
            row[f'{name}_captured'] = float(np.mean((loc - h) / (loc - bound)))
            row[f'{name}_sla'] = float(np.mean([c['sim'][name]['sla_violation_rate'] for c in cs]))
        for v in BOUNDS:
            b = np.array([c['lp'][v]['co2_kg'] for c in cs])
            row[f'lp_{v}_pct'] = float(np.mean(100 * (b / loc - 1)))
        row['calib_err_pct'] = float(np.mean([100 * (c['calibration']['model_at_local_kg'] / c['calibration']['sim_local_kg'] - 1)
                                              for c in cs]))
        row['tangent_gap_pct'] = float(np.mean([100 * (c['lp']['geo_time']['exact_model_at_lp_kg'] / c['lp']['geo_time']['co2_kg'] - 1)
                                                for c in cs if 'exact_model_at_lp_kg' in c['lp']['geo_time']]))
        row['mean_util'] = float(np.mean([np.mean(c['sim']['local']['mean_util']) for c in cs]))
        rows.append(row)
    print('\nCO2 change vs local (mean over seeds), and share of the geo_time bound captured')
    print(f"{'regions':>11} {'mon':>3} {'util':>4} {'n':>2} | {'LP time':>7} {'LP geo':>7} {'LP g+t':>7} | "
          + ' '.join(f'{h:>16}' for h in HEURISTICS[1:]) + ' | calib% tangent%')
    for r in rows:
        print(f"{r['regions']:>11} {r['month']:>3} {r['target_util']:>4} {r['n_seeds']:>2} | "
              f"{r['lp_time_pct']:+7.1f} {r['lp_geo_pct']:+7.1f} {r['lp_geo_time_pct']:+7.1f} | "
              + ' '.join(f"{r[h + '_pct']:+6.1f}% ({r[h + '_captured']:4.0%})" for h in HEURISTICS[1:])
              + f" | {r['calib_err_pct']:+.2f} {r['tangent_gap_pct']:+.2f}")
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f"{r['regions']}\nm{r['month']} u{r['target_util']}" for r in rows]
    series = [('greedy_ci', 'greedy_ci_pct'), ('merit', 'merit_pct'), ('merit_defer', 'merit_defer_pct'),
              ('forecast_greedy', 'forecast_greedy_pct'), ('LP time', 'lp_time_pct'), ('LP geo', 'lp_geo_pct'),
              ('LP geo+time (bound)', 'lp_geo_time_pct')]
    x = np.arange(len(rows))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rows)), 4))
    for k, (name, key) in enumerate(series):
        ax.bar(x + (k - len(series) / 2 + 0.5) * w, [-r[key] for r in rows], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('CO2 reduction vs local (%)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regions', default='dccluster5', help=f'comma-separated, from {list(REGION_SETS)}')
    p.add_argument('--months', default='6', help='comma-separated, 0-based (0 = January, 6 = July)')
    p.add_argument('--utils', default='0.5', help='comma-separated target mean utilizations')
    p.add_argument('--seeds', default='1')
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--sla_multiplier', type=float, default=1.5)
    p.add_argument('--rigid_fraction', type=float, default=0.0)
    p.add_argument('--local', default='noop', help='per-site cooling/battery controllers (eval_compare.py spec)')
    p.add_argument('--out_dir', default='')
    p.add_argument('--combine', default='', help='only summarize the case_*.json files of this directory')
    args = p.parse_args()

    if args.combine:
        cases = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.combine, 'case_*.json')))]
        rows = summarize(cases)
        with open(os.path.join(args.combine, 'summary.json'), 'w') as f:
            json.dump({'rows': rows, 'n_cases': len(cases)}, f, indent=1)
        plot(rows, os.path.join(args.combine, 'hardness.png'))
        print(f"\nSaved {args.combine}/summary.json and hardness.png")
        return

    out_dir = args.out_dir or os.path.join('results', 'geo_hardness', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    cases = []
    for reg in args.regions.split(','):
        for month in (int(m) for m in args.months.split(',')):
            for util in (float(u) for u in args.utils.split(',')):
                for seed in (int(s) for s in args.seeds.split(',')):
                    case = run_case(reg, month, util, seed, args)
                    cases.append(case)
                    name = f'case_{reg}_m{month}_u{util}_s{seed}.json'
                    with open(os.path.join(out_dir, name), 'w') as f:
                        json.dump(case, f, indent=1)
    rows = summarize(cases)
    plot(rows, os.path.join(out_dir, 'hardness.png'))
    print(f'\nSaved {out_dir}')


if __name__ == '__main__':
    main()
