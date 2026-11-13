"""Compare routing methods on GeoJobsDC(routing='site_queue') across network sizes: CO2, SLA, time per decision,
and the screening statistics.

Methods (all on the same episode: same jobs, CI and weather for a given seed):
    origin, greedy_ci, merit   JobRouter (geo_job_baselines.py), earliest-deadline-first starts, no battery / cooling
    index                      IndexPolicy: every site uses its nominal CI forecast (the truth when --sigma24 0)
    screen                     ScreenedIndexPolicy: nominal + box everywhere, refined forecast on boundary sites only
    index_full                 every site uses the refined forecast and the fine curve (full information)
    mpc                        B1, JointMPCPolicy (nominal forecasts); --time_limit caps each hourly joint LP
    mpc_full                   B1 with the refined forecasts
    ppo:<run_dir>              B3, a router trained by train_geo_job_ppo.py (best checkpoint of the run)
and, with --bound, the hindsight LP bound (geo_lp_oracle geo_time, battery + cooling envelope) for "captured":
    captured(m) = (origin - m) / (origin - bound)

Run from the repo root, e.g.
    python geo_scaling.py --scenario dccluster5 --days 2 --methods origin,merit,index,screen,mpc --bound   # smoke
    python geo_scaling.py --scenario n50 --days 7 --sigma24 0.1 --methods merit,index,screen,mpc --time_limit 900
One process per case (e.g. a Slurm array), then  python geo_scaling.py --combine <out_dir>
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


def make_geo(args):
    from geo_jobs import GeoJobsDC
    from geo_scenarios import site_configs
    return GeoJobsDC({'sites': site_configs(args.scenario, year=args.year), 'month': args.month,
                      'days_per_episode': args.days, 'target_util': args.util, 'sla_multiplier': args.sla_multiplier,
                      'rigid_fraction': args.rigid_fraction, 'routing': 'site_queue'})


def make_method(name, args):
    """('router', JobRouter) or ('policy', act-policy)."""
    from geo_job_baselines import JOB_ROUTERS, JobRouter
    kw = {'window_h': args.window_h, 'levels': args.levels, 'admission': args.admission}
    fc = {'sigma24': args.sigma24, 'rho': args.rho, 'fc_seed': args.seed}
    if name in JOB_ROUTERS:
        return 'router', JobRouter(name)
    if name in ('index', 'screen', 'index_full'):
        from sufficiency.screen import ScreenedIndexPolicy
        mode = {'index': 'nominal', 'screen': 'screen', 'index_full': 'refine_all'}[name]
        return 'policy', ScreenedIndexPolicy(mode=mode, levels_fine=args.levels_fine, n_samples=args.n_samples, **fc, **kw)
    if name in ('mpc', 'mpc_full'):
        from geo_mpc import JointMPCPolicy
        return 'policy', JointMPCPolicy(time_limit=args.time_limit, forecast='refined' if name == 'mpc_full' else 'nominal',
                                        **fc, **kw)
    if name.startswith('ppo:'):
        from geo_rl_jobs import PPORouter
        return 'router', PPORouter.load(name[4:], sigma24=args.sigma24, rho=args.rho, fc_seed=args.seed)
    raise ValueError(f'Unknown method {name!r}')


def run_method(geo, name, args):
    from geo_index import run_index_episode
    from geo_job_baselines import run_queue_episode
    from sufficiency.screen import screen_summary
    kind, m = make_method(name, args)
    t0 = time.time()
    if kind == 'router':
        out = run_queue_episode(geo, m, args.seed)
        d = np.array(getattr(m, 'decision_seconds', []) or [np.nan])
        out.update({'decision_s_mean': float(np.nanmean(d)), 'decision_s_max': float(np.nanmax(d))})
    else:
        out = run_index_episode(geo, m, args.seed)
        out.update(screen_summary(m))
        if 'joint_s' in m.stats:
            js = np.array(m.stats['joint_s'] or [np.nan])
            out.update({'joint_vars_max': int(max(m.stats['joint_vars'] or [0])), 'joint_s_mean': float(np.nanmean(js)),
                        'joint_s_max': float(np.nanmax(js)), 'timeouts': m.stats['timeouts']})
    out['seconds'] = time.time() - t0
    return out


def run_bound(geo, args):
    from geo_lp_oracle import solve_bound
    random.seed(args.seed)
    np.random.seed(args.seed)
    geo.reset()
    r = solve_bound(geo, 'geo_time', verbose=False, site_flex='battery+cooling')
    r.pop('util', None)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default='dccluster5', help='geo_scenarios name: dccluster5, dcc15, n50, n200s1, ...')
    p.add_argument('--methods', default='origin,merit,index,screen')
    p.add_argument('--month', type=int, default=6)
    p.add_argument('--util', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--year', type=int, default=2023)
    p.add_argument('--sla_multiplier', type=float, default=1.5)
    p.add_argument('--rigid_fraction', type=float, default=0.0)
    p.add_argument('--sigma24', type=float, default=0.0, help='CI forecast error (log sd) at 24 h lead')
    p.add_argument('--rho', type=float, default=0.3, help='refined forecast error / nominal')
    p.add_argument('--window_h', type=float, default=24)
    p.add_argument('--levels', type=int, default=4)
    p.add_argument('--levels_fine', type=int, default=8)
    p.add_argument('--n_samples', type=int, default=0, help='extra random CI draws per site for the index intervals')
    p.add_argument('--admission', default='latest', choices=('plan', 'latest'),
                   help="routing admission of the LP methods (geo_index.IndexPolicy); 'latest' meets deadlines")
    p.add_argument('--time_limit', type=float, default=None, help='seconds per hourly joint LP (mpc)')
    p.add_argument('--bound', action='store_true')
    p.add_argument('--out_dir', default='')
    p.add_argument('--combine', default='')
    args = p.parse_args()
    if args.combine:
        return combine(args.combine)

    out_dir = args.out_dir or os.path.join('results', 'geo_scaling', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    name = f'case_{args.scenario}_m{args.month}_u{args.util}_n{args.sigma24}_s{args.seed}.json'
    path = os.path.join(out_dir, name)
    case = {'args': vars(args), 'methods': {}}
    if os.path.exists(path):                      # resume: keep the methods already done
        case = json.load(open(path))
    t0 = time.time()
    geo = make_geo(args)
    case['n_sites'], case['build_s'] = geo.n_sites, time.time() - t0
    tag = f'[{args.scenario} m{args.month} u{args.util} sigma{args.sigma24} s{args.seed}]'
    print(f'{tag} {geo.n_sites} sites, built in {case["build_s"]:.0f}s', flush=True)
    for meth in args.methods.split(','):
        if meth in case['methods']:
            continue
        m = run_method(geo, meth, args)
        case['methods'][meth] = m
        extra = ''.join(f', {k} {m[k]:.3g}' for k in ('boundary_site_frac', 'forecast_frac_refined', 'joint_s_max')
                        if k in m)
        print(f"{tag} {meth:>12}: {m['co2_kg']:10.1f} kg, sla {m['sla_violations']}, decision "
              f"{m['decision_s_mean']:.3f}s mean / {m['decision_s_max']:.2f}s max, {m['seconds']:.0f}s{extra}", flush=True)
        json.dump(case, open(path, 'w'), indent=1)
    if args.bound and 'bound' not in case:
        case['bound'] = run_bound(geo, args)
        print(f"{tag} bound: {case['bound']['co2_kg']:.1f} kg ({case['bound']['solve_s']:.0f}s)", flush=True)
        json.dump(case, open(path, 'w'), indent=1)
    report(case)
    print(f'Saved {path}')


def report(case):
    ms = case['methods']
    if 'origin' not in ms:
        return
    base = ms['origin']['co2_kg']
    bound = case.get('bound', {}).get('co2_kg')
    for k, m in ms.items():
        cap = f", captured {(base - m['co2_kg']) / (base - bound):5.1%}" if bound and base > bound else ''
        print(f"  {k:>12}: {100 * (m['co2_kg'] / base - 1):+6.2f}% vs origin{cap}")


def combine(out_dir):
    cases = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'case_*.json')))]
    rows = []
    for c in cases:
        a = c['args']
        base = c['methods'].get('origin', {}).get('co2_kg')
        bound = c.get('bound', {}).get('co2_kg')
        for k, m in c['methods'].items():
            row = {'scenario': a['scenario'], 'n_sites': c['n_sites'], 'month': a['month'], 'util': a['util'],
                   'sigma24': a['sigma24'], 'seed': a['seed'], 'method': k, 'co2_kg': m['co2_kg'],
                   'sla_violations': m['sla_violations'], 'decision_s_mean': m['decision_s_mean'],
                   'decision_s_max': m['decision_s_max']}
            if base:
                row['pct_vs_origin'] = 100 * (m['co2_kg'] / base - 1)
            if base and bound and base > bound:
                row['captured'] = (base - m['co2_kg']) / (base - bound)
            for key in ('boundary_site_frac', 'boundary_item_frac', 'forecast_frac_refined', 'measurements_mean',
                        'joint_s_max', 'timeouts', 'coarse_site_s_max'):
                if key in m:
                    row[key] = m[key]
            rows.append(row)
    json.dump(rows, open(os.path.join(out_dir, 'rows.json'), 'w'), indent=1)
    keys = ['scenario', 'n_sites', 'sigma24', 'method', 'pct_vs_origin', 'captured', 'sla_violations',
            'decision_s_mean', 'boundary_site_frac', 'forecast_frac_refined']
    print(' '.join(f'{k[:12]:>12}' for k in keys))
    for r in sorted(rows, key=lambda r: (r['n_sites'], r['sigma24'], r['method'])):
        print(' '.join(f"{r[k]:>12.4g}" if isinstance(r.get(k), float) else f"{str(r.get(k, '')):>12}" for k in keys))
    print(f'Saved {out_dir}/rows.json ({len(rows)} rows from {len(cases)} cases)')


if __name__ == '__main__':
    main()
