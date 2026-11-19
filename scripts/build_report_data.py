"""Collect the single-datacenter and distributed-datacenter results on scratch into the JSON datasets of the two
results pages ("SustainDC Single-Datacenter Results", "SustainDC Geo-Distributed Results"). One file per dataset,
{"rows": [...], "meta": [...]}, meta carrying this script's text and the inputs it read.

    python scripts/build_report_data.py <out_dir> [--root /network/scratch/f/florence.cloutier/sustaindc-rl_results]

Single data center (paired % change of work-adjusted CO2 vs noop on the same episodes):
    sdc_v2_summary    grid_v2 protocol (CA, TX; 12 monthly 30-day episodes, CI 2023, held-out workload), per scenario
                      (no PV / 1 MWp PV with PV actions), location and method; runs (seeds) are averaged per episode,
                      then mean and 95% CI (t) over episodes. MPPI is read per episode (ep*/episodes.csv) since its
                      merge jobs are pending, so a partial method says how many episodes it has.
    sdc_v2_episodes   the per-episode % changes behind it (month = episode index % 12)
    sdc_v2_train      RL training-time evaluations (single PPO, MAPPO): % vs noop on the trainer's own eval episodes
    sdc_v2_mpc_iters  learned-value MPC: % vs noop per value-iteration round (training episodes)
    sdc_v1_az         grid_v1 protocol (AZ, July, 10 episodes, CI 2022): every method incl. HAPPO / MAPPO / PPO
    sdc_v4_ca         v4 runs (CA, July, 10 episodes): RL vs CEM MPC, no PV and 500 kWp PV (automatic PV rule)
Distributed (GeoJobsDC, routing site_queue; % change of CO2 vs origin, every job at its origin site):
    geo_n5            N = 5 (dccluster5), 1 day, seeds 1-3: every method, with the hindsight LP bound ("captured")
    geo_scale         N = 20 / 50 / 200, 2 days, seeds 1-2: per (N, sigma24, method, seed)
    geo_hardness      heuristic dispatchers vs perfect-foresight LP bounds (pool routing, 7-day episodes)
For a (case, method) found in several result folders, the newest folder wins (list order below).
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import re

import numpy as np
from scipy import stats

ROOT = '/network/scratch/f/florence.cloutier/sustaindc-rl_results'


def clean(x):
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not math.isfinite(x) else round(x, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def ci95(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return None
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# -- single data center: grid_v2 ------------------------------------------------------------------------------------
SCEN = {'grid_v2_nopv': 'No PV', 'grid_v2_pv1000a': 'PV 1 MWp + PV actions'}
METRICS = ('co2_work_adjusted', 'co2', 'total_energy', 'water')


def parse_method(name):
    """mpc_joint_oracle_S_s1 -> (method 'mpc_joint_oracle_S', run seed 1)."""
    m = re.fullmatch(r'(.+)_s(\d+)', name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def describe(method):
    if method.startswith('mpc_joint') or method.startswith('mppi'):
        planner = 'CEM MPC' if method.startswith('mpc_joint') else 'MPPI'
        fc = 'oracle' if 'oracle' in method else 'persistence'
        cfg = method.rsplit('_', 1)[1]
        return planner, fc, {'S': 'H48, 128 samples x 5 iters', 'H96': 'H96, block 8, replan 4'}.get(cfg, cfg)
    if method in ('mpc_final', 'mpc_best'):
        return 'Learned-value MPC', 'realistic', 'H16, ' + ('last value' if method == 'mpc_final' else 'best value in training')
    if method == 'lp_bound':
        return 'LP bound', 'oracle', 'whole episode, perfect foresight'
    if method == 'lp_replay':
        return 'LP replay', 'oracle', 'bound plan rounded to actions'
    return method, '', ''


def read_episode_files(paths):
    """{(policy label, episode seed): row} from eval_compare episodes.csv files."""
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            out[(r['policy'], int(r['seed']))] = r
    return out


def sdc_v2(root):
    summ, eps_rows, inputs = [], [], []
    for scen, scen_label in SCEN.items():
        for loc in ('ca', 'tx'):
            base = os.path.join(root, scen, 'eval', loc)
            per = {}                                     # method -> {episode seed: {run seed: {metric: pct}}}
            dropped = {}
            for d in sorted(glob.glob(os.path.join(base, '*'))):
                name = os.path.basename(d)
                if name == 'lp':
                    continue
                files = sorted(glob.glob(os.path.join(d, 'episodes.csv'))) or \
                    sorted(glob.glob(os.path.join(d, 'ep*', 'episodes.csv')))
                if not files:
                    continue
                inputs += files
                rows = read_episode_files(files)
                method, run = parse_method(name)
                for (pol, seed), r in rows.items():
                    if pol == 'noop':
                        continue
                    noop = rows.get(('noop', seed))
                    if noop is None:
                        continue
                    pct = {m: 100 * (float(r[m]) / float(noop[m]) - 1) for m in METRICS}
                    per.setdefault(method, {}).setdefault(seed, {})[run or 0] = pct
                    dropped.setdefault(method, []).append(float(r['tasks_dropped']))
            lp = os.path.join(base, 'lp')
            if os.path.isdir(lp):
                bcsv = os.path.join(lp, 'bound', 'bound.csv')
                if os.path.exists(bcsv):
                    inputs.append(bcsv)
                    for r in csv.DictReader(open(bcsv)):
                        per.setdefault('lp_bound', {})[int(r['seed'])] = {0: {'co2_work_adjusted': float(r['pct_vs_noop'])}}
                rcsv = os.path.join(lp, 'replay', 'episodes.csv')
                if os.path.exists(rcsv):
                    inputs.append(rcsv)
                    rows = read_episode_files([rcsv])
                    for (pol, seed), r in rows.items():
                        if pol != 'noop' and ('noop', seed) in rows:
                            noop = rows[('noop', seed)]
                            per.setdefault('lp_replay', {})[seed] = {0: {m: 100 * (float(r[m]) / float(noop[m]) - 1)
                                                                         for m in METRICS}}
            for method, by_ep in per.items():
                runs = sorted({k for v in by_ep.values() for k in v})
                planner, fc, cfg = describe(method)
                ep_mean = {}
                for seed, by_run in sorted(by_ep.items()):
                    ep_mean[seed] = {m: float(np.mean([v[m] for v in by_run.values() if m in v]))
                                     for m in METRICS if any(m in v for v in by_run.values())}
                    for run, v in by_run.items():
                        eps_rows.append({'scenario': scen_label, 'location': loc.upper(), 'method': method,
                                         'run_seed': run, 'episode_seed': seed, 'month': (seed - 1000) % 12,
                                         'co2_wa_pct': v['co2_work_adjusted']})
                vals = [e['co2_work_adjusted'] for e in ep_mean.values()]
                row = {'scenario': scen_label, 'location': loc.upper(), 'method': method, 'planner': planner,
                       'forecast': fc, 'config': cfg, 'runs': len(runs), 'episodes': len(vals),
                       'episodes_expected': 12, 'complete': len(vals) == 12,
                       'co2_wa_pct': float(np.mean(vals)), 'co2_wa_ci95': ci95(vals),
                       'tasks_dropped_mean': float(np.mean(dropped[method])) if method in dropped else None}
                for m in ('co2', 'total_energy', 'water'):
                    v = [e[m] for e in ep_mean.values() if m in e]
                    row[f'{m}_pct'] = float(np.mean(v)) if v else None
                summ.append(row)
    return summ, eps_rows, inputs


def sdc_v2_train(root):
    rows, inputs = [], []
    for scen, scen_label in SCEN.items():
        for path in sorted(glob.glob(os.path.join(root, scen, 'runs', 'sustaindc', '*', '*', scen, 'seed-*', 'eval',
                                                  'evals.csv'))):
            parts = path.split(os.sep)
            loc, algo = parts[-6], parts[-5]
            if algo == 'mpc':
                continue
            seed = int(re.search(r'seed-0*(\d+)-', path).group(1))
            inputs.append(path)
            for r in csv.DictReader(open(path)):
                rows.append({'scenario': scen_label, 'location': loc.upper(), 'algo': algo, 'seed': seed,
                             'env_steps': int(float(r['env_steps'])),
                             'co2_wa_pct': num(r.get('co2_work_adjusted_pct_vs_noop')),
                             'is_best': r.get('is_best') in ('1', 'True', 'true')})
    return rows, inputs


def sdc_v2_mpc_iters(root):
    rows, inputs = [], []
    for scen, scen_label in SCEN.items():
        for path in sorted(glob.glob(os.path.join(root, scen, 'runs', 'sustaindc', '*', 'mpc', scen, 'seed-*',
                                                  'history.json'))):
            loc = path.split(os.sep)[-5]
            seed = int(re.search(r'seed-0*(\d+)-', path).group(1))
            inputs.append(path)
            for h in json.load(open(path)):
                rows.append({'scenario': scen_label, 'location': loc.upper(), 'run_seed': seed,
                             'iteration': h['iteration'], 'value': 'naive' if h['value'] == 'naive' else 'learned',
                             'co2_wa_pct': h['change_pct_mean'], 'ci95': h['change_pct_ci95'],
                             'minutes': h.get('minutes')})
    return rows, inputs


def paired_table(path, extra):
    s = json.load(open(path))
    rows = []
    for label, rep in s['paired_vs_noop'].items():
        r = rep.get('co2_work_adjusted')
        if not isinstance(r, dict):
            continue
        pooled = '(mean of' in label
        method, seed = (label.split(' (')[0], None) if pooled else parse_method(label)
        rows.append({**extra, 'label': label, 'method': method, 'seed': seed, 'pooled': pooled,
                     'co2_wa_pct': r['mean_pct'], 'co2_wa_ci95': r.get('ci95_pct'),
                     'co2_pct': rep.get('co2', {}).get('mean_pct'),
                     'energy_pct': rep.get('total_energy', {}).get('mean_pct')})
    return rows


def sdc_v1_az(root):
    path = os.path.join(root, 'grid_v1', 'eval', 'az', 'combined', 'summary.json')
    return paired_table(path, {'location': 'AZ'}), [path]


def sdc_v4_ca(root):
    rows, inputs = [], []
    for d, pv in (('v4_nopv_mpc', 'No PV'), ('v4_pv500_mpc', 'PV 500 kWp (automatic rule)')):
        path = os.path.join(root, 'eval_compare', d, 'summary.json')
        inputs.append(path)
        rows += paired_table(path, {'location': 'CA', 'scenario': pv})
    return rows, inputs


# -- distributed ----------------------------------------------------------------------------------------------------
N5_DIRS = ['cmp_n5_v1', 'meas_n5_v1', 'dualw_n5_v1_index', 'dualw_n5_v1_full', 'dualw_n5_v1_r1', 'dualw_n5_v1_r2']
SCALE_DIRS = ['bn_v1', 'meas_scale_v1', 'mpc_scale_v1', 'dual_scale_v1', 'dualw_v1_index', 'dualw_v1_full',
              'dualw_v1_r1', 'dualw_v1_r2']
GEO_FIELDS = ('sla_violation_rate', 'decision_s_mean', 'decision_s_max', 'forecast_values_per_hour',
              'scalars_per_hour', 'boundary_site_frac', 'boundary_item_frac', 'linear_item_frac', 'coarse_site_s_max',
              'refine_s_per_hour', 'joint_s_mean', 'joint_s_max', 'joint_vars_max', 'timeouts', 'dw_rounds_mean',
              'dw_parallel_s_mean', 'dw_site_lps_per_hour', 'dw_converged_frac', 'dw_master_fail', 'seconds')


def geo_cases(root, dirs):
    """{(scenario, sigma, seed): {'origin': kg, 'bound': kg|None, 'n_sites', 'methods': {name: (dir, result)}}}"""
    cases, inputs = {}, []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(root, 'geo_scaling', d, 'case_*.json'))):
            c = json.load(open(path))
            a = c['args']
            key = (a['scenario'], a['sigma24'], a['seed'], a['days'])
            e = cases.setdefault(key, {'origin': None, 'bound': None, 'n_sites': c['n_sites'], 'methods': {}})
            used = False
            for m, v in c['methods'].items():
                if not isinstance(v, dict) or 'co2_kg' not in v:
                    continue
                if m == 'origin':
                    e['origin'] = v['co2_kg']
                else:
                    e['methods'][m] = (d, v)
                used = True
            b = (c.get('bound') or {}).get('co2_kg')
            if b is not None:
                e['bound'] = b
            if used:
                inputs.append(path)
    return cases, inputs


def geo_rows(cases):
    rows = []
    for (scen, sig, seed, days), e in sorted(cases.items()):
        o, b = e['origin'], e['bound']
        if o is None:
            continue
        for m, (d, v) in sorted(e['methods'].items()):
            r = {'scenario': scen, 'n_sites': e['n_sites'], 'sigma24': sig, 'seed': seed, 'days': days,
                 'method': m, 'folder': d, 'co2_kg': v['co2_kg'], 'origin_kg': o,
                 'co2_pct': 100 * (v['co2_kg'] / o - 1),
                 'captured': (o - v['co2_kg']) / (o - b) if b is not None and o > b else None}
            r.update({f: v.get(f) for f in GEO_FIELDS})
            rows.append(r)
        rows.append({'scenario': scen, 'n_sites': e['n_sites'], 'sigma24': sig, 'seed': seed, 'days': days,
                     'method': 'origin', 'folder': '', 'co2_kg': o, 'origin_kg': o, 'co2_pct': 0.0,
                     'captured': 0.0 if b is not None else None})
        if b is not None:
            rows.append({'scenario': scen, 'n_sites': e['n_sites'], 'sigma24': sig, 'seed': seed, 'days': days,
                         'method': 'bound', 'folder': '', 'co2_kg': b, 'origin_kg': o,
                         'co2_pct': 100 * (b / o - 1), 'captured': 1.0})
    return rows


def geo_hardness(root):
    rows, inputs = [], sorted(glob.glob(os.path.join(root, 'geo_hardness', 'v1', 'case_*.json')))
    for path in inputs:
        c = json.load(open(path))
        local = c['sim']['local']['co2_kg']
        b = c['lp']['geo_time'].get('co2_kg') if c['lp']['geo_time'].get('status') == 0 else None
        key = {'regions': c['regions'], 'month': 'Jan' if c['month'] == 0 else 'Jul', 'util': c['target_util'],
               'seed': c['seed']}
        for h, m in c['sim'].items():
            rows.append({**key, 'method': h, 'kind': 'heuristic', 'co2_pct': 100 * (m['co2_kg'] / local - 1),
                         'captured': (local - m['co2_kg']) / (local - b) if b and local > b else None})
        for v, r in c['lp'].items():
            if r.get('status') == 0:
                rows.append({**key, 'method': f'LP {v}', 'kind': 'bound', 'co2_pct': 100 * (r['co2_kg'] / local - 1),
                             'captured': (local - r['co2_kg']) / (local - b) if b and local > b else None})
    return rows, inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir')
    p.add_argument('--root', default=ROOT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    code = open(os.path.abspath(__file__)).read()

    def save(name, rows, inputs):
        meta = [{'label': 'build_report_data.py', 'value': code, 'type': 'code', 'lang': 'python'},
                {'label': 'Inputs', 'value': '\n'.join(sorted(set(inputs)))}]
        with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
            json.dump(clean({'rows': rows, 'meta': meta}), f, allow_nan=False)
        print(f'{name}: {len(rows)} rows from {len(set(inputs))} files')

    summ, eps, inp = sdc_v2(args.root)
    save('sdc_v2_summary', summ, inp)
    save('sdc_v2_episodes', eps, inp)
    save('sdc_v2_train', *sdc_v2_train(args.root))
    save('sdc_v2_mpc_iters', *sdc_v2_mpc_iters(args.root))
    save('sdc_v1_az', *sdc_v1_az(args.root))
    save('sdc_v4_ca', *sdc_v4_ca(args.root))
    cases, inp = geo_cases(args.root, N5_DIRS)
    save('geo_n5', geo_rows(cases), inp)
    cases, inp = geo_cases(args.root, SCALE_DIRS)
    save('geo_scale', geo_rows(cases), inp)
    save('geo_hardness', *geo_hardness(args.root))


if __name__ == '__main__':
    main()
