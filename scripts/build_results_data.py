"""Collect the single-datacenter (grid_v1) and distributed-datacenter (geo index) results on scratch into the JSON
files of the results page: one file per dataset, {"rows": [...], "meta": [...]}, meta carrying this script's text
and the inputs it read.

    python scripts/build_results_data.py <out_dir> [--root /network/scratch/f/florence.cloutier/sustaindc-rl_results]

Datasets:
    sdc_protocol           paired protocol eval vs noop per policy (grid_v1/eval/az/combined/summary.json)
    sdc_protocol_episodes  the per-episode % changes behind it
    sdc_train_curves       training-time evals of the RL runs (runs/.../eval/evals.csv)
    sdc_mpc_iters          value iterations of the learned-value MPC runs (runs/.../mpc/.../history.json)
    geo_hardness           heuristics vs perfect-foresight LP bounds (geo_hardness/v1/case_*.json)
    geo_methods_n5         all methods vs the hindsight bound at N = 5 (geo_scaling/cmp_n5_v1/case_*.json)
    geo_scaling            screening statistics against N (geo_scaling/bn_v1/case_*.json)
"""
import argparse
import csv
import glob
import json
import math
import os
import re

ROOT = '/network/scratch/f/florence.cloutier/sustaindc-rl_results'
METRICS = ('co2_work_adjusted', 'co2', 'total_energy', 'water', 'tasks_dropped', 'tasks_queued_end')


def clean(x):
    """JSON-safe: NaN / inf -> None, numpy-ish floats -> float."""
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, 6)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def pooled_label(label, s=None):
    """A row that eval_compare --combine made by pooling the '<method>_s<seed>' runs: '<method> (mean of 3 runs)'."""
    return '(mean of' in label


def method_seed(label):
    if pooled_label(label):
        return label.split(' (')[0], None
    m = re.fullmatch(r'(.+)_s(\d+)', label)
    return (m.group(1), int(m.group(2))) if m else (label, None)


# -- single datacenter ----------------------------------------------------------------------------------------------
def sdc_protocol(root):
    path = os.path.join(root, 'grid_v1', 'eval', 'az', 'combined', 'summary.json')
    s = json.load(open(path))
    rows, eps = [], []
    for label, rep in s['paired_vs_noop'].items():
        method, seed = method_seed(label)
        summ = s['summary'].get(label, {})
        for metric, r in rep.items():
            if not isinstance(r, dict) or 'mean_pct' not in r:
                continue
            absolute = summ.get(metric)
            rows.append({'policy': label, 'method': method, 'seed': seed, 'pooled': pooled_label(label, s), 'metric': metric,
                         'mean_pct': r['mean_pct'], 'ci95_pct': r.get('ci95_pct'),
                         'mean_abs': absolute[0] if isinstance(absolute, list) else None,
                         'n_episodes': len(r.get('per_episode_pct') or [])})
            if metric == 'co2_work_adjusted' and label != 'noop' and not pooled_label(label):
                for k, v in enumerate(r.get('per_episode_pct') or []):
                    eps.append({'policy': label, 'method': method, 'seed': seed if seed is not None else 0,
                                'episode_seed': s['seeds'][k], 'pct': v})
    noop = s['summary'].get('noop', {})
    for metric in METRICS:
        if metric in noop:
            rows.append({'policy': 'noop', 'method': 'noop', 'seed': None, 'pooled': True, 'metric': metric,
                         'mean_pct': 0.0, 'ci95_pct': 0.0, 'mean_abs': noop[metric][0], 'n_episodes': len(s['seeds'])})
    assert s.get('inputs_identical') is True, 'combined eval: inputs differ between policies'
    return rows, eps, [path]


def sdc_train_curves(root):
    rows, inputs = [], []
    for algo in ('happo', 'mappo', 'single_ppo'):
        for path in sorted(glob.glob(os.path.join(root, 'grid_v1', 'runs', 'sustaindc', 'az', algo, 'grid_v1',
                                                  'seed-*', 'eval', 'evals.csv'))):
            seed = int(re.search(r'seed-0*(\d+)-', path).group(1))
            inputs.append(path)
            for r in csv.DictReader(open(path)):
                rows.append({'method': algo, 'seed': seed, 'env_steps': int(float(r['env_steps'])),
                             'co2_wa_pct': num(r['co2_work_adjusted_pct_vs_noop']), 'co2_pct': num(r['co2_pct_vs_noop']),
                             'tasks_dropped': num(r['tasks_dropped']), 'is_best': r['is_best'] in ('1', 'True', 'true')})
    assert len(inputs) == 9, f'expected 9 RL runs, found {len(inputs)}'
    return rows, inputs


def sdc_mpc_iters(root):
    rows, inputs = [], []
    for path in sorted(glob.glob(os.path.join(root, 'grid_v1', 'runs', 'sustaindc', 'az', 'mpc', 'grid_v1', 'seed-*',
                                              'history.json'))):
        seed = int(re.search(r'seed-0*(\d+)-', path).group(1))
        inputs.append(path)
        for h in json.load(open(path)):
            rows.append({'run_seed': seed, 'iteration': h['iteration'],
                         'value': 'naive' if h['value'] == 'naive' else f"iter {h['iteration']}",
                         'env_steps': h['env_steps'], 'change_pct': h['change_pct_mean'], 'ci95_pct': h['change_pct_ci95'],
                         'dropped': h['dropped_mean'], 'minutes': h['minutes']})
    assert len(inputs) == 3, f'expected 3 MPC runs, found {len(inputs)}'
    return rows, inputs


# -- distributed ----------------------------------------------------------------------------------------------------
def geo_hardness(root):
    rows, inputs = [], sorted(glob.glob(os.path.join(root, 'geo_hardness', 'v1', 'case_*.json')))
    for path in inputs:
        c = json.load(open(path))
        local = c['sim']['local']['co2_kg']
        bound = c['lp']['geo_time']['co2_kg'] if c['lp']['geo_time'].get('status') == 0 else None
        key = {'regions': c['regions'], 'month': 'Jan' if c['month'] == 0 else 'Jul', 'util': c['target_util'],
               'seed': c['seed']}
        for h, m in c['sim'].items():
            rows.append({**key, 'method': h, 'kind': 'heuristic', 'co2_kg': m['co2_kg'],
                         'pct_vs_local': 100 * (m['co2_kg'] / local - 1),
                         'captured': (local - m['co2_kg']) / (local - bound) if bound and local > bound else None,
                         'sla_rate': m['sla_violation_rate']})
        for v, r in c['lp'].items():
            if r.get('status') == 0:
                rows.append({**key, 'method': f'LP {v}', 'kind': 'bound', 'co2_kg': r['co2_kg'],
                             'pct_vs_local': 100 * (r['co2_kg'] / local - 1),
                             'captured': (local - r['co2_kg']) / (local - bound) if bound and local > bound else None,
                             'sla_rate': None})
    return rows, inputs


def geo_methods_n5(root):
    rows, inputs = [], sorted(glob.glob(os.path.join(root, 'geo_scaling', 'cmp_n5_v1', 'case_*.json')))
    for path in inputs:
        c = json.load(open(path))
        base = c['methods']['origin']['co2_kg']
        bound = c.get('bound', {}).get('co2_kg')
        for k, m in c['methods'].items():
            rows.append({'seed': c['args']['seed'], 'sigma24': c['args']['sigma24'], 'method': k, 'co2_kg': m['co2_kg'],
                         'pct_vs_origin': 100 * (m['co2_kg'] / base - 1),
                         'captured': (base - m['co2_kg']) / (base - bound) if bound and base > bound else None,
                         'sla_violations': m['sla_violations'], 'sla_rate': m['sla_violation_rate'],
                         'decision_s_mean': m['decision_s_mean'], 'decision_s_max': m['decision_s_max'],
                         'boundary_site_frac': m.get('boundary_site_frac'),
                         'forecast_frac_refined': m.get('forecast_frac_refined')})
        if bound:
            rows.append({'seed': c['args']['seed'], 'sigma24': c['args']['sigma24'], 'method': 'bound', 'co2_kg': bound,
                         'pct_vs_origin': 100 * (bound / base - 1), 'captured': 1.0})
    return rows, inputs


def geo_scaling(root):
    rows, inputs = [], sorted(glob.glob(os.path.join(root, 'geo_scaling', 'bn_v1', 'case_*.json')))
    for path in inputs:
        c = json.load(open(path))
        a = c['args']
        base = c['methods']['origin']['co2_kg']
        for k, m in c['methods'].items():
            rows.append({'n_sites': c['n_sites'], 'sigma24': a['sigma24'], 'seed': a['seed'], 'method': k,
                         'co2_kg': m['co2_kg'], 'pct_vs_origin': 100 * (m['co2_kg'] / base - 1), 'jobs': m['jobs'],
                         'sla_rate': m['sla_violation_rate'], 'decision_s_mean': m['decision_s_mean'],
                         'run_minutes': m['seconds'] / 60,
                         **{f: m.get(f) for f in ('boundary_site_frac', 'boundary_site_frac_max', 'boundary_item_frac',
                                                  'forecast_frac_refined', 'measurements_mean', 'coarse_site_s_max',
                                                  'refine_s_per_hour', 'coarse_lps_per_hour', 'refine_lps_per_hour')}})
    return rows, inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir')
    p.add_argument('--root', default=ROOT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    code = open(os.path.abspath(__file__)).read()

    def save(name, rows, inputs):
        meta = [{'label': 'build_results_data.py', 'value': code, 'type': 'code', 'lang': 'python'},
                {'label': 'Inputs', 'value': '\n'.join(inputs)}]
        with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
            json.dump(clean({'rows': rows, 'meta': meta}), f, allow_nan=False)
        print(f'{name}: {len(rows)} rows from {len(inputs)} files')

    prot, eps, inp = sdc_protocol(args.root)
    save('sdc_protocol', prot, inp)
    save('sdc_protocol_episodes', eps, inp)
    save('sdc_train_curves', *sdc_train_curves(args.root))
    save('sdc_mpc_iters', *sdc_mpc_iters(args.root))
    hard, inp = geo_hardness(args.root)
    assert len(inp) == 29, f'expected 29 hardness cases, found {len(inp)}'
    save('geo_hardness', hard, inp)
    n5, inp = geo_methods_n5(args.root)
    assert len(inp) == 3, f'expected 3 N=5 cases, found {len(inp)}'
    save('geo_methods_n5', n5, inp)
    sc, inp = geo_scaling(args.root)
    assert len(inp) == 12, f'expected 12 scaling cases, found {len(inp)}'
    r = next(x for x in sc if x['n_sites'] == 200 and x['sigma24'] == 0.1 and x['seed'] == 1 and x['method'] == 'screen')
    assert abs(r['boundary_site_frac'] - 0.0768) < 1e-3, r['boundary_site_frac']   # value quoted in chat
    save('geo_scaling', sc, inp)


if __name__ == '__main__':
    main()
