"""Checks for measurement-based refinement (sufficiency/screen.py, refine='measure') and the exact sufficiency
analysis (sufficiency/exact.py). Run from the repo root (or on Slurm with scripts/slurm/geo_run.sbatch):
    python tests/verify_measurements.py [--scenarios dccluster5,n20] [--hours 3] [--draws 4] [--sigma24 0.1]

Per recorded routing decision (one bundle at one hour):
1. Linear form: for items the guard accepts, the LP's index at random costs in the box equals pi . c + kappa
   (relative error <= 1e-2 for at least 95% of (item, draw) pairs; the share within 1e-3 is reported).
2. Measurements decide the routing: the clearing with the screening's keys (surely in / out as classified, boundary
   segments keyed by their projected differences, fallback sites by their refitted curve) gives every site the same
   share as the clearing with every item's index re-solved by its LP at the refined forecast. Counted only where the
   refined forecast lies inside the nominal box (the Proposition's premise); the rest are reported.
3. Exact sufficiency (on the linearized decision, sufficiency/exact.py): every item in a relevant edge is in the
   boundary; the relevant measurement directions lie in the span of the screening's |B| - 1 differences; Ye's
   certificate holds with them at sampled costs; d*_Delta <= rank of ours <= |B| - 1. Ye's Algorithm 2 and the
   reachable-vertex count are reported beside them.
"""
import argparse
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import numpy as np

results = []


def check(name, cond, detail=''):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenarios', default='dccluster5,n20')
    p.add_argument('--hours', type=int, default=3)
    p.add_argument('--draws', type=int, default=4)
    p.add_argument('--sigma24', type=float, default=0.1)
    p.add_argument('--guard_samples', type=int, default=0, help='random draws the linearity guard also tests')
    args = p.parse_args()
    from geo_jobs import GeoJobsDC
    from geo_scenarios import site_configs
    from sufficiency.exact import RoutingDecision, compare
    from sufficiency.screen import OFFSET, ScreenedIndexPolicy
    from sufficiency.top_m import clear

    for scen in args.scenarios.split(','):
        print(f'== {scen}, sigma24 {args.sigma24}', flush=True)
        geo = GeoJobsDC({'sites': site_configs(scen), 'month': 6, 'days_per_episode': 1, 'target_util': 0.5,
                         'routing': 'site_queue'})
        random.seed(1)
        np.random.seed(1)
        geo.reset()
        pol = ScreenedIndexPolicy(mode='screen', sigma24=args.sigma24, interval='linear', refine='measure',
                                  admission='latest', record=True, n_samples=args.guard_samples)
        pol.reset(geo)
        rng = np.random.default_rng(7)
        lin_err, lin_n, agree, n_dec, exact_rows, outside, agree_out = [], 0, 0, 0, [], 0, 0
        hours = 0
        while hours < args.hours:
            if geo.t % 4 == 0 and geo.t >= 4:
                pol.records = []
                pol._replan(geo)                       # the decision as the policy makes it (keys in pol.curves)
                groups = pol._bundles(geo)
                for rec in pol.records:
                    b, keys, lab, lin_ok = rec['b'], rec['keys'], rec['lab'], rec['lin_ok']
                    sites = sorted({i for i, _ in keys})
                    W = {i: pol._window(geo, i) for i in sites}
                    # 1. linear form vs the LP at random costs in the box (shared within a region)
                    for _ in range(args.draws):
                        draw = pol.fc.sample(rng, sites, geo.t, max(W.values()))
                        nu_lp = {}
                        for i in sites:
                            _, _, cur = pol.site_curves(geo, i, {b: groups[b]}, [draw[i][:W[i]]], pol.G)
                            nu_lp[i] = cur[b][2][0] if b in cur else np.full(pol.G, np.nan)
                        for a, (i, k) in enumerate(keys):
                            if lin_ok[a]:
                                pred = rec['pis'][a] @ draw[i][:W[i]] + rec['kappas'][a]
                                lin_err.append(abs(pred - nu_lp[i][k]) / max(abs(nu_lp[i][k]), 1e-12))
                    # 2. screening keys vs every item's LP index at the refined forecast
                    nu_ref = {}
                    for i in sites:
                        _, _, cur = pol.site_curves(geo, i, {b: groups[b]}, [pol.fc.window(i, geo.t, W[i], 'refined')], pol.G)
                        nu_ref[i] = cur[b][2][0] if b in cur else np.full(pol.G, np.inf)
                    truth = np.array([nu_ref[i][k] for i, k in keys])
                    keyed = np.array([pol.curves[(i, b)][1][k] for i, k in keys])
                    x_true, x_ours = clear(truth, rec['w'], 1.0), clear(keyed, rec['w'], 1.0)
                    # routing is per site: segments of one site with equal indices may fill in either order
                    site_of = np.array([i for i, _ in keys])
                    share = lambda x: np.array([x[site_of == i].sum() for i in sites])
                    inside = all(np.all((pol.fc.window(i, geo.t, W[i], 'refined') >= rec['box'][i][0][:W[i]] - 1e-9)
                                        & (pol.fc.window(i, geo.t, W[i], 'refined') <= rec['box'][i][1][:W[i]] + 1e-9))
                                 for i in sites)
                    if inside:
                        agree += int(np.allclose(share(x_true), share(x_ours), atol=1e-9))
                        n_dec += 1
                    else:
                        outside += 1
                        agree_out += int(np.allclose(share(x_true), share(x_ours), atol=1e-9))
                    # 3. exact sufficiency on the linearized decision
                    boxes = {r: rec['box'][i] for (i, _), r in zip(keys, rec['regions'])}
                    Wmin = min(len(v) for v in rec['pis'])
                    dec = RoutingDecision(np.array([v[:Wmin] for v in rec['pis']]), rec['kappas'], rec['regions'],
                                          rec['w'], {r: (lo[:Wmin], hi[:Wmin]) for r, (lo, hi) in boxes.items()})
                    row = compare(dec, rec['regions'], rng)
                    row.update({'t': geo.t, 'bundle': int(b)})
                    exact_rows.append(row)
                    print('     t {t} b{bundle}: {items} items, {reachable_vertices} reachable vertices, d*_vert {d_vert}, '
                          'relevant edges {relevant_edges} (d*_Delta x {d_delta_x}, c {d_delta_c}), boundary {boundary_items}, '
                          'ours {ours} (rank {ours_rank}), outside ours {relevant_outside_ours:.1e}, Ye Alg2 {ye_alg2_rank}, '
                          'certificate {certificate_ours:.2f}'.format(**row), flush=True)
                hours += 1
            geo.step(pol.act(geo))
        e = np.array(lin_err)
        check('linear form matches the LP inside the box (guard-accepted items)', len(e) and np.mean(e <= 1e-2) >= 0.95,
              f'{np.mean(e <= 1e-3):.1%} within 1e-3, {np.mean(e <= 1e-2):.1%} within 1e-2 of {len(e)} (item, draw); '
              f'max rel. error {e.max():.2e}' if len(e) else 'none')
        check('measurement keys route like LP re-solves at the refined forecast (per-site shares)', agree == n_dec,
              f'{agree}/{n_dec} decisions with the refined forecast inside the box; outside it: {agree_out}/{outside}')
        check('every item of a relevant edge is in the boundary', all(r['boundary_contains_relevant'] for r in exact_rows))
        check('relevant directions lie in the span of our differences', all(r['relevant_outside_ours'] < 1e-6 for r in exact_rows),
              f"max residual {max(r['relevant_outside_ours'] for r in exact_rows):.1e}")
        check("Ye's certificate holds with our differences", all(r['certificate_ours'] == 1.0 for r in exact_rows),
              f"min share {min(r['certificate_ours'] for r in exact_rows):.2f}")
        check('d*_Delta <= rank(ours) <= |B| - 1', all(r['d_delta_c'] <= r['ours_rank'] <= max(r['boundary_items'] - 1, 0)
                                                    for r in exact_rows))
        check('d*_vert = d*_Delta in x-space (Ye Thm 3)', all(r['d_vert'] == r['d_delta_x'] for r in exact_rows))
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
