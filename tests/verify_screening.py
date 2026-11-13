"""Checks for step 5 (screening) and the B1 / index consistency. Run from the repo root:
    python tests/verify_screening.py [--scenario n20] [--hours 2] [--samples 8]
Needs the DCcluster-Opt data. ~10-20 minutes at the defaults.

1. Phase-1 toy (sufficiency/top_m.py): boundary {B, C}, one measurement c_B - c_C suffices.
2. Forecast model: exact at lead 0, ~N(0, 1) errors, the truth in the 90% box ~90% of steps, errors shared within a
   region.
3. Proposition, on the real site LPs: at a few hours, draw the true CI uniformly in each region's box (shared by the
   sites of a region), recompute every site's index curve, clear each bundle: surely-in items must be filled and
   surely-out items empty. Reported as the share of (sample, item) pairs that violate it (the intervals come from
   the box corners, so this is the empirical check the plan asks for).
4. Relevant directions (Bennouna): over the same draws, the clearing allocations x(c) differ only on boundary items,
   and dim span{x(c) - x(c')} (an estimate of d*) <= |B| - 1 per bundle.
5. Lagrangian / LP duality: the joint LP of B1 (geo_mpc) routes each bundle where the site index curves cross its
   price: sites with a positive share have their curve's first segment at or below lambda_b / work_b, sites with
   none have it at or above (up to the curve's coarse resolution).
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
    p.add_argument('--scenario', default='n20')
    p.add_argument('--hours', type=int, default=2, help='hours at which the Proposition is checked')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--sigma24', type=float, default=0.15)
    p.add_argument('--n_samples', type=int, default=0, help='extra random scenarios in the screening intervals')
    p.add_argument('--rel_tol', type=float, default=0.0, help='interval margin, relative to the index')
    p.add_argument('--no_joint', action='store_true', help='interval test for every pair (no same-region scenarios)')
    args = p.parse_args()
    from geo_jobs import GeoJobsDC
    from geo_mpc import JointMPCPolicy
    from geo_scenarios import site_configs
    from sufficiency.forecast import ForecastModel
    from sufficiency.screen import ScreenedIndexPolicy
    from sufficiency.top_m import clear, toy

    print('1. Phase-1 toy')
    out = toy()
    check('boundary {B, C}, no violations, c_B - c_C sufficient',
          out['boundary'] == ['B', 'C'] and out['violations'] == 0 and out['measurement_c_B_minus_c_C_sufficient'],
          str(out['labels']))

    print('2. Forecast model')
    rng = np.random.default_rng(0)
    ci = rng.uniform(100, 600, (3, 800))
    fc = ForecastModel(ci, ['a', 'a', 'b'], sigma24=0.1, seed=3)
    z, inside = [], []
    for t in range(0, 600, 4):
        nom = fc.window(0, t, 96, 'nominal')
        z.append(np.log(nom[1:] / ci[0, t + 1:t + 96]) / fc.sigma(96)[1:])
        lo, hi = fc.window(0, t, 96, 'low'), fc.window(0, t, 96, 'high')
        inside.append((ci[0, t:t + 96] >= lo - 1e-9) & (ci[0, t:t + 96] <= hi + 1e-9))
    z = np.concatenate(z)
    check('lead 0 exact', np.allclose(fc.window(0, 8, 5, 'nominal')[0], ci[0, 8]))
    check('errors ~ N(0, 1)', abs(z.mean()) < 0.15 and abs(z.std() - 1) < 0.15, f'mean {z.mean():.3f} sd {z.std():.3f}')
    check('truth in the 90% box', 0.85 < np.mean(inside) < 0.95, f'{np.mean(inside):.3f}')
    e0 = fc.window(0, 40, 50) / ci[0, 40:90]
    e1 = fc.window(1, 40, 50) / ci[1, 40:90]
    e2 = fc.window(2, 40, 50) / ci[2, 40:90]
    check('same region shares errors, other region does not', np.allclose(e0, e1) and not np.allclose(e0, e2))

    print(f'3-4. Proposition and relevant directions ({args.scenario}, sigma24 {args.sigma24})')
    geo = GeoJobsDC({'sites': site_configs(args.scenario), 'month': 6, 'days_per_episode': 1, 'target_util': 0.5,
                     'routing': 'site_queue'})
    random.seed(1)
    np.random.seed(1)
    geo.reset()
    pol = ScreenedIndexPolicy(mode='screen', sigma24=args.sigma24, record=True, n_samples=args.n_samples,
                              rel_tol=args.rel_tol, joint=not args.no_joint)
    excursion = []
    pol.reset(geo)
    srng = np.random.default_rng(5)
    viol = pairs = 0
    dstar_ok, support_ok, n_b = True, True, []
    checked = 0
    while checked < args.hours:
        if geo.t % 4 == 0 and geo.t >= 4:
            pol.records = []
            pol._replan(geo)
            groups = pol._bundles(geo)
            for rec in pol.records:
                b, keys, w, lab = rec['b'], rec['keys'], rec['w'], rec['lab']
                allocs = []
                for _ in range(args.samples):
                    W = {i: pol._window(geo, i) for i in {k[0] for k in keys}}
                    draw = pol.fc.sample(srng, sorted(W), geo.t, max(W.values()))
                    nus = {}
                    for i in sorted(W):
                        _, _, cur = pol.site_curves(geo, i, {b: groups[b]}, [draw[i][:W[i]]], pol.G)
                        nus[i] = cur[b][2][0] if b in cur else np.full(pol.G, np.inf)
                    nu = np.array([nus[i][k] for i, k in keys])
                    lo_, hi_ = rec['nus'].min(1), rec['nus'].max(1)
                    fin = np.isfinite(nu) & np.isfinite(hi_)
                    excursion.append(np.max(np.r_[0.0, (np.maximum(nu - hi_, lo_ - nu)[fin]) / np.abs(nu[fin])]))
                    x = clear(nu, w, 1.0)
                    allocs.append(x)
                    bad = ((lab == 1) & (x < w - 1e-9)) | ((lab == -1) & (x > 1e-9))
                    viol += int(bad.sum())
                    for a in np.flatnonzero(bad)[:3]:
                        print(f'     violation t {geo.t} b {b} item {keys[a]} label {lab[a]} w {w[a]:.4f} x {x[a]:.4f} '
                              f'nu {nu[a]:.6g} interval [{lo_[a]:.6g}, {hi_[a]:.6g}] sum w {w.sum():.3f} '
                              f'inf items {int((~np.isfinite(nu)).sum())}', flush=True)
                    pairs += len(lab)
                X = np.array(allocs)
                diff = X - X[0]
                support = np.flatnonzero(np.abs(diff).max(0) > 1e-9)
                nb = int((lab == 0).sum())
                support_ok &= set(support) <= set(np.flatnonzero(lab == 0))
                d = int(np.linalg.matrix_rank(diff, tol=1e-9)) if len(diff) > 1 else 0
                n_b.append((b, nb, len({keys[a][0] for a in np.flatnonzero(lab == 0)}), d))
                dstar_ok &= d <= max(nb - 1, 0) or nb == 0
            checked += 1
        geo.step(pol.act(geo))
    check('surely-in filled and surely-out empty for draws in the box', viol == 0,
          f'{viol} violating (sample, item) pairs of {pairs}; sampled index outside the scenario interval by up to '
          f'{100 * max(excursion):.2f}% of the index (median over samples {100 * np.median(excursion):.3f}%)')
    check('allocations change only on boundary items', support_ok)
    check('estimated d* <= boundary items - 1', dstar_ok,
          ', '.join(f'b{b}: {nb} boundary items on {ns} sites, d* {d}' for b, nb, ns, d in n_b))

    print('5. Lagrangian / LP duality: joint LP shares vs site index curves')
    random.seed(1)
    np.random.seed(1)
    geo.reset()
    idx = ScreenedIndexPolicy(mode='nominal', sigma24=0.0)
    mpc = JointMPCPolicy()
    idx.reset(geo)
    mpc.reset(geo)
    for _ in range(8):
        geo.step(idx.act(geo))
    idx._replan(geo)
    mpc._t = geo.t
    mpc._replan(geo)
    ok = tot = 0
    for b, lam in (mpc.last_duals or {}).items():
        price = lam / mpc.work[b]
        for i in range(geo.n_sites):
            cur = idx.curves.get((i, b))
            if cur is None:
                continue
            phis, nus, _ = cur
            share = mpc.last_phi[b][i]
            tot += 1
            tolr = 0.02 * abs(price) + 1e-6
            if share > 1e-6:
                ok += nus[0] <= price + tolr
            else:
                ok += nus[0] >= price - tolr
    check('shares consistent with index curves at the joint LP price', tot > 0 and ok >= 0.8 * tot,
          f'{ok}/{tot} (site, bundle) pairs')

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
