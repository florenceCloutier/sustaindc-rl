"""Sufficiency-screened index policy: spend forecasts and computation only on the sites whose routing is a close call.

Every hour, for each bundle b (the past hour's arrivals of one slack class, see geo_index.py):

1. Coarse indices, all sites (parallel in principle: each site uses only its own state and forecast). Each site
   solves its index curve (IndexPolicy: G segments of phi in [0, phi_max]) under S CI scenarios from its *nominal*
   forecast: the nominal forecast and the two corners of its uncertainty box (sufficiency/forecast.py), optionally
   plus random draws. Item (i, b, k) = segment k of site i's curve: width w = delta phi (a share of the bundle),
   index interval [lo, hi] = min / max over scenarios. Each site reports 2 numbers per item.
2. Classification (sufficiency/top_m.py, demand D = 1 bundle): surely in / surely out / boundary. Pairs of items in
   the same region share their forecast error, so for them "b possibly <= a" is decided scenario by scenario (the
   corners move both together); other pairs use the interval test. This is where the direction view pays: two
   sites on the same grid are compared through (pi_a - pi_b) . c, not through two independent boxes.
3. Refinement, boundary sites B only: a refined forecast (smaller error, forecast.py `rho`) and a finer curve
   (`levels_fine` segments). Their plans for the hour use the refined forecast too.
4. Routing (IndexPolicy.route: each arrival to the lowest index that fits): surely-in items first (nominal index as
   the tie-break when arrivals fall short of the forecast bundle), then boundary items (refined index), then
   surely-out items (only reached when arrivals exceed the bundle).

Modes (for the ablations):
    screen      as above
    nominal     no screening, no refinement: every site plans and indexes with its nominal forecast
    refine_all  no screening: every site uses the refined forecast and the fine curve (full information; the
                reference that screening tries to match with |B| / N of the refined information)
With sigma24 = 0 all three see the true CI; screening then only measures how many items are ties.

Approximations (labelled as such in the stats): the interval of an index over the box is estimated from the
scenarios (exact only when the index is monotone in c, which holds when neither capacity nor the battery's SoC
window binds; tests/verify_screening.py checks the Proposition empirically with random draws in the box).
"""
import time
import zlib

import numpy as np

from geo_index import IndexPolicy
from sufficiency.forecast import ForecastModel
from sufficiency.top_m import classify

OFFSET = 1e6          # routing-key offset per class (indices are ~0.1 g per W x step)
MODES = ('screen', 'nominal', 'refine_all')


class ScreenedIndexPolicy(IndexPolicy):
    def __init__(self, mode='screen', sigma24=0.10, q=1.645, rho=0.3, fc_seed=0, levels_fine=8, n_samples=0,
                 joint=True, tol=0.0, rel_tol=0.0, record=False, **kw):
        super().__init__(**kw)
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode, self.sigma24, self.q, self.rho, self.fc_seed = mode, sigma24, q, rho, fc_seed
        self.G_fine = int(levels_fine)
        self.n_samples, self.joint, self.tol, self.record = int(n_samples), bool(joint), float(tol), record
        self.rel_tol = float(rel_tol)   # widen each interval by rel_tol x |index| (safety margin for non-monotone indices)

    def reset(self, geo):
        super().reset(geo)
        self.regions = [cfg['location'] for cfg in geo.site_configs]
        self.fc = ForecastModel(self.ci, self.regions, self.sigma24, self.q, self.rho, self.fc_seed)
        self.refined_sites = set(range(geo.n_sites)) if self.mode == 'refine_all' else set()
        if self.mode == 'refine_all':
            self.G = self.G_fine
        self.stats['screen'] = []
        self.records = []

    def _ci_for(self, geo, i, lp):
        kind = 'refined' if i in self.refined_sites else 'nominal'
        return self.fc.window(i, geo.t, lp.W, kind)

    def _window(self, geo, i):
        """Length of site i's LP window now (as SiteLP computes it)."""
        span = int(np.ceil(geo.trace['max_dur_steps'])) + 1
        return int(min(self.H + span, self.ci.shape[1] - geo.t))

    # -----------------------------------------------------------------------------------------------------------------
    def site_curves(self, geo, i, groups, scen_ci, G):
        """Site i's plan LP and index curves under each scenario CI (W-long windows, list).
        Returns (base lp, [(v0, x0)] per scenario, {b: (phis, work, nus (S, G), lps)})."""
        base = self._site_lp(geo, i)
        W = base.W
        cis = [c[:W] for c in scen_ci]
        sols = [self._solve(base, 0.0, c) for c in cis]
        out = {}
        if not all(np.isfinite(v) for v, _ in sols):
            return base, sols, out
        for b, g in groups.items():
            lp = self._site_lp(geo, i, bundle=g)
            phi_max = self._phi_max(geo, lp, g)
            if phi_max <= 1e-6:
                continue
            phis = np.linspace(0.0, phi_max, G + 1)
            nus = np.array([self._curve(lp, phis, v0, g['work'], c)[0] for (v0, _), c in zip(sols, cis)])
            out[b] = (phis, g['work'], nus)
        return base, sols, out

    def _scenarios(self, geo, i, W, rng_per_region, refined=False):
        """CI windows for site i: nominal (or refined), box low, box high, + random draws (shared per region)."""
        pre = 'ref_' if refined else ''
        out = [self.fc.window(i, geo.t, W, 'refined' if refined else 'nominal'),
               self.fc.window(i, geo.t, W, pre + 'low'), self.fc.window(i, geo.t, W, pre + 'high')]
        if self.n_samples:
            lo, hi = out[1], out[2]
            for u in rng_per_region(self.regions[i], W):
                out.append(np.exp(np.log(lo) + (u + 1) / 2 * (np.log(hi) - np.log(lo))))
        return out

    def _replan(self, geo):
        if self.mode != 'screen':
            return super()._replan(geo)
        t = geo.t
        self.stats['replans'] += 1
        groups = self._bundles(geo)
        self.curves = {}
        self.acc[:] = 0.0
        N = geo.n_sites
        draws = {}

        def rng_per_region(region, W):
            if region not in draws:
                rng = np.random.default_rng([self.fc_seed, t, zlib.crc32(region.encode())])
                draws[region] = [rng.uniform(-1, 1, W) for _ in range(self.n_samples)]
            return [d[:W] for d in draws[region]]

        # 1. coarse curves, every site, nominal forecast + box corners
        t0 = time.time()
        lp0, site_s = self.stats['lp_solves'], np.zeros(N)
        coarse = {}
        for i in range(N):
            ts = time.time()
            W = self._window(geo, i)
            base, sols, cur = self.site_curves(geo, i, groups, self._scenarios(geo, i, W, rng_per_region), self.G)
            v0, x0 = sols[0]
            self.plans[i] = self._make_plan(base, x0, t) if x0 is not None else None
            coarse[i] = cur
            site_s[i] = time.time() - ts
        coarse_lps = self.stats['lp_solves'] - lp0

        # 2. classify the items of each bundle
        rec = {'t': t, 'n_sites': N, 'bundles': {}, 'coarse_lps': coarse_lps, 'coarse_s': time.time() - t0,
               'coarse_site_s_max': float(site_s.max()) if N else 0.0}
        labels = {}            # (i, b) -> per-segment label
        boundary_sites = set()
        for b in groups:
            keys = [(i, k) for i in range(N) if b in coarse[i] for k in range(self.G)]
            if not keys:
                continue
            nus = np.array([coarse[i][b][2][:, k] for i, k in keys])          # (items, S)
            w = np.array([np.diff(coarse[i][b][0])[k] for i, k in keys])
            ok = np.all(np.isfinite(nus), axis=1)
            fin = np.where(np.isfinite(nus), nus, np.nan)
            pad = self.rel_tol * np.nan_to_num(np.nanmax(np.abs(fin), axis=1, initial=0.0))
            # a failed solve under some scenario (numerical; the LP is always feasible) says nothing: the item could be
            # anywhere, so it competes with every item (lower end: its finite minimum, if any) and is never dismissed
            lo = np.where(np.isfinite(fin).any(1), np.nanmin(np.where(np.isfinite(fin), fin, np.inf), axis=1), -np.inf) - pad
            hi = np.where(ok, nus.max(1) + pad, np.inf)
            cheaper = None
            if self.joint:
                reg = np.array([self.regions[i] for i, _ in keys])
                same = reg[:, None] == reg[None, :]
                joint_le = np.any(nus[:, None, :] - pad[:, None, None] <= nus[None, :, :] + pad[None, :, None] + self.tol,
                                  axis=2)                                            # [b, a]: some scenario b <= a
                bad = ~ok[:, None] | ~ok[None, :]                                    # failed solves: interval test
                cheaper = np.where(same & ~bad, joint_le, lo[:, None] <= hi[None, :] + self.tol)
            lab = classify(lo, hi, w, 1.0, cheaper=cheaper, tol=self.tol)
            for (i, k), l in zip(keys, lab):
                labels.setdefault((i, b), np.zeros(self.G, dtype=int))[k] = l
            bs = {keys[a][0] for a in np.flatnonzero(lab == 0)}
            boundary_sites |= bs
            rec['bundles'][int(b)] = {'items': len(keys), 'in': int((lab == 1).sum()), 'out': int((lab == -1).sum()),
                                      'boundary_items': int((lab == 0).sum()), 'boundary_sites': len(bs),
                                      'measurements': max(len(bs) - 1, 0),
                                      'width_in': float(w[lab == 1].sum()), 'width_boundary': float(w[lab == 0].sum())}
            if self.record:
                self.records.append({'t': t, 'b': b, 'keys': keys, 'nus': nus, 'w': w, 'lab': lab,
                                     'phis': {i: coarse[i][b][0] for i, _ in keys}})

        # 3. refine the boundary sites: refined forecast and a finer curve
        lp1, t1 = self.stats['lp_solves'], time.time()
        fine = {}
        for i in sorted(boundary_sites):
            W = self._window(geo, i)
            scen = [self.fc.window(i, t, W, 'refined')]
            base, sols, cur = self.site_curves(geo, i, groups, scen, self.G_fine)
            if sols[0][1] is not None:
                self.plans[i] = self._make_plan(base, sols[0][1], t)
            fine[i] = cur
        self.refined_sites = boundary_sites
        rec.update({'boundary_sites': len(boundary_sites), 'refine_lps': self.stats['lp_solves'] - lp1,
                    'refine_s': time.time() - t1,
                    'forecast_values_refined': int(sum(self._window(geo, i) for i in boundary_sites)),
                    'forecast_values_full': int(sum(self.H for _ in range(N)))})
        self.stats['screen'].append(rec)

        # 4. routing keys: class offset + index (refined on boundary sites)
        for (i, b), lab in labels.items():
            phis_c, work, nus_c = coarse[i][b]
            if i in fine and b in fine[i]:
                phis, _, nus_f = fine[i][b]
                nu = nus_f[0]
                mid = (phis[1:] + phis[:-1]) / 2
                seg = np.minimum(np.searchsorted(phis_c[1:], mid, side='right'), self.G - 1)
                l = lab[seg]
            else:
                phis, nu, l = phis_c, nus_c[0], lab
            key = np.where(l == 1, nu - OFFSET, np.where(l == -1, nu + OFFSET, nu))
            self.curves[(i, b)] = (phis, key, work)


def screen_summary(policy):
    """Episode means of the screening stats (empty dict if the policy did not screen)."""
    recs = policy.stats.get('screen') or []
    if not recs:
        return {}
    N = recs[0]['n_sites']
    items = sum(v['items'] for r in recs for v in r['bundles'].values())
    bitems = sum(v['boundary_items'] for r in recs for v in r['bundles'].values())
    meas = [sum(v['measurements'] for v in r['bundles'].values()) for r in recs]
    return {'boundary_site_frac': float(np.mean([r['boundary_sites'] / N for r in recs])),
            'boundary_site_frac_max': float(np.max([r['boundary_sites'] / N for r in recs])),
            'boundary_item_frac': bitems / max(items, 1),
            'measurements_mean': float(np.mean(meas)),
            'coarse_lps_per_hour': float(np.mean([r['coarse_lps'] for r in recs])),
            'refine_lps_per_hour': float(np.mean([r['refine_lps'] for r in recs])),
            'coarse_site_s_max': float(np.mean([r['coarse_site_s_max'] for r in recs])),
            'refine_s_per_hour': float(np.mean([r['refine_s'] for r in recs])),
            'forecast_frac_refined': float(np.sum([r['forecast_values_refined'] for r in recs])
                                           / max(np.sum([r['forecast_values_full'] for r in recs]), 1))}
