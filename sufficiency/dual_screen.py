"""Price-coordinated routing (geo_dual.py) with sufficiency screening: refined forecasts only for the sites whose
answer to the prices is a close call.

Every hour, for the bundles of the past hour's arrivals:
1. Coordinate with nominal forecasts: Dantzig-Wolfe (geo_dual.py), every site solving only its own LP with all
   bundles, until prices lambda* place every bundle.
2. Box test (each site on its own, in parallel): re-solve the site's pricing LP at lambda* with its forecast at the
   low and the high corner of its box (sufficiency/forecast.py). The site is SETTLED when
     - its answer (shares of each bundle) is the same at the nominal forecast and both corners, and
     - the coordinator does not split it between proposals (a split site is indifferent at lambda*: at the margin).
   Otherwise it is a BOUNDARY site.
3. Refine: boundary sites get the refined forecast; coordination runs again, warm-started from the proposals so far
   (their costs re-priced under each site's current forecast) and from lambda*.
4. Re-check: settled sites are tested again at the new prices; any whose answer now changes joins the boundary, and
   step 3 repeats (at most `passes` times).

Why it is safe (Lagrangian sufficiency): if at prices lambda every site's chosen plan is optimal for its own pricing
problem and the shares place every bundle exactly, the plans are optimal for the joint LP. A settled site's answer
is the same for its nominal forecast and the box corners, so (exactly in the linear regime, where a site LP's
solution stays optimal over the box; otherwise as tested at the corners) better information about its forecast
would not change it; the boundary sites, which could change, get the refined forecast.

Modes: 'screen' (above), 'nominal' (step 1 only = DualPolicy with nominal forecasts), 'full' (refined forecasts
everywhere: the full-information reference).

stats['dscreen']: per hour, boundary sites, passes, rounds, LP solves, parallel time (sum over rounds of the slowest
site), forecast values refined.
"""
import time

import numpy as np
from scipy.optimize import linprog

from geo_dual import DualPolicy, SitePricing
from sufficiency.forecast import ForecastModel

MODES = ('screen', 'nominal', 'full')


def coordinate(sites, B, max_rounds=40, gap_tol=1e-4, cols=None, big_m=1e12):
    """Dantzig-Wolfe over `sites` (SitePricing). cols: per site, starting proposals [(cost, phi, x)], or None.
    Returns dict: phi (N, B), lam (B,), mu per site, chosen x per site, rounds, gap, site_s (parallel time),
    lps (site solves), unrouted."""
    N = len(sites)
    times, lps = [], 0
    if cols is None:
        cols = []
        for s in sites:
            t0 = time.time()
            out = s.solve(np.full(B, -1e12))                 # take nothing
            times.append(time.time() - t0)
            lps += 1
            cols.append([out if out is not None else (0.0, np.zeros(B), None)])
    site_s = max(times) if times else 0.0
    res_info = {'rounds': 0, 'gap': np.inf}
    mu = lam = None
    sizes = [len(c) for c in cols]
    for r in range(max_rounds):
        cur = [len(c) for c in cols]                         # becomes `sizes` only if this master solve succeeds
        ncol = sum(cur)
        cost = np.r_[[c[0] for cs in cols for c in cs], np.full(B, big_m)]
        A = np.zeros((N + B, ncol + B))
        k = 0
        for i, cs in enumerate(cols):
            for c in cs:
                A[i, k] = 1.0
                A[N:, k] = c[1]
                k += 1
        A[N:, ncol:] = np.eye(B)
        res = linprog(cost, A_eq=A, b_eq=np.ones(N + B), bounds=(0, None), method='highs')
        if res.status != 0:
            break
        mu, sigma, lam, sizes = res.x, res.eqlin.marginals[:N], res.eqlin.marginals[N:], cur
        master = float(res.fun)
        times, red_sum, added = [], 0.0, 0
        for i, s in enumerate(sites):
            t0 = time.time()
            out = s.solve(lam)
            times.append(time.time() - t0)
            lps += 1
            if out is None:
                continue
            red = out[0] - lam @ out[1] - sigma[i]
            red_sum += min(0.0, red)
            if red < -1e-7 * max(1.0, abs(out[0])):
                cols[i].append(out)
                added += 1
        site_s += max(times)
        res_info.update({'rounds': r + 1, 'gap': -red_sum / max(abs(master), 1e-9), 'master': master})
        if added == 0 or res_info['gap'] <= gap_tol:
            break
    phi, xs, mus = np.zeros((N, B)), [], []
    k = 0
    for i, cs in enumerate(cols):
        cs_w = cs[:sizes[i]]                                 # columns the last master weighed
        m_i = mu[k:k + len(cs_w)] if mu is not None else np.r_[1.0, np.zeros(len(cs_w) - 1)]
        phi[i] = sum(w * c[1] for w, c in zip(m_i, cs_w))
        xs.append(cs_w[int(np.argmax(m_i))][2])
        mus.append(m_i)
        k += len(cs_w)
    res_info.update({'phi': phi, 'lam': lam, 'mu': mus, 'xs': xs, 'cols': cols, 'site_s': site_s, 'lps': lps,
                     'unrouted': float(mu[-B:].sum()) if mu is not None else float(B)})
    return res_info


class DualScreenPolicy(DualPolicy):
    def __init__(self, mode='screen', sigma24=0.10, q=1.645, rho=0.3, fc_seed=0, passes=2, ans_tol=1e-6, **kw):
        super().__init__(sigma24=sigma24, rho=rho, fc_seed=fc_seed, **kw)
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode, self.q, self.passes, self.ans_tol = mode, q, int(passes), float(ans_tol)

    def reset(self, geo):
        super().reset(geo)
        self.fc = ForecastModel(self.ci, [c['location'] for c in geo.site_configs], self.sigma24, self.q, self.rho,
                                self.fc_seed)
        self.refined_sites = set(range(geo.n_sites)) if self.mode == 'full' else set()
        self.stats['dscreen'] = []

    def _ci_for(self, geo, i, lp):
        return self.fc.window(i, geo.t, lp.W, 'refined' if i in self.refined_sites else 'nominal')

    def _answers_differ(self, lp, i, geo, bids, lam, phi_nom):
        """Box test: does site i's answer at prices lam change between its nominal forecast and the box corners?"""
        n_lp = 0
        for kind in ('low', 'high'):
            out = SitePricing(lp, self.fc.window(i, geo.t, lp.W, kind), bids).solve(lam)
            n_lp += 1
            if out is None or np.abs(out[1] - phi_nom).max() > 1e-4:
                return True, n_lp
        return False, n_lp

    def _replan(self, geo):
        t = geo.t
        self.stats['replans'] += 1
        self.acc[:] = 0.0
        self.curves = {}
        groups = self._bundles(geo)
        if not groups:
            self.target = None
            return
        t0 = time.time()
        N, bids = geo.n_sites, list(groups)
        B = len(bids)
        lps = [self._site_lp(geo, i, bundle=[groups[b] for b in bids]) for i in range(N)]
        if self.mode == 'full':
            self.refined_sites = set(range(N))
        elif self.mode == 'nominal':
            self.refined_sites = set()
        else:
            self.refined_sites = set()                       # each hour starts from nominal forecasts
        sites = [SitePricing(lp, self._ci_for(geo, i, lp), bids) for i, lp in enumerate(lps)]
        # unrouted work must cost more than any site's overflow (as in geo_dual.DualPolicy)
        over = max(sum(lp.c_obj[lp.iv[c]] for c in np.flatnonzero(lp.is_bundle)) for lp in lps)
        M = self.big_m or 10.0 * max(over, 1.0)
        info = coordinate(sites, B, self.max_rounds, self.gap_tol, big_m=M)
        rounds, lp_count, par_s, passes = info['rounds'], info['lps'], info['site_s'], 0
        boundary = set()
        if self.mode == 'screen':
            checked = set()
            for p in range(self.passes):
                lam = info['lam']
                if lam is None:
                    break
                new, box_s = set(), []
                for i in range(N):
                    if i in boundary:
                        continue
                    ts = time.time()
                    split = (info['mu'][i] > self.ans_tol).sum() > 1
                    own = sites[i].solve(lam)
                    lp_count += 1
                    if split or own is None:
                        new.add(i)
                    else:
                        differ, n = self._answers_differ(lps[i], i, geo, bids, lam, own[1])
                        lp_count += n
                        if differ:
                            new.add(i)
                    box_s.append(time.time() - ts)
                    checked.add(i)
                par_s += max(box_s) if box_s else 0.0
                if not new:
                    break
                boundary |= new
                passes = p + 1
                # refine the boundary and coordinate again, warm-started: re-price the proposals under each site's
                # current forecast
                self.refined_sites = set(boundary)
                sites = [SitePricing(lp, self._ci_for(geo, i, lp), bids) for i, lp in enumerate(lps)]
                cols = []
                for i, cs in enumerate(info['cols']):
                    cols.append([(float(sites[i].c @ c[2]), c[1], c[2]) if c[2] is not None else c for c in cs])
                info = coordinate(sites, B, self.max_rounds, self.gap_tol, cols=cols, big_m=M)
                rounds += info['rounds']
                lp_count += info['lps']
                par_s += info['site_s']
        rec = {'t': t, 'n_sites': N, 'boundary_sites': len(boundary) if self.mode == 'screen' else
               (N if self.mode == 'full' else 0), 'passes': passes, 'rounds': rounds, 'lps': lp_count,
               'parallel_s': par_s, 'total_s': time.time() - t0, 'gap': info['gap'], 'unrouted': info['unrouted'],
               'forecast_values_refined': int(sum(lps[i].W for i in self.refined_sites)),
               'forecast_values_full': int(sum(lp.W for lp in lps))}
        self.stats['dscreen'].append(rec)
        for k, v in (('dw_rounds', rounds), ('dw_gap', info['gap']), ('dw_site_s', par_s), ('dw_master_s', 0.0),
                     ('dw_total_s', rec['total_s']), ('dw_unrouted', info['unrouted'])):
            self.stats[k].append(v)
        phi = info['phi']
        self.work = {b: groups[b]['work'] for b in bids}
        self.target = np.zeros((N, len(self.edges) + 1))
        for k, b in enumerate(bids):
            self.target[:, b] = phi[:, k] * groups[b]['work']
            for i in range(N):
                self.curves[(i, b)] = True
        self.last_phi = {b: phi[:, k] for k, b in enumerate(bids)}
        for i, (lp, x) in enumerate(zip(lps, info['xs'])):
            if x is not None:
                self.plans[i] = self._make_plan(lp, x[:lp.c_obj.size], t)


def dscreen_summary(policy):
    recs = policy.stats.get('dscreen') or []
    if not recs:
        return {}
    N = recs[0]['n_sites']
    return {'boundary_site_frac': float(np.mean([r['boundary_sites'] / N for r in recs])),
            'boundary_site_frac_max': float(np.max([r['boundary_sites'] / N for r in recs])),
            'forecast_frac_refined': float(np.sum([r['forecast_values_refined'] for r in recs])
                                           / max(np.sum([r['forecast_values_full'] for r in recs]), 1)),
            'forecast_values_per_hour': float(np.mean([r['forecast_values_refined'] for r in recs])),
            'ds_passes_mean': float(np.mean([r['passes'] for r in recs])),
            'ds_rounds_mean': float(np.mean([r['rounds'] for r in recs])),
            'ds_lps_per_site_hour': float(np.mean([r['lps'] / N for r in recs])),
            'ds_parallel_s_mean': float(np.mean([r['parallel_s'] for r in recs])),
            'ds_gap_max': float(np.max([r['gap'] for r in recs]))}
