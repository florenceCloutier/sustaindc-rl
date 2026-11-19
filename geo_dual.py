"""Price-coordinated routing (Dantzig-Wolfe decomposition of B1's joint LP), for GeoJobsDC(routing='site_queue').

Why. The index policy (geo_index.py) gives each site one cost curve per bundle, computed as if that bundle were the
only extra work, so a site offers the same free capacity to every bundle. B1 (geo_mpc.py) solves one joint LP over
all sites and bundles and avoids that, but it is one big centralized problem. This policy reaches B1's solution
while each site only ever solves its own LP:

    site i's pricing problem at prices lambda (one per bundle):
        min  cost_i(x)  -  sum_b lambda_b phi_{i,b}        over its own plan x and its shares phi_{i,b} in [0, 1]
        (geo_index.SiteLP with all bundles, each bundle's share a variable: the same block as in B1's joint LP)
    coordinator (master LP), over the plans proposed so far, P_i for site i:
        min  sum_{i,p} mu_{i,p} cost_{i,p}  +  M sum_b a_b
        s.t. sum_p mu_{i,p} = 1                             (each site picks a mix of its proposals)
             sum_{i,p} mu_{i,p} phi_{i,p,b} + a_b = 1       (every bundle routed in full; a_b: unrouted, cost M)
             mu, a >= 0
    Each round: solve the master, read its prices (duals lambda_b of the routing rows, sigma_i of the site rows),
    have every site solve its pricing problem, and add each plan whose reduced cost is below sigma_i. Stop when no
    site improves, or when the gap to the Lagrangian bound,
        master value - (master value + sum_i min(0, reduced cost_i)),
    is below `gap` (relative), or after `max_rounds`.
At convergence the shares sum_p mu_{i,p} phi_{i,p,b} are an optimal solution of B1's joint LP (LP duality), so
routing and execution then proceed exactly as in B1 (targets per site and bundle; geo_mpc.JointMPCPolicy).

Cost per hour: rounds x N independent site LPs (parallel in principle) + one small master LP per round.

Warm start (`warm`). Cold ('none'), every site's first proposal is "take nothing" (its answer at prices -inf), so the
first master prices are the unrouted penalty M and the first rounds only find the price range. Instead, each site's
first proposals can be its answers at
    'last'   the previous hour's prices (per bundle id; a bundle new this hour gets the highest of the others)
    'index'  the index policy's clearing prices this hour: each site's cost curve per bundle (geo_index, G levels,
             the same forecast), filled cheapest segment first until the whole bundle is placed; the price of a
             bundle is the index of the segment that completes it (x the bundle's work, to price a unit share)
    'both'   both (two proposals per site)
"Take nothing" stays among every site's proposals (without it the master can be infeasible: a site must be able to
take less than its warm answer). 'none' costs one site LP per site before the first master, 'last' two; 'index' adds
the index curves (1 + B x G LPs per site, in parallel). With `max_rounds` reached before convergence, the master is solved once more over every
proposal so the last round's answers are used.

stats: 'dw_rounds', 'dw_gap', 'dw_site_s' (sum over rounds of the slowest site's solve: the parallel time, including
the warm start), 'dw_total_s', 'dw_master_s', 'dw_site_lps' (site LPs per hour, all sites, warm start included),
'dw_warm_s' (parallel time of the warm start), 'dw_converged', 'dw_master_fail' (master solves that failed; the
last good master's mix is then used), 'dw_master_msg' (the last failure's message).
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from geo_mpc import JointMPCPolicy


class SitePricing:
    """One site's LP with every bundle and its shares phi_b as variables (the site's block of B1's joint LP)."""

    def __init__(self, lp, ci, bids):
        self.lp, self.B = lp, len(bids)
        n = lp.c_obj.size
        self.c = np.r_[lp.objective(ci), np.zeros(self.B)]
        self.iphi = n + np.arange(self.B)
        rows, cols = [], []
        for c_idx in np.flatnonzero(lp.is_bundle):          # bundle rows: sum x + overflow - phi_b = 0
            rows.append(c_idx)
            cols.append(n + lp.bid[c_idx])
        A = sp.hstack([lp.A_eq, sp.csr_matrix((lp.A_eq.shape[0], self.B))], format='csr')
        self.A_eq = A + sp.csr_matrix((-np.ones(len(rows)), (rows, cols)), shape=A.shape)
        self.b_eq = lp.rhs(0.0)
        self.A_ub = sp.hstack([lp.A_ub, sp.csr_matrix((lp.A_ub.shape[0], self.B))], format='csr')
        self.bounds = np.vstack([lp.bounds, np.c_[np.zeros(self.B), np.ones(self.B)]])

    def solve(self, lam):
        """Best plan at prices lam (B,): (cost, phi (B,), x) with cost = the site's own CO2 + penalties."""
        c = self.c.copy()
        c[self.iphi] -= lam
        res = None
        for method in ('highs', 'highs-ds', 'highs-ipm'):
            res = linprog(c, A_ub=self.A_ub, b_ub=self.lp.b_ub, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.bounds,
                          method=method)
            if res.status == 0:
                break
        if res.status != 0:
            return None
        x = res.x
        phi = x[self.iphi]
        return float(self.c @ x), phi, x


WARM = ('none', 'last', 'index', 'both')


def _master(cols, B, M):
    """Master LP over the proposals cols (per site [(cost, phi, x)]); returns (linprog result, proposals per site)."""
    N = len(cols)
    sizes = [len(c) for c in cols]
    ncol = sum(sizes)
    cost = np.r_[[c[0] for cs in cols for c in cs], np.full(B, M)]
    A = np.zeros((N + B, ncol + B))
    k = 0
    for i, cs in enumerate(cols):
        for c in cs:
            A[i, k] = 1.0
            A[N:, k] = c[1]
            k += 1
    A[N:, ncol:] = np.eye(B)
    res = linprog(cost, A_eq=A, b_eq=np.r_[np.ones(N), np.ones(B)], bounds=(0, None), method='highs')
    return res, sizes


class DualPolicy(JointMPCPolicy):
    def __init__(self, max_rounds=40, gap=1e-4, big_m=None, warm='none', **kw):
        super().__init__(**kw)
        if warm not in WARM:
            raise ValueError(f'warm must be one of {WARM}')
        self.max_rounds, self.gap_tol, self.big_m, self.warm = int(max_rounds), float(gap), big_m, warm

    def reset(self, geo):
        super().reset(geo)
        self.stats.update({'dw_rounds': [], 'dw_gap': [], 'dw_site_s': [], 'dw_total_s': [], 'dw_master_s': [],
                           'dw_unrouted': [], 'dw_site_lps': [], 'dw_warm_s': [], 'dw_converged': [],
                           'dw_master_fail': 0, 'dw_master_msg': ''})
        self.last_lam = {}

    def _index_prices(self, geo, groups, bids):
        """The index policy's clearing price of each bundle (per unit share), from every site's cost curves.
        Returns ({bid: price}, slowest site's time)."""
        segs = {b: [] for b in bids}                        # (index, width in shares)
        times = []
        for i in range(geo.n_sites):
            t0 = time.time()
            base = self._site_lp(geo, i)
            v0, _ = self._solve(base, 0.0, self._ci_for(geo, i, base))
            if np.isfinite(v0):
                for b in bids:
                    g = groups[b]
                    lp = self._site_lp(geo, i, bundle=g)
                    phi_max = self._phi_max(geo, lp, g)
                    if phi_max <= 1e-6:
                        continue
                    phis = np.linspace(0.0, phi_max, self.G + 1)
                    nus = self._curve(lp, phis, v0, g['work'], self._ci_for(geo, i, lp))[0]
                    segs[b] += [(float(nu), float(w)) for nu, w in zip(nus, np.diff(phis)) if np.isfinite(nu)]
            times.append(time.time() - t0)
        lam = {}
        for b in bids:
            s = sorted(segs[b])
            if not s:
                continue                                    # no site can take any of it: no guess
            cum, nu_star = 0.0, s[-1][0]                    # not enough room: the dearest segment offered
            for nu, w in s:
                cum += w
                if cum >= 1.0 - 1e-9:
                    nu_star = nu
                    break
            lam[b] = nu_star * groups[b]['work']
        return lam, (max(times) if times else 0.0)

    @staticmethod
    def _fill(lam, bids):
        """Price vector over bids from a partial {bid: price}; missing bundles get the highest known price."""
        if not lam:
            return None
        top = max(lam.values())
        return np.array([lam.get(b, top) for b in bids], dtype=float)

    def coordinate(self, geo, groups):
        """Dantzig-Wolfe on this hour's bundles. Returns (bids, phi (N, B), site lps, chosen x per site, info)."""
        N, bids = geo.n_sites, list(groups)
        B = len(bids)
        lps = [self._site_lp(geo, i, bundle=[groups[b] for b in bids]) for i in range(N)]
        sites = [SitePricing(lp, self._ci_for(geo, i, lp), bids) for i, lp in enumerate(lps)]
        # first proposals: each site's answers at the warm-start prices, else "take nothing" (prices -inf)
        seeds, warm_s = [], 0.0
        if self.warm in ('last', 'both'):
            lam0 = self._fill({b: self.last_lam[b] for b in bids if b in self.last_lam}, bids)
            if lam0 is not None:
                seeds.append(lam0)
        if self.warm in ('index', 'both'):
            lam_idx, warm_s = self._index_prices(geo, groups, bids)
            lam0 = self._fill(lam_idx, bids)
            if lam0 is not None:
                seeds.append(lam0)
        # Every site keeps "take nothing": the master needs each site to be able to take less (shares sum to 1 at
        # most, and each site picks a mix of its proposals), so with only warm proposals it can be infeasible.
        cols = [[] for _ in range(N)]                                   # per site: (cost, phi, x)
        times, n_lps = [], 0
        for i, s in enumerate(sites):
            t0 = time.time()
            out = s.solve(np.full(B, -1e12))
            n_lps += 1
            cols[i].append(out if out is not None else (0.0, np.zeros(B), None))
            for lam0 in seeds:
                out = s.solve(lam0)
                n_lps += 1
                if out is not None:
                    cols[i].append(out)
            times.append(time.time() - t0)
        t_site = warm_s + max(times)
        # leaving work unrouted must cost more than any site's overflow (the joint LP has no unrouted option): the
        # cost of sending a whole bundle to one site's overflow is the sum of its classes' overflow costs
        over = max(sum(lp.c_obj[lp.iv[c]] for c in np.flatnonzero(lp.is_bundle)) for lp in lps)
        M = self.big_m or 10.0 * max(over, 1.0, max(abs(c[0][0]) for c in cols))
        info = {'rounds': 0, 'gap': np.inf, 'site_s': t_site, 'warm_s': warm_s, 'master_s': 0.0, 'converged': False}
        mu = sizes = lam = None                    # from the last master solve that succeeded (recovery uses these)

        def master():
            nonlocal mu, sizes, lam
            t0 = time.time()
            res, cur = _master(cols, B, M)
            info['master_s'] += time.time() - t0
            if res.status != 0:
                self.stats['dw_master_fail'] += 1
                self.stats['dw_master_msg'] = f't={geo.t} round {info["rounds"]}: {res.message}'
                return None
            mu, sizes, lam = res.x, cur, res.eqlin.marginals[N:]
            return res

        added = 0
        for r in range(self.max_rounds):
            res = master()
            if res is None:
                break
            sigma = res.eqlin.marginals[:N]
            master_val = float(res.fun)
            # pricing: every site at prices lam
            times, red_sum, added = [], 0.0, 0
            for i, s in enumerate(sites):
                t0 = time.time()
                out = s.solve(lam)
                n_lps += 1
                times.append(time.time() - t0)
                if out is None:
                    continue
                red = out[0] - lam @ out[1] - sigma[i]
                red_sum += min(0.0, red)
                if red < -1e-7 * max(1.0, abs(out[0])):
                    cols[i].append(out)
                    added += 1
            info['site_s'] += max(times)
            info['rounds'] = r + 1
            info['gap'] = -red_sum / max(abs(master_val), 1e-9)
            info['master'] = master_val
            if added == 0 or info['gap'] <= self.gap_tol:
                info['converged'] = True
                break
        if not info['converged'] and added and info['rounds'] == self.max_rounds:
            master()                               # stopped early: weigh the last round's proposals too
        # recover shares and each site's plan (its proposal mix; the dominant proposal's x for the plan)
        phi = np.zeros((N, B))
        xs = []
        k = 0
        for i, cs in enumerate(cols):
            cs = cs[:sizes[i]] if sizes is not None else cs[:1]   # proposals the last good master weighed
            m_i = mu[k:k + len(cs)] if mu is not None else np.ones(1)
            phi[i] = sum(w * c[1] for w, c in zip(m_i, cs))
            xs.append(cs[int(np.argmax(m_i))][2])
            k += len(cs)
        info['unrouted'] = float(mu[-B:].sum()) if mu is not None else float(B)
        info['site_lps'] = n_lps
        if lam is not None:
            self.last_lam.update({b: float(lam[k]) for k, b in enumerate(bids)})
        return bids, phi, lps, xs, info

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
        bids, phi, lps, xs, info = self.coordinate(geo, groups)
        for k, v in (('dw_rounds', info['rounds']), ('dw_gap', info['gap']), ('dw_site_s', info['site_s']),
                     ('dw_master_s', info['master_s']), ('dw_total_s', time.time() - t0), ('dw_unrouted', info['unrouted']),
                     ('dw_site_lps', info['site_lps']), ('dw_warm_s', info['warm_s']), ('dw_converged', info['converged'])):
            self.stats[k].append(v)
        self.work = {b: groups[b]['work'] for b in bids}
        self.target = np.zeros((geo.n_sites, len(self.edges) + 1))
        for k, b in enumerate(bids):
            self.target[:, b] = phi[:, k] * groups[b]['work']
            for i in range(geo.n_sites):
                self.curves[(i, b)] = True
        self.last_phi = {b: phi[:, k] for k, b in enumerate(bids)}
        for i, (lp, x) in enumerate(zip(lps, xs)):
            if x is not None:
                self.plans[i] = self._make_plan(lp, x[:lp.c_obj.size], t)
