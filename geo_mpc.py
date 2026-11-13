"""B1: centralized rolling-horizon MPC for GeoJobsDC(routing='site_queue').

Every hour, one LP over all sites: every site's planning LP (geo_index.SiteLP: its queue, running jobs, battery and
cooling envelope over the window, with the past hour's arrivals as the forecast bundles) side by side, coupled by
routing shares
    phi[i, b] >= 0,   sum_i phi[i, b] = 1          (each bundle b is routed in full; a site's share of it must start
                                                   within its slack there, as in the index LP)
Variables ~ N x (window x (start slots + 5) + bundles). This is the joint problem the index policy decomposes: at the
LP optimum, the duals lambda_b of the coupling rows are the clearing prices, and each site's share sits where its
index curve crosses lambda_b / work_b (tests/verify_screening.py checks this).

Execution (same as IndexPolicy, so only routing differs): arrivals during the hour go to the site with the most of its
target share phi*[i, b] x work_b left (among sites where the job fits in time), then by marginal CO2 once every target
is met; each site re-plans its own queue, battery and setpoint every step with its own LP.

`time_limit` (s) caps each hourly solve; a solve that does not finish keeps the previous targets (or routes by
marginal CO2 if there are none) and is counted in stats['timeouts'].
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from geo_index import IndexPolicy
from sufficiency.forecast import ForecastModel


class JointMPCPolicy(IndexPolicy):
    def __init__(self, time_limit=None, sigma24=0.0, rho=0.3, fc_seed=0, forecast='nominal', **kw):
        super().__init__(**kw)
        self.time_limit = time_limit
        self.sigma24, self.rho, self.fc_seed, self.forecast = sigma24, rho, fc_seed, forecast

    def reset(self, geo):
        super().reset(geo)
        self.fc = ForecastModel(self.ci, [c['location'] for c in geo.site_configs], self.sigma24, rho=self.rho,
                                seed=self.fc_seed)
        self.target, self.work, self.last_duals = None, {}, None
        self.stats.update({'joint_vars': [], 'joint_s': [], 'timeouts': 0})

    def _ci_for(self, geo, i, lp):
        return self.fc.window(i, geo.t, lp.W, self.forecast)

    def joint_lp(self, geo, groups):
        """Assemble the joint LP. Returns (site lps, bundle ids, c, A_ub, b_ub, A_eq, b_eq, bounds, phi column index)."""
        N, bids = geo.n_sites, list(groups)
        B = len(bids)
        lps = [self._site_lp(geo, i, bundle=[groups[b] for b in bids]) for i in range(N)]
        off = np.cumsum([0] + [lp.c_obj.size for lp in lps])
        nphi = N * B
        nvar = off[-1] + nphi
        iphi = off[-1] + np.arange(nphi).reshape(N, B)
        c = np.zeros(nvar)
        eq_blocks, ub_blocks, b_eq, b_ub, bounds = [], [], [], [], []
        phi_rows, phi_cols = [], []
        row = 0
        for i, lp in enumerate(lps):
            c[off[i]:off[i + 1]] = lp.objective(self._ci_for(geo, i, lp))
            eq_blocks.append(lp.A_eq)
            ub_blocks.append(lp.A_ub)
            rhs = lp.rhs(0.0)          # bundle rows: sum x + overflow - phi[i, b] = 0
            b_eq.append(rhs)
            b_ub.append(lp.b_ub)
            bounds.append(lp.bounds)
            for c_idx in np.flatnonzero(lp.is_bundle):
                phi_rows.append(row + c_idx)
                phi_cols.append(iphi[i, lp.bid[c_idx]])
            row += lp.A_eq.shape[0]
        A_eq = sp.block_diag(eq_blocks, format='csr')
        A_eq = sp.hstack([A_eq, sp.csr_matrix((A_eq.shape[0], nphi))], format='csr')
        A_eq = A_eq + sp.csr_matrix((-np.ones(len(phi_rows)), (phi_rows, phi_cols)), shape=A_eq.shape)
        link = sp.csr_matrix((np.ones(nphi), (np.tile(np.arange(B), N), iphi.reshape(-1))), shape=(B, nvar))
        A_eq = sp.vstack([A_eq, link], format='csr')
        b_eq = np.concatenate(b_eq + [np.ones(B)])
        A_ub = sp.block_diag(ub_blocks, format='csr')
        A_ub = sp.hstack([A_ub, sp.csr_matrix((A_ub.shape[0], nphi))], format='csr')
        bounds = np.vstack(bounds + [np.c_[np.zeros(nphi), np.ones(nphi)]])
        return lps, bids, c, A_ub, np.concatenate(b_ub), A_eq, b_eq, bounds, iphi, off

    def _replan(self, geo):
        t = geo.t
        self.stats['replans'] += 1
        self.acc[:] = 0.0
        self.curves = {}
        groups = self._bundles(geo)
        if not groups:
            self.target = None
            return
        lps, bids, c, A_ub, b_ub, A_eq, b_eq, bounds, iphi, off = self.joint_lp(geo, groups)
        opts = {} if self.time_limit is None else {'time_limit': float(self.time_limit)}
        t0 = time.time()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options=opts)
        dt = time.time() - t0
        self.stats['lp_solves'] += 1
        self.stats['lp_seconds'] += dt
        self.stats['joint_vars'].append(int(c.size))
        self.stats['joint_s'].append(dt)
        if res.status != 0:
            self.stats['timeouts'] += 1
            return                                   # keep the previous targets (stale) or merit routing
        x = res.x
        phi = x[iphi]
        self.work = {b: groups[b]['work'] for b in bids}
        self.target = np.zeros((geo.n_sites, len(self.edges) + 1))
        for k, b in enumerate(bids):
            self.target[:, b] = phi[:, k] * groups[b]['work']
            for i in range(geo.n_sites):
                self.curves[(i, b)] = True           # so that route() counts the work it sends (self.acc)
        self.last_duals = {b: float(res.eqlin.marginals[-len(bids) + k]) for k, b in enumerate(bids)}
        self.last_phi = {b: phi[:, k] for k, b in enumerate(bids)}
        for i, lp in enumerate(lps):
            self.plans[i] = self._make_plan(lp, x[off[i]:off[i + 1]], t)

    def _index(self, i, b, extra_work):
        """Routing key: the more of its target share a site has left, the lower; past every target, marginal CO2."""
        if self.target is None or b not in self.work:
            return None
        rem = self.target[i, b] - self.acc[i, b] - extra_work
        return -rem / self.work[b] if rem >= 0 else 1e6 + self.A[i, self._t]

    def act(self, geo):
        self._t = geo.t
        return super().act(geo)
