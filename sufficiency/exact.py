"""Exact decision-sufficiency analysis of one routing decision (Bennouna et al. / Ye-Amin-Ozdaglar), for comparing
with the screening rule.

The decision. One bundle, items a = 1..n (segment k of site i's index curve) with widths w_a, demand D (one bundle):
    min_x  sum_a nu_a(c) x_a   s.t.  sum_a x_a = D,  0 <= x_a <= w_a          (the fluid clearing)
with linear indices nu_a(c) = pi_a . c_{r(a)} + kappa_a (sufficiency/screen.py: pi = energy profile of the marginal
work, kappa = penalties), where c_r is the CI forecast window of region r, uncertain in a box lo_r <= c_r <= hi_r.
Sites in one region share c_r. Stacking the regions' windows gives c in R^m and nu(c) = P c + kappa.

Exact quantities (the box replaces dstar.py's ball, so every test is an LP instead of a closed form):
    X*(C)        reachable optimal vertices: walk polytope edges from x*(c_nominal); a vertex is reachable iff its
                 optimality region {c in box : exists lambda, nu_f <= lambda (x_f = w_f), nu_e >= lambda (x_e = 0),
                 nu_p = lambda (partial)} is non-empty (an LP). Optimality regions tile the box, so walking edges
                 between reachable vertices finds them all.
    d*_vert      dim span{x - x' : x, x' in X*(C)}                                  (Bennouna Thm 2 / Ye Eq. 3)
    Delta(X, C)  edge directions e_j - e_i at reachable vertices whose tie face (vertex optimal and nu_i = nu_j)
                 meets the box;  d*_Delta = rank.  Ye Thm 3: span Delta = dir X*(C).
    relevant measurement directions in c-space: P^T (e_j - e_i) = pi_j - pi_i (block-placed by region). Observing
                 c along these, Q c, decides x*(c) for every c in the box (Bennouna Thm 1).
    certificate(c, Q): x*(c) stays optimal for every c' in the box with Q c' = Q c, i.e. for every edge (i -> j) at
                 x*(c): min {nu_j(c') - nu_i(c') : c' in box, Q c' = Q c} >= 0 (an LP per edge; edges that the
                 box alone settles are skipped).  Ye Algorithm 1 adds the first failing edge's direction; Algorithm 2
                 runs Algorithm 1 on sampled costs, carrying the dataset forward.

Screening's dataset for the same decision: the boundary items B (classify() with the exact linear intervals and
same-region differences) and the directions pi_a - pi_ref, a in B (|B| - 1 of them). compare() reports whether they
span the relevant directions, and whether the certificate holds with them at sampled costs.

rank / basis come from ../sufficient_stats/dstar.py.
"""
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'sufficient_stats'))
from dstar import basis, rank  # noqa: E402

from sufficiency.top_m import classify, clear  # noqa: E402

TOL = 1e-9


class RoutingDecision:
    def __init__(self, pis, kappas, regions, w, boxes, D=1.0):
        """pis: (n, W) per item; kappas: (n,); regions: (n,) labels; w: (n,); boxes: {region: (lo (W,), hi (W,))}."""
        self.n, W = len(w), pis.shape[1]
        self.w, self.D, self.kappa = np.asarray(w, float), float(D), np.asarray(kappas, float)
        self.regs = sorted(set(regions))
        off = {r: k * W for k, r in enumerate(self.regs)}
        self.m = W * len(self.regs)
        self.P = np.zeros((self.n, self.m))                    # nu = P c + kappa
        for a, r in enumerate(regions):
            self.P[a, off[r]:off[r] + W] = pis[a]
        self.lo = np.concatenate([boxes[r][0][:W] for r in self.regs])
        self.hi = np.concatenate([boxes[r][1][:W] for r in self.regs])
        self.c0 = (self.lo + self.hi) / 2

    # -- basics ---------------------------------------------------------------------------------------------------
    def nu(self, c):
        return self.P @ c + self.kappa

    def solve(self, c):
        return clear(self.nu(c), self.w, self.D)

    def sample(self, rng):
        return rng.uniform(self.lo, self.hi)

    def box_min(self, v, const=0.0):
        """min over the box of v . c + const (closed form)."""
        return const + np.minimum(v * self.lo, v * self.hi).sum()

    def status(self, x):
        full = np.flatnonzero(x >= self.w - 1e-9)
        empty = np.flatnonzero(x <= 1e-9)
        part = np.setdiff1d(np.arange(self.n), np.r_[full, empty])
        return full, part, empty

    def key(self, x):
        return tuple(np.round(x, 9))

    # -- optimality region of a vertex (LP feasibility over (c, lambda)) ------------------------------------------
    def _region_lp(self, x, extra_eq=None, objective=None):
        full, part, empty = self.status(x)
        m = self.m
        A, b, Aeq, beq = [], [], [], []
        for f in full:            # nu_f - lambda <= 0
            A.append(np.r_[self.P[f], -1.0]); b.append(-self.kappa[f])
        for e in empty:           # lambda - nu_e <= 0
            A.append(np.r_[-self.P[e], 1.0]); b.append(self.kappa[e])
        for p in part:            # nu_p = lambda
            Aeq.append(np.r_[self.P[p], -1.0]); beq.append(-self.kappa[p])
        if extra_eq is not None:
            for row, rhs in extra_eq:
                Aeq.append(np.r_[row, 0.0]); beq.append(rhs)
        bounds = list(zip(self.lo, self.hi)) + [(None, None)]
        c_obj = np.zeros(m + 1) if objective is None else np.r_[objective, 0.0]
        res = linprog(c_obj, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None,
                      A_eq=np.array(Aeq) if Aeq else None, b_eq=np.array(beq) if beq else None, bounds=bounds,
                      method='highs')
        return res

    def reachable_vertex(self, x):
        return self._region_lp(x).status == 0

    # -- X*(C): walk edges between reachable vertices -------------------------------------------------------------
    def edges(self, x):
        """Edge moves at vertex x: shift mass from i (x_i > 0) to j (x_j < w_j). Yields (i, j, next vertex)."""
        for i in np.flatnonzero(x > 1e-9):
            for j in np.flatnonzero(x < self.w - 1e-9):
                if i == j:
                    continue
                t = min(x[i], self.w[j] - x[j])
                y = x.copy()
                y[i] -= t
                y[j] += t
                yield i, j, y

    def can_be_cheaper(self, j, i):
        """Is nu_j <= nu_i for some c in the box? (closed form; necessary for edge i -> j to matter)."""
        return self.box_min(self.P[j] - self.P[i], self.kappa[j] - self.kappa[i]) <= TOL

    def reachable_set(self, max_vertices=5000):
        x0 = self.solve(self.c0)
        seen, todo, out = {self.key(x0)}, [x0], [x0]
        while todo and len(out) < max_vertices:
            x = todo.pop()
            for i, j, y in self.edges(x):
                if not self.can_be_cheaper(j, i):
                    continue
                k = self.key(y)
                if k in seen:
                    continue
                seen.add(k)
                if self.reachable_vertex(y):
                    out.append(y)
                    todo.append(y)
        return np.array(out)

    def relevant_edges(self, X):
        """Delta(X, C): (i, j) at reachable vertices whose tie face meets the box."""
        out = set()
        for x in X:
            for i, j, _ in self.edges(x):
                if (i, j) in out or not self.can_be_cheaper(j, i):
                    continue
                row = self.P[j] - self.P[i]
                if self._region_lp(x, extra_eq=[(row, self.kappa[i] - self.kappa[j])]).status == 0:
                    out.add((i, j))
        return sorted(out)

    def direction(self, i, j):
        return self.P[j] - self.P[i]

    # -- Ye's certificate with a box ------------------------------------------------------------------------------
    def certificate(self, c, Q):
        """(ok, first failing edge) for dataset Q (rows: measurement directions in c-space) at cost c."""
        x = self.solve(c)
        Qm = np.array(Q) if len(Q) else None
        worst = None
        for i, j, _ in self.edges(x):
            if not self.can_be_cheaper(j, i):
                continue
            row = self.P[j] - self.P[i]
            res = linprog(row, A_eq=Qm, b_eq=None if Qm is None else Qm @ c, bounds=list(zip(self.lo, self.hi)),
                          method='highs')
            if res.status == 0 and res.fun + self.kappa[j] - self.kappa[i] < -1e-7:
                gap = row @ c + self.kappa[j] - self.kappa[i]          # how decided it is at c
                if worst is None or gap < worst[0]:
                    worst = (gap, i, j)
        return worst is None, (None if worst is None else worst[1:])

    def alg1(self, c, Q):
        Q = list(Q)
        while True:
            ok, e = self.certificate(c, Q)
            if ok:
                return Q
            Q.append(self.direction(*e))

    def alg2(self, rng, n_samples):
        Q, T = [], 0
        for _ in range(n_samples):
            Q2 = self.alg1(self.sample(rng), Q)
            T += int(len(Q2) > len(Q))
            Q = Q2
        return Q, T

    # -- the screening rule on the same decision ------------------------------------------------------------------
    def screening(self, regions):
        """Boundary items by classify() with exact linear intervals and exact same-region differences."""
        lo = np.array([self.box_min(self.P[a], self.kappa[a]) for a in range(self.n)])
        hi = np.array([-self.box_min(-self.P[a], -self.kappa[a]) for a in range(self.n)])
        regions = np.asarray(regions)
        cheaper = lo[:, None] <= hi[None, :] + TOL
        for b in range(self.n):
            for a in range(self.n):
                if a != b and regions[a] == regions[b]:
                    cheaper[b, a] = self.can_be_cheaper(b, a)
        np.fill_diagonal(cheaper, False)
        lab = classify(lo, hi, self.w, self.D, cheaper=cheaper)
        B = np.flatnonzero(lab == 0)
        if len(B) == 0:
            return lab, B, []
        ref = B[np.argsort(self.nu(self.c0)[B])[len(B) // 2]]
        return lab, B, [self.direction(ref, a) for a in B if a != ref]


def compare(dec, regions, rng, n_alg2=40, n_cert=40):
    """Exact relevant directions vs the screening's boundary differences on one decision."""
    X = dec.reachable_set()
    W_vert = basis(X[1:] - X[0]) if len(X) > 1 else np.zeros((0, dec.n))
    E = dec.relevant_edges(X)
    Delta_x = [np.eye(dec.n)[j] - np.eye(dec.n)[i] for i, j in E]
    Delta_c = [dec.direction(i, j) for i, j in E]
    lab, B, ours = dec.screening(regions)
    span_ours = basis(ours) if ours else np.zeros((0, dec.m))
    resid = 0.0
    if Delta_c:
        Dc = np.array(Delta_c)
        resid = float(np.linalg.norm(Dc - (Dc @ span_ours.T) @ span_ours) / max(np.linalg.norm(Dc), 1e-12)) \
            if len(span_ours) else float(np.linalg.norm(Dc) > 0)
    Qy, T = dec.alg2(rng, n_alg2)
    cert = [dec.certificate(dec.sample(rng), ours)[0] for _ in range(n_cert)]
    return {'items': dec.n, 'reachable_vertices': len(X), 'd_vert': len(W_vert), 'relevant_edges': len(E),
            'd_delta_x': rank(Delta_x), 'd_delta_c': rank(Delta_c), 'boundary_items': int(len(B)),
            'ours': len(ours), 'ours_rank': rank(ours), 'relevant_outside_ours': resid,
            'ye_alg2_size': len(Qy), 'ye_alg2_rank': rank(Qy), 'ye_alg2_growths': T,
            'certificate_ours': float(np.mean(cert)),
            'boundary_contains_relevant': bool(set(np.r_[[i for i, _ in E], [j for _, j in E]].astype(int)) <= set(B.tolist()))}
